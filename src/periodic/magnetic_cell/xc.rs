// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Continuum London quadrature shared by finite cells and magnetic Bloch meshes.

use super::*;
use crate::periodic::spinor_xc::{spinor_semilocal_grid_converged, SpinorXcDiagnostics};
use std::collections::BTreeMap;

const XC_GRID_BLOCK: usize = 16_384;
const MAX_CACHED_COLLOCATION: usize = 512 * 1024 * 1024;
const PAIR_CHECK_ORDER: usize = 112;
const MAX_GRID_ORDER: usize = 128;

fn tensor_grid(
    geometry: &PeriodicGeometry,
    basis: &MolBasis,
    order: usize,
    cutoff: f64,
) -> (Vec<Vec3>, Vec<f64>) {
    let (nodes, weights) = gauss_legendre(order);
    let a = geometry.lattice().a;
    let (axes, weight_scale) = if geometry.dimension() == 2 {
        let periodic: Vec<_> = (0..3).filter(|&i| geometry.boundary().axes[i]).collect();
        let normal = a[periodic[0]].cross(a[periodic[1]]).normalized();
        let extent = basis
            .shells
            .iter()
            .map(|s| s.center.dot(normal).abs())
            .fold(0.0_f64, f64::max)
            + cutoff;
        (
            [a[periodic[0]] * 0.5, a[periodic[1]] * 0.5, normal * extent],
            geometry.measure() * extent / 4.0,
        )
    } else {
        (a.map(|v| v * 0.5), geometry.measure() / 8.0)
    };
    let mut points = Vec::with_capacity(order.pow(3));
    let mut values = Vec::with_capacity(order.pow(3));
    for i in 0..order {
        for j in 0..order {
            for k in 0..order {
                points.push(axes[0] * nodes[i] + axes[1] * nodes[j] + axes[2] * nodes[k]);
                values.push(weight_scale * weights[i] * weights[j] * weights[k]);
            }
        }
    }
    (points, values)
}

struct AtomCenteredGrid {
    molecule: Molecule,
    nuclei: usize,
}

impl AtomCenteredGrid {
    fn add_pair_centers(
        &mut self,
        geometry: &PeriodicGeometry,
        basis: &MolBasis,
        options: &MagneticScfOptions,
    ) -> Result<()> {
        self.molecule.atoms.truncate(self.nuclei);
        let mut exponents = vec![f64::INFINITY; self.nuclei];
        for shell in &basis.shells {
            for &exponent in &shell.exps {
                exponents[shell.atom] = exponents[shell.atom].min(exponent);
            }
        }
        let threshold = -options.operator_tolerance.ln();
        for i in 0..self.nuclei {
            let left = self.molecule.atoms[i].clone();
            for j in i..self.nuclei {
                let right = self.molecule.atoms[j].clone();
                let (a, b) = (exponents[i], exponents[j]);
                if !a.is_finite() || !b.is_finite() {
                    continue;
                }
                let exponent_sum = a + b;
                let displacement = right.position - left.position;
                for index in
                    geometry.nearby_translation_indices(displacement, options.basis_cutoff)?
                {
                    if i == j && index <= [0; 3] {
                        continue;
                    }
                    let offset = displacement + geometry.lattice().translation(index);
                    // Select centers of AO products in the retained overlap
                    // images. This attenuates only the choice of grid sites.
                    if a * (b / exponent_sum) * offset.norm2() > threshold {
                        continue;
                    }
                    let position = left.position + offset * (b / exponent_sum);
                    let duplicate = self.molecule.atoms.iter().any(|site| {
                        let delta = position - site.position;
                        let image = geometry.fractional(delta).map(|f| f.round() as i64);
                        (delta - geometry.lattice().translation(image)).norm2() < 1e-20
                    });
                    if !duplicate {
                        // The grid element sets the radial map and Becke
                        // radius only. Use the element of the nearer nucleus.
                        let z = match a.total_cmp(&b) {
                            std::cmp::Ordering::Greater => left.z,
                            std::cmp::Ordering::Less => right.z,
                            std::cmp::Ordering::Equal => left.z.min(right.z),
                        };
                        self.molecule
                            .atoms
                            .push(crate::system::Atom { z, position });
                    }
                }
            }
        }
        Ok(())
    }

    fn quadrature(
        &self,
        geometry: &PeriodicGeometry,
        order: usize,
        cutoff: f64,
    ) -> Result<(Vec<Vec3>, Vec<f64>)> {
        let radial = 4 * order;
        let degree = if self.has_pair_centers() {
            match order {
                MAX_GRID_ORDER => 151,
                PAIR_CHECK_ORDER.. => 131,
                _ => 13 + order,
            }
        } else {
            13 + 3 * order / 4
        };
        let options = crate::grid::GridOptions {
            prune: false,
            radial_override: Some(radial),
            angular_override: Some(degree),
            weight_cutoff: 0.0,
            ..Default::default()
        };
        // The periodic weights integrate one cell of gauge-invariant
        // contractions, including sites where the kinetic integrand is
        // sharply localized between nuclei.
        crate::periodic::grid::PeriodicGrid::dimensional_values_with_radial(
            &self.molecule,
            geometry,
            &options,
            cutoff,
            |site, z, count| {
                let scale = crate::grid::treutler_xi(z);
                if site < self.nuclei {
                    crate::grid::radial_treutler_ahlrichs(count, scale)
                } else {
                    // The core intervals resolve the small kinetic density
                    // near a stationary point of the occupied amplitude.
                    crate::grid::radial_treutler_split(count, scale, 1e-5)
                }
            },
        )
    }

    fn has_pair_centers(&self) -> bool {
        self.molecule.len() > self.nuclei
    }
}

struct GridFrame<'a> {
    cells: &'a [&'a MagneticCell],
    basis: &'a MolBasis,
    cutoff: f64,
    needs_momentum: bool,
}

struct Collocation {
    orbitals: Vec<BlochMatrix>,
    momentum: Option<Vec<[BlochMatrix; 3]>>,
}

impl Collocation {
    fn new(frame: &GridFrame<'_>, points: &[Vec3]) -> Result<Self> {
        // Use the same London images and cutoff for the covariant momentum.
        // Its contractions supply the gradient, kinetic and current densities.
        let (orbitals, momentum) = if frame.needs_momentum {
            let (orbitals, momentum) = frame
                .cells
                .iter()
                .map(|cell| cell.orbital_values_with_momentum(frame.basis, points, frame.cutoff))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip();
            (orbitals, Some(momentum))
        } else {
            let orbitals = frame
                .cells
                .iter()
                .map(|cell| cell.orbital_values(frame.basis, points, frame.cutoff))
                .collect::<Result<_>>()?;
            (orbitals, None)
        };
        Ok(Self { orbitals, momentum })
    }

    fn block(&self, begin: usize, count: usize) -> Self {
        let rows = |a: &BlochMatrix| {
            let first = begin * a.cols();
            let last = first + count * a.cols();
            let matrix = |m: &crate::linalg::Matrix| {
                crate::linalg::Matrix::from_row_major(
                    count,
                    a.cols(),
                    m.as_slice()[first..last].to_vec(),
                )
            };
            BlochMatrix {
                re: matrix(&a.re),
                im: matrix(&a.im),
            }
        };
        Self {
            orbitals: self.orbitals.iter().map(rows).collect(),
            momentum: self
                .momentum
                .as_ref()
                .map(|momentum| momentum.iter().map(|p| p.each_ref().map(rows)).collect()),
        }
    }

    fn electron_number(&self, weights: &[f64], pauli: &[[BlochMatrix; 4]]) -> f64 {
        let nao = pauli[0][0].rows();
        let nk = pauli.len() as f64;
        let mut number = crate::linalg::AccurateSum::default();
        for (point, weight) in weights.iter().enumerate() {
            let mut charge = 0.0;
            for (p, phi) in pauli.iter().zip(&self.orbitals) {
                for a in 0..nao {
                    for b in 0..nao {
                        charge += (z(&p[0], a, b) * z(phi, point, a) * z(phi, point, b).conj()).re;
                    }
                }
            }
            number.add(weight * charge / nk);
        }
        number.value()
    }
}

struct Grid {
    weights: Vec<f64>,
    points: Vec<Vec3>,
    cache: Option<Collocation>,
}

impl Grid {
    fn new(frame: &GridFrame<'_>, order: usize, sites: Option<&AtomCenteredGrid>) -> Result<Self> {
        let geometry = &frame.cells[0].geometry;
        let (mut points, weights) = if let Some(sites) = sites {
            sites.quadrature(geometry, order, frame.cutoff)?
        } else {
            tensor_grid(geometry, frame.basis, order, frame.cutoff)
        };
        let bytes_per_entry = if frame.needs_momentum { 64 } else { 16 };
        let matrix_bytes = crate::resources::checked_bytes(
            &[
                points.len(),
                frame.cells.len(),
                frame.basis.nao_sph,
                bytes_per_entry,
            ],
            "magnetic XC collocation",
        )?;
        let budget = crate::resources::remaining_bytes().map_or(MAX_CACHED_COLLOCATION, |left| {
            (left / 2).min(MAX_CACHED_COLLOCATION)
        });
        let cache = if matrix_bytes <= budget {
            let cached = Collocation::new(frame, &points)?;
            points = Vec::new();
            Some(cached)
        } else {
            None
        };
        Ok(Self {
            weights,
            points,
            cache,
        })
    }

    fn evaluate(
        &self,
        frame: &GridFrame<'_>,
        functional: &ResolvedFunctional,
        pauli: &[[BlochMatrix; 4]],
        tolerance: f64,
    ) -> Result<Evaluation> {
        if let Some(cache) = self.cache.as_ref().filter(|_| !frame.needs_momentum) {
            let orbitals = cache.orbitals.iter().collect::<Vec<_>>();
            let (energy, potential, parts) = crate::periodic::spinor_scf::noncollinear_lda_values(
                functional,
                &self.weights,
                &orbitals,
                pauli,
            )?;
            return Ok(Evaluation {
                energy,
                potential,
                parts,
                number: cache.electron_number(&self.weights, pauli),
                ..Default::default()
            });
        }
        let mut energy = crate::linalg::AccurateSum::default();
        let mut number = crate::linalg::AccurateSum::default();
        let mut parts = vec![crate::linalg::AccurateSum::default(); functional.parts.len()];
        let nao = pauli[0][0].rows();
        let mut potential = vec![BlochMatrix::zeros(2 * nao, 2 * nao); pauli.len()];
        let mut angular = SpinorXcDiagnostics {
            angular_order: 0,
            angular_energy_defect: 0.0,
            angular_operator_defect: 0.0,
            angular_refinement_history: Vec::new(),
            hermiticity_defect: 0.0,
            component_energy_defect: 0.0,
            converged: true,
        };
        let mut history = BTreeMap::<usize, [f64; 2]>::new();
        // Large grids recompute London values by block. The points and
        // weights are identical to those used by the cached path.
        let block_tolerance = tolerance / self.weights.len().div_ceil(XC_GRID_BLOCK).max(1) as f64;
        for (block, weights) in self.weights.chunks(XC_GRID_BLOCK).enumerate() {
            let begin = block * XC_GRID_BLOCK;
            let collocation = if let Some(cache) = &self.cache {
                cache.block(begin, weights.len())
            } else {
                Collocation::new(frame, &self.points[begin..begin + weights.len()])?
            };
            let phi = collocation.orbitals.iter().collect::<Vec<_>>();
            let (e, v, terms) = if let Some(momentum) = &collocation.momentum {
                let p = momentum.iter().collect::<Vec<_>>();
                let (e, v, terms, diagnostics) = spinor_semilocal_grid_converged(
                    functional,
                    weights,
                    &phi,
                    &p,
                    pauli,
                    block_tolerance,
                )?;
                angular.angular_order = angular.angular_order.max(diagnostics.angular_order);
                angular.angular_energy_defect += diagnostics.angular_energy_defect;
                angular.angular_operator_defect += diagnostics.angular_operator_defect;
                angular.hermiticity_defect += diagnostics.hermiticity_defect;
                angular.component_energy_defect += diagnostics.component_energy_defect;
                angular.converged &= diagnostics.converged;
                for [order, energy, operator] in diagnostics.angular_refinement_history {
                    let row = history.entry(order as usize).or_default();
                    row[0] += energy;
                    row[1] += operator;
                }
                (e, v, terms)
            } else {
                crate::periodic::spinor_scf::noncollinear_lda_values(
                    functional, weights, &phi, pauli,
                )?
            };
            number.add(collocation.electron_number(weights, pauli));
            energy.add(e);
            for (total, term) in parts.iter_mut().zip(terms) {
                total.add(term);
            }
            for (total, block) in potential.iter_mut().zip(v) {
                total.add_scaled(&block, 1.0);
            }
        }
        angular.angular_refinement_history = history
            .into_iter()
            .map(|(order, [e, v])| [order as f64, e, v])
            .collect();
        let energy = energy.value();
        let terms = parts.into_iter().map(|v| v.value()).collect::<Vec<_>>();
        angular.component_energy_defect = angular
            .component_energy_defect
            .max((energy - terms.iter().sum::<f64>()).abs());
        angular.hermiticity_defect = angular.hermiticity_defect.max(
            potential
                .iter()
                .map(BlochMatrix::hermiticity_error)
                .fold(0.0, f64::max),
        );
        angular.converged &=
            angular.component_energy_defect <= tolerance && angular.hermiticity_defect <= tolerance;
        Ok(Evaluation {
            energy,
            potential,
            parts: terms,
            angular: frame.needs_momentum.then_some(angular),
            number: number.value(),
            spatial: [0.0; 3],
        })
    }
}

#[derive(Default)]
pub(super) struct Evaluation {
    pub(super) energy: f64,
    pub(super) potential: Vec<BlochMatrix>,
    pub(super) parts: Vec<f64>,
    pub(super) angular: Option<SpinorXcDiagnostics>,
    number: f64,
    pub(super) spatial: [f64; 3],
}

pub(super) struct Cache {
    pub(super) order: usize,
    grids: Option<[Grid; 2]>,
    atom_centered: Option<AtomCenteredGrid>,
    last_spatial: Option<(Vec<BlochMatrix>, [f64; 3])>,
    /// Every row retains [fine order, energy, operator, number defect].
    pub(super) measurements: Vec<[f64; 4]>,
}

impl Cache {
    pub(super) fn new(
        functional: &ResolvedFunctional,
        order: usize,
        molecule: &Molecule,
    ) -> Option<Mutex<Self>> {
        (!functional.parts.is_empty()).then(|| {
            Mutex::new(Self {
                order,
                grids: None,
                atom_centered: (functional.family == Family::MetaGga).then(|| AtomCenteredGrid {
                    molecule: molecule.clone(),
                    nuclei: molecule.len(),
                }),
                last_spatial: None,
                measurements: Vec::new(),
            })
        })
    }

    pub(super) fn retained_spatial(&self, density: &[BlochMatrix]) -> Option<[f64; 3]> {
        let (previous, spatial) = self.last_spatial.as_ref()?;
        (previous.len() == density.len()
            && previous.iter().zip(density).all(|(a, b)| {
                a.rows() == b.rows()
                    && a.cols() == b.cols()
                    && a.re.as_slice() == b.re.as_slice()
                    && a.im.as_slice() == b.im.as_slice()
            }))
        .then_some(*spatial)
    }

    pub(super) fn evaluate(
        &mut self,
        cells: &[&MagneticCell],
        basis: &MolBasis,
        functional: &ResolvedFunctional,
        options: &MagneticScfOptions,
        density: &[BlochMatrix],
        overlap: &[BlochMatrix],
    ) -> Result<Evaluation> {
        self.last_spatial = None;
        let tolerance = options.operator_tolerance;
        let frame = GridFrame {
            cells,
            basis,
            cutoff: options.basis_cutoff.max(options.grid_cutoff),
            needs_momentum: functional.family != Family::Lda,
        };
        if self.grids.is_none() {
            if let Some(sites) = &mut self.atom_centered {
                sites.add_pair_centers(&cells[0].geometry, basis, options)?;
            }
        }
        let atom_centered = self.atom_centered.as_ref();
        let pair_centers = atom_centered.is_some_and(AtomCenteredGrid::has_pair_centers);
        let build = |order| Grid::new(&frame, order, atom_centered);
        if self.grids.is_none() {
            self.grids = Some([build(self.order - 8)?, build(self.order)?]);
        }
        let grids = self.grids.as_mut().expect("magnetic XC grids initialized");
        let pauli = density
            .iter()
            .map(pauli_components)
            .collect::<Result<Vec<_>>>()?;
        let exact_number = density
            .iter()
            .zip(overlap)
            .map(|(p, s)| p.trace_product(s) / density.len() as f64)
            .sum::<f64>();
        let mut coarse = grids[0].evaluate(&frame, functional, &pauli, tolerance)?;
        loop {
            let mut fine = grids[1].evaluate(&frame, functional, &pauli, tolerance)?;
            fine.spatial = [
                (fine.energy - coarse.energy).abs(),
                fine.potential
                    .iter()
                    .zip(&coarse.potential)
                    .map(|(a, b)| gap(a, b))
                    .fold(0.0_f64, f64::max),
                (fine.number - exact_number).abs(),
            ];
            self.measurements.push([
                self.order as f64,
                fine.spatial[0],
                fine.spatial[1],
                fine.spatial[2],
            ]);
            if fine.spatial.iter().all(|d| *d <= tolerance) || self.order == MAX_GRID_ORDER {
                // A spatial comparison needs angular convergence on both rules.
                if fine.angular.as_ref().is_some_and(|d| d.converged)
                    && coarse.angular.as_ref().is_some_and(|d| !d.converged)
                {
                    fine.angular = coarse.angular;
                }
                let bytes = density.iter().fold(0usize, |sum, p| {
                    sum.saturating_add(p.re.as_slice().len().saturating_mul(16))
                });
                if crate::resources::remaining_bytes().is_none_or(|left| bytes <= left) {
                    self.last_spatial = Some((density.to_vec(), fine.spatial));
                }
                return Ok(fine);
            }
            let order = if pair_centers {
                // Limit intermediate whole-cell evaluations before the
                // two high-resolution radial and angular rules.
                if self.order < PAIR_CHECK_ORDER {
                    (self.order + (self.order / 2).max(8)).min(PAIR_CHECK_ORDER)
                } else {
                    MAX_GRID_ORDER
                }
            } else {
                (self.order + 8).min(MAX_GRID_ORDER)
            };
            let next = build(order)?;
            grids.swap(0, 1);
            grids[1] = next;
            self.order = order;
            coarse = fine;
        }
    }
}
