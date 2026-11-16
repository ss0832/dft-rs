// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Energy derivatives of a periodic cell along Gamma-point displacements:
//! every atom moves together with all of its lattice images.
//!
//! The cell is sampled at Gamma (a k mesh is sampled through its supercell),
//! where the Bloch sums are real and the cell is a closed problem in a real
//! basis. The integral series of [`crate::periodic::path_series`] then take
//! the place of the molecular integrals in the transported-frame recursion
//! of the parent module, with the response operator of the cell itself. The
//! reference is converged on the functional of those series, whose lattice
//! sums are partitioned as in the self-consistent field of the cell and
//! truncated by their own tolerances, so that the path starts from an exact
//! stationary point; the difference of its energy from the supplied
//! self-consistent field is reported.

#[path = "lattice_current.rs"]
mod current;
#[path = "lattice_volume.rs"]
mod volume;
pub use volume::{
    lattice_volume_gruneisen, lattice_volume_gruneisen_with_dispersion, LatticeVolumeGruneisen,
};

use super::geometry::GeometryTaylor;
use super::reduced::{contracted_with, ContractedForceConstants, ReducedTaylor};
use super::{add, density_path, matrix_product, metric_transport, NuclearEnergyCoefficient};
use crate::linalg::{Matrix, Orthogonalizer};
use crate::periodic::dispersion::{PeriodicDispersion, PeriodicDispersionPath};
use crate::periodic::elastic::translation_free_inverse;
use crate::periodic::grid::PeriodicGrid;
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::path_series::{
    lattice_series, momentum_matrices, LatticePath, LatticeRequest, ScalarProbe,
};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::Cell;
use crate::response::complex::Complex;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::mixed::polarized_mixed;
use crate::scf::diis::Diis;
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};

/// Largest element of the orthonormalized commutator of the reference.
const REFERENCE_TOLERANCE: f64 = 1e-10;
const REFERENCE_ITERATIONS: usize = 200;

/// The cell and what its series need besides the direction.
pub(super) struct Setup<'a> {
    pub(super) cell: &'a Cell,
    pub(super) functional: &'a ResolvedFunctional,
    grid: Option<PeriodicGrid>,
    grid_shell: i64,
    ao_cutoff: f64,
    exchange: Option<f64>,
    long_range: Option<(f64, f64)>,
    /// Uniform electric field at the start of every path, in the frame of
    /// the reference lattice; `Some` keeps the field coupling of the series
    /// even when it vanishes.
    static_field: Option<[f64; 3]>,
    fixed_cartesian_bias: bool,
    /// Scalar probes `cos(G . r)` (`false`) and `sin(G . r)` (`true`) whose
    /// amplitudes are further coordinates of the paths.
    probes: Vec<([i64; 3], bool)>,
    nonlocal: Option<crate::periodic::nonlocal::PeriodicVv10Options>,
}

impl<'a> Setup<'a> {
    pub(super) fn new(
        cell: &'a Cell,
        functional: &'a ResolvedFunctional,
        result: &PeriodicScfResult,
        options: &PeriodicScfOptions,
    ) -> Result<Self> {
        Self::new_with_dispersion(cell, functional, result, options, None)
    }

    fn new_with_dispersion(
        cell: &'a Cell,
        functional: &'a ResolvedFunctional,
        result: &PeriodicScfResult,
        options: &PeriodicScfOptions,
        dispersion: Option<&PeriodicDispersion>,
    ) -> Result<Self> {
        if result.k_points.len() != 1
            || !matches!(options.occupation, OccupationScheme::Insulator)
            || !result.converged
        {
            return Err(DftError::InvalidInput(
                "the path response needs a converged insulating reference".into(),
            ));
        }
        Self::for_cell_with_dispersion(cell, functional, options, dispersion)
    }

    /// The cell without a self-consistent field of its own.
    pub(super) fn for_cell(
        cell: &'a Cell,
        functional: &'a ResolvedFunctional,
        options: &PeriodicScfOptions,
    ) -> Result<Self> {
        Self::for_cell_with_dispersion(cell, functional, options, None)
    }

    fn for_cell_with_dispersion(
        cell: &'a Cell,
        functional: &'a ResolvedFunctional,
        options: &PeriodicScfOptions,
        dispersion: Option<&PeriodicDispersion>,
    ) -> Result<Self> {
        if options.k_mesh != [1, 1, 1] {
            return Err(DftError::InvalidInput(
                "the displacement path response needs a converged insulating reference at the \
                 Gamma point of the cell; a k mesh is sampled through its supercell"
                    .into(),
            ));
        }
        if options.skala.is_some()
            || functional.required_nonlocal_kind().is_some_and(|kind| {
                !kind.is_supplied(
                    options.vv10.is_some(),
                    dispersion.is_some_and(|model| model.supplies(kind)),
                )
            })
        {
            return Err(DftError::Unsupported {
                feature: "displacement path response of a cell with a learned functional or \
                          nonlocal correlation"
                    .into(),
                missing: "the moving-geometry series of these terms in a periodic cell".into(),
            });
        }
        let grid = if functional.parts.is_empty() && options.vv10.is_none() {
            None
        } else {
            Some(PeriodicGrid::build(
                cell.molecule(),
                cell.lattice(),
                &options.grid,
                options.grid_shell,
            )?)
        };
        let cutoff = options.exchange_cutoff.unwrap_or(cell.overlap_cutoff());
        Ok(Self {
            cell,
            functional,
            grid,
            grid_shell: options.grid_shell,
            ao_cutoff: options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff()),
            exchange: (functional.hybrid.cx != 0.0).then_some(cutoff),
            long_range: (functional.hybrid.clr != 0.0).then_some((functional.hybrid.omega, cutoff)),
            static_field: None,
            fixed_cartesian_bias: false,
            probes: Vec::new(),
            nonlocal: options.vv10,
        })
    }

    pub(super) fn reduced_field(&self, path: &LatticePath, order: usize) -> Option<[Vec<f64>; 3]> {
        // Field perturbations use the reference lattice. A Cartesian bias
        // separately follows the deformed lattice vectors in this projection.
        let coupled = self.static_field.is_some() || !path.field.is_empty();
        coupled.then(|| {
            let start = self.static_field.unwrap_or([0.0; 3]);
            std::array::from_fn(|a| {
                let vector = self.cell.lattice().a[a];
                let project = |e: &[f64; 3]| {
                    (e[0] * vector.x + e[1] * vector.y + e[2] * vector.z)
                        / (2.0 * std::f64::consts::PI)
                };
                let mut series = vec![0.0; order + 1];
                series[0] = project(&start);
                if self.fixed_cartesian_bias {
                    let vector = vector.to_array();
                    for (n, deformation) in path.deformation.iter().enumerate().take(order) {
                        let shifted = deformation
                            .map(|row| row.iter().zip(vector).map(|(d, a)| d * a).sum::<f64>());
                        series[n + 1] = start.iter().zip(shifted).map(|(e, a)| e * a).sum::<f64>()
                            / (2.0 * std::f64::consts::PI);
                    }
                }
                for (n, step) in path.field.iter().enumerate().take(order) {
                    series[n + 1] += project(step);
                }
                series
            })
        })
    }

    pub(super) fn set_static_field(&mut self, field: [f64; 3]) -> Result<()> {
        if field.iter().any(|value| !value.is_finite()) {
            return Err(DftError::InvalidInput(
                "the reference electric field must be finite".into(),
            ));
        }
        if self.cell.net_charge().abs() > 1e-10 && field.iter().any(|&x| x != 0.0) {
            return Err(DftError::Unsupported {
                feature: "charged periodic lattice path under a Berry electric field".into(),
                missing: "a validated charged-cell polarization and compensation convention".into(),
            });
        }
        self.static_field = Some(field);
        Ok(())
    }

    pub(super) fn geometry(&self, path: &LatticePath, order: usize) -> Result<GeometryTaylor> {
        let series = lattice_series(&LatticeRequest {
            cell: self.cell,
            path,
            order,
            grid: self
                .grid
                .as_ref()
                .map(|grid| (grid, self.grid_shell, self.ao_cutoff)),
            exchange: self.exchange,
            long_range: self.long_range,
        })?;
        let ionic_charge = self
            .cell
            .basis
            .effective_charges(&self.cell.molecule)
            .iter()
            .sum::<f64>();
        let electron_charge_ratio = if ionic_charge == 0.0 {
            0.0
        } else {
            1.0 - self.cell.molecule.charge as f64 / ionic_charge
        };
        let mut geometry = GeometryTaylor::from_lattice(
            series,
            self.reduced_field(path, order),
            electron_charge_ratio,
        );
        if let (Some(options), Some(grid)) = (self.nonlocal, self.grid.as_ref()) {
            let (positions, lattice) = crate::periodic::path_series::nonlocal_geometry_series(
                self.cell, grid, path, order,
            )?;
            geometry = geometry.with_periodic_nonlocal(super::geometry::NonlocalTaylor::periodic(
                crate::periodic::PeriodicGeometry::new(
                    self.cell.lattice().clone(),
                    crate::periodic::features::PeriodicBoundary::new([true; 3]),
                )?,
                grid.points.clone(),
                positions,
                lattice,
                options,
            ));
        }
        Ok(geometry)
    }
}

/// Self-consistent solution of the series functional at the reference
/// geometry.
pub(super) struct Reference {
    pub(super) geometry: GeometryTaylor,
    pub(super) c: [Matrix; 2],
    pub(super) e: [Vec<f64>; 2],
    pub(super) density: [Matrix; 2],
    pub(super) n_occ: [usize; 2],
    unrestricted: bool,
    energy: f64,
    residual: f64,
    iterations: usize,
}

fn projector(c: &Matrix, n_occ: usize) -> Matrix {
    let mut p = Matrix::zeros(c.rows, c.rows);
    for m in 0..c.rows {
        for n in 0..c.rows {
            let mut value = 0.0;
            for i in 0..n_occ {
                value += c[(m, i)] * c[(n, i)];
            }
            p[(m, n)] = value;
        }
    }
    p
}

pub(super) fn reference(
    setup: &Setup,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
) -> Result<Reference> {
    let geometry = setup.geometry(&LatticePath::default(), 0)?;
    let overlap = geometry.overlap[0].clone();
    let orth = Orthogonalizer::canonical(&overlap, options.orth_threshold)?;
    let n_occ = [result.n_alpha, result.n_beta];
    if n_occ.iter().any(|n| *n > orth.n_kept)
        || result
            .density
            .iter()
            .any(|d| d.len() != 1 || d[0].re.rows != overlap.rows || d[0].re.cols != overlap.rows)
    {
        return Err(DftError::InvalidInput(
            "the Gamma-point density of the reference does not fit the basis of the cell".into(),
        ));
    }
    // The Gamma-point density is real.
    let mut density = [
        result.density[0][0].re.clone(),
        result.density[1][0].re.clone(),
    ];
    let mut diis = Diis::new(8);
    let mut residual = f64::INFINITY;
    // Preserve the historical acceptance cap while honoring a tighter
    // requested orbital tolerance for high-order independent validation.
    let tolerance = REFERENCE_TOLERANCE.min(options.conv_tol_grad);
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(DftError::InvalidInput(
            "the integral-series orbital tolerance must be positive and finite".into(),
        ));
    }
    for iteration in 1..=REFERENCE_ITERATIONS {
        let p = [vec![density[0].clone()], vec![density[1].clone()]];
        let evaluation = geometry.evaluate(setup.functional, &p, 0)?;
        let focks = [evaluation.fock[0][0].clone(), evaluation.fock[1][0].clone()];
        residual = diis.push(&focks, &density, &overlap, &orth.x);
        crate::logger::log(&format!(
            "[periodic-field-scf] iter {iteration:>3} E={:.12} residual={residual:.3e}",
            evaluation.energy[0].total()
        ));
        let converged = residual < tolerance;
        let used = if converged {
            focks.to_vec()
        } else {
            diis.extrapolate().unwrap_or_else(|| focks.to_vec())
        };
        let (e0, c0) = orth.solve_roothaan(&used[0])?;
        let (e1, c1) = orth.solve_roothaan(&used[1])?;
        density = [projector(&c0, n_occ[0]), projector(&c1, n_occ[1])];
        if converged {
            return Ok(Reference {
                geometry,
                c: [c0, c1],
                e: [e0, e1],
                density,
                n_occ,
                unrestricted: result.unrestricted,
                energy: evaluation.energy[0].total(),
                residual,
                iterations: iteration,
            });
        }
    }
    Err(DftError::ScfNotConverged {
        iterations: REFERENCE_ITERATIONS,
        error: residual,
    })
}

/// Orbital Hessian of the cell at Gamma on the virtual-occupied amplitudes,
/// `(e_a - e_i) U_ai + [C_v^T G(C_v U C_o^T + C_o U^T C_v^T) C_o]_ai`.
struct Operator<'a> {
    setup: &'a Setup<'a>,
    reference: &'a Reference,
}

impl Operator<'_> {
    fn spins(&self) -> usize {
        if self.reference.unrestricted {
            2
        } else {
            1
        }
    }

    /// Occupied and virtual counts of one spin.
    fn blocks(&self, spin: usize) -> (usize, usize) {
        let no = self.reference.n_occ[spin];
        (no, self.reference.c[spin].cols - no)
    }

    fn size(&self) -> usize {
        (0..self.spins())
            .map(|s| self.blocks(s).0 * self.blocks(s).1)
            .sum()
    }

    fn diagonal(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.size());
        for s in 0..self.spins() {
            let (no, nv) = self.blocks(s);
            let e = &self.reference.e[s];
            for a in 0..nv {
                for i in 0..no {
                    out.push(e[no + a] - e[i]);
                }
            }
        }
        out
    }

    /// AO density change of both spins from the amplitudes `u_ai`.
    fn density(&self, u: &[f64]) -> [Matrix; 2] {
        let mut out: Vec<Matrix> = Vec::with_capacity(2);
        let mut index = 0;
        for s in 0..self.spins() {
            let c = &self.reference.c[s];
            let (no, nv) = self.blocks(s);
            let mut mo = Matrix::zeros(c.cols, c.cols);
            for a in 0..nv {
                for i in 0..no {
                    mo[(no + a, i)] = u[index];
                    mo[(i, no + a)] = u[index];
                    index += 1;
                }
            }
            out.push(c.matmul(&mo).matmul(&c.transpose()));
        }
        if out.len() == 1 {
            out.push(out[0].clone());
        }
        let second = out.pop().expect("two spin densities");
        let first = out.pop().expect("two spin densities");
        [first, second]
    }

    /// First-order Kohn-Sham matrices of a density change.
    fn induced(&self, q: [Matrix; 2]) -> Result<[Matrix; 2]> {
        let p = [
            vec![self.reference.density[0].clone()],
            vec![self.reference.density[1].clone()],
        ];
        let [q0, q1] = q;
        let [mut f0, mut f1] = self.reference.geometry.evaluate_linear(
            self.setup.functional,
            &p,
            &[vec![q0], vec![q1]],
            0,
        )?;
        Ok([f0.swap_remove(0), f1.swap_remove(0)])
    }

    fn apply_real(&self, u: &[f64]) -> Result<Vec<f64>> {
        let induced = self.induced(self.density(u))?;
        let diagonal = self.diagonal();
        let mut out = Vec::with_capacity(u.len());
        for s in 0..self.spins() {
            let c = &self.reference.c[s];
            let (no, nv) = self.blocks(s);
            let mo = c.transpose().matmul(&induced[s]).matmul(c);
            for a in 0..nv {
                for i in 0..no {
                    let index = out.len();
                    out.push(diagonal[index] * u[index] + mo[(no + a, i)]);
                }
            }
        }
        Ok(out)
    }
}

impl LinearOperator for Operator<'_> {
    fn dimension(&self) -> usize {
        self.size()
    }

    fn structure(&self) -> OperatorStructure {
        OperatorStructure::Hermitian
    }

    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| self.apply_real(v))
    }
}

/// Blocks of the order-`n` density that the projector condition fixes from
/// the lower orders.
fn projector_blocks(d: &[Matrix], n: usize, n_occ: usize) -> Matrix {
    let nmo = d[0].rows;
    let mut lower = Matrix::zeros(nmo, nmo);
    for k in 1..n {
        add(&mut lower, &d[k].matmul_seq(&d[n - k]), 1.0);
    }
    let mut next = Matrix::zeros(nmo, nmo);
    for i in 0..nmo {
        for j in 0..nmo {
            if i < n_occ && j < n_occ {
                next[(i, j)] = -lower[(i, j)];
            }
            if i >= n_occ && j >= n_occ {
                next[(i, j)] = lower[(i, j)];
            }
        }
    }
    next
}

/// Energy coefficients of the cell through order `2 solved + 1` along
/// `line`, with response solves through order `solved`.
fn path(
    setup: &Setup,
    reference: &Reference,
    line: &LatticePath,
    solved: usize,
    config: &ResponseSolverConfig,
) -> Result<ReducedTaylor> {
    let total = solved
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| DftError::InvalidInput("reduced Taylor order overflow".into()))?;
    let geometry = setup.geometry(line, total)?;
    path_with_geometry(setup, reference, &geometry, solved, config)
}

/// The common response recursion, with either moving or reused fixed geometry.
fn path_with_geometry(
    setup: &Setup,
    reference: &Reference,
    geometry: &GeometryTaylor,
    solved: usize,
    config: &ResponseSolverConfig,
) -> Result<ReducedTaylor> {
    let total = geometry.overlap.len() - 1;
    let path = density_series(setup, reference, geometry, solved, total, config)?;
    Ok(ReducedTaylor {
        energy: path
            .evaluation
            .energy
            .iter()
            .map(NuclearEnergyCoefficient::total)
            .collect(),
        solved,
        converged: true,
        residuals: path.residuals,
        warnings: vec![
            "lattice sums are partitioned at the reference geometry and the reference is \
             converged on the functional of the integral series"
                .into(),
        ],
    })
}

/// The full solved density, also used by orbital-dependent correlation.
/// A post-SCF energy is not stationary in the parent's density, so it must
/// solve every required order instead of borrowing the parent's 2n+1 rule.
pub(super) struct DensitySeries {
    pub density: [Vec<Matrix>; 2],
    pub projectors: [Vec<Matrix>; 2],
    pub basis: [Vec<Matrix>; 2],
    pub evaluation: super::geometry::Evaluation,
    pub residuals: Vec<f64>,
}

pub(super) fn density_series(
    setup: &Setup,
    reference: &Reference,
    geometry: &GeometryTaylor,
    solved: usize,
    total: usize,
    config: &ResponseSolverConfig,
) -> Result<DensitySeries> {
    let operator = Operator { setup, reference };
    let diagonal = operator.diagonal();
    let spins = operator.spins();
    let n_occ = reference.n_occ;
    let b: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| metric_transport(&reference.c[s], &geometry.overlap, total));
    let mut d: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        let nmo = reference.c[s].cols;
        let mut p = Matrix::zeros(nmo, nmo);
        for i in 0..n_occ[s] {
            p[(i, i)] = 1.0;
        }
        vec![p]
    });
    let mut f: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        let nmo = reference.c[s].cols;
        let mut m = Matrix::zeros(nmo, nmo);
        for (i, &e) in reference.e[s].iter().enumerate() {
            m[(i, i)] = e;
        }
        vec![m]
    });
    let mut residuals = Vec::new();
    for n in 1..=solved {
        for s in 0..2 {
            let next = projector_blocks(&d[s], n, n_occ[s]);
            d[s].push(next);
        }
        let p: [Vec<Matrix>; 2] = std::array::from_fn(|s| density_path(&b[s], &d[s], n));
        let evaluation = geometry.evaluate(setup.functional, &p, n)?;
        let mut fn_mo: [Matrix; 2] = std::array::from_fn(|s| {
            let bt: Vec<_> = b[s][..=n].iter().map(Matrix::transpose).collect();
            matrix_product(&matrix_product(&bt, &evaluation.fock[s], n), &b[s], n)[n].clone()
        });
        let mut source = Vec::with_capacity(diagonal.len());
        for s in 0..spins {
            let nmo = reference.c[s].cols;
            let mut comm = Matrix::zeros(nmo, nmo);
            for k in 1..n {
                add(&mut comm, &f[s][k].matmul_seq(&d[s][n - k]), 1.0);
                add(&mut comm, &d[s][n - k].matmul_seq(&f[s][k]), -1.0);
            }
            let no = n_occ[s];
            for a in 0..nmo - no {
                for i in 0..no {
                    source.push(Complex::new(
                        -(fn_mo[s][(no + a, i)] + comm[(no + a, i)]),
                        0.0,
                    ));
                }
            }
        }
        let solution = krylov::solve(&operator, &source, Some(&diagonal), config)?;
        let u: Vec<f64> = solution.amplitudes.iter().map(|z| z.re).collect();
        let induced = operator.induced(operator.density(&u))?;
        let mut index = 0;
        for s in 0..spins {
            let no = n_occ[s];
            for a in 0..reference.c[s].cols - no {
                for i in 0..no {
                    d[s][n][(no + a, i)] = u[index];
                    d[s][n][(i, no + a)] = u[index];
                    index += 1;
                }
            }
        }
        if spins == 1 {
            d[1][n] = d[0][n].clone();
        }
        for s in 0..2 {
            add(
                &mut fn_mo[s],
                &reference.c[s]
                    .transpose()
                    .matmul_seq(&induced[s])
                    .matmul_seq(&reference.c[s]),
                1.0,
            );
            f[s].push(fn_mo[s].clone());
        }
        residuals.push(solution.relative_residual);
    }
    // Continuation by the projector condition alone.
    for s in 0..2 {
        for n in solved + 1..=total {
            let next = projector_blocks(&d[s], n, n_occ[s]);
            d[s].push(next);
        }
    }
    let p: [Vec<Matrix>; 2] = std::array::from_fn(|s| density_path(&b[s], &d[s], total));
    let evaluation = geometry.evaluate(setup.functional, &p, total)?;
    Ok(DensitySeries {
        density: p,
        projectors: d,
        basis: b,
        evaluation,
        residuals,
    })
}

/// The pattern of the primitive cell on every replica of a supercell whose
/// atoms are ordered replica by replica.
fn replicated(direction: &[[f64; 3]], replicas: usize) -> Vec<[f64; 3]> {
    (0..replicas)
        .flat_map(|_| direction.iter().copied())
        .collect()
}

pub(super) fn primitive_atoms(cell: &Cell, replicas: usize) -> Result<usize> {
    let natoms = cell.molecule().len();
    if replicas == 0 || natoms == 0 || natoms % replicas != 0 {
        return Err(DftError::InvalidInput(
            "the cell holds a whole number of replicas of the primitive cell".into(),
        ));
    }
    Ok(natoms / replicas)
}

/// Contracted force constants of a cell with the diagnostics of the
/// reference of the integral series.
#[derive(Clone, Debug)]
pub struct LatticeForceConstants {
    /// Derivatives of the energy per primitive cell with respect to
    /// Gamma-point displacements of its atoms.
    pub force_constants: ContractedForceConstants,
    /// Energy per primitive cell of the reference of the series, Hartree.
    pub reference_energy: f64,
    /// Its difference from the supplied self-consistent field.
    pub reference_energy_difference: f64,
    /// Largest element of the orthonormalized commutator of that reference.
    pub reference_residual: f64,
    pub reference_iterations: usize,
}

/// Third (one probe) or fourth (two probes) derivatives of the energy per
/// primitive cell, contracted with Gamma-point displacement patterns on the
/// leading indices: `Phi(u)_jk = sum_i u_i d3E / (dx_i dx_j dx_k)` and
/// `Phi(u, v)_kl = sum_ij u_i v_j d4E / (dx_i dx_j dx_k dx_l)`.
///
/// `cell` is sampled at Gamma and holds `replicas` copies of the primitive
/// cell, ordered replica by replica (one for the primitive cell itself); the
/// probes and the result refer to the atoms of the primitive cell, every
/// replica moving with its original.
#[allow(clippy::too_many_arguments)]
pub fn lattice_force_constants_along(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    probes: &[Vec<f64>],
    config: &ResponseSolverConfig,
) -> Result<LatticeForceConstants> {
    lattice_electronic_force_constants_along(
        cell, functional, result, options, replicas, probes, None, config, None,
    )
}

/// Analytic zero-field Gamma nuclear Hessian from the lattice Taylor
/// stationary energy. The cell is the Gamma supercell of the k mesh.
pub fn lattice_harmonic_force_constants(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    config: &ResponseSolverConfig,
) -> Result<LatticeTensor> {
    lattice_electronic_force_constant_tensor(
        cell, functional, result, options, replicas, 2, None, config, None,
    )
}

/// Analytic nuclear response of the stationary Berry electric enthalpy at
/// fixed Cartesian electric bias (Hartree/e/Bohr), including moving basis,
/// metric, field coupling and the field-dependent response operator.
#[allow(clippy::too_many_arguments)]
pub fn lattice_force_constants_along_at_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    probes: &[Vec<f64>],
    reference_field: [f64; 3],
    config: &ResponseSolverConfig,
) -> Result<LatticeForceConstants> {
    lattice_electronic_force_constants_along(
        cell,
        functional,
        result,
        options,
        replicas,
        probes,
        Some(reference_field),
        config,
        None,
    )
}

/// Electronic and ionic derivatives before adding the caller's separately
/// evaluated geometry-only dispersion derivative.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lattice_electronic_force_constants_along(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    probes: &[Vec<f64>],
    reference_field: Option<[f64; 3]>,
    config: &ResponseSolverConfig,
    caller_dispersion: Option<&PeriodicDispersion>,
) -> Result<LatticeForceConstants> {
    config.validate()?;
    let primitive = primitive_atoms(cell, replicas)?;
    let mut setup =
        Setup::new_with_dispersion(cell, functional, result, options, caller_dispersion)?;
    if let Some(field) = reference_field {
        setup.set_static_field(field)?;
    }
    let reference = reference(&setup, result, options)?;
    let scale = 1.0 / replicas as f64;
    let force_constants = contracted_with(3 * primitive, probes, |direction, solved| {
        let mut taylor = path(
            &setup,
            &reference,
            &LatticePath::displacement(&replicated(direction, replicas)),
            solved,
            config,
        )?;
        for value in &mut taylor.energy {
            *value *= scale;
        }
        Ok(taylor)
    })?;
    Ok(LatticeForceConstants {
        force_constants,
        reference_energy: reference.energy * scale,
        reference_energy_difference: (reference.energy - result.energy) * scale,
        reference_residual: reference.residual,
        reference_iterations: reference.iterations,
    })
}

/// Energy coefficients per primitive cell through order `2 solved + 1`
/// along the Gamma-point displacement `direction` of the atoms of the
/// primitive cell, from response solves through order `solved`.
#[allow(clippy::too_many_arguments)]
pub fn lattice_path_energy(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    direction: &[[f64; 3]],
    solved: usize,
    config: &ResponseSolverConfig,
) -> Result<ReducedTaylor> {
    lattice_path_energy_with_field(
        cell, functional, result, options, replicas, direction, solved, None, config,
    )
}

/// Analytic nuclear response of the stationary Berry electric enthalpy at
/// fixed Cartesian electric bias (Hartree/e/Bohr), including moving basis,
/// metric, field coupling and the field-dependent response operator.
#[allow(clippy::too_many_arguments)]
pub fn lattice_path_energy_at_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    direction: &[[f64; 3]],
    solved: usize,
    reference_field: [f64; 3],
    config: &ResponseSolverConfig,
) -> Result<ReducedTaylor> {
    lattice_path_energy_with_field(
        cell,
        functional,
        result,
        options,
        replicas,
        direction,
        solved,
        Some(reference_field),
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn lattice_path_energy_with_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    direction: &[[f64; 3]],
    solved: usize,
    reference_field: Option<[f64; 3]>,
    config: &ResponseSolverConfig,
) -> Result<ReducedTaylor> {
    config.validate()?;
    let primitive = primitive_atoms(cell, replicas)?;
    if direction.len() != primitive {
        return Err(DftError::InvalidInput(
            "a displacement path holds one direction for every atom of the primitive cell".into(),
        ));
    }
    let mut setup = Setup::new(cell, functional, result, options)?;
    if let Some(field) = reference_field {
        setup.set_static_field(field)?;
    }
    let reference = reference(&setup, result, options)?;
    let mut taylor = path(
        &setup,
        &reference,
        &LatticePath::displacement(&replicated(direction, replicas)),
        solved,
        config,
    )?;
    for value in &mut taylor.energy {
        *value /= replicas as f64;
    }
    Ok(taylor)
}

/// The reference of the integral series next to the supplied
/// self-consistent field.
#[derive(Clone, Debug)]
pub struct SeriesReference {
    /// Energy per primitive cell, Hartree.
    pub energy: f64,
    /// Its difference from the supplied self-consistent field.
    pub energy_difference: f64,
    /// Largest element of the orthonormalized commutator.
    pub residual: f64,
    pub iterations: usize,
}

pub(super) fn series_reference(
    reference: &Reference,
    result: &PeriodicScfResult,
    replicas: usize,
) -> SeriesReference {
    let scale = 1.0 / replicas as f64;
    SeriesReference {
        energy: reference.energy * scale,
        energy_difference: (reference.energy - result.energy) * scale,
        residual: reference.residual,
        iterations: reference.iterations,
    }
}

fn unit(n: usize, index: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    v[index] = 1.0;
    v
}

/// Symmetric strain of the Voigt components (xx, yy, zz, yz, xz, xy), whose
/// shear components are twice the tensor components.
fn voigt_strain(c: &[f64]) -> [[f64; 3]; 3] {
    [
        [c[0], 0.5 * c[5], 0.5 * c[4]],
        [0.5 * c[5], c[1], 0.5 * c[3]],
        [0.5 * c[4], 0.5 * c[3], c[2]],
    ]
}

/// Coefficients of `t^1 .. t^order` of `F(t) - 1 = (1 + 2 t eta)^(1/2) - 1`,
/// the symmetric deformation whose Lagrangian strain is exactly `t eta`:
/// `binomial(1/2, n) 2^n eta^n`.
fn lagrangian_deformation(eta: &[[f64; 3]; 3], order: usize) -> Vec<[[f64; 3]; 3]> {
    let mut out = Vec::with_capacity(order);
    let mut power = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut coefficient = 1.0;
    for n in 1..=order {
        let mut next = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    next[a][b] += power[a][c] * eta[c][b];
                }
            }
        }
        power = next;
        coefficient *= 2.0 * (0.5 - (n - 1) as f64) / n as f64;
        out.push(power.map(|row| row.map(|v| coefficient * v)));
    }
    out
}

/// Energy paths of one reference in the generalized coordinates of the
/// primitive cell: the `3 N` Gamma-point displacements of its atoms, the six
/// Voigt components of the Lagrangian strain, then the three components of
/// a uniform electric field in the frame of the reference lattice (the
/// reduced field `E . a_i` does not follow the strain). A point `x` of these
/// coordinates is the configuration `F (R + T + u)` with
/// `F = (1 + 2 eta)^(1/2)`.
struct Paths<'a> {
    setup: &'a Setup<'a>,
    reference: &'a Reference,
    dispersion: Option<&'a PeriodicDispersionPath<'a>>,
    primitive: usize,
    replicas: usize,
    config: &'a ResponseSolverConfig,
    /// Highest requested energy coefficient. The stationary energy through
    /// this order requires orbital response through `order / 2`.
    order: usize,
    /// Energy coefficients per primitive cell of the paths already run.
    cache: Vec<(Vec<f64>, Vec<f64>)>,
    /// Raw freshly checked Krylov residuals of every computed path.
    residuals: Vec<f64>,
    /// Field-only paths share their spatial integrals and Berry matrices.
    field_geometry: Option<GeometryTaylor>,
    /// Integral series of recent moving paths, keyed by the direction
    /// without its field component: the series do not depend on the field,
    /// which enters only through the reduced field. Least recently used
    /// last out, within [`GEOMETRY_CACHE_BYTES`].
    geometries: Vec<(Vec<f64>, GeometryTaylor)>,
    count: usize,
}

/// Memory budget of the geometry series reused between lattice paths.
const GEOMETRY_CACHE_BYTES: usize = 1 << 30;

impl Paths<'_> {
    fn report_reference(
        &self,
        result: &PeriodicScfResult,
    ) -> Result<(SeriesReference, Option<f64>)> {
        let dispersion = self
            .dispersion
            .map(|path| {
                Ok::<_, DftError>(
                    path.energy(&LatticePath::default(), 0)?[0] / self.replicas as f64,
                )
            })
            .transpose()?;
        let mut reference = series_reference(self.reference, result, self.replicas);
        // Both references use the same geometry-only correction, so it cancels in their difference.
        reference.energy += dispersion.unwrap_or_default();
        Ok((reference, dispersion))
    }

    fn dimension(&self) -> usize {
        3 * self.primitive + 9 + self.setup.probes.len()
    }

    fn prepare_field_geometry(&mut self) -> Result<()> {
        if self.field_geometry.is_none() {
            self.field_geometry = Some(
                self.setup
                    .geometry(
                        &LatticePath {
                            field: vec![[0.0; 3]],
                            ..Default::default()
                        },
                        0,
                    )?
                    .extend_fixed_lattice_field(self.order)?,
            );
        }
        Ok(())
    }

    /// Energy coefficients per primitive cell through the highest requested order
    /// along `x(t) = t x`, with the additional displacement `t^2 second`.
    fn energy(&mut self, x: &[f64], second: Option<&[f64]>) -> Result<Vec<f64>> {
        let mut key = x.to_vec();
        if let Some(extra) = second {
            key.extend_from_slice(extra);
        }
        if let Some((_, value)) = self.cache.iter().find(|(k, _)| *k == key) {
            return Ok(value.clone());
        }
        let n = 3 * self.primitive;
        let total = self.order;
        let solved = total / 2;
        let replicas = self.replicas;
        let atoms = |values: &[f64]| -> Vec<[f64; 3]> {
            let cell: Vec<[f64; 3]> = values.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            replicated(&cell, replicas)
        };
        let mut displacement = vec![atoms(&x[..n])];
        if let Some(extra) = second {
            displacement.push(atoms(extra));
        }
        let strain = &x[n..n + 6];
        let field = &x[n + 6..n + 9];
        let deformation = if strain.iter().all(|v| *v == 0.0) {
            Vec::new()
        } else {
            lagrangian_deformation(&voigt_strain(strain), total)
        };
        let potential: Vec<ScalarProbe> = self
            .setup
            .probes
            .iter()
            .enumerate()
            .filter_map(|(index, (wave, sine))| {
                let amplitude = x[n + 9 + index];
                (amplitude != 0.0).then(|| ScalarProbe {
                    wave: *wave,
                    sine: *sine,
                    amplitude: vec![amplitude],
                })
            })
            .collect();
        let line = LatticePath {
            displacement,
            deformation,
            field: if field.iter().all(|v| *v == 0.0) {
                Vec::new()
            } else {
                vec![[field[0], field[1], field[2]]]
            },
            potential,
        };
        let field_only = second.is_none()
            && x[..n + 6].iter().all(|v| *v == 0.0)
            && x[n + 9..].iter().all(|v| *v == 0.0);
        let taylor = if let Some(geometry) = self.field_geometry.as_mut().filter(|_| field_only) {
            let reduced = self
                .setup
                .reduced_field(&line, total)
                .unwrap_or_else(|| std::array::from_fn(|_| vec![0.0; total + 1]));
            geometry.set_reduced_lattice_field(reduced)?;
            path_with_geometry(self.setup, self.reference, geometry, solved, self.config)?
        } else {
            let mut geometry_key = x[..n + 6].to_vec();
            geometry_key.extend_from_slice(&x[n + 9..]);
            if let Some(extra) = second {
                geometry_key.extend_from_slice(extra);
            }
            let reduced = self.setup.reduced_field(&line, total);
            geometry_key.push(f64::from(u8::from(reduced.is_some())));
            let found = self.geometries.iter().position(|(k, _)| *k == geometry_key);
            let entry = match found {
                Some(position) => Some(self.geometries.remove(position)),
                None => {
                    let geometry = self.setup.geometry(&line, total)?;
                    let bytes = geometry.approximate_bytes();
                    if bytes <= GEOMETRY_CACHE_BYTES {
                        let mut held: usize = self
                            .geometries
                            .iter()
                            .map(|(_, g)| g.approximate_bytes())
                            .sum();
                        while held + bytes > GEOMETRY_CACHE_BYTES && !self.geometries.is_empty() {
                            held -= self.geometries.remove(0).1.approximate_bytes();
                        }
                        Some((geometry_key, geometry))
                    } else {
                        let mut geometry = geometry;
                        if let Some(reduced) = reduced.clone() {
                            geometry.set_reduced_lattice_field(reduced)?;
                        }
                        let taylor = path_with_geometry(
                            self.setup,
                            self.reference,
                            &geometry,
                            solved,
                            self.config,
                        )?;
                        self.geometries.clear();
                        return self.finish(key, taylor, &line);
                    }
                }
            };
            let (geometry_key, mut geometry) = entry.expect("a cached or new geometry");
            if let Some(reduced) = reduced {
                geometry.set_reduced_lattice_field(reduced)?;
            }
            let taylor =
                path_with_geometry(self.setup, self.reference, &geometry, solved, self.config);
            self.geometries.push((geometry_key, geometry));
            taylor?
        };
        self.finish(key, taylor, &line)
    }

    /// Record a computed path and return its energy coefficients per
    /// primitive cell.
    fn finish(
        &mut self,
        key: Vec<f64>,
        mut taylor: ReducedTaylor,
        line: &LatticePath,
    ) -> Result<Vec<f64>> {
        // Include dispersion before forming the harmonic and mixed internal
        // derivatives that determine the relaxed-ion strain path.
        if let Some(dispersion) = self.dispersion {
            let coefficients = dispersion.energy(line, taylor.energy.len() - 1)?;
            for (energy, correction) in taylor.energy.iter_mut().zip(coefficients) {
                *energy += correction;
            }
        }
        let scale = 1.0 / self.replicas as f64;
        self.residuals.extend(taylor.residuals);
        let value: Vec<f64> = taylor.energy.iter().map(|e| e * scale).collect();
        self.count += 1;
        self.cache.push((key, value.clone()));
        Ok(value)
    }

    /// Mixed derivative of the energy per primitive cell along generalized
    /// directions, by the polarization identity of the path coefficients.
    fn mixed(&mut self, vectors: &[Vec<f64>]) -> Result<f64> {
        let n = self.dimension();
        let result = polarized_mixed(n, vectors, |direction, total| {
            Ok(vec![self.energy(direction, None)?[total]])
        })?;
        Ok(result.value[0])
    }
}

/// Second derivatives of the series functional that fix the relaxation of
/// the atoms under strain.
struct Harmonic {
    /// Gamma-point force constants `d2E / du du`.
    force_constants: Matrix,
    /// Their pseudo-inverse without the uniform translations.
    inverse: Matrix,
    smallest_optical_eigenvalue: Option<f64>,
    /// `du / d eta_I = -Phi^+ d2E / (du d eta_I)` for the requested Voigt columns.
    relaxation: Matrix,
}

fn harmonic(paths: &mut Paths, strain_columns: usize) -> Result<Harmonic> {
    let n = 3 * paths.primitive;
    let dim = paths.dimension();
    let mut force_constants = Matrix::zeros(n, n);
    for j in 0..n {
        for k in j..n {
            let value = paths.mixed(&[unit(dim, j), unit(dim, k)])?;
            force_constants[(j, k)] = value;
            force_constants[(k, j)] = value;
        }
    }
    // A normal-strain calculation requests only the first three Voigt columns.
    let mut coupling = Matrix::zeros(n, strain_columns);
    for j in 0..n {
        for i in 0..strain_columns {
            coupling[(j, i)] = paths.mixed(&[unit(dim, j), unit(dim, n + i)])?;
        }
    }
    let (inverse, smallest_optical_eigenvalue) =
        translation_free_inverse(&force_constants, paths.primitive)?;
    let mut relaxation = inverse.matmul(&coupling);
    for value in relaxation.as_mut_slice() {
        *value = -*value;
    }
    Ok(Harmonic {
        force_constants,
        inverse,
        smallest_optical_eigenvalue,
        relaxation,
    })
}

/// A fully symmetric tensor of `order` indices over `n` coordinates, laid
/// out with the last index fastest, from its entries with non-decreasing
/// indices.
fn symmetric_tensor(
    n: usize,
    order: usize,
    mut entry: impl FnMut(&[usize]) -> Result<f64>,
) -> Result<Vec<f64>> {
    let size = (0..order).fold(1_usize, |acc, _| acc * n);
    let digits = |mut flat: usize| -> Vec<usize> {
        let mut index = vec![0; order];
        for slot in index.iter_mut().rev() {
            *slot = flat % n;
            flat /= n;
        }
        index
    };
    let mut tensor = vec![0.0; size];
    for flat in 0..size {
        let index = digits(flat);
        if index.windows(2).all(|w| w[0] <= w[1]) {
            tensor[flat] = entry(&index)?;
        }
    }
    for flat in 0..size {
        let mut index = digits(flat);
        index.sort_unstable();
        tensor[flat] = tensor[index.iter().fold(0, |acc, i| acc * n + i)];
    }
    Ok(tensor)
}

/// All derivatives of one order of the energy per primitive cell with
/// respect to Gamma-point displacements of its atoms.
#[derive(Clone, Debug)]
pub struct LatticeTensor {
    /// Fresh relative residuals of every solved response path.
    pub response_residuals: Vec<f64>,
    /// `d^order E / (dx_i dx_j ...)`, last index fastest, Hartree/Bohr^order.
    pub tensor: Vec<f64>,
    /// Number of Cartesian coordinates of the primitive cell.
    pub dimension: usize,
    pub order: usize,
    /// Number of paths that were run.
    pub paths: usize,
    pub reference: SeriesReference,
}

/// The complete cubic (`order` 3) or quartic (`order` 4) force constants of
/// a cell for Gamma-point displacements: every atom moves with all its
/// lattice images. `cell` is sampled at Gamma and holds `replicas` copies
/// of the primitive cell, ordered replica by replica.
#[allow(clippy::too_many_arguments)]
pub fn lattice_force_constant_tensor(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<LatticeTensor> {
    if !(3..=4).contains(&order) {
        return Err(DftError::InvalidInput(
            "the force-constant tensor of a cell has order three or four".into(),
        ));
    }
    lattice_electronic_force_constant_tensor(
        cell, functional, result, options, replicas, order, None, config, None,
    )
}

/// Analytic nuclear response of the stationary Berry electric enthalpy at
/// fixed Cartesian electric bias (Hartree/e/Bohr), including moving basis,
/// metric, field coupling and the field-dependent response operator.
#[allow(clippy::too_many_arguments)]
pub fn lattice_force_constant_tensor_at_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    order: usize,
    reference_field: [f64; 3],
    config: &ResponseSolverConfig,
) -> Result<LatticeTensor> {
    lattice_electronic_force_constant_tensor(
        cell,
        functional,
        result,
        options,
        replicas,
        order,
        Some(reference_field),
        config,
        None,
    )
}

/// Electronic and ionic tensor before adding the caller's separately
/// evaluated geometry-only dispersion tensor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lattice_electronic_force_constant_tensor(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    order: usize,
    reference_field: Option<[f64; 3]>,
    config: &ResponseSolverConfig,
    caller_dispersion: Option<&PeriodicDispersion>,
) -> Result<LatticeTensor> {
    if !(2..=4).contains(&order) {
        return Err(DftError::InvalidInput(
            "the force-constant tensor of a cell has order two, three or four".into(),
        ));
    }
    config.validate()?;
    let primitive = primitive_atoms(cell, replicas)?;
    let mut setup =
        Setup::new_with_dispersion(cell, functional, result, options, caller_dispersion)?;
    if let Some(field) = reference_field {
        setup.set_static_field(field)?;
    }
    let reference = reference(&setup, result, options)?;
    let mut paths = Paths {
        setup: &setup,
        reference: &reference,
        dispersion: None,
        primitive,
        replicas,
        config,
        order,
        cache: Vec::new(),
        geometries: Vec::new(),
        residuals: Vec::new(),
        field_geometry: None,
        count: 0,
    };
    let dim = paths.dimension();
    let n = 3 * primitive;
    let tensor = symmetric_tensor(n, order, |index| {
        paths.mixed(&index.iter().map(|i| unit(dim, *i)).collect::<Vec<_>>())
    })?;
    Ok(LatticeTensor {
        tensor,
        dimension: n,
        order,
        paths: paths.count,
        response_residuals: paths.residuals,
        reference: series_reference(&reference, result, replicas),
    })
}

/// Elastic constants of third or fourth order with the second-order ones
/// of the same series.
#[derive(Clone, Debug)]
pub struct LatticeElastic {
    pub order: usize,
    /// Volume of the primitive cell, Bohr^3.
    pub volume: f64,
    /// Second-order Lagrangian constants `[6 i + j]`, Hartree/Bohr^3.
    pub second_clamped: Vec<f64>,
    pub second_relaxed: Vec<f64>,
    /// Constants of the requested order over the Voigt indices, last index
    /// fastest, Hartree/Bohr^3.
    pub clamped: Vec<f64>,
    pub relaxed: Vec<f64>,
    /// `d u_(A,c) / d eta_I`, rows `3 A + c`, columns `I` (Bohr).
    pub internal_relaxation: Matrix,
    /// Fourth order only: the second-order relaxation
    /// `d2 u_(A,c) / (d eta_I d eta_J)` as `[(6 I + J) 3N + 3 A + c]`
    /// (Bohr), so that the relaxed path is
    /// `u(t) = t u' eta + t^2 u''(eta, eta) / 2`.
    pub second_internal_relaxation: Option<Vec<f64>>,
    /// Smallest force-constant eigenvalue outside the uniform translations.
    pub smallest_optical_eigenvalue: Option<f64>,
    pub paths: usize,
    /// Fresh relative residuals of the directional orbital response solves.
    pub response_residuals: Vec<f64>,
    /// Empirical dispersion energy at the reference geometry, Hartree per primitive cell.
    pub dispersion_energy_per_primitive_cell: Option<f64>,
    pub reference: SeriesReference,
}

/// Lagrangian (Brugger) elastic constants of third (`order` 3) or fourth
/// (`order` 4) order, `C_IJK.. = V^-1 d^n E / (d eta_I d eta_J d eta_K ..)`,
/// at clamped ions and with relaxed ions.
///
/// The energy along `F(t) = (1 + 2 t eta)^(1/2)` is a function of the
/// Lagrangian strain `t eta` alone. With the atoms on the path
/// `u(t) = t u' eta` of their first-order relaxation the third coefficient
/// is the relaxed one, because the energy is stationary in the internal
/// coordinates; the fourth one also needs the second-order relaxation
/// `u''(eta, eta) / 2`, obtained from the mixed third derivatives.
#[allow(clippy::too_many_arguments)]
pub fn lattice_elastic_constants(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<LatticeElastic> {
    lattice_elastic_constants_with_dispersion(
        cell, functional, result, options, replicas, order, config, None,
    )
}

/// Lagrangian elastic constants with a geometry-only periodic dispersion energy.
/// Its derivatives enter the harmonic and cubic equations for internal relaxation.
#[allow(clippy::too_many_arguments)]
pub fn lattice_elastic_constants_with_dispersion(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    order: usize,
    config: &ResponseSolverConfig,
    dispersion: Option<&PeriodicDispersion>,
) -> Result<LatticeElastic> {
    lattice_elastic_constants_at_field(
        cell, functional, result, options, replicas, order, config, dispersion, None,
    )
}

/// Lagrangian elastic constants at fixed Cartesian electric field.
/// The lattice and internal derivatives are of the stationary Berry electric
/// enthalpy and include any geometry-only dispersion correction. With no
/// field the same paths differentiate the stationary zero-field energy.
#[allow(clippy::too_many_arguments)]
pub fn lattice_elastic_constants_at_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    order: usize,
    config: &ResponseSolverConfig,
    dispersion: Option<&PeriodicDispersion>,
    reference_field: Option<[f64; 3]>,
) -> Result<LatticeElastic> {
    if !(3..=4).contains(&order) {
        return Err(DftError::InvalidInput(
            "high-order elastic constants have order three or four".into(),
        ));
    }
    config.validate()?;
    let primitive = primitive_atoms(cell, replicas)?;
    let mut setup = Setup::new_with_dispersion(cell, functional, result, options, dispersion)?;
    if let Some(field) = reference_field {
        setup.set_static_field(field)?;
        setup.fixed_cartesian_bias = true;
    }
    let reference = reference(&setup, result, options)?;
    let dispersion = dispersion
        .map(|model| PeriodicDispersionPath::new(cell, model))
        .transpose()?;
    let mut paths = Paths {
        setup: &setup,
        reference: &reference,
        dispersion: dispersion.as_ref(),
        primitive,
        replicas,
        config,
        order,
        cache: Vec::new(),
        geometries: Vec::new(),
        residuals: Vec::new(),
        field_geometry: None,
        count: 0,
    };
    let n = 3 * primitive;
    // In a one-atom primitive cell, every repeated internal displacement
    // is a uniform translation. The translation-free response space is
    // empty, so there is no first- or second-order internal relaxation.
    let harmonic = if primitive == 1 {
        Harmonic {
            force_constants: Matrix::zeros(n, n),
            inverse: Matrix::zeros(n, n),
            relaxation: Matrix::zeros(n, 6),
            smallest_optical_eigenvalue: None,
        }
    } else {
        harmonic(&mut paths, 6)?
    };
    let dim = paths.dimension();
    let volume = cell.lattice().volume().abs() / replicas as f64;
    let clamped_direction = |i: usize| unit(dim, n + i);
    let relaxed_direction = |i: usize| {
        let mut v = unit(dim, n + i);
        for j in 0..n {
            v[j] = harmonic.relaxation[(j, i)];
        }
        v
    };
    let second_clamped = symmetric_tensor(6, 2, |index| {
        Ok(paths.mixed(&[clamped_direction(index[0]), clamped_direction(index[1])])? / volume)
    })?;
    let second_relaxed = if primitive == 1 {
        second_clamped.clone()
    } else {
        symmetric_tensor(6, 2, |index| {
            Ok(paths.mixed(&[relaxed_direction(index[0]), relaxed_direction(index[1])])? / volume)
        })?
    };
    let clamped = symmetric_tensor(6, order, |index| {
        let vectors: Vec<Vec<f64>> = index.iter().map(|i| clamped_direction(*i)).collect();
        Ok(paths.mixed(&vectors)? / volume)
    })?;
    let mut second_internal_relaxation = None;
    let relaxed = if primitive == 1 {
        second_internal_relaxation = (order == 4).then(|| vec![0.0; 36 * n]);
        clamped.clone()
    } else if order == 3 {
        symmetric_tensor(6, 3, |index| {
            let vectors: Vec<Vec<f64>> = index.iter().map(|i| relaxed_direction(*i)).collect();
            Ok(paths.mixed(&vectors)? / volume)
        })?
    } else {
        // u''(eta_I, eta_J) = -Phi^+ g with g_k = d3E along the relaxed
        // directions of both strains and the displacement k.
        let mut curvature: Vec<Vec<f64>> = vec![vec![0.0; n]; 36];
        for i in 0..6 {
            for j in i..6 {
                let mut g = vec![0.0; n];
                for (k, value) in g.iter_mut().enumerate() {
                    *value =
                        paths.mixed(&[relaxed_direction(i), relaxed_direction(j), unit(dim, k)])?;
                }
                let u: Vec<f64> = (0..n)
                    .map(|a| -(0..n).map(|b| harmonic.inverse[(a, b)] * g[b]).sum::<f64>())
                    .collect();
                curvature[6 * i + j] = u.clone();
                curvature[6 * j + i] = u;
            }
        }
        second_internal_relaxation = Some(curvature.concat());
        symmetric_tensor(6, 4, |index| {
            let vectors: Vec<Vec<f64>> = index.iter().map(|i| unit(6, *i)).collect();
            let mixed = polarized_mixed(6, &vectors, |eta, total| {
                let mut x = vec![0.0; dim];
                let mut second = vec![0.0; n];
                for i in 0..6 {
                    x[n + i] = eta[i];
                    for a in 0..n {
                        x[a] += eta[i] * harmonic.relaxation[(a, i)];
                    }
                    for j in 0..6 {
                        for a in 0..n {
                            second[a] += 0.5 * eta[i] * eta[j] * curvature[6 * i + j][a];
                        }
                    }
                }
                let extra = second.iter().any(|v| *v != 0.0).then_some(&second[..]);
                Ok(vec![paths.energy(&x, extra)?[total]])
            })?;
            Ok(mixed.value[0] / volume)
        })?
    };
    let (series_reference, dispersion_energy_per_primitive_cell) =
        paths.report_reference(result)?;
    Ok(LatticeElastic {
        order,
        volume,
        second_clamped,
        second_relaxed,
        clamped,
        relaxed,
        internal_relaxation: harmonic.relaxation.clone(),
        second_internal_relaxation,
        smallest_optical_eigenvalue: harmonic.smallest_optical_eigenvalue,
        paths: paths.count,
        response_residuals: paths.residuals,
        dispersion_energy_per_primitive_cell,
        reference: series_reference,
    })
}

/// Mode Grueneisen parameters of the Gamma-point phonons.
#[derive(Clone, Debug)]
pub struct LatticeGruneisen {
    /// Angular frequencies of the Gamma modes in atomic units, in
    /// increasing order of their squares (negative: imaginary).
    pub frequencies: Vec<f64>,
    /// Mass-weighted eigenvectors as columns.
    pub modes: Matrix,
    /// `-d ln(omega_nu) / d eta_I` as `[mode][I]`; zero for the acoustic
    /// modes and for strains that were not requested.
    pub strain: Vec<[f64; 6]>,
    /// `-d ln(omega_nu) / d ln V`, the mean of the three normal components.
    pub volume: Vec<f64>,
    /// Modes of uniform translation, which have no parameter.
    pub acoustic: Vec<bool>,
    /// Gamma-point force constants of the series, Hartree/Bohr^2.
    pub force_constants: Matrix,
    /// Whether the three shear components were computed.
    pub shear: bool,
    pub paths: usize,
    pub response_residuals: Vec<f64>,
    /// Geometry-only dispersion energy at the reference, Hartree per primitive cell.
    pub dispersion_energy_per_primitive_cell: Option<f64>,
    pub reference: SeriesReference,
}

/// Mode Grueneisen parameters of the Gamma-point phonons from the third
/// derivatives of the energy: the derivative of the force constants along
/// each strain with the atoms relaxing to first order,
/// `d Phi / d eta_I`, taken to physical displacements `F u`, gives
/// `-d ln(omega) / d eta_I = -e^T D' e / (2 omega^2)`.
#[allow(clippy::too_many_arguments)]
pub fn lattice_mode_gruneisen(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    shear: bool,
    config: &ResponseSolverConfig,
) -> Result<LatticeGruneisen> {
    lattice_mode_gruneisen_with_dispersion(
        cell, functional, result, options, replicas, shear, config, None,
    )
}

/// Gamma mode strain derivatives including a geometry-only dispersion energy.
#[allow(clippy::too_many_arguments)]
pub fn lattice_mode_gruneisen_with_dispersion(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    shear: bool,
    config: &ResponseSolverConfig,
    dispersion: Option<&PeriodicDispersion>,
) -> Result<LatticeGruneisen> {
    config.validate()?;
    let primitive = primitive_atoms(cell, replicas)?;
    let setup = Setup::new_with_dispersion(cell, functional, result, options, dispersion)?;
    let reference = reference(&setup, result, options)?;
    let dispersion = dispersion
        .map(|model| PeriodicDispersionPath::new(cell, model))
        .transpose()?;
    let mut paths = Paths {
        setup: &setup,
        reference: &reference,
        dispersion: dispersion.as_ref(),
        primitive,
        replicas,
        config,
        order: 3,
        cache: Vec::new(),
        geometries: Vec::new(),
        residuals: Vec::new(),
        field_geometry: None,
        count: 0,
    };
    let strain_columns = if shear { 6 } else { 3 };
    let harmonic = harmonic(&mut paths, strain_columns)?;
    let n = 3 * primitive;
    let dim = paths.dimension();
    let phi = &harmonic.force_constants;
    let inverse_root: Vec<f64> = cell.molecule().atoms[..primitive]
        .iter()
        .map(|a| 1.0 / crate::constants::atomic_mass_au(a.z).sqrt())
        .collect();
    let weighted = |m: &Matrix| -> Matrix {
        let mut out = Matrix::zeros(n, n);
        for x in 0..n {
            for y in 0..n {
                out[(x, y)] = m[(x, y)] * inverse_root[x / 3] * inverse_root[y / 3];
            }
        }
        out
    };
    let (values, vectors) = crate::linalg::symmetric_eigen(&weighted(phi))?;
    // A uniform translation is proportional to the square roots of the
    // masses in mass-weighted coordinates.
    let mass: f64 = inverse_root.iter().map(|r| 1.0 / (r * r)).sum();
    let acoustic: Vec<bool> = (0..n)
        .map(|mode| {
            let mut overlap = 0.0;
            for c in 0..3 {
                let t: f64 = (0..primitive)
                    .map(|atom| vectors[(3 * atom + c, mode)] / inverse_root[atom])
                    .sum();
                overlap += t * t / mass;
            }
            overlap > 0.5
        })
        .collect();
    let mut strain = vec![[0.0; 6]; n];
    for i in 0..strain_columns {
        let mut probe = unit(dim, n + i);
        for j in 0..n {
            probe[j] = harmonic.relaxation[(j, i)];
        }
        let mut derivative = Matrix::zeros(n, n);
        for j in 0..n {
            for k in j..n {
                let value = paths.mixed(&[probe.clone(), unit(dim, j), unit(dim, k)])?;
                derivative[(j, k)] = value;
                derivative[(k, j)] = value;
            }
        }
        // Physical displacements F u: Phi_phys = F^-T Phi F^-1.
        let eta = voigt_strain(&unit(6, i));
        for x in 0..n {
            for y in 0..n {
                let mut value = 0.0;
                for c in 0..3 {
                    value += eta[x % 3][c] * phi[(3 * (x / 3) + c, y)]
                        + phi[(x, 3 * (y / 3) + c)] * eta[c][y % 3];
                }
                derivative[(x, y)] -= value;
            }
        }
        let dynamical = weighted(&derivative);
        for mode in 0..n {
            if acoustic[mode] || values[mode] == 0.0 {
                continue;
            }
            let mut expectation = 0.0;
            for x in 0..n {
                for y in 0..n {
                    expectation += vectors[(x, mode)] * dynamical[(x, y)] * vectors[(y, mode)];
                }
            }
            strain[mode][i] = -expectation / (2.0 * values[mode]);
        }
    }
    let volume: Vec<f64> = strain
        .iter()
        .map(|row| (row[0] + row[1] + row[2]) / 3.0)
        .collect();
    let (reference, dispersion_energy_per_primitive_cell) = paths.report_reference(result)?;
    Ok(LatticeGruneisen {
        frequencies: values.iter().map(|l| l.signum() * l.abs().sqrt()).collect(),
        modes: vectors,
        strain,
        volume,
        acoustic,
        force_constants: harmonic.force_constants.clone(),
        shear,
        paths: paths.count,
        response_residuals: paths.residuals,
        dispersion_energy_per_primitive_cell,
        reference,
    })
}

/// Intrinsic permutation symmetry of static electric-enthalpy derivatives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ElectricTensorSymmetry {
    /// Compute independent index combinations and expand their permutations.
    #[default]
    IntrinsicPermutation,
    /// Evaluate all index orders for independent acceptance diagnostics.
    Unprojected,
}

/// Electric susceptibility tensor of a Gamma sampled cell.
#[derive(Clone, Debug)]
pub struct LatticeElectricTensor {
    /// Polarization derivative order: 2 is beta, 3 is gamma.
    pub order: usize,
    pub shape: Vec<usize>,
    /// Flat row-major d^order P_a / dE_b..., without factorial normalization.
    pub tensor: Vec<f64>,
    /// Cartesian bias electric field, Hartree/(e Bohr).
    pub reference_field: [f64; 3],
    pub volume: f64,
    pub symmetry: ElectricTensorSymmetry,
    pub independent_components: usize,
    /// Measured only when every ordering is evaluated without symmetrization.
    pub permutation_residual: Option<f64>,
    pub paths: usize,
    pub response_residuals: Vec<f64>,
    pub reference: SeriesReference,
}

fn electric_tensor(
    paths: &mut Paths,
    order: usize,
    volume: f64,
    symmetry: ElectricTensorSymmetry,
) -> Result<(Vec<f64>, usize)> {
    paths.prepare_field_geometry()?;
    let dimension = paths.dimension();
    let field_offset = 3 * paths.primitive + 6;
    let rank = order + 1;
    let mut tensor = vec![0.0; 3usize.pow(rank as u32)];
    let mut independent = 0;
    for flat in 0..3usize.pow(rank as u32) {
        let mut axes = vec![0; rank];
        let mut rest = flat;
        for axis in axes.iter_mut().rev() {
            *axis = rest % 3;
            rest /= 3;
        }
        if symmetry == ElectricTensorSymmetry::IntrinsicPermutation {
            let mut canonical = axes.clone();
            canonical.sort_unstable();
            let index = canonical.iter().fold(0, |index, axis| 3 * index + axis);
            if index != flat {
                tensor[flat] = tensor[index];
                continue;
            }
        }
        let vectors: Vec<_> = axes
            .iter()
            .map(|&axis| unit(dimension, field_offset + axis))
            .collect();
        tensor[flat] = -paths.mixed(&vectors)? / volume;
        independent += 1;
    }
    Ok((tensor, independent))
}

/// Static clamped-ion beta/gamma from the common Berry-phase electric
/// enthalpy. The second orbital response gives the fourth energy derivative
/// by the 2n+1 rule; finite differences are not used in this calculation.
#[allow(clippy::too_many_arguments)]
pub fn lattice_electric_tensor(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    order: usize,
    symmetry: ElectricTensorSymmetry,
    config: &ResponseSolverConfig,
) -> Result<LatticeElectricTensor> {
    lattice_electric_tensor_at_field(
        cell, functional, result, options, replicas, order, symmetry, [0.0; 3], config,
    )
}

/// Static tensor about a self-consistent finite-field stationary state.
/// The supplied zero-field density initializes the same Berry-enthalpy SCF
/// as [`lattice_finite_field`]; the response equations use that biased
/// functional, including its field-dependent orbital Hessian.
#[allow(clippy::too_many_arguments)]
pub fn lattice_electric_tensor_at_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    order: usize,
    symmetry: ElectricTensorSymmetry,
    reference_field: [f64; 3],
    config: &ResponseSolverConfig,
) -> Result<LatticeElectricTensor> {
    if !(2..=3).contains(&order) {
        return Err(DftError::InvalidInput(
            "periodic electric tensors have polarization derivative order two or three".into(),
        ));
    }
    config.validate()?;
    let primitive = primitive_atoms(cell, replicas)?;
    let mut setup = Setup::new(cell, functional, result, options)?;
    setup.set_static_field(reference_field)?;
    let reference = reference(&setup, result, options)?;
    let mut paths = Paths {
        setup: &setup,
        reference: &reference,
        dispersion: None,
        primitive,
        replicas,
        config,
        order: order + 1,
        cache: Vec::new(),
        geometries: Vec::new(),
        residuals: Vec::new(),
        field_geometry: None,
        count: 0,
    };
    let volume = cell.lattice().volume().abs() / replicas as f64;
    let (tensor, independent_components) = electric_tensor(&mut paths, order, volume, symmetry)?;
    let rank = order + 1;
    let mut permutation_residual = 0.0_f64;
    for flat in 0..if symmetry == ElectricTensorSymmetry::Unprojected {
        tensor.len()
    } else {
        0
    } {
        let mut axes = vec![0; rank];
        let mut rest = flat;
        for axis in axes.iter_mut().rev() {
            *axis = rest % 3;
            rest /= 3;
        }
        for a in 0..rank - 1 {
            let mut other = axes.clone();
            other.swap(a, a + 1);
            let index = other.iter().fold(0, |index, axis| 3 * index + axis);
            permutation_residual = permutation_residual.max((tensor[flat] - tensor[index]).abs());
        }
    }
    Ok(LatticeElectricTensor {
        order,
        shape: vec![3; rank],
        tensor,
        reference_field,
        volume,
        symmetry,
        independent_components,
        permutation_residual: (symmetry == ElectricTensorSymmetry::Unprojected)
            .then_some(permutation_residual),
        paths: paths.count,
        response_residuals: paths.residuals,
        reference: series_reference(&reference, result, replicas),
    })
}

/// Which third derivatives [`lattice_field_response`] evaluates besides the
/// susceptibility.
#[derive(Clone, Copy, Debug, Default)]
pub struct FieldRequest {
    /// Second-order susceptibility.
    pub second_susceptibility: bool,
    /// Born charges and the displacement derivative of the susceptibility.
    pub raman: bool,
    /// Strain derivative of the susceptibility.
    pub strain_susceptibility: bool,
    /// Piezoelectric tensor and its strain derivative.
    pub piezoelectric: bool,
    /// The same derivatives with the atoms following the field and the
    /// strain to first order (relaxed ions).
    pub relaxed: bool,
}

/// Electric-field derivatives of the energy functional
/// `F = E - Omega E . P` per primitive cell at clamped ions, with the
/// polarization of the single-point Berry phase: `P_a = -Omega^-1 dF/dE_a`.
/// Field components refer to the reference lattice (fixed reduced field),
/// so that the strain derivatives are the proper ones.
#[derive(Clone, Debug)]
pub struct LatticeFieldResponse {
    /// Volume of the primitive cell, Bohr^3.
    pub volume: f64,
    /// Cartesian bias electric field, Hartree/(e Bohr).
    pub reference_field: [f64; 3],
    /// `chi_ab = dP_a / dE_b = -Omega^-1 d2F / (dE_a dE_b)`; the electronic
    /// dielectric tensor is `1 + 4 pi chi`.
    pub susceptibility: [[f64; 3]; 3],
    /// `d2P_a / (dE_b dE_c) = -Omega^-1 d3F / dE^3` as `[(3 a + b) 3 + c]`;
    /// the second-order susceptibility `chi2` is half of it.
    pub second_susceptibility: Option<Vec<f64>>,
    /// Born effective charges `Omega dP_a / du_k = -d2F / (dE_a du_k)`,
    /// rows `k = 3 A + c`, columns `a`.
    pub born_charges: Option<Matrix>,
    /// `d chi_ab / du_k = -Omega^-1 d3F / (dE_a dE_b du_k)` as
    /// `[(3 a + b) n + k]`.
    pub raman: Option<Vec<f64>>,
    /// `-Omega^-1 d3F / (dE_a dE_b d eta_I)` as `[(3 a + b) 6 + I]`.
    pub strain_susceptibility: Option<Vec<f64>>,
    /// Proper piezoelectric tensor `-Omega^-1 d2F / (dE_a d eta_I)`.
    pub piezoelectric: Option<[[f64; 6]; 3]>,
    /// `-Omega^-1 d3F / (dE_a d eta_I d eta_J)` as `[(6 a + I) 6 + J]`.
    pub second_piezoelectric: Option<Vec<f64>>,
    /// Relaxed-ion values, when requested.
    pub relaxed: Option<RelaxedFieldResponse>,
    pub paths: usize,
    /// Raw freshly checked Krylov residuals for the evaluated paths.
    pub response_residuals: Vec<f64>,
    pub reference: SeriesReference,
}

/// Electric-field derivatives with relaxed ions. Along a generalized
/// direction `x` the atoms follow `u'(x) = -Phi^+ d2F / (du dx)`; because
/// the functional is stationary in the internal coordinates, the second and
/// third derivatives along the combined directions `(x, u'(x))` are the
/// relaxed ones.
#[derive(Clone, Debug)]
pub struct RelaxedFieldResponse {
    /// `du_k / dE_a = Phi^+ Z*`, rows `k = 3 A + c`, columns `a`.
    pub field_relaxation: Matrix,
    /// `du_k / d eta_I`, rows `k`, columns `I`.
    pub strain_relaxation: Matrix,
    /// Smallest force-constant eigenvalue outside the uniform translations.
    pub smallest_optical_eigenvalue: Option<f64>,
    /// Static susceptibility: electronic plus lattice contribution.
    pub susceptibility: [[f64; 3]; 3],
    pub second_susceptibility: Option<Vec<f64>>,
    pub strain_susceptibility: Option<Vec<f64>>,
    pub piezoelectric: Option<[[f64; 6]; 3]>,
    pub second_piezoelectric: Option<Vec<f64>>,
}

/// Clamped-ion electric-field response of a cell through third order, from
/// first-order responses along combined field, displacement and strain
/// paths of the Gamma-point cell (2n + 1 rule and polarization identity).
#[allow(clippy::too_many_arguments)]
pub fn lattice_field_response(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    request: &FieldRequest,
    config: &ResponseSolverConfig,
) -> Result<LatticeFieldResponse> {
    lattice_field_response_at_field(
        cell, functional, result, options, replicas, request, [0.0; 3], config,
    )
}

/// Clamped-ion derivatives about a self-consistent biased reference.
/// Relaxed-ion formulas require ionic stationarity, which a zero-field
/// geometry generally loses under a nonzero bias.
#[allow(clippy::too_many_arguments)]
pub fn lattice_field_response_at_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    request: &FieldRequest,
    reference_field: [f64; 3],
    config: &ResponseSolverConfig,
) -> Result<LatticeFieldResponse> {
    config.validate()?;
    if request.relaxed && reference_field.iter().any(|value| *value != 0.0) {
        return Err(DftError::Unsupported {
            feature: "relaxed-ion response about a finite electric-field bias".into(),
            missing: "an ionically stationary finite-field reference and its constrained internal-coordinate response".into(),
        });
    }
    let primitive = primitive_atoms(cell, replicas)?;
    let mut setup = Setup::new(cell, functional, result, options)?;
    setup.set_static_field(reference_field)?;
    let reference = reference(&setup, result, options)?;
    let mut paths = Paths {
        setup: &setup,
        reference: &reference,
        dispersion: None,
        primitive,
        replicas,
        config,
        order: if request.second_susceptibility
            || request.raman
            || request.strain_susceptibility
            || request.piezoelectric
        {
            3
        } else {
            2
        },
        cache: Vec::new(),
        geometries: Vec::new(),
        residuals: Vec::new(),
        field_geometry: None,
        count: 0,
    };
    let n = 3 * primitive;
    let dim = paths.dimension();
    let volume = cell.lattice().volume().abs() / replicas as f64;
    let field = |a: usize| unit(dim, n + 6 + a);
    let strain = |i: usize| unit(dim, n + i);
    paths.prepare_field_geometry()?;
    let mut susceptibility = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in a..3 {
            let value = -paths.mixed(&[field(a), field(b)])? / volume;
            susceptibility[a][b] = value;
            susceptibility[b][a] = value;
        }
    }
    let second_susceptibility = if request.second_susceptibility {
        Some(
            electric_tensor(
                &mut paths,
                2,
                volume,
                ElectricTensorSymmetry::IntrinsicPermutation,
            )?
            .0,
        )
    } else {
        None
    };
    let (born_charges, raman) = if request.raman {
        let mut born = Matrix::zeros(n, 3);
        let mut tensor = vec![0.0; 9 * n];
        for k in 0..n {
            for a in 0..3 {
                born[(k, a)] = -paths.mixed(&[field(a), unit(dim, k)])?;
                for b in a..3 {
                    let value = -paths.mixed(&[field(a), field(b), unit(dim, k)])? / volume;
                    tensor[(3 * a + b) * n + k] = value;
                    tensor[(3 * b + a) * n + k] = value;
                }
            }
        }
        (Some(born), Some(tensor))
    } else {
        (None, None)
    };
    let strain_susceptibility = if request.strain_susceptibility {
        let mut tensor = vec![0.0; 54];
        for a in 0..3 {
            for b in a..3 {
                for i in 0..6 {
                    let value = -paths.mixed(&[field(a), field(b), strain(i)])? / volume;
                    tensor[(3 * a + b) * 6 + i] = value;
                    tensor[(3 * b + a) * 6 + i] = value;
                }
            }
        }
        Some(tensor)
    } else {
        None
    };
    let (piezoelectric, second_piezoelectric) = if request.piezoelectric {
        let mut first = [[0.0; 6]; 3];
        let mut second = vec![0.0; 108];
        for a in 0..3 {
            for i in 0..6 {
                first[a][i] = -paths.mixed(&[field(a), strain(i)])? / volume;
                for j in i..6 {
                    let value = -paths.mixed(&[field(a), strain(i), strain(j)])? / volume;
                    second[(6 * a + i) * 6 + j] = value;
                    second[(6 * a + j) * 6 + i] = value;
                }
            }
        }
        (Some(first), Some(second))
    } else {
        (None, None)
    };
    let relaxed = if request.relaxed {
        let harmonic = harmonic(&mut paths, 6)?;
        // du / dE_a = -Phi^+ d2F / (du dE_a).
        let mut coupling = Matrix::zeros(n, 3);
        for k in 0..n {
            for a in 0..3 {
                coupling[(k, a)] = paths.mixed(&[field(a), unit(dim, k)])?;
            }
        }
        let mut field_relaxation = harmonic.inverse.matmul(&coupling);
        for value in field_relaxation.as_mut_slice() {
            *value = -*value;
        }
        let relaxed_field = |a: usize| {
            let mut v = unit(dim, n + 6 + a);
            for j in 0..n {
                v[j] = field_relaxation[(j, a)];
            }
            v
        };
        let relaxed_strain = |i: usize| {
            let mut v = unit(dim, n + i);
            for j in 0..n {
                v[j] = harmonic.relaxation[(j, i)];
            }
            v
        };
        let mut total = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in a..3 {
                let value = -paths.mixed(&[relaxed_field(a), relaxed_field(b)])? / volume;
                total[a][b] = value;
                total[b][a] = value;
            }
        }
        let second = if request.second_susceptibility {
            Some(symmetric_tensor(3, 3, |index| {
                let vectors: Vec<Vec<f64>> = index.iter().map(|a| relaxed_field(*a)).collect();
                Ok(-paths.mixed(&vectors)? / volume)
            })?)
        } else {
            None
        };
        let strained = if request.strain_susceptibility {
            let mut tensor = vec![0.0; 54];
            for a in 0..3 {
                for b in a..3 {
                    for i in 0..6 {
                        let value = -paths.mixed(&[
                            relaxed_field(a),
                            relaxed_field(b),
                            relaxed_strain(i),
                        ])? / volume;
                        tensor[(3 * a + b) * 6 + i] = value;
                        tensor[(3 * b + a) * 6 + i] = value;
                    }
                }
            }
            Some(tensor)
        } else {
            None
        };
        let (first_piezo, second_piezo) = if request.piezoelectric {
            let mut first = [[0.0; 6]; 3];
            let mut second = vec![0.0; 108];
            for a in 0..3 {
                for i in 0..6 {
                    first[a][i] = -paths.mixed(&[relaxed_field(a), relaxed_strain(i)])? / volume;
                    for j in i..6 {
                        let value = -paths.mixed(&[
                            relaxed_field(a),
                            relaxed_strain(i),
                            relaxed_strain(j),
                        ])? / volume;
                        second[(6 * a + i) * 6 + j] = value;
                        second[(6 * a + j) * 6 + i] = value;
                    }
                }
            }
            (Some(first), Some(second))
        } else {
            (None, None)
        };
        Some(RelaxedFieldResponse {
            field_relaxation: field_relaxation.clone(),
            strain_relaxation: harmonic.relaxation.clone(),
            smallest_optical_eigenvalue: harmonic.smallest_optical_eigenvalue,
            susceptibility: total,
            second_susceptibility: second,
            strain_susceptibility: strained,
            piezoelectric: first_piezo,
            second_piezoelectric: second_piezo,
        })
    } else {
        None
    };
    Ok(LatticeFieldResponse {
        volume,
        reference_field,
        susceptibility,
        second_susceptibility,
        born_charges,
        raman,
        strain_susceptibility,
        piezoelectric,
        second_piezoelectric,
        relaxed,
        paths: paths.count,
        response_residuals: paths.residuals,
        reference: series_reference(&reference, result, replicas),
    })
}

/// Self-consistent state of a cell in a finite uniform electric field.
#[derive(Clone, Debug)]
pub struct LatticeFiniteField {
    /// `E_KS - Omega E . P` per primitive cell, Hartree.
    pub energy: f64,
    pub field: [f64; 3],
    /// Dipole per primitive cell from the single-point Berry phase, modulo
    /// the lattice vectors of the primitive cell (atomic units).
    pub dipole: [f64; 3],
    /// Volume of the primitive cell, Bohr^3.
    pub volume: f64,
    /// `-dF/du` on the atoms of the primitive cell, Hartree/Bohr.
    pub forces: Option<Vec<[f64; 3]>>,
    /// Largest element of the orthonormalized commutator.
    pub residual: f64,
    pub iterations: usize,
}

/// Self-consistent field of a cell in the finite uniform electric field
/// `field`, on the functional of the integral series with the Berry-phase
/// field term, started from the supplied zero-field state. The cell is
/// sampled at Gamma and holds `replicas` copies of the primitive cell.
pub fn lattice_finite_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    field: [f64; 3],
    forces: bool,
) -> Result<LatticeFiniteField> {
    lattice_electronic_finite_field(
        cell, functional, result, options, replicas, field, forces, None,
    )
}

/// Stationary electronic and ionic enthalpy before a separately evaluated
/// geometry-only dispersion correction is added by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lattice_electronic_finite_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    field: [f64; 3],
    forces: bool,
    caller_dispersion: Option<&PeriodicDispersion>,
) -> Result<LatticeFiniteField> {
    let primitive = primitive_atoms(cell, replicas)?;
    let mut setup =
        Setup::new_with_dispersion(cell, functional, result, options, caller_dispersion)?;
    setup.set_static_field(field)?;
    let reference = reference(&setup, result, options)?;
    let scale = 1.0 / replicas as f64;
    let phases = reference
        .geometry
        .berry_phases(&reference.density)?
        .ok_or_else(|| {
            DftError::InvalidInput("the field coupling of the cell is missing".into())
        })?;
    let lattice = cell.lattice();
    let mut dipole = [0.0; 3];
    for a in 0..3 {
        let vector = lattice.a[a];
        let weight = -phases[a] / (2.0 * std::f64::consts::PI) * scale;
        dipole[0] += weight * vector.x;
        dipole[1] += weight * vector.y;
        dipole[2] += weight * vector.z;
    }
    let forces = if forces {
        // Energy-weighted densities of the stationary state.
        let weighted: [Matrix; 2] = std::array::from_fn(|s| {
            let c = &reference.c[s];
            let mut w = Matrix::zeros(c.rows, c.rows);
            for i in 0..reference.n_occ[s] {
                let e = reference.e[s][i];
                for m in 0..c.rows {
                    for k in 0..c.rows {
                        w[(m, k)] += e * c[(m, i)] * c[(k, i)];
                    }
                }
            }
            w
        });
        let nao = reference.density[0].rows;
        let mut out = vec![[0.0; 3]; primitive];
        for (atom, force) in out.iter_mut().enumerate() {
            for (axis, value) in force.iter_mut().enumerate() {
                let mut direction = vec![[0.0; 3]; primitive];
                direction[atom][axis] = 1.0;
                let moving = setup.geometry(
                    &LatticePath::displacement(&replicated(&direction, replicas)),
                    1,
                )?;
                let p: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
                    vec![reference.density[s].clone(), Matrix::zeros(nao, nao)]
                });
                let evaluation = moving.evaluate(setup.functional, &p, 1)?;
                let mut derivative = evaluation.energy[1].total();
                for w in &weighted {
                    derivative -= w.frobenius_dot(&moving.overlap[1]);
                }
                *value = -derivative * scale;
            }
        }
        Some(out)
    } else {
        None
    };
    Ok(LatticeFiniteField {
        energy: reference.energy * scale,
        field,
        dipole,
        volume: lattice.volume().abs() * scale,
        forces,
        residual: reference.residual,
        iterations: reference.iterations,
    })
}

/// The reciprocal lattice vector of the cell with the integer components
/// `wave`.
pub(super) fn wave_vector(cell: &Cell, wave: [i64; 3]) -> crate::math::Vec3 {
    let b = cell.lattice().reciprocal();
    b[0] * wave[0] as f64 + b[1] * wave[1] as f64 + b[2] * wave[2] as f64
}

/// Density response of a cell at one wave vector.
#[derive(Clone, Debug)]
pub struct DensityResponse {
    /// Integer multiples of the reciprocal basis vectors of the sampled cell.
    pub wave: [i64; 3],
    /// The wave vector, Bohr^-1.
    pub vector: [f64; 3],
    /// `Omega^-1 d2E / d lambda^2` for the potential `lambda cos(q . r)`.
    pub cosine: f64,
    /// The same for `lambda sin(q . r)`.
    pub sine: f64,
    /// Density response function `chi(q, q)`: their sum.
    pub chi: f64,
    /// `1 + 4 pi chi / q^2`.
    pub inverse_dielectric: f64,
}

/// Density response of a cell at wave vectors commensurate with it.
#[derive(Clone, Debug)]
pub struct LatticeDensityResponse {
    /// Volume of the primitive cell, Bohr^3.
    pub volume: f64,
    pub points: Vec<DensityResponse>,
    pub paths: usize,
    pub reference: SeriesReference,
}

/// Self-consistent density response function `chi(q, q)` and inverse
/// dielectric function `1 + 4 pi chi / q^2` of a cell at reciprocal lattice
/// vectors of the sampled (Gamma-point) cell: the second derivatives of the
/// energy in the amplitudes of the potentials `cos(q . r)` and `sin(q . r)`,
/// `(Omega / 2) [chi(q, q) +- Re chi(q, -q)]`, add up to `Omega chi(q, q)`.
/// The response includes the Hartree and exchange-correlation kernels and
/// the local fields.
#[allow(clippy::too_many_arguments)]
pub fn lattice_density_response(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    waves: &[[i64; 3]],
    config: &ResponseSolverConfig,
) -> Result<LatticeDensityResponse> {
    config.validate()?;
    let primitive = primitive_atoms(cell, replicas)?;
    if waves.is_empty() || waves.iter().any(|w| w.iter().all(|c| *c == 0)) {
        return Err(DftError::InvalidInput(
            "the density response needs wave vectors different from zero".into(),
        ));
    }
    let mut setup = Setup::new(cell, functional, result, options)?;
    for wave in waves {
        setup.probes.push((*wave, false));
        setup.probes.push((*wave, true));
    }
    let reference = reference(&setup, result, options)?;
    let mut paths = Paths {
        setup: &setup,
        reference: &reference,
        dispersion: None,
        primitive,
        replicas,
        config,
        order: 2,
        cache: Vec::new(),
        geometries: Vec::new(),
        residuals: Vec::new(),
        field_geometry: None,
        count: 0,
    };
    let n = 3 * primitive;
    let dim = paths.dimension();
    let volume = cell.lattice().volume().abs() / replicas as f64;
    let mut points = Vec::with_capacity(waves.len());
    for (index, wave) in waves.iter().enumerate() {
        let slot = n + 9 + 2 * index;
        let cosine = paths.mixed(&[unit(dim, slot), unit(dim, slot)])? / volume;
        let sine = paths.mixed(&[unit(dim, slot + 1), unit(dim, slot + 1)])? / volume;
        let vector = wave_vector(cell, *wave);
        let chi = cosine + sine;
        points.push(DensityResponse {
            wave: *wave,
            vector: vector.to_array(),
            cosine,
            sine,
            chi,
            inverse_dielectric: 1.0 + 4.0 * std::f64::consts::PI * chi / vector.norm2(),
        });
    }
    Ok(LatticeDensityResponse {
        volume,
        points,
        paths: paths.count,
        reference: series_reference(&reference, result, replicas),
    })
}

/// Response of the cell to an imaginary one-electron perturbation `i B`
/// with `B` real and antisymmetric: the density changes by `i X`,
/// `X = C_v Y C_o^T - C_o Y^T C_v^T`. Exact exchange and, where
/// supplied, the meta-GGA current kernel act on this density.
struct Imaginary<'a>(Operator<'a>, Option<&'a current::LatticeCurrentKernel<'a>>);

impl Imaginary<'_> {
    fn density(&self, y: &[f64]) -> [Matrix; 2] {
        let reference = self.0.reference;
        let mut out = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        let mut index = 0;
        for s in 0..self.0.spins() {
            let c = &reference.c[s];
            let (no, nv) = self.0.blocks(s);
            let mut mo = Matrix::zeros(c.cols, c.cols);
            for a in 0..nv {
                for i in 0..no {
                    mo[(no + a, i)] = y[index];
                    mo[(i, no + a)] = -y[index];
                    index += 1;
                }
            }
            out[s] = c.matmul(&mo).matmul(&c.transpose());
        }
        if self.0.spins() == 1 {
            out[1] = out[0].clone();
        }
        out
    }

    fn apply_real(&self, y: &[f64]) -> Result<Vec<f64>> {
        let density = self.density(y);
        let induced = if let Some(current) = self.1 {
            let mut fock = self
                .0
                .reference
                .geometry
                .imaginary_exchange(self.0.setup.functional, &density);
            current.add_response(&mut fock, &density, self.0.spins());
            fock
        } else {
            self.0.induced(density)?
        };
        let diagonal = self.0.diagonal();
        let reference = self.0.reference;
        let mut out = Vec::with_capacity(y.len());
        for s in 0..self.0.spins() {
            let c = &reference.c[s];
            let (no, nv) = self.0.blocks(s);
            let mo = c.transpose().matmul(&induced[s]).matmul(c);
            for a in 0..nv {
                for i in 0..no {
                    let index = out.len();
                    out.push(diagonal[index] * y[index] + mo[(no + a, i)]);
                }
            }
        }
        Ok(out)
    }
}

impl LinearOperator for Imaginary<'_> {
    fn dimension(&self) -> usize {
        self.0.size()
    }

    fn structure(&self) -> OperatorStructure {
        OperatorStructure::Hermitian
    }

    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| self.apply_real(v))
    }
}

/// Second-order energy of the cell for the imaginary perturbation `i B`
/// (`B` real antisymmetric, the same for both spins):
/// `sum_spin sum_ai y_ai B_ai` with `(e_a - e_i) y + [B + G(X)]_ai = 0`.
fn imaginary_second_order(
    setup: &Setup,
    reference: &Reference,
    b: &Matrix,
    current: Option<&current::LatticeCurrentKernel<'_>>,
    config: &ResponseSolverConfig,
) -> Result<(f64, f64)> {
    let operator = Imaginary(Operator { setup, reference }, current);
    let spins = operator.0.spins();
    let mut projected = Vec::with_capacity(operator.0.size());
    for s in 0..spins {
        let c = &reference.c[s];
        let (no, nv) = operator.0.blocks(s);
        let mo = c.transpose().matmul(b).matmul(c);
        for a in 0..nv {
            for i in 0..no {
                projected.push(mo[(no + a, i)]);
            }
        }
    }
    if projected.is_empty() {
        return Ok((0.0, 0.0));
    }
    let source: Vec<Complex> = projected.iter().map(|v| Complex::new(-v, 0.0)).collect();
    let solution = krylov::solve(&operator, &source, Some(&operator.0.diagonal()), config)?;
    let energy: f64 = solution
        .amplitudes
        .iter()
        .zip(&projected)
        .map(|(y, b)| y.re * b)
        .sum();
    Ok((
        if spins == 1 { 2.0 * energy } else { energy },
        solution.relative_residual,
    ))
}

/// One solve for the imaginary perturbation `i B` and the contractions
/// `sum_spin sum_ai y_ai C_ai` of its amplitudes with further real
/// antisymmetric matrices `C`: half the mixed second derivatives of the
/// energy in the amplitudes of `i B` and `i C`.
pub(super) fn imaginary_mixed(
    setup: &Setup,
    reference: &Reference,
    b: &Matrix,
    others: &[Matrix],
    config: &ResponseSolverConfig,
) -> Result<Vec<f64>> {
    let operator = Imaginary(Operator { setup, reference }, None);
    let spins = operator.0.spins();
    let project = |m: &Matrix| -> Vec<f64> {
        let mut out = Vec::with_capacity(operator.0.size());
        for s in 0..spins {
            let c = &reference.c[s];
            let (no, nv) = operator.0.blocks(s);
            let mo = c.transpose().matmul(m).matmul(c);
            for a in 0..nv {
                for i in 0..no {
                    out.push(mo[(no + a, i)]);
                }
            }
        }
        out
    };
    let projected = project(b);
    if projected.is_empty() {
        return Ok(vec![0.0; others.len()]);
    }
    let source: Vec<Complex> = projected.iter().map(|v| Complex::new(-v, 0.0)).collect();
    let solution = krylov::solve(&operator, &source, Some(&operator.0.diagonal()), config)?;
    let factor = if spins == 1 { 2.0 } else { 1.0 };
    Ok(others
        .iter()
        .map(|m| {
            factor
                * solution
                    .amplitudes
                    .iter()
                    .zip(project(m))
                    .map(|(y, c)| y.re * c)
                    .sum::<f64>()
        })
        .collect())
}

/// `-(R - R^T) / 2` of `R = sum_k e_k M_k`.
pub(super) fn antisymmetric(parts: [&Matrix; 3], e: [f64; 3]) -> Matrix {
    let n = parts[0].rows;
    let mut r = Matrix::zeros(n, n);
    for k in 0..3 {
        add(&mut r, parts[k], e[k]);
    }
    let transposed = r.transpose();
    let mut out = Matrix::zeros(n, n);
    add(&mut out, &r, -0.5);
    add(&mut out, &transposed, 0.5);
    out
}

/// Orbital magnetic susceptibility of a cell at one wave vector.
#[derive(Clone, Debug)]
pub struct OrbitalSusceptibility {
    pub wave: [i64; 3],
    /// The wave vector, Bohr^-1.
    pub vector: [f64; 3],
    /// Two unit polarizations of the vector potential, transverse to the
    /// wave vector.
    pub polarization: [[f64; 3]; 2],
    /// Directions of the magnetic field, `q x e / |q|`.
    pub field: [[f64; 3]; 2],
    /// `chi(q) = -4 E2 / (Omega q^2)` per polarization, atomic units.
    pub chi: [f64; 2],
    /// Second-order paramagnetic energies per primitive cell and unit
    /// amplitude of the vector potential: the mean of the cosine and sine
    /// waves, and the uniform potential.
    pub paramagnetic: [f64; 2],
    pub uniform: [f64; 2],
    /// Relative orbital residual for each transverse polarization and
    /// its cosine, sine and uniform-potential solves, in that order.
    pub response_residuals: [[f64; 3]; 2],
}

/// Orbital magnetic susceptibility of a cell at wave vectors commensurate
/// with it.
#[derive(Clone, Debug)]
pub struct LatticeOrbitalSusceptibility {
    /// Volume of the primitive cell, Bohr^3.
    pub volume: f64,
    pub points: Vec<OrbitalSusceptibility>,
    pub reference: SeriesReference,
}

/// Orbital magnetic susceptibility `chi(q)` of a cell for the transverse
/// vector potentials `A = a e cos(q . r)` and `a e sin(q . r)` at reciprocal
/// lattice vectors `q` of the sampled (Gamma-point) cell. The field
/// `B = curl A` has the mean square `a^2 q^2 / 2`, so the second-order
/// energy `E2 a^2` gives `chi(q) = -4 E2 / (Omega q^2)`. `E2` is the mean
/// of the paramagnetic second-order energies of the two waves plus the
/// diamagnetic term `N / 4`; in a finite basis of field-independent
/// functions that term is replaced through the sum rule of the uniform
/// potential, `N / 2 = -E2_para(q = 0)`, which removes the gauge error that
/// would otherwise grow as `1 / q^2`. Exact exchange and the meta-GGA
/// current derivative act on the imaginary response. In the covariant
/// combination `tau_A - |j_A|^2/(2 rho)` the explicit vector potential
/// cancels. Its implicit density response is the current-current kernel.
#[allow(clippy::too_many_arguments)]
pub fn lattice_orbital_susceptibility(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    waves: &[[i64; 3]],
    config: &ResponseSolverConfig,
) -> Result<LatticeOrbitalSusceptibility> {
    config.validate()?;
    primitive_atoms(cell, replicas)?;
    if waves.is_empty() || waves.iter().any(|w| w.iter().all(|c| *c == 0)) {
        return Err(DftError::InvalidInput(
            "the orbital susceptibility needs wave vectors different from zero".into(),
        ));
    }
    let setup = Setup::new(cell, functional, result, options)?;
    let reference = reference(&setup, result, options)?;
    let current = (functional.family == Family::MetaGga)
        .then(|| current::LatticeCurrentKernel::new(functional, &reference))
        .transpose()?;
    let scale = 1.0 / replicas as f64;
    let volume = cell.lattice().volume().abs() * scale;
    let flat = momentum_matrices(cell, crate::math::Vec3::zero());
    let mut points = Vec::with_capacity(waves.len());
    for wave in waves {
        let vector = wave_vector(cell, *wave);
        let q = vector.to_array();
        let q2 = vector.norm2();
        let (polarization, field) = crate::periodic::magnetic_scf::transverse_frame(vector);
        let polarization = polarization.map(crate::math::Vec3::to_array);
        let parts = momentum_matrices(cell, vector);
        let mut chi = [0.0; 2];
        let mut paramagnetic = [0.0; 2];
        let mut uniform = [0.0; 2];
        let mut response_residuals = [[0.0; 3]; 2];
        for (slot, e) in polarization.into_iter().enumerate() {
            let (cosine, cosine_residual) = imaginary_second_order(
                &setup,
                &reference,
                &antisymmetric([&parts[0].0, &parts[1].0, &parts[2].0], e),
                current.as_ref(),
                config,
            )?;
            let (sine, sine_residual) = imaginary_second_order(
                &setup,
                &reference,
                &antisymmetric([&parts[0].1, &parts[1].1, &parts[2].1], e),
                current.as_ref(),
                config,
            )?;
            let (zero, zero_residual) = imaginary_second_order(
                &setup,
                &reference,
                &antisymmetric([&flat[0].0, &flat[1].0, &flat[2].0], e),
                current.as_ref(),
                config,
            )?;
            paramagnetic[slot] = 0.5 * (cosine + sine) * scale;
            uniform[slot] = zero * scale;
            chi[slot] = -4.0 * (paramagnetic[slot] - 0.5 * uniform[slot]) / (volume * q2);
            response_residuals[slot] = [cosine_residual, sine_residual, zero_residual];
        }
        points.push(OrbitalSusceptibility {
            wave: *wave,
            vector: q,
            polarization,
            field: field.map(crate::math::Vec3::to_array),
            chi,
            paramagnetic,
            uniform,
            response_residuals,
        });
    }
    Ok(LatticeOrbitalSusceptibility {
        volume,
        points,
        reference: series_reference(&reference, result, replicas),
    })
}
