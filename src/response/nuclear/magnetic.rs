// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear response about a self-consistent finite London-field reference.
//! The electronic metric, both integral endpoints, nuclear potential,
//! atom-owned quadrature and complex semilocal kernel share the existing
//! analytic integral, quadrature and stationary projector implementations.

use super::geometry::moving_grid_taylor;
use crate::basis::{n_sph, MolBasis};
use crate::dual::{Scalar, Taylor};
use crate::grid::MolecularGrid;
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{contracted_components, AnalyticShell};
use crate::integrals::london::{
    primitive_field_pair, primitive_field_repulsion, ComplexMatrix as Cm, ScalarFieldPath,
};
use crate::math::Vec3;
use crate::periodic::integrals::shifted;
use crate::response::field_path::{
    phase_exponential, point_xc_series, stationary_complex_path, ComplexPathBackend,
    ComplexPathReference, FieldPathTaylor, PointSeries,
};
use crate::response::ResponseSolverConfig;
use crate::scf::{
    magnetic::{MagneticFieldOptions, MagneticFieldResult},
    ScfOptions,
};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};
use rayon::prelude::*;

type J = Taylor<5>;
type Z = (f64, f64);

fn times(a: Z, b: Z) -> Z {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn add(a: &mut Cm, b: &Cm, weight: f64) {
    for (a, b) in a.re.as_mut_slice().iter_mut().zip(b.re.as_slice()) {
        *a += weight * b;
    }
    for (a, b) in a.im.as_mut_slice().iter_mut().zip(b.im.as_slice()) {
        *a += weight * b;
    }
}
fn zero(value: J) -> bool {
    value.c.iter().all(|c| *c == 0.0)
}
fn index(n: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    ((a * n + b) * n + c) * n + d
}
fn half_cross(b: [f64; 3], r: [J; 3]) -> [J; 3] {
    [
        (r[2].scale(b[1]) - r[1].scale(b[2])).scale(0.5),
        (r[0].scale(b[2]) - r[2].scale(b[0])).scale(0.5),
        (r[1].scale(b[0]) - r[0].scale(b[1])).scale(0.5),
    ]
}

fn half_cross_series(b: [J; 3], r: [J; 3]) -> [J; 3] {
    [
        (r[2] * b[1] - r[1] * b[2]).scale(0.5),
        (r[0] * b[2] - r[2] * b[0]).scale(0.5),
        (r[1] * b[0] - r[0] * b[1]).scale(0.5),
    ]
}

struct MovingPoint {
    values: PointSeries,
    weights: Vec<f64>,
}
struct MagneticGeometry<'a> {
    functional: &'a ResolvedFunctional,
    nao: usize,
    overlap: Vec<Cm>,
    core: [Vec<Cm>; 2],
    dipoles: [Vec<Cm>; 3],
    nuclear: Vec<f64>,
    nuclear_dipole: Vec<[f64; 3]>,
    eri: Vec<Vec<Z>>,
    eri_lr: Option<Vec<Vec<Z>>>,
    points: Vec<MovingPoint>,
    density: [Cm; 2],
}

impl MagneticGeometry<'_> {
    /// Linear Hartree/exchange coefficient at the actual geometry coefficient.
    fn interaction(&self, p: &[Vec<Cm>; 2], n: usize, fixed_geometry: bool) -> [Cm; 2] {
        let mut result = [Cm::zeros(self.nao, self.nao), Cm::zeros(self.nao, self.nao)];
        for m in 0..=if fixed_geometry { 0 } else { n } {
            let degree = n - m;
            for a in 0..self.nao {
                for b in 0..self.nao {
                    for c in 0..self.nao {
                        for d in 0..self.nao {
                            let g = self.eri[m][index(self.nao, a, b, c, d)];
                            let total = (
                                p[0][degree].re[(d, c)] + p[1][degree].re[(d, c)],
                                p[0][degree].im[(d, c)] + p[1][degree].im[(d, c)],
                            );
                            let j = times(g, total);
                            for s in 0..2 {
                                result[s].re[(a, b)] += j.0;
                                result[s].im[(a, b)] += j.1;
                                let mut exchange = self.eri[m][index(self.nao, a, d, c, b)];
                                exchange.0 *= self.functional.hybrid.cx;
                                exchange.1 *= self.functional.hybrid.cx;
                                if let Some(lr) = &self.eri_lr {
                                    let v = lr[m][index(self.nao, a, d, c, b)];
                                    exchange.0 += self.functional.hybrid.clr * v.0;
                                    exchange.1 += self.functional.hybrid.clr * v.1;
                                }
                                let k = times(
                                    exchange,
                                    (p[s][degree].re[(d, c)], p[s][degree].im[(d, c)]),
                                );
                                result[s].re[(a, b)] -= k.0;
                                result[s].im[(a, b)] -= k.1;
                            }
                        }
                    }
                }
            }
        }
        result
    }

    fn xc(
        &self,
        p: &[Vec<Cm>; 2],
        n: usize,
        fixed_geometry: bool,
        matrices: bool,
    ) -> Result<(Vec<f64>, [Cm; 2])> {
        let parts = self
            .points
            .par_iter()
            .map(|point| {
                if fixed_geometry {
                    let stationary = PointSeries {
                        w: point
                            .values
                            .w
                            .iter()
                            .map(|v| {
                                let mut x = vec![(0.0, 0.0); n + 1];
                                x[0] = v[0];
                                x
                            })
                            .collect(),
                        gw: point
                            .values
                            .gw
                            .iter()
                            .map(|v| {
                                std::array::from_fn(|a| {
                                    let mut x = vec![(0.0, 0.0); n + 1];
                                    x[0] = v[a][0];
                                    x
                                })
                            })
                            .collect(),
                        pi: point
                            .values
                            .pi
                            .iter()
                            .map(|v| {
                                std::array::from_fn(|a| {
                                    let mut x = vec![(0.0, 0.0); n + 1];
                                    x[0] = v[a][0];
                                    x
                                })
                            })
                            .collect(),
                    };
                    let mut weight = vec![0.0; n + 1];
                    weight[0] = point.weights[0];
                    point_xc_series(&stationary, &weight, self.functional, p, n, matrices)
                } else {
                    point_xc_series(
                        &point.values,
                        &point.weights,
                        self.functional,
                        p,
                        n,
                        matrices,
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let mut energy = vec![0.0; n + 1];
        let mut result = [Cm::zeros(self.nao, self.nao), Cm::zeros(self.nao, self.nao)];
        for (e, v) in parts {
            for (a, b) in energy.iter_mut().zip(e) {
                *a += b;
            }
            if let Some(v) = v {
                for s in 0..2 {
                    for (a, b) in result[s].re.as_mut_slice().iter_mut().zip(&v[s].0) {
                        *a += b;
                    }
                    for (a, b) in result[s].im.as_mut_slice().iter_mut().zip(&v[s].1) {
                        *a += b;
                    }
                }
            }
        }
        Ok((energy, result))
    }
}
impl ComplexPathBackend for MagneticGeometry<'_> {
    fn overlap(&self) -> &[Cm] {
        &self.overlap
    }
    fn fock_coefficient(&self, p: &[Vec<Cm>; 2], n: usize) -> Result<[Cm; 2]> {
        let mut result = self.interaction(p, n, false);
        let xc = self.xc(p, n, false, true)?.1;
        for s in 0..2 {
            add(&mut result[s], &self.core[s][n], 1.0);
            add(&mut result[s], &xc[s], 1.0);
        }
        Ok(result)
    }
    fn linear_kernel(&self, direction: &[Cm; 2]) -> Result<[Cm; 2]> {
        // Separate geometric derivatives from the density derivative of the
        // finite-field functional, including its complex current/tau coupling.
        let p = std::array::from_fn(|s| vec![self.reference_density(s), direction[s].clone()]);
        let mut result = self.interaction(&p, 1, true);
        let xc = self.xc(&p, 1, true, true)?.1;
        for s in 0..2 {
            add(&mut result[s], &xc[s], 1.0);
        }
        Ok(result)
    }
    fn energy(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<Vec<f64>> {
        let interaction: Vec<_> = (0..=order).map(|n| self.interaction(p, n, false)).collect();
        let mut result = self.nuclear[..=order].to_vec();
        for n in 0..=order {
            for m in 0..=n {
                for s in 0..2 {
                    result[n] += p[s][m].real_trace_product(&self.core[s][n - m]);
                    result[n] += 0.5 * p[s][m].real_trace_product(&interaction[n - m][s]);
                }
            }
        }
        let xc = self.xc(p, order, false, false)?.0;
        for (a, b) in result.iter_mut().zip(xc) {
            *a += b;
        }
        Ok(result)
    }
    fn dipole(&self, p: &[Vec<Cm>; 2], order: usize) -> Vec<[f64; 3]> {
        (0..=order)
            .map(|n| {
                std::array::from_fn(|a| {
                    let mut value = self.nuclear_dipole[n][a];
                    for m in 0..=n {
                        for s in 0..2 {
                            value -= p[s][m].real_trace_product(&self.dipoles[a][n - m]);
                        }
                    }
                    value
                })
            })
            .collect()
    }
}

impl MagneticGeometry<'_> {
    fn reference_density(&self, spin: usize) -> Cm {
        self.density[spin].clone()
    }
}

/// Analytic nuclear energy/density response through fourth order about an
/// actual finite London-field SCF reference, with an optional simultaneous
/// uniform electric field. Coordinates and directions use Bohr; energy
/// coefficients use Hartree and the factorial-normalized Taylor convention.
/// The magnetic integral degree is the explicitly reported SCF series order.
/// Semilocal and exact/range-separated exchange providers share one kernel.
pub fn magnetic_nuclear_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    options: &ScfOptions,
    field: &MagneticFieldOptions,
    reference: &MagneticFieldResult,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<FieldPathTaylor> {
    Ok(magnetic_nuclear_field_taylor(
        mol, mb, functional, grid, options, field, reference, direction, [0.0; 3], order, config,
    )?
    .report)
}

/// Factorial-normalized energy and physical AO density on a shared path.
/// This is orbital collinear response, not a SOC or spin-flip solver.
pub struct MixedMagneticNuclearTaylor {
    pub report: FieldPathTaylor,
    pub density: [Vec<Cm>; 2],
}

/// R(t)=R0+t*direction and B(t)=B0+t*magnetic_direction.
/// The fixed-field wrapper retains its original finite-field phase degree.
/// A varying field requires spin_zeeman=false: |B| is not analytic at B=0.
#[allow(clippy::too_many_arguments)]
pub fn magnetic_nuclear_field_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    options: &ScfOptions,
    field: &MagneticFieldOptions,
    reference: &MagneticFieldResult,
    direction: &[[f64; 3]],
    magnetic_direction: [f64; 3],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<MixedMagneticNuclearTaylor> {
    let varying_field = magnetic_direction.iter().any(|b| *b != 0.0);
    if magnetic_direction.iter().any(|b| !b.is_finite()) {
        return Err(DftError::InvalidInput(
            "magnetic path direction must be finite".into(),
        ));
    }
    if varying_field && field.spin_zeeman {
        return Err(DftError::Unsupported {
            feature: "varying-field nuclear response with magnitude spin Zeeman shift".into(),
            missing:
                "an analytic spin-frame Hamiltonian instead of the nonanalytic field magnitude"
                    .into(),
        });
    }
    let magnetic =
        std::array::from_fn(|a| J::from_coefficients(&[field.field[a], magnetic_direction[a]]));
    config.validate()?;
    if field.field.iter().any(|b| *b != 0.0) && reference.series_order < order + 2 {
        return Err(DftError::InvalidInput(format!(
            "finite-field nuclear order {order} requires London integral degree at least {}; rebuild the reference with a larger series_order", order + 2
        )));
    }
    if mol.is_empty()
        || order == 0
        || order > 4
        || direction.len() != mol.len()
        || direction.iter().flatten().any(|x| !x.is_finite())
        || !reference.converged
        || mb.shells.iter().any(|s| s.atom >= mol.len())
    {
        return Err(DftError::InvalidInput("finite-field nuclear response needs a converged reference, finite atomic directions and order one through four".into()));
    }
    if !mb.ecps.is_empty()
        || options.pcm.is_some()
        || options.vv10.is_some()
        || options.skala.is_some()
        || options.density_fit.is_some()
        || options.extra_hcore.is_some()
        || functional.required_nonlocal_kind().is_some()
        || options.electronic_temperature > 0.0
        || !matches!(options.kernel, crate::integrals::CoulombKernel::Coulomb)
    {
        return Err(DftError::Unsupported { feature:"finite-field complex nuclear response with additional Hamiltonian terms".into(), missing:"analytic ECP, solvent, nonlocal/learned, density-fitting, extra-core or fractional-occupation providers on this complex nuclear path".into() });
    }
    let shift = Vec3::new(reference.shift[0], reference.shift[1], reference.shift[2]);
    let mut molecule = mol.clone();
    for atom in &mut molecule.atoms {
        atom.position += shift;
    }
    let basis = shifted(mb, shift);
    let nao = basis.nao_sph;
    let width = order + 1;
    let size = nao.checked_pow(4).ok_or_else(|| {
        DftError::InvalidInput("finite-field integral tensor size overflow".into())
    })?;
    let bytes = size.checked_mul(80 + 32 * width).ok_or_else(|| {
        DftError::InvalidInput("finite-field integral cache size overflow".into())
    })?;
    if bytes > 512 * 1024 * 1024 {
        return Err(DftError::InvalidInput("finite-field nuclear integral cache exceeds its 512 MiB budget; a streamed contraction is required for this basis".into()));
    }
    let centers: Vec<[J; 3]> = molecule
        .atoms
        .iter()
        .zip(direction)
        .map(|(atom, d)| {
            std::array::from_fn(|a| J::from_coefficients(&[atom.position.to_array()[a], d[a]]))
        })
        .collect();
    let shells = basis
        .shells
        .iter()
        .map(|s| {
            let offset = s.center - molecule.atoms[s.atom].position;
            AnalyticShell::from_placed(
                s,
                std::array::from_fn(|a| centers[s.atom][a] + J::from_f64(offset.to_array()[a])),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let nuclei: Vec<_> = centers
        .iter()
        .zip(&molecule.atoms)
        .map(|(r, a)| (*r, J::from_f64(a.z as f64)))
        .collect();
    let path = ScalarFieldPath {
        electric: reference.electric_field.map(J::from_f64),
        magnetic,
        origin: [J::from_f64(0.0); 3],
        london: true,
        phase_order: if varying_field {
            reference.series_order.max(order)
        } else {
            reference.series_order
        },
        derivative_order: order,
    };
    let pairs: Vec<_> = (0..shells.len())
        .flat_map(|a| (0..shells.len()).map(move |b| (a, b)))
        .collect();
    let pair_blocks = pairs
        .par_iter()
        .map(|&(a, b)| {
            contracted_components::<J, 10, 2>([&shells[a], &shells[b]], |[pa, pb]| {
                let value = primitive_field_pair(&pa, &pb, &nuclei, &path, zero)?;
                Ok(std::array::from_fn(|i| {
                    if i % 2 == 0 {
                        value[i / 2].0
                    } else {
                        value[i / 2].1
                    }
                }))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut one: [Vec<Cm>; 5] = std::array::from_fn(|_| vec![Cm::zeros(nao, nao); width]);
    for ((a, b), block) in pairs.iter().copied().zip(pair_blocks) {
        for i in 0..n_sph(shells[a].l) {
            for j in 0..n_sph(shells[b].l) {
                let local = i * n_sph(shells[b].l) + j;
                for channel in 0..5 {
                    for n in 0..width {
                        let (mu, nu) = (basis.shells[a].sph_off + i, basis.shells[b].sph_off + j);
                        one[channel][n].re[(mu, nu)] = block[2 * channel].values[local].c[n];
                        one[channel][n].im[(mu, nu)] = block[2 * channel + 1].values[local].c[n];
                    }
                }
            }
        }
    }
    let quartets: Vec<_> = pairs
        .iter()
        .flat_map(|&(a, b)| pairs.iter().map(move |&(c, d)| [a, b, c, d]))
        .collect();
    let build_eri = |kernel| -> Result<Vec<Vec<Z>>> {
        let blocks = quartets
            .par_iter()
            .map(|s| {
                contracted_components::<J, 2, 4>(s.map(|a| &shells[a]), |p| {
                    let v = primitive_field_repulsion(p.each_ref(), kernel, &path, zero)?;
                    Ok([v.0, v.1])
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tensor = vec![vec![(0.0, 0.0); size]; width];
        for (s, block) in quartets.iter().zip(blocks) {
            let shape = s.map(|a| n_sph(shells[a].l));
            for local in 0..block[0].values.len() {
                let mut rem = local;
                let mut ao = [0; 4];
                for axis in (0..4).rev() {
                    ao[axis] = basis.shells[s[axis]].sph_off + rem % shape[axis];
                    rem /= shape[axis];
                }
                let slot = index(nao, ao[0], ao[1], ao[2], ao[3]);
                for n in 0..width {
                    tensor[n][slot] = (block[0].values[local].c[n], block[1].values[local].c[n]);
                }
            }
        }
        Ok(tensor)
    };
    let eri = build_eri(GaussianKernel::Coulomb)?;
    let eri_lr = if functional.hybrid.clr != 0.0 {
        Some(build_eri(GaussianKernel::LongRange {
            omega: J::from_f64(functional.hybrid.omega),
        })?)
    } else {
        None
    };
    let mut nuclear = J::from_f64(
        mol.charge as f64
            * Vec3::new(
                reference.electric_field[0],
                reference.electric_field[1],
                reference.electric_field[2],
            )
            .dot(shift),
    );
    let mut nuclear_dipole = [J::from_f64(0.0); 3];
    for a in 0..mol.len() {
        for axis in 0..3 {
            nuclear_dipole[axis] =
                nuclear_dipole[axis] + centers[a][axis].scale(mol.atoms[a].z as f64);
        }
        for b in 0..a {
            let distance = (0..3)
                .map(|i| (centers[a][i] - centers[b][i]).powi(2))
                .fold(J::from_f64(0.0), |a, b| a + b)
                .sqrt();
            nuclear =
                nuclear + J::from_f64(mol.atoms[a].z as f64 * mol.atoms[b].z as f64) / distance;
        }
    }
    for a in 0..3 {
        nuclear = nuclear - nuclear_dipole[a].scale(reference.electric_field[a]);
        nuclear_dipole[a] = nuclear_dipole[a] - J::from_f64(mol.charge as f64 * reference.shift[a]);
    }
    let strength = field.field.iter().map(|b| b * b).sum::<f64>().sqrt();
    let zeeman = if field.spin_zeeman {
        0.25 * crate::constants::ELECTRON_G_FACTOR * strength
    } else {
        0.0
    };
    let core = std::array::from_fn(|s| {
        (0..width)
            .map(|n| {
                let mut h = one[1][n].clone();
                add(&mut h, &one[0][n], if s == 0 { zeeman } else { -zeeman });
                h
            })
            .collect()
    });
    let mut points = Vec::new();
    if !functional.parts.is_empty() {
        let mut grid = grid
            .ok_or_else(|| {
                DftError::InvalidInput("finite-field nuclear XC requires its reference grid".into())
            })?
            .clone();
        for p in &mut grid.points {
            *p += shift;
        }
        let input: Vec<_> = centers
            .iter()
            .flat_map(|r| r.iter().map(|v| v.c[..width].to_vec()))
            .collect();
        let jets = moving_grid_taylor(&molecule, &basis, &grid, &input, order)?;
        let mut ao_centers = vec![[J::from_f64(0.0); 3]; nao];
        for (placed, shell) in basis.shells.iter().zip(&shells) {
            for a in 0..n_sph(shell.l) {
                ao_centers[placed.sph_off + a] = shell.center;
            }
        }
        for (i, jet) in jets.into_iter().enumerate() {
            let owner = grid.atom_of[i];
            let offset = grid.points[i] - molecule.atoms[owner].position;
            let r: [J; 3] =
                std::array::from_fn(|a| centers[owner][a] + J::from_f64(offset.to_array()[a]));
            let mut values = PointSeries {
                w: Vec::new(),
                gw: Vec::new(),
                pi: Vec::new(),
            };
            for mu in 0..nao {
                let slope = if varying_field {
                    half_cross_series(magnetic, ao_centers[mu])
                } else {
                    half_cross(field.field, ao_centers[mu])
                };
                let relative = std::array::from_fn(|a| r[a] - ao_centers[mu][a]);
                let gauge = if varying_field {
                    half_cross_series(magnetic, relative)
                } else {
                    half_cross(field.field, relative)
                };
                let theta = (0..3)
                    .map(|a| slope[a] * r[a])
                    .fold(J::from_f64(0.0), |a, b| a + b);
                let phase = phase_exponential(&theta.c[..width]);
                let er = J::from_coefficients(&phase.iter().map(|v| v.0).collect::<Vec<_>>());
                let ei = J::from_coefficients(&phase.iter().map(|v| v.1).collect::<Vec<_>>());
                let phi = J::from_coefficients(&jet.aos[mu][0]);
                let w = (er * phi, ei * phi);
                values
                    .w
                    .push((0..width).map(|n| (w.0.c[n], w.1.c[n])).collect());
                let mut gradients: [Vec<Z>; 3] = std::array::from_fn(|_| Vec::new());
                let mut momenta: [Vec<Z>; 3] = std::array::from_fn(|_| Vec::new());
                for a in 0..3 {
                    let grad = J::from_coefficients(&jet.aos[mu][1 + a]);
                    let gw = (
                        er * grad + ei * slope[a] * phi,
                        ei * grad - er * slope[a] * phi,
                    );
                    let pi = (
                        er * gauge[a] * phi + ei * grad,
                        ei * gauge[a] * phi - er * grad,
                    );
                    gradients[a] = (0..width).map(|n| (gw.0.c[n], gw.1.c[n])).collect();
                    momenta[a] = (0..width).map(|n| (pi.0.c[n], pi.1.c[n])).collect();
                }
                values.gw.push(gradients);
                values.pi.push(momenta);
            }
            points.push(MovingPoint {
                values,
                weights: jet.weight,
            });
        }
    }
    // A restricted reference has one shared spatial orbital gauge. The
    // spin Zeeman constant does not require independent beta orbitals.
    let density = if reference.unrestricted {
        reference.density.clone()
    } else {
        [reference.density[0].clone(), reference.density[0].clone()]
    };
    let coefficients = if reference.unrestricted {
        reference.coefficients.clone()
    } else {
        [
            reference.coefficients[0].clone(),
            reference.coefficients[0].clone(),
        ]
    };
    let backend = MagneticGeometry {
        functional,
        nao,
        overlap: one[0].clone(),
        core,
        dipoles: std::array::from_fn(|a| one[2 + a].clone()),
        nuclear: nuclear.c[..width].to_vec(),
        nuclear_dipole: (0..width).map(|n| nuclear_dipole.map(|d| d.c[n])).collect(),
        eri,
        eri_lr,
        points,
        density: density.clone(),
    };
    let state = ComplexPathReference {
        coefficients,
        energies: reference.orbital_energies.clone(),
        occupations: [reference.n_alpha, reference.n_beta],
        density,
        fock: reference.fock.clone(),
        unrestricted: reference.unrestricted,
    };
    let (result, solved) = stationary_complex_path(&backend, &state, order, config)?;
    let difference = (result.energy[0] - reference.energy).abs();
    let tolerance = config
        .acceptance()
        .max(options.conv_tol_grad)
        .max(options.e_tol);
    if difference > tolerance * reference.energy.abs().max(1.0)
        || result.reference_defect > tolerance
    {
        return Err(DftError::InvalidInput(format!("finite-field nuclear operator does not match its SCF reference: energy difference {difference:.16e}, Fock/metric defect {:.16e}; use the same fields, integral degree and functional", result.reference_defect)));
    }
    Ok(MixedMagneticNuclearTaylor {
        report: result,
        density: solved.p,
    })
}
