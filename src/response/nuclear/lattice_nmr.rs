// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear magnetic shielding in a periodic cell at Gamma.
//!
//! The shielding of nucleus `A` is the mixed second derivative of the energy
//! in the amplitude of a transverse vector potential and in the magnetic
//! moment of the nucleus. The vector potential is the wave
//! `A_w = a e sin(q . (r - R_A))` with a reciprocal lattice vector `q` of
//! the cell: its field at the nucleus is `a q (q^ x e)`, and close to the
//! nucleus it is a Landau gauge centred there, so the diamagnetic and
//! paramagnetic parts are balanced where the basis is best. The moments of
//! the periodic images of the nucleus form a lattice of dipoles with the
//! vector potential `A_d = alpha^2 m x grad_R Phi`, `Phi` the Ewald
//! potential of the lattice of nuclei without its `G = 0` component. Then
//!
//! ```text
//! sigma_(beta, q^ x e) = (1/q) d2E / (da dm_beta)
//!   = (1/q) { alpha^2 Tr[D sin(theta) (e x e_beta) . grad_R Phi]
//!             + 2 sum_ai y^(a)_ai B^(m_beta)_ai },
//! ```
//!
//! with the imaginary perturbations `B^(a) = -<sin(theta) e . grad>` and
//! `B^(m_beta) = -alpha^2 eps_(beta b c) <(d Phi / dR_b) d/dr_c>` and the
//! amplitudes `y^(a)` of the response to the wave. The value tends to the
//! shielding of a uniform field for `q -> 0`; the omitted `G = 0` component
//! is the shape-dependent contribution of the bulk magnetization.

use super::lattice::{
    antisymmetric, imaginary_mixed, primitive_atoms, reference, series_reference, wave_vector,
    SeriesReference, Setup,
};
use crate::basis::aos::cart_to_sph;
use crate::basis::{cart_components, n_sph, MolShell};
use crate::integrals::analytic::{self, GaussianKernel, Primitive};
use crate::integrals::london::{primitive, Terms};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::integrals::nearby_translation_indices;
use crate::periodic::path_series::{momentum_matrices, pair_matrices};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::screened_coulomb::in_half_space;
use crate::periodic::Cell;
use crate::response::krylov::ResponseSolverConfig;
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};
use rayon::prelude::*;

type Dual3 = crate::dual::DualN<f64, 3>;

/// Ewald splitting parameter of the lattice of nuclear dipoles, Bohr^-1.
const EWALD: f64 = 1.5;
/// Bound on the neglected terms of its sums.
const TOLERANCE: f64 = 1e-10;
/// Two primitives overlap while `mu |A - B|^2` stays below this value.
const PAIR_LOG: f64 = 40.0;

fn lifted(p: &Primitive<f64>) -> Primitive<Dual3> {
    Primitive {
        angular: p.angular,
        exponent: Dual3::constant(p.exponent),
        center: p.center.map(Dual3::constant),
    }
}

fn least(shell: &MolShell) -> f64 {
    shell.exps.iter().copied().fold(f64::INFINITY, f64::min)
}

/// `eps_(beta b c)` for `b != c`: the third index and the sign.
fn third(b: usize, c: usize) -> (usize, f64) {
    (3 - b - c, if (b + 1) % 3 == c { 1.0 } else { -1.0 })
}

/// `d/dR_b <bra| erfc(alpha |r - R|) / |r - R| |ket>` at the nucleus `image`.
fn screened_gradient(bra: &Primitive<Dual3>, ket: &Terms, image: Vec3) -> Result<[f64; 3]> {
    let centre: [Dual3; 3] = std::array::from_fn(|x| Dual3::variable(image.get(x), x));
    let mut out = [0.0; 3];
    for (coefficient, p) in &ket.items {
        let value = analytic::attraction(
            bra,
            &lifted(p),
            centre,
            Dual3::constant(-1.0),
            GaussianKernel::ShortRange {
                omega: Dual3::constant(EWALD),
            },
            1,
        )?;
        for b in 0..3 {
            out[b] += coefficient * value.im[b];
        }
    }
    Ok(out)
}

/// Shell pairs with the translation of the ket shell and the images of the
/// nucleus inside the reach of the screened kernel.
fn neighbours(cell: &Cell, nucleus: Vec3, a: usize, b: usize) -> Vec<(MolShell, Vec<Vec3>)> {
    let (sa, sb) = (&cell.basis.shells[a], &cell.basis.shells[b]);
    let (ea, eb) = (least(sa), least(sb));
    let reach = (PAIR_LOG * (ea + eb) / (ea * eb))
        .sqrt()
        .min(cell.overlap_cutoff);
    let p = ea + eb;
    let rate = (p * EWALD * EWALD / (p + EWALD * EWALD)).sqrt();
    let range = ((1.0 / TOLERANCE).ln() + 4.0).sqrt() / rate + 1.0;
    nearby_translation_indices(&cell.lattice, sb.center - sa.center, reach)
        .into_iter()
        .map(|index| {
            let mut ket = sb.clone();
            ket.center += cell.lattice.translation(index);
            let centre = (sa.center * ea + ket.center * eb) / p;
            let images = nearby_translation_indices(&cell.lattice, nucleus - centre, range)
                .into_iter()
                .map(|image| nucleus + cell.lattice.translation(image))
                .collect();
            (ket, images)
        })
        .collect()
}

/// `P_beta = eps_(beta b c) <chi_m| (d Phi / dR_b) d/dr_c |chi_n>` for the
/// Ewald potential `Phi` of the lattice of images of `nucleus`.
fn dipole_matrices(cell: &Cell, nucleus: Vec3) -> Result<[Matrix; 3]> {
    let basis = &cell.basis;
    let nao = basis.nao_sph;
    let ns = basis.shells.len();
    let shell_pairs: Vec<(usize, usize)> =
        (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    // ---- screened kernel in real space ----
    let blocks = shell_pairs
        .par_iter()
        .map(|&(a, b)| -> Result<[Vec<f64>; 3]> {
            let sa = &basis.shells[a];
            let sb = &basis.shells[b];
            let (ca, cb) = (cart_components(sa.l), cart_components(sb.l));
            let size = ca.len() * cb.len();
            let mut cart: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; size]);
            for (ket_shell, images) in neighbours(cell, nucleus, a, b) {
                for (ia, &component_a) in ca.iter().enumerate() {
                    for (ib, &component_b) in cb.iter().enumerate() {
                        let slot = ia * cb.len() + ib;
                        for pa in 0..sa.exps.len() {
                            let bra = lifted(&primitive(sa, component_a, pa));
                            for pb in 0..ket_shell.exps.len() {
                                let weight = sa.coefs[pa] * ket_shell.coefs[pb];
                                let ket = Terms::single(primitive(&ket_shell, component_b, pb));
                                for c in 0..3 {
                                    let moved = ket.derivative(c);
                                    for image in &images {
                                        let gradient = screened_gradient(&bra, &moved, *image)?;
                                        for b_axis in 0..3 {
                                            if b_axis == c {
                                                continue;
                                            }
                                            let (beta, sign) = third(b_axis, c);
                                            cart[beta][slot] += sign * weight * gradient[b_axis];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let (ta, tb) = (cart_to_sph(sa.l), cart_to_sph(sb.l));
            let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
            Ok(std::array::from_fn(|beta| {
                let mut out = vec![0.0; na * nb];
                for i in 0..na {
                    for j in 0..nb {
                        for ci in 0..ca.len() {
                            for cj in 0..cb.len() {
                                out[i * nb + j] +=
                                    ta[(ci, i)] * tb[(cj, j)] * cart[beta][ci * cb.len() + cj];
                            }
                        }
                    }
                }
                out
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(nao, nao));
    for (&(a, b), block) in shell_pairs.iter().zip(&blocks) {
        let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
        let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
        for beta in 0..3 {
            for i in 0..na {
                for j in 0..nb {
                    out[beta][(sa.sph_off + i, sb.sph_off + j)] = block[beta][i * nb + j];
                }
            }
        }
    }
    // ---- damped kernel in reciprocal space ----
    let volume = cell.lattice.volume().abs();
    let cutoff = 2.0 * EWALD * (1.0 / TOLERANCE).ln().sqrt();
    let mut waves = cell.lattice.reciprocal_within(cutoff);
    waves.retain(|g| in_half_space(*g));
    let parts: Vec<[Matrix; 3]> = waves
        .par_iter()
        .map(|g| {
            let g2 = g.norm2();
            let weight =
                4.0 * std::f64::consts::PI * (-g2 / (4.0 * EWALD * EWALD)).exp() / (volume * g2);
            let momentum = momentum_matrices(cell, *g);
            let phase = g.dot(nucleus);
            let (cosine, sine) = (phase.cos(), phase.sin());
            let mut out: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(nao, nao));
            for c in 0..3 {
                for b_axis in 0..3 {
                    if b_axis == c {
                        continue;
                    }
                    let (beta, sign) = third(b_axis, c);
                    let factor = sign * 2.0 * weight * g.get(b_axis);
                    if factor == 0.0 {
                        continue;
                    }
                    for (target, (re, im)) in out[beta].as_mut_slice().iter_mut().zip(
                        momentum[c]
                            .0
                            .as_slice()
                            .iter()
                            .zip(momentum[c].1.as_slice()),
                    ) {
                        *target += factor * (cosine * im - sine * re);
                    }
                }
            }
            out
        })
        .collect();
    for part in &parts {
        for beta in 0..3 {
            for (target, value) in out[beta]
                .as_mut_slice()
                .iter_mut()
                .zip(part[beta].as_slice())
            {
                *target += *value;
            }
        }
    }
    Ok(out)
}

/// `d_b = Tr[D <chi_m| sin(q . (r - R)) (d Phi / dR_b) |chi_n>]` for the total
/// density `D`, with `sin` expanded about every image of the nucleus in the
/// screened real-space part.
fn diamagnetic_vector(cell: &Cell, nucleus: Vec3, q: Vec3, density: &Matrix) -> Result<[f64; 3]> {
    let basis = &cell.basis;
    let ns = basis.shells.len();
    let wave = q.to_array();
    // Order of the sine series: the screened kernel confines `r` to a few
    // screening lengths of the nucleus.
    let x = q.norm() * ((1.0 / TOLERANCE).ln() + 4.0).sqrt() / EWALD;
    let mut order = 1;
    let mut term = x;
    while term > 0.1 * TOLERANCE && order < 41 {
        order += 2;
        term *= x * x / ((order - 1) * order) as f64;
    }
    let shell_pairs: Vec<(usize, usize)> =
        (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let real = shell_pairs
        .par_iter()
        .map(|&(a, b)| -> Result<[f64; 3]> {
            let sa = &basis.shells[a];
            let sb = &basis.shells[b];
            let (ca, cb) = (cart_components(sa.l), cart_components(sb.l));
            let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
            // Density block in Cartesian components.
            let mut block = Matrix::zeros(na, nb);
            for i in 0..na {
                for j in 0..nb {
                    block[(i, j)] = density[(sa.sph_off + i, sb.sph_off + j)];
                }
            }
            let weights = cart_to_sph(sa.l)
                .matmul_seq(&block)
                .matmul_seq(&cart_to_sph(sb.l).transpose());
            let mut out = [0.0; 3];
            for (ket_shell, images) in neighbours(cell, nucleus, a, b) {
                for (ia, &component_a) in ca.iter().enumerate() {
                    for (ib, &component_b) in cb.iter().enumerate() {
                        let dw = weights[(ia, ib)];
                        if dw == 0.0 {
                            continue;
                        }
                        for pa in 0..sa.exps.len() {
                            let bra = lifted(&primitive(sa, component_a, pa));
                            for pb in 0..ket_shell.exps.len() {
                                let weight = dw * sa.coefs[pa] * ket_shell.coefs[pb];
                                let ket = Terms::single(primitive(&ket_shell, component_b, pb));
                                for image in &images {
                                    // (q . rho)^n ket about this image.
                                    let mut power = ket.clone();
                                    let mut factorial = 1.0;
                                    for n in 1..=order {
                                        let mut next = Terms::empty();
                                        for k in 0..3 {
                                            if wave[k] != 0.0 {
                                                next = next.plus(
                                                    &power.shifted(k, image.get(k)).scaled(wave[k]),
                                                );
                                            }
                                        }
                                        power = next;
                                        factorial *= n as f64;
                                        if n % 2 == 0 {
                                            continue;
                                        }
                                        let sign = if (n / 2) % 2 == 0 { 1.0 } else { -1.0 };
                                        let gradient = screened_gradient(&bra, &power, *image)?;
                                        for b_axis in 0..3 {
                                            out[b_axis] +=
                                                weight * sign / factorial * gradient[b_axis];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out = [0.0; 3];
    for part in &real {
        for b in 0..3 {
            out[b] += part[b];
        }
    }
    // ---- damped kernel in reciprocal space ----
    let volume = cell.lattice.volume().abs();
    let cutoff = 2.0 * EWALD * (1.0 / TOLERANCE).ln().sqrt();
    let mut waves = cell.lattice.reciprocal_within(cutoff);
    waves.retain(|g| in_half_space(*g));
    // Re[exp(-i G' . R) Tr[D Q(G')]].
    let transform = |v: Vec3| -> f64 {
        let (re, im) = pair_matrices(cell, v);
        let phase = v.dot(nucleus);
        density.frobenius_dot(&re) * phase.cos() + density.frobenius_dot(&im) * phase.sin()
    };
    let parts: Vec<[f64; 3]> = waves
        .par_iter()
        .map(|g| {
            let g2 = g.norm2();
            let weight =
                4.0 * std::f64::consts::PI * (-g2 / (4.0 * EWALD * EWALD)).exp() / (volume * g2);
            let difference = transform(*g + q) - transform(*g - q);
            std::array::from_fn(|b| -weight * g.get(b) * difference)
        })
        .collect();
    for part in &parts {
        for b in 0..3 {
            out[b] += part[b];
        }
    }
    Ok(out)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normal(v: [f64; 3]) -> [f64; 3] {
    let size = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|x| x / size)
}

/// Shielding of one nucleus at one wave vector.
#[derive(Clone, Debug)]
pub struct ShieldingPoint {
    pub wave: [i64; 3],
    /// The wave vector, Bohr^-1.
    pub vector: [f64; 3],
    /// Two unit polarizations of the vector potential.
    pub polarization: [[f64; 3]; 2],
    /// Directions of the applied field at the nucleus, `q^ x e`.
    pub field: [[f64; 3]; 2],
    /// Induced field over applied field with opposite sign, per
    /// polarization and Cartesian component of the induced field.
    pub shielding: [[f64; 3]; 2],
    pub paramagnetic: [[f64; 3]; 2],
    pub diamagnetic: [[f64; 3]; 2],
}

/// Shielding of one nucleus.
#[derive(Clone, Debug)]
pub struct NucleusShielding {
    /// Index of the atom in the primitive cell.
    pub atom: usize,
    pub points: Vec<ShieldingPoint>,
    /// Least-squares tensor `sigma[beta][b]` over all field directions, when
    /// they span space.
    pub tensor: Option<[[f64; 3]; 3]>,
    pub isotropic: Option<f64>,
}

/// Shielding of the requested nuclei of a cell.
#[derive(Clone, Debug)]
pub struct LatticeShielding {
    pub nuclei: Vec<NucleusShielding>,
    pub reference: SeriesReference,
}

/// Nuclear magnetic shielding of the atoms `atoms` of the primitive cell at
/// the reciprocal lattice vectors `waves` of the sampled (Gamma-point) cell,
/// for two transverse polarizations each. LDA, GGA and hybrids. `cell`
/// holds `replicas` copies of the primitive cell, ordered replica by
/// replica.
#[allow(clippy::too_many_arguments)]
pub fn lattice_nmr_shielding(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    atoms: &[usize],
    waves: &[[i64; 3]],
    config: &ResponseSolverConfig,
) -> Result<LatticeShielding> {
    config.validate()?;
    let primitive_count = primitive_atoms(cell, replicas)?;
    if functional.family == Family::MetaGga {
        return Err(DftError::Unsupported {
            feature: "nuclear magnetic shielding of a cell with a meta-GGA".into(),
            missing: "the response of the gauge-invariant kinetic energy density to the \
                      vector potential"
                .into(),
        });
    }
    if atoms.iter().any(|a| *a >= primitive_count)
        || waves.is_empty()
        || waves.iter().any(|w| w.iter().all(|c| *c == 0))
    {
        return Err(DftError::InvalidInput(
            "the shielding needs atoms of the primitive cell and wave vectors different from \
             zero"
                .into(),
        ));
    }
    let setup = Setup::new(cell, functional, result, options)?;
    let reference = reference(&setup, result, options)?;
    let mut density = reference.density[0].clone();
    for (target, value) in density
        .as_mut_slice()
        .iter_mut()
        .zip(reference.density[1].as_slice())
    {
        *target += *value;
    }
    let alpha2 = crate::constants::FINE_STRUCTURE * crate::constants::FINE_STRUCTURE;
    let mut nuclei = Vec::with_capacity(atoms.len());
    for &atom in atoms {
        let nucleus = cell.molecule().atoms[atom].position;
        // B^(m_beta) = -alpha^2 P_beta, antisymmetrized.
        let dipole = dipole_matrices(cell, nucleus)?;
        let moments: Vec<Matrix> = dipole
            .iter()
            .map(|p| {
                let mut out = antisymmetric([p, p, p], [1.0, 0.0, 0.0]);
                for value in out.as_mut_slice() {
                    *value *= alpha2;
                }
                out
            })
            .collect();
        let mut points = Vec::with_capacity(waves.len());
        let mut columns: Vec<([f64; 3], [f64; 3])> = Vec::new();
        for wave in waves {
            let vector = wave_vector(cell, *wave);
            let q = vector.to_array();
            let length = vector.norm();
            let weakest = (0..3)
                .min_by(|a, b| q[*a].abs().total_cmp(&q[*b].abs()))
                .unwrap_or(0);
            let mut trial = [0.0; 3];
            trial[weakest] = 1.0;
            let first = normal(cross(q, trial));
            let second = normal(cross(q, first));
            // <sin(q . (r - R)) d/dr_k> from the two parts of exp(i q . r).
            let momentum = momentum_matrices(cell, vector);
            let phase = vector.dot(nucleus);
            let parts: Vec<Matrix> = momentum
                .iter()
                .map(|(re, im)| {
                    let mut out = Matrix::zeros(re.rows, re.cols);
                    for (target, (x, y)) in out
                        .as_mut_slice()
                        .iter_mut()
                        .zip(re.as_slice().iter().zip(im.as_slice()))
                    {
                        *target = phase.cos() * y - phase.sin() * x;
                    }
                    out
                })
                .collect();
            let d = diamagnetic_vector(cell, nucleus, vector, &density)?;
            let mut shielding = [[0.0; 3]; 2];
            let mut paramagnetic = [[0.0; 3]; 2];
            let mut diamagnetic = [[0.0; 3]; 2];
            let mut field = [[0.0; 3]; 2];
            for (slot, e) in [first, second].into_iter().enumerate() {
                let perturbation = antisymmetric([&parts[0], &parts[1], &parts[2]], e);
                let mixed = imaginary_mixed(&setup, &reference, &perturbation, &moments, config)?;
                field[slot] = cross(q, e).map(|v| v / length);
                for beta in 0..3 {
                    let mut unit = [0.0; 3];
                    unit[beta] = 1.0;
                    let direction = cross(e, unit);
                    paramagnetic[slot][beta] = 2.0 * mixed[beta] / length;
                    diamagnetic[slot][beta] = alpha2
                        * (direction[0] * d[0] + direction[1] * d[1] + direction[2] * d[2])
                        / length;
                    shielding[slot][beta] = paramagnetic[slot][beta] + diamagnetic[slot][beta];
                }
                columns.push((field[slot], shielding[slot]));
            }
            points.push(ShieldingPoint {
                wave: *wave,
                vector: q,
                polarization: [first, second],
                field,
                shielding,
                paramagnetic,
                diamagnetic,
            });
        }
        // sigma b_j = s_j in the least-squares sense: sigma = (S B^T)(B B^T)^-1.
        let mut gram = [[0.0; 3]; 3];
        let mut mixed = [[0.0; 3]; 3];
        for (b, s) in &columns {
            for i in 0..3 {
                for j in 0..3 {
                    gram[i][j] += b[i] * b[j];
                    mixed[i][j] += s[i] * b[j];
                }
            }
        }
        let determinant = gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
            - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
            + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0]);
        let tensor = (determinant.abs() > 1e-8).then(|| {
            let mut inverse = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let (a, b) = ((j + 1) % 3, (j + 2) % 3);
                    let (c, d) = ((i + 1) % 3, (i + 2) % 3);
                    inverse[i][j] =
                        (gram[a][c] * gram[b][d] - gram[a][d] * gram[b][c]) / determinant;
                }
            }
            let mut out = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        out[i][j] += mixed[i][k] * inverse[k][j];
                    }
                }
            }
            out
        });
        nuclei.push(NucleusShielding {
            atom,
            points,
            isotropic: tensor.map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.0),
            tensor,
        });
    }
    Ok(LatticeShielding {
        nuclei,
        reference: series_reference(&reference, result, replicas),
    })
}
