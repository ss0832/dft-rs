// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Ewald summation: the electrostatic energy of point charges under
//! periodic boundary conditions.
//!
//! # The problem
//!
//! The lattice sum of `1/r` is conditionally convergent — its value depends
//! on the order of summation, so "sum over shells until it stops changing"
//! is not merely slow but ill-defined. Ewald's resolution is to split each
//! point charge into a screened point plus a compensating Gaussian:
//!
//! ```text
//! 1/r = erfc(alpha r)/r  +  erf(alpha r)/r
//! ```
//!
//! The first term decays like a Gaussian and is summed in real space; the
//! second is smooth, so its lattice sum converges rapidly in reciprocal
//! space. Both halves are absolutely convergent, and `alpha` — which
//! decides only how the work is divided between them — must drop out of the
//! total. That independence is the sharpest correctness check available
//! here and is what the tests below exercise.
//!
//! # Charge neutrality
//!
//! A charged periodic cell has infinite energy. The standard treatment adds
//! a uniform compensating background, which contributes the `G = 0` term
//! `-pi Q^2 / (2 alpha^2 V)`. It is included, so a charged cell returns the
//! conventional (background-corrected) value rather than a meaningless one.

use crate::math::Vec3;
use crate::periodic::Lattice;

/// Real- and reciprocal-space cutoffs, and the splitting parameter.
#[derive(Clone, Copy, Debug)]
pub struct EwaldParams {
    /// Splitting parameter (1/Bohr). Larger shifts work to reciprocal space.
    pub alpha: f64,
    /// Real-space cutoff (Bohr).
    pub rcut: f64,
    /// Reciprocal-space cutoff (1/Bohr).
    pub gcut: f64,
}

impl EwaldParams {
    /// Cutoffs chosen for a target accuracy from the cell volume.
    ///
    /// `alpha` is set so real and reciprocal sums cost about the same, then
    /// each cutoff is taken far enough out that its neglected tail is below
    /// `eps`. The formulas are the usual asymptotic estimates: the real
    /// tail dies as `erfc(alpha r)` and the reciprocal one as
    /// `exp(-G^2/4 alpha^2)`.
    pub fn for_accuracy(lattice: &Lattice, natoms: usize, eps: f64) -> Self {
        let v = lattice.volume().abs();
        let alpha = (natoms as f64 * std::f64::consts::PI.powi(3) / (v * v)).powf(1.0 / 6.0);
        let ln = (1.0 / eps).ln().max(1.0);
        Self {
            alpha,
            rcut: ln.sqrt() / alpha,
            gcut: 2.0 * alpha * ln.sqrt(),
        }
    }
}

/// Electrostatic energy per cell of point charges `q` at positions `r`,
/// with a neutralising background when the cell is charged.
pub fn ewald_energy(lattice: &Lattice, positions: &[Vec3], charges: &[f64], p: EwaldParams) -> f64 {
    let points: Vec<_> = positions.iter().map(|r| r.to_array()).collect();
    ewald_energy_scalar(lattice, &points, charges, p, |x| (x.cos(), x.sin()), erfc)
}

/// Fixed-cell point-source energy on a moving-centre scalar path. This is
/// the same screened real/reciprocal/background decomposition as SCF.
pub(crate) fn ewald_energy_scalar<S: crate::dual::Scalar>(
    lattice: &Lattice,
    positions: &[[S; 3]],
    charges: &[f64],
    p: EwaldParams,
    cos_sin: impl Fn(S) -> (S, S),
    erfc: impl Fn(S) -> S,
) -> S {
    assert_eq!(positions.len(), charges.len());
    let reference: Vec<_> = positions
        .iter()
        .map(|r| Vec3::new(r[0].value(), r[1].value(), r[2].value()))
        .collect();
    let v = lattice.volume().abs();
    let dmax = reference
        .iter()
        .flat_map(|a| reference.iter().map(move |b| (*a - *b).norm()))
        .fold(0.0, f64::max);
    let mut real = S::from_f64(0.0);
    for translation in lattice.translations_within(p.rcut + dmax) {
        let home = translation.norm2() < 1e-20;
        for i in 0..positions.len() {
            for j in 0..positions.len() {
                if home && i == j {
                    continue;
                }
                let delta: [S; 3] = std::array::from_fn(|a| {
                    positions[i][a] - positions[j][a] + S::from_f64(translation.get(a))
                });
                let r = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                if r.value() > p.rcut || r.value() < 1e-12 {
                    continue;
                }
                real = real + (erfc(r.scale(p.alpha)) / r).scale(0.5 * charges[i] * charges[j]);
            }
        }
    }
    let mut reciprocal = S::from_f64(0.0);
    for g in lattice.reciprocal_within(p.gcut) {
        if g.norm2() < 1e-20 {
            continue;
        }
        let mut cosine = S::from_f64(0.0);
        let mut sine = cosine;
        for (r, q) in positions.iter().zip(charges) {
            let theta = r[0].scale(g.x) + r[1].scale(g.y) + r[2].scale(g.z);
            let (c, s) = cos_sin(theta);
            cosine = cosine + c.scale(*q);
            sine = sine + s.scale(*q);
        }
        reciprocal = reciprocal
            + (cosine * cosine + sine * sine).scale(
                2.0 * std::f64::consts::PI / v * (-g.norm2() / (4.0 * p.alpha * p.alpha)).exp()
                    / g.norm2(),
            );
    }
    let q2: f64 = charges.iter().map(|q| q * q).sum();
    let total: f64 = charges.iter().sum();
    real + reciprocal
        + S::from_f64(
            -p.alpha / std::f64::consts::PI.sqrt() * q2
                - std::f64::consts::PI * total * total / (2.0 * p.alpha * p.alpha * v),
        )
}

/// Electrostatic potential at `r` from the periodic array of charges plus
/// its neutralising background.
///
/// This is the same split [`ewald_energy`] uses, evaluated at a point
/// instead of contracted with the charges. Two conventions are worth
/// stating because they are what make the result well defined:
///
/// - if `r` coincides with one of the charge sites, that charge's own bare
///   Coulomb term is excluded (it is infinite) and the smooth `erf` half
///   that the reciprocal sum supplies for it is subtracted, so the value is
///   the potential from everything *else*;
/// - the `G = 0` term is the background one, `-pi Q / (alpha^2 V)`, the same
///   convention [`ewald_energy`] carries.
///
/// Together those make `E = 1/2 sum_i q_i phi(R_i)` reproduce
/// [`ewald_energy`] exactly, which is the test below and the reason this
/// function can be trusted as a reference for the periodic nuclear
/// attraction.
pub fn ewald_potential(
    lattice: &Lattice,
    positions: &[Vec3],
    charges: &[f64],
    r: Vec3,
    p: EwaldParams,
) -> f64 {
    assert_eq!(positions.len(), charges.len());
    let v = lattice.volume().abs();
    let alpha = p.alpha;
    let sqrt_pi = std::f64::consts::PI.sqrt();

    // Same trap as in `ewald_energy`: the cutoff is on the distance from
    // `r`, not on the translation length.
    let dmax = positions
        .iter()
        .map(|a| (*a - r).norm())
        .fold(0.0_f64, f64::max);
    let mut phi = 0.0;
    for t in lattice.translations_within(p.rcut + dmax) {
        for (pos, q) in positions.iter().zip(charges) {
            let d = r - *pos - t;
            let dist = d.norm();
            if dist < 1e-10 {
                phi -= q * 2.0 * alpha / sqrt_pi;
                continue;
            }
            if dist > p.rcut {
                continue;
            }
            phi += q * erfc(alpha * dist) / dist;
        }
    }

    for g in lattice.reciprocal_within(p.gcut) {
        let g2 = g.norm2();
        if g2 < 1e-20 {
            continue;
        }
        let mut s = 0.0;
        for (pos, q) in positions.iter().zip(charges) {
            s += q * g.dot(r - *pos).cos();
        }
        phi += (4.0 * std::f64::consts::PI / v) * (-g2 / (4.0 * alpha * alpha)).exp() / g2 * s;
    }

    let qtot: f64 = charges.iter().sum();
    phi -= std::f64::consts::PI * qtot / (alpha * alpha * v);
    phi
}

/// Makov–Payne correction for a CHARGED cell.
///
/// [`ewald_energy`] already includes the uniform neutralising background,
/// which is what makes a charged cell's energy finite at all. What it
/// cannot remove is the spurious interaction of the charge with its own
/// periodic images — an artefact of the boundary conditions that survives
/// to infinite cell size only as `1/L`, so it is often the largest error
/// in a charged-defect or molecular-ion calculation.
///
/// The expansion is
///
/// ```text
/// E_corr = q^2 alpha_M / (2 L)   [monopole]
///        + 2 pi q Q_r / (3 V)    [quadrupole]
/// ```
///
/// with `alpha_M` the Madelung constant of the lattice, `L = V^{1/3}`, and
/// `Q_r = Int rho(r) |r - r_0|^2 dr` the second radial moment of the TOTAL
/// charge density — nuclei AND electrons.
///
/// **Only the monopole term is computed here**, because it is the only one
/// determined by the lattice and the net charge alone. The quadrupole term
/// needs the converged electron density: for a molecular ion the second
/// radial moment is dominated by the diffuse electronic tail, not by where
/// the nuclei sit, so estimating it from nuclear positions would be a
/// number with the right units and the wrong value. Pass `q_r` to
/// [`makov_payne_with_moment`] once a density is available.
///
/// The monopole term alone is the standard leading correction and removes
/// the `1/L` error; the quadrupole term is `1/L^3` and matters at the next
/// order.
///
/// This is an asymptotic estimate, not an identity: exact in the
/// large-cell limit, approximate at realistic sizes.
pub fn makov_payne_correction(lattice: &Lattice, charge: f64) -> f64 {
    makov_payne_with_moment(lattice, charge, 0.0)
}

/// Ordered affine derivative of the charged-cell monopole correction.
pub(crate) fn monopole_strain(lattice: &Lattice, charge: f64) -> [[f64; 3]; 3] {
    if charge.abs() < 1e-14 {
        return [[0.0; 3]; 3];
    }
    // The correction is minus the Ewald energy of a unit charge and its
    // compensating background, multiplied by the square of the net charge.
    let settings = EwaldParams::for_accuracy(lattice, 1, 1e-13);
    ewald_stress(lattice, &[Vec3::zero()], &[1.0], settings)
        .map(|row| row.map(|value| -charge * charge * value))
}

/// Ordered affine Hessian of the same unit-charge monopole correction.
/// The fixed charge has no internal displacement or density response.
pub(crate) fn monopole_strain_second(lattice: &Lattice, charge: f64) -> Vec<f64> {
    if charge.abs() < 1e-14 {
        return vec![0.0; 81];
    }
    let settings = EwaldParams::for_accuracy(lattice, 1, 1e-13);
    let (mut strain, _) = ewald_strain_second(lattice, &[Vec3::zero()], &[1.0], settings);
    for value in &mut strain {
        *value *= -charge * charge;
    }
    strain
}

/// [`makov_payne_correction`] including the quadrupole term, given the
/// second radial moment `q_r` of the total charge density about its
/// centroid.
pub fn makov_payne_with_moment(lattice: &Lattice, charge: f64, q_r: f64) -> f64 {
    if charge.abs() < 1e-14 {
        return 0.0;
    }
    let v = lattice.volume().abs();
    let l = v.cbrt();
    // Madelung constant of THIS lattice, obtained rather than tabulated: a
    // single unit charge in the cell with its background has energy
    // -alpha_M / (2 L). For a simple cubic cell this reproduces 2.8373,
    // and it is equally right for a triclinic one, where no table would
    // help.
    let p = EwaldParams::for_accuracy(lattice, 1, 1e-13);
    let e_unit = ewald_energy(lattice, &[Vec3::zero()], &[1.0], p);
    let alpha_m = -2.0 * l * e_unit;
    charge * charge * alpha_m / (2.0 * l) + 2.0 * std::f64::consts::PI * charge * q_r / (3.0 * v)
}

/// Forces `-dE/dR_i` on each charge, from [`ewald_energy`].
///
/// Both halves of the split contribute. The self term and the neutralising
/// background carry no position dependence, so they drop out — which is
/// also a useful check on the split: if a force depended on `alpha`, one
/// of those two would have been mis-assigned.
pub fn ewald_forces(
    lattice: &Lattice,
    positions: &[Vec3],
    charges: &[f64],
    p: EwaldParams,
) -> Vec<Vec3> {
    let n = positions.len();
    let v = lattice.volume().abs();
    let alpha = p.alpha;
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let mut grad = vec![Vec3::zero(); n];

    // ---- Real space ----
    // The one-half of the pair sum cancels because each atom appears in
    // both slots, so the derivative is a plain sum over partners.
    let dmax = positions
        .iter()
        .flat_map(|a| positions.iter().map(move |b| (*a - *b).norm()))
        .fold(0.0_f64, f64::max);
    for t in lattice.translations_within(p.rcut + dmax) {
        let zero = t.norm2() < 1e-20;
        for i in 0..n {
            for j in 0..n {
                if zero && i == j {
                    continue;
                }
                let d = positions[i] - positions[j] + t;
                let r = d.norm();
                if r > p.rcut || r < 1e-12 {
                    continue;
                }
                let gp = -erfc(alpha * r) / (r * r)
                    - 2.0 * alpha / sqrt_pi * (-(alpha * r).powi(2)).exp() / r;
                grad[i] += d * (charges[i] * charges[j] * gp / r);
            }
        }
    }

    // ---- Reciprocal space ----
    // d|S(G)|^2 / dR_i = 2 q_i G [ Im(S) cos(G.R_i) - Re(S) sin(G.R_i) ].
    for g in lattice.reciprocal_within(p.gcut) {
        let g2 = g.norm2();
        if g2 < 1e-20 {
            continue;
        }
        let (mut cs, mut sn) = (0.0, 0.0);
        let mut phases = Vec::with_capacity(n);
        for i in 0..n {
            let ph = g.dot(positions[i]);
            let (c, s) = (ph.cos(), ph.sin());
            cs += charges[i] * c;
            sn += charges[i] * s;
            phases.push((c, s));
        }
        let f = (2.0 * std::f64::consts::PI / v) * (-g2 / (4.0 * alpha * alpha)).exp() / g2;
        for i in 0..n {
            let (c, s) = phases[i];
            grad[i] += g * (2.0 * f * charges[i] * (sn * c - cs * s));
        }
    }

    grad.iter().map(|g| *g * -1.0).collect()
}

/// Cartesian Hessian `d2E/dR_ia dR_jb` of [`ewald_energy`] for a
/// displacement of every lattice image of each charge together (the Gamma
/// point), as a `3N x 3N` row-major matrix.
///
/// Off-diagonal atom blocks are the real-space pair Hessians
/// `-q_i q_j h(d)` with `h_ab = g'' d_a d_b / r^2 + g'/r (delta_ab - d_a d_b / r^2)`,
/// `g = erfc(alpha r)/r`, plus the reciprocal
/// `(4 pi / V) f(G) q_i q_j G_a G_b cos(G . (R_i - R_j))`. The
/// translation of all charges leaves both halves unchanged, so each
/// diagonal block is minus the sum of the off-diagonal blocks of its row.
pub fn ewald_hessian(
    lattice: &Lattice,
    positions: &[Vec3],
    charges: &[f64],
    p: EwaldParams,
) -> crate::linalg::Matrix {
    let n = positions.len();
    let v = lattice.volume().abs();
    let alpha = p.alpha;
    let c = 2.0 * alpha / std::f64::consts::PI.sqrt();
    let mut h = crate::linalg::Matrix::zeros(3 * n, 3 * n);
    let dmax = positions
        .iter()
        .flat_map(|a| positions.iter().map(move |b| (*a - *b).norm()))
        .fold(0.0_f64, f64::max);
    let translations = lattice.translations_within(p.rcut + dmax);
    let gs = lattice.reciprocal_within(p.gcut);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut block = [[0.0; 3]; 3];
            // Real space: d2/dR_i dR_j of q_i q_j g(|R_i - R_j + T|).
            for t in &translations {
                let d = positions[i] - positions[j] + *t;
                let r = d.norm();
                if r > p.rcut || r < 1e-12 {
                    continue;
                }
                let e = (-(alpha * r).powi(2)).exp();
                let erfc_r = erfc(alpha * r);
                let g1 = -erfc_r / (r * r) - c * e / r;
                let g2 = 2.0 * erfc_r / (r * r * r)
                    + 2.0 * c * e / (r * r)
                    + 2.0 * c * alpha * alpha * e;
                for a in 0..3 {
                    for b in 0..3 {
                        let dd = d.get(a) * d.get(b) / (r * r);
                        let delta = if a == b { 1.0 } else { 0.0 };
                        block[a][b] -= charges[i] * charges[j] * (g2 * dd + g1 / r * (delta - dd));
                    }
                }
            }
            // Reciprocal space.
            for g in &gs {
                let g2 = g.norm2();
                if g2 < 1e-20 {
                    continue;
                }
                let f = (-g2 / (4.0 * alpha * alpha)).exp() / g2;
                let w = (4.0 * std::f64::consts::PI / v)
                    * f
                    * charges[i]
                    * charges[j]
                    * g.dot(positions[i] - positions[j]).cos();
                for a in 0..3 {
                    for b in 0..3 {
                        block[a][b] += w * g.get(a) * g.get(b);
                    }
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    h[(3 * i + a, 3 * j + b)] += block[a][b];
                    h[(3 * i + a, 3 * i + b)] -= block[a][b];
                }
            }
        }
    }
    h
}

/// Second strain derivatives `d2E/d(eps_ab) d(eps_cd)` (9 x 9, `[3a+b][3c+d]`)
/// and mixed strain-displacement derivatives `d2E/d(eps_ab) du_(i,c)`
/// (9 x 3N) of [`ewald_energy`], for positions and lattice strained
/// additively as `x -> (1 + eps) x` after a Gamma-point displacement.
///
/// Real-space pairs give `q_i q_j g_ac(d) d_b d_d / 2`; the reciprocal sum
/// uses `|G'|^2 = G^T (1+eps)^-1 (1+eps)^-T G` with
/// `d2|G'|^2 = 2 (G_a G_d d_bc + G_b G_c d_ad + G_a G_c d_bd)` and
/// `d2(1/det) = d_ab d_cd + d_ad d_bc`; the structure factor is strain
/// invariant and carries the displacement.
pub fn ewald_strain_second(
    lattice: &Lattice,
    positions: &[Vec3],
    charges: &[f64],
    p: EwaldParams,
) -> (Vec<f64>, Vec<f64>) {
    let n = positions.len();
    let v = lattice.volume().abs();
    let alpha = p.alpha;
    let c = 2.0 * alpha / std::f64::consts::PI.sqrt();
    let mut strain = vec![0.0; 81];
    let mut mixed = vec![0.0; 27 * n];
    let delta = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    // ---- real space ----
    let dmax = positions
        .iter()
        .flat_map(|a| positions.iter().map(move |b| (*a - *b).norm()))
        .fold(0.0_f64, f64::max);
    for t in lattice.translations_within(p.rcut + dmax) {
        let zero = t.norm2() < 1e-20;
        for i in 0..n {
            for j in 0..n {
                if zero && i == j {
                    continue;
                }
                let d = positions[i] - positions[j] + t;
                let r = d.norm();
                if r > p.rcut || r < 1e-12 {
                    continue;
                }
                let e = (-(alpha * r).powi(2)).exp();
                let erfc_r = erfc(alpha * r);
                let g1 = -erfc_r / (r * r) - c * e / r;
                let g2 = 2.0 * erfc_r / (r * r * r)
                    + 2.0 * c * e / (r * r)
                    + 2.0 * c * alpha * alpha * e;
                let qq = 0.5 * charges[i] * charges[j];
                let ga = |a: usize| g1 * d.get(a) / r;
                let gac = |a: usize, cc: usize| {
                    let dd = d.get(a) * d.get(cc) / (r * r);
                    g2 * dd + g1 / r * (delta(a, cc) - dd)
                };
                for a in 0..3 {
                    for b in 0..3 {
                        for cc in 0..3 {
                            for dd in 0..3 {
                                strain[(3 * a + b) * 9 + 3 * cc + dd] +=
                                    qq * gac(a, cc) * d.get(b) * d.get(dd);
                            }
                            // d/du_(U,c) with s = [i = U] - [j = U].
                            let value =
                                qq * (gac(a, cc) * d.get(b) + if b == cc { ga(a) } else { 0.0 });
                            mixed[(3 * a + b) * 3 * n + 3 * i + cc] += value;
                            mixed[(3 * a + b) * 3 * n + 3 * j + cc] -= value;
                        }
                    }
                }
            }
        }
    }
    // ---- reciprocal space ----
    for g in lattice.reciprocal_within(p.gcut) {
        let x = g.norm2();
        if x < 1e-20 {
            continue;
        }
        let (mut cs, mut sn) = (0.0, 0.0);
        let mut phases = Vec::with_capacity(n);
        for i in 0..n {
            let ph = g.dot(positions[i]);
            let (co, si) = (ph.cos(), ph.sin());
            cs += charges[i] * co;
            sn += charges[i] * si;
            phases.push((co, si));
        }
        let s2 = cs * cs + sn * sn;
        let k = 1.0 / (4.0 * alpha * alpha) + 1.0 / x;
        let f = (-x / (4.0 * alpha * alpha)).exp() / x;
        let f1 = -f * k;
        let f2 = f * (k * k + 1.0 / (x * x));
        let pre = 2.0 * std::f64::consts::PI / v;
        let dx = |a: usize, b: usize| -2.0 * g.get(a) * g.get(b);
        let ddx = |a: usize, b: usize, cc: usize, dd: usize| {
            2.0 * (g.get(a) * g.get(dd) * delta(b, cc)
                + g.get(b) * g.get(cc) * delta(a, dd)
                + g.get(a) * g.get(cc) * delta(b, dd))
        };
        // h = f(x)/det: first derivative h_ab = (-delta_ab f + f' dx_ab).
        for a in 0..3 {
            for b in 0..3 {
                let h_ab = -delta(a, b) * f + f1 * dx(a, b);
                for cc in 0..3 {
                    for dd in 0..3 {
                        let second = (delta(a, b) * delta(cc, dd) + delta(a, dd) * delta(b, cc))
                            * f
                            - delta(a, b) * f1 * dx(cc, dd)
                            - delta(cc, dd) * f1 * dx(a, b)
                            + f2 * dx(a, b) * dx(cc, dd)
                            + f1 * ddx(a, b, cc, dd);
                        strain[(3 * a + b) * 9 + 3 * cc + dd] += pre * second * s2;
                    }
                }
                // d|S|^2/du_(i,c) = 2 q_i G_c [Im(S) cos - Re(S) sin].
                for (i, (co, si)) in phases.iter().enumerate() {
                    let ds2 = 2.0 * charges[i] * (sn * co - cs * si);
                    for cc in 0..3 {
                        mixed[(3 * a + b) * 3 * n + 3 * i + cc] += pre * h_ab * ds2 * g.get(cc);
                    }
                }
            }
        }
    }
    // ---- background -pi Q^2 / (2 alpha^2 V) ----
    let qtot: f64 = charges.iter().sum();
    let e_bg = -std::f64::consts::PI * qtot * qtot / (2.0 * alpha * alpha * v);
    for a in 0..3 {
        for b in 0..3 {
            for cc in 0..3 {
                for dd in 0..3 {
                    strain[(3 * a + b) * 9 + 3 * cc + dd] +=
                        e_bg * (delta(a, b) * delta(cc, dd) + delta(a, dd) * delta(b, cc));
                }
            }
        }
    }
    (strain, mixed)
}

/// Strain derivative `dE/d(epsilon_ab)` of [`ewald_energy`], a 3x3 tensor.
///
/// Under a strain the positions and lattice vectors both transform as
/// `r -> (1 + epsilon) r`, which makes the reciprocal vectors transform
/// contravariantly, `G -> (1 - epsilon^T) G`, and the volume scale as
/// `V (1 + tr epsilon)`. Two consequences make the algebra tidy:
///
/// - `G . r` is INVARIANT, so the structure factors do not vary and only
///   the kernel and the volume do;
/// - the self term has no length in it at all, so it does not contribute.
///
/// The stress tensor is this divided by the cell volume. Returned as the
/// strain derivative rather than the stress so the caller decides the
/// sign and volume convention, which differ between communities.
pub fn ewald_stress(
    lattice: &Lattice,
    positions: &[Vec3],
    charges: &[f64],
    p: EwaldParams,
) -> [[f64; 3]; 3] {
    let n = positions.len();
    let v = lattice.volume().abs();
    let alpha = p.alpha;
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let mut out = [[0.0_f64; 3]; 3];

    // ---- Real space ----
    // d r / d epsilon_ab = r_a r_b / r, so each pair contributes
    // g'(r) r_a r_b / r with g(r) = erfc(alpha r)/r.
    let dmax = positions
        .iter()
        .flat_map(|a| positions.iter().map(move |b| (*a - *b).norm()))
        .fold(0.0_f64, f64::max);
    for t in lattice.translations_within(p.rcut + dmax) {
        let zero = t.norm2() < 1e-20;
        for i in 0..n {
            for j in 0..n {
                if zero && i == j {
                    continue;
                }
                let d = positions[i] - positions[j] + t;
                let r = d.norm();
                if r > p.rcut || r < 1e-12 {
                    continue;
                }
                let gp = -erfc(alpha * r) / (r * r)
                    - 2.0 * alpha / sqrt_pi * (-(alpha * r).powi(2)).exp() / r;
                let w = 0.5 * charges[i] * charges[j] * gp / r;
                for a in 0..3 {
                    for b in 0..3 {
                        out[a][b] += w * d.get(a) * d.get(b);
                    }
                }
            }
        }
    }

    // ---- Reciprocal space ----
    // The kernel f = exp(-G^2/4 alpha^2)/G^2 varies through G^2 alone, and
    // d(G^2)/d epsilon_ab = -2 G_a G_b.
    let mut e_recip = 0.0;
    for g in lattice.reciprocal_within(p.gcut) {
        let g2 = g.norm2();
        if g2 < 1e-20 {
            continue;
        }
        let (mut cs, mut sn) = (0.0, 0.0);
        for i in 0..n {
            let phase = g.dot(positions[i]);
            cs += charges[i] * phase.cos();
            sn += charges[i] * phase.sin();
        }
        let s2 = cs * cs + sn * sn;
        let f = (-g2 / (4.0 * alpha * alpha)).exp() / g2;
        let term = (2.0 * std::f64::consts::PI / v) * f * s2;
        e_recip += term;
        let scale = 2.0 * (1.0 / (4.0 * alpha * alpha) + 1.0 / g2);
        for a in 0..3 {
            for b in 0..3 {
                out[a][b] += term * scale * g.get(a) * g.get(b);
            }
        }
    }
    // Volume factors: the reciprocal prefactor carries 1/V and the
    // background 1/V as well, so both pick up -delta_ab from tr epsilon.
    let qtot: f64 = charges.iter().sum();
    let e_bg = -std::f64::consts::PI * qtot * qtot / (2.0 * alpha * alpha * v);
    for a in 0..3 {
        out[a][a] -= e_recip + e_bg;
    }
    out
}

/// Complementary error function.
///
/// `libm` is the only numerics dependency the project allows, and it
/// carries this; the local wrapper keeps the call site readable.
fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

#[cfg(test)]
mod tests {

    /// Strain Hessian and strain-displacement derivatives against four-step
    /// fits of central differences of the analytic strain derivative.
    #[test]
    fn ewald_strain_second_derivatives_pass_four_step_fits() {
        let base = [
            Vec3::new(5.1, 0.2, 0.0),
            Vec3::new(-0.4, 4.7, 0.3),
            Vec3::new(0.1, 0.5, 5.6),
        ];
        let positions = vec![
            Vec3::new(0.2, 0.1, -0.3),
            Vec3::new(2.4, 1.9, 2.2),
            Vec3::new(1.1, 3.3, 0.7),
        ];
        let charges = vec![1.5, -0.7, 0.4];
        let lattice0 = Lattice::new(base[0], base[1], base[2]);
        let p = EwaldParams::for_accuracy(&lattice0, 3, 1e-14);
        let deform = |v: Vec3, e: &[[f64; 3]; 3]| {
            Vec3::new(
                v.x + e[0][0] * v.x + e[0][1] * v.y + e[0][2] * v.z,
                v.y + e[1][0] * v.x + e[1][1] * v.y + e[1][2] * v.z,
                v.z + e[2][0] * v.x + e[2][1] * v.y + e[2][2] * v.z,
            )
        };
        let first = |e: &[[f64; 3]; 3], atom: usize, axis: usize, u: f64| {
            let lattice = Lattice::new(deform(base[0], e), deform(base[1], e), deform(base[2], e));
            let x: Vec<Vec3> = positions
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let mut r = *r;
                    if i == atom {
                        match axis {
                            0 => r.x += u,
                            1 => r.y += u,
                            _ => r.z += u,
                        }
                    }
                    deform(r, e)
                })
                .collect();
            ewald_stress(&lattice, &x, &charges, p)
        };
        let (strain, mixed) = ewald_strain_second(&lattice0, &positions, &charges, p);
        let zero = [[0.0; 3]; 3];
        let s0 = first(&zero, 0, 0, 0.0);
        let steps: [f64; 4] = [4e-3, 2e-3, 1e-3, 5e-4];
        let fit = |errors: &[f64]| {
            let (x, y): (Vec<f64>, Vec<f64>) = steps
                .iter()
                .zip(errors)
                .map(|(h, e)| (h.ln(), e.ln()))
                .unzip();
            let n = 4.0;
            let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
            let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
            let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
            let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
            (sxy / sxx, sxy * sxy / (sxx * syy))
        };
        for (c, d) in [(0usize, 0usize), (0, 1), (2, 1)] {
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| {
                    let mut ep = zero;
                    ep[c][d] = h;
                    let mut em = zero;
                    em[c][d] = -h;
                    let (plus, minus) = (first(&ep, 0, 0, 0.0), first(&em, 0, 0, 0.0));
                    let mut worst = 0.0_f64;
                    for a in 0..3 {
                        for b in 0..3 {
                            // A strain composed on top of eps adds delta_bc S_ad.
                            let composed = if b == c { s0[a][d] } else { 0.0 };
                            let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                            worst = worst
                                .max((fd - strain[(3 * a + b) * 9 + 3 * c + d] - composed).abs());
                        }
                    }
                    worst
                })
                .collect();
            let (slope, r2) = fit(&errors);
            println!("Ewald strain-strain column [{c}][{d}]: errors {errors:?} p = {slope:.6} R2 = {r2:.8}");
            assert!((slope - 2.0).abs() < 0.05 && r2 > 0.9999);
        }
        for (atom, axis) in [(0usize, 0usize), (2, 1)] {
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| {
                    let (plus, minus) = (first(&zero, atom, axis, h), first(&zero, atom, axis, -h));
                    let mut worst = 0.0_f64;
                    for a in 0..3 {
                        for b in 0..3 {
                            let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                            worst =
                                worst.max((fd - mixed[(3 * a + b) * 9 + 3 * atom + axis]).abs());
                        }
                    }
                    worst
                })
                .collect();
            let (slope, r2) = fit(&errors);
            println!("Ewald strain-displacement column ({atom},{axis}): errors {errors:?} p = {slope:.6} R2 = {r2:.8}");
            assert!((slope - 2.0).abs() < 0.05 && r2 > 0.9999);
        }
    }

    /// Analytic Gamma Hessian against a four-step fit of central
    /// differences of the analytic forces, with the acoustic sum rule.
    #[test]
    fn ewald_hessian_passes_a_four_step_fit_and_the_acoustic_sum_rule() {
        let lattice = Lattice::new(
            Vec3::new(5.1, 0.2, 0.0),
            Vec3::new(-0.4, 4.7, 0.3),
            Vec3::new(0.1, 0.5, 5.6),
        );
        let positions = vec![
            Vec3::new(0.2, 0.1, -0.3),
            Vec3::new(2.4, 1.9, 2.2),
            Vec3::new(1.1, 3.3, 0.7),
        ];
        let charges = vec![1.5, -0.7, 0.4];
        let p = EwaldParams::for_accuracy(&lattice, 3, 1e-14);
        let h = ewald_hessian(&lattice, &positions, &charges, p);
        for row in 0..9 {
            for b in 0..3 {
                let total: f64 = (0..3).map(|j| h[(row, 3 * j + b)]).sum();
                assert!(
                    total.abs() < 1e-12,
                    "acoustic sum rule row {row}: {total:e}"
                );
            }
            for col in 0..9 {
                assert!((h[(row, col)] - h[(col, row)]).abs() < 1e-12);
            }
        }
        let steps = [4e-3, 2e-3, 1e-3, 5e-4];
        for (atom, axis, other) in [(0usize, 0usize, 1usize), (1, 2, 2), (2, 1, 0), (1, 0, 1)] {
            let mut errors = Vec::new();
            for &step in &steps {
                let moved = |sign: f64| {
                    let mut x = positions.clone();
                    match axis {
                        0 => x[atom].x += sign * step,
                        1 => x[atom].y += sign * step,
                        _ => x[atom].z += sign * step,
                    }
                    ewald_forces(&lattice, &x, &charges, p)
                };
                let (plus, minus) = (moved(1.0), moved(-1.0));
                let mut worst = 0.0_f64;
                for b in 0..3 {
                    // H = -dF/dR.
                    let fd = -(plus[other].get(b) - minus[other].get(b)) / (2.0 * step);
                    worst = worst.max((fd - h[(3 * other + b, 3 * atom + axis)]).abs());
                }
                errors.push(worst);
            }
            let (x, y): (Vec<f64>, Vec<f64>) = steps
                .iter()
                .zip(&errors)
                .map(|(h, e)| (h.ln(), e.ln()))
                .unzip();
            let n = 4.0;
            let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
            let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
            let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
            let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
            let (slope, r2) = (sxy / sxx, sxy * sxy / (sxx * syy));
            println!("Ewald Hessian column ({atom},{axis}) row atom {other}: errors {errors:?} p = {slope:.6} R2 = {r2:.8}");
            assert!((slope - 2.0).abs() < 0.05 && r2 > 0.9999);
        }
    }
    use super::*;

    /// Rocksalt at unit spacing: the Madelung constant is a known number,
    /// `-1.747564594633...` per ion pair, and reproducing it to eight
    /// digits pins the whole construction — real sum, reciprocal sum, self
    /// term and their relative signs at once.
    ///
    /// The NaCl conventional cubic cell of edge 2 (nearest-neighbour
    /// distance 1) holds four formula units, so the energy per cell is
    /// `4 * (-M)` with the ion charges +-1.
    #[test]
    fn madelung_constant_of_rocksalt() {
        let lat = Lattice::cubic(2.0);
        let mut pos = Vec::new();
        let mut q = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    pos.push(Vec3::new(i as f64, j as f64, k as f64));
                    // Alternating sign on the simple-cubic sublattice IS
                    // the rocksalt structure.
                    q.push(if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 });
                }
            }
        }
        let p = EwaldParams {
            alpha: 2.0,
            rcut: 12.0,
            gcut: 30.0,
        };
        let e = ewald_energy(&lat, &pos, &q, p);
        // Eight ions = four formula units.
        let madelung = -e / 4.0;
        assert!(
            (madelung - 1.747_564_594_633).abs() < 1e-8,
            "Madelung constant {madelung}, expected 1.747564594633"
        );
    }

    /// The splitting parameter divides work between the two sums and must
    /// cancel from the total. This is the check that catches a wrong
    /// self-term or a missing background: those are `alpha`-dependent, so
    /// they show up here even when a single `alpha` happens to look right.
    #[test]
    fn energy_is_independent_of_the_splitting_parameter() {
        let lat = Lattice::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.2, 4.6, 0.0),
            Vec3::new(0.3, -0.7, 5.4),
        );
        let pos = vec![
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(2.4, 1.1, 3.0),
            Vec3::new(3.9, 3.3, 1.2),
        ];
        let q = vec![1.0, -2.0, 1.0];
        let mut vals = Vec::new();
        for alpha in [0.8, 1.2, 1.8, 2.5] {
            // Both cutoffs scale WITH alpha so each tail stays near 1e-14
            // regardless: the real tail dies as erfc(alpha rcut) and the
            // reciprocal one as exp(-gcut^2/4 alpha^2). Fixing the cutoffs
            // instead would make this a test of truncation error rather
            // than of the splitting cancelling.
            let p = EwaldParams {
                alpha,
                rcut: 7.0 / alpha,
                gcut: 12.0 * alpha,
            };
            vals.push(ewald_energy(&lat, &pos, &q, p));
        }
        let spread = vals.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v))
            - vals.iter().fold(f64::INFINITY, |m, &v| m.min(v));
        assert!(
            spread < 1e-9,
            "energy depends on alpha: values {vals:?}, spread {spread:e}"
        );
    }

    /// A supercell holds an integer number of primitive cells, so its
    /// energy must be that multiple of the primitive value. This is the
    /// extensivity check — it catches a background or self term that was
    /// normalised per cell instead of per charge.
    #[test]
    fn supercell_energy_is_extensive() {
        let lat = Lattice::cubic(2.0);
        let mut pos = Vec::new();
        let mut q = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    pos.push(Vec3::new(i as f64, j as f64, k as f64));
                    q.push(if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 });
                }
            }
        }
        let p = EwaldParams {
            alpha: 2.0,
            rcut: 12.0,
            gcut: 30.0,
        };
        let e1 = ewald_energy(&lat, &pos, &q, p);

        let n = [2usize, 1, 1];
        let sc = lat.supercell(n);
        let mut spos = Vec::new();
        let mut sq = Vec::new();
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let shift = lat.translation([i as i64, j as i64, k as i64]);
                    for (r, c) in pos.iter().zip(&q) {
                        spos.push(*r + shift);
                        sq.push(*c);
                    }
                }
            }
        }
        let e2 = ewald_energy(&sc, &spos, &sq, p);
        let mult = (n[0] * n[1] * n[2]) as f64;
        assert!(
            (e2 - mult * e1).abs() < 1e-8 * e1.abs(),
            "supercell energy {e2} is not {mult}x the primitive {e1}"
        );
    }

    /// Translating the whole cell cannot change the energy.
    #[test]
    fn energy_is_translation_invariant() {
        let lat = Lattice::cubic(4.0);
        let pos = vec![Vec3::new(0.3, 0.1, 0.2), Vec3::new(2.1, 2.4, 1.7)];
        let q = vec![1.0, -1.0];
        let p = EwaldParams {
            alpha: 1.5,
            rcut: 14.0,
            gcut: 22.0,
        };
        let e0 = ewald_energy(&lat, &pos, &q, p);
        let shift = Vec3::new(1.234, -0.567, 2.891);
        let moved: Vec<Vec3> = pos.iter().map(|r| *r + shift).collect();
        let e1 = ewald_energy(&lat, &moved, &q, p);
        assert!(
            (e1 - e0).abs() < 1e-10,
            "translating the cell changed the energy: {e0} vs {e1}"
        );
    }

    /// The Madelung constant extracted inside the Makov–Payne correction
    /// must be the textbook one for a simple cubic lattice, 2.8373. That
    /// number is what sets the size of the leading charged-cell error, so
    /// getting it right matters more than the correction's own accuracy.
    #[test]
    fn simple_cubic_madelung_constant_from_a_unit_charge() {
        for l in [4.0_f64, 7.0, 11.0] {
            let lat = Lattice::cubic(l);
            let p = EwaldParams::for_accuracy(&lat, 1, 1e-13);
            let e = ewald_energy(&lat, &[Vec3::zero()], &[1.0], p);
            let alpha_m = -2.0 * lat.volume().cbrt() * e;
            assert!(
                (alpha_m - 2.837_297).abs() < 1e-4,
                "simple-cubic Madelung constant {alpha_m} at L = {l}, expected 2.8373"
            );
        }
    }

    /// `E = 1/2 sum_i q_i phi(R_i)` must reproduce [`ewald_energy`] to
    /// machine precision. Every convention in the potential is exercised at
    /// once: the excluded bare self term, the `erf` self subtraction that
    /// replaces it, the reciprocal prefactor (`4 pi` here against `2 pi` in
    /// the energy) and the background. A slip in any one of them shows up
    /// here and nowhere else.
    #[test]
    fn potential_at_the_sites_reproduces_the_energy() {
        let lat = Lattice::new(
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(1.1, 6.4, 0.0),
            Vec3::new(0.3, -0.8, 7.7),
        );
        let pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.1, 1.2, 0.4),
            Vec3::new(1.0, 4.0, 5.0),
        ];
        // Deliberately charged, so the background term is under test too.
        let q = vec![1.0, -2.0, 0.5];
        let p = EwaldParams {
            alpha: 0.42,
            rcut: 18.0,
            gcut: 8.0,
        };
        let e = ewald_energy(&lat, &pos, &q, p);
        let from_phi: f64 = pos
            .iter()
            .zip(&q)
            .map(|(r, qi)| 0.5 * qi * ewald_potential(&lat, &pos, &q, *r, p))
            .sum();
        assert!(
            (e - from_phi).abs() < 1e-10,
            "energy from the potential {from_phi} differs from {e}"
        );
    }

    /// The potential at a generic point must not depend on how the work is
    /// divided between the two halves of the split.
    #[test]
    fn potential_is_independent_of_the_splitting_parameter() {
        let lat = Lattice::cubic(9.0);
        let pos = vec![Vec3::new(0.2, 0.1, 0.0), Vec3::new(4.0, 3.0, 2.0)];
        let q = vec![2.0, -2.0];
        let probe = Vec3::new(1.3, -0.7, 2.9);
        let mut vals = Vec::new();
        for alpha in [0.3_f64, 0.5, 0.8] {
            let p = EwaldParams {
                alpha,
                rcut: 5.0 / alpha,
                gcut: 10.0 * alpha,
            };
            vals.push(ewald_potential(&lat, &pos, &q, probe, p));
        }
        for w in vals.windows(2) {
            assert!(
                (w[0] - w[1]).abs() < 1e-9,
                "potential depends on alpha: {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    /// A neutral cell needs no charged-cell correction, and must get
    /// exactly zero rather than a small residue.
    #[test]
    fn makov_payne_vanishes_for_a_neutral_cell() {
        let lat = Lattice::cubic(8.0);
        assert_eq!(makov_payne_correction(&lat, 0.0), 0.0);
        assert_eq!(makov_payne_with_moment(&lat, 0.0, 12.3), 0.0);
    }

    /// The correction must remove the leading cell-size dependence of a
    /// charged system. Uncorrected, the energy of a single charge drifts
    /// as `1/L`; corrected, the drift across a range of cell sizes must be
    /// far smaller. That is the property the correction exists for, and it
    /// is checkable without any reference value.
    #[test]
    fn makov_payne_flattens_the_cell_size_dependence() {
        let pos = vec![Vec3::zero()];
        let q = vec![1.0];
        let mut raw = Vec::new();
        let mut corrected = Vec::new();
        for l in [8.0_f64, 12.0, 16.0, 20.0] {
            let lat = Lattice::cubic(l);
            let p = EwaldParams::for_accuracy(&lat, 1, 1e-13);
            let e = ewald_energy(&lat, &pos, &q, p);
            raw.push(e);
            corrected.push(e + makov_payne_correction(&lat, q[0]));
        }
        let spread = |v: &[f64]| {
            v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - v.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        let (sr, sc) = (spread(&raw), spread(&corrected));
        assert!(
            sc < 1e-9 * sr.max(1e-12) + 1e-9,
            "the correction did not flatten the size dependence: raw spread {sr:e}, \
             corrected {sc:e}"
        );
    }

    /// Forces against central differences of the energy, and their sum
    /// against zero.
    ///
    /// The sum rule is the stronger of the two here: translating every
    /// charge together cannot change a lattice sum, so the forces must
    /// cancel exactly whatever the cutoffs are. A reciprocal-space term
    /// attributed to the wrong atom passes the finite-difference check on
    /// some components and fails this one on all of them.
    #[test]
    fn forces_match_finite_differences_and_sum_to_zero() {
        let lat = Lattice::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.2, 4.6, 0.0),
            Vec3::new(0.3, -0.7, 5.4),
        );
        let pos = vec![
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(2.4, 1.1, 3.0),
            Vec3::new(3.9, 3.3, 1.2),
        ];
        let q = vec![1.0, -2.0, 1.0];
        let p = EwaldParams {
            alpha: 1.4,
            rcut: 7.0 / 1.4,
            gcut: 12.0 * 1.4,
        };
        let f = ewald_forces(&lat, &pos, &q, p);

        let sum = f.iter().fold(Vec3::zero(), |a, b| a + *b);
        let worst_sum = sum.x.abs().max(sum.y.abs()).max(sum.z.abs());
        assert!(worst_sum < 1e-9, "forces do not sum to zero: {sum:?}");

        let h = 1e-5;
        let mut worst = 0.0_f64;
        for i in 0..pos.len() {
            for ax in 0..3 {
                let at = |s: f64| -> f64 {
                    let mut r = pos.clone();
                    match ax {
                        0 => r[i].x += s,
                        1 => r[i].y += s,
                        _ => r[i].z += s,
                    }
                    ewald_energy(&lat, &r, &q, p)
                };
                let fd = -(at(h) - at(-h)) / (2.0 * h);
                worst = worst.max((f[i].get(ax) - fd).abs());
            }
        }
        assert!(
            worst < 1e-7,
            "Ewald forces vs finite differences: {worst:e}"
        );
    }

    /// The analytic strain derivative against central differences of the
    /// energy under an applied strain. A strain moves the lattice AND the
    /// positions together, which is the part easy to get half-right: a
    /// stress that strains only the cell is a different (and wrong)
    /// derivative, and it looks plausible.
    #[test]
    fn stress_matches_finite_differences() {
        let lat = Lattice::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.2, 4.6, 0.0),
            Vec3::new(0.3, -0.7, 5.4),
        );
        let pos = vec![
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(2.4, 1.1, 3.0),
            Vec3::new(3.9, 3.3, 1.2),
        ];
        let q = vec![1.0, -2.0, 1.0];
        let p = EwaldParams {
            alpha: 1.4,
            rcut: 7.0 / 1.4,
            gcut: 12.0 * 1.4,
        };
        let analytic = ewald_stress(&lat, &pos, &q, p);

        let apply = |eps: [[f64; 3]; 3]| -> (Lattice, Vec<Vec3>) {
            let strain = |v: Vec3| -> Vec3 {
                Vec3::new(
                    v.x + eps[0][0] * v.x + eps[0][1] * v.y + eps[0][2] * v.z,
                    v.y + eps[1][0] * v.x + eps[1][1] * v.y + eps[1][2] * v.z,
                    v.z + eps[2][0] * v.x + eps[2][1] * v.y + eps[2][2] * v.z,
                )
            };
            (
                Lattice::new(strain(lat.a[0]), strain(lat.a[1]), strain(lat.a[2])),
                pos.iter().map(|r| strain(*r)).collect(),
            )
        };
        let h = 1e-5;
        let mut worst = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let mut ep = [[0.0; 3]; 3];
                ep[a][b] = h;
                let (lp, rp) = apply(ep);
                let mut em = [[0.0; 3]; 3];
                em[a][b] = -h;
                let (lm, rm) = apply(em);
                let fd =
                    (ewald_energy(&lp, &rp, &q, p) - ewald_energy(&lm, &rm, &q, p)) / (2.0 * h);
                worst = worst.max((analytic[a][b] - fd).abs());
            }
        }
        assert!(
            worst < 1e-6,
            "Ewald stress vs finite differences: worst {worst:e}"
        );
    }

    /// A uniform scaling strain changes the energy by `-E` per unit strain
    /// for a Coulomb system, because every distance scales and `1/r` is
    /// homogeneous of degree -1. So the trace of the strain derivative must
    /// be exactly `-E` — an exact identity, independent of the cutoffs.
    #[test]
    fn stress_trace_is_minus_the_energy() {
        let lat = Lattice::cubic(2.0);
        let mut pos = Vec::new();
        let mut q = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    pos.push(Vec3::new(i as f64, j as f64, k as f64));
                    q.push(if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 });
                }
            }
        }
        let p = EwaldParams {
            alpha: 2.0,
            rcut: 12.0,
            gcut: 30.0,
        };
        let e = ewald_energy(&lat, &pos, &q, p);
        let s = ewald_stress(&lat, &pos, &q, p);
        let tr = s[0][0] + s[1][1] + s[2][2];
        assert!(
            (tr + e).abs() < 1e-8 * e.abs(),
            "trace {tr} should be -E = {}",
            -e
        );
    }

    /// Two well-separated charges in a large cell must approach the
    /// isolated-pair Coulomb energy, which is the molecular limit.
    #[test]
    fn large_cell_approaches_the_isolated_pair() {
        let l = 60.0;
        let lat = Lattice::cubic(l);
        let d = 2.0;
        let pos = vec![Vec3::zero(), Vec3::new(d, 0.0, 0.0)];
        let q = vec![1.0, -1.0];
        let p = EwaldParams::for_accuracy(&lat, 2, 1e-12);
        let e = ewald_energy(&lat, &pos, &q, p);
        let isolated = -1.0 / d;
        // The residual is the interaction with the periodic images, which
        // falls off as the cell grows; at L = 60 Bohr for a neutral dipole
        // it is small but not zero.
        assert!(
            (e - isolated).abs() < 1e-3,
            "large-cell energy {e} vs isolated pair {isolated}"
        );
    }
}
