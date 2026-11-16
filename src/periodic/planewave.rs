// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Plane-wave matrix elements `<mu| exp(i G . r) |nu>`.
//!
//! # Why this is the core of the periodic Coulomb problem
//!
//! Every conditionally convergent lattice sum in a periodic calculation is
//! handled the same way Ewald handles the point-charge one: split the
//! `1/r` kernel into a short-range piece summed in real space and a smooth
//! long-range piece summed in reciprocal space. The reciprocal half needs
//! the charge density's Fourier components, and for a Gaussian basis those
//! are these matrix elements. So `rho_mn(G)` is shared between the
//! periodic nuclear attraction, the periodic Coulomb (Hartree) term, and
//! the structure factors of any density-fitting scheme layered on later.
//!
//! # The closed form
//!
//! In the McMurchie–Davidson expansion a Gaussian product is a sum of
//! Hermite Gaussians centred on `P` with exponent `p`,
//!
//! ```text
//! chi_mu chi_nu = sum_tuv E_t E_u E_v Lambda_tuv(r - P; p)
//! ```
//!
//! and each Hermite Gaussian is a `P`-derivative of the plain one, whose
//! transform is elementary:
//!
//! ```text
//! Int exp(-p |r-P|^2) exp(i G . r) dr = (pi/p)^{3/2} exp(-G^2/4p) exp(i G . P)
//! ```
//!
//! Differentiating with respect to `P` brings down `i G`, so
//!
//! ```text
//! <mu| exp(i G . r) |nu> = (pi/p)^{3/2} exp(-G^2/4p) exp(i G . P)
//!                          * sum_tuv E_t E_u E_v (i G_x)^t (i G_y)^u (i G_z)^v
//! ```
//!
//! No new integral class and no quadrature — the same `ETable` the rest of
//! the code uses. Note what this gives for free: at `G = 0` every phase is
//! one, the exponential is one, and only `t=u=v=0` survives, so the result
//! must be exactly the overlap matrix. That is the first gate below.
//!
//! Results are returned as a `(real, imaginary)` pair, matching
//! [`super::integrals`], because the project's linear algebra is real.

use crate::basis::{cart_components, MolBasis, MolShell};
use crate::dual::Scalar;
use crate::integrals::hermite::ETable;
use crate::linalg::Matrix;
use crate::math::Vec3;
use rayon::prelude::*;
#[cfg(test)]
mod current_vertex_tests {
    use super::*;
    use crate::basis::{BasisSet, ElementBasis, Shell};
    use crate::system::{Atom, Molecule};
    fn gap(a: &(Matrix, Matrix), b: &(Matrix, Matrix)) -> f64 {
        a.0.max_abs_diff(&b.0).max(a.1.max_abs_diff(&b.1))
    }
    #[test]
    fn all_cartesian_current_vertices_have_four_step_centre_derivative_fits() {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.2, -0.3, 0.4),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.3, 0.5, -0.6),
            },
        ]);
        let mut definition = BasisSet::new("current-vertex-reference");
        definition.insert(ElementBasis {
            z: 1,
            shells: (0..=2)
                .map(|l| Shell {
                    l,
                    exps: vec![0.8, 0.35],
                    coefs: vec![0.6, 0.4],
                })
                .collect(),
        });
        let mb = MolBasis::build(&mol, &definition).unwrap();
        let wave = Vec3::new(0.31, -0.27, 0.19);
        let hs = [0.02_f64, 0.01, 0.005, 0.0025];
        for (left_index, right_index) in [(0, 3), (1, 5), (2, 4)] {
            let left = &mb.shells[left_index];
            let right = &mb.shells[right_index];
            let base = plane_wave_shell_pair(left, right, wave);
            for axis in 0..3 {
                let exact = plane_wave_current_shell_pair(left, right, wave, axis);
                let mut errors = [0.0; 4];
                let mut values = Vec::new();
                for (k, &h) in hs.iter().enumerate() {
                    let mut plus = right.clone();
                    let mut minus = right.clone();
                    let step = match axis {
                        0 => Vec3::new(h, 0.0, 0.0),
                        1 => Vec3::new(0.0, h, 0.0),
                        _ => Vec3::new(0.0, 0.0, h),
                    };
                    plus.center = right.center + step;
                    minus.center = right.center - step;
                    let p = plane_wave_shell_pair(left, &plus, wave);
                    let m = plane_wave_shell_pair(left, &minus, wave);
                    let mut value = base.clone();
                    for i in 0..base.0.rows {
                        for j in 0..base.0.cols {
                            let dr = (p.0[(i, j)] - m.0[(i, j)]) / (2.0 * h);
                            let di = (p.1[(i, j)] - m.1[(i, j)]) / (2.0 * h);
                            value.0[(i, j)] = -di + 0.5 * wave.get(axis) * base.0[(i, j)];
                            value.1[(i, j)] = dr + 0.5 * wave.get(axis) * base.1[(i, j)];
                        }
                    }
                    errors[k] = gap(&value, &exact);
                    values.push(value);
                }
                assert!(errors.iter().all(|&v| v.is_finite() && v > 0.0));
                let x = hs.map(f64::ln);
                let y = errors.map(f64::ln);
                let xm = x.iter().sum::<f64>() / 4.0;
                let ym = y.iter().sum::<f64>() / 4.0;
                let p = x
                    .iter()
                    .zip(y)
                    .map(|(x, y)| (x - xm) * (y - ym))
                    .sum::<f64>()
                    / x.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
                let r2 = 1.0
                    - x.iter()
                        .zip(y)
                        .map(|(x, y)| (y - ym - p * (x - xm)).powi(2))
                        .sum::<f64>()
                        / y.iter().map(|y| (y - ym).powi(2)).sum::<f64>();
                println!("field current l={},{} axis={axis}: errors={errors:?}, centre steps=0.02..0.0025 p={p:.12} R2={r2:.12}",left.l,right.l);
                assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
                for pair in errors.windows(2) {
                    assert!((pair[0] / pair[1] / 4.0 - 1.0).abs() < 0.03);
                }
                let mut rich = values[3].clone();
                for i in 0..rich.0.rows {
                    for j in 0..rich.0.cols {
                        rich.0[(i, j)] = (4.0 * values[3].0[(i, j)] - values[2].0[(i, j)]) / 3.0;
                        rich.1[(i, j)] = (4.0 * values[3].1[(i, j)] - values[2].1[(i, j)]) / 3.0;
                    }
                }
                assert!(gap(&rich, &exact) < 1e-8);
            }
        }
    }
}

/// `<mu| exp(i G . r) |nu>` for one primitive Cartesian pair, as
/// `(real, imaginary)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plane_wave_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    alpha: f64,
    beta: f64,
    a: Vec3,
    b: Vec3,
    g: Vec3,
) -> (f64, f64) {
    plane_wave_primitive_g(
        la,
        lb,
        alpha,
        beta,
        a.to_array(),
        b.to_array(),
        g.to_array(),
        |x| (x.cos(), x.sin()),
    )
}

/// The same primitive transform with moving centres and wave vectors. The
/// trigonometric callback must preserve every order in the scalar algebra.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plane_wave_primitive_g<S: Scalar>(
    la: [usize; 3],
    lb: [usize; 3],
    alpha: f64,
    beta: f64,
    a: [S; 3],
    b: [S; 3],
    g: [S; 3],
    cos_sin: impl Fn(S) -> (S, S),
) -> (S, S) {
    use super::strain_jets::hermite;
    let p = alpha + beta;
    let pc: [S; 3] = std::array::from_fn(|k| (a[k].scale(alpha) + b[k].scale(beta)).scale(1.0 / p));
    let tables: [Vec<S>; 3] =
        std::array::from_fn(|k| hermite(la[k], lb[k], a[k] - b[k], alpha, beta));
    let entry = |axis: usize, n: usize| {
        tables[axis][(la[axis] * (lb[axis] + 1) + lb[axis]) * (la[axis] + lb[axis] + 1) + n]
    };
    // sum_tuv E_t E_u E_v (i G_x)^t (i G_y)^u (i G_z)^v: the powers of `i`
    // rotate the term between the real and imaginary accumulators, so
    // collect by (t+u+v) mod 4 rather than forming a complex number.
    let mut acc = [S::from_f64(0.0); 4];
    for t in 0..=(la[0] + lb[0]) {
        let et = entry(0, t);
        let gx = g[0].powi(t as i32);
        for u in 0..=(la[1] + lb[1]) {
            let eu = entry(1, u);
            let gy = g[1].powi(u as i32);
            for v in 0..=(la[2] + lb[2]) {
                let ev = entry(2, v);
                let index = (t + u + v) % 4;
                acc[index] = acc[index] + et * eu * ev * gx * gy * g[2].powi(v as i32);
            }
        }
    }
    // i^0 = 1, i^1 = i, i^2 = -1, i^3 = -i.
    let s_re = acc[0] - acc[2];
    let s_im = acc[1] - acc[3];
    let norm2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    let pref = (norm2.scale(-0.25 / p))
        .exp()
        .scale((std::f64::consts::PI / p).powf(1.5));
    let phase = g[0] * pc[0] + g[1] * pc[1] + g[2] * pc[2];
    let (c, s) = cos_sin(phase);
    (pref * (s_re * c - s_im * s), pref * (s_re * s + s_im * c))
}

/// Contract a primitive-level complex operator over one shell pair into the
/// spherical AO block.
fn shell_block<F>(sh_a: &MolShell, sh_b: &MolShell, mut prim: F) -> (Matrix, Matrix)
where
    F: FnMut([usize; 3], [usize; 3], f64, f64) -> (f64, f64),
{
    let ca = cart_components(sh_a.l);
    let cb = cart_components(sh_b.l);
    let mut re = Matrix::zeros(ca.len(), cb.len());
    let mut im = Matrix::zeros(ca.len(), cb.len());
    for (&ea, &coa) in sh_a.exps.iter().zip(&sh_a.coefs) {
        for (&eb, &cob) in sh_b.exps.iter().zip(&sh_b.coefs) {
            let w = coa * cob;
            for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                    let (r, m) = prim(
                        [ax as usize, ay as usize, az as usize],
                        [bx as usize, by as usize, bz as usize],
                        ea,
                        eb,
                    );
                    re[(i, j)] += w * r;
                    im[(i, j)] += w * m;
                }
            }
        }
    }
    (
        crate::basis::aos::sph_block(sh_a.l, sh_b.l, &re),
        crate::basis::aos::sph_block(sh_a.l, sh_b.l, &im),
    )
}

/// Analytic spherical shell-pair Fourier form without constructing the
/// unrelated blocks of a combined basis.
pub(crate) fn plane_wave_shell_pair(sh_a: &MolShell, sh_b: &MolShell, g: Vec3) -> (Matrix, Matrix) {
    shell_block(sh_a, sh_b, |la, lb, ea, eb| {
        plane_wave_primitive(la, lb, ea, eb, sh_a.center, sh_b.center, g)
    })
}

/// Apply the same ket derivative and anticommutator contact to plain
/// integrals and to their moving-centre and moving-wave jets.
pub(crate) fn current_primitive<S: Scalar>(
    angular: [usize; 3],
    exponent: f64,
    wave_component: S,
    axis: usize,
    transform: impl Fn([usize; 3]) -> (S, S),
) -> (S, S) {
    let mut raised = angular;
    raised[axis] += 1;
    let (upper_re, upper_im) = transform(raised);
    let (base_re, base_im) = transform(angular);
    let mut derivative_re = upper_re.scale(-2.0 * exponent);
    let mut derivative_im = upper_im.scale(-2.0 * exponent);
    if angular[axis] > 0 {
        let mut lowered = angular;
        lowered[axis] -= 1;
        let (lower_re, lower_im) = transform(lowered);
        derivative_re = derivative_re + lower_re.scale(angular[axis] as f64);
        derivative_im = derivative_im + lower_im.scale(angular[axis] as f64);
    }
    let half_wave = wave_component.scale(0.5);
    (
        derivative_im + half_wave * base_re,
        -derivative_re + half_wave * base_im,
    )
}

/// Paramagnetic current vertex `<a| {exp(iQ.r),p_j}/2 |b>`.
/// Angular shifts retain the original contraction and spherical transforms.
pub(crate) fn plane_wave_current_shell_pair(
    left: &MolShell,
    right: &MolShell,
    wave: Vec3,
    axis: usize,
) -> (Matrix, Matrix) {
    shell_block(left, right, |la, lb, ea, eb| {
        current_primitive(lb, eb, wave.get(axis), axis, |angular| {
            plane_wave_primitive(la, angular, ea, eb, left.center, right.center, wave)
        })
    })
}

/// `rho_mn(G) = <mu| exp(i G . r) |nu>` over the whole basis, as
/// `(real, imaginary)`.
pub fn plane_wave_matrix(mb: &MolBasis, g: Vec3) -> (Matrix, Matrix) {
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let blocks: Vec<((usize, usize), Matrix, Matrix)> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            let (re, im) = plane_wave_shell_pair(sh_a, sh_b, g);
            ((sh_a.sph_off, sh_b.sph_off), re, im)
        })
        .collect();
    let (mut re, mut im) = (Matrix::zeros(n, n), Matrix::zeros(n, n));
    for ((oa, ob), r, m) in blocks {
        for i in 0..r.rows {
            for j in 0..r.cols {
                re[(oa + i, ob + j)] = r[(i, j)];
                im[(oa + i, ob + j)] = m[(i, j)];
            }
        }
    }
    (re, im)
}

/// Cartesian blocks of `sum_G c_k(G) rho(G)` for one shell pair and every
/// coefficient set `k`, as `(real, imaginary)`.
///
/// The loop order is the point. A reciprocal-space lattice sum needs
/// thousands of `G` vectors and never wants any of them on their own, so
/// building each matrix separately would rebuild the Hermite tables once
/// per `G` and hold `n_G * nao^2` numbers that are about to be summed away.
/// Here the tables are built once per primitive pair, the `G` loop sits
/// inside them, and the coefficient sets sit innermost — so `3 * natoms`
/// sets, which is what a nuclear derivative needs, cost one pass over the
/// mesh rather than `3 * natoms` passes.
pub(crate) fn contracted_pair_blocks(
    sh_a: &MolShell,
    sh_b: &MolShell,
    gs: &[Vec3],
    c_re: &[&[f64]],
    c_im: &[&[f64]],
) -> Vec<(Matrix, Matrix)> {
    ReciprocalContraction::new(gs, c_re, c_im).blocks(sh_a, sh_b)
}

/// Coefficient bounds and wave norms shared by contractions over many
/// shell pairs against the same reciprocal potential.
pub(crate) struct ReciprocalContraction<'a> {
    gs: &'a [Vec3],
    c_re: Vec<&'a [f64]>,
    c_im: Vec<&'a [f64]>,
    norms: Vec<f64>,
    cmax: Vec<f64>,
    tail: Vec<f64>,
    sorted: bool,
}

impl<'a> ReciprocalContraction<'a> {
    pub(crate) fn new(gs: &'a [Vec3], c_re: &[&'a [f64]], c_im: &[&'a [f64]]) -> Self {
        debug_assert_eq!(c_re.len(), c_im.len());
        let norms: Vec<_> = gs.iter().map(|g| g.norm2()).collect();
        let sorted = norms.windows(2).all(|w| w[0] <= w[1]);
        // The same screen applies to every coefficient set. A suffix
        // maximum allows an early end on a mesh ordered by wave norm.
        let cmax: Vec<f64> = (0..gs.len())
            .map(|gi| {
                c_re.iter()
                    .zip(c_im)
                    .map(|(re, im)| re[gi].abs() + im[gi].abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let mut tail = cmax.clone();
        for gi in (0..tail.len().saturating_sub(1)).rev() {
            tail[gi] = tail[gi].max(tail[gi + 1]);
        }
        Self {
            gs,
            c_re: c_re.to_vec(),
            c_im: c_im.to_vec(),
            norms,
            cmax,
            tail,
            sorted,
        }
    }

    pub(crate) fn blocks(&self, sh_a: &MolShell, sh_b: &MolShell) -> Vec<(Matrix, Matrix)> {
        let nsets = self.c_re.len();
        let (la_max, lb_max) = (sh_a.l as usize, sh_b.l as usize);
        let ca = cart_components(sh_a.l);
        let cb = cart_components(sh_b.l);
        let mut out: Vec<(Matrix, Matrix)> = (0..nsets)
            .map(|_| {
                (
                    Matrix::zeros(ca.len(), cb.len()),
                    Matrix::zeros(ca.len(), cb.len()),
                )
            })
            .collect();
        let q = sh_a.center - sh_b.center;
        let lsum = la_max + lb_max;
        let mut px = vec![0.0; lsum + 1];
        let mut py = vec![0.0; lsum + 1];
        let mut pz = vec![0.0; lsum + 1];
        for (&ea, &coa) in sh_a.exps.iter().zip(&sh_a.coefs) {
            for (&eb, &cob) in sh_b.exps.iter().zip(&sh_b.coefs) {
                let w = coa * cob;
                let p = ea + eb;
                let pc = (sh_a.center * ea + sh_b.center * eb) / p;
                // One set of Hermite tables for the whole G mesh: the expansion
                // depends on the pair, not on the operator.
                let ex = ETable::build(la_max, lb_max, q.x, ea, eb);
                let ey = ETable::build(la_max, lb_max, q.y, ea, eb);
                let ez = ETable::build(la_max, lb_max, q.z, ea, eb);
                let base = (std::f64::consts::PI / p).powf(1.5) * w;
                for (gi, g) in self.gs.iter().enumerate() {
                    let pref = base * (-self.norms[gi] / (4.0 * p)).exp();
                    // The Gaussian damping is what makes a plane-wave expansion
                    // of a smooth density finite; skipping the terms it has
                    // already killed is not an approximation beyond the one the
                    // mesh cutoff already made.
                    if pref.abs() * self.cmax[gi] < 1e-18 {
                        if self.sorted && pref.abs() * self.tail[gi] < 1e-18 {
                            break;
                        }
                        continue;
                    }
                    let phase = g.dot(pc);
                    let (cs, sn) = (phase.cos(), phase.sin());
                    px[0] = 1.0;
                    py[0] = 1.0;
                    pz[0] = 1.0;
                    for t in 1..=lsum {
                        px[t] = px[t - 1] * g.x;
                        py[t] = py[t - 1] * g.y;
                        pz[t] = pz[t - 1] * g.z;
                    }
                    for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                        let (ax, ay, az) = (ax as usize, ay as usize, az as usize);
                        for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                            let (bx, by, bz) = (bx as usize, by as usize, bz as usize);
                            let mut acc = [0.0_f64; 4];
                            for t in 0..=(ax + bx) {
                                let et = ex.get(ax, bx, t);
                                if et == 0.0 {
                                    continue;
                                }
                                for u in 0..=(ay + by) {
                                    let eu = ey.get(ay, by, u);
                                    if eu == 0.0 {
                                        continue;
                                    }
                                    for v in 0..=(az + bz) {
                                        let ev = ez.get(az, bz, v);
                                        if ev == 0.0 {
                                            continue;
                                        }
                                        acc[(t + u + v) % 4] +=
                                            et * eu * ev * px[t] * py[u] * pz[v];
                                    }
                                }
                            }
                            // rho_ij(G) for this primitive pair, prefactor and
                            // phase folded in — shared by every coefficient set.
                            let s_re = acc[0] - acc[2];
                            let s_im = acc[1] - acc[3];
                            let rre = pref * (s_re * cs - s_im * sn);
                            let rim = pref * (s_re * sn + s_im * cs);
                            for k in 0..nsets {
                                let (kr, ki) = (self.c_re[k][gi], self.c_im[k][gi]);
                                out[k].0[(i, j)] += kr * rre - ki * rim;
                                out[k].1[(i, j)] += kr * rim + ki * rre;
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

/// Cartesian blocks of `rho(G)` for one shell pair, one entry per `G`.
///
/// The transpose of [`contracted_pair_blocks`]: there the `G` index is
/// summed away and the AO indices survive, here the AO indices are the ones
/// that will be summed away and `G` survives. Both share the same
/// primitive loop, and the choice between them is which index a caller
/// contracts first — a Fourier component of the density needs this one.
pub(crate) fn pair_blocks_per_g(
    sh_a: &MolShell,
    sh_b: &MolShell,
    gs: &[Vec3],
) -> Vec<(Matrix, Matrix)> {
    let (la_max, lb_max) = (sh_a.l as usize, sh_b.l as usize);
    let ca = cart_components(sh_a.l);
    let cb = cart_components(sh_b.l);
    let mut out: Vec<(Matrix, Matrix)> = (0..gs.len())
        .map(|_| {
            (
                Matrix::zeros(ca.len(), cb.len()),
                Matrix::zeros(ca.len(), cb.len()),
            )
        })
        .collect();
    let sorted = gs.windows(2).all(|w| w[0].norm2() <= w[1].norm2());
    let q = sh_a.center - sh_b.center;
    let lsum = la_max + lb_max;
    let mut px = vec![0.0; lsum + 1];
    let mut py = vec![0.0; lsum + 1];
    let mut pz = vec![0.0; lsum + 1];
    for (&ea, &coa) in sh_a.exps.iter().zip(&sh_a.coefs) {
        for (&eb, &cob) in sh_b.exps.iter().zip(&sh_b.coefs) {
            let w = coa * cob;
            let p = ea + eb;
            let pc = (sh_a.center * ea + sh_b.center * eb) / p;
            let ex = ETable::build(la_max, lb_max, q.x, ea, eb);
            let ey = ETable::build(la_max, lb_max, q.y, ea, eb);
            let ez = ETable::build(la_max, lb_max, q.z, ea, eb);
            let base = (std::f64::consts::PI / p).powf(1.5) * w;
            for (gi, g) in gs.iter().enumerate() {
                let pref = base * (-g.norm2() / (4.0 * p)).exp();
                if pref.abs() < 1e-18 {
                    if sorted {
                        break;
                    }
                    continue;
                }
                let phase = g.dot(pc);
                let (cs, sn) = (phase.cos(), phase.sin());
                px[0] = 1.0;
                py[0] = 1.0;
                pz[0] = 1.0;
                for t in 1..=lsum {
                    px[t] = px[t - 1] * g.x;
                    py[t] = py[t - 1] * g.y;
                    pz[t] = pz[t - 1] * g.z;
                }
                for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                    let (ax, ay, az) = (ax as usize, ay as usize, az as usize);
                    for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                        let (bx, by, bz) = (bx as usize, by as usize, bz as usize);
                        let mut acc = [0.0_f64; 4];
                        for t in 0..=(ax + bx) {
                            let et = ex.get(ax, bx, t);
                            if et == 0.0 {
                                continue;
                            }
                            for u in 0..=(ay + by) {
                                let eu = ey.get(ay, by, u);
                                if eu == 0.0 {
                                    continue;
                                }
                                for v in 0..=(az + bz) {
                                    let ev = ez.get(az, bz, v);
                                    if ev == 0.0 {
                                        continue;
                                    }
                                    acc[(t + u + v) % 4] += et * eu * ev * px[t] * py[u] * pz[v];
                                }
                            }
                        }
                        let s_re = acc[0] - acc[2];
                        let s_im = acc[1] - acc[3];
                        out[gi].0[(i, j)] += pref * (s_re * cs - s_im * sn);
                        out[gi].1[(i, j)] += pref * (s_re * sn + s_im * cs);
                    }
                }
            }
        }
    }
    out
}

/// Accumulate `sum_ij D_ij rho_ij(G)` of one shell pair into `re`/`im` for
/// every `G`, with `d_cart` the density block in the Cartesian components of
/// the two shells.
///
/// The density-contracted form of [`pair_blocks_per_g`]: no per-`G` matrix
/// is formed, and on a mesh ordered by `|G|` the loop stops where the
/// Gaussian damping of the pair has killed every remaining term.
pub(crate) fn pair_density_transform(
    sh_a: &MolShell,
    sh_b: &MolShell,
    gs: &[Vec3],
    d_cart: &Matrix,
    re: &mut [f64],
    im: &mut [f64],
) {
    let (la_max, lb_max) = (sh_a.l as usize, sh_b.l as usize);
    let ca = cart_components(sh_a.l);
    let cb = cart_components(sh_b.l);
    let dmax = d_cart
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    if dmax == 0.0 {
        return;
    }
    let sorted = gs.windows(2).all(|w| w[0].norm2() <= w[1].norm2());
    let q = sh_a.center - sh_b.center;
    let lsum = la_max + lb_max;
    let mut px = vec![0.0; lsum + 1];
    let mut py = vec![0.0; lsum + 1];
    let mut pz = vec![0.0; lsum + 1];
    for (&ea, &coa) in sh_a.exps.iter().zip(&sh_a.coefs) {
        for (&eb, &cob) in sh_b.exps.iter().zip(&sh_b.coefs) {
            let w = coa * cob;
            let p = ea + eb;
            let pc = (sh_a.center * ea + sh_b.center * eb) / p;
            let ex = ETable::build(la_max, lb_max, q.x, ea, eb);
            let ey = ETable::build(la_max, lb_max, q.y, ea, eb);
            let ez = ETable::build(la_max, lb_max, q.z, ea, eb);
            let base = (std::f64::consts::PI / p).powf(1.5) * w;
            for (gi, g) in gs.iter().enumerate() {
                let pref = base * (-g.norm2() / (4.0 * p)).exp();
                if pref.abs() * dmax < 1e-18 {
                    if sorted {
                        break;
                    }
                    continue;
                }
                let phase = g.dot(pc);
                let (cs, sn) = (phase.cos(), phase.sin());
                px[0] = 1.0;
                py[0] = 1.0;
                pz[0] = 1.0;
                for t in 1..=lsum {
                    px[t] = px[t - 1] * g.x;
                    py[t] = py[t - 1] * g.y;
                    pz[t] = pz[t - 1] * g.z;
                }
                let mut acc = [0.0_f64; 4];
                for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                    let (ax, ay, az) = (ax as usize, ay as usize, az as usize);
                    for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                        let dij = d_cart[(i, j)];
                        if dij == 0.0 {
                            continue;
                        }
                        let (bx, by, bz) = (bx as usize, by as usize, bz as usize);
                        for t in 0..=(ax + bx) {
                            let et = ex.get(ax, bx, t);
                            if et == 0.0 {
                                continue;
                            }
                            for u in 0..=(ay + by) {
                                let eu = ey.get(ay, by, u);
                                if eu == 0.0 {
                                    continue;
                                }
                                for v in 0..=(az + bz) {
                                    let ev = ez.get(az, bz, v);
                                    if ev == 0.0 {
                                        continue;
                                    }
                                    acc[(t + u + v) % 4] +=
                                        dij * et * eu * ev * px[t] * py[u] * pz[v];
                                }
                            }
                        }
                    }
                }
                let s_re = acc[0] - acc[2];
                let s_im = acc[1] - acc[3];
                re[gi] += pref * (s_re * cs - s_im * sn);
                im[gi] += pref * (s_re * sn + s_im * cs);
            }
        }
    }
}

/// `sum_mn D_mn rho_mn(G)` for every `G`, as `(real, imaginary)` vectors.
///
/// Note the conjugation convention: `rho_mn(G) = <mu|exp(+iG.r)|nu>`, so the
/// Fourier coefficient of the density in the sense
/// `n(r) = (1/V) sum_G n~(G) exp(iG.r)` is the COMPLEX CONJUGATE of what
/// this returns. At `G = 0` both are the electron count, which is the
/// cheapest way to check a caller has the convention right.
pub fn plane_wave_density(mb: &MolBasis, gs: &[Vec3], d: &Matrix) -> (Vec<f64>, Vec<f64>) {
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partial: Vec<(Vec<f64>, Vec<f64>)> = pairs
        .par_iter()
        .map(|&(ia, ib)| {
            let (sh_a, sh_b) = (&mb.shells[ia], &mb.shells[ib]);
            let cart = pair_blocks_per_g(sh_a, sh_b, gs);
            let (oa, ob) = (sh_a.sph_off, sh_b.sph_off);
            let mut re = vec![0.0; gs.len()];
            let mut im = vec![0.0; gs.len()];
            for (gi, (r, m)) in cart.iter().enumerate() {
                let rs = crate::basis::aos::sph_block(sh_a.l, sh_b.l, r);
                let ms = crate::basis::aos::sph_block(sh_a.l, sh_b.l, m);
                for i in 0..rs.rows {
                    for j in 0..rs.cols {
                        let dij = d[(oa + i, ob + j)];
                        re[gi] += dij * rs[(i, j)];
                        im[gi] += dij * ms[(i, j)];
                    }
                }
            }
            (re, im)
        })
        .collect();
    let mut re = vec![0.0; gs.len()];
    let mut im = vec![0.0; gs.len()];
    for (r, m) in partial {
        for gi in 0..gs.len() {
            re[gi] += r[gi];
            im[gi] += m[gi];
        }
    }
    (re, im)
}

/// `sum_G c_k(G) rho_mn(G)` over a whole mesh of `G`, one `(real, imaginary)`
/// pair per coefficient set.
pub fn plane_wave_contractions(
    mb: &MolBasis,
    gs: &[Vec3],
    c_re: &[&[f64]],
    c_im: &[&[f64]],
) -> Vec<(Matrix, Matrix)> {
    assert_eq!(c_re.len(), c_im.len());
    for (a, b) in c_re.iter().zip(c_im) {
        assert_eq!(a.len(), gs.len());
        assert_eq!(b.len(), gs.len());
    }
    let nsets = c_re.len();
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let blocks: Vec<((usize, usize), Vec<(Matrix, Matrix)>)> = pairs
        .par_iter()
        .map(|&(ia, ib)| {
            let (sh_a, sh_b) = (&mb.shells[ia], &mb.shells[ib]);
            let cart = contracted_pair_blocks(sh_a, sh_b, gs, c_re, c_im);
            let sph = cart
                .iter()
                .map(|(r, m)| {
                    (
                        crate::basis::aos::sph_block(sh_a.l, sh_b.l, r),
                        crate::basis::aos::sph_block(sh_a.l, sh_b.l, m),
                    )
                })
                .collect();
            ((sh_a.sph_off, sh_b.sph_off), sph)
        })
        .collect();
    let mut out: Vec<(Matrix, Matrix)> = (0..nsets)
        .map(|_| (Matrix::zeros(n, n), Matrix::zeros(n, n)))
        .collect();
    for ((oa, ob), sets) in blocks {
        for (k, (r, m)) in sets.into_iter().enumerate() {
            for i in 0..r.rows {
                for j in 0..r.cols {
                    out[k].0[(oa + i, ob + j)] = r[(i, j)];
                    out[k].1[(oa + i, ob + j)] = m[(i, j)];
                }
            }
        }
    }
    out
}

/// Single-coefficient-set [`plane_wave_contractions`].
pub fn plane_wave_contraction(
    mb: &MolBasis,
    gs: &[Vec3],
    c_re: &[f64],
    c_im: &[f64],
) -> (Matrix, Matrix) {
    plane_wave_contractions(mb, gs, &[c_re], &[c_im])
        .pop()
        .expect("one coefficient set in, one out")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::system::{Atom, Molecule};

    fn water() -> (Molecule, MolBasis) {
        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.2217),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 1.4309, -0.8867),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.15, -1.4009, -0.9067),
            },
        ]);
        let set = data::load("6-31g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        (mol, mb)
    }

    /// At `G = 0` the plane wave is unity, so the result must be exactly
    /// the overlap matrix, with nothing in the imaginary part. This one
    /// check pins the prefactor, the `E`-coefficient normalisation and the
    /// spherical transform simultaneously.
    #[test]
    fn g_zero_gives_the_overlap_matrix() {
        let (_, mb) = water();
        let (re, im) = plane_wave_matrix(&mb, Vec3::zero());
        let s = crate::integrals::overlap_matrix(&mb);
        assert!(
            re.max_abs_diff(&s) < 1e-12,
            "G = 0 real part differs from the overlap matrix"
        );
        assert!(
            im.as_slice().iter().all(|v| v.abs() < 1e-14),
            "G = 0 has a non-zero imaginary part"
        );
    }

    /// `rho(-G) = rho(G)*`, because the basis functions are real. A sign
    /// error in the phase or in the powers of `i` breaks this.
    #[test]
    fn negating_g_conjugates() {
        let (_, mb) = water();
        for g in [
            Vec3::new(0.7, 0.0, 0.0),
            Vec3::new(-0.3, 1.1, 0.4),
            Vec3::new(2.0, -1.5, 0.9),
        ] {
            let (r1, i1) = plane_wave_matrix(&mb, g);
            let (r2, i2) = plane_wave_matrix(&mb, g * -1.0);
            assert!(r1.max_abs_diff(&r2) < 1e-12, "real part is not even in G");
            let mut worst = 0.0_f64;
            for k in 0..i1.rows {
                for l in 0..i1.cols {
                    worst = worst.max((i1[(k, l)] + i2[(k, l)]).abs());
                }
            }
            assert!(worst < 1e-12, "imaginary part is not odd in G: {worst:e}");
        }
    }

    /// `rho_mn(G)` must be symmetric in `(m, n)` — the operator is
    /// multiplicative and the basis real, so swapping bra and ket changes
    /// nothing. This catches an index slip in the Hermite contraction that
    /// the `G = 0` test cannot see, since the overlap is symmetric anyway.
    #[test]
    fn matrix_is_symmetric_in_the_ao_indices() {
        let (_, mb) = water();
        let g = Vec3::new(0.9, -0.6, 1.3);
        let (re, im) = plane_wave_matrix(&mb, g);
        for m in [&re, &im] {
            let mut worst = 0.0_f64;
            for i in 0..m.rows {
                for j in 0..m.cols {
                    worst = worst.max((m[(i, j)] - m[(j, i)]).abs());
                }
            }
            assert!(worst < 1e-12, "rho(G) is not symmetric: {worst:e}");
        }
    }

    /// The fast contracted path and the plain one must agree. They share
    /// the closed form but not the loop structure — the contraction hoists
    /// the Hermite tables out of the `G` loop, precomputes the powers of
    /// `G` and screens on the Gaussian damping — so this pins the
    /// rearrangement, which is where such a rewrite goes wrong.
    #[test]
    fn contraction_matches_summing_the_individual_matrices() {
        let (_, mb) = water();
        let gs = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.8, -0.5, 0.35),
            Vec3::new(-1.4, 0.2, 2.1),
            Vec3::new(0.05, 3.3, -0.9),
        ];
        let c_re = vec![1.0, -0.3, 2.2, 0.7];
        let c_im = vec![0.0, 1.1, -0.4, 0.9];
        let (re, im) = plane_wave_contraction(&mb, &gs, &c_re, &c_im);
        let n = mb.nao_sph;
        let (mut wre, mut wim) = (Matrix::zeros(n, n), Matrix::zeros(n, n));
        for (k, g) in gs.iter().enumerate() {
            let (a, b) = plane_wave_matrix(&mb, *g);
            for i in 0..n {
                for j in 0..n {
                    wre[(i, j)] += c_re[k] * a[(i, j)] - c_im[k] * b[(i, j)];
                    wim[(i, j)] += c_re[k] * b[(i, j)] + c_im[k] * a[(i, j)];
                }
            }
        }
        assert!(
            re.max_abs_diff(&wre) < 1e-11 && im.max_abs_diff(&wim) < 1e-11,
            "contracted result differs: real {:e}, imaginary {:e}",
            re.max_abs_diff(&wre),
            im.max_abs_diff(&wim)
        );
    }

    /// `sum_mn D_mn rho_mn(G)` must equal contracting the full matrix by
    /// hand, and at `G = 0` it must be the electron count `Tr[D S]`. The
    /// second half is what pins the conjugation convention, which is the
    /// only thing about a Fourier transform that is easy to get backwards.
    #[test]
    fn density_fourier_components_contract_the_matrix() {
        let (_, mb) = water();
        let n = mb.nao_sph;
        // An arbitrary symmetric "density": the point is the contraction,
        // not the physics.
        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let v = ((i * 7 + j * 3) % 5) as f64 * 0.1 - 0.2;
                d[(i, j)] = v;
                d[(j, i)] = v;
            }
        }
        let gs = vec![
            Vec3::zero(),
            Vec3::new(0.6, -0.2, 1.1),
            Vec3::new(-1.7, 0.9, 0.3),
        ];
        let (re, im) = plane_wave_density(&mb, &gs, &d);
        for (gi, g) in gs.iter().enumerate() {
            let (a, b) = plane_wave_matrix(&mb, *g);
            let (mut wr, mut wi) = (0.0, 0.0);
            for i in 0..n {
                for j in 0..n {
                    wr += d[(i, j)] * a[(i, j)];
                    wi += d[(i, j)] * b[(i, j)];
                }
            }
            assert!((re[gi] - wr).abs() < 1e-11, "real part at G index {gi}");
            assert!(
                (im[gi] - wi).abs() < 1e-11,
                "imaginary part at G index {gi}"
            );
        }
        let s = crate::integrals::overlap_matrix(&mb);
        let mut ne = 0.0;
        for i in 0..n {
            for j in 0..n {
                ne += d[(i, j)] * s[(i, j)];
            }
        }
        assert!((re[0] - ne).abs() < 1e-11, "G = 0 is not Tr[D S]");
        assert!(im[0].abs() < 1e-13, "G = 0 has an imaginary part");
    }

    /// Against direct numerical quadrature on the molecular grid. This is
    /// the independent route: the analytic closed form above and a
    /// Becke-partitioned quadrature of `chi_mu chi_nu cos(G.r)` share no
    /// code, so agreement is real evidence rather than self-consistency.
    #[test]
    fn matches_numerical_quadrature() {
        use crate::grid::{GridOptions, MolecularGrid};
        let (mol, mb) = water();
        let grid = MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 6,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let g = Vec3::new(0.8, -0.5, 0.35);
        let (re, im) = plane_wave_matrix(&mb, g);

        let n = mb.nao_sph;
        let mut qre = Matrix::zeros(n, n);
        let mut qim = Matrix::zeros(n, n);
        let batch = crate::grid::eval::eval_aos(&mb, &grid.points, false);
        for (idx, r) in grid.points.iter().enumerate() {
            let w = grid.weights[idx];
            let phase = g.dot(*r);
            let (c, s) = (phase.cos(), phase.sin());
            for i in 0..n {
                let pi = batch.phi[(idx, i)];
                if pi == 0.0 {
                    continue;
                }
                for j in 0..n {
                    let v = w * pi * batch.phi[(idx, j)];
                    qre[(i, j)] += v * c;
                    qim[(i, j)] += v * s;
                }
            }
        }
        // The quadrature is the loose side here; a few parts in 10^6 on a
        // level-6 grid is what a smooth oscillatory integrand gives.
        assert!(
            re.max_abs_diff(&qre) < 5e-6,
            "real part vs quadrature: {:e}",
            re.max_abs_diff(&qre)
        );
        assert!(
            im.max_abs_diff(&qim) < 5e-6,
            "imaginary part vs quadrature: {:e}",
            im.max_abs_diff(&qim)
        );
    }
}
