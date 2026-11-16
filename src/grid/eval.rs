// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! AO evaluation on grid-point batches: values φ_μ(r) and gradients ∇φ_μ(r)
//! in the spherical AO basis, with conservative per-shell distance
//! screening.

use crate::basis::{aos, cart_components, n_cart, n_sph, MolBasis};
use crate::linalg::Matrix;
use crate::math::Vec3;

/// AO values (and optionally derivatives) on a batch of points.
pub struct AoBatch {
    /// `npts × nao` values.
    pub phi: Matrix,
    /// `[x,y,z]` component gradients, each `npts × nao`.
    pub grad: Option<[Matrix; 3]>,
    /// Second derivatives in the order (xx, xy, xz, yy, yz, zz).
    pub hess: Option<[Matrix; 6]>,
    /// Third derivatives in the order
    /// (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz).
    pub third: Option<[Matrix; 10]>,
}

/// Packed symmetric pair for three axes or channels: xx, xy, xz, yy, yz, zz.
/// Both arguments must be in 0..3; either ordering is accepted.
pub(crate) fn second_index(u: usize, v: usize) -> usize {
    let (a, b) = if u <= v { (u, v) } else { (v, u) };
    match (a, b) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (1, 1) => 3,
        (1, 2) => 4,
        _ => 5,
    }
}

/// Index of the packed third-derivative component for sorted axes
/// `u ≤ v ≤ w` (the caller may pass any order).
pub fn third_index(mut u: usize, mut v: usize, mut w: usize) -> usize {
    if u > v {
        std::mem::swap(&mut u, &mut v);
    }
    if v > w {
        std::mem::swap(&mut v, &mut w);
    }
    if u > v {
        std::mem::swap(&mut u, &mut v);
    }
    match (u, v, w) {
        (0, 0, 0) => 0,
        (0, 0, 1) => 1,
        (0, 0, 2) => 2,
        (0, 1, 1) => 3,
        (0, 1, 2) => 4,
        (0, 2, 2) => 5,
        (1, 1, 1) => 6,
        (1, 1, 2) => 7,
        (1, 2, 2) => 8,
        _ => 9,
    }
}

/// Conservative squared screening radius of a shell: beyond it every
/// primitive contributes < `tol` relative to its coefficient scale.
fn shell_extent2(exps: &[f64], tol_log: f64) -> f64 {
    let alpha_min = exps.iter().cloned().fold(f64::INFINITY, f64::min);
    tol_log / alpha_min
}

/// Evaluate all AOs on `points`.
pub fn eval_aos(mb: &MolBasis, points: &[Vec3], want_grad: bool) -> AoBatch {
    eval_aos_full(mb, points, want_grad, false)
}

/// Evaluate AO values, gradients, and (optionally) second derivatives.
pub fn eval_aos_full(mb: &MolBasis, points: &[Vec3], want_grad: bool, want_hess: bool) -> AoBatch {
    eval_aos_deriv3(mb, points, want_grad, want_hess, false)
}

/// Evaluate AO values and derivatives up to third order.
pub fn eval_aos_deriv3(
    mb: &MolBasis,
    points: &[Vec3],
    want_grad: bool,
    want_hess: bool,
    want_third: bool,
) -> AoBatch {
    let want_hess = want_hess || want_third;
    let want_grad = want_grad || want_hess;
    let npts = points.len();
    let nao = mb.nao_sph;
    let mut phi = Matrix::zeros(npts, nao);
    let mut grad = if want_grad {
        Some([
            Matrix::zeros(npts, nao),
            Matrix::zeros(npts, nao),
            Matrix::zeros(npts, nao),
        ])
    } else {
        None
    };
    let mut hess = if want_hess {
        Some(std::array::from_fn(|_| Matrix::zeros(npts, nao)))
    } else {
        None
    };
    let mut third = if want_third {
        Some(std::array::from_fn::<Matrix, 10, _>(|_| {
            Matrix::zeros(npts, nao)
        }))
    } else {
        None
    };
    // ln(1/1e-14) ≈ 32.2: e^{−α r²} < 1e-14 outside.
    let tol_log = 32.24;

    for shell in &mb.shells {
        let l = shell.l;
        let comps = cart_components(l);
        let (ncart, nsph) = (n_cart(l), n_sph(l));
        let c2s = aos::cart_to_sph(l);
        let ext2 = shell_extent2(&shell.exps, tol_log) * 2.0; // margin
        let mut cart_val = vec![0.0; ncart];
        let mut cart_d = [vec![0.0; ncart], vec![0.0; ncart], vec![0.0; ncart]];
        let mut cart_h = std::array::from_fn::<Vec<f64>, 6, _>(|_| vec![0.0; ncart]);
        let mut cart_t = std::array::from_fn::<Vec<f64>, 10, _>(|_| vec![0.0; ncart]);
        for (ip, p) in points.iter().enumerate() {
            let d = *p - shell.center;
            let r2 = d.norm2();
            if r2 > ext2 {
                continue;
            }
            // Radial factors: g = Σ c e^{−αr²}, gp = Σ −2αc e, gpp = Σ 4α²c e
            // so that ∂x g = x·gp, ∂x∂y g = x y·gpp (+ δ_xy gp).
            let mut g = 0.0;
            let mut gp = 0.0;
            let mut gpp = 0.0;
            let mut gppp = 0.0;
            for (&a, &c) in shell.exps.iter().zip(&shell.coefs) {
                let e = c * (-a * r2).exp();
                g += e;
                gp += -2.0 * a * e;
                gpp += 4.0 * a * a * e;
                gppp += -8.0 * a * a * a * e;
            }
            let dv = [d.x, d.y, d.z];
            for (k, &(ax, ay, az)) in comps.iter().enumerate() {
                let e = [ax, ay, az];
                let mono = |ex: i32, ey: i32, ez: i32| -> f64 {
                    if ex < 0 || ey < 0 || ez < 0 {
                        0.0
                    } else {
                        d.x.powi(ex) * d.y.powi(ey) * d.z.powi(ez)
                    }
                };
                let m = mono(ax, ay, az);
                cart_val[k] = m * g;
                if want_grad {
                    // ∂g m for each axis.
                    let dm = [
                        ax as f64 * mono(ax - 1, ay, az),
                        ay as f64 * mono(ax, ay - 1, az),
                        az as f64 * mono(ax, ay, az - 1),
                    ];
                    for x in 0..3 {
                        cart_d[x][k] = dm[x] * g + m * dv[x] * gp;
                    }
                    if want_hess {
                        // ∂u∂v φ = (∂u∂v m) g + ∂u m·v·gp + ∂v m·u·gp
                        //        + m·(δ_uv gp + u v gpp).
                        let ddm = |u: usize, v: usize| -> f64 {
                            let mut ex = [ax, ay, az];
                            let mut c = 1.0;
                            for &w in &[u, v] {
                                c *= ex[w] as f64;
                                ex[w] -= 1;
                            }
                            if c == 0.0 {
                                0.0
                            } else {
                                c * mono(ex[0], ex[1], ex[2])
                            }
                        };
                        let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
                        for (idx, &(u, v)) in pairs.iter().enumerate() {
                            let delta = if u == v { gp } else { 0.0 };
                            cart_h[idx][k] = ddm(u, v) * g
                                + dm[u] * dv[v] * gp
                                + dm[v] * dv[u] * gp
                                + m * (delta + dv[u] * dv[v] * gpp);
                        }
                        let _ = e;
                        if third.is_some() {
                            // ∂u∂v∂w (m g) with the r²-chain convention
                            // (∂u f(r²) = u·f'):
                            //   m_uvw g
                            // + (m_vw u + m_uw v + m_uv w) gp
                            // + (m_w δuv + m_v δuw + m_u δvw) gp
                            // + (m_w u v + m_v u w + m_u v w) gpp
                            // + m (δvw u + δuw v + δuv w) gpp
                            // + m u v w gppp.
                            let dddm = |u: usize, v: usize, w: usize| -> f64 {
                                let mut ex = [ax, ay, az];
                                let mut c = 1.0;
                                for &q in &[u, v, w] {
                                    c *= ex[q] as f64;
                                    ex[q] -= 1;
                                }
                                if c == 0.0 {
                                    0.0
                                } else {
                                    c * mono(ex[0], ex[1], ex[2])
                                }
                            };
                            let trip = [
                                (0usize, 0usize, 0usize),
                                (0, 0, 1),
                                (0, 0, 2),
                                (0, 1, 1),
                                (0, 1, 2),
                                (0, 2, 2),
                                (1, 1, 1),
                                (1, 1, 2),
                                (1, 2, 2),
                                (2, 2, 2),
                            ];
                            let del = |a: usize, b: usize| -> f64 {
                                if a == b {
                                    1.0
                                } else {
                                    0.0
                                }
                            };
                            for (idx, &(u, v, w)) in trip.iter().enumerate() {
                                cart_t[idx][k] = dddm(u, v, w) * g
                                    + (ddm(v, w) * dv[u] + ddm(u, w) * dv[v] + ddm(u, v) * dv[w])
                                        * gp
                                    + (dm[w] * del(u, v) + dm[v] * del(u, w) + dm[u] * del(v, w))
                                        * gp
                                    + (dm[w] * dv[u] * dv[v]
                                        + dm[v] * dv[u] * dv[w]
                                        + dm[u] * dv[v] * dv[w])
                                        * gpp
                                    + m * (del(v, w) * dv[u]
                                        + del(u, w) * dv[v]
                                        + del(u, v) * dv[w])
                                        * gpp
                                    + m * dv[u] * dv[v] * dv[w] * gppp;
                            }
                        }
                    }
                }
            }
            // Cart → spherical.
            for s in 0..nsph {
                let col = shell.sph_off + s;
                let mut v = 0.0;
                for k in 0..ncart {
                    v += c2s[(k, s)] * cart_val[k];
                }
                phi[(ip, col)] = v;
                if let Some(gm) = grad.as_mut() {
                    for x in 0..3 {
                        let mut acc = 0.0;
                        for k in 0..ncart {
                            acc += c2s[(k, s)] * cart_d[x][k];
                        }
                        gm[x][(ip, col)] = acc;
                    }
                }
                if let Some(hm) = hess.as_mut() {
                    for h in 0..6 {
                        let mut acc = 0.0;
                        for k in 0..ncart {
                            acc += c2s[(k, s)] * cart_h[h][k];
                        }
                        hm[h][(ip, col)] = acc;
                    }
                }
                if let Some(tm) = third.as_mut() {
                    for t in 0..10 {
                        let mut acc = 0.0;
                        for k in 0..ncart {
                            acc += c2s[(k, s)] * cart_t[t][k];
                        }
                        tm[t][(ip, col)] = acc;
                    }
                }
            }
        }
    }
    AoBatch {
        phi,
        grad,
        hess,
        third,
    }
}

#[cfg(test)]
mod deriv3_tests {
    use super::*;
    use crate::basis::data;
    use crate::system::{Atom, Molecule};

    /// Third AO derivatives vs central differences of the second
    /// derivatives (all 10 components, s/p/d shells present).
    #[test]
    fn third_derivatives_match_fd_of_hessians() {
        let mol = Molecule::new(vec![Atom {
            z: 8,
            position: crate::math::Vec3::new(0.1, -0.2, 0.3),
        }]);
        let set = data::load("def2-svp").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let pts = [
            crate::math::Vec3::new(0.6, 0.4, -0.3),
            crate::math::Vec3::new(-0.2, 0.9, 0.5),
        ];
        let batch = eval_aos_deriv3(&mb, &pts, true, true, true);
        let third = batch.third.as_ref().unwrap();
        let h = 1e-5;
        let hpairs = [(0usize, 0usize), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        for (ip, &p) in pts.iter().enumerate() {
            for w in 0..3usize {
                let mut pp = p;
                let mut pm = p;
                match w {
                    0 => {
                        pp.x += h;
                        pm.x -= h;
                    }
                    1 => {
                        pp.y += h;
                        pm.y -= h;
                    }
                    _ => {
                        pp.z += h;
                        pm.z -= h;
                    }
                }
                let bp = eval_aos_deriv3(&mb, &[pp], true, true, false);
                let bm = eval_aos_deriv3(&mb, &[pm], true, true, false);
                let hp = bp.hess.as_ref().unwrap();
                let hm = bm.hess.as_ref().unwrap();
                for (hidx, &(u, v)) in hpairs.iter().enumerate() {
                    for ao in 0..mb.nao_sph {
                        let fd = (hp[hidx][(0, ao)] - hm[hidx][(0, ao)]) / (2.0 * h);
                        let an = third[third_index(u, v, w)][(ip, ao)];
                        assert!(
                            (an - fd).abs() < 1e-7 * (1.0 + fd.abs()),
                            "3rd ({u},{v},{w}) ao {ao}: {an} vs FD {fd}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_pair_packing_matches_all_nine_cartesian_pairs() {
        let expected = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];
        for u in 0..3 {
            for v in 0..3 {
                assert_eq!(second_index(u, v), expected[u][v]);
            }
        }
    }
    use crate::basis::data;
    use crate::system::{Atom, Molecule};

    /// AO gradients vs central finite differences of the AO values.
    #[test]
    fn ao_gradients_match_fd() {
        let set = data::load("def2-svp").unwrap();
        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.2),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 1.43, -0.89),
            },
        ]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let p0 = Vec3::new(0.31, -0.42, 0.77);
        let h = 1e-6;
        let batch = eval_aos(&mb, &[p0], true);
        let grad = batch.grad.as_ref().unwrap();
        for axis in 0..3 {
            let mut dp = p0;
            let mut dm = p0;
            match axis {
                0 => {
                    dp.x += h;
                    dm.x -= h;
                }
                1 => {
                    dp.y += h;
                    dm.y -= h;
                }
                _ => {
                    dp.z += h;
                    dm.z -= h;
                }
            }
            let bp = eval_aos(&mb, &[dp], false);
            let bm = eval_aos(&mb, &[dm], false);
            for mu in 0..mb.nao_sph {
                let fd = (bp.phi[(0, mu)] - bm.phi[(0, mu)]) / (2.0 * h);
                let an = grad[axis][(0, mu)];
                assert!(
                    (an - fd).abs() < 1e-7 * fd.abs().max(1.0),
                    "axis {axis} AO {mu}: {an} vs {fd}"
                );
            }
        }
    }

    /// AO second derivatives vs central FD of the analytic gradients.
    #[test]
    fn ao_hessians_match_fd() {
        let set = data::load("def2-svp").unwrap();
        let mol = Molecule::new(vec![Atom {
            z: 8,
            position: Vec3::new(0.1, -0.2, 0.3),
        }]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let p0 = Vec3::new(0.4, 0.15, -0.3);
        let h = 1e-6;
        let batch = eval_aos_full(&mb, &[p0], true, true);
        let hess = batch.hess.as_ref().unwrap();
        let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        for (idx, &(u, v)) in pairs.iter().enumerate() {
            let mut dp = p0;
            let mut dm = p0;
            match v {
                0 => {
                    dp.x += h;
                    dm.x -= h;
                }
                1 => {
                    dp.y += h;
                    dm.y -= h;
                }
                _ => {
                    dp.z += h;
                    dm.z -= h;
                }
            }
            let bp = eval_aos_full(&mb, &[dp], true, false);
            let bm = eval_aos_full(&mb, &[dm], true, false);
            let gp = bp.grad.as_ref().unwrap();
            let gm = bm.grad.as_ref().unwrap();
            for mu in 0..mb.nao_sph {
                let fd = (gp[u][(0, mu)] - gm[u][(0, mu)]) / (2.0 * h);
                let an = hess[idx][(0, mu)];
                assert!(
                    (an - fd).abs() < 1e-6 * fd.abs().max(1.0),
                    "hess ({u},{v}) AO {mu}: {an} vs {fd}"
                );
            }
        }
    }

    /// Grid + AO evaluation closes the loop: ∫ φ_μ φ_ν w = S_μν.
    #[test]
    fn grid_overlap_matches_analytic() {
        use crate::grid::{GridOptions, MolecularGrid};
        use crate::integrals::overlap_matrix;
        let set = data::load("sto-3g").unwrap();
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
                position: Vec3::new(0.0, -1.4309, -0.8867),
            },
        ]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let opts = GridOptions {
            level: 6,
            ..GridOptions::default()
        };
        let grid = MolecularGrid::build(&mol, &opts).unwrap();
        let batch = eval_aos(&mb, &grid.points, false);
        let n = mb.nao_sph;
        let mut s_grid = Matrix::zeros(n, n);
        for ip in 0..grid.len() {
            let w = grid.weights[ip];
            for mu in 0..n {
                let pm = batch.phi[(ip, mu)];
                if pm == 0.0 {
                    continue;
                }
                for nu in mu..n {
                    s_grid[(mu, nu)] += w * pm * batch.phi[(ip, nu)];
                }
            }
        }
        for mu in 0..n {
            for nu in 0..mu {
                s_grid[(mu, nu)] = s_grid[(nu, mu)];
            }
        }
        let s = overlap_matrix(&mb);
        let diff = s_grid.max_abs_diff(&s);
        assert!(diff < 5e-7, "grid overlap deviates by {diff}");
    }
}
