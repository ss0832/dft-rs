// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Forward-mode jets of reciprocal-space pair transforms under homogeneous
//! strain and Gamma-point displacements.
//!
//! A density-contracted pair transform
//! `sum_ij D_ij <a_i| e^{iG.r} |b_j>` with centres `A' = (1+eps)(A + u_A)`,
//! `B' = (1+eps)(B + T + u_B)` and wave vector `G' = (1+eps)^-T G` keeps the
//! phase `G' . P' = G . (P + u_P)` and changes through the Gaussian envelope
//! `exp(-|G'|^2/4p)`, the Hermite expansion coefficients of `A' - B'`, and
//! the powers of `G'`. Evaluating these with the nested dual numbers of
//! [`crate::dual`] gives exact first and second derivatives with respect to
//! the strain components and displacements that the caller seeds.

use crate::basis::{cart_components, MolShell};
use crate::dual::Scalar;
use crate::linalg::Matrix;
use crate::math::Vec3;

/// McMurchie-Davidson Hermite coefficients `E^{ij}_t` of one Cartesian
/// direction with `q = A - B`, laid out `[(i * (lb + 1) + j) * (la + lb + 1) + t]`.
pub(crate) fn hermite<S: Scalar>(la: usize, lb: usize, q: S, a: f64, b: f64) -> Vec<S> {
    let p = a + b;
    let mu = a * b / p;
    let nt = la + lb + 1;
    let index = |i: usize, j: usize, t: usize| (i * (lb + 1) + j) * nt + t;
    let mut e = vec![S::from_f64(0.0); (la + 1) * (lb + 1) * nt];
    e[index(0, 0, 0)] = (S::from_f64(-mu) * q * q).exp();
    let xpa = q * S::from_f64(-b / p);
    let xpb = q * S::from_f64(a / p);
    let half = S::from_f64(0.5 / p);
    for i in 0..=la {
        for j in 0..=lb {
            if i == 0 && j == 0 {
                continue;
            }
            let (pi, pj, x) = if i > 0 {
                (i - 1, j, xpa)
            } else {
                (i, j - 1, xpb)
            };
            for t in 0..=(i + j) {
                let mut v = x * e[index(pi, pj, t)];
                if t > 0 {
                    v = v + half * e[index(pi, pj, t - 1)];
                }
                if t < pi + pj {
                    v = v + S::from_f64((t + 1) as f64) * e[index(pi, pj, t + 1)];
                }
                e[index(i, j, t)] = v;
            }
        }
    }
    e
}

/// `(1 + eps)` applied to a jet vector.
pub(crate) fn strain_apply<S: Scalar>(eps: &[[S; 3]; 3], v: [S; 3]) -> [S; 3] {
    std::array::from_fn(|a| v[a] + eps[a][0] * v[0] + eps[a][1] * v[1] + eps[a][2] * v[2])
}

/// `(1 + eps)^-T` as a jet matrix.
pub(crate) fn inverse_transpose<S: Scalar>(eps: &[[S; 3]; 3]) -> [[S; 3]; 3] {
    let one = S::from_f64(1.0);
    let f: [[S; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| if i == j { one + eps[i][j] } else { eps[i][j] })
    });
    let det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
        - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
        + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
    // (F^-1)^T_ij = cofactor_ij / det.
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (a, b) = ((i + 1) % 3, (i + 2) % 3);
            let (c, d) = ((j + 1) % 3, (j + 2) % 3);
            (f[a][c] * f[b][d] - f[a][d] * f[b][c]) / det
        })
    })
}

/// `det(1 + eps)` as a jet.
pub(crate) fn strain_det<S: Scalar>(eps: &[[S; 3]; 3]) -> S {
    let one = S::from_f64(1.0);
    let f: [[S; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| if i == j { one + eps[i][j] } else { eps[i][j] })
    });
    f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
        - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
        + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0])
}

/// One reciprocal vector with its strained jet `G' = (1+eps)^-T G`.
pub(crate) struct StrainedWave<S: Scalar> {
    pub(crate) g: Vec3,
    pub(crate) strained: [S; 3],
    pub(crate) norm2: S,
}

pub(crate) fn strained_waves<S: Scalar>(gs: &[Vec3], eps: &[[S; 3]; 3]) -> Vec<StrainedWave<S>> {
    let m = inverse_transpose(eps);
    gs.iter()
        .map(|g| {
            let gv = [S::from_f64(g.x), S::from_f64(g.y), S::from_f64(g.z)];
            let strained: [S; 3] =
                std::array::from_fn(|e| m[e][0] * gv[0] + m[e][1] * gv[1] + m[e][2] * gv[2]);
            let norm2 =
                strained[0] * strained[0] + strained[1] * strained[1] + strained[2] * strained[2];
            StrainedWave {
                g: *g,
                strained,
                norm2,
            }
        })
        .collect()
}

/// Accumulate `sum_ij D_ij <a_i| e^{iG'.r} |b_j>` of one shell pair into
/// `re`/`im` for every strained wave. `centres` supplies the two displaced
/// centres before the strain `eps`; with `None` their phase is constant.
/// `d_cart` is the density block in Cartesian components.
#[allow(clippy::too_many_arguments)]
pub(crate) fn strained_pair_density<S: Scalar>(
    sa: &MolShell,
    sb: &MolShell,
    centres: Option<[[S; 3]; 2]>,
    eps: &[[S; 3]; 3],
    waves: &[StrainedWave<S>],
    d_cart: &Matrix,
    re: &mut [S],
    im: &mut [S],
) {
    let (la, lb) = (sa.l as usize, sb.l as usize);
    let ca = cart_components(sa.l);
    let cb = cart_components(sb.l);
    let [a_centre, b_centre] = centres
        .unwrap_or_else(|| [sa.center, sb.center].map(|centre| centre.to_array().map(S::from_f64)));
    let a_strained = strain_apply(eps, a_centre);
    let b_strained = strain_apply(eps, b_centre);
    let q: [S; 3] = std::array::from_fn(|k| a_strained[k] - b_strained[k]);
    let nt = la + lb + 1;
    let index = |i: usize, j: usize, t: usize| (i * (lb + 1) + j) * nt + t;
    let zero = S::from_f64(0.0);
    let mut powers: [Vec<S>; 3] = std::array::from_fn(|_| vec![S::from_f64(1.0); nt]);
    for (&ea, &coa) in sa.exps.iter().zip(&sa.coefs) {
        for (&eb, &cob) in sb.exps.iter().zip(&sb.coefs) {
            let p = ea + eb;
            let base = (std::f64::consts::PI / p).powf(1.5) * coa * cob;
            let tables: [Vec<S>; 3] = std::array::from_fn(|k| hermite(la, lb, q[k], ea, eb));
            // The phase G'.P' = G.(P + u_P) is strain invariant.
            let centre: [S; 3] = std::array::from_fn(|k| {
                (a_centre[k] * S::from_f64(ea) + b_centre[k] * S::from_f64(eb)) / S::from_f64(p)
            });
            let centre_value = centre.map(Scalar::value);
            for (gi, wave) in waves.iter().enumerate() {
                let envelope = wave.norm2.scale(-0.25 / p).exp().scale(base);
                if envelope.value().abs() < 1e-18 {
                    continue;
                }
                let (cs, sn) = if centres.is_none() {
                    let phase = wave.g.x * centre_value[0]
                        + wave.g.y * centre_value[1]
                        + wave.g.z * centre_value[2];
                    (S::from_f64(phase.cos()), S::from_f64(phase.sin()))
                } else {
                    cos_sin(
                        S::from_f64(wave.g.x) * centre[0]
                            + S::from_f64(wave.g.y) * centre[1]
                            + S::from_f64(wave.g.z) * centre[2],
                    )
                };
                for (k, row) in powers.iter_mut().enumerate() {
                    for t in 1..nt {
                        row[t] = row[t - 1] * wave.strained[k];
                    }
                }
                let mut acc = [zero; 4];
                for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                    let (ax, ay, az) = (ax as usize, ay as usize, az as usize);
                    for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                        let dij = d_cart[(i, j)];
                        if dij == 0.0 {
                            continue;
                        }
                        let (bx, by, bz) = (bx as usize, by as usize, bz as usize);
                        let dij = S::from_f64(dij);
                        for t in 0..=(ax + bx) {
                            let et = tables[0][index(ax, bx, t)] * powers[0][t];
                            for u in 0..=(ay + by) {
                                let eu = tables[1][index(ay, by, u)] * powers[1][u];
                                let etu = dij * et * eu;
                                for v in 0..=(az + bz) {
                                    let ev = tables[2][index(az, bz, v)] * powers[2][v];
                                    acc[(t + u + v) % 4] = acc[(t + u + v) % 4] + etu * ev;
                                }
                            }
                        }
                    }
                }
                let s_re = acc[0] - acc[2];
                let s_im = acc[1] - acc[3];
                re[gi] = re[gi] + envelope * (s_re * cs - s_im * sn);
                im[gi] = im[gi] + envelope * (s_re * sn + s_im * cs);
            }
        }
    }
}

/// Accumulate `arrays` jet transforms over every wave from all `items`,
/// in parallel over blocks of waves: each block visits every item once, so
/// the memory is one array per transform rather than one per parallel task
/// (a nested strain jet carries 100 numbers per wave). `visit` adds one
/// item's contribution on a block of waves into the block's arrays.
pub(crate) fn strained_transforms<S: Scalar + Send + Sync, P: Sync>(
    items: &[P],
    waves: &[StrainedWave<S>],
    arrays: usize,
    visit: &(dyn Fn(&P, &[StrainedWave<S>], &mut [Vec<S>]) + Sync),
) -> Vec<Vec<S>> {
    use rayon::prelude::*;
    let ng = waves.len();
    let blocks = 4 * rayon::current_num_threads().max(1);
    let chunk = ng.div_ceil(blocks).max(64);
    let starts: Vec<usize> = (0..ng).step_by(chunk).collect();
    let parts: Vec<Vec<Vec<S>>> = starts
        .par_iter()
        .map(|&start| {
            let end = (start + chunk).min(ng);
            let mut acc = vec![vec![S::from_f64(0.0); end - start]; arrays];
            for item in items {
                visit(item, &waves[start..end], &mut acc);
            }
            acc
        })
        .collect();
    let mut out: Vec<Vec<S>> = (0..arrays).map(|_| Vec::with_capacity(ng)).collect();
    for part in parts {
        for (o, p) in out.iter_mut().zip(part) {
            o.extend(p);
        }
    }
    out
}

/// `(cos x, sin x)` of a jet by the expansion about its value. Exact for
/// `f64` and for nested jets of at most second order (first-order dual
/// numbers nested once), where the nilpotent part `dx` has `dx^3 = 0`;
/// deeper jets need further terms.
pub(crate) fn cos_sin<S: Scalar>(x: S) -> (S, S) {
    // x = x0 + dx with dx nilpotent in the jet algebra: cos x = cos x0 cos dx
    // - sin x0 sin dx, with cos dx = 1 - dx^2/2 and sin dx = dx - dx^3/6 to
    // the order carried by nested second-order jets (dx^3 = 0).
    let x0 = x.value();
    let dx = x - S::from_f64(x0);
    let dx2 = dx * dx;
    let c = S::from_f64(1.0) - dx2 * S::from_f64(0.5);
    let s = dx - dx2 * dx * S::from_f64(1.0 / 6.0);
    let (c0, s0) = (x0.cos(), x0.sin());
    (
        c * S::from_f64(c0) - s * S::from_f64(s0),
        s * S::from_f64(c0) + c * S::from_f64(s0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_jets_reproduce_the_pair_density_transform() {
        let set = crate::basis::g94::parse(
            "jet-test",
            "H 0\nS 2 1.00\n 2.4 0.6\n 0.55 0.5\nP 2 1.00\n 1.6 0.6\n 0.5 0.5\nD 1 1.00\n 0.9 1.0\n****\n",
        )
        .unwrap();
        let molecule = crate::system::Molecule::new(vec![
            crate::system::Atom {
                z: 1,
                position: Vec3::new(0.1, 0.2, -0.3),
            },
            crate::system::Atom {
                z: 1,
                position: Vec3::new(1.2, 0.9, 0.8),
            },
        ]);
        let mb = crate::basis::MolBasis::build(&molecule, &set).unwrap();
        let gs = vec![
            Vec3::zero(),
            Vec3::new(0.7, -0.3, 0.4),
            Vec3::new(-1.1, 0.5, 0.9),
        ];
        let zero = [[0.0f64; 3]; 3];
        let waves = strained_waves(&gs, &zero);
        let eps = crate::periodic::strain::strain_seed();
        let strain_waves = strained_waves(&gs, &eps);
        let z = crate::dual::Jet2::constant(0.0);
        let mut max_defect = 0.0_f64;
        for sa in &mb.shells {
            for sb in &mb.shells {
                let nca = cart_components(sa.l).len();
                let ncb = cart_components(sb.l).len();
                let mut d = Matrix::zeros(nca, ncb);
                for i in 0..nca {
                    for j in 0..ncb {
                        d[(i, j)] = ((i * 7 + j * 3) as f64 * 0.31).sin();
                    }
                }
                let (mut re0, mut im0) = (vec![0.0; 3], vec![0.0; 3]);
                crate::periodic::planewave::pair_density_transform(
                    sa, sb, &gs, &d, &mut re0, &mut im0,
                );
                let (mut re1, mut im1) = (vec![0.0; 3], vec![0.0; 3]);
                let ac = [sa.center.x, sa.center.y, sa.center.z];
                let bc = [sb.center.x, sb.center.y, sb.center.z];
                strained_pair_density(
                    sa,
                    sb,
                    Some([ac, bc]),
                    &zero,
                    &waves,
                    &d,
                    &mut re1,
                    &mut im1,
                );
                for g in 0..3 {
                    assert!((re0[g] - re1[g]).abs() < 1e-12 && (im0[g] - im1[g]).abs() < 1e-12);
                }
                let centres = [ac, bc].map(|c| c.map(crate::dual::Jet2::constant));
                let (mut re0, mut im0) = ([z; 3], [z; 3]);
                let (mut re1, mut im1) = ([z; 3], [z; 3]);
                strained_pair_density(
                    sa,
                    sb,
                    Some(centres),
                    &eps,
                    &strain_waves,
                    &d,
                    &mut re0,
                    &mut im0,
                );
                strained_pair_density(sa, sb, None, &eps, &strain_waves, &d, &mut re1, &mut im1);
                for (a, b) in re0.into_iter().chain(im0).zip(re1.into_iter().chain(im1)) {
                    max_defect = max_defect.max((a.value - b.value).abs());
                    for (x, y) in a
                        .gradient
                        .into_iter()
                        .chain(a.hessian)
                        .zip(b.gradient.into_iter().chain(b.hessian))
                    {
                        max_defect = max_defect.max((x - y).abs());
                    }
                }
            }
        }
        println!("stationary-centre reciprocal ket jet maximum defect {max_defect:.12e}");
        assert!(max_defect < 1e-12);
    }
}
