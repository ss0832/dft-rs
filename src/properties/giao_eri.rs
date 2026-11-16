// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Magnetic-field derivative of the two-electron integrals that the London
//! phase factors introduce.
//!
//! With `chi_mu^B = exp(-(i/2) [B x (R_mu - O)] . (r - O)) chi_mu`, the four
//! phases of `(mn|ls)` combine into one factor
//! `exp((i/2) B . [(R_m - R_n) x r_1O + (R_l - R_s) x r_2O])`, so
//!
//! ```text
//! d(mn|ls)/dB_a = (i/2) [ (R_m - R_n) x r_1O + (R_l - R_s) x r_2O ]_a (mn|ls)
//! ```
//!
//! i.e. a moment-weighted four-centre integral. The `i` is taken out
//! throughout, as everywhere else in the magnetic layer.
//!
//! **How the moments are evaluated.** No new integral engine: a Cartesian
//! factor on one index is a shift of that index's angular momentum,
//!
//! ```text
//! (r - O)_k chi_n = (r - R_n)_k chi_n + (R_n - O)_k chi_n
//! ```
//!
//! and `(r - R_n)_k chi_n` is the SAME contraction with the Cartesian powers
//! raised by one in direction `k`. So the moment-weighted quartet is the
//! ordinary quartet with one shell raised, left Cartesian, and contracted
//! with a transform that maps each spherical component of the unraised
//! shell onto its raised Cartesian partner. Everything else — E tables,
//! the Hermite-Coulomb tensor, screening-free contraction — is the existing
//! McMurchie–Davidson path.
//!
//! **What is returned.** Not the four-index tensor (there is no reason to
//! store it) but its contraction with a density, i.e. the two-electron part
//! of `dF/dB_a`:
//!
//! ```text
//! G_a[m,n] = j_scale sum_ls P_ls dD_a(mn|ls)
//!            - (k_scale/2) sum_ls P_sl dD_a(ml|sn)
//! ```
//!
//! The exchange index order matters here in a way it never does for
//! ordinary ERIs: `(ml|sn)` and `(ml|ns)` are the same integral but carry
//! DIFFERENT phase weights, and using the wrong one produces a symmetric
//! `G_a` instead of the antisymmetric one an imaginary perturbation must
//! have.

use crate::basis::{aos, cart_components, n_cart, n_sph, MolBasis, MolShell};
use crate::integrals::eri::{eri_quartet, eri_quartet_masked};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use rayon::prelude::*;

use super::nmr::curl_terms;

/// Position of a Cartesian component in `cart_components(l)`'s fixed order.
fn cart_index(l: u8, (lx, ly, _lz): (i32, i32, i32)) -> usize {
    let l = l as i32;
    let d = (l - lx) as usize;
    d * (d + 1) / 2 + (l - lx - ly) as usize
}

#[cfg(test)]
mod orbit_regression {
    use super::*;
    use crate::basis::{data, BasisSet, ElementBasis, Shell};
    use crate::{Atom, Molecule};

    fn ordered_reference(
        mb: &MolBasis,
        p: &Matrix,
        j_scale: f64,
        k_scale: f64,
        origin: Vec3,
        kernel: CoulombKernel,
    ) -> [Matrix; 3] {
        let tables: Vec<[Matrix; 3]> = (0..=mb.max_l)
            .map(|l| std::array::from_fn(|k| moment_transform(l, k)))
            .collect();
        let mut j = std::array::from_fn(|_| Matrix::zeros(mb.nao_sph, mb.nao_sph));
        let mut k = std::array::from_fn(|_| Matrix::zeros(mb.nao_sph, mb.nao_sph));
        for a in &mb.shells {
            for b in &mb.shells {
                for c in &mb.shells {
                    for d in &mb.shells {
                        accumulate(
                            a,
                            b,
                            c,
                            d,
                            &tables,
                            p,
                            origin,
                            kernel,
                            j_scale != 0.0,
                            k_scale != 0.0,
                            &mut j,
                            &mut k,
                        );
                    }
                }
            }
        }
        std::array::from_fn(|axis| {
            let mut value = j[axis].clone();
            for (target, exchange) in value.as_mut_slice().iter_mut().zip(k[axis].as_slice()) {
                *target = j_scale * *target - 0.5 * k_scale * exchange;
            }
            value
        })
    }

    fn density(n: usize, symmetric: bool) -> Matrix {
        let mut p = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let a = 0.013 * ((i * 7 + j * 3 + 1) as f64).sin();
                let b = 0.013 * ((j * 7 + i * 3 + 1) as f64).sin();
                p[(i, j)] = if symmetric { 0.5 * (a + b) } else { a };
            }
        }
        p
    }

    fn primitive_basis(with_d: bool) -> MolBasis {
        let positions = [
            Vec3::new(0.13, -0.21, 0.19),
            Vec3::new(1.37, 0.16, -0.08),
            Vec3::new(-0.19, 1.53, 0.37),
            Vec3::new(0.42, 0.29, 1.68),
        ];
        let mol = Molecule::new(
            positions
                .into_iter()
                .enumerate()
                .map(|(i, position)| Atom {
                    z: (i + 1) as u8,
                    position,
                })
                .collect(),
        );
        let mut definition = BasisSet::new("numerical-shell-orbit-test");
        for i in 0..4 {
            let mut shells = vec![Shell {
                l: 0,
                exps: vec![0.73 + 0.11 * i as f64],
                coefs: vec![1.0],
            }];
            if i == 1 {
                shells.push(Shell {
                    l: 1,
                    exps: vec![0.61],
                    coefs: vec![1.0],
                });
            }
            if i == 2 && with_d {
                shells.push(Shell {
                    l: 2,
                    exps: vec![0.87],
                    coefs: vec![1.0],
                });
            }
            definition.insert(ElementBasis {
                z: (i + 1) as u8,
                shells,
            });
        }
        MolBasis::build(&mol, &definition).unwrap()
    }

    fn check_all_ao(mb: &MolBasis, p: &Matrix, origin: Vec3, kernel: CoulombKernel) {
        for (j, k) in [(1.0, 0.0), (0.0, -2.0), (0.37, 0.81)] {
            let old = ordered_reference(mb, p, j, k, origin, kernel);
            let new = giao_eri_fock_derivative_weighted(mb, p, j, k, origin, kernel);
            let mut maximum = 0.0_f64;
            for axis in 0..3 {
                for (index, (a, b)) in old[axis]
                    .as_slice()
                    .iter()
                    .zip(new[axis].as_slice())
                    .enumerate()
                {
                    let error = (a - b).abs();
                    maximum = maximum.max(error);
                    assert!(
                        error < 1e-11,
                        "axis={axis}, AO index={index}, J={j}, K={k}, error={error:e}"
                    );
                }
            }
            println!("shell-orbit all-AO J={j} K={k} maximum={maximum:e}");
        }
    }

    #[test]
    fn every_ao_j_and_k_matches_ordered_quartets_on_c1_sp_and_d_bases() {
        for with_d in [false, true] {
            let basis = primitive_basis(with_d);
            for symmetric in [false, true] {
                let p = density(basis.nao_sph, symmetric);
                for origin in [Vec3::zero(), Vec3::new(0.29, -0.17, 0.33)] {
                    for kernel in [
                        CoulombKernel::Coulomb,
                        CoulombKernel::Erf { omega: 0.37 },
                        CoulombKernel::Erfc { omega: 0.37 },
                    ] {
                        check_all_ao(&basis, &p, origin, kernel);
                    }
                }
            }
        }
    }

    #[test]
    fn contracted_water_and_gauge_origins_match_all_ordered_ao_entries() {
        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.13, -0.21, 0.19),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.57, 0.16, -0.08),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.19, 1.43, 0.37),
            },
        ]);
        let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        for symmetric in [false, true] {
            let p = density(basis.nao_sph, symmetric);
            for origin in [Vec3::zero(), Vec3::new(0.29, -0.17, 0.33)] {
                check_all_ao(&basis, &p, origin, CoulombKernel::Coulomb);
                check_all_ao(&basis, &p, origin, CoulombKernel::Erf { omega: 0.37 });
            }
        }
    }
}

/// `(n_cart(l+1) x n_sph(l))` matrix expanding `(r - R)_k chi^sph_j` of
/// shell `l` in the Cartesian components of shell `l+1`.
fn moment_transform(l: u8, k: usize) -> Matrix {
    let s = aos::cart_to_sph(l);
    let mut t = Matrix::zeros(n_cart(l + 1), n_sph(l));
    for (ic, &(cx, cy, cz)) in cart_components(l).iter().enumerate() {
        let mut up = (cx, cy, cz);
        match k {
            0 => up.0 += 1,
            1 => up.1 += 1,
            _ => up.2 += 1,
        }
        let ir = cart_index(l + 1, up);
        for j in 0..n_sph(l) {
            t[(ir, j)] += s[(ic, j)];
        }
    }
    t
}

/// The same shell with its angular momentum raised by one, contraction
/// coefficients untouched — multiplying by `(r - R)_k` does not renormalize
/// anything, it only shifts which Cartesian monomial each primitive carries.
fn raised(sh: &MolShell) -> MolShell {
    MolShell {
        l: sh.l + 1,
        ..sh.clone()
    }
}

/// Two-electron part of `dF/dB_a` with the factor `i` taken out: real and
/// antisymmetric.
///
/// `p` is the TOTAL density; `cx` the exact-exchange fraction of the
/// functional (1 for Hartree-Fock, 0 for a pure functional — in which case
/// only the Coulomb-like term survives, and it does NOT vanish).
pub fn giao_eri_fock_derivative(
    mb: &MolBasis,
    p: &Matrix,
    cx: f64,
    origin: Vec3,
    kernel: CoulombKernel,
) -> [Matrix; 3] {
    giao_eri_fock_derivative_weighted(mb, p, 1.0, cx, origin, kernel)
}

/// Weighted two-electron part of `dF/dB_a`, with independent Coulomb-like
/// and exchange-like coefficients.
///
/// The separation is essential for range-separated hybrids: their explicit
/// magnetic derivative is
/// `J'_C - 1/2 cx K'_C - 1/2 clr K'_erf`, with no long-range `J'_erf`.
/// Keeping the weights independent prevents that spurious Coulomb term while
/// [`giao_eri_fock_derivative`] retains its public full-range behaviour.
pub fn giao_eri_fock_derivative_weighted(
    mb: &MolBasis,
    p: &Matrix,
    j_scale: f64,
    k_scale: f64,
    origin: Vec3,
    kernel: CoulombKernel,
) -> [Matrix; 3] {
    let ns = mb.shells.len();
    let nao = mb.nao_sph;
    if j_scale == 0.0 && k_scale == 0.0 {
        return std::array::from_fn(|_| Matrix::zeros(nao, nao));
    }
    let lmax = mb.max_l;
    // Moment transforms, one per (l, direction), built once.
    let tables: Vec<[Matrix; 3]> = (0..=lmax)
        .map(|l| std::array::from_fn(|k| moment_transform(l, k)))
        .collect();

    let partials: Vec<([Matrix; 3], [Matrix; 3])> = (0..ns)
        .into_par_iter()
        .map(|ia| {
            let mut jm: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(nao, nao));
            let mut km: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(nao, nao));
            // One partial per outer shell, rather than per shell pair, keeps
            // the accumulation memory O(ns * nao^2).
            for ib in 0..=ia {
                for ic in 0..=ia {
                    let last_d = if ic == ia { ib } else { ic };
                    for id in 0..=last_d {
                        accumulate_orbit(
                            mb,
                            [ia, ib, ic, id],
                            &tables,
                            p,
                            origin,
                            kernel,
                            j_scale != 0.0,
                            k_scale != 0.0,
                            &mut jm,
                            &mut km,
                        );
                    }
                }
            }
            (jm, km)
        })
        .collect();

    let mut out: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(nao, nao));
    for (jm, km) in partials {
        for a in 0..3 {
            for (dst, (jv, kv)) in out[a]
                .as_mut_slice()
                .iter_mut()
                .zip(jm[a].as_slice().iter().zip(km[a].as_slice()))
            {
                *dst += j_scale * *jv - 0.5 * k_scale * *kv;
            }
        }
    }
    out
}

/// Reuse the ordinary moment integrals on their eightfold shell orbit.
/// The magnetic derivative itself is not eightfold symmetric: reversing
/// the first or second pair changes only that pair's phase contribution.
#[allow(clippy::too_many_arguments)]
fn accumulate_orbit(
    mb: &MolBasis,
    indices: [usize; 4],
    tables: &[[Matrix; 3]],
    p: &Matrix,
    origin: Vec3,
    kernel: CoulombKernel,
    need_j: bool,
    need_k: bool,
    jm: &mut [Matrix; 3],
    km: &mut [Matrix; 3],
) {
    let [ia, ib, ic, id] = indices;
    let (sa, sb, sc, sd) = (
        &mb.shells[ia],
        &mb.shells[ib],
        &mb.shells[ic],
        &mb.shells[id],
    );
    let ab = sa.center - sb.center;
    let cd = sc.center - sd.center;
    // Compare components exactly: a squared norm can underflow for a
    // nonzero displacement. This is an algebraic zero, not screening.
    if ab.x == 0.0 && ab.y == 0.0 && ab.z == 0.0 && cd.x == 0.0 && cd.y == 0.0 && cd.z == 0.0 {
        return;
    }
    let q0 = eri_quartet(sa, sb, sc, sd, kernel);
    let qb = eri_quartet_masked(sa, &raised(sb), sc, sd, kernel, [true, false, true, true]);
    let qd = eri_quartet_masked(sa, sb, sc, &raised(sd), kernel, [true, true, true, false]);
    let dimensions = [n_sph(sa.l), n_sph(sb.l), n_sph(sc.l), n_sph(sd.l)];
    let permutations = [
        ([0, 1, 2, 3], 1.0, 1.0),
        ([1, 0, 2, 3], -1.0, 1.0),
        ([0, 1, 3, 2], 1.0, -1.0),
        ([1, 0, 3, 2], -1.0, -1.0),
        ([2, 3, 0, 1], 1.0, 1.0),
        ([3, 2, 0, 1], 1.0, -1.0),
        ([2, 3, 1, 0], -1.0, 1.0),
        ([3, 2, 1, 0], -1.0, -1.0),
    ];
    let mut seen = Vec::with_capacity(8);
    let mut unique = Vec::with_capacity(8);
    for (permutation, sign_ab, sign_cd) in permutations {
        let key = permutation.map(|i| indices[i]);
        if !seen.contains(&key) {
            seen.push(key);
            unique.push((permutation, sign_ab, sign_cd));
        }
    }
    for i in 0..dimensions[0] {
        for j in 0..dimensions[1] {
            for k in 0..dimensions[2] {
                for l in 0..dimensions[3] {
                    let base = q0.get(i, j, k, l);
                    let mut w1 = [0.0; 3];
                    let mut w2 = [0.0; 3];
                    for axis in 0..3 {
                        w1[axis] = (sb.center.get(axis) - origin.get(axis)) * base;
                        w2[axis] = (sd.center.get(axis) - origin.get(axis)) * base;
                        for cp in 0..n_cart(sb.l + 1) {
                            let t = tables[sb.l as usize][axis][(cp, j)];
                            if t != 0.0 {
                                w1[axis] += t * qb.get(i, cp, k, l);
                            }
                        }
                        for cp in 0..n_cart(sd.l + 1) {
                            let t = tables[sd.l as usize][axis][(cp, l)];
                            if t != 0.0 {
                                w2[axis] += t * qd.get(i, j, k, cp);
                            }
                        }
                    }
                    let phase_ab: [f64; 3] = std::array::from_fn(|axis| {
                        let (x, y, _) = curl_terms(axis)[0];
                        0.5 * (ab.get(x) * w1[y] - ab.get(y) * w1[x])
                    });
                    let phase_cd: [f64; 3] = std::array::from_fn(|axis| {
                        let (x, y, _) = curl_terms(axis)[0];
                        0.5 * (cd.get(x) * w2[y] - cd.get(y) * w2[x])
                    });
                    let ao = [
                        sa.sph_off + i,
                        sb.sph_off + j,
                        sc.sph_off + k,
                        sd.sph_off + l,
                    ];
                    for &(permutation, sign_ab, sign_cd) in &unique {
                        let [m, n, r, s] = permutation.map(|index| ao[index]);
                        for axis in 0..3 {
                            let value = sign_ab * phase_ab[axis] + sign_cd * phase_cd[axis];
                            if need_j {
                                jm[axis][(m, n)] += p[(r, s)] * value;
                            }
                            if need_k {
                                km[axis][(m, s)] += p[(r, n)] * value;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// One shell quartet's contribution to the Coulomb-like and exchange-like
/// halves of the two-electron field derivative.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn accumulate(
    sa: &MolShell,
    sb: &MolShell,
    sc: &MolShell,
    sd: &MolShell,
    tables: &[[Matrix; 3]],
    p: &Matrix,
    origin: Vec3,
    kernel: CoulombKernel,
    need_j: bool,
    need_k: bool,
    jm: &mut [Matrix; 3],
    km: &mut [Matrix; 3],
) {
    let (na, nb, nc, nd) = (n_sph(sa.l), n_sph(sb.l), n_sph(sc.l), n_sph(sd.l));
    let q0 = eri_quartet(sa, sb, sc, sd, kernel);
    let qb = eri_quartet_masked(sa, &raised(sb), sc, sd, kernel, [true, false, true, true]);
    let qd = eri_quartet_masked(sa, sb, sc, &raised(sd), kernel, [true, true, true, false]);
    let size = na * nb * nc * nd;
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * nb + j) * nc + k) * nd + l;
    let tb = &tables[sb.l as usize];
    let td = &tables[sd.l as usize];
    let ncb = n_cart(sb.l + 1);
    let ncd = n_cart(sd.l + 1);
    // w1[k] = (mn| (r_1 - O)_k |ls), w2[k] the same on electron 2.
    let mut w1: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; size]);
    let mut w2: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; size]);
    for k in 0..3 {
        let shift_b = sb.center.get(k) - origin.get(k);
        let shift_d = sd.center.get(k) - origin.get(k);
        for i in 0..na {
            for j in 0..nb {
                for k2 in 0..nc {
                    for l in 0..nd {
                        let base = q0.get(i, j, k2, l);
                        let mut v1 = shift_b * base;
                        for cp in 0..ncb {
                            let t = tb[k][(cp, j)];
                            if t != 0.0 {
                                v1 += t * qb.get(i, cp, k2, l);
                            }
                        }
                        w1[k][idx(i, j, k2, l)] = v1;
                        let mut v2 = shift_d * base;
                        for cp in 0..ncd {
                            let t = td[k][(cp, l)];
                            if t != 0.0 {
                                v2 += t * qd.get(i, j, k2, cp);
                            }
                        }
                        w2[k][idx(i, j, k2, l)] = v2;
                    }
                }
            }
        }
    }
    let (oa, ob, oc, od) = (sa.sph_off, sb.sph_off, sc.sph_off, sd.sph_off);
    let ab = sa.center - sb.center;
    let cd = sc.center - sd.center;
    for a in 0..3 {
        let (jc, kc, _) = curl_terms(a)[0];
        for i in 0..na {
            for j in 0..nb {
                for k2 in 0..nc {
                    for l in 0..nd {
                        let t = idx(i, j, k2, l);
                        let dv = 0.5
                            * (ab.get(jc) * w1[kc][t] - ab.get(kc) * w1[jc][t]
                                + cd.get(jc) * w2[kc][t]
                                - cd.get(kc) * w2[jc][t]);
                        if dv == 0.0 {
                            continue;
                        }
                        if need_j {
                            jm[a][(oa + i, ob + j)] += p[(oc + k2, od + l)] * dv;
                        }
                        if need_k {
                            km[a][(oa + i, od + l)] += p[(oc + k2, ob + j)] * dv;
                        }
                    }
                }
            }
        }
    }
}

/// The bracket `dD_a(mn|ls)` itself for one shell quartet, three components
/// flattened as `[a][((i*nb + j)*nc + k)*nd + l]`.
///
/// Only the tests want the raw tensor (permutational symmetries, the
/// bra/ket cross-check, the gauge-origin shift identity); the production
/// path never materializes it.
#[cfg(test)]
fn giao_eri_quartet(
    sa: &MolShell,
    sb: &MolShell,
    sc: &MolShell,
    sd: &MolShell,
    origin: Vec3,
    kernel: CoulombKernel,
) -> [Vec<f64>; 3] {
    let (na, nb, nc, nd) = (n_sph(sa.l), n_sph(sb.l), n_sph(sc.l), n_sph(sd.l));
    let size = na * nb * nc * nd;
    let q0 = eri_quartet(sa, sb, sc, sd, kernel);
    let qb = eri_quartet_masked(sa, &raised(sb), sc, sd, kernel, [true, false, true, true]);
    let qd = eri_quartet_masked(sa, sb, sc, &raised(sd), kernel, [true, true, true, false]);
    let tb: [Matrix; 3] = std::array::from_fn(|k| moment_transform(sb.l, k));
    let td: [Matrix; 3] = std::array::from_fn(|k| moment_transform(sd.l, k));
    let ncb = n_cart(sb.l + 1);
    let ncd = n_cart(sd.l + 1);
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * nb + j) * nc + k) * nd + l;
    let mut w1: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; size]);
    let mut w2: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; size]);
    for k in 0..3 {
        let shift_b = sb.center.get(k) - origin.get(k);
        let shift_d = sd.center.get(k) - origin.get(k);
        for i in 0..na {
            for j in 0..nb {
                for k2 in 0..nc {
                    for l in 0..nd {
                        let base = q0.get(i, j, k2, l);
                        let mut v1 = shift_b * base;
                        for cp in 0..ncb {
                            v1 += tb[k][(cp, j)] * qb.get(i, cp, k2, l);
                        }
                        w1[k][idx(i, j, k2, l)] = v1;
                        let mut v2 = shift_d * base;
                        for cp in 0..ncd {
                            v2 += td[k][(cp, l)] * qd.get(i, j, k2, cp);
                        }
                        w2[k][idx(i, j, k2, l)] = v2;
                    }
                }
            }
        }
    }
    let ab = sa.center - sb.center;
    let cd = sc.center - sd.center;
    std::array::from_fn(|a| {
        let (jc, kc, _) = curl_terms(a)[0];
        (0..size)
            .map(|t| {
                0.5 * (ab.get(jc) * w1[kc][t] - ab.get(kc) * w1[jc][t] + cd.get(jc) * w2[kc][t]
                    - cd.get(kc) * w2[jc][t])
            })
            .collect()
    })
}

/// Moment-weighted quartet `(m n| (r_1 - O)_k |l s)` by raising the KET
/// index of the bra pair — the production route.
#[cfg(test)]
fn moment_quartet_via_ket(
    sa: &MolShell,
    sb: &MolShell,
    sc: &MolShell,
    sd: &MolShell,
    k: usize,
    origin: Vec3,
    kernel: CoulombKernel,
) -> Vec<f64> {
    let (na, nb, nc, nd) = (n_sph(sa.l), n_sph(sb.l), n_sph(sc.l), n_sph(sd.l));
    let q0 = eri_quartet(sa, sb, sc, sd, kernel);
    let qb = eri_quartet_masked(sa, &raised(sb), sc, sd, kernel, [true, false, true, true]);
    let tb = moment_transform(sb.l, k);
    let ncb = n_cart(sb.l + 1);
    let shift = sb.center.get(k) - origin.get(k);
    let mut out = vec![0.0; na * nb * nc * nd];
    for i in 0..na {
        for j in 0..nb {
            for k2 in 0..nc {
                for l in 0..nd {
                    let mut v = shift * q0.get(i, j, k2, l);
                    for cp in 0..ncb {
                        v += tb[(cp, j)] * qb.get(i, cp, k2, l);
                    }
                    out[((i * nb + j) * nc + k2) * nd + l] = v;
                }
            }
        }
    }
    out
}

/// Moment-weighted quartet `(m n| (r_1 - O)_k |l s)` computed by raising
/// the BRA index instead — an independent route to the same number, which
/// the tests use to gate the raise-and-transform machinery.
#[cfg(test)]
fn moment_quartet_via_bra(
    sa: &MolShell,
    sb: &MolShell,
    sc: &MolShell,
    sd: &MolShell,
    k: usize,
    origin: Vec3,
    kernel: CoulombKernel,
) -> Vec<f64> {
    let (na, nb, nc, nd) = (n_sph(sa.l), n_sph(sb.l), n_sph(sc.l), n_sph(sd.l));
    let q0 = eri_quartet(sa, sb, sc, sd, kernel);
    let qa = eri_quartet_masked(&raised(sa), sb, sc, sd, kernel, [false, true, true, true]);
    let ta = moment_transform(sa.l, k);
    let nca = n_cart(sa.l + 1);
    let shift = sa.center.get(k) - origin.get(k);
    let mut out = vec![0.0; na * nb * nc * nd];
    for i in 0..na {
        for j in 0..nb {
            for k2 in 0..nc {
                for l in 0..nd {
                    let mut v = shift * q0.get(i, j, k2, l);
                    for cp in 0..nca {
                        v += ta[(cp, i)] * qa.get(cp, j, k2, l);
                    }
                    out[((i * nb + j) * nc + k2) * nd + l] = v;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::system::{Atom, Molecule};

    fn mk(l: u8, exps: Vec<f64>, coefs: Vec<f64>, c: Vec3) -> MolShell {
        MolShell {
            l,
            atom: 0,
            center: c,
            exps,
            coefs,
            sph_off: 0,
            cart_off: 0,
        }
    }

    /// Four shells with different centres, angular momenta and contraction
    /// lengths — enough that the raise-and-transform bookkeeping cannot be
    /// right by accident.
    fn quad() -> [MolShell; 4] {
        [
            mk(1, vec![1.3, 0.4], vec![0.6, 0.5], Vec3::new(0.0, 0.1, -0.2)),
            mk(0, vec![0.9], vec![1.0], Vec3::new(0.3, -0.4, 0.5)),
            mk(2, vec![1.7], vec![1.0], Vec3::new(-0.4, 0.2, 0.1)),
            mk(1, vec![0.6], vec![1.0], Vec3::new(0.7, 0.3, -0.6)),
        ]
    }

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
                position: Vec3::new(0.0, -1.4309, -0.8867),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        (mol, mb)
    }

    /// A Cartesian factor on electron 1 can be attached to EITHER function
    /// of the bra pair. The two routes raise different shells and use
    /// different cart->sph transforms, so agreeing to machine precision
    /// gates the moment transform, the raised-shell construction and the
    /// masked-quartet indexing together.
    #[test]
    fn moment_weighted_quartet_agrees_between_bra_and_ket_routes() {
        let sh = quad();
        let o = Vec3::new(0.17, -0.41, 0.28);
        for kernel in [CoulombKernel::Coulomb, CoulombKernel::Erf { omega: 0.33 }] {
            for k in 0..3 {
                let via_ket = moment_quartet_via_ket(&sh[0], &sh[1], &sh[2], &sh[3], k, o, kernel);
                let via_bra = moment_quartet_via_bra(&sh[0], &sh[1], &sh[2], &sh[3], k, o, kernel);
                let mut worst = 0.0_f64;
                let mut scale = 0.0_f64;
                for (a, b) in via_ket.iter().zip(&via_bra) {
                    worst = worst.max((a - b).abs());
                    scale = scale.max(a.abs());
                }
                assert!(
                    worst < 1e-11 * scale.max(1.0),
                    "kernel {kernel:?}, direction {k}: routes differ by {worst:e} (scale {scale:e})"
                );
                assert!(
                    scale > 1e-8,
                    "kernel {kernel:?}, direction {k} moment is identically zero"
                );
            }
        }
    }

    /// The field-derivative bracket is symmetric under exchanging the two
    /// electron pairs, since the two phase weights enter symmetrically.
    #[test]
    fn field_derivative_bracket_is_pair_exchange_symmetric() {
        let sh = quad();
        let o = Vec3::new(0.17, -0.41, 0.28);
        let (na, nb, nc, nd) = (
            n_sph(sh[0].l),
            n_sph(sh[1].l),
            n_sph(sh[2].l),
            n_sph(sh[3].l),
        );
        for kernel in [CoulombKernel::Coulomb, CoulombKernel::Erf { omega: 0.33 }] {
            let a1 = giao_eri_quartet(&sh[0], &sh[1], &sh[2], &sh[3], o, kernel);
            let a2 = giao_eri_quartet(&sh[2], &sh[3], &sh[0], &sh[1], o, kernel);
            for comp in 0..3 {
                let mut worst = 0.0_f64;
                let mut scale = 0.0_f64;
                for i in 0..na {
                    for j in 0..nb {
                        for k in 0..nc {
                            for l in 0..nd {
                                let x = a1[comp][((i * nb + j) * nc + k) * nd + l];
                                let y = a2[comp][((k * nd + l) * na + i) * nb + j];
                                worst = worst.max((x - y).abs());
                                scale = scale.max(x.abs());
                            }
                        }
                    }
                }
                assert!(
                    worst < 1e-11 * scale.max(1.0),
                    "kernel {kernel:?}, component {comp}: pair exchange gap {worst:e} (scale {scale:e})"
                );
                assert!(
                    scale > 1e-8,
                    "kernel {kernel:?}, component {comp} is identically zero"
                );
            }
        }
    }

    /// The four London phases of `(mn|ls)` pick up the constant factors
    /// `c_m^* c_n c_l^* c_s` when the gauge origin moves, so the derivative
    /// must shift by `(gamma_n - gamma_m + gamma_s - gamma_l)` times the
    /// undifferentiated integral. Exact, and it pins the sign of each of
    /// the four phases independently.
    #[test]
    fn field_derivative_bracket_obeys_the_gauge_origin_shift_identity() {
        let sh = quad();
        let o = Vec3::new(0.17, -0.41, 0.28);
        let delta = Vec3::new(-0.33, 0.24, 0.51);
        let base = eri_quartet(&sh[0], &sh[1], &sh[2], &sh[3], CoulombKernel::Coulomb);
        let at_o = giao_eri_quartet(&sh[0], &sh[1], &sh[2], &sh[3], o, CoulombKernel::Coulomb);
        let shifted = giao_eri_quartet(
            &sh[0],
            &sh[1],
            &sh[2],
            &sh[3],
            o + delta,
            CoulombKernel::Coulomb,
        );
        let (na, nb, nc, nd) = (
            n_sph(sh[0].l),
            n_sph(sh[1].l),
            n_sph(sh[2].l),
            n_sph(sh[3].l),
        );
        for comp in 0..3 {
            let gamma = |s: &MolShell| 0.5 * (s.center - o).cross(delta).get(comp);
            let g = [gamma(&sh[0]), gamma(&sh[1]), gamma(&sh[2]), gamma(&sh[3])];
            let w = g[1] - g[0] + g[3] - g[2];
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for i in 0..na {
                for j in 0..nb {
                    for k in 0..nc {
                        for l in 0..nd {
                            let t = ((i * nb + j) * nc + k) * nd + l;
                            let want = at_o[comp][t] + w * base.get(i, j, k, l);
                            worst = worst.max((shifted[comp][t] - want).abs());
                            scale = scale.max(shifted[comp][t].abs());
                        }
                    }
                }
            }
            assert!(
                worst < 1e-11 * scale.max(1.0),
                "component {comp}: shift identity gap {worst:e} (scale {scale:e})"
            );
        }
    }

    /// The assembled Fock derivative must be ANTISYMMETRIC — the signature
    /// of the imaginary perturbation, and the one thing that goes wrong if
    /// the exchange term is contracted as `(ml|ns)` instead of `(ml|sn)`.
    /// Those are the same integral but carry different phase weights, and
    /// the wrong one produces a symmetric matrix.
    #[test]
    fn assembled_fock_derivative_is_antisymmetric() {
        let (_, mb) = water();
        let n = mb.nao_sph;
        // A physically-shaped symmetric density: the superposition-of-atoms
        // guess is enough; only its symmetry matters here.
        let mut p = Matrix::zeros(n, n);
        let mut state = 0x9E3779B9u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for i in 0..n {
            for j in i..n {
                let v = 0.3 * rng();
                p[(i, j)] = v;
                p[(j, i)] = v;
            }
        }
        for cx in [0.0, 1.0] {
            let g = giao_eri_fock_derivative(
                &mb,
                &p,
                cx,
                Vec3::new(0.21, -0.13, 0.44),
                CoulombKernel::Coulomb,
            );
            for (a, m) in g.iter().enumerate() {
                let mut worst = 0.0_f64;
                let mut scale = 0.0_f64;
                for i in 0..n {
                    for j in 0..n {
                        worst = worst.max((m[(i, j)] + m[(j, i)]).abs());
                        scale = scale.max(m[(i, j)].abs());
                    }
                }
                assert!(
                    worst < 1e-10 * scale.max(1.0),
                    "cx={cx} component {a}: antisymmetry gap {worst:e} (scale {scale:e})"
                );
                assert!(scale > 1e-8, "cx={cx} component {a} is identically zero");
            }
        }
    }

    /// The production contraction (a screened-free quartet sweep that
    /// scatters straight into J-like and K-like accumulators) must equal
    /// the naive route: materialize the whole bracket and contract it index
    /// by index. This gates the scatter bookkeeping, in particular the
    /// exchange index order.
    #[test]
    fn assembled_fock_derivative_matches_the_explicit_contraction() {
        // Deliberately off-axis: a diatomic along z has R_m - R_n parallel
        // to z, which kills the z component of every curl and would leave
        // one third of the comparison vacuous.
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 9,
                position: Vec3::new(0.91, 1.13, 0.82),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let n = mb.nao_sph;
        let o = Vec3::new(0.31, -0.22, 0.17);
        let cx = 0.63;
        let mut p = Matrix::zeros(n, n);
        let mut state = 0x1234_5678u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for i in 0..n {
            for j in i..n {
                let v = 0.25 * rng();
                p[(i, j)] = v;
                p[(j, i)] = v;
            }
        }
        // Reference: the full four-index bracket, contracted literally.
        let mut refm: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
        for sa in &mb.shells {
            for sb in &mb.shells {
                for sc in &mb.shells {
                    for sd in &mb.shells {
                        let blk = giao_eri_quartet(sa, sb, sc, sd, o, CoulombKernel::Coulomb);
                        let (na, nb, nc, nd) = (n_sph(sa.l), n_sph(sb.l), n_sph(sc.l), n_sph(sd.l));
                        for (comp, target) in refm.iter_mut().enumerate() {
                            for i in 0..na {
                                for j in 0..nb {
                                    for k in 0..nc {
                                        for l in 0..nd {
                                            let v = blk[comp][((i * nb + j) * nc + k) * nd + l];
                                            let (gi, gj, gk, gl) = (
                                                sa.sph_off + i,
                                                sb.sph_off + j,
                                                sc.sph_off + k,
                                                sd.sph_off + l,
                                            );
                                            target[(gi, gj)] += p[(gk, gl)] * v;
                                            target[(gi, gl)] -= 0.5 * cx * p[(gk, gj)] * v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let got = giao_eri_fock_derivative(&mb, &p, cx, o, CoulombKernel::Coulomb);
        for comp in 0..3 {
            let gap = got[comp].max_abs_diff(&refm[comp]);
            let scale = refm[comp]
                .as_slice()
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs()));
            assert!(
                gap < 1e-11 * scale.max(1.0),
                "component {comp}: gap {gap:e} (scale {scale:e})"
            );
            assert!(scale > 1e-6, "component {comp} reference is zero");
        }
    }

    /// RSH long-range exchange must be exchange-only: calling the Erf path
    /// must not sneak in a long-range Coulomb derivative. The same test also
    /// gates linearity of the fully contracted magnetic derivative under
    /// `erf + erfc = 1`.
    #[test]
    fn weighted_erf_exchange_matches_literal_contraction_and_kernel_complement() {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.79, 1.07, -0.61),
            },
        ]);
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let n = mb.nao_sph;
        let origin = Vec3::new(0.23, -0.19, 0.37);
        let omega = 0.33;
        let k_scale = 0.71;
        let mut p = Matrix::zeros(n, n);
        p[(0, 0)] = 0.83;
        p[(0, 1)] = -0.27;
        p[(1, 0)] = -0.27;
        p[(1, 1)] = 0.46;

        let mut reference: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
        for sa in &mb.shells {
            for sb in &mb.shells {
                for sc in &mb.shells {
                    for sd in &mb.shells {
                        let blk =
                            giao_eri_quartet(sa, sb, sc, sd, origin, CoulombKernel::Erf { omega });
                        let (na, nb, nc, nd) = (n_sph(sa.l), n_sph(sb.l), n_sph(sc.l), n_sph(sd.l));
                        for (comp, target) in reference.iter_mut().enumerate() {
                            for i in 0..na {
                                for j in 0..nb {
                                    for k in 0..nc {
                                        for l in 0..nd {
                                            let v = blk[comp][((i * nb + j) * nc + k) * nd + l];
                                            let (gi, gj, gk, gl) = (
                                                sa.sph_off + i,
                                                sb.sph_off + j,
                                                sc.sph_off + k,
                                                sd.sph_off + l,
                                            );
                                            target[(gi, gl)] -= 0.5 * k_scale * p[(gk, gj)] * v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let got = giao_eri_fock_derivative_weighted(
            &mb,
            &p,
            0.0,
            k_scale,
            origin,
            CoulombKernel::Erf { omega },
        );
        for comp in 0..3 {
            let gap = got[comp].max_abs_diff(&reference[comp]);
            let scale = reference[comp]
                .as_slice()
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs()));
            assert!(
                gap < 1e-12 * scale.max(1.0),
                "component {comp}: exchange-only gap {gap:e} (scale {scale:e})"
            );
            assert!(scale > 1e-8, "component {comp} reference is zero");
        }

        let full =
            giao_eri_fock_derivative_weighted(&mb, &p, 0.0, 1.0, origin, CoulombKernel::Coulomb);
        let lr = giao_eri_fock_derivative_weighted(
            &mb,
            &p,
            0.0,
            1.0,
            origin,
            CoulombKernel::Erf { omega },
        );
        let sr = giao_eri_fock_derivative_weighted(
            &mb,
            &p,
            0.0,
            1.0,
            origin,
            CoulombKernel::Erfc { omega },
        );
        for comp in 0..3 {
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for ((&f, &l), &s) in full[comp]
                .as_slice()
                .iter()
                .zip(lr[comp].as_slice())
                .zip(sr[comp].as_slice())
            {
                worst = worst.max((f - l - s).abs());
                scale = scale.max(f.abs());
            }
            assert!(
                worst < 1e-11 * scale.max(1.0),
                "component {comp}: Erf + Erfc complement gap {worst:e} (scale {scale:e})"
            );
        }
    }
}
