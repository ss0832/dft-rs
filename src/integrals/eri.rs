// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Two-electron repulsion integrals (ab|cd) via McMurchie–Davidson: both
//! bra and ket charge distributions are expanded in Hermite Gaussians and
//! contracted through the Hermite–Coulomb R-tensor at the composite exponent
//! `α = pq/(p+q)`. The interelectronic operator is pluggable
//! ([`CoulombKernel`]): 1/r₁₂, erf(ωr)/r₁₂, erfc(ωr)/r₁₂.
//!
//! Chemists' notation throughout: (ab|cd) = ∫∫ a(1)b(1) g(r₁₂) c(2)d(2).

use super::hermite::{ETable, RTensor};
use super::kernels::CoulombKernel;
use crate::basis::{aos, cart_components, n_cart, n_sph, MolBasis};
use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use std::f64::consts::PI;

/// One contracted shell quartet of spherical-basis ERIs,
/// `data[((ia·nb + ib)·nc + ic)·nd + id]`.
#[derive(Clone, Debug)]
pub struct Quartet {
    pub na: usize,
    pub nb: usize,
    pub nc: usize,
    pub nd: usize,
    pub data: Vec<f64>,
}

impl Quartet {
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.data[((i * self.nb + j) * self.nc + k) * self.nd + l]
    }
}

/// Raw-primitive Cartesian (ab|cd) — the slow reference path, kept for
/// closed-form tests and cross-checking the shell-quartet fast path.
#[allow(clippy::too_many_arguments)]
pub fn eri_prim_cart(
    a: f64,
    la: (i32, i32, i32),
    ca: Vec3,
    b: f64,
    lb: (i32, i32, i32),
    cb: Vec3,
    c: f64,
    lc: (i32, i32, i32),
    cc: Vec3,
    d: f64,
    ld: (i32, i32, i32),
    cd: Vec3,
    kernel: CoulombKernel,
) -> f64 {
    let p = a + b;
    let q = c + d;
    let pc = (ca * a + cb * b) / p;
    let qc = (cc * c + cd * d) / q;
    let alpha = p * q / (p + q);
    let ltot = (la.0 + la.1 + la.2 + lb.0 + lb.1 + lb.2 + lc.0 + lc.1 + lc.2 + ld.0 + ld.1 + ld.2)
        as usize;
    let pq = pc - qc;
    let mut f = vec![0.0; ltot + 1];
    kernel.boys_into(ltot, alpha, alpha * pq.norm2(), &mut f);
    let r = RTensor::build(ltot, alpha, pq, &f);
    let qab = ca - cb;
    let qcd = cc - cd;
    let ex_b = ETable::build(la.0 as usize, lb.0 as usize, qab.x, a, b);
    let ey_b = ETable::build(la.1 as usize, lb.1 as usize, qab.y, a, b);
    let ez_b = ETable::build(la.2 as usize, lb.2 as usize, qab.z, a, b);
    let ex_k = ETable::build(lc.0 as usize, ld.0 as usize, qcd.x, c, d);
    let ey_k = ETable::build(lc.1 as usize, ld.1 as usize, qcd.y, c, d);
    let ez_k = ETable::build(lc.2 as usize, ld.2 as usize, qcd.z, c, d);
    let mut sum = 0.0;
    for t in 0..=(la.0 + lb.0) as usize {
        let e1 = ex_b.get(la.0 as usize, lb.0 as usize, t);
        if e1 == 0.0 {
            continue;
        }
        for u in 0..=(la.1 + lb.1) as usize {
            let e2 = ey_b.get(la.1 as usize, lb.1 as usize, u);
            if e2 == 0.0 {
                continue;
            }
            for v in 0..=(la.2 + lb.2) as usize {
                let e3 = ez_b.get(la.2 as usize, lb.2 as usize, v);
                if e3 == 0.0 {
                    continue;
                }
                for tau in 0..=(lc.0 + ld.0) as usize {
                    let e4 = ex_k.get(lc.0 as usize, ld.0 as usize, tau);
                    if e4 == 0.0 {
                        continue;
                    }
                    for nu in 0..=(lc.1 + ld.1) as usize {
                        let e5 = ey_k.get(lc.1 as usize, ld.1 as usize, nu);
                        if e5 == 0.0 {
                            continue;
                        }
                        for phi in 0..=(lc.2 + ld.2) as usize {
                            let e6 = ez_k.get(lc.2 as usize, ld.2 as usize, phi);
                            if e6 == 0.0 {
                                continue;
                            }
                            let sign = if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
                            sum += e1
                                * e2
                                * e3
                                * e4
                                * e5
                                * e6
                                * sign
                                * r.get(t + tau, u + nu, v + phi);
                        }
                    }
                }
            }
        }
    }
    sum * 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt())
}

/// Contracted shell-quartet ERI in the spherical basis (index wrapper).
pub fn eri_shell_quartet(
    mb: &MolBasis,
    sa: usize,
    sb: usize,
    sc: usize,
    sd: usize,
    kernel: CoulombKernel,
) -> Quartet {
    eri_quartet(
        &mb.shells[sa],
        &mb.shells[sb],
        &mb.shells[sc],
        &mb.shells[sd],
        kernel,
    )
}

/// A "unit" s function (single zero-exponent primitive, coefficient 1):
/// pairing a real shell with it turns the McMurchie–Davidson quartet into a
/// genuine 3-center or 2-center Coulomb integral — the basis of the
/// density-fitting (RI) paths.
fn unit_shell(center: crate::math::Vec3) -> crate::basis::MolShell {
    crate::basis::MolShell {
        l: 0,
        atom: usize::MAX,
        center,
        exps: vec![0.0],
        coefs: vec![1.0],
        sph_off: 0,
        cart_off: 0,
    }
}

/// Three-center Coulomb block `(ab|P)` for an auxiliary shell given
/// DIRECTLY (not through a `MolBasis`): the continuum-solvation surface
/// charges are one s-type Gaussian per cavity point with a per-point
/// exponent, which no basis-set object can express.
pub fn eri_3c_shell(
    mb: &MolBasis,
    a: usize,
    b: usize,
    aux: &crate::basis::MolShell,
    kernel: CoulombKernel,
) -> Quartet {
    eri_quartet(
        &mb.shells[a],
        &mb.shells[b],
        aux,
        &unit_shell(aux.center),
        kernel,
    )
}

/// Three-centre block `(ab|P)` with ALL THREE shells given directly.
/// Needed by the continuum-solvation gradient, whose bra is a
/// raised/lowered variant that lives in no basis object.
pub fn eri_3c_shell_direct(
    sh_a: &crate::basis::MolShell,
    sh_b: &crate::basis::MolShell,
    aux: &crate::basis::MolShell,
    kernel: CoulombKernel,
) -> Quartet {
    eri_quartet(sh_a, sh_b, aux, &unit_shell(aux.center), kernel)
}

/// `(ab|g)` for a normalized s-Gaussian charge `g`, with BOTH orbital
/// shells left Cartesian and scaled by `weight`.
///
/// This is the Gaussian-charge counterpart of `deriv::nuclear_block`, and
/// it exists so the second-derivative machinery of `deriv2` can be driven
/// by continuum surface charges without duplicating any of it: the point
/// charge is just the infinite-exponent limit of this.
pub fn gauss_charge_block(
    sh_a: &crate::basis::MolShell,
    sh_b: &crate::basis::MolShell,
    aux: &crate::basis::MolShell,
    weight: f64,
    kernel: CoulombKernel,
) -> crate::linalg::Matrix {
    let q = eri_quartet_masked(
        sh_a,
        sh_b,
        aux,
        &unit_shell(aux.center),
        kernel,
        [false, false, true, true],
    );
    let mut m = crate::linalg::Matrix::zeros(q.na, q.nb);
    for i in 0..q.na {
        for j in 0..q.nb {
            m[(i, j)] = weight * q.get(i, j, 0, 0);
        }
    }
    m
}

/// As `eri_3c_shell_direct`, but leaving the BRA shell in its Cartesian
/// form. Raised/lowered shells carry `l ± 1` and are indexed by Cartesian
/// component during the derivative assembly, so they must skip the
/// spherical transform (same convention as `eri_quartet_masked`).
pub fn eri_3c_shell_direct_cart_bra(
    sh_a: &crate::basis::MolShell,
    sh_b: &crate::basis::MolShell,
    aux: &crate::basis::MolShell,
    kernel: CoulombKernel,
) -> Quartet {
    eri_quartet_masked(
        sh_a,
        sh_b,
        aux,
        &unit_shell(aux.center),
        kernel,
        [false, true, true, true],
    )
}

/// Two-center Coulomb block `(P|Q)` between two auxiliary shells.
pub fn eri_2c(aux: &MolBasis, p: usize, q: usize, kernel: CoulombKernel) -> Quartet {
    let (sp, sq) = (&aux.shells[p], &aux.shells[q]);
    eri_quartet(
        sp,
        &unit_shell(sp.center),
        sq,
        &unit_shell(sq.center),
        kernel,
    )
}

/// Three-center Coulomb block `(ab|P)`.
pub fn eri_3c(
    mb: &MolBasis,
    aux: &MolBasis,
    a: usize,
    b: usize,
    p: usize,
    kernel: CoulombKernel,
) -> Quartet {
    let sp = &aux.shells[p];
    eri_quartet(
        &mb.shells[a],
        &mb.shells[b],
        sp,
        &unit_shell(sp.center),
        kernel,
    )
}

/// Contracted shell-quartet ERI over explicit shell references.
pub fn eri_quartet(
    sh_a: &crate::basis::MolShell,
    sh_b: &crate::basis::MolShell,
    sh_c: &crate::basis::MolShell,
    sh_d: &crate::basis::MolShell,
    kernel: CoulombKernel,
) -> Quartet {
    eri_quartet_masked(sh_a, sh_b, sh_c, sh_d, kernel, [true; 4])
}

/// Like [`eri_quartet`] but each index is transformed to spherical only
/// where `sph_mask` is true — the derivative machinery keeps the
/// differentiated index Cartesian for the l±1 recombination.
pub fn eri_quartet_masked(
    sh_a: &crate::basis::MolShell,
    sh_b: &crate::basis::MolShell,
    sh_c: &crate::basis::MolShell,
    sh_d: &crate::basis::MolShell,
    kernel: CoulombKernel,
    sph_mask: [bool; 4],
) -> Quartet {
    let (la, lb, lc, ld) = (
        sh_a.l as usize,
        sh_b.l as usize,
        sh_c.l as usize,
        sh_d.l as usize,
    );
    let comps_a = cart_components(sh_a.l);
    let comps_b = cart_components(sh_b.l);
    let comps_c = cart_components(sh_c.l);
    let comps_d = cart_components(sh_d.l);
    let (nca, ncb, ncc, ncd) = (
        n_cart(sh_a.l),
        n_cart(sh_b.l),
        n_cart(sh_c.l),
        n_cart(sh_d.l),
    );
    let l_bra = la + lb;
    let l_ket = lc + ld;
    let ltot = l_bra + l_ket;
    let qab = sh_a.center - sh_b.center;
    let qcd = sh_c.center - sh_d.center;

    let mut cart = vec![0.0; nca * ncb * ncc * ncd];
    let gdim = l_bra + 1;
    let mut g = vec![0.0; gdim * gdim * gdim];
    let mut f = vec![0.0; ltot + 1];

    for (&ea, &wa) in sh_a.exps.iter().zip(&sh_a.coefs) {
        for (&eb, &wb) in sh_b.exps.iter().zip(&sh_b.coefs) {
            let p = ea + eb;
            let pcen = (sh_a.center * ea + sh_b.center * eb) / p;
            let ex_b = ETable::build(la, lb, qab.x, ea, eb);
            let ey_b = ETable::build(la, lb, qab.y, ea, eb);
            let ez_b = ETable::build(la, lb, qab.z, ea, eb);
            for (&ec, &wc) in sh_c.exps.iter().zip(&sh_c.coefs) {
                for (&ed, &wd) in sh_d.exps.iter().zip(&sh_d.coefs) {
                    let q = ec + ed;
                    let qcen = (sh_c.center * ec + sh_d.center * ed) / q;
                    let alpha = p * q / (p + q);
                    let pq = pcen - qcen;
                    kernel.boys_into(ltot, alpha, alpha * pq.norm2(), &mut f);
                    let r = RTensor::build(ltot, alpha, pq, &f);
                    let ex_k = ETable::build(lc, ld, qcd.x, ec, ed);
                    let ey_k = ETable::build(lc, ld, qcd.y, ec, ed);
                    let ez_k = ETable::build(lc, ld, qcd.z, ec, ed);
                    let pref = wa * wb * wc * wd * 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());
                    for (kc, &(cx, cy, cz)) in comps_c.iter().enumerate() {
                        for (kd, &(dx, dy, dz)) in comps_d.iter().enumerate() {
                            // Middle contraction: fold the ket Hermite
                            // expansion into G[t,u,v].
                            for slot in g.iter_mut() {
                                *slot = 0.0;
                            }
                            for tau in 0..=(cx + dx) as usize {
                                let e4 = ex_k.get(cx as usize, dx as usize, tau);
                                if e4 == 0.0 {
                                    continue;
                                }
                                for nu in 0..=(cy + dy) as usize {
                                    let e5 = ey_k.get(cy as usize, dy as usize, nu);
                                    if e5 == 0.0 {
                                        continue;
                                    }
                                    for phi in 0..=(cz + dz) as usize {
                                        let e6 = ez_k.get(cz as usize, dz as usize, phi);
                                        if e6 == 0.0 {
                                            continue;
                                        }
                                        let sign =
                                            if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
                                        let ek = sign * e4 * e5 * e6;
                                        for t in 0..=l_bra {
                                            for u in 0..=(l_bra - t) {
                                                for v in 0..=(l_bra - t - u) {
                                                    g[(t * gdim + u) * gdim + v] +=
                                                        ek * r.get(t + tau, u + nu, v + phi);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Bra dot product per (ia, ib).
                            for (ia, &(ax, ay, az)) in comps_a.iter().enumerate() {
                                for (ib, &(bx, by, bz)) in comps_b.iter().enumerate() {
                                    let mut sum = 0.0;
                                    for t in 0..=(ax + bx) as usize {
                                        let e1 = ex_b.get(ax as usize, bx as usize, t);
                                        if e1 == 0.0 {
                                            continue;
                                        }
                                        for u in 0..=(ay + by) as usize {
                                            let e2 = ey_b.get(ay as usize, by as usize, u);
                                            if e2 == 0.0 {
                                                continue;
                                            }
                                            for v in 0..=(az + bz) as usize {
                                                let e3 = ez_b.get(az as usize, bz as usize, v);
                                                if e3 == 0.0 {
                                                    continue;
                                                }
                                                sum += e1 * e2 * e3 * g[(t * gdim + u) * gdim + v];
                                            }
                                        }
                                    }
                                    cart[((ia * ncb + ib) * ncc + kc) * ncd + kd] += pref * sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Cartesian → spherical (masked), two indices at a time.
    let dim = |l: u8, sph: bool| if sph { n_sph(l) } else { n_cart(l) };
    let make_w = |la: u8, lb: u8, sa: bool, sb: bool| -> Matrix {
        if sa && sb {
            return aos::cart_to_sph_pair(la, lb).clone();
        }
        let ta = transform_or_identity(la, sa);
        let tb = transform_or_identity(lb, sb);
        let (ra, ca_) = (ta.rows, ta.cols);
        let (rb, cb_) = (tb.rows, tb.cols);
        let mut w = Matrix::zeros(ra * rb, ca_ * cb_);
        for ia in 0..ra {
            for ib in 0..rb {
                for ja in 0..ca_ {
                    for jb in 0..cb_ {
                        w[(ia * rb + ib, ja * cb_ + jb)] = ta[(ia, ja)] * tb[(ib, jb)];
                    }
                }
            }
        }
        w
    };
    let w_ab = make_w(sh_a.l, sh_b.l, sph_mask[0], sph_mask[1]);
    let w_cd = make_w(sh_c.l, sh_d.l, sph_mask[2], sph_mask[3]);
    let cart_m = Matrix::from_row_major(nca * ncb, ncc * ncd, cart);
    let half = cart_m.matmul_seq(&w_cd);
    let full = w_ab.transpose_matmul_seq(&half);
    Quartet {
        na: dim(sh_a.l, sph_mask[0]),
        nb: dim(sh_b.l, sph_mask[1]),
        nc: dim(sh_c.l, sph_mask[2]),
        nd: dim(sh_d.l, sph_mask[3]),
        data: full.as_slice().to_vec(),
    }
}

/// Pack an accumulated Cartesian quartet into a (masked) spherical one.
fn quartet_from_cart(
    la: u8,
    lb: u8,
    lc: u8,
    ld: u8,
    cart: Vec<f64>,
    sph_mask: [bool; 4],
) -> Quartet {
    let dim = |l: u8, sph: bool| if sph { n_sph(l) } else { n_cart(l) };
    let make_w = |la: u8, lb: u8, sa: bool, sb: bool| -> Matrix {
        if sa && sb {
            return aos::cart_to_sph_pair(la, lb).clone();
        }
        let ta = transform_or_identity(la, sa);
        let tb = transform_or_identity(lb, sb);
        let (ra, ca_) = (ta.rows, ta.cols);
        let (rb, cb_) = (tb.rows, tb.cols);
        let mut w = Matrix::zeros(ra * rb, ca_ * cb_);
        for ia in 0..ra {
            for ib in 0..rb {
                for ja in 0..ca_ {
                    for jb in 0..cb_ {
                        w[(ia * rb + ib, ja * cb_ + jb)] = ta[(ia, ja)] * tb[(ib, jb)];
                    }
                }
            }
        }
        w
    };
    let w_ab = make_w(la, lb, sph_mask[0], sph_mask[1]);
    let w_cd = make_w(lc, ld, sph_mask[2], sph_mask[3]);
    let cart_m = Matrix::from_row_major(n_cart(la) * n_cart(lb), n_cart(lc) * n_cart(ld), cart);
    let half = cart_m.matmul_seq(&w_cd);
    let full = w_ab.transpose_matmul_seq(&half);
    Quartet {
        na: dim(la, sph_mask[0]),
        nb: dim(lb, sph_mask[1]),
        nc: dim(lc, sph_mask[2]),
        nd: dim(ld, sph_mask[3]),
        data: full.as_slice().to_vec(),
    }
}

/// The six shell-substituted quartets the ERI gradient needs, in one pass:
/// `[a+, a-, b+, b-, c+, c-]`, each with the substituted index left
/// Cartesian. `None` where lowering an s shell is undefined.
///
/// Computing them together rather than through six `eri_quartet_masked`
/// calls is worth doing because raising or lowering a shell changes its
/// ANGULAR MOMENTUM but not its exponents or centre. So `p`, `q`, `alpha`
/// and `P-Q` are identical across all six, and therefore so are the Boys
/// values and the Hermite-Coulomb tensor `R`: one build at `L+1` serves
/// every variant, since `R.get` is valid for any `t+u+v <= l`.
///
/// The E-coefficient tables collapse the same way. `ETable::build(i,j,..)`
/// covers every order up to `(i,j)`, so ONE bra table at `(la+1, lb+1)`
/// serves the four bra substitutions and one ket table at `(lc+1, ld)`
/// serves the two ket ones — six tables per primitive quadruple instead of
/// thirty-six.
pub(crate) fn eri_quartet_deriv_set(
    sh: [&crate::basis::MolShell; 4],
    kernel: CoulombKernel,
) -> [Option<Quartet>; 6] {
    let l: [usize; 4] = [
        sh[0].l as usize,
        sh[1].l as usize,
        sh[2].l as usize,
        sh[3].l as usize,
    ];
    // Variant v = 2*pos + (0 raise, 1 lower); the substituted shell's l.
    let variant_l = |v: usize| -> Option<[u8; 4]> {
        let (pos, up) = (v / 2, v % 2 == 0);
        let mut out = [sh[0].l, sh[1].l, sh[2].l, sh[3].l];
        if up {
            out[pos] += 1;
        } else if out[pos] == 0 {
            return None;
        } else {
            out[pos] -= 1;
        }
        Some(out)
    };
    let mut carts: Vec<Option<Vec<f64>>> = (0..6)
        .map(|v| {
            variant_l(v)
                .map(|ls| vec![0.0; n_cart(ls[0]) * n_cart(ls[1]) * n_cart(ls[2]) * n_cart(ls[3])])
        })
        .collect();

    let qab = sh[0].center - sh[1].center;
    let qcd = sh[2].center - sh[3].center;
    // One extra order everywhere: the raised variants need it and the
    // lowered ones are a sub-block of the same tensor.
    let ltot_max = l[0] + l[1] + l[2] + l[3] + 1;
    let mut f = vec![0.0; ltot_max + 1];

    for (ia, (&ea, &wa)) in sh[0].exps.iter().zip(&sh[0].coefs).enumerate() {
        let _ = ia;
        for (&eb, &wb) in sh[1].exps.iter().zip(&sh[1].coefs) {
            let p = ea + eb;
            let pcen = (sh[0].center * ea + sh[1].center * eb) / p;
            // Bra tables one order up in BOTH indices: this single build
            // covers (la±1, lb) and (la, lb±1).
            let ex_b = ETable::build(l[0] + 1, l[1] + 1, qab.x, ea, eb);
            let ey_b = ETable::build(l[0] + 1, l[1] + 1, qab.y, ea, eb);
            let ez_b = ETable::build(l[0] + 1, l[1] + 1, qab.z, ea, eb);
            for (&ec, &wc) in sh[2].exps.iter().zip(&sh[2].coefs) {
                for (&ed, &wd) in sh[3].exps.iter().zip(&sh[3].coefs) {
                    let q = ec + ed;
                    let qcen = (sh[2].center * ec + sh[3].center * ed) / q;
                    let alpha = p * q / (p + q);
                    let pq = pcen - qcen;
                    kernel.boys_into(ltot_max, alpha, alpha * pq.norm2(), &mut f);
                    let r = RTensor::build(ltot_max, alpha, pq, &f);
                    let ex_k = ETable::build(l[2] + 1, l[3], qcd.x, ec, ed);
                    let ey_k = ETable::build(l[2] + 1, l[3], qcd.y, ec, ed);
                    let ez_k = ETable::build(l[2] + 1, l[3], qcd.z, ec, ed);
                    let base_pref = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());
                    for v in 0..6 {
                        let Some(ls) = variant_l(v) else { continue };
                        let cart = carts[v].as_mut().expect("variant present");
                        // Raising multiplies that shell's contraction
                        // coefficient by 2*exponent (see `raised`).
                        let w = match v {
                            0 => 2.0 * ea * wa * wb * wc * wd,
                            1 => wa * wb * wc * wd,
                            2 => wa * (2.0 * eb * wb) * wc * wd,
                            3 => wa * wb * wc * wd,
                            4 => wa * wb * (2.0 * ec * wc) * wd,
                            _ => wa * wb * wc * wd,
                        };
                        let pref = w * base_pref;
                        accumulate_quartet(
                            cart, ls, &ex_b, &ey_b, &ez_b, &ex_k, &ey_k, &ez_k, &r, pref,
                        );
                    }
                }
            }
        }
    }

    let mut out: [Option<Quartet>; 6] = Default::default();
    for v in 0..6 {
        let Some(ls) = variant_l(v) else { continue };
        let Some(cart) = carts[v].take() else {
            continue;
        };
        let mut mask = [true; 4];
        mask[v / 2] = false; // the differentiated index stays Cartesian
        out[v] = Some(quartet_from_cart(ls[0], ls[1], ls[2], ls[3], cart, mask));
    }
    out
}

/// One primitive quadruple's contribution to a Cartesian quartet, using
/// shared E-tables and a shared `R`. Split out so the six derivative
/// variants can be driven from one set of them.
#[allow(clippy::too_many_arguments)]
fn accumulate_quartet(
    cart: &mut [f64],
    ls: [u8; 4],
    ex_b: &ETable,
    ey_b: &ETable,
    ez_b: &ETable,
    ex_k: &ETable,
    ey_k: &ETable,
    ez_k: &ETable,
    r: &RTensor,
    pref: f64,
) {
    let (la, lb, lc, ld) = (ls[0], ls[1], ls[2], ls[3]);
    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let comps_c = cart_components(lc);
    let comps_d = cart_components(ld);
    let (ncb, ncc, ncd) = (n_cart(lb), n_cart(lc), n_cart(ld));
    let l_bra = la as usize + lb as usize;
    let gdim = l_bra + 1;
    let mut g = vec![0.0; gdim * gdim * gdim];
    for (kc, &(cx, cy, cz)) in comps_c.iter().enumerate() {
        for (kd, &(dx, dy, dz)) in comps_d.iter().enumerate() {
            for slot in g.iter_mut() {
                *slot = 0.0;
            }
            for tau in 0..=(cx + dx) as usize {
                let e4 = ex_k.get(cx as usize, dx as usize, tau);
                if e4 == 0.0 {
                    continue;
                }
                for nu in 0..=(cy + dy) as usize {
                    let e5 = ey_k.get(cy as usize, dy as usize, nu);
                    if e5 == 0.0 {
                        continue;
                    }
                    for phi in 0..=(cz + dz) as usize {
                        let e6 = ez_k.get(cz as usize, dz as usize, phi);
                        if e6 == 0.0 {
                            continue;
                        }
                        let sign = if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
                        let ek = sign * e4 * e5 * e6;
                        for t in 0..=l_bra {
                            for u in 0..=(l_bra - t) {
                                for v in 0..=(l_bra - t - u) {
                                    g[(t * gdim + u) * gdim + v] +=
                                        ek * r.get(t + tau, u + nu, v + phi);
                                }
                            }
                        }
                    }
                }
            }
            for (ia, &(ax, ay, az)) in comps_a.iter().enumerate() {
                for (ib, &(bx, by, bz)) in comps_b.iter().enumerate() {
                    let mut sum = 0.0;
                    for t in 0..=(ax + bx) as usize {
                        let e1 = ex_b.get(ax as usize, bx as usize, t);
                        if e1 == 0.0 {
                            continue;
                        }
                        for u in 0..=(ay + by) as usize {
                            let e2 = ey_b.get(ay as usize, by as usize, u);
                            if e2 == 0.0 {
                                continue;
                            }
                            for v in 0..=(az + bz) as usize {
                                let e3 = ez_b.get(az as usize, bz as usize, v);
                                if e3 == 0.0 {
                                    continue;
                                }
                                sum += e1 * e2 * e3 * g[(t * gdim + u) * gdim + v];
                            }
                        }
                    }
                    cart[((ia * ncb + ib) * ncc + kc) * ncd + kd] += pref * sum;
                }
            }
        }
    }
}

/// The cart→sph transform for `l`, or the Cartesian identity.
fn transform_or_identity(l: u8, sph: bool) -> Matrix {
    if sph {
        aos::cart_to_sph(l).clone()
    } else {
        Matrix::identity(n_cart(l))
    }
}

/// Schwarz screening matrix over shells: `Q[a,b] = √(max_{ij} |(ij|ij)|)`
/// with i ∈ shell a, j ∈ shell b. `|(ab|cd)| ≤ Q[a,b]·Q[c,d]`.
pub fn schwarz_matrix(mb: &MolBasis, kernel: CoulombKernel) -> Matrix {
    use rayon::prelude::*;
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (a..ns).map(move |b| (a, b))).collect();
    let vals: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let quartet = eri_shell_quartet(mb, a, b, a, b, kernel);
            let mut maxv: f64 = 0.0;
            for i in 0..quartet.na {
                for j in 0..quartet.nb {
                    maxv = maxv.max(quartet.get(i, j, i, j).abs());
                }
            }
            (a, b, maxv.sqrt())
        })
        .collect();
    let mut q = Matrix::zeros(ns, ns);
    for (a, b, v) in vals {
        q[(a, b)] = v;
        q[(b, a)] = v;
    }
    q
}

/// Dense in-core spherical ERI store (small systems and tests).
pub struct EriTensor {
    pub nao: usize,
    data: Vec<f64>,
    _allocation: crate::resources::MemoryReservation,
}

impl EriTensor {
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let n = self.nao;
        self.data[((i * n + j) * n + k) * n + l]
    }

    /// The tensor in row-major order, `(ij|kl)` at `((i n + j) n + k) n + l`.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

/// Build the full (ij|kl) tensor with 8-fold-symmetric shell-quartet fills.
/// Refuses to allocate beyond `limit_mb` MiB.
pub fn eri_incore(mb: &MolBasis, kernel: CoulombKernel, limit_mb: usize) -> Result<EriTensor> {
    use rayon::prelude::*;
    let n = mb.nao_sph;
    let bytes = crate::resources::checked_bytes(
        &[n, n, n, n, std::mem::size_of::<f64>()],
        "in-core ERI tensor",
    )?;
    let required_mb =
        bytes / crate::resources::MIB + usize::from(bytes % crate::resources::MIB != 0);
    if required_mb > limit_mb {
        return Err(DftError::ResourceLimit {
            operation: "in-core ERI tensor",
            required_mb,
            limit_mb,
        });
    }
    let ns = mb.shells.len();
    let pairs = ns
        .checked_add(1)
        .and_then(|m| ns.checked_mul(m))
        .map(|n| n / 2)
        .ok_or_else(|| DftError::InvalidInput("ERI shell-pair count overflows".into()))?;
    let count = pairs
        .checked_add(1)
        .and_then(|m| pairs.checked_mul(m))
        .map(|n| n / 2)
        .ok_or_else(|| DftError::InvalidInput("ERI shell-quartet count overflows".into()))?;
    let metadata = crate::resources::checked_bytes(
        &[
            count,
            std::mem::size_of::<(usize, usize, usize, usize)>() * 2
                + std::mem::size_of::<Quartet>(),
        ],
        "ERI quartet metadata",
    )?;
    let workspace = bytes
        .checked_add(metadata)
        .ok_or_else(|| DftError::InvalidInput("ERI assembly workspace overflows".into()))?;
    let allocation = crate::resources::reserve(bytes, "in-core ERI tensor")?;
    let _workspace = crate::resources::reserve(workspace, "in-core ERI assembly workspace")?;
    // Unique shell quartets: a ≤ b, c ≤ d, (a,b) ≤ (c,d) lexicographically.
    let mut quartets = Vec::with_capacity(count);
    for a in 0..ns {
        for b in a..ns {
            for c in a..ns {
                let d0 = if c == a { b } else { c };
                for d in d0..ns {
                    quartets.push((a, b, c, d));
                }
            }
        }
    }
    let computed: Vec<((usize, usize, usize, usize), Quartet)> = quartets
        .par_iter()
        .map(|&(a, b, c, d)| ((a, b, c, d), eri_shell_quartet(mb, a, b, c, d, kernel)))
        .collect();
    let mut data = vec![0.0; n * n * n * n];
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * n + j) * n + k) * n + l;
    for ((a, b, c, d), quartet) in computed {
        let (oa, ob, oc, od) = (
            mb.shells[a].sph_off,
            mb.shells[b].sph_off,
            mb.shells[c].sph_off,
            mb.shells[d].sph_off,
        );
        for i in 0..quartet.na {
            for j in 0..quartet.nb {
                for k in 0..quartet.nc {
                    for l in 0..quartet.nd {
                        let v = quartet.get(i, j, k, l);
                        let (gi, gj, gk, gl) = (oa + i, ob + j, oc + k, od + l);
                        // 8-fold permutational symmetry.
                        data[idx(gi, gj, gk, gl)] = v;
                        data[idx(gj, gi, gk, gl)] = v;
                        data[idx(gi, gj, gl, gk)] = v;
                        data[idx(gj, gi, gl, gk)] = v;
                        data[idx(gk, gl, gi, gj)] = v;
                        data[idx(gl, gk, gi, gj)] = v;
                        data[idx(gk, gl, gj, gi)] = v;
                        data[idx(gl, gk, gj, gi)] = v;
                    }
                }
            }
        }
    }
    Ok(EriTensor {
        nao: n,
        data,
        _allocation: allocation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::MolShell;
    use crate::math::Vec3;

    fn mk(a: f64, l: u8, c: Vec3) -> MolShell {
        MolShell {
            l,
            atom: 0,
            center: c,
            exps: vec![a],
            coefs: vec![1.0],
            sph_off: 0,
            cart_off: 0,
        }
    }

    /// (P|Q) via the unit-shell trick must equal the two-center s-Gaussian
    /// closed form 2π^{5/2}/(αβ√(α+β)) · F₀(ρR²).
    #[test]
    fn two_center_matches_closed_form() {
        let (alpha, beta) = (1.3_f64, 0.7_f64);
        let r = Vec3::new(0.0, 0.0, 2.0);
        let sp = mk(alpha, 0, Vec3::zero());
        let sq = mk(beta, 0, r);
        let q = eri_quartet(
            &sp,
            &unit_shell(sp.center),
            &sq,
            &unit_shell(sq.center),
            CoulombKernel::Coulomb,
        );
        let rho = alpha * beta / (alpha + beta);
        let f = crate::integrals::boys::boys(0, rho * r.norm2());
        let want =
            2.0 * std::f64::consts::PI.powf(2.5) / (alpha * beta * (alpha + beta).sqrt()) * f[0];
        assert!(
            (q.get(0, 0, 0, 0) - want).abs() < 1e-12 * want.abs(),
            "{} vs {want}",
            q.get(0, 0, 0, 0)
        );
    }

    /// (ab|P) with a d-type auxiliary: pinned against the primitive
    /// Cartesian reference path with an explicit zero-exponent ket.
    #[test]
    fn three_center_matches_primitive_reference() {
        let sa = mk(0.9, 1, Vec3::new(0.0, 0.1, -0.2));
        let sb = mk(1.4, 0, Vec3::new(0.3, 0.0, 0.5));
        let sp = mk(2.1, 2, Vec3::new(-0.4, 0.2, 0.1));
        let direct = eri_quartet(
            &sa,
            &sb,
            &sp,
            &unit_shell(sp.center),
            CoulombKernel::Coulomb,
        );
        let cart_ref = eri_prim_cart(
            0.9,
            (0, 1, 0),
            sa.center,
            1.4,
            (0, 0, 0),
            sb.center,
            2.1,
            (1, 1, 0),
            sp.center,
            0.0,
            (0, 0, 0),
            sp.center,
            CoulombKernel::Coulomb,
        );
        // p_y is sph index 2 (m = −1); d_xy is sph column 4 (m = −2) with
        // solid-harmonic coefficient √3 on the xy monomial.
        let got = direct.get(2, 0, 4, 0);
        let want = 3.0_f64.sqrt() * cart_ref;
        assert!(
            (got - want).abs() < 1e-12 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }
}
