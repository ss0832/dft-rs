// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Second nuclear-coordinate derivatives of integral matrices, contracted
//! with a fixed matrix into skeleton-Hessian blocks
//! `H[(A,x),(B,y)] = sum_mn m_mn d2 O_mn / dA_x dB_y`.
//!
//! For two-center operators (S, T) only the MIXED bra-ket second
//! derivative is ever computed explicitly (independent indices - the
//! single-derivative combine applied to each side); the same-center
//! blocks follow from translational invariance, d2/dAdA = -d2/dAdB.
//!
//! Nuclear attraction (three centers A, B, C) additionally needs the
//! nested same-index second derivative; with AA, AB, BB computed the
//! operator-center rows follow from the sum rules
//! AC = -AA - AB, BC = -AB - BB, CC = AA + 2AB + BB.

use super::deriv::{
    combine_bra_derivative, kinetic_kind, lowered, nuclear_block, overlap_kind, pair_block, raised,
};
use super::deriv_mat::dipole_pair_block;
use super::eri::eri_quartet_masked;
use super::kernels::CoulombKernel;
use crate::basis::{aos, cart_components, n_cart, n_sph, MolBasis, MolShell};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use rayon::prelude::*;

/// Ket-side single-derivative combine: mirror of `combine_bra_derivative`
/// acting on columns.
pub(crate) fn combine_ket_derivative(
    la: u8,
    lb: u8,
    ax: usize,
    raise_blk: &Matrix, // n_cart(la) x n_cart(lb+1)
    lower_blk: Option<&Matrix>,
) -> Matrix {
    let comps_b = cart_components(lb);
    let nca = n_cart(la);
    let mut out = Matrix::zeros(nca, comps_b.len());
    for (j, &(cx, cy, cz)) in comps_b.iter().enumerate() {
        let up = match ax {
            0 => (cx + 1, cy, cz),
            1 => (cx, cy + 1, cz),
            _ => (cx, cy, cz + 1),
        };
        let dn = match ax {
            0 => (cx - 1, cy, cz),
            1 => (cx, cy - 1, cz),
            _ => (cx, cy, cz - 1),
        };
        let lcomp = [cx, cy, cz][ax];
        let ju = comp_index(lb + 1, up);
        for i in 0..nca {
            let mut v = raise_blk[(i, ju)];
            if lcomp > 0 {
                let blk = lower_blk.expect("lowered ket block");
                let jd = comp_index(lb - 1, dn);
                v -= lcomp as f64 * blk[(i, jd)];
            }
            out[(i, j)] = v;
        }
    }
    out
}

/// Nested same-index second derivative on the bra:
///
/// ```text
/// d2 phi_u / dx dy = rr[u+x+y] - (u+y)_x rl[u+y-x] - u_y rl[u-y+x]
///                    + u_y (u-y)_x ll[u-y-x]
/// ```
///
/// Here rr spans l+2 (coefficients 4 a^2 c), rl spans l (coefficients 2 a c),
/// and ll spans l-2 (coefficients c).
pub(crate) fn combine_bra_second(
    la: u8,
    lb: u8,
    ax: usize,
    ay: usize,
    rr: &Matrix,
    rl: &Matrix,
    ll: Option<&Matrix>,
) -> Matrix {
    let comps_a = cart_components(la);
    let ncb = n_cart(lb);
    let mut out = Matrix::zeros(comps_a.len(), ncb);
    let step = |c: (i32, i32, i32), ax: usize, d: i32| -> (i32, i32, i32) {
        match ax {
            0 => (c.0 + d, c.1, c.2),
            1 => (c.0, c.1 + d, c.2),
            _ => (c.0, c.1, c.2 + d),
        }
    };
    let comp = |c: (i32, i32, i32)| [c.0, c.1, c.2];
    for (i, &u) in comps_a.iter().enumerate() {
        let uy = comp(u)[ay];
        // term 1: rr[u + y + x]
        let up2 = step(step(u, ay, 1), ax, 1);
        let i_rr = comp_index(la + 2, up2);
        // term 2: -(u+y)_x rl[u + y - x]
        let upy = step(u, ay, 1);
        let upy_x = comp(upy)[ax];
        let t2 = step(upy, ax, -1);
        // term 3: -u_y rl[u - y + x]
        let t3 = if uy > 0 {
            Some(step(step(u, ay, -1), ax, 1))
        } else {
            None
        };
        // term 4: +u_y (u-y)_x ll[u - y - x]
        let t4 = if uy > 0 {
            let umy = step(u, ay, -1);
            let umy_x = comp(umy)[ax];
            if umy_x > 0 {
                Some((step(umy, ax, -1), (uy * umy_x) as f64))
            } else {
                None
            }
        } else {
            None
        };
        for j in 0..ncb {
            let mut v = rr[(i_rr, j)];
            if t2.0 >= 0 && t2.1 >= 0 && t2.2 >= 0 {
                v -= upy_x as f64 * rl[(comp_index(la, t2), j)];
            }
            if let Some(c3) = t3 {
                if c3.0 >= 0 && c3.1 >= 0 && c3.2 >= 0 {
                    v -= uy as f64 * rl[(comp_index(la, c3), j)];
                }
            }
            if let Some((c4, f4)) = t4 {
                v += f4 * ll.expect("ll block for l>=2 components")[(comp_index(la - 2, c4), j)];
            }
            out[(i, j)] = v;
        }
    }
    out
}

fn comp_index(l: u8, c: (i32, i32, i32)) -> usize {
    cart_components(l)
        .iter()
        .position(|&x| x == c)
        .expect("cartesian component exists")
}

fn contract(m: &Matrix, oa: usize, ob: usize, sph: &Matrix) -> f64 {
    let mut acc = 0.0;
    for i in 0..sph.rows {
        for j in 0..sph.cols {
            acc += m[(oa + i, ob + j)] * sph[(i, j)];
        }
    }
    acc
}

/// Add into `hess` (3N x 3N, row-major (atom*3+ax)) the pair-operator
/// skeleton term `sum m d2 O` for S- or T-like kinds.
fn pair_operator_second<F>(mb: &MolBasis, m: &Matrix, natoms: usize, kind: F) -> Matrix
where
    F: Fn(
            &crate::integrals::hermite::ETable,
            &crate::integrals::hermite::ETable,
            &crate::integrals::hermite::ETable,
            f64,
            f64,
            (i32, i32, i32),
            (i32, i32, i32),
        ) -> f64
        + Sync
        + Copy,
{
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Matrix> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut h = Matrix::zeros(3 * natoms, 3 * natoms);
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            if sh_a.atom == sh_b.atom {
                return h; // all second derivatives cancel for one center
            }
            // Mixed d2/dA_x dB_y from the four bra/ket variant blocks.
            let ra = raised(sh_a);
            let la_lo = lowered(sh_a);
            let rb = raised(sh_b);
            let lb_lo = lowered(sh_b);
            let rr = pair_block(&ra, &rb, kind);
            let rl = lb_lo.as_ref().map(|lo| pair_block(&ra, lo, kind));
            let lr = la_lo.as_ref().map(|lo| pair_block(lo, &rb, kind));
            let ll = la_lo
                .as_ref()
                .and_then(|loa| lb_lo.as_ref().map(|lob| pair_block(loa, lob, kind)));
            let (ia, ib) = (sh_a.atom, sh_b.atom);
            for ax in 0..3 {
                // Bra derivative first (on the raised/lowered *ket* variants).
                let bra_r = combine_bra_derivative(sh_a.l, sh_b.l + 1, ax, &rr, lr.as_ref());
                let bra_l = rl
                    .as_ref()
                    .map(|rlb| combine_bra_derivative(sh_a.l, sh_b.l - 1, ax, rlb, ll.as_ref()));
                for by in 0..3 {
                    let cart = combine_ket_derivative(sh_a.l, sh_b.l, by, &bra_r, bra_l.as_ref());
                    let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                    let v = contract(m, sh_a.sph_off, sh_b.sph_off, &sph);
                    // (A,x; B,y) mixed term ...
                    h[(3 * ia + ax, 3 * ib + by)] += v;
                    h[(3 * ib + by, 3 * ia + ax)] += v;
                    // ... and the same-center blocks via d2/dAdA = -d2/dAdB.
                    h[(3 * ia + ax, 3 * ia + by)] -= v;
                    h[(3 * ib + by, 3 * ib + ax)] -= v;
                }
            }
            h
        })
        .collect();
    let mut total = Matrix::zeros(3 * natoms, 3 * natoms);
    for p in partials {
        for (t, v) in total.as_mut_slice().iter_mut().zip(p.as_slice()) {
            *t += *v;
        }
    }
    total
}

/// `sum_mn m_mn d2 S_mn` skeleton block (3N x 3N).
pub fn overlap_second_contraction(mb: &MolBasis, m: &Matrix, natoms: usize) -> Matrix {
    pair_operator_second(mb, m, natoms, overlap_kind)
}

/// `sum_mn m_mn d2 T_mn` skeleton block.
pub fn kinetic_second_contraction(mb: &MolBasis, m: &Matrix, natoms: usize) -> Matrix {
    pair_operator_second(mb, m, natoms, kinetic_kind)
}

/// `sum_mn m_mn d2 <m|r_component|n>` for a dipole operator anchored at a
/// fixed laboratory origin.
///
/// Unlike overlap and kinetic energy, translating both AO centres changes a
/// dipole integral.  The same-centre blocks therefore cannot be recovered
/// from a translational sum rule.  We evaluate the nested bra derivative and
/// the mixed bra-ket derivative for every ordered shell pair; the swapped
/// pair supplies the ket-ket and ket-bra partners.  This also handles two
/// distinct shells on the same atom without a special cancellation case.
pub fn dipole_second_contraction(
    mb: &MolBasis,
    m: &Matrix,
    natoms: usize,
    component: usize,
) -> Matrix {
    assert!(component < 3, "dipole component must be Cartesian");
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Matrix> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut h = Matrix::zeros(3 * natoms, 3 * natoms);
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            let (ia, ib) = (sh_a.atom, sh_b.atom);
            // `dipole_derivative_matrices` scatters every ordered bra block and
            // its transpose. Contracting with a symmetric density therefore
            // contributes both matrix blocks for every shell pair. The swapped
            // ordered pair supplies the ket-centre derivative; it does not
            // replace this transpose contribution.
            let pair_weight = 2.0;

            // Nested derivative on the bra, d2/dA_x dA_y.
            let ra = raised(sh_a);
            let rra = raised(&ra);
            let rl = lowered(&ra).expect("raised shell is always lowerable");
            let la = lowered(sh_a);
            let lla = la.as_ref().and_then(lowered);
            let aa_rr = dipole_pair_block(&rra, sh_b, component);
            let aa_rl = dipole_pair_block(&rl, sh_b, component);
            let aa_ll = lla
                .as_ref()
                .map(|shell| dipole_pair_block(shell, sh_b, component));
            for ax in 0..3 {
                for ay in 0..3 {
                    let cart =
                        combine_bra_second(sh_a.l, sh_b.l, ax, ay, &aa_rr, &aa_rl, aa_ll.as_ref());
                    let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                    h[(3 * ia + ax, 3 * ia + ay)] +=
                        pair_weight * contract(m, sh_a.sph_off, sh_b.sph_off, &sph);
                }
            }

            // Mixed derivative d2/dA_x dB_y.
            let rb = raised(sh_b);
            let lb = lowered(sh_b);
            let ab_rr = dipole_pair_block(&ra, &rb, component);
            let ab_rl = lb
                .as_ref()
                .map(|shell| dipole_pair_block(&ra, shell, component));
            let ab_lr = la
                .as_ref()
                .map(|shell| dipole_pair_block(shell, &rb, component));
            let ab_ll = la.as_ref().and_then(|la_shell| {
                lb.as_ref()
                    .map(|lb_shell| dipole_pair_block(la_shell, lb_shell, component))
            });
            for ax in 0..3 {
                let bra_r = combine_bra_derivative(sh_a.l, sh_b.l + 1, ax, &ab_rr, ab_lr.as_ref());
                let bra_l = ab_rl.as_ref().map(|block| {
                    combine_bra_derivative(sh_a.l, sh_b.l - 1, ax, block, ab_ll.as_ref())
                });
                for by in 0..3 {
                    let cart = combine_ket_derivative(sh_a.l, sh_b.l, by, &bra_r, bra_l.as_ref());
                    let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                    h[(3 * ia + ax, 3 * ib + by)] +=
                        pair_weight * contract(m, sh_a.sph_off, sh_b.sph_off, &sph);
                }
            }
            h
        })
        .collect();
    let mut total = Matrix::zeros(3 * natoms, 3 * natoms);
    for partial in partials {
        for (target, value) in total.as_mut_slice().iter_mut().zip(partial.as_slice()) {
            *target += *value;
        }
    }
    total
}

/// `sum_mn m_mn d2 V_mn` skeleton block, all centers (basis + operator)
/// with per-atom charges (Z_eff-ready).
/// Second derivative of `Tr[m * V]` where `V` is the one-electron operator
/// of a set of CHARGE SITES, contracted into the 3N x 3N nuclear block.
///
/// A site is `(block, owner)`: `block(bra, ket, site)` returns the
/// Cartesian `(ab|site)` matrix for the given (possibly raised/lowered)
/// shells, and `owner(site)` says which atom the site rides on. Point
/// nuclei are the case "one site per atom, owner = itself"; continuum
/// surface charges are "many sites per atom, owner = the atom whose sphere
/// carries them". Both then share this machinery — including the sum rules
/// `AC = -AA - AB`, `BC = -BA - BB`, `CC = AA + AB + BA + BB` that supply
/// the site-centre derivatives without a third derivative class.
pub(crate) fn charge_site_second_contraction<F, G>(
    mb: &MolBasis,
    natoms: usize,
    n_sites: usize,
    owner_of: G,
    m: &Matrix,
    block: F,
) -> Matrix
where
    F: Fn(&MolShell, &MolShell, usize) -> Matrix + Sync,
    G: Fn(usize) -> usize + Sync,
{
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Matrix> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut h = Matrix::zeros(3 * natoms, 3 * natoms);
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            let ra = raised(sh_a);
            let rra = raised(&ra);
            let la_lo = lowered(sh_a);
            let lla = la_lo.as_ref().and_then(lowered);
            let rb = raised(sh_b);
            let rrb = raised(&rb);
            let lb_lo = lowered(sh_b);
            let llb = lb_lo.as_ref().and_then(lowered);
            let (ia, ib) = (sh_a.atom, sh_b.atom);
            for site in 0..n_sites {
                let c = owner_of(site);
                // ---- AA: nested second derivative on the bra ----
                let blk_rr = block(&rra, sh_b, site);
                // rl needs coefs 2a c over shell l: raised-then-lowered.
                let rl_sh = lowered(&ra).expect("l+1 always lowerable");
                let blk_rl = block(&rl_sh, sh_b, site);
                let blk_ll = lla.as_ref().map(|lo| block(lo, sh_b, site));
                // ---- BB: nested second derivative on the ket ----
                let blk_rr_b = block(sh_a, &rrb, site);
                let rl_shb = lowered(&rb).expect("l+1 lowerable");
                let blk_rl_b = block(sh_a, &rl_shb, site);
                let blk_ll_b = llb.as_ref().map(|lo| block(sh_a, lo, site));
                // ---- AB: mixed ----
                let m_rr = block(&ra, &rb, site);
                let m_rl = lb_lo.as_ref().map(|lo| block(&ra, lo, site));
                let m_lr = la_lo.as_ref().map(|lo| block(lo, &rb, site));
                let m_ll = la_lo
                    .as_ref()
                    .and_then(|loa| lb_lo.as_ref().map(|lob| block(loa, lob, site)));
                for ax in 0..3 {
                    let bra_r =
                        combine_bra_derivative(sh_a.l, sh_b.l + 1, ax, &m_rr, m_lr.as_ref());
                    let bra_l = m_rl.as_ref().map(|rlb| {
                        combine_bra_derivative(sh_a.l, sh_b.l - 1, ax, rlb, m_ll.as_ref())
                    });
                    for by in 0..3 {
                        // AA(ax,by)
                        let aa = {
                            let cart = combine_bra_second(
                                sh_a.l,
                                sh_b.l,
                                ax,
                                by,
                                &blk_rr,
                                &blk_rl,
                                blk_ll.as_ref(),
                            );
                            let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                            contract(m, sh_a.sph_off, sh_b.sph_off, &sph)
                        };
                        // BB(ax,by): nested on ket = bra-nested of the
                        // transposed operator; use the ket-side second
                        // combine via transpose symmetry of the blocks.
                        let bb = {
                            let cart = combine_ket_second(
                                sh_a.l,
                                sh_b.l,
                                ax,
                                by,
                                &blk_rr_b,
                                &blk_rl_b,
                                blk_ll_b.as_ref(),
                            );
                            let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                            contract(m, sh_a.sph_off, sh_b.sph_off, &sph)
                        };
                        // AB(ax,by) mixed
                        let ab = {
                            let cart =
                                combine_ket_derivative(sh_a.l, sh_b.l, by, &bra_r, bra_l.as_ref());
                            let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                            contract(m, sh_a.sph_off, sh_b.sph_off, &sph)
                        };
                        // Sum rules: AC = -AA - AB, BC = -BA - BB,
                        // CC = AA + AB + BA + BB (indices track (ax, by)).
                        let ac = -aa - ab;
                        // BA(ax,by) = d2/dB_ax dA_by: recompute from the
                        // same mixed blocks with the axes swapped,
                        // BA(ax,by) = AB(by,ax).
                        let ba = {
                            let bra_r2 = combine_bra_derivative(
                                sh_a.l,
                                sh_b.l + 1,
                                by,
                                &m_rr,
                                m_lr.as_ref(),
                            );
                            let bra_l2 = m_rl.as_ref().map(|rlb| {
                                combine_bra_derivative(sh_a.l, sh_b.l - 1, by, rlb, m_ll.as_ref())
                            });
                            let cart = combine_ket_derivative(
                                sh_a.l,
                                sh_b.l,
                                ax,
                                &bra_r2,
                                bra_l2.as_ref(),
                            );
                            let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                            contract(m, sh_a.sph_off, sh_b.sph_off, &sph)
                        };
                        let bc = -ba - bb;
                        let cc_v = aa + ab + ba + bb;
                        let mut add = |r: usize, rx: usize, s: usize, sy: usize, v: f64| {
                            h[(3 * r + rx, 3 * s + sy)] += v;
                        };
                        add(ia, ax, ia, by, aa);
                        add(ia, ax, ib, by, ab);
                        add(ib, ax, ia, by, ba);
                        add(ib, ax, ib, by, bb);
                        add(ia, ax, c, by, ac);
                        add(c, by, ia, ax, ac);
                        add(ib, ax, c, by, bc);
                        add(c, by, ib, ax, bc);
                        add(c, ax, c, by, cc_v);
                    }
                }
            }
            h
        })
        .collect();
    let mut total = Matrix::zeros(3 * natoms, 3 * natoms);
    for p in partials {
        for (t, v) in total.as_mut_slice().iter_mut().zip(p.as_slice()) {
            *t += *v;
        }
    }
    total
}

/// Point-nucleus specialization: one site per atom, each owned by itself.
pub fn nuclear_second_contraction(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    m: &Matrix,
) -> Matrix {
    let natoms = mol.len();
    let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
    charge_site_second_contraction(
        mb,
        natoms,
        natoms,
        |s| s,
        m,
        |sa, sb, s| {
            if charges[s] == 0.0 {
                Matrix::zeros(
                    crate::basis::cart_components(sa.l).len(),
                    crate::basis::cart_components(sb.l).len(),
                )
            } else {
                nuclear_block(sa, sb, centers[s], charges[s])
            }
        },
    )
}

/// Ket-side nested second derivative (mirror of `combine_bra_second`).
pub(crate) fn combine_ket_second(
    la: u8,
    lb: u8,
    ax: usize,
    ay: usize,
    rr: &Matrix,
    rl: &Matrix,
    ll: Option<&Matrix>,
) -> Matrix {
    let comps_b = cart_components(lb);
    let nca = n_cart(la);
    let mut out = Matrix::zeros(nca, comps_b.len());
    let step = |c: (i32, i32, i32), ax: usize, d: i32| -> (i32, i32, i32) {
        match ax {
            0 => (c.0 + d, c.1, c.2),
            1 => (c.0, c.1 + d, c.2),
            _ => (c.0, c.1, c.2 + d),
        }
    };
    let comp = |c: (i32, i32, i32)| [c.0, c.1, c.2];
    for (j, &u) in comps_b.iter().enumerate() {
        let uy = comp(u)[ay];
        let up2 = step(step(u, ay, 1), ax, 1);
        let j_rr = comp_index(lb + 2, up2);
        let upy = step(u, ay, 1);
        let upy_x = comp(upy)[ax];
        let t2 = step(upy, ax, -1);
        let t3 = if uy > 0 {
            Some(step(step(u, ay, -1), ax, 1))
        } else {
            None
        };
        let t4 = if uy > 0 {
            let umy = step(u, ay, -1);
            let umy_x = comp(umy)[ax];
            if umy_x > 0 {
                Some((step(umy, ax, -1), (uy * umy_x) as f64))
            } else {
                None
            }
        } else {
            None
        };
        for i in 0..nca {
            let mut v = rr[(i, j_rr)];
            if t2.0 >= 0 && t2.1 >= 0 && t2.2 >= 0 {
                v -= upy_x as f64 * rl[(i, comp_index(lb, t2))];
            }
            if let Some(c3) = t3 {
                if c3.0 >= 0 && c3.1 >= 0 && c3.2 >= 0 {
                    v -= uy as f64 * rl[(i, comp_index(lb, c3))];
                }
            }
            if let Some((c4, f4)) = t4 {
                v += f4 * ll.expect("ll ket block")[(i, comp_index(lb - 2, c4))];
            }
            out[(i, j)] = v;
        }
    }
    out
}

/// ERI skeleton second-derivative contraction with the fixed two-electron
/// energy weight `X(mnls) = 1/2 coul P(mn)P(ls) - 1/2 exch sum_s
/// Ps(ml)Ps(ns)`:
/// `H[(A,x),(B,y)] = sum_quartets X d2(mn|ls)/dA_x dB_y`.
///
/// Canonical quartets (a<=b, c<=d, (a,b)<=(c,d)) with an explicit
/// permutation scatter. Per canonical quartet the SPHERICAL mixed
/// second-derivative tensors are built once for all six slot pairs and
/// all nine axis pairs; each distinct ordered permutation then only pays
/// a weighted sum over elements. Both the integral evaluation and the
/// Cartesian->spherical transformation therefore drop ~8x versus the
/// ordered loop, which is kept as the test-only reference. Only MIXED
/// index-pair second derivatives are evaluated; same-index blocks follow
/// from translational invariance `d_i d_i = -sum_{j!=i} d_i d_j`.
#[allow(clippy::too_many_arguments)]
pub fn eri_second_contraction(
    mb: &MolBasis,
    natoms: usize,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    kernel: CoulombKernel,
    coul_coef: f64,
    exch_coef: f64,
    schwarz: &Matrix,
    threshold: f64,
) -> Matrix {
    let ns = mb.shells.len();
    // Density screening per SHELL PAIR rather than by the global maximum:
    // the energy weight X(mnls) is bounded by the density blocks actually
    // touching the quartet, so a local bound prunes far more aggressively
    // while remaining an upper bound (the same policy the direct SCF J/K
    // build already uses).
    let pmax_shell = {
        let mut m = Matrix::zeros(ns, ns);
        for a in 0..ns {
            let (oa, na) = (mb.shells[a].sph_off, n_sph(mb.shells[a].l));
            for b in 0..ns {
                let (ob, nb) = (mb.shells[b].sph_off, n_sph(mb.shells[b].l));
                let mut mx: f64 = 0.0;
                for i in 0..na {
                    for j in 0..nb {
                        mx = mx.max(p_tot[(oa + i, ob + j)].abs());
                        mx = mx.max(p_spin[0][(oa + i, ob + j)].abs());
                        mx = mx.max(p_spin[1][(oa + i, ob + j)].abs());
                    }
                }
                m[(a, b)] = mx;
            }
        }
        m
    };
    let mut quartets = Vec::new();
    for a in 0..ns {
        for b in a..ns {
            for c in a..ns {
                let d0 = if c == a { b } else { c };
                for d in d0..ns {
                    // |X| <= max(P_ab P_cd, P_ac P_bd, P_ad P_bc) up to the
                    // coulomb/exchange prefactors, all covered by the max.
                    let dens = (pmax_shell[(a, b)] * pmax_shell[(c, d)])
                        .max(pmax_shell[(a, c)] * pmax_shell[(b, d)])
                        .max(pmax_shell[(a, d)] * pmax_shell[(b, c)])
                        .max(1e-30);
                    if schwarz[(a, b)] * schwarz[(c, d)] * dens >= threshold {
                        quartets.push((a, b, c, d));
                    }
                }
            }
        }
    }
    let partials: Vec<Matrix> = quartets
        .par_chunks(std::cmp::max(
            1,
            quartets.len() / (8 * rayon::current_num_threads()),
        ))
        .map(|chunk| {
            let mut h = Matrix::zeros(3 * natoms, 3 * natoms);
            for &(a, b, c, d) in chunk {
                let quad = [a, b, c, d];
                let sh = [&mb.shells[a], &mb.shells[b], &mb.shells[c], &mb.shells[d]];
                if sh.iter().all(|s| s.atom == sh[0].atom) {
                    continue;
                }
                let dims = [
                    n_sph(sh[0].l),
                    n_sph(sh[1].l),
                    n_sph(sh[2].l),
                    n_sph(sh[3].l),
                ];
                let nel = dims[0] * dims[1] * dims[2] * dims[3];
                let stride = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
                let d2t = quartet_mixed_second(sh, kernel);
                // Each DISTINCT ordered permutation contributes with its own
                // density weights and its own atom labels.
                let mut seen: Vec<[usize; 4]> = Vec::with_capacity(8);
                for sig in SIGMA.iter() {
                    let t = [quad[sig[0]], quad[sig[1]], quad[sig[2]], quad[sig[3]]];
                    if seen.contains(&t) {
                        continue;
                    }
                    seen.push(t);
                    let tsh = [
                        &mb.shells[t[0]],
                        &mb.shells[t[1]],
                        &mb.shells[t[2]],
                        &mb.shells[t[3]],
                    ];
                    let toffs = [
                        tsh[0].sph_off,
                        tsh[1].sph_off,
                        tsh[2].sph_off,
                        tsh[3].sph_off,
                    ];
                    let tatoms = [tsh[0].atom, tsh[1].atom, tsh[2].atom, tsh[3].atom];
                    let tdims = [dims[sig[0]], dims[sig[1]], dims[sig[2]], dims[sig[3]]];
                    // Element weights X in the PERMUTED (T) AO layout, stored
                    // against the canonical element index.
                    let mut xw = vec![0.0; nel];
                    for j0 in 0..tdims[0] {
                        let mu = toffs[0] + j0;
                        for j1 in 0..tdims[1] {
                            let nu = toffs[1] + j1;
                            for j2 in 0..tdims[2] {
                                let lam = toffs[2] + j2;
                                for j3 in 0..tdims[3] {
                                    let sg = toffs[3] + j3;
                                    let jj = [j0, j1, j2, j3];
                                    let mut el = 0usize;
                                    for k in 0..4 {
                                        el += jj[k] * stride[sig[k]];
                                    }
                                    let mut x =
                                        0.5 * coul_coef * p_tot[(mu, nu)] * p_tot[(lam, sg)];
                                    if exch_coef != 0.0 {
                                        for ps in p_spin {
                                            x -= 0.5 * exch_coef * ps[(mu, lam)] * ps[(nu, sg)];
                                        }
                                    }
                                    xw[el] = x;
                                }
                            }
                        }
                    }
                    for (ti, tj) in PAIRS {
                        // T-slot pair (ti,tj) carries canonical slots
                        // (sig[ti], sig[tj]); the stored tensor is indexed by
                        // the SORTED canonical pair, with the axis roles
                        // swapped when the order flips.
                        let (p, q) = (sig[ti], sig[tj]);
                        let (pair_idx, swap) = if p < q {
                            (pair_index(p, q), false)
                        } else {
                            (pair_index(q, p), true)
                        };
                        let block = &d2t[pair_idx];
                        let mut m_ij = [[0.0f64; 3]; 3];
                        for ax in 0..3 {
                            for by in 0..3 {
                                let comp = if swap { by * 3 + ax } else { ax * 3 + by };
                                let vals = &block[comp];
                                let mut acc = 0.0;
                                for el in 0..nel {
                                    let x = xw[el];
                                    if x != 0.0 {
                                        acc += x * vals[el];
                                    }
                                }
                                m_ij[ax][by] = acc;
                            }
                        }
                        let (ai, aj) = (tatoms[ti], tatoms[tj]);
                        for ax in 0..3 {
                            for by in 0..3 {
                                let v = m_ij[ax][by];
                                h[(3 * ai + ax, 3 * aj + by)] += v;
                                h[(3 * aj + by, 3 * ai + ax)] += v;
                                h[(3 * ai + ax, 3 * ai + by)] -= v;
                                h[(3 * aj + by, 3 * aj + ax)] -= v;
                            }
                        }
                    }
                }
            }
            h
        })
        .collect();
    let mut total = Matrix::zeros(3 * natoms, 3 * natoms);
    for p in partials {
        for (t, v) in total.as_mut_slice().iter_mut().zip(p.as_slice()) {
            *t += *v;
        }
    }
    total
}

/// Spherical mixed second derivatives of one quartet for the six slot
/// pairs (`PAIRS` order): `d2t[pair][ax*3 + by][element]`, the element
/// index running over the spherical components of the four shells in slot
/// order. Same-slot blocks follow from translational invariance,
/// `d_i d_i = -sum_{j != i} d_i d_j`.
pub(crate) fn quartet_mixed_second(
    sh: [&MolShell; 4],
    kernel: CoulombKernel,
) -> Vec<Vec<Vec<f64>>> {
    let dims = [
        n_sph(sh[0].l),
        n_sph(sh[1].l),
        n_sph(sh[2].l),
        n_sph(sh[3].l),
    ];
    let nel = dims[0] * dims[1] * dims[2] * dims[3];
    let mut d2t: Vec<Vec<Vec<f64>>> = Vec::with_capacity(6);
    for (i, j) in PAIRS {
        let ri = raised(sh[i]);
        let li = lowered(sh[i]);
        let rj = raised(sh[j]);
        let lj = lowered(sh[j]);
        let mut mask = [true; 4];
        mask[i] = false;
        mask[j] = false;
        let build = |vi: &MolShell, vj: &MolShell| {
            let mut ms: [&MolShell; 4] = [sh[0], sh[1], sh[2], sh[3]];
            ms[i] = vi;
            ms[j] = vj;
            eri_quartet_masked(ms[0], ms[1], ms[2], ms[3], kernel, mask)
        };
        let q_rr = build(&ri, &rj);
        let q_rl = lj.as_ref().map(|l| build(&ri, l));
        let q_lr = li.as_ref().map(|l| build(l, &rj));
        let q_ll = li
            .as_ref()
            .and_then(|l1| lj.as_ref().map(|l2| build(l1, l2)));
        let (l_i, l_j) = (sh[i].l, sh[j].l);
        let comps_i = cart_components(l_i);
        let comps_j = cart_components(l_j);
        let c2s_i = aos::cart_to_sph(l_i);
        let c2s_j = aos::cart_to_sph(l_j);
        let mut block = vec![vec![0.0; nel]; 9];
        let mut el = 0usize;
        let mut s_idx = [0usize; 4];
        for i0 in 0..dims[0] {
            s_idx[0] = i0;
            for i1 in 0..dims[1] {
                s_idx[1] = i1;
                for i2 in 0..dims[2] {
                    s_idx[2] = i2;
                    for i3 in 0..dims[3] {
                        s_idx[3] = i3;
                        let (si, sj) = (s_idx[i], s_idx[j]);
                        for ax in 0..3 {
                            for by in 0..3 {
                                let mut dv = 0.0;
                                for (ci, &cci) in comps_i.iter().enumerate() {
                                    let wi = c2s_i[(ci, si)];
                                    if wi == 0.0 {
                                        continue;
                                    }
                                    for (cj, &ccj) in comps_j.iter().enumerate() {
                                        let wj = c2s_j[(cj, sj)];
                                        if wj == 0.0 {
                                            continue;
                                        }
                                        dv += wi
                                            * wj
                                            * d2_element(
                                                &s_idx,
                                                i,
                                                j,
                                                cci,
                                                ccj,
                                                ax,
                                                by,
                                                l_i,
                                                l_j,
                                                &q_rr,
                                                q_rl.as_ref(),
                                                q_lr.as_ref(),
                                                q_ll.as_ref(),
                                            );
                                    }
                                }
                                block[ax * 3 + by][el] = dv;
                            }
                        }
                        el += 1;
                    }
                }
            }
        }
        d2t.push(block);
    }
    d2t
}

/// The six unordered slot pairs, in the storage order of `d2t`.
pub(crate) const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Index of the sorted slot pair `(i<j)` inside `PAIRS`.
fn pair_index(i: usize, j: usize) -> usize {
    PAIRS
        .iter()
        .position(|&p| p == (i, j))
        .expect("sorted slot pair is one of the six")
}

/// The 8-fold ERI permutation group as slot maps (`T[k] = S[SIGMA[g][k]]`).
const SIGMA: [[usize; 4]; 8] = [
    [0, 1, 2, 3],
    [1, 0, 2, 3],
    [0, 1, 3, 2],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 0, 1],
    [2, 3, 1, 0],
    [3, 2, 1, 0],
];

/// Slow reference: the original full ordered quartet loop, kept as the
/// permanent cross-check of the canonical path above.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn eri_second_contraction_ordered(
    mb: &MolBasis,
    natoms: usize,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    kernel: CoulombKernel,
    coul_coef: f64,
    exch_coef: f64,
    schwarz: &Matrix,
    threshold: f64,
) -> Matrix {
    let ns = mb.shells.len();
    let pmax = p_tot
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1e-30);
    let mut h = Matrix::zeros(3 * natoms, 3 * natoms);
    for a in 0..ns {
        for b in 0..ns {
            for c in 0..ns {
                for d in 0..ns {
                    if schwarz[(a, b)] * schwarz[(c, d)] * pmax * pmax < threshold {
                        continue;
                    }
                    let sh = [&mb.shells[a], &mb.shells[b], &mb.shells[c], &mb.shells[d]];
                    let atoms = [sh[0].atom, sh[1].atom, sh[2].atom, sh[3].atom];
                    if atoms.iter().all(|&x| x == atoms[0]) {
                        continue;
                    }
                    let dims = [
                        n_sph(sh[0].l),
                        n_sph(sh[1].l),
                        n_sph(sh[2].l),
                        n_sph(sh[3].l),
                    ];
                    let offs = [sh[0].sph_off, sh[1].sph_off, sh[2].sph_off, sh[3].sph_off];
                    let nel = dims[0] * dims[1] * dims[2] * dims[3];
                    let mut xw = vec![0.0; nel];
                    {
                        let mut el = 0usize;
                        for i0 in 0..dims[0] {
                            let mu = offs[0] + i0;
                            for i1 in 0..dims[1] {
                                let nu = offs[1] + i1;
                                for i2 in 0..dims[2] {
                                    let lam = offs[2] + i2;
                                    for i3 in 0..dims[3] {
                                        let sig = offs[3] + i3;
                                        let mut x =
                                            0.5 * coul_coef * p_tot[(mu, nu)] * p_tot[(lam, sig)];
                                        if exch_coef != 0.0 {
                                            for ps in p_spin {
                                                x -=
                                                    0.5 * exch_coef * ps[(mu, lam)] * ps[(nu, sig)];
                                            }
                                        }
                                        xw[el] = x;
                                        el += 1;
                                    }
                                }
                            }
                        }
                    }
                    for i in 0..3usize {
                        for j in (i + 1)..4usize {
                            let ri = raised(sh[i]);
                            let li = lowered(sh[i]);
                            let rj = raised(sh[j]);
                            let lj = lowered(sh[j]);
                            let mut mask = [true; 4];
                            mask[i] = false;
                            mask[j] = false;
                            let build = |vi: &MolShell, vj: &MolShell| {
                                let mut ms: [&MolShell; 4] = [sh[0], sh[1], sh[2], sh[3]];
                                ms[i] = vi;
                                ms[j] = vj;
                                eri_quartet_masked(ms[0], ms[1], ms[2], ms[3], kernel, mask)
                            };
                            let q_rr = build(&ri, &rj);
                            let q_rl = lj.as_ref().map(|l| build(&ri, l));
                            let q_lr = li.as_ref().map(|l| build(l, &rj));
                            let q_ll = li
                                .as_ref()
                                .and_then(|l1| lj.as_ref().map(|l2| build(l1, l2)));
                            let (l_i, l_j) = (sh[i].l, sh[j].l);
                            let comps_i = cart_components(l_i);
                            let comps_j = cart_components(l_j);
                            let c2s_i = aos::cart_to_sph(l_i);
                            let c2s_j = aos::cart_to_sph(l_j);
                            let mut m_ij = [[0.0f64; 3]; 3];
                            let mut el = 0usize;
                            let mut s_idx = [0usize; 4];
                            for i0 in 0..dims[0] {
                                s_idx[0] = i0;
                                for i1 in 0..dims[1] {
                                    s_idx[1] = i1;
                                    for i2 in 0..dims[2] {
                                        s_idx[2] = i2;
                                        for i3 in 0..dims[3] {
                                            s_idx[3] = i3;
                                            let x = xw[el];
                                            el += 1;
                                            if x == 0.0 {
                                                continue;
                                            }
                                            let (si, sj) = (s_idx[i], s_idx[j]);
                                            for ax in 0..3 {
                                                for by in 0..3 {
                                                    let mut dv = 0.0;
                                                    for (ci, &cci) in comps_i.iter().enumerate() {
                                                        let wi = c2s_i[(ci, si)];
                                                        if wi == 0.0 {
                                                            continue;
                                                        }
                                                        for (cj, &ccj) in comps_j.iter().enumerate()
                                                        {
                                                            let wj = c2s_j[(cj, sj)];
                                                            if wj == 0.0 {
                                                                continue;
                                                            }
                                                            dv += wi
                                                                * wj
                                                                * d2_element(
                                                                    &s_idx,
                                                                    i,
                                                                    j,
                                                                    cci,
                                                                    ccj,
                                                                    ax,
                                                                    by,
                                                                    l_i,
                                                                    l_j,
                                                                    &q_rr,
                                                                    q_rl.as_ref(),
                                                                    q_lr.as_ref(),
                                                                    q_ll.as_ref(),
                                                                );
                                                        }
                                                    }
                                                    m_ij[ax][by] += x * dv;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            let (ai, aj) = (atoms[i], atoms[j]);
                            for ax in 0..3 {
                                for by in 0..3 {
                                    let v = m_ij[ax][by];
                                    h[(3 * ai + ax, 3 * aj + by)] += v;
                                    h[(3 * aj + by, 3 * ai + ax)] += v;
                                    h[(3 * ai + ax, 3 * ai + by)] -= v;
                                    h[(3 * aj + by, 3 * aj + ax)] -= v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    h
}

/// One Cartesian-component mixed second-derivative element: apply the
/// raise/lower rule at index i (component cci, axis ax) and index j
/// (component ccj, axis by) on the four masked quartet variants.
#[allow(clippy::too_many_arguments)]
fn d2_element(
    s_idx: &[usize; 4],
    i: usize,
    j: usize,
    cci: (i32, i32, i32),
    ccj: (i32, i32, i32),
    ax: usize,
    by: usize,
    l_i: u8,
    l_j: u8,
    q_rr: &super::eri::Quartet,
    q_rl: Option<&super::eri::Quartet>,
    q_lr: Option<&super::eri::Quartet>,
    q_ll: Option<&super::eri::Quartet>,
) -> f64 {
    let step = |c: (i32, i32, i32), ax: usize, d: i32| -> (i32, i32, i32) {
        match ax {
            0 => (c.0 + d, c.1, c.2),
            1 => (c.0, c.1 + d, c.2),
            _ => (c.0, c.1, c.2 + d),
        }
    };
    let comp = |c: (i32, i32, i32)| [c.0, c.1, c.2];
    let li_x = comp(cci)[ax];
    let lj_y = comp(ccj)[by];
    let up_i = comp_index(l_i + 1, step(cci, ax, 1));
    let up_j = comp_index(l_j + 1, step(ccj, by, 1));
    let get = |q: &super::eri::Quartet, vi: usize, vj: usize| {
        let mut idx = *s_idx;
        idx[i] = vi;
        idx[j] = vj;
        q.get(idx[0], idx[1], idx[2], idx[3])
    };
    let mut v = get(q_rr, up_i, up_j);
    if lj_y > 0 {
        let dn_j = comp_index(l_j - 1, step(ccj, by, -1));
        v -= lj_y as f64 * get(q_rl.expect("rl quartet"), up_i, dn_j);
    }
    if li_x > 0 {
        let dn_i = comp_index(l_i - 1, step(cci, ax, -1));
        v -= li_x as f64 * get(q_lr.expect("lr quartet"), dn_i, up_j);
        if lj_y > 0 {
            let dn_j = comp_index(l_j - 1, step(ccj, by, -1));
            v += (li_x * lj_y) as f64 * get(q_ll.expect("ll quartet"), dn_i, dn_j);
        }
    }
    v
}

/// Second derivative of the nuclear repulsion on explicit charges.
pub fn nuclear_repulsion_hessian(mol: &Molecule, charges: &[f64]) -> Matrix {
    let n = mol.len();
    let mut h = Matrix::zeros(3 * n, 3 * n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let rij = mol.atoms[i].position - mol.atoms[j].position;
            let r = rij.norm();
            let zz = charges[i] * charges[j];
            let r3 = r * r * r;
            let r5 = r3 * r * r;
            for x in 0..3 {
                for y in 0..3 {
                    let dxy = if x == y { 1.0 } else { 0.0 };
                    // d2 (1/R) / dRi_x dRi_y for the pair (i, j)
                    let v = zz * (3.0 * rij.get(x) * rij.get(y) / r5 - dxy / r3);
                    h[(3 * i + x, 3 * i + y)] += v;
                    h[(3 * i + x, 3 * j + y)] -= v;
                }
            }
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{data, MolBasis};
    use crate::integrals::{kinetic_matrix, nuclear_matrix, overlap_matrix};
    use crate::system::Atom;

    fn mol() -> Molecule {
        Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.05, 0.2217),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.03, 1.4309, -0.8867),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, -1.4309, -0.8867),
            },
        ])
    }

    fn bump(m: &Molecule, atom: usize, ax: usize, h: f64) -> Molecule {
        let mut mm = m.clone();
        match ax {
            0 => mm.atoms[atom].position.x += h,
            1 => mm.atoms[atom].position.y += h,
            _ => mm.atoms[atom].position.z += h,
        }
        mm
    }

    fn fixed_m(n: usize) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        let mut state = 0xfeed_f00d_u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for i in 0..n {
            for j in i..n {
                let v = 0.5 * rng();
                m[(i, j)] = v;
                m[(j, i)] = v;
            }
        }
        m
    }

    /// The canonical-quartet ERI second contraction must reproduce the
    /// slow ordered-loop reference entry for entry.
    #[test]
    fn canonical_eri_second_contraction_matches_ordered_reference() {
        use crate::integrals::eri::schwarz_matrix;
        let m = mol();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&m, &set).unwrap();
        let n = mb.nao_sph;
        let p_a = fixed_m(n);
        let p_b = fixed_m(n);
        let mut p_tot = p_a.clone();
        for (t, v) in p_tot.as_mut_slice().iter_mut().zip(p_b.as_slice()) {
            *t += *v;
        }
        let kernel = CoulombKernel::Coulomb;
        let schwarz = schwarz_matrix(&mb, kernel);
        let t0 = std::time::Instant::now();
        let h_new = eri_second_contraction(
            &mb,
            m.len(),
            &p_tot,
            [&p_a, &p_b],
            kernel,
            1.0,
            0.2,
            &schwarz,
            1e-14,
        );
        let t_canon = t0.elapsed();
        let t1 = std::time::Instant::now();
        let h_ref = eri_second_contraction_ordered(
            &mb,
            m.len(),
            &p_tot,
            [&p_a, &p_b],
            kernel,
            1.0,
            0.2,
            &schwarz,
            1e-14,
        );
        let t_ordered = t1.elapsed();
        eprintln!(
            "ERI second contraction: canonical {:?} vs ordered reference {:?}",
            t_canon, t_ordered
        );
        let mut worst = 0.0f64;
        for (x, y) in h_new.as_slice().iter().zip(h_ref.as_slice()) {
            worst = worst.max((x - y).abs());
        }
        assert!(
            worst < 1e-10,
            "canonical vs ordered ERI second contraction: worst |Δ| = {worst:.3e}"
        );
    }

    /// Skeleton second contractions vs double central differences of
    /// Tr[m O] (all 3N x 3N entries; also fixed-origin dipole and E_nn).
    #[test]
    fn second_contractions_match_finite_differences() {
        let m0 = mol();
        let set = data::load("sto-3g").unwrap();
        let mb0 = MolBasis::build(&m0, &set).unwrap();
        let n = mb0.nao_sph;
        let natoms = m0.len();
        let mm = fixed_m(n);
        let charges: Vec<f64> = m0.atoms.iter().map(|a| a.z as f64).collect();

        let hs = overlap_second_contraction(&mb0, &mm, natoms);
        let ht = kinetic_second_contraction(&mb0, &mm, natoms);
        let hv = nuclear_second_contraction(&mb0, &m0, &charges, &mm);
        let hd = dipole_second_contraction(&mb0, &mm, natoms, 2);
        let hn = nuclear_repulsion_hessian(&m0, &charges);

        let energy = |mo: &Molecule| -> (f64, f64, f64, f64, f64) {
            let mb = MolBasis::build(mo, &set).unwrap();
            let s = mm.frobenius_dot(&overlap_matrix(&mb));
            let t = mm.frobenius_dot(&kinetic_matrix(&mb));
            let v = mm.frobenius_dot(&nuclear_matrix(&mb, mo, None));
            let d = mm.frobenius_dot(&crate::integrals::dipole_matrices(&mb, Vec3::zero())[2]);
            let mut enn = 0.0;
            for i in 0..mo.len() {
                for j in (i + 1)..mo.len() {
                    let r = (mo.atoms[i].position - mo.atoms[j].position).norm();
                    enn += charges[i] * charges[j] / r;
                }
            }
            (s, t, v, d, enn)
        };
        let h = 1e-4;
        for a in 0..natoms {
            for ax in 0..3 {
                for b in 0..natoms {
                    for by in 0..3 {
                        // f(+a,+b) - f(+a,-b) - f(-a,+b) + f(-a,-b) / 4h^2
                        let pp = energy(&bump(&bump(&m0, a, ax, h), b, by, h));
                        let pm = energy(&bump(&bump(&m0, a, ax, h), b, by, -h));
                        let mp = energy(&bump(&bump(&m0, a, ax, -h), b, by, h));
                        let mmm = energy(&bump(&bump(&m0, a, ax, -h), b, by, -h));
                        let fd = |sel: fn(&(f64, f64, f64, f64, f64)) -> f64| {
                            (sel(&pp) - sel(&pm) - sel(&mp) + sel(&mmm)) / (4.0 * h * h)
                        };
                        let (r, c) = (3 * a + ax, 3 * b + by);
                        let checks = [
                            ("S", fd(|e| e.0), hs[(r, c)]),
                            ("T", fd(|e| e.1), ht[(r, c)]),
                            ("V", fd(|e| e.2), hv[(r, c)]),
                            ("Dz", fd(|e| e.3), hd[(r, c)]),
                            ("Enn", fd(|e| e.4), hn[(r, c)]),
                        ];
                        for (tag, f, an) in checks {
                            assert!(
                                (an - f).abs() < 5e-6 * f.abs().max(1.0),
                                "{tag} ({a},{ax})x({b},{by}): analytic {an} vs FD {f}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// ERI skeleton second contraction vs double central differences of the
    /// fixed-weight two-electron energy.
    #[test]
    fn eri_second_contraction_matches_finite_differences() {
        use crate::integrals::{eri_incore, schwarz_matrix};
        let m0 = mol();
        let set = data::load("sto-3g").unwrap();
        let mb0 = MolBasis::build(&m0, &set).unwrap();
        let n = mb0.nao_sph;
        let natoms = m0.len();
        let mm = fixed_m(n);
        let (coul, exch) = (1.0, 0.6);
        let schwarz = schwarz_matrix(&mb0, CoulombKernel::Coulomb);
        let h2e = eri_second_contraction(
            &mb0,
            natoms,
            &mm,
            [&mm, &mm],
            CoulombKernel::Coulomb,
            coul,
            exch,
            &schwarz,
            1e-16,
        );
        let energy = |mo: &Molecule| -> f64 {
            let mb = MolBasis::build(mo, &set).unwrap();
            let eri = eri_incore(&mb, CoulombKernel::Coulomb, 256).unwrap();
            let mut e = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    for lam in 0..n {
                        for sig in 0..n {
                            let x = 0.5 * coul * mm[(mu, nu)] * mm[(lam, sig)]
                                - 0.5 * exch * 2.0 * mm[(mu, lam)] * mm[(nu, sig)];
                            e += x * eri.get(mu, nu, lam, sig);
                        }
                    }
                }
            }
            e
        };
        let h = 1e-4;
        // Spot-check a representative set of entries (full FD is slow).
        for &(a, ax, b, by) in &[
            (0usize, 0usize, 0usize, 0usize),
            (0, 2, 0, 2),
            (0, 1, 1, 1),
            (0, 2, 1, 0),
            (1, 0, 2, 2),
            (2, 1, 2, 1),
            (1, 2, 1, 2),
        ] {
            let pp = energy(&bump(&bump(&m0, a, ax, h), b, by, h));
            let pm = energy(&bump(&bump(&m0, a, ax, h), b, by, -h));
            let mp = energy(&bump(&bump(&m0, a, ax, -h), b, by, h));
            let mmm = energy(&bump(&bump(&m0, a, ax, -h), b, by, -h));
            let fd = (pp - pm - mp + mmm) / (4.0 * h * h);
            let an = h2e[(3 * a + ax, 3 * b + by)];
            assert!(
                (an - fd).abs() < 5e-6 * fd.abs().max(1.0),
                "ERI2 ({a},{ax})x({b},{by}): analytic {an} vs FD {fd}"
            );
        }
    }
}
