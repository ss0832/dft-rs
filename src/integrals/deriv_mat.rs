// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Matrix-valued first derivatives of the one-electron integrals with
//! respect to nuclear coordinates: S^(a,x), T^(a,x), V^(a,x) as full
//! spherical-AO matrices (the CPKS right-hand sides and the analytic
//! Hessian consume these; the scalar gradient contractions in `deriv.rs`
//! stay untouched for the force path).
//!
//! Same primitive identity as deriv.rs:
//! d/dA_x phi(l) = 2a phi(l+1_x) - l_x phi(l-1_x); the bra derivative is
//! assembled from raised/lowered shell blocks, the ket derivative is the
//! transpose of the bra derivative of the swapped pair, and the
//! nuclear-attraction operator-center term follows from translational
//! invariance (sum of the three center derivatives of one attraction
//! block is zero).

use super::deriv::{
    combine_bra_derivative, kinetic_kind, lowered, nuclear_block, overlap_kind, pair_block, raised,
};
use super::eri::eri_quartet_masked;
use super::kernels::CoulombKernel;
use crate::basis::{aos, cart_components, n_sph, MolBasis, MolShell};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use rayon::prelude::*;

/// One derivative direction: 3 matrices (x, y, z) for one atom.
pub type AtomDerivs = [Matrix; 3];

fn zeros3(n: usize) -> AtomDerivs {
    [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ]
}

/// Scatter a spherical bra-derivative block into per-atom matrices:
/// the bra side belongs to `atom_a`; by symmetry of the underlying
/// operator the ket derivative of (mu,nu) is the bra derivative of
/// (nu,mu), so adding the transpose into the ket atom's matrix covers it.
fn scatter_pair(
    out: &mut [AtomDerivs],
    atom_a: usize,
    sph: &Matrix,
    oa: usize,
    ob: usize,
    ax: usize,
) {
    let m = &mut out[atom_a][ax];
    for i in 0..sph.rows {
        for j in 0..sph.cols {
            m[(oa + i, ob + j)] += sph[(i, j)];
            m[(ob + j, oa + i)] += sph[(i, j)];
        }
    }
}

/// Generic driver for symmetric pair operators (S, T): returns per-atom
/// [d/dx, d/dy, d/dz] matrices.
fn pair_operator_matrices<F>(mb: &MolBasis, natoms: usize, kind: F) -> Vec<AtomDerivs>
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
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    // Ordered pairs, bra derivative only; scatter_pair adds the transpose.
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Vec<AtomDerivs>> = pairs
        .par_chunks(std::cmp::max(
            1,
            pairs.len() / (8 * rayon::current_num_threads()),
        ))
        .map(|chunk| {
            let mut out: Vec<AtomDerivs> = (0..natoms).map(|_| zeros3(n)).collect();
            for &(a, b) in chunk {
                let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
                if sh_a.atom == sh_b.atom {
                    continue; // same-center: bra+ket cancel for S/T
                }
                let raise_blk = pair_block(&raised(sh_a), sh_b, kind);
                let lower_blk = lowered(sh_a).map(|lo| pair_block(&lo, sh_b, kind));
                for ax in 0..3 {
                    let cart =
                        combine_bra_derivative(sh_a.l, sh_b.l, ax, &raise_blk, lower_blk.as_ref());
                    let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                    // Each ordered pair contributes exactly its bra-side
                    // derivative; the transpose in scatter_pair supplies the
                    // ket-side element of the swapped entry (operator
                    // symmetry), so the ordered loop covers everything once.
                    scatter_pair(&mut out, sh_a.atom, &sph, sh_a.sph_off, sh_b.sph_off, ax);
                }
            }
            out
        })
        .collect();
    reduce(partials, natoms, n)
}

fn reduce(partials: Vec<Vec<AtomDerivs>>, natoms: usize, n: usize) -> Vec<AtomDerivs> {
    let mut total: Vec<AtomDerivs> = (0..natoms).map(|_| zeros3(n)).collect();
    for p in partials {
        for (t, a) in total.iter_mut().zip(p) {
            for ax in 0..3 {
                for (tv, av) in t[ax].as_mut_slice().iter_mut().zip(a[ax].as_slice()) {
                    *tv += *av;
                }
            }
        }
    }
    total
}

/// Full overlap-derivative matrices S^(a,x).
pub fn overlap_derivative_matrices(mb: &MolBasis, natoms: usize) -> Vec<AtomDerivs> {
    pair_operator_matrices(mb, natoms, overlap_kind)
}

/// Full kinetic-derivative matrices T^(a,x).
pub fn kinetic_derivative_matrices(mb: &MolBasis, natoms: usize) -> Vec<AtomDerivs> {
    pair_operator_matrices(mb, natoms, kinetic_kind)
}

/// Full nuclear-attraction derivative matrices V^(a,x) (basis-center and
/// operator-center/Hellmann-Feynman terms; `charges` supports Z_eff).
/// Derivative matrices of the one-electron operator of arbitrary CHARGE
/// SITES (see `deriv2::charge_site_second_contraction` for the same
/// generalization at second order): `block(bra, ket, site)` gives the
/// Cartesian block and `owner_of(site)` the atom the site rides on.
pub fn charge_site_derivative_matrices<F, G>(
    mb: &MolBasis,
    natoms: usize,
    n_sites: usize,
    owner_of: G,
    block: F,
) -> Vec<AtomDerivs>
where
    F: Fn(&MolShell, &MolShell, usize) -> Matrix + Sync,
    G: Fn(usize) -> usize + Sync,
{
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Vec<AtomDerivs>> = pairs
        .par_chunks(std::cmp::max(
            1,
            pairs.len() / (8 * rayon::current_num_threads()),
        ))
        .map(|chunk| {
            let mut out: Vec<AtomDerivs> = (0..natoms).map(|_| zeros3(n)).collect();
            for &(a, b) in chunk {
                let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
                for site in 0..n_sites {
                    let c = owner_of(site);
                    // Bra derivative of THIS site's attraction block.
                    let raise_blk = block(&raised(sh_a), sh_b, site);
                    let lower_blk = lowered(sh_a).map(|lo| block(&lo, sh_b, site));
                    for ax in 0..3 {
                        let cart = combine_bra_derivative(
                            sh_a.l,
                            sh_b.l,
                            ax,
                            &raise_blk,
                            lower_blk.as_ref(),
                        );
                        let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                        // Bra-center term.
                        scatter_pair(&mut out, sh_a.atom, &sph, sh_a.sph_off, sh_b.sph_off, ax);
                        // Operator-center term: d/dC = -(d/dA + d/dB); the
                        // d/dB half of this pair appears as the d/dA half of
                        // the swapped pair, so subtracting the bra half here
                        // (with its transpose) accounts for the full -d/dA-d/dB
                        // over the ordered loop.
                        let mut neg = sph;
                        for v in neg.as_mut_slice() {
                            *v = -*v;
                        }
                        scatter_pair(&mut out, c, &neg, sh_a.sph_off, sh_b.sph_off, ax);
                    }
                }
            }
            out
        })
        .collect();
    reduce(partials, natoms, n)
}

/// Derivative matrices of a symmetric pair operator whose OPERATOR does not
/// move with the nuclei — only the two basis centres do.
///
/// This is [`charge_site_derivative_matrices`] without the operator-centre
/// term, and it is not the same thing as [`overlap_derivative_matrices`]:
/// that one skips same-centre pairs because translating both Gaussians
/// together leaves `S` and `T` unchanged. For an operator that is not
/// translationally invariant — a plane wave `exp(i G . r)`, say — the bra
/// and ket derivatives do NOT cancel on a shared centre, so every pair has
/// to be visited.
pub fn moving_center_derivative_matrices<F>(
    mb: &MolBasis,
    natoms: usize,
    block: F,
) -> Vec<AtomDerivs>
where
    F: Fn(&MolShell, &MolShell) -> Matrix + Sync,
{
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Vec<AtomDerivs>> = pairs
        .par_chunks(std::cmp::max(
            1,
            pairs.len() / (8 * rayon::current_num_threads()),
        ))
        .map(|chunk| {
            let mut out: Vec<AtomDerivs> = (0..natoms).map(|_| zeros3(n)).collect();
            for &(a, b) in chunk {
                let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
                let raise_blk = block(&raised(sh_a), sh_b);
                let lower_blk = lowered(sh_a).map(|lo| block(&lo, sh_b));
                for ax in 0..3 {
                    let cart =
                        combine_bra_derivative(sh_a.l, sh_b.l, ax, &raise_blk, lower_blk.as_ref());
                    let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                    scatter_pair(&mut out, sh_a.atom, &sph, sh_a.sph_off, sh_b.sph_off, ax);
                }
            }
            out
        })
        .collect();
    reduce(partials, natoms, n)
}

/// Point-nucleus specialization of `charge_site_derivative_matrices`.
pub fn nuclear_derivative_matrices(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
) -> Vec<AtomDerivs> {
    let natoms = mol.len();
    let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
    charge_site_derivative_matrices(
        mb,
        natoms,
        natoms,
        |s| s,
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

/// The 8-fold ERI permutation group as slot maps: the ordered quartet
/// `T[k] = S[SIGMA[g][k]]` runs over every distinct permutation of a
/// canonical quartet `S`. Generated by bra swap (01), ket swap (23) and
/// the bra-ket exchange (02)(13).
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

/// Per-quartet derivative blocks `d[pos][ax][element]` in the shell order
/// given (`pos` 3 comes from translational invariance). `None` for
/// one-center quartets, whose derivatives cancel identically.
fn quartet_derivative_blocks(
    sh: [&MolShell; 4],
    kernel: CoulombKernel,
) -> Option<Vec<Vec<Vec<f64>>>> {
    let atoms = [sh[0].atom, sh[1].atom, sh[2].atom, sh[3].atom];
    if atoms.iter().all(|&x| x == atoms[0]) {
        return None;
    }
    let dims = [
        n_sph(sh[0].l),
        n_sph(sh[1].l),
        n_sph(sh[2].l),
        n_sph(sh[3].l),
    ];
    let nel = dims[0] * dims[1] * dims[2] * dims[3];
    let mut d_store = vec![vec![vec![0.0; nel]; 3]; 4];
    for pos in 0..3 {
        let mut mask = [true; 4];
        mask[pos] = false;
        let ra = raised(sh[pos]);
        let lo = lowered(sh[pos]);
        let q_raise = {
            let mut ms: [&MolShell; 4] = [sh[0], sh[1], sh[2], sh[3]];
            ms[pos] = &ra;
            eri_quartet_masked(ms[0], ms[1], ms[2], ms[3], kernel, mask)
        };
        let q_lower = lo.as_ref().map(|l| {
            let mut ms: [&MolShell; 4] = [sh[0], sh[1], sh[2], sh[3]];
            ms[pos] = l;
            eri_quartet_masked(ms[0], ms[1], ms[2], ms[3], kernel, mask)
        });
        let l = sh[pos].l;
        let comps_l = cart_components(l);
        let c2s = aos::cart_to_sph(l);
        let mut el = 0usize;
        for i0 in 0..dims[0] {
            for i1 in 0..dims[1] {
                for i2 in 0..dims[2] {
                    for i3 in 0..dims[3] {
                        let s_idx = [i0, i1, i2, i3];
                        let s_pos = s_idx[pos];
                        for ax in 0..3 {
                            let mut d_sph = 0.0;
                            for (ic, &(cx, cy, cz)) in comps_l.iter().enumerate() {
                                let coef = c2s[(ic, s_pos)];
                                if coef == 0.0 {
                                    continue;
                                }
                                let up = match ax {
                                    0 => (cx + 1, cy, cz),
                                    1 => (cx, cy + 1, cz),
                                    _ => (cx, cy, cz + 1),
                                };
                                let lcomp = [cx, cy, cz][ax];
                                let iu = comp_index(l + 1, up);
                                let mut idx = s_idx;
                                idx[pos] = iu;
                                let mut dv = q_raise.get(idx[0], idx[1], idx[2], idx[3]);
                                if lcomp > 0 {
                                    let dn = match ax {
                                        0 => (cx - 1, cy, cz),
                                        1 => (cx, cy - 1, cz),
                                        _ => (cx, cy, cz - 1),
                                    };
                                    let id = comp_index(l - 1, dn);
                                    let mut idx2 = s_idx;
                                    idx2[pos] = id;
                                    dv -= lcomp as f64
                                        * q_lower
                                            .as_ref()
                                            .expect("lower quartet")
                                            .get(idx2[0], idx2[1], idx2[2], idx2[3]);
                                }
                                d_sph += coef * dv;
                            }
                            d_store[pos][ax][el] = d_sph;
                        }
                        el += 1;
                    }
                }
            }
        }
    }
    for ax in 0..3 {
        for el in 0..nel {
            d_store[3][ax][el] = -(d_store[0][ax][el] + d_store[1][ax][el] + d_store[2][ax][el]);
        }
    }
    Some(d_store)
}

/// Derivative Coulomb/exchange matrices for a fixed density:
/// `J^(A,x)[P]_mn = sum_ls d(mn|ls)/dA_x P_ls`,
/// `K^(A,x)[Ps]_ml = sum_ns d(mn|ls)/dA_x Ps_ns` (per spin).
///
/// Canonical quartets (a<=b, c<=d, (a,b)<=(c,d)) with an explicit
/// permutation scatter: the derivative blocks are computed once per unique
/// quartet and written into every DISTINCT ordered permutation, which
/// reproduces the full ordered loop element by element while evaluating up
/// to 8x fewer integral blocks. The slower ordered reference lives in the
/// test module and gates this against it. The fourth center's derivative
/// comes from translational invariance per quartet.
#[allow(clippy::too_many_arguments)]
pub fn eri_derivative_fock(
    mb: &MolBasis,
    natoms: usize,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    kernel: CoulombKernel,
    schwarz: &Matrix,
    threshold: f64,
) -> (Vec<AtomDerivs>, [Vec<AtomDerivs>; 2]) {
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    let pmax = p_tot
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1e-30);
    // Canonical quartets. Screening is permutation invariant (the Schwarz
    // matrix is symmetric), so filtering here matches the ordered loop.
    let mut quartets = Vec::new();
    for a in 0..ns {
        for b in a..ns {
            for c in a..ns {
                let d0 = if c == a { b } else { c };
                for d in d0..ns {
                    if schwarz[(a, b)] * schwarz[(c, d)] * pmax * pmax >= threshold {
                        quartets.push((a, b, c, d));
                    }
                }
            }
        }
    }
    let partials: Vec<(Vec<AtomDerivs>, [Vec<AtomDerivs>; 2])> = quartets
        .par_chunks(std::cmp::max(
            1,
            quartets.len() / (8 * rayon::current_num_threads()),
        ))
        .map(|chunk| {
            let mut ja: Vec<AtomDerivs> = (0..natoms).map(|_| zeros3(n)).collect();
            let mut ka: [Vec<AtomDerivs>; 2] = [
                (0..natoms).map(|_| zeros3(n)).collect(),
                (0..natoms).map(|_| zeros3(n)).collect(),
            ];
            for &(a, b, c, d) in chunk {
                let quad = [a, b, c, d];
                let sh = [&mb.shells[a], &mb.shells[b], &mb.shells[c], &mb.shells[d]];
                let d_store = match quartet_derivative_blocks(sh, kernel) {
                    Some(v) => v,
                    None => continue, // one-center quartet: derivatives cancel
                };
                let dims = [
                    n_sph(sh[0].l),
                    n_sph(sh[1].l),
                    n_sph(sh[2].l),
                    n_sph(sh[3].l),
                ];
                // Visit each DISTINCT ordered permutation exactly once.
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
                    let tdims = [
                        n_sph(tsh[0].l),
                        n_sph(tsh[1].l),
                        n_sph(tsh[2].l),
                        n_sph(tsh[3].l),
                    ];
                    let tatoms = [tsh[0].atom, tsh[1].atom, tsh[2].atom, tsh[3].atom];
                    // Element strides of the CANONICAL storage order.
                    let stride = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
                    for pos_t in 0..4 {
                        let atom = tatoms[pos_t];
                        // T-slot pos_t carries the shell of canonical slot
                        // sig[pos_t], so its derivative block is d[sig[pos_t]].
                        let dvals_ax = &d_store[sig[pos_t]];
                        for ax in 0..3 {
                            let dvals = &dvals_ax[ax];
                            for j0 in 0..tdims[0] {
                                let mu = toffs[0] + j0;
                                for j1 in 0..tdims[1] {
                                    let nu = toffs[1] + j1;
                                    for j2 in 0..tdims[2] {
                                        let lam = toffs[2] + j2;
                                        for j3 in 0..tdims[3] {
                                            let sig_ao = toffs[3] + j3;
                                            // i[sig[k]] = j[k] maps the T
                                            // element onto canonical storage.
                                            let j = [j0, j1, j2, j3];
                                            let mut el = 0usize;
                                            for k in 0..4 {
                                                el += j[k] * stride[sig[k]];
                                            }
                                            let dv = dvals[el];
                                            if dv == 0.0 {
                                                continue;
                                            }
                                            ja[atom][ax][(mu, nu)] += dv * p_tot[(lam, sig_ao)];
                                            for s in 0..2 {
                                                ka[s][atom][ax][(mu, lam)] +=
                                                    dv * p_spin[s][(nu, sig_ao)];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (ja, ka)
        })
        .collect();
    let mut jt: Vec<AtomDerivs> = (0..natoms).map(|_| zeros3(n)).collect();
    let mut kt: [Vec<AtomDerivs>; 2] = [
        (0..natoms).map(|_| zeros3(n)).collect(),
        (0..natoms).map(|_| zeros3(n)).collect(),
    ];
    for (ja, ka) in partials {
        for a in 0..natoms {
            for ax in 0..3 {
                for (t, v) in jt[a][ax]
                    .as_mut_slice()
                    .iter_mut()
                    .zip(ja[a][ax].as_slice())
                {
                    *t += *v;
                }
                for s in 0..2 {
                    for (t, v) in kt[s][a][ax]
                        .as_mut_slice()
                        .iter_mut()
                        .zip(ka[s][a][ax].as_slice())
                    {
                        *t += *v;
                    }
                }
            }
        }
    }
    (jt, kt)
}

/// Slow reference: the original full ordered quartet loop. Kept as the
/// permanent cross-check of the canonical-quartet production path
/// (project testing policy: a naive implementation stays in the tree).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn eri_derivative_fock_ordered(
    mb: &MolBasis,
    natoms: usize,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    kernel: CoulombKernel,
    schwarz: &Matrix,
    threshold: f64,
) -> (Vec<AtomDerivs>, [Vec<AtomDerivs>; 2]) {
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    let pmax = p_tot
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1e-30);
    let mut jt: Vec<AtomDerivs> = (0..natoms).map(|_| zeros3(n)).collect();
    let mut kt: [Vec<AtomDerivs>; 2] = [
        (0..natoms).map(|_| zeros3(n)).collect(),
        (0..natoms).map(|_| zeros3(n)).collect(),
    ];
    for a in 0..ns {
        for b in 0..ns {
            for c in 0..ns {
                for d in 0..ns {
                    if schwarz[(a, b)] * schwarz[(c, d)] * pmax * pmax < threshold {
                        continue;
                    }
                    let sh = [&mb.shells[a], &mb.shells[b], &mb.shells[c], &mb.shells[d]];
                    let d_store = match quartet_derivative_blocks(sh, kernel) {
                        Some(v) => v,
                        None => continue,
                    };
                    let dims = [
                        n_sph(sh[0].l),
                        n_sph(sh[1].l),
                        n_sph(sh[2].l),
                        n_sph(sh[3].l),
                    ];
                    let offs = [sh[0].sph_off, sh[1].sph_off, sh[2].sph_off, sh[3].sph_off];
                    let atoms = [sh[0].atom, sh[1].atom, sh[2].atom, sh[3].atom];
                    for pos in 0..4 {
                        let atom = atoms[pos];
                        for ax in 0..3 {
                            let dvals = &d_store[pos][ax];
                            let mut el = 0usize;
                            for i0 in 0..dims[0] {
                                let mu = offs[0] + i0;
                                for i1 in 0..dims[1] {
                                    let nu = offs[1] + i1;
                                    for i2 in 0..dims[2] {
                                        let lam = offs[2] + i2;
                                        for i3 in 0..dims[3] {
                                            let sig = offs[3] + i3;
                                            let dv = dvals[el];
                                            el += 1;
                                            if dv != 0.0 {
                                                jt[atom][ax][(mu, nu)] += dv * p_tot[(lam, sig)];
                                                for s in 0..2 {
                                                    kt[s][atom][ax][(mu, lam)] +=
                                                        dv * p_spin[s][(nu, sig)];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (jt, kt)
}

/// Cartesian component index (panics on invalid = programming error).
fn comp_index(l: u8, c: (i32, i32, i32)) -> usize {
    cart_components(l)
        .iter()
        .position(|&x| x == c)
        .expect("cartesian component exists")
}

/// Dipole pair block `<a| r_comp |b>` (origin at 0) for arbitrary shells.
pub(crate) fn dipole_pair_block(sh_a: &MolShell, sh_b: &MolShell, comp: usize) -> Matrix {
    use super::hermite::ETable;
    use super::one_electron::{axis_moment, moment_table};
    let comps_a = cart_components(sh_a.l);
    let comps_b = cart_components(sh_b.l);
    let q = sh_a.center - sh_b.center;
    let mut blk = Matrix::zeros(comps_a.len(), comps_b.len());
    for (&ea, &ca) in sh_a.exps.iter().zip(&sh_a.coefs) {
        for (&eb, &cb) in sh_b.exps.iter().zip(&sh_b.coefs) {
            let p = ea + eb;
            let pcent = (sh_a.center * ea + sh_b.center * eb) / p;
            let ex = ETable::build(sh_a.l as usize, sh_b.l as usize, q.x, ea, eb);
            let ey = ETable::build(sh_a.l as usize, sh_b.l as usize, q.y, ea, eb);
            let ez = ETable::build(sh_a.l as usize, sh_b.l as usize, q.z, ea, eb);
            let m = [
                moment_table(1, p, pcent.x),
                moment_table(1, p, pcent.y),
                moment_table(1, p, pcent.z),
            ];
            let w = ca * cb;
            for (i, &(ax, ay, az)) in comps_a.iter().enumerate() {
                for (j, &(bx, by, bz)) in comps_b.iter().enumerate() {
                    let ords = [
                        if comp == 0 { 1 } else { 0 },
                        if comp == 1 { 1 } else { 0 },
                        if comp == 2 { 1 } else { 0 },
                    ];
                    let fx = axis_moment(&ex, &m[0], ax, bx, ords[0]);
                    let fy = axis_moment(&ey, &m[1], ay, by, ords[1]);
                    let fz = axis_moment(&ez, &m[2], az, bz, ords[2]);
                    blk[(i, j)] += w * fx * fy * fz;
                }
            }
        }
    }
    blk
}

/// Full dipole-integral derivative matrices `d<mu| r_c |nu>/dR_(a,x)`,
/// indexed `[dipole component][atom][axis]`. The operator origin is fixed
/// (no operator-center term); same-atom pairs do NOT cancel here.
pub fn dipole_derivative_matrices(mb: &MolBasis, natoms: usize) -> [Vec<AtomDerivs>; 3] {
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let mut out: [Vec<AtomDerivs>; 3] = [
        (0..natoms).map(|_| zeros3(n)).collect(),
        (0..natoms).map(|_| zeros3(n)).collect(),
        (0..natoms).map(|_| zeros3(n)).collect(),
    ];
    for comp in 0..3 {
        let partials: Vec<Vec<AtomDerivs>> = pairs
            .par_chunks(std::cmp::max(
                1,
                pairs.len() / (8 * rayon::current_num_threads()),
            ))
            .map(|chunk| {
                let mut acc: Vec<AtomDerivs> = (0..natoms).map(|_| zeros3(n)).collect();
                for &(a, b) in chunk {
                    let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
                    let raise_blk = dipole_pair_block(&raised(sh_a), sh_b, comp);
                    let lower_blk = lowered(sh_a).map(|lo| dipole_pair_block(&lo, sh_b, comp));
                    for ax in 0..3 {
                        let cart = combine_bra_derivative(
                            sh_a.l,
                            sh_b.l,
                            ax,
                            &raise_blk,
                            lower_blk.as_ref(),
                        );
                        let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                        scatter_pair(&mut acc, sh_a.atom, &sph, sh_a.sph_off, sh_b.sph_off, ax);
                    }
                }
                acc
            })
            .collect();
        out[comp] = reduce(partials, natoms, n);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{data, MolBasis};
    use crate::integrals::{kinetic_matrix, nuclear_matrix, overlap_matrix};
    use crate::system::{Atom, Molecule};

    fn mol() -> Molecule {
        Molecule::new(vec![
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
                position: Vec3::new(0.1, -1.4309, -0.8867),
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

    /// The canonical-quartet derivative Fock must reproduce the slow
    /// ordered-loop reference element for element (permutation scatter
    /// equivalence — the whole point of the optimization).
    #[test]
    fn canonical_derivative_fock_matches_ordered_reference() {
        use crate::integrals::eri::schwarz_matrix;
        let m = mol();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&m, &set).unwrap();
        let n = mb.nao_sph;
        // Deterministic asymmetric-in-spin densities (no symmetry luck).
        let mut p_a = Matrix::zeros(n, n);
        let mut p_b = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                let s = ((mu * 7 + nu * 3) % 11) as f64 / 11.0 - 0.5;
                let t = ((mu * 5 + nu * 13) % 9) as f64 / 9.0 - 0.5;
                p_a[(mu, nu)] = 0.3 * s;
                p_b[(mu, nu)] = 0.2 * t;
            }
        }
        for mu in 0..n {
            for nu in 0..mu {
                let sa = 0.5 * (p_a[(mu, nu)] + p_a[(nu, mu)]);
                p_a[(mu, nu)] = sa;
                p_a[(nu, mu)] = sa;
                let sb = 0.5 * (p_b[(mu, nu)] + p_b[(nu, mu)]);
                p_b[(mu, nu)] = sb;
                p_b[(nu, mu)] = sb;
            }
        }
        let mut p_tot = p_a.clone();
        for (t, v) in p_tot.as_mut_slice().iter_mut().zip(p_b.as_slice()) {
            *t += *v;
        }
        let kernel = CoulombKernel::Coulomb;
        let schwarz = schwarz_matrix(&mb, kernel);
        let t0 = std::time::Instant::now();
        let (j_new, k_new) =
            eri_derivative_fock(&mb, m.len(), &p_tot, [&p_a, &p_b], kernel, &schwarz, 1e-14);
        let t_canon = t0.elapsed();
        let t1 = std::time::Instant::now();
        let (j_ref, k_ref) = eri_derivative_fock_ordered(
            &mb,
            m.len(),
            &p_tot,
            [&p_a, &p_b],
            kernel,
            &schwarz,
            1e-14,
        );
        let t_ordered = t1.elapsed();
        // Visible with `cargo test -- --nocapture`; the reference is serial
        // so this is an integral-count indication, not a wall-clock claim.
        eprintln!(
            "derivative Fock: canonical {:?} vs ordered reference {:?}",
            t_canon, t_ordered
        );
        let mut worst = 0.0f64;
        for a in 0..m.len() {
            for ax in 0..3 {
                for (x, y) in j_new[a][ax].as_slice().iter().zip(j_ref[a][ax].as_slice()) {
                    worst = worst.max((x - y).abs());
                }
                for s in 0..2 {
                    for (x, y) in k_new[s][a][ax]
                        .as_slice()
                        .iter()
                        .zip(k_ref[s][a][ax].as_slice())
                    {
                        worst = worst.max((x - y).abs());
                    }
                }
            }
        }
        assert!(
            worst < 1e-12,
            "canonical vs ordered derivative Fock: worst |Δ| = {worst:.3e}"
        );
    }

    /// S^(a,x), T^(a,x), V^(a,x) against central differences of the full
    /// matrices (every element, every atom, every direction).
    #[test]
    fn derivative_matrices_match_finite_differences() {
        let m = mol();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&m, &set).unwrap();
        let charges: Vec<f64> = m.atoms.iter().map(|a| a.z as f64).collect();
        let ds = overlap_derivative_matrices(&mb, m.len());
        let dt = kinetic_derivative_matrices(&mb, m.len());
        let dv = nuclear_derivative_matrices(&mb, &m, &charges);
        let h = 1e-5;
        for a in 0..m.len() {
            for ax in 0..3 {
                let mp = bump(&m, a, ax, h);
                let mm = bump(&m, a, ax, -h);
                let mbp = MolBasis::build(&mp, &set).unwrap();
                let mbm = MolBasis::build(&mm, &set).unwrap();
                let refs = [
                    (&ds[a][ax], overlap_matrix(&mbp), overlap_matrix(&mbm), "S"),
                    (&dt[a][ax], kinetic_matrix(&mbp), kinetic_matrix(&mbm), "T"),
                    (
                        &dv[a][ax],
                        nuclear_matrix(&mbp, &mp, None),
                        nuclear_matrix(&mbm, &mm, None),
                        "V",
                    ),
                ];
                for (an, fp, fm, tag) in refs {
                    for i in 0..mb.nao_sph {
                        for j in 0..mb.nao_sph {
                            let fd = (fp[(i, j)] - fm[(i, j)]) / (2.0 * h);
                            assert!(
                                (an[(i, j)] - fd).abs() < 5e-9,
                                "{tag} atom {a} ax {ax} ({i},{j}): {} vs FD {fd}",
                                an[(i, j)]
                            );
                        }
                    }
                }
            }
        }
    }

    /// Dipole-integral derivative matrices vs central differences.
    #[test]
    fn dipole_derivative_matrices_match_finite_differences() {
        use crate::integrals::dipole_matrices;
        let m = mol();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&m, &set).unwrap();
        let dd = dipole_derivative_matrices(&mb, m.len());
        let h = 1e-5;
        for a in 0..m.len() {
            for ax in 0..3 {
                let mbp = MolBasis::build(&bump(&m, a, ax, h), &set).unwrap();
                let mbm = MolBasis::build(&bump(&m, a, ax, -h), &set).unwrap();
                let dp = dipole_matrices(&mbp, Vec3::zero());
                let dm = dipole_matrices(&mbm, Vec3::zero());
                for comp in 0..3 {
                    for i in 0..mb.nao_sph {
                        for j in 0..mb.nao_sph {
                            let fd = (dp[comp][(i, j)] - dm[comp][(i, j)]) / (2.0 * h);
                            let an = dd[comp][a][ax][(i, j)];
                            assert!(
                                (an - fd).abs() < 5e-9,
                                "D{comp} atom {a} ax {ax} ({i},{j}): {an} vs FD {fd}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// J^(a,x)[P] and K^(a,x)[P] against central differences of the full
    /// in-core J/K builds at displaced geometries (fixed density).
    #[test]
    fn derivative_fock_matrices_match_finite_differences() {
        use crate::integrals::{eri_incore, schwarz_matrix};
        let m = mol();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&m, &set).unwrap();
        let n = mb.nao_sph;
        // Deterministic symmetric density stand-in.
        let mut p = Matrix::zeros(n, n);
        let mut state = 0x1357_9bdf_u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for i in 0..n {
            for j in i..n {
                let v = 0.4 * rng();
                p[(i, j)] = v;
                p[(j, i)] = v;
            }
        }
        let schwarz = schwarz_matrix(&mb, CoulombKernel::Coulomb);
        let (dj, dk) = eri_derivative_fock(
            &mb,
            m.len(),
            &p,
            [&p, &p],
            CoulombKernel::Coulomb,
            &schwarz,
            1e-16,
        );
        let jk_at = |mm: &Molecule| -> (Matrix, Matrix) {
            let mbx = MolBasis::build(mm, &set).unwrap();
            let eri = eri_incore(&mbx, CoulombKernel::Coulomb, 256).unwrap();
            let mut j = Matrix::zeros(n, n);
            let mut k = Matrix::zeros(n, n);
            for mu in 0..n {
                for nu in 0..n {
                    for lam in 0..n {
                        for sig in 0..n {
                            j[(mu, nu)] += eri.get(mu, nu, lam, sig) * p[(lam, sig)];
                            k[(mu, lam)] += eri.get(mu, nu, lam, sig) * p[(nu, sig)];
                        }
                    }
                }
            }
            (j, k)
        };
        let h = 1e-5;
        for a in 0..m.len() {
            for ax in 0..3 {
                let (jp, kp) = jk_at(&bump(&m, a, ax, h));
                let (jm, km) = jk_at(&bump(&m, a, ax, -h));
                for i in 0..n {
                    for jcol in 0..n {
                        let fdj = (jp[(i, jcol)] - jm[(i, jcol)]) / (2.0 * h);
                        assert!(
                            (dj[a][ax][(i, jcol)] - fdj).abs() < 2e-7,
                            "J atom {a} ax {ax} ({i},{jcol}): {} vs FD {fdj}",
                            dj[a][ax][(i, jcol)]
                        );
                        let fdk = (kp[(i, jcol)] - km[(i, jcol)]) / (2.0 * h);
                        assert!(
                            (dk[0][a][ax][(i, jcol)] - fdk).abs() < 2e-7,
                            "K atom {a} ax {ax} ({i},{jcol}): {} vs FD {fdk}",
                            dk[0][a][ax][(i, jcol)]
                        );
                    }
                }
            }
        }
    }
}
