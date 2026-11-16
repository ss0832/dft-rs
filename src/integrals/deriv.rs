// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Center derivatives of the integral matrices, contracted immediately with
//! density-like matrices into per-atom gradient contributions.
//!
//! Everything reduces to the primitive identity
//! `∂/∂A_x φ(l) = 2a·φ(l+1_x) − l_x·φ(l−1_x)`: the raising term is an
//! ordinary integral over a "raised" shell whose contraction coefficients
//! absorb the per-primitive factor 2a, the lowering term an ordinary
//! integral over the l−1 shell — so every derivative reuses the untouched
//! MD integral routines. Operator-center terms (nuclear attraction) come
//! from translational invariance.

#![allow(clippy::items_after_test_module)]

use super::eri::Quartet;
use super::hermite::{ETable, RTensor};
use super::kernels::CoulombKernel;
use crate::basis::{aos, cart_components, n_cart, MolBasis, MolShell};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Shell with l+1 and per-primitive coefficients 2a·c (raising partner).
pub(crate) fn raised(sh: &MolShell) -> MolShell {
    MolShell {
        l: sh.l + 1,
        atom: sh.atom,
        center: sh.center,
        exps: sh.exps.clone(),
        coefs: sh
            .exps
            .iter()
            .zip(&sh.coefs)
            .map(|(&a, &c)| 2.0 * a * c)
            .collect(),
        sph_off: 0,
        cart_off: 0,
    }
}

/// Shell with l−1 (same coefficients); `None` for s shells.
pub(crate) fn lowered(sh: &MolShell) -> Option<MolShell> {
    if sh.l == 0 {
        return None;
    }
    Some(MolShell {
        l: sh.l - 1,
        atom: sh.atom,
        center: sh.center,
        exps: sh.exps.clone(),
        coefs: sh.coefs.clone(),
        sph_off: 0,
        cart_off: 0,
    })
}

/// Index of Cartesian component `(x,y,z)` within `cart_components(l)`,
/// or None when any exponent is negative.
fn comp_index(l: u8, c: (i32, i32, i32)) -> Option<usize> {
    if c.0 < 0 || c.1 < 0 || c.2 < 0 {
        return None;
    }
    cart_components(l).iter().position(|&x| x == c)
}

/// Assemble the Cartesian derivative block `∂/∂A_ax` of a generic pair
/// integral from raised/lowered blocks:
/// `d[i,j] = raise[i+1_ax, j] − l_ax(i)·lower[i−1_ax, j]`.
pub(crate) fn combine_bra_derivative(
    la: u8,
    lb: u8,
    ax: usize,
    raise_blk: &Matrix,         // (n_cart(la+1)) × n_cart(lb)
    lower_blk: Option<&Matrix>, // (n_cart(la−1)) × n_cart(lb)
) -> Matrix {
    let comps_a = cart_components(la);
    let ncb = n_cart(lb);
    let mut out = Matrix::zeros(comps_a.len(), ncb);
    for (i, &(cx, cy, cz)) in comps_a.iter().enumerate() {
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
        let iu = comp_index(la + 1, up).expect("raised component exists");
        for j in 0..ncb {
            let mut v = raise_blk[(iu, j)];
            if lcomp > 0 {
                let blk = lower_blk.expect("lowered block exists for l>0");
                let id = comp_index(la - 1, dn).expect("lowered component exists");
                v -= lcomp as f64 * blk[(id, j)];
            }
            out[(i, j)] = v;
        }
    }
    out
}

/// Generic Cartesian pair-block builder used for S and T derivative parts.
pub(crate) fn pair_block<F>(sh_a: &MolShell, sh_b: &MolShell, kind: F) -> Matrix
where
    F: Fn(&ETable, &ETable, &ETable, f64, f64, (i32, i32, i32), (i32, i32, i32)) -> f64,
{
    let comps_a = cart_components(sh_a.l);
    let comps_b = cart_components(sh_b.l);
    let q = sh_a.center - sh_b.center;
    let mut blk = Matrix::zeros(comps_a.len(), comps_b.len());
    for (&ea, &ca) in sh_a.exps.iter().zip(&sh_a.coefs) {
        for (&eb, &cb) in sh_b.exps.iter().zip(&sh_b.coefs) {
            let ex = ETable::build(sh_a.l as usize, sh_b.l as usize + 2, q.x, ea, eb);
            let ey = ETable::build(sh_a.l as usize, sh_b.l as usize + 2, q.y, ea, eb);
            let ez = ETable::build(sh_a.l as usize, sh_b.l as usize + 2, q.z, ea, eb);
            let w = ca * cb;
            for (i, &pa) in comps_a.iter().enumerate() {
                for (j, &pb) in comps_b.iter().enumerate() {
                    blk[(i, j)] += w * kind(&ex, &ey, &ez, ea, eb, pa, pb);
                }
            }
        }
    }
    blk
}

fn s1d(e: &ETable, la: i32, lb: i32, p: f64) -> f64 {
    if la < 0 || lb < 0 {
        return 0.0;
    }
    e.get(la as usize, lb as usize, 0) * (PI / p).sqrt()
}

pub(crate) fn overlap_kind(
    ex: &ETable,
    ey: &ETable,
    ez: &ETable,
    ea: f64,
    eb: f64,
    a: (i32, i32, i32),
    b: (i32, i32, i32),
) -> f64 {
    let p = ea + eb;
    s1d(ex, a.0, b.0, p) * s1d(ey, a.1, b.1, p) * s1d(ez, a.2, b.2, p)
}

pub(crate) fn kinetic_kind(
    ex: &ETable,
    ey: &ETable,
    ez: &ETable,
    ea: f64,
    eb: f64,
    a: (i32, i32, i32),
    b: (i32, i32, i32),
) -> f64 {
    let p = ea + eb;
    let t1 = |e: &ETable, la: i32, lb: i32| -> f64 {
        eb * (2 * lb + 1) as f64 * s1d(e, la, lb, p)
            - 2.0 * eb * eb * s1d(e, la, lb + 2, p)
            - 0.5 * (lb * (lb - 1)) as f64 * s1d(e, la, lb - 2, p)
    };
    t1(ex, a.0, b.0) * s1d(ey, a.1, b.1, p) * s1d(ez, a.2, b.2, p)
        + s1d(ex, a.0, b.0, p) * t1(ey, a.1, b.1) * s1d(ez, a.2, b.2, p)
        + s1d(ex, a.0, b.0, p) * s1d(ey, a.1, b.1, p) * t1(ez, a.2, b.2)
}

/// Nuclear-attraction Cartesian block of a shell pair for ONE center C with
/// charge z (positive → the −z/r attraction included).
pub(crate) fn nuclear_block(sh_a: &MolShell, sh_b: &MolShell, cc: Vec3, z: f64) -> Matrix {
    nuclear_block_kernel(sh_a, sh_b, cc, z, CoulombKernel::Coulomb)
}

/// [`nuclear_block`] with the `1/r` kernel replaced. The screened variants
/// are what a periodic lattice sum needs: only the `erfc` half of the Ewald
/// split converges in real space.
pub(crate) fn nuclear_block_kernel(
    sh_a: &MolShell,
    sh_b: &MolShell,
    cc: Vec3,
    z: f64,
    kernel: CoulombKernel,
) -> Matrix {
    let comps_a = cart_components(sh_a.l);
    let comps_b = cart_components(sh_b.l);
    let (la, lb) = (sh_a.l as usize, sh_b.l as usize);
    let ltot = la + lb;
    let q = sh_a.center - sh_b.center;
    let mut blk = Matrix::zeros(comps_a.len(), comps_b.len());
    let mut f = vec![0.0; ltot + 1];
    for (&ea, &ca) in sh_a.exps.iter().zip(&sh_a.coefs) {
        for (&eb, &cb) in sh_b.exps.iter().zip(&sh_b.coefs) {
            let p = ea + eb;
            let pcen = (sh_a.center * ea + sh_b.center * eb) / p;
            let ex = ETable::build(la, lb, q.x, ea, eb);
            let ey = ETable::build(la, lb, q.y, ea, eb);
            let ez = ETable::build(la, lb, q.z, ea, eb);
            let w = ca * cb * 2.0 * PI / p;
            let pc = pcen - cc;
            kernel.boys_into(ltot, p, p * pc.norm2(), &mut f);
            let r = RTensor::build(ltot, p, pc, &f);
            for (i, &(axx, ayy, azz)) in comps_a.iter().enumerate() {
                for (j, &(bxx, byy, bzz)) in comps_b.iter().enumerate() {
                    let mut sum = 0.0;
                    for t in 0..=(axx + bxx) as usize {
                        let e1 = ex.get(axx as usize, bxx as usize, t);
                        if e1 == 0.0 {
                            continue;
                        }
                        for u in 0..=(ayy + byy) as usize {
                            let e2 = ey.get(ayy as usize, byy as usize, u);
                            if e2 == 0.0 {
                                continue;
                            }
                            for v in 0..=(azz + bzz) as usize {
                                let e3 = ez.get(azz as usize, bzz as usize, v);
                                if e3 == 0.0 {
                                    continue;
                                }
                                sum += e1 * e2 * e3 * r.get(t, u, v);
                            }
                        }
                    }
                    blk[(i, j)] -= w * z * sum;
                }
            }
        }
    }
    blk
}

/// Contract a spherical derivative block with the matrix `m`'s
/// corresponding block.
fn contract(m: &Matrix, oa: usize, ob: usize, sph: &Matrix) -> f64 {
    let mut acc = 0.0;
    for i in 0..sph.rows {
        for j in 0..sph.cols {
            acc += m[(oa + i, ob + j)] * sph[(i, j)];
        }
    }
    acc
}

/// `out[A] += scale · Σ_μν m_μν ∂S_μν/∂R_A` (energy-weighted-density
/// contraction for the Pulay term uses `m = W`, `scale = −1`... the caller
/// chooses).
pub fn overlap_gradient(mb: &MolBasis, m: &Matrix, natoms: usize, scale: f64) -> Vec<Vec3> {
    pair_operator_gradient(mb, m, natoms, scale, |sh_a, sh_b| {
        (
            pair_block(&raised(sh_a), sh_b, overlap_kind),
            lowered(sh_a).map(|lo| pair_block(&lo, sh_b, overlap_kind)),
        )
    })
}

/// `out[A] += scale · Σ_μν m_μν ∂T_μν/∂R_A`.
pub fn kinetic_gradient(mb: &MolBasis, m: &Matrix, natoms: usize, scale: f64) -> Vec<Vec3> {
    pair_operator_gradient(mb, m, natoms, scale, |sh_a, sh_b| {
        (
            pair_block(&raised(sh_a), sh_b, kinetic_kind),
            lowered(sh_a).map(|lo| pair_block(&lo, sh_b, kinetic_kind)),
        )
    })
}

/// Shared driver for translation-invariant pair operators (S, T): the bra
/// derivative is computed explicitly, the ket derivative is its negative.
fn pair_operator_gradient<F>(
    mb: &MolBasis,
    m: &Matrix,
    natoms: usize,
    scale: f64,
    blocks: F,
) -> Vec<Vec3>
where
    F: Fn(&MolShell, &MolShell) -> (Matrix, Option<Matrix>) + Sync,
{
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Vec<Vec3>> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut out = vec![Vec3::zero(); natoms];
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            if sh_a.atom == sh_b.atom {
                return out; // same-center pair: derivative cancels
            }
            // blocks() convention: first = block over the RAISED bra shell
            // (coefs folded with 2a), second = block over the LOWERED bra
            // shell (original coefs).
            let (raise_blk, lower_blk) = blocks(sh_a, sh_b);
            for ax in 0..3 {
                let cart =
                    combine_bra_derivative(sh_a.l, sh_b.l, ax, &raise_blk, lower_blk.as_ref());
                let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                let val = scale * contract(m, sh_a.sph_off, sh_b.sph_off, &sph);
                let v = match ax {
                    0 => Vec3::new(val, 0.0, 0.0),
                    1 => Vec3::new(0.0, val, 0.0),
                    _ => Vec3::new(0.0, 0.0, val),
                };
                // Ordered elements each carry a bra-side AND a ket-side
                // derivative; with symmetric m the ket side of (μν) equals
                // the bra side of (νμ), so the ordered-pair loop over bra
                // sides captures exactly half — factor 2 restores the total.
                out[sh_a.atom] += v * 2.0;
            }
            out
        })
        .collect();
    let mut total = vec![Vec3::zero(); natoms];
    for p in partials {
        for (t, v) in total.iter_mut().zip(p) {
            *t += v;
        }
    }
    total
}

/// `out[A] += scale · Σ_μν m_μν ∂V_μν/∂R_A` including the operator-center
/// (Hellmann–Feynman) terms, with per-atom charges (ECP-ready).
pub fn nuclear_gradient(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    m: &Matrix,
    scale: f64,
) -> Vec<Vec3> {
    let natoms = mol.len();
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Vec<Vec3>> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut out = vec![Vec3::zero(); natoms];
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            let ra = raised(sh_a);
            let lo = lowered(sh_a);
            for (c_idx, atom) in mol.atoms.iter().enumerate() {
                let z = charges[c_idx];
                if z == 0.0 {
                    continue;
                }
                // Bra derivative for THIS operator center.
                let raise_blk = nuclear_block(&ra, sh_b, atom.position, z);
                let lower_blk = lo
                    .as_ref()
                    .map(|l| nuclear_block(l, sh_b, atom.position, z));
                for ax in 0..3 {
                    let cart =
                        combine_bra_derivative(sh_a.l, sh_b.l, ax, &raise_blk, lower_blk.as_ref());
                    let sph = aos::sph_block(sh_a.l, sh_b.l, &cart);
                    let val = scale * contract(m, sh_a.sph_off, sh_b.sph_off, &sph);
                    let v = match ax {
                        0 => Vec3::new(val, 0.0, 0.0),
                        1 => Vec3::new(0.0, val, 0.0),
                        _ => Vec3::new(0.0, 0.0, val),
                    };
                    // Factor 2: each ordered element carries bra + ket
                    // derivative sides (equal under symmetric m); the loop
                    // enumerates bra sides only. The operator center C
                    // receives −Σ(bra + ket) by translational invariance —
                    // the same factor.
                    out[sh_a.atom] += v * 2.0;
                    out[c_idx] -= v * 2.0;
                }
            }
            out
        })
        .collect();
    let mut total = vec![Vec3::zero(); natoms];
    for p in partials {
        for (t, v) in total.iter_mut().zip(p) {
            *t += v;
        }
    }
    total
}

/// Two-electron gradient: `out[A] += Σ (∂(μν|λσ)/∂R_A) · X_μν,λσ` where the
/// contraction weight is the Coulomb/exchange density combination
/// `X = ½ P_μν P_λσ − ¼·cx_eff Σ_s P^s_μλ P^s_νσ` — i.e. exactly the
/// derivative of `½Tr[P J] − ½ Σ_s cx_eff Tr[P^s K^s]`.
///
/// Screening uses the Schwarz matrix times the largest density magnitude.
#[allow(clippy::too_many_arguments)]
pub fn eri_gradient(
    mb: &MolBasis,
    natoms: usize,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    kernel: CoulombKernel,
    coul_coef: f64,
    exch_coef: f64,
    schwarz: &Matrix,
    threshold: f64,
) -> Vec<Vec3> {
    let channels = [(p_spin[0], exch_coef), (p_spin[1], exch_coef)];
    eri_gradient_channels(
        mb, natoms, p_tot, &channels, kernel, coul_coef, schwarz, threshold,
    )
}

/// Weighted real exchange channels of definite transpose parity. A complex
/// Hermitian spin density supplies symmetric real and antisymmetric imaginary
/// Pauli components. All channels share one set of derivative quartets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eri_gradient_channels(
    mb: &MolBasis,
    natoms: usize,
    p_tot: &Matrix,
    channels: &[(&Matrix, f64)],
    kernel: CoulombKernel,
    coul_coef: f64,
    schwarz: &Matrix,
    threshold: f64,
) -> Vec<Vec3> {
    let ns = mb.shells.len();
    let mut quartets = Vec::new();
    // Largest density element in each shell-pair block. Screening on the
    // GLOBAL maximum, as this used to, keeps every quartet whose integral
    // is large even when the density that multiplies it is negligible —
    // which for anything bigger than a few atoms is most of them. The
    // per-block bound is the one the contraction actually obeys: the
    // Coulomb part is weighted by P_cd (and P_ab), the exchange part by the
    // four crossed blocks.
    let shell_pmax = {
        let mut m = Matrix::zeros(ns, ns);
        for a in 0..ns {
            let (oa, na) = (mb.shells[a].sph_off, crate::basis::n_sph(mb.shells[a].l));
            for b in 0..ns {
                let (ob, nb) = (mb.shells[b].sph_off, crate::basis::n_sph(mb.shells[b].l));
                let mut v = 0.0_f64;
                for i in 0..na {
                    for j in 0..nb {
                        v = v.max(p_tot[(oa + i, ob + j)].abs());
                        for (p, _) in channels {
                            v = v.max(p[(oa + i, ob + j)].abs());
                        }
                    }
                }
                m[(a, b)] = v;
            }
        }
        m
    };
    for a in 0..ns {
        for b in a..ns {
            for c in a..ns {
                let d0 = if c == a { b } else { c };
                for d in d0..ns {
                    let dens = shell_pmax[(a, b)]
                        .max(shell_pmax[(c, d)])
                        .max(shell_pmax[(a, c)])
                        .max(shell_pmax[(a, d)])
                        .max(shell_pmax[(b, c)])
                        .max(shell_pmax[(b, d)]);
                    if schwarz[(a, b)] * schwarz[(c, d)] * dens < threshold {
                        continue;
                    }
                    quartets.push((a, b, c, d));
                }
            }
        }
    }
    // One task per quartet, NOT chunked. Chunking would save ~10^5 small
    // allocations, and was tried: benzene/6-31G went from 9.96 s to
    // 10.83 s. Quartet cost varies by orders of magnitude with the angular
    // momenta involved, so rayon's work stealing across many small tasks
    // beats fixed chunks by more than the allocations cost.
    let partials: Vec<Vec<Vec3>> = quartets
        .par_iter()
        .map(|&(a, b, c, d)| {
            let mut out = vec![Vec3::zero(); natoms];
            {
                // Degeneracy = number of distinct permutation images, in closed
                // form. The eight-fold group is generated by swapping within
                // the bra, within the ket, and bra with ket, so each coincidence
                // among (a,b), (c,d) and the two pairs halves the orbit. This
                // used to be counted by building a HashSet per quartet — one
                // heap allocation and eight hashes for every one of the ~10^5
                // quartets, for a number that is three comparisons.
                let deg = 8.0
                    / (if a == b { 2.0 } else { 1.0 }
                        * if c == d { 2.0 } else { 1.0 }
                        * if a == c && b == d { 2.0 } else { 1.0 });
                // Derivative quartets for the first three centers; the fourth
                // follows from translational invariance.
                let sh = [&mb.shells[a], &mb.shells[b], &mb.shells[c], &mb.shells[d]];
                let mut center_grad = [Vec3::zero(); 4];
                // All six raised/lowered quartets in one pass: they share the
                // exponents and centres, hence the Boys values and the R-tensor,
                // and their E-tables are sub-blocks of one build per side.
                let qs = crate::integrals::eri::eri_quartet_deriv_set(sh, kernel);
                for pos in 0..3 {
                    let q_raise = qs[2 * pos].as_ref().expect("raised quartet always exists");
                    let q_lower = qs[2 * pos + 1].as_ref();
                    for ax in 0..3 {
                        let val = contract_deriv_quartet(
                            sh, pos, ax, q_raise, q_lower, p_tot, channels, coul_coef,
                        );
                        let v = deg
                            * match ax {
                                0 => Vec3::new(val, 0.0, 0.0),
                                1 => Vec3::new(0.0, val, 0.0),
                                _ => Vec3::new(0.0, 0.0, val),
                            };
                        center_grad[pos] += v;
                    }
                }
                center_grad[3] = -(center_grad[0] + center_grad[1] + center_grad[2]);
                for pos in 0..4 {
                    out[sh[pos].atom] += center_grad[pos];
                }
            }
            out
        })
        .collect();
    let mut total = vec![Vec3::zero(); natoms];
    for p in partials {
        for (t, v) in total.iter_mut().zip(p) {
            *t += v;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{data, MolBasis};
    use crate::integrals::{
        eri_incore, kinetic_matrix, nuclear_matrix, overlap_matrix, schwarz_matrix,
    };
    use crate::system::Atom;

    fn test_mol(shift: Option<(usize, usize, f64)>) -> Molecule {
        let mut mol = Molecule::new(vec![
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
        ]);
        if let Some((atom, ax, h)) = shift {
            match ax {
                0 => mol.atoms[atom].position.x += h,
                1 => mol.atoms[atom].position.y += h,
                _ => mol.atoms[atom].position.z += h,
            }
        }
        mol
    }

    fn fixed_m(n: usize) -> Matrix {
        // Deterministic symmetric test matrix.
        let mut m = Matrix::zeros(n, n);
        let mut state = 0x2468_ace0_u64;
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

    /// Each contracted derivative vs FD of the corresponding fixed-matrix
    /// energy expression — isolates every gradient term.
    #[test]
    fn contracted_derivatives_match_fd() {
        let set = data::load("sto-3g").unwrap();
        let mol0 = test_mol(None);
        let mb0 = MolBasis::build(&mol0, &set).unwrap();
        let n = mb0.nao_sph;
        let m = fixed_m(n);
        let natoms = 3;
        let h = 1e-5;
        let charges: Vec<f64> = mol0.atoms.iter().map(|a| a.z as f64).collect();

        let gs = overlap_gradient(&mb0, &m, natoms, 1.0);
        let gt = kinetic_gradient(&mb0, &m, natoms, 1.0);
        let gv = nuclear_gradient(&mb0, &mol0, &charges, &m, 1.0);
        let schwarz = schwarz_matrix(&mb0, CoulombKernel::Coulomb);
        let g2 = eri_gradient(
            &mb0,
            natoms,
            &m,
            [&m, &m],
            CoulombKernel::Coulomb,
            1.0,
            0.7,
            &schwarz,
            1e-14,
        );

        for atom in 0..natoms {
            for ax in 0..3 {
                let e = |mol: &Molecule| -> (f64, f64, f64, f64) {
                    let mb = MolBasis::build(mol, &set).unwrap();
                    let s = overlap_matrix(&mb);
                    let t = kinetic_matrix(&mb);
                    let v = nuclear_matrix(&mb, mol, None);
                    let eri = eri_incore(&mb, CoulombKernel::Coulomb, 256).unwrap();
                    let mut e2 = 0.0;
                    for mu in 0..n {
                        for nu in 0..n {
                            for lam in 0..n {
                                for sig in 0..n {
                                    let x = 0.5 * m[(mu, nu)] * m[(lam, sig)]
                                        - 0.5
                                            * 0.7
                                            * (m[(mu, lam)] * m[(nu, sig)]
                                                + m[(mu, lam)] * m[(nu, sig)]);
                                    e2 += x * eri.get(mu, nu, lam, sig);
                                }
                            }
                        }
                    }
                    (
                        m.frobenius_dot(&s),
                        m.frobenius_dot(&t),
                        m.frobenius_dot(&v),
                        e2,
                    )
                };
                let (sp, tp, vp, e2p) = e(&test_mol(Some((atom, ax, h))));
                let (sm, tm, vm, e2m) = e(&test_mol(Some((atom, ax, -h))));
                let pick = |g: &Vec<Vec3>| g[atom].get(ax);
                let checks = [
                    ("S", (sp - sm) / (2.0 * h), pick(&gs)),
                    ("T", (tp - tm) / (2.0 * h), pick(&gt)),
                    ("V", (vp - vm) / (2.0 * h), pick(&gv)),
                    ("ERI", (e2p - e2m) / (2.0 * h), pick(&g2)),
                ];
                for (label, fd, an) in checks {
                    assert!(
                        (an - fd).abs() < 5e-6 * fd.abs().max(1.0),
                        "{label} atom {atom} ax {ax}: analytic {an} vs FD {fd}"
                    );
                }
            }
        }
    }
}

/// Contract one derivative quartet with the J/K density combination
/// `X(μνλσ) = ½·coul·P(μν)P(λσ) − ½·exch·Σ_s P^s(μλ)P^s(νσ)`. The
/// differentiated index arrives Cartesian (raised l+1 / lowered l−1); it is
/// recombined per component and transformed to spherical on the fly.
#[allow(clippy::too_many_arguments)]
fn contract_deriv_quartet(
    sh: [&MolShell; 4],
    pos: usize,
    ax: usize,
    q_raise: &Quartet,
    q_lower: Option<&Quartet>,
    p_tot: &Matrix,
    channels: &[(&Matrix, f64)],
    coul_coef: f64,
) -> f64 {
    let l = sh[pos].l;
    let comps_l = cart_components(l);
    let c2s = aos::cart_to_sph(l);
    let sph_dims = [
        crate::basis::n_sph(sh[0].l),
        crate::basis::n_sph(sh[1].l),
        crate::basis::n_sph(sh[2].l),
        crate::basis::n_sph(sh[3].l),
    ];
    let offs = [sh[0].sph_off, sh[1].sph_off, sh[2].sph_off, sh[3].sph_off];
    let get = |q: &Quartet, idx: [usize; 4]| q.get(idx[0], idx[1], idx[2], idx[3]);
    let mut val = 0.0;
    let dims = |p: usize| if p == pos { sph_dims[pos] } else { sph_dims[p] };
    for i0 in 0..dims(0) {
        for i1 in 0..dims(1) {
            for i2 in 0..dims(2) {
                for i3 in 0..dims(3) {
                    let s_idx = [i0, i1, i2, i3];
                    // Derivative element with the pos index spherical s:
                    // d_sph(s) = Σ_cart c2s[cart, s]·d_cart(cart).
                    let s_pos = s_idx[pos];
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
                        let iu = comp_index(l + 1, up).expect("raised comp");
                        let mut idx = s_idx;
                        idx[pos] = iu;
                        let mut d = get(q_raise, idx);
                        if lcomp > 0 {
                            let dn = match ax {
                                0 => (cx - 1, cy, cz),
                                1 => (cx, cy - 1, cz),
                                _ => (cx, cy, cz - 1),
                            };
                            let id = comp_index(l - 1, dn).expect("lowered comp");
                            let mut idx2 = s_idx;
                            idx2[pos] = id;
                            d -= lcomp as f64 * get(q_lower.expect("lower quartet"), idx2);
                        }
                        d_sph += coef * d;
                    }
                    if d_sph == 0.0 {
                        continue;
                    }
                    let (mu, nu, lam, sig) =
                        (offs[0] + i0, offs[1] + i1, offs[2] + i2, offs[3] + i3);
                    // The per-element exchange weight is NOT invariant under
                    // the individual index swaps, so the canonical-quartet ×
                    // degeneracy shortcut requires its σ-orbit-symmetrized
                    // form ½[Pˢ(μλ)Pˢ(νσ) + Pˢ(μσ)Pˢ(νλ)] (the summed energy
                    // is unchanged by the integral symmetries).
                    let mut x = 0.5 * coul_coef * p_tot[(mu, nu)] * p_tot[(lam, sig)];
                    for &(ps, exch_coef) in channels {
                        if exch_coef != 0.0 {
                            x -= 0.25
                                * exch_coef
                                * (ps[(mu, lam)] * ps[(nu, sig)] + ps[(mu, sig)] * ps[(nu, lam)]);
                        }
                    }
                    val += d_sph * x;
                }
            }
        }
    }
    val
}

/// One spherical shell-quartet derivative with respect to a Cartesian
/// component of one centre.
///
/// Periodic contractions do not have the eight-fold density symmetry used
/// by [`eri_gradient`], because each shell carries a distinct lattice
/// translation.  They therefore need the derivative tensor itself rather
/// than the molecular J/K contraction above.  The fourth-centre derivative
/// follows exactly from translational invariance.
pub(crate) fn eri_quartet_center_derivatives(
    sh: [&MolShell; 4],
    kernel: CoulombKernel,
) -> Vec<[Quartet; 3]> {
    let set = crate::integrals::eri::eri_quartet_deriv_set(sh, kernel);
    let sph_dims = [
        crate::basis::n_sph(sh[0].l),
        crate::basis::n_sph(sh[1].l),
        crate::basis::n_sph(sh[2].l),
        crate::basis::n_sph(sh[3].l),
    ];
    let empty = || Quartet {
        na: sph_dims[0],
        nb: sph_dims[1],
        nc: sph_dims[2],
        nd: sph_dims[3],
        data: vec![0.0; sph_dims.iter().product()],
    };
    let mut all: Vec<[Quartet; 3]> = (0..4).map(|_| std::array::from_fn(|_| empty())).collect();
    for center in 0..3 {
        let q_raise = set[2 * center]
            .as_ref()
            .expect("the raised derivative quartet always exists");
        let q_lower = set[2 * center + 1].as_ref();
        let l = sh[center].l;
        let components = cart_components(l);
        let c2s = aos::cart_to_sph(l);
        for axis in 0..3 {
            for i0 in 0..sph_dims[0] {
                for i1 in 0..sph_dims[1] {
                    for i2 in 0..sph_dims[2] {
                        for i3 in 0..sph_dims[3] {
                            let spherical = [i0, i1, i2, i3];
                            let mut value = 0.0;
                            for (cart_index, &(cx, cy, cz)) in components.iter().enumerate() {
                                let coefficient = c2s[(cart_index, spherical[center])];
                                if coefficient == 0.0 {
                                    continue;
                                }
                                let raised_component = match axis {
                                    0 => (cx + 1, cy, cz),
                                    1 => (cx, cy + 1, cz),
                                    _ => (cx, cy, cz + 1),
                                };
                                let mut raised_index = spherical;
                                raised_index[center] =
                                    comp_index(l + 1, raised_component).expect("raised component");
                                let mut derivative = q_raise.get(
                                    raised_index[0],
                                    raised_index[1],
                                    raised_index[2],
                                    raised_index[3],
                                );
                                let power = [cx, cy, cz][axis];
                                if power > 0 {
                                    let lowered_component = match axis {
                                        0 => (cx - 1, cy, cz),
                                        1 => (cx, cy - 1, cz),
                                        _ => (cx, cy, cz - 1),
                                    };
                                    let mut lowered_index = spherical;
                                    lowered_index[center] = comp_index(l - 1, lowered_component)
                                        .expect("lowered component");
                                    derivative -= power as f64
                                        * q_lower.expect("lowered derivative quartet").get(
                                            lowered_index[0],
                                            lowered_index[1],
                                            lowered_index[2],
                                            lowered_index[3],
                                        );
                                }
                                value += coefficient * derivative;
                            }
                            let flat =
                                ((i0 * sph_dims[1] + i1) * sph_dims[2] + i2) * sph_dims[3] + i3;
                            all[center][axis].data[flat] = value;
                        }
                    }
                }
            }
        }
    }
    for axis in 0..3 {
        for index in 0..all[3][axis].data.len() {
            all[3][axis].data[index] =
                -(all[0][axis].data[index] + all[1][axis].data[index] + all[2][axis].data[index]);
        }
    }
    all
}
