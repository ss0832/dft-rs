// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! One-electron integral matrices over the spherical AO basis: overlap S,
//! kinetic T, nuclear attraction V, and arbitrary-order multipole moments.
//!
//! Every matrix is assembled per shell pair in the Cartesian primitive
//! basis, contracted, transformed to real solid harmonics
//! (`basis::aos::sph_block`), and scattered symmetrically.

use super::hermite::{ETable, RTensor};
use super::kernels::CoulombKernel;
use crate::basis::{aos, cart_components, MolBasis};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Per-axis E tables of one primitive pair.
struct EAxes {
    x: ETable,
    y: ETable,
    z: ETable,
}

impl EAxes {
    fn build(imax: usize, jmax: usize, q: Vec3, a: f64, b: f64) -> Self {
        Self {
            x: ETable::build(imax, jmax, q.x, a, b),
            y: ETable::build(imax, jmax, q.y, a, b),
            z: ETable::build(imax, jmax, q.z, a, b),
        }
    }
}

fn overlap_1d(e: &ETable, la: i32, lb: i32, p: f64) -> f64 {
    if la < 0 || lb < 0 {
        return 0.0;
    }
    e.get(la as usize, lb as usize, 0) * (PI / p).sqrt()
}

/// 1D kinetic factor: T(la,lb) = b(2lb+1)S(la,lb) − 2b²S(la,lb+2)
/// − ½·lb(lb−1)·S(la,lb−2).
fn kinetic_1d(e: &ETable, la: i32, lb: i32, b: f64, p: f64) -> f64 {
    b * (2 * lb + 1) as f64 * overlap_1d(e, la, lb, p)
        - 2.0 * b * b * overlap_1d(e, la, lb + 2, p)
        - 0.5 * (lb * (lb - 1)) as f64 * overlap_1d(e, la, lb - 2, p)
}

/// Assemble a symmetric one-electron matrix from a per-shell-pair Cartesian
/// block builder (runs shell pairs in parallel).
fn assemble<F>(mb: &MolBasis, block: F) -> Matrix
where
    F: Fn(usize, usize) -> Matrix + Sync,
{
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (a..ns).map(move |b| (a, b))).collect();
    let blocks: Vec<(usize, usize, Matrix)> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let cart = block(a, b);
            let sph = aos::sph_block(mb.shells[a].l, mb.shells[b].l, &cart);
            (a, b, sph)
        })
        .collect();
    let mut m = Matrix::zeros(mb.nao_sph, mb.nao_sph);
    for (a, b, sph) in blocks {
        let (oa, ob) = (mb.shells[a].sph_off, mb.shells[b].sph_off);
        for i in 0..sph.rows {
            for j in 0..sph.cols {
                m[(oa + i, ob + j)] = sph[(i, j)];
                m[(ob + j, oa + i)] = sph[(i, j)];
            }
        }
    }
    m
}

/// Overlap matrix S (spherical AO basis).
pub fn overlap_matrix(mb: &MolBasis) -> Matrix {
    assemble(mb, |sa, sb| {
        let (a_sh, b_sh) = (&mb.shells[sa], &mb.shells[sb]);
        let (la, lb) = (a_sh.l as i32, b_sh.l as i32);
        let comps_a = cart_components(a_sh.l);
        let comps_b = cart_components(b_sh.l);
        let q = a_sh.center - b_sh.center;
        let mut block = Matrix::zeros(comps_a.len(), comps_b.len());
        for (&ea, &ca) in a_sh.exps.iter().zip(&a_sh.coefs) {
            for (&eb, &cb) in b_sh.exps.iter().zip(&b_sh.coefs) {
                let p = ea + eb;
                let e = EAxes::build(la as usize, lb as usize, q, ea, eb);
                let w = ca * cb;
                for (i, &(ax, ay, az)) in comps_a.iter().enumerate() {
                    for (j, &(bx, by, bz)) in comps_b.iter().enumerate() {
                        block[(i, j)] += w
                            * overlap_1d(&e.x, ax, bx, p)
                            * overlap_1d(&e.y, ay, by, p)
                            * overlap_1d(&e.z, az, bz, p);
                    }
                }
            }
        }
        block
    })
}

/// Kinetic-energy matrix T (spherical AO basis).
pub fn kinetic_matrix(mb: &MolBasis) -> Matrix {
    assemble(mb, |sa, sb| {
        let (a_sh, b_sh) = (&mb.shells[sa], &mb.shells[sb]);
        let (la, lb) = (a_sh.l as i32, b_sh.l as i32);
        let comps_a = cart_components(a_sh.l);
        let comps_b = cart_components(b_sh.l);
        let q = a_sh.center - b_sh.center;
        let mut block = Matrix::zeros(comps_a.len(), comps_b.len());
        for (&ea, &ca) in a_sh.exps.iter().zip(&a_sh.coefs) {
            for (&eb, &cb) in b_sh.exps.iter().zip(&b_sh.coefs) {
                let p = ea + eb;
                // jmax + 2 to reach the lb+2 overlaps in the kinetic form.
                let e = EAxes::build(la as usize, lb as usize + 2, q, ea, eb);
                let w = ca * cb;
                for (i, &(ax, ay, az)) in comps_a.iter().enumerate() {
                    for (j, &(bx, by, bz)) in comps_b.iter().enumerate() {
                        let sx = overlap_1d(&e.x, ax, bx, p);
                        let sy = overlap_1d(&e.y, ay, by, p);
                        let sz = overlap_1d(&e.z, az, bz, p);
                        let tx = kinetic_1d(&e.x, ax, bx, eb, p);
                        let ty = kinetic_1d(&e.y, ay, by, eb, p);
                        let tz = kinetic_1d(&e.z, az, bz, eb, p);
                        block[(i, j)] += w * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
                    }
                }
            }
        }
        block
    })
}

/// Nuclear-attraction matrix V = Σ_C (−Z_C) ⟨a| 1/|r−C| |b⟩.
///
/// `charges` supplies the per-atom charge (later: Z_eff for ECP atoms);
/// pass `None` to use the plain atomic numbers of `mol`.
pub fn nuclear_matrix(mb: &MolBasis, mol: &Molecule, charges: Option<&[f64]>) -> Matrix {
    nuclear_matrix_kernel(mb, mol, charges, CoulombKernel::Coulomb)
}

/// Nuclear attraction through an arbitrary Coulomb kernel.
///
/// `Erfc { omega }` gives the SCREENED attraction, which is what a
/// periodic calculation sums in real space: the bare `1/r` lattice sum is
/// only conditionally convergent, while the screened one decays like a
/// Gaussian and converges absolutely. The screening enters solely through
/// the Boys function, so the whole Hermite contraction is shared with the
/// unscreened case rather than duplicated.
pub fn nuclear_matrix_kernel(
    mb: &MolBasis,
    mol: &Molecule,
    charges: Option<&[f64]>,
    kernel: CoulombKernel,
) -> Matrix {
    let z: Vec<f64> = match charges {
        Some(c) => c.to_vec(),
        None => mol.atoms.iter().map(|a| a.z as f64).collect(),
    };
    let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
    assemble(mb, |sa, sb| {
        nuclear_pair_block_kernel(&mb.shells[sa], &mb.shells[sb], &centers, &z, kernel)
    })
}

/// Gaussian-damped nuclear attraction with kernel exp(-eta |r-C|²)/|r-C|.
/// Subtract this contribution from the bare nuclear attraction to obtain
/// the regularized nuclear potential. Gaussian contraction coefficients
/// retain the original AO normalization.
pub(crate) fn nuclear_penetration_matrix(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    eta: f64,
) -> Matrix {
    assemble(mb, |sa, sb| {
        let (a, b) = (&mb.shells[sa], &mb.shells[sb]);
        let (la, lb) = (a.l as usize, b.l as usize);
        let comps_a = cart_components(a.l);
        let comps_b = cart_components(b.l);
        let degree = la + lb;
        let mut block = Matrix::zeros(comps_a.len(), comps_b.len());
        let mut boys = vec![0.0; degree + 1];
        for (atom, &charge) in mol.atoms.iter().zip(charges) {
            if charge == 0.0 {
                continue;
            }
            for (&ea, &ca) in a.exps.iter().zip(&a.coefs) {
                let exponent = ea + eta;
                let shift = (atom.position - a.center) * (eta / exponent);
                let center = a.center + shift;
                let factor = (-ea * (eta / exponent) * (a.center - atom.position).norm2()).exp();
                for (&eb, &cb) in b.exps.iter().zip(&b.coefs) {
                    let p = exponent + eb;
                    let q = center - b.center;
                    let product = center + (b.center - center) * (eb / p);
                    let distance = product - atom.position;
                    let tables: [_; 3] = std::array::from_fn(|axis| {
                        translated_hermite(la, lb, q.get(axis), shift.get(axis), exponent, eb)
                    });
                    CoulombKernel::Coulomb.boys_into(degree, p, p * distance.norm2(), &mut boys);
                    let r = RTensor::build(degree, p, distance, &boys);
                    let weight = -charge * ca * cb * factor * (2.0 * PI / p);
                    for (i, &(ax, ay, az)) in comps_a.iter().enumerate() {
                        for (j, &(bx, by, bz)) in comps_b.iter().enumerate() {
                            let [x, y, z] = [
                                ((ax, bx), &tables[0]),
                                ((ay, by), &tables[1]),
                                ((az, bz), &tables[2]),
                            ]
                            .map(|((left, right), table)| {
                                &table[left as usize * (lb + 1) + right as usize]
                            });
                            let mut value = 0.0;
                            for (t, &ext) in x.iter().enumerate() {
                                for (u, &eyu) in y.iter().enumerate() {
                                    for (v, &ezv) in z.iter().enumerate() {
                                        value += ext * eyu * ezv * r.get(t, u, v);
                                    }
                                }
                            }
                            block[(i, j)] += weight * value;
                        }
                    }
                }
            }
        }
        block
    })
}

fn translated_hermite(la: usize, lb: usize, q: f64, shift: f64, a: f64, b: f64) -> Vec<Vec<f64>> {
    let table = ETable::build(la, lb, q, a, b);
    let mut result = Vec::with_capacity((la + 1) * (lb + 1));
    // (x-A)^i = sum_k binomial(i,k) (A'-A)^(i-k) (x-A')^k
    // after combining the damping Gaussian into the first primitive.
    for i in 0..=la {
        let mut shifts = vec![0.0; i + 1];
        shifts[i] = 1.0;
        for k in (0..i).rev() {
            shifts[k] = shifts[k + 1] * (k + 1) as f64 * shift / (i - k) as f64;
        }
        for j in 0..=lb {
            let mut coefficients = vec![0.0; i + j + 1];
            for (k, &factor) in shifts.iter().enumerate() {
                for (t, value) in coefficients.iter_mut().enumerate().take(k + j + 1) {
                    *value += factor * table.get(k, j, t);
                }
            }
            result.push(coefficients);
        }
    }
    result
}

/// Cartesian attraction block for a contracted shell pair and point charges.
/// Spherical transformation and symmetric scatter belong to the caller.
pub(crate) fn nuclear_pair_block_kernel(
    a_sh: &crate::basis::MolShell,
    b_sh: &crate::basis::MolShell,
    centers: &[Vec3],
    z: &[f64],
    kernel: CoulombKernel,
) -> Matrix {
    let (la, lb) = (a_sh.l as usize, b_sh.l as usize);
    let ltot = la + lb;
    let comps_a = cart_components(a_sh.l);
    let comps_b = cart_components(b_sh.l);
    let q = a_sh.center - b_sh.center;
    let mut block = Matrix::zeros(comps_a.len(), comps_b.len());
    let mut f = vec![0.0; ltot + 1];
    for (&ea, &ca) in a_sh.exps.iter().zip(&a_sh.coefs) {
        for (&eb, &cb) in b_sh.exps.iter().zip(&b_sh.coefs) {
            let p = ea + eb;
            let pc_center = (a_sh.center * ea + b_sh.center * eb) / p;
            let e = EAxes::build(la, lb, q, ea, eb);
            let w = ca * cb * 2.0 * PI / p;
            for (zc, cc) in z.iter().zip(centers) {
                if *zc == 0.0 {
                    continue;
                }
                let pc = pc_center - *cc;
                kernel.boys_into(ltot, p, p * pc.norm2(), &mut f);
                let r = RTensor::build(ltot, p, pc, &f);
                for (i, &(ax, ay, az)) in comps_a.iter().enumerate() {
                    for (j, &(bx, by, bz)) in comps_b.iter().enumerate() {
                        let mut sum = 0.0;
                        for t in 0..=(ax + bx) as usize {
                            let ext = e.x.get(ax as usize, bx as usize, t);
                            if ext == 0.0 {
                                continue;
                            }
                            for u in 0..=(ay + by) as usize {
                                let eyu = e.y.get(ay as usize, by as usize, u);
                                if eyu == 0.0 {
                                    continue;
                                }
                                for v in 0..=(az + bz) as usize {
                                    let ezv = e.z.get(az as usize, bz as usize, v);
                                    if ezv == 0.0 {
                                        continue;
                                    }
                                    sum += ext * eyu * ezv * r.get(t, u, v);
                                }
                            }
                        }
                        block[(i, j)] -= w * zc * sum;
                    }
                }
            }
        }
    }
    block
}

/// 1D Hermite moment table `M_t^e = ∫ Λ_t (x−C)^e dx` for `t ≤ e ≤ emax`,
/// by the ladder `M_t^{e+1} = t·M_{t−1}^e + X_PC·M_t^e + M_{t+1}^e/(2p)`.
pub(crate) fn moment_table(emax: usize, p: f64, x_pc: f64) -> Vec<Vec<f64>> {
    let root = (PI / p).sqrt();
    let mut m = vec![vec![0.0; emax + 2]; emax + 1];
    m[0][0] = root;
    for e in 0..emax {
        for t in 0..=(e + 1) {
            let mut v = x_pc * m[e][t];
            if t > 0 {
                v += t as f64 * m[e][t - 1];
            }
            v += m[e][t + 1] / (2.0 * p);
            m[e + 1][t] = v;
        }
    }
    m
}

/// All multipole matrices of total order `order` about `center`:
/// returns `(e,f,g) → ⟨a| (x−Cx)^e (y−Cy)^f (z−Cz)^g |b⟩` for every
/// monomial with `e+f+g = order`, in `cart_components(order)` order.
pub fn moment_matrices(mb: &MolBasis, center: Vec3, order: u8) -> Vec<((i32, i32, i32), Matrix)> {
    cart_components(order)
        .into_iter()
        .map(|power| {
            let m = assemble(mb, |sa, sb| {
                moment_pair_block(&mb.shells[sa], &mb.shells[sb], center, power)
            });
            (power, m)
        })
        .collect()
}

/// Cartesian shell-pair block of `(x−Cx)^e (y−Cy)^f (z−Cz)^g` about
/// `center`, in `cart_components` order for both shells.
pub fn moment_pair_block(
    a_sh: &crate::basis::MolShell,
    b_sh: &crate::basis::MolShell,
    center: Vec3,
    (me, mf, mg): (i32, i32, i32),
) -> Matrix {
    let (la, lb) = (a_sh.l as i32, b_sh.l as i32);
    let comps_a = cart_components(a_sh.l);
    let comps_b = cart_components(b_sh.l);
    let q = a_sh.center - b_sh.center;
    let emax = (me + mf + mg) as usize;
    let mut block = Matrix::zeros(comps_a.len(), comps_b.len());
    for (&ea, &ca) in a_sh.exps.iter().zip(&a_sh.coefs) {
        for (&eb, &cb) in b_sh.exps.iter().zip(&b_sh.coefs) {
            let p = ea + eb;
            let pcent = (a_sh.center * ea + b_sh.center * eb) / p;
            let e = EAxes::build(la as usize, lb as usize, q, ea, eb);
            let mx = moment_table(emax, p, pcent.x - center.x);
            let my = moment_table(emax, p, pcent.y - center.y);
            let mz = moment_table(emax, p, pcent.z - center.z);
            let w = ca * cb;
            for (i, &(ax, ay, az)) in comps_a.iter().enumerate() {
                for (j, &(bx, by, bz)) in comps_b.iter().enumerate() {
                    let fx = axis_moment(&e.x, &mx, ax, bx, me);
                    let fy = axis_moment(&e.y, &my, ay, by, mf);
                    let fz = axis_moment(&e.z, &mz, az, bz, mg);
                    block[(i, j)] += w * fx * fy * fz;
                }
            }
        }
    }
    block
}

/// Σ_t E^{ab}_t · M_t^e for one axis.
pub(crate) fn axis_moment(e: &ETable, m: &[Vec<f64>], la: i32, lb: i32, order: i32) -> f64 {
    let tmax = ((la + lb) as usize).min(order as usize);
    let mut sum = 0.0;
    for t in 0..=tmax {
        sum += e.get(la as usize, lb as usize, t) * m[order as usize][t];
    }
    sum
}

/// Convenience: the three dipole matrices ⟨a|(r−C)|b⟩ as `[x, y, z]`.
pub fn dipole_matrices(mb: &MolBasis, center: Vec3) -> [Matrix; 3] {
    let mut all = moment_matrices(mb, center, 1);
    // cart_components(1) order: (1,0,0), (0,1,0), (0,0,1).
    let z = all.pop().unwrap().1;
    let y = all.pop().unwrap().1;
    let x = all.pop().unwrap().1;
    [x, y, z]
}
