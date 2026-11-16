// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Magnetic-property operator integrals: the one-electron layer of GIAO
//! NMR shielding.
//!
//! The shielding of nucleus `N` is the mixed second derivative
//!
//! ```text
//! sigma_N,ab = d^2 E / dB_a dm_N,b
//! ```
//!
//! with `B` the external field and `m_N` the nuclear magnetic moment. It
//! splits into a DIAMAGNETIC part — the expectation of the `B`-`m` second
//! derivative operator over the unperturbed density — and a PARAMAGNETIC
//! part, the magnetic-field density response contracted with the nuclear
//! magnetic-dipole operator.
//!
//! This module provides the operator integrals both parts are built from.
//! All of them come out of the SAME McMurchie–Davidson machinery the rest
//! of the code uses — [`ETable`] for the Hermite expansion and [`RTensor`]
//! for the Coulomb kernel — so there is no second integral engine to keep
//! in step:
//!
//! - [`electric_field_operator`]: `<mu| (r-C)_j / |r-C|^3 |nu>`, which is
//!   `d/dC_j` of the nuclear attraction and therefore the same Hermite
//!   contraction with one extra order in direction `j`;
//! - [`paramagnetic_operator`]: `<mu| (r_C x grad)_b / |r-C|^3 |nu>`, the
//!   nuclear magnetic-dipole operator, built from the field integral
//!   through the ket-derivative relation
//!   `d_k|b> = b_k |b - e_k> - 2 beta |b + e_k>`;
//! - [`angular_momentum_operator`]: `<mu| ((r-O) x grad)_b |nu>`, the
//!   orbital angular momentum about a chosen origin, which is the
//!   one-electron part of the magnetic-field perturbation;
//! - [`gradient_operator`]: `<mu| d_k |nu>`, which the two above share and
//!   which pins the origin dependence of the angular momentum.
//!
//! The magnetic perturbation is IMAGINARY and ANTISYMMETRIC. Both magnetic
//! operators here are real and antisymmetric in `(mu, nu)`, the factor `i`
//! having been taken out — that is why the magnetic response needs its own
//! solver ([`super::magnetic`]) rather than the symmetric one, and it is
//! also what the tests gate on: a sign slip in the curl produces a matrix
//! with a symmetric part and hence a shielding that is finite but wrong.
//!
//! **Scope.** What is here is the one-electron operator layer, gated. The
//! GIAO field derivative of the two-electron integrals lives in
//! [`super::giao_eri`] and the CPHF / shielding assembly in
//! [`super::magnetic`].
//!
//! **Gauge origin.** Every routine that needs one takes it explicitly as
//! `origin`. The London orbital is
//!
//! ```text
//! chi_mu^B(r) = exp(-(i/2) [B x (R_mu - O)] . (r - O)) chi_mu(r)
//! ```
//!
//! which makes the phase manifestly covariant under a shift of `O`: moving
//! the gauge origin to `O + delta` multiplies each London orbital by the
//! CONSTANT phase `exp((i/2) B . ((R_mu - O) x delta))` times one common
//! r-dependent phase that the change of `A` cancels. A diagonal unitary on
//! the basis changes no observable — that is exactly why GIAO shieldings
//! are gauge-origin independent, and every derivative matrix here obeys the
//! resulting first-order identity
//!
//! ```text
//! X_a(O + delta) = X_a(O) + (gamma_n - gamma_m) * X_a(0-th order),
//! gamma_m = (1/2) ((R_m - O) x delta)_a
//! ```
//!
//! which the tests use as a sharp, oracle-free check.

use crate::basis::{cart_components, MolBasis, MolShell};
use crate::error::{DftError, Result};
use crate::integrals::hermite::{ETable, RTensor};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use rayon::prelude::*;

/// The two `(j, k, sign)` triples of the cross product `(r x grad)_b`:
/// `L_x = y d_z - z d_y`, and cyclically.
pub(crate) fn curl_terms(b: usize) -> [(usize, usize, f64); 2] {
    match b {
        0 => [(1, 2, 1.0), (2, 1, -1.0)],
        1 => [(2, 0, 1.0), (0, 2, -1.0)],
        _ => [(0, 1, 1.0), (1, 0, -1.0)],
    }
}

/// `<la| (r-C)_j / |r-C|^3 |lb>` for one primitive Cartesian pair.
///
/// This is `d/dC_j` of the nuclear attraction integral, so it is the same
/// Hermite contraction with one extra order in direction `j`.
#[allow(clippy::too_many_arguments)]
fn field_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    j: usize,
    alpha: f64,
    beta: f64,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> f64 {
    let p = alpha + beta;
    let pp = (a * alpha + b * beta) / p;
    let pc = pp - c;
    let q = a - b;
    let ex = ETable::build(la[0], lb[0], q.x, alpha, beta);
    let ey = ETable::build(la[1], lb[1], q.y, alpha, beta);
    let ez = ETable::build(la[2], lb[2], q.z, alpha, beta);
    let (dt, du, dv) = match j {
        0 => (1usize, 0usize, 0usize),
        1 => (0, 1, 0),
        _ => (0, 0, 1),
    };
    let ltot = la[0] + la[1] + la[2] + lb[0] + lb[1] + lb[2] + 1;
    let mut f = vec![0.0; ltot + 1];
    CoulombKernel::Coulomb.boys_into(ltot, p, p * pc.norm2(), &mut f);
    let r = RTensor::build(ltot, p, pc, &f);
    let mut v = 0.0;
    for t in 0..=(la[0] + lb[0]) {
        let et = ex.get(la[0], lb[0], t);
        if et == 0.0 {
            continue;
        }
        for u in 0..=(la[1] + lb[1]) {
            let eu = ey.get(la[1], lb[1], u);
            if eu == 0.0 {
                continue;
            }
            for w in 0..=(la[2] + lb[2]) {
                let ew = ez.get(la[2], lb[2], w);
                if ew == 0.0 {
                    continue;
                }
                v += et * eu * ew * r.get(t + dt, u + du, w + dv);
            }
        }
    }
    // Nuclear attraction of a UNIT charge is -2pi/p sum E R; one extra
    // Hermite order in j differentiates it with respect to C_j, and the
    // outer sign turns dV/dC into the field itself.
    -v * 2.0 * std::f64::consts::PI / p
}

/// `<la| (r-O)_j |lb>` for one primitive pair: the overlap with one
/// Cartesian factor. `(r-O)_j |lb> = |lb + e_j> + (B_j - O_j)|lb>`.
#[allow(clippy::too_many_arguments)]
fn moment_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    j: usize,
    alpha: f64,
    beta: f64,
    a: Vec3,
    b: Vec3,
    o: Vec3,
) -> f64 {
    let mut lbp = lb;
    lbp[j] += 1;
    overlap_primitive(la, lbp, alpha, beta, a, b)
        + (b.get(j) - o.get(j)) * overlap_primitive(la, lb, alpha, beta, a, b)
}

/// Plain primitive overlap from the E-coefficient table.
fn overlap_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    alpha: f64,
    beta: f64,
    a: Vec3,
    b: Vec3,
) -> f64 {
    let p = alpha + beta;
    let q = a - b;
    let pref = (std::f64::consts::PI / p).sqrt();
    let mut prod = 1.0;
    for (axis, qa) in [q.x, q.y, q.z].into_iter().enumerate() {
        prod *=
            ETable::build(la[axis], lb[axis], qa, alpha, beta).get(la[axis], lb[axis], 0) * pref;
    }
    prod
}

/// Apply `d_k` to the ket of a primitive-level bilinear form:
/// `d_k|b> = b_k |b - e_k> - 2 beta |b + e_k>`.
fn ket_derivative<F>(lb: [usize; 3], k: usize, beta: f64, mut f: F) -> f64
where
    F: FnMut([usize; 3]) -> f64,
{
    let mut s = 0.0;
    if lb[k] >= 1 {
        let mut lbm = lb;
        lbm[k] -= 1;
        s += lb[k] as f64 * f(lbm);
    }
    let mut lbp = lb;
    lbp[k] += 1;
    s - 2.0 * beta * f(lbp)
}

/// Contract a primitive-level operator over one shell pair into the
/// spherical AO block.
fn shell_block<F>(sh_a: &MolShell, sh_b: &MolShell, mut prim: F) -> Matrix
where
    F: FnMut([usize; 3], [usize; 3], f64, f64) -> f64,
{
    let ca = cart_components(sh_a.l);
    let cb = cart_components(sh_b.l);
    let mut cart = Matrix::zeros(ca.len(), cb.len());
    for (&ea, &coa) in sh_a.exps.iter().zip(&sh_a.coefs) {
        for (&eb, &cob) in sh_b.exps.iter().zip(&sh_b.coefs) {
            let w = coa * cob;
            for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                    cart[(i, j)] += w * prim(
                        [ax as usize, ay as usize, az as usize],
                        [bx as usize, by as usize, bz as usize],
                        ea,
                        eb,
                    );
                }
            }
        }
    }
    crate::basis::aos::sph_block(sh_a.l, sh_b.l, &cart)
}

/// Assemble an `ncomp`-component one-electron operator over all shell pairs.
fn assemble_n<F>(mb: &MolBasis, ncomp: usize, prim: F) -> Vec<Matrix>
where
    F: Fn(usize, [usize; 3], [usize; 3], f64, f64, Vec3, Vec3) -> f64 + Sync,
{
    let n = mb.nao_sph;
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let mut out: Vec<Matrix> = (0..ncomp).map(|_| Matrix::zeros(n, n)).collect();
    for (comp, target) in out.iter_mut().enumerate() {
        let blocks: Vec<((usize, usize), Matrix)> = pairs
            .par_iter()
            .map(|&(a, b)| {
                let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
                let blk = shell_block(sh_a, sh_b, |la, lb, ea, eb| {
                    prim(comp, la, lb, ea, eb, sh_a.center, sh_b.center)
                });
                ((sh_a.sph_off, sh_b.sph_off), blk)
            })
            .collect();
        for ((oa, ob), blk) in blocks {
            for i in 0..blk.rows {
                for j in 0..blk.cols {
                    target[(oa + i, ob + j)] = blk[(i, j)];
                }
            }
        }
    }
    out
}

/// Assemble a three-component one-electron operator over all shell pairs.
fn assemble3<F>(mb: &MolBasis, prim: F) -> [Matrix; 3]
where
    F: Fn(usize, [usize; 3], [usize; 3], f64, f64, Vec3, Vec3) -> f64 + Sync,
{
    let mut v = assemble_n(mb, 3, prim);
    let z = v.pop().expect("three components");
    let y = v.pop().expect("three components");
    let x = v.pop().expect("three components");
    [x, y, z]
}

/// Centre of every spherical AO (the origin of its London phase).
pub(crate) fn ao_centres(mb: &MolBasis) -> Vec<Vec3> {
    let mut c = vec![Vec3::zero(); mb.nao_sph];
    for sh in &mb.shells {
        for k in 0..crate::basis::n_sph(sh.l) {
            c[sh.sph_off + k] = sh.center;
        }
    }
    c
}

/// 1D overlap factor `E^{ij}_0 sqrt(pi/p)`, zero outside the valid range.
fn overlap_1d(e: &ETable, i: i32, j: i32, p: f64) -> f64 {
    if i < 0 || j < 0 {
        return 0.0;
    }
    e.get(i as usize, j as usize, 0) * (std::f64::consts::PI / p).sqrt()
}

/// Primitive kinetic-energy integral, differentiating the KET:
/// `T(i,j) = b(2j+1)S(i,j) - 2b^2 S(i,j+2) - (1/2) j(j-1) S(i,j-2)`.
fn kinetic_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    alpha: f64,
    beta: f64,
    a: Vec3,
    b: Vec3,
) -> f64 {
    let p = alpha + beta;
    let q = a - b;
    let e: [ETable; 3] = [
        ETable::build(la[0], lb[0] + 2, q.x, alpha, beta),
        ETable::build(la[1], lb[1] + 2, q.y, alpha, beta),
        ETable::build(la[2], lb[2] + 2, q.z, alpha, beta),
    ];
    let mut s = [0.0; 3];
    let mut t = [0.0; 3];
    for axis in 0..3 {
        let (i, j) = (la[axis] as i32, lb[axis] as i32);
        s[axis] = overlap_1d(&e[axis], i, j, p);
        t[axis] = beta * (2 * j + 1) as f64 * s[axis]
            - 2.0 * beta * beta * overlap_1d(&e[axis], i, j + 2, p)
            - 0.5 * (j * (j - 1)) as f64 * overlap_1d(&e[axis], i, j - 2, p);
    }
    t[0] * s[1] * s[2] + s[0] * t[1] * s[2] + s[0] * s[1] * t[2]
}

/// Primitive nuclear attraction `-sum_C Z_C <la| 1/|r-C| |lb>`.
#[allow(clippy::too_many_arguments)]
fn nuclear_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    alpha: f64,
    beta: f64,
    a: Vec3,
    b: Vec3,
    centers: &[Vec3],
    charges: &[f64],
) -> f64 {
    let p = alpha + beta;
    let pp = (a * alpha + b * beta) / p;
    let q = a - b;
    let ex = ETable::build(la[0], lb[0], q.x, alpha, beta);
    let ey = ETable::build(la[1], lb[1], q.y, alpha, beta);
    let ez = ETable::build(la[2], lb[2], q.z, alpha, beta);
    let ltot = la[0] + la[1] + la[2] + lb[0] + lb[1] + lb[2];
    let mut f = vec![0.0; ltot + 1];
    let mut total = 0.0;
    for (&zc, &cc) in charges.iter().zip(centers) {
        if zc == 0.0 {
            continue;
        }
        let pc = pp - cc;
        CoulombKernel::Coulomb.boys_into(ltot, p, p * pc.norm2(), &mut f);
        let r = RTensor::build(ltot, p, pc, &f);
        let mut v = 0.0;
        for t in 0..=(la[0] + lb[0]) {
            let et = ex.get(la[0], lb[0], t);
            if et == 0.0 {
                continue;
            }
            for u in 0..=(la[1] + lb[1]) {
                let eu = ey.get(la[1], lb[1], u);
                if eu == 0.0 {
                    continue;
                }
                for w in 0..=(la[2] + lb[2]) {
                    let ew = ez.get(la[2], lb[2], w);
                    if ew == 0.0 {
                        continue;
                    }
                    v += et * eu * ew * r.get(t, u, w);
                }
            }
        }
        total -= zc * v * 2.0 * std::f64::consts::PI / p;
    }
    total
}

/// Electric-field integrals `<mu| (r-C)_j / |r-C|^3 |nu>`.
///
/// The building block of the paramagnetic term, and on its own the
/// electronic electric field at a point.
pub fn electric_field_operator(mb: &MolBasis, c: Vec3) -> [Matrix; 3] {
    assemble3(mb, |j, la, lb, ea, eb, ca, cb| {
        field_primitive(la, lb, j, ea, eb, ca, cb, c)
    })
}

/// The momentum (gradient) operator `<mu| d_k |nu>`.
///
/// Real and antisymmetric, since `d_k` is anti-Hermitian; the factor `-i`
/// of the physical momentum has been taken out.
pub fn gradient_operator(mb: &MolBasis) -> [Matrix; 3] {
    assemble3(mb, |k, la, lb, ea, eb, ca, cb| {
        ket_derivative(lb, k, eb, |l| overlap_primitive(la, l, ea, eb, ca, cb))
    })
}

/// The nuclear magnetic-dipole operator
/// `<mu| (r_C x grad)_b / |r-C|^3 |nu>` at nucleus centre `c`.
///
/// Real and ANTISYMMETRIC in `(mu, nu)` — the `i` of the imaginary
/// perturbation has been taken out.
pub fn paramagnetic_operator(mb: &MolBasis, c: Vec3) -> [Matrix; 3] {
    assemble3(mb, |b, la, lb, ea, eb, ca, cb| {
        let mut total = 0.0;
        for (jdir, kdir, sign) in curl_terms(b) {
            total += sign
                * ket_derivative(lb, kdir, eb, |l| {
                    field_primitive(la, l, jdir, ea, eb, ca, cb, c)
                });
        }
        total
    })
}

/// Orbital angular momentum `<mu| ((r-O) x grad)_b |nu>` about origin `o`.
///
/// The one-electron part of the magnetic-field perturbation. Real and
/// antisymmetric, and built from the same overlap/E-table path as
/// [`gradient_operator`] rather than a separate one.
pub fn angular_momentum_operator(mb: &MolBasis, o: Vec3) -> [Matrix; 3] {
    assemble3(mb, |b, la, lb, ea, eb, ca, cb| {
        let mut total = 0.0;
        for (jdir, kdir, sign) in curl_terms(b) {
            total += sign
                * ket_derivative(lb, kdir, eb, |l| {
                    moment_primitive(la, l, jdir, ea, eb, ca, cb, o)
                });
        }
        total
    })
}

/// GIAO derivative of the overlap matrix with respect to the magnetic
/// field, `dS/dB_a`, with the factor `i` taken out.
///
/// Each London orbital carries `exp(-i/2 (B x R_mu) . r)`, so
///
/// ```text
/// dS_mn/dB_a = (i/2) [ (R_m - R_n) x <m| r |n> ]_a
/// ```
///
/// The bracket is REAL and antisymmetric in `(m, n)`; the returned matrices
/// are that bracket, so the caller supplies the `i/2`. Note what the
/// difference `R_m - R_n` means: a one-centre block contributes nothing, so
/// this term exists only because the two basis functions have different
/// gauge origins — which is the whole mechanism by which GIAOs remove the
/// gauge-origin dependence.
///
/// This is the first of the pieces the magnetic-field CPHF needs; the
/// others are [`giao_core_hamiltonian_derivative`] here and
/// [`super::giao_eri::giao_eri_fock_derivative`].
pub fn giao_overlap_derivative(mb: &MolBasis) -> [Matrix; 3] {
    giao_overlap_derivative_about(mb, Vec3::zero())
}

/// [`giao_overlap_derivative`] about an explicit gauge origin: the dipole
/// factor is `<m| (r - O) |n>`, so the returned matrices carry the origin
/// the London phases were written about.
pub fn giao_overlap_derivative_about(mb: &MolBasis, origin: Vec3) -> [Matrix; 3] {
    let dip = crate::integrals::dipole_matrices(mb, origin);
    let n = mb.nao_sph;
    let mut out = [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ];
    let centre = ao_centres(mb);
    for (a, target) in out.iter_mut().enumerate() {
        let (j, k, _) = curl_terms(a)[0];
        for m in 0..n {
            for p in 0..n {
                let d = centre[m] - centre[p];
                target[(m, p)] = 0.5 * (d.get(j) * dip[k][(m, p)] - d.get(k) * dip[j][(m, p)]);
            }
        }
    }
    out
}

/// Core Hamiltonian with the BRA multiplied by one Cartesian factor:
/// `<m| (r - O)_j h |n>` for `j = x, y, z`, with `h = T + V`.
///
/// This is the piece the GIAO core-Hamiltonian derivative is built from,
/// and the reason it cannot be shortcut: `h` carries the kinetic operator,
/// which does not commute with `r`, so `<m| r_j h |n>` and `<m| h r_j |n>`
/// are genuinely different matrices. Their DIFFERENCE is the commutator
/// `<m| [r_j, h] |n> = <m| d_j |n>`, which is what the test gates on.
pub fn moment_weighted_core_hamiltonian(
    mol: &Molecule,
    mb: &MolBasis,
    origin: Vec3,
) -> Result<[Matrix; 3]> {
    if !mb.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "magnetic properties with ECP basis sets".into(),
            missing: "the core potential's moment-weighted integrals".into(),
        });
    }
    let charges = mb.effective_charges(mol);
    let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
    Ok(assemble3(mb, |j, la, lb, ea, eb, ca, cb| {
        let core = |l: [usize; 3]| {
            kinetic_primitive(l, lb, ea, eb, ca, cb)
                + nuclear_primitive(l, lb, ea, eb, ca, cb, &centers, &charges)
        };
        let mut lap = la;
        lap[j] += 1;
        // (r-O)_j chi_a = (r-A)_j chi_a + (A-O)_j chi_a; the first factor
        // raises the bra's Cartesian power, the second is a plain scalar.
        core(lap) + (ca.get(j) - origin.get(j)) * core(la)
    }))
}

/// GIAO derivative of the core Hamiltonian, `dh/dB_a`, with the factor `i`
/// taken out — the returned matrices are real and antisymmetric.
///
/// Three contributions, from the bra phase, the ket phase, and the
/// operator's own field dependence in the symmetric gauge
/// `h(B) = T + V + (1/2) B.L_O + (1/8) (B x (r-O))^2`:
///
/// ```text
/// dh_mn/dB_a = (i/2) [ <m| (d_m x r_O)_a h |n> - <m| h (d_n x r_O)_a |n> ]
///            + (1/2) <m| L_{O,a} |n>,     d_m = R_m - O,  r_O = r - O
/// ```
///
/// With `W_a[m,n] = <m| (d_m x r_O)_a h |n>` the first bracket is
/// `W_a - W_a^T` (h is Hermitian and everything real), so what is returned
/// is `(1/2)(W_a - W_a^T) - (1/2) L_a`, the `L` term picking up its sign
/// from `L = r x p = -i (r x grad)`.
pub fn giao_core_hamiltonian_derivative(
    mol: &Molecule,
    mb: &MolBasis,
    origin: Vec3,
) -> Result<[Matrix; 3]> {
    let g = moment_weighted_core_hamiltonian(mol, mb, origin)?;
    let lop = angular_momentum_operator(mb, origin);
    let centre = ao_centres(mb);
    let n = mb.nao_sph;
    let mut out = [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ];
    for (a, target) in out.iter_mut().enumerate() {
        let (j, k, _) = curl_terms(a)[0];
        for m in 0..n {
            let dm = centre[m] - origin;
            for p in 0..n {
                let dp = centre[p] - origin;
                let w_mp = dm.get(j) * g[k][(m, p)] - dm.get(k) * g[j][(m, p)];
                let w_pm = dp.get(j) * g[k][(p, m)] - dp.get(k) * g[j][(p, m)];
                target[(m, p)] = 0.5 * (w_mp - w_pm) - 0.5 * lop[a][(m, p)];
            }
        }
    }
    Ok(out)
}

/// The mixed `B`-`m_N` second derivative of the London-basis one-electron
/// matrix, `d^2 h_mn / dB_a dm_{N,b}` — the non-response ("diamagnetic")
/// half of the shielding. Returned as `out[a][b]`, real and SYMMETRIC in
/// `(m, n)`.
///
/// Three pieces, all of them field integrals at the nucleus with at most
/// one extra Cartesian factor on the ket:
///
/// - the operator's own mixed derivative `A_B^{(a)} . A_N^{(b)}`, i.e.
///   `(1/2) [ delta_ab (r_O . r_N) - r_{N,a} r_{O,b} ] / r_N^3`;
/// - the London phases differentiating the nuclear dipole operator,
///   `(1/2) <m| (R_mn x r_O)_a (r_N x grad)_b / r_N^3 |n>`;
/// - the term left over when the momentum operator hits the phase's own
///   `r` dependence, `-(1/2) <m| (d_n x A_N^{(b)})_a |n>`.
///
/// Dropping the last two is the common way to get a plausible but wrong
/// GIAO shielding: they are exactly what carries the gauge-origin
/// dependence that must cancel against the response term.
///
/// With `giao = false` only the operator's own mixed derivative is kept —
/// the common-gauge-origin form, which is what a non-London basis gives.
pub fn shielding_mixed_operator(
    mb: &MolBasis,
    nucleus: Vec3,
    origin: Vec3,
    giao: bool,
) -> [[Matrix; 3]; 3] {
    let n = mb.nao_sph;
    let ef = if giao {
        electric_field_operator(mb, nucleus)
    } else {
        [
            Matrix::zeros(n, n),
            Matrix::zeros(n, n),
            Matrix::zeros(n, n),
        ]
    };
    // Q[c*3 + d] = <m| (r-O)_d (r-N)_c / |r-N|^3 |n>.
    let q = assemble_n(mb, 9, |cd, la, lb, ea, eb, ca, cb| {
        let (c, d) = (cd / 3, cd % 3);
        let mut lbp = lb;
        lbp[d] += 1;
        field_primitive(la, lbp, c, ea, eb, ca, cb, nucleus)
            + (cb.get(d) - origin.get(d)) * field_primitive(la, lb, c, ea, eb, ca, cb, nucleus)
    });
    // Y[c*3 + b] = <m| (r-O)_c (r_N x grad)_b / |r-N|^3 |n>.
    let y = if giao {
        assemble_n(mb, 9, |cb_idx, la, lb, ea, eb, ca, cbv| {
            let (c, bcomp) = (cb_idx / 3, cb_idx % 3);
            let mut total = 0.0;
            for (jdir, kdir, sign) in curl_terms(bcomp) {
                total += sign
                    * ket_derivative(lb, kdir, eb, |l| {
                        let mut lp = l;
                        lp[c] += 1;
                        field_primitive(la, lp, jdir, ea, eb, ca, cbv, nucleus)
                            + (cbv.get(c) - origin.get(c))
                                * field_primitive(la, l, jdir, ea, eb, ca, cbv, nucleus)
                    });
            }
            total
        })
    } else {
        (0..9).map(|_| Matrix::zeros(n, n)).collect()
    };
    let centre = ao_centres(mb);
    let zero = || {
        [
            Matrix::zeros(n, n),
            Matrix::zeros(n, n),
            Matrix::zeros(n, n),
        ]
    };
    let mut out = [zero(), zero(), zero()];
    for a in 0..3 {
        let (jc, kc, _) = curl_terms(a)[0];
        for b in 0..3 {
            let mut m = Matrix::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    let rmn = centre[r] - centre[c];
                    let dn = centre[c] - origin;
                    // Operator mixed derivative.
                    let mut v = -q[a * 3 + b][(r, c)];
                    if a == b {
                        v += q[0][(r, c)] + q[4][(r, c)] + q[8][(r, c)];
                    }
                    // Phase x nuclear-dipole operator.
                    v += rmn.get(jc) * y[kc * 3 + b][(r, c)] - rmn.get(kc) * y[jc * 3 + b][(r, c)];
                    // Momentum acting on the phase's own r dependence:
                    // (d_n x A_N^{(b)})_a = delta_ab (d_n . F) - F_a d_{n,b}.
                    let mut extra = -dn.get(b) * ef[a][(r, c)];
                    if a == b {
                        extra += dn.get(0) * ef[0][(r, c)]
                            + dn.get(1) * ef[1][(r, c)]
                            + dn.get(2) * ef[2][(r, c)];
                    }
                    m[(r, c)] = 0.5 * (v - extra);
                }
            }
            out[a][b] = m;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::system::{Atom, Molecule};

    /// Water, in Bohr, with a basis carrying p functions so the curl and
    /// the ket-derivative relation are actually exercised.
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
        let set = data::load("6-31g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        (mol, mb)
    }

    /// The field integral is `d/dC` of the nuclear attraction, so it must
    /// match central differences of that integral. This pins the extra
    /// Hermite order, the prefactor and the sign together — and every
    /// magnetic operator below is built on top of it.
    #[test]
    fn field_integral_is_the_nuclear_attraction_derivative() {
        let (_, mb) = water();
        let c = Vec3::new(0.13, -0.27, 0.71);
        let ef = electric_field_operator(&mb, c);
        let h = 1e-5;
        for j in 0..3 {
            let at = |s: f64| -> Matrix {
                let mut cc = c;
                match j {
                    0 => cc.x += s,
                    1 => cc.y += s,
                    _ => cc.z += s,
                }
                let m = Molecule::new(vec![Atom { z: 1, position: cc }]);
                crate::integrals::nuclear_matrix(&mb, &m, Some(&[1.0]))
            };
            let (vp, vm) = (at(h), at(-h));
            let mut worst = 0.0_f64;
            for i in 0..mb.nao_sph {
                for k in 0..mb.nao_sph {
                    // V = -Z <mu|1/|r-C||nu>, so dV/dC_j = +field_j.
                    let fd = (vp[(i, k)] - vm[(i, k)]) / (2.0 * h);
                    worst = worst.max((ef[j][(i, k)] + fd).abs());
                }
            }
            assert!(worst < 1e-7, "field component {j}: worst FD gap {worst:e}");
        }
    }

    /// The gradient operator must reproduce the overlap derivative: moving
    /// the ket shell is exactly `-<mu|d|nu>` for the functions on it. This
    /// gates `ket_derivative`, which both magnetic operators route through.
    #[test]
    fn gradient_operator_matches_the_overlap_derivative() {
        let (mol, mb) = water();
        let g = gradient_operator(&mb);
        let h = 1e-5;
        // Displace atom 1 (an H) and compare with the columns its AOs own.
        let at = |s: f64, ax: usize| -> Matrix {
            let mut m = mol.clone();
            match ax {
                0 => m.atoms[1].position.x += s,
                1 => m.atoms[1].position.y += s,
                _ => m.atoms[1].position.z += s,
            }
            let set = data::load("6-31g").unwrap();
            crate::integrals::overlap_matrix(&MolBasis::build(&m, &set).unwrap())
        };
        let owned: Vec<usize> = mb
            .shells
            .iter()
            .filter(|s| s.atom == 1)
            .flat_map(|s| s.sph_off..s.sph_off + crate::basis::n_sph(s.l))
            .collect();
        for ax in 0..3 {
            let (sp, sm) = (at(h, ax), at(-h, ax));
            let mut worst = 0.0_f64;
            for i in 0..mb.nao_sph {
                if owned.contains(&i) {
                    // Both sides move; that case is the sum rule, not this.
                    continue;
                }
                for &j in &owned {
                    let fd = (sp[(i, j)] - sm[(i, j)]) / (2.0 * h);
                    // d/dB <mu|nu(B)> = -<mu|d|nu>
                    worst = worst.max((fd + g[ax][(i, j)]).abs());
                }
            }
            assert!(worst < 1e-7, "gradient axis {ax}: worst FD gap {worst:e}");
        }
    }

    /// Every magnetic operator must be ANTISYMMETRIC. That is the
    /// signature of an imaginary perturbation, and it is not automatic: a
    /// sign slip in the curl, or dropping either half of the
    /// ket-derivative relation, leaves a symmetric part behind and the
    /// shielding comes out finite but wrong.
    #[test]
    fn magnetic_operators_are_antisymmetric() {
        let (mol, mb) = water();
        let ops = [
            (
                "paramagnetic on O",
                paramagnetic_operator(&mb, mol.atoms[0].position),
            ),
            // Off-centre too: antisymmetry must not be an accident of the
            // operator sitting on a nucleus.
            (
                "paramagnetic off-centre",
                paramagnetic_operator(&mb, Vec3::new(0.3, 0.4, -0.2)),
            ),
            (
                "angular momentum",
                angular_momentum_operator(&mb, Vec3::zero()),
            ),
            ("gradient", gradient_operator(&mb)),
        ];
        for (name, op) in &ops {
            for (b, m) in op.iter().enumerate() {
                let mut worst = 0.0_f64;
                let mut scale = 0.0_f64;
                for i in 0..m.rows {
                    for j in 0..m.cols {
                        worst = worst.max((m[(i, j)] + m[(j, i)]).abs());
                        scale = scale.max(m[(i, j)].abs());
                    }
                }
                assert!(
                    worst < 1e-10 * scale.max(1.0),
                    "{name} component {b}: worst antisymmetry gap {worst:e} (scale {scale:e})"
                );
                assert!(scale > 1e-8, "{name} component {b} is identically zero");
            }
        }
    }

    /// The GIAO overlap field-derivative must be antisymmetric (the
    /// perturbation is imaginary) and must VANISH within a single centre
    /// — a one-centre block has `R_m - R_n = 0`. That second property is
    /// the point of London orbitals: the term exists only because two
    /// functions carry different gauge origins.
    #[test]
    fn giao_overlap_derivative_is_antisymmetric_and_one_centre_free() {
        let (_, mb) = water();
        let ds = giao_overlap_derivative(&mb);
        let owner = mb.sph_ao_atoms();
        let mut off_centre_scale = 0.0_f64;
        for (a, m) in ds.iter().enumerate() {
            for i in 0..mb.nao_sph {
                for j in 0..mb.nao_sph {
                    assert!(
                        (m[(i, j)] + m[(j, i)]).abs() < 1e-12,
                        "component {a} not antisymmetric at ({i},{j})"
                    );
                    if owner[i] == owner[j] {
                        assert!(
                            m[(i, j)].abs() < 1e-12,
                            "component {a} is non-zero within one centre at ({i},{j}): {}",
                            m[(i, j)]
                        );
                    } else {
                        off_centre_scale = off_centre_scale.max(m[(i, j)].abs());
                    }
                }
            }
        }
        assert!(
            off_centre_scale > 1e-6,
            "the two-centre part is identically zero, so nothing was tested"
        );
    }

    /// Multiplying the BRA by one Cartesian factor and evaluating the
    /// primitive one power higher — the pattern every moment-weighted
    /// operator below is built on — must reproduce the multipole path,
    /// which is implemented independently (Hermite moment tables rather
    /// than raised angular momenta).
    #[test]
    fn raised_bra_moment_reproduces_the_dipole_matrix() {
        let (_, mb) = water();
        let o = Vec3::new(0.21, -0.34, 0.15);
        let got = assemble3(&mb, |j, la, lb, ea, eb, ca, cb| {
            let mut lap = la;
            lap[j] += 1;
            overlap_primitive(lap, lb, ea, eb, ca, cb)
                + (ca.get(j) - o.get(j)) * overlap_primitive(la, lb, ea, eb, ca, cb)
        });
        let want = crate::integrals::dipole_matrices(&mb, o);
        for j in 0..3 {
            let gap = got[j].max_abs_diff(&want[j]);
            assert!(gap < 1e-11, "moment axis {j}: worst gap {gap:e}");
        }
    }

    /// `<m| r_j h |n>` and `<m| h r_j |n>` differ by exactly the commutator
    /// `[r_j, h] = [r_j, T] = d_j`. This is the identity the tempting
    /// simplification of the GIAO core-Hamiltonian derivative violates: if
    /// `h` could be pulled out of the moment, the difference would be zero.
    #[test]
    fn moment_weighted_core_hamiltonian_carries_the_kinetic_commutator() {
        let (mol, mb) = water();
        let g = moment_weighted_core_hamiltonian(&mol, &mb, Vec3::new(0.4, -0.1, 0.7)).unwrap();
        let grad = gradient_operator(&mb);
        for j in 0..3 {
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for m in 0..mb.nao_sph {
                for n in 0..mb.nao_sph {
                    let comm = g[j][(m, n)] - g[j][(n, m)];
                    worst = worst.max((comm - grad[j][(m, n)]).abs());
                    scale = scale.max(g[j][(m, n)].abs());
                }
            }
            assert!(
                worst < 1e-9,
                "axis {j}: [r,h] gap {worst:e} (moment scale {scale:e})"
            );
            assert!(scale > 1e-6, "axis {j} moment is identically zero");
        }
    }

    /// Shifting the gauge origin `O -> O + delta` multiplies every London
    /// orbital by the CONSTANT phase `exp((i/2) B.((R_mu - O) x delta))`
    /// (times one common r-dependent phase that the change of `A` cancels).
    /// A diagonal unitary on the basis, so every first-order field
    /// derivative must obey
    /// `X_a(O') = X_a(O) + (gamma_n - gamma_m) X^0` exactly.
    ///
    /// This is the sharpest oracle-free check available on `dh/dB`: its
    /// proof consumes the kinetic commutator above, so an implementation
    /// that drops the angular-momentum term, or gets either phase sign
    /// wrong, fails it.
    #[test]
    fn giao_derivatives_obey_the_gauge_origin_shift_identity() {
        let (mol, mb) = water();
        let o = Vec3::new(0.31, -0.47, 0.19);
        let delta = Vec3::new(-0.23, 0.61, 0.37);
        let n = mb.nao_sph;
        let centre = ao_centres(&mb);
        let s0 = crate::integrals::overlap_matrix(&mb);
        let charges = mb.effective_charges(&mol);
        let mut h0 = crate::integrals::kinetic_matrix(&mb);
        let v = crate::integrals::nuclear_matrix(&mb, &mol, Some(&charges));
        for (a, b) in h0.as_mut_slice().iter_mut().zip(v.as_slice()) {
            *a += *b;
        }
        let cases: [(&str, [Matrix; 3], [Matrix; 3], &Matrix); 2] = [
            (
                "overlap",
                giao_overlap_derivative_about(&mb, o),
                giao_overlap_derivative_about(&mb, o + delta),
                &s0,
            ),
            (
                "core Hamiltonian",
                giao_core_hamiltonian_derivative(&mol, &mb, o).unwrap(),
                giao_core_hamiltonian_derivative(&mol, &mb, o + delta).unwrap(),
                &h0,
            ),
        ];
        for (name, at_o, at_shifted, zeroth) in &cases {
            for a in 0..3 {
                let gamma: Vec<f64> = centre
                    .iter()
                    .map(|&r| 0.5 * (r - o).cross(delta).get(a))
                    .collect();
                let mut worst = 0.0_f64;
                let mut scale = 0.0_f64;
                for m in 0..n {
                    for p in 0..n {
                        let want = at_o[a][(m, p)] + (gamma[p] - gamma[m]) * zeroth[(m, p)];
                        worst = worst.max((at_shifted[a][(m, p)] - want).abs());
                        scale = scale.max(at_shifted[a][(m, p)].abs());
                    }
                }
                assert!(
                    worst < 1e-9 * scale.max(1.0),
                    "{name} component {a}: shift identity gap {worst:e} (scale {scale:e})"
                );
                assert!(scale > 1e-6, "{name} component {a} is identically zero");
            }
        }
    }

    /// The mixed `B`-`m_N` operator must be SYMMETRIC (it is a second
    /// derivative of a Hermitian matrix element with respect to two real
    /// parameters), and it must obey the same gauge-origin shift identity —
    /// with the nuclear-dipole operator as the zeroth-order factor, since
    /// only the `B` derivative sees the phase.
    #[test]
    fn mixed_field_moment_operator_is_symmetric_and_gauge_covariant() {
        let (_, mb) = water();
        let nucleus = Vec3::new(0.0, 0.0, 0.2217);
        let o = Vec3::new(0.13, 0.41, -0.28);
        let delta = Vec3::new(0.35, -0.19, 0.52);
        let n = mb.nao_sph;
        let centre = ao_centres(&mb);
        let at_o = shielding_mixed_operator(&mb, nucleus, o, true);
        let at_shifted = shielding_mixed_operator(&mb, nucleus, o + delta, true);
        let pso = paramagnetic_operator(&mb, nucleus);
        for a in 0..3 {
            let gamma: Vec<f64> = centre
                .iter()
                .map(|&r| 0.5 * (r - o).cross(delta).get(a))
                .collect();
            for b in 0..3 {
                let mut asym = 0.0_f64;
                let mut worst = 0.0_f64;
                let mut scale = 0.0_f64;
                for m in 0..n {
                    for p in 0..n {
                        asym = asym.max((at_o[a][b][(m, p)] - at_o[a][b][(p, m)]).abs());
                        let want = at_o[a][b][(m, p)] + (gamma[p] - gamma[m]) * pso[b][(m, p)];
                        worst = worst.max((at_shifted[a][b][(m, p)] - want).abs());
                        scale = scale.max(at_o[a][b][(m, p)].abs());
                    }
                }
                assert!(
                    asym < 1e-9 * scale.max(1.0),
                    "component ({a},{b}) is not symmetric: {asym:e} (scale {scale:e})"
                );
                assert!(
                    worst < 1e-8 * scale.max(1.0),
                    "component ({a},{b}): shift identity gap {worst:e} (scale {scale:e})"
                );
            }
        }
    }

    /// On a ONE-CENTRE system with the gauge origin on the atom every
    /// London phase is identically 1, so the GIAO mixed operator must
    /// collapse onto the common-gauge one. An exact equality that pins the
    /// two code paths against each other.
    #[test]
    fn one_centre_giao_collapses_onto_the_common_gauge_operator() {
        let mol = Molecule::new(vec![Atom {
            z: 4,
            position: Vec3::new(0.37, -0.21, 0.58),
        }]);
        let set = data::load("6-31g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let at = mol.atoms[0].position;
        let giao = shielding_mixed_operator(&mb, at, at, true);
        let common = shielding_mixed_operator(&mb, at, at, false);
        let mut scale = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let gap = giao[a][b].max_abs_diff(&common[a][b]);
                for v in giao[a][b].as_slice() {
                    scale = scale.max(v.abs());
                }
                assert!(gap < 1e-12, "component ({a},{b}) differs by {gap:e}");
            }
        }
        assert!(scale > 1e-6, "the operator is identically zero");
        // The GIAO derivative of the overlap and of h likewise reduce.
        let ds = giao_overlap_derivative_about(&mb, at);
        for (a, m) in ds.iter().enumerate() {
            for v in m.as_slice() {
                assert!(
                    v.abs() < 1e-14,
                    "overlap derivative {a} is non-zero one-centre"
                );
            }
        }
        let dh = giao_core_hamiltonian_derivative(&mol, &mb, at).unwrap();
        let lop = angular_momentum_operator(&mb, at);
        for a in 0..3 {
            let mut worst = 0.0_f64;
            for i in 0..mb.nao_sph {
                for j in 0..mb.nao_sph {
                    worst = worst.max((dh[a][(i, j)] + 0.5 * lop[a][(i, j)]).abs());
                }
            }
            assert!(
                worst < 1e-10,
                "dh/dB {a} does not reduce to -L/2: {worst:e}"
            );
        }
    }

    /// Angular momentum about a shifted origin differs from the original
    /// by exactly `-(d x grad)`. An exact identity linking the operator to
    /// the momentum integrals — it catches an origin threaded
    /// inconsistently through the two halves of the curl, which
    /// antisymmetry alone would not.
    #[test]
    fn angular_momentum_origin_shift_is_exact() {
        let (_, mb) = water();
        let d = Vec3::new(0.37, -0.21, 0.58);
        let l1 = angular_momentum_operator(&mb, Vec3::zero());
        let l2 = angular_momentum_operator(&mb, d);
        let grad = gradient_operator(&mb);
        for b in 0..3 {
            let (j, k, _) = curl_terms(b)[0];
            let mut worst = 0.0_f64;
            for r in 0..mb.nao_sph {
                for c in 0..mb.nao_sph {
                    let cross = d.get(j) * grad[k][(r, c)] - d.get(k) * grad[j][(r, c)];
                    worst = worst.max((l2[b][(r, c)] - (l1[b][(r, c)] - cross)).abs());
                }
            }
            assert!(
                worst < 1e-10,
                "origin shift, component {b}: worst {worst:e}"
            );
        }
    }
}
