// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Real Wigner-D matrices `D^(l)(U)` for `l = 0..=4`, rotating the
//! project's real solid harmonics (`basis::aos::cart_to_sph`).
//!
//! # Construction (exact, not fitted)
//!
//! A 3x3 matrix `U` acts linearly on a point `r`, and therefore acts
//! linearly on the space of degree-`l` Cartesian monomials too:
//! `(U r)^α = Σ_β T(U)_{α,β} r^β` for every multi-index `α` of degree `l`
//! (`α` ranges over `basis::cart_components(l)`). `T(U)` is built here by
//! plain multinomial expansion — `monomial_action` — and is a genuine
//! representation of `GL(3)` on that monomial space: `T(U₁U₂) = T(U₁)T(U₂)`
//! follows immediately by substitution (apply `U₂` to `r`, then `U₁` to the
//! result), with no appeal to orthogonality anywhere in the proof.
//!
//! `cart_to_sph(l)` (`C`, `ncart × nsph`) expresses each real solid
//! harmonic `S_m` as `S_m(r) = Σ_α C_{α,m} r^α`. Because every `S_m` is a
//! harmonic polynomial (annihilated by the Laplacian) and the Laplacian
//! commutes with the linear substitution `r ↦ Ur` for orthogonal `U`,
//! `S_m(Ur)` is again a degree-`l` harmonic polynomial, hence (since
//! `cart_to_sph` supplies a full, linearly independent basis of that
//! `(2l+1)`-dimensional space — proved by the orthonormality test in
//! `aos.rs`) it lies **exactly** in `span{S_{m'}}`. Concretely, the
//! monomial-coefficient vector of `S_m(Ur)` is column `m` of `T(U)ᵀ C`, and
//! that vector is guaranteed to lie in `Range(C)`, so solving
//! `C · d = (T(U)ᵀC)[:, m]` for `d` (via the *exact* left inverse of `C`,
//! not a least-squares fit — `C` has full column rank because distinct
//! harmonics are linearly independent polynomials, so `CᵀC` is invertible
//! and `C⁺ = (CᵀC)⁻¹Cᵀ` satisfies `C⁺C = I` exactly) recovers the unique
//! `d = D(U)[m, :]ᵀ`. Collecting all `m`:
//!
//! ```text
//! D(U) = Cᵀ T(U) C (CᵀC)⁻¹
//! ```
//!
//! The "pseudo-inverse" here is doing exact algebra, not approximation: the
//! residual `C d − (T(U)ᵀC)[:,m]` is exactly zero for every `U`, `m`
//! because of the harmonic-invariance argument above, not merely small in a
//! least-squares sense. `(CᵀC)⁻¹` is applied via `linalg::solve_linear_matrix`
//! (a direct LU solve) rather than forming the inverse explicitly.
//!
//! # Transpose convention (resolved, not a free choice)
//!
//! This file's `wigner_d(l, U)` satisfies
//!
//! ```text
//! S_lm(U r) = Σ_m' D^(l)_{m,m'}(U) S_lm'(r)          (no transpose)
//! ```
//!
//! with **no** transpose anywhere — `U` is applied to the point directly.
//! This is not a stylistic pick: it is the only convention consistent with
//! the other two invariants this module also has to satisfy, and demanding
//! the transposed formula `S_lm(Uᵀr) = Σ D_{m,m'}(U) S_lm'(r)` instead
//! breaks both of them:
//!
//! - **Composition.** Substituting `r ↦ U₂r` then `r ↦ U₁r` and comparing
//!   with substituting `r ↦ (U₁U₂)r` directly shows the *no-transpose*
//!   `D` obeys `D(U₁U₂) = D(U₁)D(U₂)` — a genuine homomorphism (see
//!   `composition_law` below, gate 2). Defining `D'(U) := D(Uᵀ)` instead
//!   (to satisfy the transposed formula) turns this into the
//!   *anti*-homomorphism `D'(U₁)D'(U₂) = D'(U₂U₁)`, which fails gate 2
//!   for any non-commuting pair.
//! - **The `l=1` identification.** `cart_to_sph(1)` is exactly a signed
//!   permutation matrix (see below), so `D(U)` for `l=1` is `U` itself
//!   with rows/columns relabelled by that permutation — i.e. genuinely
//!   `U`, not `Uᵀ`. `D'(U) = D(Uᵀ)` would instead reproduce a relabelling
//!   of `Uᵀ`, contradicting "`D^(1)(U)` equals `U` up to the project's
//!   real-harmonic ordering".
//!
//! So: **no transpose**. `wigner_d`'s `U` argument is plugged straight into
//! `S_lm(U r)`.
//!
//! # Real-harmonic ordering (read from `aos.rs`, not guessed)
//!
//! `aos.rs`'s own test (`s_and_p_are_permutations`) pins down the l=1
//! order: Cartesian rows are `(x, y, z)`, columns are `m = 0, +1, −1`, and
//! the table is `x → col 1, y → col 2, z → col 0`. So the p-shell's
//! `(m=0, m=+1, m=−1)` column order corresponds to Cartesian
//! **`(z, x, y)`**, not `(x, y, z)` and not `(y, z, x)`. `l1_matches_u_up_to_ordering`
//! below encodes exactly that permutation and checks it against `wigner_d(1, ·)`.

use crate::basis::aos::cart_to_sph;
use crate::basis::cart_components;
use crate::error::Result;
use crate::linalg::{solve_linear_matrix, Matrix};
use crate::math::Mat3;

/// Highest angular momentum this file builds a Wigner-D for.
pub const MAX_L: u8 = 4;

#[inline]
fn mat3_entry(u: &Mat3, row: usize, col: usize) -> f64 {
    u.col[col].get(row)
}

/// Position of the exponent tuple `t` within `comps` (linear search — these
/// lists have at most 15 entries for `l ≤ 4`, so this is never a hot path).
fn comp_index(comps: &[(i32, i32, i32)], t: (i32, i32, i32)) -> usize {
    comps
        .iter()
        .position(|&c| c == t)
        .expect("monomial exponent missing from cart_components — degree mismatch")
}

/// Multiply a degree-`d` polynomial (coefficients over `cart_components(d)`)
/// by the linear form `cx·x + cy·y + cz·z`, returning the coefficients of
/// the degree-`(d+1)` product over `cart_components(d+1)`.
fn mul_by_linear(coeffs: &[f64], d: u8, (cx, cy, cz): (f64, f64, f64)) -> Vec<f64> {
    let comps_d = cart_components(d);
    let comps_d1 = cart_components(d + 1);
    let mut out = vec![0.0_f64; comps_d1.len()];
    for (idx, &(i, j, k)) in comps_d.iter().enumerate() {
        let c = coeffs[idx];
        if c == 0.0 {
            continue;
        }
        out[comp_index(&comps_d1, (i + 1, j, k))] += c * cx;
        out[comp_index(&comps_d1, (i, j + 1, k))] += c * cy;
        out[comp_index(&comps_d1, (i, j, k + 1))] += c * cz;
    }
    out
}

/// Expand `(U r)_x^{ax} (U r)_y^{ay} (U r)_z^{az}` into monomial
/// coefficients over `cart_components(ax+ay+az)`, by repeated
/// multiplication of the three linear forms `(Ur)_x`, `(Ur)_y`, `(Ur)_z`
/// (the rows of `U`).
fn expand_rotated_monomial(u: &Mat3, ax: i32, ay: i32, az: i32) -> Vec<f64> {
    let lx = (
        mat3_entry(u, 0, 0),
        mat3_entry(u, 0, 1),
        mat3_entry(u, 0, 2),
    );
    let ly = (
        mat3_entry(u, 1, 0),
        mat3_entry(u, 1, 1),
        mat3_entry(u, 1, 2),
    );
    let lz = (
        mat3_entry(u, 2, 0),
        mat3_entry(u, 2, 1),
        mat3_entry(u, 2, 2),
    );
    let mut coeffs = vec![1.0_f64];
    let mut d = 0u8;
    for _ in 0..ax {
        coeffs = mul_by_linear(&coeffs, d, lx);
        d += 1;
    }
    for _ in 0..ay {
        coeffs = mul_by_linear(&coeffs, d, ly);
        d += 1;
    }
    for _ in 0..az {
        coeffs = mul_by_linear(&coeffs, d, lz);
        d += 1;
    }
    coeffs
}

/// The `ncart × ncart` representation of `U` acting on degree-`l`
/// monomials: `(U r)^α = Σ_β T_{α,β} r^β`. A genuine representation of
/// `GL(3)` restricted to degree-`l` monomials — `T(U₁U₂) = T(U₁)T(U₂)` —
/// which is exactly what makes `wigner_d`'s composition law (gate 2) hold.
fn monomial_action(l: u8, u: &Mat3) -> Matrix {
    let comps = cart_components(l);
    let n = comps.len();
    let mut t = Matrix::zeros(n, n);
    for (row, &(ax, ay, az)) in comps.iter().enumerate() {
        let poly = expand_rotated_monomial(u, ax, ay, az);
        for (col, &c) in poly.iter().enumerate() {
            t[(row, col)] = c;
        }
    }
    t
}

/// The real Wigner-D matrix `D^(l)(U)`, `(2l+1) × (2l+1)`, satisfying
/// `S_lm(U r) = Σ_m' D_{m,m'}(U) S_lm'(r)` (see the module docs for why this
/// is the correct, non-transposed convention, and why the pseudo-inverse
/// construction below is exact rather than a least-squares fit).
///
/// `l` must be `<= MAX_L` (4); this is a project-internal helper over a
/// fixed, small range, not a general-purpose API, so an out-of-range `l`
/// panics rather than returning `DftError` — the same tradeoff `aos.rs`
/// makes for `cart_to_sph`.
pub fn wigner_d(l: u8, u: &Mat3) -> Result<Matrix> {
    assert!(l <= MAX_L, "wigner_d: l={l} exceeds MAX_L={MAX_L}");
    let c = cart_to_sph(l);
    let t = monomial_action(l, u);
    let tc = t.matmul(c);
    let ct_tc = c.transpose_matmul(&tc);
    let ct_c = c.transpose_matmul(c);
    // D = (Cᵀ T C)(CᵀC)⁻¹. Solve (CᵀC) Dᵀ = (Cᵀ T C)ᵀ for Dᵀ instead of
    // forming (CᵀC)⁻¹ explicitly — CᵀC is symmetric positive definite (C
    // has full column rank, see module docs), so this is a well-posed LU
    // solve either way, just without materializing the inverse.
    let d_t = solve_linear_matrix(&ct_c, &ct_tc.transpose())?;
    Ok(d_t.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::n_sph;
    use crate::math::Vec3;

    /// Tiny inline linear congruential generator (Numerical-Recipes-style
    /// constants) — deterministic, fixed-seed, no external `rand` crate
    /// (dependencies are frozen) and no `Math.random`-style nondeterminism,
    /// per the project's testing rules.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next_f64()
        }
    }

    fn rotate_vec(axis: Vec3, angle: f64, v: Vec3) -> Vec3 {
        v * angle.cos() + axis.cross(v) * angle.sin() + axis * (axis.dot(v) * (1.0 - angle.cos()))
    }

    /// Rodrigues' rotation formula: a proper (orthogonal, det=+1) rotation
    /// by `angle` about (unit-normalized) `axis`.
    fn rotation_from_axis_angle(axis: Vec3, angle: f64) -> Mat3 {
        let ax = axis.normalized();
        Mat3::from_columns(
            rotate_vec(ax, angle, Vec3::new(1.0, 0.0, 0.0)),
            rotate_vec(ax, angle, Vec3::new(0.0, 1.0, 0.0)),
            rotate_vec(ax, angle, Vec3::new(0.0, 0.0, 1.0)),
        )
    }

    fn sample_rotations(seed: u64, n: usize) -> Vec<Mat3> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut axis = Vec3::new(
                    rng.next_range(-1.0, 1.0),
                    rng.next_range(-1.0, 1.0),
                    rng.next_range(-1.0, 1.0),
                );
                if axis.norm() < 1e-3 {
                    axis = Vec3::new(1.0, 0.0, 0.0);
                }
                let angle = rng.next_range(0.05, 2.0 * std::f64::consts::PI - 0.05);
                rotation_from_axis_angle(axis, angle)
            })
            .collect()
    }

    fn sample_points(seed: u64, n: usize) -> Vec<Vec3> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                Vec3::new(
                    rng.next_range(-2.0, 2.0),
                    rng.next_range(-2.0, 2.0),
                    rng.next_range(-2.0, 2.0),
                )
            })
            .collect()
    }

    fn mat3_matmul(a: &Mat3, b: &Mat3) -> Mat3 {
        Mat3::from_columns(
            a.mul_vec(b.col[0]),
            a.mul_vec(b.col[1]),
            a.mul_vec(b.col[2]),
        )
    }

    /// Evaluate `S_lm` (harmonic column `m_col`) at `r`, through the exact
    /// same `cart_to_sph` path `wigner_d` itself uses — so this gate tests
    /// `D` against the project's own definition of the harmonics, not a
    /// second, independently-invented one.
    fn eval_harmonic(l: u8, m_col: usize, r: Vec3) -> f64 {
        let c = cart_to_sph(l);
        let comps = cart_components(l);
        comps
            .iter()
            .enumerate()
            .map(|(row, &(ax, ay, az))| {
                c[(row, m_col)] * r.x.powi(ax) * r.y.powi(ay) * r.z.powi(az)
            })
            .sum()
    }

    /// Gate 1: the defining property, `S_lm(U r) = Σ_m' D_{m,m'}(U) S_lm'(r)`
    /// (see module docs for why no transpose appears), l=0..=4, 8 rotations
    /// x 20 points x (2l+1) harmonics each.
    #[test]
    fn defining_property_matches_rotated_harmonics() {
        let rotations = sample_rotations(0xC0FFEE_u64, 8);
        let points = sample_points(0xFACADE_u64, 20);
        let mut worst = 0.0_f64;
        for l in 0..=MAX_L {
            let ns = n_sph(l);
            for u in &rotations {
                let d = wigner_d(l, u).unwrap();
                for &r in &points {
                    let ur = u.mul_vec(r);
                    for m in 0..ns {
                        let lhs = eval_harmonic(l, m, ur);
                        let rhs: f64 = (0..ns).map(|mp| d[(m, mp)] * eval_harmonic(l, mp, r)).sum();
                        worst = worst.max((lhs - rhs).abs());
                    }
                }
            }
        }
        assert!(
            worst < 1e-12,
            "gate1 worst-case |S(Ur) - D S(r)| = {worst:e}"
        );
        eprintln!("gate1 worst-case error: {worst:e}");
    }

    /// Gate 2: composition law, `D(U1)D(U2) == D(U1 U2)`, l=0..=4, all 64
    /// pairs among 8 rotations.
    #[test]
    fn composition_law() {
        let rotations = sample_rotations(0xBEEF_u64, 8);
        let mut worst = 0.0_f64;
        for l in 0..=MAX_L {
            for u1 in &rotations {
                let d1 = wigner_d(l, u1).unwrap();
                for u2 in &rotations {
                    let d2 = wigner_d(l, u2).unwrap();
                    let u12 = mat3_matmul(u1, u2);
                    let d12 = wigner_d(l, &u12).unwrap();
                    let lhs = d1.matmul(&d2);
                    worst = worst.max(lhs.max_abs_diff(&d12));
                }
            }
        }
        assert!(
            worst < 1e-12,
            "gate2 worst-case |D(U1)D(U2) - D(U1U2)| = {worst:e}"
        );
        eprintln!("gate2 worst-case error: {worst:e}");
    }

    /// Gate 3: orthogonality, `D(U)ᵀ D(U) == I`, l=0..=4, 8 rotations.
    #[test]
    fn orthogonality() {
        let rotations = sample_rotations(0x1234_u64, 8);
        let mut worst = 0.0_f64;
        for l in 0..=MAX_L {
            let id = Matrix::identity(n_sph(l));
            for u in &rotations {
                let d = wigner_d(l, u).unwrap();
                let dtd = d.transpose_matmul(&d);
                worst = worst.max(dtd.max_abs_diff(&id));
            }
        }
        assert!(worst < 1e-12, "gate3 worst-case |DᵀD - I| = {worst:e}");
        eprintln!("gate3 worst-case error: {worst:e}");
    }

    /// Gate 4: `D^(0)(U) == [1]` exactly, and `D^(1)(U)` equals `U`
    /// relabelled by the project's real-harmonic ordering `(z, x, y)`
    /// (read from `aos.rs`'s own `s_and_p_are_permutations` test — see
    /// module docs).
    #[test]
    fn l0_is_identity_and_l1_matches_u_up_to_ordering() {
        let rotations = sample_rotations(0x5678_u64, 8);
        let mut worst_l0 = 0.0_f64;
        let mut worst_l1 = 0.0_f64;
        // harmonic column m (0,1,2 = m=0,+1,-1) -> Cartesian row/col index.
        let perm = [2usize, 0, 1]; // m=0 -> z(2), m=+1 -> x(0), m=-1 -> y(1)
        for u in &rotations {
            let d0 = wigner_d(0, u).unwrap();
            worst_l0 = worst_l0.max((d0[(0, 0)] - 1.0).abs());

            let d1 = wigner_d(1, u).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    let expect = mat3_entry(u, perm[i], perm[j]);
                    worst_l1 = worst_l1.max((d1[(i, j)] - expect).abs());
                }
            }
        }
        assert_eq!(worst_l0, 0.0, "D^(0)(U) must be exactly [1]");
        assert!(
            worst_l1 < 1e-12,
            "gate4 worst-case |D^(1)(U) - perm(U)| = {worst_l1:e}"
        );
        eprintln!("gate4 worst-case l0 error: {worst_l0:e}, l1 error: {worst_l1:e}");
    }

    /// Gate 5: `det D^(l)(U) == +1` for a proper rotation `U`, l=0..=4.
    #[test]
    fn determinant_is_plus_one() {
        fn determinant(m: &Matrix) -> f64 {
            let n = m.rows;
            let mut a: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| m[(i, j)]).collect())
                .collect();
            let mut det = 1.0_f64;
            for col in 0..n {
                let mut pivot_row = col;
                let mut pivot_val = a[col][col].abs();
                for row in (col + 1)..n {
                    if a[row][col].abs() > pivot_val {
                        pivot_val = a[row][col].abs();
                        pivot_row = row;
                    }
                }
                if pivot_val < 1e-300 {
                    return 0.0;
                }
                if pivot_row != col {
                    a.swap(pivot_row, col);
                    det = -det;
                }
                det *= a[col][col];
                for row in (col + 1)..n {
                    let factor = a[row][col] / a[col][col];
                    for k in col..n {
                        a[row][k] -= factor * a[col][k];
                    }
                }
            }
            det
        }

        let rotations = sample_rotations(0x9ABC_u64, 8);
        let mut worst = 0.0_f64;
        for l in 0..=MAX_L {
            for u in &rotations {
                let d = wigner_d(l, u).unwrap();
                let det = determinant(&d);
                worst = worst.max((det - 1.0).abs());
            }
        }
        assert!(worst < 1e-9, "gate5 worst-case |det D - 1| = {worst:e}");
        eprintln!("gate5 worst-case error: {worst:e}");
    }
}
