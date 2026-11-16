// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Dense linear algebra: a row-major `Matrix` wrapper plus the solvers the
//! SCF needs.
//!
//! The heavy O(n³) work — symmetric eigendecomposition, matmul, LU solve —
//! is delegated to **faer** (pure Rust; no LAPACK/BLAS). All faer calls in
//! the crate live in this module so a faer upgrade touches one file.
//!
//! Unlike the orthonormal-AO semiempirical ancestors, a Gaussian AO basis is
//! non-orthogonal: this module additionally provides the canonical
//! orthogonalizer `X` (with linear-dependence removal) and the Roothaan
//! generalized eigensolve `F C = S C ε` built on it.

use crate::error::{DftError, Result};
use std::sync::OnceLock;

/// Error-compensated accumulation for cancellation-sensitive residuals and
/// integral contractions. Product rounding is recovered with an IEEE FMA.
#[derive(Clone, Copy, Default)]
pub(crate) struct AccurateSum {
    sum: f64,
    correction: f64,
    correction_error: f64,
}

impl AccurateSum {
    #[inline]
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        let residual = if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        let corrected = self.correction + residual;
        let lost = if self.correction.abs() >= residual.abs() {
            (self.correction - corrected) + residual
        } else {
            (residual - corrected) + self.correction
        };
        self.correction_error = Self::upper_add(self.correction_error, lost.abs());
        self.correction = corrected;
        self.sum = next;
    }

    /// Upward-rounded addition of nonnegative bounds. An exact zero does
    /// not introduce an absolute floor; only an observed loss enlarges it.
    #[inline]
    fn upper_add(a: f64, b: f64) -> f64 {
        if b == 0.0 {
            return a;
        }
        let value = a + b;
        if value.is_finite() {
            f64::from_bits(value.to_bits() + 1)
        } else {
            value
        }
    }

    #[inline]
    pub(crate) fn add_product(&mut self, a: f64, b: f64) {
        let product = a * b;
        self.add(product);
        self.add(a.mul_add(b, -product));
        if a != 0.0
            && b != 0.0
            && product.abs() < 2.0 * f64::MIN_POSITIVE / f64::EPSILON
            && Self::product_loses_subnormal_bits(a, b)
        {
            self.correction_error = Self::upper_add(self.correction_error, f64::from_bits(1));
        }
    }

    /// Even a normal product can have an unrepresentable FMA remainder.
    /// Decode only that tiny range; the integer significand product is
    /// exact in u128 and identifies bits below the subnormal quantum.
    #[inline]
    fn product_loses_subnormal_bits(a: f64, b: f64) -> bool {
        let decode = |x: f64| {
            let bits = x.to_bits() & 0x7fff_ffff_ffff_ffff;
            let field = (bits >> 52) as i32;
            let fraction = bits & ((1u64 << 52) - 1);
            if field == 0 {
                (fraction, -1074)
            } else {
                (fraction | (1u64 << 52), field - 1023 - 52)
            }
        };
        let (ma, ea) = decode(a);
        let (mb, eb) = decode(b);
        let product = u128::from(ma) * u128::from(mb);
        ea + eb + (product.trailing_zeros() as i32) < -1074
    }

    /// Rounded value and its retained low part. The general two-sum
    /// transform also handles severe cancellation of the leading sum.
    #[inline]
    pub(crate) fn parts(self) -> (f64, f64) {
        let high = self.sum + self.correction;
        let recovered = high - self.sum;
        let low = (self.sum - (high - recovered)) + (self.correction - recovered);
        (high, low)
    }

    #[inline]
    pub(crate) fn value(self) -> f64 {
        self.sum + self.correction
    }

    /// A posteriori absolute error of the represented dot product: the
    /// final low part plus upward-rounded actual correction losses. Input
    /// approximation errors remain separate. Inlined callers requesting
    /// only `value` or `parts` need not retain the unused error tracking.
    #[inline]
    pub(crate) fn roundoff_bound(self) -> f64 {
        Self::upper_add(self.parts().1.abs(), self.correction_error)
    }
}

#[cfg(test)]
mod compensated_sum_tests {
    use super::AccurateSum;

    #[test]
    fn actual_correction_losses_bound_a_cascaded_cancellation() {
        let theta = 2.0_f64.powi(-54);
        let mut sum = AccurateSum::default();
        for value in [1.0, theta, theta * theta, -theta, -theta * theta, -1.0] {
            sum.add_product(value, 1.0);
        }
        // The exact represented sum is zero. The second-level addition
        // loses theta^2, which must be covered by the recorded bound.
        assert_eq!(sum.value(), -theta * theta);
        assert!(sum.roundoff_bound() >= sum.value().abs());
        assert!(sum.roundoff_bound() < 2.0 * theta * theta);
    }

    #[test]
    fn only_observed_underflow_adds_an_absolute_bound() {
        let mut zero = AccurateSum::default();
        zero.add_product(0.0, 0.5);
        assert_eq!(zero.roundoff_bound(), 0.0);
        let mut tiny = AccurateSum::default();
        tiny.add_product(f64::from_bits(1), 0.5);
        assert_eq!(tiny.value(), 0.0);
        assert!(tiny.roundoff_bound() >= f64::from_bits(1));
        let mut exact_normal = AccurateSum::default();
        exact_normal.add_product(f64::MIN_POSITIVE, 1.0);
        assert_eq!(exact_normal.roundoff_bound(), 0.0);
        let mut normal = AccurateSum::default();
        normal.add_product(
            f64::from_bits(f64::MIN_POSITIVE.to_bits() + 1),
            f64::from_bits(1.0_f64.to_bits() + 1),
        );
        assert!(normal.value().is_normal());
        assert!(normal.roundoff_bound() >= f64::from_bits(1));
    }
}

/// Switch faer's dense linear algebra (eigensolves, matmuls, factorizations)
/// to the multi-threaded path once, lazily. faer 0.24 defaults to sequential;
/// `Par::rayon(0)` uses the shared rayon global pool so it composes with the
/// crate's own `par_iter` loops instead of oversubscribing. Numerically
/// identical — only the scheduling changes. `DFT_RS_FAER_THREADS=1` (or `0`)
/// forces the sequential path for benchmarking/reproducibility.
#[inline]
pub fn ensure_parallelism() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let seq = std::env::var("DFT_RS_FAER_THREADS")
            .ok()
            .map(|v| {
                let v = v.trim();
                v == "0" || v == "1"
            })
            .unwrap_or(false);
        let par = if seq {
            faer::Par::Seq
        } else {
            faer::Par::rayon(0)
        };
        faer::set_global_parallelism(par);
    });
}

fn parallelism() -> faer::Par {
    let default = faer::get_global_parallelism();
    match (default, crate::resources::current_usage()) {
        (faer::Par::Rayon(_), Some(scope)) => {
            faer::Par::rayon(default.degree().min(scope.cpu_threads))
        }
        _ => default,
    }
}

/// Row-major dense matrix.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
    _allocation: crate::resources::MemoryReservation,
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("data", &self.data)
            .finish()
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let allocation = crate::resources::reserve_or_unwind(
            &[self.rows, self.cols, std::mem::size_of::<f64>()],
            "matrix copy",
        );
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
            _allocation: allocation,
        }
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let allocation = crate::resources::reserve_or_unwind(
            &[rows, cols, std::mem::size_of::<f64>()],
            "dense matrix",
        );
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            _allocation: allocation,
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        let allocation = crate::resources::reserve_or_unwind(
            &[data.capacity(), std::mem::size_of::<f64>()],
            "imported dense matrix",
        );
        assert_eq!(rows * cols, data.len(), "matrix data length mismatch");
        Self {
            rows,
            cols,
            data,
            _allocation: allocation,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// Largest absolute element difference (test/convergence helper).
    pub fn max_abs_diff(&self, other: &Matrix) -> f64 {
        debug_assert_eq!(self.data.len(), other.data.len());
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// Standard `self · other` via faer's blocked/SIMD kernel operating on
    /// **borrowed views** of the row-major data (no per-call copy), at the
    /// global parallelism. Dominant large-N SCF cost.
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        self.matmul_par(other, parallelism())
    }

    /// Sequential matmul (faer kernel, `Par::Seq`) on borrowed views. Use
    /// inside an already-parallel loop (e.g. per-perturbation CPKS solves):
    /// the outer loop supplies the parallelism, so each matmul must stay
    /// single-threaded.
    pub fn matmul_seq(&self, other: &Matrix) -> Matrix {
        self.matmul_par(other, faer::Par::Seq)
    }

    /// Compensated contractions for small residuals in a nonorthogonal metric.
    /// Ordinary large matrix operations continue to use the faer kernels.
    pub(crate) fn matmul_accurate(&self, other: &Matrix, transpose_self: bool) -> Matrix {
        let (rows, inner) = if transpose_self {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        assert_eq!(inner, other.rows, "matmul dimension mismatch");
        let mut out = Matrix::zeros(rows, other.cols);
        for i in 0..rows {
            for j in 0..other.cols {
                let mut sum = AccurateSum::default();
                for k in 0..inner {
                    sum.add_product(
                        if transpose_self {
                            self[(k, i)]
                        } else {
                            self[(i, k)]
                        },
                        other[(k, j)],
                    );
                }
                out[(i, j)] = sum.value();
            }
        }
        out
    }

    /// Error-compensated `self^T operator self`. Retain the low part of
    /// `operator self` until the final contraction; separate rounded matrix
    /// products can lose a small physical commutator or metric coefficient.
    pub(crate) fn congruence_accurate(&self, operator: &Matrix) -> Matrix {
        assert_eq!((operator.rows, operator.cols), (self.rows, self.rows));
        let mut high = Matrix::zeros(self.rows, self.cols);
        let mut low = high.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut sum = AccurateSum::default();
                for k in 0..self.rows {
                    sum.add_product(operator[(i, k)], self[(k, j)]);
                }
                (high[(i, j)], low[(i, j)]) = sum.parts();
            }
        }
        let mut value = Matrix::zeros(self.cols, self.cols);
        for i in 0..self.cols {
            for j in 0..self.cols {
                let mut sum = AccurateSum::default();
                for k in 0..self.rows {
                    sum.add_product(self[(k, i)], high[(k, j)]);
                    sum.add_product(self[(k, i)], low[(k, j)]);
                }
                value[(i, j)] = sum.value();
            }
        }
        value
    }

    /// Compute `self^T · other` without materializing `self^T`.
    pub fn transpose_matmul_seq(&self, other: &Matrix) -> Matrix {
        self.matmul_with_transposes(other, true, false, faer::Par::Seq)
    }

    /// Compute `self · other^T` without materializing `other^T`.
    pub fn matmul_transpose_seq(&self, other: &Matrix) -> Matrix {
        self.matmul_with_transposes(other, false, true, faer::Par::Seq)
    }

    /// Compute `self^T · other` at global parallelism.
    pub fn transpose_matmul(&self, other: &Matrix) -> Matrix {
        self.matmul_with_transposes(other, true, false, parallelism())
    }

    fn matmul_par(&self, other: &Matrix, par: faer::Par) -> Matrix {
        self.matmul_with_transposes(other, false, false, par)
    }

    fn matmul_with_transposes(
        &self,
        other: &Matrix,
        transpose_self: bool,
        transpose_other: bool,
        par: faer::Par,
    ) -> Matrix {
        let (m, k_left) = if transpose_self {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        let (k_right, n) = if transpose_other {
            (other.cols, other.rows)
        } else {
            (other.rows, other.cols)
        };
        assert_eq!(k_left, k_right, "matmul dimension mismatch");
        let k = k_left;
        let mut out = Matrix::zeros(m, n);
        if m == 0 || n == 0 || k == 0 {
            return out;
        }
        let a = faer::MatRef::from_row_major_slice(&self.data, self.rows, self.cols);
        let b = faer::MatRef::from_row_major_slice(&other.data, other.rows, other.cols);
        let a = if transpose_self { a.transpose() } else { a };
        let b = if transpose_other { b.transpose() } else { b };
        let c = faer::MatMut::from_row_major_slice_mut(out.data.as_mut_slice(), m, n);
        faer::linalg::matmul::matmul(c, faer::Accum::Replace, a, b, 1.0, par);
        out
    }

    /// Return `scale · C C^T`, where `C` is the leading `count` columns
    /// (density build `P = f · C_occ C_occᵀ` without dense temporaries).
    pub fn leading_columns_gram(&self, count: usize, scale: f64) -> Matrix {
        assert!(count <= self.cols, "leading column count out of bounds");
        let mut out = Matrix::zeros(self.rows, self.rows);
        if self.rows == 0 || count == 0 {
            return out;
        }
        let full = faer::MatRef::from_row_major_slice(&self.data, self.rows, self.cols);
        let columns = full.subcols(0, count);
        let target =
            faer::MatMut::from_row_major_slice_mut(out.data.as_mut_slice(), self.rows, self.rows);
        faer::linalg::matmul::matmul(
            target,
            faer::Accum::Replace,
            columns,
            columns.transpose(),
            scale,
            parallelism(),
        );
        out
    }

    /// Frobenius inner product `Σ_ij A_ij B_ij` (energies `½ Σ P(H+F)`, DIIS
    /// B-matrix elements).
    pub fn frobenius_dot(&self, other: &Matrix) -> f64 {
        debug_assert_eq!(self.data.len(), other.data.len());
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}
impl std::ops::IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

/// Physical orthonormal-metric commutator, evaluated as A - A^T with
/// A = (F X)^T (P S X). Factoring before the subtraction avoids amplifying
/// large AO-space cancellation by the inverse overlap metric afterwards.
pub(crate) fn metric_commutator(f: &Matrix, p: &Matrix, s: &Matrix, x: &Matrix) -> Matrix {
    let fx = f.matmul_accurate(x, false);
    let sx = s.matmul_accurate(x, false);
    let psx = p.matmul_accurate(&sx, false);
    let a = fx.matmul_accurate(&psx, true);
    let mut out = Matrix::zeros(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..i {
            let value = a[(i, j)] - a[(j, i)];
            out[(i, j)] = value;
            out[(j, i)] = -value;
        }
    }
    out
}

/// Symmetric eigendecomposition (faer, pure Rust — no LAPACK/BLAS).
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvalues in **ascending**
/// order and eigenvectors as the **columns** of the returned matrix, so
/// `A = V diag(λ) Vᵀ`. faer's ordering is not guaranteed ascending; the
/// aufbau occupation relies on it, so it is enforced here.
pub fn symmetric_eigen(a: &Matrix) -> Result<(Vec<f64>, Matrix)> {
    let n = a.rows;
    if a.cols != n {
        return Err(DftError::LinearAlgebra(
            "symmetric_eigen requires a square matrix".to_string(),
        ));
    }
    if n == 0 {
        return Ok((Vec::new(), Matrix::zeros(0, 0)));
    }
    let _workspace = dense_solver_workspace(n)?;
    let fa = faer::MatRef::from_row_major_slice(a.as_slice(), n, n);
    let eigen = fa
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| DftError::LinearAlgebra(format!("faer eigendecomposition failed: {e:?}")))?;
    let s = eigen.S();
    let u = eigen.U();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| s[i].partial_cmp(&s[j]).unwrap_or(std::cmp::Ordering::Equal));
    let values: Vec<f64> = order.iter().map(|&k| s[k]).collect();
    let mut vectors = Matrix::zeros(n, n);
    for (new_col, &old_col) in order.iter().enumerate() {
        for i in 0..n {
            vectors[(i, new_col)] = u[(i, old_col)];
        }
    }
    Ok((values, vectors))
}

/// Inverse of a complex Cartesian tensor after scaling by its maximum norm.
/// The returned pair is `(inverse(A/scale), scale)`: callers form ratios
/// without constructing an unrepresentably large physical inverse. The
/// determinant test matches the real transport inverse condition; scaling
/// does not add a floor or change the operator.
pub(crate) fn normalized_complex3_inverse(
    a: &[[crate::response::complex::Complex; 3]; 3],
) -> Result<Option<([[crate::response::complex::Complex; 3]; 3], f64)>> {
    use crate::response::complex::Complex;
    use faer::linalg::solvers::Solve;
    let _workspace = dense_solver_workspace(6)?;
    if a.iter().flatten().any(|z| !z.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "complex transport tensor must be finite".into(),
        ));
    }
    let scale = a.iter().flatten().map(|z| z.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(None);
    }
    let m = a.map(|row| row.map(|z| z / scale));
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() <= 1e-13 {
        return Ok(None);
    }
    // Real representation [Re A, -Im A; Im A, Re A] of the complex solve.
    let real = faer::Mat::<f64>::from_fn(6, 6, |i, j| {
        let z = m[i % 3][j % 3];
        match (i < 3, j < 3) {
            (true, true) | (false, false) => z.re,
            (true, false) => -z.im,
            (false, true) => z.im,
        }
    });
    let identity = faer::Mat::<f64>::from_fn(6, 3, |i, j| if i == j { 1.0 } else { 0.0 });
    let inverse = real.partial_piv_lu().solve(&identity);
    let result: [[Complex; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| Complex::new(inverse[(i, j)], inverse[(i + 3, j)]))
    });
    if result.iter().flatten().any(|z| !z.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "complex transport inverse exceeds the finite floating-point range".into(),
        ));
    }
    Ok(Some((result, scale)))
}

/// Solve `A x = b` via faer's partial-pivot LU (pure Rust — no LAPACK/BLAS).
///
/// faer's LU does not error on a singular system (it returns a huge,
/// low-residual solution); reject non-finite or unreasonably large solutions
/// so DIIS-style callers using `.ok()` fall back gracefully.
pub fn solve_linear(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    let n = a.rows;
    if a.cols != n || b.len() != n {
        return Err(DftError::LinearAlgebra(
            "solve_linear dimension mismatch".to_string(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    use faer::linalg::solvers::Solve;
    let _workspace = dense_solver_workspace(n)?;
    let fa = faer::MatRef::from_row_major_slice(a.as_slice(), n, n);
    let rhs = faer::Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
    let lu = fa.partial_piv_lu();
    let x = lu.solve(&rhs);
    let sol: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();
    let bmax = b.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
    if sol.iter().any(|v| !v.is_finite()) || sol.iter().any(|v| v.abs() > 1.0e8 * bmax) {
        return Err(DftError::LinearAlgebra(
            "singular or ill-conditioned linear solve".to_string(),
        ));
    }
    Ok(sol)
}

/// Solve `A X = B` for several right-hand sides with one LU
/// factorization.  This is substantially cheaper than calling
/// [`solve_linear`] once per column and is used to prepare reusable linear
/// response operators such as PCM `K^{-1}R`.
pub fn solve_linear_matrix(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    let n = a.rows;
    if a.cols != n || b.rows != n {
        return Err(DftError::LinearAlgebra(
            "solve_linear_matrix dimension mismatch".to_string(),
        ));
    }
    if n == 0 || b.cols == 0 {
        return Ok(Matrix::zeros(n, b.cols));
    }
    use faer::linalg::solvers::Solve;
    let _workspace = dense_solver_workspace(n.max(b.cols))?;
    let fa = faer::MatRef::from_row_major_slice(a.as_slice(), n, n);
    let rhs = faer::Mat::<f64>::from_fn(n, b.cols, |i, j| b[(i, j)]);
    let x = fa.partial_piv_lu().solve(&rhs);
    let mut out = Matrix::zeros(n, b.cols);
    for i in 0..n {
        for j in 0..b.cols {
            out[(i, j)] = x[(i, j)];
        }
    }
    let bmax = b
        .as_slice()
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()))
        .max(1.0);
    if out.as_slice().iter().any(|value| !value.is_finite())
        || out
            .as_slice()
            .iter()
            .any(|value| value.abs() > 1.0e8 * bmax)
    {
        return Err(DftError::LinearAlgebra(
            "matrix solve is singular or ill-conditioned".to_string(),
        ));
    }
    Ok(out)
}

fn dense_solver_workspace(n: usize) -> Result<crate::resources::MemoryReservation> {
    let padded = n
        .checked_add(8)
        .ok_or_else(|| DftError::InvalidInput("dense solver workspace size overflows".into()))?;
    let bytes = crate::resources::checked_bytes(
        &[padded, padded, 64, std::mem::size_of::<f64>()],
        "dense solver workspace allowance",
    )?;
    crate::resources::reserve(bytes, "dense solver workspace allowance")
}

/// Canonical orthogonalizer for a non-orthogonal AO overlap matrix `S`.
///
/// `X = U_kept · diag(1/√λ)` over the eigenpairs of `S` with `λ > thresh`,
/// so `Xᵀ S X = I` on the retained subspace. Columns with `λ ≤ thresh`
/// (near-linear dependence, e.g. diffuse-heavy bases) are dropped; the SCF
/// then runs in the `n_kept`-dimensional orthonormal basis.
#[derive(Clone, Debug)]
pub struct Orthogonalizer {
    /// `nao × n_kept` transformation, `Xᵀ S X = I`.
    pub x: Matrix,
    /// Retained dimension (`= nao − n_dropped`).
    pub n_kept: usize,
    /// Number of near-null overlap eigenvectors removed.
    pub n_dropped: usize,
    /// Smallest retained overlap eigenvalue (conditioning diagnostic).
    pub min_kept_eigenvalue: f64,
}

impl Orthogonalizer {
    /// Build from the overlap matrix. `thresh` is the eigenvalue cutoff
    /// (typical: 1e-7; PySCF default is comparable).
    pub fn canonical(s: &Matrix, thresh: f64) -> Result<Self> {
        let n = s.rows;
        if s.cols != n {
            return Err(DftError::LinearAlgebra(
                "orthogonalizer requires a square overlap matrix".to_string(),
            ));
        }
        let (vals, vecs) = symmetric_eigen(s)?;
        if let Some(&lowest) = vals.first() {
            if lowest < -1e-8 {
                return Err(DftError::LinearAlgebra(format!(
                    "overlap matrix has a significantly negative eigenvalue ({lowest:.3e}); \
                     the AO overlap is broken (integral bug or corrupted basis)"
                )));
            }
        }
        let kept: Vec<usize> = (0..n).filter(|&i| vals[i] > thresh).collect();
        let n_kept = kept.len();
        if n_kept == 0 && n > 0 {
            return Err(DftError::LinearAlgebra(
                "all overlap eigenvalues fall below the linear-dependence threshold".to_string(),
            ));
        }
        let mut x = Matrix::zeros(n, n_kept);
        let mut min_kept = f64::INFINITY;
        for (col, &idx) in kept.iter().enumerate() {
            let inv_sqrt = 1.0 / vals[idx].sqrt();
            min_kept = min_kept.min(vals[idx]);
            for row in 0..n {
                x[(row, col)] = vecs[(row, idx)] * inv_sqrt;
            }
        }
        // Refine the metric of the retained eigenspace, without changing its
        // rank or cutoff. Eigenvector roundoff amplified by small retained
        // eigenvalues can otherwise set a false SCF residual lower bound.
        for _ in 0..2 {
            let gram = x.matmul_accurate(&s.matmul_accurate(&x, false), true);
            let mut correction = Matrix::identity(n_kept);
            for i in 0..n_kept {
                for j in 0..n_kept {
                    correction[(i, j)] -= 0.5 * (gram[(i, j)] - if i == j { 1.0 } else { 0.0 });
                }
            }
            x = x.matmul_accurate(&correction, false);
        }
        Ok(Self {
            x,
            n_kept,
            n_dropped: n - n_kept,
            min_kept_eigenvalue: if n_kept == 0 { 0.0 } else { min_kept },
        })
    }

    /// Solve the Roothaan generalized eigenproblem `F C = S C ε` through this
    /// orthogonalizer: diagonalize `F' = Xᵀ F X`, back-transform `C = X C'`.
    ///
    /// Returns `(ε, C)` with ε ascending and `C` an `nao × n_kept` matrix of
    /// MO coefficient columns satisfying `Cᵀ S C = I`.
    pub fn solve_roothaan(&self, f: &Matrix) -> Result<(Vec<f64>, Matrix)> {
        let fp = self.x.transpose_matmul(&f.matmul(&self.x));
        let (eps, cp) = symmetric_eigen(&fp)?;
        let c = self.x.matmul(&cp);
        Ok((eps, c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_products_retain_small_terms_and_product_roundoff() {
        let mut sum = AccurateSum::default();
        for value in [1e16, 1.0, -1e16] {
            sum.add_product(value, 1.0);
        }
        assert_eq!(sum.value(), 1.0);
        let mut product = AccurateSum::default();
        product.add_product(1.0 + 2.0_f64.powi(-27), 1.0 - 2.0_f64.powi(-27));
        product.add_product(-1.0, 1.0);
        assert_eq!(product.value(), -2.0_f64.powi(-54));
    }

    #[test]
    fn congruence_retains_inner_product_low_parts_in_every_ordered_entry() {
        // Dyadic inputs give independent exact polynomial values. Rounding
        // the inner product before the outer contraction loses the e^2 term.
        let e = 2.0_f64.powi(-27);
        let c = Matrix::from_row_major(2, 2, vec![1.0 + e, 1.0 - 2.0 * e, 1.0, 1.0]);
        let f = Matrix::from_row_major(2, 2, vec![1.0 + e, 0.0, 0.0, -(1.0 + 3.0 * e)]);
        let actual = c.congruence_accurate(&f);
        let exact = [
            3.0 * e * e + e * e * e,
            -3.0 * e - 3.0 * e * e - 2.0 * e * e * e,
            -3.0 * e - 3.0 * e * e - 2.0 * e * e * e,
            -6.0 * e + 4.0 * e * e * e,
        ];
        for (index, (&value, &expected)) in actual.as_slice().iter().zip(&exact).enumerate() {
            assert_eq!(value, expected, "ordered congruence entry {index}");
        }
        let rounded = c.matmul_accurate(&f.matmul_accurate(&c, false), true);
        assert_ne!(rounded[(0, 0)], exact[0]);
    }

    #[test]
    fn factored_metric_commutator_matches_a_closed_nonorthogonal_model() {
        let s = Matrix::from_row_major(2, 2, vec![0.25, 0.0, 0.0, 4.0]);
        let x = Matrix::from_row_major(2, 2, vec![2.0, 0.0, 0.0, 0.5]);
        let f = Matrix::from_row_major(2, 2, vec![-250.0, 3e-13, 3e-13, 8.0]);
        let p = Matrix::from_row_major(2, 2, vec![4.0, 0.0, 0.0, 0.0]);
        let comm = metric_commutator(&f, &p, &s, &x);
        assert_eq!(comm[(0, 0)], 0.0);
        assert_eq!(comm[(1, 1)], 0.0);
        assert_eq!(comm[(1, 0)], 3e-13);
        assert_eq!(comm[(0, 1)], -3e-13);
    }

    #[test]
    fn eigen_diagonalizes_known_matrix() {
        // [[2,1],[1,2]] -> eigenvalues 1, 3.
        let a = Matrix::from_row_major(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
        let mut lam = Matrix::zeros(2, 2);
        lam[(0, 0)] = vals[0];
        lam[(1, 1)] = vals[1];
        let recon = vecs.matmul(&lam).matmul(&vecs.transpose());
        assert!(recon.max_abs_diff(&a) < 1e-10);
    }

    #[test]
    fn eigen_reconstructs_asymmetric_eigenvectors() {
        // Distinct eigenvalues -> eigenvector matrix is NOT symmetric; catches transpose bugs.
        let a = Matrix::from_row_major(3, 3, vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 1.0]);
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        assert!(vals[0] <= vals[1] && vals[1] <= vals[2]);
        let mut lam = Matrix::zeros(3, 3);
        for i in 0..3 {
            lam[(i, i)] = vals[i];
        }
        let recon = vecs.matmul(&lam).matmul(&vecs.transpose());
        assert!(recon.max_abs_diff(&a) < 1e-9);
        for j in 0..3 {
            let nj: f64 = (0..3).map(|i| vecs[(i, j)] * vecs[(i, j)]).sum();
            assert!((nj - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn solve_linear_basic() {
        let a = Matrix::from_row_major(2, 2, vec![3.0, 2.0, 1.0, 2.0]);
        let x = solve_linear(&a, &[7.0, 5.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn solve_diis_saddle_point() {
        // DIIS-like bordered system with known solution c0 = c1 = 0.5, λ = 0.75.
        let a = Matrix::from_row_major(3, 3, vec![1.0, 0.5, -1.0, 0.5, 1.0, -1.0, -1.0, -1.0, 0.0]);
        let x = solve_linear(&a, &[0.0, 0.0, -1.0]).unwrap();
        assert!((x[0] - 0.5).abs() < 1e-9);
        assert!((x[1] - 0.5).abs() < 1e-9);
    }

    /// A well-conditioned overlap-like SPD matrix: X must satisfy XᵀSX = I
    /// with no columns dropped.
    #[test]
    fn canonical_orthogonalizer_full_rank() {
        // S = I + small symmetric off-diagonals (a plausible AO overlap).
        let s = Matrix::from_row_major(
            3,
            3,
            vec![1.0, 0.30, 0.10, 0.30, 1.0, 0.25, 0.10, 0.25, 1.0],
        );
        let orth = Orthogonalizer::canonical(&s, 1e-7).unwrap();
        assert_eq!(orth.n_kept, 3);
        assert_eq!(orth.n_dropped, 0);
        let xtsx = orth.x.transpose_matmul(&s.matmul(&orth.x));
        assert!(xtsx.max_abs_diff(&Matrix::identity(3)) < 1e-10);
    }

    /// Two identical basis vectors → one exactly-null overlap eigenvalue.
    /// The orthogonalizer must drop it and still give XᵀSX = I on the
    /// retained subspace.
    #[test]
    fn canonical_orthogonalizer_drops_linear_dependence() {
        // Rows/cols 0 and 1 are duplicates: S has eigenvalue 0.
        let s = Matrix::from_row_major(3, 3, vec![1.0, 1.0, 0.2, 1.0, 1.0, 0.2, 0.2, 0.2, 1.0]);
        let orth = Orthogonalizer::canonical(&s, 1e-7).unwrap();
        assert_eq!(orth.n_dropped, 1);
        assert_eq!(orth.n_kept, 2);
        let xtsx = orth.x.transpose_matmul(&s.matmul(&orth.x));
        assert!(xtsx.max_abs_diff(&Matrix::identity(2)) < 1e-10);
    }

    /// Generalized eigenproblem F C = S C ε with a known analytic solution:
    /// the H2-like two-level model. For F = [[a, b], [b, a]], S = [[1, s], [s, 1]],
    /// the solutions are ε± = (a ± b)/(1 ± s) with C normalized by CᵀSC = I.
    #[test]
    fn roothaan_solve_two_level_model() {
        let (a, b, s_off) = (-1.0, -0.6, 0.4);
        let f = Matrix::from_row_major(2, 2, vec![a, b, b, a]);
        let s = Matrix::from_row_major(2, 2, vec![1.0, s_off, s_off, 1.0]);
        let orth = Orthogonalizer::canonical(&s, 1e-10).unwrap();
        let (eps, c) = orth.solve_roothaan(&f).unwrap();
        let e_plus = (a + b) / (1.0 + s_off); // bonding (lower for negative b)
        let e_minus = (a - b) / (1.0 - s_off);
        assert!((eps[0] - e_plus.min(e_minus)).abs() < 1e-12);
        assert!((eps[1] - e_plus.max(e_minus)).abs() < 1e-12);
        // CᵀSC = I and F C = S C ε.
        let ctsc = c.transpose_matmul(&s.matmul(&c));
        assert!(ctsc.max_abs_diff(&Matrix::identity(2)) < 1e-12);
        let fc = f.matmul(&c);
        let mut sce = s.matmul(&c);
        for i in 0..2 {
            for j in 0..2 {
                sce[(i, j)] *= eps[j];
            }
        }
        assert!(fc.max_abs_diff(&sce) < 1e-12);
    }
}
