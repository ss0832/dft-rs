// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed support/Pauli normalization for reference operator projection.
//! This algebraic helper changes no energy, coefficients, SCF or geometry.

use crate::linalg::{solve_linear_matrix, symmetric_eigen, Matrix};
use crate::{DftError, Result};

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}
fn largest(matrix: &Matrix) -> f64 {
    matrix
        .as_slice()
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f64::max)
}
fn finite(matrix: &Matrix) -> bool {
    matrix.as_slice().iter().all(|x| x.is_finite())
}
fn symmetric(matrix: &Matrix) -> bool {
    if matrix.rows != matrix.cols || !finite(matrix) {
        return false;
    }
    let tolerance = 64.0 * f64::EPSILON * (1.0 + largest(matrix));
    (0..matrix.rows).all(|i| (0..i).all(|j| (matrix[(i, j)] - matrix[(j, i)]).abs() <= tolerance))
}

#[derive(Clone, Debug)]
pub struct SupportProjection {
    /// C=[Y,CB], where C^T S C=I. No orbital sign convention is imposed.
    pub transform_c: Matrix,
    /// diag(I, sqrt(1 Eh) [kappa I-CB^T Ffull CB]^(-1/2)).
    pub metric_weight: Matrix,
    pub frozen_pauli_block: Matrix,
    /// Unmodified eigenvalues; singular/indefinite blocks are errors.
    pub raw_pauli_eigenvalues: Vec<f64>,
    pub physical_support_dimension: usize,
    pub frozen_dimension: usize,
    pub orthonormality_error: f64,
    pub pauli_condition_number: f64,
    pub weight_condition_number: f64,
    pub normalized_pauli_identity_error: f64,
    /// Common dimensional convention, never fitted to an observable.
    pub energy_scale_hartree: f64,
}

impl SupportProjection {
    pub fn new(
        complement_y: &Matrix,
        frozen_cb: &Matrix,
        full_fock: &Matrix,
        overlap: &Matrix,
        kappa: f64,
    ) -> Result<Self> {
        let n = overlap.rows;
        let ny = complement_y.cols;
        let nb = frozen_cb.cols;
        if n == 0
            || ny == 0
            || nb == 0
            || ny.checked_add(nb) != Some(n)
            || complement_y.rows != n
            || frozen_cb.rows != n
            || !finite(complement_y)
            || !finite(frozen_cb)
            || !symmetric(overlap)
            || !symmetric(full_fock)
            || full_fock.rows != n
            || !kappa.is_finite()
            || kappa <= 0.0
        {
            return invalid("support projection requires complete finite S-orthonormal complement/frozen spaces, Hermitian F/S and positive kappa");
        }
        let overlap_values = symmetric_eigen(overlap)?.0;
        if overlap_values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return invalid("support projection overlap is not strictly positive definite");
        }
        let mut c = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..ny {
                c[(i, j)] = complement_y[(i, j)];
            }
            for j in 0..nb {
                c[(i, ny + j)] = frozen_cb[(i, j)];
            }
        }
        let gram = c.transpose_matmul_seq(&overlap.matmul_seq(&c));
        let orthonormality_error = gram.max_abs_diff(&Matrix::identity(n));
        if !orthonormality_error.is_finite() || orthonormality_error > 1e-10 {
            return invalid("support projection refuses nonorthogonal or incomplete Y/CB spaces");
        }
        let fbb = frozen_cb.transpose_matmul_seq(&full_fock.matmul_seq(frozen_cb));
        let mut m = Matrix::zeros(nb, nb);
        for i in 0..nb {
            for j in 0..nb {
                m[(i, j)] = if i == j { kappa } else { 0.0 } - fbb[(i, j)];
            }
        }
        let (eigenvalues, vectors) = symmetric_eigen(&m)?;
        if eigenvalues.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return invalid("the frozen Pauli block kappa I-CB^T F CB is singular or indefinite; no eigenvalue floor or PSD repair is allowed");
        }
        let mut wb = Matrix::zeros(nb, nb);
        for k in 0..nb {
            let factor = 1.0 / eigenvalues[k].sqrt();
            for i in 0..nb {
                for j in 0..nb {
                    wb[(i, j)] += vectors[(i, k)] * vectors[(j, k)] * factor;
                }
            }
        }
        let mut weight = Matrix::identity(n);
        for i in 0..nb {
            for j in 0..nb {
                weight[(ny + i, ny + j)] = wb[(i, j)];
            }
        }
        let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = eigenvalues.iter().copied().fold(0.0, f64::max);
        let pauli_condition_number = maximum / minimum;
        let weight_minimum = 1.0_f64.min(1.0 / maximum.sqrt());
        let weight_maximum = 1.0_f64.max(1.0 / minimum.sqrt());
        let weight_condition_number = weight_maximum / weight_minimum;
        let identity_error = wb
            .matmul_seq(&m)
            .matmul_seq(&wb)
            .max_abs_diff(&Matrix::identity(nb));
        if !finite(&weight)
            || !pauli_condition_number.is_finite()
            || !weight_condition_number.is_finite()
            || !identity_error.is_finite()
        {
            return invalid("Pauli normalization overflowed; no finite replacement or spectral clipping is applied");
        }
        Ok(Self {
            transform_c: c,
            metric_weight: weight,
            frozen_pauli_block: m,
            raw_pauli_eigenvalues: eigenvalues,
            physical_support_dimension: ny,
            frozen_dimension: nb,
            orthonormality_error,
            pauli_condition_number,
            weight_condition_number,
            normalized_pauli_identity_error: identity_error,
            energy_scale_hartree: 1.0,
        })
    }

    /// Supply this result, not W V W, as an operator-projection sample target.
    /// The existing fitter applies the returned metric_weight on both sides.
    pub fn transform_operator(&self, operator: &Matrix) -> Result<Matrix> {
        if !symmetric(operator) || operator.rows != self.transform_c.rows {
            return invalid("support operator must be a finite Hermitian full-AO matrix");
        }
        let transformed = self
            .transform_c
            .transpose_matmul_seq(&operator.matmul_seq(&self.transform_c));
        if !finite(&transformed) {
            return invalid("support operator transformation overflowed");
        }
        Ok(transformed)
    }
    pub fn transform_dictionary(&self, dictionary: &[Matrix]) -> Result<Vec<Matrix>> {
        dictionary
            .iter()
            .map(|operator| self.transform_operator(operator))
            .collect()
    }
    /// Diagnostic only: do not pass the weighted result to a fitter that
    /// will apply metric_weight again.
    pub fn weighted_operator(&self, operator: &Matrix) -> Result<Matrix> {
        let transformed = self.transform_operator(operator)?;
        let weighted = self
            .metric_weight
            .matmul_seq(&transformed)
            .matmul_seq(&self.metric_weight);
        if !finite(&weighted) {
            return invalid("support metric multiplication overflowed");
        }
        Ok(weighted)
    }
}

/// A fixed-dimensional raw AO chart embedded orthogonally to frozen orbitals.
/// This is reference-generation algebra, not a density-independent replacement
/// for the total-density energy functional. The reference basis must contain
/// every raw AO direction; an incomplete least-squares image is refused.
#[derive(Clone, Debug)]
pub struct OrthogonalReducedChart {
    /// Full-reference coefficients of raw-coordinate functions: T^T S T=Sraw.
    pub transform: Matrix,
    /// Full-reference coefficients in orthonormal raw coordinates.
    pub orthonormal_transform: Matrix,
    /// Exact metric left inverse on this chart, L T=I.
    pub density_left_inverse: Matrix,
    pub raw_overlap: Matrix,
    pub retained_metric_eigenvalues: Vec<f64>,
    pub representation_error: f64,
    pub metric_error: f64,
    pub frozen_overlap: f64,
}

pub(crate) fn positive_power(matrix: &Matrix, exponent: f64) -> Result<Matrix> {
    let (values, vectors) = symmetric_eigen(matrix)?;
    if values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return invalid("the reduced chart metric must be strictly positive definite");
    }
    let mut result = Matrix::zeros(matrix.rows, matrix.cols);
    for (k, value) in values.iter().enumerate() {
        let scale = value.powf(exponent);
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                result[(i, j)] += vectors[(i, k)] * vectors[(j, k)] * scale;
            }
        }
    }
    if !finite(&result) {
        return invalid("the reduced chart metric power overflowed");
    }
    Ok(result)
}

/// Fréchet derivative of a positive metric square root or inverse square
/// root. The divided differences below are regular at repeated eigenvalues;
/// they avoid subtracting nearly equal roots and require no eigenvector gauge.
pub(crate) fn metric_power_direction(
    matrix: &Matrix,
    direction: &Matrix,
    inverse: bool,
) -> Result<Matrix> {
    if !symmetric(matrix) || !symmetric(direction) || direction.rows != matrix.rows {
        return invalid("metric derivative requires matching finite symmetric matrices");
    }
    let (values, vectors) = symmetric_eigen(matrix)?;
    if values.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return invalid("metric derivative requires a positive definite metric");
    }
    let mut transformed = vectors.transpose_matmul_seq(&direction.matmul_seq(&vectors));
    for i in 0..values.len() {
        for j in 0..values.len() {
            let a = values[i].sqrt();
            let b = values[j].sqrt();
            transformed[(i, j)] *= if inverse {
                -1. / (a * b * (a + b))
            } else {
                1. / (a + b)
            };
        }
    }
    Ok(vectors
        .matmul_seq(&transformed)
        .matmul_transpose_seq(&vectors))
}

struct MatrixDirection {
    value: Matrix,
    derivative: Matrix,
}

/// Matrix Taylor coefficients use factorial-normalized powers. The square
/// root recurrence solves a Sylvester equation in the reference eigenbasis;
/// it keeps noncommuting products and remains regular at repeated roots.
#[derive(Clone)]
pub(crate) struct MatrixTaylor(pub(crate) Vec<Matrix>);
impl MatrixTaylor {
    pub(crate) fn constant(value: &Matrix, order: usize) -> Self {
        let mut coefficients = vec![Matrix::zeros(value.rows, value.cols); order + 1];
        coefficients[0] = value.clone();
        Self(coefficients)
    }
    pub(crate) fn transpose(&self) -> Self {
        Self(self.0.iter().map(Matrix::transpose).collect())
    }
    pub(crate) fn product(&self, other: &Self) -> Self {
        Self(crate::response::nuclear::matrix_product(
            &self.0,
            &other.0,
            self.0.len() - 1,
        ))
    }
    pub(crate) fn subtract(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (a, b) in result.0.iter_mut().zip(&other.0) {
            for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
                *x -= y;
            }
        }
        result
    }
    pub(crate) fn solve(&self, rhs: &Self) -> Result<Self> {
        let mut result = Vec::<Matrix>::with_capacity(self.0.len());
        for n in 0..self.0.len() {
            let mut residual = rhs.0[n].clone();
            for k in 1..=n {
                let product = self.0[k].matmul_seq(&result[n - k]);
                for (a, b) in residual.as_mut_slice().iter_mut().zip(product.as_slice()) {
                    *a -= b;
                }
            }
            result.push(solve_linear_matrix(&self.0[0], &residual)?);
        }
        Ok(Self(result))
    }
    pub(crate) fn power(&self, inverse: bool) -> Result<Self> {
        if self.0.is_empty()
            || self
                .0
                .iter()
                .any(|m| !symmetric(m) || m.rows != self.0[0].rows)
        {
            return invalid("metric Taylor power needs matching finite symmetric coefficients");
        }
        let (values, vectors) = symmetric_eigen(&self.0[0])?;
        if values.iter().any(|v| *v <= 0. || !v.is_finite()) {
            return invalid("metric Taylor power requires a positive reference metric");
        }
        let mut root = vec![positive_power(&self.0[0], 0.5)?];
        for n in 1..self.0.len() {
            let mut residual = self.0[n].clone();
            for k in 1..n {
                let product = root[k].matmul_seq(&root[n - k]);
                for (a, b) in residual.as_mut_slice().iter_mut().zip(product.as_slice()) {
                    *a -= b;
                }
            }
            let mut coefficient = vectors.transpose_matmul_seq(&residual.matmul_seq(&vectors));
            for i in 0..values.len() {
                for j in 0..values.len() {
                    coefficient[(i, j)] /= values[i].sqrt() + values[j].sqrt();
                }
            }
            root.push(
                vectors
                    .matmul_seq(&coefficient)
                    .matmul_transpose_seq(&vectors),
            );
        }
        let root = Self(root);
        if inverse {
            root.solve(&Self::constant(
                &Matrix::identity(values.len()),
                self.0.len() - 1,
            ))
        } else {
            Ok(root)
        }
    }
}
impl MatrixDirection {
    fn new(value: &Matrix, derivative: &Matrix) -> Self {
        Self {
            value: value.clone(),
            derivative: derivative.clone(),
        }
    }
    fn transpose(&self) -> Self {
        Self {
            value: self.value.transpose(),
            derivative: self.derivative.transpose(),
        }
    }
    fn product(&self, other: &Self) -> Self {
        let mut derivative = self.derivative.matmul_seq(&other.value);
        let second = self.value.matmul_seq(&other.derivative);
        for (a, b) in derivative.as_mut_slice().iter_mut().zip(second.as_slice()) {
            *a += b;
        }
        Self {
            value: self.value.matmul_seq(&other.value),
            derivative,
        }
    }
    fn subtract(&self, other: &Self) -> Self {
        let mut value = self.value.clone();
        let mut derivative = self.derivative.clone();
        for (a, b) in value.as_mut_slice().iter_mut().zip(other.value.as_slice()) {
            *a -= b;
        }
        for (a, b) in derivative
            .as_mut_slice()
            .iter_mut()
            .zip(other.derivative.as_slice())
        {
            *a -= b;
        }
        Self { value, derivative }
    }
    fn solve(&self, rhs: &Self) -> Result<Self> {
        let value = solve_linear_matrix(&self.value, &rhs.value)?;
        let mut derivative = rhs.derivative.clone();
        let product = self.derivative.matmul_seq(&value);
        for (a, b) in derivative.as_mut_slice().iter_mut().zip(product.as_slice()) {
            *a -= b;
        }
        Ok(Self {
            value,
            derivative: solve_linear_matrix(&self.value, &derivative)?,
        })
    }
    fn power(&self, inverse: bool) -> Result<Self> {
        Ok(Self {
            value: positive_power(&self.value, if inverse { -0.5 } else { 0.5 })?,
            derivative: metric_power_direction(&self.value, &self.derivative, inverse)?,
        })
    }
}

impl OrthogonalReducedChart {
    /// Higher-order coefficients of exactly the same common metric/Pauli
    /// chart. Inputs are full, cross, raw metric and normalized frozen span.
    pub fn taylor_transform(&self, coefficients: [&[Matrix]; 4]) -> Result<Vec<Matrix>> {
        let order = coefficients[0].len().checked_sub(1).ok_or_else(|| {
            DftError::InvalidInput("common chart Taylor coefficients are empty".into())
        })?;
        let shapes = [
            (self.transform.rows, self.transform.rows),
            (self.transform.rows, self.transform.cols),
            (self.transform.cols, self.transform.cols),
            (
                self.transform.rows,
                coefficients[3].first().map_or(0, |m| m.cols),
            ),
        ];
        for (series, (rows, cols)) in coefficients.iter().zip(shapes) {
            if series.len() != order + 1
                || series
                    .iter()
                    .any(|m| m.rows != rows || m.cols != cols || !finite(m))
            {
                return invalid("common chart Taylor dimensions or coefficients are invalid");
            }
        }
        let [s, cross, raw, b] = coefficients.map(|v| MatrixTaylor(v.to_vec()));
        let image = s.solve(&cross)?.product(&raw.power(true)?);
        let removed = b.product(&b.transpose().product(&s.product(&image)));
        let projected = image.subtract(&removed);
        let gram = projected.transpose().product(&s.product(&projected));
        let transform = projected
            .product(&gram.power(true)?)
            .product(&raw.power(false)?);
        if transform.0[0].max_abs_diff(&self.transform) > 1e-9
            || transform.0.iter().any(|m| !finite(m))
        {
            return invalid("Taylor common chart differs from its fixed scalar definition");
        }
        Ok(transform.0)
    }
    /// Differentiate the same metric/Pauli chart used by `new`. Directions
    /// are [full overlap, full/raw cross overlap, raw overlap, frozen B].
    /// Frozen orbitals may themselves be metric-transported; their derivative
    /// must then be supplied, rather than silently treating their AO rows fixed.
    pub fn directional_transform(
        &self,
        full_overlap: &Matrix,
        cross_overlap: &Matrix,
        frozen: &Matrix,
        directions: [&Matrix; 4],
    ) -> Result<Matrix> {
        if full_overlap.rows != self.transform.rows
            || !symmetric(full_overlap)
            || cross_overlap.rows != self.transform.rows
            || cross_overlap.cols != self.transform.cols
            || frozen.rows != self.transform.rows
            || !finite(cross_overlap)
            || !finite(frozen)
        {
            return invalid("common-chart derivative input spaces do not match the saved chart");
        }
        let s = MatrixDirection::new(full_overlap, directions[0]);
        let cross = MatrixDirection::new(cross_overlap, directions[1]);
        let raw = MatrixDirection::new(&self.raw_overlap, directions[2]);
        let b = MatrixDirection::new(frozen, directions[3]);
        if [
            (&s.value, &s.derivative),
            (&cross.value, &cross.derivative),
            (&raw.value, &raw.derivative),
            (&b.value, &b.derivative),
        ]
        .iter()
        .any(|(v, d)| v.rows != d.rows || v.cols != d.cols || !finite(d))
        {
            return invalid("common-chart derivative dimensions or values are invalid");
        }
        let image = s.solve(&cross)?.product(&raw.power(true)?);
        let removed = b.product(&b.transpose().product(&s.product(&image)));
        let projected = image.subtract(&removed);
        let gram = projected.transpose().product(&s.product(&projected));
        let transformed = projected
            .product(&gram.power(true)?)
            .product(&raw.power(false)?);
        if transformed.value.max_abs_diff(&self.transform) > 1e-9
            || !finite(&transformed.derivative)
        {
            return invalid("differentiated common chart differs from its fixed scalar definition");
        }
        Ok(transformed.derivative)
    }
    /// Construct Q R (R^T Q^T S Q R)^(-1/2) in orthonormal raw coordinates.
    /// Q=I-B B^T S excludes the frozen occupied subspace. No eigenvalue is
    /// clipped, and no direction or electron is removed to repair a failure.
    pub fn new(
        full_overlap: &Matrix,
        cross_overlap: &Matrix,
        raw_overlap: &Matrix,
        frozen_orbitals: &Matrix,
        rank_tolerance: f64,
    ) -> Result<Self> {
        let n = full_overlap.rows;
        let m = raw_overlap.rows;
        if n == 0
            || m == 0
            || !symmetric(full_overlap)
            || !symmetric(raw_overlap)
            || cross_overlap.rows != n
            || cross_overlap.cols != m
            || frozen_orbitals.rows != n
            || !finite(cross_overlap)
            || !finite(frozen_orbitals)
            || !rank_tolerance.is_finite()
            || rank_tolerance <= 0.0
            || rank_tolerance >= 1.0
        {
            return invalid("orthogonal reduced chart requires compatible finite metrics, overlaps, frozen orbitals and a rank tolerance in (0,1)");
        }
        // A linear solve alone does not certify a physical positive AO metric.
        if symmetric_eigen(full_overlap)?.0.iter().any(|v| *v <= 0.0) {
            return invalid("the full-reference chart metric is not positive definite");
        }
        let frozen_gram =
            frozen_orbitals.transpose_matmul_seq(&full_overlap.matmul_seq(frozen_orbitals));
        if frozen_gram.max_abs_diff(&Matrix::identity(frozen_orbitals.cols)) > 1e-10 {
            return invalid("the reduced chart requires orthonormal frozen orbitals");
        }
        let raw_inverse_root = positive_power(raw_overlap, -0.5)?;
        let raw_root = positive_power(raw_overlap, 0.5)?;
        let image = solve_linear_matrix(full_overlap, cross_overlap)?.matmul_seq(&raw_inverse_root);
        let image_gram = image.transpose_matmul_seq(&full_overlap.matmul_seq(&image));
        let representation_error = image_gram.max_abs_diff(&Matrix::identity(m));
        if representation_error > 1e-9 {
            return Err(DftError::Unsupported {
                feature: "orthogonal common reduced chart".into(),
                missing: "the reference basis must contain the entire common raw AO span; extend the reference basis instead of normalizing an incomplete image".into(),
            });
        }
        let removed = frozen_orbitals
            .matmul_seq(&frozen_orbitals.transpose_matmul_seq(&full_overlap.matmul_seq(&image)));
        let mut projected = image;
        for (value, remove) in projected.as_mut_slice().iter_mut().zip(removed.as_slice()) {
            *value -= remove;
        }
        let gram = projected.transpose_matmul_seq(&full_overlap.matmul_seq(&projected));
        let retained_metric_eigenvalues = symmetric_eigen(&gram)?.0;
        if retained_metric_eigenvalues
            .iter()
            .any(|v| !v.is_finite() || *v <= rank_tolerance)
        {
            return Err(DftError::Unsupported {
                feature: "orthogonal common reduced chart".into(),
                missing: "frozen exclusion loses the declared common AO rank; a different shared dictionary is required, without deleting endpoint-specific directions".into(),
            });
        }
        let orthonormal_transform = projected.matmul_seq(&positive_power(&gram, -0.5)?);
        let transform = orthonormal_transform.matmul_seq(&raw_root);
        let metric_error = transform
            .transpose_matmul_seq(&full_overlap.matmul_seq(&transform))
            .max_abs_diff(raw_overlap);
        let frozen_overlap =
            largest(&frozen_orbitals.transpose_matmul_seq(&full_overlap.matmul_seq(&transform)));
        let density_left_inverse =
            solve_linear_matrix(raw_overlap, &transform.transpose_matmul_seq(full_overlap))?;
        if !finite(&transform)
            || !finite(&density_left_inverse)
            || metric_error > 1e-9 * (1.0 + largest(raw_overlap))
            || frozen_overlap > 1e-9
            || density_left_inverse
                .matmul_seq(&transform)
                .max_abs_diff(&Matrix::identity(m))
                > 1e-9
        {
            return invalid(
                "orthogonal reduced chart failed its raw metric or frozen-exclusion identities",
            );
        }
        Ok(Self {
            transform,
            orthonormal_transform,
            density_left_inverse,
            raw_overlap: raw_overlap.clone(),
            retained_metric_eigenvalues,
            representation_error,
            metric_error,
            frozen_overlap,
        })
    }

    /// The operator pullback is dual to `lift_density` under Tr(P F).
    pub fn pullback_operator(&self, operator: &Matrix) -> Result<Matrix> {
        if !symmetric(operator) || operator.rows != self.transform.rows {
            return invalid("full-reference operator has the wrong reduced-chart shape");
        }
        let result = self
            .transform
            .transpose_matmul_seq(&operator.matmul_seq(&self.transform));
        if !finite(&result) {
            return invalid("reduced-chart operator transformation overflowed");
        }
        Ok(result)
    }

    /// Preserve the raw metric particle count and occupied-projector identity.
    pub fn lift_density(&self, density: &Matrix) -> Result<Matrix> {
        if !symmetric(density) || density.rows != self.transform.cols {
            return invalid("raw density has the wrong reduced-chart shape");
        }
        let result = self
            .transform
            .matmul_seq(density)
            .matmul_transpose_seq(&self.transform);
        if !finite(&result) {
            return invalid("reduced-chart density lifting overflowed");
        }
        Ok(result)
    }

    /// Recover raw coordinates; density outside the chart is projected, not
    /// silently normalized. The caller must inspect the recovered trace.
    pub fn pullback_density(&self, density: &Matrix) -> Result<Matrix> {
        if !symmetric(density) || density.rows != self.transform.rows {
            return invalid("full-reference density has the wrong reduced-chart shape");
        }
        let result = self
            .density_left_inverse
            .matmul_seq(density)
            .matmul_transpose_seq(&self.density_left_inverse);
        if !finite(&result) {
            return invalid("reduced-chart density pullback overflowed");
        }
        Ok(result)
    }
}
