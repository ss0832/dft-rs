// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Retained low parts of cancellation-sensitive matrix Taylor coefficients.
//!
//! A compensated dot product alone is insufficient when its rounded result
//! is multiplied by another large coefficient. These matrices retain that
//! rounding remainder through transport, density and response assembly.

use crate::linalg::{AccurateSum, Matrix};
use crate::response::{ResponseSolver, ResponseSolverConfig};
use crate::xc::ResolvedFunctional;
use crate::Result;

#[derive(Clone)]
pub(crate) struct MatrixParts {
    pub high: Matrix,
    pub low: Matrix,
}

pub(crate) type Series = Vec<MatrixParts>;

impl MatrixParts {
    pub fn new(high: Matrix) -> Self {
        let low = Matrix::zeros(high.rows, high.cols);
        Self { high, low }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(Matrix::zeros(rows, cols))
    }

    pub fn transpose(&self) -> Self {
        Self {
            high: self.high.transpose(),
            low: self.low.transpose(),
        }
    }

    pub fn add_scaled(&mut self, other: &Self, scale: f64) {
        for i in 0..self.high.rows {
            for j in 0..self.high.cols {
                let mut sum = AccurateSum::default();
                sum.add_product(self.high[(i, j)], 1.0);
                sum.add_product(self.low[(i, j)], 1.0);
                sum.add_product(other.high[(i, j)], scale);
                sum.add_product(other.low[(i, j)], scale);
                (self.high[(i, j)], self.low[(i, j)]) = sum.parts();
            }
        }
    }

    pub fn add_entry(&mut self, row: usize, col: usize, value: f64) {
        let mut sum = AccurateSum::default();
        sum.add_product(self.high[(row, col)], 1.0);
        sum.add_product(self.low[(row, col)], 1.0);
        sum.add_product(value, 1.0);
        (self.high[(row, col)], self.low[(row, col)]) = sum.parts();
    }

    pub fn max_abs(&self) -> f64 {
        self.high
            .as_slice()
            .iter()
            .zip(self.low.as_slice())
            .map(|(h, l)| (h + l).abs())
            .fold(0.0, f64::max)
    }
}

pub(super) fn retain(values: &[Matrix]) -> Series {
    values.iter().cloned().map(MatrixParts::new).collect()
}

pub(super) fn high(values: &[MatrixParts]) -> Vec<Matrix> {
    values.iter().map(|v| v.high.clone()).collect()
}

pub(super) fn low(values: &[MatrixParts]) -> Vec<Matrix> {
    values.iter().map(|v| v.low.clone()).collect()
}

fn accumulate(
    sum: &mut AccurateSum,
    a: &MatrixParts,
    b: &MatrixParts,
    i: usize,
    k: usize,
    j: usize,
) {
    sum.add_product(a.high[(i, k)], b.high[(k, j)]);
    sum.add_product(a.high[(i, k)], b.low[(k, j)]);
    sum.add_product(a.low[(i, k)], b.high[(k, j)]);
    sum.add_product(a.low[(i, k)], b.low[(k, j)]);
}

pub(crate) fn coefficient(a: &[MatrixParts], b: &[MatrixParts], n: usize) -> MatrixParts {
    let mut out = MatrixParts::zeros(a[0].high.rows, b[0].high.cols);
    for i in 0..out.high.rows {
        for j in 0..out.high.cols {
            let mut sum = AccurateSum::default();
            for k in 0..=n {
                for p in 0..a[k].high.cols {
                    accumulate(&mut sum, &a[k], &b[n - k], i, p, j);
                }
            }
            (out.high[(i, j)], out.low[(i, j)]) = sum.parts();
        }
    }
    out
}

pub(super) fn product(a: &[MatrixParts], b: &[MatrixParts], order: usize) -> Series {
    (0..=order).map(|n| coefficient(a, b, n)).collect()
}

pub(super) fn congruence(b: &[MatrixParts], f: &[MatrixParts], order: usize) -> Series {
    let transpose: Series = b.iter().map(MatrixParts::transpose).collect();
    product(&transpose, &product(f, b, order), order)
}

pub(super) fn density(b: &[MatrixParts], d: &[MatrixParts], order: usize) -> Series {
    let transpose: Series = b.iter().map(MatrixParts::transpose).collect();
    product(&product(b, d, order), &transpose, order)
}

pub(super) fn commutator(f: &[MatrixParts], d: &[MatrixParts], n: usize) -> MatrixParts {
    let mut out = coefficient(f, d, n);
    out.add_scaled(&coefficient(d, f, n), -1.0);
    out
}

pub(super) fn trace_product(a: &[MatrixParts], b: &[MatrixParts], n: usize) -> f64 {
    trace_product_parts(a, b, n).0
}

pub(super) fn trace_product_parts(a: &[MatrixParts], b: &[MatrixParts], n: usize) -> (f64, f64) {
    let mut sum = AccurateSum::default();
    for k in 0..=n {
        for i in 0..a[k].high.rows {
            for j in 0..a[k].high.cols {
                sum.add_product(a[k].high[(i, j)], b[n - k].high[(j, i)]);
                sum.add_product(a[k].high[(i, j)], b[n - k].low[(j, i)]);
                sum.add_product(a[k].low[(i, j)], b[n - k].high[(j, i)]);
                sum.add_product(a[k].low[(i, j)], b[n - k].low[(j, i)]);
            }
        }
    }
    sum.parts()
}

fn inverse_sqrt(metric: &[MatrixParts], order: usize) -> Series {
    let dim = metric[0].high.rows;
    let mut x = metric.to_vec();
    x[0].add_scaled(&MatrixParts::new(Matrix::identity(dim)), -1.0);
    let mut power = vec![MatrixParts::zeros(dim, dim); order + 1];
    power[0] = MatrixParts::new(Matrix::identity(dim));
    let mut inverse = power.clone();
    let mut binomial = 1.0;
    for k in 1..=order + 4 {
        power = product(&power, &x, order);
        binomial *= (-0.5 - (k - 1) as f64) / k as f64;
        for n in 0..=order {
            inverse[n].add_scaled(&power[n], binomial);
        }
    }
    inverse
}

pub(super) fn transport(c: &Matrix, overlap: &[Matrix], order: usize) -> Series {
    let mut reference = vec![MatrixParts::zeros(c.rows, c.cols); order + 1];
    reference[0] = MatrixParts::new(c.clone());
    let metric = congruence(&reference, &retain(overlap), order);
    product(&reference, &inverse_sqrt(&metric, order), order)
}

pub(super) fn frames(d: &[MatrixParts], occupied: usize) -> (Series, f64) {
    let dim = d[0].high.rows;
    let order = d.len() - 1;
    let mut q = vec![MatrixParts::zeros(dim, dim); order + 1];
    q[0] = MatrixParts::new(Matrix::identity(dim));
    for n in 1..=order {
        for i in 0..dim {
            for j in 0..dim {
                let sign = if j < occupied { 1.0 } else { -1.0 };
                q[n].high[(i, j)] = sign * d[n].high[(i, j)];
                q[n].low[(i, j)] = sign * d[n].low[(i, j)];
            }
        }
    }
    let transpose: Series = q.iter().map(MatrixParts::transpose).collect();
    let metric = product(&transpose, &q, order);
    let frame = product(&q, &inverse_sqrt(&metric, order), order);
    let mut gram = congruence(
        &frame,
        &{
            let mut identity = vec![MatrixParts::zeros(dim, dim); order + 1];
            identity[0] = MatrixParts::new(Matrix::identity(dim));
            identity
        },
        order,
    );
    gram[0].add_scaled(&MatrixParts::new(Matrix::identity(dim)), -1.0);
    let error = gram.iter().map(MatrixParts::max_abs).fold(0.0, f64::max);
    (frame, error)
}

pub(super) struct IntegerPath {
    pub b: [Series; 2],
    pub d: [Series; 2],
    pub p: [Series; 2],
    pub f: [Series; 2],
    pub evaluation: super::geometry::Evaluation,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
    pub coefficient_equations: Vec<super::CoefficientEquationDiagnostics>,
}

fn source(solver: &ResponseSolver<'_>, commutator: &[MatrixParts; 2]) -> Vec<f64> {
    super::orbital_source(solver, &std::array::from_fn(|s| commutator[s].high.clone()))
}

fn add_amplitudes(solver: &ResponseSolver<'_>, d: &mut [Series; 2], n: usize, amplitudes: &[f64]) {
    let mut index = 0;
    for (s, path) in d.iter_mut().enumerate().take(solver.spins()) {
        let no = solver.dims.n_occ[s];
        for a in 0..solver.dims.n_vir[s] {
            for i in 0..no {
                path[n].add_entry(no + a, i, amplitudes[index]);
                path[n].high[(i, no + a)] = path[n].high[(no + a, i)];
                path[n].low[(i, no + a)] = path[n].low[(no + a, i)];
                index += 1;
            }
        }
    }
    if solver.spins() == 1 {
        d[1][n] = d[0][n].clone();
    }
}

fn coefficient_amplitudes(solver: &ResponseSolver<'_>, d: &[Series; 2], n: usize) -> [Vec<f64>; 2] {
    let mut values = std::array::from_fn(|_| Vec::with_capacity(solver.u_len()));
    for (s, path) in d.iter().enumerate().take(solver.spins()) {
        let no = solver.dims.n_occ[s];
        for a in 0..solver.dims.n_vir[s] {
            for i in 0..no {
                values[0].push(path[n].high[(no + a, i)]);
                values[1].push(path[n].low[(no + a, i)]);
            }
        }
    }
    values
}

/// Fresh residual of A*u=-b for the coefficient actually eligible for return.
/// Scaling follows the original source, without a denominator floor.
fn coefficient_image_residual(source: &[f64], images: &[&[f64]]) -> f64 {
    if source.iter().any(|v| !v.is_finite())
        || images
            .iter()
            .any(|image| source.len() != image.len() || image.iter().any(|v| !v.is_finite()))
    {
        return f64::INFINITY;
    }
    let scale = source.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let mut residual = 0.0_f64;
    let mut norm = 0.0_f64;
    for (i, &b) in source.iter().enumerate() {
        let mut sum = crate::linalg::AccurateSum::default();
        sum.add_product(b, 1.0);
        for image in images {
            sum.add_product(image[i], 1.0);
        }
        if scale == 0.0 {
            if sum.value() != 0.0 {
                return f64::INFINITY;
            }
            continue;
        }
        residual = residual.hypot(sum.value() / scale);
        norm = norm.hypot(b / scale);
    }
    if scale == 0.0 {
        0.0
    } else {
        residual / norm
    }
}

/// Apply the original operator to both represented parts before summation.
/// Only an exactly zero vector can omit its operator application.
fn coefficient_parts_linear_residual(
    source: &[f64],
    parts: &[Vec<f64>; 2],
    mut apply: impl FnMut(&[f64]) -> Result<Vec<f64>>,
) -> Result<f64> {
    let mut images = [Vec::new(), Vec::new()];
    for (image, part) in images.iter_mut().zip(parts) {
        *image = if part.iter().all(|v| *v == 0.0) {
            vec![0.0; part.len()]
        } else {
            apply(part)?
        };
    }
    Ok(coefficient_image_residual(
        source,
        &[&images[0], &images[1]],
    ))
}

#[cfg(test)]
fn coefficient_linear_residual(source: &[f64], image: &[f64]) -> f64 {
    coefficient_image_residual(source, &[image])
}

/// Require a fresh relative linear solve even for a tiny nonzero source.
/// The separately reconstructed physical equation has its absolute target;
/// dividing its rounding error by a tiny source is a different diagnostic.
fn coefficient_resolved(
    norm: f64,
    linear_residual: Option<f64>,
    config: &ResponseSolverConfig,
) -> bool {
    norm <= config.tolerance * f64::EPSILON
        && linear_residual.map_or(norm == 0.0, |r| r <= config.acceptance())
}

/// Keep the best physical iterate when further floating-point residual
/// replacements stagnate. This changes neither the requested inner tolerance
/// nor the independently rebuilt final acceptance equations.
pub(crate) struct RefinementHistory<T> {
    best: Option<T>,
    pub(crate) norm: f64,
    progress_norm: f64,
    stalled: usize,
}

impl<T: Clone> RefinementHistory<T> {
    pub(crate) fn new() -> Self {
        Self {
            best: None,
            norm: f64::INFINITY,
            progress_norm: f64::INFINITY,
            stalled: 0,
        }
    }

    pub(crate) fn observe(&mut self, norm: f64, value: &T) -> Result<bool> {
        self.observe_if(norm, value, true)
    }

    fn observe_if(&mut self, norm: f64, value: &T, eligible: bool) -> Result<bool> {
        if !norm.is_finite() {
            return Err(crate::DftError::InvalidInput(
                "non-finite physical residual during retained refinement".into(),
            ));
        }
        if eligible && (self.best.is_none() || norm < self.norm) {
            self.best = Some(value.clone());
            self.norm = norm;
        }
        // A residual replacement must produce a meaningful contraction to
        // restart its budget. Retain smaller iterates even when roundoff
        // makes their improvement too slow to justify another replacement.
        if norm <= 0.5 * self.progress_norm {
            self.progress_norm = norm;
            self.stalled = 0;
        } else {
            self.stalled += 1;
        }
        Ok(self.stalled >= 3)
    }

    pub(crate) fn finish(self) -> T {
        self.best
            .expect("refinement evaluates its physical source before updating")
    }
}

/// Mixed-precision iterative refinement of the original physical equations.
/// Inner solves use the requested relative tolerance. Residual replacement
/// uses the retained matrix coefficients, rather than requesting an inner
/// tolerance below machine precision for a large higher-order source.
pub(super) fn integer_path(
    geometry: &super::geometry::GeometryTaylor,
    functional: &ResolvedFunctional,
    solver: &ResponseSolver<'_>,
    reference: &[Matrix; 2],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<IntegerPath> {
    let mut b: [Series; 2] =
        std::array::from_fn(|s| transport(&reference[s], &geometry.overlap[..=order], order));
    let mut d: [Series; 2] = std::array::from_fn(|s| {
        let mut projector = Matrix::zeros(reference[s].cols, reference[s].cols);
        for i in 0..solver.dims.n_occ[s] {
            projector[(i, i)] = 1.0;
        }
        vec![MatrixParts::new(projector)]
    });
    // A sub-ulp reference rotation is retained too: a zeroth-order residual
    // multiplied by a large fourth coefficient otherwise contaminates the
    // diagonal stationarity equations even after their OV solve converges.
    let mut reference_history = RefinementHistory::new();
    for refinement in 0..=12 {
        let p = std::array::from_fn(|s| density(&b[s][..1], &d[s], 0));
        let (_, ao) = geometry.evaluate_retained(functional, &p, 0)?;
        let f: [Series; 2] = std::array::from_fn(|s| congruence(&b[s][..1], &ao[s], 0));
        let rhs = source(
            solver,
            &std::array::from_fn(|s| commutator(&f[s], &d[s], 0)),
        );
        let norm = rhs.iter().fold(0.0_f64, |a, &x| a.hypot(x));
        let stalled = reference_history.observe(norm, &b)?;
        if norm <= config.tolerance * f64::EPSILON || stalled || refinement == 12 {
            break;
        }
        let solved = solver.solve_ov_report_with_structure(
            &rhs,
            config,
            crate::response::krylov::OperatorStructure::Hermitian,
        )?;
        let mut rotation: [Series; 2] =
            std::array::from_fn(|s| vec![MatrixParts::new(Matrix::identity(reference[s].cols))]);
        let mut index = 0;
        for (s, r) in rotation.iter_mut().enumerate().take(solver.spins()) {
            let no = solver.dims.n_occ[s];
            let dim = reference[s].cols;
            let mut generator = MatrixParts::zeros(dim, dim);
            for a in 0..solver.dims.n_vir[s] {
                for i in 0..no {
                    let u = solved.amplitudes[index].re;
                    generator.high[(no + a, i)] = u;
                    generator.high[(i, no + a)] = -u;
                    index += 1;
                }
            }
            let mut power = vec![MatrixParts::new(Matrix::identity(dim))];
            let mut factor = 1.0;
            for k in 1..=4 {
                power = product(&power, std::slice::from_ref(&generator), 0);
                factor /= k as f64;
                r[0].add_scaled(&power[0], factor);
            }
        }
        if solver.spins() == 1 {
            rotation[1] = rotation[0].clone();
        }
        for s in 0..2 {
            // A constant rotation has no other Taylor coefficients.
            for coefficient in &mut b[s] {
                *coefficient = coefficient_product(coefficient, &rotation[s][0]);
            }
        }
    }
    b = reference_history.finish();
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut coefficient_equations = Vec::new();
    for n in 1..=order {
        for (s, path) in d.iter_mut().enumerate() {
            let dim = path[0].high.rows;
            path.push(MatrixParts::zeros(dim, dim));
            let lower = coefficient(path, path, n);
            let no = solver.dims.n_occ[s];
            for i in 0..dim {
                for j in 0..dim {
                    if (i < no) == (j < no) {
                        let sign = if i < no { -1.0 } else { 1.0 };
                        path[n].high[(i, j)] = sign * lower.high[(i, j)];
                        path[n].low[(i, j)] = sign * lower.low[(i, j)];
                    }
                }
            }
        }
        let mut count = 0;
        let mut initial_norm = None;
        let mut initial_source = None;
        let mut inner_linear_residual: Option<f64> = None;
        let mut last_linear_residual = f64::INFINITY;
        let mut history = RefinementHistory::new();
        for refinement in 0..=8 {
            let p = std::array::from_fn(|s| density(&b[s], &d[s], n));
            let (_, ao) = geometry.evaluate_retained(functional, &p, n)?;
            let f: [Series; 2] = std::array::from_fn(|s| congruence(&b[s], &ao[s], n));
            let rhs = source(
                solver,
                &std::array::from_fn(|s| commutator(&f[s], &d[s], n)),
            );
            let norm = rhs.iter().fold(0.0_f64, |a, &x| a.hypot(x));
            initial_norm.get_or_insert(norm);
            let original = initial_source.get_or_insert_with(|| rhs.clone());
            // The free OV block starts at zero; its current entries are the
            // full amplitude, including every retained correction so far.
            let amplitudes = coefficient_amplitudes(solver, &d, n);
            let linear = coefficient_parts_linear_residual(original, &amplitudes, |part| {
                solver.hess_vec(part)
            })?;
            last_linear_residual = linear;
            let eligible = linear.is_finite()
                && linear <= config.acceptance()
                && (inner_linear_residual.is_some() || original.iter().all(|v| *v == 0.0));
            let current = std::array::from_fn::<_, 2, _>(|s| d[s][n].clone());
            let stalled = history.observe_if(norm, &(current, linear), eligible)?;
            // A lower-order OV residual multiplies higher density
            // coefficients in the diagonal commutator equations. Stopping
            // each order at the final absolute gate therefore does not
            // control the complete higher-order physical residual. Resolve
            // the retained low part here, using ordinary relative inner
            // solves and bounded mixed-precision residual replacement.
            if (eligible && coefficient_resolved(norm, Some(linear), config))
                || stalled
                || refinement == 8
            {
                break;
            }
            let solved = solver.solve_ov_report_with_structure(
                &rhs,
                config,
                crate::response::krylov::OperatorStructure::Hermitian,
            )?;
            count += solved.iterations;
            inner_linear_residual =
                Some(inner_linear_residual.map_or(solved.relative_residual, |r| {
                    r.max(solved.relative_residual)
                }));
            add_amplitudes(
                solver,
                &mut d,
                n,
                &solved.amplitudes.iter().map(|z| z.re).collect::<Vec<_>>(),
            );
        }
        if history.best.is_none() {
            return Err(crate::DftError::ResponseNotConverged {
                iterations: count,
                residual: last_linear_residual,
            });
        }
        let reconstructed_norm = history.norm;
        let (best, linear_residual) = history.finish();
        let denominator = initial_norm.expect("the physical source was evaluated");
        let relative = if denominator == 0.0 {
            reconstructed_norm
        } else {
            reconstructed_norm / denominator
        };
        coefficient_equations.push(super::CoefficientEquationDiagnostics {
            source_norm: denominator,
            reconstructed_norm,
            reconstructed_relative_residual: relative,
            linear_relative_residual: linear_residual,
            inner_linear_relative_residual: inner_linear_residual.unwrap_or(0.0),
        });
        for s in 0..2 {
            d[s][n] = best[s].clone();
        }
        residuals.push(linear_residual);
        iterations.push(count);
    }
    let p = std::array::from_fn(|s| density(&b[s], &d[s], order));
    let (evaluation, ao) = geometry.evaluate_retained(functional, &p, order)?;
    let f = std::array::from_fn(|s| congruence(&b[s], &ao[s], order));
    Ok(IntegerPath {
        b,
        d,
        p,
        f,
        evaluation,
        residuals,
        iterations,
        coefficient_equations,
    })
}

fn coefficient_product(a: &MatrixParts, b: &MatrixParts) -> MatrixParts {
    coefficient(std::slice::from_ref(a), std::slice::from_ref(b), 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use crate::dual::{Scalar, Taylor};

    #[test]
    fn tiny_nonzero_coefficients_require_a_fresh_relative_linear_solve() {
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        assert!(coefficient_resolved(0.0, None, &config));
        for initial in [1e-300, 1e-40, 1e-30, 1.0, 1e30, 1e300] {
            assert!(!coefficient_resolved(initial, None, &config));
            let resolved =
                (initial * 0.5 * config.acceptance()).min(0.5 * config.tolerance * f64::EPSILON);
            assert!(resolved > 0.0);
            assert!(coefficient_resolved(
                resolved,
                Some(0.5 * config.acceptance()),
                &config
            ));
            assert!(!coefficient_resolved(resolved, Some(1.0), &config));
        }
        // Neither an absolute-only nor a relative-only criterion suffices.
        assert!(!coefficient_resolved(1e-30, None, &config));
        assert!(!coefficient_resolved(1e-6, Some(0.0), &config));
    }

    #[test]
    fn fresh_linear_residuals_match_a_closed_dense_solution_at_every_source_scale() {
        use crate::response::complex::Complex;
        use crate::response::krylov::{solve, LinearOperator, OperatorStructure};
        struct Dense;
        impl LinearOperator for Dense {
            fn dimension(&self) -> usize {
                2
            }
            fn structure(&self) -> OperatorStructure {
                OperatorStructure::Hermitian
            }
            fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
                Ok(vec![x[0] * 3.0 + x[1], x[0] + x[1] * 2.0])
            }
        }
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        for scale in [1e-300, 1e-30, 1.0, 1e30, 1e300] {
            let rhs = [Complex::from(0.7 * scale), Complex::from(-0.3 * scale)];
            let result = solve(&Dense, &rhs, None, &config).unwrap();
            let x = [
                result.amplitudes[0].re / scale,
                result.amplitudes[1].re / scale,
            ];
            // Independent inverse [[2,-1],[-1,3]]/5 of the supplied matrix.
            assert!((x[0] - 0.34).abs() < 1e-12);
            assert!((x[1] + 0.32).abs() < 1e-12);
            let fresh =
                (3.0 * x[0] + x[1] - 0.7).hypot(x[0] + 2.0 * x[1] + 0.3) / 0.7_f64.hypot(0.3);
            assert!(fresh <= config.acceptance());
            assert!(result.relative_residual <= config.acceptance());
        }
    }

    #[test]
    fn returned_coefficient_residual_checks_the_actual_amplitude_at_every_scale() {
        for scale in [1e-300, 1e-30, 1.0, 1e30, 1e300] {
            let source = [0.7 * scale, -0.3 * scale];
            // Independent inverse of [[3,1],[1,2]] for A*u=-source.
            let amplitude = [-0.34 * scale, 0.32 * scale];
            let image = [
                3.0 * amplitude[0] + amplitude[1],
                amplitude[0] + 2.0 * amplitude[1],
            ];
            let measured = coefficient_linear_residual(&source, &image);
            assert!(measured.is_finite() && measured < 1e-12);
            let wrong_amplitude = [amplitude[0] + 0.001 * scale, amplitude[1]];
            let wrong_image = [
                3.0 * wrong_amplitude[0] + wrong_amplitude[1],
                wrong_amplitude[0] + 2.0 * wrong_amplitude[1],
            ];
            assert!(coefficient_linear_residual(&source, &wrong_image) > 1e-4);
        }
        assert_eq!(coefficient_linear_residual(&[0.0], &[0.0]), 0.0);
        assert!(!coefficient_linear_residual(&[0.0], &[1.0]).is_finite());
        assert!(!coefficient_linear_residual(&[1.0], &[]).is_finite());
        assert!(!coefficient_linear_residual(&[1.0], &[f64::NAN]).is_finite());
    }

    #[test]
    fn returned_coefficient_keeps_low_parts_in_the_original_equation() {
        let mu = 1e-30;
        let source = [-mu, -mu];
        let parts = [vec![1.0, 1.0], vec![1e-17, -1e-17]];
        let apply = |x: &[f64]| vec![x[0] - x[1] + mu * x[0], x[1] - x[0] + mu * x[1]];
        let rounded = [parts[0][0] + parts[1][0], parts[0][1] + parts[1][1]];
        assert_eq!(coefficient_linear_residual(&source, &apply(&rounded)), 0.0);
        let mut calls = 0;
        let measured = coefficient_parts_linear_residual(&source, &parts, |x| {
            calls += 1;
            Ok(apply(x))
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert!((measured / 2e13 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn retained_operator_skips_only_exact_zero_parts() {
        let mut calls = 0;
        let measured =
            coefficient_parts_linear_residual(&[-1e-300], &[vec![0.0], vec![1e-300]], |x| {
                calls += 1;
                Ok(x.to_vec())
            })
            .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(measured, 0.0);
        assert_eq!(coefficient_image_residual(&[0.0], &[&[1.0], &[-1.0]]), 0.0);
        assert!(!coefficient_image_residual(&[0.0], &[&[1e-300]]).is_finite());
    }

    #[test]
    fn physical_iterate_selection_cannot_discard_the_accepted_linear_solution() {
        let source = 1e-30;
        let solved = -0.5 * source;
        let initial_norm = source;
        let reconstructed_norm = 2.0 * source;
        let initial_linear = coefficient_linear_residual(&[source], &[0.0]);
        let solved_linear = coefficient_linear_residual(&[source], &[2.0 * solved]);
        assert_eq!(initial_linear, 1.0);
        assert_eq!(solved_linear, 0.0);

        // The former unconditional best-physical policy returns the unsolved
        // initial coefficient even though the inner solve was exact.
        let mut former = RefinementHistory::new();
        former.observe(initial_norm, &0.0).unwrap();
        former.observe(reconstructed_norm, &solved).unwrap();
        assert_eq!(former.finish(), 0.0);

        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let mut selected = RefinementHistory::new();
        selected
            .observe_if(initial_norm, &0.0, initial_linear <= config.acceptance())
            .unwrap();
        selected
            .observe_if(
                reconstructed_norm,
                &solved,
                solved_linear <= config.acceptance(),
            )
            .unwrap();
        assert_eq!(selected.norm, reconstructed_norm);
        let returned = selected.finish();
        assert_eq!(returned, solved);
        assert_eq!(
            coefficient_linear_residual(&[source], &[2.0 * returned]),
            0.0
        );
        assert!(reconstructed_norm < config.tolerance * f64::EPSILON);
    }

    #[test]
    fn stalled_refinement_keeps_its_best_complete_physical_iterate() {
        let mut history = RefinementHistory::new();
        for (index, norm) in [1.0, 0.1, 0.2, 0.11, 0.12].into_iter().enumerate() {
            assert_eq!(history.observe(norm, &index).unwrap(), index == 4);
        }
        assert_eq!(history.norm, 0.1);
        assert_eq!(history.finish(), 1);
        let mut improving = RefinementHistory::new();
        for index in 0..100 {
            assert!(!improving.observe(0.5_f64.powi(index), &index).unwrap());
        }
        assert_eq!(improving.finish(), 99);
        let mut slow = RefinementHistory::new();
        for (index, norm) in [1.0, 0.99, 0.98, 0.97].into_iter().enumerate() {
            assert_eq!(slow.observe(norm, &index).unwrap(), index == 3);
        }
        assert_eq!(slow.finish(), 3, "retain improvements during stagnation");
        assert!(RefinementHistory::new().observe(f64::NAN, &0).is_err());
    }

    fn closed_transport<S: Scalar>(t: S, c: &Matrix) -> [[S; 2]; 2] {
        let s = [
            [
                S::from_f64(1.7) + S::from_f64(0.31) * t + S::from_f64(0.07) * t * t,
                S::from_f64(0.23) + S::from_f64(0.17) * t - S::from_f64(0.09) * t * t,
            ],
            [
                S::from_f64(0.23) + S::from_f64(0.17) * t - S::from_f64(0.09) * t * t,
                S::from_f64(1.3) - S::from_f64(0.19) * t - S::from_f64(0.05) * t * t,
            ],
        ];
        let mut m = [[S::from_f64(0.0); 2]; 2];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                for a in 0..2 {
                    for b in 0..2 {
                        *value = *value + S::from_f64(c[(a, i)] * c[(b, j)]) * s[a][b];
                    }
                }
            }
        }
        let root = (m[0][0] * m[1][1] - m[0][1] * m[1][0]).sqrt();
        let scale = (m[0][0] + m[1][1] + S::from_f64(2.0) * root).sqrt();
        let determinant = (m[0][0] + root) * (m[1][1] + root) - m[0][1] * m[1][0];
        let inverse = [
            [
                (m[1][1] + root) * scale / determinant,
                -m[0][1] * scale / determinant,
            ],
            [
                -m[1][0] * scale / determinant,
                (m[0][0] + root) * scale / determinant,
            ],
        ];
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                S::from_f64(c[(i, 0)]) * inverse[0][j] + S::from_f64(c[(i, 1)]) * inverse[1][j]
            })
        })
    }

    #[test]
    fn all_retained_transport_derivatives_fit_closed_two_by_two_roots() {
        let overlap = vec![
            Matrix::from_row_major(2, 2, vec![1.7, 0.23, 0.23, 1.3]),
            Matrix::from_row_major(2, 2, vec![0.31, 0.17, 0.17, -0.19]),
            Matrix::from_row_major(2, 2, vec![0.07, -0.09, -0.09, -0.05]),
            Matrix::zeros(2, 2),
            Matrix::zeros(2, 2),
        ];
        // Distinct first/second metric coefficients do not commute. A
        // dense constant rotation leaves every tested matrix entry nonzero.
        // Transport expects a metric-orthonormal reference. Obtain its
        // closed-form root independently, then apply a dense rotation.
        let identity = Matrix::identity(2);
        let root = closed_transport(0.0, &identity);
        let angle = 0.37_f64;
        let rotation = Matrix::from_row_major(
            2,
            2,
            vec![angle.cos(), -angle.sin(), angle.sin(), angle.cos()],
        );
        let c = Matrix::from_row_major(2, 2, root.into_iter().flatten().collect())
            .matmul_seq(&rotation);
        let values = transport(&c, &overlap, 4);
        let first = c.transpose_matmul(&overlap[1].matmul_seq(&c));
        let second = c.transpose_matmul(&overlap[2].matmul_seq(&c));
        assert!(
            first
                .matmul_seq(&second)
                .max_abs_diff(&second.matmul_seq(&first))
                > 1e-3
        );
        let names = (1..=4)
            .flat_map(|n| {
                (0..2).flat_map(move |i| {
                    (0..2)
                        .map(move |j| format!("retained transport derivative={n} row={i} col={j}"))
                })
            })
            .collect::<Vec<_>>();
        let stages = [
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let oracle = |n: usize, t: f64, i: usize, j: usize| {
                    let root = closed_transport(Taylor::<4>::from_coefficients(&[t, 1.0]), &c);
                    root[i][j].c[n] * (1..=n).product::<usize>() as f64
                };
                (1..=4)
                    .flat_map(|n| {
                        let values = &values;
                        let oracle = &oracle;
                        (0..2).flat_map(move |i| {
                            (0..2).map(move |j| {
                                let analytic = (values[n].high[(i, j)] + values[n].low[(i, j)])
                                    * (1..=n).product::<usize>() as f64;
                                std::array::from_fn(|q| {
                                    let h = steps[q];
                                    (stencil.first(
                                        oracle(n - 1, h, i, j),
                                        oracle(n - 1, -h, i, j),
                                        oracle(n - 1, 2.0 * h, i, j),
                                        oracle(n - 1, -2.0 * h, i, j),
                                        h,
                                    ) - analytic)
                                        .abs()
                                })
                            })
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
        components::assert_no_failures("all16 retained transport derivatives", &failures);
    }

    #[test]
    fn intermediate_remainders_survive_a_second_contraction() {
        let a = Matrix::from_row_major(1, 2, vec![1.0, 1.0]);
        let b = Matrix::from_row_major(2, 1, vec![1.0, 2.0_f64.powi(-54)]);
        let ab = product(&retain(&[a]), &retain(&[b]), 0);
        assert_eq!(ab[0].high[(0, 0)], 1.0);
        assert_eq!(ab[0].low[(0, 0)], 2.0_f64.powi(-54));
        let mut residual = ab[0].clone();
        residual.add_entry(0, 0, -1.0);
        assert_eq!(residual.high[(0, 0)], 2.0_f64.powi(-54));
    }
}
