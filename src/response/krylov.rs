// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Matrix-free CG, MINRES and restarted complex GMRES. Acceptance always
//! uses a fresh residual of the supplied operator, never the projected one.

use super::complex::{dot, norm, Complex};
use crate::{DftError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorStructure {
    PositiveDefinite,
    Hermitian,
    General,
}

pub trait LinearOperator {
    fn dimension(&self) -> usize;
    fn structure(&self) -> OperatorStructure;
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>>;
}

/// Extend an existing real linear action to complex amplitudes. Sharing this
/// adapter keeps the real/imaginary convention identical across the magnetic,
/// transported-frame and restricted-open-shell response operators.
pub(crate) fn apply_real_linear(
    x: &[Complex],
    image: impl Fn(&[f64]) -> Result<Vec<f64>>,
) -> Result<Vec<Complex>> {
    let real: Vec<f64> = x.iter().map(|v| v.re).collect();
    let imaginary: Vec<f64> = x.iter().map(|v| v.im).collect();
    let imaginary_is_zero = imaginary.iter().all(|v| *v == 0.0);
    // Pure imaginary perturbations need one real-kernel evaluation. For an
    // entirely zero vector retain one call so the underlying dimension and
    // reference validation still runs.
    let a = if !imaginary_is_zero && real.iter().all(|v| *v == 0.0) {
        vec![0.0; x.len()]
    } else {
        image(&real)?
    };
    let b = if imaginary_is_zero {
        vec![0.0; a.len()]
    } else {
        image(&imaginary)?
    };
    if a.len() != x.len() || b.len() != x.len() {
        return Err(DftError::InvalidInput(
            "real linear operator changed the amplitude dimension during complex extension".into(),
        ));
    }
    Ok(a.into_iter()
        .zip(b)
        .map(|(re, im)| Complex::new(re, im))
        .collect())
}

/// Controls of every matrix-free response solve.
///
/// The iteration stops when the relative residual reaches `tolerance`. The
/// solution is then accepted only if a freshly recomputed residual of the
/// original operator is at most `accept_residual` (by default the same
/// value). `divergence_factor` stops a solve whose residual grows by that
/// factor over the smallest residual seen. `min_denominator` refuses a solve
/// in which a zeroth-order (preconditioner) denominator is smaller than the
/// stated value: such a near-singular or resonant response is reported
/// instead of being regularized. `preconditioner_floor` affects only the
/// preconditioner, never the operator.
#[derive(Clone, Debug)]
pub struct ResponseSolverConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub gmres_restart: usize,
    pub preconditioner_floor: f64,
    /// Acceptance bound of the fresh relative residual; `None` = `tolerance`.
    pub accept_residual: Option<f64>,
    pub divergence_factor: f64,
    pub min_denominator: f64,
}

impl Default for ResponseSolverConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 500,
            gmres_restart: 40,
            preconditioner_floor: 1e-3,
            accept_residual: None,
            divergence_factor: 1e8,
            min_denominator: 0.0,
        }
    }
}

impl ResponseSolverConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.tolerance.is_finite()
            || self.tolerance <= 0.0
            || self.tolerance >= 1.0
            || self.max_iterations == 0
            || self.gmres_restart == 0
            || !self.preconditioner_floor.is_finite()
            || self.preconditioner_floor <= 0.0
            || self
                .accept_residual
                .is_some_and(|a| !a.is_finite() || a < self.tolerance || a >= 1.0)
            || self.divergence_factor.is_nan()
            || self.divergence_factor <= 1.0
            || !self.min_denominator.is_finite()
            || self.min_denominator < 0.0
        {
            return Err(DftError::InvalidInput(
                "invalid response solver configuration: need 0 < tolerance <= accept_residual < 1, \
                 positive iteration counts and preconditioner floor, divergence_factor > 1 and a \
                 non-negative min_denominator"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The acceptance bound of the fresh residual.
    pub fn acceptance(&self) -> f64 {
        self.accept_residual.unwrap_or(self.tolerance)
    }

    /// Retain the requested relative bound and also bound the absolute
    /// equation residual. Only stopping criteria change; the operator and
    /// its physical denominators are untouched.
    pub(crate) fn for_absolute_residual(&self, source_norm: f64) -> Result<Self> {
        self.validate()?;
        if !source_norm.is_finite() || source_norm < 0.0 {
            return Err(DftError::InvalidInput(
                "absolute response acceptance requires a finite nonnegative source norm".into(),
            ));
        }
        let mut config = self.clone();
        if source_norm > 0.0 {
            let bound = self.acceptance() / source_norm;
            config.tolerance = self.tolerance.min(bound);
            config.accept_residual = Some(self.acceptance().min(bound));
        }
        config.validate()?;
        Ok(config)
    }
}

/// A relative linear residual alone does not bound a reconstructed physical
/// coefficient equation. Retry with a stricter solve only when the fresh
/// projector/stationarity check fails. Previous work arrays are released
/// before retrying; neither physical operators nor acceptance gates change.
pub(crate) fn with_physical_acceptance<T>(
    config: &ResponseSolverConfig,
    mut calculate: impl FnMut(&ResponseSolverConfig) -> Result<(T, f64)>,
) -> Result<T> {
    config.validate()?;
    let acceptance = config.acceptance();
    let mut inner = config.clone();
    for attempt in 0..4 {
        let (value, residual) = calculate(&inner)?;
        if residual.is_finite() && residual >= 0.0 && residual <= acceptance {
            return Ok(value);
        }
        drop(value);
        if !residual.is_finite() || residual < 0.0 || attempt == 3 {
            return Err(DftError::ResponseNotConverged {
                iterations: attempt + 1,
                residual,
            });
        }
        crate::logger::log(&format!(
            "[response] physical coefficient refinement {}: absolute residual={residual:.3e}, requested={acceptance:.3e}",
            attempt + 1
        ));
        inner.tolerance *= 0.5 * acceptance / residual;
        inner.accept_residual = Some(inner.tolerance);
        inner.validate()?;
    }
    unreachable!("the bounded physical response refinement returns on every terminal branch")
}

#[derive(Clone, Debug)]
pub struct LinearSolution {
    pub amplitudes: Vec<Complex>,
    pub relative_residual: f64,
    pub iterations: usize,
    pub method: &'static str,
}

fn apply(op: &impl LinearOperator, x: &[Complex]) -> Result<Vec<Complex>> {
    let y = op.apply(x)?;
    if y.len() != op.dimension() || y.iter().any(|v| !v.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "response operator returned an invalid image".into(),
        ));
    }
    Ok(y)
}

#[cfg(test)]
mod physical_acceptance_tests {
    use super::*;

    struct Diagonal;
    impl LinearOperator for Diagonal {
        fn dimension(&self) -> usize {
            64
        }
        fn structure(&self) -> OperatorStructure {
            OperatorStructure::PositiveDefinite
        }
        fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
            Ok(x.iter()
                .enumerate()
                .map(|(i, x)| *x * (i + 1) as f64)
                .collect())
        }
    }

    #[test]
    fn source_norm_changes_stopping_criteria_without_changing_the_model() {
        let config = ResponseSolverConfig {
            tolerance: 1e-8,
            ..Default::default()
        };
        for source in [0.0, 0.1, 1.0, 1000.0] {
            let inner = config.for_absolute_residual(source).unwrap();
            assert!(inner.tolerance <= config.tolerance);
            assert!(inner.acceptance() <= config.acceptance());
            if source > 0.0 {
                assert!(inner.acceptance() <= config.acceptance() / source);
            }
            assert_eq!(inner.preconditioner_floor, config.preconditioner_floor);
            assert_eq!(inner.min_denominator, config.min_denominator);
        }
        assert!(config.for_absolute_residual(-1.0).is_err());
        assert!(config.for_absolute_residual(f64::INFINITY).is_err());
    }

    #[test]
    fn fresh_absolute_equations_trigger_a_stricter_relative_solve() {
        let operator = Diagonal;
        let rhs: Vec<_> = (0..64)
            .map(|i| Complex::from(100.0 + 0.31 * i as f64))
            .collect();
        let config = ResponseSolverConfig {
            tolerance: 1e-8,
            ..Default::default()
        };
        let mut calls = 0;
        let solution = with_physical_acceptance(&config, |inner| {
            calls += 1;
            let value = solve(&operator, &rhs, None, inner)?;
            let actual = operator.apply(&value.amplitudes)?;
            let defect = actual
                .iter()
                .zip(&rhs)
                .map(|(a, b)| (*a - *b).abs())
                .fold(0.0_f64, f64::max);
            Ok((value, defect))
        })
        .unwrap();
        assert!(calls > 1);
        assert!(solution.relative_residual <= config.tolerance);
        let actual = operator.apply(&solution.amplitudes).unwrap();
        assert!(actual
            .iter()
            .zip(&rhs)
            .all(|(a, b)| (*a - *b).abs() <= config.acceptance()));
    }

    #[test]
    fn irreducible_or_nonfinite_physical_errors_are_never_accepted() {
        let config = ResponseSolverConfig::default();
        let mut calls = 0;
        let failed = with_physical_acceptance(&config, |_| {
            calls += 1;
            Ok(((), 7.0 * config.acceptance()))
        });
        assert!(failed.is_err());
        assert_eq!(calls, 4);
        assert!(with_physical_acceptance(&config, |_| Ok(((), f64::NAN))).is_err());
    }
}

fn residual(op: &impl LinearOperator, b: &[Complex], x: &[Complex]) -> Result<Vec<Complex>> {
    Ok(b.iter().zip(apply(op, x)?).map(|(&b, ax)| b - ax).collect())
}

/// Positive two-sided scaling preserves Hermiticity, including an indefinite
/// constraint saddle system. With S=D^-1/2, solve S A S y=S b and return x=S y.
/// The caller always recomputes the residual of A x=b after this transformation.
struct DiagonallyScaled<'a, O> {
    original: &'a O,
    inverse_sqrt: Vec<f64>,
}

impl<O: LinearOperator> LinearOperator for DiagonallyScaled<'_, O> {
    fn dimension(&self) -> usize {
        self.original.dimension()
    }
    fn structure(&self) -> OperatorStructure {
        self.original.structure()
    }
    fn apply(&self, y: &[Complex]) -> Result<Vec<Complex>> {
        let x: Vec<_> = y
            .iter()
            .zip(&self.inverse_sqrt)
            .map(|(&v, &s)| v * s)
            .collect();
        let mut value = apply(self.original, &x)?;
        for (v, &s) in value.iter_mut().zip(&self.inverse_sqrt) {
            *v = *v * s;
        }
        Ok(value)
    }
}

pub fn solve(
    op: &impl LinearOperator,
    rhs: &[Complex],
    diagonal: Option<&[f64]>,
    config: &ResponseSolverConfig,
) -> Result<LinearSolution> {
    config.validate()?;
    if rhs.len() != op.dimension() || rhs.iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "response RHS has an invalid dimension or value".into(),
        ));
    }
    if let Some(d) = diagonal {
        if d.len() != rhs.len() || d.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(DftError::InvalidInput(
                "response preconditioner must be finite and positive".into(),
            ));
        }
    }
    if let Some(smallest) = diagonal.and_then(|d| d.iter().copied().reduce(f64::min)) {
        if smallest < config.min_denominator {
            return Err(DftError::Unsupported {
                feature: format!("response with a zeroth-order denominator of {smallest:.3e}"),
                missing: format!(
                    "a resolved (non-degenerate, non-resonant) response; the denominator is below \
                     response_min_denominator = {:.3e}",
                    config.min_denominator
                ),
            });
        }
    }
    let method = match op.structure() {
        OperatorStructure::PositiveDefinite => "CG",
        OperatorStructure::Hermitian => "MINRES",
        OperatorStructure::General => "GMRES",
    };
    let scale = rhs
        .iter()
        .fold(0.0_f64, |m, z| m.max(z.re.abs()).max(z.im.abs()));
    if scale == 0.0 {
        return Ok(LinearSolution {
            amplitudes: vec![Complex::ZERO; rhs.len()],
            relative_residual: 0.0,
            iterations: 0,
            method,
        });
    }
    // Solve an equivalent scaled RHS, avoiding underflow in inner products
    // for symmetry-forbidden or high-order response sources.
    let normalized: Vec<_> = rhs.iter().map(|&b| b / scale).collect();
    let preconditioned = diagonal
        .filter(|_| op.structure() != OperatorStructure::PositiveDefinite)
        .map(|d| DiagonallyScaled {
            original: op,
            inverse_sqrt: d
                .iter()
                .map(|v| v.max(config.preconditioner_floor).sqrt().recip())
                .collect(),
        });
    let minimum = |source: &[Complex], controls: &ResponseSolverConfig, hermitian| {
        let Some(scaled) = &preconditioned else {
            return minimum_residual(op, source, controls, hermitian);
        };
        let transformed: Vec<_> = source
            .iter()
            .zip(&scaled.inverse_sqrt)
            .map(|(&v, &s)| v * s)
            .collect();
        let mut result = minimum_residual(scaled, &transformed, controls, hermitian)?;
        for (x, &s) in result.amplitudes.iter_mut().zip(&scaled.inverse_sqrt) {
            *x = *x * s;
        }
        Ok(result)
    };
    let iterate = |source: &[Complex], controls: &ResponseSolverConfig| match op.structure() {
        OperatorStructure::PositiveDefinite => cg(op, source, diagonal, controls),
        OperatorStructure::Hermitian => minimum(source, controls, true),
        OperatorStructure::General => minimum(source, controls, false),
    };
    let mut result = iterate(&normalized, config)?;
    for x in &mut result.amplitudes {
        *x = *x * scale;
    }
    if result.amplitudes.iter().any(|x| !x.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "response amplitudes overflow".into(),
        ));
    }
    let mut correction_roundoff = vec![Complex::ZERO; rhs.len()];
    loop {
        let fresh = residual(op, rhs, &result.amplitudes)?;
        let scaled: Vec<_> = fresh.iter().map(|&r| r / scale).collect();
        result.relative_residual = norm(&scaled) / norm(&normalized);
        if result.relative_residual <= config.acceptance() {
            return Ok(result);
        }
        if result.iterations >= config.max_iterations {
            return Err(DftError::ResponseNotConverged {
                iterations: result.iterations,
                residual: result.relative_residual,
            });
        }
        // Rescaling a converged Krylov iterate changes the rounding in an
        // integral-direct action. Refine the original equation instead of
        // rejecting that iterate immediately or accepting its scaled residual.
        // Every correction uses the same physical operator and remaining
        // iteration budget; no diagonal shift or relaxed tolerance is used.
        let correction_scale = fresh
            .iter()
            .fold(0.0_f64, |m, z| m.max(z.re.abs()).max(z.im.abs()));
        let source: Vec<_> = fresh.iter().map(|&r| r / correction_scale).collect();
        let mut remaining = config.clone();
        remaining.max_iterations -= result.iterations;
        // The correction only has to reduce the current original residual
        // below its requested bound. Solving that tiny correction to the
        // original relative tolerance would demand needless accuracy (and
        // can stagnate in integral contractions). The final original-scale
        // acceptance above remains unchanged.
        remaining.tolerance = (0.25 * config.tolerance / result.relative_residual)
            .max(config.tolerance)
            .min(0.25);
        remaining.accept_residual = None;
        #[cfg(test)]
        eprintln!("original-scale response refinement iterations={} residual={:.16e} correction_tolerance={:.16e}", result.iterations, result.relative_residual, remaining.tolerance);
        let correction = match iterate(&source, &remaining) {
            Err(DftError::ResponseNotConverged { iterations, .. }) => {
                return Err(DftError::ResponseNotConverged {
                    iterations: result.iterations + iterations,
                    residual: result.relative_residual,
                });
            }
            value => value?,
        };
        result.iterations += correction.iterations;
        for ((x, roundoff), dx) in result
            .amplitudes
            .iter_mut()
            .zip(&mut correction_roundoff)
            .zip(correction.amplitudes)
        {
            let increment = dx * correction_scale - *roundoff;
            let next = *x + increment;
            *roundoff = (next - *x) - increment;
            *x = next;
        }
    }
}

fn cg(
    op: &impl LinearOperator,
    b: &[Complex],
    diagonal: Option<&[f64]>,
    cfg: &ResponseSolverConfig,
) -> Result<LinearSolution> {
    let bnorm = norm(b);
    let precondition = |r: &[Complex]| -> Vec<Complex> {
        r.iter()
            .enumerate()
            .map(|(i, &v)| v / diagonal.map_or(1.0, |d| d[i].max(cfg.preconditioner_floor)))
            .collect()
    };
    let mut x = vec![Complex::ZERO; b.len()];
    let mut r = b.to_vec();
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z).re;
    let mut best = 1.0_f64;
    for iteration in 1..=cfg.max_iterations {
        let ap = apply(op, &p)?;
        let pap = dot(&p, &ap);
        if pap.re <= 0.0 || !pap.is_finite() || pap.im.abs() > 1e-10 * norm(&p) * norm(&ap) {
            return Err(DftError::LinearAlgebra(
                "CG requires a Hermitian positive-definite operator".into(),
            ));
        }
        let alpha = rz / pap.re;
        for i in 0..x.len() {
            x[i] += p[i] * alpha;
            r[i] -= ap[i] * alpha;
        }
        let current = norm(&r) / bnorm;
        if current <= cfg.tolerance || iteration % 20 == 0 || current > cfg.divergence_factor * best
        {
            r = residual(op, b, &x)?;
            let rr = norm(&r) / bnorm;
            // A recursively updated residual may be spuriously tiny after
            // cancellation. Only verified residuals belong in the divergence
            // history; otherwise restarting from the true residual can look
            // like divergence by many orders of magnitude.
            best = best.min(rr);
            if rr > cfg.divergence_factor * best {
                return Err(DftError::ResponseNotConverged {
                    iterations: iteration,
                    residual: rr,
                });
            }
            if rr <= cfg.tolerance || current <= cfg.tolerance {
                // Hand a recursively converged candidate to the mandatory
                // original-scale check in solve(). If its fresh residual is
                // larger, solve() computes a small correction from zero;
                // repeatedly updating a large accumulated x here can lose
                // that correction to cancellation.
                return Ok(LinearSolution {
                    amplitudes: x,
                    relative_residual: rr,
                    iterations: iteration,
                    method: "CG",
                });
            }
            // A fresh residual requires a fresh conjugate-gradient cycle.
            z = precondition(&r);
            p = z.clone();
            rz = dot(&r, &z).re;
            continue;
        }
        z = precondition(&r);
        let next = dot(&r, &z).re;
        let beta = next / rz;
        for i in 0..p.len() {
            p[i] = z[i] + p[i] * beta;
        }
        rz = next;
    }
    let rr = norm(&residual(op, b, &x)?) / bnorm;
    exhausted(x, rr, cfg.max_iterations, "CG", cfg)
}

fn rotation(a: Complex, b: Complex) -> (f64, Complex) {
    if b.abs() == 0.0 {
        return (1.0, Complex::ZERO);
    }
    if a.abs() == 0.0 {
        return (0.0, b.conj() / b.abs());
    }
    let length = a.abs().hypot(b.abs());
    (a.abs() / length, (a / a.abs()) * b.conj() / length)
}

fn minimum_residual(
    op: &impl LinearOperator,
    b: &[Complex],
    cfg: &ResponseSolverConfig,
    hermitian: bool,
) -> Result<LinearSolution> {
    let bnorm = norm(b);
    let n = b.len();
    let mut x = vec![Complex::ZERO; n];
    let mut iterations = 0;
    let method = if hermitian { "MINRES" } else { "GMRES" };
    let mut best = 1.0_f64;
    while iterations < cfg.max_iterations {
        let r = residual(op, b, &x)?;
        let beta = norm(&r);
        if beta / bnorm <= cfg.tolerance {
            return Ok(LinearSolution {
                amplitudes: x,
                relative_residual: beta / bnorm,
                iterations,
                method,
            });
        }
        let width = if hermitian { n } else { cfg.gmres_restart }
            .min(n)
            .min(cfg.max_iterations - iterations);
        let base = x.clone();
        let mut v = vec![r.iter().map(|&z| z / beta).collect::<Vec<_>>()];
        let mut h = vec![vec![Complex::ZERO; width]; width + 1];
        let mut g = vec![Complex::ZERO; width + 1];
        g[0] = beta.into();
        let mut rotations: Vec<(f64, Complex)> = Vec::new();
        let mut previous_beta = 0.0;
        for j in 0..width {
            let mut w = apply(op, &v[j])?;
            if hermitian {
                // Hermitian Lanczos: only the three-term recurrence is
                // needed. QR of its tridiagonal relation defines MINRES.
                let alpha = dot(&v[j], &w);
                if alpha.im.abs() > 1e-10 * norm(&w).max(f64::MIN_POSITIVE) {
                    return Err(DftError::LinearAlgebra(
                        "MINRES requires a Hermitian operator".into(),
                    ));
                }
                h[j][j] = alpha.re.into();
                for i in 0..n {
                    w[i] -= v[j][i] * alpha.re;
                }
                if j > 0 {
                    h[j - 1][j] = previous_beta.into();
                    for i in 0..n {
                        w[i] -= v[j - 1][i] * previous_beta;
                    }
                }
            } else {
                // Two MGS passes protect orthogonality near resonances.
                for _ in 0..2 {
                    for k in 0..=j {
                        let projection = dot(&v[k], &w);
                        h[k][j] += projection;
                        for i in 0..n {
                            w[i] -= v[k][i] * projection;
                        }
                    }
                }
            }
            let next_norm = norm(&w);
            h[j + 1][j] = next_norm.into();
            previous_beta = next_norm;
            for (k, &(c, s)) in rotations.iter().enumerate() {
                let top = h[k][j];
                let bottom = h[k + 1][j];
                h[k][j] = top * c + bottom * s;
                h[k + 1][j] = -top * s.conj() + bottom * c;
            }
            let (c, s) = rotation(h[j][j], h[j + 1][j]);
            let top = h[j][j];
            let bottom = h[j + 1][j];
            h[j][j] = top * c + bottom * s;
            h[j + 1][j] = Complex::ZERO;
            let top = g[j];
            let bottom = g[j + 1];
            g[j] = top * c + bottom * s;
            g[j + 1] = -top * s.conj() + bottom * c;
            rotations.push((c, s));
            iterations += 1;
            let mut y = g[..=j].to_vec();
            for row in (0..=j).rev() {
                let mut remainder = y[row];
                for col in row + 1..=j {
                    remainder -= h[row][col] * y[col];
                }
                if h[row][row].abs() <= f64::MIN_POSITIVE {
                    return Err(DftError::LinearAlgebra(
                        "singular projected response operator".into(),
                    ));
                }
                y[row] = remainder / h[row][row];
            }
            x.clone_from(&base);
            for k in 0..=j {
                for i in 0..n {
                    x[i] += v[k][i] * y[k];
                }
            }
            let rr = norm(&residual(op, b, &x)?) / bnorm;
            if rr <= cfg.tolerance {
                return Ok(LinearSolution {
                    amplitudes: x,
                    relative_residual: rr,
                    iterations,
                    method,
                });
            }
            best = best.min(rr);
            if rr > cfg.divergence_factor * best {
                return Err(DftError::ResponseNotConverged {
                    iterations,
                    residual: rr,
                });
            }
            if next_norm <= f64::MIN_POSITIVE {
                break;
            }
            v.push(w.iter().map(|&z| z / next_norm).collect());
        }
    }
    let rr = norm(&residual(op, b, &x)?) / bnorm;
    exhausted(x, rr, iterations, method, cfg)
}

/// End of the iteration budget: hand an iterate within the acceptance bound
/// to the fresh-residual check of [`solve`], refuse anything else.
fn exhausted(
    amplitudes: Vec<Complex>,
    relative_residual: f64,
    iterations: usize,
    method: &'static str,
    cfg: &ResponseSolverConfig,
) -> Result<LinearSolution> {
    if relative_residual <= cfg.acceptance() {
        Ok(LinearSolution {
            amplitudes,
            relative_residual,
            iterations,
            method,
        })
    } else {
        Err(DftError::ResponseNotConverged {
            iterations,
            residual: relative_residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_restarts_use_verified_residual_history_and_match_dense_lu() {
        use crate::linalg::{solve_linear, Matrix};
        struct Dense(Matrix);
        impl LinearOperator for Dense {
            fn dimension(&self) -> usize {
                self.0.rows
            }
            fn structure(&self) -> OperatorStructure {
                OperatorStructure::PositiveDefinite
            }
            fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
                Ok((0..self.0.rows)
                    .map(|i| {
                        let row: Vec<_> = (0..self.0.cols)
                            .map(|j| Complex::from(self.0[(i, j)]))
                            .collect();
                        dot(&row, x)
                    })
                    .collect())
            }
        }
        let op = Dense(Matrix::from_row_major(
            4,
            4,
            vec![
                0.35872355597260974,
                0.284318778354338,
                0.24436786058057983,
                0.2213755122565804,
                0.284318778354338,
                0.31598268411707037,
                0.27431210553598945,
                0.21526318563612173,
                0.24436786058057983,
                0.27431210553598945,
                0.24343595926792308,
                0.18820853188605974,
                0.2213755122565804,
                0.21526318563612173,
                0.18820853188605974,
                0.15487017507243792,
            ],
        ));
        let b = [
            0.49802897079787617,
            0.463856731867568,
            1.0,
            -0.8124141116671257,
        ];
        let expected = solve_linear(&op.0, &b).unwrap();
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let rhs: Vec<_> = b.into_iter().map(Complex::from).collect();
        let solution = solve(&op, &rhs, None, &config).unwrap();
        let difference = solution
            .amplitudes
            .iter()
            .zip(expected)
            .map(|(a, b)| (a.re - b).abs())
            .fold(0.0_f64, f64::max);
        println!("verified CG restart iterations={} fresh residual={:.16e} dense LU difference={difference:.16e}", solution.iterations, solution.relative_residual);
        assert!(solution.relative_residual <= config.tolerance && difference < 2e-5);
    }

    #[test]
    fn real_operator_extension_matches_an_independent_complex_dense_action() {
        let matrix = [[2.0, 3.0, -1.0], [0.0, -0.5, 4.0], [1.0, 0.0, 5.0]];
        let x = [
            Complex::new(0.2, -0.3),
            Complex::new(-0.7, 0.4),
            Complex::new(0.9, 0.8),
        ];
        let actual = apply_real_linear(&x, |v| {
            Ok(matrix
                .iter()
                .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
                .collect())
        })
        .unwrap();
        for i in 0..3 {
            let expected = (0..3).fold(Complex::ZERO, |sum, j| sum + x[j] * matrix[i][j]);
            assert!((actual[i] - expected).abs() < 1e-15);
        }
        let calls = std::cell::Cell::new(0);
        let real = [Complex::new(0.7, 0.0), Complex::new(-0.4, 0.0)];
        let result = apply_real_linear(&real, |v| {
            calls.set(calls.get() + 1);
            Ok(v.iter().map(|a| 2.0 * a).collect())
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(result, real.map(|v| v * 2.0));
        assert!(apply_real_linear(&real, |_| Ok(vec![1.0])).is_err());
        for input in [real.map(|v| Complex::new(0.0, v.re)), [Complex::ZERO; 2]] {
            calls.set(0);
            let result = apply_real_linear(&input, |v| {
                calls.set(calls.get() + 1);
                Ok(v.iter().map(|a| 2.0 * a).collect())
            })
            .unwrap();
            assert_eq!(calls.get(), 1);
            assert_eq!(result, input.map(|v| v * 2.0));
            assert!(apply_real_linear(&input, |_| Ok(vec![1.0])).is_err());
        }
    }

    struct Diagonal(Vec<f64>, OperatorStructure);
    impl LinearOperator for Diagonal {
        fn dimension(&self) -> usize {
            self.0.len()
        }
        fn structure(&self) -> OperatorStructure {
            self.1
        }
        fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
            Ok(x.iter().zip(&self.0).map(|(&v, &d)| v * d).collect())
        }
    }

    #[test]
    fn positive_scaling_accelerates_indefinite_and_complex_response_without_changing_residuals() {
        struct Weighted {
            diagonal: Vec<f64>,
            general: bool,
        }
        impl LinearOperator for Weighted {
            fn dimension(&self) -> usize {
                self.diagonal.len()
            }
            fn structure(&self) -> OperatorStructure {
                if self.general {
                    OperatorStructure::General
                } else {
                    OperatorStructure::Hermitian
                }
            }
            fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
                Ok((0..x.len())
                    .map(|i| {
                        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                        let mut value = x[i]
                            * Complex::new(
                                sign * self.diagonal[i],
                                if self.general {
                                    -0.17 * self.diagonal[i]
                                } else {
                                    0.0
                                },
                            );
                        if i > 0 {
                            value += x[i - 1]
                                * (0.03 * (self.diagonal[i] * self.diagonal[i - 1]).sqrt());
                        }
                        if i + 1 < x.len() {
                            value += x[i + 1]
                                * (0.03 * (self.diagonal[i] * self.diagonal[i + 1]).sqrt());
                        }
                        value
                    })
                    .collect())
            }
        }
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            max_iterations: 20,
            preconditioner_floor: 1e-8,
            ..Default::default()
        };
        for general in [false, true] {
            let op = Weighted {
                diagonal: (0..32)
                    .map(|i| 10.0_f64.powf(-6.0 + 12.0 * i as f64 / 31.0))
                    .collect(),
                general,
            };
            let expected: Vec<_> = op
                .diagonal
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    Complex::new((i as f64 + 0.3).sin(), (i as f64 + 0.7).cos()) / d.sqrt()
                })
                .collect();
            let rhs = op.apply(&expected).unwrap();
            assert!(solve(&op, &rhs, None, &config).is_err());
            let solved = solve(&op, &rhs, Some(&op.diagonal), &config).unwrap();
            let error: Vec<_> = solved
                .amplitudes
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| a - b)
                .collect();
            let original = norm(&residual(&op, &rhs, &solved.amplitudes).unwrap()) / norm(&rhs);
            let relative_error = norm(&error) / norm(&expected);
            println!("preconditioned {} iterations={} original_residual={original:.16e} solution_error={relative_error:.16e}",
                     solved.method, solved.iterations);
            assert!(original <= config.acceptance());
            assert!((original - solved.relative_residual).abs() < 1e-15);
            assert!(relative_error < 1e-10);
            assert!(solved.iterations < config.max_iterations);
        }
    }

    #[test]
    fn small_denominators_are_refused_not_floored() {
        let op = Diagonal(vec![1.0, 1e-7, 2.0], OperatorStructure::PositiveDefinite);
        let rhs = vec![Complex::ONE; 3];
        let config = ResponseSolverConfig {
            min_denominator: 1e-6,
            ..Default::default()
        };
        let error = solve(&op, &rhs, Some(&op.0), &config).unwrap_err();
        assert!(matches!(error, DftError::Unsupported { .. }), "{error}");
        // Without the bound the same physical denominator is solved exactly.
        let solution = solve(&op, &rhs, Some(&op.0), &ResponseSolverConfig::default()).unwrap();
        assert!((solution.amplitudes[1].re - 1e7).abs() < 1e-3);
    }

    #[test]
    fn acceptance_bound_and_configuration_are_validated() {
        let op = Diagonal(vec![1.0, 3.0, 7.0, 11.0], OperatorStructure::General);
        let rhs = vec![Complex::new(1.0, 0.5); 4];
        // Two GMRES iterations cannot resolve four distinct eigenvalues.
        let strict = ResponseSolverConfig {
            max_iterations: 2,
            gmres_restart: 2,
            ..Default::default()
        };
        assert!(matches!(
            solve(&op, &rhs, None, &strict),
            Err(DftError::ResponseNotConverged { .. })
        ));
        let relaxed = ResponseSolverConfig {
            accept_residual: Some(0.9),
            ..strict.clone()
        };
        let accepted = solve(&op, &rhs, None, &relaxed).unwrap();
        assert!(accepted.relative_residual > strict.tolerance && accepted.relative_residual <= 0.9);
        for invalid in [
            ResponseSolverConfig {
                accept_residual: Some(1e-12),
                ..Default::default()
            },
            ResponseSolverConfig {
                divergence_factor: 1.0,
                ..Default::default()
            },
            ResponseSolverConfig {
                min_denominator: -1.0,
                ..Default::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
}
