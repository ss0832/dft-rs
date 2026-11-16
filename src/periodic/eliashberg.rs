// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Isotropic Migdal-Eliashberg equations on the imaginary axis for a discrete
//! phonon spectrum `a2F = 1/2 sum_nu w_nu lambda_nu delta(w - w_nu)`, so that
//! `lambda(n) = sum_nu lambda_nu w_nu^2 / (w_nu^2 + (2 pi n T)^2)`.
//!
//! With the positive Matsubara frequencies `w_n = pi T (2n + 1)`,
//! `n = 0..N-1`, up to the energy cutoff, and the even gap symmetry
//! `Delta_(-n-1) = Delta_n`, the nonlinear equations are
//!
//! ```text
//! Z_n = 1 + (pi T / w_n) sum_m [lambda(n - m) - lambda(n + m + 1)] w_m / sqrt(w_m^2 + Delta_m^2)
//! Z_n Delta_n = pi T sum_m [lambda(n - m) + lambda(n + m + 1) - 2 mu*] Delta_m / sqrt(w_m^2 + Delta_m^2)
//! ```
//!
//! with the Coulomb pseudopotential `mu*` acting unscaled on every
//! frequency below the cutoff. The cutoff acts on `mu*` and on the gap
//! only: `Z` is summed over all frequencies, the part beyond the cutoff
//! being the normal-state tail of [`Spectrum::tail`]. For `Delta -> 0` the square roots become
//! `w_m` and the equations are the linearized gap equation solved by
//! [`crate::periodic::electron_phonon::eliashberg_tc`]; both share the
//! spectrum, the Matsubara grid and the kernel of this module, so the
//! nonlinear gap vanishes exactly where the largest eigenvalue of the
//! linearized kernel is one. The gap on the real axis follows from the
//! Matsubara values by the Vidberg-Serene continued-fraction (Thiele)
//! interpolant, which is ill-conditioned in double precision for many
//! points.

use crate::error::{DftError, Result};
use crate::response::complex::Complex;
use rayon::prelude::*;

/// Largest number of positive Matsubara frequencies.
pub(crate) const MAX_MATSUBARA: usize = 4000;

/// Included modes `(w_nu, lambda_nu)` with positive frequency and coupling.
pub(crate) struct Spectrum {
    modes: Vec<(f64, f64)>,
}

impl Spectrum {
    pub(crate) fn new(frequencies: &[f64], mode_lambda: &[f64]) -> Self {
        let modes = frequencies
            .iter()
            .zip(mode_lambda)
            .filter(|(w, l)| **w > 0.0 && **l > 0.0)
            .map(|(w, l)| (*w, *l))
            .collect();
        Self { modes }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.modes.is_empty()
    }

    /// Highest included mode frequency.
    pub(crate) fn highest(&self) -> f64 {
        self.modes.iter().fold(0.0_f64, |m, (w, _)| m.max(*w))
    }

    /// `lambda` at the bosonic Matsubara energy `nu`.
    pub(crate) fn lambda_at(&self, nu: f64) -> f64 {
        self.modes
            .iter()
            .map(|(w, l)| l * w * w / (w * w + nu * nu))
            .sum()
    }

    /// `lambda(2 pi T k)` for `k = 0..2n+2`, the indices reached by
    /// `|i - m|` and `i + m + 1` on an `n`-point grid.
    pub(crate) fn table(&self, temperature: f64, n: usize) -> Vec<f64> {
        let pi_t = std::f64::consts::PI * temperature;
        (0..2 * n + 2)
            .map(|k| self.lambda_at(2.0 * pi_t * k as f64))
            .collect()
    }
}

impl Spectrum {
    /// Normal-state tail of the renormalization sum beyond the `n`-point grid,
    /// `sum_(m >= n) [lambda(m - i) - lambda(m + i + 1)]` for `i = 0..n`
    /// (there `Delta = 0`, so the weights of `Z` are one). The cutoff acts on
    /// the pseudopotential only; `Z` is summed over all frequencies. With
    /// `S(j) = sum_(k >= j) lambda(2 pi T k)` the tail is
    /// `S(n - i) - S(n + i + 1)`; `S` is accumulated downward from
    /// `K = 2n + 2` with the Euler-Maclaurin tail of every Lorentzian,
    /// `int_K^inf f + f(K)/2 - f'(K)/12`.
    pub(crate) fn tail(&self, temperature: f64, n: usize, table: &[f64]) -> Vec<f64> {
        let c = 2.0 * std::f64::consts::PI * temperature;
        let k = 2 * n + 2;
        let kf = k as f64;
        let start: f64 = self
            .modes
            .iter()
            .map(|(w, l)| {
                let d = w * w + c * c * kf * kf;
                let integral = l * w / c * (std::f64::consts::FRAC_PI_2 - (c * kf / w).atan());
                let f = l * w * w / d;
                let df = -2.0 * l * w * w * c * c * kf / (d * d);
                integral + 0.5 * f - df / 12.0
            })
            .sum();
        let mut s = vec![0.0; k + 1];
        s[k] = start;
        for j in (1..k).rev() {
            s[j] = s[j + 1] + table[j];
        }
        (0..n).map(|i| s[n - i] - s[n + i + 1]).collect()
    }
}

/// Number of positive Matsubara frequencies `pi T (2n + 1)` up to `cutoff`.
pub(crate) fn matsubara_count(temperature: f64, cutoff: f64) -> usize {
    let pi_t = std::f64::consts::PI * temperature;
    let n_max = ((cutoff / pi_t - 1.0) / 2.0).ceil().max(1.0) as usize;
    n_max.min(MAX_MATSUBARA)
}

/// Matsubara frequency `w_m = pi T (2m + 1)`.
pub(crate) fn matsubara_frequency(temperature: f64, m: usize) -> f64 {
    std::f64::consts::PI * temperature * (2 * m + 1) as f64
}

/// Symmetric part of the kernel (without the factor `pi T`):
/// `lambda(i - m) + lambda(i + m + 1) - 2 mu*`.
pub(crate) fn kernel(table: &[f64], i: usize, m: usize, mu_star: f64) -> f64 {
    table[i.abs_diff(m)] + table[i + m + 1] - 2.0 * mu_star
}

/// Renormalization `Z_n = 1 + (pi T / w_n) {sum_m [lambda(n - m) - lambda(n + m + 1)] a_m + tail_n}`
/// with `a_m = 1` in the normal state and `w_m / sqrt(w_m^2 + Delta_m^2)` in
/// the superconducting state, and the normal-state `tail` of
/// [`Spectrum::tail`] for the frequencies beyond the cutoff.
pub(crate) fn renormalization(table: &[f64], tail: &[f64], pi_t: f64, weights: &[f64]) -> Vec<f64> {
    let n = weights.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let w = pi_t * (2 * i + 1) as f64;
            let sum: f64 = (0..n)
                .map(|m| (table[i.abs_diff(m)] - table[i + m + 1]) * weights[m])
                .sum();
            1.0 + pi_t / w * (sum + tail[i])
        })
        .collect()
}

/// Controls of the nonlinear gap iteration.
#[derive(Clone, Copy, Debug)]
pub struct GapOptions {
    /// Relative max-norm change of `Delta` at convergence.
    pub tolerance: f64,
    pub max_iter: usize,
    /// Linear mixing parameter of the fixed-point iteration.
    pub mixing: f64,
    /// Number of previous iterates of the Anderson acceleration
    /// (0: plain linear mixing).
    pub anderson_depth: usize,
    /// Constant starting gap (Hartree); default `0.2` times the highest mode.
    pub initial_gap: Option<f64>,
}

impl Default for GapOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iter: 2000,
            mixing: 0.3,
            anderson_depth: 5,
            initial_gap: None,
        }
    }
}

/// Solution of the nonlinear gap equations on the positive Matsubara axis.
#[derive(Clone, Debug)]
pub struct EliashbergGap {
    /// `w_n = pi T (2n + 1)` (Hartree).
    pub matsubara: Vec<f64>,
    pub z: Vec<f64>,
    /// `Delta(i w_n)` (Hartree); identically zero in the normal state.
    pub delta: Vec<f64>,
    pub iterations: usize,
    /// Relative max-norm change of the last iteration.
    pub residual: f64,
    pub converged: bool,
}

/// One application of the gap equations to `delta`: the new gap and the
/// renormalization evaluated at `delta`.
fn gap_map(
    table: &[f64],
    tail: &[f64],
    pi_t: f64,
    mu_star: f64,
    w: &[f64],
    delta: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = w.len();
    let roots: Vec<f64> = w.iter().zip(delta).map(|(w, d)| w.hypot(*d)).collect();
    let weights: Vec<f64> = w.iter().zip(&roots).map(|(w, r)| w / r).collect();
    let ratios: Vec<f64> = delta.iter().zip(&roots).map(|(d, r)| d / r).collect();
    let z = renormalization(table, tail, pi_t, &weights);
    let next: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let sum: f64 = (0..n)
                .map(|m| kernel(table, i, m, mu_star) * ratios[m])
                .sum();
            pi_t * sum / z[i]
        })
        .collect();
    (next, z)
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Solve `A x = b` for a small dense system by Gaussian elimination with
/// partial pivoting; `None` when singular.
fn solve_small(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if !(a[pivot][col].abs() > 1e-300) {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    Some(x)
}

/// Nonlinear isotropic gap `Delta(i w_n)` at `temperature` (`kT`, Hartree)
/// for the discrete spectrum `(frequencies, mode_lambda)` (Hartree),
/// `mu_star` and the Matsubara energy `cutoff` (Hartree).
///
/// The fixed-point iteration `Delta <- F(Delta)` uses linear mixing with
/// Anderson acceleration (`GapOptions::anderson_depth`). Above `T_c` the
/// iteration collapses to the normal state (`max |Delta|` below `1e-14` times
/// the highest mode), which is returned as `Delta = 0` with
/// `converged = true`.
pub fn eliashberg_gap(
    frequencies: &[f64],
    mode_lambda: &[f64],
    mu_star: f64,
    cutoff: f64,
    temperature: f64,
    options: &GapOptions,
) -> Result<EliashbergGap> {
    if !(temperature > 0.0 && temperature.is_finite()) {
        return Err(DftError::InvalidInput(
            "the Eliashberg temperature must be positive".into(),
        ));
    }
    if !(cutoff > 0.0 && cutoff.is_finite()) {
        return Err(DftError::InvalidInput(
            "the Matsubara cutoff must be positive".into(),
        ));
    }
    if !(options.tolerance > 0.0 && options.mixing > 0.0 && options.mixing <= 1.0) {
        return Err(DftError::InvalidInput(
            "gap tolerance must be positive and mixing in (0, 1]".into(),
        ));
    }
    let spectrum = Spectrum::new(frequencies, mode_lambda);
    let pi_t = std::f64::consts::PI * temperature;
    let n = matsubara_count(temperature, cutoff);
    let w: Vec<f64> = (0..n)
        .map(|m| matsubara_frequency(temperature, m))
        .collect();
    let table = spectrum.table(temperature, n);
    let tail = spectrum.tail(temperature, n, &table);
    let top = spectrum.highest();
    let normal_state = |iterations: usize| -> EliashbergGap {
        EliashbergGap {
            matsubara: w.clone(),
            z: renormalization(&table, &tail, pi_t, &vec![1.0; n]),
            delta: vec![0.0; n],
            iterations,
            residual: 0.0,
            converged: true,
        }
    };
    if spectrum.is_empty() {
        return Ok(normal_state(0));
    }
    let start = options.initial_gap.unwrap_or(0.2 * top);
    if !(start > 0.0) {
        return Err(DftError::InvalidInput(
            "the initial gap must be positive".into(),
        ));
    }
    let mut x = vec![start; n];
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut previous: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut residual = f64::INFINITY;
    for iteration in 1..=options.max_iter {
        let (g, z) = gap_map(&table, &tail, pi_t, mu_star, &w, &x);
        let scale = max_abs(&g);
        if scale < 1e-14 * top {
            return Ok(normal_state(iteration));
        }
        let f: Vec<f64> = g.iter().zip(&x).map(|(g, x)| g - x).collect();
        residual = max_abs(&f) / scale;
        if residual < options.tolerance {
            return Ok(EliashbergGap {
                matsubara: w,
                z,
                delta: g,
                iterations: iteration,
                residual,
                converged: true,
            });
        }
        if options.anderson_depth > 0 {
            if let Some((px, pf)) = &previous {
                let dx: Vec<f64> = x.iter().zip(px).map(|(a, b)| a - b).collect();
                let df: Vec<f64> = f.iter().zip(pf).map(|(a, b)| a - b).collect();
                history.push((dx, df));
                if history.len() > options.anderson_depth {
                    history.remove(0);
                }
            }
        }
        previous = Some((x.clone(), f.clone()));
        let m = history.len();
        let mut next: Vec<f64> = x
            .iter()
            .zip(&f)
            .map(|(x, f)| x + options.mixing * f)
            .collect();
        if m > 0 {
            let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
            let mut a: Vec<Vec<f64>> = (0..m)
                .map(|i| (0..m).map(|j| dot(&history[i].1, &history[j].1)).collect())
                .collect();
            let b: Vec<f64> = (0..m).map(|i| dot(&history[i].1, &f)).collect();
            let trace: f64 = (0..m).map(|i| a[i][i]).sum();
            for (i, row) in a.iter_mut().enumerate() {
                row[i] += 1e-12 * trace;
            }
            match solve_small(a, b) {
                Some(gamma) if gamma.iter().all(|c| c.is_finite()) => {
                    for (j, c) in gamma.iter().enumerate() {
                        for (k, value) in next.iter_mut().enumerate() {
                            *value -= c * (history[j].0[k] + options.mixing * history[j].1[k]);
                        }
                    }
                }
                _ => history.clear(),
            }
        }
        x = next;
    }
    let (g, z) = gap_map(&table, &tail, pi_t, mu_star, &w, &x);
    Ok(EliashbergGap {
        matsubara: w,
        z,
        delta: g,
        iterations: options.max_iter,
        residual,
        converged: false,
    })
}

/// Vidberg-Serene (Thiele) continued-fraction interpolant through
/// `(points, values)`, evaluated at `targets`.
///
/// With `g_0(z_i) = f_i` and
/// `g_p(z) = (g_(p-1)(z_(p-1)) - g_(p-1)(z)) / ((z - z_(p-1)) g_(p-1)(z))`
/// the coefficients are `a_p = g_p(z_p)` and the interpolant
/// `f(z) = A_N(z)/B_N(z)` follows from the three-term recursion
/// `A_(n+1) = A_n + (z - z_(n-1)) a_n A_(n-1)`, `A_0 = 0`, `A_1 = a_0`
/// (`B` likewise with `B_0 = B_1 = 1`).
pub fn pade_continuation(
    points: &[Complex],
    values: &[Complex],
    targets: &[Complex],
) -> Result<Vec<Complex>> {
    let n = points.len();
    if n == 0 || values.len() != n {
        return Err(DftError::InvalidInput(
            "Pade continuation needs equally many (at least one) points and values".into(),
        ));
    }
    let mut g = values.to_vec();
    let mut coefficients = Vec::with_capacity(n);
    coefficients.push(g[0]);
    for p in 1..n {
        let previous = g.clone();
        for i in p..n {
            let denominator = (points[i] - points[p - 1]) * previous[i];
            g[i] = (previous[p - 1] - previous[i]) / denominator;
        }
        if !g[p].is_finite() {
            return Err(DftError::InvalidInput(format!(
                "Pade coefficient {p} is not finite (repeated points or exhausted precision); use fewer points"
            )));
        }
        coefficients.push(g[p]);
    }
    let mut out = Vec::with_capacity(targets.len());
    for &z in targets {
        let (mut a_prev, mut a) = (Complex::ZERO, coefficients[0]);
        let (mut b_prev, mut b) = (Complex::ONE, Complex::ONE);
        for p in 1..n {
            let step = (z - points[p - 1]) * coefficients[p];
            let a_next = a + step * a_prev;
            let b_next = b + step * b_prev;
            a_prev = a;
            b_prev = b;
            a = a_next;
            b = b_next;
            // Common rescaling keeps the recursion in range.
            let scale = a.abs().max(b.abs());
            if scale > 1e100 || (scale < 1e-100 && scale > 0.0) {
                a_prev = a_prev / scale;
                b_prev = b_prev / scale;
                a = a / scale;
                b = b / scale;
            }
        }
        let value = a / b;
        if !value.is_finite() {
            return Err(DftError::InvalidInput(
                "Pade interpolant is not finite at a target point".into(),
            ));
        }
        out.push(value);
    }
    Ok(out)
}

/// `Delta(omega + i eta)` for every `omega` by continuing the first `points`
/// Matsubara values `Delta(i w_n)` to the real axis with
/// [`pade_continuation`]. A normal-state gap (identically zero) continues to
/// zero.
pub fn real_axis_gap(
    gap: &EliashbergGap,
    points: usize,
    omega: &[f64],
    eta: f64,
) -> Result<Vec<Complex>> {
    let count = points.min(gap.matsubara.len());
    if count == 0 {
        return Err(DftError::InvalidInput(
            "the real-axis continuation needs at least one Matsubara point".into(),
        ));
    }
    if gap.delta.iter().all(|d| *d == 0.0) {
        return Ok(vec![Complex::ZERO; omega.len()]);
    }
    let z: Vec<Complex> = gap.matsubara[..count]
        .iter()
        .map(|w| Complex::new(0.0, *w))
        .collect();
    let v: Vec<Complex> = gap.delta[..count]
        .iter()
        .map(|d| Complex::from(*d))
        .collect();
    let targets: Vec<Complex> = omega.iter().map(|w| Complex::new(*w, eta)).collect();
    pade_continuation(&z, &v, &targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::electron_phonon::{allen_dynes, eliashberg_tc};

    /// Least-squares line `y = a + b x`: slope and R^2.
    fn line_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    #[test]
    fn pade_reproduces_a_rational_function() {
        let f = |z: Complex| {
            (z + Complex::from(0.3)) / ((z - Complex::new(0.5, 0.2)) * (z + Complex::from(1.1)))
        };
        let points: Vec<Complex> = (0..8)
            .map(|k| Complex::new(0.0, 0.25 * (k as f64 + 1.0)))
            .collect();
        let values: Vec<Complex> = points.iter().map(|z| f(*z)).collect();
        let targets = [
            Complex::new(0.4, 0.3),
            Complex::new(-0.7, 0.1),
            Complex::new(1.3, 0.8),
        ];
        let out = pade_continuation(&points, &values, &targets).unwrap();
        for (t, o) in targets.iter().zip(&out) {
            let exact = f(*t);
            let error = (*o - exact).abs() / exact.abs();
            println!("Pade at {t:?}: {o:?}, exact {exact:?}, relative error {error:e}");
            assert!(error < 1e-10, "{error}");
        }
        assert!(pade_continuation(&[], &[], &targets).is_err());
        assert!(pade_continuation(&points, &values[..3], &targets).is_err());
    }

    /// The nonlinear gap vanishes above the linearized `T_c` and shows the
    /// mean-field law `Delta_0^2 ~ (T_c - T)` below it.
    #[test]
    fn gap_is_consistent_with_the_linearized_critical_temperature() {
        let w = 1e-3;
        let cutoff = 20.0 * w;
        let tc = eliashberg_tc(&[w], &[1.0], 0.1, cutoff, 0.02 * w)
            .unwrap()
            .unwrap();
        let options = GapOptions::default();
        let gap = |t: f64| eliashberg_gap(&[w], &[1.0], 0.1, cutoff, t, &options).unwrap();
        let above = gap(1.05 * tc);
        assert!(above.converged && above.delta.iter().all(|d| *d == 0.0));
        for factor in [0.95, 0.90] {
            let g = gap(factor * tc);
            println!(
                "T = {factor} tc: Delta_0 {:e}, Z_0 {}, {} iterations, residual {:e}",
                g.delta[0], g.z[0], g.iterations, g.residual
            );
            assert!(g.converged && g.delta[0] > 0.0);
        }
        let ts = [0.08, 0.04, 0.02, 0.01];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for t in ts {
            let g = gap(tc * (1.0 - t));
            println!(
                "t = {t}: Delta_0 {:e}, {} iterations, residual {:e}",
                g.delta[0], g.iterations, g.residual
            );
            assert!(g.converged && g.delta[0] > 0.0, "t = {t}");
            x.push(t.ln());
            y.push((g.delta[0] * g.delta[0]).ln());
        }
        let (slope, r2) = line_fit(&x, &y);
        println!(
            "tc {tc:e}; log Delta_0^2 vs log t over t = {}..{}: slope {slope:.6}, R^2 {r2:.8}",
            ts[0], ts[3]
        );
        assert!(
            (slope - 1.0).abs() < 0.05 && r2 > 0.999,
            "slope {slope}, R^2 {r2}"
        );
    }

    /// Weak-coupling BCS ratio `2 Delta_0 / T_c = 3.53`, with `Delta_0` from
    /// the Pade continuation to the real axis.
    #[test]
    fn weak_coupling_gap_ratio_is_the_bcs_value() {
        let w = 1e-3;
        let cutoff = 20.0 * w;
        let tc = eliashberg_tc(&[w], &[0.3], 0.0, cutoff, 0.003 * w)
            .unwrap()
            .unwrap();
        let gap =
            eliashberg_gap(&[w], &[0.3], 0.0, cutoff, tc / 8.0, &GapOptions::default()).unwrap();
        assert!(gap.converged);
        let d0 = gap.delta[0];
        let omega = [0.1 * d0];
        let real = real_axis_gap(&gap, 64, &omega, 1e-8 * w).unwrap()[0];
        let ratio = 2.0 * real.re / tc;
        println!(
            "lambda 0.3: tc {tc:e}, {} Matsubara points, Delta(i w_0) {d0:e}, Re \
                Delta(omega -> 0) {:e} (Im {:e}), 2 Delta_0/tc {ratio:.6}, \
                Delta_0/Delta(i w_0) {:.6}, {} iterations",
            gap.delta.len(),
            real.re,
            real.im,
            real.re / d0,
            gap.iterations
        );
        assert!((3.40..=3.70).contains(&ratio), "{ratio}");
        assert!((real.re / d0 - 1.0).abs() < 0.03);
    }

    #[test]
    fn strong_coupling_ratio_exceeds_the_bcs_value() {
        let w = 1e-3;
        let cutoff = 20.0 * w;
        let tc = eliashberg_tc(&[w], &[2.5], 0.1, cutoff, 0.02 * w)
            .unwrap()
            .unwrap();
        let gap =
            eliashberg_gap(&[w], &[2.5], 0.1, cutoff, tc / 4.0, &GapOptions::default()).unwrap();
        assert!(gap.converged);
        let ratio = 2.0 * gap.delta[0] / tc;
        println!(
            "lambda 2.5: tc {tc:e}, Delta(i w_0) {:e}, 2 Delta(i w_0)/tc {ratio:.6}, {} iterations",
            gap.delta[0], gap.iterations
        );
        assert!(ratio > 3.9, "{ratio}");
    }

    /// Critical temperature of an Einstein spectrum: the strong-coupling
    /// limit `T_c -> 0.1827 w sqrt(lambda)` (`mu* = 0`), and the values of the
    /// independent pure-Python oracle `validation/validate_eliashberg_oracle.py`
    /// (all Matsubara indices of both signs, untruncated `Z`, power iteration),
    /// which is the physical reference. Agreement requires the full
    /// (untruncated) renormalization sum, whose part beyond the cutoff is
    /// `Spectrum::tail`: with `Z` truncated at the cutoff `T_c` is high by
    /// 1.3 % at 10 w (`lambda` = 0.6, `mu* = 0.1`).
    ///
    /// The earlier criterion, agreement with Allen-Dynes within 10 % at
    /// `mu* = 0.1` for `lambda` = 0.6, 1.0, 1.5 (cutoff 40 w), was wrong: the
    /// solver gave a ratio of 1.29 at `lambda` = 0.6 and the oracle confirms the
    /// solver. An imaginary-axis `mu*` defined at a hard Matsubara cutoff is
    /// cutoff dependent and is not the `mu*` of the Allen-Dynes fit, so with
    /// `mu* > 0` the two are not comparable; at `mu* = 0` they agree within
    /// 4 %, the accuracy of the fit.
    #[test]
    fn eliashberg_tc_matches_the_independent_oracle() {
        let w = 1e-3;
        let tc = |lambda: f64, mu: f64, cutoff: f64, t_min: f64| {
            eliashberg_tc(&[w], &[lambda], mu, cutoff * w, t_min * w)
                .unwrap()
                .unwrap()
                / w
        };
        // (a) Strong coupling.
        let strong = tc(100.0, 0.0, 200.0, 0.1);
        let limit = 0.1827 * 100.0_f64.sqrt();
        println!(
            "strong coupling: T_c/w {strong:.6}, 0.1827 sqrt(lambda) {limit:.6}, ratio {:.6}",
            strong / limit
        );
        assert!((strong / limit - 1.0).abs() < 0.03);
        // (b) mu* = 0, cutoff 20 w: oracle and Allen-Dynes.
        for (lambda, oracle) in [(0.6, 0.055992), (1.0, 0.114619), (1.5, 0.169146)] {
            let value = tc(lambda, 0.0, 20.0, 0.02);
            let ad = allen_dynes(lambda, 0.0, 1.0, 1.0, true).unwrap();
            println!(
                "mu* 0, lambda {lambda}, cutoff 20 w: T_c/w {value:.6}, oracle {oracle:.6} (relative {:.5}), Allen-Dynes {ad:.6}, ratio {:.4}",
                value / oracle - 1.0,
                value / ad
            );
            assert!((value / oracle - 1.0).abs() < 0.005, "lambda {lambda}");
            assert!((1.0..=1.06).contains(&(value / ad)), "lambda {lambda}");
        }
        // (c) mu* = 0.1: the oracle at two cutoffs, rising with the cutoff.
        for (lambda, at10, at40) in [(0.6, 0.027838, 0.030209), (1.0, 0.079912, 0.083071)] {
            let (low, high) = (tc(lambda, 0.1, 10.0, 0.02), tc(lambda, 0.1, 40.0, 0.02));
            println!(
                "mu* 0.1, lambda {lambda}: T_c/w {low:.6} at 10 w (oracle {at10:.6}, relative {:.5}), {high:.6} at 40 w (oracle {at40:.6}, relative {:.5})",
                low / at10 - 1.0,
                high / at40 - 1.0
            );
            assert!(
                (low / at10 - 1.0).abs() < 0.005,
                "lambda {lambda}, cutoff 10 w"
            );
            assert!(
                (high / at40 - 1.0).abs() < 0.005,
                "lambda {lambda}, cutoff 40 w"
            );
            assert!(high > low, "lambda {lambda}");
        }
        // (d) No solution at weak coupling.
        assert!(eliashberg_tc(&[w], &[0.05], 0.2, 40.0 * w, 0.02 * w)
            .unwrap()
            .is_none());
    }
}
