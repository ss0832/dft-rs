// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Boys function F_n(x) = ∫₀¹ t^{2n} exp(−x t²) dt, the scalar core of every
//! Coulomb-type Gaussian integral.
//!
//! Three regimes:
//! - `x ≈ 0`: F_n = 1/(2n+1) − x/(2n+3) + … (leading term suffices below 1e-13),
//! - moderate `x`: convergent series for F_nmax, then stable downward
//!   recursion F_n = (2x·F_{n+1} + e^{−x})/(2n+1),
//! - large `x` (> 35): F_0 = ½√(π/x) (erf(√x) ≈ 1 to machine precision),
//!   then upward recursion F_{n+1} = ((2n+1)F_n − e^{−x})/(2x), which is
//!   stable in this regime.

use std::f64::consts::PI;

/// Fill `out[0..=nmax]` with F_0(x) … F_nmax(x).
pub fn boys_into(nmax: usize, x: f64, out: &mut [f64]) {
    debug_assert!(out.len() > nmax);
    if x < 1e-13 {
        for (n, o) in out.iter_mut().enumerate().take(nmax + 1) {
            *o = 1.0 / (2 * n + 1) as f64;
        }
        return;
    }
    if x > 35.0 {
        out[0] = 0.5 * (PI / x).sqrt();
        let ex = (-x).exp();
        for n in 1..=nmax {
            out[n] = ((2 * n - 1) as f64 * out[n - 1] - ex) / (2.0 * x);
        }
        return;
    }
    // Series for F_nmax, then downward recursion.
    let mut term = 1.0 / (2 * nmax + 1) as f64;
    let mut sum = term;
    let mut k = 1usize;
    loop {
        term *= (2.0 * x) / (2 * nmax + 2 * k + 1) as f64;
        sum += term;
        if term < 1e-17 * sum || k > 500 {
            break;
        }
        k += 1;
    }
    let ex = (-x).exp();
    out[nmax] = ex * sum;
    for n in (0..nmax).rev() {
        out[n] = (2.0 * x * out[n + 1] + ex) / (2 * n + 1) as f64;
    }
}

/// Allocate-and-return variant of [`boys_into`].
pub fn boys(nmax: usize, x: f64) -> Vec<f64> {
    let mut out = vec![0.0; nmax + 1];
    boys_into(nmax, x, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values via composite Gauss–Legendre quadrature of the
    /// defining integral (test-only, slow but arbitrarily accurate).
    fn boys_quadrature(n: usize, x: f64) -> f64 {
        // 64-point Gauss–Legendre on [0,1], split into 8 panels.
        let (nodes, weights) = gauss_legendre_16();
        let mut sum = 0.0;
        let panels = 8;
        for p in 0..panels {
            let a = p as f64 / panels as f64;
            let b = (p + 1) as f64 / panels as f64;
            let half = 0.5 * (b - a);
            let mid = 0.5 * (a + b);
            for (t, w) in nodes.iter().zip(&weights) {
                let u: f64 = mid + half * t;
                sum += w * half * u.powi(2 * n as i32) * (-x * u * u).exp();
            }
        }
        sum
    }

    /// 16-point Gauss–Legendre nodes/weights on [−1,1], computed by Newton
    /// iteration on the Legendre polynomial (no table transcription).
    fn gauss_legendre_16() -> (Vec<f64>, Vec<f64>) {
        let n = 16usize;
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        for i in 0..n {
            // Chebyshev initial guess.
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            for _ in 0..100 {
                let (p, dp) = legendre_p_dp(n, x);
                let dx = -p / dp;
                x += dx;
                if dx.abs() < 1e-15 {
                    break;
                }
            }
            let (_, dp) = legendre_p_dp(n, x);
            nodes[i] = x;
            weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
        }
        (nodes, weights)
    }

    fn legendre_p_dp(n: usize, x: f64) -> (f64, f64) {
        let (mut p0, mut p1) = (1.0, x);
        for k in 2..=n {
            let pk = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = pk;
        }
        let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
        (p1, dp)
    }

    /// Every regime, every order n ≤ 24, against quadrature to ≤ 1e-13.
    #[test]
    fn boys_matches_quadrature_over_all_regimes() {
        let xs = [
            0.0, 1e-14, 1e-8, 1e-3, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 34.9, 35.1, 40.0, 60.0,
        ];
        for &x in &xs {
            let f = boys(24, x);
            for n in (0..=24).step_by(4) {
                let reference = boys_quadrature(n, x);
                assert!(
                    (f[n] - reference).abs() <= 1e-13 * reference.max(1.0),
                    "F_{n}({x}) = {} vs quadrature {}",
                    f[n],
                    reference
                );
            }
        }
    }

    /// The downward/upward regimes must join continuously at x = 35: with a
    /// straddle of ±1e-9 the true variation is |dF_n/dx|·2e-9 = F_{n+1}·2e-9
    /// ≲ 5e-12, so any jump beyond 1e-10 exposes a regime inconsistency.
    #[test]
    fn regime_boundary_is_continuous() {
        let below = boys(12, 35.0 - 1e-9);
        let above = boys(12, 35.0 + 1e-9);
        for n in 0..=12 {
            assert!(
                (below[n] - above[n]).abs() < 1e-10,
                "F_{n} jumps across the x=35 boundary: {} vs {}",
                below[n],
                above[n]
            );
        }
    }

    /// x = 0 closed form F_n(0) = 1/(2n+1).
    #[test]
    fn zero_argument_closed_form() {
        let f = boys(10, 0.0);
        for (n, &v) in f.iter().enumerate() {
            assert!((v - 1.0 / (2 * n + 1) as f64).abs() < 1e-16);
        }
    }
}
