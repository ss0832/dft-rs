// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

/// Independent log-log least squares for error(h) = prefactor * h^exponent.
/// Requires measured positive errors; zero/roundoff-saturated errors cannot
/// establish an asymptotic order and must not be replaced by numerical floors.
pub struct ErrorPowerFit {
    pub exponent: f64,
    pub prefactor: f64,
    pub r_squared: f64,
    pub log_rmse: f64,
}

pub fn error_power_fit(steps: &[f64], errors: &[f64]) -> ErrorPowerFit {
    assert_eq!(steps.len(), errors.len());
    assert!(steps.len() >= 4);
    assert!(steps
        .iter()
        .chain(errors)
        .all(|x| x.is_finite() && *x > 0.0));
    let x: Vec<_> = steps.iter().map(|h| h.ln()).collect();
    let y: Vec<_> = errors.iter().map(|e| e.ln()).collect();
    let n = steps.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let xx = x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>();
    let xy = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - mean_x) * (b - mean_y))
        .sum::<f64>();
    assert!(xx > 0.0);
    let exponent = xy / xx;
    let intercept = mean_y - exponent * mean_x;
    let sse = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (b - intercept - exponent * a).powi(2))
        .sum::<f64>();
    let sst = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>();
    assert!(sst > 0.0);
    let fit = ErrorPowerFit {
        exponent,
        prefactor: intercept.exp(),
        r_squared: 1.0 - sse / sst,
        log_rmse: (sse / n).sqrt(),
    };
    assert!(fit.prefactor.is_finite() && fit.prefactor > 0.0);
    assert!(fit.log_rmse.is_finite() && fit.log_rmse >= 0.0);
    fit
}
