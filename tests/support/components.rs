// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Four-step fits of every component of a derivative tensor. Include
//! together with `support/convergence.rs` as `mod convergence`.

/// Four-step fit of one component; passes when p is within 0.2 of the
/// expected order and R2 exceeds 0.999. A zero or non-finite error cannot
/// establish an order and fails the fit (no floor is applied).
pub fn component_fit_of_order(
    label: &str,
    steps: &[f64; 4],
    errors: &[f64; 4],
    order: f64,
) -> bool {
    if !errors.iter().all(|e| e.is_finite() && *e > 0.0) {
        println!("{label}: steps {steps:?} errors {errors:?} FAILED (zero or non-finite error)");
        return false;
    }
    if errors.windows(2).all(|pair| pair[0] == pair[1]) {
        println!(
            "{label}: steps {steps:?} errors {errors:?} FAILED (constant error has no convergence fit)"
        );
        return false;
    }
    let fit = super::convergence::error_power_fit(steps, errors);
    let pass = (fit.exponent - order).abs() < 0.2 && fit.r_squared > 0.999;
    println!(
        "{label}: steps {steps:?} errors {errors:?} p={:.6} (expected {order}) R2={:.8} C={:.3e}{}",
        fit.exponent,
        fit.r_squared,
        fit.prefactor,
        if pass { "" } else { " FAILED" }
    );
    pass
}

/// Fits of a group of components that share their finite-difference
/// evaluations, by second-order central differences. `errors_at(steps)`
/// returns the four errors of every component. A component that fails on
/// `intervals[0]` is recomputed on the following intervals in order (a
/// finer one against higher-order terms, then a coarser one against noise);
/// every fit is printed and a failed one is marked FAILED. A component
/// failing on every interval is a failure.
pub fn fit_components<F>(
    names: &[String],
    intervals: &[[f64; 4]],
    mut errors_at: F,
    failures: &mut Vec<String>,
) where
    F: FnMut(&[f64; 4]) -> Vec<[f64; 4]>,
{
    let mut open: Vec<usize> = (0..names.len()).collect();
    for (n, steps) in intervals.iter().enumerate() {
        let errors = errors_at(steps);
        assert_eq!(errors.len(), names.len());
        open.retain(|&k| {
            let label = if n == 0 {
                names[k].clone()
            } else {
                format!("{} (interval {})", names[k], n + 1)
            };
            !component_fit_of_order(&label, steps, &errors[k], 2.0)
        });
        if open.is_empty() {
            return;
        }
    }
    failures.extend(open.into_iter().map(|k| names[k].clone()));
}

pub fn assert_no_failures(what: &str, failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{what}: {} components failed on every interval:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
