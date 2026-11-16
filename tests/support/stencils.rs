// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Staged component fits with a fourth-order stencil as the last resort.
//! Include together with `support/convergence.rs` and
//! `support/components.rs`.

/// Central-difference stencil of a first derivative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stencil {
    /// `(f(h) - f(-h)) / 2h`, error of order h^2.
    Second,
    /// `(-f(2h) + 8 f(h) - 8 f(-h) + f(-2h)) / 12h`, error of order h^4.
    Fourth,
}

impl Stencil {
    pub fn order(self) -> f64 {
        match self {
            Self::Second => 2.0,
            Self::Fourth => 4.0,
        }
    }

    /// The derivative estimate from the values at +-h and (for the fourth
    /// order stencil) +-2h.
    pub fn first(self, plus: f64, minus: f64, plus2: f64, minus2: f64, h: f64) -> f64 {
        match self {
            Self::Second => (plus - minus) / (2.0 * h),
            Self::Fourth => (-plus2 + 8.0 * plus - 8.0 * minus + minus2) / (12.0 * h),
        }
    }
}

/// As `components::fit_components`, with a stencil per stage: a component
/// whose second-order central differences cannot isolate their h^2 term (an
/// anomalously small h^2 coefficient cancelling against the h^4 term, with
/// noise below) is fitted with the fourth-order stencil against order 4.
pub fn fit_components_staged<F>(
    names: &[String],
    stages: &[([f64; 4], Stencil)],
    mut errors_at: F,
    failures: &mut Vec<String>,
) where
    F: FnMut(&[f64; 4], Stencil) -> Vec<[f64; 4]>,
{
    fit_components_staged_with_accuracy(
        names,
        stages,
        |steps, stencil| {
            errors_at(steps, stencil)
                .into_iter()
                .map(|errors| FitErrors {
                    errors,
                    richardson: None,
                })
                .collect()
        },
        failures,
    );
}

pub struct FitErrors {
    pub errors: [f64; 4],
    pub richardson: Option<(f64, f64)>,
}

impl FitErrors {
    fn accepts_accuracy(&self, label: &str, steps: &[f64; 4]) -> bool {
        self.richardson.is_none_or(|(error, limit)| {
            let passed = error.is_finite() && error < limit;
            let status = if passed { "" } else { " FAILED" };
            println!(
                "{label}: steps {steps:?} Richardson error={error:.9e} limit={limit:.9e}{status}"
            );
            passed
        })
    }
}

/// Require the convergence fit and optional Richardson accuracy on the
/// same interval and keep all failures before trying the next interval.
pub fn fit_components_staged_with_accuracy<F>(
    names: &[String],
    stages: &[([f64; 4], Stencil)],
    mut errors_at: F,
    failures: &mut Vec<String>,
) where
    F: FnMut(&[f64; 4], Stencil) -> Vec<FitErrors>,
{
    let mut open: Vec<usize> = (0..names.len()).collect();
    // Reuse the ordinary central-difference component gate for the common
    // first stage, retaining precisely its unresolved components below.
    let start = if let Some((steps, Stencil::Second)) = stages.first() {
        let mut unresolved = Vec::new();
        let fits = errors_at(steps, Stencil::Second);
        super::components::fit_components(
            names,
            std::slice::from_ref(steps),
            |_| fits.iter().map(|fit| fit.errors).collect(),
            &mut unresolved,
        );
        open.retain(|&k| {
            let accuracy = fits[k].accepts_accuracy(&names[k], steps);
            !accuracy || unresolved.contains(&names[k])
        });
        if open.is_empty() {
            return;
        }
        1
    } else {
        0
    };
    for (n, (steps, stencil)) in stages.iter().enumerate().skip(start) {
        let fits = errors_at(steps, *stencil);
        assert_eq!(fits.len(), names.len());
        open.retain(|&k| {
            let label = match (n, stencil) {
                (0, _) => names[k].clone(),
                (_, Stencil::Second) => format!("{} (interval {})", names[k], n + 1),
                (_, Stencil::Fourth) => {
                    format!("{} (interval {}, fourth-order stencil)", names[k], n + 1)
                }
            };
            let fit = &fits[k];
            let rate = super::components::component_fit_of_order(
                &label,
                steps,
                &fit.errors,
                stencil.order(),
            );
            let accuracy = fit.accepts_accuracy(&label, steps);
            !(rate && accuracy)
        });
        if open.is_empty() {
            return;
        }
    }
    failures.extend(open.into_iter().map(|k| names[k].clone()));
}
