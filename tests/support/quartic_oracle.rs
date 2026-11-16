// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent lower-response validation when third differences of gradients
//! lose their small truncation error to cancellation. Original failed
//! gradient intervals remain in the acceptance output.
use super::stencils::{self, Stencil};
use std::collections::BTreeMap;

pub fn fit_from_second(
    label: &str,
    exact: f64,
    mut second: impl FnMut(f64) -> f64,
    failures: &mut Vec<String>,
) {
    let mut cache = BTreeMap::new();
    let mut at = |t: f64| *cache.entry(t.to_bits()).or_insert_with(|| second(t));
    let centre = at(0.0);
    let names = [format!("{label} independent displaced-Hessian oracle")];
    stencils::fit_components_staged(
        &names,
        &[
            ([0.04, 0.03, 0.02, 0.015], Stencil::Second),
            ([0.01, 0.0075, 0.005, 0.00375], Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], Stencil::Fourth),
        ],
        |steps, stencil| {
            vec![std::array::from_fn(|k| {
                let h = steps[k];
                let near = (at(h) - 2.0 * centre + at(-h)) / h.powi(2);
                let measured = if stencil == Stencil::Fourth {
                    let far = (at(2.0 * h) - 2.0 * centre + at(-2.0 * h)) / (4.0 * h.powi(2));
                    (4.0 * near - far) / 3.0
                } else {
                    near
                };
                (measured - exact).abs()
            })]
        },
        failures,
    );
}
