// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared, independent all-component nuclear derivative acceptance harness.
use super::{components, convergence, stencils};
use dft_rs::driver::{Controls, System, Value};
use std::collections::BTreeMap;

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key)
        .unwrap_or_else(|| panic!("missing nuclear report field `{key}`"))
        .numeric_array()
        .unwrap_or_else(|| panic!("non-numeric nuclear report field `{key}`"))
        .1
}
pub fn fit(
    label: &str,
    base: &System,
    controls: &Controls,
    stages: &[([f64; 4], stencils::Stencil)],
    channels: &[&str],
    evaluate: impl Fn(&str, &System, &Controls) -> Value,
) -> Value {
    let dim = 3 * base.positions.len();
    let gradient = evaluate("gradient", base, controls);
    let g: Vec<_> = channels
        .iter()
        .map(|p| values(&gradient, &format!("{p}gradient")))
        .collect();
    let hessian = evaluate("hessian", base, controls);
    let h: Vec<_> = channels
        .iter()
        .map(|p| values(&hessian, &format!("{p}hessian")))
        .collect();
    assert!(g.iter().all(|g| g.len() == dim));
    assert!(h.iter().all(|h| h.len() == dim * dim));
    for (force, derivative) in values(&gradient, "forces").iter().zip(&g[0]) {
        assert_eq!(*force, -derivative);
    }
    let count = dim + dim * dim;
    let names: Vec<_> = channels
        .iter()
        .flat_map(|prefix| {
            (0..dim)
                .map(move |i| format!("{label} {prefix}gradient {i}"))
                .chain((0..dim).flat_map(move |i| {
                    (0..dim).map(move |j| format!("{label} {prefix}Hessian ({i},{j})"))
                }))
        })
        .collect();
    // Different stencils can share a displacement. Cache the independent
    // relaxed samples across stages while retaining every attempted fit.
    let mut samples = BTreeMap::new();
    let mut errors = |steps: &[f64; 4], stencil: stencils::Stencil, columns: Option<&[usize]>| {
        let mut errors = vec![[0.0; 4]; channels.len() * count];
        for (k, &step) in steps.iter().enumerate() {
            for column in 0..dim {
                if columns.is_some_and(|active| !active.contains(&column)) {
                    continue;
                }
                let mut sample = |offset: f64| {
                    samples
                        .entry((column, offset.to_bits()))
                        .or_insert_with(|| {
                            let mut moved = base.clone();
                            moved.positions[column / 3][column % 3] += offset;
                            let report = evaluate("gradient", &moved, controls);
                            (
                                channels
                                    .iter()
                                    .map(|p| {
                                        report
                                            .get(
                                                if p.is_empty() {
                                                    "energy".to_string()
                                                } else {
                                                    format!("{p}nuclear_energy")
                                                }
                                                .as_str(),
                                            )
                                            .unwrap()
                                            .as_f64()
                                            .unwrap()
                                    })
                                    .collect::<Vec<_>>(),
                                channels
                                    .iter()
                                    .map(|p| values(&report, &format!("{p}gradient")))
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .clone()
                };
                let (ep, gp) = sample(step);
                let (em, gm) = sample(-step);
                let (ep2, gp2, em2, gm2) = if stencil == stencils::Stencil::Fourth {
                    let (e2p, g2p) = sample(2.0 * step);
                    let (e2m, g2m) = sample(-2.0 * step);
                    (e2p, g2p, e2m, g2m)
                } else {
                    (
                        vec![0.0; channels.len()],
                        vec![vec![0.0; dim]; channels.len()],
                        vec![0.0; channels.len()],
                        vec![vec![0.0; dim]; channels.len()],
                    )
                };
                for c in 0..channels.len() {
                    errors[c * count + column][k] =
                        (stencil.first(ep[c], em[c], ep2[c], em2[c], step) - g[c][column]).abs();
                    for row in 0..dim {
                        errors[c * count + dim + row * dim + column][k] =
                            (stencil.first(gp[c][row], gm[c][row], gp2[c][row], gm2[c][row], step)
                                - h[c][row * dim + column])
                                .abs();
                    }
                }
            }
        }
        errors
    };
    let steps = [0.04, 0.03, 0.02, 0.015];
    let initial = errors(&steps, stencils::Stencil::Second, None);
    for (name, e) in names.iter().zip(&initial) {
        if e.iter().all(|v| *v > 0.0 && v.is_finite()) {
            println!(
                "{name} log fit RMSE={:.3e}",
                convergence::error_power_fit(&steps, e).log_rmse
            );
        }
    }
    let mut failed = Vec::new();
    components::fit_components(&names, &[steps], |_| initial.clone(), &mut failed);
    if !failed.is_empty() {
        println!("Initial failed components retained: {failed:?}");
        let indices = failed
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|n| n == name)
                    .expect("component name")
            })
            .collect::<Vec<_>>();
        let selected = indices
            .iter()
            .map(|i| names[*i].clone())
            .collect::<Vec<_>>();
        let mut columns = indices
            .iter()
            .map(|i| {
                let local = *i % count;
                if local < dim {
                    local
                } else {
                    (local - dim) % dim
                }
            })
            .collect::<Vec<_>>();
        columns.sort_unstable();
        columns.dedup();
        failed.clear();
        // Re-evaluate only columns needed by failed components. Previously
        // accepted entries retain their independent initial four-step fits.
        stencils::fit_components_staged(
            &selected,
            stages,
            |h, stencil| {
                let measured = errors(h, stencil, Some(&columns));
                indices.iter().map(|i| measured[*i]).collect()
            },
            &mut failed,
        );
    }
    components::assert_no_failures(label, &failed);
    hessian
}
