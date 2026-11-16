// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Third and fourth derivatives of the energy of a periodic cell contracted
//! with Gamma-point displacement patterns, on the common driver: independent
//! four-step fits of central differences of the analytic Gamma-point force
//! constants (validation only), Richardson checks, the acoustic sum rule and
//! the energy of the reference of the integral series. Measured so far
//! (docs/response.md, "Periodic path response"): the Gamma-point
//! directional cubic derivative over [0.08, 0.04, 0.02, 0.01] Bohr, with
//! p = 1.998664, R^2 = 0.99999988 and Richardson defect 9.360e-9 (a fit
//! of the largest entry error). Every one of the 36 contracted entries is
//! now fitted on its own; an entry that fails on the first interval is
//! recomputed on a finer and then a coarser one.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use components::{assert_no_failures, fit_components};

use dft_rs::driver::{run, Controls, System, Value};

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]];
/// Displacement pattern without a rigid translation of the cell.
const PROBE: [f64; 6] = [0.3, -0.2, 0.1, -0.3, 0.2, -0.1];

/// The cell with its atoms displaced by `scale` times the pattern.
fn chain(scale: f64) -> System {
    let mut positions = POSITIONS.to_vec();
    for (atom, position) in positions.iter_mut().enumerate() {
        for (axis, value) in position.iter_mut().enumerate() {
            *value += scale * PROBE[3 * atom + axis];
        }
    }
    System {
        numbers: vec![1, 1],
        positions,
        cell: Some(CELL),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions (see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(20.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(2)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(200)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn probe() -> Value {
    Value::Array(PROBE.iter().map(|x| Value::Float(*x)).collect())
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn matrix(report: &Value, key: &str) -> Vec<f64> {
    let values = report.get(key).unwrap().numeric_array().unwrap().1;
    assert_eq!(values.len(), 36);
    values
}

/// Analytic Gamma-point force constants of the displaced cell.
fn hessian(scale: f64) -> Vec<f64> {
    matrix(
        &run("phonons", &chain(scale), &controls(&[])).unwrap(),
        "force_constants",
    )
}

fn largest(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Largest sum of a row over the atoms: a rigid translation of the cell
/// changes no derivative.
fn acoustic(phi: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for row in 0..6 {
        for axis in 0..3 {
            let sum: f64 = (0..2).map(|atom| phi[6 * row + 3 * atom + axis]).sum();
            worst = worst.max(sum.abs());
        }
    }
    worst
}

#[test]
fn periodic_cubic_force_constants_pass_a_four_step_fit_of_the_analytic_force_constants() {
    let report = run(
        "periodic-cubic-fc-along",
        &chain(0.0),
        &controls(&[("probe", probe())]),
    )
    .unwrap();
    let phi = matrix(&report, "contracted_force_constants");
    let difference = number(&report, "series_reference_energy_difference");
    let residual = number(&report, "series_reference_residual");
    const INTERVALS: [[f64; 4]; 3] = [
        [0.08, 0.04, 0.02, 0.01],
        [0.04, 0.02, 0.01, 0.005],
        [0.16, 0.08, 0.04, 0.02],
    ];
    let names: Vec<String> = (0..36)
        .map(|i| format!("cubic contracted force constant [{}][{}]", i / 6, i % 6))
        .collect();
    let mut failures = Vec::new();
    let mut first_differences = Vec::new();
    fit_components(
        &names,
        &INTERVALS,
        |steps| {
            let fd: Vec<Vec<f64>> = steps
                .iter()
                .map(|&h| {
                    let (plus, minus) = (hessian(h), hessian(-h));
                    (0..36).map(|i| (plus[i] - minus[i]) / (2.0 * h)).collect()
                })
                .collect();
            if first_differences.is_empty() {
                first_differences = fd.clone();
            }
            (0..36)
                .map(|i| std::array::from_fn(|s| (fd[s][i] - phi[i]).abs()))
                .collect()
        },
        &mut failures,
    );
    let fd = &first_differences;
    let richardson: Vec<f64> = (0..36).map(|i| (4.0 * fd[3][i] - fd[2][i]) / 3.0).collect();
    let worst = largest(&richardson, &phi);
    let sum_rule = acoustic(&phi);
    println!(
        "periodic cubic force constants along the pattern: Richardson max difference {worst:e}; \
         acoustic sum {sum_rule:e}; series reference energy difference {difference:e}, \
         residual {residual:e}"
    );
    assert_no_failures("periodic cubic contracted force constants", &failures);
    assert!(worst < 1e-4, "Richardson: {worst:e}");
    assert!(sum_rule < 1e-5, "acoustic sum rule: {sum_rule:e}");
    assert!(difference.abs() < 1e-6, "series reference: {difference:e}");
}

#[test]
fn periodic_quartic_force_constants_pass_a_four_step_fit_of_the_analytic_force_constants() {
    let report = run(
        "periodic-quartic-fc-along",
        &chain(0.0),
        &controls(&[("probe", probe())]),
    )
    .unwrap();
    let phi = matrix(&report, "contracted_force_constants");
    let centre = hessian(0.0);
    const INTERVALS: [[f64; 4]; 3] = [
        [0.2, 0.14, 0.1, 0.07],
        [0.1, 0.07, 0.05, 0.035],
        [0.28, 0.2, 0.14, 0.1],
    ];
    let names: Vec<String> = (0..36)
        .map(|i| format!("quartic contracted force constant [{}][{}]", i / 6, i % 6))
        .collect();
    let mut failures = Vec::new();
    fit_components(
        &names,
        &INTERVALS,
        |steps| {
            let fd: Vec<Vec<f64>> = steps
                .iter()
                .map(|&h| {
                    let (plus, minus) = (hessian(h), hessian(-h));
                    (0..36)
                        .map(|i| (plus[i] - 2.0 * centre[i] + minus[i]) / (h * h))
                        .collect()
                })
                .collect();
            (0..36)
                .map(|i| std::array::from_fn(|s| (fd[s][i] - phi[i]).abs()))
                .collect()
        },
        &mut failures,
    );
    let sum_rule = acoustic(&phi);
    println!("periodic quartic force constants along the pattern: acoustic sum {sum_rule:e}");
    assert_no_failures("periodic quartic contracted force constants", &failures);
    assert!(sum_rule < 1e-4, "acoustic sum rule: {sum_rule:e}");
}
