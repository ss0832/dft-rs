// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Direct finite-field initialization: identical stationary energy, and an
//! independent four-step check of its analytic gradient on the common driver.

#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::{run, Controls, System, Value};

fn system(displacement: f64) -> System {
    System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7 + displacement]],
        cell: None,
        pbc: [false; 3],
    }
}
fn controls(guess: &str) -> Controls {
    Controls::from_value(
        &Value::object()
            .with("xc", "pbe")
            .with("basis", "sto-3g")
            .with("conv_tol", 1e-12)
            .with("bfield", [0.03, 0.01, 0.02])
            .with("bfield_initial_guess", guess),
    )
    .unwrap()
}
fn number(report: &Value, key: &str) -> f64 {
    report.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn london_core_seed_and_converged_zero_field_seed_reach_the_same_state() {
    let core = run("bfield", &system(0.0), &controls("core")).unwrap();
    let warm = run("bfield", &system(0.0), &controls("zero-field")).unwrap();
    let difference = (number(&core, "field_energy") - number(&warm, "field_energy")).abs();
    println!("finite-field core/warm energy difference={difference:.12e}; core residual={:.12e}; warm residual={:.12e}",
        number(&core,"field_residual"), number(&warm,"field_residual"));
    assert!(difference < 1e-11);
    assert!(matches!(core.get("energy"), Some(Value::Null)));
    assert!(core.get("zero_field_reference_converged") == Some(&Value::Null));
    assert!(matches!(warm.get("energy"), Some(Value::Float(_))));
    assert!(number(&core, "field_residual") < 1e-6);
}

#[test]
fn directly_initialized_field_gradient_has_an_independent_four_step_fit() {
    let report = run("bfield-gradient", &system(0.0), &controls("core")).unwrap();
    let gradient = report.get("gradient").unwrap().numeric_array().unwrap().1[5];
    let energy = |h| {
        number(
            &run("bfield", &system(h), &controls("core")).unwrap(),
            "field_energy",
        )
    };
    let steps = [0.04, 0.02, 0.01, 0.005];
    let estimates: Vec<_> = steps
        .iter()
        .map(|h| (energy(*h) - energy(-*h)) / (2.0 * h))
        .collect();
    let errors: Vec<_> = estimates.iter().map(|v| (v - gradient).abs()).collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - gradient).abs();
    println!("direct-core finite-field gradient={gradient:.12e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e}", fit.exponent, fit.r_squared);
    assert!((fit.exponent - 2.0).abs() < 0.15 && fit.r_squared > 0.999);
    assert!(richardson < 1e-7);
}

#[test]
fn axial_diamagnetic_polynomial_is_not_mistaken_for_a_truncation_error() {
    let atom = System {
        numbers: vec![1],
        positions: vec![[0.0; 3]],
        cell: None,
        pbc: [false; 3],
    };
    let options = |spin| {
        Controls::from_value(
            &Value::object()
                .with("xc", "hf")
                .with("multiplicity", 2i64)
                .with("unrestricted", true)
                .with("bfield", [0.0, 0.0, 0.01])
                .with("spin_zeeman", spin)
                .with("bfield_initial_guess", "core")
                .with("conv_tol", 1e-12),
        )
        .unwrap()
    };
    let without = run("bfield", &atom, &options(false)).unwrap();
    let with = run("bfield", &atom, &options(true)).unwrap();
    assert_eq!(number(&with, "series_order"), 2.0);
    assert_eq!(number(&with, "series_tail"), 0.0);
    let expected = dft_rs::constants::ELECTRON_G_FACTOR * 0.01 / 4.0;
    let defect =
        (number(&with, "field_energy") - number(&without, "field_energy") - expected).abs();
    println!("terminating axial B^2 polynomial: omitted tail=0; Zeeman defect={defect:.12e}");
    assert!(defect < 1e-13);
}
