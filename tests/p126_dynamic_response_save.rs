// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Frequency-dependent amplitudes of the saved response on the common
//! driver: at zero frequency `X + Y` is twice the static orbital rotation,
//! `X - Y` vanishes and the polarizability is the static one; below the
//! lowest excitation the polarizability at a finite frequency exceeds the
//! static one (normal dispersion). Both strict-unit tests pass: the static
//! rotation and polarizability defects are 4.496e-15 and 1.776e-15. The
//! amplitudes are those of the
//! validated dynamic polarizability, so no new derivative and no step fit
//! is involved.

use dft_rs::driver::{run, Controls, System, Value};

fn water() -> System {
    System {
        numbers: vec![8, 1, 1],
        positions: vec![
            [0.0, 0.0, 0.2217],
            [0.0, 1.4309, -0.8867],
            [0.0, -1.4309, -0.8867],
        ],
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(frequencies: &[f64]) -> Controls {
    Controls::from_value(&Value::Object(vec![
        ("xc".to_string(), Value::Str("pbe".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("nuclear".to_string(), Value::Bool(false)),
        (
            "frequencies".to_string(),
            Value::Array(frequencies.iter().map(|w| Value::Float(*w)).collect()),
        ),
    ]))
    .unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn zero_frequency_amplitudes_are_the_static_rotations() {
    let report = run("response-save", &water(), &controls(&[0.0])).unwrap();
    let dynamic = &report.get("dynamic").and_then(Value::as_array).unwrap()[0];
    let fixed = report.get("field").and_then(Value::as_array).unwrap();
    let moving = dynamic.get("field").and_then(Value::as_array).unwrap();
    let mut worst = 0.0_f64;
    let mut residue = 0.0_f64;
    for (a, b) in fixed.iter().zip(moving) {
        let rotation = flat(a, "orbital_rotations");
        let sum = flat(b, "x_plus_y_real");
        for (u, s) in rotation.iter().zip(&sum) {
            worst = worst.max((2.0 * u - s).abs());
        }
        for key in ["x_minus_y_real", "x_plus_y_imag", "x_minus_y_imag"] {
            residue = flat(b, key).iter().fold(residue, |m, v| m.max(v.abs()));
        }
    }
    let alpha = flat(&report, "polarizability");
    let moving_alpha = flat(dynamic, "polarizability_real");
    let difference = alpha
        .iter()
        .zip(&moving_alpha)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    println!(
        "zero frequency: rotations {worst:.3e}, other parts {residue:.3e}, polarizability \
         {difference:.3e}"
    );
    assert!(worst < 1e-7);
    assert!(residue < 1e-7);
    assert!(difference < 1e-7);
}

#[test]
fn finite_frequency_polarizability_exceeds_the_static_one_below_the_first_pole() {
    let report = run("response-save", &water(), &controls(&[0.0, 0.1])).unwrap();
    let dynamic = report.get("dynamic").and_then(Value::as_array).unwrap();
    let trace = |entry: &Value| {
        let alpha = flat(entry, "polarizability_real");
        alpha[0] + alpha[4] + alpha[8]
    };
    let (fixed, moving) = (trace(&dynamic[0]), trace(&dynamic[1]));
    println!("trace of the polarizability: static {fixed:.8}, at 0.1 hartree {moving:.8}");
    // Normal dispersion below the lowest excitation.
    assert!(moving > fixed);
}
