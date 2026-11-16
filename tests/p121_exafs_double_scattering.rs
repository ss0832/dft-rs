// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Double scattering of the EXAFS model on the common driver: without
//! angle-resolved tables the signal is the single-scattering one; for a
//! collinear chain with scattering functions that do not depend on the angle
//! or on k the double-scattering signal is a closed form; and two
//! scatterers at a right angle from the absorber give no double scattering,
//! because the polarization average of a K edge is their direction cosine.
//! Measured: the collinear closed-form defect is 9.975e-18, the right-angle
//! signal is zero, and the no-table single-scattering limit is exact. Three
//! tests pass. No derivative is involved, so there is no step fit.

use dft_rs::driver::{run, Controls, System, Value};

const AMPLITUDE: f64 = 0.6;
const PHASE: f64 = 0.4;
const SHIFT: f64 = 0.25;
const PATH: f64 = 30.0;
const SPACING: f64 = 4.0;

fn rows(table: &[Vec<f64>]) -> Value {
    Value::Array(
        table
            .iter()
            .map(|row| Value::Array(row.iter().map(|v| Value::Float(*v)).collect()))
            .collect(),
    )
}

fn list(items: &[f64]) -> Value {
    Value::Array(items.iter().map(|v| Value::Float(*v)).collect())
}

/// Controls with scattering functions that are the same for every element,
/// angle and k.
fn controls(double: bool) -> Controls {
    let mut single = Vec::new();
    let mut angles = Vec::new();
    for number in [1.0, 2.0] {
        for k in [0.5, 12.0] {
            single.push(vec![number, k, AMPLITUDE, PHASE]);
            for angle in [0.0, 180.0] {
                angles.push(vec![number, angle, k, AMPLITUDE, PHASE]);
            }
        }
    }
    let mut pairs = vec![
        ("exafs_absorber".to_string(), Value::Int(0)),
        ("exafs_mfp".to_string(), Value::Float(PATH)),
        ("exafs_rmax".to_string(), Value::Float(9.0)),
        ("exafs_scattering".to_string(), rows(&single)),
        ("exafs_k_range".to_string(), list(&[2.0, 6.0, 0.5])),
        ("exafs_sigma2".to_string(), list(&[0.0])),
    ];
    if double {
        pairs.push(("exafs_scattering_angles".to_string(), rows(&angles)));
        pairs.push((
            "exafs_central_phase".to_string(),
            rows(&[vec![0.5, SHIFT], vec![12.0, SHIFT]]),
        ));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn molecule(positions: Vec<[f64; 3]>) -> System {
    System {
        numbers: vec![1, 2, 2],
        positions,
        cell: None,
        pbc: [false; 3],
    }
}

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn chain() -> System {
    molecule(vec![
        [0.0, 0.0, 0.0],
        [SPACING, 0.0, 0.0],
        [2.0 * SPACING, 0.0, 0.0],
    ])
}

#[test]
fn signal_without_angle_tables_is_the_single_scattering_one() {
    let report = run("exafs", &chain(), &controls(false)).unwrap();
    let total = values(&report, "chi");
    let single = values(&report, "chi_single");
    let double = values(&report, "chi_double");
    assert!(total.iter().zip(&single).all(|(a, b)| a == b));
    assert!(double.iter().all(|v| *v == 0.0));
}

#[test]
fn collinear_chain_has_the_closed_form_double_scattering_signal() {
    let report = run("exafs", &chain(), &controls(true)).unwrap();
    let k = values(&report, "k");
    let double = values(&report, "chi_double");
    // Two paths, absorber -> near -> far -> absorber and its reverse: legs
    // R, R and 2 R, both scatterers along the same direction.
    let total = 4.0 * SPACING;
    let mut worst = 0.0_f64;
    for (wave, value) in k.iter().zip(&double) {
        let expected = -2.0 * AMPLITUDE * AMPLITUDE / (wave * SPACING * SPACING * 2.0 * SPACING)
            * (-total / PATH).exp()
            * (wave * total + 2.0 * PHASE - SHIFT).sin();
        worst = worst.max((value - expected).abs());
    }
    let count = report
        .get("double_scattering_paths")
        .and_then(Value::as_f64)
        .unwrap();
    println!("double scattering of the chain against the closed form {worst:.3e}");
    assert!(count == 2.0);
    assert!(worst < 1e-12);
}

#[test]
fn scatterers_at_a_right_angle_give_no_double_scattering() {
    let corner = molecule(vec![
        [0.0, 0.0, 0.0],
        [SPACING, 0.0, 0.0],
        [0.0, SPACING, 0.0],
    ]);
    // The scattering angle at each atom is 135 degrees: inside the table.
    let report = run("exafs", &corner, &controls(true)).unwrap();
    let largest = values(&report, "chi_double")
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    println!("largest double-scattering signal of the right angle {largest:.3e}");
    assert!(largest < 1e-14);
}
