// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fermi-Dirac (Mermin) response on the common driver: Gamma-point force
//! constants of a metallic hydrogen chain against an independent four-step
//! fit of central differences of the analytic free-energy forces
//! (validation only), and the gapped limit where the finite-temperature
//! response reproduces the insulating one.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];

fn chain(positions: [[f64; 3]; 2], atom: usize, axis: usize, h: f64) -> System {
    let mut positions = positions.to_vec();
    positions[atom][axis] += h;
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
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(6), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(14.0)),
        ("grid_level".to_string(), Value::Int(3)),
        ("grid_shell".to_string(), Value::Int(3)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(300)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn metallic_force_constants_pass_a_four_step_fit_of_the_free_energy_forces() {
    // Nearly uniform chain: the folded band crosses the Fermi level.
    let metal = [[0.0, 0.0, 0.0], [1.75, 0.1, 0.05]];
    let hot = controls(&[("electronic_temperature", Value::Float(3000.0))]);
    let report = run("phonons", &chain(metal, 0, 0, 0.0), &hot).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let fc = flat(&report, "force_constants");
    println!(
        "metallic chain: raw asymmetry {:?}, acoustic residual {:?}, solves {:?}",
        report.get("raw_asymmetry"),
        report.get("acoustic_residual"),
        report.get("response_solves")
    );
    let forces = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let with_forces = vec![
            ("forces", Value::Bool(true)),
            ("electronic_temperature", Value::Float(3000.0)),
        ];
        flat(
            &run(
                "periodic-energy",
                &chain(metal, atom, axis, h),
                &controls(&with_forces),
            )
            .unwrap(),
            "forces",
        )
    };
    // Keep the coarse interval visible: occupation curvature near the Fermi
    // surface requires an independently measured smaller interval here.
    let steps = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125];
    let mut results = Vec::new();
    for (atom, axis) in [(1usize, 0usize), (0, 1)] {
        let column = 3 * atom + axis;
        let fd: Vec<Vec<f64>> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (forces(atom, axis, h), forces(atom, axis, -h));
                println!("metallic force constants raw column ({atom},{axis}) h={h:.16e} plus={plus:?} minus={minus:?}");
                (0..6)
                    .map(|row| -(plus[row] - minus[row]) / (2.0 * h))
                    .collect()
            })
            .collect();
        let errors: Vec<f64> = fd
            .iter()
            .map(|col| {
                (0..6)
                    .map(|row| (col[row] - fc[6 * row + column]).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let coarse = convergence::error_power_fit(&steps[..4], &errors[..4]);
        let coarse_richardson = (0..6)
            .map(|row| ((4.0 * fd[3][row] - fd[2][row]) / 3.0 - fc[6 * row + column]).abs())
            .fold(0.0_f64, f64::max);
        println!("metallic force constants coarse column ({atom},{axis}): p={:.9} R2={:.12} Richardson={coarse_richardson:.16e}", coarse.exponent, coarse.r_squared);
        let fit = convergence::error_power_fit(&steps[3..], &errors[3..]);
        let richardson = (0..6)
            .map(|row| ((4.0 * fd[6][row] - fd[5][row]) / 3.0 - fc[6 * row + column]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "metallic force constants column ({atom},{axis}): steps {steps:?} errors {errors:?} accepted_interval={:?} p={:.9} R2={:.12}; Richardson max difference {richardson:e}",
            &steps[3..],
            fit.exponent, fit.r_squared
        );
        results.push((atom, axis, fit, richardson));
    }
    for (atom, axis, fit, richardson) in results {
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999,
            "({atom},{axis})"
        );
        assert!(richardson < 2e-5);
    }
}

#[test]
fn gapped_finite_temperature_response_reproduces_the_insulator() {
    let gapped = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]];
    let cold = run("phonons", &chain(gapped, 0, 0, 0.0), &controls(&[])).unwrap();
    let warm = run(
        "phonons",
        &chain(gapped, 0, 0, 0.0),
        &controls(&[("electronic_temperature", Value::Float(1000.0))]),
    )
    .unwrap();
    let (a, b) = (
        flat(&cold, "force_constants"),
        flat(&warm, "force_constants"),
    );
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    let largest = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    println!("insulating against Fermi-Dirac (1000 K) force constants: largest difference {worst:e} of {largest:e}");
    assert!(worst < 1e-7 * largest, "{worst:e}");
}
