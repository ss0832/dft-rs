// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Actual common-driver contracts for an explicit zero-correction primitive.
//! Nonzero derivative accuracy is covered by the independent displaced-pose
//! acceptance suite; this identity fixture tests routing, E0 and unit blocks.

use dft_rs::driver::{self, Controls, System, Value};
use dft_rs::DftError;

fn system() -> System {
    System::molecule(vec![1, 1], vec![[-0.7, 0.23, -0.15], [1.1, -0.31, 0.42]])
}

fn settings(order: usize, threads: usize) -> Value {
    Value::object()
        .with("fragment_mode", "rigid-evaluate")
        .with(
            "fragment_model_json",
            include_str!("data/rigid_pose_identity.json"),
        )
        .with("derivative_order", order)
        .with("cpu_threads", threads)
        .with("memory_mb", 384)
}

fn run(settings: Value) -> dft_rs::Result<Value> {
    driver::run(
        "continuous-species",
        &system(),
        &Controls::from_value(&settings)?,
    )
}

fn number(value: &Value, key: &str) -> f64 {
    value.get(key).unwrap().as_f64().unwrap()
}

fn flatten(value: &Value) -> Vec<f64> {
    match value {
        Value::Array(rows) => rows.iter().flat_map(flatten).collect(),
        _ => vec![value.as_f64().unwrap()],
    }
}

#[test]
fn unsupported_references_and_orders_do_not_start_a_fallback_scf() {
    for reference in ["uks", "uhf", "roks", "rohf", "gks", "ghf"] {
        let mut input = settings(0, 1);
        input.set("reference", reference);
        assert!(
            matches!(run(input), Err(DftError::Unsupported { .. })),
            "{reference}"
        );
    }
    assert!(matches!(
        run(settings(3, 1)),
        Err(DftError::Unsupported { .. })
    ));
    let mut input = settings(0, 1);
    input.set("reference", "unknown");
    assert!(matches!(run(input), Err(DftError::InvalidInput(_))));
}

#[test]
fn invalid_integer_and_unknown_controls_are_not_ignored() {
    for value in [Value::Bool(true), Value::Float(1.5), Value::Int(-1)] {
        let mut input = settings(0, 1);
        input.set("derivative_order", value);
        assert!(matches!(run(input), Err(DftError::InvalidInput(_))));
    }
    let mut input = settings(0, 1);
    input.set("unimplemented_pose_control", true);
    assert!(matches!(run(input), Err(DftError::InvalidInput(_))));
}

#[test]
fn e0_is_included_once_and_order_zero_does_not_advertise_derivatives() {
    let actual = run(settings(0, 1)).unwrap();
    let ordinary = Value::object()
        .with("basis", "sto-3g")
        .with("xc", "pbe")
        .with("grid_level", 1)
        .with("grid_weight_cutoff", 0.)
        .with("guess", "core")
        .with("max_iter", 300)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-11)
        .with("cpu_threads", 1);
    let reference = driver::run(
        "energy",
        &system(),
        &Controls::from_value(&ordinary).unwrap(),
    )
    .unwrap();
    assert!((number(&actual, "energy") - number(&reference, "energy") - 0.2).abs() < 1e-10);
    assert!(
        (number(&actual, "electronic_energy") + number(&actual, "classical_energy")
            - number(&actual, "energy"))
        .abs()
            < 1e-12
    );
    assert!((number(&actual, "internal_energy_reference") - 0.2).abs() < 1e-15);
    for name in [
        "forces",
        "center_gradient",
        "center_hessian",
        "angle_hessian",
    ] {
        assert!(actual.get(name).is_none(), "{name}");
    }
}

#[test]
fn analytic_hessian_uses_independent_scoped_pools_and_separate_angle_units() {
    let reports: Vec<_> = [1, 2]
        .into_iter()
        .map(|threads| std::thread::spawn(move || run(settings(2, threads)).unwrap()))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|job| job.join().unwrap())
        .collect();
    for report in &reports {
        assert_eq!(report.get("response_paths").unwrap().as_i64(), Some(78));
        assert_eq!(report.get("converged").unwrap().as_bool(), Some(true));
        let units = report.get("units").unwrap();
        for (key, label) in [
            ("center_gradient", "hartree/bohr"),
            ("euler_gradient", "hartree/radian"),
            ("center_hessian", "hartree/bohr^2"),
            ("center_angle_hessian", "hartree/bohr/radian"),
            ("angle_hessian", "hartree/radian^2"),
        ] {
            assert_eq!(units.get(key).unwrap().as_str(), Some(label));
        }
        for name in ["euler_gradient", "center_angle_hessian", "angle_hessian"] {
            assert!(
                flatten(report.get(name).unwrap())
                    .iter()
                    .all(|x| x.abs() < 1e-11),
                "{name}"
            );
        }
        let g = flatten(report.get("center_gradient").unwrap());
        let f = flatten(report.get("forces").unwrap());
        for (a, b) in g.iter().zip(f) {
            assert!((a + b).abs() < 1e-14);
        }
        assert!(dft_rs::driver::units::complete(&mut report.clone()).is_empty());
    }
    for key in [
        "energy",
        "center_gradient",
        "center_hessian",
        "angle_hessian",
    ] {
        for (a, b) in flatten(reports[0].get(key).unwrap())
            .iter()
            .zip(flatten(reports[1].get(key).unwrap()))
        {
            assert!((a - b).abs() < 1e-10, "{key}: {a} {b}");
        }
    }
}
