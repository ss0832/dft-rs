// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Actual common-driver nuclear-A shapes, scaling and reference boundaries.
//! Independent complete derivative fits are in p_hyperfine_nuclear.rs.

use dft_rs::driver::{Controls, System, Value};
use dft_rs::DftError;

fn system() -> System {
    System::molecule(
        vec![6, 1, 1, 1],
        vec![
            [0.13, -0.21, 0.17],
            [2.05, 0.25, 0.38],
            [-0.65, 1.76, -0.44],
            [-1.19, -1.48, 0.85],
        ],
    )
}

fn parameters() -> Value {
    Value::object()
        .with("xc", "hf")
        .with("reference", "uhf")
        .with("basis", "sto-3g")
        .with("multiplicity", 2usize)
        .with("unrestricted", true)
        .with("dispersion", "none")
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-12)
        .with("conv_tol_dp", 1e-13)
        .with("cpu_threads", 1usize)
        .with("memory_mb", 512usize)
}

#[test]
fn common_driver_returns_complete_hf_nuclear_tensors_and_preserves_static_values() {
    let system = system();
    let static_parameters = parameters();
    let controls = static_parameters
        .clone()
        .with("nuclear_order", 2usize)
        .with("derivative_nuclei", vec![0usize])
        .with("nuclear_g", vec![-2.0, 1.0, 1.0, 1.0])
        .with("response_tolerance", 1e-12)
        .with("response_max_iterations", 1000usize);
    let report = dft_rs::driver::run(
        "hyperfine",
        &system,
        &Controls::from_value(&controls).unwrap(),
    )
    .unwrap();
    assert_eq!(report.get("converged"), Some(&Value::Bool(true)));
    let nuclear = report.get("hyperfine_nuclear").unwrap();
    assert_eq!(nuclear.get("order"), Some(&Value::Int(2)));
    assert_eq!(nuclear.get("converged"), Some(&Value::Bool(true)));
    assert_eq!(
        nuclear.get("derivative_nuclei"),
        Some(&Value::Array(vec![Value::Int(0)]))
    );
    for part in ["contact", "dipolar", "total"] {
        for derivative in ["gradient", "hessian"] {
            let key = format!("{part}_{derivative}");
            let (shape, values) = nuclear
                .get(&format!("{key}_per_g"))
                .unwrap()
                .numeric_array()
                .unwrap();
            let (scaled_shape, scaled) = nuclear.get(&key).unwrap().numeric_array().unwrap();
            let mut expected = vec![1, 12];
            if derivative == "hessian" {
                expected.push(12);
            }
            if part != "contact" {
                expected.extend([3, 3]);
            }
            assert_eq!(shape, expected);
            assert_eq!(shape, scaled_shape);
            for (value, actual) in values.iter().zip(scaled) {
                assert!(value.is_finite() && actual.is_finite());
                assert!((actual + 2.0 * value).abs() < 1e-10);
            }
        }
    }
    let static_report = dft_rs::driver::run(
        "hyperfine",
        &system,
        &Controls::from_value(&static_parameters).unwrap(),
    )
    .unwrap();
    assert!(static_report.get("hyperfine_nuclear").is_none());
    for key in [
        "isotropic_coupling_per_g_mhz",
        "dipolar_coupling_per_g_mhz",
        "total_coupling_per_g_mhz",
    ] {
        let original = static_report.get(key).unwrap().numeric_array().unwrap();
        let current = report.get(key).unwrap().numeric_array().unwrap();
        assert_eq!(original.0, current.0);
        for (a, b) in original.1.iter().zip(current.1) {
            assert!((a - b).abs() < 1e-8);
        }
    }
    println!("RUST_NUCLEAR_A_REPORT {}", report.to_json());
}

#[test]
fn common_driver_nuclear_preflight_distinguishes_capability_and_malformed_input() {
    for reference in ["roks", "rohf", "gks", "ghf"] {
        let controls = parameters()
            .with("basis", "invalid-test-basis")
            .with("reference", reference)
            .with("nuclear_order", 1usize);
        let error = dft_rs::driver::run(
            "hyperfine",
            &system(),
            &Controls::from_value(&controls).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(error, DftError::Unsupported { .. }));
    }
    for (order, unsupported) in [
        (Value::Int(3), true),
        (Value::Int(-1), false),
        (Value::Bool(true), false),
    ] {
        let controls = parameters()
            .with("basis", "invalid-test-basis")
            .with("nuclear_order", order);
        let error = dft_rs::driver::run(
            "hyperfine",
            &system(),
            &Controls::from_value(&controls).unwrap(),
        )
        .unwrap_err();
        if unsupported {
            assert!(matches!(error, DftError::Unsupported { .. }));
        } else {
            assert!(matches!(error, DftError::InvalidInput(_)));
        }
    }
}
