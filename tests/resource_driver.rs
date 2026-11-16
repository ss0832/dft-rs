// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Global resource controls survive feature-local control reconstruction.

use dft_rs::driver::{run, Controls, System, Value};

fn system() -> System {
    System::molecule(vec![1, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
}

#[test]
fn resource_controls_reject_floats_booleans_negative_zero_and_overflowing_values() {
    for (key, value) in [
        ("cpu_threads", Value::Float(1.0)),
        ("memory_mb", Value::Float(32.0)),
        ("cpu_threads", Value::Bool(true)),
        ("memory_mb", Value::Bool(false)),
        ("cpu_threads", Value::Int(-1)),
        ("memory_mb", Value::Int(-1)),
        ("cpu_threads", Value::Int(0)),
        ("memory_mb", Value::Int(0)),
        ("cpu_threads", Value::Int(i64::MAX)),
        ("memory_mb", Value::Int(i64::MAX)),
    ] {
        let mut controls = Controls::new();
        controls.insert(key, value).unwrap();
        let message = run("energy", &system(), &controls).unwrap_err().to_string();
        assert!(message.contains(key), "{message}");
    }
}

#[test]
fn reference_normalization_keeps_resource_limits_and_original_request() {
    let structure = system();
    for reference in ["rks", "rhf", "uks", "uhf", "auto"] {
        let mut controls = Controls::new();
        controls
            .insert("reference", Value::Str(reference.into()))
            .unwrap();
        let baseline = run("energy", &structure, &controls).unwrap();
        let expected = baseline.get("energy").unwrap().as_f64().unwrap();
        controls.insert("cpu_threads", Value::Int(2)).unwrap();
        controls.insert("memory_mb", Value::Int(32)).unwrap();
        let report = run("energy", &structure, &controls).unwrap();
        assert!((report.get("energy").unwrap().as_f64().unwrap() - expected).abs() < 1e-11);
        let resources = report.get("resources").unwrap();
        assert_eq!(resources.get("cpu_threads").unwrap().as_i64(), Some(2));
        assert_eq!(resources.get("memory_mb").unwrap().as_i64(), Some(32));
        assert_eq!(
            controls.to_value().get("cpu_threads").unwrap().as_i64(),
            Some(2)
        );
    }
}

#[test]
fn explicit_null_resources_inherit_defaults_but_unknown_feature_controls_still_fail() {
    let mut controls = Controls::new();
    controls
        .insert("reference", Value::Str("rks".into()))
        .unwrap();
    controls.insert("cpu_threads", Value::Null).unwrap();
    controls.insert("memory_mb", Value::Null).unwrap();
    let report = run("energy", &system(), &controls).unwrap();
    assert!(report.get("resources").is_none());
    controls
        .insert("unknown_contract_control", Value::Bool(true))
        .unwrap();
    let message = run("energy", &system(), &controls).unwrap_err().to_string();
    assert!(message.contains("unknown_contract_control"));
    assert!(!message.contains("cpu_threads") && !message.contains("memory_mb"));
}
