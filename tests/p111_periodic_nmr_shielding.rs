// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear magnetic shielding in a periodic cell on the common driver: a
//! molecule in a large cell, where the shielding at the smallest wave
//! vectors approaches the molecular value, with the full tensor from wave
//! vectors along two axes. Prepared with the implementation. The
//! large-cell limit against the molecular shielding with a common gauge
//! origin at the nucleus, and the independent four-step fits against
//! self-consistent calculations in a finite periodic vector potential, are
//! owed to the acceptance run and are not replaced by these checks.

use dft_rs::driver::{run, Controls, System, Value};

fn hydrogen(edge: f64) -> System {
    System {
        numbers: vec![1, 1],
        positions: vec![[0.2, 0.1, 0.0], [0.9, 0.5, 1.2]],
        cell: Some([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
        pbc: [true; 3],
    }
}

fn controls() -> Controls {
    let pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("overlap_cutoff".to_string(), Value::Float(20.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(2)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(200)),
        (
            "q_points".to_string(),
            Value::Array(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    .iter()
                    .map(|x| Value::Float(*x))
                    .collect(),
            ),
        ),
    ];
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

#[test]
fn shielding_of_a_molecule_in_a_large_cell_has_a_full_tensor() {
    let report = run("periodic-nmr-shielding", &hydrogen(14.0), &controls()).unwrap();
    let nuclei = match report.get("nuclei") {
        Some(Value::Array(list)) => list.clone(),
        other => panic!("nuclei missing: {other:?}"),
    };
    assert_eq!(nuclei.len(), 2);
    for nucleus in &nuclei {
        let tensor = nucleus
            .get("shielding_tensor_ppm")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        let isotropic = nucleus
            .get("isotropic_ppm")
            .and_then(Value::as_f64)
            .unwrap();
        println!("periodic shielding tensor (ppm) {tensor:?}, isotropic {isotropic:.6}");
        assert_eq!(tensor.len(), 9);
        assert!(tensor.iter().all(|x| x.is_finite()));
        // Proton shieldings of a hydrogen molecule are tens of ppm.
        assert!(isotropic > 5.0 && isotropic < 60.0);
    }
}
