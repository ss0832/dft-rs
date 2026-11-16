// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Symmetry limits of the magnetoelectric response and of the Berry
//! curvature dipole on the common driver: both vanish with inversion
//! symmetry, and the magnetoelectric response needs an open shell.
//! Prepared with the implementation. Independent four-step fits (the
//! electric dipole of finite magnetic field states with a spin-orbit term,
//! and the curvature dipole against differences of the anomalous Hall
//! response over the chemical potential) are owed to the acceptance run and
//! are not replaced by these checks.

use dft_rs::driver::{run, Controls, System, Value};

fn controls(pairs: Vec<(&str, Value)>) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    ))
    .unwrap()
}

fn largest(report: &Value, key: &str) -> f64 {
    report
        .get(key)
        .unwrap()
        .numeric_array()
        .unwrap()
        .1
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()))
}

#[test]
fn magnetoelectric_response_vanishes_with_inversion_and_not_without() {
    let base = || {
        vec![
            ("xc", Value::Str("pbe".into())),
            ("basis", Value::Str("sto-3g".into())),
            ("unrestricted", Value::Bool(true)),
        ]
    };
    // Triplet dioxygen has a centre of inversion.
    let oxygen = System {
        numbers: vec![8, 8],
        positions: vec![[0.0, 0.0, -1.14], [0.0, 0.0, 1.14]],
        cell: None,
        pbc: [false; 3],
    };
    let mut triplet = base();
    triplet.push(("multiplicity", Value::Int(3)));
    let symmetric = largest(
        &run("magnetoelectric", &oxygen, &controls(triplet)).unwrap(),
        "magnetoelectric_tensor",
    );
    // The hydroxyl radical has none.
    let hydroxyl = System {
        numbers: vec![8, 1],
        positions: vec![[0.1, -0.05, 0.0], [0.9, 0.7, 1.3]],
        cell: None,
        pbc: [false; 3],
    };
    let mut doublet = base();
    doublet.push(("multiplicity", Value::Int(2)));
    let polar = largest(
        &run("magnetoelectric", &hydroxyl, &controls(doublet)).unwrap(),
        "magnetoelectric_tensor",
    );
    println!("magnetoelectric tensor: centrosymmetric {symmetric:.3e}, polar {polar:.3e}");
    assert!(polar > 1e-10);
    assert!(symmetric < 1e-6 * polar.max(1e-6));
}

#[test]
fn magnetoelectric_response_refuses_a_closed_shell() {
    let water = System {
        numbers: vec![8, 1, 1],
        positions: vec![
            [0.0, 0.0, 0.2217],
            [0.0, 1.4309, -0.8867],
            [0.0, -1.4309, -0.8867],
        ],
        cell: None,
        pbc: [false; 3],
    };
    let result = run(
        "magnetoelectric",
        &water,
        &controls(vec![
            ("xc", Value::Str("pbe".into())),
            ("basis", Value::Str("sto-3g".into())),
        ]),
    );
    assert!(result.is_err());
}

#[test]
fn berry_curvature_dipole_vanishes_in_a_centrosymmetric_cell() {
    // A hydrogen molecule centred in the cell: inversion about its midpoint.
    let system = System {
        numbers: vec![1, 1],
        positions: vec![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]],
        cell: Some([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]),
        pbc: [true; 3],
    };
    let report = run(
        "berry-dipole",
        &system,
        &controls(vec![
            ("xc", Value::Str("svwn5".into())),
            ("basis", Value::Str("sto-3g".into())),
            (
                "k_mesh",
                Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
            ),
            (
                "transport_k_mesh",
                Value::Array(vec![Value::Int(4), Value::Int(3), Value::Int(3)]),
            ),
            ("transport_temperature_k", Value::Float(3000.0)),
            ("berry_dipole_doping", Value::Float(0.2)),
        ]),
    )
    .unwrap();
    let dipole = largest(&report, "berry_curvature_dipole");
    let weight = report.get("fermi_weight").and_then(Value::as_f64).unwrap();
    println!("Berry curvature dipole {dipole:.3e} with Fermi weight {weight:.3e}");
    assert!(weight > 0.0);
    assert!(dipole < 1e-8);
}
