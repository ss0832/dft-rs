// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spin-dipolar hyperfine tensors on the common driver: they are symmetric
//! and traceless, vanish for the spherical spin density of a hydrogen atom,
//! and turn with the molecule and its state-selecting electric field.
//! The independent operator derivative fit is in p127_hyperfine_tensors.

use dft_rs::driver::{run, Controls, System, Value};

fn controls(field: Option<[f64; 3]>) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("pbe".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("multiplicity".to_string(), Value::Int(2)),
        ("unrestricted".to_string(), Value::Bool(true)),
    ];
    if let Some(field) = field {
        pairs.push((
            "field".into(),
            Value::Array(field.into_iter().map(Value::Float).collect()),
        ));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

/// Tensors as `[nucleus][3 a + b]`.
fn tensors(system: &System, field: Option<[f64; 3]>) -> Vec<Vec<f64>> {
    let report = run("hyperfine", system, &controls(field)).unwrap();
    report
        .get("dipolar_coupling_per_g_mhz")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|tensor| tensor.numeric_array().unwrap().1)
        .collect()
}

fn hydroxyl(swap: bool) -> System {
    // The second geometry is the first turned by a quarter about z:
    // (x, y, z) -> (-y, x, z).
    let base = [[0.1, -0.05, 0.0], [0.9, 0.7, 1.3]];
    System {
        numbers: vec![8, 1],
        positions: base
            .iter()
            .map(|p| if swap { [-p[1], p[0], p[2]] } else { *p })
            .collect(),
        cell: None,
        pbc: [false; 3],
    }
}

#[test]
fn dipolar_tensors_are_symmetric_and_traceless() {
    for tensor in tensors(&hydroxyl(false), None) {
        let scale = tensor.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let trace = tensor[0] + tensor[4] + tensor[8];
        let asymmetry = (tensor[1] - tensor[3])
            .abs()
            .max((tensor[2] - tensor[6]).abs())
            .max((tensor[5] - tensor[7]).abs());
        println!("dipolar tensor scale {scale:.4e}, trace {trace:.3e}, asymmetry {asymmetry:.3e}");
        assert!(scale > 1e-6);
        assert!(trace.abs() < 1e-8 * scale.max(1.0));
        assert!(asymmetry < 1e-8 * scale.max(1.0));
    }
}

#[test]
fn hydrogen_atom_has_no_dipolar_coupling() {
    let atom = System {
        numbers: vec![1],
        positions: vec![[0.3, -0.2, 0.1]],
        cell: None,
        pbc: [false; 3],
    };
    let largest = tensors(&atom, None)[0]
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    println!("largest dipolar component of the hydrogen atom {largest:.3e}");
    assert!(largest < 1e-8);
}

#[test]
fn dipolar_tensors_turn_with_the_molecule() {
    // Zero-field OH has a degenerate pi manifold: independently initialized
    // determinants can select different azimuthal states. Rotate a transverse
    // field with the geometry to compare the same selected electronic state.
    let field = [0.02, 0.03, 0.01];
    let first = tensors(&hydroxyl(false), Some(field));
    let second = tensors(&hydroxyl(true), Some([-field[1], field[0], field[2]]));
    // R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]; T' = R T R^T.
    let rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let mut worst = 0.0_f64;
    for (a, b) in first.iter().zip(&second) {
        for i in 0..3 {
            for j in 0..3 {
                let mut expected = 0.0;
                for k in 0..3 {
                    for l in 0..3 {
                        expected += rotation[i][k] * a[3 * k + l] * rotation[j][l];
                    }
                }
                worst = worst.max((b[3 * i + j] - expected).abs());
            }
        }
    }
    println!("rotation of the dipolar tensors {worst:.3e}");
    assert!(worst < 1e-6);
}
