// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electronic g tensor with London orbitals on the common driver: it does
//! not depend on the gauge origin or on a rigid translation of the
//! molecule, and for an atom with the origin on its nucleus it is the
//! common-origin tensor. The first acceptance run used the hydroxyl radical,
//! whose 2Pi ground state has a degenerate pair of pi orbitals: the
//! translated calculation converged to a hole rotated within that pair
//! (energy 7.4e-9 Hartree apart, g changed by 7.6e2 ppm), so the
//! translation test now uses the water cation, whose 2B1 hole is not
//! degenerate. Closed-shell London shieldings of water were translation
//! invariant to 1e-7 ppm in the same check.
//! These are symmetry and consistency checks; an independent four-step fit
//! (the mixed derivative of the energy of a two-component state in a finite
//! magnetic field) needs a calculation that does not exist yet and is owed
//! to the acceptance run.

use dft_rs::driver::{run, Controls, System, Value};

/// Water cation (Bohr), distorted away from C2v: an open shell whose
/// singly occupied orbital is not degenerate.
fn water_cation(shift: [f64; 3]) -> System {
    let positions = [
        [0.0, 0.0, 0.23],
        [0.0, 1.4309, -0.8867],
        [0.12, -1.4809, -0.9067],
    ];
    System {
        numbers: vec![8, 1, 1],
        positions: positions
            .iter()
            .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
            .collect(),
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("pbe0".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("multiplicity".to_string(), Value::Int(2)),
        ("unrestricted".to_string(), Value::Bool(true)),
        ("conv_tol".to_string(), Value::Float(1e-11)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn cation_controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![("charge", Value::Int(1))];
    pairs.extend(extra.iter().cloned());
    controls(&pairs)
}

fn tensor(report: &Value) -> Vec<f64> {
    report.get("g_tensor").unwrap().numeric_array().unwrap().1
}

fn list(items: &[f64]) -> Value {
    Value::Array(items.iter().map(|v| Value::Float(*v)).collect())
}

fn largest_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn london_tensor_does_not_depend_on_the_origin_or_on_a_translation() {
    let origin = |o: [f64; 3]| ("gauge_origin", list(&o));
    let common = ("gauge", Value::Str("common-origin".into()));
    let reference = tensor(
        &run(
            "g-tensor",
            &water_cation([0.0; 3]),
            &cation_controls(&[origin([0.0; 3])]),
        )
        .unwrap(),
    );
    let moved = tensor(
        &run(
            "g-tensor",
            &water_cation([0.0; 3]),
            &cation_controls(&[origin([3.0, -2.0, 1.0])]),
        )
        .unwrap(),
    );
    let translated = tensor(
        &run(
            "g-tensor",
            &water_cation([2.0, 1.0, -1.0]),
            &cation_controls(&[origin([0.0; 3])]),
        )
        .unwrap(),
    );
    let fixed = |o: [f64; 3]| {
        tensor(
            &run(
                "g-tensor",
                &water_cation([0.0; 3]),
                &cation_controls(&[origin(o), common.clone()]),
            )
            .unwrap(),
        )
    };
    let london_origin = largest_difference(&reference, &moved);
    let london_translation = largest_difference(&reference, &translated);
    let common_origin = largest_difference(&fixed([0.0; 3]), &fixed([3.0, -2.0, 1.0]));
    println!(
        "g tensor: London origin {london_origin:.3e}, translation {london_translation:.3e}; \
         common origin under the same origin shift {common_origin:.3e}"
    );
    assert!(london_origin < 1e-9);
    assert!(london_translation < 1e-9);
    assert!(common_origin > 1e-9);
}

#[test]
fn atom_with_the_origin_on_its_nucleus_has_the_common_origin_tensor() {
    // Boron: one centre, so the London phases are those of one origin.
    let atom = System {
        numbers: vec![5],
        positions: vec![[0.4, -0.3, 0.2]],
        cell: None,
        pbc: [false; 3],
    };
    let origin = ("gauge_origin", list(&[0.4, -0.3, 0.2]));
    let london = tensor(&run("g-tensor", &atom, &controls(std::slice::from_ref(&origin))).unwrap());
    let common = tensor(
        &run(
            "g-tensor",
            &atom,
            &controls(&[origin, ("gauge", Value::Str("common-origin".into()))]),
        )
        .unwrap(),
    );
    let difference = largest_difference(&london, &common);
    println!("one-centre g tensor: London against common origin {difference:.3e}");
    assert!(difference < 1e-9);
}
