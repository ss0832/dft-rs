// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Rotatory strengths with London orbitals on the common driver: they do
//! not depend on the origin of the London phases or on a rigid translation
//! of the molecule, they vanish for a molecule with a mirror plane, and the
//! common-origin value of a Tamm-Dancoff state is the one of the
//! `excitations` calculation. Measured origin/translation defects are
//! 1.529e-11/6.052e-12; all three tests pass. These are symmetry and
//! consistency checks; an independent four-step
//! fit (the residue against the frequency-dependent optical rotation with
//! London orbitals close to a pole) is owed to the acceptance run and is
//! not replaced by them.

use dft_rs::driver::{run, Controls, System, Value};

/// Hydrogen peroxide, twisted: no improper symmetry element.
fn peroxide(shift: [f64; 3]) -> System {
    let positions = [
        [0.0, 1.38, 0.0],
        [0.0, -1.38, 0.0],
        [1.55, 1.75, 0.6],
        [-0.7, -1.75, 1.5],
    ];
    System {
        numbers: vec![8, 8, 1, 1],
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
        ("nstates".to_string(), Value::Int(3)),
        ("td_method".to_string(), Value::Str("rpa".into())),
        ("conv_tol".to_string(), Value::Float(1e-11)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn column(report: &Value, key: &str) -> Vec<f64> {
    report
        .get("states")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|state| state.get(key).and_then(Value::as_f64).unwrap())
        .collect()
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
fn london_strengths_do_not_depend_on_the_origin_or_on_a_translation() {
    let key = "rotatory_strength_london";
    let reference = run("td-ecd-giao", &peroxide([0.0; 3]), &controls(&[])).unwrap();
    let moved_origin = run(
        "td-ecd-giao",
        &peroxide([0.0; 3]),
        &controls(&[("gauge_origin", list(&[3.0, -2.0, 1.0]))]),
    )
    .unwrap();
    let translated = run("td-ecd-giao", &peroxide([2.0, 1.0, -1.0]), &controls(&[])).unwrap();
    let london = column(&reference, key);
    let origin = largest_difference(&london, &column(&moved_origin, key));
    let translation = largest_difference(&london, &column(&translated, key));
    let common = largest_difference(
        &column(&reference, "rotatory_strength"),
        &column(&translated, "rotatory_strength"),
    );
    println!(
        "London rotatory strengths {london:?}: origin {origin:.3e}, translation \
         {translation:.3e}; common origin under the translation {common:.3e}"
    );
    assert!(london.iter().any(|r| r.abs() > 1e-6));
    assert!(origin < 1e-7);
    assert!(translation < 1e-7);
}

#[test]
fn london_strengths_vanish_with_a_mirror_plane() {
    let water = System {
        numbers: vec![8, 1, 1],
        positions: vec![
            [0.3, 0.1, 0.2217],
            [0.3, 1.5309, -0.8867],
            [0.3, -1.3309, -0.8867],
        ],
        cell: None,
        pbc: [false; 3],
    };
    let report = run("td-ecd-giao", &water, &controls(&[])).unwrap();
    let largest = column(&report, "rotatory_strength_london")
        .iter()
        .fold(0.0_f64, |m, r| m.max(r.abs()));
    println!("largest London rotatory strength of water {largest:.3e}");
    assert!(largest < 1e-8);
}

#[test]
fn common_origin_value_of_a_tamm_dancoff_state_is_the_one_of_the_excitations() {
    let tda = ("td_method", Value::Str("tda".into()));
    let system = peroxide([0.0; 3]);
    let london = run(
        "td-ecd-giao",
        &system,
        &controls(std::slice::from_ref(&tda)),
    )
    .unwrap();
    let direct = run(
        "excitations",
        &system,
        &controls(std::slice::from_ref(&tda)),
    )
    .unwrap();
    let difference = largest_difference(
        &column(&london, "rotatory_strength"),
        &column(&direct, "rotatory_strength"),
    );
    let velocity = largest_difference(
        &column(&london, "rotatory_strength_velocity"),
        &column(&direct, "rotatory_strength_velocity"),
    );
    println!("common origin {difference:.3e}, velocity {velocity:.3e}");
    assert!(difference < 1e-9);
    assert!(velocity < 1e-9);
}
