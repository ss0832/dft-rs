// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Relaxed-ion electric response of cells on the common driver: a cell with
//! one atom has no internal coordinate, so its relaxed values are the
//! clamped ones; the lattice part of the static susceptibility is the Born
//! charges contracted with the field relaxation; and the static
//! susceptibility against central differences of the dipole of the cell in
//! a finite field with the atoms on their first-order relaxation, by
//! four-step fits of all nine components (finite differences are
//! validation only; a component that fails on the first interval is
//! recomputed on a finer and then a coarser one).

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use components::{assert_no_failures, fit_components};

use dft_rs::driver::{run, Controls, System, Value};

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions (the partition-weight cutoff otherwise
        // adds noise near 1e-13 to displaced gradients; see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(1), Value::Int(1), Value::Int(1)]),
        ),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn relaxed() -> (&'static str, Value) {
    ("relaxed_ion", Value::Bool(true))
}

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

/// Lithium hydride without symmetry in a cubic cell; `shift` displaces the
/// atoms (six Cartesian values).
fn hydride(shift: &[f64]) -> System {
    let base = [[0.0, 0.0, 0.0], [3.3, 0.2, 0.1]];
    System {
        numbers: vec![3, 1],
        positions: (0..2)
            .map(|atom| std::array::from_fn(|axis| base[atom][axis] + shift[3 * atom + axis]))
            .collect(),
        cell: Some([[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]]),
        pbc: [true; 3],
    }
}

#[test]
fn one_atom_cell_has_no_lattice_contribution() {
    let system = System {
        numbers: vec![2],
        positions: vec![[0.0, 0.0, 0.0]],
        cell: Some([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]),
        pbc: [true; 3],
    };
    let report = run("electrostriction", &system, &controls(&[relaxed()])).unwrap();
    let difference = values(&report, "relaxed_susceptibility")
        .iter()
        .zip(&values(&report, "susceptibility"))
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let motion = values(&report, "field_relaxation")
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    println!("one-atom cell: relaxed against clamped {difference:.3e}, relaxation {motion:.3e}");
    assert!(difference < 1e-10);
    assert!(motion < 1e-10);
}

#[test]
fn lattice_susceptibility_is_the_born_charges_on_the_field_relaxation() {
    let report = run(
        "periodic-raman",
        &hydride(&[0.0; 6]),
        &controls(&[relaxed()]),
    )
    .unwrap();
    let clamped = values(&report, "susceptibility");
    let relaxed = values(&report, "relaxed_susceptibility");
    let born = values(&report, "born_charges");
    let motion = values(&report, "field_relaxation");
    let volume = report.get("volume").and_then(Value::as_f64).unwrap();
    let mut worst = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            let lattice: f64 = (0..6)
                .map(|k| born[3 * k + a] * motion[3 * k + b])
                .sum::<f64>()
                / volume;
            worst = worst.max((relaxed[3 * a + b] - clamped[3 * a + b] - lattice).abs());
        }
    }
    let gain = (0..3).map(|a| relaxed[4 * a] - clamped[4 * a]).sum::<f64>();
    println!("lattice susceptibility identity {worst:.3e}, trace gain {gain:.6e}");
    assert!(worst < 1e-8);
    assert!(gain > 0.0);
}

#[test]
fn static_susceptibility_passes_four_step_fits_of_the_dipole_in_a_field_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.008, 0.004, 0.002, 0.001],
        [0.004, 0.002, 0.001, 0.0005],
        [0.016, 0.008, 0.004, 0.002],
    ];
    const AXES: [&str; 3] = ["x", "y", "z"];
    let report = run(
        "periodic-raman",
        &hydride(&[0.0; 6]),
        &controls(&[relaxed()]),
    )
    .unwrap();
    let chi = values(&report, "relaxed_susceptibility");
    let motion = values(&report, "field_relaxation");
    let volume = report.get("volume").and_then(Value::as_f64).unwrap();
    assert_eq!((chi.len(), motion.len()), (9, 18));
    println!("relaxed static susceptibility {chi:?}");
    let mut failures = Vec::new();
    for b in 0..3 {
        // The atoms follow the field along axis b to first order.
        let dipole = |strength: f64| {
            let shift: Vec<f64> = (0..6).map(|k| strength * motion[3 * k + b]).collect();
            let mut vector = [0.0; 3];
            vector[b] = strength;
            let field = Value::Array(vector.iter().map(|v| Value::Float(*v)).collect());
            values(
                &run(
                    "periodic-efield",
                    &hydride(&shift),
                    &controls(&[("field", field)]),
                )
                .unwrap(),
                "dipole",
            )
        };
        let names: Vec<String> = (0..3)
            .map(|a| format!("chi_{}{}", AXES[a], AXES[b]))
            .collect();
        fit_components(
            &names,
            &INTERVALS,
            |steps| {
                let mut errors = vec![[0.0; 4]; 3];
                for (s, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (dipole(h), dipole(-h));
                    for a in 0..3 {
                        errors[a][s] =
                            ((plus[a] - minus[a]) / (2.0 * h * volume) - chi[3 * a + b]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    assert_no_failures("relaxed static susceptibility", &failures);
}
