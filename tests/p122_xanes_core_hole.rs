// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Core-hole relaxation of the K-edge calculation on the common driver: for
//! one electron in Hartree-Fock the orbital energy is the exact ionization
//! energy, so the relaxation shift vanishes; for the oxygen 1s orbital of
//! water in Hartree-Fock the orbitals relax around the hole, so the relaxed
//! ionization energy lies below the frozen-orbital estimate and the hole
//! stays in the core orbital. Measured: one-electron shift 2.220e-16
//! Hartree; water O 1s relaxed ionization 19.906435 Hartree, shift -0.335436
//! Hartree and minimum core-hole overlap 0.934376. Both tests pass. No
//! derivative is involved, so there is no step fit.

use dft_rs::driver::{run, Controls, System, Value};

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("hf".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("core_hole".to_string(), Value::Str("delta-scf".into())),
        ("conv_tol".to_string(), Value::Float(1e-11)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

#[test]
fn one_electron_molecule_has_no_relaxation_shift() {
    let system = System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        cell: None,
        pbc: [false; 3],
    };
    let report = run(
        "xanes",
        &system,
        &controls(&[
            ("charge", Value::Int(1)),
            ("multiplicity", Value::Int(2)),
            ("unrestricted", Value::Bool(true)),
            ("core_orbitals", Value::Array(vec![Value::Int(0)])),
            ("nstates", Value::Int(1)),
        ]),
    )
    .unwrap();
    let shift = number(&report, "core_hole_shift");
    let ionization = number(&report, "core_ionization_energy");
    println!("one electron: ionization {ionization:.10}, shift {shift:.3e}");
    assert!(ionization > 0.0);
    assert!(shift.abs() < 1e-8);
}

#[test]
fn oxygen_core_hole_of_water_relaxes_and_stays_in_the_core() {
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
    let report = run(
        "xanes",
        &water,
        &controls(&[("xanes_element", Value::Int(8)), ("nstates", Value::Int(2))]),
    )
    .unwrap();
    let shift = number(&report, "core_hole_shift");
    let ionization = number(&report, "core_ionization_energy");
    let overlap = number(&report, "core_hole_overlap");
    println!(
        "oxygen 1s: relaxed ionization {ionization:.6}, shift {shift:.6}, smallest overlap \
         {overlap:.6}"
    );
    assert!(ionization > 10.0);
    assert!(shift < 0.0);
    assert!(overlap > 0.5);
}
