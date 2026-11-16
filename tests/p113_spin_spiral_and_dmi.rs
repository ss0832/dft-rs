// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Commensurate spin spirals and their chirality-odd spin-orbit energy on
//! the common driver: invariance of the spiral energy under a global spin
//! rotation and under a reversal of the wave vector without spin-orbit
//! coupling, the ferromagnetic limit at zero wave vector, the vanishing of
//! the chirality-odd energy with inversion symmetry, and a four-step fit of
//! the self-consistent chirality-odd energy against its first-order value
//! in the spin-orbit scale (finite differences are validation only).
//! Prepared with the implementation and not yet run: the step interval of
//! the fit was chosen beforehand, and the measured exponent, fit quality
//! and resolved interval are recorded here by the acceptance run.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
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

fn list(values: &[f64]) -> Value {
    Value::Array(values.iter().map(|v| Value::Float(*v)).collect())
}

fn mesh(n: i64) -> Value {
    Value::Array(vec![Value::Int(n), Value::Int(1), Value::Int(1)])
}

/// A wire of hydrogen atoms: one local moment per cell, inversion symmetric.
fn wire(spacing: f64) -> System {
    System {
        numbers: vec![1],
        positions: vec![[0.0, 0.0, 0.0]],
        cell: Some([[spacing, 0.0, 0.0], [0.0; 3], [0.0; 3]]),
        pbc: [true, false, false],
    }
}

/// The same wire with a helium atom off its axis: no centre of inversion,
/// the mirror plane of the wire and the helium atoms remains.
fn polar_wire() -> System {
    System {
        numbers: vec![1, 2],
        positions: vec![[0.0, 0.0, 0.0], [2.4, 1.5, 0.0]],
        cell: Some([[5.0, 0.0, 0.0], [0.0; 3], [0.0; 3]]),
        pbc: [true, false, false],
    }
}

#[test]
fn spiral_energy_is_invariant_under_spin_rotation_and_reversal_of_the_wave_vector() {
    let system = wire(5.0);
    let spiral = |q: f64, axis: [f64; 3]| {
        run(
            "lab-frame-spin-spiral",
            &system,
            &controls(&[
                ("k_mesh", mesh(2)),
                ("spiral_q", list(&[q, 0.0, 0.0])),
                ("spiral_axis", list(&axis)),
            ]),
        )
        .unwrap()
    };
    let reference = spiral(0.5, [0.0, 0.0, 1.0]);
    let energy = number(&reference, "energy");
    let rotated = number(&spiral(0.5, [1.0, 0.0, 0.0]), "energy");
    let reversed = number(&spiral(-0.5, [0.0, 0.0, 1.0]), "energy");
    let projection = number(&reference, "spiral_spin_projection");
    println!(
        "spiral energy {energy:.10} rotated axis {:.3e} reversed q {:.3e} projection {projection:.6}",
        rotated - energy,
        reversed - energy
    );
    // The spins settle against the seed field and keep their moment.
    assert!(projection < -0.3);
    assert!((rotated - energy).abs() < 1e-7);
    assert!((reversed - energy).abs() < 1e-7);
}

#[test]
fn zero_wave_vector_gives_the_ferromagnet_above_the_antiferromagnet() {
    let system = wire(5.0);
    let spiral = |q: f64| {
        run(
            "lab-frame-spin-spiral",
            &system,
            &controls(&[("k_mesh", mesh(2)), ("spiral_q", list(&[q, 0.0, 0.0]))]),
        )
        .unwrap()
    };
    let ferro = spiral(0.0);
    let anti = spiral(0.5);
    let moment = ferro
        .get("magnetization")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    let norm = moment.iter().map(|m| m * m).sum::<f64>().sqrt();
    println!(
        "ferromagnet moment {moment:?} energy {:.10} antiferromagnet {:.10}",
        number(&ferro, "energy"),
        number(&anti, "energy")
    );
    // One filled spinor band of a single spin direction, against the seed
    // field along the first frame vector (x for the default axis).
    assert!((norm - 1.0).abs() < 1e-6);
    assert!(moment[0] < 0.0);
    assert!(number(&anti, "energy") < number(&ferro, "energy"));
}

#[test]
fn chirality_odd_energy_vanishes_with_inversion_symmetry() {
    let report = run(
        "dmi",
        &wire(5.0),
        &controls(&[
            ("k_mesh", mesh(4)),
            ("spiral_q", list(&[0.25, 0.0, 0.0])),
            ("nuclear_soc_scale", Value::Float(1000.0)),
        ]),
    )
    .unwrap();
    let odd = number(&report, "dmi_energy");
    let even = number(&report, "soc_even_energy");
    let defect = number(&report, "chirality_rotation_energy_defect");
    println!(
        "inversion symmetric wire: odd {odd:.3e} even {even:.3e} rotation defect {defect:.3e}"
    );
    assert!(defect < 1e-8);
    assert!(odd.abs() < 1e-10);
}

#[test]
fn self_consistent_chirality_odd_energy_passes_a_four_step_fit_of_its_first_order() {
    let system = polar_wire();
    // Spin-orbit scales as the steps: the first-order value is linear in the
    // scale. Only s shells in one mirror plane: complex conjugation maps one
    // chirality at a scale onto the other at the opposite scale, so the
    // chirality-odd energy is odd in the scale and the remainder per unit
    // scale is quadratic.
    let steps = [2000.0, 1000.0, 500.0, 250.0];
    let mut slopes = Vec::new();
    let mut errors = Vec::new();
    for &scale in &steps {
        let report = run(
            "dmi",
            &system,
            &controls(&[
                ("k_mesh", mesh(4)),
                ("spiral_q", list(&[0.25, 0.0, 0.0])),
                ("nuclear_soc_scale", Value::Float(scale)),
                ("dmi_self_consistent", Value::Bool(true)),
            ]),
        )
        .unwrap();
        let first = number(&report, "dmi_energy") / scale;
        let full = number(&report, "dmi_energy_self_consistent") / scale;
        slopes.push(first);
        errors.push((full - first).abs());
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "chirality-odd spin-orbit energy per unit scale {slopes:?}: steps {steps:?} errors \
         {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!(slopes[0].abs() > 1e-12);
    assert!((fit.exponent - 2.0).abs() < 0.3 && fit.r_squared > 0.98);
}
