// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent spin-orbit spinor state and its dynamic polarizability on
//! the common driver: without spin-orbit coupling the spinor response is the
//! collinear one; the antisymmetric imaginary part per unit spin-orbit scale
//! against the first-order magneto-optical tensor by a four-step fit; the
//! self-consistent spin-orbit energy is variational and quadratic in the
//! coupling, by a second four-step fit (finite differences are validation
//! only). The first run passed the collinear limit (2.459e-11) and the
//! quadratic energy (p = 1.998227, R^2 = 0.99999995 over [8, 4, 2, 1]) and
//! failed the antisymmetric fit over the spin-orbit scales [8, 4, 2, 1]:
//! errors [1.581e-5, 1.047e-5, 5.857e-6, 3.081e-6], p = 0.79, R^2 = 0.9908.
//! The hydroxyl radical has a nearly degenerate empty pi spin orbital, so the
//! response is far from linear at the larger scales (local exponents 0.59,
//! 0.84, 0.93); the fit now uses the resolved interval [2, 1, 0.5, 0.25]
//! with the same bounds.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

fn hydroxyl() -> System {
    System {
        numbers: vec![8, 1],
        positions: vec![[0.1, -0.05, 0.0], [0.9, 0.7, 1.3]],
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("pbe".into())),
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

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn list(items: &[f64]) -> Value {
    Value::Array(items.iter().map(|v| Value::Float(*v)).collect())
}

#[test]
fn without_spin_orbit_coupling_the_spinor_response_is_the_collinear_one() {
    let system = hydroxyl();
    let frequency = 0.1;
    // At this frequency the response amplifies small differences between
    // independently converged collinear and spinor stationary states.
    let converged_controls = |extra: &[(&str, Value)]| {
        let mut requested = vec![
            ("conv_tol", Value::Float(1e-13)),
            ("conv_tol_dp", Value::Float(1e-12)),
            ("conv_tol_grad", Value::Float(1e-12)),
            ("response_tolerance", Value::Float(1e-12)),
            ("max_iter", Value::Int(150)),
        ];
        requested.extend_from_slice(extra);
        controls(&requested)
    };
    let spinor = run(
        "spinor-polarizability",
        &system,
        &converged_controls(&[
            ("soc_scale", Value::Float(0.0)),
            ("frequencies", list(&[frequency])),
        ]),
    )
    .unwrap();
    let collinear = run(
        "magneto-optical",
        &system,
        &converged_controls(&[
            ("magneto_optical_source", Value::Str("spin-orbit".into())),
            ("frequency", Value::Float(frequency)),
        ]),
    )
    .unwrap();
    let real = values(&spinor, "polarizability_real");
    let imag = values(&spinor, "polarizability_imag");
    let expected = values(&collinear, "polarizability");
    let difference = real
        .iter()
        .zip(&expected)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let imaginary = imag.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let spin = values(&spinor, "spin_expectation");
    let shift = number(&spinor, "spin_orbit_energy");
    println!(
        "spinor against collinear polarizability {difference:.3e}, imaginary part \
         {imaginary:.3e}, spin {spin:?}, energy shift {shift:.3e}"
    );
    assert!(difference < 1e-6);
    assert!(imaginary < 1e-8);
    assert!(shift.abs() < 1e-8);
    assert!((spin[2] - 0.5).abs() < 1e-6 && spin[0].abs() < 1e-6 && spin[1].abs() < 1e-6);
}

#[test]
fn antisymmetric_part_passes_a_four_step_fit_of_the_magneto_optical_tensor() {
    let system = hydroxyl();
    let frequency = 0.1;
    let first = run(
        "magneto-optical",
        &system,
        &controls(&[
            ("magneto_optical_source", Value::Str("spin-orbit".into())),
            ("frequency", Value::Float(frequency)),
        ]),
    )
    .unwrap();
    // magneto_optical_axial[c][spin axis]: the column of the spins along z.
    let axial = values(&first, "magneto_optical_axial");
    let slope = [axial[2], axial[5], axial[8]];
    // The spin-orbit scale is the step: the antisymmetric imaginary part per
    // unit scale tends to the first-order tensor at least linearly.
    let steps = [2.0, 1.0, 0.5, 0.25];
    let mut errors = Vec::new();
    for &scale in &steps {
        let report = run(
            "spinor-polarizability",
            &system,
            &controls(&[
                ("soc_scale", Value::Float(scale)),
                ("frequencies", list(&[frequency])),
            ]),
        )
        .unwrap();
        let vector = values(&report, "polarizability_axial");
        errors.push((0..3).fold(0.0_f64, |m, c| m.max((vector[c] / scale - slope[c]).abs())));
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "antisymmetric polarizability per unit spin-orbit scale against {slope:?}: steps \
         {steps:?} errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!(slope.iter().any(|s| s.abs() > 1e-12));
    assert!(fit.exponent > 0.8 && fit.r_squared > 0.98);
}

#[test]
fn self_consistent_spin_orbit_energy_is_variational_and_quadratic_in_the_coupling() {
    let system = hydroxyl();
    // Real collinear orbitals have no first-order spin-orbit energy, and the
    // collinear state is a trial state of the spinor SCF: the energy change
    // is not positive and quadratic in the effective charges.
    let steps = [8.0, 4.0, 2.0, 1.0];
    let mut errors = Vec::new();
    for &scale in &steps {
        let report = run(
            "spin-orbit",
            &system,
            &controls(&[
                ("soc_charges", list(&[8.0 * scale, scale])),
                ("spin_orbit_self_consistent", Value::Bool(true)),
            ]),
        )
        .unwrap();
        let change = values(&report, "self_consistent_spin_orbit_energies")[0];
        assert!(change <= 1e-10);
        errors.push(change.abs());
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "self-consistent spin-orbit energy: steps {steps:?} changes {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.2 && fit.r_squared > 0.99);
}
