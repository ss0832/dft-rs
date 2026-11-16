// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular Fermi-Dirac (Mermin) references: the zero-temperature limit,
//! the thermodynamic identity dA/dT = -S, the analytic free-energy gradient
//! and the refusal of integer-occupation response routes, each checked
//! independently of the implementation.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K;

fn stretched_h2(bond: f64) -> System {
    System::molecule(vec![1, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, bond]])
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("basis".to_string(), Value::Str("6-31g".into())),
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("grid_level".to_string(), Value::Int(4)),
        ("conv_tol".to_string(), Value::Float(1e-13)),
        ("conv_tol_dp".to_string(), Value::Float(1e-11)),
        ("max_iter".to_string(), Value::Int(400)),
    ];
    pairs.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn fit(label: &str, steps: &[f64], errors: &[f64]) {
    let fit = convergence::error_power_fit(steps, errors);
    println!(
        "{label}: steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!(
        (fit.exponent - 2.0).abs() < 0.02,
        "{label}: p = {}",
        fit.exponent
    );
    assert!(fit.r_squared > 0.99999, "{label}: R2 = {}", fit.r_squared);
}

#[test]
fn occupations_conserve_electrons_and_vanish_at_low_temperature() {
    let system = stretched_h2(3.0);
    let cold = run(
        "free-energy",
        &system,
        &controls(&[("electronic_temperature", Value::Float(1.0))]),
    )
    .unwrap();
    let zero = run("energy", &system, &controls(&[])).unwrap();
    assert!((number(&cold, "free_energy") - number(&zero, "energy")).abs() < 1e-10);
    assert!(number(&cold, "entropy") < 1e-12);
    let hot = run(
        "free-energy",
        &system,
        &controls(&[("electronic_temperature", Value::Float(6000.0))]),
    )
    .unwrap();
    let occupations = hot.get("occupations").unwrap().as_array().unwrap();
    for channel in occupations {
        let (_, f) = channel.numeric_array().unwrap();
        let total: f64 = f.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "electrons per spin {total}");
        assert!(f.iter().all(|x| (0.0..=1.0).contains(x)));
    }
    let entropy = number(&hot, "entropy");
    assert!(entropy > 1e-3, "entropy {entropy}");
    let kt = number(&hot, "kt");
    assert!(
        (number(&hot, "internal_energy") - number(&hot, "free_energy") - kt * entropy).abs()
            < 1e-14
    );
    // A is a minimum over occupations: the smeared free energy lies below
    // the integer-occupation energy evaluated at the same temperature.
    assert!(number(&hot, "free_energy") < number(&zero, "energy"));
}

#[test]
fn free_energy_temperature_derivative_is_minus_the_entropy() {
    let system = stretched_h2(3.0);
    let at = |kelvin: f64| {
        run(
            "free-energy",
            &system,
            &controls(&[("electronic_temperature", Value::Float(kelvin))]),
        )
        .unwrap()
    };
    let center = at(6000.0);
    // dA/dT in Hartree/K equals -S k_B (S reported in units of k_B).
    let analytic = -number(&center, "entropy") * BOLTZMANN_HARTREE_PER_K;
    let steps = [400.0, 200.0, 100.0, 50.0];
    let errors: Vec<f64> = steps
        .iter()
        .map(|&h| {
            let derivative = (number(&at(6000.0 + h), "free_energy")
                - number(&at(6000.0 - h), "free_energy"))
                / (2.0 * h);
            ((derivative - analytic) / analytic).abs()
        })
        .collect();
    fit("dA/dT + S (relative)", &steps, &errors);
}

#[test]
fn analytic_free_energy_gradient_passes_a_four_step_fit() {
    let temperature = [
        ("electronic_temperature", Value::Float(6000.0)),
        ("gradient", Value::Bool(true)),
    ];
    let report = run("free-energy", &stretched_h2(3.0), &controls(&temperature)).unwrap();
    let (_, gradient) = report.get("gradient").unwrap().numeric_array().unwrap();
    assert!((gradient[2] + gradient[5]).abs() < 1e-8, "translation sum");
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let at = |bond: f64| {
        number(
            &run(
                "free-energy",
                &stretched_h2(bond),
                &controls(&[("electronic_temperature", Value::Float(6000.0))]),
            )
            .unwrap(),
            "free_energy",
        )
    };
    let errors: Vec<f64> = steps
        .iter()
        .map(|&h| ((at(3.0 + h) - at(3.0 - h)) / (2.0 * h) - gradient[5]).abs())
        .collect();
    fit("dA/dR(H2 z)", &steps, &errors);
}

#[test]
fn spectral_static_response_and_integer_projector_refusals_are_distinct() {
    let hot = [("electronic_temperature", Value::Float(6000.0))];
    let system = stretched_h2(3.0);
    let electric = run("polarizability", &system, &controls(&hot)).unwrap();
    assert!(electric
        .get("response_population_model")
        .and_then(Value::as_str)
        .unwrap()
        .contains("Fermi-Dirac"));
    let (_, residuals) = electric
        .get("response_residuals")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert!(residuals.len() == 3 && residuals.iter().all(|r| *r <= 1e-10));
    // The spectral static routes include population response. The distinct
    // integer-projector solver must still reject a smeared reference.
    let method = dft_rs::driver::method::MolecularMethod::from_controls(&controls(&hot)).unwrap();
    let setup =
        dft_rs::driver::method::prepare_molecular(&system.numbers, &system.positions, &method)
            .unwrap();
    let reference = dft_rs::driver::method::run_molecular_reference(&setup).unwrap();
    assert!(matches!(
        dft_rs::response::ResponseSolver::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options
        ),
        Err(dft_rs::DftError::Unsupported { .. })
    ));
    let hessian = run("hessian", &stretched_h2(3.0), &controls(&hot)).unwrap();
    assert!(
        hessian
            .get("ensemble_response_residuals")
            .unwrap()
            .as_array()
            .unwrap()
            .len()
            == 6
    );
}
