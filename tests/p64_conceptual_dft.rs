// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Conceptual DFT routes. The analytic grand-canonical hardness and Fukui
//! functions are compared with independent four-step central differences of
//! the Fermi level and populations of fractional-electron ensembles; the
//! integer and Koopmans routes are checked against their defining energies
//! and normalization sum rules.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

fn lih() -> System {
    System::molecule(vec![3, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]])
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("basis".to_string(), Value::Str("6-31g".into())),
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("grid_level".to_string(), Value::Int(4)),
        ("conv_tol".to_string(), Value::Float(1e-13)),
        ("conv_tol_dp".to_string(), Value::Float(1e-11)),
        ("max_iter".to_string(), Value::Int(500)),
    ];
    pairs.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn number(report: &Value, path: &[&str]) -> f64 {
    let mut v = report;
    for key in path {
        v = v.get(key).unwrap();
    }
    v.as_f64().unwrap()
}

fn list(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
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

const KELVIN: f64 = 8000.0;

fn hot(extra: &[(&str, Value)]) -> Controls {
    let mut all = vec![("electronic_temperature", Value::Float(KELVIN))];
    all.extend_from_slice(extra);
    controls(&all)
}

fn offset(alpha: f64, beta: f64) -> Value {
    Value::Array(vec![Value::Float(alpha), Value::Float(beta)])
}

#[test]
fn grand_canonical_hardness_and_fukui_match_fractional_electron_fits() {
    let report = run("conceptual-dft", &lih(), &hot(&[])).unwrap();
    assert_eq!(
        report.get("route").and_then(Value::as_str),
        Some("grand_canonical")
    );
    assert!(number(&report, &["response_relative_residual"]) <= 1e-10);
    assert!(number(&report, &["response_number_defect"]) < 1e-10);
    let eta = number(&report, &["global", "hardness"]);
    let mu = number(&report, &["global", "chemical_potential"]);
    let fukui = list(&report, "fukui");
    assert!(
        (fukui.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "Fukui normalization"
    );
    // The reference Fermi level is the exact dA/dN: compare with the free
    // energy of fractional ensembles (adding h/2 electrons per spin).
    let state = |h: f64| {
        run(
            "free-energy",
            &lih(),
            &hot(&[("electron_offset", offset(0.5 * h, 0.5 * h))]),
        )
        .unwrap()
    };
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut mu_errors = Vec::new();
    let mut eta_errors = Vec::new();
    let mut fukui_errors = Vec::new();
    for &h in &steps {
        let plus = state(h);
        let minus = state(-h);
        let a = |r: &Value| number(r, &["free_energy"]);
        let mu_fd = (a(&plus) - a(&minus)) / (2.0 * h);
        mu_errors.push((mu_fd - mu).abs());
        let level = |r: &Value| {
            let v = list(r, "chemical_potentials");
            0.5 * (v[0] + v[1])
        };
        eta_errors.push(((level(&plus) - level(&minus)) / (2.0 * h) - eta).abs());
        let charges = |shift: f64| {
            let r = run(
                "charges",
                &lih(),
                &hot(&[("electron_offset", offset(0.5 * shift, 0.5 * shift))]),
            )
            .unwrap();
            list(&r, "mulliken")
        };
        let (qp, qm) = (charges(h), charges(-h));
        // Population = Z - q, so d(population)/dN = -dq/dN.
        fukui_errors.push((-(qp[1] - qm[1]) / (2.0 * h) - fukui[1]).abs());
    }
    fit("mu = dA/dN", &steps, &mu_errors);
    fit("eta = d mu/dN", &steps, &eta_errors);
    fit("fukui(H)", &steps, &fukui_errors);
}

#[test]
fn unrestricted_hardness_matrix_is_symmetric_and_matches_spin_resolved_fits() {
    // The doublet cation has distinct alpha and beta channels, so the
    // Maxwell relation d mu_alpha / d N_beta = d mu_beta / d N_alpha is a
    // genuine test rather than a consequence of spin symmetry.
    let cation = [("charge", Value::Int(1)), ("multiplicity", Value::Int(2))];
    let report = run("spin-conceptual-dft", &lih(), &hot(&cation)).unwrap();
    let matrix = report
        .get("hardness_matrix_per_spin")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    let defect = number(&report, &["hardness_matrix_symmetry_defect"]);
    println!("hardness matrix {matrix:?} raw symmetry defect {defect:e}");
    assert!(
        defect < 1e-7 * matrix[0].abs(),
        "Maxwell relation defect {defect:e}"
    );
    let ns = number(&report, &["charge_spin_coupling"]);
    let sn = number(&report, &["charge_spin_coupling_transpose"]);
    assert!((ns - sn).abs() < 1e-8 * ns.abs().max(1e-3), "{ns} vs {sn}");
    // Cross-spin response: d mu_sigma / d N_alpha of the closed-shell
    // unrestricted ensemble, against alpha-only fractional ensembles. (The
    // cation's single-electron beta channel is effectively gapped at this
    // temperature, so its Fermi level is not a smooth function of N on any
    // practical step; the closed-shell channels both carry Fermi weight.)
    let closed = run("spin-conceptual-dft", &lih(), &hot(&[])).unwrap();
    let matrix = closed
        .get("hardness_matrix_per_spin")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    let unrestricted = |h: f64| {
        run(
            "free-energy",
            &lih(),
            &hot(&[
                ("unrestricted", Value::Bool(true)),
                ("electron_offset", offset(h, 0.0)),
            ]),
        )
        .unwrap()
    };
    // The interval starting at 0.04 is pre-asymptotic for this channel (the
    // quartic term still moves the step-halving ratio from 3.90 to 3.99);
    // the fit is taken on the resolved interval 0.02..0.0025.
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for (row, label) in [
        (0usize, "d mu_alpha / d N_alpha"),
        (1, "d mu_beta / d N_alpha"),
    ] {
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| {
                let level = |r: &Value| list(r, "chemical_potentials")[row];
                ((level(&unrestricted(h)) - level(&unrestricted(-h))) / (2.0 * h) - matrix[row * 2])
                    .abs()
            })
            .collect();
        fit(label, &steps, &errors);
    }
}

#[test]
fn integer_and_koopmans_routes_obey_their_definitions_and_sum_rules() {
    let tight = controls(&[]);
    let report = run("conceptual-dft", &lih(), &tight).unwrap();
    assert_eq!(report.get("route").and_then(Value::as_str), Some("integer"));
    let e0 = number(&report, &["energy"]);
    let cation = run(
        "energy",
        &lih(),
        &controls(&[("charge", Value::Int(1)), ("multiplicity", Value::Int(2))]),
    )
    .unwrap();
    let ip = number(&cation, &["energy"]) - e0;
    assert!((number(&report, &["ionization_energy"]) - ip).abs() < 1e-10);
    for key in ["fukui_plus", "fukui_minus"] {
        let f = list(&report, key);
        assert!(
            (f.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "{key} sums to one electron"
        );
    }
    assert!(list(&report, "dual_descriptor").iter().sum::<f64>().abs() < 1e-9);
    let mu = number(&report, &["global", "chemical_potential"]);
    let ea = number(&report, &["electron_affinity"]);
    assert!((mu + 0.5 * (ip + ea)).abs() < 1e-12);

    let koopmans = run(
        "conceptual-dft",
        &lih(),
        &controls(&[("route", Value::Str("koopmans".into()))]),
    )
    .unwrap();
    let orbitals = run("orbitals", &lih(), &controls(&[])).unwrap();
    let homo = orbitals.get("channels").unwrap().as_array().unwrap()[0]
        .get("homo")
        .unwrap()
        .as_f64()
        .unwrap();
    assert!((number(&koopmans, &["ionization_energy"]) + homo).abs() < 1e-10);
    let f_plus = list(&koopmans, "fukui_plus");
    assert!((f_plus.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}
