// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic nuclear gradients and Hessians of CASSCF and MC-PDFT energies
//! from nuclear Taylor paths with the complete orbital and CI response.
//!
//! Oracles (finite differences are validation only): independent four-step
//! fits of central differences of reconverged energies against the
//! analytic gradients, and of central differences of analytic gradients
//! against Hessian columns; the translational sum rules of gradients and
//! Hessians. Lithium hydride with the sigma pair active has a unique
//! active-space solution at every displaced geometry; the orbital gradient
//! of every displaced reference is converged to 1e-12 because the PDFT
//! energy follows the orbitals linearly.
//!
//! The translated functionals are not smooth functions of the geometry:
//! the translated polarization has a kink at R = 1, and the fully
//! translated gradient-corrected form carries the first derivative of the
//! joined polarization through its gradient terms, so grid points that
//! cross the joints make finite differences of open-shell-like densities
//! irregular (the first run fitted p = 1.80, R^2 = 0.917 for translated PBE
//! gradients and p = -0.49 for a fully translated PBE Hessian column over
//! [0.02, 0.0025]). The four-step fits therefore use smooth cases that
//! cover every term: the fully translated local functional with the active
//! orbital and CI response (one state, two averaged states, hybrid), and
//! the fully translated gradient-corrected functional of a closed-shell
//! determinant, whose on-top ratio R = 1 lies inside the polynomial joint.
//! The open active space with gradient-corrected parents is checked against
//! extrapolated small-step differences.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

fn lithium_hydride(atom: usize, axis: usize, shift: f64) -> System {
    let mut positions = vec![[0.0, 0.0, 0.0], [0.15, -0.1, 3.015]];
    positions[atom][axis] += shift;
    System::molecule(vec![3, 1], positions)
}

fn controls(extra: &[(&'static str, Value)]) -> Controls {
    let mut pairs = vec![
        ("basis", Value::Str("sto-3g".into())),
        ("xc", Value::Str("hf".into())),
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-11)),
        ("cas", Value::Array(vec![Value::Int(2), Value::Int(2)])),
        ("casscf_gradient_tol", Value::Float(1e-12)),
        ("casscf_energy_tol", Value::Float(1e-14)),
        ("ci_tol", Value::Float(1e-14)),
    ];
    for (k, v) in extra {
        pairs.retain(|(key, _)| key != k);
        pairs.push((k, v.clone()));
    }
    Controls::from_value(&Value::Object(
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
    ))
    .unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

const STEPS: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];
/// Candidate interval of the small off-diagonal Hessian element: over
/// STEPS its last error reached the 1e-9 noise floor of the analytic
/// gradients (local fully translated functional, (3,5) = 0.0042: errors
/// [3.143e-7, 7.554e-8, 2.244e-8, 6.948e-9], p = 1.825, R^2 = 0.9978).
/// The wider interval also failed, p = 1.975, R^2 = 0.998346. Both failed
/// runs used a 1e-10 orbital tolerance. The new run tightens the reference
/// to 1e-12 and keeps the same fit bounds; print every sampled error.
/// Tightening alone leaves those errors unchanged. Refining the local
/// on-top quadrature to level 6 resolves the wide interval (p = 1.924922,
/// R^2 = 0.999179); the measured small-step failures remain documented in
/// docs/response.md. Translation joints can spoil finite differences on a
/// coarse grid even when the wavefunction response has converged.
const WIDE: [f64; 4] = [0.04, 0.02, 0.01, 0.005];

fn fit(label: &str, analytic: f64, errors: &[f64]) {
    fit_over(label, analytic, &STEPS, errors);
}

fn fit_over(label: &str, analytic: f64, steps: &[f64], errors: &[f64]) {
    let fit = convergence::error_power_fit(steps, errors);
    println!(
        "{label}: analytic {analytic:.10} steps {steps:?} errors {errors:?} p={:.9} R2={:.12} C={:.12e} log_rmse={:.12e}",
        fit.exponent, fit.r_squared, fit.prefactor, fit.log_rmse
    );
    assert!(
        (fit.exponent - 2.0).abs() < 0.1,
        "{label}: p = {}",
        fit.exponent
    );
    assert!(fit.r_squared > 0.999, "{label}: R2 = {}", fit.r_squared);
}

/// Gradient components against energies and the translational sum rule.
fn check_gradients(
    feature: &str,
    extra: &[(&'static str, Value)],
    energy_key: &str,
    gradient_key: &str,
) {
    let mut request = extra.to_vec();
    request.push(("derivative_order", Value::Int(1)));
    let report = run(feature, &lithium_hydride(0, 0, 0.0), &controls(&request)).unwrap();
    let gradients = flat(&report, gradient_key);
    let states = gradients.len() / 6;
    for state in 0..states {
        let g = &gradients[6 * state..6 * state + 6];
        let sum: Vec<f64> = (0..3).map(|x| g[x] + g[3 + x]).collect();
        let largest = sum.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        println!("{feature} {energy_key} state {state}: gradient {g:?}, translation {largest:.3e}");
        assert!(largest < 1e-7);
        for component in [5usize, 3, 2] {
            let (atom, axis) = (component / 3, component % 3);
            let energy = |h: f64| {
                flat(
                    &run(feature, &lithium_hydride(atom, axis, h), &controls(extra)).unwrap(),
                    energy_key,
                )[state]
            };
            let errors: Vec<f64> = STEPS
                .iter()
                .map(|&h| ((energy(h) - energy(-h)) / (2.0 * h) - g[component]).abs())
                .collect();
            fit(
                &format!("{feature} {energy_key} state {state} component {component}"),
                g[component],
                &errors,
            );
        }
    }
}

/// A Hessian column against analytic gradients and the sum rules.
fn check_hessian(
    feature: &str,
    extra: &[(&'static str, Value)],
    gradient_key: &str,
    hessian_key: &str,
) {
    let mut second = extra.to_vec();
    second.push(("derivative_order", Value::Int(2)));
    let report = run(feature, &lithium_hydride(0, 0, 0.0), &controls(&second)).unwrap();
    println!(
        "{feature} reference gradient={:.12e} path_stationarity={:.12e} solve_residual={:.12e}",
        report.get("orbital_gradient").unwrap().as_f64().unwrap(),
        report
            .get("derivative_stationarity")
            .unwrap()
            .as_f64()
            .unwrap(),
        report
            .get("derivative_solve_residual")
            .unwrap()
            .as_f64()
            .unwrap(),
    );
    let hessian = flat(&report, hessian_key);
    let mut first = extra.to_vec();
    first.push(("derivative_order", Value::Int(1)));
    let column = 5usize;
    let gradient = |h: f64| {
        flat(
            &run(feature, &lithium_hydride(1, 2, h), &controls(&first)).unwrap(),
            gradient_key,
        )
    };
    let all = [0.04, 0.02, 0.01, 0.005, 0.0025];
    let samples: Vec<(f64, Vec<f64>, Vec<f64>)> = all
        .iter()
        .map(|&h| (h, gradient(h), gradient(-h)))
        .collect();
    // Large elements over the standard interval; the small off-diagonal
    // element over the resolved interval.
    for (row, steps) in [(5usize, &STEPS), (2, &STEPS), (3, &WIDE)] {
        let analytic = hessian[6 * row + column];
        let all_errors: Vec<f64> = samples
            .iter()
            .map(|(h, plus, minus)| ((plus[row] - minus[row]) / (2.0 * h) - analytic).abs())
            .collect();
        println!(
            "{feature} {hessian_key} ({row},{column}) all steps {all:?} errors {all_errors:?}"
        );
        let errors: Vec<f64> = steps
            .iter()
            .map(|h| {
                let (_, plus, minus) = samples.iter().find(|(s, _, _)| s == h).unwrap();
                ((plus[row] - minus[row]) / (2.0 * h) - analytic).abs()
            })
            .collect();
        fit_over(
            &format!("{feature} {hessian_key} ({row},{column})"),
            analytic,
            steps,
            &errors,
        );
    }
    let mut largest = 0.0_f64;
    for i in 0..6 {
        let sum: f64 = (0..2).map(|a| hessian[6 * i + 3 * a + i % 3]).sum::<f64>();
        largest = largest.max(sum.abs());
        for j in 0..6 {
            assert!((hessian[6 * i + j] - hessian[6 * j + i]).abs() < 1e-12);
        }
    }
    println!("{feature} {hessian_key}: translational sum rule {largest:.3e}");
    assert!(largest < 1e-6);
}

#[test]
fn casscf_gradient_and_hessian_pass_four_step_fits() {
    check_gradients("casscf", &[], "cas_energies", "gradients");
    check_hessian("casscf", &[], "gradients", "hessians");
}

fn ints(values: &[i64]) -> Value {
    Value::Array(values.iter().map(|v| Value::Int(*v)).collect())
}

#[test]
fn mcpdft_gradients_pass_four_step_fits() {
    let local = [("on_top_functional", Value::Str("ftsvwn5".into()))];
    check_gradients("mcpdft", &local, "pdft_energies", "gradients");
    let hybrid = [
        ("on_top_functional", Value::Str("ftsvwn5".into())),
        ("pdft_lambda", Value::Float(0.25)),
    ];
    check_gradients("mcpdft", &hybrid, "pdft_energies", "gradients");
    let closed = [
        ("on_top_functional", Value::Str("ftpbe".into())),
        ("cas", ints(&[0, 0])),
    ];
    check_gradients("mcpdft", &closed, "pdft_energies", "gradients");
}

#[test]
fn mcpdft_hessians_pass_four_step_fits() {
    let local = [
        ("on_top_functional", Value::Str("ftsvwn5".into())),
        ("grid_level", Value::Int(6)),
    ];
    check_hessian("mcpdft", &local, "gradients", "hessians");
    let closed = [
        ("on_top_functional", Value::Str("ftpbe".into())),
        ("cas", ints(&[0, 0])),
    ];
    check_hessian("mcpdft", &closed, "gradients", "hessians");
}

#[test]
fn gradient_corrected_open_active_space_approaches_small_step_differences() {
    // Richardson extrapolation of the two smallest central differences of
    // reconverged energies; and small-step differences of analytic
    // gradients for the Hessian.
    for functional in ["ftpbe", "tpbe"] {
        let extra = [("on_top_functional", Value::Str(functional.into()))];
        let mut first = extra.to_vec();
        first.push(("derivative_order", Value::Int(1)));
        let analytic = flat(
            &run("mcpdft", &lithium_hydride(0, 0, 0.0), &controls(&first)).unwrap(),
            "gradients",
        )[5];
        let energy = |h: f64| {
            flat(
                &run("mcpdft", &lithium_hydride(1, 2, h), &controls(&extra)).unwrap(),
                "pdft_energies",
            )[0]
        };
        let central = |h: f64| (energy(h) - energy(-h)) / (2.0 * h);
        let (a, b) = (central(0.0025), central(0.00125));
        let extrapolated = (4.0 * b - a) / 3.0;
        println!("{functional}: analytic gradient {analytic:.12}, extrapolated differences {extrapolated:.12}, defect {:.3e}", analytic - extrapolated);
        assert!((analytic - extrapolated).abs() < 2e-7);
    }
    let mut second = vec![
        ("on_top_functional", Value::Str("ftpbe".into())),
        ("derivative_order", Value::Int(2)),
    ];
    let analytic = flat(
        &run("mcpdft", &lithium_hydride(0, 0, 0.0), &controls(&second)).unwrap(),
        "hessians",
    )[6 * 5 + 5];
    second.pop();
    second.push(("derivative_order", Value::Int(1)));
    let gradient = |h: f64| {
        flat(
            &run("mcpdft", &lithium_hydride(1, 2, h), &controls(&second)).unwrap(),
            "gradients",
        )[5]
    };
    for h in [0.0005, 0.00025] {
        let defect = (gradient(h) - gradient(-h)) / (2.0 * h) - analytic;
        println!("ftpbe Hessian (5,5) {analytic:.12} at h = {h}: defect {defect:.3e}");
        assert!(defect.abs() < 1e-6);
    }
}

#[test]
fn state_averaged_gradients_of_both_states_pass_four_step_fits() {
    let extra = [
        ("on_top_functional", Value::Str("ftsvwn5".into())),
        ("cas_roots", Value::Int(2)),
    ];
    // The state-specific PDFT energies of the averaged reference are
    // reported in reference-state order as state_total.
    check_gradients("mcpdft", &extra, "state_total", "gradients");
    check_gradients("mcpdft", &extra, "cas_energies", "cas_gradients");
}
