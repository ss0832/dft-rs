// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance of the excited-state geometry relaxation and of the
//! integral-direct gradient on the common driver.
//!
//! `excited-optimization`: the relaxed first singlet Tamm-Dancoff state of
//! formaldehyde has a non-increasing trajectory and a gradient below the
//! requested tolerance; the separate `td-state-gradient` feature gives the
//! same energy and gradient at the optimum; central differences of the
//! excited-state energy at the optimum pass an independent four-step fit
//! against the reported gradient (finite differences are validation only),
//! and the diagonal curvature there is positive.
//!
//! `streamed-gradient`: equal to the stored-integral `gradient` feature,
//! translation invariant, and the independent four-step fit of central
//! energy differences.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};

/// Formaldehyde (Bohr), slightly pyramidal so that the relaxation is not
/// held on the planar saddle of the n -> pi* state by symmetry.
fn formaldehyde() -> System {
    System::molecule(
        vec![6, 8, 1, 1],
        vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.28],
            [0.15, 1.77, -1.10],
            [0.15, -1.77, -1.10],
        ],
    )
}

/// Water distorted away from C2v (Bohr).
fn water() -> System {
    System::molecule(
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.23],
            [0.0, 1.4309, -0.8867],
            [0.12, -1.4809, -0.9067],
        ],
    )
}

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

fn tight(xc: &str) -> Vec<(&'static str, Value)> {
    vec![
        ("basis", Value::Str("sto-3g".into())),
        ("xc", Value::Str(xc.into())),
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-12)),
    ]
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn states(report: &Value) -> &[Value] {
    report.get("states").unwrap().as_array().unwrap()
}

fn amplitude_request(xc: &str) -> Vec<(&'static str, Value)> {
    let mut request = tight(xc);
    request.push(("nstates", Value::Int(4)));
    request.push(("td_method", Value::Str("tda".into())));
    request.push(("amplitudes", Value::Bool(true)));
    request
}

/// Phase-independent similarity of two amplitude sets.
fn amplitude_similarity(a: &Value, b: &Value) -> f64 {
    let x = vector(a, "amplitudes");
    let y = vector(b, "amplitudes");
    x.iter().zip(&y).map(|(p, q)| (p * q).abs()).sum()
}

/// Total energy of the excited state that matches `reference` at `system`.
fn followed_energy(system: &System, xc: &str, reference: &Value) -> f64 {
    let report = run("excitations", system, &controls(&amplitude_request(xc))).unwrap();
    let best = states(&report)
        .iter()
        .max_by(|u, v| {
            amplitude_similarity(reference, u)
                .partial_cmp(&amplitude_similarity(reference, v))
                .unwrap()
        })
        .unwrap();
    let similarity = amplitude_similarity(reference, best);
    assert!(
        similarity > 0.9,
        "root following lost the state: {similarity}"
    );
    number(&report, "energy") + number(best, "energy")
}

fn at(positions: &[f64], numbers: &[u8]) -> System {
    System::molecule(
        numbers.to_vec(),
        positions.chunks(3).map(|r| [r[0], r[1], r[2]]).collect(),
    )
}

#[test]
fn relaxed_excited_state_is_a_minimum_of_the_followed_state() {
    let xc = "hf";
    let gtol = 1e-5;
    let mut request = tight(xc);
    request.push(("root", Value::Int(1)));
    request.push(("nstates", Value::Int(4)));
    request.push(("gtol", Value::Float(gtol)));
    request.push(("max_steps", Value::Int(200)));
    let start = formaldehyde();
    let report = run("excited-optimization", &start, &controls(&request)).unwrap();
    assert_eq!(
        report
            .get("optimization_converged")
            .and_then(Value::as_bool),
        Some(true)
    );
    let energies = vector(&report, "trajectory_energies");
    for pair in energies.windows(2) {
        assert!(pair[1] <= pair[0] + 1e-12, "energy rose: {pair:?}");
    }
    let gradient = vector(&report, "gradient");
    let largest = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    println!(
        "excited-state relaxation: {} iterations, energy {:.10} -> {:.10}, gradient max {largest:.3e}",
        number(&report, "iterations"),
        energies[0],
        energies[energies.len() - 1]
    );
    assert!(largest < gtol);
    assert!(energies[energies.len() - 1] < energies[0]);

    // The separate excited-state gradient feature at the optimum.
    let optimum = at(&vector(&report, "positions"), &start.numbers);
    let mut single = tight(xc);
    single.push(("root", Value::Int(1)));
    single.push(("nstates", Value::Int(4)));
    let direct = run("td-state-gradient", &optimum, &controls(&single)).unwrap();
    let energy_difference =
        (number(&direct, "excited_state_energy") - number(&report, "excited_state_energy")).abs();
    let gradient_difference = vector(&direct, "gradient")
        .iter()
        .zip(&gradient)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "against td-state-gradient: energy {energy_difference:.3e}, gradient {gradient_difference:.3e}"
    );
    assert!(energy_difference < 1e-9);
    assert!(gradient_difference < 1e-8);

    // Independent four-step fit and positive diagonal curvature at the optimum.
    let unperturbed = run("excitations", &optimum, &controls(&amplitude_request(xc))).unwrap();
    let state = states(&unperturbed)[0].clone();
    let centre = followed_energy(&optimum, xc, &state);
    assert!((centre - number(&report, "excited_state_energy")).abs() < 1e-9);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    // Oxygen z and the out-of-plane coordinate of the first hydrogen.
    for component in [5usize, 6] {
        let (atom, axis) = (component / 3, component % 3);
        let energy_at = |h: f64| {
            let mut system = optimum.clone();
            system.positions[atom][axis] += h;
            followed_energy(&system, xc, &state)
        };
        let samples: Vec<(f64, f64)> = steps
            .iter()
            .map(|&h| (energy_at(h), energy_at(-h)))
            .collect();
        let errors: Vec<f64> = steps
            .iter()
            .zip(&samples)
            .map(|(&h, (plus, minus))| ((plus - minus) / (2.0 * h) - gradient[component]).abs())
            .collect();
        let curvature = (samples[0].0 + samples[0].1 - 2.0 * centre) / (steps[0] * steps[0]);
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "optimum component {component}: analytic {:.3e} curvature {curvature:.6} steps \
             {steps:?} errors {errors:?} p={:.6} R2={:.8}",
            gradient[component], fit.exponent, fit.r_squared
        );
        assert!(curvature > 0.0);
        assert!((fit.exponent - 2.0).abs() < 0.05, "p = {}", fit.exponent);
        assert!(fit.r_squared > 0.9999, "R2 = {}", fit.r_squared);
    }
}

#[test]
fn integral_direct_gradient_is_the_stored_integral_gradient() {
    use stencils::Stencil;
    // Retain the original water case, and add a genuinely C1 four-atom
    // geometry. Fit every Cartesian entry for both reference functionals.
    let ammonia = System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.17, -0.11, 0.23],
            [1.43, 0.81, -0.62],
            [-0.63, 1.59, -0.43],
            [-1.12, -1.36, -0.71],
        ],
    );
    for (label, system) in [("water", water()), ("C1 ammonia", ammonia)] {
        for xc in ["hf", "pbe"] {
            let mut request = tight(xc);
            request.push(("grid_weight_cutoff", Value::Float(0.0)));
            request.push(("conv_tol_grad", Value::Float(1e-11)));
            let streamed = run("streamed-gradient", &system, &controls(&request)).unwrap();
            let stored = run("gradient", &system, &controls(&request)).unwrap();
            assert_eq!(
                streamed.get("integral_storage").and_then(Value::as_str),
                Some("none")
            );
            let g = vector(&streamed, "gradient");
            let difference = g
                .iter()
                .zip(&vector(&stored, "gradient"))
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            let energy_difference = (number(&streamed, "energy") - number(&stored, "energy")).abs();
            let translation = number(&streamed, "translation_residual");
            println!(
            "{label} {xc}: streamed against stored gradient {difference:.3e}, energy {energy_difference:.3e}, \
             translation {translation:.3e}"
        );
            assert!(difference < 1e-9);
            assert!(energy_difference < 1e-10);
            assert!(translation < 1e-8);
            assert!(
                g.iter().all(|entry| entry.abs() > 1e-8),
                "zero Cartesian gradient entry"
            );
            let energy_at = |component: usize, h: f64| {
                let mut displaced = system.clone();
                displaced.positions[component / 3][component % 3] += h;
                number(
                    &run("energy", &displaced, &controls(&request)).unwrap(),
                    "energy",
                )
            };
            let names: Vec<_> = (0..g.len())
                .map(|k| format!("{label} {xc} streamed gradient component {k}"))
                .collect();
            let mut strict_passed = vec![false; g.len()];
            let mut failures = Vec::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
                    ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], Stencil::Fourth),
                ],
                |steps, stencil| {
                    (0..g.len()).map(|k| {
                    let errors = steps.map(|h| {
                        let (plus,minus) = (energy_at(k,h),energy_at(k,-h));
                        let (plus2,minus2) = if stencil == Stencil::Fourth {
                            (energy_at(k,2.0*h),energy_at(k,-2.0*h))
                        } else { (0.0,0.0) };
                        (stencil.first(plus,minus,plus2,minus2,h) - g[k]).abs()
                    });
                    if errors.iter().all(|e| e.is_finite() && *e > 0.0) {
                        let fit = convergence::error_power_fit(steps,&errors);
                        // Preserve the original tighter gates in addition to
                        // the shared staged component diagnostics.
                        let pass = (fit.exponent-stencil.order()).abs() < 0.05 && fit.r_squared > 0.9999;
                        strict_passed[k] |= pass;
                        println!("{} strict {:?}: steps {steps:?} errors {errors:?} p={:.6} R2={:.8} C={:.3e} log_rmse={:.3e} pass={pass}",
                            names[k],stencil,fit.exponent,fit.r_squared,fit.prefactor,fit.log_rmse);
                    }
                    errors
                }).collect()
                },
                &mut failures,
            );
            components::assert_no_failures("streamed gradient", &failures);
            assert!(
                strict_passed.iter().all(|pass| *pass),
                "original p/R2 gates failed: {strict_passed:?}"
            );
        }
    }
}
