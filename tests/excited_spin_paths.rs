// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete-component triplet nuclear-path contracts, independent of any
//! exact-spin claim for unrestricted or generalized references.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use stencils::Stencil;

fn molecule() -> System {
    System::molecule(
        vec![1; 4],
        vec![
            [0.13, 0.21, -0.07],
            [1.48, -0.19, 0.32],
            [-0.43, 1.71, 0.58],
            [0.36, -0.81, 2.05],
        ],
    )
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut values = vec![
        ("xc".into(), Value::Str("hf".into())),
        ("basis".into(), Value::Str("sto-3g".into())),
        ("charge".into(), Value::Int(2)),
        ("td_spin".into(), Value::Str("triplet".into())),
        ("nstates".into(), Value::Int(3)),
        ("conv_tol".into(), Value::Float(1e-13)),
        ("conv_tol_dp".into(), Value::Float(1e-12)),
        ("conv_tol_grad".into(), Value::Float(1e-11)),
        ("td_tolerance".into(), Value::Float(1e-11)),
        ("grid_weight_cutoff".into(), Value::Float(0.0)),
    ];
    values.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Controls::from_value(&Value::Object(values)).unwrap()
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn gradient(system: &System) -> Vec<f64> {
    vector(
        &run(
            "td-state-gradient",
            system,
            &controls(&[("td_gradient_route", Value::Str("path".into()))]),
        )
        .unwrap(),
        "gradient",
    )
}

fn energy(system: &System) -> f64 {
    let report = run("excitations", system, &controls(&[])).unwrap();
    let states = report.get("states").and_then(Value::as_array).unwrap();
    // Require an isolated root on every displaced scalar reference. The
    // small spatial singles matrix is solved densely by this scalar route.
    assert_eq!(states.len(), 3);
    assert!(number(&states[1], "energy") - number(&states[0], "energy") > 1e-3);
    number(&report, "energy") + number(&states[0], "energy")
}

#[test]
fn triplet_hf_gradient_and_hessian_have_complete_four_step_fits() {
    let system = molecule();
    let report = run("td-hessian", &system, &controls(&[])).unwrap();
    let analytic_gradient = vector(&report, "gradient");
    let analytic_hessian = vector(&report, "hessian");
    assert_eq!(analytic_gradient.len(), 12);
    assert_eq!(analytic_hessian.len(), 144);
    assert_eq!(
        report.get("provenance").and_then(Value::as_str),
        Some("analytic")
    );
    assert!(report
        .get("response_converged")
        .and_then(Value::as_bool)
        .unwrap());
    let mut failures = Vec::new();
    let stages = [
        ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
        ([0.04, 0.02, 0.01, 0.005], Stencil::Fourth),
    ];
    for column in 0..12 {
        let mut labels = vec![format!("triplet gradient {column}")];
        labels.extend((0..12).map(|row| format!("triplet Hessian {row},{column}")));
        stencils::fit_components_staged(
            &labels,
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 13];
                for (index, &h) in steps.iter().enumerate() {
                    let shifted = |step| {
                        let mut point = system.clone();
                        point.positions[column / 3][column % 3] += step;
                        point
                    };
                    let plus = shifted(h);
                    let minus = shifted(-h);
                    let gp = gradient(&plus);
                    let gm = gradient(&minus);
                    let ep = energy(&plus);
                    let em = energy(&minus);
                    let (gp2, gm2, ep2, em2) = if stencil == Stencil::Fourth {
                        let p = shifted(2.0 * h);
                        let m = shifted(-2.0 * h);
                        (gradient(&p), gradient(&m), energy(&p), energy(&m))
                    } else {
                        (vec![0.0; 12], vec![0.0; 12], 0.0, 0.0)
                    };
                    errors[0][index] =
                        (stencil.first(ep, em, ep2, em2, h) - analytic_gradient[column]).abs();
                    for row in 0..12 {
                        errors[row + 1][index] = (stencil
                            .first(gp[row], gm[row], gp2[row], gm2[row], h)
                            - analytic_hessian[12 * row + column])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("triplet HF nuclear derivatives", &failures);
}

#[test]
fn optimization_path_preserves_the_triplet_state_and_fresh_gradient() {
    let report = run(
        "excited-optimization",
        &molecule(),
        &controls(&[
            ("td_gradient_route", Value::Str("path".into())),
            ("max_steps", Value::Int(1)),
            ("gtol", Value::Float(1e-9)),
        ]),
    )
    .unwrap();
    assert_eq!(
        report.get("td_spin").and_then(Value::as_str),
        Some("triplet")
    );
    assert_eq!(
        report.get("td_gradient_route").and_then(Value::as_str),
        Some("path")
    );
    assert!(number(&report, "response_residual") < 1e-8);
    let points = report.get("positions").unwrap().numeric_array().unwrap().1;
    let mut endpoint = molecule();
    endpoint.positions = points.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let fresh = gradient(&endpoint);
    let saved = vector(&report, "gradient");
    for (a, b) in fresh.iter().zip(saved) {
        assert!((a - b).abs() < 1e-9);
    }
    assert!((energy(&endpoint) - number(&report, "excited_state_energy")).abs() < 1e-10);
    let energies = vector(&report, "trajectory_energies");
    assert_eq!(
        energies.len(),
        2,
        "one actual optimization step must be accepted"
    );
    assert!(energies[1] < energies[0]);
}
