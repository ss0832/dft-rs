// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic nuclear derivatives of mixed-reference spin-flip states on the
//! common driver: every gradient component against central differences of
//! the state energy of the direct solution, and every Hessian element
//! against central differences of the analytic gradient, each by a
//! four-step fit (finite differences are validation only; a component that
//! fails on the first interval is recomputed on a finer and then a coarser
//! one), the agreement of the path series with the direct solution at the
//! reference geometry, and the translation sum rule.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use components::{assert_no_failures, fit_components};

use dft_rs::driver::{run, Controls, System, Value};

/// Methylene, bent and without symmetry, with Cartesian coordinate `coord`
/// (3 atom + axis) moved by `shift`.
fn methylene(coord: usize, shift: f64) -> System {
    let mut positions = vec![[0.0, 0.0, 0.1], [0.0, 1.87, -0.75], [0.3, -1.80, -0.80]];
    positions[coord / 3][coord % 3] += shift;
    System {
        numbers: vec![6, 1, 1],
        positions,
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("pbe0".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions (the partition-weight cutoff otherwise
        // adds noise near 1e-13 to displaced gradients; see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        ("multiplicity".to_string(), Value::Int(3)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

/// Total energy of the lowest singlet from the direct solution.
fn singlet_energy(system: &System) -> f64 {
    let report = run("mrsf", system, &controls(&[])).unwrap();
    report.get("singlets").and_then(Value::as_array).unwrap()[0]
        .get("energy")
        .and_then(Value::as_f64)
        .unwrap()
}

/// Gradient of the lowest singlet from the analytic route, flattened.
fn state_gradient(system: &System) -> Vec<f64> {
    values(
        &run("mrsf-state", system, &controls(&[])).unwrap(),
        "gradient",
    )
}

#[test]
fn gradient_and_hessian_pass_four_step_fits_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.04, 0.02, 0.01, 0.005],
        [0.02, 0.01, 0.005, 0.0025],
        [0.08, 0.04, 0.02, 0.01],
    ];
    const AXES: [&str; 3] = ["x", "y", "z"];
    let reference = methylene(0, 0.0);
    let gradient = state_gradient(&reference);
    let hessian = values(
        &run("mrsf-hessian", &reference, &controls(&[])).unwrap(),
        "hessian",
    );
    assert_eq!((gradient.len(), hessian.len()), (9, 81));
    println!("spin-flip state gradient {gradient:?}");
    println!("spin-flip state Hessian {hessian:?}");
    let mut failures = Vec::new();
    for coord in 0..9 {
        let mut names = vec![format!("gradient dE/d{}{}", AXES[coord % 3], coord / 3)];
        names.extend((0..9).map(|r| format!("Hessian H[{r}][{coord}]")));
        fit_components(
            &names,
            &INTERVALS,
            |steps| {
                let mut errors = vec![[0.0; 4]; 10];
                for (s, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (methylene(coord, h), methylene(coord, -h));
                    let energy = (singlet_energy(&plus) - singlet_energy(&minus)) / (2.0 * h);
                    errors[0][s] = (energy - gradient[coord]).abs();
                    let (gp, gm) = (state_gradient(&plus), state_gradient(&minus));
                    for r in 0..9 {
                        errors[1 + r][s] =
                            ((gp[r] - gm[r]) / (2.0 * h) - hessian[9 * r + coord]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    assert_no_failures("spin-flip state gradient and Hessian", &failures);
}

#[test]
fn path_series_starts_from_the_direct_solution_and_obeys_the_translation_sum_rule() {
    for label in ["S0", "S1", "T0"] {
        let report = run(
            "mrsf-state",
            &methylene(0, 0.0),
            &controls(&[("mrsf_state", Value::Str(label.into()))]),
        )
        .unwrap();
        let error = report
            .get("omega_reference_error")
            .and_then(Value::as_f64)
            .unwrap();
        let gradient = values(&report, "gradient");
        let sum = (0..3).fold(0.0_f64, |m, axis| {
            m.max(
                (0..3)
                    .map(|atom| gradient[3 * atom + axis])
                    .sum::<f64>()
                    .abs(),
            )
        });
        println!("{label}: series against direct solution {error:.3e}, translation sum {sum:.3e}");
        assert!(error < 1e-8);
        assert!(sum < 1e-6);
    }
}
