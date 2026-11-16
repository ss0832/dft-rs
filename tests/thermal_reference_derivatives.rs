// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All Cartesian components of the finite-temperature gradient and Hessian.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};

fn property(system: &System, settings: &Value, feature: &str) -> Value {
    let report = run(feature, system, &Controls::from_value(settings).unwrap())
        .unwrap_or_else(|error| panic!("{feature} at {:?}: {error}", system.positions));
    assert_eq!(
        report.get("converged").and_then(Value::as_bool),
        Some(true),
        "{feature}: {}",
        report.to_json()
    );
    report
}

fn entries(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn every_gradient_and_hessian_component_has_four_step_fits() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
    );
    let dimension = 3 * system.numbers.len();
    let reference =
        std::env::var("DFT_RS_THERMAL_TEST_REFERENCE").unwrap_or_else(|_| "roks".into());
    let xc = std::env::var("DFT_RS_THERMAL_TEST_XC").unwrap_or_else(|_| "wb97x-d".into());
    let settings = Value::object()
        .with("reference", reference.as_str())
        .with("xc", xc.as_str())
        .with("basis", "sto-3g")
        .with("charge", 1)
        .with("multiplicity", 2)
        .with("electronic_temperature", 120000.0)
        .with("grid_level", 0)
        .with("vv10_grid_level", 0)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-12)
        .with("response_tolerance", 1e-9);
    let gradient = entries(&property(&system, &settings, "gradient"), "gradient");
    let hessian = entries(&property(&system, &settings, "hessian"), "hessian");
    assert_eq!(gradient.len(), dimension);
    assert_eq!(hessian.len(), dimension * dimension);
    let mut failures = Vec::new();
    for column in 0..dimension {
        let names = std::iter::once(format!("{reference}/{xc} gradient[{column}]"))
            .chain((0..dimension).map(|row| format!("{reference}/{xc} Hessian[{row},{column}]")))
            .collect::<Vec<_>>();
        let shifted = |step: f64| {
            let mut geometry = system.clone();
            geometry.positions[column / 3][column % 3] += step;
            let report = property(&geometry, &settings, "gradient");
            let mut f = vec![report.get("energy").and_then(Value::as_f64).unwrap()];
            f.extend(entries(&report, "gradient"));
            f
        };
        stencils::fit_components_staged(
            &names,
            &[
                ([0.06, 0.04, 0.025, 0.016], stencils::Stencil::Second),
                ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
                // Components with competing quadratic and quartic terms
                // must recover the asymptotic slope on a smaller interval.
                ([0.004, 0.0028, 0.002, 0.0014], stencils::Stencil::Second),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; names.len()];
                for (k, &h) in steps.iter().enumerate() {
                    let plus = shifted(h);
                    let minus = shifted(-h);
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (shifted(2.0 * h), shifted(-2.0 * h))
                    } else {
                        (plus.clone(), minus.clone())
                    };
                    for (i, error) in errors.iter_mut().enumerate() {
                        let exact = if i == 0 {
                            gradient[column]
                        } else {
                            hessian[(i - 1) * dimension + column]
                        };
                        error[k] = (stencil.first(plus[i], minus[i], plus2[i], minus2[i], h)
                            - exact)
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures(
        "thermal molecular gradient and ordered Cartesian Hessian",
        &failures,
    );
}
