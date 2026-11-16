// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every registered semilocal component is checked through the generic core.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::periodic::spinor_xc::{spinor_semilocal_point, SpinorXcPoint};
use dft_rs::xc::{
    resolve, FunctionalSpec, Kind, COMBINED_NAMES, CORRELATION_NAMES, EXCHANGE_NAMES,
};

#[test]
fn every_registered_semilocal_provider_has_a_four_step_energy_fit() {
    let point = SpinorXcPoint {
        density: [0.9, 0.1, 0.12, 0.2],
        gradient: [
            [0.1, 0.02, -0.04],
            [0.01, 0.02, 0.005],
            [0.001, -0.012, 0.002],
            [0.03, 0.01, 0.005],
        ],
        tau: [0.9, 0.02, 0.03, 0.04],
        current: [
            [0.02, -0.01, 0.03],
            [0.005, 0.001, 0.002],
            [-0.003, 0.004, 0.001],
            [0.002, 0.003, -0.005],
        ],
    };
    let specs: Vec<_> = COMBINED_NAMES
        .iter()
        .map(|name| FunctionalSpec::Combined((*name).into()))
        .chain(EXCHANGE_NAMES.iter().map(|name| FunctionalSpec::Split {
            x: (*name).into(),
            c: "none".into(),
        }))
        .chain(CORRELATION_NAMES.iter().map(|name| FunctionalSpec::Split {
            x: "slater".into(),
            c: (*name).into(),
        }))
        .collect();
    let direction: [f64; 32] = std::array::from_fn(|i| 0.1 * ((i + 1) as f64 * 0.37).sin());
    let values = point.descriptors();
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut checked = 0;
    let mut failures = Vec::new();
    for spec in specs {
        let mut functional = resolve(&spec).unwrap();
        if matches!(&spec, FunctionalSpec::Split { c, .. } if c != "none") {
            functional
                .parts
                .retain(|(_, part)| part.kind() == Kind::Correlation);
        }
        if functional.parts.is_empty() {
            continue;
        }
        let reference = spinor_semilocal_point(&functional, &point, 16).unwrap();
        let exact = reference
            .gradient
            .iter()
            .zip(direction)
            .map(|(v, d)| v * d)
            .sum::<f64>();
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let evaluate = |sign: f64| {
                spinor_semilocal_point(
                    &functional,
                    &SpinorXcPoint::from_descriptors(std::array::from_fn(|i| {
                        values[i] + sign * h * direction[i]
                    })),
                    16,
                )
                .unwrap()
                .energy
            };
            let estimate = (evaluate(1.0) - evaluate(-1.0)) / (2.0 * h);
            estimates.push(estimate);
            errors.push((estimate - exact).abs());
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
        let parts = (reference.energy - reference.component_energies.iter().sum::<f64>()).abs();
        println!(
            "spinor registry {:?}: semilocal-only nonlocal={:?} steps={steps:?} \
            errors={errors:?} p={:.9} R2={:.10} Richardson={rich:.3e} \
            parts={parts:.3e}",
            spec,
            functional.required_nonlocal_term(),
            fit.exponent,
            fit.r_squared
        );
        if (fit.exponent - 2.0).abs() >= 0.02
            || fit.r_squared <= 0.99999
            || rich >= 1e-8
            || parts >= 1e-12
        {
            failures.push(format!("{spec:?}"));
        }
        checked += 1;
    }
    println!("spinor registry checked={checked} failures={failures:?}");
    assert_eq!(
        checked,
        COMBINED_NAMES.len() - 1 + EXCHANGE_NAMES.len() + CORRELATION_NAMES.len()
    );
    assert!(failures.is_empty());
}
