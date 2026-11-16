// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common-driver mixed electric response, independently differentiated in
//! every Cartesian field coordinate at an asymmetric neutral periodic cell.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use std::collections::BTreeMap;
use stencils::Stencil;

fn fit(rank: usize) {
    let system = System {
        numbers: vec![3, 1],
        positions: vec![[0.17, 0.41, -0.19], [1.09, -0.82, 2.45]],
        cell: Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
        pbc: [true; 3],
    };
    let bias = [0.0013, -0.0021, 0.0017];
    let options = Value::object()
        .with("xc", "b2plyp")
        .with("reference", "rks")
        .with(
            "basis",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/periodic_lih.g94"),
        )
        .with("dispersion", "none")
        .with("k_mesh", [1, 1, 1])
        .with("overlap_cutoff", 5.5)
        .with("xc_ao_cutoff", 16.0)
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-12)
        .with("response_tolerance", 1e-12)
        .with("max_iter", 500);
    let features = [
        "periodic-efield",
        "dielectric-tensor",
        "periodic-beta",
        "periodic-gamma",
    ];
    let keys = [
        "polarization",
        "susceptibility",
        "second_susceptibility",
        "third_susceptibility",
    ];
    let calculate = |order: usize, field: [f64; 3]| {
        let mut input = options.clone().with("field", field);
        if order == 1 {
            input = input.with("response_order", 1);
        } else if order == 4 {
            input = input.with("use_symmetry", false);
        }
        let report = run(
            features[order - 1],
            &system,
            &Controls::from_value(&input).unwrap(),
        )
        .unwrap();
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        for key in [
            "response_residuals",
            "amplitude_response_residuals",
            "projector_residuals",
            "stationarity_residuals",
            "metric_residuals",
            "response_number_residuals",
        ] {
            let values = report.get(key).unwrap().numeric_array().unwrap().1;
            assert!(
                values.iter().all(|v| v.is_finite() && v.abs() < 1e-9),
                "{key}"
            );
        }
        report
            .get(keys[order - 1])
            .unwrap()
            .numeric_array()
            .unwrap()
            .1
    };
    let analytic = calculate(rank, bias);
    assert_eq!(analytic.len(), 3usize.pow(rank as u32));
    println!("correlated electric rank={rank} tensor={analytic:?}");
    let names: Vec<_> = (0..analytic.len())
        .map(|i| format!("correlated electric rank={rank} ordered component={i}"))
        .collect();
    let mut samples = BTreeMap::new();
    let mut failures = Vec::new();
    // Differentiate the lower response at independently converged biased
    // references. This checks all ordered mixed entries, not contractions.
    stencils::fit_components_staged(
        &names,
        &[
            ([0.002, 0.001, 0.0005, 0.00025], Stencil::Second),
            ([0.001, 0.0005, 0.00025, 0.000125], Stencil::Second),
            ([0.004, 0.003, 0.002, 0.001], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; analytic.len()];
            for axis in 0..3 {
                let mut sample = |h: f64| {
                    samples
                        .entry((axis, h.to_bits()))
                        .or_insert_with(|| {
                            let mut field = bias;
                            field[axis] += h;
                            calculate(rank - 1, field)
                        })
                        .clone()
                };
                for (s, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (sample(h), sample(-h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (sample(2.0 * h), sample(-2.0 * h))
                    } else {
                        (vec![0.0; plus.len()], vec![0.0; plus.len()])
                    };
                    for i in 0..plus.len() {
                        errors[3 * i + axis][s] = (stencil
                            .first(plus[i], minus[i], plus2[i], minus2[i], h)
                            - analytic[3 * i + axis])
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("correlated electric common-driver matrix", &failures);
}

#[test]
fn all_nine_linear_components() {
    fit(2);
}

#[test]
fn all_twenty_seven_quadratic_components() {
    fit(3);
}

#[test]
fn all_eighty_one_cubic_components() {
    fit(4);
}
