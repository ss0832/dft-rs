// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Correlated alpha with second-order orbital and PT2 amplitude response.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::{run, Controls, System, Value};

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).unwrap().as_f64().unwrap()
}
fn projected(report: &Value, key: &str, direction: [f64; 3]) -> f64 {
    let (_, tensor) = report.get(key).unwrap().numeric_array().unwrap();
    (0..3)
        .map(|a| {
            (0..3)
                .map(|b| direction[a] * tensor[3 * a + b] * direction[b])
                .sum::<f64>()
        })
        .sum()
}
fn fit(label: &str, exact: f64, estimates: &[f64], steps: &[f64; 4]) {
    let errors: Vec<_> = estimates
        .iter()
        .map(|value| (value - exact).abs())
        .collect();
    let fit = convergence::error_power_fit(steps, &errors);
    let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
    println!("{label} value={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e}",fit.exponent,fit.r_squared);
    assert!(
        (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999,
        "{label}"
    );
    assert!(richardson < 1e-5, "{label}");
}

#[test]
fn water_correlated_alpha_includes_separate_ss_os_and_distinct_target_response() {
    for model in [
        "b2plyp",
        "dsd-pbep86",
        "dsd-blyp",
        "pwpb95",
        "wb97m2",
        "pbe",
        "tpss",
        "pbe0",
    ] {
        if std::env::var("DFT_RS_TEST_MODEL")
            .ok()
            .is_some_and(|selected| selected != model)
        {
            continue;
        }
        // The rounded local-correlation join is not globally differentiable.
        // Keep the original failed interval in the validation record and
        // resolve a smooth geometry and four-step interval without changing
        // either the published functional or the acceptance thresholds.
        let scale = if model == "dsd-pbep86" { 0.8 } else { 1.0 };
        let steps = if model == "dsd-pbep86" {
            [0.01, 0.005, 0.0025, 0.00125]
        } else {
            [0.04, 0.02, 0.01, 0.005]
        };
        let system = System::molecule(
            vec![8, 1, 1],
            vec![
                [0., 0., 0.],
                [1.43 * scale, 0., 1.10 * scale],
                [-1.43 * scale, 0., 1.10 * scale],
            ],
        );
        let controls = |field: f64| {
            let mut v = Value::object()
                .with("xc", model)
                .with("basis", "sto-3g")
                .with("grid_level", 0usize)
                .with("vv10_grid_level", 1usize)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-12)
                .with("conv_tol_dp", 1e-12)
                .with("field", [0., 0., field]);
            if ["pbe", "tpss", "pbe0"].contains(&model) {
                v.set("pt2_same_spin", 0.2);
                v.set("pt2_opposite_spin", 0.3);
            }
            Controls::from_value(&v).unwrap()
        };
        let mut alpha_controls = controls(0.0);
        alpha_controls
            .insert("response_tolerance", Value::Float(1e-12))
            .unwrap();
        let report = run("polarizability", &system, &alpha_controls).unwrap();
        assert_eq!(report.get("raw_asymmetry"), Some(&Value::Null));
        let centre = run("energy", &system, &controls(0.0)).unwrap();
        let mut estimates: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        for h in steps {
            let plus = run("energy", &system, &controls(h)).unwrap();
            let minus = run("energy", &system, &controls(-h)).unwrap();
            for (index, key) in ["energy", "pt2_same_spin_alpha", "pt2_opposite_spin"]
                .into_iter()
                .enumerate()
            {
                estimates[index].push(
                    -(number(&plus, key) - 2.0 * number(&centre, key) + number(&minus, key))
                        / (h * h),
                );
            }
        }
        for (index, key) in [
            "polarizability",
            "pt2_same_spin_alpha_polarizability",
            "pt2_opposite_spin_polarizability",
        ]
        .into_iter()
        .enumerate()
        {
            fit(
                &format!("water {model} {key}"),
                projected(&report, key, [0., 0., 1.]),
                &estimates[index],
                &steps,
            );
        }
        for key in [
            "response_residuals",
            "amplitude_response_residuals",
            "response_number_residuals",
            "projector_residuals",
            "stationarity_residuals",
            "orbital_frame_residuals",
        ] {
            let (_, values) = report.get(key).unwrap().numeric_array().unwrap();
            let raw = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            println!("water {model} {key} max={raw:.12e}");
            assert!(raw < 1e-9, "{model} {key}");
        }
    }
}

#[test]
fn biased_lithium_hydride_alpha_and_zero_pt2_limits() {
    let system = System::molecule(vec![3, 1], vec![[0., 0., 0.], [0.2, -0.1, 3.0]]);
    let direction = [0.2, -0.3, 0.7];
    let controls = |field: f64| {
        Controls::from_value(
            &Value::object()
                .with("xc", "b2plyp")
                .with("basis", "sto-3g")
                .with("grid_level", 0usize)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-12)
                .with("conv_tol_dp", 1e-12)
                .with("field", direction.map(|u| (field + 0.003) * u)),
        )
        .unwrap()
    };
    let mut alpha_controls = controls(0.0);
    alpha_controls
        .insert("response_tolerance", Value::Float(1e-12))
        .unwrap();
    let report = run("polarizability", &system, &alpha_controls).unwrap();
    let centre = number(&run("energy", &system, &controls(0.)).unwrap(), "energy");
    let steps = [0.01, 0.005, 0.0025, 0.00125];
    let estimates: Vec<_> = steps
        .iter()
        .map(|&h| {
            -(number(&run("energy", &system, &controls(h)).unwrap(), "energy") - 2.0 * centre
                + number(&run("energy", &system, &controls(-h)).unwrap(), "energy"))
                / (h * h)
        })
        .collect();
    fit(
        "biased LiH B2PLYP alpha contraction",
        projected(&report, "polarizability", direction),
        &estimates,
        &steps,
    );
    let hydrogen = System::molecule(vec![1, 1], vec![[0., 0., 0.], [0., 0., 1.4]]);
    let mut plain = Controls::from_value(
        &Value::object()
            .with("xc", "pbe0")
            .with("basis", "sto-3g")
            .with("grid_level", 0usize)
            .with("conv_tol", 1e-13)
            .with("conv_tol_grad", 1e-12)
            .with("conv_tol_dp", 1e-12)
            .with("response_tolerance", 1e-12),
    )
    .unwrap();
    let parent = run("polarizability", &hydrogen, &plain).unwrap();
    plain.insert("pt2_same_spin", Value::Float(0.0)).unwrap();
    plain
        .insert("pt2_opposite_spin", Value::Float(0.0))
        .unwrap();
    let zero = run("polarizability", &hydrogen, &plain).unwrap();
    let (_, expected) = parent
        .get("polarizability")
        .unwrap()
        .numeric_array()
        .unwrap();
    let (_, actual) = zero.get("polarizability").unwrap().numeric_array().unwrap();
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(a, b)| (a - b).abs() < 1e-10));
    for key in [
        "pt2_polarizability_correction",
        "pt2_same_spin_alpha_polarizability",
        "pt2_same_spin_beta_polarizability",
    ] {
        let (_, values) = zero.get(key).unwrap().numeric_array().unwrap();
        assert!(values.iter().all(|v| v.abs() < 1e-14));
    }
}
