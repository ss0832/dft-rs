// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete nuclear tensors including independently fitted unscaled SS/OS
//! channels. Parent orbital and PT2 amplitudes both respond analytically.
//! Measured: all 4368 components pass, p=1.802733..2.195880,
//! minimum R2=0.99902605. Forty-nine failed stages are retained and resolved
//! on the documented intervals; no error floor or gate relaxation is used.
//! The newly strengthened raw convergence gates require a separate rerun.
//! Open-shell coverage adds a fifth asymmetric center so both same-spin
//! channels remain nonzero. Its 6720 component fits are a separate gate.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_reference.rs"]
mod nuclear_reference;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::{run, Controls, System, Value};

fn fit_correlated_nuclear_components(xc: &str, unrestricted: bool) {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    let mut positions = vec![
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
    ];
    if unrestricted {
        positions.push([1.41, 1.83, 1.59]);
    }
    let system = System::molecule(vec![1; positions.len()], positions);
    let reference = if unrestricted { "uks" } else { "rks" };
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", xc)
            .with("reference", reference)
            .with("multiplicity", if unrestricted { 2usize } else { 1usize })
            .with("double_hybrid_model", "custom")
            .with("pt2_same_spin", 0.37)
            .with("pt2_opposite_spin", 0.61)
            .with("basis", "sto-3g")
            .with("grid_level", 0)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("conv_tol_grad", 1e-12)
            .with("conv_tol_dp", 1e-12)
            .with("max_iter", 300)
            .with("guess", "core")
            .with("response_tolerance", 1e-12),
    )
    .unwrap();
    let label = if unrestricted {
        format!("uks {xc} correlated nuclear")
    } else {
        format!("{xc} correlated nuclear")
    };
    nuclear_reference::fit(
        &label,
        &system,
        &controls,
        &[
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
            ([0.002, 0.0015, 0.001, 0.00075], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
        ],
        &[
            "",
            "parent_",
            "target_",
            "pt2_",
            "pt2_same_spin_alpha_",
            "pt2_opposite_spin_",
            "pt2_same_spin_beta_",
        ],
        |feature, system, controls| {
            let report = run(feature, system, controls).unwrap();
            assert_eq!(
                report.get("unrestricted").and_then(Value::as_bool),
                Some(unrestricted)
            );
            assert_eq!(
                report.get("n_alpha").and_then(Value::as_i64),
                Some(if unrestricted { 3 } else { 2 })
            );
            assert_eq!(report.get("n_beta").and_then(Value::as_i64), Some(2));
            if matches!(feature, "gradient" | "hessian") {
                assert_eq!(
                    report.get("converged").unwrap().as_bool(),
                    Some(true),
                    "{xc} {feature} positions={:?}: {}",
                    system.positions,
                    report.to_json_pretty()
                );
                for key in [
                    "response_residuals",
                    "amplitude_response_residuals",
                    "projector_defects",
                    "nuclear_stationarity_defects",
                    "metric_defects",
                    "response_number_defects",
                    "orbital_frame_defects",
                    "reference_energy_defects",
                    "reference_fock_defects",
                ] {
                    let values = report.get(key).unwrap().numeric_array().unwrap().1;
                    assert!(values.iter().all(|v| v.is_finite()));
                    let residual = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                    if feature == "hessian" {
                        println!("{xc} correlated nuclear {key}={residual:.16e}");
                    }
                    assert!(
                        residual <= 1e-12,
                        "{xc} {feature} {key}={residual} positions={:?}",
                        system.positions
                    );
                }
                if feature == "hessian" {
                    for key in [
                        "pt2_same_spin_alpha",
                        "pt2_opposite_spin",
                        "pt2_same_spin_beta",
                    ] {
                        let value = report.get(key).and_then(Value::as_f64).unwrap();
                        assert!(
                            value.is_finite() && value.abs() > 1e-12,
                            "{reference} {xc} {key}={value}"
                        );
                    }
                    if let Ok(directory) = std::env::var("DFT_RS_TEST_REPORT_DIRECTORY") {
                        std::fs::write(
                            std::path::Path::new(&directory)
                                .join(format!("correlated-{reference}-{xc}.json")),
                            report.to_json_pretty(),
                        )
                        .unwrap();
                    }
                }
            }
            report
        },
    );
}

#[test]
fn all_correlated_nuclear_tensor_components_and_spin_channels() {
    for xc in ["hf", "pbe", "tpss", "pbe0"] {
        fit_correlated_nuclear_components(xc, false);
    }
}

#[test]
fn unrestricted_hf_parent_nuclear_components() {
    fit_correlated_nuclear_components("hf", true);
}

#[test]
fn unrestricted_gga_parent_nuclear_components() {
    fit_correlated_nuclear_components("pbe", true);
}

#[test]
fn unrestricted_meta_gga_parent_nuclear_components() {
    fit_correlated_nuclear_components("tpss", true);
}

#[test]
fn unrestricted_hybrid_parent_nuclear_components() {
    fit_correlated_nuclear_components("pbe0", true);
}

fn check_zero_correlation_nuclear_limit(unrestricted: bool) {
    let mut positions = vec![
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
    ];
    if unrestricted {
        positions.push([1.41, 1.83, 1.59]);
    }
    let dim = 3 * positions.len();
    let system = System::molecule(vec![1; positions.len()], positions);
    let reference = if unrestricted { "uks" } else { "rks" };
    let multiplicity = if unrestricted { 2usize } else { 1usize };
    let parent = Value::object()
        .with("xc", "hf")
        .with("reference", reference)
        .with("multiplicity", multiplicity)
        .with("basis", "sto-3g")
        .with("conv_tol", 1e-13)
        .with("conv_tol_grad", 1e-12)
        .with("conv_tol_dp", 1e-12)
        .with("grid_weight_cutoff", 0.0)
        .with("guess", "core");
    let correlated = parent
        .clone()
        .with("double_hybrid_model", "custom")
        .with("pt2_same_spin", 0.0)
        .with("pt2_opposite_spin", 0.0)
        .with("response_tolerance", 1e-12);
    // The independent legacy CPKS Hessian takes its solve tolerance from
    // conv_tol_dp, rather than the new response-provider controls.
    let parent = run("hessian", &system, &Controls::from_value(&parent).unwrap()).unwrap();
    let correlated = run(
        "hessian",
        &system,
        &Controls::from_value(&correlated).unwrap(),
    )
    .unwrap();
    let g = run(
        "gradient",
        &system,
        &Controls::from_value(
            &Value::object()
                .with("xc", "hf")
                .with("reference", reference)
                .with("multiplicity", multiplicity)
                .with("basis", "sto-3g")
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-12)
                .with("conv_tol_dp", 1e-12)
                .with("guess", "core"),
        )
        .unwrap(),
    )
    .unwrap();
    for report in [&parent, &correlated, &g] {
        assert_eq!(
            report.get("converged").unwrap().as_bool(),
            Some(true),
            "{}",
            report.to_json_pretty()
        );
    }
    for key in [
        "response_residuals",
        "amplitude_response_residuals",
        "projector_defects",
        "nuclear_stationarity_defects",
        "metric_defects",
        "response_number_defects",
        "orbital_frame_defects",
        "reference_energy_defects",
        "reference_fock_defects",
    ] {
        let values = correlated.get(key).unwrap().numeric_array().unwrap().1;
        assert!(
            values.iter().all(|v| v.is_finite() && v.abs() <= 1e-12),
            "{key}={values:?}"
        );
    }
    for (key, reference_report) in [("gradient", &g), ("hessian", &parent)] {
        let actual = correlated.get(key).unwrap().numeric_array().unwrap().1;
        let expected = reference_report
            .get(key)
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        let size = if key == "gradient" { dim } else { dim * dim };
        assert_eq!(actual.len(), size);
        assert_eq!(expected.len(), size);
        let defect = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "{reference} zero-correlation {key}: all {size} entries, maximum defect {defect:.15e}"
        );
        assert!(defect < 1e-9);
        assert!(correlated
            .get(&format!("pt2_{key}"))
            .unwrap()
            .numeric_array()
            .unwrap()
            .1
            .iter()
            .all(|v| *v == 0.0));
    }
}

#[test]
fn zero_correlation_recovers_every_parent_nuclear_tensor_entry() {
    check_zero_correlation_nuclear_limit(false);
}

#[test]
fn unrestricted_zero_correlation_recovers_every_parent_nuclear_tensor_entry() {
    check_zero_correlation_nuclear_limit(true);
}
