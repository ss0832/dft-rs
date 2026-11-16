// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Main-group and transition-metal regression beyond the smallest molecules.
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/molecules.rs"]
mod molecules;
use dft_rs::driver::{run, Controls, Value};

#[test]
fn phosphine_correlated_alpha_passes_independent_four_step_dipole_fit() {
    let system = molecules::phosphine();
    let controls = |field: f64| {
        Controls::from_value(
            &Value::object()
                .with("xc", "b2plyp")
                .with("basis", "sto-3g")
                .with("grid_level", 0usize)
                .with("field", [0., 0., 0.003 + field])
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-12)
                .with("conv_tol_dp", 1e-12)
                .with("response_tolerance", 1e-12),
        )
        .unwrap()
    };
    let alpha = run("polarizability", &system, &controls(0.)).unwrap();
    assert_eq!(alpha.get("n_alpha").unwrap().as_f64(), Some(9.0));
    assert_eq!(alpha.get("n_beta").unwrap().as_f64(), Some(9.0));
    let (_, tensor) = alpha
        .get("polarizability")
        .unwrap()
        .numeric_array()
        .unwrap();
    let exact = tensor[8];
    let steps = [0.01, 0.005, 0.0025, 0.00125];
    let estimates: Vec<_> = steps
        .iter()
        .map(|h| {
            let dipole = |field| {
                let r = run("dipole", &system, &controls(field)).unwrap();
                r.get("dipole").unwrap().numeric_array().unwrap().1[2]
            };
            (dipole(*h) - dipole(-*h)) / (2. * h)
        })
        .collect();
    let errors: Vec<_> = estimates.iter().map(|v| (v - exact).abs()).collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    let rich = ((4. * estimates[3] - estimates[2]) / 3. - exact).abs();
    println!("PH3 correlated alpha value={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
    assert!((fit.exponent - 2.).abs() < 0.05 && fit.r_squared > 0.999);
    assert!(rich < 1e-5);
    for key in [
        "response_residuals",
        "amplitude_response_residuals",
        "response_number_residuals",
    ] {
        assert!(alpha
            .get(key)
            .unwrap()
            .numeric_array()
            .unwrap()
            .1
            .iter()
            .all(|v| v.abs() < 1e-9));
    }
}

#[test]
#[ignore = "Long transition-metal SCF and electric-property regression"]
fn ferrocene_closed_shell_reference_has_correct_population_and_inversion_dipole() {
    let system = molecules::ferrocene();
    assert_eq!(system.numbers.len(), 21);
    assert_eq!(
        system.numbers.iter().map(|z| *z as usize).sum::<usize>(),
        96
    );
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "pbe")
            .with("basis", "sto-3g")
            .with("grid_level", 0usize)
            .with("guess", "sad")
            .with("max_iter", 300usize)
            .with("conv_tol", 1e-10)
            .with("conv_tol_grad", 1e-8)
            .with("conv_tol_dp", 1e-9),
    )
    .unwrap();
    let report = run("dipole", &system, &controls).unwrap();
    assert_eq!(report.get("n_alpha").unwrap().as_f64(), Some(48.0));
    assert_eq!(report.get("n_beta").unwrap().as_f64(), Some(48.0));
    assert_eq!(report.get("converged"), Some(&Value::Bool(true)));
    let (_, dipole) = report.get("dipole").unwrap().numeric_array().unwrap();
    println!(
        "ferrocene PBE energy={:?} dipole={dipole:?}",
        report.get("energy")
    );
    assert!(dipole.iter().all(|v| v.abs() < 1e-5));
}
