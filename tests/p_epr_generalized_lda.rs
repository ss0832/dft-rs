// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Actual public-driver regression for the three-electron SOC/GKS reference
//! that stalls without exact local-spin XC and global spin relaxation.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;

use dft_rs::driver::{Controls, System, Value};

#[test]
fn c1_three_electron_gks_reaches_the_unchanged_stationarity_gates_for_g_and_a() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let system = System::molecule(
                vec![1; 4],
                vec![
                    [0.17, -0.11, 0.23],
                    [1.77, 0.18, -0.53],
                    [-0.31, 1.74, -0.45],
                    [-1.2, -1.31, -0.61],
                ],
            );
            c1_geometry::assert_c1(&system.numbers, &system.positions);
            let parameters = Value::object()
                .with("reference", "gks")
                .with("xc", "svwn5")
                .with("basis", "sto-3g")
                .with("charge", 1i64)
                .with("multiplicity", 2usize)
                .with("guess", "core")
                .with("dispersion", "none")
                .with("grid_level", 2usize)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol", 1e-12)
                .with("conv_tol_dp", 1e-11)
                .with("conv_tol_grad", 1e-10)
                .with("max_iter", 500usize)
                .with("epr_projection", "single-kramers-pair")
                .with("epr_projection_tolerance", 1e-8)
                .with("gauge", "common-origin")
                .with("gauge_origin", system.positions[0]);
            for feature in ["g-tensor", "hyperfine"] {
                let report = dft_rs::driver::run(
                    feature,
                    &system,
                    &Controls::from_value(&parameters).unwrap(),
                )
                .unwrap();
                assert_eq!(report.get("reference").unwrap().as_str(), Some("gks"));
                assert_eq!(report.get("converged").unwrap().as_bool(), Some(true));
                assert_eq!(
                    report.get("operator_converged").unwrap().as_bool(),
                    Some(true)
                );
                let number = |key| report.get(key).unwrap().as_f64().unwrap();
                assert!(number("orbital_residual") < 1e-10);
                assert!(number("density_change") < 1e-11);
                assert!(number("occupation_consistency") < 1e-11);
                assert!(number("particle_number_error") < 1e-11);
                assert!(number("energy_change") < 1e-12);
                assert!(number("spinor_recovery_cycles") > 0.0);
                assert!(number("scf_iterations") <= 500.0);
                assert!(number("spinor_last_segment_iterations") <= 8.0);
                assert!(number("spinor_rotation_energy_defect") < 1e-12);
                assert!((number("energy") - -1.6769848451687972).abs() < 1e-10);
                let key = if feature == "g-tensor" {
                    "g_tensor"
                } else {
                    "total_coupling_per_g_mhz"
                };
                let (shape, values) = report.get(key).unwrap().numeric_array().unwrap();
                assert_eq!(
                    shape,
                    if feature == "g-tensor" {
                        vec![3, 3]
                    } else {
                        vec![4, 3, 3]
                    }
                );
                assert!(values.iter().all(|v| v.is_finite()));
                println!(
                    "Public {feature} C1 H4+ GKS E={} residual={} cycles={} total_iterations={}",
                    number("energy"),
                    number("orbital_residual"),
                    number("spinor_recovery_cycles"),
                    number("scf_iterations")
                );
            }
        })
        .unwrap();
}

#[test]
fn c1_methyl_gks_reaches_the_original_stationarity_gates_for_g_and_a() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let system = System::molecule(
                vec![6, 1, 1, 1],
                vec![
                    [0.17, -0.11, 0.23],
                    [1.77, 0.18, -0.53],
                    [-0.31, 1.74, -0.45],
                    [-1.2, -1.31, -0.61],
                ],
            );
            c1_geometry::assert_c1(&system.numbers, &system.positions);
            let parameters = Value::object()
                .with("reference", "gks")
                .with("xc", "svwn5")
                .with("basis", "sto-3g")
                .with("charge", 0i64)
                .with("multiplicity", 2usize)
                .with("guess", "core")
                .with("dispersion", "none")
                .with("grid_level", 2usize)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol", 1e-12)
                .with("conv_tol_dp", 1e-11)
                .with("conv_tol_grad", 1e-10)
                .with("max_iter", 500usize)
                .with("epr_projection", "single-kramers-pair")
                .with("epr_projection_tolerance", 1e-8)
                .with("gauge", "common-origin")
                .with("gauge_origin", system.positions[0]);
            for feature in ["g-tensor", "hyperfine"] {
                let report = dft_rs::driver::run(
                    feature,
                    &system,
                    &Controls::from_value(&parameters).unwrap(),
                )
                .unwrap();
                assert_eq!(report.get("reference").unwrap().as_str(), Some("gks"));
                assert_eq!(report.get("converged").unwrap().as_bool(), Some(true));
                assert_eq!(
                    report.get("operator_converged").unwrap().as_bool(),
                    Some(true)
                );
                let number = |key| report.get(key).unwrap().as_f64().unwrap();
                assert!(number("orbital_residual") < 1e-10);
                assert!(number("density_change") < 1e-11);
                assert!(number("occupation_consistency") < 1e-11);
                assert!(number("particle_number_error") < 1e-11);
                assert!(number("energy_change") < 1e-12);
                assert!(number("spinor_recovery_cycles") > 0.0);
                assert!(number("scf_iterations") <= 500.0);
                assert!(number("spinor_last_segment_iterations") <= 8.0);
                assert!(number("spinor_rotation_energy_defect") < 1e-12);
                assert!((number("energy") - -38.86163683564086).abs() < 1e-10);
                let key = if feature == "g-tensor" {
                    "g_tensor"
                } else {
                    "total_coupling_per_g_mhz"
                };
                let (shape, values) = report.get(key).unwrap().numeric_array().unwrap();
                assert_eq!(
                    shape,
                    if feature == "g-tensor" {
                        vec![3, 3]
                    } else {
                        vec![4, 3, 3]
                    }
                );
                assert!(values.iter().all(|v| v.is_finite()));
                println!(
                    "Public {feature} C1 methyl GKS E={} residual={} cycles={} total_iterations={}",
                    number("energy"),
                    number("orbital_residual"),
                    number("spinor_recovery_cycles"),
                    number("scf_iterations")
                );
            }
        })
        .unwrap();
}
