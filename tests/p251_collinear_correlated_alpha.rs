// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All nine restricted/unrestricted correlated energy-Hessian components,
//! including separate unscaled SS-alpha, OS and SS-beta energy oracles.
//! Parent and distinct target responses use the same generic provider path.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::{run, Controls, System, Value};

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}
fn energy(v: &Value, key: &str) -> f64 {
    if key == "target_difference" {
        return energy(v, "reference_energy") - energy(v, "orbital_reference_energy");
    }
    v.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn restricted_and_unrestricted_full_correlated_tensors_have_independent_energy_fits() {
    for unrestricted in [false, true] {
        for xc in ["hf", "pbe0"] {
            if std::env::var("DFT_RS_TEST_MODEL")
                .ok()
                .is_some_and(|v| v != xc)
            {
                continue;
            }
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
            let base = Value::object()
                .with("xc", xc)
                .with("basis", "sto-3g")
                .with("unrestricted", unrestricted)
                .with("multiplicity", if unrestricted { 2usize } else { 1 })
                .with("pt2_same_spin", 0.2)
                .with("pt2_opposite_spin", 0.3)
                .with("double_hybrid_target_xc", "pbe")
                .with("grid_level", 0usize)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-12)
                .with("conv_tol_dp", 1e-12)
                .with("response_tolerance", 1e-12)
                .with("max_iter", 300usize)
                .with("guess", "core")
                .with("dispersion", "none");
            let evaluate = |id: &str, v: &Value| {
                let mut v = v.clone();
                if id == "energy" {
                    if let Value::Object(entries) = &mut v {
                        entries.retain(|(key, _)| key != "response_tolerance");
                    }
                }
                run(id, &system, &Controls::from_value(&v).unwrap()).unwrap()
            };
            let report = evaluate("polarizability", &base);
            assert!(report.get("converged").unwrap().as_bool().unwrap());
            for key in [
                "response_residuals",
                "amplitude_response_residuals",
                "response_number_residuals",
                "projector_residuals",
                "stationarity_residuals",
                "orbital_frame_residuals",
            ] {
                let raw = values(&report, key)
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max);
                println!("{xc} U={unrestricted} {key} max={raw:.12e}");
                assert!(raw < 1e-9, "{xc} U={unrestricted} {key}");
            }
            let tensors = [
                "polarizability",
                "parent_polarizability",
                "target_polarizability_correction",
                "pt2_polarizability_correction",
                "pt2_same_spin_alpha_polarizability",
                "pt2_opposite_spin_polarizability",
                "pt2_same_spin_beta_polarizability",
            ];
            let keys = [
                "energy",
                "orbital_reference_energy",
                "target_difference",
                "pt2_correlation",
                "pt2_same_spin_alpha",
                "pt2_opposite_spin",
                "pt2_same_spin_beta",
            ];
            let exact: Vec<_> = tensors.iter().map(|key| values(&report, key)).collect();
            for a in 0..3 {
                for b in 0..3 {
                    let i = 3 * a + b;
                    assert!((exact[0][i] - exact[1][i] - exact[2][i] - exact[3][i]).abs() < 1e-10);
                    assert!(
                        (exact[3][i] - 0.2 * (exact[4][i] + exact[6][i]) - 0.3 * exact[5][i]).abs()
                            < 1e-10
                    );
                }
            }
            let sample = |a: usize, x: f64, b: usize, y: f64| {
                let mut v = base.clone();
                let mut field = [0.0; 3];
                field[a] += x;
                field[b] += y;
                v.set("field", field);
                let value = evaluate("energy", &v);
                keys.map(|key| energy(&value, key))
            };
            let zero = sample(0, 0.0, 0, 0.0);
            let intervals = [
                [0.008, 0.006, 0.004, 0.003],
                [0.002, 0.0015, 0.001, 0.00075],
                [0.016, 0.012, 0.008, 0.006],
                [0.0008, 0.0006, 0.0004, 0.0003],
            ];
            let mut failures = Vec::new();
            for a in 0..3 {
                for b in a..3 {
                    let names = tensors
                        .iter()
                        .flat_map(|key| {
                            let mut names = vec![format!("{xc} U={unrestricted} {key}({a},{b})")];
                            if a != b {
                                names.push(format!("{xc} U={unrestricted} {key}({b},{a})"));
                            }
                            names
                        })
                        .collect::<Vec<_>>();
                    components::fit_components(
                        &names,
                        &intervals,
                        |steps| {
                            let mut errors = vec![[0.0; 4]; names.len()];
                            for (k, &h) in steps.iter().enumerate() {
                                let (plus, minus, pm, mp) = if a == b {
                                    (sample(a, h, b, 0.0), sample(a, -h, b, 0.0), zero, zero)
                                } else {
                                    (
                                        sample(a, h, b, h),
                                        sample(a, -h, b, -h),
                                        sample(a, h, b, -h),
                                        sample(a, -h, b, h),
                                    )
                                };
                                for c in 0..keys.len() {
                                    let fd = if a == b {
                                        -(plus[c] + minus[c] - 2.0 * zero[c]) / (h * h)
                                    } else {
                                        -(plus[c] + minus[c] - pm[c] - mp[c]) / (4.0 * h * h)
                                    };
                                    let stride = if a == b { 1 } else { 2 };
                                    errors[stride * c][k] = (fd - exact[c][3 * a + b]).abs();
                                    if a != b {
                                        errors[stride * c + 1][k] =
                                            (fd - exact[c][3 * b + a]).abs();
                                    }
                                }
                            }
                            errors
                        },
                        &mut failures,
                    );
                }
            }
            components::assert_no_failures("full correlated electric tensors", &failures);
            let mut zero = base.clone();
            zero.set("pt2_same_spin", 0.0);
            zero.set("pt2_opposite_spin", 0.0);
            if let Value::Object(entries) = &mut zero {
                entries.retain(|(key, _)| key != "double_hybrid_target_xc");
            }
            let limit = evaluate("polarizability", &zero);
            let mut plain = zero.clone();
            if let Value::Object(entries) = &mut plain {
                entries.retain(|(key, _)| !key.starts_with("pt2_"));
            }
            let parent = evaluate("polarizability", &plain);
            let defect = values(&limit, "polarizability")
                .iter()
                .zip(values(&parent, "polarizability"))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            println!(
                "{xc} U={unrestricted} zero-coefficient complete-tensor parent limit {defect:.12e}"
            );
            assert!(defect < 1e-9);
            assert!(values(&limit, "pt2_polarizability_correction")
                .iter()
                .all(|v| *v == 0.0));
        }
    }
}
