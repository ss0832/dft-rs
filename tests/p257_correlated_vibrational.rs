// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every Hessian and APT entry returned by correlated VCD, against displaced
//! gradient and dipole calculations on their independent public drivers.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::{run, Controls, System, Value};

fn values(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}
fn calculate(id: &str, system: &System, control: &Value) -> Value {
    run(id, system, &Controls::from_value(control).unwrap()).unwrap_or_else(|e| panic!("{id}: {e}"))
}

#[test]
fn every_correlated_vcd_hessian_and_polar_tensor_component_has_a_displaced_property_fit() {
    let models = ["hf", "svwn5", "pbe", "tpss", "pbe0"];
    if let Ok(model) = std::env::var("DFT_RS_TEST_MODEL") {
        assert!(
            models.contains(&model.as_str()),
            "unknown acceptance model: {model}"
        );
    }
    for xc in models {
        for u in [false, true] {
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
            if u {
                positions.push([1.41, 1.83, 1.59]);
            }
            let numbers = vec![1; positions.len()];
            let dim = 3 * positions.len();
            let system = System::molecule(numbers.clone(), positions.clone());
            let controls = Value::object()
                .with("xc", xc)
                .with("basis", "sto-3g")
                .with("reference", if u { "uks" } else { "rks" })
                .with("multiplicity", if u { 2usize } else { 1usize })
                .with("guess", "core")
                .with("max_iter", 300usize)
                .with("grid_level", 0usize)
                .with("grid_weight_cutoff", 0.0)
                .with("dispersion", "none")
                .with("conv_tol", 1e-14)
                .with("conv_tol_dp", 1e-14)
                .with("conv_tol_grad", 1e-14)
                .with("response_tolerance", 1e-12)
                .with("pt2_same_spin", 0.31)
                .with("pt2_opposite_spin", 0.47);
            let vcd = calculate(
                "vcd",
                &system,
                &controls
                    .clone()
                    .with("gauge", "common-origin")
                    .with("gauge_origin", vec![0.0; 3]),
            );
            assert_eq!(vcd.get("response_converged"), Some(&Value::Bool(true)));
            assert_eq!(vcd.get("converged"), Some(&Value::Bool(true)));
            let hessian = values(&vcd, "hessian");
            let polar = values(&vcd, "dipole_derivatives");
            let separate = calculate("hessian", &system, &controls);
            assert!(hessian
                .iter()
                .zip(values(&separate, "hessian"))
                .all(|(a, b)| (a - b).abs() < 1e-12));
            let mut failures = Vec::new();
            for x in 0..dim {
                let mut names: Vec<_> = (0..dim)
                    .map(|r| format!("{xc} U={u} VCD Hessian ({r},{x})"))
                    .collect();
                names.extend((0..3).map(|r| format!("{xc} U={u} VCD APT ({r},{x})")));
                stencils::fit_components_staged(
                    &names,
                    &[
                        ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                        (
                            [0.005, 0.00375, 0.0025, 0.001875],
                            stencils::Stencil::Second,
                        ),
                        ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
                        ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
                    ],
                    |steps, stencil| {
                        let mut errors = vec![[0.0; 4]; dim + 3];
                        for (k, &h) in steps.iter().enumerate() {
                            let shifts = if stencil == stencils::Stencil::Fourth {
                                vec![-h, h, -2.0 * h, 2.0 * h]
                            } else {
                                vec![-h, h]
                            };
                            let states: Vec<_> = shifts
                                .into_iter()
                                .map(|shift| {
                                    let mut p = positions.clone();
                                    p[x / 3][x % 3] += shift;
                                    let moved = System::molecule(numbers.clone(), p);
                                    (
                                        values(
                                            &calculate("gradient", &moved, &controls),
                                            "gradient",
                                        ),
                                        values(
                                            &calculate(
                                                "dipole",
                                                &moved,
                                                &controls.clone().with("origin", "zero"),
                                            ),
                                            "dipole",
                                        ),
                                    )
                                })
                                .collect();
                            for r in 0..dim {
                                errors[r][k] = (stencil.first(
                                    states[1].0[r],
                                    states[0].0[r],
                                    states.get(3).map_or(0.0, |s| s.0[r]),
                                    states.get(2).map_or(0.0, |s| s.0[r]),
                                    h,
                                ) - hessian[r * dim + x])
                                    .abs();
                            }
                            for r in 0..3 {
                                errors[dim + r][k] = (stencil.first(
                                    states[1].1[r],
                                    states[0].1[r],
                                    states.get(3).map_or(0.0, |s| s.1[r]),
                                    states.get(2).map_or(0.0, |s| s.1[r]),
                                    h,
                                ) - polar[r * dim + x])
                                    .abs();
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
            }
            components::assert_no_failures("all VCD correlated Hessian/APT entries", &failures);
        }
    }
}
