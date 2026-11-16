// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete electric, momentum and angular-momentum transition vectors on
//! all collinear parent families. The independently fitted propagator
//! ingredients are covered by properties::pt2_transition::tests. These
//! molecular checks measure parent limits, coefficient linearity, actual
//! magnetic/electric corrections and fresh pole/normalization residuals.
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::skala::{random_tensors, write_tensors, SkalaSizes};

fn calculate(system: &System, controls: &Value) -> Value {
    run(
        "ecd-spectrum",
        system,
        &Controls::from_value(controls).unwrap(),
    )
    .unwrap()
}
fn flat(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}
fn moments(report: &Value) -> Vec<f64> {
    report
        .get("states")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|state| {
            [
                "transition_dipole",
                "transition_momentum",
                "transition_angular_momentum",
            ]
            .into_iter()
            .flat_map(|key| flat(state, key))
            .collect::<Vec<_>>()
        })
        .collect()
}
#[test]
fn every_parent_family_has_real_pt2_electric_and_magnetic_transition_corrections() {
    let models = ["hf", "svwn5", "pbe", "tpss", "pbe0", "skala"];
    if let Ok(name) = std::env::var("DFT_RS_TEST_MODEL") {
        assert!(
            models.contains(&name.as_str()),
            "unknown acceptance parent: {name}"
        );
    }
    let path = std::env::temp_dir().join(format!(
        "dft_rs_pt2_transitions_{}.model",
        std::process::id()
    ));
    write_tensors(
        &path,
        &random_tensors(
            7,
            SkalaSizes {
                width: 6,
                channels: 2,
                shells: 1,
                correlation: 2,
                layers: 1,
                post: 2,
            },
        ),
    )
    .unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
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
            let system = System::molecule(vec![1; positions.len()], positions);
            let mut controls = Value::object()
                .with("xc", xc)
                .with("basis", "sto-3g")
                .with("reference", if u { "uks" } else { "rks" })
                .with("multiplicity", if u { 2usize } else { 1usize })
                .with("guess", "core")
                .with("max_iter", 300usize)
                .with("grid_level", 0usize)
                .with("grid_weight_cutoff", 0.0)
                .with("dispersion", "none")
                .with("conv_tol", 1e-13)
                .with("conv_tol_dp", 1e-13)
                .with("conv_tol_grad", 1e-13)
                .with("nstates", 2usize)
                .with("td_tolerance", 1e-10)
                .with("broadening_ev", 0.2)
                .with("points", 101usize);
            if xc == "skala" {
                controls.set("skala_model", path.to_str().unwrap());
            }
            let ordinary = calculate(&system, &controls);
            let parent = moments(&ordinary);
            let zero = calculate(
                &system,
                &controls
                    .clone()
                    .with("pt2_same_spin", 0.0)
                    .with("pt2_opposite_spin", 0.0),
            );
            let defect = parent
                .iter()
                .zip(moments(&zero))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                defect < 2e-11,
                "{xc} U={u} zero-correlation moment defect {defect:e}"
            );
            let first = calculate(
                &system,
                &controls
                    .clone()
                    .with("pt2_same_spin", 0.031)
                    .with("pt2_opposite_spin", 0.047),
            );
            let second = calculate(
                &system,
                &controls
                    .clone()
                    .with("pt2_same_spin", 0.062)
                    .with("pt2_opposite_spin", 0.094),
            );
            assert_eq!(first.get("response_converged"), Some(&Value::Bool(true)));
            let one = moments(&first);
            let two = moments(&second);
            let linear = one
                .iter()
                .zip(&two)
                .zip(&parent)
                .map(|((a, b), p)| (b - p - 2.0 * (a - p)).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                linear < 5e-11,
                "{xc} U={u} transition coefficient linearity {linear:e}"
            );
            let change: Vec<_> = one
                .iter()
                .zip(&parent)
                .map(|(a, b)| (a - b).abs())
                .collect();
            let electric = change
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 9 < 3)
                .map(|(_, v)| *v)
                .fold(0.0_f64, f64::max);
            let magnetic = change
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 9 >= 6)
                .map(|(_, v)| *v)
                .fold(0.0_f64, f64::max);
            assert!(
                electric > 1e-8 && magnetic > 1e-8,
                "{xc} U={u} missing electric/magnetic correction {electric:e}/{magnetic:e}"
            );
            for key in [
                "pt2_transition_pole_residuals",
                "pt2_transition_normalization_defects",
            ] {
                assert!(
                    flat(&first, key).iter().all(|v| *v < 1e-10),
                    "{xc} U={u} {key}"
                );
            }
            println!("{xc} U={u}: parent limit={defect:.12e}, coefficient linearity={linear:.12e}, electric correction={electric:.12e}, magnetic correction={magnetic:.12e}, pole residuals={:?}",flat(&first,"pt2_transition_pole_residuals"));
        }
    }
}

#[test]
fn named_double_hybrid_uses_the_existing_ecd_entry_for_both_references() {
    for u in [false, true] {
        let system = System::molecule(
            vec![8, 1, 1],
            vec![
                [0.13, -0.21, 0.19],
                [1.57, 0.16, -0.08],
                [-0.19, 1.93, 0.37],
            ],
        );
        let controls = Value::object()
            .with("xc", "b2plyp")
            .with("basis", "sto-3g")
            .with("reference", if u { "uks" } else { "rks" })
            .with("charge", if u { 1i32 } else { 0i32 })
            .with("multiplicity", if u { 2usize } else { 1usize })
            .with("dispersion", "none")
            .with("grid_level", 0usize)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("nstates", 2usize)
            .with("broadening_ev", 0.2)
            .with("points", 101usize);
        let report = calculate(&system, &controls);
        assert_eq!(report.get("response_converged"), Some(&Value::Bool(true)));
        assert!(moments(&report).iter().all(|v| v.is_finite()));
        assert_eq!(flat(&report, "pt2_excitation_energy_corrections").len(), 2);
        assert!(report
            .get("model")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("property gradients"));
    }
}
