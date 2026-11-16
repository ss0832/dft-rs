// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Canonical common-driver EPR contract. Separate actual native, CLI and ASE
//! artifact tests establish their surface-specific acceptance.

use dft_rs::driver::{Controls, System, Value};

fn parameters(reference: &str) -> Value {
    let mut p = Value::object()
        .with("reference", reference)
        .with("basis", "sto-3g")
        .with("multiplicity", 2usize)
        .with("unrestricted", true)
        .with("dispersion", "none")
        .with("grid_level", 2usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-10)
        .with("gauge", "common-origin");
    if matches!(reference, "roks" | "gks") {
        p.set("xc", "pbe");
    }
    if matches!(reference, "gks" | "ghf") {
        p.set("epr_projection", "single-kramers-pair");
        p.set("spin_axis", [0.36, -0.48, 0.8]);
    }
    p
}

fn run(id: &str, system: &System, p: &Value) -> dft_rs::Result<Value> {
    let mut p = p.clone();
    if id == "hyperfine"
        && matches!(p.get("reference"), Some(Value::Str(s)) if matches!(s.as_str(), "rohf" | "roks"))
    {
        if let Value::Object(ref mut entries) = p {
            entries.retain(|(name, _)| name != "gauge");
        }
    }
    dft_rs::driver::run(id, system, &Controls::from_value(&p)?)
}

#[test]
fn all_four_explicit_references_execute_atomic_hyperfine_and_g_tensor() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let h = System::molecule(vec![1], vec![[0.13, -0.17, 0.23]]);
            let mut collinear = Vec::new();
            for reference in ["rohf", "roks", "ghf", "gks"] {
                let p = parameters(reference);
                let gt = run("g-tensor", &h, &p).unwrap();
                assert_eq!(gt.get("reference"), Some(&Value::Str(reference.into())));
                assert_eq!(gt.get("converged"), Some(&Value::Bool(true)));
                assert!(gt
                    .get("g_tensor")
                    .unwrap()
                    .numeric_array()
                    .unwrap()
                    .1
                    .iter()
                    .all(|v| v.is_finite()));
                let mut nuclear = p.clone();
                nuclear.set("nuclear_g", vec![-2.7]);
                let at = run("hyperfine", &h, &nuclear).unwrap();
                let per_g = at
                    .get("total_coupling_per_g_mhz")
                    .unwrap()
                    .numeric_array()
                    .unwrap()
                    .1;
                let physical = at
                    .get("total_coupling_mhz")
                    .unwrap()
                    .numeric_array()
                    .unwrap()
                    .1;
                for (a, b) in per_g.iter().zip(&physical) {
                    assert!((-2.7 * a - b).abs() < 1e-10);
                }
                assert_eq!(at.get("reference"), Some(&Value::Str(reference.into())));
                let g = gt.get("g_tensor").unwrap().numeric_array().unwrap().1;
                let energy = gt.get("energy").unwrap().as_f64().unwrap();
                if matches!(reference, "rohf" | "roks") {
                    collinear.push((g.clone(), per_g.clone(), energy));
                } else {
                    let (ordinary_g, ordinary_a, ordinary_energy) =
                        &collinear[usize::from(reference == "gks")];
                    assert!(
                        (energy - ordinary_energy).abs() < 1e-10,
                        "{reference} collinear energy limit"
                    );
                    assert!(
                        (gt.get("spin_squared").unwrap().as_f64().unwrap() - 0.75).abs() < 1e-10
                    );
                    assert_eq!(gt.get("operator_converged"), Some(&Value::Bool(true)));
                    for a in 0..3 {
                        for b in 0..3 {
                            let actual_g = (0..3).map(|c| g[3 * a + c] * g[3 * b + c]).sum::<f64>();
                            let expected_g = (0..3)
                                .map(|c| ordinary_g[3 * a + c] * ordinary_g[3 * b + c])
                                .sum::<f64>();
                            let actual_a = (0..3)
                                .map(|c| per_g[3 * a + c] * per_g[3 * b + c])
                                .sum::<f64>();
                            let expected_a = (0..3)
                                .map(|c| ordinary_a[3 * a + c] * ordinary_a[3 * b + c])
                                .sum::<f64>();
                            assert!(
                                (actual_g - expected_g).abs() < 1e-9,
                                "{reference} g Gram({a},{b})"
                            );
                            assert!(
                                (actual_a - expected_a).abs() < 1e-6,
                                "{reference} A Gram({a},{b})"
                            );
                        }
                    }
                }
                println!(
                    "{reference} atomic g={} A={} MHz/per nuclear g",
                    gt.get("g_tensor").unwrap().to_json(),
                    at.get("total_coupling_per_g_mhz").unwrap().to_json()
                );
                println!(
                    "EPR_REPORT_JSON {}",
                    Value::object().with("g", gt).with("A", at).to_json()
                );
            }
        })
        .unwrap();
}

#[test]
fn dedicated_epr_preflight_preserves_reference_and_model_boundaries() {
    let h = System::molecule(vec![1], vec![[0.13, -0.17, 0.23]]);
    for id in ["hyperfine", "g-tensor"] {
        for reference in ["rohf", "ghf"] {
            let mut p = parameters(reference);
            p.set("xc", "pbe");
            let error = run(id, &h, &p).unwrap_err();
            assert!(error
                .to_string()
                .contains("require the Hartree-Fock functional"));
        }
        let mut london = parameters("gks");
        london.set("gauge", "london");
        assert!(matches!(
            run(id, &h, &london),
            Err(dft_rs::DftError::Unsupported { .. })
        ));
        let mut pair = parameters("gks");
        pair.set("epr_projection", Value::Null);
        assert!(matches!(
            run(id, &h, &pair),
            Err(dft_rs::DftError::Unsupported { .. })
        ));
        let mut higher = parameters("ghf");
        higher.set("multiplicity", 4usize);
        let lithium = System::molecule(vec![3], vec![[0.13, -0.17, 0.23]]);
        assert!(matches!(
            run(id, &lithium, &higher),
            Err(dft_rs::DftError::Unsupported { .. })
        ));
        let mut dh = parameters("roks");
        dh.set("double_hybrid_model", "custom");
        assert!(matches!(
            run(id, &h, &dh),
            Err(dft_rs::DftError::Unsupported { .. })
        ));
        let mut unknown = parameters("rohf");
        unknown.set("unused_epr_control", 1usize);
        assert!(run(id, &h, &unknown)
            .unwrap_err()
            .to_string()
            .contains("unused_epr_control"));
    }
}
