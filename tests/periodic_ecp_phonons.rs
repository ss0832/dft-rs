// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full ordered CPKS Hessians of a C1 scalar-ECP cell. Every family uses
//! the same driver and fit gates; functional names only choose test inputs.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use std::collections::BTreeMap;
use stencils::Stencil;

fn scalar(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn array(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn fit(xc: &str, reference: &str) {
    let system = System {
        numbers: vec![1, 11, 1],
        positions: vec![[0.13, -0.21, 0.07], [2.0, 0.45, -0.31], [-0.46, 1.29, 1.62]],
        cell: Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
        pbc: [true; 3],
    };
    let options = Value::object()
        .with("xc", xc)
        .with("reference", reference)
        .with("charge", if reference == "rks" { 1 } else { 0 })
        .with("multiplicity", if reference == "rks" { 1 } else { 2 })
        .with(
            "basis",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/periodic_scalar_ecp.g94"
            ),
        )
        .with("dispersion", "none")
        .with("k_mesh", [1, 1, 1])
        .with("overlap_cutoff", 5.5)
        .with("xc_ao_cutoff", 16.0)
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("max_iter", 500);
    let calculate = |feature, geometry: &System, input: &Value| {
        let result = run(feature, geometry, &Controls::from_value(input).unwrap())
            .unwrap_or_else(|error| panic!("{feature} {xc} {reference} failed: {error}"));
        assert_eq!(
            result.get("converged").and_then(Value::as_bool),
            Some(true),
            "{feature} {xc} {reference} did not converge at {:?}: {result:?}",
            geometry.positions
        );
        result
    };
    let phonon_options = options.clone().with("response_tolerance", 1e-11);
    let report = calculate("phonons", &system, &phonon_options);
    assert_eq!(
        report.get("reference_type").and_then(Value::as_str),
        Some(reference)
    );
    let hessian = array(&report, "raw_force_constants");
    assert_eq!(hessian.len(), 81);
    assert!(scalar(&report, "raw_asymmetry") < 1e-8);
    assert!(scalar(&report, "acoustic_residual") < 1e-8);
    println!("scalar ECP CPKS {xc} {reference} energy={:.17e} raw_asymmetry={:.17e} acoustic={:.17e} Hessian={hessian:?} frequencies={:?} response_solves={:?}",
        scalar(&report,"energy"),scalar(&report,"raw_asymmetry"),scalar(&report,"acoustic_residual"),
        array(&report,"frequencies_cm1"),report.get("response_solves"));
    let force_options = options.with("forces", true);
    let labels: Vec<_> = (0..9)
        .flat_map(|a| {
            (0..9).map(move |b| format!("scalar ECP {xc} {reference} raw Hessian [{a},{b}]"))
        })
        .collect();
    let mut samples = BTreeMap::new();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 81];
            for b in 0..9 {
                let mut forces = |h: f64| {
                    samples
                        .entry((b, h.to_bits()))
                        .or_insert_with(|| {
                            let mut geometry = system.clone();
                            geometry.positions[b / 3][b % 3] += h;
                            let force = calculate("periodic-energy", &geometry, &force_options);
                            let values = array(&force, "forces");
                            println!(
                                "scalar ECP {xc} {reference} force sample [{b}] h={h:.17e} E={:.17e} SCF={:.17e} F={values:?}",
                                scalar(&force, "energy"),
                                scalar(&force, "orbital_residual"),
                            );
                            values
                        })
                        .clone()
                };
                for (s, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (forces(h), forces(-h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (forces(2.0 * h), forces(-2.0 * h))
                    } else {
                        (vec![0.0; 9], vec![0.0; 9])
                    };
                    for a in 0..9 {
                        errors[9 * a + b][s] = (-stencil
                            .first(plus[a], minus[a], plus2[a], minus2[a], h)
                            - hessian[9 * a + b])
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("scalar ECP relaxed CPKS Hessian", &failures);
    let mut richardson = 0.0_f64;
    for a in 0..9 {
        for b in 0..9 {
            let derivative = |h: f64| {
                -(samples[&(b, h.to_bits())][a] - samples[&(b, (-h).to_bits())][a]) / (2.0 * h)
            };
            richardson = richardson.max(
                ((4.0 * derivative(0.005) - derivative(0.01)) / 3.0 - hessian[9 * a + b]).abs(),
            );
        }
    }
    println!("scalar ECP {xc} {reference} maximum Richardson defect={richardson:.17e}");
    assert!(richardson < 2e-7);
}

macro_rules! family {
    ($name:ident, $xc:literal, $reference:literal) => {
        #[test]
        fn $name() {
            fit($xc, $reference);
        }
    };
}

family!(restricted_hf, "hf", "rks");
family!(unrestricted_hf, "hf", "uks");
family!(restricted_lda, "svwn5", "rks");
family!(unrestricted_lda, "svwn5", "uks");
family!(restricted_gga, "pbe", "rks");
family!(unrestricted_gga, "pbe", "uks");
family!(restricted_meta_gga, "tpss", "rks");
family!(unrestricted_meta_gga, "tpss", "uks");
family!(restricted_hybrid, "pbe0", "rks");
family!(unrestricted_hybrid, "pbe0", "uks");
