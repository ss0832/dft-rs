// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Native consumer acceptance for the fixed core-s/valence-sp AO template.
//! The earlier 13-AO curve preflight failed chemical accuracy (MAE ~0.060071 Eh).
//! These gates validate accounting, serialization and analytic derivatives only.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/reduced_fragment.rs"]
pub mod fixture;

use dft_rs::driver::continuous_species::reduced::{
    evaluate_candidate, evaluate_candidate_with_response, ReducedEvaluation,
};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{
    generate_methyl, GenerationOptions, ProjectionRule, ReducedFragment,
};
use dft_rs::pseudo_sub::reduction_basis::{OneCenterBasisTemplate, CORE_VALENCE_SP_TEMPLATE_ID};
use dft_rs::response::ResponseSolverConfig;
use std::collections::BTreeMap;
use std::sync::OnceLock;

fn candidate() -> &'static ReducedFragment {
    static MODEL: OnceLock<ReducedFragment> = OnceLock::new();
    MODEL.get_or_init(|| {
        let (source, spec) = fixture::ethane();
        let mut options = fixture::construction_options();
        options.basis_template = OneCenterBasisTemplate::CarbonCoreValenceSp;
        options.projection_rule = ProjectionRule::PhysicalSupportPauliNormalized;
        assert_eq!(options.electrostatic_points.len(), 47);
        assert_eq!(options.electrostatic_weight, 1e6);
        generate_methyl(&source, spec, options).unwrap()
    })
}
fn origin() -> Vec3 {
    Vec3::new(0.11, 0.07, -0.09)
}

#[test]
fn fine_nuclear_gradient_absolute_accuracy_all_fifteen_components() {
    let model = candidate();
    let saved = model.to_json().unwrap();
    let y = model.geometry_map.reference_coordinates();
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let baseline = evaluate_candidate(model, &y, field, origin()).unwrap();
    assert_eq!(baseline.gradient.len(), 5);
    let mut accepted = false;
    // This absolute-accuracy check supplements the unchanged 27-component
    // four-step order test. Retain every coarse result before any fallback.
    for h in [0.001, 0.0005] {
        let mut all_pass = true;
        let mut maximum = 0_f64;
        for component in 0..15 {
            let atom = component / 3;
            let axis = component % 3;
            let energy = |amount: f64| {
                let mut coords = y.clone();
                let mut p = coords[atom].to_array();
                p[axis] += amount;
                coords[atom] = Vec3::new(p[0], p[1], p[2]);
                evaluate_candidate(model,&coords,field,origin()).unwrap_or_else(|e|panic!("fine nuclear gradient component={component} displacement={amount:.16e}: {e}")).energy
            };
            let plus = energy(h);
            let minus = energy(-h);
            let fd = (plus - minus) / (2. * h);
            let analytic = baseline.gradient[atom].get(axis);
            let error = (fd - analytic).abs();
            let pass = error.is_finite() && error <= 1e-6;
            all_pass &= pass;
            maximum = maximum.max(error);
            println!("core13 fine gradient[{atom},{axis}] h={h:.16e} Eplus={plus:.16e} Eminus={minus:.16e} analytic={analytic:.16e} FD={fd:.16e} absoluteError={error:.16e} bound=1e-6 units=Eh/bohr pass={pass} no_floor=true");
        }
        println!("core13 fine gradient interval h={h:.16e} components=15 maxAbsoluteError={maximum:.16e} all_pass={all_pass}");
        if all_pass {
            accepted = true;
            break;
        }
    }
    assert_eq!(saved, model.to_json().unwrap());
    assert!(accepted,"all fifteen nuclear gradient components must meet 1e-6 Eh/bohr on a predeclared fine interval; raw failed intervals retained");
}

#[test]
fn fixed_core_template_accounting_roundtrip_and_every_force_dipole_alpha_component() {
    let model = candidate();
    model.validate().unwrap();
    assert_eq!(
        model.options.basis_template,
        OneCenterBasisTemplate::CarbonCoreValenceSp
    );
    assert_eq!(
        model.options.projection_rule,
        ProjectionRule::PhysicalSupportPauliNormalized
    );
    let ledger = &model.electron_ledger;
    assert_eq!(
        (
            ledger.full_electron_count,
            ledger.frozen_electron_count,
            ledger.active_electron_count
        ),
        (18, 8, 10)
    );
    assert_eq!(ledger.fragments[0].effective_ionic_charge, 1);
    assert_eq!(ledger.full_total_charge, ledger.reduced_total_charge);
    assert_eq!(ledger.reduced_total_charge, 0);
    assert_eq!(
        (
            model.diagnostics.full_ao_count,
            model.diagnostics.reduced_ao_count
        ),
        (16, 13)
    );
    assert_eq!(
        model.local_amplitudes.len() + model.nonlocal_coefficients.len(),
        39
    );
    assert_eq!(
        model.gaussian_charges.len()
            + model
                .gaussian_body_dipoles
                .iter()
                .map(|v| v.len())
                .sum::<usize>(),
        12
    );
    assert!(model.diagnostics.reference_energy_reconstruction_error < 1e-12);
    let saved = model.to_json().unwrap();
    assert!(saved.contains(CORE_VALENCE_SP_TEMPLATE_ID));
    let restored = ReducedFragment::from_json(&saved).unwrap();
    assert_eq!(restored.to_json().unwrap(), saved);
    assert_eq!(
        restored.construction_fingerprint,
        model.construction_fingerprint
    );
    assert_eq!(
        restored.internal_energy_reference.to_bits(),
        model.internal_energy_reference.to_bits()
    );
    // One ordinary default-generation regression, independent of the one 13-AO
    // candidate: introducing this template must not change the legacy chart.
    let defaults = GenerationOptions::default();
    assert_eq!(defaults.basis_template, OneCenterBasisTemplate::Hydrogen1s);
    assert_eq!(defaults.projection_rule, ProjectionRule::FullAoMetric);
    let (source, spec) = fixture::ethane();
    let legacy = generate_methyl(&source, spec, defaults).unwrap();
    assert_eq!(legacy.construction_fingerprint, "fnv1a64:d5f4cc45da1c31c1");
    assert_ne!(
        model.construction_fingerprint,
        legacy.construction_fingerprint
    );
    let y = model.geometry_map.reference_coordinates();
    assert_eq!(y.len(), 5);
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let baseline = evaluate_candidate_with_response(
        model,
        &y,
        field,
        origin(),
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    assert_eq!(
        (
            baseline.full_electrons,
            baseline.frozen_electrons,
            baseline.active_electrons
        ),
        (18, 8, 10)
    );
    assert_eq!(
        (
            baseline.ao_count,
            baseline.occupied_spatial_orbitals,
            baseline.response_variables
        ),
        (13, 5, 40)
    );
    assert!((baseline.active_density_trace - 10.).abs() < 1e-9);
    assert!(baseline.physical_total_charge.abs() < 1e-12);
    assert!(baseline.scf_orbital_residual <= 1e-11);
    assert_eq!(
        baseline.candidate_status,
        "candidate-not-chemically-validated"
    );
    assert_eq!(baseline.shape_status, "not-generated");
    assert_eq!(
        baseline.response_status,
        "active-only-no-internal-polarization"
    );
    assert!(
        (baseline.energy - baseline.electronic_energy - baseline.classical_energy).abs() < 1e-10
    );
    assert_eq!((baseline.gradient.len(), baseline.forces.len()), (5, 5));
    for (force, gradient) in baseline.forces.iter().zip(&baseline.gradient) {
        assert!((*force + *gradient).norm() < 1e-14);
    }
    let repeated = evaluate_candidate(&restored, &y, field, origin()).unwrap();
    assert!((repeated.energy - baseline.energy).abs() < 1e-11);
    assert!((repeated.model_dipole - baseline.model_dipole).norm() < 1e-10);
    for (a, b) in repeated.forces.iter().zip(&baseline.forces) {
        assert!((*a - *b).norm() < 1e-9);
    }
    let alpha = baseline
        .active_model_polarizability
        .expect("active-model analytic static alpha must be present");
    assert!(alpha.iter().flatten().all(|v| v.is_finite()));
    let residuals = baseline
        .response_residuals
        .as_ref()
        .expect("fresh CPKS residuals must be reported");
    // The fixed model reports its null chart-parameter response, then x/y/z fields.
    assert_eq!(residuals.len(), 4);
    assert_eq!(
        residuals[0], 0.0,
        "null parameter response is structurally zero"
    );
    assert!(residuals
        .iter()
        .all(|r| r.is_finite() && *r <= ResponseSolverConfig::default().tolerance));
    println!("core13 fingerprint={} E0={:.16e} E={:.16e} Ne=18/8/10 ion=1 AO=13 occupied=5 responseVariables=40 active_only_alpha={alpha:?} CPKSresiduals={residuals:?} status=chemical-accuracy-unestablished prior_curve_MAE_Eh=0.060071 no_winner_selection=true",model.construction_fingerprint,model.internal_energy_reference,baseline.energy);
    let intervals = [[0.02, 0.01, 0.005, 0.0025], [0.04, 0.02, 0.01, 0.005]];
    let names: Vec<_> = (0..27)
        .map(|i| {
            if i < 15 {
                format!("core13 gradient[{},{}]", i / 3, i % 3)
            } else if i < 18 {
                format!("core13 dipole[{}]", i - 15)
            } else {
                format!(
                    "core13 active-only alpha[{},{}]",
                    (i - 18) / 3,
                    (i - 18) % 3
                )
            }
        })
        .collect();
    let mut cache = BTreeMap::<(usize, u64), ReducedEvaluation>::new();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &intervals,
        |steps| {
            let mut errors = vec![[0.; 4]; 27];
            for component in 0..27 {
                let (channel, analytic) = if component < 15 {
                    (
                        component,
                        baseline.gradient[component / 3].get(component % 3),
                    )
                } else if component < 18 {
                    (
                        15 + component - 15,
                        baseline.model_dipole.get(component - 15),
                    )
                } else {
                    (
                        15 + (component - 18) % 3,
                        alpha[(component - 18) / 3][(component - 18) % 3],
                    )
                };
                let mut estimates = [0.; 4];
                for (k, &h) in steps.iter().enumerate() {
                    let mut value = |amount: f64| {
                        let v = cache.entry((channel, amount.to_bits())).or_insert_with(|| {
                            let mut coords = y.clone();
                            let mut f = field;
                            if channel < 15 {
                                let axis = channel % 3;
                                let mut p = coords[channel / 3].to_array();
                                p[axis] += amount;
                                coords[channel / 3] = Vec3::new(p[0], p[1], p[2]);
                            } else {
                                let mut p = f.to_array();
                                p[channel - 15] += amount;
                                f = Vec3::new(p[0], p[1], p[2]);
                            }
                            evaluate_candidate(model, &coords, f, origin()).unwrap_or_else(|e| {
                                panic!(
                                    "core13 derivative channel={channel} amount={amount:.16e}: {e}"
                                )
                            })
                        });
                        if component < 18 {
                            v.energy
                        } else {
                            v.model_dipole.get((component - 18) / 3)
                        }
                    };
                    estimates[k] = (value(h) - value(-h)) / (2. * h)
                        * if (15..18).contains(&component) {
                            -1.
                        } else {
                            1.
                        };
                    errors[component][k] = (estimates[k] - analytic).abs();
                }
                let extrapolated = (4. * estimates[3] - estimates[2]) / 3.;
                println!("{} interval={steps:?} analytic={analytic:.16e} finestAbsoluteError={:.16e} RichardsonError={:.16e} no_floor=true",names[component],errors[component][3],(extrapolated-analytic).abs());
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "core13 candidate all15 force/3 dipole/9 active alpha components",
        &failures,
    );
    assert_eq!(saved, model.to_json().unwrap());
}
