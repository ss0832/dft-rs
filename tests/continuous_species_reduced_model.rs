// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-candidate reduced-model energy/force consistency, not chemical accuracy.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::continuous_species::reduced::{evaluate_candidate, ReducedEvaluation};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{generate_methyl, GenerationOptions, ReducedFragment};
use dft_rs::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec};
use dft_rs::system::{Atom, Molecule};
use std::collections::BTreeMap;
use std::sync::OnceLock;

const STEPS: [[f64; 4]; 2] = [[0.02, 0.01, 0.005, 0.0025], [0.04, 0.02, 0.01, 0.005]];

fn source_molecule() -> Molecule {
    let positions = [
        [-1.46, -0.14, 0.23],
        [-2.13, 1.69, 0.68],
        [-2.28, -1.57, 1.31],
        [-2.19, -0.43, -1.65],
        [1.35, 0.17, -0.12],
        [2.05, 2.08, 0.16],
        [2.10, -0.92, 1.45],
        [2.01, -0.84, -1.67],
    ];
    Molecule::new(
        positions
            .iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i == 0 || i == 4 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

fn spec() -> FragmentSpec {
    FragmentSpec {
        explicit_atoms: vec![4, 5, 6, 7],
        retained_atoms: vec![0, 1, 2, 3],
        port: AttachmentPort {
            fragment_atom: 4,
            host_atom: 0,
            reference_atom: 1,
        },
        single_bonds: vec![[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]],
        electronic_centers: 1,
    }
}

fn candidate() -> &'static ReducedFragment {
    static CANDIDATE: OnceLock<ReducedFragment> = OnceLock::new();
    CANDIDATE.get_or_init(|| {
        generate_methyl(&source_molecule(), spec(), GenerationOptions::default()).expect(
            "canonical C1 ethane generation must succeed before evaluating its fixed candidate",
        )
    })
}

fn field() -> Vec3 {
    Vec3::new(0.0013, -0.0009, 0.0017)
}
fn origin() -> Vec3 {
    Vec3::new(0.11, 0.07, -0.09)
}

fn baseline() -> &'static ReducedEvaluation {
    static BASELINE: OnceLock<ReducedEvaluation> = OnceLock::new();
    BASELINE.get_or_init(|| {
        evaluate_candidate(
            candidate(),
            &candidate().geometry_map.reference_coordinates(),
            field(),
            origin(),
        )
        .expect("fixed candidate baseline SCF must converge")
    })
}

fn shifted(vector: Vec3, axis: usize, amount: f64) -> Vec3 {
    let mut coordinates = vector.to_array();
    coordinates[axis] += amount;
    Vec3::new(coordinates[0], coordinates[1], coordinates[2])
}

#[test]
fn fixed_candidate_smoke_accounting_and_roundtrip() {
    let model = candidate();
    let value = baseline();
    assert_eq!(
        (
            value.full_electrons,
            value.frozen_electrons,
            value.active_electrons
        ),
        (18, 8, 10)
    );
    assert_eq!(model.electron_ledger.fragments[0].effective_ionic_charge, 1);
    assert!(value.physical_total_charge.abs() < 1e-12);
    assert!((value.active_density_trace - 10.0).abs() < 1e-9);
    assert_eq!(value.ao_count, 9);
    assert_eq!(value.occupied_spatial_orbitals, 5);
    assert_eq!(value.response_variables, 20);
    assert!(value.grid_points > 0 && value.scf_iterations > 0);
    assert!(value.scf_orbital_residual <= 1e-11);
    assert!(value.energy.is_finite());
    assert!((value.energy - value.electronic_energy - value.classical_energy).abs() < 1e-10);
    assert_eq!(
        value.fragment_internal_energy,
        model.internal_energy_reference
    );
    assert_eq!(
        value.energy,
        value.energy_without_internal_reference + value.fragment_internal_energy
    );
    assert_eq!(
        value.classical_energy,
        value.classical_energy_without_internal_reference + value.fragment_internal_energy
    );
    assert_eq!(value.gradient.len(), 5);
    assert_eq!(value.forces.len(), 5);
    for (force, gradient) in value.forces.iter().zip(&value.gradient) {
        assert!((*force + *gradient).norm() < 1e-14);
    }
    let sum = value.gradient.iter().fold(Vec3::zero(), |sum, &g| sum + g);
    assert!((sum + field() * value.physical_total_charge).norm() < 1e-7);
    assert!(
        (value.model_dipole - value.permanent_ionic_dipole - value.active_electronic_dipole).norm()
            < 1e-12
    );
    assert_eq!(value.candidate_status, "candidate-not-chemically-validated");
    assert_eq!(value.shape_status, "not-generated");
    assert_eq!(
        value.response_status,
        "active-only-no-internal-polarization"
    );
    let coordinates = model.geometry_map.reference_coordinates();
    let expanded = model.geometry_map.expand(&coordinates).unwrap();
    for (actual, expected) in expanded.atoms.iter().zip(&source_molecule().atoms) {
        assert_eq!(actual.z, expected.z);
        assert!((actual.position - expected.position).norm() < 1e-13);
    }
    let saved = model.to_json().unwrap();
    let restored = ReducedFragment::from_json(&saved).unwrap();
    assert_eq!(saved, restored.to_json().unwrap());
    let repeated = evaluate_candidate(&restored, &coordinates, field(), origin()).unwrap();
    assert!((repeated.energy - value.energy).abs() < 1e-11);
    assert!((repeated.model_dipole - value.model_dipole).norm() < 1e-10);
    for (a, b) in repeated.gradient.iter().zip(&value.gradient) {
        assert!((*a - *b).norm() < 1e-9);
    }
    use dft_rs::driver::{Controls, System, Value};
    let system = System::molecule(
        model
            .spec
            .retained_atoms
            .iter()
            .map(|&i| model.source.atoms[i].z)
            .chain([1])
            .collect(),
        coordinates.iter().map(|v| v.to_array()).collect(),
    );
    let controls = Controls::from_value(
        &Value::object()
            .with("fragment_mode", "evaluate")
            .with("fragment_model_json", saved)
            .with("field", field())
            .with("charge_origin", origin())
            .with("cpu_threads", 1),
    )
    .unwrap();
    let report = dft_rs::driver::run("continuous-species", &system, &controls).unwrap();
    for (name, expected) in [
        (
            "energy_without_internal_reference",
            value.energy_without_internal_reference,
        ),
        (
            "classical_energy_without_internal_reference",
            value.classical_energy_without_internal_reference,
        ),
        ("fragment_internal_energy", value.fragment_internal_energy),
    ] {
        assert!((report.get(name).unwrap().as_f64().unwrap() - expected).abs() < 1e-11);
        assert_eq!(
            report.get("units").unwrap().get(name).unwrap().as_str(),
            Some("hartree")
        );
    }
    assert_eq!(
        report.get("scf_energy_convention").unwrap().as_str(),
        Some(dft_rs::driver::continuous_species::FRAGMENT_SCF_ENERGY_CONVENTION)
    );
    println!("fixed candidate fingerprint={} E={:.16e} fullNe={} frozenNe={} activeNe={} Q={} activeTrace={:.16e} AO={} occupied={} responseVariables={} grid={} SCFresidual={:.9e} gap={:?} status={} shape={} response={}",
        model.construction_fingerprint,value.energy,value.full_electrons,value.frozen_electrons,
        value.active_electrons,value.physical_total_charge,value.active_density_trace,value.ao_count,
        value.occupied_spatial_orbitals,value.response_variables,value.grid_points,value.scf_orbital_residual,
        value.frontier_gap,value.candidate_status,value.shape_status,value.response_status);
}

#[test]
fn all_fifteen_reduced_nuclear_gradient_components_have_four_step_energy_fits() {
    let model = candidate();
    let coordinates = model.geometry_map.reference_coordinates();
    let analytic = &baseline().gradient;
    assert_eq!(3 * coordinates.len(), 15);
    let names: Vec<_> = (0..15)
        .map(|i| format!("fixed reduced nuclear gradient[{},{}]", i / 3, i % 3))
        .collect();
    let mut cache = BTreeMap::<(usize, u64), f64>::new();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &STEPS,
        |steps| {
            let mut errors = vec![[0.0; 4]; 15];
            for component in 0..15 {
                let atom = component / 3;
                let axis = component % 3;
                let mut estimates = [0.0; 4];
                for (k, &h) in steps.iter().enumerate() {
                    let mut energy = |amount: f64| {
                        *cache.entry((component, amount.to_bits())).or_insert_with(|| {
                    let mut displaced = coordinates.clone();
                    displaced[atom] = shifted(displaced[atom], axis, amount);
                    evaluate_candidate(model, &displaced, field(), origin())
                        .unwrap_or_else(|error| panic!("nuclear component {component}, displacement {amount:.16e}: {error}")).energy
                })
                    };
                    estimates[k] = (energy(h) - energy(-h)) / (2.0 * h);
                    errors[component][k] = (estimates[k] - analytic[atom].get(axis)).abs();
                }
                let richardson = (4.0 * estimates[3] - estimates[2]) / 3.0;
                println!("{} interval={steps:?} finest_h={:.9e} absolute_error={:.16e} Richardson_error={:.16e} units=hartree/bohr", names[component],steps[3],errors[component][3],(richardson-analytic[atom].get(axis)).abs());
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("every reduced nuclear gradient component", &failures);
    assert_eq!(model.to_json().unwrap(), candidate().to_json().unwrap());
}

#[test]
fn all_three_model_dipole_components_have_four_step_field_energy_fits() {
    let model = candidate();
    let coordinates = model.geometry_map.reference_coordinates();
    let analytic = baseline().model_dipole;
    let names: Vec<_> = (0..3)
        .map(|axis| format!("fixed reduced -E_F[{axis}]"))
        .collect();
    let mut cache = BTreeMap::<(usize, u64), f64>::new();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &STEPS,
        |steps| {
            let mut errors = vec![[0.0; 4]; 3];
            for axis in 0..3 {
                let mut estimates = [0.0; 4];
                for (k, &h) in steps.iter().enumerate() {
                    let mut energy = |amount: f64| {
                        *cache.entry((axis, amount.to_bits())).or_insert_with(|| {
                            evaluate_candidate(
                                model,
                                &coordinates,
                                shifted(field(), axis, amount),
                                origin(),
                            )
                            .unwrap_or_else(|error| {
                                panic!("field axis {axis}, displacement {amount:.16e}: {error}")
                            })
                            .energy
                        })
                    };
                    estimates[k] = -(energy(h) - energy(-h)) / (2.0 * h);
                    errors[axis][k] = (estimates[k] - analytic.get(axis)).abs();
                }
                let richardson = (4.0 * estimates[3] - estimates[2]) / 3.0;
                println!("{} interval={steps:?} finest_h={:.9e} absolute_error={:.16e} Richardson_error={:.16e} units=e*bohr", names[axis],steps[3],errors[axis][3],(richardson-analytic.get(axis)).abs());
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("every reduced static-model dipole component", &failures);
}
