// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Prepared common-mode contract gates; candidate accuracy is tested separately.
use dft_rs::driver::continuous_species::fragment_driver;
pub use dft_rs::driver::continuous_species::reduced;
pub use dft_rs::{driver, math, pseudo_sub, system, DftError, Result};

use driver::{Controls, System, ToValue, Value};
use math::Vec3;
use pseudo_sub::reduction::{generate_methyl, GenerationOptions, ReducedFragment};
use pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec};
use std::sync::OnceLock;
use system::{Atom, Molecule};

fn explicit() -> System {
    System::molecule(
        vec![6, 1, 1, 1, 6, 1, 1, 1],
        vec![
            [-1.46, -0.14, 0.23],
            [-2.13, 1.69, 0.68],
            [-2.28, -1.57, 1.31],
            [-2.19, -0.43, -1.65],
            [1.35, 0.17, -0.12],
            [2.05, 2.08, 0.16],
            [2.10, -0.92, 1.45],
            [2.01, -0.84, -1.67],
        ],
    )
}

fn spec() -> Value {
    Value::object()
        .with("explicit_atoms", vec![4_usize, 5, 6, 7])
        .with("retained_atoms", vec![0_usize, 1, 2, 3])
        .with("port", [4_usize, 0, 1])
        .with(
            "single_bonds",
            vec![[0_usize, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]],
        )
        .with("electronic_centers", 1_usize)
}

fn generate_controls(spec: Value) -> Controls {
    Controls::from_value(
        &Value::object()
            .with("fragment_mode", "generate")
            .with("fragment_spec", spec),
    )
    .unwrap()
}

fn generated() -> &'static Value {
    static REPORT: OnceLock<Value> = OnceLock::new();
    REPORT.get_or_init(|| fragment_driver::run(&explicit(), &generate_controls(spec())).unwrap())
}

fn candidate() -> ReducedFragment {
    ReducedFragment::from_json(generated().get("model_json").unwrap().as_str().unwrap()).unwrap()
}

fn reduced_system(model: &ReducedFragment) -> System {
    let coordinates = model.geometry_map.reference_coordinates();
    System::molecule(
        vec![6, 1, 1, 1, 1],
        coordinates.iter().map(|r| r.to_array()).collect(),
    )
}

fn evaluation(model: Value) -> Controls {
    Controls::from_value(
        &Value::object()
            .with("fragment_mode", "evaluate")
            .with("fragment_model_json", model)
            .with("field", [0.0013, -0.0017, 0.0009])
            .with("charge_origin", [0.11, -0.07, 0.19]),
    )
    .unwrap()
}

#[test]
fn generation_matches_direct_rust_and_saves_physical_ledger() {
    let report = generated();
    let model = candidate();
    let source = explicit();
    let molecule = Molecule::new(
        source
            .numbers
            .iter()
            .zip(&source.positions)
            .map(|(&z, r)| Atom {
                z,
                position: Vec3::new(r[0], r[1], r[2]),
            })
            .collect(),
    );
    let direct = generate_methyl(
        &molecule,
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
        },
        GenerationOptions::default(),
    )
    .unwrap();
    assert_eq!(model.to_json().unwrap(), direct.to_json().unwrap());
    let counts = report.get("counts").unwrap();
    assert_eq!(counts.get("full_electrons").unwrap().as_i64(), Some(18));
    assert_eq!(counts.get("frozen_electrons").unwrap().as_i64(), Some(8));
    assert_eq!(counts.get("active_electrons").unwrap().as_i64(), Some(10));
    assert_eq!(model.baseline_ionic_charge, 1.0);
    assert_eq!(model.source.atoms[4].z, 6);
    assert_eq!(
        report.get("candidate_status").unwrap().as_str(),
        Some("candidate-not-chemically-validated")
    );
    assert!(report.get("model_json").unwrap().as_str().is_some());
    let mut checked = report.clone();
    assert!(driver::units::complete(&mut checked).is_empty());
}

#[test]
fn text_object_and_common_driver_match_direct_evaluation() {
    let model = candidate();
    let system = reduced_system(&model);
    let text = Value::Str(model.to_json().unwrap());
    let report = fragment_driver::run(&system, &evaluation(text)).unwrap();
    let common = driver::run(
        "continuous-species",
        &system,
        &evaluation(model.to_value().unwrap()),
    )
    .unwrap();
    let direct = reduced::evaluate_candidate(
        &model,
        &model.geometry_map.reference_coordinates(),
        Vec3::new(0.0013, -0.0017, 0.0009),
        Vec3::new(0.11, -0.07, 0.19),
    )
    .unwrap();
    assert!((report.get("energy").unwrap().as_f64().unwrap() - direct.energy).abs() < 1e-12);
    assert_eq!(report.get("gradient"), Some(&direct.gradient.to_value()));
    assert_eq!(report.get("forces"), Some(&direct.forces.to_value()));
    assert_eq!(report.get("dipole"), Some(&direct.model_dipole.to_value()));
    for key in [
        "energy",
        "gradient",
        "forces",
        "dipole",
        "counts",
        "candidate_status",
        "response_status",
    ] {
        assert_eq!(report.get(key), common.get(key), "{key}");
    }
    assert!(
        (report
            .get("active_density_trace")
            .unwrap()
            .as_f64()
            .unwrap()
            - 10.0)
            .abs()
            < 1e-9
    );
    assert_eq!(
        report.get("shape_status").unwrap().as_str(),
        Some("not-generated")
    );
    assert_eq!(
        report.get("response_status").unwrap().as_str(),
        Some("active-only-no-internal-polarization")
    );
    assert!(report.get("polarizability").is_none());
    let mut checked = report;
    assert!(driver::units::complete(&mut checked).is_empty());
}

#[test]
fn saved_sector_labels_scope_and_unknown_controls_are_rejected() {
    let model = candidate();
    let system = reduced_system(&model);
    for (name, value) in [
        ("charge", Value::Int(1)),
        ("multiplicity", Value::Int(3)),
        ("xc", Value::Str("lda".into())),
        ("basis", Value::Str("sto-3g".into())),
        ("grid_level", Value::Int(2)),
        ("ecp", Value::Bool(true)),
        ("frequency", Value::Float(0.1)),
        ("unknown_key", Value::Bool(true)),
    ] {
        let mut controls = evaluation(model.to_value().unwrap());
        controls.insert(name, value).unwrap();
        assert!(
            fragment_driver::run(&system, &controls).is_err(),
            "accepted {name}"
        );
    }
    let mut wrong_labels = system.clone();
    wrong_labels.numbers[4] = 6;
    assert!(fragment_driver::run(&wrong_labels, &evaluation(model.to_value().unwrap())).is_err());
    let mut periodic = system.clone();
    periodic.pbc = [true, false, false];
    assert!(fragment_driver::run(&periodic, &evaluation(model.to_value().unwrap())).is_err());
    let mut invalid = model.to_value().unwrap();
    invalid.set("model_version", "future-invalid");
    assert!(fragment_driver::run(&system, &evaluation(invalid)).is_err());
    let mut invalid = model.to_value().unwrap();
    invalid.set("surprise", 1_usize);
    assert!(fragment_driver::run(&system, &evaluation(invalid)).is_err());
}

#[test]
fn legacy_default_projection_rule_is_the_only_optional_saved_field() {
    let model = candidate();
    let system = reduced_system(&model);
    let mut legacy = model.to_value().unwrap();
    let Value::Object(root) = &mut legacy else {
        panic!("model object")
    };
    let (_, Value::Object(options)) = root.iter_mut().find(|(key, _)| key == "options").unwrap()
    else {
        panic!("options object")
    };
    let old_len = options.len();
    options.retain(|(key, _)| key != "projection_rule");
    assert_eq!(options.len() + 1, old_len);
    let restored = ReducedFragment::from_value(&legacy).unwrap();
    assert_eq!(restored.options.projection_rule.id(), "full-ao-metric-v1");
    let result = fragment_driver::run(&system, &evaluation(Value::Str(legacy.to_json()))).unwrap();
    assert!(result.get("energy").unwrap().as_f64().unwrap().is_finite());
    assert_eq!(
        result
            .get("counts")
            .unwrap()
            .get("active_electrons")
            .unwrap()
            .as_i64(),
        Some(10)
    );
    // A second missing option must still fail; an unknown option cannot fill
    // its place and exploit equal object lengths.
    let Value::Object(root) = &mut legacy else {
        unreachable!()
    };
    let (_, Value::Object(options)) = root.iter_mut().find(|(key, _)| key == "options").unwrap()
    else {
        unreachable!()
    };
    options.retain(|(key, _)| key != "functional");
    options.push(("unknown_replacement".into(), Value::Str("pbe".into())));
    assert!(fragment_driver::run(&system, &evaluation(legacy)).is_err());
}

#[test]
fn malformed_generation_contracts_fail_before_reference_work() {
    for value in [Value::Null, Value::Str("[]".into()), Value::Str("{".into())] {
        assert!(fragment_driver::run(&explicit(), &generate_controls(value)).is_err());
    }
    let mut unknown = spec();
    unknown.set("surprise", true);
    assert!(fragment_driver::run(&explicit(), &generate_controls(unknown)).is_err());
    let mut unsupported = generate_controls(spec());
    unsupported
        .insert("fragment_basis_template", Value::Str("unknown".into()))
        .unwrap();
    assert!(fragment_driver::run(&explicit(), &unsupported).is_err());
    let mut unsupported = generate_controls(spec());
    unsupported
        .insert("fragment_projection_rule", Value::Str("unknown".into()))
        .unwrap();
    assert!(fragment_driver::run(&explicit(), &unsupported).is_err());
    let mut unsupported = generate_controls(spec());
    unsupported.insert("frequency", Value::Float(0.1)).unwrap();
    assert!(fragment_driver::run(&explicit(), &unsupported).is_err());
}
