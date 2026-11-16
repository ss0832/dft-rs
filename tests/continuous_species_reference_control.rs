// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Fixed-reference controls: physical residual checks, no chemical acceptance.
use dft_rs::driver::continuous_species::reference_control::{
    diagnose_candidate_reference, ReferenceControlKind, ReferenceControlSeed,
    ReferenceControlStatus,
};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{generate_methyl, GenerationOptions};
use dft_rs::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec, RigidExpansion};
use dft_rs::system::{Atom, Molecule};
fn source() -> Molecule {
    let p = [
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
        p.iter()
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
fn check(options: GenerationOptions, label: &str) {
    let candidate = generate_methyl(&source(), spec(), options).unwrap();
    let saved = candidate.to_json().unwrap();
    let report = diagnose_candidate_reference(&candidate).unwrap();
    println!("{label} reference-control report={report:#?}");
    assert_eq!(
        (
            report.full_ao_count,
            report.full_electrons,
            report.frozen_electrons,
            report.active_electrons,
            report.integer_adapter_charge,
            report.physical_total_charge
        ),
        (16, 18, 8, 10, 8, 0)
    );
    assert_eq!(report.outcomes.len(), 6);
    assert!(report.exact_reference_reconstruction_error < 1e-10);
    assert!(report.projected_reference_reconstruction_error < 1e-10);
    for outcome in &report.outcomes {
        match &outcome.value {
            None => {
                assert_eq!(outcome.status, ReferenceControlStatus::Failed);
                assert!(outcome.error.is_some());
            }
            Some(v) => {
                assert!(v.energy.is_finite());
                assert!(v.metric_density_difference.is_finite());
                assert!(v.frozen_electron_contamination.is_finite());
                if outcome.status == ReferenceControlStatus::Converged {
                    assert!((v.active_trace - 10.0).abs() < 1e-9);
                    assert!(v.physical_orbital_residual <= 1e-11);
                }
            }
        }
        if outcome.kind == ReferenceControlKind::ExactConstrainedFrozen {
            assert_eq!(outcome.status, ReferenceControlStatus::Converged);
            let v = outcome.value.as_ref().unwrap();
            assert!(v.energy_difference_from_full.abs() < 1e-10);
            assert!(v.frozen_electron_contamination.abs() < 1e-9);
        }
        // A reference-seeded exact static control must retain the known
        // stationary active solution. Core-seed branch changes remain reported.
        if outcome.kind == ReferenceControlKind::UnprojectedStatic
            && outcome.seed == ReferenceControlSeed::ActiveReference
        {
            assert_eq!(outcome.status, ReferenceControlStatus::Converged);
            let v = outcome.value.as_ref().unwrap();
            assert!(v.energy_difference_from_full.abs() < 1e-9);
            assert!(v.metric_density_difference < 1e-7);
        }
    }
    assert_eq!(saved, candidate.to_json().unwrap());
}
#[test]
fn baseline_full_ao_reference_controls() {
    check(GenerationOptions::default(), "baseline");
}
#[test]
fn esp_full_ao_reference_controls() {
    let source = source();
    let map = RigidExpansion::methyl(&source, spec()).unwrap();
    let y = map.reference_coordinates();
    let frame = map.body_frame_kinematics(&y).unwrap().unwrap();
    let direction = (y[4] - y[0]) * (1.0 / (y[4] - y[0]).norm());
    let mut points: Vec<_> = (0..4).map(|i| source.atoms[i].position).collect();
    for atom in 0..4 {
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut shift = [0.0; 3];
                shift[axis] = sign * 0.05;
                points.push(source.atoms[atom].position + Vec3::new(shift[0], shift[1], shift[2]));
            }
        }
    }
    for r in [0.6, 0.9, 1.2, 1.6, 2.0, 2.4, 3.0] {
        points.push(frame.center - direction * r);
    }
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            for r in [1.1, 2.2] {
                points.push(frame.center + frame.orientation.col[axis] * (sign * r));
            }
        }
    }
    assert_eq!(points.len(), 47);
    check(
        GenerationOptions {
            electrostatic_points: points,
            electrostatic_weight: 1e6,
            ..GenerationOptions::default()
        },
        "fixed-ESP",
    );
}
