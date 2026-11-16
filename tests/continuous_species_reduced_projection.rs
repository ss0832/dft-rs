// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! A generator/projection regression, not a chemical-accuracy acceptance claim.
use dft_rs::driver::continuous_species::reduced::evaluate_candidate;
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{
    generate_methyl, GenerationOptions, ProjectionRule, ReducedFragment,
};
use dft_rs::pseudo_sub::reduction_basis::OneCenterBasisTemplate;
use dft_rs::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec};
use dft_rs::system::{Atom, Molecule};

fn construction() -> (Molecule, FragmentSpec) {
    let points = [
        [-1.46, -0.14, 0.23],
        [-2.13, 1.69, 0.68],
        [-2.28, -1.57, 1.31],
        [-2.19, -0.43, -1.65],
        [1.35, 0.17, -0.12],
        [2.05, 2.08, 0.16],
        [2.10, -0.92, 1.45],
        [2.01, -0.84, -1.67],
    ];
    let molecule = Molecule::new(
        points
            .into_iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i == 0 || i == 4 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    );
    let spec = FragmentSpec {
        explicit_atoms: vec![4, 5, 6, 7],
        retained_atoms: vec![0, 1, 2, 3],
        port: AttachmentPort {
            fragment_atom: 4,
            host_atom: 0,
            reference_atom: 1,
        },
        single_bonds: vec![[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]],
        electronic_centers: 1,
    };
    (molecule, spec)
}

#[test]
fn reduced_metric_generation_keeps_one_reference_and_the_fixed_ao_contract() {
    let (source, spec) = construction();
    let mut options = GenerationOptions {
        basis_template: OneCenterBasisTemplate::CarbonCoreValenceSp,
        projection_rule: ProjectionRule::ReducedAoMetric,
        electrostatic_weight: 10000.0,
        ..Default::default()
    };
    // A declared construction protocol only. No held-out molecular data or
    // target curve energies enter these existing electrostatic projection rows.
    for &atom in &spec.retained_atoms {
        for shift in [
            Vec3::zero(),
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::new(-0.3, 0.0, 0.0),
            Vec3::new(0.0, 0.3, 0.0),
            Vec3::new(0.0, -0.3, 0.0),
            Vec3::new(0.0, 0.0, 0.3),
            Vec3::new(0.0, 0.0, -0.3),
        ] {
            options
                .electrostatic_points
                .push(source.atoms[atom].position + shift);
        }
    }
    let candidate = generate_methyl(&source, spec, options).unwrap();
    let encoded = candidate.to_json().unwrap();
    let restored = ReducedFragment::from_json(&encoded).unwrap();
    assert_eq!(restored.to_json().unwrap(), encoded);
    assert_eq!(
        restored.options.projection_rule,
        ProjectionRule::ReducedAoMetric
    );
    assert_eq!(candidate.electron_ledger.active_electron_count, 10);
    assert_eq!(candidate.diagnostics.reduced_ao_count, 13);
    assert!(candidate.diagnostics.sigma_charge_defect < 1e-12);
    assert!(candidate.diagnostics.sigma_dipole_defect < 1e-11);
    assert!(candidate.diagnostics.reference_energy_reconstruction_error < 1e-12);
    let reference = candidate.geometry_map.reference_coordinates();
    let center = candidate.geometry_map.electronic_center_index().unwrap();
    let direction = (reference[center] - reference[0]) / (reference[center] - reference[0]).norm();
    let mut energies = Vec::new();
    for offset in [-0.1, 0.0, 0.1] {
        let mut coordinates = reference.clone();
        coordinates[center] += direction * offset;
        let value =
            evaluate_candidate(&candidate, &coordinates, Vec3::zero(), Vec3::zero()).unwrap();
        assert_eq!(value.ao_count, 13);
        assert_eq!(value.active_electrons, 10);
        assert!((value.active_density_trace - 10.0).abs() < 1e-9);
        energies.push(value.energy_without_internal_reference);
        println!(
            "reduced-AO construction offset={offset:.8} energy_without_C={:.16e} residual={:.12e}",
            value.energy_without_internal_reference, value.scf_orbital_residual
        );
    }
    // Independent pre-integration projection harness: direct cross-overlap
    // contraction and the existing few-center consumer supplied these values.
    let expected = [0.00048727625781, 0.0, -0.001776155365355];
    for (i, &energy) in energies.iter().enumerate() {
        assert!((energy - energies[1] - expected[i]).abs() < 1e-8);
    }
    // These values still fail the independent all-atom chemical accuracy gate.
    // A passing implementation regression must not promote the model status.
    assert_eq!(
        candidate.candidate_status(),
        "candidate-not-chemically-validated"
    );
    assert_eq!(candidate.to_json().unwrap(), encoded);
}

#[test]
fn reduced_metric_refuses_a_noncontained_ao_template() {
    let (source, spec) = construction();
    let options = GenerationOptions {
        projection_rule: ProjectionRule::ReducedAoMetric,
        ..Default::default()
    };
    assert!(generate_methyl(&source, spec, options)
        .unwrap_err()
        .to_string()
        .contains("carbon AO subset"));
    assert_eq!(
        ProjectionRule::from_id("reduced-ao-metric-v1").unwrap(),
        ProjectionRule::ReducedAoMetric
    );
}
