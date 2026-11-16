// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Binding and held-out scaffold fixtures. No chemical accuracy is asserted.
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{generate_methyl, GenerationOptions, ReducedFragment};
use dft_rs::pseudo_sub::reduction_application::bind_methyl;
use dft_rs::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec, RigidExpansion};
use dft_rs::system::{Atom, Molecule};
use std::sync::OnceLock;

fn molecule(labels: &[u8], positions: &[[f64; 3]]) -> Molecule {
    assert_eq!(labels.len(), positions.len());
    Molecule::new(
        labels
            .iter()
            .zip(positions)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

fn original() -> (Molecule, FragmentSpec) {
    let source = molecule(
        &[6, 1, 1, 1, 6, 1, 1, 1],
        &[
            [-1.46, -0.14, 0.23],
            [-2.13, 1.69, 0.68],
            [-2.28, -1.57, 1.31],
            [-2.19, -0.43, -1.65],
            [1.35, 0.17, -0.12],
            [2.05, 2.08, 0.16],
            [2.10, -0.92, 1.45],
            [2.01, -0.84, -1.67],
        ],
    );
    (
        source,
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
    )
}

fn candidate() -> &'static ReducedFragment {
    static CANDIDATE: OnceLock<ReducedFragment> = OnceLock::new();
    CANDIDATE.get_or_init(|| {
        let (source, spec) = original();
        generate_methyl(&source, spec, GenerationOptions::default()).unwrap()
    })
}

/// C3H8 sigma-connected scaffold, kept separate from ethane construction.
pub fn propane() -> (Molecule, FragmentSpec) {
    let source = molecule(
        &[6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1],
        &[
            [2.8, 0.1, 0.2],
            [0.0, 0.0, 0.0],
            [-1.0, 2.6, 0.3],
            [3.5, 1.8, 0.9],
            [3.6, -1.0, 1.6],
            [3.4, -0.8, -1.6],
            [-0.6, -0.8, 1.7],
            [-0.7, -0.9, -1.6],
            [-0.3, 3.5, 1.9],
            [-3.0, 2.7, 0.2],
            [-0.4, 3.6, -1.3],
        ],
    );
    (
        source,
        FragmentSpec {
            explicit_atoms: vec![0, 3, 4, 5],
            retained_atoms: vec![1, 2, 6, 7, 8, 9, 10],
            port: AttachmentPort {
                fragment_atom: 0,
                host_atom: 1,
                reference_atom: 2,
            },
            single_bonds: vec![
                [0, 1],
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 2],
                [1, 6],
                [1, 7],
                [2, 8],
                [2, 9],
                [2, 10],
            ],
            electronic_centers: 1,
        },
    )
}

/// Branched C4H10 sigma-connected scaffold, not construction data.
pub fn isobutane() -> (Molecule, FragmentSpec) {
    let source = molecule(
        &[6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[
            [2.8, 0.1, 0.2],
            [0.0, 0.0, 0.0],
            [-1.0, 2.6, 0.3],
            [-0.9, -1.4, 2.3],
            [3.5, 1.8, 0.9],
            [3.6, -1.0, 1.6],
            [3.4, -0.8, -1.6],
            [-0.3, 3.5, 1.9],
            [-3.0, 2.7, 0.2],
            [-0.4, 3.6, -1.3],
            [0.0, -3.2, 2.7],
            [-2.9, -1.7, 2.1],
            [-0.3, -0.2, 3.8],
            [0.0, -1.0, -1.8],
        ],
    );
    (
        source,
        FragmentSpec {
            explicit_atoms: vec![0, 4, 5, 6],
            retained_atoms: vec![1, 2, 3, 7, 8, 9, 10, 11, 12, 13],
            port: AttachmentPort {
                fragment_atom: 0,
                host_atom: 1,
                reference_atom: 2,
            },
            single_bonds: vec![
                [0, 1],
                [0, 4],
                [0, 5],
                [0, 6],
                [1, 2],
                [1, 3],
                [1, 13],
                [2, 7],
                [2, 8],
                [2, 9],
                [3, 10],
                [3, 11],
                [3, 12],
            ],
            electronic_centers: 1,
        },
    )
}

fn rotate(v: Vec3) -> Vec3 {
    let axis = Vec3::new(2.0, -1.0, 3.0) / 14.0_f64.sqrt();
    let (s, c) = 0.413_f64.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

#[test]
fn immutable_original_parameters_and_explicit_application_accounting() {
    let candidate = candidate();
    let bytes = candidate.to_json().unwrap();
    for (source, spec, full, active) in [
        {
            let (s, p) = propane();
            (s, p, 26, 18)
        },
        {
            let (s, p) = isobutane();
            (s, p, 34, 26)
        },
    ] {
        let original_input = format!("{source:?}");
        let bound = bind_methyl(candidate, &source, spec.clone()).unwrap();
        bound.validate().unwrap();
        assert_eq!(bound.original_model_json().as_bytes(), bytes.as_bytes());
        assert_eq!(
            bound.candidate().to_json().unwrap().as_bytes(),
            bytes.as_bytes()
        );
        assert_eq!(candidate.to_json().unwrap(), bytes);
        assert_eq!(format!("{:?}", bound.application_source()), original_input);
        assert_eq!(format!("{source:?}"), original_input);
        let ledger = bound.electron_ledger();
        assert_eq!(
            (
                ledger.full_electron_count,
                ledger.frozen_electron_count,
                ledger.active_electron_count
            ),
            (full, 8, active)
        );
        assert_eq!(ledger.fragments[0].effective_ionic_charge, 1);
        assert_eq!(ledger.reduced_total_charge, 0);
        assert!(active / 2 <= bound.reduced_ao_count());
        for ((_, actual), (_, original)) in bound
            .geometry_map()
            .internal_offsets()
            .iter()
            .zip(candidate.geometry_map.internal_offsets())
        {
            for axis in 0..3 {
                assert_eq!(actual.get(axis).to_bits(), original.get(axis).to_bits());
            }
        }
        let rigid = bound.rigid_reference_geometry();
        assert_eq!(
            rigid.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
            source.atoms.iter().map(|a| a.z).collect::<Vec<_>>()
        );
        assert_eq!(
            (rigid.charge, rigid.multiplicity),
            (source.charge, source.multiplicity)
        );
        for &i in &spec.retained_atoms {
            for axis in 0..3 {
                assert_eq!(
                    rigid.atoms[i].position.get(axis).to_bits(),
                    source.atoms[i].position.get(axis).to_bits()
                );
            }
        }
        assert!(spec
            .explicit_atoms
            .iter()
            .any(|&i| (rigid.atoms[i].position - source.atoms[i].position).norm() > 0.01));
        let expanded = bound
            .geometry_map()
            .expand(&bound.geometry_map().reference_coordinates())
            .unwrap();
        for (a, b) in expanded.atoms.iter().zip(&rigid.atoms) {
            assert!((a.position - b.position).norm() < 1e-13);
        }
    }
}

#[test]
fn application_rotation_translation_and_model_byte_identity() {
    let candidate = candidate();
    let (source, spec) = propane();
    let reference = bind_methyl(candidate, &source, spec.clone()).unwrap();
    let shift = Vec3::new(0.91, -1.37, 0.42);
    let mut transformed = source.clone();
    for atom in &mut transformed.atoms {
        atom.position = rotate(atom.position) + shift;
    }
    let changed = bind_methyl(candidate, &transformed, spec).unwrap();
    assert_eq!(
        reference.original_model_json(),
        changed.original_model_json()
    );
    assert_eq!(reference.fingerprints().model, changed.fingerprints().model);
    assert_ne!(
        reference.fingerprints().source,
        changed.fingerprints().source
    );
    assert_ne!(
        reference.fingerprints().application,
        changed.fingerprints().application
    );
    for (a, b) in reference
        .rigid_reference_geometry()
        .atoms
        .iter()
        .zip(&changed.rigid_reference_geometry().atoms)
    {
        assert!((rotate(a.position) + shift - b.position).norm() < 2e-13);
    }
    let y = reference.geometry_map().reference_coordinates();
    let transformed_y: Vec<_> = y.iter().map(|&v| rotate(v) + shift).collect();
    let a = reference.geometry_map().evaluate(&y).unwrap();
    let b = changed.geometry_map().evaluate(&transformed_y).unwrap();
    // J transforms as R J R^T, independently for every 3x3 coordinate block.
    for full in 0..source.len() {
        for reduced in 0..y.len() {
            for axis in 0..3 {
                let mut cart = [0.0; 3];
                cart[axis] = 1.0;
                let rotated_axis = rotate(Vec3::new(cart[0], cart[1], cart[2]));
                let original_column = Vec3::new(
                    a.jacobian[(3 * full, 3 * reduced + axis)],
                    a.jacobian[(3 * full + 1, 3 * reduced + axis)],
                    a.jacobian[(3 * full + 2, 3 * reduced + axis)],
                );
                let mut transformed_column = Vec3::zero();
                for k in 0..3 {
                    transformed_column += Vec3::new(
                        b.jacobian[(3 * full, 3 * reduced + k)],
                        b.jacobian[(3 * full + 1, 3 * reduced + k)],
                        b.jacobian[(3 * full + 2, 3 * reduced + k)],
                    ) * rotated_axis.get(k);
                }
                assert!((rotate(original_column) - transformed_column).norm() < 1e-12);
            }
        }
    }
}

#[test]
fn exact_offsets_refuse_malformed_maps_and_application_charts() {
    let candidate = candidate();
    let (mut source, mut spec) = propane();
    source.charge = 1;
    assert!(bind_methyl(candidate, &source, spec.clone()).is_err());
    source.charge = 0;
    source.multiplicity = 3;
    assert!(bind_methyl(candidate, &source, spec.clone()).is_err());
    source.multiplicity = 1;
    spec.electronic_centers = 2;
    assert!(bind_methyl(candidate, &source, spec.clone()).is_err());
    spec.electronic_centers = 1;
    let mut offsets: Vec<_> = spec
        .explicit_atoms
        .iter()
        .copied()
        .filter(|&i| i != spec.port.fragment_atom)
        .zip(
            candidate
                .geometry_map
                .internal_offsets()
                .iter()
                .map(|(_, x)| *x),
        )
        .collect();
    offsets.swap(0, 1);
    assert!(RigidExpansion::methyl_with_internal_offsets(&source, spec.clone(), &offsets).is_err());
    offsets.swap(0, 1);
    offsets[0].1 = Vec3::new(f64::NAN, 0.0, 0.0);
    assert!(RigidExpansion::methyl_with_internal_offsets(&source, spec.clone(), &offsets).is_err());
    source.atoms[spec.port.reference_atom].position = source.atoms[spec.port.host_atom].position;
    assert!(bind_methyl(candidate, &source, spec).is_err());
}
