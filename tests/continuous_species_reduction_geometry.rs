// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Independent rigid-map and generalized-force tests; no electronic fit.
pub use dft_rs::{linalg, math, pseudo_sub, system, DftError, Result};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::pseudo_sub::reduction_geometry;

use math::Vec3;
use reduction_geometry::{AttachmentPort, FragmentSpec, RigidExpansion};
use system::{Atom, Molecule};

fn molecule() -> Molecule {
    let positions = [
        [-0.4, 0.2, 0.1],
        [2.3, 0.8, -0.2],
        [-1.0, 1.6, 0.7],
        [-1.2, -0.7, 1.1],
        [-0.9, -0.5, -1.5],
        [2.7, 2.4, 0.9],
        [3.5, -0.2, 0.7],
        [2.9, 0.3, -2.1],
    ];
    Molecule::new(
        positions
            .iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i < 2 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

fn spec() -> FragmentSpec {
    FragmentSpec {
        explicit_atoms: vec![1, 5, 6, 7],
        retained_atoms: vec![0, 2, 3, 4],
        port: AttachmentPort {
            fragment_atom: 1,
            host_atom: 0,
            reference_atom: 2,
        },
        single_bonds: vec![[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]],
        electronic_centers: 1,
    }
}

fn map() -> RigidExpansion {
    RigidExpansion::methyl(&molecule(), spec()).unwrap()
}

fn shifted(point: Vec3, axis: usize, delta: f64) -> Vec3 {
    let mut p = point.to_array();
    p[axis] += delta;
    Vec3::new(p[0], p[1], p[2])
}

fn coordinates() -> Vec<Vec3> {
    let mut y = map().reference_coordinates();
    y[4] += Vec3::new(0.11, -0.07, 0.04);
    y[1] += Vec3::new(0.04, 0.03, 0.02);
    y
}

fn max_geometry_error(a: &Molecule, b: &Molecule) -> f64 {
    a.atoms
        .iter()
        .zip(&b.atoms)
        .map(|(a, b)| (a.position - b.position).norm())
        .fold(0., f64::max)
}

#[test]
fn reference_expansion_identity_and_bookkeeping_independent_geometry() {
    let reference = molecule();
    let expansion = map();
    assert_eq!(expansion.full_atom_count(), 8);
    assert_eq!(expansion.reduced_coordinate_count(), 5);
    assert_eq!(expansion.retained_atoms(), &[0, 2, 3, 4]);
    assert_eq!(expansion.fragment_atoms(), &[1, 5, 6, 7]);
    assert_eq!(expansion.electronic_center_index(), Some(4));
    assert_eq!(expansion.reference_molecule().len(), 8);
    assert_eq!(expansion.fragment_spec(), Some(&spec()));
    assert_eq!(expansion.internal_offsets().len(), 3);
    let restored = expansion
        .expand(&expansion.reference_coordinates())
        .unwrap();
    assert!(max_geometry_error(&reference, &restored) < 2e-14);
    assert_eq!(
        restored.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
        vec![6, 6, 1, 1, 1, 1, 1, 1]
    );
    let charged = reference.clone().with_charge(1).with_multiplicity(2);
    let charged_map = RigidExpansion::methyl(&charged, spec()).unwrap();
    let charged_result = charged_map.evaluate(&coordinates()).unwrap();
    let neutral_result = expansion.evaluate(&coordinates()).unwrap();
    assert!(max_geometry_error(&charged_result.molecule, &neutral_result.molecule) < 1e-14);
    assert!(
        charged_result
            .jacobian
            .max_abs_diff(&neutral_result.jacobian)
            < 1e-14
    );
    assert_eq!(charged_result.molecule.charge, 1);
    assert_eq!(charged_result.molecule.multiplicity, 2);
    let mut permuted_spec = spec();
    permuted_spec.retained_atoms = vec![4, 0, 3, 2];
    permuted_spec.explicit_atoms = vec![7, 1, 5, 6];
    let permuted = RigidExpansion::methyl(&reference, permuted_spec).unwrap();
    assert!(
        max_geometry_error(
            &permuted.expand(&permuted.reference_coordinates()).unwrap(),
            &reference
        ) < 2e-14
    );
    let forces: Vec<_> = (0..8)
        .map(|i| Vec3::new(0.2 + i as f64 * 0.03, -0.17 + i as f64 * 0.01, 0.11))
        .collect();
    let original = expansion
        .pullback_forces(&expansion.reference_coordinates(), &forces)
        .unwrap();
    let reordered = permuted
        .pullback_forces(&permuted.reference_coordinates(), &forces)
        .unwrap();
    for (new, old) in [3usize, 0, 2, 1, 4].into_iter().enumerate() {
        assert!((reordered[new] - original[old]).norm() < 1e-13);
    }
    // Ordinary identities keep all physical atoms, and add no electronic center.
    for z in [1, 9, 17] {
        let ordinary = Molecule::new(vec![Atom {
            z,
            position: Vec3::new(0.3, -0.4, 0.7),
        }]);
        let identity = RigidExpansion::identity(&ordinary).unwrap();
        assert_eq!(identity.electronic_center_index(), None);
        assert!(identity.fragment_atoms().is_empty());
        assert!(identity.fragment_spec().is_none());
        assert!(identity.internal_offsets().is_empty());
        assert!(identity
            .body_frame_kinematics(&identity.reference_coordinates())
            .unwrap()
            .is_none());
        let evaluated = identity.evaluate(&[Vec3::new(-0.2, 0.7, 0.9)]).unwrap();
        assert_eq!(evaluated.molecule.atoms[0].z, z);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(evaluated.jacobian[(i, j)], if i == j { 1. } else { 0. });
            }
        }
        let report = identity
            .force_report(
                &identity.reference_coordinates(),
                &[Vec3::new(0.2, 0.4, -0.1)],
            )
            .unwrap();
        assert_eq!(report.reduced_forces, [Vec3::new(0.2, 0.4, -0.1)]);
        assert!(report.fragment_translation_force.is_none());
        assert!(report.fragment_torque_body.is_none());
    }
}

#[test]
fn shared_body_frame_every_active_derivative_and_exact_center_blocks() {
    let expansion = map();
    let y = coordinates();
    let exact = expansion.body_frame_kinematics(&y).unwrap().unwrap();
    let anchors = [0usize, 4, 1];
    let entries: Vec<_> = (0..3)
        .flat_map(|column| {
            (0..3).flat_map(move |i| {
                anchors
                    .into_iter()
                    .filter(move |&a| column != 2 || a != 1)
                    .flat_map(move |a| (0..3).map(move |j| (column, i, a, j)))
            })
        })
        .collect();
    assert_eq!(entries.len(), 72);
    let names: Vec<_> = entries
        .iter()
        .map(|(c, i, a, j)| format!("body frame U[{c},{i}]/y[{a},{j}]"))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
        |steps| {
            let mut errors = vec![[0.; 4]; entries.len()];
            for (k, &h) in steps.iter().enumerate() {
                for &a in &anchors {
                    for j in 0..3 {
                        let mut p = y.clone();
                        p[a] = shifted(p[a], j, h);
                        let mut n = y.clone();
                        n[a] = shifted(n[a], j, -h);
                        let p = expansion.body_frame_kinematics(&p).unwrap().unwrap();
                        let n = expansion.body_frame_kinematics(&n).unwrap().unwrap();
                        for (kc, &(column, i, atom, axis)) in entries.iter().enumerate() {
                            if atom == a && axis == j {
                                errors[kc][k] = ((p.orientation.col[column].get(i)
                                    - n.orientation.col[column].get(i))
                                    / (2. * h)
                                    - exact.orientation_jacobian[(3 * column + i, 3 * a + j)])
                                    .abs();
                            }
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all72 active shared body-frame derivatives", &failures);
    for a in 0..y.len() {
        for axis in 0..3 {
            for i in 0..3 {
                assert_eq!(
                    exact.center_jacobian[(i, 3 * a + axis)],
                    if a == 4 && i == axis { 1. } else { 0. }
                );
            }
            for column in 0..3 {
                for i in 0..3 {
                    if !entries.contains(&(column, i, a, axis)) {
                        assert_eq!(
                            exact.orientation_jacobian[(3 * column + i, 3 * a + axis)],
                            0.
                        );
                        for h in [0.04, 0.02, 0.01, 0.005] {
                            let mut p = y.clone();
                            p[a] = shifted(p[a], axis, h);
                            let mut n = y.clone();
                            n[a] = shifted(n[a], axis, -h);
                            let p = expansion.body_frame_kinematics(&p).unwrap().unwrap();
                            let n = expansion.body_frame_kinematics(&n).unwrap().unwrap();
                            assert_eq!(
                                p.orientation.col[column].get(i) - n.orientation.col[column].get(i),
                                0.
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_nonstructural_expansion_jacobian_entry_four_step_fit() {
    let expansion = map();
    let y = coordinates();
    let exact = expansion.evaluate(&y).unwrap();
    let anchors = [0usize, 4, 1];
    let entries: Vec<_> = [5usize, 6, 7]
        .into_iter()
        .flat_map(|full| {
            (0..3).flat_map(move |i| {
                anchors
                    .into_iter()
                    .flat_map(move |reduced| (0..3).map(move |axis| (full, i, reduced, axis)))
            })
        })
        .collect();
    assert_eq!(entries.len(), 81);
    let names: Vec<_> = entries
        .iter()
        .map(|(a, i, b, j)| format!("expansion J[{a},{i};{b},{j}]"))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
        |steps| {
            let mut errors = vec![[0.; 4]; entries.len()];
            for (k, &h) in steps.iter().enumerate() {
                for &reduced in &anchors {
                    for axis in 0..3 {
                        let mut p = y.clone();
                        p[reduced] = shifted(p[reduced], axis, h);
                        let mut n = y.clone();
                        n[reduced] = shifted(n[reduced], axis, -h);
                        let p = expansion.expand(&p).unwrap();
                        let n = expansion.expand(&n).unwrap();
                        for (c, &(full, i, r, j)) in entries.iter().enumerate() {
                            if r == reduced && j == axis {
                                errors[c][k] = ((p.atoms[full].position.get(i)
                                    - n.atoms[full].position.get(i))
                                    / (2. * h)
                                    - exact.jacobian[(3 * full + i, 3 * reduced + axis)])
                                    .abs();
                            }
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all81 nonlinear expansion Jacobian components", &failures);
    // All other entries are exact identity/zero blocks fixed by topology,
    // not discarded because a fitted error happens to be small.
    for reduced in 0..y.len() {
        for axis in 0..3 {
            for h in [0.04, 0.02, 0.01, 0.005] {
                let mut p = y.clone();
                p[reduced] = shifted(p[reduced], axis, h);
                let mut n = y.clone();
                n[reduced] = shifted(n[reduced], axis, -h);
                let p = expansion.expand(&p).unwrap();
                let n = expansion.expand(&n).unwrap();
                for full in 0..8 {
                    for i in 0..3 {
                        if entries.contains(&(full, i, reduced, axis)) {
                            continue;
                        }
                        let expected = exact.jacobian[(3 * full + i, 3 * reduced + axis)];
                        assert!(expected == 0. || expected == 1.);
                        let fd = (p.atoms[full].position.get(i) - n.atoms[full].position.get(i))
                            / (2. * h);
                        assert!((fd - expected).abs() < 1e-12);
                    }
                }
            }
        }
    }
}

// Independent scalar energy on full physical coordinates and closed-form
// forces. It has no frame/Jacobian code and no reduction-specific parameters.
fn energy_forces(molecule: &Molecule) -> (f64, Vec<Vec3>) {
    let mut energy = 0.;
    let mut forces = vec![Vec3::zero(); molecule.len()];
    for (a, atom) in molecule.atoms.iter().enumerate() {
        let p = atom.position;
        let w = 0.07 + 0.013 * a as f64;
        let linear = Vec3::new(
            0.11 + 0.02 * a as f64,
            -0.17 + 0.01 * a as f64,
            0.23 + 0.007 * a as f64,
        );
        energy += w * p.x.powi(4)
            + 0.23 * p.x * p.y * p.z
            + 0.17 * p.y.powi(3)
            + 0.11 * p.z.powi(4)
            + linear.dot(p);
        forces[a] = Vec3::new(
            -(4. * w * p.x.powi(3) + 0.23 * p.y * p.z + linear.x),
            -(0.23 * p.x * p.z + 0.51 * p.y * p.y + linear.y),
            -(0.23 * p.x * p.y + 0.44 * p.z.powi(3) + linear.z),
        );
    }
    for a in 0..molecule.len() {
        for b in a + 1..molecule.len() {
            let r = molecule.atoms[a].position - molecule.atoms[b].position;
            let eta = 0.13 + 0.003 * (a + b) as f64;
            let pair = (0.3 + 0.017 * (a * b + 1) as f64) * (-eta * r.norm2()).exp();
            energy += pair;
            let force = r * (2. * eta * pair);
            forces[a] += force;
            forces[b] -= force;
        }
    }
    (energy, forces)
}

#[test]
fn every_generalized_cartesian_force_four_step_fit_and_anchor_chain() {
    let expansion = map();
    let y = coordinates();
    let full = expansion.expand(&y).unwrap();
    let (_, forces) = energy_forces(&full);
    let report = expansion.force_report(&y, &forces).unwrap();
    assert_eq!(report.cartesian_units, "hartree/bohr");
    assert_eq!(report.torque_units, "hartree/radian");
    assert!(
        (report.reduced_forces[1] - forces[2]).norm() > 1e-3,
        "removed atoms must transmit frame forces to retained anchors"
    );
    let names: Vec<_> = (0..y.len())
        .flat_map(|a| (0..3).map(move |i| format!("force pullback[{a},{i}]")))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
        |steps| {
            let mut errors = vec![[0.; 4]; names.len()];
            for a in 0..y.len() {
                for axis in 0..3 {
                    for (k, &h) in steps.iter().enumerate() {
                        let mut p = y.clone();
                        p[a] = shifted(p[a], axis, h);
                        let mut n = y.clone();
                        n[a] = shifted(n[a], axis, -h);
                        let e_p = energy_forces(&expansion.expand(&p).unwrap()).0;
                        let e_n = energy_forces(&expansion.expand(&n).unwrap()).0;
                        errors[3 * a + axis][k] =
                            (-(e_p - e_n) / (2. * h) - report.reduced_forces[a].get(axis)).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all reduced Cartesian force components", &failures);
    let expected = spec()
        .explicit_atoms
        .iter()
        .fold(Vec3::zero(), |sum, &a| sum + forces[a]);
    assert!((report.fragment_translation_force.unwrap() - expected).norm() < 1e-12);
}

fn rotate_axis(point: Vec3, axis: usize, angle: f64) -> Vec3 {
    let mut a = [0.; 3];
    a[axis] = 1.;
    let unit = Vec3::new(a[0], a[1], a[2]);
    point * angle.cos()
        + unit.cross(point) * angle.sin()
        + unit * (unit.dot(point) * (1. - angle.cos()))
}

#[test]
fn independent_rigid_orientation_torque_four_step_fit() {
    let expansion = map();
    let y = coordinates();
    let full = expansion.expand(&y).unwrap();
    let (_, forces) = energy_forces(&full);
    let report = expansion.force_report(&y, &forces).unwrap();
    let exact = report.fragment_torque_global.unwrap();
    let origin = full.atoms[1].position;
    let names: Vec<_> = (0..3)
        .map(|i| format!("rigid fragment torque[{i}]"))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
        |steps| {
            let mut errors = vec![[0.; 4]; 3];
            for axis in 0..3 {
                for (k, &h) in steps.iter().enumerate() {
                    let evaluate = |angle| {
                        let mut moved = full.clone();
                        for a in [1, 5, 6, 7] {
                            moved.atoms[a].position =
                                origin + rotate_axis(full.atoms[a].position - origin, axis, angle);
                        }
                        energy_forces(&moved).0
                    };
                    errors[axis][k] =
                        (-(evaluate(h) - evaluate(-h)) / (2. * h) - exact.get(axis)).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "three independent rigid fragment torque components",
        &failures,
    );
}

#[test]
fn rigid_covariance_internal_geometry_and_force_transforms() {
    let expansion = map();
    let y = coordinates();
    let full = expansion.expand(&y).unwrap();
    for a in [1, 5, 6, 7] {
        for b in [1, 5, 6, 7] {
            let original = (molecule().atoms[a].position - molecule().atoms[b].position).norm();
            assert!(
                ((full.atoms[a].position - full.atoms[b].position).norm() - original).abs() < 2e-14
            );
        }
    }
    let translation = Vec3::new(0.37, -0.29, 0.41);
    let translated: Vec<_> = y.iter().map(|&p| p + translation).collect();
    let shifted = expansion.expand(&translated).unwrap();
    for (a, b) in shifted.atoms.iter().zip(&full.atoms) {
        assert!((a.position - b.position - translation).norm() < 2e-14);
    }
    let rotate = |p: Vec3| Vec3::new(p.y, p.z, p.x);
    let rotated: Vec<_> = y.iter().copied().map(rotate).collect();
    let restored = expansion.expand(&rotated).unwrap();
    for (a, b) in restored.atoms.iter().zip(&full.atoms) {
        assert!((a.position - rotate(b.position)).norm() < 2e-14);
    }
    let forces: Vec<_> = (0..8)
        .map(|i| {
            Vec3::new(
                0.13 + 0.03 * i as f64,
                -0.21 + 0.01 * i as f64,
                0.17 - 0.02 * i as f64,
            )
        })
        .collect();
    let original = expansion.force_report(&y, &forces).unwrap();
    let rotated_forces: Vec<_> = forces.iter().copied().map(rotate).collect();
    let transformed = expansion.force_report(&rotated, &rotated_forces).unwrap();
    for (a, b) in transformed
        .reduced_forces
        .iter()
        .zip(&original.reduced_forces)
    {
        assert!((*a - rotate(*b)).norm() < 1e-13);
    }
    assert!(
        (transformed.fragment_torque_global.unwrap()
            - rotate(original.fragment_torque_global.unwrap()))
        .norm()
            < 1e-13
    );
    assert!(
        (transformed.fragment_torque_body.unwrap() - original.fragment_torque_body.unwrap()).norm()
            < 1e-13
    );
}

#[test]
fn malformed_topology_composition_and_frame_are_refused() {
    let reference = molecule();
    let mut bad = spec();
    bad.electronic_centers = 2;
    assert!(RigidExpansion::methyl(&reference, bad).is_err());
    let mut bad = spec();
    bad.explicit_atoms[3] = 5;
    assert!(RigidExpansion::methyl(&reference, bad).is_err());
    let mut bad = spec();
    bad.retained_atoms[0] = 1;
    assert!(RigidExpansion::methyl(&reference, bad).is_err());
    let mut bad = spec();
    bad.port.reference_atom = 5;
    assert!(RigidExpansion::methyl(&reference, bad).is_err());
    let mut bad = spec();
    bad.single_bonds.pop();
    assert!(RigidExpansion::methyl(&reference, bad).is_err());
    let mut bad = spec();
    bad.single_bonds.push([1, 0]);
    assert!(RigidExpansion::methyl(&reference, bad).is_err());
    let mut bad = reference.clone();
    bad.atoms[1].z = 8;
    assert!(RigidExpansion::methyl(&bad, spec()).is_err());
    let expansion = map();
    let mut y = coordinates();
    y[1] = y[0] + (y[4] - y[0]) * 0.37;
    assert!(expansion.expand(&y).is_err());
    assert!(expansion.evaluate(&[]).is_err());
    assert!(expansion.pullback_forces(&coordinates(), &[]).is_err());
    let mut bad = reference;
    bad.atoms[2].position.x = f64::NAN;
    assert!(RigidExpansion::identity(&bad).is_err());
}
