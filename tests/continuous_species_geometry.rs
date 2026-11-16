// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

pub use dft_rs::{math, pseudo_sub, system, DftError, Result};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "../src/driver/continuous_species/frame.rs"]
mod frame;
#[path = "../src/driver/continuous_species/steric.rs"]
mod steric;
use frame::AttachedFrame;
use math::Vec3;
use steric::{steric_overlap, StericAtom};

fn positions() -> Vec<Vec3> {
    vec![
        Vec3::new(0.13, -0.21, 0.17),
        Vec3::new(1.34, 0.82, -0.23),
        Vec3::new(-0.74, 1.13, 0.91),
    ]
}

fn geometry(points: &[Vec3]) -> system::Molecule {
    system::Molecule::new(
        points
            .iter()
            .zip([8, 1, 1])
            .map(|(&position, z)| system::Atom { z, position })
            .collect(),
    )
}

fn shifted(point: Vec3, axis: usize, delta: f64) -> Vec3 {
    let mut xyz = point.to_array();
    xyz[axis] += delta;
    Vec3::new(xyz[0], xyz[1], xyz[2])
}

fn attached() -> AttachedFrame {
    AttachedFrame {
        anchors: [0, 1, 2],
        offset: Vec3::new(0.27, -0.19, 0.33),
        min_anchor_distance: 1e-8,
        min_sine_angle: 1e-6,
    }
}

#[test]
fn attached_frame_all_27_jacobian_entries_four_step_fits() {
    let positions = positions();
    let frame = attached();
    let value = frame.evaluate(&geometry(&positions)).unwrap();
    let mut failures = Vec::new();
    for atom in 0..3 {
        for axis in 0..3 {
            let names: Vec<_> = (0..3)
                .map(|i| format!("attached-frame J[{atom},{i},{axis}]"))
                .collect();
            components::fit_components(
                &names,
                &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
                |steps| {
                    let mut errors = vec![[0.0; 4]; 3];
                    for (k, &h) in steps.iter().enumerate() {
                        let evaluate = |delta| {
                            let mut p = positions.clone();
                            p[atom] = shifted(p[atom], axis, delta);
                            frame.evaluate(&geometry(&p)).unwrap().center
                        };
                        let finite = (evaluate(h) - evaluate(-h)) * (0.5 / h);
                        for i in 0..3 {
                            errors[i][k] =
                                (finite.get(i) - value.atom_jacobians[atom][i][axis]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("every attached-frame Jacobian entry", &failures);
}

#[test]
fn frame_covariance_and_degeneracy_rejections() {
    let p = positions();
    let frame = attached();
    let point = frame.evaluate(&geometry(&p)).unwrap();
    let translation = Vec3::new(0.31, -0.42, 0.23);
    let shifted: Vec<_> = p.iter().map(|&x| x + translation).collect();
    assert!(
        (frame.evaluate(&geometry(&shifted)).unwrap().center - point.center - translation).norm()
            < 1e-14
    );
    let rotate = |v: Vec3| Vec3::new(v.y, v.z, v.x);
    let rotated: Vec<_> = p.iter().copied().map(rotate).collect();
    assert!(
        (frame.evaluate(&geometry(&rotated)).unwrap().center - rotate(point.center)).norm() < 1e-14
    );
    for i in 0..3 {
        for j in 0..3 {
            let sum = point.atom_jacobians.iter().map(|a| a[i][j]).sum::<f64>();
            assert!((sum - if i == j { 1.0 } else { 0.0 }).abs() < 1e-14);
        }
    }
    let mut coincident = p.clone();
    coincident[1] = coincident[0];
    assert!(frame.evaluate(&geometry(&coincident)).is_err());
    let mut collinear = p.clone();
    collinear[2] = p[0] + (p[1] - p[0]) * 0.3;
    assert!(frame.evaluate(&geometry(&collinear)).is_err());
    assert!(AttachedFrame {
        anchors: [0, 0, 2],
        ..attached()
    }
    .evaluate(&geometry(&p))
    .is_err());
}

fn environment(p: &[Vec3]) -> Vec<StericAtom> {
    p.iter()
        .enumerate()
        .map(|(i, &position)| StericAtom {
            position,
            exponent: [0.47, 0.81, 0.59][i],
            hardness: [0.31, 0.73, 0.23][i],
        })
        .collect()
}

#[test]
fn steric_every_nonstructural_center_force_component_four_step_fits() {
    let p = positions();
    let center = Vec3::new(0.37, -0.29, 0.83);
    let mask = [true, false, false];
    let value = steric_overlap(center, 0.63, &environment(&p), &mask).unwrap();
    assert_eq!(value.atom_gradients[0], Vec3::zero());
    let sum = value
        .atom_gradients
        .iter()
        .fold(value.center_gradient, |sum, &g| sum + g);
    assert!(sum.norm() < 1e-14);
    let mut failures = Vec::new();
    for site in 1..=3 {
        let names: Vec<_> = (0..3)
            .map(|i| format!("steric gradient[{site},{i}]"))
            .collect();
        components::fit_components(
            &names,
            &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
            |steps| {
                let mut errors = vec![[0.0; 4]; 3];
                for (k, &h) in steps.iter().enumerate() {
                    for axis in 0..3 {
                        let energy = |delta| {
                            let mut moved = p.clone();
                            let mut c = center;
                            if site == 3 {
                                c = shifted(c, axis, delta);
                            } else {
                                moved[site] = shifted(moved[site], axis, delta);
                            }
                            steric_overlap(c, 0.63, &environment(&moved), &mask)
                                .unwrap()
                                .value
                        };
                        let exact = if site == 3 {
                            value.center_gradient
                        } else {
                            value.atom_gradients[site]
                        };
                        errors[axis][k] =
                            ((energy(h) - energy(-h)) / (2.0 * h) - exact.get(axis)).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("steric every active gradient component", &failures);
    assert!(
        steric_overlap(center, 0.63, &environment(&p), &[true, true, true])
            .unwrap()
            .value
            == 0.0
    );
    assert!(steric_overlap(center, 0.63, &environment(&p), &[]).is_err());
}
