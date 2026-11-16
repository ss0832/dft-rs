// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear-derivative contracts for RI-J density fitting.

use dft_rs::basis::{data, MolBasis};
use dft_rs::gradient::{analytic_gradient, numerical_gradient};
use dft_rs::math::Vec3;
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;

fn molecule() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(-0.31, 0.22, -0.63),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.27, -0.18, 0.71),
        },
    ])
}

fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        density_fit: Some("def2-universal-jfit".to_string()),
        ..ScfOptions::default()
    }
}

#[test]
fn ri_j_derivatives_are_energy_consistent_and_obey_rigid_motion_sum_rules() {
    let mol = molecule();
    let set = data::load("def2-svp").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let functional = ResolvedFunctional::hartree_fock();
    let opts = options();

    let error = analytic_gradient(&mol, &mb, &opts, &functional, None)
        .err()
        .expect("RI-J analytic nuclear derivatives must not masquerade as exact J derivatives");
    assert!(error.to_string().contains("RI-J"));

    let gradient = numerical_gradient(&mol, &set, &opts, &functional, None, 1e-4).unwrap();
    let force_sum = gradient.iter().copied().fold(Vec3::zero(), |a, b| a + b);
    assert!(
        force_sum.norm() < 2e-7,
        "RI-J translation sum = {force_sum:?}"
    );
    let torque = mol
        .atoms
        .iter()
        .zip(&gradient)
        .fold(Vec3::zero(), |sum, (atom, g)| sum + atom.position.cross(*g));
    assert!(torque.norm() < 2e-7, "RI-J rotational torque = {torque:?}");

    let energy = |m: &Molecule| {
        let basis = MolBasis::build(m, &set).unwrap();
        run_scf(m, &basis, &opts, &functional, None).unwrap().energy
    };
    let axis = Vec3::new(0.4, -0.8, 1.1);
    let u = axis * (1.0 / axis.norm());
    let angle: f64 = 0.61;
    let moved = Molecule::new(
        mol.atoms
            .iter()
            .map(|atom| {
                let p = atom.position;
                Atom {
                    z: atom.z,
                    position: p * angle.cos()
                        + u.cross(p) * angle.sin()
                        + u * (u.dot(p) * (1.0 - angle.cos()))
                        + Vec3::new(1.7, -2.2, 0.9),
                }
            })
            .collect(),
    );
    assert!((energy(&mol) - energy(&moved)).abs() < 2e-9);
}
