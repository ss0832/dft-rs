// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent Poisson-sum oracle for the diffuse Li p image-tail failure.
use dft_rs::basis::aos::cart_to_sph;
use dft_rs::basis::data;
use dft_rs::periodic::bloch::BlochMatrix;
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::{Atom, Molecule, Vec3};
use std::f64::consts::PI;

fn fixture(cutoff: f64) -> Cell {
    let a = 3.964 / dft_rs::constants::BOHR_TO_ANGSTROM;
    let molecule = Molecule::new(vec![
        Atom {
            z: 3,
            position: Vec3::zero(),
        },
        Atom {
            z: 9,
            position: Vec3::new(a / 2.0, 0.0, 0.0),
        },
    ]);
    Cell::new(
        molecule,
        Lattice::new(
            Vec3::new(0.0, a / 2.0, a / 2.0),
            Vec3::new(a / 2.0, 0.0, a / 2.0),
            Vec3::new(a / 2.0, a / 2.0, 0.0),
        ),
        &data::load("sto-3g").unwrap(),
        cutoff,
    )
    .unwrap()
}

// Fourier transform of x exp(-a r^2): -i q_x (pi/a)^(3/2)
// exp(-q^2/(4a))/(2a). This independently avoids all real-space overlap
// integrals and image cutoffs. Only the fixed angular convention is shared.
fn poisson_p_block(cell: &Cell, k: Vec3, reciprocal_cutoff: f64) -> (Vec<usize>, BlochMatrix) {
    let shells: Vec<_> = cell.basis().shells.iter().filter(|s| s.l == 1).collect();
    let indices: Vec<_> = shells
        .iter()
        .flat_map(|s| (0..3).map(move |i| s.sph_off + i))
        .collect();
    let mut result = BlochMatrix::zeros(indices.len(), indices.len());
    let angular = cart_to_sph(1);
    let mut waves = cell.lattice().reciprocal_within(reciprocal_cutoff);
    waves.push(Vec3::zero());
    for g in waves {
        let q = g + k;
        let q_array = q.to_array();
        let mut values = Vec::new();
        for shell in &shells {
            let radial: f64 = shell
                .exps
                .iter()
                .zip(&shell.coefs)
                .map(|(&a, &c)| c * (PI / a).powf(1.5) * (-q.norm2() / (4.0 * a)).exp() / (2.0 * a))
                .sum();
            let phase = q.dot(shell.center);
            for i in 0..3 {
                let component: f64 = (0..3).map(|axis| q_array[axis] * angular[(axis, i)]).sum();
                let imaginary = -radial * component;
                values.push((imaginary * phase.sin(), imaginary * phase.cos()));
            }
        }
        for i in 0..values.len() {
            for j in 0..values.len() {
                result.re[(i, j)] += (values[i].0 * values[j].0 + values[i].1 * values[j].1)
                    / cell.lattice().volume();
                result.im[(i, j)] += (values[i].0 * values[j].1 - values[i].1 * values[j].0)
                    / cell.lattice().volume();
            }
        }
    }
    (indices, result)
}

#[test]
fn diffuse_p_overlap_requires_converged_images_and_matches_independent_poisson_sum() {
    let coarse = fixture(12.0);
    let medium = fixture(30.0);
    let converged = fixture(42.0);
    for k in [
        Vec3::zero(),
        Vec3::new(0.043, 0.017, -0.021),
        Vec3::new(-0.031, 0.027, 0.052),
    ] {
        let (indices, expected) = poisson_p_block(&converged, k, 24.0);
        let (_, finer) = poisson_p_block(&converged, k, 30.0);
        assert!(expected.re.max_abs_diff(&finer.re) < 1e-12);
        assert!(expected.im.max_abs_diff(&finer.im) < 1e-12);
        let mut errors = Vec::new();
        for cell in [&coarse, &medium, &converged] {
            let (real, imaginary) = cell.overlap(k);
            let mut maximum = 0.0_f64;
            for (i, &mu) in indices.iter().enumerate() {
                for (j, &nu) in indices.iter().enumerate() {
                    maximum = maximum.max((real[(mu, nu)] - expected.re[(i, j)]).abs());
                    maximum = maximum.max((imaginary[(mu, nu)] - expected.im[(i, j)]).abs());
                }
            }
            errors.push(maximum);
        }
        println!("k={k:?}: all complex p-block errors at12/30/42Bohr={errors:?}");
        assert!(
            errors[0] > 0.3,
            "fixture no longer reproduces the diffuse-tail deficiency"
        );
        assert!(errors[1] < 1e-6);
        assert!(errors[2] < 5e-11);
    }
}
