// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact singular values and complex AO reconstruction of a rectangular
//! transition with independently specified orthogonal columns.

use dft_rs::dfpt::orbitals::{adjoint, matmul};
use dft_rs::linalg::Matrix;
use dft_rs::periodic::BlochMatrix;
use dft_rs::properties::natural_transition_orbitals::decompose;

#[test]
fn rectangular_complex_transition_recovers_weak_and_strong_pairs() {
    let mut transition = BlochMatrix::zeros(4, 3);
    transition.re[(0, 0)] = 0.3;
    transition.im[(1, 0)] = 0.4;
    transition.im[(2, 1)] = -0.2;
    transition.re[(3, 2)] = 4e-9;
    let particles = BlochMatrix::from_pair((Matrix::identity(4), Matrix::zeros(4, 4)));
    let holes = BlochMatrix::from_pair((
        Matrix::from_row_major(4, 3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.]),
        Matrix::zeros(4, 3),
    ));
    let result = decompose(&transition, &particles, &holes).unwrap();
    assert_eq!(result.singular_values.len(), 3);
    for (&actual, expected) in result.singular_values.iter().zip([0.5, 0.2, 4e-9]) {
        assert!((actual - expected).abs() < 5e-15);
    }
    let mut weighted = result.particle_coefficients.clone();
    for (k, &sigma) in result.singular_values.iter().enumerate() {
        for a in 0..weighted.rows() {
            weighted.re[(a, k)] *= sigma;
            weighted.im[(a, k)] *= sigma;
        }
    }
    let reconstructed = matmul(&weighted, &adjoint(&result.hole_coefficients).unwrap()).unwrap();
    let direct = matmul(&transition, &adjoint(&holes).unwrap()).unwrap();
    let error = reconstructed.rms_diff(&direct);
    println!(
        "complex NTO singular values {:?}, AO reconstruction RMS {error:.3e}",
        result.singular_values
    );
    assert!(error < 5e-15 && result.reconstruction_residual < 5e-15);
    assert!((result.transition_weight - 0.29).abs() < 5e-15);
}
