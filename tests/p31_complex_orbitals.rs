// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::dfpt::orbitals::{adjoint, band_density, generalized_orbitals, matmul};
use dft_rs::linalg::Matrix;
use dft_rs::periodic::bloch::BlochMatrix;
use dft_rs::periodic::spinor_scf::{pauli_components, spin_diagonal};

fn identity(n: usize) -> BlochMatrix {
    BlochMatrix {
        re: Matrix::identity(n),
        im: Matrix::zeros(n, n),
    }
}

#[test]
fn complex_generalized_orbitals_resolve_exact_degeneracy_and_metric() {
    let mut s = identity(3);
    s.re[(0, 0)] = 1.4;
    s.re[(1, 1)] = 0.9;
    s.im[(0, 1)] = 0.2;
    s.im[(1, 0)] = -0.2;
    // F=epsilon*S has complete degeneracy. Doubled-real eigenvectors do
    // not have a distinguished ordering within this eigenspace.
    let mut f = s.clone();
    f.scale(-0.7);
    let orbitals = generalized_orbitals(&f, &s, 1e-9).unwrap();
    assert_eq!(orbitals.energies.len(), 3);
    assert!(orbitals.energies.iter().all(|e| (e + 0.7).abs() < 1e-12));
    assert!(orbitals.eigen_residual < 1e-12);
    assert!(orbitals.metric_defect < 1e-12);
    let density = band_density(&orbitals, &[0.8, 0.3, 0.1]).unwrap();
    assert!(density.hermiticity_error() < 1e-12);
    assert!((density.trace_product(&s) - 1.2).abs() < 1e-12);
    let mo = matmul(
        &adjoint(&orbitals.coefficients).unwrap(),
        &matmul(
            &s,
            &matmul(&density, &matmul(&s, &orbitals.coefficients).unwrap()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((mo.re[(i, j)] - if i == j { [0.8, 0.3, 0.1][i] } else { 0.0 }).abs() < 1e-12);
            assert!(mo.im[(i, j)].abs() < 1e-12);
        }
    }
}

#[test]
fn complex_spectrum_and_overlap_rank_match_closed_forms() {
    let s = identity(2);
    let mut f = BlochMatrix::zeros(2, 2);
    f.re[(0, 0)] = -0.4;
    f.re[(1, 1)] = 0.8;
    f.re[(0, 1)] = 0.3;
    f.re[(1, 0)] = 0.3;
    f.im[(0, 1)] = 0.2;
    f.im[(1, 0)] = -0.2;
    let orbitals = generalized_orbitals(&f, &s, 1e-9).unwrap();
    let splitting = 0.49_f64.sqrt();
    assert!((orbitals.energies[0] - (0.2 - splitting)).abs() < 1e-12);
    assert!((orbitals.energies[1] - (0.2 + splitting)).abs() < 1e-12);
    assert!(orbitals.eigen_residual < 1e-12);
    let mut reduced_s = identity(2);
    reduced_s.re[(1, 1)] = 1e-14;
    let reduced_f = identity(2);
    let reduced = generalized_orbitals(&reduced_f, &reduced_s, 1e-9).unwrap();
    assert_eq!(reduced.energies.len(), 1);
    assert!((reduced.energies[0] - 1.0).abs() < 1e-12);
    let mut malformed = s;
    malformed.im = Matrix::zeros(1, 2);
    assert!(generalized_orbitals(&f, &malformed, 1e-9).is_err());
    assert!(band_density(&orbitals, &[1.1, 0.0]).is_err());
}

#[test]
fn equal_spin_blocks_keep_a_filled_pair_exactly_unpolarized() {
    let mut overlap = identity(2);
    overlap.re[(0, 1)] = 0.23;
    overlap.re[(1, 0)] = 0.23;
    let mut fock = BlochMatrix::zeros(2, 2);
    fock.re[(0, 0)] = -0.8;
    fock.re[(1, 1)] = 0.3;
    fock.re[(0, 1)] = 0.07;
    fock.re[(1, 0)] = 0.07;
    fock.im[(0, 1)] = 0.15;
    fock.im[(1, 0)] = -0.15;
    let spatial = generalized_orbitals(&fock, &overlap, 1e-9).unwrap();
    let spin_overlap = spin_diagonal(&overlap);
    let mut spin_fock = spin_diagonal(&fock);
    let spinor = generalized_orbitals(&spin_fock, &spin_overlap, 1e-9).unwrap();
    for (pair, energy) in spinor.energies.chunks_exact(2).zip(spatial.energies) {
        assert_eq!(pair[0], pair[1]);
        assert!((pair[0] - energy).abs() < 1e-12);
    }
    let density = band_density(&spinor, &[1.0, 1.0, 0.0, 0.0]).unwrap();
    assert!((density.trace_product(&spin_overlap) - 2.0).abs() < 1e-12);
    let pauli = pauli_components(&density).unwrap();
    for component in &pauli[1..] {
        assert!(component
            .re
            .as_slice()
            .iter()
            .chain(component.im.as_slice())
            .all(|&value| value == 0.0));
    }
    // A finite spin-dependent perturbation must leave the repeated-block
    // route and change the spectrum, including when the overlap is unchanged.
    spin_fock.re[(0, 2)] = 0.04;
    spin_fock.re[(2, 0)] = 0.04;
    let perturbed = generalized_orbitals(&spin_fock, &spin_overlap, 1e-9).unwrap();
    assert!(perturbed.energies[1] - perturbed.energies[0] > 0.05);
    assert!(perturbed.eigen_residual < 1e-12);
    assert!(perturbed.metric_defect < 1e-12);
}
