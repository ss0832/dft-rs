// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::data;
use dft_rs::dfpt::coulomb::FiniteQHartree;
use dft_rs::math::Vec3;
use dft_rs::periodic::bloch::BlochMatrix;
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::{Atom, Molecule};

#[test]
fn reciprocal_hartree_matches_the_ewald_band_splitting() {
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.7),
            },
        ]),
        Lattice::cubic(9.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap();
    let k = Vec3::new(0.0, 0.0, 0.0);
    let (s, _) = cell.overlap(k);
    let mut density = BlochMatrix::zeros(2, 2);
    density.re[(0, 0)] = 1.0;
    density.re[(1, 1)] = 1.0;
    density.re[(0, 1)] = -s[(0, 0)] / s[(0, 1)];
    density.re[(1, 0)] = density.re[(0, 1)];
    let images = cell
        .density_images_from_k(
            &[k],
            &[(density.re.clone(), density.im.clone())],
            2.0 * cell.hartree_real_space_cutoff(),
        )
        .unwrap();
    let (j, _) = cell.hartree(&images, k).unwrap();
    let splitting = |d: &dft_rs::linalg::Matrix| {
        (d[(0, 0)] + d[(0, 1)]) / (s[(0, 0)] + s[(0, 1)])
            - (d[(0, 0)] - d[(0, 1)]) / (s[(0, 0)] - s[(0, 1)])
    };
    let expected = splitting(&j);
    for cutoff in [7.0, cell.hartree_real_space_cutoff()] {
        let kernel = FiniteQHartree::new(&cell, &[k], k, 16.0, cutoff).unwrap();
        let zero = BlochMatrix::zeros(2, 2);
        let actual = kernel.apply(&[vec![density.clone()], vec![zero]]).unwrap();
        let error = (splitting(&actual[0][0].re) - expected).abs();
        if cutoff == 7.0 {
            // This deliberately truncated metric range cannot be reused
            // for the Coulomb products of a diffuse basis.
            assert!(error > 1e-5);
        } else {
            assert!(
                error < 1e-9,
                "reciprocal/Ewald band splitting error {error:e}"
            );
            let mut ewald_energy = 0.0;
            let mut reciprocal_energy = 0.0;
            for i in 0..2 {
                for jcol in 0..2 {
                    let reciprocal = actual[0][0].re[(i, jcol)];
                    assert!(
                        (reciprocal - j[(i, jcol)]).abs() < 1e-9,
                        "Hartree matrix component ({i},{jcol}) differs between Ewald and reciprocal sums"
                    );
                    ewald_energy += 0.5 * density.re[(jcol, i)] * j[(i, jcol)];
                    reciprocal_energy += 0.5 * density.re[(jcol, i)] * reciprocal;
                }
            }
            assert!((reciprocal_energy - ewald_energy).abs() < 1e-9);
        }
    }
}
