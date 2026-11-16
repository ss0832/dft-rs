// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Ordered parallel Hartree assembly retains every independent C1 gradient.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::linalg::Matrix;
use dft_rs::periodic::integrals::ImageBlock;
use dft_rs::periodic::screened_coulomb::{
    screened_hartree_derivatives, screened_hartree_image_blocks, HartreeScreening,
};
use dft_rs::periodic::Lattice;
use dft_rs::{Atom, Molecule, Vec3};

const POSITIONS: [[f64; 3]; 4] = [
    [0.1, -0.2, 0.3],
    [1.35, 0.1, 0.2],
    [-0.3, 3.2, 0.9],
    [0.2, 3.4, 2.2],
];

fn basis_at(positions: &[[f64; 3]; 4]) -> MolBasis {
    let mut basis = BasisSet::new("C1 mixed compact/diffuse Hartree regression");
    basis.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![2.4, 0.55],
            coefs: vec![0.6, 0.5],
        }],
    });
    MolBasis::build(
        &Molecule::new(
            positions
                .iter()
                .map(|p| Atom {
                    z: 1,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        ),
        &basis,
    )
    .unwrap()
}

#[test]
fn every_c1_hartree_gradient_component_retains_a_four_step_energy_fit() {
    c1_geometry::assert_c1(&[1; 4], &POSITIONS);
    let basis = basis_at(&POSITIONS);
    let lattice = Lattice::new(
        Vec3::new(8.7, 0.31, 0.13),
        Vec3::new(0.2, 9.2, 0.43),
        Vec3::new(0.11, 0.27, 9.7),
    );
    let mut d = Matrix::zeros(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            d[(i, j)] = 0.15 + 0.08 * (0.7 * (1 + i + j) as f64).sin();
        }
    }
    let density = vec![ImageBlock {
        translation: Vec3::zero(),
        matrix: d,
    }];
    let screening = HartreeScreening {
        compact_exponent: 1.0,
        tolerance: 1e-13,
        max_translation: Some(4.5),
    };
    // This is an analytic Gaussian Hartree term; no XC grid or pruning is used.
    let analytic = screened_hartree_derivatives(&basis, &lattice, &density, &screening, 4).unwrap();
    let energy = |basis: &MolBasis| {
        screened_hartree_image_blocks(basis, &lattice, &density, &screening)
            .unwrap()
            .iter()
            .filter(|b| b.translation.norm() < 1e-8)
            .map(|b| 0.5 * b.matrix.frobenius_dot(&density[0].matrix))
            .sum::<f64>()
    };
    assert!((energy(&basis) - analytic.energy).abs() < 1e-11);
    let total = analytic.gradient.iter().fold(Vec3::zero(), |a, b| a + *b);
    assert!(total.norm() < 1e-10);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let names: Vec<_> = (0..4)
        .flat_map(|atom| {
            (0..3).map(move |axis| format!("ordered parallel Hartree atom {atom} axis {axis}"))
        })
        .collect();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            (steps, stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            (0..4)
                .flat_map(|atom| (0..3).map(move |axis| (atom, axis)))
                .map(|(atom, axis)| {
                    steps.map(|h| {
                        let mut plus = POSITIONS;
                        let mut minus = POSITIONS;
                        plus[atom][axis] += h;
                        minus[atom][axis] -= h;
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            let mut p2 = POSITIONS;
                            let mut m2 = POSITIONS;
                            p2[atom][axis] += 2.0 * h;
                            m2[atom][axis] -= 2.0 * h;
                            (energy(&basis_at(&p2)), energy(&basis_at(&m2)))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(
                            energy(&basis_at(&plus)),
                            energy(&basis_at(&minus)),
                            plus2,
                            minus2,
                            h,
                        ) - analytic.gradient[atom].get(axis))
                        .abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("all C1 Hartree gradient components", &failures);
}
