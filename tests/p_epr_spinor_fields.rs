// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Closed-form spin-rotation and full-complex expectation controls. These
//! do not establish a noncollinear effective-spin manifold or a g tensor.

use dft_rs::properties::epr::spinor as spinor_epr;

use dft_rs::basis::MolBasis;
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::periodic::spinor_scf::from_pauli;
use dft_rs::periodic::BlochMatrix;
use dft_rs::{Atom, Molecule};
use spinor_epr::spinor_hyperfine_field;

#[test]
fn generalized_complex_spin_rotations_keep_all_contact_field_components() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(generalized_rotations)
        .unwrap();
}

fn generalized_rotations() {
    let set = dft_rs::basis::data::load("sto-3g").unwrap();
    let molecule = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.13, -0.17, 0.23),
    }]);
    let basis = MolBasis::build(&molecule, &set).unwrap();
    let overlap = dft_rs::integrals::overlap_matrix(&basis);
    let mut orbital = Matrix::zeros(1, 1);
    orbital[(0, 0)] = 1.0 / overlap[(0, 0)];
    let collinear = dft_rs::properties::hyperfine::spin_hyperfine_tensor(
        &basis,
        &orbital,
        molecule.atoms[0].position,
        1.0,
    )
    .unwrap();
    for axis in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.36, -0.48, 0.8]] {
        let mut pauli = std::array::from_fn::<_, 4, _>(|_| BlochMatrix::zeros(1, 1));
        pauli[0].re = orbital.clone();
        for a in 0..3 {
            pauli[a + 1].re[(0, 0)] = axis[a] * orbital[(0, 0)];
        }
        let density = from_pauli(&pauli).unwrap();
        let report = spinor_hyperfine_field(
            &basis,
            &overlap,
            &density,
            molecule.atoms[0].position,
            1e-12,
        )
        .unwrap();
        assert!(report.collinearity_defect < 1e-12);
        let tensor = report.collinear_spin_tensor.as_ref().unwrap();
        for a in 0..3 {
            assert!((report.contact[a] - 0.5 * axis[a] * collinear.contact).abs() < 1e-9);
            assert!((report.spin_expectation[a] - 0.5 * axis[a]).abs() < 1e-12);
            for b in 0..3 {
                assert!((tensor.total[a][b] - collinear.total[a][b]).abs() < 1e-9);
            }
        }
        assert!(report.poisson_residual < 1e-10);
        assert!(report.hermiticity_residual < 1e-12);
        println!(
            "Generalized H spin={:?}, hyperfine nuclear field={:?}MHz per nuclear g",
            report.spin_expectation, report.total
        );
    }
}

#[test]
fn a_noncollinear_texture_returns_a_field_without_fabricating_a_spin_tensor() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(noncollinear_texture)
        .unwrap();
}

fn noncollinear_texture() {
    let set = dft_rs::basis::data::load("sto-3g").unwrap();
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.13, -0.17, 0.23),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.41, 0.29, -0.31),
        },
    ]);
    let basis = MolBasis::build(&molecule, &set).unwrap();
    let overlap = dft_rs::integrals::overlap_matrix(&basis);
    let mut density = BlochMatrix::zeros(4, 4);
    let scale = 1.0 / (overlap[(0, 0)] + overlap[(1, 1)]);
    density.re[(0, 0)] = scale;
    density.re[(3, 3)] = scale;
    density.im[(0, 3)] = -scale;
    density.im[(3, 0)] = scale;
    let report = spinor_hyperfine_field(
        &basis,
        &overlap,
        &density,
        molecule.atoms[0].position,
        1e-12,
    )
    .unwrap();
    assert!(report.collinearity_defect > 1e-3);
    assert!(report.collinear_spin_tensor.is_none());
    assert!(report.total.iter().all(|v| v.is_finite()));
    assert!(report.contact[2].abs() > 1e-6);
    assert!(
        (report.total[0] - report.contact[0] - report.dipolar[0] - report.orbital[0]).abs() < 1e-12
    );
    println!(
        "Noncollinear field={:?}, collinearity defect={}, effective tensor absent",
        report.total, report.collinearity_defect
    );
}
