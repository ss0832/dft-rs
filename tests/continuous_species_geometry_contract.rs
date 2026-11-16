// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! The same spin/count contract guards every affine derivative order.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::GridOptions;
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::scalar_geometry::{AffineFrozenGeometry, CollinearAffineFrozenGeometry};
use dft_rs::response::krylov::ResponseSolverConfig;
use dft_rs::scf::{Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[test]
fn every_affine_derivative_refuses_truncated_counts_and_conflicting_spins() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(-0.61, 0.13, -0.09),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.72, -0.08, 0.17),
        },
    ]);
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let frozen = Matrix::zeros(basis.nao_sph, 0);
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let options = ScfOptions {
        reference: Reference::Rhf,
        ..Default::default()
    };
    let grid = GridOptions {
        weight_cutoff: 0.,
        ..Default::default()
    };
    let r = molecule
        .atoms
        .iter()
        .map(|a| a.position)
        .collect::<Vec<_>>();
    let config = ResponseSolverConfig::default();
    let direction = [[0.1, -0.2, 0.3], [-0.3, 0.2, 0.1]];
    for count in [0usize, 3] {
        let model = AffineFrozenGeometry {
            full_reference: &molecule,
            full_basis: &basis,
            raw_reference: &molecule,
            raw_basis: &basis,
            frozen_reference: &frozen,
            full_atom_owners: &[0, 1],
            functional: &functional,
            options: &options,
            grid_options: &grid,
            active_electrons: count,
            reference_energy: 0.,
        };
        assert!(model.evaluate(&r, None).is_err());
        assert!(model.hessian(&r, &config).is_err());
        assert!(model
            .directional_second_derivative(&r, &direction, &config)
            .is_err());
    }
    let model = CollinearAffineFrozenGeometry {
        full_reference: &molecule,
        full_basis: &basis,
        raw_reference: &molecule,
        raw_basis: &basis,
        frozen_reference: [&frozen; 2],
        full_atom_owners: &[0, 1],
        functional: &functional,
        options: &options,
        grid_options: &grid,
        active_electrons: [1, 1],
        reference_energy: 0.,
    };
    assert!(model.evaluate(&r, None).is_err());
    assert!(model.hessian(&r, &config).is_err());
    assert!(model
        .directional_second_derivative(&r, &direction, &config)
        .is_err());
}
