// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Full-RPA stability is required before returning a normalizable state.
use dft_rs::basis::{data, MolBasis};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::response::tddft::{TdaSolver, TdaSpin};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[test]
fn negative_and_zero_squared_frequencies_are_not_zero_energy_states() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0., 0., 0.),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.2, -0.1, 1.4),
        },
    ]);
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let options = ScfOptions::default();
    let reference = run_scf(&molecule, &basis, &options, &functional, None).unwrap();
    assert!(reference.converged);
    let solver = TdaSolver::new(
        &molecule,
        &basis,
        &functional,
        None,
        &reference,
        &options,
        TdaSpin::Singlet,
    )
    .unwrap();
    assert_eq!(solver.dim(), 1);
    let minus = Matrix::identity(1);
    for value in [-1., 0.] {
        let plus = Matrix::from_row_major(1, 1, vec![value]);
        assert!(
            solver.rpa_from_hessians(&plus, &minus, 1).is_err(),
            "non-positive squared frequency {value} was returned as a state"
        );
    }
    let plus = Matrix::from_row_major(1, 1, vec![4.]);
    let states = solver.rpa_from_hessians(&plus, &minus, 1).unwrap();
    assert!((states[0].energy - 2.).abs() < 1e-14);
    // Stored amplitudes are X+Y, not X alone. The metric partner is
    // (A+B)(X+Y)/omega = X-Y and their scalar product is one.
    let xpy = states[0].x[0][(0, 0)];
    assert!((xpy * (4. * xpy / 2.) - 1.).abs() < 1e-14);
}
