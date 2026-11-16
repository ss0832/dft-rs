// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Unit, meaning, owner and convergence contracts; numerical derivatives
//! remain those of the already independently validated common driver.

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::typed::MolecularProblem;
use dft_rs::indices::{Atom, AtomicOrbital, IndexSpace, SpinComponent};
use dft_rs::quantities::*;
use dft_rs::{DftError, Matrix, Molecule};

#[test]
fn physical_units_meanings_layout_and_invalid_inputs() {
    assert_eq!(
        std::mem::size_of::<Coordinates<Bohr>>(),
        std::mem::size_of::<Vec<[f64; 3]>>()
    );
    assert_eq!(
        std::mem::align_of::<Coordinates<Bohr>>(),
        std::mem::align_of::<Vec<[f64; 3]>>()
    );
    assert_eq!(
        std::mem::size_of::<Energy<Hartree, PerMolecule>>(),
        std::mem::size_of::<f64>()
    );
    let coordinates =
        Coordinates::<Angstrom>::new(vec![[0.529177210903, -1.058354421806, 0.0]]).unwrap();
    let pointer = coordinates.as_slice().as_ptr();
    let atomic = coordinates.into_bohr().unwrap();
    assert_eq!(atomic.as_slice().as_ptr(), pointer);
    assert_eq!(atomic.as_slice(), &[[1.0, -2.0, 0.0]]);
    let energy = Energy::<Hartree, PerMolecule>::new(1.0).unwrap();
    assert_eq!(energy.into_ev().unwrap().value(), 27.211386245988);
    assert_eq!(
        energy.into_kcal_per_mol().unwrap().kcal_per_mol(),
        627.509474063
    );
    let gradient = EnergyGradient::<HartreePerBohr>::new(vec![[1.0, -2.0, 3.0]]).unwrap();
    let force = gradient.clone().into_force();
    assert_eq!(force.as_slice(), &[[-1.0, 2.0, -3.0]]);
    assert_eq!(force.into_energy_gradient(), gradient);
    let physical = gradient.into_ev_per_angstrom().unwrap();
    assert!((physical.as_slice()[0][0] - 51.4220674763259).abs() < 1e-12);
    assert!(Coordinates::<Bohr>::new(vec![[f64::NAN, 0.0, 0.0]]).is_err());
    assert!(Coordinates::<Angstrom>::new(vec![[f64::MAX; 3]])
        .unwrap()
        .into_bohr()
        .is_err());
    assert!(CartesianHessian::<HartreePerBohr2>::new(Matrix::zeros(2, 2)).is_err());
    assert!(Stress::<HartreePerBohr3>::new([[f64::INFINITY; 3]; 3]).is_err());
    let h = CartesianHessian::<HartreePerBohr2>::new(Matrix::identity(3))
        .unwrap()
        .into_ev_per_angstrom2()
        .unwrap();
    assert!((h.as_matrix()[(0, 0)] - 97.1736242922822).abs() < 1e-10);
    let stress = Stress::<HartreePerBohr3>::new([[1.0; 3]; 3])
        .unwrap()
        .into_gigapascal()
        .unwrap();
    assert!((stress.as_array()[0][0] - 29421.0156965221).abs() < 1e-6);
}

#[test]
fn owners_generations_ranges_and_xyz_unit_boundary() {
    let a = IndexSpace::<AtomicOrbital>::new(3).unwrap();
    let b = IndexSpace::<AtomicOrbital>::new(3).unwrap();
    let id = a.id(1).unwrap();
    assert_eq!(a.clone().resolve(id).unwrap(), 1);
    assert!(b.resolve(id).is_err());
    assert!(a.id(3).is_err());
    assert!(IndexSpace::<Atom>::new(0).unwrap().id(0).is_err());
    let geometry =
        Molecule::from_xyz_str("1\nExplicit Angstrom input\nHe 0.529177210903 0 0\n").unwrap();
    assert_eq!(geometry.atoms[0].position.to_array(), [1.0, 0.0, 0.0]);
    assert!(Molecule::from_xyz_str("1\nInvalid input\nHe NaN 0 0\n").is_err());
}

fn method() -> MolecularMethod {
    MolecularMethod {
        xc: Some("hf".into()),
        basis: "sto-3g".into(),
        dispersion: "none".into(),
        conv_tol: 1e-13,
        conv_tol_dp: 1e-11,
        conv_tol_grad: Some(1e-11),
        max_iter: 200,
        guess: Some("core".into()),
        grid_weight_cutoff: 0.0,
        ..Default::default()
    }
}

#[test]
fn checked_solution_preserves_every_existing_gradient_and_hessian_entry() {
    // Nonplanar asymmetric ammonia; no coordinate component is suppressed.
    let numbers = [7, 1, 1, 1];
    let positions = vec![
        [0.13, -0.21, 0.17],
        [1.82, 0.16, -0.39],
        [-0.37, 1.68, -0.28],
        [-0.53, -0.64, 1.87],
    ];
    let method = method();
    let problem = MolecularProblem::new(
        &numbers,
        Coordinates::new(positions.clone()).unwrap(),
        &method,
    )
    .unwrap();
    let solution = problem.run().unwrap();
    let setup = prepare_molecular(&numbers, &positions, &method).unwrap();
    let raw = run_molecular_reference(&setup).unwrap();
    assert_eq!(solution.energy().value(), raw.total_energy());
    let gradient = solution.nuclear_gradient().unwrap();
    let expected = dft_rs::gradient::assemble_gradient(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &raw.scf,
        &setup.options,
    )
    .unwrap();
    for (actual, expected) in gradient.as_slice().iter().zip(&expected) {
        assert_eq!(*actual, expected.to_array());
    }
    let hessian = solution.cartesian_hessian().unwrap();
    let expected = dft_rs::hessian::analytic::analytic_hessian(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        &raw.scf,
        setup.grid.as_ref(),
    )
    .unwrap();
    assert_eq!(hessian.as_matrix().as_slice(), expected.as_slice());
    let a = problem.ao_ids().id(0).unwrap();
    assert_eq!(
        solution
            .density_element(SpinComponent::Alpha, a, a)
            .unwrap(),
        raw.scf.density[0][(0, 0)]
    );
    let moved = problem
        .with_coordinates(
            Coordinates::new(positions.iter().map(|r| [r[0] + 0.2, r[1], r[2]]).collect()).unwrap(),
        )
        .unwrap();
    assert!(moved.position(problem.atom_ids().id(0).unwrap()).is_err());
    assert!(solution
        .density_element(SpinComponent::Alpha, moved.ao_ids().id(0).unwrap(), a)
        .is_err());
    assert_eq!(solution.energy().value(), raw.total_energy());
    assert_eq!(
        solution.initial_guess().as_density()[0].as_slice(),
        raw.scf.density[0].as_slice()
    );
    println!("Typed ammonia HF/STO-3G energy={:.16e} gradient={:?} Hessian max={:.16e} raw orbital residual={:.16e}",
        solution.energy().value(), gradient.as_slice(),
        hessian.as_matrix().as_slice().iter().map(|v| v.abs()).fold(0.0, f64::max),
        solution.diagnostics().orbital_residual);
}

#[test]
fn unfinished_scf_has_a_trial_energy_but_no_converged_transition() {
    let method = MolecularMethod {
        max_iter: 1,
        ..method()
    };
    let problem = MolecularProblem::new(
        &[8, 1, 1],
        Coordinates::new(vec![[0.1, -0.2, 0.3], [1.5, 0.1, -0.8], [-0.4, 1.7, -0.3]]).unwrap(),
        &method,
    )
    .unwrap();
    let trial = problem.iterate().unwrap();
    assert!(trial.trial_energy().unwrap().value().is_finite());
    assert!(!trial.diagnostics().converged);
    println!(
        "Unconverged trial energy={:.16e} raw orbital residual={:.16e}",
        trial.trial_energy().unwrap().value(),
        trial.diagnostics().orbital_residual
    );
    // The established public SCF contract still rejects this same input.
    let strict = prepare_molecular(
        &[8, 1, 1],
        &[[0.1, -0.2, 0.3], [1.5, 0.1, -0.8], [-0.4, 1.7, -0.3]],
        &method,
    )
    .unwrap();
    assert!(matches!(
        run_molecular_reference(&strict),
        Err(DftError::ScfNotConverged { .. })
    ));
    assert!(matches!(
        trial.into_converged(),
        Err(DftError::ScfNotConverged { .. })
    ));
}
