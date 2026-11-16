// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::data;
use dft_rs::dfpt::density::{PeriodicDensitySolver, PeriodicResponseOptions, ThermalSource};
use dft_rs::dfpt::thermodynamic::Ensemble;
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

#[test]
fn screened_thermal_response_matches_independent_temperature_scf_derivatives() {
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
    let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let options = PeriodicScfOptions {
        occupation: OccupationScheme::FermiDirac {
            temperature_kelvin: 40000.0,
        },
        e_tol: 1e-12,
        p_tol: 1e-11,
        conv_tol_grad: 1e-10,
        max_iter: 250,
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(10),
            angular_override: Some(7),
            prune: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(reference.converged);
    let request = PeriodicResponseOptions {
        reciprocal_cutoff: 14.0,
        ..Default::default()
    };
    let solver =
        PeriodicDensitySolver::new(&cell, &functional, &reference, &options, &request).unwrap();
    let delta_temperature = 400.0 * BOLTZMANN_HARTREE_PER_K;
    let source = ThermalSource {
        delta_temperature_hartree: delta_temperature,
        ensembles: [Ensemble::FixedNumber { delta_number: 0.0 }; 2],
    };
    let response = solver.solve_thermal(source).unwrap();
    assert!(response.residuals[0] <= 1e-10);
    assert!(response.value.occupation_equation_defect < 1e-12);
    assert!(response
        .value
        .number_constraint_defects
        .iter()
        .all(|x| *x < 1e-12));
    assert!(response.value.delta_entropy > 0.0);
    assert!(
        (response.value.delta_free_energy + reference.entropy_term / 40000.0 * 400.0).abs() < 1e-10
    );
    let shifted = |temperature_kelvin| {
        let mut shifted = options.clone();
        shifted.occupation = OccupationScheme::FermiDirac { temperature_kelvin };
        let result = run_periodic_scf(&cell, &functional, &shifted).unwrap();
        assert!(result.converged);
        result
    };
    let coarse_plus = shifted(40040.0);
    let coarse_minus = shifted(39960.0);
    let fine_plus = shifted(40020.0);
    let fine_minus = shifted(39980.0);
    let extrapolate = |coarse: f64, fine: f64| (4.0 * fine - coarse) / 3.0;
    for spin in 0..2 {
        for index in 0..4 {
            let coarse = (coarse_plus.density[spin][0].re.as_slice()[index]
                - coarse_minus.density[spin][0].re.as_slice()[index])
                * 400.0
                / 80.0;
            let fine = (fine_plus.density[spin][0].re.as_slice()[index]
                - fine_minus.density[spin][0].re.as_slice()[index])
                * 400.0
                / 40.0;
            let expected = extrapolate(coarse, fine);
            let actual = response.value.density[spin][0].re.as_slice()[index];
            assert!(
                (actual - expected).abs() < 1e-7,
                "spin {spin}, AO {index}: analytic {actual:.14e}, independent {expected:.14e}"
            );
        }
    }
    for (actual, cp, cm, fp, fm) in [
        (
            response.value.delta_internal_energy,
            coarse_plus.energy,
            coarse_minus.energy,
            fine_plus.energy,
            fine_minus.energy,
        ),
        (
            response.value.delta_free_energy,
            coarse_plus.free_energy,
            coarse_minus.free_energy,
            fine_plus.free_energy,
            fine_minus.free_energy,
        ),
    ] {
        assert!((actual - extrapolate((cp - cm) * 5.0, (fp - fm) * 10.0)).abs() < 1e-7);
    }
    let number_source = ThermalSource {
        delta_temperature_hartree: 0.0,
        ensembles: [
            Ensemble::FixedNumber { delta_number: 0.01 },
            Ensemble::FixedNumber {
                delta_number: -0.02,
            },
        ],
    };
    let number = solver.solve_thermal(number_source).unwrap();
    assert!((number.value.delta_numbers[0] - 0.01).abs() < 1e-12);
    assert!((number.value.delta_numbers[1] + 0.02).abs() < 1e-12);
    let mut grand = request.clone();
    grand.fixed_number = false;
    let grand =
        PeriodicDensitySolver::new(&cell, &functional, &reference, &options, &grand).unwrap();
    let reservoir = grand
        .solve_thermal(ThermalSource {
            delta_temperature_hartree: 0.0,
            ensembles: [Ensemble::FixedChemicalPotential { delta_mu: 0.001 }; 2],
        })
        .unwrap();
    assert!(reservoir.value.delta_numbers.iter().all(|n| *n > 0.0));
    assert_eq!(reservoir.value.delta_chemical_potentials, [Some(0.001); 2]);
    assert!(reservoir.value.occupation_equation_defect < 1e-12);
    assert!(grand.solve_thermal(source).is_err());
}
