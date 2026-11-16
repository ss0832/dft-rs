// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Water finite-field acceptance for molecular and periodic response paths.
//! Finite fields are independent validation, never production derivatives.
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/electric.rs"]
mod electric;

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::periodic::dielectric::dielectric_tensors;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::properties::dipole_moment;
use dft_rs::response::nuclear::{
    lattice_electric_tensor, lattice_finite_field, ElectricTensorSymmetry,
};
use dft_rs::response::{electric_field_taylor, ResponseSolverConfig};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

const DIRECTION: [f64; 3] = [0.2, -0.3, 0.7];

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.43, 0.0, 1.10),
        },
        Atom {
            z: 1,
            position: Vec3::new(-1.43, 0.0, 1.10),
        },
    ])
}

fn grid_options() -> GridOptions {
    GridOptions {
        radial_override: Some(10),
        angular_override: Some(7),
        prune: false,
        ..Default::default()
    }
}

#[test]
fn molecular_water_alpha_beta_gamma_match_biased_and_unbiased_finite_fields() {
    let molecule = water();
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    for model in ["hf", "pbe0"] {
        let functional = resolve(&FunctionalSpec::Combined(model.into())).unwrap();
        let grid = (!functional.parts.is_empty())
            .then(|| MolecularGrid::build(&molecule, &grid_options()).unwrap());
        for bias in [0.0, 0.003] {
            let options = ScfOptions {
                max_iter: 300,
                e_tol: 1e-14,
                p_tol: 1e-14,
                conv_tol_grad: 1e-14,
                external_field: Some(Vec3::new(0.0, 0.0, bias)),
                ..Default::default()
            };
            let reference =
                run_scf(&molecule, &basis, &options, &functional, grid.as_ref()).unwrap();
            assert!(reference.converged);
            let response = electric_field_taylor(
                &molecule,
                &basis,
                &functional,
                grid.as_ref(),
                &reference,
                &options,
                DIRECTION,
                3,
                &config,
            )
            .unwrap();
            assert!(response.converged);
            assert!(response.residuals.iter().all(|value| *value <= 1e-12));
            let values: Vec<_> = (1..=3)
                .map(|order| {
                    let factorial = match order {
                        1 => 1.0,
                        2 => 2.0,
                        _ => 6.0,
                    };
                    (
                        order,
                        factorial
                            * electric::projection(
                                response.value.dipole_coefficients[order],
                                DIRECTION,
                            ),
                    )
                })
                .collect();
            electric::check_fields(
                &format!("water molecular {model} bias={bias}"),
                &values,
                |field| {
                    let mut shifted = options.clone();
                    shifted.external_field = Some(Vec3::new(
                        field * DIRECTION[0],
                        field * DIRECTION[1],
                        bias + field * DIRECTION[2],
                    ));
                    let state =
                        run_scf(&molecule, &basis, &shifted, &functional, grid.as_ref()).unwrap();
                    assert!(state.converged);
                    let mu = dipole_moment(&molecule, &basis, &state);
                    electric::projection([mu.x, mu.y, mu.z], DIRECTION)
                },
                1e-5,
            );
            println!("water molecular {model} bias={bias}: response={:?} reference={:.12e} projector={:?} particle_number={:?}",response.residuals,response.reference.orbital_residual,response.value.projector_defects,response.value.particle_number_derivatives);
        }
    }
}

#[test]
fn periodic_water_beta_gamma_match_independent_berry_finite_fields() {
    let cell = Cell::new(
        water(),
        Lattice::cubic(12.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let options = PeriodicScfOptions {
        max_iter: 300,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        grid_shell: 1,
        ..Default::default()
    };
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(reference.converged);
    let beta = lattice_electric_tensor(
        &cell,
        &functional,
        &reference,
        &options,
        1,
        2,
        ElectricTensorSymmetry::Unprojected,
        &config,
    )
    .unwrap();
    let gamma = lattice_electric_tensor(
        &cell,
        &functional,
        &reference,
        &options,
        1,
        3,
        ElectricTensorSymmetry::Unprojected,
        &config,
    )
    .unwrap();
    let exact = [
        (2, electric::contraction(&beta.tensor, 3, DIRECTION)),
        (3, electric::contraction(&gamma.tensor, 4, DIRECTION)),
    ];
    electric::check_fields(
        "water periodic hf",
        &exact,
        |field| {
            let state = lattice_finite_field(
                &cell,
                &functional,
                &reference,
                &options,
                1,
                DIRECTION.map(|u| field * u),
                false,
            )
            .unwrap();
            assert!(state.residual < 1e-13);
            electric::projection(state.dipole, DIRECTION) / state.volume
        },
        1e-6,
    );
    assert!(
        beta.permutation_residual.unwrap() < 1e-10 && gamma.permutation_residual.unwrap() < 1e-10
    );
    assert!(beta
        .response_residuals
        .iter()
        .chain(&gamma.response_residuals)
        .all(|value| *value <= 1e-12));
    println!("water periodic hf: beta_permutation={:.12e} gamma_permutation={:.12e} beta_response={:?} gamma_response={:?} reference={:.12e}",beta.permutation_residual.unwrap(),gamma.permutation_residual.unwrap(),beta.response_residuals,gamma.response_residuals,gamma.reference.residual);
}

#[test]
fn water_dfpt_dielectric_approaches_the_independent_molecular_limit() {
    let molecule = water();
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let molecular_options = ScfOptions {
        max_iter: 300,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        ..Default::default()
    };
    let molecular = run_scf(&molecule, &basis, &molecular_options, &functional, None).unwrap();
    assert!(molecular.converged);
    let response = electric_field_taylor(
        &molecule,
        &basis,
        &functional,
        None,
        &molecular,
        &molecular_options,
        DIRECTION,
        1,
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    let exact = electric::projection(response.value.dipole_coefficients[1], DIRECTION);
    let options = PeriodicScfOptions {
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        grid_shell: 1,
        ..Default::default()
    };
    let edges: [f64; 4] = [18.0, 22.0, 26.0, 30.0];
    let mut errors = Vec::new();
    for edge in edges {
        let cell = Cell::new(
            water(),
            Lattice::cubic(edge),
            &data::load("sto-3g").unwrap(),
            7.0,
        )
        .unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged);
        let result = dielectric_tensors(
            &cell,
            &functional,
            &reference,
            &options,
            &ResponseSolverConfig::default(),
            false,
        )
        .unwrap();
        let mut alpha = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                alpha += edge.powi(3) * DIRECTION[a] * result.susceptibility[a][b] * DIRECTION[b];
            }
        }
        errors.push((alpha - exact).abs());
        assert!(alpha > 0.0);
        assert!(result.raw_asymmetry < 1e-8);
        assert!(result.imaginary_residual < 1e-10);
        assert!(result.fock_image_residual < 1e-8);
        assert!(result
            .field_solves
            .iter()
            .all(|(_, residual)| *residual <= 1e-10));
        println!("water DFPT hf L={edge}: alpha={alpha:.15e} molecular={exact:.15e} error={:.12e} asymmetry={:.12e} imaginary={:.12e} image_fock={:.12e} solves={:?}",errors.last().unwrap(),result.raw_asymmetry,result.imaginary_residual,result.fock_image_residual,result.field_solves);
    }
    // Additional physical finite-volume convergence, distinct from the
    // finite-field O(h^2) derivative fits above. Dipolar local fields decay
    // as inverse volume; a finite Berry mesh is not identified with d/dk.
    let inverse_edges: Vec<_> = edges.iter().map(|edge| 1.0 / edge).collect();
    let fit = convergence::error_power_fit(&inverse_edges, &errors);
    println!(
        "water DFPT dilute limit: h=1/L {inverse_edges:?} errors={errors:?} p={:.9} R2={:.12}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 3.0).abs() < 0.15 && fit.r_squared > 0.999);
}
