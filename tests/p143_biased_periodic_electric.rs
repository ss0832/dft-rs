// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Derivatives of a stationary Berry enthalpy at nonzero electric bias.
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/electric.rs"]
mod electric;

use dft_rs::basis::data;
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::nuclear::{
    lattice_electric_tensor_at_field, lattice_field_response_at_field, lattice_finite_field,
    ElectricTensorSymmetry, FieldRequest,
};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

#[test]
fn biased_periodic_alpha_beta_gamma_pass_four_step_fits() {
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(-0.1, -0.05, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.1, 0.05, 0.7),
            },
        ]),
        Lattice::cubic(9.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
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
    let config = ResponseSolverConfig {
        tolerance: 1e-13,
        ..Default::default()
    };
    let direction = [0.2, -0.3, 0.7];
    let bias = direction.map(|u| 0.02 * u);
    for model in ["hf", "pbe", "tpss", "pbe0"] {
        let functional = resolve(&FunctionalSpec::Combined(model.into())).unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged);
        let first = lattice_field_response_at_field(
            &cell,
            &functional,
            &reference,
            &options,
            1,
            &FieldRequest::default(),
            bias,
            &config,
        )
        .unwrap();
        let mut responses = Vec::new();
        let chi: Vec<_> = first.susceptibility.iter().flatten().copied().collect();
        let mut exact = vec![(1, electric::contraction(&chi, 2, direction))];
        for order in 2..=3 {
            let response = lattice_electric_tensor_at_field(
                &cell,
                &functional,
                &reference,
                &options,
                1,
                order,
                ElectricTensorSymmetry::Unprojected,
                bias,
                &config,
            )
            .unwrap();
            assert_eq!(response.reference_field, bias);
            assert_eq!(
                response.independent_components,
                3usize.pow((order + 1) as u32)
            );
            assert!(response.permutation_residual.unwrap() < 1e-10);
            exact.push((
                order,
                electric::contraction(&response.tensor, order + 1, direction),
            ));
            responses.push(response);
        }
        let mut field_reference = 0.0_f64;
        electric::check_fields(
            &format!("biased periodic {model}"),
            &exact,
            |field| {
                let state = lattice_finite_field(
                    &cell,
                    &functional,
                    &reference,
                    &options,
                    1,
                    std::array::from_fn(|a| bias[a] + field * direction[a]),
                    false,
                )
                .unwrap();
                field_reference = field_reference.max(state.residual);
                electric::projection(state.dipole, direction) / state.volume
            },
            1e-6,
        );
        assert!(field_reference < 1e-13);
        assert!(first.response_residuals.iter().all(|r| *r <= 1e-13));
        for response in &responses {
            assert!(response.reference.residual < 1e-13);
            assert!(response.response_residuals.iter().all(|r| *r <= 1e-13));
            println!("biased periodic {model} order={} reference={:.12e} permutation={:.12e} response={:?}",
                response.order, response.reference.residual, response.permutation_residual.unwrap(), response.response_residuals);
        }
        println!("biased periodic {model} reference_field={bias:?} field_reference={field_reference:.12e} alpha_reference={:.12e} alpha_response={:?}", first.reference.residual, first.response_residuals);
    }
}
