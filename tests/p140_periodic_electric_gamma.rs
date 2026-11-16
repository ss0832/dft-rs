// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fourth electric-enthalpy derivatives against independently converged
//! finite-field Berry dipoles on four distinct field steps.
#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::basis::data;
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::nuclear::{
    lattice_electric_tensor, lattice_finite_field, ElectricTensorSymmetry,
};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

#[test]
fn static_periodic_gamma_has_independent_four_step_field_fits_for_generic_parents() {
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
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0", "wb97x"] {
        if std::env::var("DFT_RS_TEST_MODEL")
            .ok()
            .is_some_and(|selected| selected != xc)
        {
            continue;
        }
        let functional = resolve(&FunctionalSpec::Combined(xc.into())).unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged);
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
        assert_eq!(gamma.shape, vec![3; 4]);
        assert_eq!(gamma.tensor.len(), 81);
        assert_eq!(gamma.independent_components, 81);
        let symmetric = lattice_electric_tensor(
            &cell,
            &functional,
            &reference,
            &options,
            1,
            3,
            ElectricTensorSymmetry::IntrinsicPermutation,
            &config,
        )
        .unwrap();
        assert_eq!(symmetric.independent_components, 15);
        assert_eq!(symmetric.permutation_residual, None);
        let symmetry_difference = gamma
            .tensor
            .iter()
            .zip(&symmetric.tensor)
            .map(|(raw, expanded)| (raw - expanded).abs())
            .fold(0.0_f64, f64::max);
        assert!(symmetry_difference < 1e-12);
        let mut exact = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        exact += gamma.tensor[((3 * a + b) * 3 + c) * 3 + d]
                            * direction[a]
                            * direction[b]
                            * direction[c]
                            * direction[d];
                    }
                }
            }
        }
        assert!(exact.abs() > 1e-8);
        let mut cache = Vec::<(f64, f64)>::new();
        let mut max_reference_residual = 0.0_f64;
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut estimates = Vec::new();
        for h in steps {
            let mut values = Vec::new();
            for scale in [2.0 * h, h, -h, -2.0 * h] {
                let dipole =
                    if let Some((_, value)) = cache.iter().find(|(field, _)| *field == scale) {
                        *value
                    } else {
                        let finite = lattice_finite_field(
                            &cell,
                            &functional,
                            &reference,
                            &options,
                            1,
                            direction.map(|u| scale * u),
                            false,
                        )
                        .unwrap();
                        max_reference_residual = max_reference_residual.max(finite.residual);
                        let value = finite
                            .dipole
                            .iter()
                            .zip(direction)
                            .map(|(mu, u)| mu * u)
                            .sum::<f64>()
                            / finite.volume;
                        cache.push((scale, value));
                        value
                    };
                values.push(dipole);
            }
            estimates.push(
                (values[0] - 2.0 * values[1] + 2.0 * values[2] - values[3]) / (2.0 * h.powi(3)),
            );
        }
        let errors: Vec<_> = estimates
            .iter()
            .map(|value| (value - exact).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
        println!("{xc} periodic gamma: value={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e} raw_permutation={:.12e} field_reference={max_reference_residual:.12e} response={:?} paths={}",
            fit.exponent, fit.r_squared, gamma.permutation_residual.unwrap(), gamma.response_residuals, gamma.paths);
        println!("{xc} static permutation expansion: evaluated={} total={} raw_difference={symmetry_difference:.12e}", symmetric.independent_components, symmetric.tensor.len());
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999,
            "{xc}"
        );
        assert!(richardson < 1e-6, "{xc}: {richardson:e}");
        assert!(gamma.permutation_residual.unwrap() < 1e-10);
        assert!(max_reference_residual < 1e-13);
        assert!(gamma.response_residuals.iter().all(|value| *value < 1e-12));
    }
}
