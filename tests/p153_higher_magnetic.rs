// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Orbital magnetic energy derivatives against independent finite-field
//! London SCF calculations. No finite difference enters production.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::field_path::field_path_taylor;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::magnetic::{magnetic_field_scf, MagneticFieldOptions};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

#[test]
fn london_second_and_fourth_magnetic_energy_derivatives_have_four_step_fits() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ]);
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let direction = [0.6, -0.2, 0.1];
    let steps = [0.16, 0.08, 0.04, 0.02];
    for model in ["hf", "pbe", "tpss", "pbe0"] {
        let functional = resolve(&FunctionalSpec::Combined(model.into())).unwrap();
        let grid = (!functional.parts.is_empty()).then(|| {
            MolecularGrid::build(
                &molecule,
                &GridOptions {
                    level: 2,
                    ..Default::default()
                },
            )
            .unwrap()
        });
        let options = ScfOptions {
            e_tol: 1e-13,
            p_tol: 1e-13,
            max_iter: 250,
            ..Default::default()
        };
        let reference = run_scf(&molecule, &basis, &options, &functional, grid.as_ref()).unwrap();
        assert!(reference.converged);
        let exact = field_path_taylor(
            &molecule,
            &basis,
            &functional,
            grid.as_ref(),
            &reference,
            &options,
            [0.0; 3],
            direction,
            Vec3::zero(),
            true,
            4,
            &ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(exact.residuals.iter().all(|r| r.abs() < 1e-10));
        let at = |t: f64| {
            let result = magnetic_field_scf(
                &molecule,
                &basis,
                &functional,
                grid.as_ref(),
                &reference,
                &options,
                &MagneticFieldOptions {
                    field: direction.map(|b| t * b),
                    spin_zeeman: false,
                    series_order: Some(12),
                    series_tolerance: 1e-13,
                },
            )
            .unwrap();
            assert!(result.converged && result.residual < 1e-10);
            result.energy
        };
        let zero = at(0.0);
        assert!((zero - exact.energy[0]).abs() < 1e-9);
        let mut estimates = [Vec::new(), Vec::new()];
        for h in steps {
            let plus = at(h);
            let minus = at(-h);
            estimates[0].push((plus - 2.0 * zero + minus) / (h * h));
            estimates[1].push(
                (at(2.0 * h) - 4.0 * plus + 6.0 * zero - 4.0 * minus + at(-2.0 * h)) / h.powi(4),
            );
        }
        for (i, n) in [2, 4].into_iter().enumerate() {
            let value = exact.energy[n] * if n == 2 { 2.0 } else { 24.0 };
            let errors: Vec<_> = estimates[i].iter().map(|v| (v - value).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[i][3] - estimates[i][2]) / 3.0 - value).abs();
            println!("{model} London magnetic energy order={n} value={value:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
            assert!(rich < 1e-6);
        }
    }
}
