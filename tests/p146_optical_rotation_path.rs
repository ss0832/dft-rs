// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Optical-rotation nuclear derivatives on independently displaced states.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::dynamic::{
    dynamic_optical_rotation, dynamic_optical_rotation_velocity, Frequency,
};
use dft_rs::response::nuclear::optical_tensor_path;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

#[test]
fn optical_rotation_path_has_unprojected_four_step_fits_in_both_gauges() {
    let coordinates = [
        [-0.7, 0., 0.],
        [0.7, 0., 0.],
        [0.2, 0.8, 1.3],
        [-0.4, -0.8, 1.8],
    ];
    let direction = [
        [0., 0., 0.],
        [0., 0., 0.],
        [0.3, 0.1, -0.2],
        [-0.1, 0.2, 0.1],
    ];
    let molecule = |step: f64| {
        Molecule::new(
            coordinates
                .iter()
                .zip(direction)
                .map(|(r, u)| Atom {
                    z: 1,
                    position: Vec3::new(r[0] + step * u[0], r[1] + step * u[1], r[2] + step * u[2]),
                })
                .collect(),
        )
    };
    let data = data::load("sto-3g").unwrap();
    let grid_options = GridOptions {
        radial_override: Some(10),
        angular_override: Some(7),
        prune: false,
        ..Default::default()
    };
    let options = ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        max_iter: 250,
        ..Default::default()
    };
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let origin = Vec3::new(0.1, 0.2, -0.1);
    let omega = 0.05;
    for model in ["hf", "pbe", "tpss", "pbe0"] {
        if std::env::var("DFT_RS_TEST_MODEL")
            .ok()
            .is_some_and(|selected| selected != model)
        {
            continue;
        }
        let functional = resolve(&FunctionalSpec::Combined(model.into())).unwrap();
        let centre = molecule(0.0);
        let basis = MolBasis::build(&centre, &data).unwrap();
        let grid = (!functional.parts.is_empty())
            .then(|| MolecularGrid::build(&centre, &grid_options).unwrap());
        let scf = run_scf(&centre, &basis, &options, &functional, grid.as_ref()).unwrap();
        assert!(scf.converged);
        for velocity in [false, true] {
            let path = optical_tensor_path(
                &centre,
                &basis,
                &functional,
                grid.as_ref(),
                &scf,
                &options,
                &direction,
                1,
                omega,
                origin,
                velocity,
                &config,
            )
            .unwrap();
            assert!(path.converged);
            let norm = path.rotation[1]
                .iter()
                .flatten()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(norm > 1e-8);
            let dynamic = |step: f64, origin| {
                let mol = molecule(step);
                let basis = MolBasis::build(&mol, &data).unwrap();
                let grid = (!functional.parts.is_empty())
                    .then(|| MolecularGrid::build(&mol, &grid_options).unwrap());
                let scf = run_scf(&mol, &basis, &options, &functional, grid.as_ref()).unwrap();
                assert!(scf.converged);
                let frequency = Frequency::Retarded {
                    omega,
                    damping: 0.0,
                };
                let result = if velocity {
                    dynamic_optical_rotation_velocity(
                        &mol,
                        &basis,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &options,
                        frequency,
                        origin,
                        &config,
                    )
                } else {
                    dynamic_optical_rotation(
                        &mol,
                        &basis,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &options,
                        frequency,
                        origin,
                        &config,
                    )
                }
                .unwrap();
                assert!(result.residuals.iter().all(|r| *r < 1e-10));
                result.value.tensor.map(|row| row.map(|z| z.re))
            };
            let independent = dynamic(0.0, origin);
            let base_error = independent
                .iter()
                .flatten()
                .zip(path.rotation[0].iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                base_error < 1e-8,
                "{model} velocity={velocity} baseline {base_error:e}"
            );
            let steps = [0.08, 0.04, 0.02, 0.01];
            let mut estimates = Vec::new();
            for h in steps {
                let plus = dynamic(h, origin);
                let minus = dynamic(-h, origin);
                let tensor: Vec<_> = plus
                    .iter()
                    .flatten()
                    .zip(minus.iter().flatten())
                    .map(|(p, m)| (p - m) / (2.0 * h))
                    .collect();
                estimates.push(tensor);
            }
            let errors: Vec<_> = estimates
                .iter()
                .map(|tensor| {
                    tensor
                        .iter()
                        .zip(path.rotation[1].iter().flatten())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson = estimates[3]
                .iter()
                .zip(&estimates[2])
                .zip(path.rotation[1].iter().flatten())
                .map(|((fine, coarse), exact)| ((4.0 * fine - coarse) / 3.0 - exact).powi(2))
                .sum::<f64>()
                .sqrt();
            println!("{model} optical rotation velocity={velocity} norm={norm:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e} baseline={base_error:.12e} residuals={:?}",fit.exponent,fit.r_squared,path.residuals);
            assert!(
                (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999,
                "{model} velocity={velocity}"
            );
            assert!(richardson < 1e-6);
            if velocity {
                let moved = dynamic(0.0, origin + Vec3::new(0.3, -0.2, 0.1));
                let trace = |tensor: &[[f64; 3]; 3]| tensor[0][0] + tensor[1][1] + tensor[2][2];
                let origin_defect = (trace(&moved) - trace(&independent)).abs();
                println!("{model} modified velocity origin trace defect={origin_defect:.12e}");
                assert!(origin_defect < 1e-9);
            }
        }
    }
}
