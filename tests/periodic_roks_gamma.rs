// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Constrained periodic references and every physical common-orbital rotation.
pub use dft_rs::{linalg, logger, math, periodic, scf, xc, DftError, Result};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
pub use dft_rs::periodic::roks;

use dft_rs::driver::periodic::{prepare_periodic, PeriodicMethod, PeriodicSetup};
use dft_rs::driver::{Controls, System, Value};
use linalg::Matrix;
use periodic::BlochMatrix;
use roks::{GammaRoksProblem, MeshRoksProblem};

fn setup(functional: &str, closed: bool) -> PeriodicSetup {
    let mut positions = vec![[0.13, -0.21, 0.07], [1.91, 0.20, -0.19], [0.4, 1.7, 0.33]];
    if closed {
        positions.pop();
    }
    let system = System {
        numbers: vec![1; positions.len()],
        positions,
        cell: Some([[8.5, 0.0, 0.0], [0.31, 9.3, 0.0], [0.27, -0.19, 10.1]]),
        pbc: [true; 3],
    };
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", functional)
            .with("basis", "sto-3g")
            .with("dispersion", "none")
            .with("multiplicity", if closed { 1 } else { 2 })
            .with("overlap_cutoff", 6.0)
            .with("grid_shell", 1)
            .with("grid_level", 1)
            .with("grid_weight_cutoff", 0.0)
            .with("guess", "core")
            .with("conv_tol", 1e-13)
            .with("conv_tol_grad", 1e-10)
            .with("conv_tol_dp", 1e-11)
            .with("max_iter", 300),
    )
    .unwrap();
    prepare_periodic(&system, &PeriodicMethod::from_controls(&controls).unwrap()).unwrap()
}

fn rotate(c: &Matrix, p: usize, q: usize, theta: f64) -> Matrix {
    let mut next = c.clone();
    let (s, co) = theta.sin_cos();
    for i in 0..c.rows {
        next[(i, p)] = co * c[(i, p)] + s * c[(i, q)];
        next[(i, q)] = -s * c[(i, p)] + co * c[(i, q)];
    }
    next
}

fn rotate_bloch(c: &BlochMatrix, p: usize, q: usize, angle: f64, imaginary: bool) -> BlochMatrix {
    if !imaginary {
        return BlochMatrix {
            re: rotate(&c.re, p, q, angle),
            im: rotate(&c.im, p, q, angle),
        };
    }
    let mut rotated = c.clone();
    let (s, co) = angle.sin_cos();
    for row in 0..c.rows() {
        rotated.re[(row, p)] = co * c.re[(row, p)] + s * c.im[(row, q)];
        rotated.im[(row, p)] = co * c.im[(row, p)] - s * c.re[(row, q)];
        rotated.re[(row, q)] = co * c.re[(row, q)] + s * c.im[(row, p)];
        rotated.im[(row, q)] = co * c.im[(row, q)] - s * c.re[(row, p)];
    }
    rotated
}

#[test]
fn mesh_roks_conjugate_orbital_gradients() {
    let mut setup = setup("hf", false);
    setup.options.k_mesh = [3, 1, 1];
    let problem = MeshRoksProblem::new(&setup.cell, &setup.functional, &setup.options).unwrap();
    let mut orbitals = problem.initial_orbitals().unwrap();
    let rotations = [(0, 1, 0.14), (0, 2, -0.21), (1, 2, 0.17)];
    for (p, q, theta) in rotations {
        for (k, band) in orbitals.iter_mut().enumerate() {
            band.coefficients = rotate_bloch(&band.coefficients, p, q, theta, false);
            if k != 0 {
                let angle = if k == 1 { 0.7 * theta } else { -0.7 * theta };
                band.coefficients = rotate_bloch(&band.coefficients, p, q, angle, true);
            }
        }
    }
    let state = problem.evaluate(&orbitals).unwrap();
    let directions: Vec<_> = rotations
        .into_iter()
        .flat_map(|(p, q, _)| [(0, p, q, false), (1, p, q, false), (1, p, q, true)])
        .collect();
    let names: Vec<_> = directions
        .iter()
        .map(|(k, p, q, im)| format!("ROKS rotation k={k} [{p},{q}] imaginary={im}"))
        .collect();
    let mut failures = Vec::new();
    use stencils::Stencil;
    stencils::fit_components_staged(
        &names,
        &[
            ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.06, 0.04, 0.03, 0.02], Stencil::Fourth),
        ],
        |steps, stencil| {
            directions
                .iter()
                .map(|&(k, p, q, imaginary)| {
                    let gradient = &state.orbital_gradient[k];
                    let component = if imaginary {
                        gradient.im[(p, q)]
                    } else {
                        gradient.re[(p, q)]
                    };
                    let analytic = component * if k == 0 { 1.0 / 3.0 } else { 2.0 / 3.0 };
                    let energy = |h| {
                        let mut trial = orbitals.clone();
                        trial[k].coefficients =
                            rotate_bloch(&orbitals[k].coefficients, p, q, h, imaginary);
                        if k != 0 {
                            let sign = if imaginary { -1.0 } else { 1.0 };
                            trial[2].coefficients =
                                rotate_bloch(&orbitals[2].coefficients, p, q, sign * h, imaginary);
                        }
                        problem.evaluate(&trial).unwrap().energy
                    };
                    steps.map(|h| {
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (energy(2.0 * h), energy(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(energy(h), energy(-h), plus2, minus2, h) - analytic).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures(
        "all independent conjugate mesh orbital gradients",
        &failures,
    );
}

#[test]
fn all_common_orbital_energy_derivatives_and_open_shell_references() {
    for functional in ["hf", "pbe"] {
        let setup = setup(functional, false);
        let problem =
            GammaRoksProblem::new(&setup.cell, &setup.functional, &setup.options).unwrap();
        let mut c = problem.initial_orbitals().unwrap();
        for (p, q, theta) in [(0, 1, 0.14), (0, 2, -0.21), (1, 2, 0.17)] {
            c = rotate(&c, p, q, theta);
        }
        let state = problem.evaluate(&c).unwrap();
        let steps = [0.04, 0.02, 0.01, 0.005];
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let analytic = state.orbital_gradient[(p, q)];
            assert!(analytic.abs() > 1e-5);
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| {
                    let plus = problem.evaluate(&rotate(&c, p, q, h)).unwrap().energy;
                    let minus = problem.evaluate(&rotate(&c, p, q, -h)).unwrap().energy;
                    ((plus - minus) / (2.0 * h) - analytic).abs()
                })
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            assert!(fit.prefactor > 0.0 && fit.log_rmse.is_finite());
            println!("Gamma ROKS {functional} rotation[{p},{q}] steps={steps:?} errors={errors:?} p={} R2={}", fit.exponent, fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.2 && fit.r_squared > 0.999);
        }
        let result = problem.run().unwrap();
        assert_eq!((result.n_alpha, result.n_beta), (2, 1));
        assert!(result.state.orbital_residual < 1e-10);
        assert!(result.density_change < 1e-11 && result.energy_change < 1e-13);
        assert!(result.state.metric_defect < 1e-12);
        let density = &result.state.density;
        let subset = density[0]
            .matmul_seq(&problem.overlap)
            .matmul_seq(&density[1]);
        assert!(subset.max_abs_diff(&density[1]) < 1e-12);
        for (spin, count) in [problem.n_alpha, problem.n_beta].iter().enumerate() {
            assert!((density[spin].frobenius_dot(&problem.overlap) - *count as f64).abs() < 1e-12);
        }
        println!("Gamma ROKS {functional} actual E={} residual={} dP={} dE={} metric={} iterations={} Exc={} Eel={} coefficient_shape={:?}", result.state.energy, result.state.orbital_residual, result.density_change, result.energy_change, result.state.metric_defect, result.iterations, result.state.xc_energy, result.state.electronic_energy, [result.coefficients.rows, result.coefficients.cols]);
        assert!(result
            .state
            .fock
            .iter()
            .all(|f| f.as_slice().iter().all(|x| x.is_finite())));
    }
}

#[test]
fn closed_shell_limit_matches_periodic_rks_and_unsupported_scopes_reject() {
    for functional in ["hf", "pbe"] {
        let setup = setup(functional, true);
        let problem =
            GammaRoksProblem::new(&setup.cell, &setup.functional, &setup.options).unwrap();
        let common = problem.run().unwrap();
        let restricted =
            periodic::scf::run_periodic_scf(&setup.cell, &setup.functional, &setup.options)
                .unwrap();
        assert!(restricted.converged);
        println!(
            "Gamma ROKS closed {functional} E={} RKS={} defect={}",
            common.state.energy,
            restricted.energy,
            (common.state.energy - restricted.energy).abs()
        );
        assert!((common.state.energy - restricted.energy).abs() < 1e-10);
        let mut unsupported = setup.options.clone();
        unsupported.k_mesh = [2, 1, 1];
        assert!(GammaRoksProblem::new(&setup.cell, &setup.functional, &unsupported).is_err());
        unsupported.k_mesh = [1; 3];
        unsupported.occupation = periodic::occupations::OccupationScheme::FermiDirac {
            temperature_kelvin: 300.0,
        };
        assert!(GammaRoksProblem::new(&setup.cell, &setup.functional, &unsupported).is_err());
    }
}
