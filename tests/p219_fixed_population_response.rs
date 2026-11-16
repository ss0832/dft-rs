// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent fixed-population electric and moving-metric response oracles.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::hessian::analytic::analytic_hessian_full;
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::response::ensemble::EnsembleSolver;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::occupations::FixedOccupations;

fn method(xc: &str, unrestricted: bool) -> MolecularMethod {
    MolecularMethod {
        xc: Some(xc.into()),
        basis: "6-31g".into(),
        unrestricted,
        grid_level: 4,
        dispersion: "none".into(),
        max_iter: 500,
        conv_tol: 1e-14,
        conv_tol_grad: Some(1e-12),
        conv_tol_dp: 1e-12,
        field: Some(Vec3::new(0.001, -0.002, 0.003)),
        fixed_occupations: Some(FixedOccupations {
            populations: [
                vec![0.68, 0.22, 0.08, 0.02],
                if unrestricted {
                    vec![0.61, 0.26, 0.10, 0.03]
                } else {
                    vec![0.68, 0.22, 0.08, 0.02]
                },
            ],
        }),
        ..Default::default()
    }
}

fn geometry(displacement: f64) -> [[f64; 3]; 2] {
    [[0.0, 0.0, -0.8], [0.2, 0.1, 0.8 + displacement]]
}

fn weighted(scf: &dft_rs::scf::ScfResult) -> Matrix {
    let mut sum = Matrix::zeros(scf.overlap.rows, scf.overlap.cols);
    for spin in 0..2 {
        let c = &scf.mo_coeff[spin];
        for i in 0..c.cols {
            for a in 0..c.rows {
                for b in 0..c.rows {
                    sum[(a, b)] +=
                        c[(a, i)] * c[(b, i)] * scf.occupations[spin][i] * scf.mo_energy[spin][i];
                }
            }
        }
    }
    sum
}

fn fit(label: &str, steps: &[f64], errors: &[f64]) -> bool {
    let f = convergence::error_power_fit(steps, errors);
    println!("fixed response {label} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} C={:.12e} log_rmse={:.12e}",
        f.exponent, f.r_squared, f.prefactor, f.log_rmse);
    (f.exponent - 2.0).abs() < 0.025 && f.r_squared > 0.9999
}

#[test]
fn electric_and_moving_metric_responses_have_independent_four_step_fits() {
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut accepted = true;
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0", "cam-b3lyp"] {
        for unrestricted in [false, true] {
            let m = method(xc, unrestricted);
            let at = |t, field| {
                let mut model = m.clone();
                model.field = Some(field);
                let setup = prepare_molecular(&[1, 1], &geometry(t), &model).unwrap();
                let reference = run_molecular_reference(&setup).unwrap();
                (setup, reference)
            };
            let field = m.field.unwrap();
            let (setup, r) = at(0.0, field);
            let solver = EnsembleSolver::new(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &r.scf,
                &setup.options,
            )
            .unwrap();
            let electric = solver.polarizability(&config).unwrap();
            let analytic = analytic_hessian_full(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                &r.scf,
                setup.grid.as_ref(),
            )
            .unwrap();
            let steps = [0.04, 0.02, 0.01, 0.005];
            let mut errors = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut hessian_estimates = Vec::new();
            let mut apt_estimates = Vec::new();
            for h in steps {
                let samples: Vec<_> = [-1.0, 1.0]
                    .iter()
                    .map(|sign| {
                        let (s, r) = at(sign * h, field);
                        let gradient = dft_rs::gradient::assemble_gradient(
                            &s.molecule,
                            &s.basis,
                            &s.functional,
                            s.grid.as_ref(),
                            &r.scf,
                            &s.options,
                        )
                        .unwrap();
                        let moment =
                            dft_rs::properties::dipole_moment(&s.molecule, &s.basis, &r.scf);
                        (
                            gradient[1].z,
                            r.scf.density.clone(),
                            weighted(&r.scf),
                            moment.z,
                        )
                    })
                    .collect();
                let hessian = (samples[1].0 - samples[0].0) / (2.0 * h);
                let apt = (samples[1].3 - samples[0].3) / (2.0 * h);
                hessian_estimates.push(hessian);
                apt_estimates.push(apt);
                errors[0].push((hessian - analytic.hessian[(5, 5)]).abs());
                errors[1].push((apt - analytic.dipole_derivatives[(2, 5)]).abs());
                let pdiff = (0..2)
                    .flat_map(|spin| {
                        (0..setup.basis.nao_sph)
                            .flat_map(move |i| (0..setup.basis.nao_sph).map(move |j| (spin, i, j)))
                    })
                    .map(|(spin, i, j)| {
                        ((samples[1].1[spin][(i, j)] - samples[0].1[spin][(i, j)]) / (2.0 * h)
                            - analytic.density_derivatives[5][spin][(i, j)])
                            .abs()
                    })
                    .fold(0.0_f64, f64::max);
                errors[2].push(pdiff);
                errors[3].push(
                    samples[1]
                        .2
                        .as_slice()
                        .iter()
                        .zip(samples[0].2.as_slice())
                        .zip(analytic.energy_weighted_density_derivatives[5].as_slice())
                        .map(|((a, b), c)| ((a - b) / (2.0 * h) - c).abs())
                        .fold(0.0_f64, f64::max),
                );
            }
            for (name, error) in ["Hessian", "APT", "density", "weighted density"]
                .iter()
                .zip(&errors)
            {
                accepted &= fit(
                    &format!("{xc} unrestricted={unrestricted} {name}"),
                    &steps,
                    error,
                );
            }
            let hs = &hessian_estimates;
            let aps = &apt_estimates;
            let richardson_h = ((4.0 * hs[3] - hs[2]) / 3.0 - analytic.hessian[(5, 5)]).abs();
            let richardson_apt =
                ((4.0 * aps[3] - aps[2]) / 3.0 - analytic.dipole_derivatives[(2, 5)]).abs();
            accepted &= richardson_h < 2e-7 && richardson_apt < 2e-7;
            let steps = [0.004, 0.002, 0.001, 0.0005];
            let mut electric_errors = Vec::new();
            let mut electric_estimates = Vec::new();
            for h in steps {
                let moments: Vec<_> = [-1.0, 1.0]
                    .iter()
                    .map(|sign| {
                        let (s, r) = at(0.0, field + Vec3::new(0.0, 0.0, sign * h));
                        dft_rs::properties::dipole_moment(&s.molecule, &s.basis, &r.scf).z
                    })
                    .collect();
                let estimate = (moments[1] - moments[0]) / (2.0 * h);
                electric_estimates.push(estimate);
                electric_errors.push((estimate - electric.tensor[2][2]).abs());
            }
            accepted &= fit(
                &format!("{xc} unrestricted={unrestricted} alpha_zz"),
                &steps,
                &electric_errors,
            );
            let es = &electric_estimates;
            let richardson_electric = ((4.0 * es[3] - es[2]) / 3.0 - electric.tensor[2][2]).abs();
            accepted &= richardson_electric < 2e-7;
            let residual = analytic
                .ensemble_response_residuals
                .iter()
                .copied()
                .chain(electric.responses.iter().map(|r| r.relative_residual))
                .fold(0.0_f64, f64::max);
            let number = analytic
                .ensemble_number_defects
                .iter()
                .flatten()
                .copied()
                .chain(electric.responses.iter().flat_map(|r| r.number_defects))
                .fold(0.0_f64, f64::max);
            println!("fixed response {xc} unrestricted={unrestricted} equation={residual:.12e} number={number:.12e} raw_asymmetry={:.12e} Richardson_H={richardson_h:.12e} Richardson_APT={richardson_apt:.12e} Richardson_alpha={richardson_electric:.12e}",analytic.raw_asymmetry);
            accepted &= residual < 1e-10 && number < 1e-10 && analytic.raw_asymmetry < 1e-8;
        }
    }
    assert!(
        accepted,
        "all measured derivative fits and residuals must pass without floors or discarded steps"
    );
}

#[test]
fn integer_and_uniform_population_limits_preserve_counts_and_gauges() {
    for xc in ["hf", "pbe", "tpss", "pbe0"] {
        let mut fixed = method(xc, false);
        fixed.fixed_occupations = Some(FixedOccupations {
            populations: [vec![1.0], vec![1.0]],
        });
        let mut ground = fixed.clone();
        ground.fixed_occupations = None;
        let calculate = |m: &MolecularMethod| {
            let s = prepare_molecular(&[1, 1], &geometry(0.0), m).unwrap();
            let r = run_molecular_reference(&s).unwrap();
            analytic_hessian_full(
                &s.molecule,
                &s.basis,
                &s.options,
                &s.functional,
                &r.scf,
                s.grid.as_ref(),
            )
            .unwrap()
        };
        let a = calculate(&fixed);
        let b = calculate(&ground);
        assert!(a.hessian.max_abs_diff(&b.hessian) < 1e-9);
        assert!(a.dipole_derivatives.max_abs_diff(&b.dipole_derivatives) < 1e-9);
        fixed.fixed_occupations = Some(FixedOccupations {
            populations: [vec![0.25; 4], vec![0.25; 4]],
        });
        let s = prepare_molecular(&[1, 1], &geometry(0.0), &fixed).unwrap();
        let r = run_molecular_reference(&s).unwrap();
        let solver = EnsembleSolver::new(
            &s.molecule,
            &s.basis,
            &s.functional,
            s.grid.as_ref(),
            &r.scf,
            &s.options,
        )
        .unwrap();
        let config = ResponseSolverConfig::default();
        let response = solver.polarizability(&config).unwrap();
        assert!(response.tensor.iter().flatten().all(|v| v.abs() < 1e-14));
        assert!(response
            .responses
            .iter()
            .all(|r| r.number_defects == [0.0; 2]));
        assert!(solver.solve(None, [1.0, 1.0], &config).is_err());
        assert!(solver
            .solve(
                Some(&[Matrix::zeros(1, 1), Matrix::zeros(1, 1)]),
                [0.0; 2],
                &config
            )
            .is_err());
    }
}

#[test]
fn thermal_electric_response_retains_canonical_population_derivatives() {
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut accepted = true;
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0"] {
        let mut m = method(xc, false);
        m.fixed_occupations = None;
        m.electronic_temperature = 80000.0;
        let at = |field| {
            let mut model = m.clone();
            model.field = Some(field);
            let setup = prepare_molecular(&[1, 1], &geometry(0.0), &model).unwrap();
            let reference = run_molecular_reference(&setup).unwrap();
            (setup, reference)
        };
        let field = m.field.unwrap();
        let (s, r) = at(field);
        let solver = EnsembleSolver::new(
            &s.molecule,
            &s.basis,
            &s.functional,
            s.grid.as_ref(),
            &r.scf,
            &s.options,
        )
        .unwrap();
        let electric = solver.polarizability(&config).unwrap();
        assert!(solver.population_model().contains("Fermi-Dirac"));
        let steps = [0.004, 0.002, 0.001, 0.0005];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let moments: Vec<_> = [-1.0, 1.0]
                .iter()
                .map(|sign| {
                    let (s, r) = at(field + Vec3::new(0.0, 0.0, sign * h));
                    dft_rs::properties::dipole_moment(&s.molecule, &s.basis, &r.scf).z
                })
                .collect();
            let estimate = (moments[1] - moments[0]) / (2.0 * h);
            estimates.push(estimate);
            errors.push((estimate - electric.tensor[2][2]).abs());
        }
        accepted &= fit(&format!("thermal {xc} alpha_zz"), &steps, &errors);
        let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - electric.tensor[2][2]).abs();
        accepted &= richardson < 2e-7;
        let gauge = solver
            .solve(
                Some(&[r.scf.overlap.clone(), r.scf.overlap.clone()]),
                [0.0; 2],
                &config,
            )
            .unwrap();
        let gauge_defect = gauge
            .delta_density
            .iter()
            .flat_map(|p| p.as_slice())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        println!("thermal electric {xc} Richardson={richardson:.12e} gauge_defect={gauge_defect:.12e} residuals={:?}",electric.responses.iter().map(|r|r.relative_residual).collect::<Vec<_>>());
        accepted &= gauge_defect < 1e-10
            && electric.responses.iter().all(|r| {
                r.relative_residual < 1e-10 && r.number_defects.iter().all(|v| *v < 1e-10)
            });
    }
    assert!(accepted);
}

#[test]
fn cached_and_direct_ensemble_kernels_have_the_same_physical_response() {
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0", "cam-b3lyp"] {
        let m = method(xc, true);
        let s = prepare_molecular(&[1, 1], &geometry(0.0), &m).unwrap();
        let r = run_molecular_reference(&s).unwrap();
        let solve = |options: &dft_rs::scf::ScfOptions| {
            EnsembleSolver::new(
                &s.molecule,
                &s.basis,
                &s.functional,
                s.grid.as_ref(),
                &r.scf,
                options,
            )
            .unwrap()
            .polarizability(&config)
            .unwrap()
        };
        let cached = solve(&s.options);
        let mut direct_options = s.options.clone();
        direct_options.incore_mb = 0;
        direct_options.eri_mode = dft_rs::scf::EriMode::Direct {
            screen_threshold: 1e-14,
        };
        let direct = solve(&direct_options);
        let mut difference = 0.0_f64;
        for axis in 0..3 {
            for spin in 0..2 {
                difference = difference.max(
                    cached.responses[axis].delta_density[spin]
                        .max_abs_diff(&direct.responses[axis].delta_density[spin]),
                );
            }
            for a in 0..3 {
                difference =
                    difference.max((cached.tensor[a][axis] - direct.tensor[a][axis]).abs());
            }
        }
        println!(
            "fixed response cached/direct {xc} maximum_density_tensor_difference={difference:.12e}"
        );
        assert!(difference < 1e-10);
    }
}

#[test]
fn fixed_population_translation_and_potential_gauge_identities_are_unmodified() {
    for (xc, unrestricted) in [("hf", false), ("tpss", true), ("pbe0", false)] {
        let m = method(xc, unrestricted);
        let s = prepare_molecular(&[1, 1], &geometry(0.0), &m).unwrap();
        let r = run_molecular_reference(&s).unwrap();
        let h = analytic_hessian_full(
            &s.molecule,
            &s.basis,
            &s.options,
            &s.functional,
            &r.scf,
            s.grid.as_ref(),
        )
        .unwrap();
        let mut translation = 0.0_f64;
        let mut apt_translation = 0.0_f64;
        for axis in 0..3 {
            for row in 0..6 {
                translation =
                    translation.max((h.hessian[(row, axis)] + h.hessian[(row, 3 + axis)]).abs());
            }
            for observable in 0..3 {
                apt_translation = apt_translation.max(
                    (h.dipole_derivatives[(observable, axis)]
                        + h.dipole_derivatives[(observable, 3 + axis)])
                        .abs(),
                );
            }
        }
        let solver = EnsembleSolver::new(
            &s.molecule,
            &s.basis,
            &s.functional,
            s.grid.as_ref(),
            &r.scf,
            &s.options,
        )
        .unwrap();
        let gauge = solver
            .solve(
                Some(&[r.scf.overlap.clone(), r.scf.overlap.clone()]),
                [0.0; 2],
                &ResponseSolverConfig::default(),
            )
            .unwrap();
        let defect = gauge
            .delta_density
            .iter()
            .flat_map(|p| p.as_slice())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        println!("fixed response identities {xc} unrestricted={unrestricted} Hessian_translation={translation:.12e} APT_translation={apt_translation:.12e} potential_gauge_density={defect:.12e}");
        assert!(translation < 1e-8 && apt_translation < 1e-8 && defect < 1e-10);
    }
}
