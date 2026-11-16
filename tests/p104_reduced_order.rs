// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance tests of the `2n + 1` rule on nuclear paths.
//!
//! Oracle. The energy coefficients from a first-order response are compared
//! with those of the complete third-order response of the nuclear Taylor
//! driver along the same path, and the Cartesian third derivatives with the
//! mixed derivatives of the complete route: the same integral series, but a
//! density path that carries its second- and third-order response.
//! The two response-route comparisons passed the phase B baseline repeat.
//! An additional four-step fit against independently converged displaced
//! SCF energies also passes for HF/PBE/PBE0: p=1.995439646..2.002158542,
//! minimum R2=0.999998593470; raw measurements are in docs/response.md.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::krylov::ResponseSolverConfig;
use dft_rs::response::nuclear::{
    cubic_force_constants_along, cubic_force_constants_reduced, mixed_nuclear_response,
    nuclear_taylor, quartic_force_constants_along, reduced_order_taylor,
};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-11,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

fn config() -> ResponseSolverConfig {
    ResponseSolverConfig {
        tolerance: 1e-11,
        ..ResponseSolverConfig::default()
    }
}

/// A bent chain of three hydrogen-like centres with two electrons: HeH+ and
/// a hydrogen atom would be open shells, so the molecule is H3+.
fn molecule() -> Molecule {
    let mut mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.05, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.55, -0.1, 0.2),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.6, 1.4, -0.15),
        },
    ]);
    mol.charge = 1;
    mol
}

#[test]
fn reduced_cubic_derivative_passes_independent_scf_energy_fits() {
    let mol = molecule();
    let direction = [[0.3, -0.2, 0.5], [-0.4, 0.6, 0.1], [0.2, 0.1, -0.7]];
    let steps = [0.08, 0.04, 0.02, 0.01];
    let set = data::load("sto-3g").unwrap();
    let opts = ScfOptions {
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        ..options()
    };
    let solver = ResponseSolverConfig {
        tolerance: 1e-12,
        ..config()
    };
    for name in ["hf", "pbe", "pbe0"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let reference_at = |t: f64| {
            let mut displaced = mol.clone();
            for (atom, d) in displaced.atoms.iter_mut().zip(&direction) {
                atom.position.x += t * d[0];
                atom.position.y += t * d[1];
                atom.position.z += t * d[2];
            }
            let basis = MolBasis::build(&displaced, &set).unwrap();
            let grid = (!functional.parts.is_empty()).then(|| {
                MolecularGrid::build(
                    &displaced,
                    &GridOptions {
                        level: 3,
                        ..GridOptions::default()
                    },
                )
                .unwrap()
            });
            let scf = run_scf(&displaced, &basis, &opts, &functional, grid.as_ref()).unwrap();
            assert!(
                scf.converged,
                "{name}: displaced SCF t={t} did not converge"
            );
            (displaced, basis, grid, scf)
        };
        let (centre, basis, grid, scf) = reference_at(0.0);
        let reduced = reduced_order_taylor(
            &centre,
            &basis,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &direction,
            1,
            &solver,
        )
        .unwrap();
        assert!(reduced.converged);
        let analytic = 6.0 * reduced.energy[3];
        let mut energies = std::collections::HashMap::<u64, f64>::new();
        let mut energy_at = |t: f64| {
            *energies
                .entry(t.to_bits())
                .or_insert_with(|| reference_at(t).3.energy)
        };
        let estimates: Vec<f64> = steps
            .iter()
            .map(|&h| {
                // Third derivative of the independently converged SCF energy;
                // finite differences are used only by this validation oracle.
                (energy_at(2.0 * h) - 2.0 * energy_at(h) + 2.0 * energy_at(-h)
                    - energy_at(-2.0 * h))
                    / (2.0 * h * h * h)
            })
            .collect();
        let errors: Vec<f64> = estimates.iter().map(|v| (v - analytic).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - analytic).abs();
        println!(
            "{name}: reduced cubic derivative={analytic:.15e} steps={steps:?} \
             errors={errors:?} p={:.9} R2={:.12} C={:.12e} log_rmse={:.12e} \
             Richardson={richardson:.12e} residuals={:?}",
            fit.exponent, fit.r_squared, fit.prefactor, fit.log_rmse, reduced.residuals
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.08,
            "{name}: p={}",
            fit.exponent
        );
        assert!(fit.r_squared > 0.9999, "{name}: R2={}", fit.r_squared);
        assert!(richardson < 2e-6, "{name}: Richardson={richardson:e}");
    }
}

#[test]
fn first_order_response_gives_the_third_order_energy_coefficient() {
    let mol = molecule();
    let direction = vec![[0.3, -0.2, 0.5], [-0.4, 0.6, 0.1], [0.2, 0.1, -0.7]];
    for name in ["hf", "pbe", "pbe0"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let opts = options();
        let grid = (!func.parts.is_empty()).then(|| {
            MolecularGrid::build(
                &mol,
                &GridOptions {
                    level: 3,
                    ..GridOptions::default()
                },
            )
            .unwrap()
        });
        let scf = run_scf(&mol, &mb, &opts, &func, grid.as_ref()).unwrap();
        assert!(scf.converged);
        let complete = nuclear_taylor(
            &mol,
            &mb,
            &func,
            grid.as_ref(),
            &scf,
            &opts,
            &direction,
            3,
            &config(),
        )
        .unwrap();
        let reduced = reduced_order_taylor(
            &mol,
            &mb,
            &func,
            grid.as_ref(),
            &scf,
            &opts,
            &direction,
            1,
            &config(),
        )
        .unwrap();
        assert_eq!(reduced.energy.len(), 4);
        let mut worst = 0.0_f64;
        for n in 0..=3 {
            worst = worst.max((reduced.energy[n] - complete.value.energy_coefficients[n]).abs());
        }
        println!(
            "{name}: coefficients from the first-order response {:?} against the complete \
             response {:?}; largest difference {worst:.3e}",
            reduced.energy, complete.value.energy_coefficients
        );
        assert!(worst < 1e-8, "{name}: difference {worst:e}");

        let cubic =
            cubic_force_constants_reduced(&mol, &mb, &func, grid.as_ref(), &scf, &opts, &config())
                .unwrap();
        let n = cubic.dimension;
        let unit = |coordinate: usize| -> Vec<[f64; 3]> {
            let mut d = vec![[0.0; 3]; mol.len()];
            d[coordinate / 3][coordinate % 3] = 1.0;
            d
        };
        let mut defect = 0.0_f64;
        for (i, j, k) in [(0, 0, 0), (0, 4, 8), (2, 2, 5), (1, 3, 6), (7, 7, 7)] {
            let mixed = mixed_nuclear_response(
                &mol,
                &mb,
                &func,
                grid.as_ref(),
                &scf,
                &opts,
                &[unit(i), unit(j), unit(k)],
                &config(),
            )
            .unwrap();
            let value = cubic.tensor[(i * n + j) * n + k];
            defect = defect.max((value - mixed.value.energy_derivative).abs());
        }
        println!(
            "{name}: cubic force constants from {} paths; largest difference to the complete \
             mixed response {defect:.3e}; roundoff estimate {:.3e}",
            cubic.paths, cubic.accumulation_roundoff
        );
        assert!(defect < 1e-7, "{name}: difference {defect:e}");
    }
}

/// The contracted force constants are the contractions of the complete
/// mixed derivatives: third order against the full cubic tensor, fourth
/// order against the mixed fourth derivatives of the complete response.
#[test]
fn contracted_force_constants_match_the_complete_mixed_derivatives() {
    let mol = molecule();
    let func = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let opts = options();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 3,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    assert!(scf.converged);
    let n = 3 * mol.len();
    let probe: Vec<f64> = (0..n).map(|i| 0.3 * ((i % 4) as f64 - 1.5)).collect();
    let other: Vec<f64> = (0..n).map(|i| 0.2 * ((i % 3) as f64 - 1.0)).collect();
    let cubic =
        cubic_force_constants_reduced(&mol, &mb, &func, Some(&grid), &scf, &opts, &config())
            .unwrap();
    let along = cubic_force_constants_along(
        &mol,
        &mb,
        &func,
        Some(&grid),
        &scf,
        &opts,
        &probe,
        &config(),
    )
    .unwrap();
    let mut cubic_defect = 0.0_f64;
    for j in 0..n {
        for k in 0..n {
            let expected: f64 = (0..n)
                .map(|i| probe[i] * cubic.tensor[(i * n + j) * n + k])
                .sum();
            cubic_defect = cubic_defect.max((along.matrix[(j, k)] - expected).abs());
        }
    }
    println!(
        "cubic contraction: {} paths, largest difference to the full tensor {cubic_defect:.3e}",
        along.paths
    );
    assert!(cubic_defect < 1e-7);

    let quartic = quartic_force_constants_along(
        &mol,
        &mb,
        &func,
        Some(&grid),
        &scf,
        &opts,
        &probe,
        &other,
        &config(),
    )
    .unwrap();
    let pattern = |values: &[f64]| -> Vec<[f64; 3]> {
        values.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    };
    let unit = |coordinate: usize| -> Vec<[f64; 3]> {
        let mut d = vec![[0.0; 3]; mol.len()];
        d[coordinate / 3][coordinate % 3] = 1.0;
        d
    };
    let mut quartic_defect = 0.0_f64;
    for (k, l) in [(0, 0), (1, 5), (4, 8)] {
        let mixed = mixed_nuclear_response(
            &mol,
            &mb,
            &func,
            Some(&grid),
            &scf,
            &opts,
            &[pattern(&probe), pattern(&other), unit(k), unit(l)],
            &config(),
        )
        .unwrap();
        quartic_defect =
            quartic_defect.max((quartic.matrix[(k, l)] - mixed.value.energy_derivative).abs());
    }
    println!(
        "quartic contraction: {} paths, largest difference to the complete mixed response \
         {quartic_defect:.3e}",
        quartic.paths
    );
    assert!(quartic_defect < 1e-6);
}
