// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Noncanonical fixed-potential Newton Hessians against independent rotations.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::linalg::Matrix;
use dft_rs::scf::soscf::{rotate, OrbitalHessian};

#[test]
fn noncanonical_orbital_curvature_has_independent_four_step_fits() {
    let c = [Matrix::identity(4), Matrix::identity(4)];
    // Deliberately stale preconditioning energies cannot define the operator.
    let eps = [vec![-7.0, -3.0, 0.1, 4.0], vec![-6.0, -2.0, 0.4, 3.0]];
    let f = Matrix::from_row_major(
        4,
        4,
        vec![
            -0.8, 0.17, 0.03, -0.04, 0.17, -0.4, 0.06, 0.02, 0.03, 0.06, 0.2, 0.15, -0.04, 0.02,
            0.15, 0.7,
        ],
    );
    let fock = [f.clone(), f];
    for spins in [1, 2] {
        let oh = OrbitalHessian {
            c: &c,
            eps: &eps,
            n_occ: [2, 2],
            spins,
        };
        let tangent: Vec<_> = (0..oh.len()).map(|i| 0.11 + 0.07 * i as f64).collect();
        let exact = oh.fock_commutator(&tangent, &fock);
        let steps = [0.04, 0.02, 0.01, 0.005];
        let gradients = |h: f64| {
            let step: Vec<_> = tangent.iter().map(|u| u * h).collect();
            let mut rotated = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
            rotate(&oh, &step, &mut rotated);
            OrbitalHessian {
                c: &rotated,
                eps: &eps,
                n_occ: [2, 2],
                spins,
            }
            .gradient(&fock)
        };
        let samples: Vec<_> = steps
            .iter()
            .map(|h| {
                let p = gradients(*h);
                let m = gradients(-h);
                p.iter()
                    .zip(m)
                    .map(|(a, b)| (a - b) / (2.0 * h))
                    .collect::<Vec<_>>()
            })
            .collect();
        for i in 0..exact.len() {
            let errors: Vec<_> = samples.iter().map(|v| (v[i] - exact[i]).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson = ((4.0 * samples[3][i] - samples[2][i]) / 3.0 - exact[i]).abs();
            println!("orbital Newton spins={spins} slot={i} analytic={:.14e} steps={steps:?} errors={errors:?} p={:.12} R2={:.12} C={:.12e} log_rmse={:.12e} Richardson={richardson:.12e}",exact[i],fit.exponent,fit.r_squared,fit.prefactor,fit.log_rmse);
            assert!((fit.exponent - 2.0).abs() < 0.025 && fit.r_squared > 0.9999);
            assert!(richardson < 2e-7 + 1e-8 * exact[i].abs());
        }
    }
}

#[test]
fn finite_orbital_work_retains_tiny_diagonal_population_changes() {
    let c = [Matrix::identity(2), Matrix::identity(2)];
    let eps = [vec![-1.0, 0.5], vec![-1.0, 0.5]];
    let oh = OrbitalHessian {
        c: &c,
        eps: &eps,
        n_occ: [1, 1],
        spins: 1,
    };
    for angle in [0.02_f64, 1e-5, 1e-8, 1e-11] {
        let d = oh.finite_density_change(&[angle]);
        let diagonal = angle.sin().powi(2);
        let mixed = angle.sin() * angle.cos();
        assert!((d[0][(0, 0)] + diagonal).abs() < 2e-14 * diagonal);
        assert!((d[0][(1, 1)] - diagonal).abs() < 2e-14 * diagonal);
        assert!((d[0][(0, 1)] - mixed).abs() < 2e-14 * mixed);
        assert!((d[0][(0, 0)] + d[0][(1, 1)]).abs() < 2e-14 * diagonal);
        assert_eq!(d[0], d[1]);
    }
}
