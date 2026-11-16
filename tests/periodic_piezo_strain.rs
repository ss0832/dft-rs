// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact-source conversion tests; physical input-tensor validation is separate.
pub use dft_rs::error;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "../src/geomech/piezoelectric.rs"]
mod piezoelectric;

#[test]
fn all_strain_coefficients_match_an_independent_nonlinear_stress_oracle() {
    // A dense positive-definite compliance with all couplings nonzero.
    let s: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| if i == j { 0.6 + 0.07 * i as f64 } else { 0.025 })
    });
    let e: [[f64; 6]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| 0.11 + 0.04 * i as f64 + 0.017 * j as f64));
    let analytic = piezoelectric::strain_coefficients(&e, &s).unwrap();
    let c = dft_rs::geomech::compliance(&s).unwrap();
    // Minimize .5 eta.C.eta + sum(eta^4)/4 - stress.eta independently.
    // The cubic constitutive term produces a genuine central-difference h^2 error.
    let polarization = |axis: usize, stress: f64| -> [f64; 3] {
        let mut eta = [0.0; 6];
        for _ in 0..20 {
            let mut jac = dft_rs::linalg::Matrix::zeros(6, 6);
            let residual: Vec<f64> = (0..6)
                .map(|i| {
                    for j in 0..6 {
                        jac[(i, j)] = c[i][j];
                    }
                    jac[(i, i)] += 3.0 * eta[i] * eta[i];
                    (0..6).map(|j| c[i][j] * eta[j]).sum::<f64>() + eta[i] * eta[i] * eta[i]
                        - if i == axis { stress } else { 0.0 }
                })
                .collect();
            let step = dft_rs::linalg::solve_linear(&jac, &residual).unwrap();
            for i in 0..6 {
                eta[i] -= step[i];
            }
        }
        std::array::from_fn(|i| (0..6).map(|j| e[i][j] * eta[j]).sum())
    };
    let steps = [0.04, 0.02, 0.01, 0.005];
    for j in 0..6 {
        let values: Vec<[f64; 3]> = steps
            .iter()
            .map(|&h| {
                let plus = polarization(j, h);
                let minus = polarization(j, -h);
                std::array::from_fn(|i| (plus[i] - minus[i]) / (2.0 * h))
            })
            .collect();
        for i in 0..3 {
            let errors: Vec<f64> = values
                .iter()
                .map(|v| (v[i] - analytic[i][j]).abs())
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            println!(
                "piezo d[{i},{j}] steps={steps:?} errors={errors:?} p={} R2={}",
                fit.exponent, fit.r_squared
            );
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
        }
    }
}

#[test]
fn engineering_shear_units_and_invalid_inputs_are_explicit() {
    let mut e = [[0.0; 6]; 3];
    e[0][3] = 2.0;
    let mut s = [[0.0; 6]; 6];
    s[3][3] = 0.25;
    assert_eq!(
        piezoelectric::strain_coefficients(&e, &s).unwrap()[0][3],
        0.5
    );
    e[1][1] = f64::NAN;
    assert!(piezoelectric::strain_coefficients(&e, &s).is_err());
    e[1][1] = 0.0;
    s[0][1] = 1.0;
    assert!(piezoelectric::strain_coefficients(&e, &s).is_err());
    s[0][1] = 0.0;
    s[2][2] = f64::INFINITY;
    assert!(piezoelectric::strain_coefficients(&e, &s).is_err());
}
