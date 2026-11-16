// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static clamped-ion electrostriction at fixed reduced electric field.
//! Reconverged strained references provide an independent four-step fit.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::data;
use dft_rs::math::Vec3;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::nuclear::{lattice_field_response, FieldRequest};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

#[test]
fn electrostriction_volume_and_reduced_field_factors_have_a_four_step_fit() {
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let options = PeriodicScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        max_iter: 250,
        ..Default::default()
    };
    let at = |strain: f64, derivative: bool| {
        let molecule = Molecule::new(
            [[-0.1, -0.05, -0.7], [0.1, 0.05, 0.7]]
                .iter()
                .map(|r| Atom {
                    z: 1,
                    position: Vec3::new(r[0], r[1], (1.0 + strain) * r[2]),
                })
                .collect(),
        );
        let cell = Cell::new(
            molecule,
            Lattice::new(
                Vec3::new(9.0, 0.0, 0.0),
                Vec3::new(0.0, 9.0, 0.0),
                Vec3::new(0.0, 0.0, 9.0 * (1.0 + strain)),
            ),
            &data::load("sto-3g").unwrap(),
            7.0,
        )
        .unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged);
        let result = lattice_field_response(
            &cell,
            &functional,
            &reference,
            &options,
            1,
            &FieldRequest {
                second_susceptibility: false,
                raman: false,
                strain_susceptibility: derivative,
                piezoelectric: false,
                relaxed: false,
            },
            &ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.response_residuals.iter().all(|v| v.abs() < 1e-10));
        result
    };
    let reference = at(0.0, true);
    let raw = reference.strain_susceptibility.as_ref().unwrap();
    // d(1/Omega)/d eta_zz=-1/Omega. Electric covectors follow
    // E_lab=F^-T E_reference, so each z index supplies one inverse stretch.
    let exact: Vec<_> = (0..3)
        .flat_map(|a| {
            (0..3).map(move |b| raw[(3 * a + b) * 6 + 2] - reference.susceptibility[a][b])
        })
        .collect();
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut estimates = Vec::new();
    for h in steps {
        let plus = at(h, false);
        let minus = at(-h, false);
        estimates.push(
            (0..3)
                .flat_map(|a| {
                    let p = &plus.susceptibility;
                    let m = &minus.susceptibility;
                    (0..3).map(move |b| {
                        let power = i32::from(a == 2) + i32::from(b == 2);
                        (p[a][b] / (1.0 + h).powi(power) - m[a][b] / (1.0 - h).powi(power))
                            / (2.0 * h)
                    })
                })
                .collect::<Vec<_>>(),
        );
    }
    let errors: Vec<_> = estimates
        .iter()
        .map(|e| {
            e.iter()
                .zip(&exact)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    let rich = estimates[3]
        .iter()
        .zip(&estimates[2])
        .zip(&exact)
        .map(|((a, b), v)| ((4.0 * a - b) / 3.0 - v).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("HF clamped electrostriction eta_zz steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e} raw_residual={:.12e}",fit.exponent,fit.r_squared,reference.response_residuals.iter().copied().fold(0.0_f64,f64::max));
    assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
    assert!(rich < 1e-7);
}
