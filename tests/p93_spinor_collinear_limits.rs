// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact empty-spin and zero-magnetization AO limits with independent fits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::periodic::spinor_scf::from_pauli;
use dft_rs::periodic::spinor_xc::spinor_semilocal_grid;
use dft_rs::periodic::BlochMatrix;
use dft_rs::xc::{resolve, FunctionalSpec};

#[test]
fn empty_spin_and_zero_spin_limits_are_rotation_covariant_and_variational() {
    let mut phi = BlochMatrix::zeros(1, 1);
    phi.re[(0, 0)] = 1.0;
    phi.im[(0, 0)] = 0.1;
    let mut momentum = std::array::from_fn(|_| BlochMatrix::zeros(1, 1));
    for (d, (re, im)) in [(0.2, 0.3), (0.1, -0.2), (-0.3, 0.1)]
        .into_iter()
        .enumerate()
    {
        momentum[d].re[(0, 0)] = re;
        momentum[d].im[(0, 0)] = im;
    }
    let theta = 0.47_f64;
    let axis = [theta.sin(), 0.0, theta.cos()];
    let steps = [0.04, 0.02, 0.01, 0.005];
    for polarization in [0.0, 1.0] {
        for name in ["svwn5", "pbe", "tpss", "r2scan", "m06l"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let density = |direction: [f64; 3]| {
                let mut p = std::array::from_fn(|_| BlochMatrix::zeros(1, 1));
                p[0].re[(0, 0)] = 0.9;
                for d in 0..3 {
                    p[d + 1].re[(0, 0)] = 0.9 * polarization * direction[d];
                }
                p
            };
            let p = density([0.0, 0.0, 1.0]);
            let evaluate = |p: [BlochMatrix; 4]| {
                spinor_semilocal_grid(&functional, &[0.7], &[&phi], &[&momentum], &[p], 16).unwrap()
            };
            let reference = evaluate(p.clone());
            let mut delta = p.clone();
            for m in &mut delta {
                m.scale(0.2);
            }
            let analytic = from_pauli(&delta).unwrap().trace_product(&reference.1[0]);
            let mut estimates = Vec::new();
            let mut errors = Vec::new();
            for h in steps {
                let displaced = |sign: f64| {
                    let mut varied = p.clone();
                    for m in &mut varied {
                        m.scale(1.0 + sign * 0.2 * h);
                    }
                    evaluate(varied).0
                };
                let estimate = (displaced(1.0) - displaced(-1.0)) / (2.0 * h);
                estimates.push(estimate);
                errors.push((estimate - analytic).abs());
            }
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - analytic).abs();
            let rotated = evaluate(density(axis));
            let rotation = (rotated.0 - reference.0).abs();
            println!(
                "collinear spin limit {name} polarization={polarization}: \
                steps={steps:?} errors={errors:?} p={:.9} R2={:.10} \
                Richardson={rich:.3e} rotation={rotation:.3e}",
                fit.exponent, fit.r_squared
            );
            assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
            assert!(rich < 1e-8);
            assert!(rotation < 1e-12);
            assert!(
                reference.1[0].hermiticity_error() < 1e-12
                    && rotated.1[0].hermiticity_error() < 1e-12
            );
        }
    }
}
