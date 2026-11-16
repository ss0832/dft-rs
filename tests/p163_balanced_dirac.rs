// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent centered Gaussian radial integrals and Dirac eigenvalue.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::integrals::analytic_basis::AnalyticShell;
use dft_rs::integrals::dirac::{matrix_coefficients, restricted_kinetic_balance, GaussianNucleus};
use dft_rs::properties::no_pair::no_pair_projector_taylor;
use std::f64::consts::PI;
type Jet = Taylor<5>;

fn parameters<S: Scalar>(t: S) -> (S, S, S, S) {
    let one = S::from_f64(1.0);
    (
        (one + t.scale(0.4)).powi(2).scale(0.8),
        (one - t.scale(0.1)).scale(0.35),
        (one + t.scale(0.07)).scale(5.0),
        (one + t.scale(0.05)).scale(2.0),
    )
}

/// Direct radial integration of normalized s Gaussians against erf(omega*r)/r.
/// No Gaussian integral engine, basis derivative or eigensolver is used.
fn oracle<S: Scalar>(a: S, r: S, c: S, z: S) -> [S; 5] {
    let f = S::from_f64;
    let omega = f(1.5_f64.sqrt()) / r;
    let b = a.scale(2.0);
    let k = a.scale(1.5);
    let expectation = (b / f(PI)).sqrt().scale(2.0) * omega / (b + omega * omega).sqrt();
    let v = -z * expectation;
    let w = -z * a * a / (c * c) * expectation * (f(1.0) / b + f(0.5) / (b + omega * omega));
    let s = k / (c * c).scale(2.0);
    let d = w / s - (c * c).scale(2.0);
    let trace = v + d;
    let determinant = v * d - (c * c * k).scale(2.0);
    let electronic =
        determinant.scale(2.0) / (trace - (trace * trace - determinant.scale(4.0)).sqrt());
    [v, w, k, s, electronic]
}

fn production(a: Jet, r: Jet, c: Jet, z: Jet) -> [Jet; 5] {
    let zero = Jet::from_f64(0.0);
    let shell = AnalyticShell::normalized(0, [zero; 3], vec![a], vec![Jet::from_f64(1.0)]).unwrap();
    let nucleus = GaussianNucleus {
        center: [zero; 3],
        charge: z,
        rms_radius: r,
    };
    let integrals = restricted_kinetic_balance(&[shell], &[nucleus], c, 4).unwrap();
    let h = matrix_coefficients(
        &integrals.hamiltonian_real,
        Some(&integrals.hamiltonian_imaginary),
        4,
    )
    .unwrap();
    let s = matrix_coefficients(&integrals.overlap, None, 4).unwrap();
    for m in h.iter().chain(&s) {
        assert!(m.hermiticity_error() < 1e-13);
    }
    let response = no_pair_projector_taylor(&h, &s, 4, -c.value() * c.value(), 1e-12).unwrap();
    assert_eq!(response.electronic_dimension, 2);
    let mut energy = zero;
    for n in 0..5 {
        for k in 0..=n {
            let p = &response.projector[k];
            let hh = &h[n - k];
            for i in 0..4 {
                for j in 0..4 {
                    energy.c[n] +=
                        0.5 * (p.re[(i, j)] * hh.re[(j, i)] - p.im[(i, j)] * hh.im[(j, i)]);
                }
            }
        }
    }
    println!(
        "balanced Dirac raw projector={:?} metric={:?} relative_commutator={:?} number={:?}",
        response.projector_defects,
        response.metric_defects,
        response.relative_commutator_defects,
        response.number_defects
    );
    [
        integrals.potential.values[0],
        integrals.small_potential_real.values[0],
        integrals.kinetic.values[0],
        integrals.overlap.values[2 * 4 + 2],
        energy,
    ]
}

#[test]
fn finite_nucleus_balanced_integrals_projector_and_energy_have_independent_four_step_fits() {
    let t = Jet::from_coefficients(&[0.0, 1.0]);
    let (a, r, c, z) = parameters(t);
    let actual = production(a, r, c, z);
    let expected = oracle(a, r, c, z);
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        for n in 0..5 {
            let error = (got.c[n] - want.c[n]).abs();
            assert!(error < 1e-10, "component={i} order={n}: {error:.12e}");
        }
    }
    let steps = [0.08, 0.04, 0.02, 0.01];
    for (component, label) in [
        (0, "large potential"),
        (1, "small potential"),
        (4, "electronic energy"),
    ] {
        for n in 1..=4 {
            let exact = actual[component].c[n] * n as f64;
            let estimates: Vec<_> = steps
                .iter()
                .map(|&h| {
                    let eval = |x| {
                        let (a, r, c, z) = parameters(Jet::from_coefficients(&[x, 1.0]));
                        oracle(a, r, c, z)[component].c[n - 1]
                    };
                    (eval(h) - eval(-h)) / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - exact).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
            println!("balanced Dirac {label} order={n} value={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
            assert!(rich < 1e-7);
        }
    }
    let (a, r, _, z) = parameters(0.0_f64);
    let steps = [0.1_f64, 0.05, 0.025, 0.0125];
    let nr = oracle(a, r, 5.0, z)[0] + 1.5 * a;
    let errors: Vec<_> = steps
        .iter()
        .map(|&inverse_c| {
            (production(
                Jet::from_f64(a),
                Jet::from_f64(r),
                Jet::from_f64(1.0 / inverse_c),
                Jet::from_f64(z),
            )[4]
            .value()
                - nr)
                .abs()
        })
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "balanced Dirac nonrelativistic limit steps={steps:?} errors={errors:?} p={:.9} R2={:.12}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
    let steps = [0.08, 0.04, 0.02, 0.01];
    let errors: Vec<_> = steps
        .iter()
        .map(|&radius| {
            let got = production(
                Jet::from_f64(a),
                Jet::from_f64(radius),
                Jet::from_f64(5.0),
                Jet::from_f64(z),
            );
            (got[0].value() + z * 2.0 * (2.0 * a / PI).sqrt()).abs()
        })
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "balanced Dirac point nucleus limit steps={steps:?} errors={errors:?} p={:.9} R2={:.12}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
}

#[test]
fn finite_nucleus_spin_orbit_matrix_keeps_complex_entries_and_translation_invariance() {
    let shell = |center| AnalyticShell::normalized(1, center, vec![0.8], vec![1.0]).unwrap();
    let nucleus = |center| GaussianNucleus {
        center,
        charge: 2.0,
        rms_radius: 0.35,
    };
    let origin =
        restricted_kinetic_balance(&[shell([0.0; 3])], &[nucleus([0.0; 3])], 5.0, 0).unwrap();
    let shift = [0.3, -0.2, 0.1];
    let translated =
        restricted_kinetic_balance(&[shell(shift)], &[nucleus(shift)], 5.0, 0).unwrap();
    let h = |data: &dft_rs::integrals::dirac::BalancedDiracIntegrals<f64>| {
        dft_rs::periodic::BlochMatrix {
            re: dft_rs::linalg::Matrix::from_row_major(
                12,
                12,
                data.hamiltonian_real.values.clone(),
            ),
            im: dft_rs::linalg::Matrix::from_row_major(
                12,
                12,
                data.hamiltonian_imaginary.values.clone(),
            ),
        }
    };
    assert!(h(&origin).hermiticity_error() < 1e-13);
    assert!(h(&origin).re.max_abs_diff(&h(&translated).re) < 1e-13);
    assert!(h(&origin).im.max_abs_diff(&h(&translated).im) < 1e-13);
    // For a radial potential: spin-dependent W/(4c^2) = <V'/r>*sigma.L/(4c^2).
    // A normalized p shell has sigma.L eigenvalues +1 (four states), -2 (two).
    let b = 1.6_f64;
    let omega = 1.5_f64.sqrt() / 0.35;
    let coefficient = 2.0 * (4.0 / 3.0) * b.powf(1.5) / PI.sqrt()
        * (omega * omega / (b + omega * omega)).powf(1.5)
        / (4.0 * 25.0);
    let diag = origin.small_potential_real.values[0];
    let mut spin = dft_rs::periodic::BlochMatrix::zeros(6, 6);
    for i in 0..6 {
        for j in 0..6 {
            spin.re[(i, j)] =
                origin.small_potential_real.values[i * 6 + j] - if i == j { diag } else { 0.0 };
            spin.im[(i, j)] = origin.small_potential_imaginary.values[i * 6 + j];
        }
    }
    let (eigenvalues, _) = dft_rs::linalg::symmetric_eigen(&spin.real_embedding()).unwrap();
    let mut eigenvalues = eigenvalues;
    eigenvalues.sort_by(f64::total_cmp);
    for (i, &value) in eigenvalues.iter().enumerate() {
        let expected = if i < 4 {
            -2.0 * coefficient
        } else {
            coefficient
        };
        assert!(
            (value - expected).abs() < 1e-12,
            "spin-orbit eigenvalue {i}: {value} vs {expected}"
        );
    }
    println!(
        "balanced Dirac closed spin-orbit coefficient={coefficient:.15e} Hermiticity={:.12e}",
        h(&origin).hermiticity_error()
    );
}

fn displaced_oracles(t: Jet) -> [Jet; 3] {
    let f = Jet::from_f64;
    let a = f(0.8);
    let c = f(5.0);
    let z = f(2.0);
    let radius = f(0.35) + t.scale(0.02);
    let other_radius = f(0.3) - t.scale(0.01);
    let displacement = [
        f(0.4) - t.scale(0.3),
        f(0.2) + t.scale(0.03),
        f(-0.1) + t.scale(0.04),
    ];
    let r2 = displacement.iter().fold(f(0.0), |sum, &x| sum + x * x);
    let r = r2.sqrt();
    let omega = f(1.5_f64.sqrt()) / radius;
    let k = (a.scale(2.0) * omega * omega / (a.scale(2.0) + omega * omega)).sqrt();
    let v = -z * (k * r).erf() / r;
    // Laplacian of the Gaussian-convolved potential fixes the trace of W:
    // W = 3*a*<V> + (1/4)*Laplacian(<V>), for a normalized s Gaussian.
    let w = (a * v.scale(3.0) + z * k.powi(3).scale(1.0 / PI.sqrt()) * (-k * k * r2).exp())
        / (c * c).scale(4.0);
    let nuclear_width = (f(1.5) / (radius * radius + other_radius * other_radius)).sqrt();
    let nuclear = z * (nuclear_width * r).erf() / r;
    [v, w, nuclear]
}

#[test]
fn displaced_basis_finite_nuclei_and_nuclear_interaction_have_four_step_fits() {
    let f = Jet::from_f64;
    let t = Jet::from_coefficients(&[0.0, 1.0]);
    let origin = [t.scale(0.2), f(0.0), f(0.0)];
    let target = [
        f(0.4) - t.scale(0.1),
        f(0.2) + t.scale(0.03),
        f(-0.1) + t.scale(0.04),
    ];
    let radius = f(0.35) + t.scale(0.02);
    let other_radius = f(0.3) - t.scale(0.01);
    let shell = AnalyticShell::normalized(0, origin, vec![f(0.8)], vec![f(1.0)]).unwrap();
    let nucleus = GaussianNucleus {
        center: target,
        charge: f(2.0),
        rms_radius: radius,
    };
    let single =
        restricted_kinetic_balance(std::slice::from_ref(&shell), &[nucleus], f(5.0), 4).unwrap();
    let pair = restricted_kinetic_balance(
        &[shell],
        &[
            nucleus,
            GaussianNucleus {
                center: origin,
                charge: f(1.0),
                rms_radius: other_radius,
            },
        ],
        f(5.0),
        4,
    )
    .unwrap();
    let actual = [
        single.potential.values[0],
        single.small_potential_real.values[0],
        pair.nuclear_repulsion,
    ];
    let expected = displaced_oracles(t);
    for (a, b) in actual.iter().zip(expected) {
        for n in 0..5 {
            assert!((a.c[n] - b.c[n]).abs() < 1e-11);
        }
    }
    let steps = [0.08, 0.04, 0.02, 0.01];
    for (component, label) in [
        "displaced large potential",
        "displaced small potential",
        "Gaussian nuclear interaction",
    ]
    .iter()
    .enumerate()
    {
        for n in 1..=4 {
            let exact = actual[component].c[n] * n as f64;
            let estimates: Vec<_> = steps
                .iter()
                .map(|&h| {
                    let plus = displaced_oracles(Jet::from_coefficients(&[h, 1.0]));
                    let minus = displaced_oracles(Jet::from_coefficients(&[-h, 1.0]));
                    (plus[component].c[n - 1] - minus[component].c[n - 1]) / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - exact).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
            println!("balanced Dirac {label} order={n} value={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
            assert!(rich < 1e-7);
        }
    }
}

fn radial_spin_orbit(t: Jet) -> Jet {
    let (a, r, c, z) = parameters(t);
    let omega = Jet::from_f64(1.5_f64.sqrt()) / r;
    let b = a.scale(2.0);
    z * b.powf(1.5).scale(4.0 / (3.0 * PI.sqrt())) * (omega * omega / (b + omega * omega)).powf(1.5)
        / (c * c).scale(4.0)
}

#[test]
fn finite_nucleus_spin_orbit_derivatives_have_independent_four_step_fits() {
    let t = Jet::from_coefficients(&[0.0, 1.0]);
    let (a, r, c, z) = parameters(t);
    let zero = Jet::from_f64(0.0);
    let shell = AnalyticShell::normalized(1, [zero; 3], vec![a], vec![Jet::from_f64(1.0)]).unwrap();
    let nucleus = GaussianNucleus {
        center: [zero; 3],
        charge: z,
        rms_radius: r,
    };
    let integrals = restricted_kinetic_balance(&[shell], &[nucleus], c, 4).unwrap();
    // CCA p order is [z,x,y]. <x|L_z|y>=-i fixes the absolute sign.
    let actual = -integrals.small_potential_imaginary.values[6 + 2];
    let expected = radial_spin_orbit(t);
    for n in 0..5 {
        assert!((actual.c[n] - expected.c[n]).abs() < 1e-12);
    }
    let steps = [0.08, 0.04, 0.02, 0.01];
    for n in 1..=4 {
        let exact = actual.c[n] * n as f64;
        let estimates: Vec<_> = steps
            .iter()
            .map(|&h| {
                (radial_spin_orbit(Jet::from_coefficients(&[h, 1.0])).c[n - 1]
                    - radial_spin_orbit(Jet::from_coefficients(&[-h, 1.0])).c[n - 1])
                    / (2.0 * h)
            })
            .collect();
        let errors: Vec<_> = estimates.iter().map(|x| (x - exact).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
        println!("balanced Dirac spin-orbit order={n} value={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
        assert!(rich < 1e-7);
    }
}
