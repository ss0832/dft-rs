// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Independent radial magnetic-balance integrals and no-pair energy response.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::integrals::analytic_basis::AnalyticShell;
use dft_rs::integrals::dirac::{
    matrix_coefficients, restricted_kinetic_balance, restricted_magnetic_balance, GaussianNucleus,
};
use dft_rs::properties::no_pair::no_pair_projector_taylor;
use std::f64::consts::PI;
type Jet = Taylor<5>;

fn parameters(t: Jet) -> (Jet, Jet, Jet, Jet, [Jet; 3]) {
    let f = Jet::from_f64;
    let one = f(1.0);
    (
        f(0.8) * (one + t.scale(0.2)).powi(2),
        f(0.35) / (one - t.scale(0.11)),
        f(5.0) * (one + t.scale(0.07)),
        f(2.0) * (one + t.scale(0.13)),
        [
            f(0.3) / (one - t.scale(0.21)),
            f(-0.2) / (one + t.scale(0.12)),
            f(0.4) * (one + t.scale(0.3)).powi(2),
        ],
    )
}
fn closed(a: Jet, r: Jet, c: Jet, z: Jet, b: [Jet; 3]) -> [Jet; 7] {
    let f = Jet::from_f64;
    let one = f(1.0);
    let omega = f(1.5_f64.sqrt()) / r;
    let beta = a.scale(2.0);
    let v = -z * (beta / f(PI)).sqrt().scale(2.0) * omega / (beta + omega * omega).sqrt();
    // Direct radial integrals: <r^2 V>=3<V>/(4a)-d<V>/da/2.
    let rv = v * (one / (a.scale(2.0)) + one / (beta + omega * omega).scale(2.0));
    let b2 = b.iter().fold(f(0.0), |s, &x| s + x * x);
    let strength = b2.sqrt();
    let t = a.scale(1.5) + b2 / (a.scale(16.0));
    let w = rv * (a * a.scale(4.0) + b2.scale(1.0 / 6.0));
    let w_spin = rv * a.scale(4.0 / 3.0);
    let mut energy = f(0.0);
    let mut p_ls = f(0.0);
    for sign in [-1.0, 1.0] {
        let tb = t + strength.scale(0.5 * sign);
        let wb = w + w_spin * strength.scale(sign);
        let d = wb / (tb.scale(2.0)) - (c * c).scale(2.0);
        let delta = v - d;
        let radius = (delta * delta + (c * c * tb).scale(8.0)).sqrt();
        energy = energy + (v + (c * c * tb).scale(4.0) / (radius + delta)).scale(0.5);
        p_ls = p_ls + (c * c).scale(2.0) / radius.scale(2.0);
    }
    [
        t,
        -b[1].scale(0.5),
        w / (c * c).scale(4.0),
        -w_spin * b[1] / (c * c).scale(4.0),
        t / (c * c).scale(2.0),
        energy,
        p_ls,
    ]
}
fn production(a: Jet, r: Jet, c: Jet, z: Jet, b: [Jet; 3]) -> [Jet; 7] {
    let f = Jet::from_f64;
    let zero = f(0.0);
    let shell = AnalyticShell::normalized(0, [zero; 3], vec![a], vec![f(1.0)]).unwrap();
    let nucleus = GaussianNucleus {
        center: [zero; 3],
        charge: z,
        rms_radius: r,
    };
    let data = restricted_magnetic_balance(&[shell], &[nucleus], c, b, [zero; 3], 4).unwrap();
    let h =
        matrix_coefficients(&data.hamiltonian_real, Some(&data.hamiltonian_imaginary), 4).unwrap();
    let s = matrix_coefficients(&data.overlap, Some(&data.overlap_imaginary), 4).unwrap();
    for x in h.iter().chain(&s) {
        assert!(x.hermiticity_error() < 1e-12);
    }
    let response = no_pair_projector_taylor(&h, &s, 4, -c.value().powi(2), 1e-12).unwrap();
    for values in [
        &response.projector_defects,
        &response.metric_defects,
        &response.relative_commutator_defects,
        &response.number_defects,
    ] {
        assert!(values.iter().all(|x| *x < 1e-10));
    }
    let mut energy = zero;
    let mut p_ls = zero;
    for n in 0..5 {
        p_ls.c[n] = (response.projector[n].re[(0, 2)] + response.projector[n].re[(1, 3)]) * 0.5;
        for k in 0..=n {
            for i in 0..4 {
                for j in 0..4 {
                    energy.c[n] += (response.projector[k].re[(i, j)] * h[n - k].re[(j, i)]
                        - response.projector[k].im[(i, j)] * h[n - k].im[(j, i)])
                        * 0.5;
                }
            }
        }
    }
    println!(
        "RMB raw projector={:?} metric={:?} commutator={:?} number={:?}",
        response.projector_defects,
        response.metric_defects,
        response.relative_commutator_defects,
        response.number_defects
    );
    [
        (data.balanced_kinetic_real.values[0] + data.balanced_kinetic_real.values[3]).scale(0.5),
        data.balanced_kinetic_imaginary.values[1],
        (data.small_potential_real.values[0] + data.small_potential_real.values[3]).scale(0.5),
        data.small_potential_imaginary.values[1],
        (data.overlap.values[10] + data.overlap.values[15]).scale(0.5),
        energy,
        p_ls,
    ]
}
#[test]
fn magnetic_balance_and_no_pair_response_have_independent_radial_four_step_fits() {
    let (a, r, c, z, b) = parameters(Jet::from_coefficients(&[0.0, 1.0]));
    let actual = production(a, r, c, z, b);
    let exact = closed(a, r, c, z, b);
    for j in 0..7 {
        for n in 0..5 {
            let error = (actual[j].c[n] - exact[j].c[n]).abs();
            println!("RMB radial component={j} coefficient={n} error={error:.12e}");
            assert!(error < 1e-10);
        }
    }
    let steps = [0.08, 0.04, 0.02, 0.01];
    let mut passed = true;
    for j in 0..7 {
        for n in 1..=4 {
            let target = n as f64 * actual[j].c[n];
            let estimates: Vec<_> = steps
                .iter()
                .map(|&h| {
                    let evaluate = |x| {
                        let (a, r, c, z, b) = parameters(Jet::from_coefficients(&[x, 1.0]));
                        closed(a, r, c, z, b)[j].c[n - 1]
                    };
                    (evaluate(h) - evaluate(-h)) / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
            println!("RMB response component={j} order={n} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            passed &= (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999 && rich < 1e-7;
        }
    }
    assert!(passed);
}
#[test]
fn magnetic_balance_has_rkb_zero_field_time_reversal_and_translation_limits() {
    let shell = |center| AnalyticShell::normalized(0, center, vec![0.8], vec![1.0]).unwrap();
    let nucleus = |center| GaussianNucleus {
        center,
        charge: 2.0,
        rms_radius: 0.35,
    };
    let zero = [0.0; 3];
    let b = [0.3, -0.2, 0.4];
    let displacement = [0.2, -0.3, 0.1];
    let base = restricted_kinetic_balance(&[shell(zero)], &[nucleus(zero)], 5.0, 0).unwrap();
    let rmb =
        restricted_magnetic_balance(&[shell(zero)], &[nucleus(zero)], 5.0, zero, zero, 0).unwrap();
    for (a, b) in base
        .hamiltonian_real
        .values
        .iter()
        .zip(&rmb.hamiltonian_real.values)
    {
        assert!((a - b).abs() < 1e-12);
    }
    let plus =
        restricted_magnetic_balance(&[shell(zero)], &[nucleus(zero)], 5.0, b, zero, 0).unwrap();
    let minus = restricted_magnetic_balance(
        &[shell(zero)],
        &[nucleus(zero)],
        5.0,
        b.map(|x| -x),
        zero,
        0,
    )
    .unwrap();
    let moved = restricted_magnetic_balance(
        &[shell(displacement)],
        &[nucleus(displacement)],
        5.0,
        b,
        displacement,
        0,
    )
    .unwrap();
    assert!(plus.overlap_imaginary.values.iter().any(|x| x.abs() > 1e-8));
    let map = [1, 0, 3, 2];
    let signs = [1.0, -1.0, 1.0, -1.0];
    for (p, m, t) in [
        (
            &plus.hamiltonian_real,
            &minus.hamiltonian_real,
            &moved.hamiltonian_real,
        ),
        (&plus.overlap, &minus.overlap, &moved.overlap),
    ] {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (p.values[4 * i + j] - signs[i] * signs[j] * m.values[4 * map[i] + map[j]])
                        .abs()
                        < 1e-12
                );
                assert!((p.values[4 * i + j] - t.values[4 * i + j]).abs() < 1e-12);
            }
        }
    }
    for (p, m, t) in [
        (
            &plus.hamiltonian_imaginary,
            &minus.hamiltonian_imaginary,
            &moved.hamiltonian_imaginary,
        ),
        (
            &plus.overlap_imaginary,
            &minus.overlap_imaginary,
            &moved.overlap_imaginary,
        ),
    ] {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (p.values[4 * i + j] + signs[i] * signs[j] * m.values[4 * map[i] + map[j]])
                        .abs()
                        < 1e-12
                );
                assert!((p.values[4 * i + j] - t.values[4 * i + j]).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn finite_field_electronic_sector_approaches_the_pauli_limit() {
    let f = Jet::from_f64;
    let (a, r, _, z, b) = parameters(f(0.0));
    let omega = f(1.5_f64.sqrt()) / r;
    let v =
        -z * (a.scale(2.0 / PI)).sqrt().scale(2.0) * omega / (a.scale(2.0) + omega * omega).sqrt();
    let b2 = b.iter().map(|x| x.value().powi(2)).sum::<f64>();
    let target = v.value() + 1.5 * a.value() + b2 / (16.0 * a.value());
    let steps = [0.05, 0.025, 0.0125, 0.00625];
    let errors: Vec<_> = steps
        .iter()
        .map(|&h| (production(a, r, f(1.0 / h), z, b)[5].value() - target).abs())
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "RMB Pauli limit inverse_c={steps:?} errors={errors:?} p={:.9} R2={:.12}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
}
