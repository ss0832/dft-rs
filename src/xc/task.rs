// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! TASK exchange: Aschebrock and Kuemmel, PRResearch 1, 033082 (2019),
//! equations (21), (23), (30). The mTASK parameters follow Neupane et al.,
//! PRMaterials 5, 063803 (2021), supplemental equations (10)--(11).
//! These are exchange components; their named recipes add PW92 correlation.

use super::common::RHO_CUTOFF;
use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

const PARAMETER_NAMES: &[&str] = &[
    "c", "d", "h0", "a0", "a1", "a2", "b0", "b1", "b2", "b3", "b4",
];

/// Rational Chebyshev expansion T_n((x-1)/(x+1)).
fn rational_series<S: Scalar>(x: S, coefficients: &[S]) -> S {
    let one = S::from_f64(1.0);
    let mapped = (x - one) / (x + one);
    let mut previous = one;
    let mut current = mapped;
    let mut result = coefficients[0] + coefficients[1] * current;
    for &coefficient in &coefficients[2..] {
        let next = S::from_f64(2.0) * mapped * current - previous;
        result = result + coefficient * next;
        previous = current;
        current = next;
    }
    result
}

fn exchange<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    let one = S::from_f64(1.0);
    let zero = S::from_f64(0.0);
    let kf2 = (6.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0);
    let mut energy = zero;
    for (rho, sigma, kinetic) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let kinetic = if kinetic.value() < 1e-20 {
            S::from_f64(1e-20)
        } else {
            kinetic
        };
        let p = sigma / (S::from_f64(4.0 * kf2) * rho.powf(8.0 / 3.0));
        let uniform = S::from_f64(0.3 * kf2) * rho.powf(5.0 / 3.0);
        let pauli_fraction = one - sigma / (S::from_f64(8.0) * rho * kinetic);
        // Keep the documented fractional-Pauli convention rather than clipping
        // the resulting alpha. Inside this clamp, alpha still depends on tau.
        let pauli_fraction = if pauli_fraction.value() < 1e-10 {
            S::from_f64(1e-10)
        } else {
            pauli_fraction
        };
        let alpha = kinetic * pauli_fraction / uniform;
        let gx = if p.value() == 0.0 {
            // exp(-c/p^(1/4)) has an exactly flat smooth extension at p=0.
            one
        } else {
            let exponent = -th[0] / p.powf(0.25);
            if exponent.value().abs() < 0.01 {
                // Avoid subtraction loss in the asymptotic two-dimensional limit.
                let mut term = exponent;
                let mut sum = term;
                for n in 2..=8 {
                    term = term * exponent / S::from_f64(n as f64);
                    sum = sum + term;
                }
                -sum
            } else {
                one - exponent.exp()
            }
        };
        let h1 = rational_series(p, &th[3..6]);
        let interpolation = rational_series(alpha, &th[6..11]);
        let enhancement =
            th[2] * gx + (one - interpolation) * (h1 - th[2]) * (th[1] * gx.ln()).exp();
        energy =
            energy - S::from_f64(super::common::X_FACTOR_C) * rho.powf(4.0 / 3.0) * enhancement;
    }
    energy
}

/// Original TASK exchange with the published rounded coefficient set.
pub fn task_x() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "task",
        Kind::Exchange,
        vec![
            4.9479, 10.0, 1.174, 0.938719, -0.076371, -0.0150899, -0.628591, -2.10315, -0.5,
            0.103153, 0.128591,
        ],
        PARAMETER_NAMES,
        exchange,
        true
    )
}

/// mTASK exchange with the published modified upper bound and coefficients.
pub fn mtask_x() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "mtask",
        Kind::Exchange,
        vec![
            4.9479,
            10.0,
            1.29,
            0.924374,
            -0.09276847,
            -0.017143,
            -0.639572,
            -2.087488,
            -0.625,
            -0.162512,
            0.014572,
        ],
        PARAMETER_NAMES,
        exchange,
        true
    )
}
