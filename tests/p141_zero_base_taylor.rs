// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Integer powers at a zero constant coefficient: exact polynomial and
//! nested-potential limits plus independent four-step derivative fits.
#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::dual::{DualN, Scalar};
use dft_rs::response::series::{evaluate_series, evaluate_series_many};

type D1 = DualN<f64, 1>;
type D2 = DualN<D1, 1>;
type D3 = DualN<D2, 1>;
type D4 = DualN<D3, 1>;

fn variable(x: f64) -> D4 {
    D4::variable(D3::variable(D2::variable(D1::variable(x, 0), 0), 0), 0)
}

fn derivatives(x: D4) -> [f64; 5] {
    [
        x.re.re.re.re,
        x.re.re.re.im[0],
        x.re.re.im[0].im[0],
        x.re.im[0].im[0].im[0],
        x.im[0].im[0].im[0].im[0],
    ]
}

#[test]
fn nested_integer_powers_preserve_zero_and_negative_power_derivatives() {
    for power in 0..=8 {
        let integer = derivatives(variable(0.0).powi(power));
        let real = derivatives(variable(0.0).powf(power as f64));
        for order in 0..=4 {
            let expected = if order == power as usize {
                (1..=order).product::<usize>() as f64
            } else {
                0.0
            };
            assert_eq!(integer[order], expected);
            assert_eq!(real[order], expected);
        }
    }
    let actual = derivatives(variable(1.2).powi(-3));
    let mut factor = 1.0;
    for (order, value) in actual.into_iter().enumerate() {
        if order > 0 {
            factor *= -(order as f64 + 2.0);
        }
        assert!((value - factor * 1.2_f64.powi(-3 - order as i32)).abs() < 1e-11);
    }
    assert_eq!(variable(1.0).powi(i32::MIN).re.re.re.im[0], i32::MIN as f64);
    assert!(!variable(0.0).powi(-1).value().is_finite());
    assert!(derivatives(variable(2.0).powi(-1100))
        .iter()
        .all(|x| x.is_finite()));
}

#[test]
fn nested_integer_power_derivatives_have_independent_four_step_fits() {
    let x = variable(0.0);
    let polynomial = (0..=8).fold(D4::from_f64(0.0), |sum, power| sum + x.powi(power));
    let actual = derivatives(polynomial);
    let steps: [f64; 4] = [0.08, 0.04, 0.02, 0.01];
    for order in 1..=4 {
        // Explicit scalar polynomial derivatives, independent of dual arithmetic.
        let lower = |x: f64| -> f64 {
            (order - 1..=8)
                .map(|power| {
                    let factor = (0..order - 1).fold(1.0, |f, k| f * (power - k) as f64);
                    factor * (0..power - order + 1).fold(1.0, |v, _| v * x)
                })
                .sum()
        };
        let errors: Vec<_> = steps
            .iter()
            .map(|&h| ((lower(h) - lower(-h)) / (2.0 * h) - actual[order]).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "nested zero-base order={order} steps={steps:?} errors={errors:?} p={:.9} R2={:.12}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.05);
        assert!(fit.r_squared > 0.999);
    }
}

#[test]
fn nested_negative_power_derivatives_have_independent_four_step_fits() {
    let actual = derivatives((D4::from_f64(1.2) + variable(0.0).scale(0.2)).powi(-3));
    let steps: [f64; 4] = [0.08, 0.04, 0.02, 0.01];
    for order in 1..=4 {
        let factor = (0..order - 1).fold(1.0, |f, k| f * (-3.0 - k as f64) * 0.2);
        let lower = |t: f64| factor * (1.2 + 0.2 * t).powi(-2 - order as i32);
        let errors: Vec<_> = steps
            .iter()
            .map(|&h| ((lower(h) - lower(-h)) / (2.0 * h) - actual[order]).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!("nested negative-power order={order} steps={steps:?} errors={errors:?} p={:.9} R2={:.12}", fit.exponent, fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
    }
}

#[test]
fn zero_base_integer_powers_preserve_exact_coefficients_and_nested_tangents() {
    let mut input = vec![0.0; 9];
    input[1] = 1.0;
    for power in 2..=8 {
        let coefficients =
            evaluate_series(std::slice::from_ref(&input), |v| v[0].powf(power as f64)).unwrap();
        for (order, value) in coefficients.iter().enumerate() {
            assert_eq!(*value, if order == power { 1.0 } else { 0.0 });
        }
    }
    let values = evaluate_series_many(&[input], |v| {
        let x = DualN::<_, 1>::variable(v[0], 0);
        let y = x.powf(6.0);
        vec![y.re, y.im[0]]
    })
    .unwrap();
    assert_eq!(values[0][6], 1.0);
    assert_eq!(values[1][5], 6.0);
    assert!(values.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn zero_base_third_and_fourth_derivatives_have_independent_four_step_fits() {
    let coefficients = evaluate_series(&[vec![0.0, 1.0, 0.0, 0.0, 0.0]], |v| {
        v[0].powf(3.0) + v[0].powf(4.0) + v[0].powf(5.0) + v[0].powf(6.0)
    })
    .unwrap();
    let thirds = 6.0 * coefficients[3];
    let fourths = 24.0 * coefficients[4];
    assert_eq!(thirds, 6.0);
    assert_eq!(fourths, 24.0);
    // Direct scalar multiplication is independent of runtime Taylor power.
    let polynomial = |x: f64| x * x * x + x * x * x * x + x * x * x * x * x + x * x * x * x * x * x;
    let steps: [f64; 4] = [0.08, 0.04, 0.02, 0.01];
    let mut errors3 = Vec::new();
    let mut errors4 = Vec::new();
    for h in steps {
        let derivative3 = (polynomial(2.0 * h) - 2.0 * polynomial(h) + 2.0 * polynomial(-h)
            - polynomial(-2.0 * h))
            / (2.0 * h * h * h);
        let derivative4 = (polynomial(2.0 * h) - 4.0 * polynomial(h) + 6.0 * polynomial(0.0)
            - 4.0 * polynomial(-h)
            + polynomial(-2.0 * h))
            / (h * h * h * h);
        errors3.push((derivative3 - thirds).abs());
        errors4.push((derivative4 - fourths).abs());
    }
    for (name, errors) in [("third", errors3), ("fourth", errors4)] {
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "zero-base {name}: steps={steps:?} errors={errors:?} p={:.9} R2={:.12}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 1e-6);
        assert!(fit.r_squared > 0.999999999);
    }
}
