// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent finite-field dipole stencils with raw four-step error fits.
use super::convergence;

pub fn projection(vector: [f64; 3], direction: [f64; 3]) -> f64 {
    vector
        .iter()
        .zip(direction)
        .map(|(value, u)| value * u)
        .sum()
}

pub fn contraction(tensor: &[f64], rank: usize, direction: [f64; 3]) -> f64 {
    tensor
        .iter()
        .enumerate()
        .map(|(mut index, value)| {
            let mut weight = 1.0;
            for _ in 0..rank {
                weight *= direction[index % 3];
                index /= 3;
            }
            value * weight
        })
        .sum()
}

pub fn check_fields(
    label: &str,
    exact: &[(usize, f64)],
    mut dipole: impl FnMut(f64) -> f64,
    richardson_bound: f64,
) {
    let steps: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];
    let mut cache = Vec::<(f64, f64)>::new();
    let mut sample = |field: f64| {
        if let Some((_, value)) = cache.iter().find(|(key, _)| *key == field) {
            return *value;
        }
        let value = dipole(field);
        cache.push((field, value));
        value
    };
    let centre = sample(0.0);
    for &(order, value) in exact {
        assert!(
            value.abs() > 1e-8,
            "{label} derivative {order} must be nonzero"
        );
        let estimates: Vec<_> = steps
            .iter()
            .map(|&h| match order {
                1 => (sample(h) - sample(-h)) / (2.0 * h),
                2 => (sample(h) - 2.0 * centre + sample(-h)) / (h * h),
                3 => {
                    (sample(2.0 * h) - 2.0 * sample(h) + 2.0 * sample(-h) - sample(-2.0 * h))
                        / (2.0 * h * h * h)
                }
                _ => panic!("invalid validation derivative order"),
            })
            .collect();
        let errors: Vec<_> = estimates
            .iter()
            .map(|estimate| (estimate - value).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - value).abs();
        println!("{label} derivative={order} value={value:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e}",fit.exponent,fit.r_squared);
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999,
            "{label} derivative {order}"
        );
        assert!(
            richardson < richardson_bound,
            "{label} derivative {order}: {richardson:e}"
        );
    }
}
