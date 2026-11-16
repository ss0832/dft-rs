// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Active-descriptor contracts and independent complete first-to-fourth fits.
use super::*;
use crate::acceptance::{
    components, stencils, symmetric_cartesian_directions as symmetric_directions,
};

#[test]
fn external_directional_scalar_constructor_preserves_type_inference() {
    let coefficients = evaluate_series(&[vec![0.0]], |_| {
        let scalar = DirectionalTaylorScalar::from_f64(0.0);
        assert_eq!(scalar.value(), 0.0);
        scalar.re.re
    })
    .unwrap();
    assert_eq!(coefficients, vec![0.0]);
}

fn body<S: Scalar, const N: usize>(variables: &[S], parameters: &[S]) -> S {
    let mut argument = parameters[0];
    for (i, &variable) in variables.iter().take(N).enumerate() {
        argument = argument + S::from_f64(0.17 + 0.031 * i as f64) * variable;
    }
    argument.exp()
}

fn assert_coefficients_equal(left: &XcTaylorOutput, right: &XcTaylorOutput) {
    let left = left.energy.iter().chain(left.potentials.iter().flatten());
    let right = right.energy.iter().chain(right.potentials.iter().flatten());
    for (&a, &b) in left.zip(right) {
        assert!(a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0));
    }
}

fn compare_full_lanes<const N: usize>() {
    for order in 0..=5 {
        let input = XcTaylorInput {
            variables: std::array::from_fn(|i| {
                (0..=order)
                    .map(|k| {
                        if k == 0 {
                            0.19 + 0.037 * i as f64
                        } else {
                            (0.13 + 0.071 * i as f64) * 0.07_f64.powi(k as i32)
                        }
                    })
                    .collect()
            }),
            parameters: Some(vec![(0..=order)
                .map(|k| 0.23 * 0.11_f64.powi(k as i32))
                .collect()]),
        };
        let direction = std::array::from_fn(|i| {
            (0..=order)
                .map(|k| (0.29 - 0.017 * i as f64) * 0.09_f64.powi(k as i32))
                .collect()
        });
        let full = input
            .evaluate_directional_potentials(&direction, &[0.23], body::<_, N>)
            .unwrap();
        let active = input
            .evaluate_directional_potentials_for::<N>(&direction, &[0.23], body::<_, N>)
            .unwrap();
        assert_coefficients_equal(&full.value, &active.value);
        assert_coefficients_equal(&full.directional, &active.directional);
        let full = input.evaluate_potentials(&[0.23], body::<_, N>).unwrap();
        let active = input
            .evaluate_potentials_for::<N>(&[0.23], body::<_, N>)
            .unwrap();
        assert_coefficients_equal(&full, &active);
        let energy = input.evaluate(&[0.23], body::<_, N>).unwrap();
        assert_eq!(energy, active.energy);
        for lane in &active.potentials[N..] {
            assert_eq!(lane, &vec![0.0; order + 1]);
        }
    }
}

#[test]
fn active_lanes_preserve_every_coefficient_and_parameter_path() {
    compare_full_lanes::<2>();
    compare_full_lanes::<5>();
    compare_full_lanes::<7>();
}

fn complete_fits<const N: usize>() {
    let base = std::array::from_fn(|i| 0.19 + 0.037 * i as f64);
    let argument = 0.23
        + (0..N)
            .map(|i| (0.17 + 0.031 * i as f64) * base[i])
            .sum::<f64>();
    let value = argument.exp();
    let mut failures = Vec::new();
    let mut fits = 0;
    for rank in 1..=4 {
        let factorial = (1..=rank).product::<usize>() as f64;
        for (node, vector) in symmetric_directions::directions(N, rank)
            .into_iter()
            .enumerate()
        {
            let tangent = std::array::from_fn(|i| if i < N { vector[i] } else { 0.0 });
            let input = XcTaylorInput::linear(base, tangent, rank).unwrap();
            let ordinary = input
                .evaluate_potentials_for::<N>(&[0.23], body::<_, N>)
                .unwrap();
            let direction = std::array::from_fn(|i| {
                let mut coefficients = vec![0.0; rank + 1];
                coefficients[0] = tangent[i];
                coefficients
            });
            let mixed = input
                .evaluate_directional_potentials_for::<N>(&direction, &[0.23], body::<_, N>)
                .unwrap();
            let slope = (0..N)
                .map(|i| (0.17 + 0.031 * i as f64) * tangent[i])
                .sum::<f64>();
            let expected = value * slope.powi(rank as i32);
            let mut analytic = vec![
                ordinary.energy[rank] * factorial,
                (0..N)
                    .map(|i| tangent[i] * ordinary.potentials[i][rank - 1])
                    .sum::<f64>()
                    * factorial
                    / rank as f64,
            ];
            if rank >= 2 {
                analytic.push(
                    (0..N)
                        .map(|i| tangent[i] * mixed.directional.potentials[i][rank - 2])
                        .sum::<f64>()
                        * factorial
                        / (rank * (rank - 1)) as f64,
                );
            }
            for &entry in &analytic {
                assert!((entry - expected).abs() < 1e-12);
            }
            let names = (0..analytic.len())
                .map(|route| format!("active XC N={N} rank={rank} node={node} route={route}"))
                .collect::<Vec<_>>();
            // Closed exponential derivatives supply the displaced oracle;
            // no AD/Taylor evaluation is used in these finite differences.
            let lower = |x: f64| value * slope.powi(rank as i32 - 1) * (slope * x).exp();
            stencils::fit_components_staged(
                &names,
                &[([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second)],
                |steps, stencil| {
                    analytic
                        .iter()
                        .map(|&entry| {
                            steps.map(|h| {
                                (stencil.first(
                                    lower(h),
                                    lower(-h),
                                    lower(2.0 * h),
                                    lower(-2.0 * h),
                                    h,
                                ) - entry)
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            let h = 0.01;
            let richardson = stencils::Stencil::Fourth.first(
                lower(h),
                lower(-h),
                lower(2.0 * h),
                lower(-2.0 * h),
                h,
            );
            assert!((richardson - expected).abs() < 1e-9);
            fits += analytic.len();
        }
    }
    components::assert_no_failures("complete active XC derivative tensors", &failures);
    println!("active XC N={N} complete first-through-fourth derivative fits={fits}");
}

#[test]
fn active_lda_complete_first_through_fourth_derivatives() {
    complete_fits::<2>();
}

#[test]
fn active_gga_complete_first_through_fourth_derivatives() {
    complete_fits::<5>();
}

#[test]
fn full_meta_complete_first_through_fourth_derivatives() {
    complete_fits::<7>();
}
