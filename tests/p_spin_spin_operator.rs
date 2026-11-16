// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::dual::Scalar;
use dft_rs::integrals::analytic_basis::AnalyticShell;
use dft_rs::response::nuclear::spin_spin_operators::spin_spin_integrals;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

fn shell(center: [f64; 3], exponent: f64) -> AnalyticShell<f64> {
    AnalyticShell::normalized(0, center, vec![exponent], vec![1.0]).unwrap()
}

fn gaussian_pair_tensor(r: [f64; 3], a: f64, b: f64) -> [f64; 9] {
    let r2 = r.iter().map(|x| x * x).sum::<f64>();
    let distance = r2.sqrt();
    let u = (2.0 * a * b / (a + b)).sqrt() * distance;
    let overlap = (2.0 * (a * b).sqrt() / (a + b)).powf(1.5) * (-a * b / (a + b) * r2).exp();
    let radial = (Scalar::erf(u)
        - 2.0 * u / std::f64::consts::PI.sqrt() * (-u * u).exp() * (1.0 + 2.0 * u * u / 3.0))
        / distance.powi(3)
        / (1.0 - overlap * overlap);
    std::array::from_fn(|i| {
        radial * (if i / 3 == i % 3 { 1.0 } else { 0.0 } - 3.0 * r[i / 3] * r[i % 3] / r2)
    })
}

fn moving_pair_tensor<S: Scalar>(r: [S; 3], a: f64, b: f64, order: usize) -> [S; 9] {
    let c = S::from_f64;
    let shells = [
        AnalyticShell::normalized(0, [c(0.0); 3], vec![c(a)], vec![c(1.0)]).unwrap(),
        AnalyticShell::normalized(0, r, vec![c(b)], vec![c(1.0)]).unwrap(),
    ];
    let r2 = r.iter().fold(c(0.0), |sum, &x| sum + x * x);
    let overlap = c((2.0 * (a * b).sqrt() / (a + b)).powf(1.5)) * (c(-a * b / (a + b)) * r2).exp();
    let denominator = c(1.0) - overlap * overlap;
    let p = [
        [c(1.0) / denominator, -overlap / denominator],
        [-overlap / denominator, c(1.0) / denominator],
    ];
    let mut result = [c(0.0); 9];
    for mu in 0..2 {
        for nu in 0..2 {
            for kappa in 0..2 {
                for lambda in 0..2 {
                    let block = spin_spin_integrals(
                        &shells[mu],
                        &shells[nu],
                        &shells[kappa],
                        &shells[lambda],
                        order,
                    )
                    .unwrap();
                    assert_eq!(block.shape, [1; 4]);
                    let weight =
                        c(0.5) * (p[mu][nu] * p[kappa][lambda] - p[mu][kappa] * p[nu][lambda]);
                    for (value, operator) in result.iter_mut().zip(block.values[0]) {
                        *value = *value + weight * operator;
                    }
                }
            }
        }
    }
    result
}

fn pair_tensor(r: [f64; 3], a: f64, b: f64) -> [f64; 9] {
    moving_pair_tensor(r, a, b, 0)
}

#[test]
fn antisymmetric_gaussian_pair_matches_closed_tensor() {
    // Also retain direct normalized-shell construction as an interface check.
    assert_eq!(shell([0.0; 3], 0.7).l, 0);
    for (r, a, b) in [([1.2, 0.7, -0.4], 0.7, 1.1), ([3.1, -1.3, 0.9], 1.3, 0.4)] {
        let actual = pair_tensor(r, a, b);
        let expected = gaussian_pair_tensor(r, a, b);
        for i in 0..9 {
            println!(
                "gaussian-pair component-{i} actual={} expected={}",
                actual[i], expected[i]
            );
            assert!(
                (actual[i] - expected[i]).abs() < 2e-13,
                "component {i}: {} versus {}",
                actual[i],
                expected[i]
            );
        }
        assert!((actual[0] + actual[4] + actual[8]).abs() < 2e-13);
        for i in 0..3 {
            for j in 0..3 {
                assert!((actual[3 * i + j] - actual[3 * j + i]).abs() < 2e-13);
            }
        }
    }
}

#[test]
fn spin_spin_operator_is_invariant_under_common_translation() {
    let positions = [
        [0.1, -0.2, 0.3],
        [1.2, 0.7, -0.4],
        [-0.3, 0.8, 1.1],
        [0.9, -1.2, 0.6],
    ];
    let shift = [2.3, -1.7, 0.9];
    let shells: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(i, &position)| shell(position, 0.6 + 0.2 * i as f64))
        .collect();
    let moved: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(i, position)| {
            shell(
                std::array::from_fn(|axis| position[axis] + shift[axis]),
                0.6 + 0.2 * i as f64,
            )
        })
        .collect();
    let original = spin_spin_integrals(&shells[0], &shells[1], &shells[2], &shells[3], 0).unwrap();
    let translated = spin_spin_integrals(&moved[0], &moved[1], &moved[2], &moved[3], 0).unwrap();
    for entry in 0..9 {
        assert!((original.values[0][entry] - translated.values[0][entry]).abs() < 2e-14);
    }
}

#[test]
fn moving_pair_gradient_and_hessian_have_complete_four_step_fits() {
    use dft_rs::dual::Taylor;
    let r = [1.2, 0.7, -0.4];
    let directions = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.8, 0.6, 0.0],
        [0.6, 0.0, 0.8],
        [0.0, 0.8, 0.6],
    ];
    let mut failures = Vec::new();
    let center = gaussian_pair_tensor(r, 0.7, 1.1);
    for (direction_index, direction) in directions.iter().enumerate() {
        let path: [Taylor<3>; 3] =
            std::array::from_fn(|axis| Taylor::from_coefficients(&[r[axis], direction[axis]]));
        let analytic = moving_pair_tensor(path, 0.7, 1.1, 2);
        for order in 1..=2 {
            if order == 1 && direction_index >= 3 {
                continue;
            }
            let names: Vec<_> = (0..9)
                .map(|entry| {
                    format!("spin-spin order-{order} direction-{direction_index} entry-{entry}")
                })
                .collect();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    let mut errors = vec![[0.0; 4]; 9];
                    for (step_index, &h) in steps.iter().enumerate() {
                        let displaced = |scale| {
                            gaussian_pair_tensor(
                                std::array::from_fn(|axis| r[axis] + scale * h * direction[axis]),
                                0.7,
                                1.1,
                            )
                        };
                        let plus = displaced(1.0);
                        let minus = displaced(-1.0);
                        let plus2 = displaced(2.0);
                        let minus2 = displaced(-2.0);
                        for entry in 0..9 {
                            let measured = if order == 1 {
                                stencil.first(
                                    plus[entry],
                                    minus[entry],
                                    plus2[entry],
                                    minus2[entry],
                                    h,
                                )
                            } else if stencil == stencils::Stencil::Second {
                                (plus[entry] - 2.0 * center[entry] + minus[entry]) / (h * h)
                            } else {
                                (-plus2[entry] + 16.0 * plus[entry] - 30.0 * center[entry]
                                    + 16.0 * minus[entry]
                                    - minus2[entry])
                                    / (12.0 * h * h)
                            };
                            let expected =
                                analytic[entry].c[order] * if order == 2 { 2.0 } else { 1.0 };
                            errors[entry][step_index] = (measured - expected).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("complete Gaussian-pair operator derivatives", &failures);
}
