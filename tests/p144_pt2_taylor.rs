// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! High-order pair response against independent dense noncanonical solves.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::linalg::{solve_linear, Matrix};
use dft_rs::properties::pt2::Pt2Problem;
use dft_rs::properties::pt2_response::{linear_response, taylor_response};
use dft_rs::xc::double_hybrid::Pt2Coefficients;

#[test]
fn noncanonical_pt2_energy_orders_one_through_four_and_amplitudes_have_four_step_fits() {
    let problem = Pt2Problem {
        energies: [vec![-0.8, -0.4, 0.2, 0.6], vec![-0.7, -0.4, 0.3, 0.6]],
        occupied: [2, 2],
        ovov: std::array::from_fn(|s| {
            (0..16)
                .map(|p| {
                    let (ia, jb) = (p / 4, p % 4);
                    0.08 * (((ia + 1) * (jb + 1) + s) as f64).cos()
                })
                .collect()
        }),
    };
    let scales = [1.0, 0.2, 0.1, 0.03, 0.01, 0.003, 0.001];
    let fock: Vec<[Matrix; 2]> = (1..=6)
        .map(|n| {
            std::array::from_fn(|s| {
                let mut f = Matrix::zeros(4, 4);
                for i in 0..4 {
                    for j in 0..4 {
                        if (i < 2) == (j < 2) {
                            f[(i, j)] = 0.08 / n as f64 * ((i + j + 2 + s + n) as f64).cos();
                        }
                    }
                }
                f
            })
        })
        .collect();
    let integrals: Vec<[Vec<f64>; 3]> = scales[1..=4]
        .iter()
        .map(|scale| {
            problem
                .ovov
                .each_ref()
                .map(|v| v.iter().map(|g| scale * g).collect())
        })
        .collect();
    let coefficients = Pt2Coefficients {
        same_spin: 0.31,
        opposite_spin: 0.47,
    };
    let response = taylor_response(&problem, &integrals, &fock[..4], coefficients).unwrap();
    let linear = linear_response(&problem, &integrals[0], &fock[0], coefficients).unwrap();
    assert_eq!(linear.correlation, response.correlation[1]);
    let entries: Vec<_> = (0..2)
        .flat_map(|i| {
            (0..2).flat_map(move |a| (0..2).flat_map(move |j| (0..2).map(move |b| [i, a, j, b])))
        })
        .collect();
    let dense = |theta: f64| {
        let mut energy = [0.0; 4];
        let mut derivative = [0.0; 4];
        let mut amplitudes = Vec::new();
        for (ch, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let mut a = Matrix::zeros(16, 16);
            let mut da = Matrix::zeros(16, 16);
            let mut g = vec![0.0; 16];
            let mut dg = [0.0; 16];
            let polynomial: f64 = scales
                .iter()
                .enumerate()
                .map(|(n, c)| c * theta.powi(n as i32))
                .sum();
            let gradient: f64 = scales
                .iter()
                .enumerate()
                .skip(1)
                .map(|(n, c)| n as f64 * c * theta.powi(n as i32 - 1))
                .sum();
            let value = |spin: usize, i: usize, j: usize, derivative: bool| {
                let base = if !derivative && i == j {
                    problem.energies[spin][i]
                } else {
                    0.0
                };
                base + fock
                    .iter()
                    .enumerate()
                    .map(|(k, f)| {
                        let n = k + 1;
                        f[spin][(i, j)]
                            * if derivative {
                                n as f64 * theta.powi(n as i32 - 1)
                            } else {
                                theta.powi(n as i32)
                            }
                    })
                    .sum::<f64>()
            };
            for (p, &[i, av, j, b]) in entries.iter().enumerate() {
                let exchanged = ((i * 2 + b) * 2 + j) * 2 + av;
                let raw = problem.ovov[ch][p]
                    - if s == t {
                        problem.ovov[ch][exchanged]
                    } else {
                        0.0
                    };
                g[p] = raw * polynomial;
                dg[p] = raw * gradient;
                for (q, &[k, c, l, d]) in entries.iter().enumerate() {
                    for (matrix, differentiated) in [(&mut a, false), (&mut da, true)] {
                        if av == c && j == l && b == d {
                            matrix[(p, q)] += value(s, i, k, differentiated);
                        }
                        if i == k && av == c && b == d {
                            matrix[(p, q)] += value(t, j, l, differentiated);
                        }
                        if i == k && j == l && b == d {
                            matrix[(p, q)] -= value(s, 2 + av, 2 + c, differentiated);
                        }
                        if i == k && av == c && j == l {
                            matrix[(p, q)] -= value(t, 2 + b, 2 + d, differentiated);
                        }
                    }
                }
            }
            let solution = solve_linear(&a, &g).unwrap();
            let weight = if s == t { 0.25 } else { 1.0 };
            for p in 0..16 {
                energy[ch] += weight * g[p] * solution[p];
                derivative[ch] += weight * 2.0 * dg[p] * solution[p];
                for q in 0..16 {
                    derivative[ch] -= weight * solution[p] * da[(p, q)] * solution[q];
                }
            }
            amplitudes.extend(solution);
        }
        energy[3] = coefficients.same_spin * (energy[0] + energy[2])
            + coefficients.opposite_spin * energy[1];
        derivative[3] = coefficients.same_spin * (derivative[0] + derivative[2])
            + coefficients.opposite_spin * derivative[1];
        (energy, derivative, amplitudes)
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    let centre = dense(0.0);
    let channels = [
        &response.same_spin_alpha,
        &response.opposite_spin,
        &response.same_spin_beta,
        &response.correlation,
    ];
    for order in 1..=4 {
        let factorial = [1.0, 1.0, 2.0, 6.0, 24.0][order];
        for (ch, values) in channels.iter().enumerate() {
            let exact = factorial * values[order];
            assert!(exact.abs() > 1e-8);
            let errors: Vec<_> = steps
                .iter()
                .map(|&h| {
                    let plus = dense(h);
                    let minus = dense(-h);
                    let estimate = match order {
                        1 => (plus.0[ch] - minus.0[ch]) / (2.0 * h),
                        2 => (plus.1[ch] - minus.1[ch]) / (2.0 * h),
                        3 => (plus.1[ch] - 2.0 * centre.1[ch] + minus.1[ch]) / (h * h),
                        _ => {
                            (dense(2.0 * h).1[ch] - 2.0 * plus.1[ch] + 2.0 * minus.1[ch]
                                - dense(-2.0 * h).1[ch])
                                / (2.0 * h * h * h)
                        }
                    };
                    (estimate - exact).abs()
                })
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            println!("PT2 channel={ch} order={order} derivative={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12}", fit.exponent, fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
        }
    }
    let errors: Vec<_> = steps
        .iter()
        .map(|&h| {
            dense(h)
                .2
                .iter()
                .zip(&centre.2)
                .zip(dense(-h).2)
                .zip(response.amplitude_coefficients[2].iter().flatten())
                .map(|(((p, c), m), exact)| ((p - 2.0 * c + m) / (h * h) - 2.0 * exact).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "PT2 second amplitude steps={steps:?} errors={errors:?} p={:.9} R2={:.12} residuals={:?}",
        fit.exponent, fit.r_squared, response.residuals
    );
    assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
    assert!(response.residuals.iter().all(|r| *r < 1e-12));
    let zero = taylor_response(
        &problem,
        &integrals,
        &fock[..4],
        Pt2Coefficients {
            same_spin: 0.0,
            opposite_spin: 0.0,
        },
    )
    .unwrap();
    assert!(zero.correlation.iter().all(|v| *v == 0.0));
}
