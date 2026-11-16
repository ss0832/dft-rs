// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::acceptance::components;

#[test]
fn gaussian_shift_matches_independent_cartesian_moments_and_four_step_fits() {
    let centres = [
        [0.17, -0.31, 0.29],
        [0.73, 0.21, -0.13],
        [-0.37, 0.83, 0.43],
        [0.31, -0.17, 1.19],
    ];
    let exponents = [0.63, 1.17, 0.89, 1.43];
    let kernels = [
        GaussianKernel::Coulomb,
        GaussianKernel::LongRange { omega: 0.7 },
        GaussianKernel::ShortRange { omega: 0.7 },
    ];
    let mut maximum = 0.0_f64;
    let mut failures = Vec::new();
    for (case, angular) in [
        [[0, 0, 0]; 4],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[2, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 0]],
    ]
    .into_iter()
    .enumerate()
    {
        let primitives: [_; 4] = std::array::from_fn(|i| Primitive {
            center: centres[i],
            exponent: exponents[i],
            angular: angular[i],
        });
        let p = primitives.each_ref();
        for (kind, &kernel) in kernels.iter().enumerate() {
            let bra = [0.23, -0.19, 0.31];
            let ket = [-0.17, 0.29, 0.13];
            let fast = repulsion_coefficients(p, kernel, bra, ket, 0, 4).unwrap();
            let exact =
                super::super::primitive_repulsion_coefficients(p, kernel, bra, ket, 0, 4, 0, |x| {
                    x == 0.0
                })
                .unwrap();
            for (a, b) in fast.iter().zip(exact) {
                maximum = maximum.max((a.0 - b.0).abs()).max((a.1 - b.1).abs());
                assert!((a.0 - b.0).abs().max((a.1 - b.1).abs()) < 2e-12);
            }
            // A one-axis Fourier phase makes the independently expanded
            // high-order Cartesian moment oracle economical. Neither
            // finite-field oracle nor fitting uses the shifted algorithm.
            let bra = [0.0, 0.0, 0.81];
            let ket = [0.0, 0.0, -0.63];
            let fast = repulsion_coefficients(p, kernel, bra, ket, 0, 4).unwrap();
            let moment = super::super::primitive_repulsion_coefficients(
                p,
                kernel,
                bra,
                ket,
                0,
                12,
                0,
                |x| x == 0.0,
            )
            .unwrap();
            let finite = |h: f64| {
                moment
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(a, b), (n, &(x, y))| {
                        (a + x * h.powi(n as i32), b + y * h.powi(n as i32))
                    })
            };
            let names: Vec<_> = (1..=4)
                .map(|n| format!("Gaussian phase case={case} kernel={kind} d{n}"))
                .collect();
            components::fit_components(
                &names,
                &[
                    [0.16, 0.12, 0.08, 0.06],
                    [0.08, 0.06, 0.04, 0.03],
                    [0.32, 0.24, 0.16, 0.12],
                ],
                |steps| {
                    let mut errors = vec![[0.0; 4]; 4];
                    for (k, &h) in steps.iter().enumerate() {
                        let a = finite(h);
                        let b = finite(2.0 * h);
                        let estimates = [
                            a.1 / h,
                            2.0 * (a.0 - fast[0].0) / (h * h),
                            (b.1 - 2.0 * a.1) / h.powi(3),
                            (2.0 * b.0 - 8.0 * a.0 + 6.0 * fast[0].0) / h.powi(4),
                        ];
                        for n in 1..=4 {
                            let factorial = (1..=n).product::<usize>() as f64;
                            let derivative = if n % 2 == 0 { fast[n].0 } else { fast[n].1 };
                            errors[n - 1][k] = (estimates[n - 1] - factorial * derivative).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    println!("shifted Gaussian versus spatial moments maximum {maximum:.12e}");
    components::assert_no_failures("Gaussian Fourier phase", &failures);
}

#[test]
fn one_electron_gaussian_shift_fits_every_independent_operator_coefficient() {
    let nuclei = [([0.31, -0.17, 0.63], 1.3), ([-0.41, 0.23, 0.71], 0.7)];
    let electric = [0.21, -0.31, 0.17];
    let magnetic = [0.7, -0.6, 0.0];
    let a = Primitive {
        angular: [1, 0, 0],
        exponent: 0.63,
        center: [0.17, -0.31, 0.29],
    };
    let b = Primitive {
        angular: [0, 1, 1],
        exponent: 1.17,
        center: [0.73, 0.21, 0.29],
    };
    let separation = std::array::from_fn(|i| a.center[i] - b.center[i]);
    let k = super::super::half_cross(magnetic, separation);
    let exact = pair_coefficients(&a, &b, k, &nuclei, electric, magnetic, b.center, 4)
        .unwrap()
        .unwrap();
    let finite = |h: f64| {
        super::super::primitive_field_pair(
            &a,
            &b,
            &nuclei,
            &super::super::ScalarFieldPath {
                electric: electric.map(|v| h * v),
                magnetic: magnetic.map(|v| h * v),
                origin: [0.0; 3],
                london: true,
                phase_order: 12,
                derivative_order: 0,
            },
            |x| x == 0.0,
        )
        .unwrap()
    };
    let zero = finite(0.0);
    for q in 0..5 {
        assert!((zero[q].0 - exact[q][0].0).abs() < 1e-13);
    }
    let mut locations = Vec::new();
    let mut names = Vec::new();
    for n in 1..=4 {
        for q in 0..5 {
            for part in 0..2 {
                if q != 1 && part != n % 2 {
                    assert_eq!(
                        if part == 0 {
                            exact[q][n].0
                        } else {
                            exact[q][n].1
                        },
                        0.0
                    );
                    continue;
                }
                locations.push((n, q, part));
                names.push(format!(
                    "one-electron shifted Gaussian d{n} operator={q} part={part}"
                ));
            }
        }
    }
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[
            [0.24, 0.18, 0.12, 0.09],
            [0.12, 0.09, 0.06, 0.045],
            [0.4, 0.3, 0.2, 0.15],
        ],
        |steps| {
            let mut errors = vec![[0.0; 4]; locations.len()];
            for (t, &h) in steps.iter().enumerate() {
                let samples = [finite(h), finite(-h), finite(2.0 * h), finite(-2.0 * h)];
                for (index, &(n, q, part)) in locations.iter().enumerate() {
                    let get = |x: (f64, f64)| if part == 0 { x.0 } else { x.1 };
                    let [p, m, p2, m2] = samples.map(|s| get(s[q]));
                    let z = get(zero[q]);
                    let fd = match n {
                        1 => (p - m) / (2.0 * h),
                        2 => (p + m - 2.0 * z) / (h * h),
                        3 => (p2 - m2 - 2.0 * p + 2.0 * m) / (2.0 * h.powi(3)),
                        _ => (p2 + m2 - 4.0 * p - 4.0 * m + 6.0 * z) / h.powi(4),
                    };
                    errors[index][t] =
                        (fd - get(exact[q][n]) * (1..=n).product::<usize>() as f64).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("shifted one-electron coefficients", &failures);
}
