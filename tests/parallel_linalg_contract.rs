// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Borrowed row-major faer views and outer Rayon scheduling against scalar
//! matrix-product oracles. These are algebra/storage tests, not timings or
//! physical derivative acceptance. Binary-rational inputs additionally make
//! exact equality independent of summation order.

use dft_rs::linalg::{ensure_parallelism, Matrix};
use rayon::prelude::*;

fn matrix(rows: usize, cols: usize, seed: usize, dyadic: bool) -> Matrix {
    Matrix::from_row_major(
        rows,
        cols,
        (0..rows * cols)
            .map(|i| {
                if dyadic {
                    (((i * 17 + seed * 13) % 37) as f64 - 18.0) / 16.0
                } else {
                    ((i * 7 + seed * 11) as f64 * 0.17).sin() / 3.0
                }
            })
            .collect(),
    )
}

fn scalar_product(a: &Matrix, ta: bool, b: &Matrix, tb: bool) -> Matrix {
    let (rows, inner) = if ta {
        (a.cols, a.rows)
    } else {
        (a.rows, a.cols)
    };
    let (other_inner, cols) = if tb {
        (b.cols, b.rows)
    } else {
        (b.rows, b.cols)
    };
    assert_eq!(inner, other_inner);
    let mut out = Matrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..inner {
                let left = if ta { a[(k, i)] } else { a[(i, k)] };
                let right = if tb { b[(j, k)] } else { b[(k, j)] };
                sum += left * right;
            }
            out[(i, j)] = sum;
        }
    }
    out
}

fn check(actual: &Matrix, expected: &Matrix, exact: bool) {
    assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
    let mut worst = 0.0_f64;
    for i in 0..actual.rows {
        for j in 0..actual.cols {
            let got = actual[(i, j)];
            let want = expected[(i, j)];
            assert!(got.is_finite() && want.is_finite());
            worst = worst.max((got - want).abs());
            if exact {
                assert_eq!(got, want, "entry ({i},{j})");
            } else {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "entry ({i},{j}): {got} vs {want}"
                );
            }
        }
    }
    println!(
        "shape={}x{} exact={exact} max_abs_error={worst:.16e}",
        actual.rows, actual.cols
    );
}

#[test]
fn rectangular_and_transposed_views_match_scalar_products() {
    ensure_parallelism();
    for (m, k, n) in [
        (0, 5, 3),
        (3, 0, 4),
        (3, 5, 0),
        (1, 1, 1),
        (7, 3, 11),
        (17, 23, 31),
        (129, 193, 97),
    ] {
        for exact in [true, false] {
            let a = matrix(m, k, 1, exact);
            let b = matrix(k, n, 2, exact);
            let expected = scalar_product(&a, false, &b, false);
            check(&a.matmul(&b), &expected, exact);
            check(&a.matmul_seq(&b), &expected, exact);

            let at = matrix(k, m, 3, exact);
            let expected = scalar_product(&at, true, &b, false);
            check(&at.transpose_matmul(&b), &expected, exact);
            check(&at.transpose_matmul_seq(&b), &expected, exact);

            let bt = matrix(n, k, 4, exact);
            let expected = scalar_product(&a, false, &bt, true);
            check(&a.matmul_transpose_seq(&bt), &expected, exact);
        }
    }
}

#[test]
fn occupied_gram_matches_explicit_density_contraction() {
    ensure_parallelism();
    for (rows, cols) in [(0, 7), (7, 0), (1, 3), (7, 11), (37, 23)] {
        let c = matrix(rows, cols, 5, true);
        for occupied in [0, cols / 2, cols] {
            for scale in [0.0, 2.0, -1.5] {
                let mut expected = Matrix::zeros(rows, rows);
                for i in 0..rows {
                    for j in 0..rows {
                        for k in 0..occupied {
                            expected[(i, j)] += scale * c[(i, k)] * c[(j, k)];
                        }
                    }
                }
                check(&c.leading_columns_gram(occupied, scale), &expected, true);
            }
        }
    }
}

#[test]
fn outer_rayon_tasks_keep_sequential_faer_results() {
    let tasks: Vec<_> = (0..16)
        .map(|seed| (matrix(19, 31, seed, true), matrix(31, 23, seed + 1, true)))
        .collect();
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let actual: Vec<_> =
            pool.install(|| tasks.par_iter().map(|(a, b)| a.matmul_seq(b)).collect());
        for ((a, b), got) in tasks.iter().zip(actual) {
            check(&got, &scalar_product(a, false, b, false), true);
        }
    }
}
