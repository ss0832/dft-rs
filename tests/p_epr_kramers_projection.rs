// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Projection mathematics, not physical acceptance of a spin-contaminated
//! GKS/GHF doublet or a substitute for its self-consistent reference.

use dft_rs::properties::epr::doublet as doublet_epr;

use dft_rs::linalg::Matrix;
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;
use doublet_epr::KramersPair;

fn sigma(axis: usize, n: usize) -> BlochMatrix {
    let mut operator = BlochMatrix::zeros(2 * n, 2 * n);
    for a in 0..n {
        match axis {
            0 => {
                operator.re[(a, n + a)] = 1.;
                operator.re[(n + a, a)] = 1.;
            }
            1 => {
                operator.im[(a, n + a)] = -1.;
                operator.im[(n + a, a)] = 1.;
            }
            _ => {
                operator.re[(a, a)] = 1.;
                operator.re[(n + a, n + a)] = -1.;
            }
        }
    }
    operator
}

#[test]
fn a_rotated_one_electron_pair_preserves_all_nine_gram_entries() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(rotated_pair)
        .unwrap();
}

fn rotated_pair() {
    let mut occupied = BlochMatrix::zeros(2, 1);
    occupied.re[(0, 0)] = 0.8;
    occupied.re[(1, 0)] = 0.36;
    occupied.im[(1, 0)] = 0.48;
    let pair = KramersPair::new(&occupied, &Matrix::identity(1), 1e-12).unwrap();
    assert!(pair.occupied_metric_defect < 1e-12);
    assert!(pair.determinant_overlap.abs() < 1e-12);
    let g: [[f64; 3]; 3] = std::array::from_fn(|a| {
        pair.pauli_coefficients(&sigma(a, 1), 1e-12)
            .unwrap()
            .map(|x| 2. * x)
    });
    for a in 0..3 {
        for b in 0..3 {
            let gram = (0..3).map(|c| g[a][c] * g[b][c]).sum::<f64>();
            assert!(
                (gram - if a == b { 4. } else { 0. }).abs() < 1e-12,
                "Gram({a},{b})"
            );
        }
    }
    println!("Single-electron Kramers projection g={g:?}");
}

fn value(matrix: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(matrix.re[(i, j)], matrix.im[(i, j)])
}
fn determinant3(matrix: &BlochMatrix, rows: [usize; 3]) -> Complex {
    let [a, b, c] = rows;
    value(matrix, a, 0)
        * (value(matrix, b, 1) * value(matrix, c, 2) - value(matrix, b, 2) * value(matrix, c, 1))
        - value(matrix, a, 1)
            * (value(matrix, b, 0) * value(matrix, c, 2)
                - value(matrix, b, 2) * value(matrix, c, 0))
        + value(matrix, a, 2)
            * (value(matrix, b, 0) * value(matrix, c, 1)
                - value(matrix, b, 1) * value(matrix, c, 0))
}

#[test]
fn three_electron_singular_overlap_transitions_match_fock_space_all_entries() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(three_electron_projection)
        .unwrap();
}

fn three_electron_projection() {
    let mut all = BlochMatrix::zeros(6, 6);
    for i in 0..6 {
        all.re[(i, i)] = 1.;
    }
    for (a, b, angle, imaginary) in [
        (0, 4, 0.31, true),
        (1, 3, -0.27, false),
        (2, 5, 0.49, true),
        (0, 2, 0.23, false),
        (3, 4, 0.19, true),
    ] {
        let c: f64 = f64::cos(angle);
        let s = f64::sin(angle);
        for row in 0..6 {
            let x = value(&all, row, a);
            let y = value(&all, row, b);
            let (u, v) = if imaginary {
                (
                    x * c + y * Complex::new(0., s),
                    x * Complex::new(0., s) + y * c,
                )
            } else {
                (x * c - y * s, x * s + y * c)
            };
            all.re[(row, a)] = u.re;
            all.im[(row, a)] = u.im;
            all.re[(row, b)] = v.re;
            all.im[(row, b)] = v.im;
        }
    }
    let mut occupied = BlochMatrix::zeros(6, 3);
    let mut partner = occupied.clone();
    for a in 0..6 {
        for i in 0..3 {
            occupied.re[(a, i)] = all.re[(a, i)];
            occupied.im[(a, i)] = all.im[(a, i)];
        }
    }
    for a in 0..3 {
        for i in 0..3 {
            let u = -value(&occupied, 3 + a, i).conj();
            let v = value(&occupied, a, i).conj();
            partner.re[(a, i)] = u.re;
            partner.im[(a, i)] = u.im;
            partner.re[(3 + a, i)] = v.re;
            partner.im[(3 + a, i)] = v.im;
        }
    }
    let pair = KramersPair::new(&occupied, &Matrix::identity(3), 1e-12).unwrap();
    let states = (0u32..64)
        .filter(|mask| mask.count_ones() == 3)
        .collect::<Vec<_>>();
    let amplitudes = |columns: &BlochMatrix| {
        states
            .iter()
            .map(|&mask| {
                let rows = (0..6).filter(|&i| mask & (1 << i) != 0).collect::<Vec<_>>();
                determinant3(columns, rows.try_into().unwrap())
            })
            .collect::<Vec<_>>()
    };
    let bra = amplitudes(&occupied);
    let ket = amplitudes(&partner);
    for axis in 0..3 {
        let mut operator = sigma(axis, 3);
        // Add a real-antisymmetric orbital current: it is time odd and
        // tests spatial as well as transverse complex spin contractions.
        for spin in 0..2 {
            operator.im[(3 * spin, 3 * spin + 1)] = 0.17 * (axis + 1) as f64;
            operator.im[(3 * spin + 1, 3 * spin)] = -0.17 * (axis + 1) as f64;
        }
        let expectation = |left: &[Complex], right: &[Complex]| {
            let mut total = Complex::ZERO;
            for (k, &mask) in states.iter().enumerate() {
                for j in 0..6 {
                    if mask & (1 << j) == 0 {
                        continue;
                    }
                    let after = mask ^ (1 << j);
                    let sign_j = if (mask & ((1 << j) - 1)).count_ones() % 2 == 0 {
                        1.
                    } else {
                        -1.
                    };
                    for i in 0..6 {
                        if after & (1 << i) != 0 {
                            continue;
                        }
                        let sign_i = if (after & ((1 << i) - 1)).count_ones() % 2 == 0 {
                            1.
                        } else {
                            -1.
                        };
                        let row = states
                            .iter()
                            .position(|&m| m == (after | (1 << i)))
                            .unwrap();
                        total += left[row].conj()
                            * right[k]
                            * value(&operator, i, j)
                            * (sign_i * sign_j);
                    }
                }
            }
            total
        };
        let off = expectation(&bra, &ket);
        let upper = expectation(&bra, &bra);
        let lower = expectation(&ket, &ket);
        let expected = [off.re, -off.im, 0.5 * (upper.re - lower.re)];
        let actual = pair.pauli_coefficients(&operator, 1e-12).unwrap();
        for component in 0..3 {
            assert!(
                (actual[component] - expected[component]).abs() < 1e-12,
                "Fock-space projection mismatch ({axis},{component})"
            );
        }
    }
}
