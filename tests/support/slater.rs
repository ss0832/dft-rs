// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent Slater-overlap oracle shared by pair-response acceptance.
use super::{Complex, Pt2Coefficients};

// Bare Laplace determinants of the occupied-orbital cross overlap. This
// oracle neither differentiates an orbital nor applies a fermion operator.
fn determinant(a: &[Vec<Complex>]) -> Complex {
    if a.is_empty() {
        return Complex::ONE;
    }
    (0..a.len()).fold(Complex::ZERO, |sum, j| {
        let minor: Vec<Vec<_>> = a
            .iter()
            .skip(1)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(k, v)| (k != j).then_some(*v))
                    .collect()
            })
            .collect();
        sum + a[0][j] * determinant(&minor) * if j % 2 == 0 { 1.0 } else { -1.0 }
    })
}

pub fn finite_state(
    n: [usize; 2],
    o: [usize; 2],
    base: &[Vec<Complex>; 3],
    change: &[Vec<Complex>; 3],
    t: f64,
    channel: Option<usize>,
    weights: Pt2Coefficients,
) -> Vec<(Vec<usize>, Complex)> {
    let reference: Vec<_> = (0..o[0]).chain((0..o[1]).map(|i| n[0] + i)).collect();
    let mut result = if channel.is_none() {
        vec![(reference.clone(), Complex::ONE)]
    } else {
        Vec::new()
    };
    for (c, (s, u)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        if channel.is_some_and(|v| v != c) {
            continue;
        }
        let (ns, nu, vs, vu) = (o[s], o[u], n[s] - o[s], n[u] - o[u]);
        let scale = if channel.is_some() {
            1.0
        } else {
            if c == 1 {
                weights.opposite_spin
            } else {
                weights.same_spin
            }
            .sqrt()
        };
        for i in 0..ns {
            for a in 0..vs {
                for j in 0..nu {
                    for b in 0..vu {
                        if s == u && (i >= j || a >= b) {
                            continue;
                        }
                        let index = ((i * vs + a) * nu + j) * vu + b;
                        let mut occupation = reference.clone();
                        // Replacing the corresponding occupied Slater columns gives
                        // the T2 sign from a permutation, without an operator action.
                        occupation[if s == 0 { i } else { o[0] + i }] =
                            if s == 0 { ns + a } else { n[0] + ns + a };
                        occupation[if u == 0 { j } else { o[0] + j }] =
                            if u == 0 { nu + b } else { n[0] + nu + b };
                        let inversions = (0..occupation.len())
                            .map(|i| {
                                (i + 1..occupation.len())
                                    .filter(|&j| occupation[i] > occupation[j])
                                    .count()
                            })
                            .sum::<usize>();
                        occupation.sort_unstable();
                        let value = (base[c][index]
                            + change[c][index] * t
                            + base[c][index] * (0.09 * t * t))
                            * scale
                            * if inversions % 2 == 0 { 1.0 } else { -1.0 };
                        result.push((occupation, value));
                    }
                }
            }
        }
    }
    result
}

pub fn slater_overlap(
    left: &[(Vec<usize>, Complex)],
    right: &[(Vec<usize>, Complex)],
    cross: &[Vec<Complex>],
) -> Complex {
    left.iter()
        .filter(|(_, c)| *c != Complex::ZERO)
        .flat_map(|(l, cl)| {
            right
                .iter()
                .filter(|(_, c)| *c != Complex::ZERO)
                .map(move |(r, cr)| {
                    let occupied: Vec<Vec<_>> = l
                        .iter()
                        .map(|&i| r.iter().map(|&j| cross[i][j]).collect())
                        .collect();
                    cl.conj() * *cr * determinant(&occupied)
                })
        })
        .fold(Complex::ZERO, |a, b| a + b)
}
