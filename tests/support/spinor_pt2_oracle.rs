// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent stationary spin-orbital PT2 sum over unique substitutions.
use super::dft_rs::{
    integrals::{eri::EriTensor, london::ComplexMatrix as Cm},
    response::complex::Complex as C,
};

fn z(m: &Cm, i: usize, j: usize) -> C {
    C::new(m.re[(i, j)], m.im[(i, j)])
}

pub fn stationary_pt2(c: &Cm, e: &[f64], occupied: usize, eri: &EriTensor) -> f64 {
    let n = c.re.rows / 2;
    let density = |i, a| {
        (0..n * n)
            .map(|p| {
                let mu = p / n;
                let nu = p % n;
                (0..2).fold(C::ZERO, |v, s| {
                    v + z(c, s * n + mu, i) * z(c, s * n + nu, a).conj()
                })
            })
            .collect::<Vec<_>>()
    };
    let g = |i, a, j, b| {
        let left = density(i, a);
        let right = density(j, b);
        let mut out = C::ZERO;
        for mu in 0..n {
            for nu in 0..n {
                for la in 0..n {
                    for ka in 0..n {
                        out += left[mu * n + nu] * right[la * n + ka] * eri.get(mu, nu, la, ka);
                    }
                }
            }
        }
        out
    };
    let mut correlation = 0.0;
    for i in 0..occupied {
        for j in i + 1..occupied {
            for a in occupied..e.len() {
                for b in a + 1..e.len() {
                    correlation +=
                        (g(i, a, j, b) - g(i, b, j, a)).abs().powi(2) / (e[i] + e[j] - e[a] - e[b]);
                }
            }
        }
    }
    correlation
}
