// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Unisolvent generic directions for every symmetric Cartesian component.
pub fn directions(rank: usize) -> Vec<[f64; 3]> {
    let mut result = Vec::new();
    for i in 0..=rank {
        for j in 0..=rank - i {
            let mut d = [1.0, 0.23 + 0.31 * i as f64, -0.47 + 0.29 * j as f64];
            let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            for v in &mut d {
                *v /= norm;
            }
            result.push(d);
        }
    }
    // Shifted triangular interpolation nodes are unisolvent for bivariate
    // polynomials of total degree rank. Check numerical independence too.
    let exponents: Vec<_> = (0..=rank)
        .flat_map(|a| (0..=rank - a).map(move |b| [a, b, rank - a - b]))
        .collect();
    assert_eq!(result.len(), exponents.len());
    let mut matrix: Vec<Vec<f64>> = result
        .iter()
        .map(|d| {
            exponents
                .iter()
                .map(|e| (0..3).map(|a| d[a].powi(e[a] as i32)).product())
                .collect()
        })
        .collect();
    let mut smallest = f64::INFINITY;
    for k in 0..matrix.len() {
        let pivot = (k..matrix.len())
            .max_by(|&a, &b| matrix[a][k].abs().total_cmp(&matrix[b][k].abs()))
            .unwrap();
        matrix.swap(k, pivot);
        let v = matrix[k][k];
        smallest = smallest.min(v.abs());
        assert!(
            v.abs() > 1e-12,
            "generic tensor directions lost independence"
        );
        for i in k + 1..matrix.len() {
            let scale = matrix[i][k] / v;
            for j in k..matrix.len() {
                matrix[i][j] -= scale * matrix[k][j];
            }
        }
    }
    println!(
        "generic symmetric rank={rank} directions={} minimum pivot {smallest:.12e}",
        result.len()
    );
    result
}
