// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete unisolvent generic directions in arbitrary dimension.
pub fn directions(dimension: usize, rank: usize) -> Vec<Vec<f64>> {
    assert!(dimension > 0 && rank > 0);
    fn compositions(left: usize, slots: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if slots == 1 {
            prefix.push(left);
            out.push(prefix.clone());
            prefix.pop();
            return;
        }
        for value in 0..=left {
            prefix.push(value);
            compositions(left - value, slots - 1, prefix, out);
            prefix.pop();
        }
    }
    let mut nodes = Vec::new();
    compositions(rank, dimension, &mut Vec::new(), &mut nodes);
    let expected = (1..=rank).fold(1usize, |count, k| count * (dimension + k - 1) / k);
    assert_eq!(nodes.len(), expected);
    // Degree-rank barycentric simplex nodes are unisolvent for homogeneous
    // polynomials. Apply M=D+u v^T to make every Cartesian component nonzero.
    // D is positive diagonal and u,v positive, so det(M)=det(D)*(1+v^T D^-1 u)>0.
    // Normalization rescales rows by nonzero factors and preserves independence.
    let result = nodes
        .iter()
        .map(|node| {
            let common: f64 = node
                .iter()
                .enumerate()
                .map(|(j, &n)| (1.0 + 0.03 * j as f64) * n as f64)
                .sum();
            let mut d: Vec<_> = node
                .iter()
                .enumerate()
                .map(|(i, &n)| {
                    (0.81 + 0.017 * i as f64) * n as f64 + 0.13 * (1.0 + 0.02 * i as f64) * common
                })
                .collect();
            let norm = d.iter().fold(0.0_f64, |v, &x| v.hypot(x));
            for x in &mut d {
                *x /= norm;
            }
            assert!(d.iter().all(|x| x.is_finite() && *x != 0.0));
            d
        })
        .collect();
    println!("unisolvent symmetric dimension={dimension} rank={rank} independent directions={expected}; invertible positive diagonal plus rank-one map");
    result
}
