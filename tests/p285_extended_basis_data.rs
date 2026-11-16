// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every nonzero exponent/coefficient of every element in the thirty new bases
//! is compared with the independently downloaded BSE JSON export. The runtime
//! parser reads the separate Gaussian94 export. No network is used by tests.

use dft_rs::basis::data::{bundled_names, load};
use dft_rs::driver::Value;

#[test]
fn every_added_angular_contraction_matches_the_independent_json_export() {
    let raw = include_str!("oracle_data/bse_extended_bases.json");
    let data = Value::parse_json(raw).unwrap();
    let bases = data.get("bases").unwrap().as_array().unwrap();
    assert_eq!(bases.len(), 30);
    assert_eq!(bundled_names().len(), 78);
    let mut elements = 0usize;
    let mut primitive_pairs = 0usize;
    for row in bases {
        let name = row.get("name").unwrap().as_str().unwrap();
        let basis = load(name).unwrap();
        let Value::Object(expected) = row.get("elements").unwrap() else {
            panic!("element map required")
        };
        assert_eq!(
            (1..=118).filter(|&z| basis.has_element(z)).count(),
            expected.len()
        );
        for (z, shells) in expected {
            let actual = basis.element(z.parse().unwrap()).unwrap();
            let mut available: Vec<_> = actual
                .shells
                .iter()
                .map(|s| {
                    let mut pairs: Vec<_> = s
                        .exps
                        .iter()
                        .copied()
                        .zip(s.coefs.iter().copied())
                        .filter(|(_, c)| *c != 0.0)
                        .collect();
                    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
                    (s.l, pairs)
                })
                .collect();
            for shell in shells.as_array().unwrap() {
                let values = shell.as_array().unwrap();
                let l = values[0].as_i64().unwrap() as u8;
                let exps = values[1].numeric_array().unwrap().1;
                let coefs = values[2].numeric_array().unwrap().1;
                let pairs: Vec<_> = exps.into_iter().zip(coefs).collect();
                let index = available
                    .iter()
                    .position(|entry| entry.0 == l && entry.1 == pairs)
                    .unwrap_or_else(|| {
                        panic!("{name} Z={z} l={l}: independent BSE primitive coefficients differ")
                    });
                primitive_pairs += pairs.len();
                available.swap_remove(index);
            }
            assert!(
                available.is_empty(),
                "{name} Z={z}: unaccounted contraction"
            );
            elements += 1;
        }
    }
    println!("thirty bases: {elements} elements and {primitive_pairs} nonzero primitive pairs match the independent JSON oracle exactly");
}
