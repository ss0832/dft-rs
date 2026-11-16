// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Sufficient geometric C1 certificate for nonsymmetric derivative fixtures.
//! Distinct element-labelled distance fingerprints force every atom to be
//! fixed, and noncoplanarity then excludes any remaining improper symmetry.

pub fn assert_c1(numbers: &[u8], positions: &[[f64; 3]]) {
    assert_eq!(numbers.len(), positions.len());
    assert!(
        positions.iter().flatten().all(|v| v.is_finite()),
        "C1 certificate requires finite coordinates"
    );
    assert!(
        numbers.len() >= 4,
        "a molecular C1 certificate needs four noncoplanar centres"
    );
    let distance = |i: usize, j: usize| {
        (0..3)
            .map(|a| (positions[i][a] - positions[j][a]).powi(2))
            .sum::<f64>()
    };
    let mut signatures = Vec::new();
    let mut scale = 0.0_f64;
    for i in 0..numbers.len() {
        let mut signature = (0..numbers.len())
            .map(|j| (numbers[j], distance(i, j)))
            .collect::<Vec<_>>();
        signature.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        for &(_, d) in &signature {
            scale = scale.max(d);
        }
        signatures.push(signature);
    }
    assert!(scale.is_finite() && scale > 0.0);
    let mut separation = f64::INFINITY;
    for i in 0..numbers.len() {
        for j in i + 1..numbers.len() {
            if numbers[i] != numbers[j] {
                continue;
            }
            let defect = signatures[i]
                .iter()
                .zip(&signatures[j])
                .map(|(a, b)| if a.0 == b.0 { (a.1 - b.1).abs() } else { scale })
                .fold(0.0_f64, f64::max);
            separation = separation.min(defect);
            assert!(
                defect > 64.0 * f64::EPSILON * scale,
                "distance fingerprints do not certify that atoms {i},{j} are fixed"
            );
        }
    }
    let mut volume = 0.0_f64;
    for i in 1..numbers.len() {
        for j in i + 1..numbers.len() {
            for k in j + 1..numbers.len() {
                let a: [f64; 3] = std::array::from_fn(|s| positions[i][s] - positions[0][s]);
                let b: [f64; 3] = std::array::from_fn(|s| positions[j][s] - positions[0][s]);
                let c: [f64; 3] = std::array::from_fn(|s| positions[k][s] - positions[0][s]);
                let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
                volume = volume.max(det.abs());
            }
        }
    }
    assert!(
        volume > 64.0 * f64::EPSILON * scale.powf(1.5),
        "coplanar centres do not certify C1"
    );
    println!(
        "C1 certificate: centres={} fingerprint_separation={separation:.12e} Bohr^2 tetrahedron_determinant={volume:.12e} Bohr^3",
        numbers.len()
    );
}
