// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact finite isotropic nuclear-spin spectra: independent AB closed forms,
//! equivalent-spin and arbitrary-I limits, and an independent full product-space
//! Hamiltonian/detection oracle. No electronic derivative is introduced here;
//! the shielding/J response retains its separate complete-component acceptance.

use dft_rs::properties::nuclear_spin::{nuclear_spin_spectrum, SpinLine};
use dft_rs::{symmetric_eigen, Matrix};

fn broaden(lines: &[SpinLine], x: f64) -> f64 {
    lines
        .iter()
        .map(|line| line.weight / (1.0 + (x - line.offset_hz).powi(2)))
        .sum()
}

#[test]
fn every_ab_transition_matches_its_closed_form_and_signed_carrier_limit() {
    let (a, b) = (27.25f64, 73.8f64);
    for j in [-44.0f64, -15.0, 15.0, 44.0] {
        for sign in [-1.0, 1.0] {
            let report = nuclear_spin_spectrum(
                &[sign * 400e6; 2],
                &[sign * a, sign * b],
                &[1, 1],
                &[(0, 1, j)],
                &[vec![0, 1]],
                4,
            )
            .unwrap();
            let r = ((b - a).powi(2) + j * j).sqrt();
            let mut expected = Vec::new();
            for q in [-1.0, 1.0] {
                for s in [-1.0, 1.0] {
                    expected.push((
                        (a + b) * 0.5 + s * (r + q * j) * 0.5,
                        0.5 * (1.0 - q * j / r),
                    ));
                }
            }
            expected.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert_eq!(report.lines[0].len(), 4);
            let mut defect = 0.0f64;
            for (line, &(position, weight)) in report.lines[0].iter().zip(&expected) {
                defect = defect
                    .max((line.offset_hz - position).abs())
                    .max((line.weight - weight).abs());
            }
            println!("AB J={j} sign={sign} all four position/weight defect={defect:.12e} eigen residual={:.12e} sum rule={:.12e}", report.eigen_residual_hz,report.sum_rule_defects[0]);
            assert!(defect < 1e-11);
            assert!(report.eigen_residual_hz < 1e-11);
            assert!(report.sum_rule_defects[0].abs() < 1e-13);
        }
    }
}

#[test]
fn equivalent_spins_and_arbitrary_single_spin_obey_transverse_sum_rules() {
    let equivalent = nuclear_spin_spectrum(
        &[400e6; 2],
        &[43.7; 2],
        &[1, 1],
        &[(0, 1, 18.0)],
        &[vec![0, 1]],
        4,
    )
    .unwrap();
    for step in 0..121 {
        let x = 36.0 + 0.13 * step as f64;
        assert!(
            (broaden(&equivalent.lines[0], x) - 2.0 / (1.0 + (x - 43.7).powi(2))).abs() < 1e-12
        );
    }
    for twice in 1..=9 {
        let report =
            nuclear_spin_spectrum(&[-400e6], &[-81.3], &[twice], &[], &[vec![0]], twice + 1)
                .unwrap();
        assert!(report.sum_rule_defects[0].abs() < 1e-13);
        for line in &report.lines[0] {
            assert!((line.offset_hz - 81.3).abs() < 1e-12);
        }
    }
}

fn product(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.rows * b.rows, a.cols * b.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            for k in 0..b.rows {
                for l in 0..b.cols {
                    out[(i * b.rows + k, j * b.cols + l)] = a[(i, j)] * b[(k, l)];
                }
            }
        }
    }
    out
}

fn dense_operators(spins: &[usize], selected: usize) -> [Matrix; 3] {
    let mut result = std::array::from_fn(|_| {
        let mut unit = Matrix::zeros(1, 1);
        unit[(0, 0)] = 1.0;
        unit
    });
    for (i, &twice) in spins.iter().enumerate() {
        let size = twice + 1;
        let mut local = std::array::from_fn::<_, 3, _>(|_| Matrix::zeros(size, size));
        if i == selected {
            for m in 0..size {
                local[2][(m, m)] = m as f64 - 0.5 * twice as f64;
                if m < twice {
                    let amplitude = 0.5 * (((twice - m) * (m + 1)) as f64).sqrt();
                    local[0][(m + 1, m)] = amplitude;
                    local[0][(m, m + 1)] = amplitude;
                    // Iy = i times this real antisymmetric matrix.
                    local[1][(m + 1, m)] = -amplitude;
                    local[1][(m, m + 1)] = amplitude;
                }
            }
        } else {
            for matrix in &mut local {
                for m in 0..size {
                    matrix[(m, m)] = 1.0;
                }
            }
        }
        for axis in 0..3 {
            result[axis] = product(&result[axis], &local[axis]);
        }
    }
    result
}

#[test]
fn complete_mixed_spin_spectrum_matches_independent_dense_kronecker_oracle() {
    let carriers = [1000.0, 1000.0, -730.0];
    let offsets = [17.3, -9.7, 13.9];
    let spins = [1, 2, 1];
    let pairs = [(0, 1, 27.1), (0, 2, -14.3), (1, 2, 8.4)];
    let groups = [vec![0, 1], vec![2]];
    let solved = nuclear_spin_spectrum(&carriers, &offsets, &spins, &pairs, &groups, 12).unwrap();
    let operators: Vec<_> = (0..3).map(|i| dense_operators(&spins, i)).collect();
    let mut h = Matrix::zeros(12, 12);
    for (i, ops) in operators.iter().enumerate() {
        for r in 0..12 {
            h[(r, r)] -= (carriers[i] + offsets[i]) * ops[2][(r, r)];
        }
    }
    for &(i, j, coupling) in &pairs {
        for axis in 0..3 {
            let block = operators[i][axis].matmul(&operators[j][axis]);
            let factor = if axis == 1 { -coupling } else { coupling };
            for (v, b) in h.as_mut_slice().iter_mut().zip(block.as_slice()) {
                *v += factor * b;
            }
        }
    }
    let (energies, vectors) = symmetric_eigen(&h).unwrap();
    for (channel, group) in groups.iter().enumerate() {
        let mut detect = Matrix::zeros(12, 12);
        let mut norm = 0.0;
        for &i in group {
            let spin = 0.5 * spins[i] as f64;
            norm += spin * (spin + 1.0);
            for (v, x) in detect
                .as_mut_slice()
                .iter_mut()
                .zip(operators[i][0].as_slice())
            {
                *v += x;
            }
        }
        let detect = vectors.transpose_matmul(&detect.matmul(&vectors));
        let scale = 6.0 * group.len() as f64 / (12.0 * norm);
        let mut lines = Vec::new();
        for i in 0..12 {
            for j in i + 1..12 {
                lines.push(SpinLine {
                    offset_hz: energies[j] - energies[i] - carriers[group[0]].abs(),
                    weight: scale * detect[(i, j)].powi(2),
                });
            }
        }
        let mut defect = 0.0f64;
        for k in 0..301 {
            let x = -130.0 + k as f64;
            defect = defect.max((broaden(&solved.lines[channel], x) - broaden(&lines, x)).abs());
        }
        println!(
            "mixed-spin channel={channel} dense spectral defect={defect:.12e} residual={:.12e}",
            solved.eigen_residual_hz
        );
        assert!(defect < 1e-10);
        assert!(solved.sum_rule_defects[channel].abs() < 1e-12);
    }
}

#[test]
fn malformed_models_and_resource_overflow_are_rejected_before_allocation() {
    assert!(nuclear_spin_spectrum(&[1.0; 2], &[0.0; 2], &[1, 1], &[], &[vec![0, 1]], 3).is_err());
    assert!(
        nuclear_spin_spectrum(&[1.0], &[0.0], &[usize::MAX], &[], &[vec![0]], usize::MAX).is_err()
    );
    assert!(nuclear_spin_spectrum(
        &[1.0; 2],
        &[0.0; 2],
        &[1, 1],
        &[(0, 1, f64::NAN)],
        &[vec![0]],
        4
    )
    .is_err());
    assert!(nuclear_spin_spectrum(&[1.0, 2.0], &[0.0; 2], &[1, 1], &[], &[vec![0, 1]], 4).is_err());
}
