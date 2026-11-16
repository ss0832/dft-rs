// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Total spin diagnostics for molecular complex spinor references.

use std::f64::consts::TAU;

use crate::dfpt::orbitals::{adjoint, matmul};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::periodic::spinor_scf::spin_diagonal;
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

pub struct SpinDiagnostics {
    pub expectation: [f64; 3],
    pub squared: f64,
    pub sectors: Option<SpinSectorWeights>,
}

pub struct SpinSectorWeights {
    /// The pair contains `2S` and the probability of the entire spin sector.
    pub weights: Vec<(usize, f64)>,
    pub normalization_defect: f64,
    pub squared_moment_defect: f64,
}

/// Compute the Wick spin moment of a determinant or independent-occupation
/// ensemble. Resolve total-spin sectors when the occupations are integral.
pub fn molecular_spin_diagnostics(
    coefficients: &BlochMatrix,
    occupations: &[f64],
    overlap: &Matrix,
) -> Result<SpinDiagnostics> {
    if overlap.rows != overlap.cols
        || coefficients.rows() != 2 * overlap.rows
        || coefficients.cols() != occupations.len()
        || occupations.iter().any(|f| !(0.0..=1.0).contains(f))
    {
        return Err(DftError::InvalidInput(
            "spin diagnostics need spin-major orbitals, the spatial overlap and valid occupations"
                .into(),
        ));
    }
    let selected: Vec<_> = occupations
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 0.0)
        .collect();
    let mut occupied = BlochMatrix::zeros(coefficients.rows(), selected.len());
    for (target, &(source, _)) in selected.iter().enumerate() {
        for row in 0..occupied.rows() {
            occupied.re[(row, target)] = coefficients.re[(row, source)];
            occupied.im[(row, target)] = coefficients.im[(row, source)];
        }
    }
    let adj = adjoint(&occupied)?;
    let spin_overlap = spin_diagonal(&BlochMatrix::from_pair((
        overlap.clone(),
        Matrix::zeros(overlap.rows, overlap.cols),
    )));
    let metric = matmul(&adj, &matmul(&spin_overlap, &occupied)?)?;
    let n_active = selected.len();
    if metric.re.max_abs_diff(&Matrix::identity(n_active)) > 1e-9
        || metric.im.as_slice().iter().any(|v| v.abs() > 1e-9)
    {
        return Err(DftError::InvalidInput(
            "spin diagnostics require occupied orbitals orthonormal in the AO overlap".into(),
        ));
    }
    let spin = crate::response::spin_spin::total_spin_constraints(overlap)?
        .iter()
        .map(|operator| matmul(&adj, &matmul(operator, &occupied)?))
        .collect::<Result<Vec<_>>>()?;
    let mut expectation = [0.0; 3];
    let mut exchange = 0.0;
    for (a, operator) in spin.iter().enumerate() {
        for (i, &(_, fi)) in selected.iter().enumerate() {
            expectation[a] += fi * operator.re[(i, i)];
            for (j, &(_, fj)) in selected.iter().enumerate() {
                let re = operator.re[(i, j)];
                let im = operator.im[(i, j)];
                exchange += fi * fj * (re * re + im * im);
            }
        }
    }
    let n_electrons: f64 = occupations.iter().sum();
    let squared = expectation.iter().map(|s| s * s).sum::<f64>() + 0.75 * n_electrons - exchange;
    let sectors = if selected.iter().all(|&(_, &f)| f == 1.0) {
        Some(project_sectors(&spin, selected.len(), squared)?)
    } else {
        None
    };
    Ok(SpinDiagnostics {
        expectation,
        squared,
        sectors,
    })
}

fn project_sectors(spin: &[BlochMatrix], n: usize, squared: f64) -> Result<SpinSectorWeights> {
    let identity = BlochMatrix::from_pair((Matrix::identity(n), Matrix::zeros(n, n)));
    let (nodes, weights) = crate::grid::gauss_legendre(n / 2 + 1);
    let azimuths: Vec<_> = (0..=n)
        .map(|k| (TAU * k as f64 / (n + 1) as f64).sin_cos())
        .collect();
    let mut average = vec![0.0; n + 1];
    let mut counts = vec![0.0; n + 1];
    let tolerance = 1e-9;
    for (z, w) in nodes.into_iter().zip(weights) {
        let radial = (1.0 - z * z).sqrt();
        for &(sin, cos) in &azimuths {
            let mut up = identity.clone();
            up.scale(0.5);
            for (operator, axis) in spin.iter().zip([radial * cos, radial * sin, z]) {
                up.add_scaled(operator, axis);
            }
            up.make_hermitian();
            // Diagonalizing the occupied compression of the spin-up
            // projector gives independent Bernoulli modes for this axis.
            let (eigenvalues, _) = symmetric_eigen(&up.real_embedding())?;
            counts.fill(0.0);
            counts[0] = 1.0;
            for (i, pair) in eigenvalues.chunks_exact(2).enumerate() {
                let raw = 0.5 * (pair[0] + pair[1]);
                if !(-tolerance..=1.0 + tolerance).contains(&raw) {
                    return Err(DftError::InvalidInput(
                        "occupied spin projection eigenvalue lies outside [0, 1]".into(),
                    ));
                }
                let p = raw.clamp(0.0, 1.0);
                for k in (1..=i + 1).rev() {
                    counts[k] = (1.0 - p) * counts[k] + p * counts[k - 1];
                }
                counts[0] *= 1.0 - p;
            }
            let weight = w / (2.0 * (n + 1) as f64);
            for (total, count) in average.iter_mut().zip(&counts) {
                *total += weight * count;
            }
        }
    }
    // Each axis probability is a polynomial of degree at most N. Fourier
    // terms up to degree N average out over N+1 azimuths. The surviving polar
    // polynomial is integrated exactly by the chosen Gauss–Legendre rule.
    // The rotational average has q_M = sum_(S>=|M|) p_S/(2S+1).
    let weights: Vec<_> = (n % 2..=n)
        .step_by(2)
        .map(|spin2| {
            let k = (n + spin2) / 2;
            let p = (spin2 + 1) as f64 * (average[k] - average.get(k + 1).copied().unwrap_or(0.0));
            (spin2, p)
        })
        .collect();
    let normalization_defect = (weights.iter().map(|&(_, p)| p).sum::<f64>() - 1.0).abs();
    let projected_squared: f64 = weights
        .iter()
        .map(|&(spin2, p)| {
            let s = spin2 as f64 * 0.5;
            p * s * (s + 1.0)
        })
        .sum();
    let squared_moment_defect = (projected_squared - squared).abs();
    if weights.iter().any(|&(_, p)| p < -tolerance)
        || normalization_defect > tolerance
        || squared_moment_defect > tolerance * (1.0 + squared.abs())
    {
        return Err(DftError::InvalidInput(
            "total-spin sector weights did not reproduce the normalization and spin moment".into(),
        ));
    }
    Ok(SpinSectorWeights {
        weights,
        normalization_defect,
        squared_moment_defect,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinant_sectors_and_global_spin_rotation() {
        let overlap = Matrix::identity(2);
        let mut c = BlochMatrix::zeros(4, 2);
        let mut check = |second_row: usize, expected: [f64; 2]| {
            c.re[(0, 0)] = 1.0;
            c.re[(second_row, 1)] = 1.0;
            for _ in 0..2 {
                let result = molecular_spin_diagnostics(&c, &[1.0, 1.0], &overlap).unwrap();
                let sectors = result.sectors.unwrap();
                assert!((result.squared - 2.0 * expected[1]).abs() < 1e-13);
                for ((s2, p), q) in sectors.weights.into_iter().zip(expected) {
                    assert!(s2 == 0 || s2 == 2);
                    assert!((p - q).abs() < 1e-13, "2S={s2}: {p} vs {q}");
                }
                let (s, t) = (0.17_f64.sin(), 0.17_f64.cos());
                for a in 0..2 {
                    for i in 0..2 {
                        let (up_r, down_i) = (c.re[(a, i)], c.im[(a + 2, i)]);
                        let (up_i, down_r) = (c.im[(a, i)], c.re[(a + 2, i)]);
                        c.re[(a, i)] = t * up_r + s * down_i;
                        c.im[(a + 2, i)] = -s * up_r + t * down_i;
                        c.im[(a, i)] = t * up_i - s * down_r;
                        c.re[(a + 2, i)] = s * up_i + t * down_r;
                    }
                }
            }
            c = BlochMatrix::zeros(4, 2);
        };
        check(2, [1.0, 0.0]); // shared spatial orbital, singlet
        check(1, [0.0, 1.0]); // two parallel spins, triplet
        check(3, [0.5, 0.5]); // orthogonal antiparallel spinors, equal mixture
    }

    #[test]
    fn fractional_occupations_match_independent_bernoulli_ensemble() {
        let c = BlochMatrix::from_pair((Matrix::identity(2), Matrix::zeros(2, 2)));
        let result = molecular_spin_diagnostics(&c, &[0.7, 0.2], &Matrix::identity(1)).unwrap();
        let single_occupation = 0.7 * (1.0 - 0.2) + (1.0 - 0.7) * 0.2;
        assert!((result.squared - 0.75 * single_occupation).abs() < 1e-14);
        assert!((result.expectation[2] - 0.25).abs() < 1e-14);
        assert!(result.sectors.is_none());
    }
}
