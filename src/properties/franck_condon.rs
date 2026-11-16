// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Franck-Condon integrals between the vibrational states of two harmonic
//! surfaces with different frequencies, rotated normal modes and displaced
//! minima.
//!
//! The normal coordinates of the final surface are `Q' = J Q + K` in terms
//! of those of the initial one (mass-weighted, atomic units). With
//! `J' = w'^(1/2) J w^(-1/2)`, `d = w'^(1/2) K`, `Q = (1 + J'^T J')^-1`,
//! `P = J' Q J'^T` and `R = Q J'^T`,
//!
//! ```text
//! <0'|0> = 2^(N/2) det(Q)^(1/2) prod_k (w'_k / w_k)^(1/4) exp(-d^T (1 - P) d / 2),
//! <v'+1_i|v> = [2/(v'_i+1)]^(1/2) [(1-P) d]_i <v'|v>
//!            + sum_j [v'_j/(v'_i+1)]^(1/2) (2P-1)_ij <v'-1_j|v>
//!            + sum_j [v_j/(v'_i+1)]^(1/2) 2 R_ji <v'|v-1_j>,
//! <v'|v+1_i> = -[2/(v_i+1)]^(1/2) (R d)_i <v'|v>
//!            + sum_j [v_j/(v_i+1)]^(1/2) (2Q-1)_ij <v'|v-1_j>
//!            + sum_j [v'_j/(v_i+1)]^(1/2) 2 R_ij <v'-1_j|v>.
//! ```
//!
//! For equal frequencies and `J = 1` these are the integrals of displaced
//! oscillators, whose squares from the ground state are the Poisson weights
//! of the Huang-Rhys factors `d_k^2 / 2`.

use crate::linalg::{solve_linear_matrix, Matrix};
use crate::{DftError, Result};
use std::collections::HashMap;

/// Two harmonic surfaces and the relation between their normal coordinates.
#[derive(Clone, Debug)]
pub struct Surfaces {
    /// Frequencies of the initial surface, Hartree.
    pub initial: Vec<f64>,
    /// Frequencies of the final surface, Hartree.
    pub final_: Vec<f64>,
    /// `Q' = rotation Q + shift`: rows final modes, columns initial modes.
    pub rotation: Matrix,
    pub shift: Vec<f64>,
}

/// One vibronic line.
#[derive(Clone, Debug)]
pub struct Line {
    /// Quanta of the initial state.
    pub initial: Vec<usize>,
    /// Quanta of the final state.
    pub final_: Vec<usize>,
    /// Vibrational energy of the final state minus that of the initial one,
    /// without the zero-point energies, Hartree.
    pub energy: f64,
    /// Population of the initial state times the squared overlap.
    pub intensity: f64,
}

/// Recursion matrices of a pair of surfaces.
struct Recursion {
    n: usize,
    /// `(1 - P) d` and `R d`.
    final_shift: Vec<f64>,
    initial_shift: Vec<f64>,
    /// `2P - 1`, `2Q - 1` and `2R` (rows initial, columns final).
    final_pair: Matrix,
    initial_pair: Matrix,
    mixed: Matrix,
    origin: f64,
    memo: HashMap<(Vec<usize>, Vec<usize>), f64>,
}

fn determinant(m: &Matrix) -> f64 {
    // Gaussian elimination with partial pivoting.
    let n = m.rows;
    let mut a = m.clone();
    let mut det = 1.0;
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[(row, col)].abs() > a[(pivot, col)].abs() {
                pivot = row;
            }
        }
        if a[(pivot, col)] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            for k in 0..n {
                let t = a[(col, k)];
                a[(col, k)] = a[(pivot, k)];
                a[(pivot, k)] = t;
            }
            det = -det;
        }
        det *= a[(col, col)];
        for row in col + 1..n {
            let factor = a[(row, col)] / a[(col, col)];
            for k in col..n {
                let value = a[(col, k)];
                a[(row, k)] -= factor * value;
            }
        }
    }
    det
}

impl Recursion {
    fn new(surfaces: &Surfaces) -> Result<Self> {
        let n = surfaces.initial.len();
        if surfaces.final_.len() != n
            || surfaces.shift.len() != n
            || surfaces.rotation.rows != n
            || surfaces.rotation.cols != n
            || surfaces
                .initial
                .iter()
                .chain(&surfaces.final_)
                .any(|w| !(w.is_finite() && *w > 0.0))
        {
            return Err(DftError::InvalidInput(
                "Franck-Condon integrals need two sets of positive frequencies of equal size"
                    .into(),
            ));
        }
        // J' = w'^(1/2) J w^(-1/2).
        let mut scaled = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                scaled[(i, j)] = surfaces.final_[i].sqrt() * surfaces.rotation[(i, j)]
                    / surfaces.initial[j].sqrt();
            }
        }
        let d: Vec<f64> = (0..n)
            .map(|i| surfaces.final_[i].sqrt() * surfaces.shift[i])
            .collect();
        let mut metric = scaled.transpose_matmul_seq(&scaled);
        for i in 0..n {
            metric[(i, i)] += 1.0;
        }
        let q = solve_linear_matrix(&metric, &Matrix::identity(n))?;
        let r = q.matmul_transpose_seq(&scaled);
        let p = scaled.matmul_seq(&r);
        let mut final_shift = vec![0.0; n];
        let mut initial_shift = vec![0.0; n];
        let mut exponent = 0.0;
        for i in 0..n {
            for j in 0..n {
                let complement = if i == j { 1.0 } else { 0.0 } - p[(i, j)];
                final_shift[i] += complement * d[j];
                initial_shift[i] += r[(i, j)] * d[j];
            }
            exponent += d[i] * final_shift[i];
        }
        let mut final_pair = Matrix::zeros(n, n);
        let mut initial_pair = Matrix::zeros(n, n);
        let mut mixed = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let unit = if i == j { 1.0 } else { 0.0 };
                final_pair[(i, j)] = 2.0 * p[(i, j)] - unit;
                initial_pair[(i, j)] = 2.0 * q[(i, j)] - unit;
                mixed[(i, j)] = 2.0 * r[(i, j)];
            }
        }
        let ratio: f64 = surfaces
            .final_
            .iter()
            .zip(&surfaces.initial)
            .map(|(a, b)| a / b)
            .product();
        let det = determinant(&q);
        if !(det > 0.0) {
            return Err(DftError::LinearAlgebra(
                "the Franck-Condon recursion matrix is not positive definite".into(),
            ));
        }
        let origin =
            2.0_f64.powf(0.5 * n as f64) * det.sqrt() * ratio.powf(0.25) * (-0.5 * exponent).exp();
        Ok(Self {
            n,
            final_shift,
            initial_shift,
            final_pair,
            initial_pair,
            mixed,
            origin,
            memo: HashMap::new(),
        })
    }

    /// `<final|initial>`.
    fn overlap(&mut self, final_: &[usize], initial: &[usize]) -> f64 {
        if let Some(value) = self.memo.get(&(final_.to_vec(), initial.to_vec())) {
            return *value;
        }
        let value = if let Some(i) = (0..self.n).find(|&i| final_[i] > 0) {
            // Lower the final state in mode i.
            let mut lower = final_.to_vec();
            lower[i] -= 1;
            let scale = (lower[i] + 1) as f64;
            let mut sum =
                (2.0 / scale).sqrt() * self.final_shift[i] * self.overlap(&lower, initial);
            for j in 0..self.n {
                if lower[j] > 0 {
                    let mut twice = lower.clone();
                    twice[j] -= 1;
                    sum += (lower[j] as f64 / scale).sqrt()
                        * self.final_pair[(i, j)]
                        * self.overlap(&twice, initial);
                }
                if initial[j] > 0 {
                    let mut other = initial.to_vec();
                    other[j] -= 1;
                    sum += (initial[j] as f64 / scale).sqrt()
                        * self.mixed[(j, i)]
                        * self.overlap(&lower, &other);
                }
            }
            sum
        } else if let Some(i) = (0..self.n).find(|&i| initial[i] > 0) {
            // The final state is the vibrational ground state: lower the
            // initial state in mode i.
            let mut lower = initial.to_vec();
            lower[i] -= 1;
            let scale = (lower[i] + 1) as f64;
            let mut sum =
                -(2.0 / scale).sqrt() * self.initial_shift[i] * self.overlap(final_, &lower);
            for j in 0..self.n {
                if lower[j] > 0 {
                    let mut twice = lower.clone();
                    twice[j] -= 1;
                    sum += (lower[j] as f64 / scale).sqrt()
                        * self.initial_pair[(i, j)]
                        * self.overlap(final_, &twice);
                }
            }
            sum
        } else {
            self.origin
        };
        self.memo.insert((final_.to_vec(), initial.to_vec()), value);
        value
    }
}

/// Every assignment of at most `per_mode` quanta per mode and `total` in all.
fn assignments(modes: usize, per_mode: usize, total: usize) -> Vec<Vec<usize>> {
    fn extend(
        mode: usize,
        remaining: usize,
        per_mode: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if mode == current.len() {
            out.push(current.clone());
            return;
        }
        for n in 0..=per_mode.min(remaining) {
            current[mode] = n;
            extend(mode + 1, remaining - n, per_mode, current, out);
        }
        current[mode] = 0;
    }
    let mut out = Vec::new();
    extend(0, total, per_mode, &mut vec![0; modes], &mut out);
    out
}

/// Vibronic lines between two harmonic surfaces. Initial states carry the
/// Boltzmann populations of `kt` (Hartree; zero keeps the vibrational ground
/// state only) with at most `max_hot` quanta in all; final states have at
/// most `max_quanta` quanta per mode and `max_total` in all. Lines below
/// `threshold` are dropped. The second value is the sum of the kept
/// intensities, which is one for the complete set.
pub fn vibronic_lines(
    surfaces: &Surfaces,
    kt: f64,
    max_hot: usize,
    max_quanta: usize,
    max_total: usize,
    threshold: f64,
) -> Result<(Vec<Line>, f64)> {
    if !(kt.is_finite() && kt >= 0.0) {
        return Err(DftError::InvalidInput(
            "the vibrational temperature must not be negative".into(),
        ));
    }
    let mut recursion = Recursion::new(surfaces)?;
    let n = recursion.n;
    // Partition function of the initial harmonic oscillators without the
    // zero-point energy.
    let partition: f64 = if kt > 0.0 {
        surfaces
            .initial
            .iter()
            .map(|w| 1.0 / (1.0 - (-w / kt).exp()))
            .product()
    } else {
        1.0
    };
    let hot = if kt > 0.0 { max_hot } else { 0 };
    let initial_states = assignments(n, hot, hot);
    let final_states = assignments(n, max_quanta, max_total);
    let mut lines = Vec::new();
    let mut covered = 0.0;
    for initial in &initial_states {
        let initial_energy: f64 = initial
            .iter()
            .zip(&surfaces.initial)
            .map(|(v, w)| *v as f64 * w)
            .sum();
        let population = if kt > 0.0 {
            (-initial_energy / kt).exp() / partition
        } else {
            1.0
        };
        if population < threshold {
            continue;
        }
        for final_ in &final_states {
            let overlap = recursion.overlap(final_, initial);
            let intensity = population * overlap * overlap;
            if intensity < threshold {
                continue;
            }
            let final_energy: f64 = final_
                .iter()
                .zip(&surfaces.final_)
                .map(|(v, w)| *v as f64 * w)
                .sum();
            covered += intensity;
            lines.push(Line {
                initial: initial.clone(),
                final_: final_.clone(),
                energy: final_energy - initial_energy,
                intensity,
            });
        }
    }
    lines.sort_by(|a, b| a.energy.total_cmp(&b.energy));
    Ok((lines, covered))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Displaced oscillators of equal frequency give the Poisson weights of
    /// the Huang-Rhys factor.
    #[test]
    fn displaced_oscillator_gives_poisson_weights() {
        let w = 0.01;
        let shift = 12.0;
        let s = 0.5 * w * shift * shift;
        let surfaces = Surfaces {
            initial: vec![w],
            final_: vec![w],
            rotation: Matrix::identity(1),
            shift: vec![shift],
        };
        let (lines, covered) = vibronic_lines(&surfaces, 0.0, 0, 12, 12, 0.0).unwrap();
        let mut term = (-s).exp();
        for (n, line) in lines.iter().enumerate() {
            assert!(
                (line.intensity - term).abs() < 1e-12,
                "quantum {n}: {} against {term}",
                line.intensity
            );
            term *= s / (n + 1) as f64;
        }
        assert!((covered - 1.0).abs() < 1e-6);
    }

    /// The intensities of a complete set of final states sum to the
    /// population of each initial state, also with rotated modes, changed
    /// frequencies and a finite temperature.
    #[test]
    fn rotated_surfaces_keep_the_sum_rule() {
        let angle = 0.3_f64;
        let mut rotation = Matrix::zeros(2, 2);
        rotation[(0, 0)] = angle.cos();
        rotation[(0, 1)] = -angle.sin();
        rotation[(1, 0)] = angle.sin();
        rotation[(1, 1)] = angle.cos();
        let surfaces = Surfaces {
            initial: vec![0.008, 0.012],
            final_: vec![0.007, 0.014],
            rotation,
            shift: vec![4.0, -3.0],
        };
        let (_, cold) = vibronic_lines(&surfaces, 0.0, 0, 30, 30, 0.0).unwrap();
        assert!((cold - 1.0).abs() < 1e-6, "cold sum {cold}");
        let kt = 0.004;
        let (lines, hot) = vibronic_lines(&surfaces, kt, 2, 30, 30, 0.0).unwrap();
        // Populations of the initial states with at most two quanta.
        let partition: f64 = surfaces
            .initial
            .iter()
            .map(|w| 1.0 / (1.0 - (-w / kt).exp()))
            .product();
        let mut expected = 0.0;
        for a in 0..=2 {
            for b in 0..=2 - a {
                let energy = a as f64 * surfaces.initial[0] + b as f64 * surfaces.initial[1];
                expected += (-energy / kt).exp() / partition;
            }
        }
        assert!(
            (hot - expected).abs() < 1e-5,
            "hot sum {hot} against {expected}"
        );
        assert!(lines.iter().any(|line| line.energy < 0.0));
    }
}
