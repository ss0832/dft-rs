// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! CDIIS (Pulay commutator DIIS) in a non-orthogonal AO basis.
//!
//! The error vector is the orthonormalized commutator
//! `e = Xᵀ (F P S − S P F) X` per spin channel (concatenated for UHF);
//! at convergence e → 0. Extrapolation coefficients solve the bordered
//! system `[[B, −1], [−1ᵀ, 0]] [c, λ] = [0, −1]` with a pivot-guarded
//! elimination that reports near-singularity so the caller can fall back to
//! the undamped Fock instead of applying a garbage extrapolation.

// Eliminate the last affine coefficient before forming the error Gram
// matrix. Nearly equal error vectors then retain their independent small
// differences instead of subtracting large, already-rounded dot products.
use crate::linalg::{AccurateSum, Matrix};

pub struct Diis {
    max_vecs: usize,
    focks: Vec<Vec<Matrix>>, // per stored iterate: one Fock per spin
    errors: Vec<Vec<f64>>,   // flattened concatenated error vectors
}

impl Diis {
    pub fn new(max_vecs: usize) -> Self {
        Self {
            max_vecs,
            focks: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.focks.clear();
        self.errors.clear();
    }

    /// Store physical Focks with a caller-supplied metric commutator. This
    /// supports a common extrapolation over distinct Bloch overlap metrics.
    pub(crate) fn push_with_errors(&mut self, focks: &[Matrix], errors: Vec<f64>) {
        self.focks.push(focks.to_vec());
        self.errors.push(errors);
        if self.focks.len() > self.max_vecs {
            self.focks.remove(0);
            self.errors.remove(0);
        }
    }

    /// Compute the per-spin commutator errors and push the iterate. Returns
    /// the largest |e| element (the DIIS error norm used for gating).
    pub fn push(&mut self, focks: &[Matrix], densities: &[Matrix], s: &Matrix, x: &Matrix) -> f64 {
        let mut err = Vec::new();
        let mut max_abs: f64 = 0.0;
        for (f, p) in focks.iter().zip(densities) {
            let e = crate::linalg::metric_commutator(f, p, s, x);
            for &v in e.as_slice() {
                max_abs = max_abs.max(v.abs());
                err.push(v);
            }
        }
        self.focks.push(focks.to_vec());
        self.errors.push(err);
        if self.focks.len() > self.max_vecs {
            self.focks.remove(0);
            self.errors.remove(0);
        }
        max_abs
    }

    /// Extrapolated Fock matrices, or `None` when the subspace is too small
    /// or irreparably singular.
    ///
    /// The B matrix is normalized by its largest element before solving —
    /// near convergence B_ij ~ ‖e‖² underflows any absolute pivot guard,
    /// which would silently disable DIIS exactly when it is needed to
    /// stabilize the endgame (observed as a slowly creeping commutator).
    /// If the scaled system is still singular, the OLDEST vectors are
    /// dropped until it is solvable.
    pub fn extrapolate(&self) -> Option<Vec<Matrix>> {
        let m_total = self.focks.len();
        for skip in 0..m_total.saturating_sub(1) {
            let m = m_total - skip;
            if m < 2 {
                break;
            }
            let focks = &self.focks[skip..];
            let errors = &self.errors[skip..];
            let Some(coeffs) = affine_error_coefficients(errors) else {
                continue; // drop the oldest vector and retry
            };
            let nspin = focks[0].len();
            let mut out = Vec::with_capacity(nspin);
            for spin in 0..nspin {
                out.push(affine_fock(focks, &coeffs[..m], spin));
            }
            return Some(out);
        }
        None
    }
}

fn affine_error_coefficients(errors: &[Vec<f64>]) -> Option<Vec<f64>> {
    let reference = errors.last()?;
    let columns = errors.len().checked_sub(1)?;
    if columns == 0
        || errors.iter().any(|e| e.len() != reference.len())
        || errors.iter().flatten().any(|e| !e.is_finite())
    {
        return None;
    }
    let scale = errors.iter().flatten().fold(0.0_f64, |a, b| a.max(b.abs()));
    if scale == 0.0 {
        let mut coefficients = vec![0.0; errors.len()];
        coefficients[columns] = 1.0;
        return Some(coefficients);
    }
    let differences: Vec<Vec<f64>> = errors[..columns]
        .iter()
        .map(|e| {
            e.iter()
                .zip(reference)
                .map(|(&value, &origin)| {
                    let difference = value - origin;
                    if difference.is_finite() {
                        difference / scale
                    } else {
                        value / scale - origin / scale
                    }
                })
                .collect()
        })
        .collect();
    let mut gram = vec![0.0; columns * columns];
    let mut rhs = vec![0.0; columns];
    let mut largest = 0.0_f64;
    for i in 0..columns {
        for j in 0..columns {
            let mut sum = AccurateSum::default();
            for (&a, &b) in differences[i].iter().zip(&differences[j]) {
                sum.add_product(a, b);
            }
            gram[i * columns + j] = sum.value();
            largest = largest.max(sum.value().abs());
        }
        let mut sum = AccurateSum::default();
        for (&a, &b) in differences[i].iter().zip(reference) {
            sum.add_product(-a, b / scale);
        }
        rhs[i] = sum.value();
    }
    if largest <= 0.0 || !largest.is_finite() {
        return None;
    }
    for value in &mut gram {
        *value /= largest;
    }
    for value in &mut rhs {
        *value /= largest;
    }
    // The same relative pivot guard is used on the affine least-squares
    // system; no physical SCF convergence threshold is changed.
    let mut coefficients = solve_bordered_small(&mut gram, &mut rhs, columns)?;
    let mut last = AccurateSum::default();
    last.add_product(1.0, 1.0);
    for &coefficient in &coefficients {
        last.add_product(-coefficient, 1.0);
    }
    coefficients.push(last.value());
    Some(coefficients)
}

fn affine_fock(focks: &[Vec<Matrix>], coefficients: &[f64], spin: usize) -> Matrix {
    let last = focks.len() - 1;
    let reference = &focks[last][spin];
    let mut output = reference.clone();
    for (entry, value) in output.as_mut_slice().iter_mut().enumerate() {
        let mut correction = AccurateSum::default();
        for (i, fock_set) in focks[..last].iter().enumerate() {
            correction.add_product(
                coefficients[i],
                fock_set[spin].as_slice()[entry] - reference.as_slice()[entry],
            );
        }
        // The bordered problem imposes sum(c)=1. Enforce that affine
        // constraint before multiplying large common Fock entries, rather
        // than amplifying its solve roundoff.
        correction.add_product(*value, 1.0);
        *value = correction.value();
    }
    output
}

/// Gaussian elimination with partial pivoting for the small bordered DIIS
/// system; returns `None` on a near-singular pivot (caller falls back).
fn solve_bordered_small(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // Pivot.
        let mut best = col;
        let mut best_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best_val {
                best = row;
                best_val = v;
            }
        }
        if best_val < 1e-14 {
            return None;
        }
        if best != col {
            for k in 0..n {
                a.swap(col * n + k, best * n + k);
            }
            b.swap(col, best);
        }
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= a[row * n + k] * x[k];
        }
        x[row] = sum / a[row * n + row];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bordered_solver_known_solution() {
        // B = [[1, 0.5], [0.5, 1]] with the border: c = [0.5, 0.5], λ = 0.75.
        let n = 3;
        let mut a = vec![1.0, 0.5, -1.0, 0.5, 1.0, -1.0, -1.0, -1.0, 0.0];
        let mut b = vec![0.0, 0.0, -1.0];
        let x = solve_bordered_small(&mut a, &mut b, n).unwrap();
        assert!((x[0] - 0.5).abs() < 1e-12);
        assert!((x[1] - 0.5).abs() < 1e-12);
        assert!((x[2] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn singular_system_returns_none() {
        // Two identical error vectors → singular B.
        let n = 3;
        let mut a = vec![1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0];
        let mut b = vec![0.0, 0.0, -1.0];
        assert!(solve_bordered_small(&mut a, &mut b, n).is_none());
    }

    /// DIIS coefficients must sum to 1 and reproduce a stationary point:
    /// if one stored Fock has zero error it gets coefficient ≈ 1.
    #[test]
    fn extrapolation_prefers_zero_error_iterate() {
        let mut diis = Diis::new(8);
        let s = Matrix::identity(2);
        let x = Matrix::identity(2);
        // Iterate 1: F and P that do NOT commute → nonzero error.
        let f1 = Matrix::from_row_major(2, 2, vec![1.0, 0.4, 0.4, 2.0]);
        let p1 = Matrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 0.0]);
        diis.push(&[f1], &[p1], &s, &x);
        // Iterate 2: diagonal F commutes with diagonal P → zero error.
        let f2 = Matrix::from_row_major(2, 2, vec![1.5, 0.0, 0.0, 2.5]);
        let p2 = Matrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 0.0]);
        diis.push(std::slice::from_ref(&f2), &[p2], &s, &x);
        let out = diis.extrapolate().unwrap();
        assert!(out[0].max_abs_diff(&f2) < 1e-10);
    }

    #[test]
    fn extrapolation_preserves_a_common_operator_with_large_coefficients() {
        let mut diis = Diis::new(3);
        let fock = Matrix::from_row_major(2, 2, vec![1024.0, 0.125, 0.125, -256.0]);
        // Nearly parallel errors require extrapolation outside their convex
        // hull. The answer for identical physical operators is independent
        // of the coefficients and of their rounded sum.
        diis.push_with_errors(std::slice::from_ref(&fock), vec![1.0, 1e-5]);
        diis.push_with_errors(std::slice::from_ref(&fock), vec![1.00001, 1.1e-5]);
        let output = diis.extrapolate().unwrap();
        assert_eq!(output[0].as_slice(), fock.as_slice());
    }

    #[test]
    fn affine_extrapolation_retains_the_closed_binary_correction() {
        let base = 2.0_f64.powi(40);
        let delta = 2.0_f64.powi(-12);
        let scale = 2.0_f64.powi(30);
        let focks: Vec<_> = [base + delta, base + 2.0 * delta, base - delta]
            .into_iter()
            .map(|value| vec![Matrix::from_row_major(1, 1, vec![value])])
            .collect();
        let coefficients = [scale, 1.0, -scale];
        // Exact binary arithmetic gives base + 2*scale*delta + 2*delta.
        // Summing the three large weighted operators loses the last term.
        let expected = base + 2.0 * scale * delta + 2.0 * delta;
        let output = affine_fock(&focks, &coefficients, 0);
        assert_eq!(output[(0, 0)], expected);
        let rounded: f64 = focks
            .iter()
            .zip(coefficients)
            .map(|(fock, coefficient)| fock[0][(0, 0)] * coefficient)
            .sum();
        assert_ne!(rounded, expected);
    }

    #[test]
    fn independent_small_error_differences_survive_a_large_common_residual() {
        let epsilon = 2.0_f64.powi(-30);
        let expected = vec![
            -2.0_f64.powi(30),
            -2.0_f64.powi(30),
            1.0 + 2.0_f64.powi(30) * 2.0,
        ];
        for exponent in [-200, 0, 200] {
            let scale = 2.0_f64.powi(exponent);
            let errors = vec![
                vec![scale * (1.0 + epsilon), scale],
                vec![scale, scale * (1.0 + epsilon)],
                vec![scale, scale],
            ];
            let coefficients = affine_error_coefficients(&errors).unwrap();
            assert_eq!(coefficients, expected);
            let mut diis = Diis::new(3);
            for (index, error) in errors.into_iter().enumerate() {
                let a = 1.0 + if index == 0 { epsilon } else { 0.0 };
                let b = 1.0 + if index == 1 { epsilon } else { 0.0 };
                let matrix =
                    Matrix::from_row_major(3, 3, vec![1024.0, a, b, a, 2.0, 0.0, b, 0.0, 3.0]);
                diis.push_with_errors(&[matrix], error);
            }
            let output = diis.extrapolate().unwrap();
            assert_eq!(
                output[0].as_slice(),
                &[1024.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
            );
        }
    }
}
