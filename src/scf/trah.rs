// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Trust-region augmented-Hessian orbital steps. A Davidson subspace stores
//! Hessian products, never the full orbital Hessian. The lowest root of
//! [0, alpha*g^T; alpha*g, H] determines both the stabilizing shift and
//! the rotation. Alpha is bracketed until the rotation fits the trust sphere.
//! Actual/model energy agreement controls acceptance and the next radius.

use crate::linalg::{symmetric_eigen, Matrix};
use crate::{DftError, Result};

/// Choice of integer-occupation orbital globalization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrbitalOptimizer {
    /// Use TRAH on difficult nonlinear SCF paths; retain cheap local iterations.
    #[default]
    Auto,
    /// Start TRAH as soon as a determinant orbital frame is available.
    Trah,
    /// Retain the existing DIIS, truncated Newton and quasi-Newton sequence.
    Legacy,
}

pub(crate) struct Step {
    pub rotation: Vec<f64>,
    pub predicted_work: f64,
    pub residual: f64,
    pub shift: f64,
    pub products: usize,
}

pub(crate) struct TrustRegion {
    pub radius: f64,
    maximum: f64,
}

impl TrustRegion {
    pub fn new(radius: f64) -> Result<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(DftError::InvalidInput(
                "SCF trust radius must be positive and finite".into(),
            ));
        }
        Ok(Self {
            radius,
            maximum: (8.0 * radius).min(1.0).max(radius),
        })
    }

    /// Reject non-decreasing or invalid physical energies. A tiny accepted
    /// step is not convergence: the caller retains every physical SCF gate.
    pub fn assess(&mut self, predicted: f64, actual: f64, norm: f64) -> (bool, f64) {
        let ratio = actual / predicted;
        let accepted = predicted < 0.0 && actual < 0.0 && ratio.is_finite() && ratio > 0.05;
        if !accepted || ratio < 0.25 {
            self.radius *= 0.5;
        } else if ratio > 0.75 && norm > 0.8 * self.radius {
            self.radius = (1.5 * self.radius).min(self.maximum);
        }
        (accepted, ratio)
    }
}

/// Evaluate actual energy work without losing it to subtraction of large
/// endpoint energies. Near cancellation, integrate the analytic Fock work
/// on the straight density segment. Two independent Gaussian orders must
/// agree; an unresolved integral returns to the physical SCF iteration.
/// This changes no energy, density, or stationarity convergence threshold.
pub(crate) fn energy_work(
    before: f64,
    after: f64,
    mut slope: impl FnMut(f64) -> Result<f64>,
) -> Result<Option<f64>> {
    let difference = after - before;
    if difference.abs() > 64.0 * f64::EPSILON * (before.abs() + after.abs()) {
        return Ok(Some(difference));
    }
    let x2 = 1.0 / 3.0_f64.sqrt();
    let second = 0.5 * (slope(0.5 * (1.0 - x2))? + slope(0.5 * (1.0 + x2))?);
    let x3 = (3.0_f64 / 5.0).sqrt();
    let third = (5.0 / 18.0) * (slope(0.5 * (1.0 - x3))? + slope(0.5 * (1.0 + x3))?)
        + (4.0 / 9.0) * slope(0.5)?;
    let error = (third - second).abs();
    let resolved = third.is_finite() && error <= 0.01 * third.abs();
    crate::logger::log(&format!(
        "[scf] TRAH density-path work endpoint={difference:.12e} gauss2={second:.12e} gauss3={third:.12e} discrepancy={error:.12e} resolved={resolved}"
    ));
    Ok(resolved.then_some(third))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = crate::linalg::AccurateSum::default();
    for (&x, &y) in a.iter().zip(b) {
        sum.add_product(x, y);
    }
    sum.value()
}
fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn append_basis(basis: &mut Vec<Vec<f64>>, mut x: Vec<f64>) -> bool {
    let original = norm(&x);
    if original == 0.0 || !original.is_finite() {
        return false;
    }
    // Two passes keep the reduced symmetric eigenproblem well conditioned.
    for _ in 0..2 {
        for v in basis.iter() {
            let projection = dot(v, &x);
            for (a, b) in x.iter_mut().zip(v) {
                *a -= projection * b;
            }
        }
    }
    let size = norm(&x);
    if size <= 1e-12 * original {
        return false;
    }
    for a in &mut x {
        *a /= size;
    }
    basis.push(x);
    true
}

/// The hard case has a zero leading AH eigenvector component: the gradient
/// is orthogonal to the lowest negative-curvature mode. Solve the equivalent
/// reduced trust-region secular equation, retaining that boundary mode.
fn hard_case(h: &Matrix, g: &[f64], radius: f64) -> Result<(Vec<f64>, f64)> {
    let (e, u) = symmetric_eigen(h)?;
    let n = g.len();
    let q: Vec<f64> = (0..n)
        .map(|j| (0..n).map(|i| u[(i, j)] * g[i]).sum())
        .collect();
    let lower = (-e[0]).max(0.0);
    let coefficients = |shift: f64| -> Option<Vec<f64>> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let d = e[i] + shift;
            if d.abs() <= 32.0 * f64::EPSILON * (1.0 + e[i].abs() + shift) {
                if q[i].abs() > 32.0 * f64::EPSILON * norm(g) {
                    return None;
                }
                out.push(0.0);
            } else {
                out.push(-q[i] / d);
            }
        }
        Some(out)
    };
    let (mut y, shift) = if let Some(y) = coefficients(lower).filter(|x| norm(x) <= radius) {
        (y, lower)
    } else {
        let mut hi = lower + (norm(g) / radius).max(1.0);
        while coefficients(hi).is_none_or(|x| norm(&x) > radius) {
            hi = 2.0 * hi + 1.0;
        }
        let mut lo = lower;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if coefficients(mid).is_none_or(|x| norm(&x) > radius) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (coefficients(hi).expect("positive shifted denominator"), hi)
    };
    if shift == lower && lower > 0.0 {
        y[0] += (radius * radius - dot(&y, &y)).max(0.0).sqrt();
    }
    let s = (0..n)
        .map(|i| (0..n).map(|j| u[(i, j)] * y[j]).sum())
        .collect();
    Ok((s, -shift))
}

fn reduced_step(h: &Matrix, g: &[f64], radius: f64) -> Result<(Vec<f64>, f64)> {
    let n = g.len();
    let solve = |alpha: f64| -> Result<Option<(Vec<f64>, f64)>> {
        let mut ah = Matrix::zeros(n + 1, n + 1);
        for i in 0..n {
            ah[(0, i + 1)] = alpha * g[i];
            ah[(i + 1, 0)] = alpha * g[i];
            for j in 0..n {
                ah[(i + 1, j + 1)] = h[(i, j)];
            }
        }
        let (values, vectors) = symmetric_eigen(&ah)?;
        let head = vectors[(0, 0)];
        if head.abs() < 1e-12 {
            return Ok(None);
        }
        Ok(Some((
            (0..n)
                .map(|i| vectors[(i + 1, 0)] / (alpha * head))
                .collect(),
            values[0],
        )))
    };
    let Some(first) = solve(1.0)? else {
        return hard_case(h, g, radius);
    };
    if norm(&first.0) <= radius {
        return Ok(first);
    }
    let mut lo = 1.0;
    let mut hi = 2.0;
    let mut best = loop {
        let Some(candidate) = solve(hi)? else {
            return hard_case(h, g, radius);
        };
        if norm(&candidate.0) <= radius {
            break candidate;
        }
        lo = hi;
        hi *= 2.0;
        if !hi.is_finite() {
            return Err(DftError::InvalidInput("AH trust scaling overflow".into()));
        }
    };
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        let Some(candidate) = solve(mid)? else {
            return hard_case(h, g, radius);
        };
        if norm(&candidate.0) > radius {
            lo = mid;
        } else {
            hi = mid;
            best = candidate;
        }
        if (radius - norm(&best.0)).abs() <= 1e-8 * radius {
            break;
        }
    }
    Ok(best)
}

/// Matrix-free augmented-Hessian solve, generic over the physical orbital
/// manifold and its complete analytic density kernel. No finite differences.
pub(crate) fn direction(
    diagonal: &[f64],
    gradient: &[f64],
    mut hessian: impl FnMut(&[f64]) -> Result<Vec<f64>>,
    radius: f64,
    max_products: usize,
    relative_tolerance: f64,
) -> Result<Step> {
    let n = gradient.len();
    if n == 0
        || diagonal.len() != n
        || max_products == 0
        || !radius.is_finite()
        || radius <= 0.0
        || !relative_tolerance.is_finite()
        || relative_tolerance <= 0.0
        || diagonal.iter().chain(gradient).any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "invalid augmented-Hessian dimensions or controls".into(),
        ));
    }
    let gnorm = norm(gradient);
    if gnorm == 0.0 {
        return Ok(Step {
            rotation: vec![0.0; n],
            predicted_work: 0.0,
            residual: 0.0,
            shift: 0.0,
            products: 0,
        });
    }
    let limit = max_products.min(n);
    let mut basis = Vec::new();
    append_basis(&mut basis, gradient.to_vec());
    if limit > 1 {
        let lowest = diagonal
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let mut unit = vec![0.0; n];
        unit[lowest] = 1.0;
        append_basis(&mut basis, unit);
    }
    let mut images: Vec<Vec<f64>> = Vec::new();
    loop {
        while images.len() < basis.len() {
            let image = hessian(&basis[images.len()])?;
            if image.len() != n || image.iter().any(|x| !x.is_finite()) {
                return Err(DftError::InvalidInput(
                    "invalid orbital Hessian product".into(),
                ));
            }
            images.push(image);
        }
        let m = basis.len();
        let mut h = Matrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                h[(i, j)] = 0.5 * (dot(&basis[i], &images[j]) + dot(&basis[j], &images[i]));
            }
        }
        let g: Vec<f64> = basis.iter().map(|b| dot(b, gradient)).collect();
        let (reduced, shift) = reduced_step(&h, &g, radius)?;
        let mut rotation = vec![0.0; n];
        let mut image = vec![0.0; n];
        for j in 0..m {
            for i in 0..n {
                rotation[i] += reduced[j] * basis[j][i];
                image[i] += reduced[j] * images[j][i];
            }
        }
        let residual_vector: Vec<f64> = (0..n)
            .map(|i| gradient[i] + image[i] - shift * rotation[i])
            .collect();
        let residual = norm(&residual_vector);
        let step = Step {
            predicted_work: dot(gradient, &rotation) + 0.5 * dot(&rotation, &image),
            rotation,
            residual,
            shift,
            products: images.len(),
        };
        if residual <= relative_tolerance * gnorm || m >= limit {
            return Ok(step);
        }
        let correction = residual_vector
            .iter()
            .zip(diagonal)
            .map(|(r, d)| -r / (d - shift).abs().max(1e-8))
            .collect();
        if !append_basis(&mut basis, correction) && !append_basis(&mut basis, residual_vector) {
            return Ok(step);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn augmented_root_respects_boundary_and_negative_curvature() {
        for (diagonal, g, radius) in [
            (vec![1.0, 3.0], vec![0.7, -0.4], 0.1),
            (vec![-2.0, 3.0], vec![0.0, 0.5], 0.2),
            (vec![1e-6, 2.0], vec![1e-8, -3e-8], 0.2),
        ] {
            let step = direction(
                &diagonal,
                &g,
                |x| Ok(x.iter().zip(&diagonal).map(|(a, b)| a * b).collect()),
                radius,
                8,
                1e-10,
            )
            .unwrap();
            assert!(norm(&step.rotation) <= radius * (1.0 + 1e-12));
            assert!(step.predicted_work < 0.0);
            assert!(step.residual < 1e-9);
            if diagonal[0] < 0.0 {
                assert!(step.rotation[0].abs() > 0.1);
            }
            eprintln!("TRAH quadratic radius={radius:e} norm={:e} predicted={:e} shift={:e} residual={:e} products={}",norm(&step.rotation),step.predicted_work,step.shift,step.residual,step.products);
        }
    }
    #[test]
    fn trah_density_path_resolves_energy_cancellation() {
        // The nonzero cubic density energy lies below an endpoint ULP.
        // Its exact analytic gradient still determines the physical work.
        let value = energy_work(-142.0, -142.0, |t| Ok(-1e-20 * (1.0 + t + t * t)))
            .unwrap()
            .unwrap();
        assert!((value / (-11e-20 / 6.0) - 1.0).abs() < 1e-14);
        let unresolved = energy_work(-142.0, -142.0, |t| Ok((20.0 * t).exp())).unwrap();
        assert!(unresolved.is_none());
    }

    #[test]
    fn trust_radius_tracks_actual_model_agreement() {
        let mut t = TrustRegion::new(0.2).unwrap();
        assert!(!t.assess(-1.0, 0.1, 0.2).0);
        assert_eq!(t.radius, 0.1);
        assert!(t.assess(-1.0, -0.1, 0.1).0);
        assert_eq!(t.radius, 0.05);
        assert!(t.assess(-1.0, -0.9, 0.05).0);
        assert!((t.radius - 0.075).abs() < 1e-15);
        assert!(!t.assess(-1.0, f64::NAN, 0.05).0);
    }
}
