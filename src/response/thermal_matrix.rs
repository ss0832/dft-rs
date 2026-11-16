// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Noncommuting finite-temperature matrix response without eigenvector
//! derivatives or spectral denominator floors. Moving metrics use the
//! same analytic transport as molecular and no-pair response.

use super::nuclear::{matrix_product, metric_transport};
use super::ResponseSolverConfig;
use crate::linalg::{solve_linear_matrix, Matrix, Orthogonalizer};
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct ThermalMatrixTaylor {
    /// Contravariant density coefficients, C f C^H, not an idempotent projector.
    pub density: Vec<BlochMatrix>,
    pub metric_inverse: Vec<BlochMatrix>,
    pub metric_defects: Vec<f64>,
    pub hermiticity_defects: Vec<f64>,
    pub relative_commutator_defects: Vec<f64>,
    pub complex_embedding_defects: Vec<f64>,
    pub maximum_inverse_residual: f64,
    pub exponential_tail_estimate: f64,
    pub exponential_terms: usize,
    pub doubling_steps: usize,
}

fn norm(a: &Matrix) -> f64 {
    a.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn add(a: &mut Matrix, b: &Matrix, factor: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += factor * y;
    }
}

fn identity(n: usize, order: usize) -> Vec<Matrix> {
    let mut out = vec![Matrix::zeros(n, n); order + 1];
    out[0] = Matrix::identity(n);
    out
}

fn inverse(a: &[Matrix], order: usize) -> Result<(Vec<Matrix>, f64)> {
    let n = a[0].rows;
    let mut out = vec![solve_linear_matrix(&a[0], &Matrix::identity(n))?];
    for k in 1..=order {
        let mut rhs = Matrix::zeros(n, n);
        for j in 1..=k {
            add(&mut rhs, &a[j].matmul_seq(&out[k - j]), -1.0);
        }
        out.push(solve_linear_matrix(&a[0], &rhs)?);
    }
    let check = matrix_product(a, &out, order);
    let mut residual = 0.0_f64;
    for k in 0..=order {
        let mut defect = check[k].clone();
        if k == 0 {
            add(&mut defect, &Matrix::identity(n), -1.0);
        }
        let scale = (0..=k)
            .map(|j| norm(&a[j]) * norm(&out[k - j]) * n as f64)
            .sum::<f64>();
        let raw = norm(&defect);
        residual = residual.max(if scale == 0.0 { raw } else { raw / scale });
    }
    Ok((out, residual))
}

fn valid(a: &BlochMatrix, n: usize) -> bool {
    a.rows() == n
        && a.cols() == n
        && a.im.rows == n
        && a.im.cols == n
        && a.re
            .as_slice()
            .iter()
            .chain(a.im.as_slice())
            .all(|x| x.is_finite())
        && a.hermiticity_error() <= 1e-10
}

/// f(H/kT) in a full-rank moving metric, with factorial-normalized H/S/kT
/// coefficients and chemical potential already included in H. Missing higher
/// coefficients are exactly zero. The reference kT must be positive.
///
/// The matrix exponential is evaluated after scaling its reference norm below
/// one half. Bounded logistic doubling then uses
/// f(2X)=f(X)^2/[f(X)^2+(I-f(X))^2]. The denominator has spectrum in [1/2,1]
/// for physical occupations; no physical gap is inverted or regularized.
pub fn fermi_matrix_taylor(
    hamiltonian: &[BlochMatrix],
    overlap: &[BlochMatrix],
    temperature_energy: &[f64],
    order: usize,
    overlap_cutoff: f64,
    config: &ResponseSolverConfig,
) -> Result<ThermalMatrixTaylor> {
    config.validate()?;
    let n = hamiltonian.first().map_or(0, BlochMatrix::rows);
    if n == 0
        || order > 4
        || hamiltonian.len() > order + 1
        || overlap.is_empty()
        || overlap.len() > order + 1
        || temperature_energy.is_empty()
        || temperature_energy.len() > order + 1
        || temperature_energy.iter().any(|x| !x.is_finite())
        || temperature_energy[0] <= 0.0
        || !overlap_cutoff.is_finite()
        || overlap_cutoff <= 0.0
        || hamiltonian.iter().chain(overlap).any(|a| !valid(a, n))
    {
        return Err(DftError::InvalidInput("thermal matrix response needs matching finite Hermitian H/S series, positive kT/cutoff and order zero through four".into()));
    }
    let size = 2 * n;
    let pad = |a: &[BlochMatrix]| {
        (0..=order)
            .map(|k| {
                a.get(k)
                    .map_or_else(|| Matrix::zeros(size, size), BlochMatrix::real_embedding)
            })
            .collect::<Vec<_>>()
    };
    let h = pad(hamiltonian);
    let s = pad(overlap);
    let orth = Orthogonalizer::canonical(&s[0], overlap_cutoff)?;
    if orth.n_kept != size {
        return Err(DftError::InvalidInput("thermal matrix response requires a full-rank positive overlap; no Nambu state may be dropped".into()));
    }
    let basis = metric_transport(&orth.x, &s, order);
    let transpose: Vec<_> = basis.iter().map(Matrix::transpose).collect();
    let fock = matrix_product(&matrix_product(&transpose, &h, order), &basis, order);
    let mut inv_t = vec![0.0; order + 1];
    inv_t[0] = 1.0 / temperature_energy[0];
    for k in 1..=order {
        inv_t[k] = -(1..=k)
            .map(|j| temperature_energy.get(j).copied().unwrap_or(0.0) * inv_t[k - j])
            .sum::<f64>()
            / temperature_energy[0];
    }
    let mut x = vec![Matrix::zeros(size, size); order + 1];
    for k in 0..=order {
        for j in 0..=k {
            add(&mut x[k], &fock[j], inv_t[k - j]);
        }
    }
    let bound = (0..size)
        .map(|i| (0..size).map(|j| x[0][(i, j)].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    if !bound.is_finite() || x.iter().flat_map(Matrix::as_slice).any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "thermal matrix temperature/parameter path exceeds finite arithmetic".into(),
        ));
    }
    let doublings = if bound <= 0.5 {
        0
    } else {
        (bound.log2() + 1.0).ceil() as usize
    };
    let scale = 0.5_f64.powi(doublings as i32);
    for a in &mut x {
        for v in a.as_mut_slice() {
            *v *= scale;
        }
    }
    let unit = identity(size, order);
    let mut exponential = unit.clone();
    let mut term = unit.clone();
    let mut tail = f64::INFINITY;
    let mut terms = 0;
    for k in 1..=config.max_iterations {
        term = matrix_product(&term, &x, order);
        for a in &mut term {
            for v in a.as_mut_slice() {
                *v /= k as f64;
            }
        }
        tail = 0.0;
        for j in 0..=order {
            add(&mut exponential[j], &term[j], 1.0);
            let a = norm(&term[j]);
            let b = norm(&exponential[j]);
            tail = tail.max(if b == 0.0 { a } else { a / b });
        }
        terms = k;
        if k >= order + 8 && tail < 0.01 * config.tolerance {
            break;
        }
    }
    if !tail.is_finite() || tail >= 0.01 * config.tolerance {
        return Err(DftError::LinearAlgebra(format!(
            "thermal matrix exponential did not converge: relative last term {tail:.3e}"
        )));
    }
    add(&mut exponential[0], &unit[0], 1.0);
    let (mut density, mut inverse_residual) = inverse(&exponential, order)?;
    for _ in 0..doublings {
        let square = matrix_product(&density, &density, order);
        let mut complement = unit.clone();
        for k in 0..=order {
            add(&mut complement[k], &density[k], -1.0);
        }
        let mut denominator = matrix_product(&complement, &complement, order);
        for k in 0..=order {
            add(&mut denominator[k], &square[k], 1.0);
        }
        let (inv, residual) = inverse(&denominator, order)?;
        inverse_residual = inverse_residual.max(residual);
        density = matrix_product(&inv, &square, order);
    }
    let left = matrix_product(&fock, &density, order);
    let right = matrix_product(&density, &fock, order);
    let mut commutators = Vec::new();
    for k in 0..=order {
        let mut difference = left[k].clone();
        add(&mut difference, &right[k], -1.0);
        let scale = (0..=k)
            .map(|j| norm(&fock[j]) * norm(&density[k - j]) * size as f64)
            .sum::<f64>();
        let raw = norm(&difference);
        commutators.push(if scale == 0.0 { raw } else { raw / scale });
    }
    let metric = matrix_product(&matrix_product(&transpose, &s, order), &basis, order);
    let metric_defects = (0..=order)
        .map(|k| {
            let mut d = metric[k].clone();
            add(&mut d, &unit[k], -1.0);
            norm(&d)
        })
        .collect::<Vec<_>>();
    let ao = matrix_product(&matrix_product(&basis, &density, order), &transpose, order);
    let sinv = matrix_product(&basis, &transpose, order);
    let mut embedding = Vec::new();
    let convert = |data: Vec<Matrix>, defects: &mut Vec<f64>| {
        data.into_iter()
            .map(|a| {
                let b = BlochMatrix::from_real_embedding(&a);
                defects.push(a.max_abs_diff(&b.real_embedding()));
                b
            })
            .collect::<Vec<_>>()
    };
    let density = convert(ao, &mut embedding);
    let mut unused_embedding = Vec::new();
    let metric_inverse = convert(sinv, &mut unused_embedding);
    for (a, b) in embedding.iter_mut().zip(unused_embedding) {
        *a = a.max(b);
    }
    let hermiticity = density
        .iter()
        .map(BlochMatrix::hermiticity_error)
        .collect::<Vec<_>>();
    if density
        .iter()
        .chain(&metric_inverse)
        .flat_map(|a| a.re.as_slice().iter().chain(a.im.as_slice()))
        .any(|v| !v.is_finite())
        || inverse_residual > config.acceptance()
        || commutators
            .iter()
            .chain(&metric_defects)
            .chain(&hermiticity)
            .chain(&embedding)
            .any(|x| !x.is_finite() || *x > config.acceptance())
    {
        return Err(DftError::LinearAlgebra("thermal matrix original inverse/commutator/metric/embedding residual exceeds acceptance".into()));
    }
    Ok(ThermalMatrixTaylor {
        density,
        metric_inverse,
        metric_defects,
        hermiticity_defects: hermiticity,
        relative_commutator_defects: commutators,
        complex_embedding_defects: embedding,
        maximum_inverse_residual: inverse_residual,
        exponential_tail_estimate: tail,
        exponential_terms: terms,
        doubling_steps: doublings,
    })
}
