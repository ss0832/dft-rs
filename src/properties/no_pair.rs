// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electronic-sector projection and metric/projector response of a supplied
//! four-component Dirac operator. Hamiltonian and overlap providers must
//! already implement kinetic/magnetic balance and the intended nuclear model.
//! This component does not manufacture those integrals or a Dirac SCF state.
//!
//! Projection follows the separated electronic subspace as a whole. Spin and
//! intra-sector degeneracies require no eigenvector derivative. Coupling to
//! the negative-energy sector is retained in every projector coefficient.

use crate::linalg::{Matrix, Orthogonalizer};
use crate::periodic::BlochMatrix;
use crate::response::nuclear::{matrix_product, metric_transport};
use crate::{DftError, Result};

pub mod operator;
pub use operator::{no_pair_operator_taylor, NoPairOperatorTaylor};
pub mod stationary;

/// Factorial-normalized coefficients of the contravariant electronic
/// projector P (P S P=P) and of the covariant operator S P H P S.
#[derive(Clone, Debug)]
pub struct NoPairTaylor {
    pub projector: Vec<BlochMatrix>,
    pub projected_hamiltonian: Vec<BlochMatrix>,
    pub electronic_dimension: usize,
    pub spectral_gap: f64,
    pub projector_defects: Vec<f64>,
    pub metric_defects: Vec<f64>,
    pub commutator_defects: Vec<f64>,
    pub relative_commutator_defects: Vec<f64>,
    pub number_defects: Vec<f64>,
    pub complex_embedding_defects: Vec<f64>,
    pub hamiltonian_embedding_defects: Vec<f64>,
    pub relative_hamiltonian_embedding_defects: Vec<f64>,
}

fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}
fn largest(a: &Matrix) -> f64 {
    a.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}
fn trace(a: &Matrix) -> f64 {
    (0..a.rows).map(|i| a[(i, i)]).sum()
}

/// Analytic no-pair projector through any supplied order up to four.
/// Missing higher H/S coefficients mean exact zero, not numerical sampling.
/// `separator` is inside the electron/negative-energy gap (for a free Dirac
/// operator shifted by -c^2 it is -c^2). A full-rank positive overlap is
/// required; no spectral denominator is regularized or changed.
pub fn no_pair_projector_taylor(
    hamiltonian: &[BlochMatrix],
    overlap: &[BlochMatrix],
    order: usize,
    separator: f64,
    overlap_cutoff: f64,
) -> Result<NoPairTaylor> {
    let dimension = hamiltonian.first().map_or(0, BlochMatrix::rows);
    if dimension == 0
        || dimension % 4 != 0
        || overlap.is_empty()
        || order > 4
        || !separator.is_finite()
        || !overlap_cutoff.is_finite()
        || overlap_cutoff <= 0.0
        || hamiltonian.len() > order + 1
        || overlap.len() > order + 1
        || hamiltonian.iter().chain(overlap).any(|a| {
            a.rows() != dimension
                || a.cols() != dimension
                || a.im.rows != dimension
                || a.im.cols != dimension
                || a.re
                    .as_slice()
                    .iter()
                    .chain(a.im.as_slice())
                    .any(|x| !x.is_finite())
                || a.hermiticity_error() > 1e-10
        })
    {
        return Err(DftError::InvalidInput("no-pair projection needs Hermitian four-component H/S series, finite gap separator, positive overlap cutoff and order zero through four".into()));
    }
    let size = 2 * dimension;
    let pad = |data: &[BlochMatrix]| -> Vec<Matrix> {
        (0..=order)
            .map(|n| {
                data.get(n)
                    .map_or_else(|| Matrix::zeros(size, size), BlochMatrix::real_embedding)
            })
            .collect()
    };
    let h = pad(hamiltonian);
    let s = pad(overlap);
    let orth = Orthogonalizer::canonical(&s[0], overlap_cutoff)?;
    if orth.n_kept != size {
        return Err(DftError::InvalidInput("no-pair projection needs a linearly independent balanced basis and full-rank positive overlap".into()));
    }
    let (eigenvalues, eigenvectors) = orth.solve_roothaan(&h[0])?;
    if eigenvalues.contains(&separator) {
        return Err(DftError::InvalidInput(
            "the no-pair separator intersects the Dirac spectrum".into(),
        ));
    }
    let positive: Vec<_> = (0..size).filter(|&i| eigenvalues[i] > separator).collect();
    let negative: Vec<_> = (0..size).filter(|&i| eigenvalues[i] < separator).collect();
    if positive.is_empty()
        || negative.is_empty()
        || positive.len() % 2 != 0
        || negative.len() % 2 != 0
    {
        return Err(DftError::InvalidInput("the no-pair separator must split nonempty complex electronic and negative-energy subspaces".into()));
    }
    let occupied = positive.len();
    let gap = eigenvalues[positive[0]] - eigenvalues[*negative.last().unwrap()];
    let indices: Vec<_> = positive.into_iter().chain(negative).collect();
    let energies: Vec<_> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut c = Matrix::zeros(size, size);
    for (j, &old) in indices.iter().enumerate() {
        for i in 0..size {
            c[(i, j)] = eigenvectors[(i, old)];
        }
    }
    // Reuse the moving-overlap transport from the common nuclear response.
    let basis = metric_transport(&c, &s, order);
    let bt: Vec<_> = basis.iter().map(Matrix::transpose).collect();
    let f = matrix_product(&matrix_product(&bt, &h, order), &basis, order);
    let mut d = vec![Matrix::zeros(size, size); order + 1];
    for i in 0..occupied {
        d[0][(i, i)] = 1.0;
    }
    for n in 1..=order {
        let mut lower = Matrix::zeros(size, size);
        let mut commutator = lower.clone();
        for k in 1..n {
            add(&mut lower, &d[k].matmul_seq(&d[n - k]), 1.0);
            add(&mut commutator, &f[k].matmul_seq(&d[n - k]), 1.0);
            add(&mut commutator, &d[n - k].matmul_seq(&f[k]), -1.0);
        }
        for i in 0..size {
            for j in 0..size {
                if i < occupied && j < occupied {
                    d[n][(i, j)] = -lower[(i, j)];
                }
                if i >= occupied && j >= occupied {
                    d[n][(i, j)] = lower[(i, j)];
                }
            }
        }
        for a in occupied..size {
            for i in 0..occupied {
                let value = -(f[n][(a, i)] + commutator[(a, i)]) / (energies[a] - energies[i]);
                d[n][(a, i)] = value;
                d[n][(i, a)] = value;
            }
        }
    }
    let p = matrix_product(&matrix_product(&basis, &d, order), &bt, order);
    let projected = matrix_product(
        &matrix_product(&s, &p, order),
        &matrix_product(&h, &matrix_product(&p, &s, order), order),
        order,
    );
    let mut projector_defects = Vec::new();
    if p.iter()
        .chain(&projected)
        .any(|a| a.as_slice().iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::LinearAlgebra(
            "no-pair projector coefficients overflow".into(),
        ));
    }
    let mut metric_defects = Vec::new();
    let mut commutator_defects = Vec::new();
    let mut relative_commutator_defects = Vec::new();
    let mut number_defects = Vec::new();
    let d2 = matrix_product(&d, &d, order);
    let fd = matrix_product(&f, &d, order);
    let df = matrix_product(&d, &f, order);
    let metric = matrix_product(&matrix_product(&bt, &s, order), &basis, order);
    let number = matrix_product(&p, &s, order);
    for n in 0..=order {
        let mut defect = d2[n].clone();
        add(&mut defect, &d[n], -1.0);
        projector_defects.push(largest(&defect));
        let mut defect = metric[n].clone();
        if n == 0 {
            add(&mut defect, &Matrix::identity(size), -1.0);
        }
        metric_defects.push(largest(&defect));
        let mut defect = fd[n].clone();
        add(&mut defect, &df[n], -1.0);
        let raw = largest(&defect);
        commutator_defects.push(raw);
        let scale = (0..=n)
            .map(|k| largest(&f[k]) * largest(&d[n - k]) * size as f64)
            .sum::<f64>();
        relative_commutator_defects.push(if scale == 0.0 { raw } else { raw / scale });
        number_defects.push(
            (0.5 * trace(&number[n]) - if n == 0 { occupied as f64 / 2.0 } else { 0.0 }).abs(),
        );
    }
    let convert = |values: Vec<Matrix>| -> (Vec<BlochMatrix>, Vec<f64>, Vec<f64>) {
        let mut output = Vec::new();
        let mut absolute = Vec::new();
        let mut relative = Vec::new();
        for a in values {
            let value = BlochMatrix::from_real_embedding(&a);
            let defect = a.max_abs_diff(&value.real_embedding());
            let norm = largest(&a);
            absolute.push(defect);
            relative.push(if norm == 0.0 { defect } else { defect / norm });
            output.push(value);
        }
        (output, absolute, relative)
    };
    let (projector, complex_embedding_defects, _) = convert(p);
    let (
        projected_hamiltonian,
        hamiltonian_embedding_defects,
        relative_hamiltonian_embedding_defects,
    ) = convert(projected);
    Ok(NoPairTaylor {
        projector,
        projected_hamiltonian,
        electronic_dimension: occupied / 2,
        spectral_gap: gap,
        projector_defects,
        metric_defects,
        commutator_defects,
        relative_commutator_defects,
        number_defects,
        complex_embedding_defects,
        hamiltonian_embedding_defects,
        relative_hamiltonian_embedding_defects,
    })
}
