// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use crate::dfpt::orbitals::{adjoint, matmul};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

pub(super) fn identity(n: usize) -> BlochMatrix {
    BlochMatrix::from_pair((Matrix::identity(n), Matrix::zeros(n, n)))
}

pub(super) fn max_abs(matrix: &BlochMatrix) -> f64 {
    matrix
        .re
        .as_slice()
        .iter()
        .zip(matrix.im.as_slice())
        .map(|(r, i)| r.hypot(*i))
        .fold(0.0, f64::max)
}

pub(super) fn validate_hermitian(series: &[BlochMatrix], n: usize, order: usize) -> Result<()> {
    if n == 0
        || series.is_empty()
        || series.len() > order + 1
        || series.iter().any(|a| {
            [a.re.rows, a.re.cols, a.im.rows, a.im.cols] != [n; 4]
                || a.re
                    .as_slice()
                    .iter()
                    .chain(a.im.as_slice())
                    .any(|x| !x.is_finite())
                || a.hermiticity_error() > 1e-10
        })
    {
        return Err(DftError::InvalidInput(
            "two-component operators require finite matching Hermitian coefficients".into(),
        ));
    }
    Ok(())
}

pub(super) fn block(
    matrix: &BlochMatrix,
    row: usize,
    column: usize,
    rows: usize,
    cols: usize,
) -> BlochMatrix {
    let mut out = BlochMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            out.re[(i, j)] = matrix.re[(row + i, column + j)];
            out.im[(i, j)] = matrix.im[(row + i, column + j)];
        }
    }
    out
}

pub(super) fn product(
    a: &[BlochMatrix],
    b: &[BlochMatrix],
    order: usize,
) -> Result<Vec<BlochMatrix>> {
    (0..=order)
        .map(|n| {
            let mut out = BlochMatrix::zeros(a[0].rows(), b[0].cols());
            for (i, left) in a.iter().take(n + 1).enumerate() {
                if let Some(right) = b.get(n - i) {
                    out.add_scaled(&matmul(left, right)?, 1.0);
                }
            }
            Ok(out)
        })
        .collect()
}

pub(super) fn congruence(
    b: &[BlochMatrix],
    a: &[BlochMatrix],
    order: usize,
) -> Result<Vec<BlochMatrix>> {
    let bt = b.iter().map(adjoint).collect::<Result<Vec<_>>>()?;
    product(&bt, &product(a, b, order)?, order)
}

/// Principal square root and its inverse with analytic moving-metric
/// coefficients. Root derivatives solve R0 Rk + Rk R0 = Ak - sum Ri Rj.
/// The doubled-real representation retains all complex Hermitian entries.
pub(super) fn root_and_inverse(
    a: &[BlochMatrix],
    order: usize,
    cutoff: f64,
) -> Result<(Vec<BlochMatrix>, Vec<BlochMatrix>)> {
    let n = 2 * a[0].rows();
    let mut base = a[0].clone();
    base.make_hermitian();
    let (values, vectors) = symmetric_eigen(&base.real_embedding())?;
    if values.iter().any(|x| !x.is_finite() || *x <= cutoff) {
        return Err(DftError::LinearAlgebra(
            "two-component direct rotation requires a full-rank positive metric and large-component projection"
                .into(),
        ));
    }
    let diagonal: Vec<_> = values.iter().map(|v| v.sqrt()).collect();
    let transpose = vectors.transpose();
    let spectral = |power: f64| {
        let mut d = Matrix::zeros(n, n);
        for (i, &v) in diagonal.iter().enumerate() {
            d[(i, i)] = v.powf(power);
        }
        vectors.matmul_seq(&d).matmul_seq(&transpose)
    };
    let mut root = vec![spectral(1.0)];
    let mut inv = vec![spectral(-1.0)];
    for k in 1..=order {
        let mut rhs = a
            .get(k)
            .map_or_else(|| Matrix::zeros(n, n), BlochMatrix::real_embedding);
        for i in 1..k {
            let term = root[i].matmul_seq(&root[k - i]);
            for (value, subtract) in rhs.as_mut_slice().iter_mut().zip(term.as_slice()) {
                *value -= subtract;
            }
        }
        let mut transformed = transpose.matmul_seq(&rhs).matmul_seq(&vectors);
        for i in 0..n {
            for j in 0..n {
                transformed[(i, j)] /= diagonal[i] + diagonal[j];
            }
        }
        root.push(vectors.matmul_seq(&transformed).matmul_seq(&transpose));
        let mut lower = Matrix::zeros(n, n);
        for i in 1..=k {
            let term = root[i].matmul_seq(&inv[k - i]);
            for (value, subtract) in lower.as_mut_slice().iter_mut().zip(term.as_slice()) {
                *value -= subtract;
            }
        }
        inv.push(inv[0].matmul_seq(&lower));
    }
    let convert = |values: Vec<Matrix>| {
        values
            .iter()
            .map(|a| {
                let mut c = BlochMatrix::from_real_embedding(a);
                c.make_hermitian();
                c
            })
            .collect()
    };
    Ok((convert(root), convert(inv)))
}
