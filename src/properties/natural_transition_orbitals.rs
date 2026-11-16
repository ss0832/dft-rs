// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Singular-value representation of a directed one-particle transition.

use crate::dfpt::orbitals::{adjoint, matmul};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::periodic::{band_transport::hermitian_eigen, BlochMatrix};
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct NaturalTransitionOrbitals {
    pub singular_values: Vec<f64>,
    pub particle_coefficients: BlochMatrix,
    pub hole_coefficients: BlochMatrix,
    /// The squared Frobenius norm of the original transition matrix.
    pub transition_weight: f64,
    /// Frobenius defect in the original orthonormal transition frame.
    pub reconstruction_residual: f64,
}

fn finite(matrix: &BlochMatrix) -> bool {
    matrix.im.rows == matrix.re.rows
        && matrix.im.cols == matrix.re.cols
        && matrix
            .re
            .as_slice()
            .iter()
            .chain(matrix.im.as_slice())
            .all(|x| x.is_finite())
}

/// Restrict a coefficient matrix to contiguous columns in original order.
pub(crate) fn columns(m: &BlochMatrix, start: usize, count: usize) -> BlochMatrix {
    let mut out = BlochMatrix::zeros(m.rows(), count);
    for i in 0..m.rows() {
        for j in 0..count {
            out.re[(i, j)] = m.re[(i, start + j)];
            out.im[(i, j)] = m.im[(i, start + j)];
        }
    }
    out
}

pub(crate) fn real(real: Matrix) -> BlochMatrix {
    let imaginary = Matrix::zeros(real.rows, real.cols);
    BlochMatrix::from_pair((real, imaginary))
}

/// Decompose `T = U sigma V^H` in orthonormal particle (row) and hole
/// (column) frames. The supplied coefficient matrices express the frames
/// in a common AO basis, including a spin-major AO basis for spinors.
pub fn decompose(
    transition: &BlochMatrix,
    particles: &BlochMatrix,
    holes: &BlochMatrix,
) -> Result<NaturalTransitionOrbitals> {
    let (nv, no) = (transition.rows(), transition.cols());
    if particles.cols() != nv
        || holes.cols() != no
        || particles.rows() != holes.rows()
        || [transition, particles, holes].iter().any(|m| !finite(m))
    {
        return Err(DftError::InvalidInput(
            "natural transition orbitals need finite matching transition and orbital matrices"
                .into(),
        ));
    }
    let scale = transition
        .re
        .as_slice()
        .iter()
        .chain(transition.im.as_slice())
        .fold(0.0_f64, |max, &x| max.max(x.abs()));
    if scale == 0.0 {
        return Ok(NaturalTransitionOrbitals {
            singular_values: Vec::new(),
            particle_coefficients: BlochMatrix::zeros(particles.rows(), 0),
            hole_coefficients: BlochMatrix::zeros(holes.rows(), 0),
            transition_weight: 0.0,
            reconstruction_residual: 0.0,
        });
    }
    // The Hermitian dilation resolves small singular values without
    // squaring the condition number. Positive eigenvectors have equally
    // normalized particle and hole blocks.
    let dim = nv + no;
    let mut dilation = BlochMatrix::zeros(dim, dim);
    for a in 0..nv {
        for i in 0..no {
            let x = transition.re[(a, i)] / scale;
            let y = transition.im[(a, i)] / scale;
            dilation.re[(a, nv + i)] = x;
            dilation.re[(nv + i, a)] = x;
            dilation.im[(a, nv + i)] = y;
            dilation.im[(nv + i, a)] = -y;
        }
    }
    let (eigenvalues, vectors) = if dilation.im.as_slice().iter().all(|&x| x == 0.0) {
        let (values, vectors) = symmetric_eigen(&dilation.re)?;
        (values, real(vectors))
    } else {
        hermitian_eigen(&dilation)?
    };
    let largest = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let roundoff = 64.0 * f64::EPSILON * dim as f64 * largest;
    let nonzero: Vec<_> = eigenvalues
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, value)| **value > roundoff)
        .take(nv.min(no))
        .collect();
    let rank = nonzero.len();
    let mut u = BlochMatrix::zeros(nv, rank);
    let mut v = BlochMatrix::zeros(no, rank);
    let mut singular_values = Vec::with_capacity(rank);
    for (k, (j, eigenvalue)) in nonzero.into_iter().enumerate() {
        singular_values.push(scale * eigenvalue);
        for a in 0..nv {
            u.re[(a, k)] = std::f64::consts::SQRT_2 * vectors.re[(a, j)];
            u.im[(a, k)] = std::f64::consts::SQRT_2 * vectors.im[(a, j)];
        }
        for i in 0..no {
            v.re[(i, k)] = std::f64::consts::SQRT_2 * vectors.re[(nv + i, j)];
            v.im[(i, k)] = std::f64::consts::SQRT_2 * vectors.im[(nv + i, j)];
        }
    }
    let particle_coefficients = matmul(particles, &u)?;
    let hole_coefficients = matmul(holes, &v)?;
    for (k, &sigma) in singular_values.iter().enumerate() {
        for a in 0..nv {
            u.re[(a, k)] *= sigma;
            u.im[(a, k)] *= sigma;
        }
    }
    let mut reconstructed = matmul(&u, &adjoint(&v)?)?;
    reconstructed.add_scaled(transition, -1.0);
    Ok(NaturalTransitionOrbitals {
        singular_values,
        particle_coefficients,
        hole_coefficients,
        transition_weight: transition.trace_product(transition),
        reconstruction_residual: reconstructed.trace_product(&reconstructed).sqrt(),
    })
}
