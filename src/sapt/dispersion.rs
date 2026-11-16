// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Coupled dispersion from the frequency-dependent density susceptibilities
//! of the monomers (Casimir-Polder integral).
//!
//! In the occupied-virtual basis of unit transition densities of a monomer
//! (one spatial block for a closed shell, alpha and beta blocks otherwise),
//! the linear response at the imaginary frequency `iu` of the time-
//! dependent Kohn-Sham (or Hartree-Fock) equations is
//! `R(u) = [(A + B) + u^2 (A - B)^-1]^-1`, which is
//! `sum_n w_n t_n t_n^T / (w_n^2 + u^2)` over the RPA states with the
//! transition vectors `t_n = X_n + Y_n`. With the Coulomb coupling `W` of
//! the unit transition densities of the two monomers,
//! `E_disp = -f (2/pi) int_0^inf du Tr[R_A(u) W R_B(u) W^T]`, the integral
//! form of `-f sum_nm (t_n^T W t_m)^2 / (w_n + w_m)`; `f` is the spin
//! factor of the sum over states. The integral is evaluated by
//! Gauss-Legendre points on `u = u0 (1 + s)/(1 - s)` with `u0 = 1/2`.

use crate::integrals::eri::EriTensor;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::Result;

/// Coulomb coupling `W[p][q] = (t_p | t_q)` of the transition densities of
/// monomer A (AO block `0..nao_a` of the dimer integrals) with those of B
/// (block `nao_a..`).
pub(crate) fn coupling_matrix(
    pa: &[Matrix],
    pb: &[Matrix],
    eri: &EriTensor,
    nao_a: usize,
    nao_b: usize,
) -> Matrix {
    use rayon::prelude::*;
    let rows: Vec<Vec<f64>> = pa
        .par_iter()
        .map(|p_n| {
            let mut field = Matrix::zeros(nao_b, nao_b);
            for lam in 0..nao_b {
                for sg in 0..nao_b {
                    let mut v = 0.0;
                    for mu in 0..nao_a {
                        for nu in 0..nao_a {
                            let pv = p_n[(mu, nu)];
                            if pv != 0.0 {
                                v += pv * eri.get(mu, nu, lam + nao_a, sg + nao_a);
                            }
                        }
                    }
                    field[(lam, sg)] = v;
                }
            }
            pb.iter().map(|p_m| p_m.frobenius_dot(&field)).collect()
        })
        .collect();
    Matrix::from_row_major(pa.len(), pb.len(), rows.concat())
}

/// Inverse of a symmetric positive definite matrix by its eigenvectors.
fn inverse(m: &Matrix) -> Result<Matrix> {
    let (values, vectors) = symmetric_eigen(m)?;
    if values.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
        return Err(crate::DftError::InvalidInput(
            "the orbital Hessian of a monomer is not positive definite: its dynamic response \
             (and the Casimir-Polder dispersion) is undefined"
                .into(),
        ));
    }
    let mut scaled = vectors.clone();
    for (k, v) in values.iter().enumerate() {
        for i in 0..scaled.rows {
            scaled[(i, k)] /= v;
        }
    }
    Ok(scaled.matmul_transpose_seq(&vectors))
}

/// Response of one monomer along the imaginary frequency axis.
pub(crate) enum Response {
    /// `(A + B, (A - B)^-1)`.
    Coupled(Matrix, Matrix),
    /// Orbital-energy differences of the uncoupled response.
    Uncoupled(Vec<f64>),
}

impl Response {
    pub(crate) fn coupled(apb: &Matrix, amb: &Matrix) -> Result<Self> {
        Ok(Response::Coupled(apb.clone(), inverse(amb)?))
    }

    fn at(&self, u: f64) -> Result<Matrix> {
        match self {
            Response::Coupled(apb, amb_inverse) => {
                let mut m = apb.clone();
                for (x, y) in m.as_mut_slice().iter_mut().zip(amb_inverse.as_slice()) {
                    *x += u * u * y;
                }
                inverse(&m)
            }
            Response::Uncoupled(gaps) => {
                let n = gaps.len();
                let mut m = Matrix::zeros(n, n);
                for (i, d) in gaps.iter().enumerate() {
                    m[(i, i)] = d / (d * d + u * u);
                }
                Ok(m)
            }
        }
    }
}

/// `-f (2/pi) int_0^inf du Tr[R_A(u) W R_B(u) W^T]` with `points` and with
/// `2 points` Gauss-Legendre points; returns the finer value and the change
/// between the two.
pub(crate) fn casimir_polder(
    a: &Response,
    b: &Response,
    w: &Matrix,
    spin_factor: f64,
    points: usize,
) -> Result<(f64, f64)> {
    let integral = |n: usize| -> Result<f64> {
        let (nodes, weights) = crate::grid::gauss_legendre(n);
        let u0 = 0.5;
        let mut sum = 0.0;
        for (&s, &weight) in nodes.iter().zip(&weights) {
            let u = u0 * (1.0 + s) / (1.0 - s);
            let jacobian = 2.0 * u0 / ((1.0 - s) * (1.0 - s));
            let ra = a.at(u)?;
            let rb = b.at(u)?;
            let left = ra.matmul(w);
            let right = rb.matmul_transpose_seq(w);
            // Tr[left right] with left (dA x dB) and right (dB x dA).
            let mut trace = 0.0;
            for p in 0..left.rows {
                for q in 0..left.cols {
                    trace += left[(p, q)] * right[(q, p)];
                }
            }
            sum += weight * jacobian * trace;
        }
        Ok(-spin_factor * 2.0 / std::f64::consts::PI * sum)
    };
    let coarse = integral(points)?;
    let fine = integral(2 * points)?;
    Ok((fine, (fine - coarse).abs()))
}
