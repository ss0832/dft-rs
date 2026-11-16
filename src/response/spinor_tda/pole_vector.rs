// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Normalized isolated eigenvector jets for pole and residue differentiation.
use super::*;

/// One normalized isolated eigenpair path and fresh coefficient diagnostics.
#[derive(Clone, Debug)]
pub struct SpinorPoleTaylor {
    pub energies: Vec<f64>,
    /// Factorial-normalized column vectors, in a fixed-base overlap gauge.
    pub vectors: Vec<BlochMatrix>,
    pub equation_residuals: Vec<f64>,
    pub normalization_defects: Vec<f64>,
    pub nearest_gap: f64,
}

fn axpy(out: &mut BlochMatrix, re: f64, im: f64, x: &BlochMatrix) {
    for i in 0..out.rows() {
        out.re[(i, 0)] += re * x.re[(i, 0)] - im * x.im[(i, 0)];
        out.im[(i, 0)] += re * x.im[(i, 0)] + im * x.re[(i, 0)];
    }
}

/// Isolated Hermitian eigenpair coefficients through order two. The phase
/// is fixed by a positive real overlap with the base vector; normalization
/// is enforced through each order. Degenerate subspaces are refused.
pub fn spinor_tda_eigenpair_taylor(
    operator: &[BlochMatrix],
    root: usize,
    config: &SpinorTdaConfig,
) -> Result<SpinorPoleTaylor> {
    let spectral = prepare_pole(operator, root, config)?;
    let energies = pole_energy_coefficients(operator, root, &spectral, config)?;
    if let Some(order) = energies.iter().position(|v| !v.is_finite()) {
        return Err(DftError::ResponseNotConverged {
            iterations: order,
            residual: f64::INFINITY,
        });
    }
    let dim = operator[0].rows();
    let base = columns(&spectral.coefficients, root, 1);
    let first_action = matmul(&operator[1], &base)?;
    let mut first = BlochMatrix::zeros(dim, 1);
    let mut nearest_gap = f64::INFINITY;
    for (k, &other_energy) in spectral.energies.iter().enumerate() {
        if k == root {
            continue;
        }
        let denominator = energies[0] - other_energy;
        nearest_gap = nearest_gap.min(denominator.abs());
        let other = columns(&spectral.coefficients, k, 1);
        let coupling = matmul(&adjoint(&other)?, &first_action)?;
        axpy(
            &mut first,
            coupling.re[(0, 0)] / denominator,
            coupling.im[(0, 0)] / denominator,
            &other,
        );
    }
    if first
        .re
        .as_slice()
        .iter()
        .chain(first.im.as_slice())
        .any(|v| !v.is_finite())
    {
        return Err(DftError::ResponseNotConverged {
            iterations: 1,
            residual: f64::INFINITY,
        });
    }
    let mut vectors = vec![base, first];
    if operator.len() == 3 {
        let mut action = matmul(&operator[2], &vectors[0])?;
        axpy(&mut action, 1.0, 0.0, &matmul(&operator[1], &vectors[1])?);
        axpy(&mut action, -energies[1], 0.0, &vectors[1]);
        let mut second = BlochMatrix::zeros(dim, 1);
        for (k, &other_energy) in spectral.energies.iter().enumerate() {
            if k == root {
                continue;
            }
            let other = columns(&spectral.coefficients, k, 1);
            let coupling = matmul(&adjoint(&other)?, &action)?;
            let denominator = energies[0] - other_energy;
            axpy(
                &mut second,
                coupling.re[(0, 0)] / denominator,
                coupling.im[(0, 0)] / denominator,
                &other,
            );
        }
        let norm = matmul(&adjoint(&vectors[1])?, &vectors[1])?.re[(0, 0)];
        axpy(&mut second, -0.5 * norm, 0.0, &vectors[0]);
        vectors.push(second);
    }
    let mut equation_residuals = Vec::new();
    let mut normalization_defects = Vec::new();
    for k in 0..operator.len() {
        let mut residual = BlochMatrix::zeros(dim, 1);
        let (mut norm_re, mut norm_im) = (0.0, 0.0);
        for m in 0..=k {
            axpy(
                &mut residual,
                1.0,
                0.0,
                &matmul(&operator[m], &vectors[k - m])?,
            );
            axpy(&mut residual, -energies[m], 0.0, &vectors[k - m]);
            let overlap = matmul(&adjoint(&vectors[m])?, &vectors[k - m])?;
            norm_re += overlap.re[(0, 0)];
            norm_im += overlap.im[(0, 0)];
        }
        if k == 0 {
            norm_re -= 1.0;
        }
        let norm = norm_re.hypot(norm_im);
        let error = (0..dim).fold(0.0_f64, |v, i| {
            v.hypot(residual.re[(i, 0)]).hypot(residual.im[(i, 0)])
        });
        let defect = if error.is_finite() && norm.is_finite() {
            error.max(norm)
        } else {
            f64::INFINITY
        };
        if defect > config.tolerance {
            return Err(DftError::ResponseNotConverged {
                iterations: k,
                residual: defect,
            });
        }
        equation_residuals.push(error);
        normalization_defects.push(norm);
    }
    Ok(SpinorPoleTaylor {
        energies,
        vectors,
        equation_residuals,
        normalization_defects,
        nearest_gap,
    })
}
