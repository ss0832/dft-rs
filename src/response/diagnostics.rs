// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Residuals recomputed from the actual reference Fock, overlap, orbitals
//! and density, independently of the SCF stopping flag.

use crate::linalg::Matrix;
use crate::scf::ScfResult;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct ReferenceDiagnostics {
    pub converged: bool,
    pub iterations: usize,
    /// max |[C†FC, C†SP SC]|, Hartree.
    pub orbital_residual: f64,
    /// Relative Frobenius residual of FC - SC epsilon.
    pub eigen_residual: f64,
    pub orthonormality_defect: f64,
    pub projector_defect: f64,
    pub particle_number_error: f64,
}
fn max_abs(m: &Matrix) -> f64 {
    m.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}
pub fn reference_diagnostics(scf: &ScfResult) -> Result<ReferenceDiagnostics> {
    if scf.electronic_temperature > 0.0 || !scf.has_integer_occupations() {
        return Err(DftError::Unsupported {
            feature: "integer-occupation orbital response of a fractional molecular reference"
                .into(),
            missing: "this route requires an integer projector; fractional responses need \
                      all population-weighted orbital rotations and, for varying \
                      populations, their explicit response"
                .into(),
        });
    }
    density_diagnostics(scf)
}

/// Actual electronic residuals without imposing an integer-projector model.
/// A fractional reference must validate its own population constraint; the
/// returned projector defect remains the physical non-idempotency measure.
pub(crate) fn density_diagnostics(scf: &ScfResult) -> Result<ReferenceDiagnostics> {
    let nao = scf.overlap.rows;
    if scf.overlap.cols != nao || scf.overlap.as_slice().iter().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput("invalid reference overlap".into()));
    }
    let mut result = ReferenceDiagnostics {
        converged: scf.converged,
        iterations: scf.iterations,
        orbital_residual: 0.0,
        eigen_residual: 0.0,
        orthonormality_defect: 0.0,
        projector_defect: 0.0,
        particle_number_error: 0.0,
    };
    for s in 0..2 {
        let c = &scf.mo_coeff[s];
        let p = &scf.density[s];
        let f = &scf.fock[s];
        if c.rows != nao
            || c.cols != scf.mo_energy[s].len()
            || [scf.n_alpha, scf.n_beta][s] > c.cols
            || [p, f].iter().any(|m| m.rows != nao || m.cols != nao)
            || [c, p, f]
                .iter()
                .any(|m| m.as_slice().iter().any(|x| !x.is_finite()))
            || scf.mo_energy[s].iter().any(|e| !e.is_finite())
        {
            return Err(DftError::InvalidInput(
                "invalid reference electronic matrices".into(),
            ));
        }
        let sc = scf.overlap.matmul_seq(c);
        let fc = f.matmul_seq(c);
        let mut eigen = fc.clone();
        for i in 0..nao {
            for j in 0..c.cols {
                eigen[(i, j)] -= sc[(i, j)] * scf.mo_energy[s][j];
            }
        }
        let norm_fc = fc.as_slice().iter().fold(0.0_f64, |m, v| m.hypot(*v));
        let norm_eigen = eigen.as_slice().iter().fold(0.0_f64, |m, v| m.hypot(*v));
        result.eigen_residual = result.eigen_residual.max(if norm_fc == 0.0 {
            norm_eigen
        } else {
            norm_eigen / norm_fc
        });
        let metric = c.transpose().matmul_seq(&sc);
        result.orthonormality_defect = result
            .orthonormality_defect
            .max(metric.max_abs_diff(&Matrix::identity(c.cols)));
        let d = sc.transpose().matmul_seq(p).matmul_seq(&sc);
        let fmo = c.transpose().matmul_seq(&fc);
        let mut comm = fmo.matmul_seq(&d);
        let df = d.matmul_seq(&fmo);
        for (v, &other) in comm.as_mut_slice().iter_mut().zip(df.as_slice()) {
            *v -= other;
        }
        result.orbital_residual = result.orbital_residual.max(max_abs(&comm));
        let mut projector = d.matmul_seq(&d);
        for (v, &other) in projector.as_mut_slice().iter_mut().zip(d.as_slice()) {
            *v -= other;
        }
        result.projector_defect = result.projector_defect.max(max_abs(&projector));
        let number = p.frobenius_dot(&scf.overlap);
        let expected = if s == 0 { scf.n_alpha } else { scf.n_beta };
        result.particle_number_error += (number - expected as f64).abs();
    }
    Ok(result)
}

/// Maximum within-occupied/within-virtual C^T F C - epsilon defect.
/// Call after reference_diagnostics has checked the electronic matrices.
/// Common-orbital stationary states need not have zero spin F_vo.
pub(crate) fn semicanonical_residual(scf: &ScfResult) -> f64 {
    let mut residual = 0.0_f64;
    for spin in 0..2 {
        let c = &scf.mo_coeff[spin];
        let f = c.transpose().matmul_seq(&scf.fock[spin]).matmul_seq(c);
        let occupied = [scf.n_alpha, scf.n_beta][spin];
        for i in 0..c.cols {
            for j in 0..c.cols {
                if (i < occupied) == (j < occupied) {
                    let expected = if i == j { scf.mo_energy[spin][i] } else { 0.0 };
                    residual = residual.max((f[(i, j)] - expected).abs());
                }
            }
        }
    }
    residual
}
