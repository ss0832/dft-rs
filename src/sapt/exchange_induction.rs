// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Occupied projectors and their first variations for the Heitler-London
//! exchange-quenching model.

use crate::linalg::{symmetric_eigen, Matrix};
use crate::{DftError, Result};

pub(super) struct OccupiedProjector {
    pub(super) coefficients: Matrix,
    dual: Matrix,
    pub(super) density: Matrix,
    pub(super) inverse_metric: Matrix,
}

impl OccupiedProjector {
    pub(super) fn new(coefficients: Matrix, overlap: &Matrix) -> Result<Self> {
        let count = coefficients.cols;
        let inverse_metric = if count == 0 {
            Matrix::zeros(0, 0)
        } else {
            let metric = coefficients.transpose_matmul(&overlap.matmul(&coefficients));
            let (values, vectors) = symmetric_eigen(&metric)?;
            if values.iter().any(|&v| !v.is_finite() || v <= 1e-12) {
                return Err(DftError::InvalidInput(
                    "SAPT: singular occupied-overlap metric (monomers too close?)".into(),
                ));
            }
            let mut scaled = vectors.clone();
            for i in 0..count {
                for j in 0..count {
                    scaled[(i, j)] /= values[j];
                }
            }
            scaled.matmul_transpose_seq(&vectors)
        };
        let dual = coefficients.matmul(&inverse_metric);
        let density = dual.matmul_transpose_seq(&coefficients);
        Ok(Self {
            coefficients,
            dual,
            density,
            inverse_metric,
        })
    }

    /// Recover the monomer orbital tangent dC = dP S_fragment C and
    /// project it out of the combined occupied space. Intermolecular
    /// overlap enters the projection after dC has been formed.
    pub(super) fn polarization_direction(
        &self,
        dp: &Matrix,
        monomer_overlap: &Matrix,
        overlap: &Matrix,
        offset: usize,
    ) -> Matrix {
        let mut local_coefficients = Matrix::zeros(dp.rows, self.coefficients.cols);
        for i in 0..local_coefficients.rows {
            for j in 0..local_coefficients.cols {
                local_coefficients[(i, j)] = self.coefficients[(offset + i, j)];
            }
        }
        let local = dp.matmul(&monomer_overlap.matmul(&local_coefficients));
        let mut dc = Matrix::zeros(self.coefficients.rows, self.coefficients.cols);
        for i in 0..local.rows {
            for j in 0..local.cols {
                dc[(offset + i, j)] = local[(i, j)];
            }
        }
        let counterterm = self.density.matmul(&overlap.matmul(&dc));
        let mut projected = dc;
        for (entry, correction) in projected
            .as_mut_slice()
            .iter_mut()
            .zip(counterterm.as_slice())
        {
            *entry -= correction;
        }
        projected
    }

    /// Differentiate D = C (C^T S C)^-1 C^T by contracting the projected
    /// orbital tangent times dual^T and its transpose with the Fock matrix.
    pub(super) fn energy_derivative(&self, projected: &Matrix, fock: &Matrix) -> f64 {
        2.0 * projected.frobenius_dot(&fock.matmul(&self.dual))
    }
}
