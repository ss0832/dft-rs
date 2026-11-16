// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Orbital spectra of the spin-summed and signed spin densities.

use super::NaturalAtomicOrbitals;
use crate::basis::MolBasis;
use crate::grid::eval::eval_aos;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct NaturalDensityOrbitals {
    /// Density eigenvalues in ascending order.
    pub weights: Vec<f64>,
    /// Spatial AO rows, with one natural orbital per column.
    pub ao_coefficients: Matrix,
    /// Maximum reconstruction defect in the orthonormal NAO basis.
    pub reconstruction_residual: f64,
    /// Difference between the spectral trace and the direct density trace.
    pub trace_residual: f64,
}

impl NaturalDensityOrbitals {
    /// Diagonalize the spin density in the orthonormal NAO frame and retain
    /// the complete signed spectrum, including paired positive/negative
    /// spin polarization and any vanishing eigenvalues.
    pub fn from_spin_density(nao: &NaturalAtomicOrbitals) -> Result<Self> {
        let mut magnetization = nao.spin_density_nao[0].clone();
        for (x, beta) in magnetization
            .as_mut_slice()
            .iter_mut()
            .zip(nao.spin_density_nao[1].as_slice())
        {
            *x -= beta;
        }
        Self::from_density(nao, &magnetization)
    }

    /// Diagonalize the spin-summed density (zero to two electrons per orbital).
    pub fn from_charge_density(nao: &NaturalAtomicOrbitals) -> Result<Self> {
        Self::from_density(nao, &nao.density_nao)
    }

    fn from_density(nao: &NaturalAtomicOrbitals, density: &Matrix) -> Result<Self> {
        let (weights, eigenvectors) = symmetric_eigen(density)?;
        let mut weighted = eigenvectors.clone();
        for (j, &weight) in weights.iter().enumerate() {
            for i in 0..weighted.rows {
                weighted[(i, j)] *= weight;
            }
        }
        let rebuilt = weighted.matmul_transpose_seq(&eigenvectors);
        let trace: f64 = (0..density.rows).map(|i| density[(i, i)]).sum();
        Ok(Self {
            trace_residual: weights.iter().sum::<f64>() - trace,
            reconstruction_residual: density.max_abs_diff(&rebuilt),
            weights,
            ao_coefficients: nao.nao_coefficients.matmul(&eigenvectors),
        })
    }

    /// Evaluate the weighted sum of orbital densities at the requested Bohr
    /// coordinates. Values are in electrons per cubic Bohr.
    pub fn values_at(&self, basis: &MolBasis, points: &[Vec3]) -> Result<Vec<f64>> {
        if basis.nao_sph != self.ao_coefficients.rows
            || self.weights.len() != self.ao_coefficients.cols
            || points
                .iter()
                .flat_map(|p| p.to_array())
                .any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "natural density evaluation needs the matching AO dimension and finite points"
                    .into(),
            ));
        }
        let amplitudes = eval_aos(basis, points, false)
            .phi
            .matmul(&self.ao_coefficients);
        Ok((0..points.len())
            .map(|i| {
                self.weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * amplitudes[(i, j)].powi(2))
                    .sum()
            })
            .collect())
    }
}

/// Head-Gordon nonlinear effectively-unpaired weights from spatial occupations.
/// The occupations are used as supplied, including any numerical bound defect.
pub fn effectively_unpaired_weights(occupations: &[f64]) -> Vec<f64> {
    occupations
        .iter()
        .map(|&n| (n * (2.0 - n)).powi(2))
        .collect()
}
