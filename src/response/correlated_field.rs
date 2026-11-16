// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Noncanonical PT2 amplitude/orbital response of an analytic complex field
//! Hamiltonian. The parent, integral transform and amplitude equations are
//! shared with molecular nuclear and electric response.
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::london::{repulsion_series, ComplexMatrix, FieldPath};
use crate::properties::pt2::Pt2Problem;
use crate::properties::pt2_response::{complex_taylor_response, complex_transform_path};
use crate::response::complex::Complex;
use crate::response::field_path::{
    field_wavefunction_path_for_model, FieldEnergyModel, FieldWavefunctionPath,
};
use crate::response::ResponseSolverConfig;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Factorial-normalized coefficients; the three spin channels are unscaled.
pub struct CorrelatedFieldTaylor {
    pub energy: Vec<f64>,
    pub parent: FieldWavefunctionPath,
    pub same_spin_alpha: Vec<f64>,
    pub opposite_spin: Vec<f64>,
    pub same_spin_beta: Vec<f64>,
    pub correlation: Vec<f64>,
    pub amplitudes: Vec<[Vec<Complex>; 3]>,
    pub amplitude_residuals: Vec<f64>,
    /// Fresh physical amplitude-equation residuals normalized by the source.
    pub amplitude_relative_residuals: Vec<f64>,
    pub imaginary_energy_defect: f64,
}

pub(crate) const ENERGY_CHANNELS: usize = 9;
impl CorrelatedFieldTaylor {
    /// Shared physical channel order for molecular energy derivatives.
    pub(crate) fn energy_channels(&self, order: usize) -> [f64; ENERGY_CHANNELS] {
        [
            self.energy[order],
            self.parent.parent.energy[order],
            self.parent.target_energy[order],
            self.correlation[order],
            self.same_spin_alpha[order],
            self.opposite_spin[order],
            self.same_spin_beta[order],
            self.parent.parent_nonlocal_energy[order],
            self.parent.target_nonlocal_energy[order],
        ]
    }
}

/// Full complex parent response, every London AO integral endpoint and
/// complete noncanonical amplitudes. All supplied core coefficients are
/// factorial normalized in the same real path parameter. No 2n+1
/// truncation of the parent density is used for post-SCF correlation.
#[allow(clippy::too_many_arguments)]
pub fn correlated_field_taylor(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<&ResolvedFunctional>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    path: &FieldPath,
    core_coefficients: Option<&[[ComplexMatrix; 2]]>,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedFieldTaylor> {
    correlated_field_taylor_for_model(
        molecule,
        basis,
        parent,
        target.map(FieldEnergyModel::new),
        grid,
        reference,
        options,
        coefficients,
        path,
        core_coefficients,
        memory_mb,
        config,
    )
}

/// Explicitly composed target model on the same parent and response equations.
#[allow(clippy::too_many_arguments)]
pub fn correlated_field_taylor_for_model(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    path: &FieldPath,
    core_coefficients: Option<&[[ComplexMatrix; 2]]>,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedFieldTaylor> {
    coefficients.validate()?;
    let width = path
        .order
        .checked_add(1)
        .ok_or_else(|| DftError::InvalidInput("complex PT2 path order overflows".into()))?;
    let bytes = basis
        .nao_sph
        .checked_pow(4)
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| n.checked_mul(16 * 8))
        .ok_or_else(|| {
            DftError::InvalidInput("complex PT2 coefficient workspace overflows".into())
        })?;
    let required = bytes.div_ceil(1024 * 1024);
    if required > memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "complex PT2 field coefficient tensors",
            required_mb: required,
            limit_mb: memory_mb,
        });
    }
    let data = field_wavefunction_path_for_model(
        molecule,
        basis,
        parent,
        target,
        grid,
        reference,
        options,
        path,
        core_coefficients,
        config,
    )?;
    let mut integrals: Vec<Vec<Complex>> = repulsion_series(basis, path, None)?
        .into_iter()
        .map(|(real, imaginary)| {
            real.into_iter()
                .zip(imaginary)
                .map(|(re, im)| Complex::new(re, im))
                .collect()
        })
        .collect();
    integrals.resize_with(width, || vec![Complex::ZERO; basis.nao_sph.pow(4)]);
    let transformed = complex_transform_path(
        basis.nao_sph,
        &integrals,
        &data.orbitals,
        &reference.mo_energy,
        data.occupied,
    )?;
    let base = Pt2Problem {
        energies: reference.mo_energy.clone(),
        occupied: data.occupied,
        ovov: transformed[0]
            .ovov
            .each_ref()
            .map(|g| g.iter().map(|z| z.re).collect()),
    };
    let base_imaginary = transformed[0]
        .ovov
        .iter()
        .flatten()
        .map(|g| g.im.abs())
        .fold(0.0_f64, f64::max);
    if base_imaginary > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: base_imaginary,
        });
    }
    let fock: Vec<_> = (1..width)
        .map(|n| [data.mo_fock[0][n].clone(), data.mo_fock[1][n].clone()])
        .collect();
    let integrals: Vec<_> = transformed.into_iter().skip(1).map(|p| p.ovov).collect();
    let pairs = complex_taylor_response(&base, &integrals, &fock, coefficients)?;
    let imaginary_energy_defect = pairs
        .same_spin_alpha
        .iter()
        .chain(&pairs.same_spin_beta)
        .chain(&pairs.opposite_spin)
        .chain(&pairs.correlation)
        .map(|z| z.im.abs())
        .fold(base_imaginary, f64::max);
    if imaginary_energy_defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: path.order,
            residual: imaginary_energy_defect,
        });
    }
    let reals = |v: &[Complex]| v.iter().map(|z| z.re).collect::<Vec<_>>();
    let correlation = reals(&pairs.correlation);
    let energy = data
        .target_energy
        .iter()
        .zip(&correlation)
        .map(|(a, b)| a + b)
        .collect();
    Ok(CorrelatedFieldTaylor {
        energy,
        parent: data,
        same_spin_alpha: reals(&pairs.same_spin_alpha),
        opposite_spin: reals(&pairs.opposite_spin),
        same_spin_beta: reals(&pairs.same_spin_beta),
        correlation,
        amplitudes: pairs.amplitude_coefficients,
        amplitude_residuals: pairs.residuals,
        amplitude_relative_residuals: pairs.relative_residuals,
        imaginary_energy_defect,
    })
}
