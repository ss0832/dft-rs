// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete static beta and gamma tensors from analytic projector response.
//! Every ordered field tuple is solved independently. Raw components are never
//! filled by symmetry; interchange diagnostics include the observed dipole leg.

use super::{ResponseReport, ResponseSolverConfig};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

mod variational;
pub use variational::static_beta_variational;

pub(crate) const QUADRUPOLE_AXES: [(usize, usize); 6] =
    [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
pub(crate) const BUCKINGHAM_FACTOR: f64 = 3.0;

pub(crate) fn quadrupole_modes() -> [[[f64; 3]; 3]; 5] {
    let a = 1.0 / 2.0_f64.sqrt();
    let b = 1.0 / 6.0_f64.sqrt();
    [
        [[a, 0.0, 0.0], [0.0, -a, 0.0], [0.0, 0.0, 0.0]],
        [[b, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, -2.0 * b]],
        [[0.0, a, 0.0], [a, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, a], [0.0, 0.0, 0.0], [a, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, a], [0.0, a, 0.0]],
    ]
}

/// Coefficients of the six unique Cartesian monomials in G:rr/2.
pub(crate) fn quadrupole_weights(mode: [[f64; 3]; 3]) -> [f64; 6] {
    QUADRUPOLE_AXES.map(|(a, b)| if a == b { 0.5 * mode[a][b] } else { mode[a][b] })
}

#[derive(Clone, Debug)]
pub struct StaticElectricTensor {
    /// Number of field derivatives: 2 for beta, 3 for gamma.
    pub order: usize,
    /// Row-major [dipole axis][ordered field axes], actual derivatives.
    pub tensor: Vec<f64>,
    pub accumulation_roundoff: Vec<f64>,
    /// Field-leg interchange, measured on independently evaluated raw entries.
    pub intrinsic_permutation_defect: f64,
    pub intrinsic_permutation_comparisons: usize,
    /// Dipole/field-leg interchange implied by a differentiable static energy.
    pub energy_permutation_defect: f64,
    pub energy_permutation_comparisons: usize,
    pub frobenius_norm: f64,
    /// beta_i = sum_j (beta_ijj + beta_jij + beta_jji)/3.
    pub beta_vector: Option<[f64; 3]>,
    pub beta_vector_norm: Option<f64>,
    /// sum_ij (gamma_iijj + gamma_ijij + gamma_ijji)/15.
    pub gamma_isotropic: Option<f64>,
    pub ordered_field_evaluations: usize,
    pub directional_evaluations: usize,
    /// Maximum raw Taylor-coefficient defects over all polarization solves.
    pub projector_defect: f64,
    pub stationarity_defect: f64,
    pub particle_number_defect: f64,
    /// Whether idempotency is an active density constraint for this reference.
    pub projector_constraint: bool,
    pub fixed_population_defect: Option<f64>,
    pub fermi_density_defect: Option<f64>,
}

fn tuple(mut flat: usize, rank: usize) -> Vec<usize> {
    let mut out = vec![0; rank];
    for axis in out.iter_mut().rev() {
        *axis = flat % 3;
        flat /= 3;
    }
    out
}

fn index(axes: &[usize]) -> usize {
    axes.iter().fold(0, |n, &a| 3 * n + a)
}

#[allow(clippy::too_many_arguments)]
pub fn static_electric_tensor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<StaticElectricTensor>> {
    config.validate()?;
    if !(2..=3).contains(&order) {
        return Err(DftError::InvalidInput(
            "static hyperpolarizability requires order=2 (beta) or 3 (gamma)".into(),
        ));
    }
    let rank = order + 1;
    let size = 3usize.pow(rank as u32);
    let mut tensor = vec![0.0; size];
    let mut accumulation_roundoff = vec![0.0; size];
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    let mut directional_evaluations = 0;
    let mut projector_defect = 0.0_f64;
    let mut stationarity_defect = 0.0_f64;
    let mut particle_number_defect = 0.0_f64;
    let mut fixed_population_defect: Option<f64> = None;
    let mut fermi_density_defect: Option<f64> = None;
    let ordered_field_evaluations = 3usize.pow(order as u32);
    for flat in 0..ordered_field_evaluations {
        let axes = tuple(flat, order);
        let directions: Vec<_> = axes
            .iter()
            .map(|&axis| {
                let mut d = vec![0.0; 3];
                d[axis] = 1.0;
                d
            })
            .collect();
        let mixed = super::mixed::polarized_mixed(3, &directions, |d, n| {
            let report = super::electric_field_taylor(
                mol,
                mb,
                functional,
                grid,
                scf,
                opts,
                [d[0], d[1], d[2]],
                n,
                config,
            )?;
            converged &= report.converged;
            residuals.extend(report.residuals);
            iterations.extend(report.iterations);
            warnings.extend(report.warnings);
            if let Some(d) = &report.value.fixed_population {
                fixed_population_defect = Some(
                    fixed_population_defect
                        .unwrap_or(0.0)
                        .max(d.maximum_defect()),
                );
            }
            if let Some(d) = &report.value.thermal_population {
                let maximum = d.density_defects.iter().copied().fold(0.0_f64, f64::max);
                fermi_density_defect = Some(fermi_density_defect.unwrap_or(0.0).max(maximum));
            }
            for &defect in &report.value.projector_defects {
                projector_defect = projector_defect.max(defect);
            }
            for &defect in &report.value.stationarity_defects {
                stationarity_defect = stationarity_defect.max(defect);
            }
            for &defect in report.value.particle_number_derivatives.iter().flatten() {
                particle_number_defect = particle_number_defect.max(defect.abs());
            }
            Ok(report.value.dipole_coefficients[n].to_vec())
        })?;
        directional_evaluations += mixed.directional_evaluations;
        for a in 0..3 {
            let i = a * ordered_field_evaluations + flat;
            tensor[i] = mixed.value[a];
            accumulation_roundoff[i] = mixed.accumulation_roundoff[a];
        }
    }
    let mut report = complete_tensor(
        order,
        tensor,
        accumulation_roundoff,
        ordered_field_evaluations,
        directional_evaluations,
        projector_defect,
        stationarity_defect,
        particle_number_defect,
        fixed_population_defect,
        residuals,
        iterations,
        warnings,
        converged,
        functional,
        scf,
        config,
        if scf.electronic_temperature > 0.0 {
            "static clamped-nuclei canonical Fermi-Dirac Mermin response; analytic perturbations at fixed electronic temperature"
        } else if scf.fixed_occupation_residual.is_some() {
            "static clamped-nuclei prescribed-population KS/HF; analytic isospectral orbital response; adiabatic XC and connected nonlocal/linear-continuum kernels"
        } else {
            "static clamped-nuclei integer-occupation KS/HF analytic \
            density-projector response; adiabatic XC; self-consistent nonlocal and \
            linear solvent kernels when selected"
        },
    )?;
    report.value.fermi_density_defect = fermi_density_defect;
    Ok(report)
}

pub(crate) struct TensorContractions {
    pub intrinsic_permutation_defect: f64,
    pub energy_permutation_defect: f64,
    pub intrinsic_permutation_comparisons: usize,
    pub energy_permutation_comparisons: usize,
    pub beta_vector: Option<[f64; 3]>,
    pub beta_vector_norm: Option<f64>,
    pub gamma_isotropic: Option<f64>,
    pub frobenius_norm: f64,
}

/// Cartesian tensor contractions independent of a particular reference space.
pub(crate) fn tensor_contractions(order: usize, tensor: &[f64]) -> TensorContractions {
    let rank = order + 1;
    let mut intrinsic_permutation_defect = 0.0_f64;
    let mut energy_permutation_defect = 0.0_f64;
    let mut intrinsic_permutation_comparisons = 0;
    let mut energy_permutation_comparisons = 0;
    for (flat, &value) in tensor.iter().enumerate() {
        let axes = tuple(flat, rank);
        for left in 0..rank {
            for right in left + 1..rank {
                // Count each unequal-index pair once, before any averaging.
                if axes[left] >= axes[right] {
                    continue;
                }
                let mut other = axes.clone();
                other.swap(left, right);
                let defect = (value - tensor[index(&other)]).abs();
                if left == 0 {
                    energy_permutation_defect = energy_permutation_defect.max(defect);
                    energy_permutation_comparisons += 1;
                } else {
                    intrinsic_permutation_defect = intrinsic_permutation_defect.max(defect);
                    intrinsic_permutation_comparisons += 1;
                }
            }
        }
    }
    let beta_vector = (order == 2).then(|| {
        std::array::from_fn(|i| {
            (0..3)
                .map(|j| {
                    (tensor[index(&[i, j, j])]
                        + tensor[index(&[j, i, j])]
                        + tensor[index(&[j, j, i])])
                        / 3.0
                })
                .sum::<f64>()
        })
    });
    let beta_vector_norm = beta_vector.map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt());
    let gamma_isotropic = (order == 3).then(|| {
        (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| {
                tensor[index(&[i, i, j, j])]
                    + tensor[index(&[i, j, i, j])]
                    + tensor[index(&[i, j, j, i])]
            })
            .sum::<f64>()
            / 15.0
    });
    let frobenius_norm = tensor.iter().map(|x| x * x).sum::<f64>().sqrt();
    TensorContractions {
        intrinsic_permutation_defect,
        energy_permutation_defect,
        intrinsic_permutation_comparisons,
        energy_permutation_comparisons,
        beta_vector,
        beta_vector_norm,
        gamma_isotropic,
        frobenius_norm,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn complete_tensor(
    order: usize,
    tensor: Vec<f64>,
    accumulation_roundoff: Vec<f64>,
    ordered_field_evaluations: usize,
    directional_evaluations: usize,
    projector_defect: f64,
    stationarity_defect: f64,
    particle_number_defect: f64,
    fixed_population_defect: Option<f64>,
    residuals: Vec<f64>,
    iterations: Vec<usize>,
    mut warnings: Vec<String>,
    mut converged: bool,
    functional: &ResolvedFunctional,
    scf: &ScfResult,
    config: &ResponseSolverConfig,
    model: &'static str,
) -> Result<ResponseReport<StaticElectricTensor>> {
    let rank = order + 1;
    let TensorContractions {
        intrinsic_permutation_defect,
        energy_permutation_defect,
        intrinsic_permutation_comparisons,
        energy_permutation_comparisons,
        beta_vector,
        beta_vector_norm,
        gamma_isotropic,
        frobenius_norm,
    } = tensor_contractions(order, &tensor);
    let scale = tensor.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
    if intrinsic_permutation_defect.max(energy_permutation_defect) > config.tolerance * scale {
        converged = false;
        warnings.push(
            "raw static tensor violates the requested energy/field interchange tolerance".into(),
        );
    }
    if accumulation_roundoff
        .iter()
        .any(|&x| x > config.tolerance * scale)
    {
        warnings.push(
            "electric tensor polarization roundoff exceeds the requested scaled tolerance".into(),
        );
    }
    warnings.sort();
    warnings.dedup();
    Ok(ResponseReport {
        value: StaticElectricTensor {
            order,
            tensor,
            accumulation_roundoff,
            intrinsic_permutation_defect,
            intrinsic_permutation_comparisons,
            energy_permutation_defect,
            energy_permutation_comparisons,
            frobenius_norm,
            beta_vector,
            beta_vector_norm,
            gamma_isotropic,
            ordered_field_evaluations,
            directional_evaluations,
            projector_defect,
            stationarity_defect,
            particle_number_defect,
            projector_constraint: scf.fixed_occupation_residual.is_none()
                && scf.electronic_temperature == 0.0,
            fixed_population_defect,
            fermi_density_defect: None,
        },
        shape: vec![3; rank],
        functional: functional.description.clone(),
        reference: if scf.fixed_occupation_residual.is_some() || scf.electronic_temperature > 0.0 {
            super::diagnostics::density_diagnostics(scf)?
        } else {
            super::diagnostics::reference_diagnostics(scf)?
        },
        units: if order == 2 {
            "atomic units; e^3 Bohr^3/Hartree^2"
        } else {
            "atomic units; e^4 Bohr^4/Hartree^3"
        },
        model,
        axis_convention: "row-major [dipole Cartesian axis][ordered field Cartesian axes]; actual \
            derivatives without factorial scaling; raw components preserved",
        converged,
        residuals,
        iterations,
        warnings,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn static_beta(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<StaticElectricTensor>> {
    static_electric_tensor(mol, mb, functional, grid, scf, opts, 2, config)
}

#[allow(clippy::too_many_arguments)]
pub fn static_gamma(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<StaticElectricTensor>> {
    static_electric_tensor(mol, mb, functional, grid, scf, opts, 3, config)
}
