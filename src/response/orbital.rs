// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Relaxed orbital eigenvalue derivatives with the moving overlap metric.
//! A degenerate eigenspace carries a projected derivative operator. Its
//! diagonal in one arbitrary MO frame is not a unique orbital gradient.

use super::nuclear::nuclear_taylor;
use super::{ResponseReport, ResponseSolverConfig};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct OrbitalGradientOptions {
    /// Cartesian IDs 3*atom+axis, in the requested order; None means all.
    pub coordinates: Option<Vec<usize>>,
    /// Maximum energy span of a grouped eigenspace, in Hartree.
    pub degeneracy_tolerance: f64,
}
impl Default for OrbitalGradientOptions {
    fn default() -> Self {
        Self {
            coordinates: None,
            degeneracy_tolerance: 1e-8,
        }
    }
}
impl OrbitalGradientOptions {
    pub fn validate(&self, n_atoms: usize) -> Result<Vec<usize>> {
        let coordinates = self
            .coordinates
            .clone()
            .unwrap_or_else(|| (0..3 * n_atoms).collect());
        let mut sorted = coordinates.clone();
        sorted.sort_unstable();
        if !self.degeneracy_tolerance.is_finite()
            || self.degeneracy_tolerance < 0.0
            || coordinates.is_empty()
            || sorted.iter().any(|&i| i >= 3 * n_atoms)
            || sorted.windows(2).any(|w| w[0] == w[1])
        {
            return Err(DftError::InvalidInput(
                "orbital gradients require unique nonempty Cartesian IDs 3*atom+axis and \
                a finite non-negative degeneracy tolerance"
                    .into(),
            ));
        }
        Ok(coordinates)
    }
}

#[derive(Clone, Debug)]
pub struct OrbitalCluster {
    /// 0: alpha, 1: beta. Orbital IDs are zero-based within that spin.
    pub spin: usize,
    pub orbitals: Vec<usize>,
    pub energies: Vec<f64>,
    pub energy_spread: f64,
    /// One unsymmetrized projected d(B^T F B)/dR matrix per coordinate.
    pub derivative_matrices: Vec<Matrix>,
    /// Largest entry of [D_a,D_b] for distinct requested coordinates.
    pub commutator_defect: f64,
}
impl OrbitalCluster {
    fn directional_matrix(&self, weights: &[f64]) -> Result<Matrix> {
        if weights.len() != self.derivative_matrices.len() || weights.iter().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "orbital cluster direction must have one finite weight per requested coordinate"
                    .into(),
            ));
        }
        let mut out = Matrix::zeros(self.orbitals.len(), self.orbitals.len());
        for (weight, matrix) in weights.iter().zip(&self.derivative_matrices) {
            super::nuclear::add(&mut out, matrix, *weight);
        }
        Ok(out)
    }
    /// Eigenvalues of sum_a weights[a] D_a. These are directional slopes
    /// at exact degeneracy; at a nonzero energy spread they describe the
    /// degenerate limiting model, not the individual eigenvalue derivatives.
    /// Sorting separate directions does not define multivariate branches.
    pub fn directional_slopes(&self, weights: &[f64]) -> Result<Vec<f64>> {
        Ok(symmetric_eigen(&self.directional_matrix(weights)?)?.0)
    }
    /// Spectrum of diag(reference energies) + step * sum_a weights[a] D_a.
    /// This first-order projected Hamiltonian retains any small reference
    /// splitting rather than replacing near-degenerate energies by a mean.
    pub fn linearized_energies(&self, weights: &[f64], step: f64) -> Result<Vec<f64>> {
        if !step.is_finite() {
            return Err(DftError::InvalidInput(
                "orbital cluster step must be finite".into(),
            ));
        }
        let mut matrix = self.directional_matrix(weights)?;
        for v in matrix.as_mut_slice() {
            *v *= step;
        }
        for (i, energy) in self.energies.iter().enumerate() {
            matrix[(i, i)] += energy;
        }
        Ok(symmetric_eigen(&matrix)?.0)
    }
}

#[derive(Clone, Debug)]
pub struct OrbitalEnergyGradients {
    pub coordinates: Vec<usize>,
    pub orbital_energies: [Vec<f64>; 2],
    pub occupations: [usize; 2],
    /// [spin][orbital][requested coordinate], Hartree/Bohr. None for every
    /// member of a grouped eigenspace; use the corresponding cluster instead.
    pub gradients: [Vec<Option<Vec<f64>>>; 2],
    pub clusters: Vec<OrbitalCluster>,
    pub degeneracy_tolerance: f64,
    /// Measured for the complete Cartesian tensor at zero reference field.
    /// Includes individual gradients and full projected cluster matrices.
    pub translation_residual: Option<f64>,
    pub hermiticity_defects: Vec<f64>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub metric_defects: Vec<f64>,
    pub particle_number_coefficients: Vec<[f64; 2]>,
    pub reference_energy_errors: Vec<f64>,
    pub reference_fock_errors: Vec<f64>,
}

fn groups(energies: &[f64], tolerance: f64) -> Vec<Vec<usize>> {
    // Limit the total span, not each adjacent gap: avoid chained clusters
    // that would silently group well-separated endpoints.
    let mut groups = Vec::new();
    let mut start = 0;
    while start < energies.len() {
        let mut end = start + 1;
        while end < energies.len() && energies[end] - energies[start] <= tolerance {
            end += 1;
        }
        if end - start > 1 {
            groups.push((start..end).collect());
        }
        start = end;
    }
    groups
}

fn assemble(
    scf: &ScfResult,
    coordinates: Vec<usize>,
    derivatives: &[[Matrix; 2]],
    tolerance: f64,
) -> OrbitalEnergyGradients {
    let mut gradients: [Vec<Option<Vec<f64>>>; 2] = std::array::from_fn(|s| {
        (0..scf.mo_energy[s].len())
            .map(|i| Some(derivatives.iter().map(|d| d[s][(i, i)]).collect()))
            .collect()
    });
    let mut clusters = Vec::new();
    for spin in 0..2 {
        for orbitals in groups(&scf.mo_energy[spin], tolerance) {
            for &i in &orbitals {
                gradients[spin][i] = None;
            }
            let energies: Vec<_> = orbitals.iter().map(|&i| scf.mo_energy[spin][i]).collect();
            let derivative_matrices: Vec<_> = derivatives
                .iter()
                .map(|d| {
                    let mut out = Matrix::zeros(orbitals.len(), orbitals.len());
                    for (i, &p) in orbitals.iter().enumerate() {
                        for (j, &q) in orbitals.iter().enumerate() {
                            out[(i, j)] = d[spin][(p, q)];
                        }
                    }
                    out
                })
                .collect();
            let mut commutator_defect = 0.0_f64;
            for a in 0..derivative_matrices.len() {
                for b in 0..a {
                    let ab = derivative_matrices[a].matmul_seq(&derivative_matrices[b]);
                    let ba = derivative_matrices[b].matmul_seq(&derivative_matrices[a]);
                    commutator_defect = commutator_defect.max(ab.max_abs_diff(&ba));
                }
            }
            clusters.push(OrbitalCluster {
                spin,
                orbitals,
                energy_spread: energies[energies.len() - 1] - energies[0],
                energies,
                derivative_matrices,
                commutator_defect,
            });
        }
    }
    OrbitalEnergyGradients {
        coordinates,
        orbital_energies: scf.mo_energy.clone(),
        occupations: [scf.n_alpha, scf.n_beta],
        gradients,
        clusters,
        degeneracy_tolerance: tolerance,
        translation_residual: None,
        hermiticity_defects: Vec::new(),
        projector_defects: Vec::new(),
        stationarity_defects: Vec::new(),
        metric_defects: Vec::new(),
        particle_number_coefficients: Vec::new(),
        reference_energy_errors: Vec::new(),
        reference_fock_errors: Vec::new(),
    }
}

/// Analytic relaxed eigenvalue gradients, including Pulay, integral, grid
/// and self-consistent density response. There are no displaced SCFs.
#[allow(clippy::too_many_arguments)]
pub fn orbital_energy_gradients(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    options: &OrbitalGradientOptions,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<OrbitalEnergyGradients>> {
    config.validate()?;
    let coordinates = options.validate(mol.len())?;
    let mut derivatives = Vec::new();
    let mut reports = Vec::new();
    for &coordinate in &coordinates {
        let mut direction = vec![[0.0; 3]; mol.len()];
        direction[coordinate / 3][coordinate % 3] = 1.0;
        let report = nuclear_taylor(mol, mb, functional, grid, scf, opts, &direction, 1, config)?;
        derivatives.push(report.value.orthonormal_fock_coefficients[1].clone());
        reports.push(report);
    }
    let mut value = assemble(scf, coordinates, &derivatives, options.degeneracy_tolerance);
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    for (report, d) in reports.iter().zip(&derivatives) {
        converged &= report.converged;
        residuals.extend_from_slice(&report.residuals);
        iterations.extend_from_slice(&report.iterations);
        for warning in &report.warnings {
            if !warnings.contains(warning) {
                warnings.push(warning.clone());
            }
        }
        value.hermiticity_defects.push(
            (0..2)
                .map(|s| d[s].max_abs_diff(&d[s].transpose()))
                .fold(0.0, f64::max),
        );
        value
            .projector_defects
            .push(report.value.projector_defects[0]);
        value
            .stationarity_defects
            .push(report.value.stationarity_defects[0]);
        value.metric_defects.push(report.value.metric_defects[0]);
        value
            .particle_number_coefficients
            .push(report.value.particle_number_coefficients[0]);
        value
            .reference_energy_errors
            .push(report.value.reference_energy_error);
        value
            .reference_fock_errors
            .push(report.value.reference_fock_error);
    }
    let acceptance = config
        .tolerance
        .max(reports[0].reference.orbital_residual)
        .max(reports[0].reference.eigen_residual)
        .max(1e-14);
    if value.hermiticity_defects.iter().any(|&v| v > acceptance) {
        converged = false;
        warnings.push(
            "projected orbital derivative Hermiticity exceeds the reference/response tolerance"
                .into(),
        );
    }
    if !value.clusters.is_empty() {
        warnings.push(
            "grouped orbital eigenvalues return projected derivative matrices and \
            absent individual gradients; directional slopes require exact \
            degeneracy, and near-degenerate spectra retain the reported reference \
            splitting"
                .into(),
        );
    }
    let zero_field = opts
        .external_field
        .is_none_or(|v| v.x == 0.0 && v.y == 0.0 && v.z == 0.0);
    if value.coordinates.len() == 3 * mol.len() && zero_field {
        let mut defect = 0.0_f64;
        for spin in 0..2 {
            for orbital in 0..scf.mo_energy[spin].len() {
                if let Some(g) = &value.gradients[spin][orbital] {
                    for axis in 0..3 {
                        let sum: f64 = value
                            .coordinates
                            .iter()
                            .zip(g)
                            .filter(|(c, _)| **c % 3 == axis)
                            .map(|(_, v)| v)
                            .sum();
                        defect = defect.max(sum.abs());
                    }
                }
            }
        }
        for cluster in &value.clusters {
            for axis in 0..3 {
                let mut sum = Matrix::zeros(cluster.orbitals.len(), cluster.orbitals.len());
                for (&coordinate, matrix) in
                    value.coordinates.iter().zip(&cluster.derivative_matrices)
                {
                    if coordinate % 3 == axis {
                        super::nuclear::add(&mut sum, matrix, 1.0);
                    }
                }
                defect = defect.max(sum.as_slice().iter().fold(0.0_f64, |a, v| a.max(v.abs())));
            }
        }
        value.translation_residual = Some(defect);
    }
    Ok(ResponseReport {
        shape: vec![2, scf.mo_energy[0].len(), value.coordinates.len()],
        value,
        functional: functional.description.clone(),
        reference: reports[0].reference.clone(),
        units: "Hartree/Bohr; orbital energies and degeneracy tolerance in Hartree",
        model: "relaxed static all-electron integer-occupation collinear KS/HF \
            eigenvalue response; moving Gaussian metric and Becke grid; adiabatic \
            semilocal XC",
        axis_convention: "[spin alpha,beta][orbital][requested Cartesian coordinate 3*atom+axis]; \
            degenerate subspaces return full projected matrices in the reference MO \
            frame",
        converged,
        residuals,
        iterations,
        warnings,
    })
}

#[derive(Clone, Debug)]
pub struct FrontierGap {
    pub spin: usize,
    pub homo: usize,
    pub lumo: usize,
    pub gap: f64,
    /// None when either frontier level belongs to a grouped eigenspace.
    pub gradient: Option<Vec<f64>>,
}
#[derive(Clone, Debug)]
pub struct FrontierGapGradient {
    pub gaps: [Option<FrontierGap>; 2],
    pub orbitals: OrbitalEnergyGradients,
}

/// Per-spin neutral-reference HOMO/LUMO orbital gaps, not charge-addition
/// fundamental gaps. Empty occupied or virtual spaces yield an absent gap.
#[allow(clippy::too_many_arguments)]
pub fn frontier_gap_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    options: &OrbitalGradientOptions,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<FrontierGapGradient>> {
    let report = orbital_energy_gradients(mol, mb, functional, grid, scf, opts, options, config)?;
    let gaps = std::array::from_fn(|spin| {
        let lumo = report.value.occupations[spin];
        if lumo == 0 || lumo >= scf.mo_energy[spin].len() {
            return None;
        }
        let homo = lumo - 1;
        let gradient = report.value.gradients[spin][homo]
            .as_ref()
            .zip(report.value.gradients[spin][lumo].as_ref())
            .map(|(h, l)| l.iter().zip(h).map(|(l, h)| l - h).collect());
        Some(FrontierGap {
            spin,
            homo,
            lumo,
            gap: scf.mo_energy[spin][lumo] - scf.mo_energy[spin][homo],
            gradient,
        })
    });
    Ok(ResponseReport {
        value: FrontierGapGradient {
            gaps,
            orbitals: report.value,
        },
        shape: vec![2, options.validate(mol.len())?.len()],
        functional: report.functional,
        reference: report.reference,
        units: report.units,
        model: report.model,
        axis_convention: "[spin alpha,beta][requested Cartesian coordinate]; gap = \
            neutral-reference LUMO - HOMO; absent derivative at grouped frontier \
            eigenvalues",
        converged: report.converged,
        residuals: report.residuals,
        iterations: report.iterations,
        warnings: report.warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn degenerate_operator_retains_directional_spectra_and_near_splitting() {
        let mut x = Matrix::zeros(2, 2);
        x[(0, 1)] = 1.0;
        x[(1, 0)] = 1.0;
        let mut z = Matrix::zeros(2, 2);
        z[(0, 0)] = 1.0;
        z[(1, 1)] = -1.0;
        let cluster = OrbitalCluster {
            spin: 0,
            orbitals: vec![1, 2],
            energies: vec![0.4; 2],
            energy_spread: 0.0,
            derivative_matrices: vec![x, z],
            commutator_defect: 2.0,
        };
        let slopes = cluster.directional_slopes(&[0.6, 0.8]).unwrap();
        assert!((slopes[0] + 1.0).abs() < 1e-14 && (slopes[1] - 1.0).abs() < 1e-14);
        // Summing separately sorted axis slopes would falsely give +/-1.4.
        let linear = cluster.linearized_energies(&[0.6, 0.8], 0.02).unwrap();
        assert!((linear[0] - 0.38).abs() < 1e-14 && (linear[1] - 0.42).abs() < 1e-14);
        let mut near = cluster.clone();
        near.energies[1] += 0.01;
        near.energy_spread = 0.01;
        let near_linear = near.linearized_energies(&[0.6, 0.8], 0.0).unwrap();
        assert!((near_linear[0] - 0.4).abs() < 1e-14 && (near_linear[1] - 0.41).abs() < 1e-14);
        assert!(cluster.directional_slopes(&[1.0]).is_err());
        assert!(cluster.linearized_energies(&[1.0, 0.0], f64::NAN).is_err());
        assert_eq!(groups(&[0.0, 0.9, 1.8], 1.0), vec![vec![0, 1]]);
    }
}
