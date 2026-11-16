// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Stationary laboratory-frame quantum-mass derivatives at fixed Gaussian bases.
//! Charges, correlation parameters and any expectation-position targets stay fixed.
//! No center-of-mass subtraction or identical-particle statistics is introduced.

use super::correlation::CorrelationModel;
use super::response::{CoupledDensityResponse, CoupledResponseSolver};
use super::{MulticomponentResult, QuantumParticle};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::{dipole_matrices, kinetic_matrix, overlap_matrix};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::ResponseSolverConfig;
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Derivatives in particle order, with masses expressed in electron masses.
#[derive(Debug)]
pub struct MassResponse {
    /// Hartree per electron mass.
    pub gradient: Vec<f64>,
    /// All ordered relaxed entries, Hartree per squared electron mass.
    pub hessian: Matrix,
    pub raw_asymmetry: f64,
    /// One complete coupled density response per perturbed mass.
    pub responses: Vec<CoupledDensityResponse>,
    /// Signed reference particle numbers minus one.
    pub reference_number_defects: Vec<f64>,
    /// Signed particle-number responses, indexed [perturbed mass][particle].
    pub response_number_defects: Vec<Vec<f64>>,
    /// Reconstructed minus reported kinetic energies, Hartree.
    pub kinetic_energy_defects: Vec<f64>,
}

fn operators(particles: &[QuantumParticle], state: &MulticomponentResult) -> Result<Vec<Matrix>> {
    if !state.converged || particles.is_empty() || particles.len() != state.quantum.len() {
        return Err(DftError::InvalidInput(
            "mass derivatives require a matching converged coupled reference".into(),
        ));
    }
    particles
        .iter()
        .zip(&state.quantum)
        .map(|(p, q)| {
            if !p.mass.is_finite()
                || p.mass <= 0.0
                || p.basis.nao_sph == 0
                || p.basis.nao_sph != q.density.rows
                || q.density.rows != q.density.cols
                || q.overlap.rows != q.density.rows
                || q.overlap.cols != q.density.cols
                || !q.kinetic_energy.is_finite()
                || q.density.as_slice().iter().any(|v| !v.is_finite())
            {
                return Err(DftError::InvalidInput(
                    "mass derivative particle masses or reference AO dimensions are invalid".into(),
                ));
            }
            let overlap = overlap_matrix(&p.basis);
            let mut kinetic = kinetic_matrix(&p.basis);
            for v in kinetic.as_mut_slice() {
                *v /= p.mass;
            }
            let energy = q.density.frobenius_dot(&kinetic);
            let bound = 64.0 * f64::EPSILON * (energy.abs() + q.kinetic_energy.abs());
            let position_matches = dipole_matrices(&p.basis, Vec3::zero())
                .iter()
                .zip(q.position)
                .all(|(operator, reference)| {
                    let current = q.density.frobenius_dot(operator);
                    reference.is_finite()
                        && current.is_finite()
                        && (current - reference).abs()
                            <= 128.0 * f64::EPSILON * (1.0 + current.abs() + reference.abs())
                });
            if !energy.is_finite()
                || (energy - q.kinetic_energy).abs() > bound
                || overlap.max_abs_diff(&q.overlap) > 64.0 * f64::EPSILON
                || !position_matches
            {
                return Err(DftError::InvalidInput(
                    "mass derivatives require the reference particle masses and Gaussian bases"
                        .into(),
                ));
            }
            for v in kinetic.as_mut_slice() {
                *v = -*v / p.mass;
            }
            if kinetic.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput(
                    "mass derivative is not representable".into(),
                ));
            }
            Ok(kinetic)
        })
        .collect()
}

/// Hellmann-Feynman gradient for the matching jointly stationary reference.
/// The basis, charge and correlation parameters must be those of `state`.
pub fn gradient(particles: &[QuantumParticle], state: &MulticomponentResult) -> Result<Vec<f64>> {
    operators(particles, state)?
        .iter()
        .zip(&state.quantum)
        .map(|(t, q)| {
            let value = q.density.frobenius_dot(t);
            if !value.is_finite() {
                return Err(DftError::InvalidInput(
                    "mass derivative is not representable".into(),
                ));
            }
            Ok(value)
        })
        .collect()
}

/// Full relaxed mass Hessian, solving each ordered column independently.
/// `F'_i=-T_i/m_i^2` and `E''_ij=Tr(P'_j F'_i)+delta_ij 2 Tr(P_i T_i)/m_i^3`.
/// The matching coupled solver includes the position KKT equations when present.
#[allow(clippy::too_many_arguments)]
pub fn response(
    molecule: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    state: &MulticomponentResult,
    config: &ResponseSolverConfig,
) -> Result<MassResponse> {
    let sources = operators(particles, state)?;
    let gradient: Vec<_> = sources
        .iter()
        .zip(&state.quantum)
        .map(|(t, q)| q.density.frobenius_dot(t))
        .collect();
    let solver = CoupledResponseSolver::new(
        molecule,
        basis,
        options,
        functional,
        grid,
        particles,
        correlation,
        state,
    )?;
    let n = particles.len();
    let e = basis.nao_sph;
    let zero = [Matrix::zeros(e, e), Matrix::zeros(e, e)];
    let overlaps: Vec<_> = particles.iter().map(|p| overlap_matrix(&p.basis)).collect();
    let reference_number_defects = state
        .quantum
        .iter()
        .zip(&overlaps)
        .map(|(q, s)| q.density.frobenius_dot(s) - 1.0)
        .collect();
    let kinetic_energy_defects = particles
        .iter()
        .zip(&state.quantum)
        .zip(&gradient)
        .map(|((p, q), g)| -p.mass * g - q.kinetic_energy)
        .collect();
    let mut hessian = Matrix::zeros(n, n);
    let mut responses = Vec::with_capacity(n);
    let mut response_number_defects = Vec::with_capacity(n);
    for j in 0..n {
        let mut q: Vec<_> = particles
            .iter()
            .map(|p| Matrix::zeros(p.basis.nao_sph, p.basis.nao_sph))
            .collect();
        q[j] = sources[j].clone();
        let r = solver.solve(&zero, &q, config)?;
        response_number_defects.push(
            r.quantum_density
                .iter()
                .zip(&overlaps)
                .map(|(p, s)| p.frobenius_dot(s))
                .collect(),
        );
        for i in 0..n {
            hessian[(i, j)] = r.quantum_density[i].frobenius_dot(&sources[i]);
            if i == j {
                hessian[(i, j)] -= 2.0 * gradient[i] / particles[i].mass;
            }
        }
        responses.push(r);
    }
    if gradient
        .iter()
        .chain(hessian.as_slice())
        .any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "mass response is not representable".into(),
        ));
    }
    let raw_asymmetry = hessian.max_abs_diff(&hessian.transpose());
    Ok(MassResponse {
        gradient,
        hessian,
        raw_asymmetry,
        responses,
        reference_number_defects,
        response_number_defects,
        kinetic_energy_defects,
    })
}
