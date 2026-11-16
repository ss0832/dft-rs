// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Local stability of a converged electronic/quantum-particle reference.
//!
//! The lowest eigenvalues of the coupled orbital Hessian
//! ([`CoupledResponseSolver::apply_real`], normalized to one half of the
//! variational energy Hessian) come from a Davidson iteration with the
//! orbital-gap preconditioner. A negative eigenvalue marks a saddle point:
//! [`follow`] rotates the occupied orbitals of every species along the mode
//! and re-converges the coupled problem from there, keeping the first trial
//! step that lowers the energy, until the lowest curvature is positive.

use super::correlation::CorrelationModel;
use super::response::CoupledResponseSolver;
use super::{solve_from, MulticomponentOptions, MulticomponentResult, QuantumParticle};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Lowest coupled curvatures and their unit modes.
#[derive(Clone, Debug)]
pub struct CoupledStability {
    /// Ascending eigenvalues of the normalized coupled orbital Hessian
    /// (Hartree; the energy curvature along a unit mode is twice the value).
    pub eigenvalues: Vec<f64>,
    pub modes: Vec<Vec<f64>>,
    /// Raw `|A x - lambda x|` of every returned pair.
    pub residuals: Vec<f64>,
    pub iterations: usize,
    pub dimension: usize,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Lowest `roots` eigenpairs of the coupled orbital Hessian to the raw
/// residual `tolerance`.
pub fn lowest_curvatures(
    solver: &CoupledResponseSolver,
    roots: usize,
    tolerance: f64,
    max_iterations: usize,
) -> Result<CoupledStability> {
    let n = solver.dimension();
    if n == 0 || roots == 0 || !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(DftError::InvalidInput(
            "coupled stability needs a nonempty rotation space, at least one root and a positive tolerance"
                .into(),
        ));
    }
    let roots = roots.min(n);
    let gaps = solver.orbital_gaps();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| gaps[*a].total_cmp(&gaps[*b]));
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut images: Vec<Vec<f64>> = Vec::new();
    let add =
        |vector: Vec<f64>, basis: &mut Vec<Vec<f64>>, images: &mut Vec<Vec<f64>>| -> Result<bool> {
            let mut v = vector;
            for _ in 0..2 {
                for b in basis.iter() {
                    let c = dot(&v, b);
                    for (x, y) in v.iter_mut().zip(b) {
                        *x -= c * y;
                    }
                }
            }
            let norm = dot(&v, &v).sqrt();
            if norm < 1e-10 {
                return Ok(false);
            }
            for x in v.iter_mut() {
                *x /= norm;
            }
            images.push(solver.apply_real(&v)?);
            basis.push(v);
            Ok(true)
        };
    for &k in order.iter().take((2 * roots).min(n)) {
        let mut v = vec![0.0; n];
        v[k] = 1.0;
        add(v, &mut basis, &mut images)?;
    }
    // Small references can use the entire rotation space without restarting.
    let limit = roots.saturating_mul(20).min(n);
    for iteration in 1..=max_iterations {
        let m = basis.len();
        let mut small = Matrix::zeros(m, m);
        for i in 0..m {
            for j in 0..=i {
                let value = 0.5 * (dot(&basis[i], &images[j]) + dot(&basis[j], &images[i]));
                small[(i, j)] = value;
                small[(j, i)] = value;
            }
        }
        let (values, vectors) = symmetric_eigen(&small)?;
        let mut eigenvalues = Vec::with_capacity(roots);
        let mut modes = Vec::with_capacity(roots);
        let mut residuals = Vec::with_capacity(roots);
        let mut corrections = Vec::new();
        for r in 0..roots.min(m) {
            let mut x = vec![0.0; n];
            let mut ax = vec![0.0; n];
            for k in 0..m {
                let y = vectors[(k, r)];
                for p in 0..n {
                    x[p] += y * basis[k][p];
                    ax[p] += y * images[k][p];
                }
            }
            let residual: Vec<f64> = ax.iter().zip(&x).map(|(a, b)| a - values[r] * b).collect();
            let norm = dot(&residual, &residual).sqrt();
            if norm > tolerance {
                corrections.push(
                    residual
                        .iter()
                        .zip(&gaps)
                        .map(|(value, gap)| {
                            let d = values[r] - gap;
                            value
                                / if d.abs() < 1e-4 {
                                    1e-4_f64.copysign(d)
                                } else {
                                    d
                                }
                        })
                        .collect::<Vec<f64>>(),
                );
            }
            eigenvalues.push(values[r]);
            modes.push(x);
            residuals.push(norm);
        }
        if corrections.is_empty() {
            return Ok(CoupledStability {
                eigenvalues,
                modes,
                residuals,
                iterations: iteration,
                dimension: n,
            });
        }
        if basis.len() + corrections.len() > limit {
            // Restart from the current Ritz vectors.
            basis.clear();
            images.clear();
            for mode in modes {
                add(mode, &mut basis, &mut images)?;
            }
        }
        let mut grew = false;
        for correction in corrections {
            grew |= add(correction, &mut basis, &mut images)?;
        }
        if !grew {
            return Err(DftError::InvalidInput(
                "the coupled stability iteration stagnated before its residual tolerance".into(),
            ));
        }
    }
    Err(DftError::InvalidInput(format!(
        "the coupled stability iteration did not reach the residual {tolerance:.1e} in \
         {max_iterations} iterations"
    )))
}

/// Outcome of following negative coupled curvatures.
pub struct FollowedReference {
    pub result: MulticomponentResult,
    /// Stability of the returned reference.
    pub stability: CoupledStability,
    /// Lowest eigenvalue of every analysed reference, the first being the
    /// starting one.
    pub history: Vec<f64>,
    /// Starting minus final energy (Hartree).
    pub energy_lowering: f64,
    pub steps: usize,
}

/// Analyse the reference and, when `follow`, step along every negative
/// lowest mode (trial lengths 1, 1/2, 1/4 of the unit mode in normalized
/// amplitudes) and re-converge, until the lowest curvature exceeds
/// `-threshold` or `max_steps` steps were taken.
#[allow(clippy::too_many_arguments)]
pub fn follow(
    molecule: &Molecule,
    electronic_basis: &MolBasis,
    electronic_options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    options: &MulticomponentOptions,
    reference: MulticomponentResult,
    roots: usize,
    threshold: f64,
    follow_modes: bool,
    max_steps: usize,
) -> Result<FollowedReference> {
    let start = reference.energy;
    let mut result = reference;
    let mut history = Vec::new();
    let mut steps = 0;
    loop {
        let (stability, trials) = {
            let solver = CoupledResponseSolver::new(
                molecule,
                electronic_basis,
                electronic_options,
                functional,
                grid,
                particles,
                correlation,
                &result,
            )?;
            let stability = lowest_curvatures(&solver, roots, 1e-6, 200)?;
            let unstable = stability.eigenvalues[0] < -threshold;
            let trials = if follow_modes && unstable && steps < max_steps {
                [1.0, 0.5, 0.25]
                    .iter()
                    .map(|&t| solver.rotated_state(&stability.modes[0], t))
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            (stability, trials)
        };
        history.push(stability.eigenvalues[0]);
        if trials.is_empty() {
            if follow_modes && stability.eigenvalues[0] < -threshold {
                return Err(DftError::ScfNotConverged {
                    iterations: steps,
                    error: stability.eigenvalues[0],
                });
            }
            return Ok(FollowedReference {
                energy_lowering: start - result.energy,
                result,
                stability,
                history,
                steps,
            });
        }
        let mut accepted = None;
        for trial in &trials {
            let candidate = solve_from(
                molecule,
                electronic_basis,
                electronic_options,
                functional,
                grid,
                particles,
                correlation,
                options,
                trial,
            )?;
            if candidate.converged && candidate.energy < result.energy - 1e-9 {
                accepted = Some(candidate);
                break;
            }
        }
        match accepted {
            Some(candidate) => {
                result = candidate;
                steps += 1;
            }
            None => {
                return Err(DftError::InvalidInput(format!(
                    "no trial step along the coupled mode with curvature {:.6e} lowered the \
                     energy after re-convergence",
                    stability.eigenvalues[0]
                )))
            }
        }
    }
}
