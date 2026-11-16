// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-geometry variational density constraints on integer determinants.

use super::{Reference, ScfOptions, ScfResult};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{solve_linear, symmetric_eigen, Matrix};
use crate::response::krylov::OperatorStructure;
use crate::response::{ResponseSolver, ResponseSolverConfig};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// One expectation constraint Tr[P_alpha Q_alpha + P_beta Q_beta] = target.
/// The supplied real symmetric AO operators are fixed at the given geometry.
#[derive(Clone, Debug)]
pub struct CollinearConstraint {
    pub operators: [Matrix; 2],
    pub target: f64,
}

/// Outer iteration on the Lagrange multipliers in Hartree.
#[derive(Clone, Debug)]
pub struct ConstraintOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub max_multiplier_step: f64,
    pub response: ResponseSolverConfig,
}

impl Default for ConstraintOptions {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            tolerance: 1e-9,
            max_multiplier_step: 0.5,
            response: ResponseSolverConfig::default(),
        }
    }
}

/// Constrained energy and fresh derivatives of the multiplier dual.
pub struct ConstrainedResult<State> {
    pub state: State,
    pub energy: f64,
    pub lagrangian: f64,
    pub multipliers: Vec<f64>,
    pub expectations: Vec<f64>,
    pub residuals: Vec<f64>,
    pub susceptibility: Matrix,
    pub response_residuals: Vec<f64>,
    pub iterations: usize,
}

pub(crate) struct Trial<State> {
    pub state: State,
    pub biased_energy: f64,
    pub expectations: Vec<f64>,
}

pub(crate) fn maximize_dual<State>(
    targets: &[f64],
    initial: &[f64],
    options: &ConstraintOptions,
    evaluate: impl Fn(&[f64], Option<&State>) -> Result<Trial<State>>,
    differentiate: impl Fn(&State) -> Result<(Matrix, Vec<f64>)>,
) -> Result<ConstrainedResult<State>> {
    let n = targets.len();
    if n == 0
        || initial.len() != n
        || targets.iter().chain(initial).any(|x| !x.is_finite())
        || options.max_iterations == 0
        || !(options.tolerance.is_finite() && options.tolerance > 0.0)
        || !(options.max_multiplier_step.is_finite() && options.max_multiplier_step > 0.0)
    {
        return Err(DftError::InvalidInput(
            "density constraints require finite targets and matching initial multipliers, positive tolerances and iteration limit".into(),
        ));
    }
    options.response.validate()?;
    let mut multipliers = initial.to_vec();
    let mut trial = evaluate(&multipliers, None)?;
    let residual = |trial: &Trial<State>| {
        trial
            .expectations
            .iter()
            .zip(targets)
            .map(|(q, t)| q - t)
            .collect::<Vec<_>>()
    };
    let norm = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut residuals = residual(&trial);
    for iteration in 1..=options.max_iterations {
        let defect = norm(&residuals);
        let lagrangian = trial.biased_energy
            - multipliers
                .iter()
                .zip(targets)
                .map(|(x, t)| x * t)
                .sum::<f64>();
        crate::logger::log(&format!(
            "[cdft] iter {iteration:>3} L={lagrangian:.12} |Q-target|={defect:.6e}"
        ));
        let (susceptibility, response_residuals) = differentiate(&trial.state)?;
        let scale = susceptibility
            .as_slice()
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        if susceptibility.as_slice().iter().any(|x| !x.is_finite())
            || susceptibility.max_abs_diff(&susceptibility.transpose()) > scale * 1e-7 + 1e-12
        {
            return Err(DftError::LinearAlgebra(
                "constrained DFT multiplier response is nonfinite or violates reciprocity".into(),
            ));
        }
        let mut symmetric = susceptibility.clone();
        for i in 0..n {
            for j in 0..i {
                let x = 0.5 * (susceptibility[(i, j)] + susceptibility[(j, i)]);
                symmetric[(i, j)] = x;
                symmetric[(j, i)] = x;
            }
        }
        let eigenvalues = symmetric_eigen(&symmetric)?.0;
        let rank_tolerance = scale * n as f64 * 64.0 * f64::EPSILON;
        let singular = eigenvalues[n - 1] >= -rank_tolerance;
        if defect <= options.tolerance && !singular {
            return Ok(ConstrainedResult {
                energy: trial.biased_energy
                    - multipliers
                        .iter()
                        .zip(&trial.expectations)
                        .map(|(x, q)| x * q)
                        .sum::<f64>(),
                lagrangian,
                state: trial.state,
                multipliers,
                expectations: trial.expectations,
                residuals,
                susceptibility,
                response_residuals,
                iterations: iteration,
            });
        }
        if singular {
            return Err(DftError::InvalidInput(format!(
                "density constraints have no independent concave multiplier response (largest eigenvalue {:.6e}, residual {defect:.6e}); check linearly dependent or unreachable targets and the reference stability",
                eigenvalues[n - 1]
            )));
        }
        let mut step = solve_linear(&symmetric, &residuals)?;
        let scaling = (options.max_multiplier_step / norm(&step)).min(1.0);
        for value in &mut step {
            *value *= scaling;
        }
        let mut fraction = 1.0;
        let mut next = None;
        for _ in 0..14 {
            let candidate: Vec<_> = multipliers
                .iter()
                .zip(&step)
                .map(|(x, d)| x - fraction * d)
                .collect();
            match evaluate(&candidate, Some(&trial.state)) {
                Ok(state) => {
                    let r = residual(&state);
                    if norm(&r) < defect {
                        next = Some((candidate, state, r));
                        break;
                    }
                }
                Err(DftError::ScfNotConverged { .. }) => {}
                Err(error) => return Err(error),
            }
            fraction *= 0.5;
        }
        let Some((candidate, state, r)) = next else {
            return Err(DftError::LinearAlgebra(format!(
                "constrained DFT line search stalled at iteration {iteration} with constraint residual {defect:.6e}"
            )));
        };
        (multipliers, trial, residuals) = (candidate, state, r);
    }
    Err(DftError::LinearAlgebra(format!(
        "constrained DFT exhausted {} multiplier iterations with residual {:.6e}",
        options.max_iterations,
        norm(&residuals)
    )))
}

pub(super) fn run_biased_collinear(
    mol: &Molecule,
    basis: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    operators: &[Matrix; 2],
    seed: Option<&[Matrix; 2]>,
) -> Result<ScfResult> {
    let electrons = mol
        .n_electrons()?
        .checked_sub(basis.n_core_electrons as usize)
        .ok_or_else(|| DftError::InvalidInput("ECP core exceeds electron count".into()))?;
    let (alpha, beta) = mol.n_alpha_beta_for(electrons)?;
    let restricted = opts.reference != Reference::Uhf && alpha == beta;
    let average = restricted.then(|| {
        let mut matrix = operators[0].clone();
        for (a, &b) in matrix
            .as_mut_slice()
            .iter_mut()
            .zip(operators[1].as_slice())
        {
            *a = 0.5 * (*a + b);
        }
        [matrix.clone(), matrix]
    });
    super::run_scf_in_potential_with_completion(
        mol,
        basis,
        opts,
        functional,
        grid,
        seed,
        None,
        None,
        Some(super::fields::DensityExtension::SpinPotential(
            average.as_ref().unwrap_or(operators),
        )),
        true,
    )
}

/// Solve one fixed set of longitudinal charge and spin constraints.
/// With common_orbitals, use the closed/open/virtual ROKS manifold.
#[allow(clippy::too_many_arguments)]
pub fn run_collinear(
    mol: &Molecule,
    basis: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    common_orbitals: bool,
    constraints: &[CollinearConstraint],
    initial: &[f64],
    options: &ConstraintOptions,
) -> Result<ConstrainedResult<(ScfResult, Option<Matrix>)>> {
    if opts.occupation.is_some()
        || opts.fixed_occupations.is_some()
        || opts.electronic_temperature != 0.0
        || opts.electron_offset != [0.0; 2]
        || opts.asymptotic_correction.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "variational constrained DFT".into(),
            missing: "an integer aufbau determinant with a variational energy and Fock".into(),
        });
    }
    let n = basis.nao_sph;
    for matrix in constraints.iter().flat_map(|c| &c.operators) {
        if matrix.rows != n
            || matrix.cols != n
            || matrix.as_slice().iter().any(|x| !x.is_finite())
            || matrix.max_abs_diff(&matrix.transpose()) > 1e-12
        {
            return Err(DftError::InvalidInput(
                "each density constraint needs finite symmetric spin/AO operators".into(),
            ));
        }
    }
    let targets: Vec<_> = constraints.iter().map(|c| c.target).collect();
    maximize_dual(
        &targets,
        initial,
        options,
        |lambda, seed: Option<&(ScfResult, Option<Matrix>)>| {
            let operators: [Matrix; 2] = std::array::from_fn(|s| {
                let mut total = Matrix::zeros(n, n);
                for (c, &value) in constraints.iter().zip(lambda) {
                    for (x, &q) in total
                        .as_mut_slice()
                        .iter_mut()
                        .zip(c.operators[s].as_slice())
                    {
                        *x += value * q;
                    }
                }
                total
            });
            let (scf, orbitals) = if common_orbitals {
                let (scf, orbitals) = super::roks::run_roks_in_spin_potential(
                    mol,
                    basis,
                    opts,
                    functional,
                    grid,
                    Some(&operators),
                )?;
                (scf, Some(orbitals))
            } else {
                (
                    run_biased_collinear(
                        mol,
                        basis,
                        opts,
                        functional,
                        grid,
                        &operators,
                        seed.map(|state| &state.0.density),
                    )?,
                    None,
                )
            };
            if !scf.converged {
                return Err(DftError::ScfNotConverged {
                    iterations: scf.iterations,
                    error: scf.common_orbital_residual.unwrap_or(f64::INFINITY),
                });
            }
            let expectations = constraints
                .iter()
                .map(|c| expectation(c, &scf.density))
                .collect();
            Ok(Trial {
                biased_energy: scf.energy,
                expectations,
                state: (scf, orbitals),
            })
        },
        |(scf, orbitals)| {
            operator_susceptibility(
                mol,
                basis,
                opts,
                functional,
                grid,
                scf,
                orbitals.as_ref(),
                constraints,
                &options.response,
            )
        },
    )
}

/// Static one-body susceptibility at the biased stationary determinant.
pub(crate) fn operator_susceptibility(
    mol: &Molecule,
    basis: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: Option<&Matrix>,
    constraints: &[CollinearConstraint],
    config: &ResponseSolverConfig,
) -> Result<(Matrix, Vec<f64>)> {
    let n = basis.nao_sph;
    let size = constraints.len();
    if let Some(orbitals) = orbitals.as_ref().filter(|_| scf.n_alpha != scf.n_beta) {
        let operators: Vec<_> = constraints
            .iter()
            .map(|c| {
                c.operators
                    .clone()
                    .map(|re| crate::integrals::london::ComplexMatrix {
                        re,
                        im: Matrix::zeros(n, n),
                    })
            })
            .collect();
        let response = crate::response::nuclear::open_shell_operator_response(
            mol,
            basis,
            functional,
            grid,
            scf,
            orbitals,
            opts,
            &operators,
            crate::response::Frequency::Static,
            config,
        )?;
        let values = response
            .susceptibility
            .iter()
            .flatten()
            .map(|v| v.re)
            .collect();
        return Ok((
            Matrix::from_row_major(size, size, values),
            response.residuals,
        ));
    }
    let solver = ResponseSolver::new(mol, basis, functional, grid, scf, opts)?;
    let (occupied, virtuals) = solver.ov_dims();
    let length: usize = (0..if scf.unrestricted { 2 } else { 1 })
        .map(|s| occupied[s] * virtuals[s])
        .sum();
    if length == 0 {
        return Err(DftError::Unsupported {
            feature: "density constraints on a saturated determinant".into(),
            missing: "an occupied-to-empty orbital rotation for a variable constraint expectation"
                .into(),
        });
    }
    let mut jacobian = Matrix::zeros(size, size);
    let mut residuals = Vec::with_capacity(size);
    for (b, c) in constraints.iter().enumerate() {
        let mut source = vec![0.0; length];
        if scf.unrestricted {
            solver.project_ov([&c.operators[0], &c.operators[1]], &mut source);
        } else {
            let mut average = c.operators[0].clone();
            for (a, &b) in average
                .as_mut_slice()
                .iter_mut()
                .zip(c.operators[1].as_slice())
            {
                *a = 0.5 * (*a + b);
            }
            solver.project_ov([&average, &average], &mut source);
        }
        let solution =
            solver.solve_ov_report_with_structure(&source, config, OperatorStructure::Hermitian)?;
        let vector: Vec<_> = solution.amplitudes.iter().map(|v| v.re).collect();
        let density = solver.delta_density(&vector);
        for (a, observable) in constraints.iter().enumerate() {
            jacobian[(a, b)] = expectation(observable, &density);
        }
        residuals.push(solver.ov_equation_residual(&vector, &source)?);
    }
    Ok((jacobian, residuals))
}

fn expectation(constraint: &CollinearConstraint, density: &[Matrix; 2]) -> f64 {
    constraint
        .operators
        .iter()
        .zip(density)
        .map(|(q, p)| q.frobenius_dot(p))
        .sum()
}

#[cfg(test)]
#[path = "constrained_tests.rs"]
mod tests;
