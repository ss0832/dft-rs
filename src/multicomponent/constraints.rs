// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Expectation-position constraints inside the simultaneous species SCF.
//! L = E + sum lambda_I . (<r_I> - R_I). Nuclear inner solves enforce the
//! position at fixed Fock, before the complete species SCF is reconverged.
//! Physical energy excludes constraint work. Derivatives use the full KKT
//! orbital/position system, including electronic and multiplier response.

use super::correlation::CorrelationModel;
use super::{MulticomponentOptions, MulticomponentResult, QuantumParticle};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{Matrix, Orthogonalizer, symmetric_eigen};
use crate::response::ResponseSolverConfig;
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct PositionConstraints {
    pub targets: Vec<[f64; 3]>,
    /// +lambda . (<r>-target), in Hartree/Bohr.
    pub multipliers: Vec<[f64; 3]>,
    pub residual: f64,
    pub iterations: usize,
    pub residual_history: Vec<f64>,
}

fn maximum(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn range_solve(a: &Matrix, rhs: &[f64], tolerance: f64) -> Result<Vec<f64>> {
    let (e, c) = symmetric_eigen(a)?;
    let threshold = 128.0 * f64::EPSILON * maximum(&e);
    let mut out = vec![0.0; rhs.len()];
    for (k, v) in e.iter().enumerate() {
        let b: f64 = rhs.iter().enumerate().map(|(i, r)| c[(i, k)] * r).sum();
        if v.abs() > threshold {
            for i in 0..out.len() {
                out[i] += c[(i, k)] * b / v;
            }
        }
    }
    let defect: Vec<_> = (0..out.len())
        .map(|i| (0..out.len()).map(|j| a[(i, j)] * out[j]).sum::<f64>() - rhs[i])
        .collect();
    if maximum(&defect) > tolerance || out.iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(format!(
            "quantum position constraint is outside the basis response range: residual {:.12e}",
            maximum(&defect)
        )));
    }
    Ok(out)
}

pub(super) fn operators(moments: &[Matrix; 3], overlap: &Matrix, target: [f64; 3]) -> [Matrix; 3] {
    std::array::from_fn(|a| {
        Matrix::from_row_major(
            overlap.rows,
            overlap.cols,
            moments[a]
                .as_slice()
                .iter()
                .zip(overlap.as_slice())
                .map(|(r, s)| r - target[a] * s)
                .collect(),
        )
    })
}

fn orbital_error(c: &Matrix, operators: &[Matrix; 3]) -> Vec<f64> {
    operators
        .iter()
        .map(|a| c.transpose_matmul_seq(a).matmul_seq(c)[(0, 0)])
        .collect()
}

pub(super) struct ConstrainedOrbital {
    pub fock: Matrix,
    pub energies: Vec<f64>,
    pub coefficient: Vec<f64>,
    pub multipliers: [f64; 3],
    pub iterations: usize,
}

/// Analytic fixed-Fock nuclear susceptibility avoids inverting the
/// unconstrained electronic/nuclear translational Goldstone mode.
pub(super) fn orbital(
    physical: &Matrix,
    orth: &Orthogonalizer,
    operators: &[Matrix; 3],
    initial: [f64; 3],
    tolerance: f64,
) -> Result<ConstrainedOrbital> {
    let mut lambda = initial;
    let evaluate = |lambda: [f64; 3]| -> Result<_> {
        let mut f = physical.clone();
        for a in 0..3 {
            super::add(&mut f, &operators[a], lambda[a]);
        }
        let (e, c) = orth.solve_roothaan(&f)?;
        let occupied = Matrix::from_row_major(c.rows, 1, (0..c.rows).map(|i| c[(i, 0)]).collect());
        let r = orbital_error(&occupied, operators);
        Ok((f, e, c, r))
    };
    let mut state = evaluate(lambda)?;
    for iteration in 0..100 {
        if maximum(&state.3) <= tolerance {
            return Ok(ConstrainedOrbital {
                fock: state.0,
                energies: state.1,
                coefficient: (0..state.2.rows).map(|i| state.2[(i, 0)]).collect(),
                multipliers: lambda,
                iterations: iteration,
            });
        }
        let transformed: Vec<_> = operators
            .iter()
            .map(|a| state.2.transpose_matmul_seq(a).matmul_seq(&state.2))
            .collect();
        let mut jacobian = Matrix::zeros(3, 3);
        for a in 0..3 {
            for b in 0..3 {
                for k in 1..state.1.len() {
                    let gap = state.1[0] - state.1[k];
                    let numerator = 2.0 * transformed[a][(0, k)] * transformed[b][(k, 0)];
                    if gap == 0.0 && numerator != 0.0 {
                        return Err(DftError::InvalidInput("position-constrained nuclear ground state is exactly degenerate in an active position channel".into()));
                    }
                    if gap != 0.0 {
                        jacobian[(a, b)] += numerator / gap;
                    }
                }
            }
        }
        let step = range_solve(
            &jacobian,
            &state.3.iter().map(|r| -r).collect::<Vec<_>>(),
            tolerance,
        )?;
        let mut fraction = 1.0;
        let mut accepted = None;
        for _ in 0..32 {
            let next = std::array::from_fn(|a| lambda[a] + fraction * step[a]);
            let trial = evaluate(next)?;
            if maximum(&trial.3) < maximum(&state.3) {
                accepted = Some((next, trial));
                break;
            }
            fraction *= 0.5;
        }
        (lambda, state) = accepted.ok_or_else(|| {
            DftError::LinearAlgebra(format!(
                "nuclear position multiplier line search failed at residual {:.12e}",
                maximum(&state.3)
            ))
        })?;
    }
    Err(DftError::ScfNotConverged {
        iterations: 100,
        error: maximum(&state.3),
    })
}

/// Retract a normalized orbital step onto its position manifold. Null
/// control directions are inspected rather than physically regularized.
pub(super) fn retract(
    coefficient: &mut [f64],
    overlap: &Matrix,
    orth: &Orthogonalizer,
    operators: &[Matrix; 3],
    tolerance: f64,
) -> Result<()> {
    let c = Matrix::from_row_major(coefficient.len(), 1, coefficient.to_vec());
    let mut y = orth.x.transpose_matmul_seq(&overlap.matmul_seq(&c));
    for _ in 0..40 {
        let norm = y.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in y.as_mut_slice() {
            *x /= norm;
        }
        let c = orth.x.matmul_seq(&y);
        let r = orbital_error(&c, operators);
        if maximum(&r) <= tolerance {
            coefficient.copy_from_slice(c.as_slice());
            return Ok(());
        }
        let gradients: Vec<_> = operators
            .iter()
            .map(|a| {
                let mut g = orth.x.transpose_matmul_seq(&a.matmul_seq(&c));
                let along = y.frobenius_dot(&g);
                super::add(&mut g, &y, -along);
                g
            })
            .collect();
        let jacobian = Matrix::from_row_major(
            3,
            3,
            (0..3)
                .flat_map(|a| {
                    (0..3).map({
                        let g = &gradients;
                        move |b| 2.0 * g[a].frobenius_dot(&g[b])
                    })
                })
                .collect(),
        );
        let step = range_solve(
            &jacobian,
            &r.iter().map(|v| -v).collect::<Vec<_>>(),
            tolerance,
        )?;
        for a in 0..3 {
            super::add(&mut y, &gradients[a], step[a]);
        }
    }
    Err(DftError::LinearAlgebra(
        "nuclear position-manifold retraction did not converge".into(),
    ))
}

/// Constrain every selected particle's expectation position. Each nuclear
/// inner solve enforces its position analytically, and the outer loop
/// reconverges all electronic/nuclear physical density terms.
#[allow(clippy::too_many_arguments)]
pub fn solve(
    molecule: &Molecule,
    basis: &MolBasis,
    electronic: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    options: &MulticomponentOptions,
    targets: &[[f64; 3]],
    config: &ResponseSolverConfig,
) -> Result<MulticomponentResult> {
    solve_with_gradient(
        molecule,
        basis,
        electronic,
        functional,
        grid,
        particles,
        correlation,
        options,
        targets,
        config,
        None,
    )
}

/// Constrain each expectation position while solving in a finite electric
/// field gradient. The constraint work is excluded from the physical energy.
#[allow(clippy::too_many_arguments)]
pub fn solve_in_field_gradient(
    molecule: &Molecule,
    basis: &MolBasis,
    electronic: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    options: &MulticomponentOptions,
    targets: &[[f64; 3]],
    config: &ResponseSolverConfig,
    gradient: &super::QuadrupoleField,
) -> Result<MulticomponentResult> {
    solve_with_gradient(
        molecule,
        basis,
        electronic,
        functional,
        grid,
        particles,
        correlation,
        options,
        targets,
        config,
        Some(gradient),
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_with_gradient(
    molecule: &Molecule,
    basis: &MolBasis,
    electronic: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    options: &MulticomponentOptions,
    targets: &[[f64; 3]],
    config: &ResponseSolverConfig,
    gradient: Option<&super::QuadrupoleField>,
) -> Result<MulticomponentResult> {
    config.validate()?;
    if targets.len() != particles.len() || targets.iter().flatten().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput(
            "position targets require three finite Bohr coordinates per quantum particle".into(),
        ));
    }
    let constraints = PositionConstraints {
        targets: targets.to_vec(),
        multipliers: vec![[0.0; 3]; targets.len()],
        residual: f64::INFINITY,
        iterations: 0,
        residual_history: Vec::new(),
    };
    let mut options = options.clone();
    options.tolerance = options.tolerance.min(config.acceptance());
    super::solve_with_multipliers(
        molecule,
        basis,
        electronic,
        functional,
        grid,
        particles,
        correlation,
        &options,
        Some(&constraints),
        None,
        gradient,
    )
}
