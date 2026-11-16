// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit constrained solves for gauge, particle-number and Goldstone
//! directions. On the allowed space the physical operator is untouched;
//! the complementary identity only closes the augmented linear system.

use super::complex::{dot, norm, Complex};
use super::krylov::{
    solve, LinearOperator, LinearSolution, OperatorStructure, ResponseSolverConfig,
};
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct ConstraintSpace {
    dimension: usize,
    basis: Vec<Vec<Complex>>,
}
impl ConstraintSpace {
    /// Orthonormalize independently supplied excluded directions, rejecting
    /// zero or numerically dependent modes. The constraint is Q†x=0.
    pub fn new(dimension: usize, modes: &[Vec<Complex>]) -> Result<Self> {
        let mut basis: Vec<Vec<Complex>> = Vec::new();
        for mode in modes {
            if mode.len() != dimension || mode.iter().any(|z| !z.is_finite()) {
                return Err(DftError::InvalidInput(
                    "invalid response constraint mode".into(),
                ));
            }
            let scale = mode
                .iter()
                .fold(0.0_f64, |m, z| m.max(z.re.abs()).max(z.im.abs()));
            if scale == 0.0 {
                return Err(DftError::InvalidInput(
                    "zero response constraint mode".into(),
                ));
            }
            let mut vector: Vec<_> = mode.iter().map(|&z| z / scale).collect();
            let initial = norm(&vector);
            for _ in 0..2 {
                for q in &basis {
                    let a = dot(q, &vector);
                    for (v, &q) in vector.iter_mut().zip(q) {
                        *v -= q * a;
                    }
                }
            }
            let length = norm(&vector);
            if length <= 1e-12 * initial {
                return Err(DftError::InvalidInput(
                    "linearly dependent response constraint modes".into(),
                ));
            }
            for z in &mut vector {
                *z = *z / length;
            }
            basis.push(vector);
        }
        Ok(Self { dimension, basis })
    }
    pub fn project(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        if x.len() != self.dimension || x.iter().any(|z| !z.is_finite()) {
            return Err(DftError::InvalidInput(
                "invalid constrained response vector".into(),
            ));
        }
        let mut p = x.to_vec();
        // Reorthogonalization preserves a tiny allowed component next to a
        // large excluded reaction. One subtraction leaves an O(eps) excluded
        // remainder that can dominate a symmetry-forbidden response source.
        for _ in 0..2 {
            for q in &self.basis {
                let a = dot(q, &p);
                for (p, &q) in p.iter_mut().zip(q) {
                    *p -= q * a;
                }
            }
        }
        Ok(p)
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn excluded_rank(&self) -> usize {
        self.basis.len()
    }
}

struct ConstrainedOperator<'a, O> {
    physical: &'a O,
    space: &'a ConstraintSpace,
}
impl<O: LinearOperator> LinearOperator for ConstrainedOperator<'_, O> {
    fn dimension(&self) -> usize {
        self.physical.dimension()
    }
    fn structure(&self) -> OperatorStructure {
        self.physical.structure()
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        let p = self.space.project(x)?;
        let image = self.physical.apply(&p)?;
        let mut result = self.space.project(&image)?;
        for ((r, &x), &p) in result.iter_mut().zip(x).zip(&p) {
            *r += x - p;
        }
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub struct ConstrainedSolution {
    pub solution: LinearSolution,
    /// Q†(b-Ax), the Lagrange multipliers in Ax+Q lambda=b.
    pub multipliers: Vec<Complex>,
    pub constraint_defect: f64,
}

pub fn solve_constrained(
    op: &impl LinearOperator,
    rhs: &[Complex],
    space: &ConstraintSpace,
    config: &ResponseSolverConfig,
) -> Result<ConstrainedSolution> {
    if op.dimension() != space.dimension {
        return Err(DftError::InvalidInput(
            "constraint/operator dimension mismatch".into(),
        ));
    }
    let projected = space.project(rhs)?;
    let augmented = ConstrainedOperator {
        physical: op,
        space,
    };
    let mut solution = solve(&augmented, &projected, None, config)?;
    solution.amplitudes = space.project(&solution.amplitudes)?;
    let image = op.apply(&solution.amplitudes)?;
    if image.len() != rhs.len() || image.iter().any(|z| !z.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "invalid constrained operator image".into(),
        ));
    }
    let projected_image = space.project(&image)?;
    let residual: Vec<_> = rhs.iter().zip(&image).map(|(&b, &ax)| b - ax).collect();
    let multipliers = space.basis.iter().map(|q| dot(q, &residual)).collect();
    let constraint_defect = space
        .basis
        .iter()
        .fold(0.0_f64, |m, q| m.max(dot(q, &solution.amplitudes).abs()));
    // Check the original allowed-space equation Q b - Q A x. Forming
    // Q(b-Ax) first loses the small physical residual to cancellation against
    // a large excluded acoustic/gauge reaction, although the two expressions
    // are identical in exact arithmetic. Multipliers above retain b-Ax.
    let residual: Vec<_> = projected
        .iter()
        .zip(projected_image)
        .map(|(&b, ax)| b - ax)
        .collect();
    let norm_rhs = norm(&projected);
    let residual_norm = norm(&residual);
    solution.relative_residual = if norm_rhs == 0.0 {
        residual_norm
    } else {
        residual_norm / norm_rhs
    };
    if solution.relative_residual > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: solution.iterations,
            residual: solution.relative_residual,
        });
    }
    Ok(ConstrainedSolution {
        solution,
        multipliers,
        constraint_defect,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Optical;
    impl LinearOperator for Optical {
        fn dimension(&self) -> usize {
            2
        }
        fn structure(&self) -> OperatorStructure {
            OperatorStructure::General
        }
        fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
            let optical = (x[0] - x[1]) * 0.00005;
            Ok(vec![optical, -optical])
        }
    }

    #[test]
    fn fresh_optical_residual_survives_a_large_acoustic_reaction() {
        let space = ConstraintSpace::new(2, &[vec![Complex::ONE; 2]]).unwrap();
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        for amplitude in [1e-4, 1e-8, 1e-12, 1e-15] {
            let rhs = vec![
                Complex::new(1.0 + amplitude, 0.3 - amplitude),
                Complex::new(1.0 - amplitude, 0.3 + amplitude),
            ];
            let allowed = space.project(&rhs).unwrap();
            let actual = solve_constrained(&Optical, &rhs, &space, &config).unwrap();
            for (x, b) in actual.solution.amplitudes.iter().zip(&allowed) {
                assert!((*x - *b / 0.0001).abs() <= 1e-12 * (b.abs() / 0.0001));
            }
            assert!(actual.solution.relative_residual <= 1e-12);
            assert!(actual.constraint_defect <= 1e-15);
            println!("acoustic reaction amplitude={amplitude:e} fresh optical residual={:e} constraint={:e}",actual.solution.relative_residual,actual.constraint_defect);
        }
    }
}
