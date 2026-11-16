// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Equality constraints on a full-spin response, imposed by Lagrange
//! multipliers. The physical Hessian is never shifted or gap-regularized.

use super::*;
use crate::response::complex::{dot, norm};

pub(super) struct ConstraintSpace {
    pub basis: Vec<Vec<Complex>>,
    pub transforms: Vec<Vec<Complex>>,
    pub vertices: Vec<Vec<Complex>>,
    pub discarded_norms: Vec<f64>,
}

impl ConstraintSpace {
    pub fn new(response: &Response<'_>, operators: &[BlochMatrix]) -> Result<Self> {
        let mut basis: Vec<Vec<Complex>> = Vec::new();
        let mut transforms: Vec<Vec<Complex>> = Vec::new();
        let mut vertices = Vec::new();
        let mut discarded_norms = Vec::new();
        let coefficient_norm = |c: &BlochMatrix| {
            c.re.as_slice()
                .iter()
                .chain(c.im.as_slice())
                .fold(0.0_f64, |a, &b| a.hypot(b))
        };
        let scale = coefficient_norm(&response.orbitals.occupied)
            * coefficient_norm(&response.orbitals.empty);
        for (index, operator) in operators.iter().enumerate() {
            let vertex = response.orbitals.projected(operator)?;
            if vertex.iter().any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput(
                    "non-finite projected spinor expectation constraint".into(),
                ));
            }
            let mut remaining = vertex.clone();
            let mut transform = vec![Complex::ZERO; operators.len()];
            transform[index] = Complex::ONE;
            // Reorthogonalization distinguishes redundant constraints from
            // physical small eigenvalues. The bound concerns only floating-
            // point projection/rank, never an excitation denominator.
            for _ in 0..2 {
                for (column, map) in basis.iter().zip(&transforms) {
                    let weight = dot(column, &remaining);
                    for (a, b) in remaining.iter_mut().zip(column) {
                        *a -= weight * *b;
                    }
                    for (a, b) in transform.iter_mut().zip(map) {
                        *a -= weight * *b;
                    }
                }
            }
            let length = norm(&remaining);
            let bound =
                32.0 * f64::EPSILON * operator.rows() as f64 * scale * coefficient_norm(operator);
            if !length.is_finite() || !bound.is_finite() {
                return Err(DftError::InvalidInput(
                    "spinor constraint projection exceeds finite response arithmetic".into(),
                ));
            }
            if length > bound {
                for x in &mut remaining {
                    *x = *x / length;
                }
                for x in &mut transform {
                    *x = *x / length;
                }
                basis.push(remaining);
                transforms.push(transform);
                discarded_norms.push(0.0);
            } else {
                discarded_norms.push(length);
            }
            vertices.push(vertex);
        }
        Ok(Self {
            basis,
            transforms,
            vertices,
            discarded_norms,
        })
    }

    pub fn diagnostics(
        &self,
        solution: &[Complex],
        dimension: usize,
    ) -> (Vec<Complex>, Vec<Complex>) {
        let errors = self
            .vertices
            .iter()
            .map(|v| dot(v, &solution[..dimension]))
            .collect();
        let mut multipliers = vec![Complex::ZERO; self.vertices.len()];
        for (map, &lambda) in self.transforms.iter().zip(&solution[dimension..]) {
            for (out, &weight) in multipliers.iter_mut().zip(map) {
                *out += weight * lambda;
            }
        }
        (errors, multipliers)
    }
}

pub(super) struct Constrained<'a, 'b> {
    pub response: &'a Response<'b>,
    pub constraints: &'a ConstraintSpace,
}

impl LinearOperator for Constrained<'_, '_> {
    fn dimension(&self) -> usize {
        self.response.dimension() + self.constraints.basis.len()
    }
    fn structure(&self) -> OperatorStructure {
        self.response.structure()
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        if x.len() != self.dimension() {
            return Err(DftError::InvalidInput(
                "invalid constrained spinor response dimension".into(),
            ));
        }
        let n = self.response.dimension();
        let mut result = self.response.apply(&x[..n])?;
        for (column, &lambda) in self.constraints.basis.iter().zip(&x[n..]) {
            for (out, &value) in result.iter_mut().zip(column) {
                *out += value * lambda;
            }
        }
        result.extend(self.constraints.basis.iter().map(|v| dot(v, &x[..n])));
        Ok(result)
    }
}
