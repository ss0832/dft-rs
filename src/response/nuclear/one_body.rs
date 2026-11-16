// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Compensated one-body expectations in the same moving AO representation
//! as the stationary nuclear density. Metric response is already contained
//! in that density; applying a second metric transform would double count it.

use crate::linalg::{AccurateSum, Matrix};
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct OneBodyCoefficients {
    /// Factorial-normalized coefficients of the expectation value.
    pub coefficients: Vec<f64>,
    /// Accumulation-only absolute bounds. Input/response errors are separate.
    pub accumulation_bounds: Vec<f64>,
}

/// Contract every coefficient with both represented density parts. The
/// weights select total density `[1,1]` or spin density `[1,-1]`, for example.
/// No high+low matrix is rounded before its product is accumulated.
pub fn nuclear_one_body_coefficients(
    high: &[[Matrix; 2]],
    low: &[[Matrix; 2]],
    operators: &[Matrix],
    weights: [f64; 2],
) -> Result<OneBodyCoefficients> {
    let width = operators.len();
    if width == 0
        || high.len() < width
        || low.len() < width
        || weights.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput("one-body response requires finite spin weights and complete density/operator coefficients".into()));
    }
    let n = operators[0].rows;
    if n == 0
        || operators
            .iter()
            .chain(high[..width].iter().flatten())
            .chain(low[..width].iter().flatten())
            .any(|m| m.rows != n || m.cols != n || m.as_slice().iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "one-body coefficients require finite square matrices in the same AO space".into(),
        ));
    }
    let mut coefficients = Vec::with_capacity(width);
    let mut accumulation_bounds = Vec::with_capacity(width);
    for degree in 0..width {
        let mut sum = AccurateSum::default();
        for k in 0..=degree {
            for (spin, weight) in weights.iter().enumerate() {
                for mu in 0..n {
                    for nu in 0..n {
                        let o = *weight * operators[k][(nu, mu)];
                        sum.add_product(o, high[degree - k][spin][(mu, nu)]);
                        sum.add_product(o, low[degree - k][spin][(mu, nu)]);
                    }
                }
            }
        }
        let value = sum.value();
        let bound = sum.roundoff_bound();
        if !value.is_finite() || !bound.is_finite() {
            return Err(DftError::InvalidInput(
                "nonfinite one-body contraction or accumulation bound".into(),
            ));
        }
        coefficients.push(value);
        accumulation_bounds.push(bound);
    }
    Ok(OneBodyCoefficients {
        coefficients,
        accumulation_bounds,
    })
}
