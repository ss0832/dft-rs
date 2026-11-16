// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Density-independent Gaussian overlap with an explicit immutable pair mask.
use crate::math::Vec3;
use crate::{DftError, Result};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct StericAtom {
    pub position: Vec3,
    pub exponent: f64,
    /// Fixed environment hardness in Bohr^-3.
    pub hardness: f64,
}

pub struct StericOverlap {
    /// Unit-strength overlap. Multiply by a strength in Hartree for energy.
    pub value: f64,
    pub center_gradient: Vec3,
    pub atom_gradients: Vec<Vec3>,
}

/// Excluded pairs contribute neither energy nor force. The caller must encode
/// self/intrafragment/neighbor exclusions explicitly; no distance-based mask
/// changes or generalized-site pair double counting occur in this function.
pub fn steric_overlap(
    center: Vec3,
    exponent: f64,
    environment: &[StericAtom],
    excluded: &[bool],
) -> Result<StericOverlap> {
    if excluded.len() != environment.len()
        || !exponent.is_finite()
        || exponent <= 0.0
        || (0..3).any(|i| !center.get(i).is_finite())
        || environment.iter().any(|a| {
            !a.exponent.is_finite()
                || a.exponent <= 0.0
                || !a.hardness.is_finite()
                || a.hardness < 0.0
                || (0..3).any(|i| !a.position.get(i).is_finite())
        })
    {
        return Err(DftError::InvalidInput("steric overlap requires finite centers, positive exponents, nonnegative hardness and an explicit complete exclusion mask".into()));
    }
    let mut result = StericOverlap {
        value: 0.0,
        center_gradient: Vec3::zero(),
        atom_gradients: vec![Vec3::zero(); environment.len()],
    };
    for (index, atom) in environment.iter().enumerate() {
        if excluded[index] {
            continue;
        }
        let p = exponent + atom.exponent;
        let rho = exponent * atom.exponent / p;
        let r = center - atom.position;
        let value = atom.hardness * (PI / p).powf(1.5) * (-rho * r.norm2()).exp();
        let gradient = r * (-2.0 * rho * value);
        result.value += value;
        result.center_gradient += gradient;
        result.atom_gradients[index] = gradient * -1.0;
    }
    Ok(result)
}
