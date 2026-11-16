// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Conversion of proper piezoelectric stress coefficients to strain coefficients.

use crate::error::{DftError, Result};

/// Compute `d[i,J] = sum_K e[i,K] S[K,J]` at fixed electric field.
///
/// Voigt order is xx, yy, zz, yz, xz, xy, with engineering shear strain
/// `(eps_xx, eps_yy, eps_zz, 2 eps_yz, 2 eps_xz, 2 eps_xy)` and unscaled
/// shear stress. Thus no additional factors of two belong in this product.
/// For `e` in C/m^2 and `S` in Pa^-1 the result is C/N, equivalently m/V.
/// Atomic input units give e*bohr/hartree. Inputs must use the same axes,
/// electric boundary condition and clamped- or relaxed-ion convention.
///
/// This algebraic conversion does not establish mechanical stability or
/// validate the electronic reference that supplied either input tensor.
pub fn strain_coefficients(
    stress_coefficients: &[[f64; 6]; 3],
    compliance: &[[f64; 6]; 6],
) -> Result<[[f64; 6]; 3]> {
    if stress_coefficients.iter().flatten().any(|v| !v.is_finite())
        || compliance.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "piezoelectric strain coefficients need finite stress coefficients and compliance"
                .into(),
        ));
    }
    let scale = compliance
        .iter()
        .flatten()
        .fold(0.0_f64, |a, b| a.max(b.abs()));
    if (0..6).any(|i| (0..6).any(|j| (compliance[i][j] - compliance[j][i]).abs() > 1e-12 * scale)) {
        return Err(DftError::InvalidInput(
            "piezoelectric compliance must be symmetric in engineering Voigt notation".into(),
        ));
    }
    let result = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (0..6)
                .map(|k| stress_coefficients[i][k] * compliance[k][j])
                .sum()
        })
    });
    if result.iter().flatten().any(|v: &f64| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "piezoelectric strain coefficient conversion overflowed".into(),
        ));
    }
    Ok(result)
}
