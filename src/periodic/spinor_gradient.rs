// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic complex-exchange contribution, not a total GKS nuclear force.
//!
//! Images are the normalized inverse Bloch coefficients used by the energy
//! backend. Both Hermitian image parities are retained. A Pauli decomposition
//! uses this same quadratic contraction for every component, with coefficient
//! -cx/4 or -clr/4. No collinear state or real-density projection is substituted.
//! This primitive assumes field-free real Gaussian AOs; London phases, Pulay,
//! Coulomb and semilocal XC nuclear derivatives are separate contributions.

use crate::basis::MolBasis;
use crate::integrals::CoulombKernel;
use crate::math::Vec3;
use crate::periodic::exchange_quartets::ExchangeCells;
use crate::periodic::gradient::{exchange_gradient_on_cells, ExchangeDerivativeScreening};
use crate::periodic::integrals::ImageBlock;
use crate::periodic::PeriodicGeometry;
use crate::{DftError, Result};
use std::collections::BTreeMap;

/// Real and imaginary inverse Fourier components of one Hermitian Pauli block.
pub struct ComplexImageDensity<'a> {
    pub real: &'a [ImageBlock],
    pub imaginary: &'a [ImageBlock],
}

pub struct ExchangeNuclearDerivative {
    /// Fixed-AO-density exchange contribution, Hartree/Bohr; excludes Pulay.
    pub gradient: Vec<Vec3>,
    /// Raw affine derivative in Hartree, at fixed reduced k and density.
    /// This is not the stress divided by the cell measure.
    pub strain_derivative: [[f64; 3]; 3],
}

fn key(t: Vec3) -> Result<[i64; 3]> {
    let values = [t.x, t.y, t.z];
    if values
        .iter()
        .any(|v| !v.is_finite() || v.abs() > (i64::MAX as f64) / 2e6)
    {
        return Err(DftError::InvalidInput(
            "exchange image translation is not representable".into(),
        ));
    }
    Ok(values.map(|v| (v * 1e6).round() as i64))
}

fn validate_images(mb: &MolBasis, images: &[ImageBlock], parity: f64) -> Result<Vec<[i64; 3]>> {
    let mut indexed = BTreeMap::new();
    for block in images {
        if block.matrix.rows != mb.nao_sph
            || block.matrix.cols != mb.nao_sph
            || block.matrix.as_slice().iter().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "complex exchange density has invalid AO dimensions or values".into(),
            ));
        }
        let k = key(block.translation)?;
        if indexed.insert(k, &block.matrix).is_some() {
            return Err(DftError::InvalidInput(
                "complex exchange density repeats an image".into(),
            ));
        }
    }
    for (k, a) in &indexed {
        let opposite = k.map(|v| -v);
        let b = indexed.get(&opposite).ok_or_else(|| {
            DftError::InvalidInput("complex exchange density is missing the opposite image".into())
        })?;
        for i in 0..mb.nao_sph {
            for j in 0..mb.nao_sph {
                let scale = 1.0 + a[(i, j)].abs().max(b[(j, i)].abs());
                if (a[(i, j)] - parity * b[(j, i)]).abs() > 1e-12 * scale {
                    return Err(DftError::InvalidInput(
                        "complex exchange density violates Hermitian image parity".into(),
                    ));
                }
            }
        }
    }
    Ok(indexed.into_keys().collect())
}

/// Differentiate precisely the existing dimensional exchange quartet sum.
/// Geometry selection is unchanged; exact mode does not inherit the legacy
/// scalar-force density screens. Density coefficients and kernel parameters
/// are held fixed, and real/imaginary Frobenius contributions add with + signs.
pub fn complex_exchange_nuclear_derivative(
    mb: &MolBasis,
    geometry: &PeriodicGeometry,
    natoms: usize,
    density: ComplexImageDensity<'_>,
    cutoff: f64,
    kernel: CoulombKernel,
    coefficient: f64,
) -> Result<ExchangeNuclearDerivative> {
    if !cutoff.is_finite()
        || cutoff <= 0.0
        || !coefficient.is_finite()
        || natoms == 0
        || mb.nao_sph == 0
        || mb.shells.is_empty()
        || mb.shells.iter().any(|s| s.atom >= natoms)
        || density.real.is_empty()
    {
        return Err(DftError::InvalidInput("complex exchange derivative needs a finite coefficient, positive cutoff, atoms and AO basis".into()));
    }
    match kernel {
        CoulombKernel::Coulomb => {}
        CoulombKernel::Erf { omega } => {
            if !omega.is_finite() || omega < 0.0 {
                return Err(DftError::InvalidInput(
                    "erf range must be finite and nonnegative".into(),
                ));
            }
        }
        _ => {
            return Err(DftError::Unsupported {
                feature: "this complex exchange derivative kernel".into(),
                missing: "a validated full or nonnegative-range erf Coulomb kernel".into(),
            })
        }
    }
    let real_keys = validate_images(mb, density.real, 1.0)?;
    if real_keys != validate_images(mb, density.imaginary, -1.0)? {
        return Err(DftError::InvalidInput(
            "complex exchange density components have different image sets".into(),
        ));
    }
    let cells = ExchangeCells::for_geometry(mb, geometry, cutoff)?;
    let (mut gradient, mut strain) = exchange_gradient_on_cells(
        mb,
        natoms,
        &cells,
        (density.real, density.real),
        kernel,
        coefficient,
        ExchangeDerivativeScreening::Exact,
    );
    let (imaginary_gradient, imaginary_strain) = exchange_gradient_on_cells(
        mb,
        natoms,
        &cells,
        (density.imaginary, density.imaginary),
        kernel,
        coefficient,
        ExchangeDerivativeScreening::Exact,
    );
    for (g, im) in gradient.iter_mut().zip(imaginary_gradient) {
        *g += im;
    }
    for (row, imaginary) in strain.iter_mut().zip(imaginary_strain) {
        for (entry, im) in row.iter_mut().zip(imaginary) {
            *entry += im;
        }
    }
    if gradient
        .iter()
        .any(|g| !g.x.is_finite() || !g.y.is_finite() || !g.z.is_finite())
        || strain.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(DftError::LinearAlgebra(
            "nonfinite complex exchange derivative".into(),
        ));
    }
    Ok(ExchangeNuclearDerivative {
        gradient,
        strain_derivative: strain,
    })
}

pub mod stationary;
pub use stationary::{DimensionalNuclearGradient, SpinorNuclearGradient};

pub mod stress;
pub use stress::{DimensionalStress, SpinorStress};

#[cfg(test)]
pub(crate) use crate::acceptance::components;
#[cfg(test)]
pub(crate) use crate::acceptance::stencils;
#[cfg(test)]
#[path = "spinor_gradient/tests.rs"]
mod tests;
