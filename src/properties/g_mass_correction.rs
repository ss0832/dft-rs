// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear coefficients of the isotropic relativistic mass correction to g.
//! Gauge correction and spin-orbit response are separate required terms;
//! this module does not return a complete electronic g tensor.

use crate::basis::MolBasis;
use crate::constants::{ELECTRON_G_FACTOR, FINE_STRUCTURE};
use crate::response::nuclear::kinetic_operators::nuclear_kinetic_operator_path;
use crate::response::nuclear::one_body::{nuclear_one_body_coefficients, OneBodyCoefficients};
use crate::response::nuclear::NuclearTaylor;
use crate::{DftError, Result};

/// Return `-alpha^2*g_e/(2S) Tr[(P_alpha-P_beta) T]` along the same moving
/// nuclear path as the supplied stationary response. Coefficient k is the
/// derivative/k!, in inverse bohr^k. The caller must enforce the reference
/// domain, fixed spin normalization and response convergence separately.
pub fn mass_correction_nuclear_coefficients(
    basis: &MolBasis,
    direction: &[[f64; 3]],
    response: &NuclearTaylor,
    twice_spin: usize,
    order: usize,
) -> Result<OneBodyCoefficients> {
    if twice_spin == 0 || direction != response.direction.as_slice() {
        return Err(DftError::InvalidInput(
            "g mass correction requires positive spin and the actual response direction".into(),
        ));
    }
    if response.field_direction.iter().any(|value| *value != 0.0) {
        return Err(DftError::Unsupported {
            feature: "mixed electric-field g mass correction".into(),
            missing: "mixed-field mass-correction acceptance".into(),
        });
    }
    let operators = nuclear_kinetic_operator_path(basis, direction, order)?;
    let mut result = nuclear_one_body_coefficients(
        &response.density_coefficients,
        &response.density_corrections,
        &operators,
        [1.0, -1.0],
    )?;
    let factor = -FINE_STRUCTURE.powi(2) * ELECTRON_G_FACTOR / twice_spin as f64;
    for value in &mut result.coefficients {
        *value *= factor;
    }
    for bound in &mut result.accumulation_bounds {
        *bound *= factor.abs();
    }
    Ok(result)
}
