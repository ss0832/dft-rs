// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic Fermi-Dirac occupation response at fixed particle number or
//! chemical potential. All temperatures here are k_B T in Hartree.

use crate::{DftError, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ensemble {
    FixedNumber { delta_number: f64 },
    FixedChemicalPotential { delta_mu: f64 },
}

pub fn fermi(energy: f64, mu: f64, temperature: f64) -> Result<(f64, f64)> {
    if !energy.is_finite() || !mu.is_finite() || !temperature.is_finite() || temperature <= 0.0 {
        return Err(DftError::InvalidInput(
            "Fermi response requires finite energy/mu and positive k_B T".into(),
        ));
    }
    let x = (energy - mu) / temperature;
    let tail = (-x.abs()).exp();
    let occupation = if x >= 0.0 {
        tail / (1.0 + tail)
    } else {
        1.0 / (1.0 + tail)
    };
    let derivative = -tail / (temperature * (1.0 + tail).powi(2));
    Ok((occupation, derivative))
}

/// Stable (f(a)-f(b))/(a-b), including the exactly degenerate limit.
/// The scaled hyperbolic identity avoids subtracting nearly full bands.
pub fn divided_difference(a: f64, b: f64, mu: f64, temperature: f64) -> Result<f64> {
    fermi(a, mu, temperature)?;
    fermi(b, mu, temperature)?;
    let gap = a - b;
    if gap.abs() <= 1e-5 * temperature {
        return Ok(fermi(0.5 * a + 0.5 * b, mu, temperature)?.1);
    }
    let midpoint = (0.5 * a + 0.5 * b - mu) / temperature;
    let delta = (0.5 * a - 0.5 * b) / temperature;
    let scale = midpoint.abs().max(delta.abs());
    let scaled_cosh = |x: f64| 0.5 * ((x.abs() - scale).exp() + (-x.abs() - scale).exp());
    let scaled_sinh =
        delta.signum() * 0.5 * ((delta.abs() - scale).exp() - (-delta.abs() - scale).exp());
    let value = -scaled_sinh / ((scaled_cosh(midpoint) + scaled_cosh(delta)) * gap);
    if !value.is_finite() {
        return Err(DftError::InvalidInput(
            "unrepresentable Fermi divided difference".into(),
        ));
    }
    Ok(value)
}

#[derive(Clone, Debug)]
pub struct OccupationResponse {
    pub delta_occupations: Vec<f64>,
    pub delta_mu: Option<f64>,
    pub delta_number: f64,
}

pub fn occupation_response(
    energies: &[f64],
    weights: &[f64],
    delta_energies: &[f64],
    mu: f64,
    temperature: f64,
    delta_temperature: f64,
    ensemble: Ensemble,
) -> Result<OccupationResponse> {
    let n = energies.len();
    if n == 0
        || weights.len() != n
        || delta_energies.len() != n
        || !delta_temperature.is_finite()
        || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        || !weights.iter().any(|v| *v > 0.0)
        || delta_energies.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "invalid weighted occupation-response inputs".into(),
        ));
    }
    let slopes: Vec<_> = energies
        .iter()
        .map(|&e| fermi(e, mu, temperature).map(|p| p.1))
        .collect::<Result<_>>()?;
    let forcing: Vec<_> = energies
        .iter()
        .zip(delta_energies)
        .map(|(&e, &de)| de - (e - mu) * delta_temperature / temperature)
        .collect();
    let denominator: f64 = weights.iter().zip(&slopes).map(|(w, s)| w * s).sum();
    let unconstrained: f64 = weights
        .iter()
        .zip(&slopes)
        .zip(&forcing)
        .map(|((w, s), f)| w * s * f)
        .sum();
    let shift = match ensemble {
        Ensemble::FixedNumber { delta_number } => {
            if !delta_number.is_finite() {
                return Err(DftError::InvalidInput("non-finite number source".into()));
            }
            if denominator == 0.0 {
                if delta_number != 0.0 {
                    return Err(DftError::InvalidInput(
                        "no thermally active band can carry the requested number change".into(),
                    ));
                }
                None
            } else {
                Some((unconstrained - delta_number) / denominator)
            }
        }
        Ensemble::FixedChemicalPotential { delta_mu } => {
            if !delta_mu.is_finite() {
                return Err(DftError::InvalidInput("non-finite reservoir source".into()));
            }
            Some(delta_mu)
        }
    };
    let delta_occupations: Vec<_> = slopes
        .iter()
        .zip(forcing)
        .map(|(s, f)| s * (f - shift.unwrap_or(0.0)))
        .collect();
    let delta_number: f64 = weights
        .iter()
        .zip(&delta_occupations)
        .map(|(w, f)| w * f)
        .sum();
    if delta_occupations.iter().any(|v| !v.is_finite())
        || !delta_number.is_finite()
        || shift.is_some_and(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "unrepresentable occupation response".into(),
        ));
    }
    Ok(OccupationResponse {
        delta_occupations,
        delta_mu: shift,
        delta_number,
    })
}
