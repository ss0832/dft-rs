// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Canonical phase selection for a fixed-band variational singlet model.

use super::critical_temperature::leading_pairing_mode;
use super::{
    anomalous_factor, solve_gap_taylor, validate_data, GapFunctional, GapFunctionalData, GapState,
    GapTaylor, NormalBandSpectrum, SuppliedBandFunctional,
};
use crate::dual::{Scalar, Taylor};
use crate::response::complex::Complex;
use crate::response::krylov::ResponseSolverConfig;
use crate::{DftError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandPhase {
    Normal,
    Superconducting,
}

#[derive(Clone, Debug)]
pub struct BandPhaseResult {
    pub phase: BandPhase,
    pub solution: GapTaylor,
    /// Normal-state chemical potential in Hartree.
    pub normal_chemical_potential: f64,
    pub normal_number_residual: f64,
    pub normal_pairing_eigenvalue: f64,
    /// Smallest eigenvalue of the identity minus the weighted pairing map.
    /// A paired solution has a zero mode from its global phase.
    pub pairing_stability_margin: f64,
    /// F(selected) - F(normal), in Hartree at the same electron number and kT.
    pub condensation_free_energy: f64,
    /// Anomalous-density interaction contribution to F, in Hartree.
    pub pairing_energy: f64,
    pub entropy_over_boltzmann: f64,
    pub normal_entropy_over_boltzmann: f64,
}

/// Select a thermodynamic phase of the supplied, fixed-band singlet model.
///
/// A uniform mass factor permits the reciprocal variational interaction
/// K/(1+Z). The Helmholtz functional is sum(w*epsilon*n) + chi^H*w*K/(1+Z)*w*chi
/// minus the independent-quasiparticle entropy term. Each band represents two
/// spin states. `parameters` has the layout of [`SuppliedBandFunctional`].
///
/// A paired stationary solution is accepted only with a negative condensation
/// free energy and a nonnegative pairing certificate within the solver
/// tolerance. If a certified solution cannot be resolved, return an error.
pub fn select_band_phase(
    functional: &SuppliedBandFunctional,
    parameters: &[f64],
    electron_number: f64,
    config: &ResponseSolverConfig,
) -> Result<BandPhaseResult> {
    config.validate()?;
    let n = functional.bands();
    let zero = vec![0.0; n];
    let data = functional.evaluate(parameters, 0.0, &zero, &zero)?;
    validate_data(&data, n)?;
    let z = data.mass_renormalization[0];
    if data.mass_renormalization.iter().any(|&zi| zi != z) {
        return Err(DftError::Unsupported {
            feature: "canonical superconducting phase selection with band-dependent mass".into(),
            missing: "a reciprocal variational free-energy functional consistent with the gap and number equations".into(),
        });
    }
    if (0..n).any(|i| (0..i).any(|j| data.kernel[i * n + j] != data.kernel[j * n + i])) {
        return Err(DftError::InvalidInput(
            "canonical phase selection requires an exactly reciprocal supplied pairing kernel"
                .into(),
        ));
    }
    let normal = NormalBandSpectrum {
        energies: &data.energies,
        weights: &data.weights,
        electron_number,
    };
    let kt = data.temperature_energy;
    let (mu, number_residual) =
        normal.chemical_potential(kt, normal.capacity(n)?, config.acceptance())?;
    let normal_energies: Vec<_> = data.energies.iter().map(|e| (e - mu).abs()).collect();
    let normal_population = pairing_population(&data, normal_energies.iter().map(|e| e * e))?;
    let normal_mode = leading_pairing_mode(&data, &normal_population, config)?;
    let normal_margin = 1.0 - normal_mode.leading_eigenvalue;
    let normal_entropy = entropy(&data, &normal_energies);
    let jets: Vec<_> = parameters
        .iter()
        .copied()
        .map(Taylor::<1>::from_f64)
        .collect();
    let solve = |gap| {
        solve_gap_taylor(
            functional,
            &jets,
            Taylor::from_f64(electron_number),
            &GapState {
                gap,
                chemical_potential: mu,
            },
            0,
            config,
        )
    };
    if normal_margin >= 0.0 {
        return Ok(BandPhaseResult {
            phase: BandPhase::Normal,
            solution: solve(vec![Complex::ZERO; n])?,
            normal_chemical_potential: mu,
            normal_number_residual: number_residual,
            normal_pairing_eigenvalue: normal_mode.leading_eigenvalue,
            pairing_stability_margin: normal_margin,
            condensation_free_energy: 0.0,
            pairing_energy: 0.0,
            entropy_over_boltzmann: normal_entropy,
            normal_entropy_over_boltzmann: normal_entropy,
        });
    }

    // The unstable eigenvector fixes a deterministic relative phase. Use the
    // bound |chi_i| <= 1/2 and the normal-state curvature to scale the seed.
    let max_gap = data
        .kernel
        .chunks_exact(n)
        .map(|row| {
            row.iter()
                .zip(&data.weights)
                .map(|(k, w)| 0.5 * w * (k / (1.0 + z)).abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    let amplitude = max_gap * (-normal_margin / normal_mode.leading_eigenvalue).sqrt();
    let mode_max = normal_mode
        .gap_mode
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    if !amplitude.is_finite() || amplitude <= 0.0 {
        return Err(DftError::LinearAlgebra(
            "non-finite or unresolved seed amplitude for canonical superconducting phase".into(),
        ));
    }
    let mut last_rejection = String::new();
    for factor in [1.0, 2.0, 0.5, 4.0] {
        let gap = normal_mode
            .gap_mode
            .iter()
            .map(|&value| Complex::from((value / mode_max) * (amplitude * factor)))
            .collect();
        let solution = match solve(gap) {
            Ok(solution) => solution,
            Err(error @ DftError::ResponseNotConverged { .. }) => {
                last_rejection = error.to_string();
                continue;
            }
            Err(error) => return Err(error),
        };
        let population = pairing_population(
            &data,
            solution.quasiparticle_energy_squared[0].iter().copied(),
        )?;
        let margin = 1.0 - leading_pairing_mode(&data, &population, config)?.leading_eigenvalue;
        let chi_real = &solution.anomalous_real[0];
        let chi_imaginary = &solution.anomalous_imaginary[0];
        let mut pairing_energy = 0.0;
        for i in 0..n {
            let wi = data.weights[i] / (1.0 + z);
            for j in 0..n {
                pairing_energy += wi
                    * data.kernel[i * n + j]
                    * data.weights[j]
                    * (chi_real[i] * chi_real[j] + chi_imaginary[i] * chi_imaginary[j]);
            }
        }
        let electronic_entropy = entropy(&data, &solution.quasiparticle_energies);
        // Center the one-electron term on the normal chemical potential.
        // Its contribution cancels at fixed particle number.
        let band_energy_change: f64 = data
            .energies
            .iter()
            .zip(&data.weights)
            .zip(&solution.occupations[0])
            .map(|((&e, &w), &occupation)| {
                let f = crate::periodic::occupations::fermi_and_hole((e - mu) / kt).0;
                w * (e - mu) * (occupation - 2.0 * f)
            })
            .sum();
        let free_energy =
            band_energy_change + pairing_energy - kt * (electronic_entropy - normal_entropy);

        // The negative entropy is convex in the squared Bloch radius
        // (1-n)^2+4|chi|^2 and its tangent has slope 1/(4*q).
        // At a stationary solution this bounds every constrained free-energy
        // change by sum(w/q*|delta chi|^2) + delta chi^H*w*K/(1+Z)*w*delta chi
        // and a nonnegative number term. The weighted map tests this bound.
        if margin >= -config.acceptance() && free_energy.is_finite() && free_energy < 0.0 {
            return Ok(BandPhaseResult {
                phase: BandPhase::Superconducting,
                solution,
                normal_chemical_potential: mu,
                normal_number_residual: number_residual,
                normal_pairing_eigenvalue: normal_mode.leading_eigenvalue,
                pairing_stability_margin: margin,
                condensation_free_energy: free_energy,
                pairing_energy,
                entropy_over_boltzmann: electronic_entropy,
                normal_entropy_over_boltzmann: normal_entropy,
            });
        }
        last_rejection =
            format!("pairing stability margin {margin:.6e}, condensation free energy {free_energy:.6e} Hartree");
    }
    Err(DftError::LinearAlgebra(format!(
        "no certified canonical superconducting phase from the unstable normal mode: {last_rejection}"
    )))
}

fn pairing_population(
    data: &GapFunctionalData<f64>,
    energy_squared: impl Iterator<Item = f64>,
) -> Result<Vec<f64>> {
    data.weights
        .iter()
        .zip(energy_squared)
        .map(|(&w, e2)| Ok(w * anomalous_factor(e2, data.temperature_energy)?))
        .collect()
}

fn entropy(data: &GapFunctionalData<f64>, quasiparticle_energies: &[f64]) -> f64 {
    // Two thermal quasiparticles per band. Evaluate the exponential only
    // with a nonpositive argument, including when the occupation underflows.
    data.weights
        .iter()
        .zip(quasiparticle_energies)
        .map(|(&w, &e)| {
            let boltzmann = (-e / data.temperature_energy).exp();
            2.0 * w
                * (boltzmann.ln_1p()
                    + (e * (boltzmann / (1.0 + boltzmann))) / data.temperature_energy)
        })
        .sum()
}
