// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Canonical normal-state singlet pairing instability on fixed normal bands.

use super::{anomalous_factor, validate_data, GapFunctional, GapFunctionalData};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::periodic::occupations::fermi_and_hole;
use crate::response::krylov::ResponseSolverConfig;
use crate::{DftError, Result};

/// Spin-degenerate normal bands. Energies are in Hartree and each weight
/// counts a pair of spin states. The normal spectrum stays fixed during
/// the temperature search.
#[derive(Clone, Copy, Debug)]
pub struct NormalBandSpectrum<'a> {
    pub energies: &'a [f64],
    pub weights: &'a [f64],
    pub electron_number: f64,
}

/// A bracketed crossing of the largest eigenvalue through one.
#[derive(Clone, Debug)]
pub struct CriticalTemperature {
    /// Thermal energy k_B T_c in Hartree.
    pub temperature_energy: f64,
    /// Final cold and hot bounds on k_B T_c in Hartree.
    pub temperature_energy_bracket: [f64; 2],
    pub chemical_potential: f64,
    pub leading_eigenvalue: f64,
    /// A unit Euclidean-norm real eigenmode in the original gap coordinates.
    /// Its amplitude is arbitrary at a linear instability.
    pub gap_mode: Vec<f64>,
    pub occupations: Vec<f64>,
    pub mass_renormalization: Vec<f64>,
    /// Raw eigenvector defect of the original unsymmetrized operator.
    pub eigen_residual: f64,
    /// Raw defect of the original (1 + Z) Delta + K w q Delta = 0.
    pub gap_residual: f64,
    pub number_residual: f64,
    /// Largest difference K_ij - K_ji in Hartree.
    pub kernel_asymmetry: f64,
    /// Count of evaluated temperatures, including both initial endpoints.
    pub evaluations: usize,
}

impl NormalBandSpectrum<'_> {
    pub(super) fn capacity(self, bands: usize) -> Result<f64> {
        let capacity = 2.0 * self.weights.iter().sum::<f64>();
        if self.energies.len() != bands
            || self.weights.len() != bands
            || bands == 0
            || self.energies.iter().any(|x| !x.is_finite())
            || self.weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
            || !capacity.is_finite()
            || !self.electron_number.is_finite()
            || self.electron_number <= 0.0
            || self.electron_number >= capacity
        {
            return Err(DftError::InvalidInput(
                "SCDFT critical temperature requires finite fixed bands, positive weights and a strictly interior electron number".into(),
            ));
        }
        Ok(capacity)
    }

    fn number_residual(self, mu: f64, kt: f64) -> (f64, f64) {
        // Collect filled-state count separately so that thermal tails in a
        // band gap are retained even when a Fermi occupation rounds to one.
        let mut filled = 0.0;
        let mut thermal = 0.0;
        for (&energy, &weight) in self.energies.iter().zip(self.weights) {
            let x = (energy - mu) / kt;
            let (particle, hole) = fermi_and_hole(x);
            if x < 0.0 {
                filled += 2.0 * weight;
                thermal -= 2.0 * weight * hole;
            } else {
                thermal += 2.0 * weight * particle;
            }
        }
        let residual = (filled - self.electron_number) + thermal;
        if residual != 0.0 || filled != self.electron_number {
            return (residual, residual);
        }
        // A zero linear-space residual can conceal underflow in both
        // thermal tails. Compare log weights after subtracting their
        // common leading exponent to retain the sign inside a deep gap.
        let nearest = self
            .energies
            .iter()
            .map(|e| (e - mu).abs())
            .fold(f64::INFINITY, f64::min);
        let mut log_tails = [f64::NEG_INFINITY; 2];
        for (&energy, &weight) in self.energies.iter().zip(self.weights) {
            let distance = (energy - mu).abs();
            let log_tail = weight.ln() - (distance - nearest) / kt - (-distance / kt).exp().ln_1p();
            let slot = usize::from(energy < mu);
            log_tails[slot] = log_add(log_tails[slot], log_tail);
        }
        (residual, log_tails[0] - log_tails[1])
    }

    pub(super) fn chemical_potential(
        self,
        kt: f64,
        capacity: f64,
        acceptance: f64,
    ) -> Result<(f64, f64)> {
        // Shift both band edges by the single-level chemical potential at
        // the requested filling. The lower (upper) bound puts every
        // occupation at or below (above) the target mean occupation.
        let offset = kt * (self.electron_number.ln() - (capacity - self.electron_number).ln());
        let mut low = self.energies.iter().copied().fold(f64::INFINITY, f64::min) + offset;
        let mut high = self
            .energies
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            + offset;
        if !low.is_finite() || !high.is_finite() {
            return Err(DftError::LinearAlgebra(
                "non-finite normal-state SCDFT chemical-potential bounds".into(),
            ));
        }
        for _ in 0..256 {
            let mid = low.midpoint(high);
            let (_, sign) = self.number_residual(mid, kt);
            if sign == 0.0 || mid == low || mid == high {
                break;
            }
            if sign < 0.0 {
                low = mid;
            } else {
                high = mid;
            }
        }
        let (mu, (residual, _)) = [low, low.midpoint(high), high]
            .map(|candidate| {
                let (residual, sign) = self.number_residual(candidate, kt);
                (candidate, (residual.abs(), sign.abs()))
            })
            .into_iter()
            .min_by(|(_, (a, sign_a)), (_, (b, sign_b))| {
                a.total_cmp(b).then_with(|| sign_a.total_cmp(sign_b))
            })
            .unwrap();
        if !residual.is_finite() || residual > acceptance {
            return Err(DftError::LinearAlgebra(format!(
                "normal-state SCDFT number residual {residual:.6e} exceeds {acceptance:.6e}"
            )));
        }
        Ok((mu, residual))
    }
}

fn log_add(first: f64, second: f64) -> f64 {
    let largest = first.max(second);
    if largest == f64::NEG_INFINITY {
        largest
    } else {
        largest + (first.min(second) - largest).exp().ln_1p()
    }
}

pub(super) struct PairingMode {
    pub leading_eigenvalue: f64,
    pub gap_mode: Vec<f64>,
    pub eigen_residual: f64,
    pub gap_residual: f64,
    pub kernel_asymmetry: f64,
}

/// Largest eigenpair of the gap map with the supplied populations w_i q_i.
/// The same weighted operator tests the normal and paired phase curvatures.
pub(super) fn leading_pairing_mode(
    data: &GapFunctionalData<f64>,
    population: &[f64],
    config: &ResponseSolverConfig,
) -> Result<PairingMode> {
    let n = population.len();

    let mass: Vec<_> = data.mass_renormalization.iter().map(|z| 1.0 + z).collect();
    // B = -D K D with D_i = sqrt(w_i q_i / (1 + Z_i)) is similar to
    // the original linearized gap map. The gap vector is M^(-1/2) u
    // with M_i = w_i q_i (1 + Z_i).
    let diagonal: Vec<_> = population
        .iter()
        .zip(&mass)
        .map(|(wq, m)| wq.sqrt() / m.sqrt())
        .collect();
    let mut operator = Matrix::zeros(n, n);
    let mut kernel_asymmetry = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            operator[(i, j)] = -data.kernel[i * n + j] * diagonal[i] * diagonal[j];
            kernel_asymmetry =
                kernel_asymmetry.max((data.kernel[i * n + j] - data.kernel[j * n + i]).abs());
        }
    }
    if operator.as_slice().iter().any(|x| !x.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "non-finite weighted SCDFT linearized gap operator".into(),
        ));
    }
    let (eigenvalues, vectors) = symmetric_eigen(&operator)?;
    let lambda = eigenvalues[n - 1];
    let mut mode: Vec<_> = (0..n)
        .map(|i| vectors[(i, n - 1)] / (population[i].sqrt() * mass[i].sqrt()))
        .collect();
    let norm = mode.iter().fold(0.0_f64, |s, x| s.hypot(*x));
    if !lambda.is_finite() || !norm.is_finite() || norm == 0.0 {
        return Err(DftError::LinearAlgebra(
            "non-finite original SCDFT linearized gap eigenpair".into(),
        ));
    }
    let pivot = mode
        .iter()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap();
    let signed_norm = norm.copysign(*pivot);
    for x in &mut mode {
        *x /= signed_norm;
    }
    let mut eigen_residual = 0.0_f64;
    let mut gap_residual = 0.0_f64;
    for i in 0..n {
        let image = (0..n)
            .map(|j| -data.kernel[i * n + j] * population[j] * mode[j])
            .sum::<f64>();
        let diagonal_image = mass[i] * mode[i];
        // Check the original reciprocal-kernel entries, including the
        // triangle that the symmetric eigensolver does not read.
        eigen_residual = eigen_residual.hypot(image / mass[i] - lambda * mode[i]);
        gap_residual = gap_residual.hypot(diagonal_image - image);
    }
    if !eigen_residual.is_finite() || eigen_residual > config.acceptance() * lambda.abs().max(1.0) {
        return Err(DftError::LinearAlgebra(format!(
            "original SCDFT linearized gap eigenvector residual {eigen_residual:.6e}"
        )));
    }
    Ok(PairingMode {
        leading_eigenvalue: lambda,
        gap_mode: mode,
        eigen_residual,
        gap_residual,
        kernel_asymmetry,
    })
}

/// Solve for the linearized static singlet onset in a caller-supplied
/// cold-to-hot bracket in Hartree (k_B T). The cold endpoint must have
/// eigenvalue at least one and the hot endpoint at most one. The leading
/// eigenvalue is evaluated on the canonical normal reference at every
/// temperature. The bracket selects a crossing if there are several.
///
/// `parameters_at` supplies the functional parameters at each requested
/// thermal energy. The provider's output energies and weights must match
/// the fixed normal bands, while its kernel and mass may change with
/// temperature and the solved chemical potential.
pub fn critical_temperature(
    functional: &impl GapFunctional,
    normal: NormalBandSpectrum<'_>,
    bracket: [f64; 2],
    config: &ResponseSolverConfig,
    parameters_at: impl Fn(f64) -> Vec<f64>,
) -> Result<CriticalTemperature> {
    config.validate()?;
    let n = functional.bands();
    let capacity = normal.capacity(n)?;
    if !bracket[0].is_finite()
        || !bracket[1].is_finite()
        || bracket[0] <= 0.0
        || bracket[0] >= bracket[1]
    {
        return Err(DftError::InvalidInput(
            "SCDFT critical temperature needs a finite, positive, ascending bracket".into(),
        ));
    }
    let zero_gap = vec![0.0; n];
    let at_temperature = |kt| {
        let (mu, number_residual) = normal.chemical_potential(kt, capacity, config.acceptance())?;
        let data = functional.evaluate(&parameters_at(kt), mu, &zero_gap, &zero_gap)?;
        validate_data(&data, n)?;
        if data.energies != normal.energies
            || data.weights != normal.weights
            || data.temperature_energy != kt
        {
            return Err(DftError::InvalidInput(
                "SCDFT critical temperature needs fixed normal bands at the requested thermal energy".into(),
            ));
        }
        let xi: Vec<_> = data.energies.iter().map(|e| e - mu).collect();
        let population: Vec<_> = xi
            .iter()
            .zip(&data.weights)
            .map(|(&x, &w)| Ok(w * anomalous_factor(x * x, kt)?))
            .collect::<Result<_>>()?;
        let mode = leading_pairing_mode(&data, &population, config)?;
        Ok(CriticalTemperature {
            temperature_energy: kt,
            temperature_energy_bracket: [kt; 2],
            chemical_potential: mu,
            leading_eigenvalue: mode.leading_eigenvalue,
            gap_mode: mode.gap_mode,
            occupations: xi.iter().map(|&x| 2.0 * fermi_and_hole(x / kt).0).collect(),
            mass_renormalization: data.mass_renormalization,
            eigen_residual: mode.eigen_residual,
            gap_residual: mode.gap_residual,
            number_residual,
            kernel_asymmetry: mode.kernel_asymmetry,
            evaluations: 0,
        })
    };
    let mut cold = at_temperature(bracket[0])?;
    let mut hot = at_temperature(bracket[1])?;
    if cold.leading_eigenvalue < 1.0
        || hot.leading_eigenvalue > 1.0
        || (cold.leading_eigenvalue == 1.0 && hot.leading_eigenvalue == 1.0)
    {
        return Err(DftError::InvalidInput(format!(
            "SCDFT critical temperature is not bracketed by a pairing-unstable cold endpoint and stable hot endpoint (lambda_cold={:.12e}, lambda_hot={:.12e})",
            cold.leading_eigenvalue, hot.leading_eigenvalue
        )));
    }
    let mut evaluations = 2;
    loop {
        let bounds = [cold.temperature_energy, hot.temperature_energy];
        let cold_error = (cold.leading_eigenvalue - 1.0).abs();
        let hot_error = (hot.leading_eigenvalue - 1.0).abs();
        let best = if cold_error <= hot_error { &cold } else { &hot };
        let narrow = bounds[1] - bounds[0] <= config.tolerance * bounds[1];
        if narrow
            && (best.leading_eigenvalue - 1.0).abs() <= config.acceptance()
            && best.gap_residual <= config.acceptance()
        {
            let mut result = best.clone();
            result.temperature_energy_bracket = bounds;
            result.evaluations = evaluations;
            return Ok(result);
        }
        let midpoint = bounds[0].midpoint(bounds[1]);
        if evaluations - 2 >= config.max_iterations
            || midpoint == bounds[0]
            || midpoint == bounds[1]
        {
            return Err(DftError::ResponseNotConverged {
                iterations: evaluations,
                residual: (best.leading_eigenvalue - 1.0).abs().max(best.gap_residual),
            });
        }
        let trial = at_temperature(midpoint)?;
        if trial.leading_eigenvalue >= 1.0 {
            cold = trial;
        } else {
            hot = trial;
        }
        evaluations += 1;
    }
}
