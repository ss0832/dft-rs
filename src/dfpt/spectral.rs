// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Finite-q rectangular spectral density response and complex screened
//! solves. The density connects bra bands at k+q to ket bands at k; no
//! same-q Hermitian projection is made. Number correction is mesh-global
//! and belongs only to the static uniform sector.

use super::thermodynamic::{divided_difference, fermi};
use crate::response::complex::Complex;
use crate::response::krylov::{
    self, LinearOperator, LinearSolution, OperatorStructure, ResponseSolverConfig,
};
use crate::response::Frequency;
use crate::{DftError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct BandSpace {
    pub energies: Vec<f64>,
    pub occupations: Vec<f64>,
    pub thermal: Option<(f64, f64)>,
}

impl BandSpace {
    pub fn integer(energies: Vec<f64>, occupied: usize) -> Result<Self> {
        if occupied > energies.len() {
            return Err(DftError::InvalidInput(
                "occupied band count exceeds band space".into(),
            ));
        }
        let occupations = (0..energies.len())
            .map(|i| if i < occupied { 1.0 } else { 0.0 })
            .collect();
        let result = Self {
            energies,
            occupations,
            thermal: None,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn thermal(energies: Vec<f64>, mu: f64, temperature: f64) -> Result<Self> {
        let occupations = energies
            .iter()
            .map(|&e| fermi(e, mu, temperature).map(|p| p.0))
            .collect::<Result<_>>()?;
        let result = Self {
            energies,
            occupations,
            thermal: Some((mu, temperature)),
        };
        result.validate()?;
        Ok(result)
    }
    fn validate(&self) -> Result<()> {
        if self.energies.is_empty()
            || self.energies.len() != self.occupations.len()
            || self.energies.iter().any(|v| !v.is_finite())
            || self
                .occupations
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(DftError::InvalidInput("invalid spectral band space".into()));
        }
        for (&e, &f) in self.energies.iter().zip(&self.occupations) {
            if let Some((mu, temperature)) = self.thermal {
                if (fermi(e, mu, temperature)?.0 - f).abs() > 1e-14 {
                    return Err(DftError::InvalidInput(
                        "occupations do not match the thermal distribution".into(),
                    ));
                }
            } else if f != 0.0 && f != 1.0 {
                return Err(DftError::InvalidInput(
                    "fractional occupations require a thermal reference".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BandPair {
    pub bra: BandSpace,
    pub ket: BandSpace,
    /// Brillouin-zone integration weight, not a per-band weight.
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct SpectralPropagator {
    pairs: Vec<BandPair>,
    transfer: [f64; 3],
    frequency: Frequency,
    fixed_number: bool,
}

impl SpectralPropagator {
    pub fn new(
        pairs: Vec<BandPair>,
        transfer: [f64; 3],
        frequency: Frequency,
        fixed_number: bool,
    ) -> Result<Self> {
        frequency.validate()?;
        if pairs.is_empty() || transfer.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "invalid spectral mesh or transfer".into(),
            ));
        }
        let mut total_weight = 0.0;
        let thermal = pairs[0].ket.thermal;
        for pair in &pairs {
            pair.bra.validate()?;
            pair.ket.validate()?;
            if !pair.weight.is_finite()
                || pair.weight <= 0.0
                || pair.bra.thermal != thermal
                || pair.ket.thermal != thermal
            {
                return Err(DftError::InvalidInput(
                    "spectral pairs require positive weights and one thermal distribution".into(),
                ));
            }
            total_weight += pair.weight;
        }
        if (total_weight - 1.0).abs() > 1e-12 {
            return Err(DftError::InvalidInput(
                "spectral mesh weights must sum to one".into(),
            ));
        }
        let result = Self {
            pairs,
            transfer,
            frequency,
            fixed_number,
        };
        if result.fixed_number && result.transfer == [0.0; 3] && !result.is_uniform() {
            return Err(DftError::InvalidInput(
                "fixed-number uniform response needs identical bra/ket band spaces".into(),
            ));
        }
        Ok(result)
    }
    pub fn dimension(&self) -> usize {
        self.pairs
            .iter()
            .map(|p| p.bra.energies.len() * p.ket.energies.len())
            .sum()
    }
    pub fn shapes(&self) -> Vec<(usize, usize)> {
        self.pairs
            .iter()
            .map(|p| (p.bra.energies.len(), p.ket.energies.len()))
            .collect()
    }
    pub fn is_uniform(&self) -> bool {
        self.transfer == [0.0; 3] && self.pairs.iter().all(|p| p.bra == p.ket)
    }
    pub fn apply(&self, potential: &[Complex]) -> Result<Vec<Complex>> {
        if potential.len() != self.dimension() || potential.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput("invalid spectral potential".into()));
        }
        let z = self.frequency.energy()?;
        let static_sector = z == Complex::ZERO;
        let uniform = self.is_uniform();
        let mut density = vec![Complex::ZERO; potential.len()];
        let mut offset = 0;
        let mut trace = Complex::ZERO;
        let mut trace_slope = 0.0;
        for pair in &self.pairs {
            let rows = pair.bra.energies.len();
            let cols = pair.ket.energies.len();
            for a in 0..rows {
                for b in 0..cols {
                    let ea = pair.bra.energies[a];
                    let eb = pair.ket.energies[b];
                    let gap = ea - eb;
                    let coefficient = if static_sector {
                        if let Some((mu, temperature)) = pair.ket.thermal {
                            Complex::from(divided_difference(ea, eb, mu, temperature)?)
                        } else if gap == 0.0 {
                            if pair.bra.occupations[a] != pair.ket.occupations[b] {
                                return Err(DftError::LinearAlgebra("integer response is singular between degenerate bands of different occupation".into()));
                            }
                            Complex::ZERO
                        } else {
                            ((pair.bra.occupations[a] - pair.ket.occupations[b]) / gap).into()
                        }
                    } else {
                        let difference = if let Some((mu, temperature)) = pair.ket.thermal {
                            -gap * divided_difference(ea, eb, mu, temperature)?
                        } else {
                            pair.ket.occupations[b] - pair.bra.occupations[a]
                        };
                        if difference == 0.0 {
                            Complex::ZERO
                        } else {
                            Complex::from(difference) / (z - Complex::from(gap))
                        }
                    };
                    let index = offset + a * cols + b;
                    density[index] = coefficient * potential[index];
                    if static_sector && uniform && a == b {
                        trace += density[index] * pair.weight;
                        trace_slope += coefficient.re * pair.weight;
                    }
                }
            }
            offset += rows * cols;
        }
        if static_sector && uniform && self.fixed_number && trace_slope != 0.0 {
            let delta_mu = trace / trace_slope;
            let mut offset = 0;
            for pair in &self.pairs {
                let n = pair.ket.energies.len();
                if let Some((mu, temperature)) = pair.ket.thermal {
                    for a in 0..n {
                        density[offset + a * n + a] -=
                            delta_mu * fermi(pair.ket.energies[a], mu, temperature)?.1;
                    }
                }
                offset += n * n;
            }
        }
        if density.iter().any(|v| !v.is_finite()) {
            return Err(DftError::LinearAlgebra(
                "singular or non-finite spectral density response".into(),
            ));
        }
        Ok(density)
    }
}

/// Density-to-induced-potential kernel in the same flattened band-pair
/// layout. Implementations must include all physical interaction channels.
pub trait DensityKernel {
    fn apply(&self, density: &[Complex]) -> Result<Vec<Complex>>;
}

struct Screened<'a, K> {
    propagator: &'a SpectralPropagator,
    kernel: &'a K,
}
impl<K: DensityKernel> LinearOperator for Screened<'_, K> {
    fn dimension(&self) -> usize {
        self.propagator.dimension()
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }
    fn apply(&self, density: &[Complex]) -> Result<Vec<Complex>> {
        let induced = self.kernel.apply(density)?;
        let image = self.propagator.apply(&induced)?;
        Ok(density.iter().zip(image).map(|(&p, chi)| p - chi).collect())
    }
}

/// Solve delta P = chi_0 (V_ext + K delta P) without constructing chi_0 K.
pub fn screened_response(
    propagator: &SpectralPropagator,
    kernel: &impl DensityKernel,
    external: &[Complex],
    config: &ResponseSolverConfig,
) -> Result<LinearSolution> {
    let rhs = propagator.apply(external)?;
    krylov::solve(&Screened { propagator, kernel }, &rhs, None, config)
}
