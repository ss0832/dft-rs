// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Longitudinal spin susceptibility of a collinear thermal reference,
//! including the self-consistent spin XC kernel. Spin populations exchange
//! particles at fixed total number and fixed reference spin chemical-potential
//! difference. A common chemical-potential response imposes the number
//! constraint without imposing pointwise cancellation of the spin densities.

use crate::dfpt::density::{PeriodicDensitySolver, PeriodicResponseOptions};
use crate::dfpt::orbitals::matmul;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::Cell;
use crate::response::krylov::{LinearOperator, ResponseSolverConfig};
use crate::response::ResponseReport;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct LongitudinalSpinSusceptibility {
    /// dM/dB per cell in atomic units, M = -g(N_alpha-N_beta)/4.
    pub susceptibility: f64,
    pub bare_susceptibility: f64,
    pub enhancement_factor: Option<f64>,
    pub spin_g_factor: f64,
    /// Per atomic magnetic field; no spin or charge projection is applied.
    pub delta_numbers: [f64; 2],
    /// Unconstrained Zeeman response before the common multiplier is solved.
    pub unconstrained_delta_numbers: [f64; 2],
    /// d common mu / dB, in Hartree per atomic magnetic field. None means
    /// that all retained bands are inert and the multiplier is undetermined.
    pub chemical_potential_response: Option<f64>,
    pub bare_chemical_potential_response: Option<f64>,
    /// d total N / d common mu per cell, with and without the induced kernel.
    pub charge_compressibility: f64,
    pub bare_charge_compressibility: f64,
    pub reference_chemical_potentials: [Option<f64>; 2],
    /// (mu_alpha - mu_beta)/2, held fixed during the response.
    pub reference_spin_bias: Option<f64>,
    pub charge_constraint_residual: f64,
    pub bare_charge_constraint_residual: f64,
    /// Fresh residual of the constrained source and the actual density operator.
    pub constrained_equation_residual: f64,
    /// Raw pointwise sum of spin density responses; generally nonzero for
    /// unequal channels, even though its overlap trace is constrained to zero.
    pub spin_density_cancellation_residual: f64,
    pub density_hermitian_residual: f64,
    pub density: [Vec<BlochMatrix>; 2],
    /// Negative response flags an instability of this stationary reference.
    pub nonnegative_susceptibility: bool,
}

struct NumberConstrainedDensity {
    density: [Vec<BlochMatrix>; 2],
    multiplier: Option<f64>,
    compressibility: f64,
    unconstrained_numbers: [f64; 2],
}

fn constrain_number(
    field: &[Vec<BlochMatrix>; 2],
    reservoir: &[Vec<BlochMatrix>; 2],
    overlap: &[BlochMatrix],
) -> Result<NumberConstrainedDensity> {
    let unconstrained_numbers = numbers(field, overlap);
    let reservoir_numbers = numbers(reservoir, overlap);
    let compressibility = reservoir_numbers.iter().sum::<f64>();
    let unconstrained_number = unconstrained_numbers.iter().sum::<f64>();
    let multiplier = if compressibility == 0.0 {
        if unconstrained_number != 0.0 {
            return Err(DftError::InvalidInput(
                "a vanishing common charge compressibility cannot carry the required fixed-number spin response".into(),
            ));
        }
        None
    } else {
        Some(-unconstrained_number / compressibility)
    };
    if !compressibility.is_finite() || multiplier.is_some_and(|mu| !mu.is_finite()) {
        return Err(DftError::InvalidInput(
            "unrepresentable common chemical-potential spin response".into(),
        ));
    }
    let mut density = field.clone();
    if let Some(mu) = multiplier {
        for (p, reservoir) in density.iter_mut().flatten().zip(reservoir.iter().flatten()) {
            p.add_scaled(reservoir, mu);
        }
    }
    Ok(NumberConstrainedDensity {
        density,
        multiplier,
        compressibility,
        unconstrained_numbers,
    })
}

fn constrained_residual(
    solver: &PeriodicDensitySolver,
    overlap: &[BlochMatrix],
    source: &[Vec<BlochMatrix>; 2],
    constrained: &NumberConstrainedDensity,
) -> Result<f64> {
    let metric_density: [Vec<_>; 2] = [0, 1]
        .map(|spin| {
            constrained.density[spin]
                .iter()
                .zip(overlap)
                .map(|(p, s)| matmul(s, &matmul(p, s)?))
                .collect::<Result<_>>()
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| DftError::InvalidInput("invalid constrained spin dimensions".into()))?;
    let amplitudes = solver.potential_to_bands(&metric_density)?;
    let mut external = source.clone();
    for spin in &mut external {
        for (v, s) in spin.iter_mut().zip(overlap) {
            v.add_scaled(s, -constrained.multiplier.unwrap_or(0.0));
        }
    }
    let rhs = solver.independent_response(&solver.potential_to_bands(&external)?)?;
    let image = solver.apply(&amplitudes)?;
    let residual = rhs
        .iter()
        .zip(image)
        .fold(0.0_f64, |norm, (&b, a)| norm.hypot((b - a).abs()));
    let rhs_norm = rhs.iter().fold(0.0_f64, |norm, z| norm.hypot(z.abs()));
    Ok(if rhs_norm == 0.0 {
        residual
    } else {
        residual / rhs_norm
    })
}

fn numbers(density: &[Vec<BlochMatrix>; 2], overlap: &[BlochMatrix]) -> [f64; 2] {
    std::array::from_fn(|spin| {
        density[spin]
            .iter()
            .zip(overlap)
            .map(|(p, s)| p.trace_product(s))
            .sum::<f64>()
            / overlap.len() as f64
    })
}

/// Static spin-only susceptibility at fixed total number and fixed baseline
/// spin bias. Two grand-canonical responses determine the physical common
/// charge multiplier. The bare result uses the same orbitals, mesh and
/// occupations, with its own number multiplier and the induced kernel removed.
pub fn longitudinal_spin_susceptibility(
    cell: &Cell,
    functional: &ResolvedFunctional,
    reference: &PeriodicScfResult,
    scf_options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
    reciprocal_cutoff: f64,
    spin_g_factor: f64,
) -> Result<ResponseReport<LongitudinalSpinSusceptibility>> {
    if !spin_g_factor.is_finite() || spin_g_factor <= 0.0 {
        return Err(DftError::InvalidInput(
            "spin_g_factor must be positive and finite".into(),
        ));
    }
    if !matches!(scf_options.occupation, OccupationScheme::FermiDirac { temperature_kelvin } if temperature_kelvin.is_finite() && temperature_kelvin > 0.0)
    {
        return Err(DftError::InvalidInput(
            "exchange-enhanced Pauli response needs a positive-temperature Fermi-Dirac reference"
                .into(),
        ));
    }
    let options = PeriodicResponseOptions {
        fixed_number: false,
        reciprocal_cutoff,
        solver: solver.clone(),
        ..PeriodicResponseOptions::default()
    };
    let response_solver =
        PeriodicDensitySolver::new(cell, functional, reference, scf_options, &options)?;
    let zeeman = spin_g_factor / 4.0;
    let source: [Vec<_>; 2] = std::array::from_fn(|spin| {
        reference
            .overlap
            .iter()
            .cloned()
            .map(|mut s| {
                s.scale(if spin == 0 { zeeman } else { -zeeman });
                s
            })
            .collect()
    });
    let bare = response_solver.density_to_aos(
        &response_solver.independent_response(&response_solver.potential_to_bands(&source)?)?,
    )?;
    let reservoir_source: [Vec<_>; 2] = std::array::from_fn(|_| {
        reference
            .overlap
            .iter()
            .cloned()
            .map(|mut s| {
                s.scale(-1.0);
                s
            })
            .collect()
    });
    let bare_reservoir = response_solver.density_to_aos(
        &response_solver
            .independent_response(&response_solver.potential_to_bands(&reservoir_source)?)?,
    )?;
    let response = response_solver.solve(&source)?;
    let reservoir_response = response_solver.solve(&reservoir_source)?;
    let constrained = constrain_number(
        &response.value.density,
        &reservoir_response.value.density,
        &reference.overlap,
    )?;
    let constrained_bare = constrain_number(&bare, &bare_reservoir, &reference.overlap)?;
    let equation_residual =
        constrained_residual(&response_solver, &reference.overlap, &source, &constrained)?;
    let population = numbers(&constrained.density, &reference.overlap);
    let bare_population = numbers(&constrained_bare.density, &reference.overlap);
    let susceptibility = -zeeman * (population[0] - population[1]);
    let bare_susceptibility = -zeeman * (bare_population[0] - bare_population[1]);
    let charge_residual = (population[0] + population[1]).abs();
    let bare_charge_residual = (bare_population[0] + bare_population[1]).abs();
    let mut spin_residual = 0.0_f64;
    let mut hermitian_residual = 0.0_f64;
    for (a, b) in constrained.density[0].iter().zip(&constrained.density[1]) {
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                spin_residual = spin_residual
                    .max((a.re[(i, j)] + b.re[(i, j)]).abs())
                    .max((a.im[(i, j)] + b.im[(i, j)]).abs());
                for p in [a, b] {
                    hermitian_residual = hermitian_residual
                        .max((p.re[(i, j)] - p.re[(j, i)]).abs())
                        .max((p.im[(i, j)] + p.im[(j, i)]).abs());
                }
            }
        }
    }
    let mut warnings = response.warnings;
    for warning in reservoir_response.warnings {
        if !warnings.contains(&warning) {
            warnings.push(warning);
        }
    }
    if susceptibility < 0.0 {
        warnings.push("Negative longitudinal susceptibility flags a spin instability of the stationary reference; it is not projected onto a stable value.".into());
    }
    if charge_residual > solver.tolerance {
        warnings
            .push("The raw total-number response exceeds the requested response tolerance.".into());
    }
    if constrained.compressibility < 0.0 {
        warnings.push("Negative common charge compressibility flags a charge instability of the stationary reference; its sign and the physical constraint are retained.".into());
    }
    if equation_residual > solver.acceptance() {
        warnings.push("The freshly evaluated constrained density-equation residual exceeds the response acceptance bound.".into());
    }
    let mut residuals = response.residuals;
    residuals.extend(reservoir_response.residuals);
    residuals.push(equation_residual);
    let mut iterations = response.iterations;
    iterations.extend(reservoir_response.iterations);
    Ok(ResponseReport {
        value: LongitudinalSpinSusceptibility { susceptibility, bare_susceptibility,
            enhancement_factor: if bare_susceptibility != 0.0 { Some(susceptibility / bare_susceptibility) } else { None },
            spin_g_factor, delta_numbers: population, charge_constraint_residual: charge_residual,
            unconstrained_delta_numbers: constrained.unconstrained_numbers,
            chemical_potential_response: constrained.multiplier,
            bare_chemical_potential_response: constrained_bare.multiplier,
            charge_compressibility: constrained.compressibility,
            bare_charge_compressibility: constrained_bare.compressibility,
            reference_chemical_potentials: reference.chemical_potential,
            reference_spin_bias: reference.chemical_potential[0].zip(reference.chemical_potential[1]).map(|(a,b)| 0.5*(a-b)),
            bare_charge_constraint_residual: bare_charge_residual,
            constrained_equation_residual: equation_residual,
            spin_density_cancellation_residual: spin_residual, density_hermitian_residual: hermitian_residual,
            density: constrained.density, nonnegative_susceptibility: susceptibility >= 0.0 },
        shape: vec![1], functional: response.functional, reference: response.reference,
        units: "atomic units of spin magnetic moment per magnetic field per cell",
        model: "3D static adiabatic semilocal collinear thermal spin response; fixed total number and fixed baseline spin chemical-potential difference with a common charge multiplier; spin populations exchange particles; spin-only Zeeman term, no orbital or SOC contribution",
        axis_convention: "longitudinal spin quantization axis; M=-g(N_alpha-N_beta)/4, F_alpha^B=+g S/4, F_beta^B=-g S/4",
        converged: response.converged && reservoir_response.converged && charge_residual <= solver.tolerance && bare_charge_residual <= solver.tolerance && equation_residual <= solver.acceptance(),
        residuals, iterations, warnings,
    })
}
