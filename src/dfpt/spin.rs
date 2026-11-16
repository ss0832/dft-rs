// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Longitudinal finite-q spin-Zeeman response of a collinear reference.
//! This local spin probe has no orbital vector potential or transverse SOC.

use super::density::{PeriodicDensityResponse, PeriodicDensitySolver, PeriodicResponseOptions};
use super::field::fourier_observable;
use crate::math::Vec3;
use crate::periodic::field::{periodic_field_operators, PeriodicFieldOptions};
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::{BlochMatrix, Cell, PeriodicGeometry};
use crate::response::complex::Complex;
use crate::response::ResponseReport;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct PeriodicLongitudinalSpinResponse {
    pub density_response: PeriodicDensityResponse,
    /// Fourier amplitude B_z in the reduced magnetic atomic unit of (p+A)^2/2.
    pub magnetic_field: Complex,
    pub spin_g_factor: f64,
    /// Induced electron numbers per primitive cell, not divided by its volume.
    pub spin_numbers: [Complex; 2],
    pub total_number: Complex,
    /// delta M_z = -g (delta N_alpha - delta N_beta)/4 in atomic units.
    pub magnetic_moment: Complex,
    /// delta M_z / B_z; undefined for an exactly zero probe amplitude.
    pub susceptibility: Option<Complex>,
}

/// H'_alpha(q)=+g B_z N(q)/4, H'_beta(q)=-g B_z N(q)/4.
/// The spin-only local source reuses the full density/spin screening kernel.
/// Uniform static Pauli equilibrium needs a common number multiplier and is
/// deliberately left to the existing static susceptibility driver.
pub fn periodic_longitudinal_spin_response(
    cell: &Cell,
    functional: &ResolvedFunctional,
    reference: &PeriodicScfResult,
    scf_options: &PeriodicScfOptions,
    options: &PeriodicResponseOptions,
    magnetic_field: Complex,
    spin_g_factor: f64,
) -> Result<ResponseReport<PeriodicLongitudinalSpinResponse>> {
    if !magnetic_field.is_finite() || !spin_g_factor.is_finite() || spin_g_factor <= 0.0 {
        return Err(DftError::InvalidInput(
            "longitudinal spin response needs a finite field and positive finite spin g factor"
                .into(),
        ));
    }
    let bloch_uniform = cell.lattice().a.iter().all(|a| {
        let coordinate = a.dot(options.q) / (2.0 * std::f64::consts::PI);
        (coordinate - coordinate.round()).abs() <= 64.0 * f64::EPSILON * coordinate.abs().max(1.0)
    });
    let thermal = matches!(scf_options.occupation, OccupationScheme::FermiDirac { temperature_kelvin } if temperature_kelvin>0.0);
    if options.frequency.energy()? == Complex::ZERO
        && (options.q == Vec3::zero() || (thermal && bloch_uniform))
    {
        return Err(DftError::Unsupported {
            feature: "static Bloch-uniform longitudinal spin probe without a common number multiplier".into(),
            missing: "a common-total-number equilibrium ensemble; use pauli-spin-susceptibility with exchange_enhanced=true for a spatially uniform field. A general intra-cell reciprocal-lattice spin probe at finite temperature needs its own common multiplier; the equilibrium static limit differs from collisionless finite-frequency response".into(),
        });
    }
    let mut request = options.clone();
    // At nonzero q the perturbation has zero macroscopic average. At finite
    // frequency the diagonal collisionless occupation response vanishes.
    // Neither case uses a separate equilibrium spin-number constraint.
    request.fixed_number = false;
    let solver = PeriodicDensitySolver::new(cell, functional, reference, scf_options, &request)?;
    let geometry = PeriodicGeometry::new(
        cell.lattice().clone(),
        crate::periodic::features::PeriodicBoundary::FULL_3D,
    )?;
    let vertices = periodic_field_operators(
        cell.molecule(),
        cell.basis(),
        &geometry,
        &PeriodicFieldOptions {
            k_mesh: scf_options.k_mesh,
            basis_cutoff: cell.overlap_cutoff(),
            wave: request.q,
            frequency: request.frequency,
            scalar_potential: Complex::ONE,
            ..Default::default()
        },
    )?;
    let source: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
        vertices
            .density_vertex
            .iter()
            .map(|v| {
                let factor =
                    magnetic_field * (if spin == 0 { 1.0 } else { -1.0 }) * (spin_g_factor / 4.0);
                let mut matrix = BlochMatrix::zeros(v.rows(), v.cols());
                for i in 0..v.rows() {
                    for j in 0..v.cols() {
                        let entry = factor * Complex::new(v.re[(i, j)], v.im[(i, j)]);
                        matrix.re[(i, j)] = entry.re;
                        matrix.im[(i, j)] = entry.im;
                    }
                }
                matrix
            })
            .collect()
    });
    let report = solver.solve(&source)?;
    let zero = vertices
        .density_vertex
        .iter()
        .map(|v| BlochMatrix::zeros(v.rows(), v.cols()))
        .collect::<Vec<_>>();
    let spin_numbers = [
        fourier_observable(
            &vertices.density_vertex,
            &[report.value.density[0].clone(), zero.clone()],
        )?,
        fourier_observable(
            &vertices.density_vertex,
            &[zero, report.value.density[1].clone()],
        )?,
    ];
    let moment = (spin_numbers[0] - spin_numbers[1]) * (-spin_g_factor / 4.0);
    let susceptibility = if magnetic_field == Complex::ZERO {
        None
    } else {
        Some(moment / magnetic_field)
    };
    let mut warnings = report.warnings;
    warnings.push("Longitudinal spin-only Zeeman probe; orbital minimal coupling, transverse spin, spin-orbit coupling and spin relaxation are not included.".into());
    warnings.push("In this collinear spin-conserving model a spatially uniform finite-frequency spin field has zero collisionless response. It is not the equilibrium Pauli limit; do not impose their equality.".into());
    Ok(ResponseReport {
        value: PeriodicLongitudinalSpinResponse {
            density_response: report.value, magnetic_field, spin_g_factor,
            spin_numbers, total_number: spin_numbers[0]+spin_numbers[1],
            magnetic_moment: moment, susceptibility,
        },
        shape: report.shape, functional: report.functional, reference: report.reference,
        units: "per-cell electron number and atomic magnetic moment; magnetic field in reduced atomic units; frequency Hartree; q inverse Bohr",
        model: "3D collinear longitudinal spin-Zeeman response with adiabatic self-consistent density/spin screening; no orbital or transverse magnetic response",
        axis_convention: "spin alpha,beta; density[spin][ket k][bra AO at k+q][ket AO at k]; exp(i q.r-i z.t); H'_s=+/-g B_z N(q)/4",
        converged: report.converged, residuals: report.residuals, iterations: report.iterations, warnings,
    })
}
