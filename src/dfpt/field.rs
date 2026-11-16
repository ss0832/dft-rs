// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Screened periodic electric/magnetic Fourier response in independent
//! shifted Bloch spaces. Number current includes its explicit contact term.

use super::density::{PeriodicDensityResponse, PeriodicDensitySolver, PeriodicResponseOptions};
use crate::math::Vec3;
use crate::periodic::field::{periodic_field_operators, PeriodicFieldOptions};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::{BlochMatrix, Cell, PeriodicGeometry};
use crate::response::complex::Complex;
use crate::response::ResponseReport;
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};

#[derive(Clone, Debug, Default)]
pub struct PeriodicFieldResponseOptions {
    pub response: PeriodicResponseOptions,
    /// Electron scalar potential energy, V = -phi, in Hartree.
    pub scalar_potential: Complex,
    /// Reduced vector potential in the Hamiltonian (p+A)^2/2.
    pub vector_potential: [Complex; 3],
}

#[derive(Clone, Debug)]
pub struct PeriodicFieldResponse {
    pub density_response: PeriodicDensityResponse,
    pub k_points: Vec<Vec3>,
    pub target_k_points: Vec<Vec3>,
    pub scalar_potential: Complex,
    pub vector_potential: [Complex; 3],
    pub electric_field: [Complex; 3],
    pub magnetic_field: [Complex; 3],
    /// Complex cell-integrated density variation at the input wave vector.
    pub number_density: Complex,
    pub charge_density: Complex,
    pub paramagnetic_number_current: [Complex; 3],
    pub diamagnetic_number_current: [Complex; 3],
    pub number_current: [Complex; 3],
    pub charge_current: [Complex; 3],
    pub reference_electrons: f64,
    /// Raw |z delta_n - q.delta_j| in the actual finite AO basis.
    /// It is not projected away or used to change a physical denominator.
    pub continuity_defect: f64,
}

/// Hilbert-Schmidt contraction Tr(vertex^dagger density), including all
/// imaginary parts; a same-q density is not a Hermitian matrix.
pub fn fourier_observable(
    vertex: &[BlochMatrix],
    density: &[Vec<BlochMatrix>; 2],
) -> Result<Complex> {
    if vertex.is_empty()
        || density.iter().any(|s| s.len() != vertex.len())
        || vertex.iter().enumerate().any(|(k, v)| {
            v.re.rows != v.im.rows
                || v.re.cols != v.im.cols
                || density.iter().any(|s| {
                    let d = &s[k];
                    d.re.rows != v.re.rows
                        || d.re.cols != v.re.cols
                        || d.im.rows != v.re.rows
                        || d.im.cols != v.re.cols
                })
        })
    {
        return Err(DftError::InvalidInput(
            "invalid periodic observable mesh or complex AO shapes".into(),
        ));
    }
    let mut result = Complex::ZERO;
    for spin in density {
        for (v, d) in vertex.iter().zip(spin) {
            for i in 0..v.rows() {
                for j in 0..v.cols() {
                    let a = Complex::new(v.re[(i, j)], v.im[(i, j)]);
                    let b = Complex::new(d.re[(i, j)], d.im[(i, j)]);
                    if !a.is_finite() || !b.is_finite() {
                        return Err(DftError::InvalidInput(
                            "nonfinite periodic observable matrix".into(),
                        ));
                    }
                    result += a.conj() * b;
                }
            }
        }
    }
    Ok(result / vertex.len() as f64)
}

/// Common Rust/CLI/native Python/ASE field response driver. The SCF plan is
/// checked by the density solver; bra operators are constructed at k+q even
/// for a noncommensurate transfer. Hartree/XC screening is solved by GMRES.
pub fn periodic_field_response(
    cell: &Cell,
    functional: &ResolvedFunctional,
    reference: &PeriodicScfResult,
    scf_options: &PeriodicScfOptions,
    options: &PeriodicFieldResponseOptions,
) -> Result<ResponseReport<PeriodicFieldResponse>> {
    if !options.scalar_potential.is_finite()
        || options.vector_potential.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "periodic field amplitudes must be finite".into(),
        ));
    }
    if functional.family == Family::MetaGga {
        return Err(DftError::Unsupported {
            feature: "periodic meta-GGA field/current response".into(),
            missing:
                "gauge-covariant kinetic-density sources, explicit current and contact kernels"
                    .into(),
        });
    }
    if !cell.basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "periodic ECP field/current response".into(),
            missing: "the nonlocal-core covariant current and vector-potential vertices".into(),
        });
    }
    let solver =
        PeriodicDensitySolver::new(cell, functional, reference, scf_options, &options.response)?;
    let geometry = PeriodicGeometry::new(
        cell.lattice().clone(),
        crate::periodic::features::PeriodicBoundary::FULL_3D,
    )?;
    let fields = periodic_field_operators(
        &cell.molecule,
        &cell.basis,
        &geometry,
        &PeriodicFieldOptions {
            k_mesh: scf_options.k_mesh,
            // Minimal coupling uses the same one-electron image reach as
            // S/T. The induced Hartree kernel retains its independent,
            // longer AO-product reach and reciprocal convergence cutoff.
            basis_cutoff: cell.overlap_cutoff(),
            wave: options.response.q,
            frequency: options.response.frequency,
            scalar_potential: options.scalar_potential,
            vector_potential: options.vector_potential,
            ..Default::default()
        },
    )?;
    if fields.k_points != reference.k_points {
        return Err(DftError::InvalidInput(
            "field vertices and reference integration meshes differ".into(),
        ));
    }
    let report = solver.solve(&[fields.first_order.clone(), fields.first_order.clone()])?;
    let number_density = fourier_observable(&fields.density_vertex, &report.value.density)?;
    let mut paramagnetic = [Complex::ZERO; 3];
    for axis in 0..3 {
        paramagnetic[axis] =
            fourier_observable(&fields.current_vertex[axis], &report.value.density)?;
    }
    // Observable differentiation pairs the probe at -q with A at +q,
    // giving N(0), not N(2q). The latter belongs to two source derivatives.
    let reference_electrons = fourier_observable(&reference.overlap, &reference.density)?.re;
    let diamagnetic = options.vector_potential.map(|a| a * reference_electrons);
    let number_current = std::array::from_fn(|a| paramagnetic[a] + diamagnetic[a]);
    let charge_current = number_current.map(|j| -j);
    let z = options.response.frequency.energy()?;
    let divergence = (0..3).fold(Complex::ZERO, |s, a| {
        s + number_current[a] * options.response.q.get(a)
    });
    let continuity_defect = (z * number_density - divergence).abs();
    let mut warnings = report.warnings;
    warnings.push(
        "Current is the reduced electron number current; electric charge \
        density/current has the opposite sign. Reports are integrated per \
        primitive cell, not divided by its volume."
            .into(),
    );
    warnings.push(
        "Continuity defect is measured in the finite Gaussian basis. A screened \
        response residual does not certify basis gauge completeness; converge \
        the AO and reciprocal cutoffs separately."
            .into(),
    );
    warnings.push(
        "The zero-field collinear HF/KS model includes orbital minimal \
        coupling, adiabatic semilocal density screening and the \
        self-consistent full/range-separated exchange kernel. Spin Zeeman, \
        current-dependent XC, spinor/SOC and flux-compatible finite-field \
        references require their additional physical kernels."
            .into(),
    );
    if options.response.q == Vec3::zero() && z == Complex::ZERO {
        warnings.push(
            "Static uniform potentials do not represent a uniform electric or \
            magnetic field; Berry electric response and magnetic flux cells remain \
            separate calculations."
                .into(),
        );
    }
    Ok(ResponseReport {
        value: PeriodicFieldResponse {
            density_response: report.value, k_points: fields.k_points,
            target_k_points: fields.target_k_points,
            scalar_potential: options.scalar_potential, vector_potential: options.vector_potential,
            electric_field: fields.electric_field, magnetic_field: fields.magnetic_field,
            number_density, charge_density: -number_density,
            paramagnetic_number_current: paramagnetic,
            diamagnetic_number_current: diamagnetic, number_current, charge_current,
            reference_electrons, continuity_defect,
        },
        shape: report.shape, functional: report.functional, reference: report.reference,
        units: "per-cell induced electron number/current; reduced potentials: Hartree and momentum; q: inverse Bohr; frequency: Hartree",
        model: "3D zero-field collinear orbital minimal-coupling response; reciprocal Hartree, adiabatic LDA/GGA density and full/range-separated exact-exchange kernels",
        axis_convention: "density[spin][ket k][bra AO at k+q][ket AO at k]; Cartesian current x,y,z; exp(i q.r-i z.t)",
        converged: report.converged, residuals: report.residuals, iterations: report.iterations, warnings,
    })
}
