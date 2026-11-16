// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Screened orbital electric/magnetic response in a physical periodic subspace.
//! The Bloch transfer is explicitly fractional; folded uniform number constraints
//! never rely on a small Cartesian wave or spectral denominator floor.

use super::density::{band_space, reference_diagnostics};
use super::field::fourier_observable;
use super::orbitals::{adjoint, generalized_orbitals, matmul, BandOrbitals};
use super::spectral::{BandPair, SpectralPropagator};
use super::xc::FiniteQXc;
use crate::basis::MolBasis;
use crate::math::Vec3;
use crate::periodic::coulomb_boundary::DimensionalCoulombBackend;
use crate::periodic::field::{periodic_field_operators, PeriodicFieldOptions};
use crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
use crate::periodic::quasiperiodic::PeriodicTransfer;
use crate::periodic::scf_boundary::{
    DimensionalScfBackend, DimensionalScfOptions, DimensionalScfReport,
};
use crate::periodic::{BlochMatrix, PeriodicGeometry};
use crate::response::complex::Complex;
use crate::response::diagnostics::ReferenceDiagnostics;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::{Frequency, ResponseReport};
use crate::system::Molecule;
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct DimensionalFieldResponseOptions {
    /// Coordinates in reciprocal lattice vectors; open axes must be exactly zero.
    /// Integer coordinates label a uniform Bloch sector but a microscopic field.
    pub transfer_fractional: [f64; 3],
    pub frequency: Frequency,
    pub fixed_number: bool,
    /// Electron scalar potential V=-phi, in Hartree.
    pub scalar_potential: Complex,
    /// Reduced vector potential in momentum units; p -> p+A.
    pub vector_potential: [Complex; 3],
    pub solver: ResponseSolverConfig,
}
impl Default for DimensionalFieldResponseOptions {
    fn default() -> Self {
        Self {
            transfer_fractional: [0.0; 3],
            frequency: Frequency::Static,
            fixed_number: true,
            scalar_potential: Complex::ZERO,
            vector_potential: [Complex::ZERO; 3],
            solver: ResponseSolverConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DimensionalFieldResponse {
    pub transfer_fractional: [f64; 3],
    pub reduced_transfer_fractional: [f64; 3],
    pub wave: Vec3,
    pub reduced_transfer: Vec3,
    pub frequency: Frequency,
    pub k_points: Vec<Vec3>,
    pub target_k_points: Vec<Vec3>,
    pub density: [Vec<BlochMatrix>; 2],
    pub scalar_potential: Complex,
    pub vector_potential: [Complex; 3],
    pub electric_field: [Complex; 3],
    pub magnetic_field: [Complex; 3],
    pub number_density: Complex,
    pub charge_density: Complex,
    pub paramagnetic_number_current: [Complex; 3],
    pub diamagnetic_number_current: [Complex; 3],
    pub number_current: [Complex; 3],
    pub charge_current: [Complex; 3],
    pub reference_electrons: f64,
    pub temperature_hartree: f64,
    pub fixed_number: bool,
    pub occupation_consistency_defect: f64,
    /// max component difference between n and 2n open Fourier quadratures,
    /// applied to the final spin-summed AO response; no projected residual.
    pub hartree_quadrature_defect: f64,
    pub continuity_defect: f64,
    /// Numerical stationary-reference validation, not a provider-name match.
    pub reference_operator_defect: f64,
}

pub struct DimensionalDensitySolver<'a> {
    ket: [Vec<BandOrbitals>; 2],
    bra: [Vec<BandOrbitals>; 2],
    propagators: [SpectralPropagator; 2],
    coulomb: &'a DimensionalCoulombBackend,
    transfer: PeriodicTransfer,
    xc: Option<FiniteQXc>,
    nao: usize,
    pub reference_diagnostics: ReferenceDiagnostics,
    pub occupation_consistency_defect: f64,
    pub reference_operator_defect: f64,
}
impl<'a> DimensionalDensitySolver<'a> {
    pub fn new(
        backend: &'a DimensionalScfBackend<'_>,
        basis: &MolBasis,
        geometry: &PeriodicGeometry,
        functional: &ResolvedFunctional,
        reference: &DimensionalScfReport,
        scf_options: &DimensionalScfOptions,
        options: &DimensionalFieldResponseOptions,
    ) -> Result<Self> {
        options.frequency.validate()?;
        options.solver.validate()?;
        let transfer = PeriodicTransfer::new(geometry, options.transfer_fractional)?;
        if functional.hybrid.cx != 0.0 || functional.hybrid.clr != 0.0 {
            return Err(DftError::Unsupported {
                feature: "dimensional finite-q hybrid response".into(),
                missing: "complex covariant exact-exchange kernel".into(),
            });
        }
        let r = &reference.reference;
        let nk = r.k_points.len();
        if nk == 0
            || r.k_points != backend.k_points()
            || r.overlap.len() != nk
            || r.density.iter().any(|s| s.len() != nk)
            || r.fock.iter().any(|s| s.len() != nk)
            || r.occupations.iter().any(|s| s.len() != nk)
            || reference.boundary != geometry.boundary()
        {
            return Err(DftError::InvalidInput(
                "dimensional reference mesh or boundary mismatch".into(),
            ));
        }
        let fresh = backend.evaluate(&r.density)?;
        let defect = |a: &BlochMatrix, b: &BlochMatrix| {
            a.re.as_slice()
                .iter()
                .zip(b.re.as_slice())
                .chain(a.im.as_slice().iter().zip(b.im.as_slice()))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        };
        if r.overlap
            .iter()
            .chain(r.fock.iter().flatten())
            .any(|m| m.rows() != basis.nao_sph || m.cols() != basis.nao_sph)
        {
            return Err(DftError::InvalidInput(
                "dimensional reference AO shape mismatch".into(),
            ));
        }
        let reference_operator_defect = r
            .overlap
            .iter()
            .zip(backend.overlap())
            .map(|(a, b)| defect(a, b))
            .chain(
                r.fock
                    .iter()
                    .flatten()
                    .zip(fresh.fock.iter().flatten())
                    .map(|(a, b)| defect(a, b)),
            )
            .fold(0.0, f64::max);
        if !reference_operator_defect.is_finite()
            || reference_operator_defect > scf_options.operator_tolerance
        {
            return Err(DftError::InvalidInput("reference operators do not match the requested dimensional energy model and quadrature".into()));
        }
        let mut ket: [Vec<_>; 2] = std::array::from_fn(|_| Vec::new());
        for spin in 0..2 {
            for ik in 0..nk {
                ket[spin].push(generalized_orbitals(
                    &r.fock[spin][ik],
                    &r.overlap[ik],
                    scf_options.scf.orth_threshold,
                )?);
            }
        }
        let points: Vec<_> = r
            .k_points
            .iter()
            .map(|&k| k + transfer.reduced_cartesian())
            .collect();
        let bra = if transfer.is_uniform() {
            ket.clone()
        } else {
            let (overlap, fock) = backend.reference_fock_at(&r.density, &points)?;
            let mut bra: [Vec<_>; 2] = std::array::from_fn(|_| Vec::new());
            for spin in 0..2 {
                for ik in 0..nk {
                    bra[spin].push(generalized_orbitals(
                        &fock[spin][ik],
                        &overlap[ik],
                        scf_options.scf.orth_threshold,
                    )?);
                }
            }
            bra
        };
        let make = |spin: usize| -> Result<SpectralPropagator> {
            let pairs = (0..nk)
                .map(|ik| {
                    Ok(BandPair {
                        ket: band_space(
                            ket[spin][ik].energies.clone(),
                            r,
                            spin,
                            scf_options.scf.occupation,
                        )?,
                        bra: band_space(
                            bra[spin][ik].energies.clone(),
                            r,
                            spin,
                            scf_options.scf.occupation,
                        )?,
                        weight: 1.0 / nk as f64,
                    })
                })
                .collect::<Result<_>>()?;
            SpectralPropagator::new(
                pairs,
                transfer.reduced_fractional(),
                options.frequency,
                options.fixed_number,
            )
        };
        let propagators = [make(0)?, make(1)?];
        let xc = backend
            .xc_grid()
            .map(|grid| -> Result<_> {
                let shifted = grid.at_dimensional_k_points(
                    basis,
                    geometry,
                    &points,
                    scf_options
                        .scf
                        .xc_ao_cutoff
                        .unwrap_or(scf_options.basis_cutoff),
                )?;
                FiniteQXc::new(functional, grid, &shifted, &r.density)
            })
            .transpose()?;
        let (reference_diagnostics, occupation_consistency_defect) =
            reference_diagnostics(r, &ket)?;
        Ok(Self {
            ket,
            bra,
            propagators,
            coulomb: backend.coulomb_backend(),
            transfer,
            xc,
            nao: basis.nao_sph,
            reference_diagnostics,
            occupation_consistency_defect,
            reference_operator_defect,
        })
    }
    pub fn potential_to_bands(&self, potential: &[Vec<BlochMatrix>; 2]) -> Result<Vec<Complex>> {
        let mut out = Vec::with_capacity(self.dimension());
        for spin in 0..2 {
            if potential[spin].len() != self.ket[spin].len() {
                return Err(DftError::InvalidInput(
                    "dimensional source mesh mismatch".into(),
                ));
            }
            for (ik, v) in potential[spin].iter().enumerate() {
                if v.rows() != self.nao || v.cols() != self.nao {
                    return Err(DftError::InvalidInput(
                        "dimensional source AO shape mismatch".into(),
                    ));
                }
                let m = matmul(
                    &adjoint(&self.bra[spin][ik].coefficients)?,
                    &matmul(v, &self.ket[spin][ik].coefficients)?,
                )?;
                for i in 0..m.rows() {
                    for j in 0..m.cols() {
                        out.push(Complex::new(m.re[(i, j)], m.im[(i, j)]));
                    }
                }
            }
        }
        Ok(out)
    }
    pub fn density_to_aos(&self, density: &[Complex]) -> Result<[Vec<BlochMatrix>; 2]> {
        if density.len() != self.dimension() || density.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "invalid dimensional band density".into(),
            ));
        }
        let mut out: [Vec<_>; 2] = std::array::from_fn(|_| Vec::new());
        let mut offset = 0;
        for spin in 0..2 {
            for ik in 0..self.ket[spin].len() {
                let a = &self.bra[spin][ik].coefficients;
                let b = &self.ket[spin][ik].coefficients;
                let mut d = BlochMatrix::zeros(a.cols(), b.cols());
                for i in 0..d.rows() {
                    for j in 0..d.cols() {
                        let z = density[offset];
                        offset += 1;
                        d.re[(i, j)] = z.re;
                        d.im[(i, j)] = z.im;
                    }
                }
                out[spin].push(matmul(&matmul(a, &d)?, &adjoint(b)?)?);
            }
        }
        Ok(out)
    }
    pub fn independent_response(&self, potential: &[Complex]) -> Result<Vec<Complex>> {
        if potential.len() != self.dimension() {
            return Err(DftError::InvalidInput(
                "dimensional band potential shape mismatch".into(),
            ));
        }
        let split = self.propagators[0].dimension();
        let mut out = self.propagators[0].apply(&potential[..split])?;
        out.extend(self.propagators[1].apply(&potential[split..])?);
        Ok(out)
    }
    fn hartree(&self, density: &[Vec<BlochMatrix>; 2], coarse: bool) -> Result<Vec<BlochMatrix>> {
        let mut total = density[0].clone();
        for (p, b) in total.iter_mut().zip(&density[1]) {
            p.add_scaled(b, 1.0);
        }
        self.coulomb
            .finite_q_hartree(&self.transfer, &total, coarse)
    }
    pub fn induced_potential(&self, density: &[Complex]) -> Result<Vec<Complex>> {
        let ao = self.density_to_aos(density)?;
        let h = self.hartree(&ao, false)?;
        let mut v = [h.clone(), h];
        if let Some(xc) = &self.xc {
            let x = xc.apply(&ao)?;
            for spin in 0..2 {
                for (v, x) in v[spin].iter_mut().zip(&x[spin]) {
                    v.add_scaled(x, 1.0);
                }
            }
        }
        self.potential_to_bands(&v)
    }
}
impl LinearOperator for DimensionalDensitySolver<'_> {
    fn dimension(&self) -> usize {
        self.propagators.iter().map(|p| p.dimension()).sum()
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }
    fn apply(&self, density: &[Complex]) -> Result<Vec<Complex>> {
        let induced = self.independent_response(&self.induced_potential(density)?)?;
        Ok(density.iter().zip(induced).map(|(&d, i)| d - i).collect())
    }
}

/// Common screened field driver. The stationary reference is validated against
/// the requested operators; q need not belong to the integration mesh.
pub fn dimensional_field_response(
    molecule: &Molecule,
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    functional: &ResolvedFunctional,
    reference: &DimensionalScfReport,
    scf_options: &DimensionalScfOptions,
    options: &DimensionalFieldResponseOptions,
) -> Result<ResponseReport<DimensionalFieldResponse>> {
    if !basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "dimensional ECP field/current response".into(),
            missing: "gauge-covariant local/nonlocal ECP field vertices and contact terms; \
                ECP ground-state and shifted-reference operators are available"
                .into(),
        });
    }
    if !options.scalar_potential.is_finite()
        || options.vector_potential.iter().any(|a| !a.is_finite())
    {
        return Err(DftError::InvalidInput(
            "dimensional field amplitudes must be finite".into(),
        ));
    }
    if functional.family == Family::MetaGga {
        return Err(DftError::Unsupported {
            feature: "dimensional meta-GGA field/current response".into(),
            missing: "gauge-covariant kinetic-density, explicit current and contact kernels".into(),
        });
    }
    let transfer = PeriodicTransfer::new(geometry, options.transfer_fractional)?;
    let backend = DimensionalScfBackend::new(molecule, basis, geometry, functional, scf_options)?;
    let solver = DimensionalDensitySolver::new(
        &backend,
        basis,
        geometry,
        functional,
        reference,
        scf_options,
        options,
    )?;
    let fields = periodic_field_operators(
        molecule,
        basis,
        geometry,
        &PeriodicFieldOptions {
            k_mesh: scf_options.scf.k_mesh,
            basis_cutoff: scf_options.basis_cutoff,
            wave: transfer.cartesian(),
            frequency: options.frequency,
            scalar_potential: options.scalar_potential,
            vector_potential: options.vector_potential,
            ..Default::default()
        },
    )?;
    let rhs = solver.independent_response(
        &solver.potential_to_bands(&[fields.first_order.clone(), fields.first_order.clone()])?,
    )?;
    let solution = krylov::solve(&solver, &rhs, None, &options.solver)?;
    let density = solver.density_to_aos(&solution.amplitudes)?;
    let coarse = solver.hartree(&density, true)?;
    let fine = solver.hartree(&density, false)?;
    let quadrature = coarse
        .iter()
        .zip(&fine)
        .map(|(a, b)| {
            a.re.as_slice()
                .iter()
                .zip(b.re.as_slice())
                .chain(a.im.as_slice().iter().zip(b.im.as_slice()))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    let number_density = fourier_observable(&fields.density_vertex, &density)?;
    let mut paramagnetic = [Complex::ZERO; 3];
    for a in 0..3 {
        paramagnetic[a] = fourier_observable(&fields.current_vertex[a], &density)?;
    }
    let reference_electrons =
        fourier_observable(&reference.reference.overlap, &reference.reference.density)?.re;
    let diamagnetic = options.vector_potential.map(|a| a * reference_electrons);
    let number_current = std::array::from_fn(|a| paramagnetic[a] + diamagnetic[a]);
    let continuity_defect = (options.frequency.energy()? * number_density
        - (0..3).fold(Complex::ZERO, |s, a| {
            s + number_current[a] * transfer.cartesian().get(a)
        }))
    .abs();
    let mut warnings = reference.warnings.clone();
    warnings.push(
        "Zero-field collinear orbital minimal coupling with adiabatic LDA/GGA \
        screening; spin Zeeman, current XC, SOC, static uniform Berry fields \
        and magnetic flux references retain separate physical kernels."
            .into(),
    );
    warnings.push(
        "Numbers and currents are integrated per primitive cell. The raw \
        continuity defect measures finite-basis gauge completeness \
        independently of GMRES convergence; converge all AO, lattice and grid \
        controls."
            .into(),
    );
    if quadrature > scf_options.electrostatics.tolerance {
        warnings.push("Final Hartree quadrature defect exceeds the requested tolerance.".into());
    }
    let temperature = scf_options.scf.occupation.temperature_kelvin() * BOLTZMANN_HARTREE_PER_K;
    if temperature > 0.0 {
        warnings.push(
            "Thermal projector non-idempotency is physical; inspect occupation_consistency_defect."
                .into(),
        );
    }
    let converged = reference.converged
        && solution.relative_residual <= options.solver.acceptance()
        && quadrature <= scf_options.electrostatics.tolerance;
    Ok(ResponseReport {
        value:DimensionalFieldResponse {
            transfer_fractional:transfer.fractional(),reduced_transfer_fractional:transfer.reduced_fractional(),
            wave:transfer.cartesian(),reduced_transfer:transfer.reduced_cartesian(),frequency:options.frequency,
            k_points:reference.reference.k_points.clone(),target_k_points:reference.reference.k_points.iter().map(|&k|k+transfer.reduced_cartesian()).collect(),
            density,scalar_potential:options.scalar_potential,vector_potential:options.vector_potential,
            electric_field:fields.electric_field,magnetic_field:fields.magnetic_field,number_density,charge_density:-number_density,
            paramagnetic_number_current:paramagnetic,diamagnetic_number_current:diamagnetic,number_current,charge_current:number_current.map(|j|-j),
            reference_electrons,temperature_hartree:temperature,fixed_number:options.fixed_number,
            occupation_consistency_defect:solver.occupation_consistency_defect,hartree_quadrature_defect:quadrature,
            continuity_defect,reference_operator_defect:solver.reference_operator_defect,
        },
        shape:vec![2,reference.reference.k_points.len(),basis.nao_sph,basis.nao_sph],functional:functional.description.clone(),
        reference:solver.reference_diagnostics.clone(),
        units:"per-cell induced electron number/current; scalar Hartree; reduced vector momentum; wave inverse Bohr; frequency Hartree",
        model:"1D/2D/3D zero-field collinear orbital minimal coupling; dimensional quasiperiodic Hartree and adiabatic LDA/GGA screening",
        axis_convention:"density[spin][ket k][bra AO at k+reduced q][ket AO at k]; field wave is unfolded G+q; exp(i wave.r-i z.t)",
        converged,residuals:vec![solution.relative_residual],iterations:vec![solution.iterations],warnings,
    })
}
