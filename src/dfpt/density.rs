// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent finite-q density response with analytic Gaussian
//! Hartree, semilocal XC and full/range-separated exchange kernels.
//! Bra and ket orbitals are independent;
//! incommensurate shifted operators use the original SCF integration mesh.

use super::coulomb::FiniteQHartree;
use super::exchange::FiniteQExchange;
use super::orbitals::{adjoint, generalized_orbitals, matmul, BandOrbitals};
use super::spectral::{BandPair, BandSpace, SpectralPropagator};
use super::thermodynamic::{fermi, occupation_response, Ensemble};
use super::xc::FiniteQXc;
use crate::integrals::CoulombKernel;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
use crate::periodic::scf::{reference_fock_at, PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::xc::{AoDerivativeOrder, PeriodicXcGrid};
use crate::periodic::Cell;
use crate::response::complex::Complex;
use crate::response::diagnostics::ReferenceDiagnostics;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::{Frequency, ResponseReport};
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct PeriodicResponseOptions {
    /// Cartesian transfer in inverse Bohr, with no restriction to the SCF mesh.
    pub q: Vec3,
    pub frequency: Frequency,
    /// Static uniform particle numbers are constrained separately by spin,
    /// matching the existing collinear periodic reference convention.
    pub fixed_number: bool,
    /// Spherical cutoff for the complete reciprocal Hartree sum, inverse Bohr.
    pub reciprocal_cutoff: f64,
    pub solver: ResponseSolverConfig,
}
impl Default for PeriodicResponseOptions {
    fn default() -> Self {
        Self {
            q: Vec3::zero(),
            frequency: Frequency::Static,
            fixed_number: true,
            reciprocal_cutoff: 6.0,
            solver: ResponseSolverConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PeriodicDensityResponse {
    pub q: Vec3,
    pub frequency: Frequency,
    pub density: [Vec<BlochMatrix>; 2],
    pub fourier_density: Vec<Complex>,
    pub reciprocal_waves: Vec<Vec3>,
    pub temperature_hartree: f64,
    pub fixed_number: bool,
    pub reciprocal_cutoff: f64,
    /// max |C† S P S C - diag(f)|, including fractional occupations.
    pub occupation_consistency_defect: f64,
}

#[derive(Clone, Debug)]
pub struct DielectricResponse {
    pub q: Vec3,
    pub frequency: Frequency,
    pub temperature_hartree: f64,
    pub fixed_number: bool,
    pub reciprocal_cutoff: f64,
    pub reciprocal_waves: Vec<Vec3>,
    /// Fully screened density response to external scalar potential,
    /// in bohr^-3 / Hartree. Row is density mode; column is source mode.
    pub screened_chi: Vec<Vec<Complex>>,
    /// epsilon^-1 = I + v chi, with v on the density (row) index.
    pub inverse_epsilon: Vec<Vec<Complex>>,
}

/// Uniform static thermodynamic direction. Temperature is k_B T in Hartree;
/// each spin channel uses the same ensemble as the screening operator.
#[derive(Clone, Copy, Debug)]
pub struct ThermalSource {
    pub delta_temperature_hartree: f64,
    pub ensembles: [Ensemble; 2],
}

#[derive(Clone, Debug)]
pub struct PeriodicThermalResponse {
    pub source: ThermalSource,
    pub temperature_hartree: f64,
    pub density: [Vec<BlochMatrix>; 2],
    pub fourier_density: Vec<Complex>,
    pub reciprocal_waves: Vec<Vec3>,
    pub delta_chemical_potentials: [Option<f64>; 2],
    pub delta_numbers: [f64; 2],
    pub occupation_equation_defect: f64,
    pub number_constraint_defects: [f64; 2],
    /// Entropy variation in units of k_B; energies in Hartree.
    pub delta_entropy: f64,
    pub delta_internal_energy: f64,
    pub delta_free_energy: f64,
    pub occupation_consistency_defect: f64,
}

pub struct PeriodicDensitySolver {
    ket: [Vec<BandOrbitals>; 2],
    bra: [Vec<BandOrbitals>; 2],
    propagators: [SpectralPropagator; 2],
    hartree: FiniteQHartree,
    xc: Option<FiniteQXc>,
    exchange: Vec<(f64, FiniteQExchange)>,
    options: PeriodicResponseOptions,
    diagnostics: ReferenceDiagnostics,
    occupation_consistency_defect: f64,
    temperature: f64,
    chemical_potentials: [Option<f64>; 2],
    functional: String,
    nao: usize,
}

pub(crate) fn band_space(
    energies: Vec<f64>,
    reference: &PeriodicScfResult,
    spin: usize,
    occupation: OccupationScheme,
) -> Result<BandSpace> {
    let occupied = if spin == 0 {
        reference.n_alpha
    } else {
        reference.n_beta
    };
    match occupation {
        OccupationScheme::Insulator => BandSpace::integer(energies, occupied),
        OccupationScheme::FermiDirac { temperature_kelvin } => {
            if let Some(mu) = reference.chemical_potential[spin] {
                BandSpace::thermal(energies, mu, temperature_kelvin * BOLTZMANN_HARTREE_PER_K)
            } else if occupied == 0 || occupied == energies.len() {
                BandSpace::integer(energies, occupied)
            } else {
                Err(DftError::InvalidInput(
                    "thermal periodic reference is missing its chemical potential".into(),
                ))
            }
        }
    }
}

impl PeriodicDensitySolver {
    pub fn new(
        cell: &Cell,
        functional: &ResolvedFunctional,
        reference: &PeriodicScfResult,
        scf_options: &PeriodicScfOptions,
        options: &PeriodicResponseOptions,
    ) -> Result<Self> {
        reference.validate_derivative_request(cell, functional, scf_options)?;
        options.frequency.validate()?;
        options.solver.validate()?;
        if !options.q.x.is_finite()
            || !options.q.y.is_finite()
            || !options.q.z.is_finite()
            || !options.reciprocal_cutoff.is_finite()
            || options.reciprocal_cutoff <= 0.0
        {
            return Err(DftError::InvalidInput(
                "invalid periodic response transfer or reciprocal cutoff".into(),
            ));
        }
        if scf_options.skala.is_some() {
            return Err(DftError::Unsupported {
                feature: "finite-q learned-functional density response".into(),
                missing: "the graph kernel between independent shifted Bloch spaces".into(),
            });
        }
        let nk = reference.k_points.len();
        if nk == 0
            || reference.overlap.len() != nk
            || reference.density.iter().any(|s| s.len() != nk)
            || reference.fock.iter().any(|s| s.len() != nk)
            || reference.occupations.iter().any(|s| s.len() != nk)
        {
            return Err(DftError::InvalidInput(
                "invalid periodic reference mesh dimensions".into(),
            ));
        }
        let ket: [Vec<_>; 2] = [0, 1]
            .map(|spin| {
                (0..nk)
                    .map(|ik| {
                        generalized_orbitals(
                            &reference.fock[spin][ik],
                            &reference.overlap[ik],
                            scf_options.orth_threshold,
                        )
                    })
                    .collect::<Result<_>>()
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| DftError::InvalidInput("invalid spin channels".into()))?;
        let shifted_points: Vec<_> = reference.k_points.iter().map(|&k| k + options.q).collect();
        let bra = if options.q == Vec3::zero() {
            ket.clone()
        } else {
            let fock =
                reference_fock_at(cell, functional, reference, scf_options, &shifted_points)?;
            let overlap: Vec<_> = shifted_points
                .iter()
                .map(|&k| BlochMatrix::from_pair(cell.overlap(k)))
                .collect();
            let mut orbitals: [Vec<_>; 2] = std::array::from_fn(|_| Vec::new());
            for spin in 0..2 {
                for ik in 0..nk {
                    orbitals[spin].push(generalized_orbitals(
                        &fock[spin][ik],
                        &overlap[ik],
                        scf_options.orth_threshold,
                    )?);
                }
            }
            orbitals
        };
        let make_propagator = |spin: usize| -> Result<SpectralPropagator> {
            let pairs = (0..nk)
                .map(|ik| {
                    Ok(BandPair {
                        ket: band_space(
                            ket[spin][ik].energies.clone(),
                            reference,
                            spin,
                            scf_options.occupation,
                        )?,
                        bra: band_space(
                            bra[spin][ik].energies.clone(),
                            reference,
                            spin,
                            scf_options.occupation,
                        )?,
                        weight: 1.0 / nk as f64,
                    })
                })
                .collect::<Result<_>>()?;
            SpectralPropagator::new(
                pairs,
                [options.q.x, options.q.y, options.q.z],
                options.frequency,
                options.fixed_number,
            )
        };
        let propagators = [make_propagator(0)?, make_propagator(1)?];
        let cutoff = scf_options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff);
        let xc = if functional.parts.is_empty() {
            None
        } else {
            let grid = PeriodicXcGrid::build(
                cell,
                &reference.k_points,
                &scf_options.grid,
                scf_options.grid_shell,
                cutoff,
                AoDerivativeOrder::for_scf(functional.family),
            )?;
            let shifted = grid.at_k_points(cell, &shifted_points, cutoff)?;
            Some(FiniteQXc::new(
                functional,
                &grid,
                &shifted,
                &reference.density,
            )?)
        };
        let hartree = FiniteQHartree::new(
            cell,
            &reference.k_points,
            options.q,
            options.reciprocal_cutoff,
            // Coulomb AO products extend beyond the short-range metric
            // cutoff. Use the same real-space reach as the reference
            // Hartree build, including images absent from S(k).
            cell.hartree_real_space_cutoff(),
        )?;
        let mut exchange = Vec::new();
        for (coefficient, kernel) in [
            (functional.hybrid.cx, CoulombKernel::Coulomb),
            (
                functional.hybrid.clr,
                CoulombKernel::Erf {
                    omega: functional.hybrid.omega,
                },
            ),
        ] {
            if coefficient != 0.0 {
                exchange.push((
                    coefficient,
                    FiniteQExchange::new(
                        cell,
                        &reference.k_points,
                        options.q,
                        scf_options.exchange_cutoff.unwrap_or(cell.overlap_cutoff),
                        kernel,
                    )?,
                ));
            }
        }
        let (diagnostics, occupation_consistency_defect) = reference_diagnostics(reference, &ket)?;
        Ok(Self {
            ket,
            bra,
            propagators,
            hartree,
            xc,
            exchange,
            options: options.clone(),
            diagnostics,
            occupation_consistency_defect,
            temperature: scf_options.occupation.temperature_kelvin() * BOLTZMANN_HARTREE_PER_K,
            chemical_potentials: reference.chemical_potential,
            functional: functional.description.clone(),
            nao: cell.nao(),
        })
    }

    pub fn reciprocal_waves(&self) -> &[Vec3] {
        &self.hartree.waves
    }
    pub fn plane_wave_source(&self, mode: usize) -> Result<[Vec<BlochMatrix>; 2]> {
        let forms = self.hartree.forms(mode)?.to_vec();
        Ok([forms.clone(), forms])
    }
    /// Transform arbitrary complex AO potentials to the rectangular band
    /// layout. Nuclear/basis sources must additionally provide their metric
    /// and explicit observable derivatives through their perturbation driver.
    pub fn potential_to_bands(&self, potential: &[Vec<BlochMatrix>; 2]) -> Result<Vec<Complex>> {
        let mut output = Vec::with_capacity(self.dimension());
        for spin in 0..2 {
            if potential[spin].len() != self.ket[spin].len() {
                return Err(DftError::InvalidInput(
                    "invalid periodic source mesh".into(),
                ));
            }
            for (ik, v) in potential[spin].iter().enumerate() {
                if v.rows() != self.nao || v.cols() != self.nao {
                    return Err(DftError::InvalidInput(
                        "periodic source AO shape mismatch".into(),
                    ));
                }
                let mo = matmul(
                    &adjoint(&self.bra[spin][ik].coefficients)?,
                    &matmul(v, &self.ket[spin][ik].coefficients)?,
                )?;
                for i in 0..mo.rows() {
                    for j in 0..mo.cols() {
                        output.push(Complex::new(mo.re[(i, j)], mo.im[(i, j)]));
                    }
                }
            }
        }
        Ok(output)
    }
    pub fn density_to_aos(&self, density: &[Complex]) -> Result<[Vec<BlochMatrix>; 2]> {
        if density.len() != self.dimension() || density.iter().any(|x| !x.is_finite()) {
            return Err(DftError::InvalidInput(
                "invalid rectangular band density".into(),
            ));
        }
        let mut output: [Vec<_>; 2] = std::array::from_fn(|_| Vec::new());
        let mut offset = 0;
        for spin in 0..2 {
            for ik in 0..self.ket[spin].len() {
                let left = &self.bra[spin][ik].coefficients;
                let right = &self.ket[spin][ik].coefficients;
                let mut d = BlochMatrix::zeros(left.cols(), right.cols());
                for i in 0..d.rows() {
                    for j in 0..d.cols() {
                        let z = density[offset];
                        d.re[(i, j)] = z.re;
                        d.im[(i, j)] = z.im;
                        offset += 1;
                    }
                }
                output[spin].push(matmul(&matmul(left, &d)?, &adjoint(right)?)?);
            }
        }
        Ok(output)
    }
    pub fn independent_response(&self, potential: &[Complex]) -> Result<Vec<Complex>> {
        if potential.len() != self.dimension() {
            return Err(DftError::InvalidInput(
                "periodic potential dimension mismatch".into(),
            ));
        }
        let split = self.propagators[0].dimension();
        let mut result = self.propagators[0].apply(&potential[..split])?;
        result.extend(self.propagators[1].apply(&potential[split..])?);
        Ok(result)
    }
    pub fn induced_potential(&self, density: &[Complex]) -> Result<Vec<Complex>> {
        let ao = self.density_to_aos(density)?;
        let mut v = self.hartree.apply(&ao)?;
        if let Some(xc) = &self.xc {
            let f = xc.apply(&ao)?;
            for spin in 0..2 {
                for (v, f) in v[spin].iter_mut().zip(&f[spin]) {
                    v.add_scaled(f, 1.0);
                }
            }
        }
        for (coefficient, exchange) in &self.exchange {
            for spin in 0..2 {
                for (potential, image) in v[spin].iter_mut().zip(exchange.apply(&ao[spin])?) {
                    potential.add_scaled(&image, -coefficient);
                }
            }
        }
        self.potential_to_bands(&v)
    }
    pub fn solve(
        &self,
        external: &[Vec<BlochMatrix>; 2],
    ) -> Result<ResponseReport<PeriodicDensityResponse>> {
        let rhs = self.independent_response(&self.potential_to_bands(external)?)?;
        self.solve_band_source(&rhs)
    }

    fn solve_band_source(
        &self,
        rhs: &[Complex],
    ) -> Result<ResponseReport<PeriodicDensityResponse>> {
        let solution = krylov::solve(self, rhs, None, &self.options.solver)?;
        let density = self.density_to_aos(&solution.amplitudes)?;
        let fourier_density = self.hartree.density_fourier(&density)?;
        let mut warnings = Vec::new();
        if self.temperature > 0.0 {
            warnings.push(
                "Reference projector defect includes physical fractional occupations; \
                use occupation_consistency_defect to assess the thermal density."
                    .into(),
            );
        }
        if self.diagnostics.eigen_residual > self.options.solver.tolerance {
            warnings.push(
                "Reference eigenvalue residual exceeds the requested response tolerance.".into(),
            );
        }
        if self.diagnostics.orbital_residual > self.options.solver.tolerance {
            warnings.push(
                "Reference stationarity residual exceeds the requested response tolerance.".into(),
            );
        }
        Ok(ResponseReport {
            value: PeriodicDensityResponse {
                q: self.options.q,
                frequency: self.options.frequency,
                density,
                fourier_density,
                reciprocal_waves: self.hartree.waves.clone(),
                temperature_hartree: self.temperature,
                fixed_number: self.options.fixed_number,
                reciprocal_cutoff: self.options.reciprocal_cutoff,
                occupation_consistency_defect: self.occupation_consistency_defect,
            },
            shape: vec![2, self.ket[0].len(), self.nao, self.nao],
            functional: self.functional.clone(),
            reference: self.diagnostics.clone(),
            units: "AO density response in hartree^-1; Fourier density in bohr^-3/hartree",
            model: "3D adiabatic collinear HF/KS finite-q density response; reciprocal Hartree, semilocal XC and localized full/range-separated exact exchange",
            axis_convention: "density[spin][k][bra AO at k+q][ket AO at k]; Fourier modes G+q",
            converged: true,
            residuals: vec![solution.relative_residual],
            iterations: vec![solution.iterations],
            warnings,
        })
    }

    fn thermal_occupations(
        &self,
        potential: &[Complex],
        source: ThermalSource,
    ) -> Result<(Vec<Complex>, [Option<f64>; 2])> {
        if self.options.q != Vec3::zero()
            || self.options.frequency.energy()? != Complex::ZERO
            || self.temperature <= 0.0
            || !source.delta_temperature_hartree.is_finite()
            || potential.len() != self.dimension()
        {
            return Err(DftError::InvalidInput(
                "thermodynamic response requires uniform static finite-temperature bands and a finite source".into(),
            ));
        }
        let mut output = vec![Complex::ZERO; self.dimension()];
        let mut shifts = [None; 2];
        let mut offset = 0;
        for spin in 0..2 {
            if matches!(source.ensembles[spin], Ensemble::FixedNumber { .. })
                != self.options.fixed_number
            {
                return Err(DftError::InvalidInput(
                    "thermal source ensemble must match the screening ensemble".into(),
                ));
            }
            let mut indices = Vec::new();
            let mut energies = Vec::new();
            let mut delta_energies = Vec::new();
            let mut weights = Vec::new();
            for k in &self.ket[spin] {
                let n = k.energies.len();
                for i in 0..n {
                    indices.push(offset + i * n + i);
                    energies.push(k.energies[i]);
                    delta_energies.push(potential[offset + i * n + i].re);
                    weights.push(1.0 / self.ket[spin].len() as f64);
                }
                offset += n * n;
            }
            if let Some(mu) = self.chemical_potentials[spin] {
                let response = occupation_response(
                    &energies,
                    &weights,
                    &delta_energies,
                    mu,
                    self.temperature,
                    source.delta_temperature_hartree,
                    source.ensembles[spin],
                )?;
                shifts[spin] = response.delta_mu;
                for (index, value) in indices.into_iter().zip(response.delta_occupations) {
                    output[index] = Complex::from(value);
                }
            } else if !matches!(
                source.ensembles[spin],
                Ensemble::FixedNumber { delta_number: 0.0 }
            ) {
                return Err(DftError::InvalidInput(
                    "an empty/full spin channel without a finite chemical potential cannot carry a reservoir or number source".into(),
                ));
            }
        }
        Ok((output, shifts))
    }

    /// Solve self-consistent temperature, particle-number, or chemical-potential
    /// response. Occupation sources and multiplier corrections cover the whole
    /// k mesh. No displaced temperature SCF is used by this production route.
    pub fn solve_thermal(
        &self,
        source: ThermalSource,
    ) -> Result<ResponseReport<PeriodicThermalResponse>> {
        let zero = vec![Complex::ZERO; self.dimension()];
        let (rhs, _) = self.thermal_occupations(&zero, source)?;
        let solution = krylov::solve(self, &rhs, None, &self.options.solver)?;
        let (occupations, shifts) =
            self.thermal_occupations(&self.induced_potential(&solution.amplitudes)?, source)?;
        let density = self.density_to_aos(&solution.amplitudes)?;
        let fourier_density = self.hartree.density_fourier(&density)?;
        let mut numbers = [0.0; 2];
        let mut number_defects = [0.0; 2];
        let mut defect = 0.0_f64;
        let mut entropy = 0.0;
        let mut delta_entropy = 0.0;
        let mut delta_energy = 0.0;
        let mut offset = 0;
        for spin in 0..2 {
            let weight = 1.0 / self.ket[spin].len() as f64;
            for k in &self.ket[spin] {
                let n = k.energies.len();
                for (i, &energy) in k.energies.iter().enumerate() {
                    let index = offset + i * n + i;
                    let df = solution.amplitudes[index];
                    defect = defect.max((df - occupations[index]).abs());
                    numbers[spin] += weight * df.re;
                    delta_energy += weight * energy * df.re;
                    if let Some(mu) = self.chemical_potentials[spin] {
                        let f = fermi(energy, mu, self.temperature)?.0;
                        if f > 0.0 {
                            entropy -= weight * f * f.ln();
                        }
                        if f < 1.0 {
                            entropy -= weight * (1.0 - f) * (1.0 - f).ln();
                        }
                        delta_entropy += weight * (energy - mu) / self.temperature * df.re;
                    }
                }
                offset += n * n;
            }
            if let Ensemble::FixedNumber { delta_number } = source.ensembles[spin] {
                number_defects[spin] = (numbers[spin] - delta_number).abs();
            }
        }
        let mut warnings = Vec::new();
        if self.occupation_consistency_defect > self.options.solver.tolerance {
            warnings.push("Reference occupation consistency exceeds response tolerance.".into());
        }
        if self.diagnostics.eigen_residual > self.options.solver.tolerance
            || self.diagnostics.orbital_residual > self.options.solver.tolerance
        {
            warnings.push("Reference electronic residual exceeds response tolerance.".into());
        }
        Ok(ResponseReport {
            value: PeriodicThermalResponse {
                source, temperature_hartree: self.temperature, density, fourier_density,
                reciprocal_waves: self.hartree.waves.clone(), delta_chemical_potentials: shifts,
                delta_numbers: numbers, occupation_equation_defect: defect,
                number_constraint_defects: number_defects, delta_entropy,
                delta_internal_energy: delta_energy,
                delta_free_energy: delta_energy-self.temperature*delta_entropy
                    -entropy*source.delta_temperature_hartree,
                occupation_consistency_defect: self.occupation_consistency_defect,
            },
            shape: vec![2,self.ket[0].len(),self.nao,self.nao],
            functional: self.functional.clone(), reference: self.diagnostics.clone(),
            units: "density variation dimensionless; Fourier variation bohr^-3; energy variation hartree; entropy variation k_B",
            model: "3D collinear Mermin HF/KS with temperature-independent XC and full/range-separated exact exchange; uniform static thermodynamic response",
            axis_convention: "density[spin][k][bra AO][ket AO]; path-direction variations",
            converged: true, residuals: vec![solution.relative_residual],
            iterations: vec![solution.iterations], warnings,
        })
    }

    /// Compute selected entries of the inverse dielectric matrix while
    /// retaining the complete configured reciprocal set in every induced
    /// Hartree application. Selecting output modes does not truncate the
    /// internal screening space.
    pub fn dielectric(&self, modes: &[usize]) -> Result<ResponseReport<DielectricResponse>> {
        if modes.is_empty()
            || modes.iter().any(|&i| i >= self.hartree.waves.len())
            || modes
                .iter()
                .enumerate()
                .any(|(i, m)| modes[..i].contains(m))
        {
            return Err(DftError::InvalidInput(
                "dielectric modes must be unique valid reciprocal indices".into(),
            ));
        }
        let n = modes.len();
        let mut chi = vec![vec![Complex::ZERO; n]; n];
        let mut inverse = chi.clone();
        let mut residuals = Vec::new();
        let mut iterations = Vec::new();
        let mut warnings = Vec::new();
        for (column, &mode) in modes.iter().enumerate() {
            let report = self.solve(&self.plane_wave_source(mode)?)?;
            residuals.extend(report.residuals);
            iterations.extend(report.iterations);
            for warning in report.warnings {
                if !warnings.contains(&warning) {
                    warnings.push(warning);
                }
            }
            for (row, &output) in modes.iter().enumerate() {
                chi[row][column] = report.value.fourier_density[output];
                inverse[row][column] = chi[row][column]
                    * (4.0 * std::f64::consts::PI / self.hartree.waves[output].norm2())
                    + Complex::from(if row == column { 1.0 } else { 0.0 });
            }
        }
        Ok(ResponseReport {
            value: DielectricResponse {
                q: self.options.q,
                frequency: self.options.frequency,
                temperature_hartree: self.temperature,
                fixed_number: self.options.fixed_number,
                reciprocal_cutoff: self.options.reciprocal_cutoff,
                reciprocal_waves: modes.iter().map(|&i| self.hartree.waves[i]).collect(),
                screened_chi: chi,
                inverse_epsilon: inverse,
            },
            shape: vec![n, n],
            functional: self.functional.clone(),
            reference: self.diagnostics.clone(),
            units: "dimensionless inverse dielectric; screened chi in bohr^-3/hartree",
            model: "3D adiabatic collinear HF/KS including full/range-separated exchange; epsilon^-1 = I + v chi with screened chi",
            axis_convention:
                "[density reciprocal mode G+q][external-potential reciprocal mode G'+q]",
            converged: true,
            residuals,
            iterations,
            warnings,
        })
    }
}

impl LinearOperator for PeriodicDensitySolver {
    fn dimension(&self) -> usize {
        self.propagators.iter().map(|p| p.dimension()).sum()
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }
    fn apply(&self, density: &[Complex]) -> Result<Vec<Complex>> {
        let image = self.independent_response(&self.induced_potential(density)?)?;
        Ok(density.iter().zip(image).map(|(&d, i)| d - i).collect())
    }
}

pub(crate) fn reference_diagnostics(
    reference: &PeriodicScfResult,
    orbitals: &[Vec<BandOrbitals>; 2],
) -> Result<(ReferenceDiagnostics, f64)> {
    let mut out = ReferenceDiagnostics {
        converged: reference.converged,
        iterations: reference.iterations,
        orbital_residual: 0.0,
        eigen_residual: 0.0,
        orthonormality_defect: 0.0,
        projector_defect: 0.0,
        particle_number_error: 0.0,
    };
    let mut consistency = 0.0_f64;
    let nk = reference.k_points.len();
    for spin in 0..2 {
        let mut number = 0.0;
        for ik in 0..nk {
            let c = &orbitals[spin][ik].coefficients;
            let sc = matmul(&reference.overlap[ik], c)?;
            let d = matmul(&adjoint(&sc)?, &matmul(&reference.density[spin][ik], &sc)?)?;
            let f = matmul(&adjoint(c)?, &matmul(&reference.fock[spin][ik], c)?)?;
            let comm1 = matmul(&f, &d)?;
            let comm2 = matmul(&d, &f)?;
            let dd = matmul(&d, &d)?;
            if reference.occupations[spin][ik].len() != d.rows() {
                return Err(DftError::InvalidInput(
                    "reference occupations differ from orbital rank".into(),
                ));
            }
            for i in 0..d.rows() {
                for j in 0..d.cols() {
                    let value = Complex::new(d.re[(i, j)], d.im[(i, j)]);
                    let expected = if i == j {
                        reference.occupations[spin][ik][i]
                    } else {
                        0.0
                    };
                    consistency = consistency.max((value - Complex::from(expected)).abs());
                    out.projector_defect = out
                        .projector_defect
                        .max((Complex::new(dd.re[(i, j)], dd.im[(i, j)]) - value).abs());
                    out.orbital_residual = out.orbital_residual.max(
                        Complex::new(
                            comm1.re[(i, j)] - comm2.re[(i, j)],
                            comm1.im[(i, j)] - comm2.im[(i, j)],
                        )
                        .abs(),
                    );
                }
            }
            out.eigen_residual = out.eigen_residual.max(orbitals[spin][ik].eigen_residual);
            out.orthonormality_defect = out
                .orthonormality_defect
                .max(orbitals[spin][ik].metric_defect);
            number += reference.density[spin][ik].trace_product(&reference.overlap[ik]) / nk as f64;
        }
        out.particle_number_error += (number
            - if spin == 0 {
                reference.n_alpha as f64
            } else {
                reference.n_beta as f64
            })
        .abs();
    }
    Ok((out, consistency))
}
