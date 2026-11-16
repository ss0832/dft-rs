// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Experimental simultaneous electronic and quantum-particle SCF.
//!
//! The Hamiltonian contains every selected particle's kinetic energy and
//! Coulomb couplings. It does not construct a Born-Oppenheimer potential
//! surface. Distinguishable quantum particles use normalized Hartree orbitals;
//! electronic exchange and XC retain the ordinary shared electronic provider.
//! Energies are laboratory-frame variational-model energies. In an entirely
//! quantum molecule the center-of-mass motion is retained, not subtracted.

pub mod annihilation;
pub mod constraints;
pub mod correlation;
pub mod derivatives;
mod electric;
pub use electric::QuadrupoleField;
pub mod fixed_grid;
pub mod mass;
mod orbital;
pub mod relativistic;
pub mod response;
pub mod stability;

use crate::basis::MolBasis;
use crate::grid::{MolecularGrid, eval::eval_aos};
use crate::integrals::{
    CoulombKernel, dipole_matrices, kinetic_matrix, moment_matrices, nuclear_matrix, overlap_matrix,
};
use crate::linalg::{Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::scf::{JkEngine, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};
use correlation::CorrelationModel;
use rayon::prelude::*;

/// One distinguishable quantum particle. `replaced_atom` removes that point
/// nucleus from the classical Hamiltonian; `None` adds a particle (e.g. positron).
#[derive(Clone, Debug)]
pub struct QuantumParticle {
    pub label: String,
    /// Charge in units of positive elementary charge.
    pub charge: f64,
    /// Mass in electron masses.
    pub mass: f64,
    pub basis: MolBasis,
    pub replaced_atom: Option<usize>,
    /// The channel of the correlation model whose partner density includes
    /// this particle (`None`: uncorrelated).
    pub correlation_channel: Option<usize>,
}

/// Uncontracted normalized Gaussian particle basis at a freely supplied center.
/// The centers are variational-basis parameters, not point Coulomb sources.
pub fn gaussian_particle_basis(center: Vec3, exponents: &[f64], lmax: u8) -> Result<MolBasis> {
    if exponents.is_empty() || lmax > 3 || center.to_array().iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "a particle Gaussian basis needs finite coordinates, exponents and lmax <= 3".into(),
        ));
    }
    let mut basis = crate::basis::BasisSet::new("quantum-particle");
    basis.insert(crate::basis::ElementBasis {
        z: 1,
        shells: (0..=lmax)
            .flat_map(|l| {
                exponents.iter().map(move |&e| crate::basis::Shell {
                    l,
                    exps: vec![e],
                    coefs: vec![1.0],
                })
            })
            .collect(),
    });
    MolBasis::build(
        &Molecule::new(vec![crate::system::Atom {
            z: 1,
            position: center,
        }]),
        &basis,
    )
}

#[derive(Clone, Debug)]
pub struct MulticomponentOptions {
    pub tolerance: f64,
    pub max_iterations: usize,
    /// Fraction of the new normalized orbital in a damped SCF step.
    pub mixing: f64,
}

impl Default for MulticomponentOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 500,
            mixing: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QuantumState {
    pub coefficient: Vec<f64>,
    pub density: Matrix,
    pub overlap: Matrix,
    pub fock: Matrix,
    pub orbital_energies: Vec<f64>,
    pub kinetic_energy: f64,
    pub classical_energy: f64,
    pub particle_number: f64,
    pub position: [f64; 3],
    pub position_variance: [f64; 3],
    pub orbital_residual: f64,
    pub density_residual: f64,
}

/// Starting point of a coupled solution: electronic spin densities and the
/// normalized coefficients of every quantum-particle orbital.
#[derive(Clone, Debug)]
pub struct InitialState {
    pub electronic_density: [Matrix; 2],
    pub quantum: Vec<Vec<f64>>,
}

/// No ordinary `ScfResult` is exposed: derivatives must differentiate both
/// species and the interspecies functional, rather than freeze its potential.
#[derive(Clone, Debug)]
pub struct MulticomponentResult {
    pub(crate) electronic_reference: ScfResult,
    pub energy: f64,
    pub electronic_kinetic: f64,
    pub electronic_one_body: f64,
    pub electronic_coulomb: f64,
    pub electronic_exchange_correlation: f64,
    pub electronic_nonlocal_correlation: f64,
    pub electron_quantum_coulomb: f64,
    pub quantum_pair_coulomb: f64,
    pub interspecies_correlation: f64,
    pub classical_energy: f64,
    pub electronic_density: [Matrix; 2],
    pub quantum: Vec<QuantumState>,
    pub electronic_orbital_residual: f64,
    pub electronic_density_residual: f64,
    pub iterations: usize,
    pub converged: bool,
    pub energy_history: Vec<f64>,
    pub residual_history: Vec<f64>,
    /// Present for an expectation-position constrained stationary reference.
    pub position_constraints: Option<constraints::PositionConstraints>,
}

fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (a, b) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *a += scale * b;
    }
}
fn block(a: &Matrix, offset: usize, n: usize) -> Matrix {
    Matrix::from_row_major(
        n,
        n,
        (0..n)
            .flat_map(|i| (0..n).map(move |j| a[(offset + i, offset + j)]))
            .collect(),
    )
}

fn electronic_particle_potential(
    correction: &Matrix,
    sources: &[Matrix],
    particles: &[QuantumParticle],
) -> Matrix {
    let mut operator = correction.clone();
    for (source, particle) in sources.iter().zip(particles) {
        add(
            &mut operator,
            &block(source, 0, correction.rows),
            -particle.charge,
        );
    }
    operator
}
fn projector(c: &[f64]) -> Matrix {
    Matrix::from_row_major(
        c.len(),
        c.len(),
        c.iter()
            .flat_map(|a| c.iter().map(move |b| a * b))
            .collect(),
    )
}

fn mix_orbital(current: &[f64], next: &[f64], metric: &Matrix, step: f64) -> Vec<f64> {
    let dot: f64 = (0..current.len())
        .map(|a| {
            (0..current.len())
                .map(|b| current[a] * metric[(a, b)] * next[b])
                .sum::<f64>()
        })
        .sum();
    let sign = if dot < 0.0 { -1.0 } else { 1.0 };
    let mut mixed: Vec<_> = current
        .iter()
        .zip(next)
        .map(|(a, b)| (1.0 - step) * a + step * sign * b)
        .collect();
    let norm = projector(&mixed).frobenius_dot(metric).sqrt();
    for value in &mut mixed {
        *value /= norm;
    }
    mixed
}

fn max_abs(a: &Matrix) -> f64 {
    a.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn append_basis(target: &mut MolBasis, source: &MolBasis) {
    for shell in &source.shells {
        let mut shell = shell.clone();
        shell.sph_off += target.nao_sph;
        shell.cart_off += target.nao_cart;
        target.shells.push(shell);
    }
    target.nao_sph += source.nao_sph;
    target.nao_cart += source.nao_cart;
    target.max_l = target.max_l.max(source.max_l);
}

fn orbital_density(phi: &Matrix, c: &Matrix, occupations: &[f64]) -> Vec<f64> {
    let values = phi.matmul_seq(c);
    (0..phi.rows)
        .map(|g| {
            occupations
                .iter()
                .enumerate()
                .map(|(i, f)| f * values[(g, i)].powi(2))
                .sum()
        })
        .collect()
}

struct CorrelationGrid {
    weights: Vec<f64>,
    phi: Vec<Matrix>,
}
struct CorrelationEvaluation {
    energy: f64,
    potentials: Vec<Matrix>,
}

impl CorrelationGrid {
    /// Total electron density on the grid.
    fn electron_density(&self, electronic: &ScfResult) -> Vec<f64> {
        let mut electron = orbital_density(
            &self.phi[0],
            &electronic.mo_coeff[0],
            &electronic.occupations[0],
        );
        let beta = orbital_density(
            &self.phi[0],
            &electronic.mo_coeff[1],
            &electronic.occupations[1],
        );
        for (a, b) in electron.iter_mut().zip(beta) {
            *a += b;
        }
        electron
    }

    /// Summed density of the particles of every correlation channel.
    fn partner_densities(
        &self,
        coefficients: &[Vec<f64>],
        particles: &[QuantumParticle],
        channels: usize,
    ) -> Vec<Vec<f64>> {
        let mut partner = vec![vec![0.0; self.weights.len()]; channels];
        for (i, particle) in particles.iter().enumerate() {
            let Some(channel) = particle.correlation_channel else {
                continue;
            };
            let c = Matrix::from_row_major(coefficients[i].len(), 1, coefficients[i].clone());
            let density = orbital_density(&self.phi[i + 1], &c, &[1.0]);
            for (a, b) in partner[channel].iter_mut().zip(density) {
                *a += b;
            }
        }
        partner
    }

    fn evaluate(
        &self,
        electronic: &ScfResult,
        coefficients: &[Vec<f64>],
        particles: &[QuantumParticle],
        model: &CorrelationModel,
    ) -> Result<CorrelationEvaluation> {
        let zero = || -> Vec<Matrix> {
            self.phi
                .iter()
                .map(|p| Matrix::zeros(p.cols, p.cols))
                .collect()
        };
        if model.is_zero() {
            return Ok(CorrelationEvaluation {
                energy: 0.0,
                potentials: zero(),
            });
        }
        let electron = self.electron_density(electronic);
        let partner = self.partner_densities(coefficients, particles, model.len());
        let mut energy = 0.0;
        let mut electron_potential = vec![0.0; electron.len()];
        let mut channel_potential = vec![vec![0.0; electron.len()]; model.len()];
        for (channel, provider) in model.channels().iter().enumerate() {
            if provider.is_zero() {
                continue;
            }
            let points = electron
                .par_iter()
                .zip(&partner[channel])
                .map(|(&e, &p)| provider.evaluate(e, p))
                .collect::<Result<Vec<_>>>()?;
            for (g, point) in points.iter().enumerate() {
                energy += self.weights[g] * point.energy;
                electron_potential[g] += point.electron_potential;
                channel_potential[channel][g] = point.partner_potential;
            }
        }
        let potentials = self
            .phi
            .par_iter()
            .enumerate()
            .map(|(s, phi)| {
                let mut potential = Matrix::zeros(phi.cols, phi.cols);
                let values = if s == 0 {
                    &electron_potential
                } else if let Some(channel) = particles[s - 1].correlation_channel {
                    &channel_potential[channel]
                } else {
                    return potential;
                };
                for (g, (v, w)) in values.iter().zip(&self.weights).enumerate() {
                    let v = w * v;
                    for i in 0..phi.cols {
                        for j in 0..=i {
                            potential[(i, j)] += v * phi[(g, i)] * phi[(g, j)];
                        }
                    }
                }
                for i in 0..phi.cols {
                    for j in 0..i {
                        potential[(j, i)] = potential[(i, j)];
                    }
                }
                potential
            })
            .collect();
        Ok(CorrelationEvaluation { energy, potentials })
    }
}

/// Simultaneously solve the electronic and quantum-particle orbital equations.
/// The electronic provider is reused unchanged; correlation between species is
/// a separate supplied provider. Nuclear exchange between distinct localized
/// Hartree orbitals, relativistic effects and internal-frame mass polarization
/// are outside this experimental model. No finite differences are used.
#[allow(clippy::too_many_arguments)]
pub fn solve(
    molecule: &Molecule,
    electronic_basis: &MolBasis,
    electronic_options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    options: &MulticomponentOptions,
) -> Result<MulticomponentResult> {
    solve_with_multipliers(
        molecule,
        electronic_basis,
        electronic_options,
        functional,
        grid,
        particles,
        correlation,
        options,
        None,
        None,
        None,
    )
}

/// Solve the stationary electronic and quantum-particle reference with a
/// finite five-mode electric field gradient.
#[allow(clippy::too_many_arguments)]
pub fn solve_in_field_gradient(
    molecule: &Molecule,
    electronic_basis: &MolBasis,
    electronic_options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    options: &MulticomponentOptions,
    field_gradient: &QuadrupoleField,
) -> Result<MulticomponentResult> {
    solve_with_multipliers(
        molecule,
        electronic_basis,
        electronic_options,
        functional,
        grid,
        particles,
        correlation,
        options,
        None,
        None,
        Some(field_gradient),
    )
}

/// [`solve`] from a supplied starting state instead of the core guess.
#[allow(clippy::too_many_arguments)]
pub fn solve_from(
    molecule: &Molecule,
    electronic_basis: &MolBasis,
    electronic_options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    options: &MulticomponentOptions,
    initial: &InitialState,
) -> Result<MulticomponentResult> {
    solve_with_multipliers(
        molecule,
        electronic_basis,
        electronic_options,
        functional,
        grid,
        particles,
        correlation,
        options,
        None,
        Some(initial),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve_with_multipliers(
    molecule: &Molecule,
    electronic_basis: &MolBasis,
    electronic_options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    options: &MulticomponentOptions,
    constraints: Option<&constraints::PositionConstraints>,
    initial: Option<&InitialState>,
    field_gradient: Option<&QuadrupoleField>,
) -> Result<MulticomponentResult> {
    if let Some(gradient) = field_gradient {
        gradient.validate()?;
    }
    if particles.is_empty()
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || options.max_iterations == 0
        || !options.mixing.is_finite()
        || options.mixing <= 0.0
        || options.mixing > 1.0
    {
        return Err(DftError::InvalidInput("multicomponent SCF requires particles, a positive finite tolerance, iterations and 0 < mixing <= 1".into()));
    }
    let mut positions = constraints.cloned();
    if electronic_options.extra_hcore.is_some()
        || electronic_options.pcm.is_some()
        || electronic_options.occupation.is_some()
        || electronic_options.electronic_temperature != 0.0
        || electronic_options.fixed_occupations.is_some()
        || electronic_options.kernel != CoulombKernel::Coulomb
    {
        return Err(DftError::Unsupported { feature: "multicomponent SCF with external embedding, solvent or non-Aufbau/thermal populations".into(), missing: "the corresponding jointly variational species energy provider".into() });
    }
    let mut charges = electronic_basis.effective_charges(molecule);
    let mut used = vec![false; molecule.len()];
    for particle in particles {
        if !particle.mass.is_finite()
            || particle.mass <= 0.0
            || !particle.charge.is_finite()
            || particle.basis.nao_sph == 0
            || !particle.basis.ecps.is_empty()
            || particle
                .correlation_channel
                .is_some_and(|c| c >= correlation.len())
        {
            return Err(DftError::InvalidInput("quantum particles need a finite positive mass, finite charge, a nonempty ECP-free basis and a correlation channel of the supplied model".into()));
        }
        if let Some(atom) = particle.replaced_atom {
            if atom >= molecule.len()
                || used[atom]
                || charges[atom] != particle.charge
                || electronic_basis.ecps.iter().any(|ecp| ecp.atom == atom)
            {
                return Err(DftError::InvalidInput("each quantum nucleus must replace one distinct matching all-electron point charge".into()));
            }
            used[atom] = true;
            charges[atom] = 0.0;
        }
    }
    let n = electronic_basis.nao_sph;
    let mut combined = electronic_basis.clone();
    let mut offsets = vec![0];
    for particle in particles {
        offsets.push(combined.nao_sph);
        append_basis(&mut combined, &particle.basis);
    }
    let engine = JkEngine::new(
        &combined,
        electronic_options.effective_eri_mode(combined.nao_sph),
        CoulombKernel::Coulomb,
    )?;
    let mut electronic_cached = electronic_options.clone();
    electronic_cached.eri_mode = electronic_options.effective_eri_mode(n);
    let electronic_engine = JkEngine::from_options(molecule, electronic_basis, &electronic_cached)?;
    let electronic_long_range = if functional.hybrid.clr != 0.0 {
        Some(JkEngine::new(
            electronic_basis,
            electronic_cached.eri_mode,
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
        )?)
    } else {
        None
    };
    let coulomb = |density: &Matrix, offset: usize| -> Result<Matrix> {
        let mut padded = Matrix::zeros(combined.nao_sph, combined.nao_sph);
        for i in 0..density.rows {
            for j in 0..density.cols {
                padded[(offset + i, offset + j)] = density[(i, j)];
            }
        }
        let zero = Matrix::zeros(combined.nao_sph, combined.nao_sph);
        Ok(engine.build(&combined, &padded, [&zero, &zero], false)?.0)
    };
    let mut correction = nuclear_matrix(electronic_basis, molecule, Some(&charges));
    add(
        &mut correction,
        &nuclear_matrix(electronic_basis, molecule, None),
        -1.0,
    );
    // On ECP atoms the ordinary reference already uses effective charges.
    if !electronic_basis.ecps.is_empty() {
        correction = nuclear_matrix(electronic_basis, molecule, Some(&charges));
        add(
            &mut correction,
            &nuclear_matrix(
                electronic_basis,
                molecule,
                Some(&electronic_basis.effective_charges(molecule)),
            ),
            -1.0,
        );
    }
    if let Some(gradient) = field_gradient {
        gradient.add_potential(&mut correction, electronic_basis, -1.0);
    }
    let mut classical_energy = 0.0;
    for i in 0..charges.len() {
        for j in 0..i {
            classical_energy += charges[i] * charges[j]
                / (molecule.atoms[i].position - molecule.atoms[j].position).norm();
        }
    }
    let field = electronic_options.external_field.unwrap_or(Vec3::zero());
    for (atom, z) in molecule.atoms.iter().zip(&charges) {
        classical_energy -= z * atom.position.dot(field);
    }
    if let Some(gradient) = field_gradient {
        classical_energy += gradient.classical_energy(molecule, &charges);
    }
    let overlaps: Vec<_> = particles.iter().map(|p| overlap_matrix(&p.basis)).collect();
    let orth: Vec<_> = overlaps
        .iter()
        .map(|s| Orthogonalizer::canonical(s, electronic_options.orth_threshold))
        .collect::<Result<_>>()?;
    let kinetic: Vec<_> = particles
        .iter()
        .map(|p| {
            let mut t = kinetic_matrix(&p.basis);
            for v in t.as_mut_slice() {
                *v /= p.mass;
            }
            t
        })
        .collect();
    let external: Vec<_> = particles
        .iter()
        .map(|p| {
            let mut v = nuclear_matrix(&p.basis, molecule, Some(&charges));
            for x in v.as_mut_slice() {
                *x *= -p.charge;
            }
            for (axis, d) in dipole_matrices(&p.basis, Vec3::zero()).iter().enumerate() {
                add(&mut v, d, -p.charge * field.to_array()[axis]);
            }
            if let Some(gradient) = field_gradient {
                gradient.add_potential(&mut v, &p.basis, p.charge);
            }
            v
        })
        .collect();
    let core: Vec<_> = kinetic
        .iter()
        .zip(&external)
        .map(|(t, v)| {
            let mut h = t.clone();
            add(&mut h, v, 1.0);
            h
        })
        .collect();
    let mut coefficients: Vec<Vec<f64>> = core
        .iter()
        .zip(&orth)
        .map(|(h, o)| {
            o.solve_roothaan(h)
                .map(|(_, c)| (0..c.rows).map(|i| c[(i, 0)]).collect())
        })
        .collect::<Result<_>>()?;
    if let Some(initial) = initial {
        let n_electronic = electronic_basis.nao_sph;
        if initial.quantum.len() != particles.len()
            || initial
                .quantum
                .iter()
                .zip(particles)
                .any(|(c, p)| c.len() != p.basis.nao_sph || c.iter().any(|v| !v.is_finite()))
            || initial
                .electronic_density
                .iter()
                .any(|d| d.rows != n_electronic || d.cols != n_electronic)
        {
            return Err(DftError::InvalidInput(
                "the initial coupled state does not match the electronic and quantum bases".into(),
            ));
        }
        coefficients = initial.quantum.clone();
    }
    let moments: Vec<_> = particles
        .iter()
        .map(|p| dipole_matrices(&p.basis, Vec3::zero()))
        .collect();
    let position_operators = positions.as_ref().map(|p| {
        moments
            .iter()
            .enumerate()
            .map(|(i, r)| constraints::operators(r, &overlaps[i], p.targets[i]))
            .collect::<Vec<_>>()
    });
    if let Some(operators) = &position_operators {
        for i in 0..particles.len() {
            constraints::retract(
                &mut coefficients[i],
                &overlaps[i],
                &orth[i],
                &operators[i],
                options.tolerance,
            )?;
        }
    }
    let correlation_grid = grid.map(|grid| CorrelationGrid {
        weights: grid.weights.clone(),
        phi: std::iter::once(electronic_basis)
            .chain(particles.iter().map(|p| &p.basis))
            .map(|b| eval_aos(b, &grid.points, false).phi)
            .collect(),
    });
    // Bootstrap in the supplied quantum charge distribution. Requiring a
    // separate point-nucleus BO solution can fail even when the actual
    // multicomponent problem is well defined. Reuse the prepared caches.
    let particle_sources = |coefficients: &[Vec<f64>]| -> Result<Vec<Matrix>> {
        coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| coulomb(&projector(c), offsets[i + 1]))
            .collect()
    };
    let mut sources = particle_sources(&coefficients)?;
    let initial_operator = electronic_particle_potential(&correction, &sources, particles);
    let mut electronic = crate::scf::run_scf_in_potential(
        molecule,
        electronic_basis,
        electronic_options,
        functional,
        grid,
        initial.map(|state| &state.electronic_density),
        None,
        Some(crate::scf::FrozenPotential {
            operator: &initial_operator,
            coulomb: &electronic_engine,
            long_range: electronic_long_range.as_ref(),
            classical_energy: None,
        }),
    )?;
    if !electronic.converged {
        return Err(DftError::ScfNotConverged {
            iterations: electronic.iterations,
            error: f64::INFINITY,
        });
    }
    let eorth = Orthogonalizer::canonical(&electronic.overlap, electronic_options.orth_threshold)?;
    let empty_correlation = || CorrelationEvaluation {
        energy: 0.0,
        potentials: std::iter::once(n)
            .chain(particles.iter().map(|p| p.basis.nao_sph))
            .map(|n| Matrix::zeros(n, n))
            .collect(),
    };
    if correlation_grid.is_none() && !correlation.is_zero() {
        return Err(DftError::InvalidInput(
            "a nonzero interspecies correlation provider requires a molecular grid".into(),
        ));
    }
    let mut energy_history = Vec::new();
    let mut residual_history = Vec::new();
    let mut newton_available = true;
    for iteration in 1..=options.max_iterations {
        let densities: Vec<_> = coefficients.iter().map(|c| projector(c)).collect();
        if iteration > 1 {
            sources = particle_sources(&coefficients)?;
        }
        let frozen = if let Some(g) = &correlation_grid {
            g.evaluate(&electronic, &coefficients, particles, correlation)?
        } else {
            empty_correlation()
        };
        let mut extra = electronic_particle_potential(&correction, &sources, particles);
        add(&mut extra, &frozen.potentials[0], 1.0);
        electronic = crate::scf::run_scf_in_potential(
            molecule,
            electronic_basis,
            electronic_options,
            functional,
            grid,
            Some(&electronic.density),
            None,
            Some(crate::scf::FrozenPotential {
                operator: &extra,
                coulomb: &electronic_engine,
                long_range: electronic_long_range.as_ref(),
                classical_energy: None,
            }),
        )?;
        let p = electronic.p_total();
        let je = coulomb(&p, 0)?;
        let actual = if let Some(g) = &correlation_grid {
            g.evaluate(&electronic, &coefficients, particles, correlation)?
        } else {
            empty_correlation()
        };
        let mut quantum = Vec::new();
        let mut mapped = Vec::new();
        let mut pair = 0.0;
        for i in 0..particles.len() {
            let dim = particles[i].basis.nao_sph;
            let mut fock = core[i].clone();
            add(
                &mut fock,
                &block(&je, offsets[i + 1], dim),
                -particles[i].charge,
            );
            for j in 0..particles.len() {
                if i != j {
                    let v = block(&sources[j], offsets[i + 1], dim);
                    add(&mut fock, &v, particles[i].charge * particles[j].charge);
                    if j < i {
                        pair += particles[i].charge
                            * particles[j].charge
                            * densities[i].frobenius_dot(&v);
                    }
                }
            }
            add(&mut fock, &actual.potentials[i + 1], 1.0);
            let (energies, next) =
                if let (Some(positions), Some(operators)) = (&mut positions, &position_operators) {
                    let constrained = constraints::orbital(
                        &fock,
                        &orth[i],
                        &operators[i],
                        positions.multipliers[i],
                        options.tolerance,
                    )?;
                    positions.multipliers[i] = constrained.multipliers;
                    positions.iterations += constrained.iterations;
                    fock = constrained.fock;
                    (constrained.energies, constrained.coefficient)
                } else {
                    let (e, c) = orth[i].solve_roothaan(&fock)?;
                    (e, (0..dim).map(|k| c[(k, 0)]).collect::<Vec<_>>())
                };
            let map = projector(&next);
            let mut delta = map.clone();
            add(&mut delta, &densities[i], -1.0);
            let sx = overlaps[i].matmul_seq(&orth[i].x);
            let density_residual = max_abs(&sx.transpose_matmul_seq(&delta).matmul_seq(&sx));
            let orbital_residual = max_abs(&crate::linalg::metric_commutator(
                &fock,
                &densities[i],
                &overlaps[i],
                &orth[i].x,
            ));
            let position = std::array::from_fn(|a| densities[i].frobenius_dot(&moments[i][a]));
            let second = moment_matrices(&particles[i].basis, Vec3::zero(), 2);
            let variance = std::array::from_fn(|a| {
                let key = match a {
                    0 => (2, 0, 0),
                    1 => (0, 2, 0),
                    _ => (0, 0, 2),
                };
                densities[i].frobenius_dot(
                    &second
                        .iter()
                        .find(|(k, _)| *k == key)
                        .expect("Cartesian second moment")
                        .1,
                ) - position[a] * position[a]
            });
            quantum.push(QuantumState {
                coefficient: coefficients[i].clone(),
                density: densities[i].clone(),
                overlap: overlaps[i].clone(),
                fock,
                orbital_energies: energies,
                kinetic_energy: densities[i].frobenius_dot(&kinetic[i]),
                classical_energy: densities[i].frobenius_dot(&external[i]),
                particle_number: densities[i].frobenius_dot(&overlaps[i]),
                position,
                position_variance: variance,
                orbital_residual,
                density_residual,
            });
            mapped.push(next);
        }
        let mut ec = 0.0_f64;
        let mut ed = 0.0_f64;
        for s in 0..2 {
            let mut f = electronic.fock[s].clone();
            add(&mut f, &actual.potentials[0], 1.0);
            add(&mut f, &frozen.potentials[0], -1.0);
            ec = ec.max(max_abs(&crate::linalg::metric_commutator(
                &f,
                &electronic.density[s],
                &electronic.overlap,
                &eorth.x,
            )));
            let (_, c) = eorth.solve_roothaan(&f)?;
            let mut map = c.leading_columns_gram([electronic.n_alpha, electronic.n_beta][s], 1.0);
            add(&mut map, &electronic.density[s], -1.0);
            let sx = electronic.overlap.matmul_seq(&eorth.x);
            ed = ed.max(max_abs(&sx.transpose_matmul_seq(&map).matmul_seq(&sx)));
        }
        let ep: f64 = sources
            .iter()
            .zip(particles)
            .map(|(j, q)| -q.charge * p.frobenius_dot(&block(j, 0, n)))
            .sum();
        let qenergy: f64 = quantum
            .iter()
            .map(|q| q.kinetic_energy + q.classical_energy)
            .sum();
        let energy = electronic.e_elec - p.frobenius_dot(&frozen.potentials[0])
            + qenergy
            + pair
            + actual.energy
            + classical_energy;
        let position_residual = positions.as_mut().map_or(0.0, |p| {
            p.residual = quantum
                .iter()
                .zip(&p.targets)
                .flat_map(|(q, t)| (0..3).map(move |a| (q.position[a] - t[a]).abs()))
                .fold(0.0_f64, f64::max);
            p.residual_history.push(p.residual);
            crate::logger::log(&format!(
                "[quantum-position] iteration {iteration:4} energy {energy:.14} residual {:.12e}",
                p.residual
            ));
            p.residual
        });
        let residual = quantum
            .iter()
            .fold(ec.max(ed).max(position_residual), |r, q| {
                r.max(q.orbital_residual)
                    .max(q.density_residual)
                    .max((q.particle_number - 1.0).abs())
            });
        if !energy.is_finite() || !residual.is_finite() {
            return Err(DftError::LinearAlgebra(
                "non-finite coupled species energy or residual".into(),
            ));
        }
        energy_history.push(energy);
        residual_history.push(residual);
        crate::logger::log(&format!(
            "[multicomponent] iteration {iteration:4} energy {energy:.14} residual {residual:.6e}"
        ));
        let converged = electronic.converged && residual <= options.tolerance;
        if converged || iteration == options.max_iterations {
            let kinetic_e = p.frobenius_dot(&kinetic_matrix(electronic_basis));
            let electronic_j = electronic_engine
                .build(
                    electronic_basis,
                    &p,
                    [&electronic.density[0], &electronic.density[1]],
                    false,
                )?
                .0;
            let coulomb_e = 0.5 * p.frobenius_dot(&electronic_j);
            let mut one = electronic.h_core.clone();
            add(&mut one, &extra, -1.0);
            add(&mut one, &correction, 1.0);
            let one_energy = p.frobenius_dot(&one);
            // Extract the full electronic exchange/XC contribution, including
            // short/long-range exact exchange, from the shared energy identity.
            let electronic_xc = electronic.e_elec
                - one_energy
                - coulomb_e
                - ep
                - p.frobenius_dot(&frozen.potentials[0])
                - electronic.e_nlc;
            let electronic_reference = electronic.clone();
            return Ok(MulticomponentResult {
                electronic_reference,
                energy,
                electronic_kinetic: kinetic_e,
                electronic_one_body: one_energy,
                electronic_coulomb: coulomb_e,
                electronic_exchange_correlation: electronic_xc,
                electronic_nonlocal_correlation: electronic.e_nlc,
                electron_quantum_coulomb: ep,
                quantum_pair_coulomb: pair,
                interspecies_correlation: actual.energy,
                classical_energy,
                electronic_density: electronic.density,
                quantum,
                electronic_orbital_residual: ec,
                electronic_density_residual: ed,
                iterations: iteration,
                converged,
                energy_history,
                residual_history,
                position_constraints: positions,
            });
        }
        if quantum.iter().all(|q| {
            q.orbital_residual <= options.tolerance && q.density_residual <= options.tolerance
        }) {
            // A stationary or one-dimensional quantum space can still leave
            // electronic interspecies self-consistency to be converged.
            continue;
        }
        // Nonlinear interspecies correlation makes a fixed-mixing orbital
        // map unstable even when the physical stationary solution is stable.
        // Search the actual coupled functional at fixed electronic density.
        // At an exactly rounded energy equality require a smaller fresh
        // commutator; never replace a failed residual by an energy threshold.
        let mut old_quantum_energy = actual.energy;
        for i in 0..particles.len() {
            let dim = particles[i].basis.nao_sph;
            let mut h = core[i].clone();
            add(
                &mut h,
                &block(&je, offsets[i + 1], dim),
                -particles[i].charge,
            );
            old_quantum_energy += densities[i].frobenius_dot(&h);
            for j in 0..i {
                old_quantum_energy += particles[i].charge
                    * particles[j].charge
                    * densities[i].frobenius_dot(&block(&sources[j], offsets[i + 1], dim));
            }
        }
        let old_commutator = quantum
            .iter()
            .fold(0.0_f64, |m, q| m.max(q.orbital_residual));
        let newton = if residual < 1e-3 && !correlation.is_zero() && newton_available {
            match orbital::proposal(
                &electronic,
                &quantum,
                &orth,
                correlation_grid.as_ref(),
                particles,
                correlation,
                &coulomb,
                &offsets,
            ) {
                Ok(proposal) => proposal,
                Err(DftError::Unsupported { .. }) => {
                    // A ground-state provider needs its energy and two
                    // potentials. A missing higher kernel disables only
                    // this optional optimizer, never a physical term.
                    // Response requests still require their full kernels.
                    newton_available = false;
                    crate::logger::log(
                        "[multicomponent] optional quantum Newton kernel unavailable; retaining variational species SCF",
                    );
                    None
                }
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        let mut accepted = None;
        for fallback in [false, true] {
            let proposals = if fallback {
                quantum
                    .iter()
                    .enumerate()
                    .map(|(i, q)| {
                        let c = Matrix::from_row_major(
                            coefficients[i].len(),
                            1,
                            coefficients[i].clone(),
                        );
                        let fc = q.fock.matmul_seq(&c);
                        let lambda = c.transpose_matmul_seq(&fc)[(0, 0)];
                        let mut g = orth[i].x.transpose_matmul_seq(&fc);
                        let y = orth[i].x.transpose_matmul_seq(&overlaps[i].matmul_seq(&c));
                        add(&mut g, &y, -lambda);
                        let delta = orth[i].x.matmul_seq(&g);
                        coefficients[i]
                            .iter()
                            .enumerate()
                            .map(|(a, v)| v - delta[(a, 0)])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            } else {
                newton.as_ref().unwrap_or(&mapped).clone()
            };
            let mut step = options.mixing;
            for backtrack in 0..40 {
                let mut trial: Vec<Vec<f64>> = coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, c)| mix_orbital(c, &proposals[i], &overlaps[i], step))
                    .collect();
                if let Some(operators) = &position_operators {
                    for i in 0..particles.len() {
                        constraints::retract(
                            &mut trial[i],
                            &overlaps[i],
                            &orth[i],
                            &operators[i],
                            options.tolerance,
                        )?;
                    }
                }
                let td: Vec<_> = trial.iter().map(|c| projector(c)).collect();
                let ts = if particles.len() > 1 {
                    td.iter()
                        .enumerate()
                        .map(|(i, p)| coulomb(p, offsets[i + 1]))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    Vec::new()
                };
                let tc = if let Some(g) = &correlation_grid {
                    g.evaluate(&electronic, &trial, particles, correlation)?
                } else {
                    empty_correlation()
                };
                let mut trial_energy = tc.energy;
                let mut trial_commutator = 0.0_f64;
                let mut initial_work = 0.0;
                let mut final_work = 0.0;
                for i in 0..particles.len() {
                    let dim = particles[i].basis.nao_sph;
                    let mut h = core[i].clone();
                    add(
                        &mut h,
                        &block(&je, offsets[i + 1], dim),
                        -particles[i].charge,
                    );
                    trial_energy += td[i].frobenius_dot(&h);
                    for j in 0..particles.len() {
                        if i != j {
                            let v = block(&ts[j], offsets[i + 1], dim);
                            let charge = particles[i].charge * particles[j].charge;
                            if j < i {
                                trial_energy += charge * td[i].frobenius_dot(&v);
                            }
                            add(&mut h, &v, charge);
                        }
                    }
                    add(&mut h, &tc.potentials[i + 1], 1.0);
                    if let (Some(positions), Some(operators)) = (&positions, &position_operators) {
                        for a in 0..3 {
                            add(&mut h, &operators[i][a], positions.multipliers[i][a]);
                        }
                    }
                    for (c, f, work) in [
                        (&coefficients[i], &quantum[i].fock, &mut initial_work),
                        (&trial[i], &h, &mut final_work),
                    ] {
                        let c = Matrix::from_row_major(c.len(), 1, c.clone());
                        let fc = f.matmul_seq(&c);
                        let sc = overlaps[i].matmul_seq(&c);
                        let lambda = c.transpose_matmul_seq(&fc)[(0, 0)];
                        for a in 0..c.rows {
                            *work += 2.0
                                * (trial[i][a] - coefficients[i][a])
                                * (fc[(a, 0)] - lambda * sc[(a, 0)]);
                        }
                    }
                    trial_commutator = trial_commutator.max(max_abs(
                        &crate::linalg::metric_commutator(&h, &td[i], &overlaps[i], &orth[i].x),
                    ));
                }
                // Endpoint work uses analytic gradients on the normalized
                // orbital path. It resolves a descending local step when a
                // total energy difference has rounded away; the fresh orbital
                // residual must decrease as well. No energy/error floor is used.
                let resolved_descent =
                    initial_work < 0.0 && final_work < 0.0 && trial_commutator < old_commutator;
                if trial_energy < old_quantum_energy
                    || resolved_descent
                    || trial_energy == old_quantum_energy && trial_commutator < old_commutator
                {
                    if backtrack > 0 || fallback {
                        crate::logger::log(&format!(
                            "[multicomponent] quantum step {step:.6e} backtracks {backtrack} gradient_fallback {fallback}"
                        ));
                    }
                    accepted = Some(trial);
                    break;
                }
                step *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }
        coefficients=accepted.ok_or_else(||DftError::LinearAlgebra(format!("multicomponent quantum orbital line search failed at measured residual {residual:.12e}")))?;
    }
    unreachable!("a positive iteration limit returns its final measured state")
}
