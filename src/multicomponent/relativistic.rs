// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed no-pair electronic SCF coupled to distinguishable quantum nuclei.

use super::{add, max_abs, mix_orbital, projector, MulticomponentOptions};
use crate::integrals::analytic_basis::{AnalyticShell, IntegralBlock};
use crate::integrals::dirac::{
    electron_quantum_coulomb, shell_offsets, BalancedCoulombFields, BalancedCoulombInput,
};
use crate::linalg::{metric_commutator, Matrix, Orthogonalizer};
use crate::periodic::spinor_scf::{
    run_spinor_scf, BiasedSpinorBackend, SpinorEnergyBackend, SpinorScfConfig, SpinorScfReport,
};
use crate::periodic::BlochMatrix;
use crate::properties::no_pair::stationary::{NoPairBackend, NoPairSpace};
use crate::{DftError, Result};

mod nuclear_coulomb;
mod response;

pub use response::{JointElectricOperators, JointElectricResponse};

/// One distinguishable quantum nucleus. The supplied covariant one-body
/// operator includes the mass-scaled kinetic and any classical potentials.
pub struct BalancedQuantumNucleus<'a> {
    pub shells: &'a [AnalyticShell<f64>],
    pub overlap: &'a Matrix,
    pub one_body: &'a Matrix,
    pub charge: f64,
    pub intrinsic_rms_radius: Option<f64>,
}

/// The full electronic backend must use the same balanced shell order,
/// speed of light and fixed sector. Its energy excludes the quantum
/// nucleus and its Coulomb coupling, which are added here.
pub struct JointNoPairModel<'a, B> {
    pub electronic: &'a B,
    pub electronic_space: &'a NoPairSpace,
    pub electronic_shells: &'a [AnalyticShell<f64>],
    pub speed_of_light: f64,
    pub nucleus: BalancedQuantumNucleus<'a>,
}

/// The electronic provider supplies the energy of the electronic sector
/// and its fixed classical nuclei. Each quantum nucleus supplies its own
/// covariant one-body operator. All electron–nuclear and inter-nuclear
/// quantum Coulomb terms are assembled from the same trial densities.
pub struct JointNoPairSystem<'a, B> {
    pub electronic: &'a B,
    pub electronic_space: &'a NoPairSpace,
    pub electronic_shells: &'a [AnalyticShell<f64>],
    pub speed_of_light: f64,
    pub nuclei: &'a [BalancedQuantumNucleus<'a>],
}

pub struct JointNoPairState {
    /// Inner reference in the positive-energy metric, including the
    /// electronic potential from the reported nuclear density. Its bias
    /// component follows the generic spinor backend's bookkeeping.
    pub electronic: SpinorScfReport,
    pub electronic_density: BlochMatrix,
    pub nuclear_density: Matrix,
    pub nuclear_fock: Matrix,
    pub nuclear_orbital_residual: f64,
    pub nuclear_density_residual: f64,
    pub nuclear_particle_number_error: f64,
    pub electronic_energy: f64,
    pub nuclear_one_body_energy: f64,
    pub interaction_energy: f64,
    pub reciprocity_residual: f64,
    /// Projected electronic bias expectation minus the full-AO contraction.
    pub projection_residual: f64,
    pub energy: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub struct JointNuclearState {
    pub density: Matrix,
    pub fock: Matrix,
    pub orbital_residual: f64,
    pub density_residual: f64,
    pub particle_number_error: f64,
    pub one_body_energy: f64,
}

pub struct JointNoPairSystemState {
    pub electronic: SpinorScfReport,
    pub electronic_density: BlochMatrix,
    pub nuclei: Vec<JointNuclearState>,
    pub electronic_energy: f64,
    pub electron_nuclear_energy: f64,
    pub external_field_energy: f64,
    pub nuclear_pair_energy: f64,
    pub electron_nuclear_reciprocity_residual: f64,
    pub max_electron_nuclear_reciprocity_defect: f64,
    pub max_nuclear_pair_reciprocity_defect: f64,
    /// Projected electronic bias expectation minus the full-AO contraction.
    pub projection_residual: f64,
    pub energy: f64,
    pub iterations: usize,
    pub converged: bool,
}

struct NuclearIteration {
    offsets: Vec<usize>,
    orth: Orthogonalizer,
    sx: Matrix,
    coefficient: Vec<f64>,
    density: Matrix,
}

impl NuclearIteration {
    fn new(nucleus: &BalancedQuantumNucleus<'_>, cutoff: f64) -> Result<Self> {
        let offsets = shell_offsets(nucleus.shells)?;
        let n = *offsets.last().unwrap();
        let valid = |m: &Matrix| {
            m.rows == n
                && m.cols == n
                && m.as_slice().iter().all(|x| x.is_finite())
                && (0..n).all(|i| (0..i).all(|j| (m[(i, j)] - m[(j, i)]).abs() <= 1e-10))
        };
        if n == 0 || !valid(nucleus.overlap) || !valid(nucleus.one_body) {
            return Err(DftError::InvalidInput(
                "joint no-pair SCF needs finite symmetric nuclear matrices matching the nuclear basis".into(),
            ));
        }
        let orth = Orthogonalizer::canonical(nucleus.overlap, cutoff)?;
        if orth.n_kept != n {
            return Err(DftError::InvalidInput(
                "joint no-pair SCF requires full-rank nuclear overlaps".into(),
            ));
        }
        let (_, start) = orth.solve_roothaan(nucleus.one_body)?;
        let coefficient: Vec<_> = (0..n).map(|i| start[(i, 0)]).collect();
        Ok(Self {
            offsets,
            sx: nucleus.overlap.matmul_seq(&orth.x),
            orth,
            density: projector(&coefficient),
            coefficient,
        })
    }
}

fn block(matrix: &Matrix) -> IntegralBlock<f64, 2> {
    IntegralBlock {
        shape: [matrix.rows, matrix.cols],
        values: matrix.as_slice().to_vec(),
    }
}

fn matrix(block: IntegralBlock<f64, 2>) -> Matrix {
    Matrix::from_row_major(block.shape[0], block.shape[1], block.values)
}

impl<B: SpinorEnergyBackend> JointNoPairModel<'_, B> {
    /// Solve the coupled reference with a single quantum nucleus.
    pub fn solve(
        &self,
        electronic_config: &SpinorScfConfig,
        options: &MulticomponentOptions,
    ) -> Result<JointNoPairState> {
        let mut result = JointNoPairSystem {
            electronic: self.electronic,
            electronic_space: self.electronic_space,
            electronic_shells: self.electronic_shells,
            speed_of_light: self.speed_of_light,
            nuclei: std::slice::from_ref(&self.nucleus),
        }
        .solve(electronic_config, options)?;
        let nuclear = result.nuclei.pop().expect("single nucleus was supplied");
        Ok(JointNoPairState {
            electronic: result.electronic,
            electronic_density: result.electronic_density,
            nuclear_density: nuclear.density,
            nuclear_fock: nuclear.fock,
            nuclear_orbital_residual: nuclear.orbital_residual,
            nuclear_density_residual: nuclear.density_residual,
            nuclear_particle_number_error: nuclear.particle_number_error,
            electronic_energy: result.electronic_energy,
            nuclear_one_body_energy: nuclear.one_body_energy,
            interaction_energy: result.electron_nuclear_energy,
            reciprocity_residual: result.electron_nuclear_reciprocity_residual,
            projection_residual: result.projection_residual,
            energy: result.energy,
            iterations: result.iterations,
            converged: result.converged,
        })
    }
}

impl<B: SpinorEnergyBackend> JointNoPairSystem<'_, B> {
    fn coupling(
        &self,
        nucleus: &BalancedQuantumNucleus<'_>,
        [re, im]: &[IntegralBlock<f64, 2>; 2],
        nuclear: &Matrix,
    ) -> Result<BalancedCoulombFields<f64>> {
        electron_quantum_coulomb(BalancedCoulombInput {
            electronic_shells: self.electronic_shells,
            nuclear_shells: nucleus.shells,
            electronic_density_real: re,
            electronic_density_imaginary: im,
            nuclear_density: &block(nuclear),
            speed_of_light: self.speed_of_light,
            nuclear_charge: nucleus.charge,
            intrinsic_rms_radius: nucleus.intrinsic_rms_radius,
            order: 0,
        })
    }

    /// Solve all species in one energy functional. Each Coulomb pair enters
    /// the energy once and contributes to both conjugate one-body operators.
    pub fn solve(
        &self,
        electronic_config: &SpinorScfConfig,
        options: &MulticomponentOptions,
    ) -> Result<JointNoPairSystemState> {
        self.solve_in_field(electronic_config, options, None)
    }

    /// Solve at a specified uniform electric field in the same fixed sector.
    /// The balanced basis, classical centres and projector stay fixed.
    pub fn solve_in_field(
        &self,
        electronic_config: &SpinorScfConfig,
        options: &MulticomponentOptions,
        perturbation: Option<(&JointElectricOperators, [f64; 3])>,
    ) -> Result<JointNoPairSystemState> {
        self.solve_in_electric_field(electronic_config, options, perturbation)
    }

    /// Solve in the same fixed sector with uniform and gradient electric vertices.
    pub fn solve_in_electric_field<const N: usize>(
        &self,
        electronic_config: &SpinorScfConfig,
        options: &MulticomponentOptions,
        perturbation: Option<(&JointElectricOperators<N>, [f64; N])>,
    ) -> Result<JointNoPairSystemState> {
        if self.nuclei.is_empty()
            || !options.tolerance.is_finite()
            || options.tolerance <= 0.0
            || options.max_iterations == 0
            || !(0.0 < options.mixing && options.mixing <= 1.0)
        {
            return Err(DftError::InvalidInput(
                "joint no-pair SCF needs nuclei, positive species tolerances, iterations and mixing at most one".into(),
            ));
        }
        let mut nuclear: Vec<_> = self
            .nuclei
            .iter()
            .map(|nucleus| NuclearIteration::new(nucleus, electronic_config.overlap_cutoff))
            .collect::<Result<_>>()?;
        let dimension = self.electronic_space.basis().rows();
        let zero = block(&Matrix::zeros(dimension, dimension));
        let empty = [zero.clone(), zero];
        let electronic_backend = NoPairBackend::new(self.electronic_space, self.electronic)?;
        // The external potential is linear in the electric-field amplitudes.
        // Its vertices do not change the balanced basis or positive projector.
        let field_operators = if let Some((operators, field)) = perturbation {
            if field.iter().any(|x| !x.is_finite()) {
                return Err(DftError::InvalidInput(
                    "joint no-pair electric field requires finite components".into(),
                ));
            }
            operators.validate_dimensions(dimension, nuclear.iter().map(|n| n.density.rows))?;
            let mut electron_field = BlochMatrix::zeros(dimension, dimension);
            let mut nuclear_fields: Vec<_> = nuclear
                .iter()
                .map(|n| Matrix::zeros(n.density.rows, n.density.cols))
                .collect();
            let mut classical_field_energy = 0.0;
            for (axis, amplitude) in field.into_iter().enumerate() {
                electron_field.add_scaled(&operators.electronic[axis], amplitude);
                classical_field_energy += operators.classical[axis] * amplitude;
                for (f, ops) in nuclear_fields.iter_mut().zip(&operators.nuclear) {
                    add(f, &ops[axis], amplitude);
                }
            }
            Some((electron_field, nuclear_fields, classical_field_energy))
        } else {
            None
        };
        for iteration in 1..=options.max_iterations {
            let mut full_bias = field_operators.as_ref().map_or_else(
                || BlochMatrix::zeros(dimension, dimension),
                |(electron, _, _)| electron.clone(),
            );
            for (nucleus, state) in self.nuclei.iter().zip(&nuclear) {
                let fields = self.coupling(nucleus, &empty, &state.density)?;
                add(&mut full_bias.re, &matrix(fields.electronic_real), 1.0);
                add(&mut full_bias.im, &matrix(fields.electronic_imaginary), 1.0);
            }
            let bias = self.electronic_space.project_operator(&full_bias)?;
            let backend = BiasedSpinorBackend::new(&electronic_backend, Some(vec![bias]), None)?;
            let electronic = run_spinor_scf(&backend, electronic_config)?;
            let electronic_density = self.electronic_space.lift_density(&electronic.density[0])?;
            let electronic_blocks = [block(&electronic_density.re), block(&electronic_density.im)];
            let mut electron_nuclear_energy = 0.0;
            let mut reciprocity_sum = 0.0;
            let mut reciprocity_max = 0.0_f64;
            let mut focks = Vec::with_capacity(nuclear.len());
            for (nucleus, state) in self.nuclei.iter().zip(&nuclear) {
                let fields = self.coupling(nucleus, &electronic_blocks, &state.density)?;
                electron_nuclear_energy += fields.energy;
                reciprocity_sum += fields.reciprocity_residual;
                reciprocity_max = reciprocity_max.max(fields.reciprocity_residual.abs());
                let mut fock = matrix(fields.nuclear);
                add(&mut fock, nucleus.one_body, 1.0);
                if let Some((_, fields, _)) = &field_operators {
                    add(&mut fock, &fields[focks.len()], 1.0);
                }
                focks.push(fock);
            }
            let mut nuclear_pair_energy = 0.0;
            let mut nuclear_pair_defect = 0.0_f64;
            for a in 0..nuclear.len() {
                for b in 0..a {
                    let pair = nuclear_coulomb::pair(
                        [&self.nuclei[b], &self.nuclei[a]],
                        [&nuclear[b].density, &nuclear[a].density],
                        [&nuclear[b].offsets, &nuclear[a].offsets],
                    )?;
                    nuclear_pair_energy += pair.energy;
                    nuclear_pair_defect = nuclear_pair_defect.max(pair.reciprocity_defect);
                    add(&mut focks[b], &pair.first, 1.0);
                    add(&mut focks[a], &pair.second, 1.0);
                }
            }
            let bias_energy = backend.bias_energy(&electronic.density);
            // The projected electronic bias includes both the external field
            // and the quantum nuclear attraction. Subtract both contractions
            // when comparing it against their full-AO expectations.
            let (electronic_field_energy, external_field_energy) =
                field_operators
                    .as_ref()
                    .map_or((0.0, 0.0), |(electron, fields, classical)| {
                        let electronic = electronic_density.trace_product(electron);
                        let nuclear = nuclear
                            .iter()
                            .zip(fields)
                            .map(|(n, f)| n.density.frobenius_dot(f))
                            .sum::<f64>();
                        (electronic, classical + electronic + nuclear)
                    });
            let projection_residual =
                bias_energy - electron_nuclear_energy - electronic_field_energy;
            let electronic_energy = electronic.fields.energy - bias_energy;
            // Count each reciprocal Coulomb pair once in the joint energy.
            let mut energy = electronic_energy
                + electron_nuclear_energy
                + nuclear_pair_energy
                + external_field_energy;
            let mut residual = electronic
                .orbital_residual
                .max(reciprocity_max)
                .max(nuclear_pair_defect)
                .max(projection_residual.abs());
            let mut nuclei = Vec::with_capacity(nuclear.len());
            let mut proposals = Vec::with_capacity(nuclear.len());
            for ((nucleus, current), fock) in self.nuclei.iter().zip(&nuclear).zip(focks) {
                let (_, orbitals) = current.orth.solve_roothaan(&fock)?;
                let next: Vec<_> = (0..current.density.rows)
                    .map(|i| orbitals[(i, 0)])
                    .collect();
                let mut change = projector(&next);
                add(&mut change, &current.density, -1.0);
                let density_residual = max_abs(
                    &current
                        .sx
                        .transpose_matmul_seq(&change)
                        .matmul_seq(&current.sx),
                );
                let orbital_residual = max_abs(&metric_commutator(
                    &fock,
                    &current.density,
                    nucleus.overlap,
                    &current.orth.x,
                ));
                let particle_number_error = current.density.frobenius_dot(nucleus.overlap) - 1.0;
                let one_body_energy = current.density.frobenius_dot(nucleus.one_body);
                energy += one_body_energy;
                residual = residual
                    .max(density_residual)
                    .max(orbital_residual)
                    .max(particle_number_error.abs());
                nuclei.push(JointNuclearState {
                    density: current.density.clone(),
                    fock,
                    orbital_residual,
                    density_residual,
                    particle_number_error,
                    one_body_energy,
                });
                proposals.push(next);
            }
            if !energy.is_finite() || !residual.is_finite() {
                return Err(DftError::LinearAlgebra(
                    "non-finite joint no-pair SCF energy or residual".into(),
                ));
            }
            crate::logger::log(&format!(
                "[joint-no-pair] iteration {iteration:4} energy {energy:.14} residual {residual:.6e}"
            ));
            let converged = electronic.converged && residual <= options.tolerance;
            if converged || iteration == options.max_iterations {
                return Ok(JointNoPairSystemState {
                    electronic,
                    electronic_density,
                    nuclei,
                    electronic_energy,
                    electron_nuclear_energy,
                    external_field_energy,
                    nuclear_pair_energy,
                    electron_nuclear_reciprocity_residual: reciprocity_sum,
                    max_electron_nuclear_reciprocity_defect: reciprocity_max,
                    max_nuclear_pair_reciprocity_defect: nuclear_pair_defect,
                    projection_residual,
                    energy,
                    iterations: iteration,
                    converged,
                });
            }
            for ((current, next), nucleus) in nuclear.iter_mut().zip(proposals).zip(self.nuclei) {
                current.coefficient =
                    mix_orbital(&current.coefficient, &next, nucleus.overlap, options.mixing);
                if current.coefficient.iter().any(|value| !value.is_finite()) {
                    return Err(DftError::LinearAlgebra(
                        "joint no-pair nuclear damping lost the orbital norm".into(),
                    ));
                }
                current.density = projector(&current.coefficient);
            }
        }
        unreachable!("positive iteration limit and final-state return")
    }
}

#[cfg(test)]
mod tests;
