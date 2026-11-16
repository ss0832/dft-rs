// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-sector orbital response of coupled Dirac electrons and quantum nuclei.

use super::{add, block, matrix, nuclear_coulomb, JointNoPairSystem, JointNoPairSystemState};
use crate::dfpt::orbitals::adjoint;
use crate::integrals::{analytic_basis, dirac};
use crate::linalg::{Matrix, Orthogonalizer};
use crate::periodic::BlochMatrix;
use crate::properties::no_pair::stationary::DiracCoulombBackend;
use crate::response::complex::{dot, Complex};
use crate::response::electric_tensor::{quadrupole_modes, quadrupole_weights, QUADRUPOLE_AXES};
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::spinor::PairedOrbitals;
use crate::{DftError, Result};

/// Covariant electric perturbation operators for all particle sectors.
/// The first three entries are minus the dipole. The optional last five
/// entries are minus the field-conjugate traceless charge moments.
pub struct JointElectricOperators<const N: usize = 3> {
    pub(super) electronic: [BlochMatrix; N],
    pub(super) nuclear: Vec<[Matrix; N]>,
    pub(super) classical: [f64; N],
}

const DIPOLE_POWERS: [[usize; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

impl JointElectricOperators<3> {
    /// The fixed classical contribution is minus its total dipole.
    pub fn new<B>(model: &JointNoPairSystem<'_, B>, classical: [f64; 3]) -> Result<Self> {
        Self::from_moments(model, DIPOLE_POWERS, dipole_weights(), [0.0; 3], classical)
    }

    /// Total permanent dipole at laboratory zero, including classical charges.
    pub fn dipole(&self, state: &JointNoPairSystemState) -> Result<[f64; 3]> {
        self.field_conjugate_moments(state)
    }
}

fn dipole_weights<const M: usize>() -> [[f64; M]; 3] {
    std::array::from_fn(|a| std::array::from_fn(|b| f64::from(u8::from(a == b))))
}

impl JointElectricOperators<8> {
    /// Uniform field and five orthonormal traceless gradient modes at one origin.
    pub fn with_field_gradient<B>(
        model: &JointNoPairSystem<'_, B>,
        classical: &[dirac::GaussianNucleus<f64>],
        origin: [f64; 3],
    ) -> Result<Self> {
        let mut powers = [[0; 3]; 9];
        powers[..3].copy_from_slice(&DIPOLE_POWERS);
        for (power, (a, b)) in powers[3..].iter_mut().zip(QUADRUPOLE_AXES) {
            power[a] += 1;
            power[b] += 1;
        }
        let mut weights = [[0.0; 9]; 8];
        weights[..3].copy_from_slice(&dipole_weights());
        for (row, mode) in weights[3..].iter_mut().zip(quadrupole_modes()) {
            row[3..].copy_from_slice(&quadrupole_weights(mode));
        }
        // The intrinsic isotropic Gaussian variance enters only the trace
        // of the Cartesian second moments and cancels in every gradient mode.
        let mut source = [0.0; 8];
        for nucleus in classical {
            let r = std::array::from_fn::<_, 3, _>(|a| nucleus.center[a] - origin[a]);
            let mut moments = [0.0; 9];
            moments[..3].copy_from_slice(&r);
            for (moment, (a, b)) in moments[3..].iter_mut().zip(QUADRUPOLE_AXES) {
                *moment = r[a] * r[b];
            }
            for (value, row) in source.iter_mut().zip(weights) {
                *value -= nucleus.charge * row.iter().zip(moments).map(|(w, m)| w * m).sum::<f64>();
            }
        }
        Self::from_moments(model, powers, weights, origin, source)
    }
}

impl<const N: usize> JointElectricOperators<N> {
    fn from_moments<B, const M: usize>(
        model: &JointNoPairSystem<'_, B>,
        powers: [[usize; 3]; M],
        weights: [[f64; M]; N],
        origin: [f64; 3],
        classical: [f64; N],
    ) -> Result<Self> {
        if classical.iter().chain(&origin).any(|x| !x.is_finite()) {
            return Err(DftError::InvalidInput(
                "nonfinite joint electric origin or classical source".into(),
            ));
        }
        let moments = dirac::balanced_electric_moments(
            model.electronic_shells,
            model.speed_of_light,
            powers,
            origin,
        )?;
        let dimension = moments[0].rows();
        let electronic = weights.map(|row| {
            let mut operator = BlochMatrix::zeros(dimension, dimension);
            for (weight, moment) in row.into_iter().zip(&moments).filter(|(w, _)| *w != 0.0) {
                operator.add_scaled(moment, weight);
            }
            operator
        });
        let nuclear = model
            .nuclei
            .iter()
            .map(|nucleus| {
                let offsets = dirac::shell_offsets(nucleus.shells)?;
                let n = *offsets.last().unwrap();
                let mut operators = std::array::from_fn(|_| Matrix::zeros(n, n));
                for (i, a) in nucleus.shells.iter().enumerate() {
                    for (j, b) in nucleus.shells.iter().enumerate() {
                        for (term, power) in powers.into_iter().enumerate() {
                            let block = analytic_basis::moment(a, b, power, origin)?;
                            let [rows, columns] = block.shape;
                            for (operator, weights) in operators.iter_mut().zip(weights) {
                                let weight = -nucleus.charge * weights[term];
                                if weight == 0.0 {
                                    continue;
                                }
                                for row in 0..rows {
                                    for column in 0..columns {
                                        operator[(offsets[i] + row, offsets[j] + column)] +=
                                            weight * block.values[row * columns + column];
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(operators)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            electronic,
            nuclear,
            classical,
        })
    }

    pub(super) fn validate_dimensions(
        &self,
        electronic: usize,
        nuclear: impl ExactSizeIterator<Item = usize>,
    ) -> Result<()> {
        if self.electronic.iter().any(|o| o.rows() != electronic)
            || self.nuclear.len() != nuclear.len()
            || self
                .nuclear
                .iter()
                .zip(nuclear)
                .any(|(ops, n)| ops.iter().any(|o| o.rows != n))
        {
            return Err(DftError::InvalidInput(
                "joint no-pair electric operators do not match the electronic and nuclear dimensions".into(),
            ));
        }
        Ok(())
    }

    /// Negative total field derivatives including the fixed classical charges.
    /// For a gradient mode the Buckingham quadrupole is three times this value.
    pub fn field_conjugate_moments(&self, state: &JointNoPairSystemState) -> Result<[f64; N]> {
        self.validate_dimensions(
            state.electronic_density.rows(),
            state.nuclei.iter().map(|n| n.density.rows),
        )?;
        Ok(std::array::from_fn(|a| {
            -self.classical[a]
                - state.electronic_density.trace_product(&self.electronic[a])
                - state
                    .nuclei
                    .iter()
                    .zip(&self.nuclear)
                    .map(|(state, operators)| state.density.frobenius_dot(&operators[a]))
                    .sum::<f64>()
        }))
    }
}

pub struct JointElectricResponse<const N: usize = 3> {
    /// Total field-conjugate response, [observable][field], in atomic units.
    pub tensor: [[Complex; N]; N],
    pub frequency: Complex,
    pub relative_residuals: [f64; N],
    pub iterations: [usize; N],
    /// Maximum raw electronic/nuclear stationary orbital residual, Hartree.
    pub reference_residual: f64,
}

struct CoupledResponse<'a, 'b, 'c> {
    model: &'a JointNoPairSystem<'b, DiracCoulombBackend<'c>>,
    frames: Vec<PairedOrbitals>,
    nuclear_offsets: Vec<Vec<usize>>,
    frequency: Complex,
}

fn real_operator(re: Matrix) -> BlochMatrix {
    let im = Matrix::zeros(re.rows, re.cols);
    BlochMatrix { re, im }
}

impl CoupledResponse<'_, '_, '_> {
    fn hermitian_kernel(&self, density: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        let full = self.model.electronic_space.lift_density(&density[0])?;
        // J-K is the linear derivative of the electronic HF Fock. The
        // electron-quantum Coulomb map also returns the nuclear image of the
        // electronic variation and the electronic image of each nuclear variation.
        let mut electronic = self.model.electronic.density_kernel(&full)?;
        let electronic_density = [block(&full.re), block(&full.im)];
        let mut nuclear = Vec::with_capacity(self.model.nuclei.len());
        for (nucleus, variation) in self.model.nuclei.iter().zip(&density[1..]) {
            // Coulomb sees the position density of a nucleus. The antisymmetric
            // imaginary AO part does not contribute to this scalar density.
            let fields = self
                .model
                .coupling(nucleus, &electronic_density, &variation.re)?;
            add(&mut electronic.re, &matrix(fields.electronic_real), 1.0);
            add(
                &mut electronic.im,
                &matrix(fields.electronic_imaginary),
                1.0,
            );
            nuclear.push(matrix(fields.nuclear));
        }
        for a in 0..nuclear.len() {
            for b in 0..a {
                let pair = nuclear_coulomb::pair(
                    [&self.model.nuclei[b], &self.model.nuclei[a]],
                    [&density[b + 1].re, &density[a + 1].re],
                    [&self.nuclear_offsets[b], &self.nuclear_offsets[a]],
                )?;
                add(&mut nuclear[b], &pair.first, 1.0);
                add(&mut nuclear[a], &pair.second, 1.0);
            }
        }
        let electronic = self.model.electronic_space.project_operator(&electronic)?;
        Ok(std::iter::once(electronic)
            .chain(nuclear.into_iter().map(real_operator))
            .collect())
    }

    fn kernel(&self, density: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        // The physical Fock derivative acts on Hermitian density variations.
        // Extend it complex-linearly to the independent +z and -z amplitudes.
        // D = H + iK with H = (D + D^dagger)/2 and K = (D - D^dagger)/(2i).
        let mut real = density.to_vec();
        let mut imag = Vec::with_capacity(density.len());
        for (real, d) in real.iter_mut().zip(density) {
            let conjugate = adjoint(d)?;
            real.add_scaled(&conjugate, 1.0);
            real.scale(0.5);
            let mut anti = d.clone();
            anti.add_scaled(&conjugate, -1.0);
            imag.push(rotate_by_i(anti, -0.5));
        }
        let mut result = self.hermitian_kernel(&real)?;
        if imag.iter().any(|d| {
            d.re.as_slice()
                .iter()
                .chain(d.im.as_slice())
                .any(|&x| x != 0.0)
        }) {
            for (out, imaginary) in result.iter_mut().zip(self.hermitian_kernel(&imag)?) {
                out.add_scaled(&rotate_by_i(imaginary, 1.0), 1.0);
            }
        }
        Ok(result)
    }

    fn vertex<const N: usize>(
        &self,
        field: usize,
        operators: &JointElectricOperators<N>,
    ) -> Result<Vec<Complex>> {
        let mut output = self.frames[0].projected(
            &self
                .model
                .electronic_space
                .project_operator(&operators.electronic[field])?,
        )?;
        for (frame, source) in self.frames[1..].iter().zip(&operators.nuclear) {
            output.extend(frame.projected(&real_operator(source[field].clone()))?);
        }
        Ok(output)
    }
}

fn rotate_by_i(mut matrix: BlochMatrix, scale: f64) -> BlochMatrix {
    std::mem::swap(&mut matrix.re, &mut matrix.im);
    for x in matrix.re.as_mut_slice() {
        *x *= -scale;
    }
    for x in matrix.im.as_mut_slice() {
        *x *= scale;
    }
    matrix
}

impl LinearOperator for CoupledResponse<'_, '_, '_> {
    fn dimension(&self) -> usize {
        self.frames.iter().map(PairedOrbitals::dimension).sum()
    }

    fn structure(&self) -> OperatorStructure {
        if self.frequency.im == 0.0 {
            OperatorStructure::Hermitian
        } else {
            OperatorStructure::General
        }
    }

    fn apply(&self, amplitudes: &[Complex]) -> Result<Vec<Complex>> {
        // Each species conserves its own occupation. There are no orbital
        // rotations between species, only coupling through the induced Focks.
        let mut remaining = amplitudes;
        let mut density = Vec::with_capacity(self.frames.len());
        for frame in &self.frames {
            let (local, rest) = remaining.split_at(frame.dimension());
            density.push(frame.density_response(local)?);
            remaining = rest;
        }
        let mut remaining = amplitudes;
        let mut result = Vec::with_capacity(amplitudes.len());
        for (frame, induced) in self.frames.iter().zip(self.kernel(&density)?) {
            let (local, rest) = remaining.split_at(frame.dimension());
            let mut image = frame.projected(&induced)?;
            frame.add_free_action(&mut image, local, self.frequency);
            result.extend(image);
            remaining = rest;
        }
        Ok(result)
    }
}

impl JointNoPairSystem<'_, DiracCoulombBackend<'_>> {
    /// Coupled electric response in the same fixed sector as the reference.
    /// The nuclear and electronic forward/backward amplitudes are solved together.
    pub fn electric_response<const N: usize>(
        &self,
        state: &JointNoPairSystemState,
        operators: &JointElectricOperators<N>,
        frequency: Complex,
        overlap_cutoff: f64,
        config: &ResponseSolverConfig,
    ) -> Result<JointElectricResponse<N>> {
        config.validate()?;
        let reference_residual = state
            .nuclei
            .iter()
            .map(|state| state.orbital_residual)
            .fold(state.electronic.orbital_residual, f64::max);
        if !state.converged
            || !frequency.is_finite()
            || reference_residual > config.acceptance()
            || state.nuclei.len() != self.nuclei.len()
        {
            return Err(DftError::InvalidInput(
                "joint no-pair response requires a matching converged reference, finite frequency and an accepted stationary residual".into(),
            ));
        }
        operators.validate_dimensions(
            self.electronic_space.basis().rows(),
            self.nuclei.iter().map(|n| n.overlap.rows),
        )?;
        let ([band], [occupations]) = (
            state.electronic.bands.as_slice(),
            state.electronic.occupations.as_slice(),
        ) else {
            return Err(DftError::InvalidInput(
                "joint no-pair response requires one electronic orbital frame".into(),
            ));
        };
        if occupations.len() != band.energies.len()
            || band.coefficients.rows() != self.electronic_space.basis().cols()
            || band.coefficients.cols() != occupations.len()
        {
            return Err(DftError::InvalidInput(
                "joint no-pair electronic orbital and occupation dimensions disagree".into(),
            ));
        }
        let occupied = occupations.iter().take_while(|&&x| x == 1.0).count();
        if occupied == 0 || occupations[occupied..].iter().any(|&x| x != 0.0) {
            return Err(DftError::Unsupported {
                feature: "fractional fixed-sector joint electric response".into(),
                missing: "coupled occupation derivatives".into(),
            });
        }
        let mut frames = vec![PairedOrbitals::new(
            &band.coefficients,
            &band.energies,
            occupied,
        )];
        let mut nuclear_offsets = Vec::with_capacity(self.nuclei.len());
        for (nucleus, state) in self.nuclei.iter().zip(&state.nuclei) {
            let orth = Orthogonalizer::canonical(nucleus.overlap, overlap_cutoff)?;
            let (energies, orbitals) = orth.solve_roothaan(&state.fock)?;
            frames.push(PairedOrbitals::new(&real_operator(orbitals), &energies, 1));
            nuclear_offsets.push(dirac::shell_offsets(nucleus.shells)?);
        }
        let operator = CoupledResponse {
            model: self,
            frames,
            nuclear_offsets,
            frequency,
        };
        let vertices = (0..N)
            .map(|axis| operator.vertex(axis, operators))
            .collect::<Result<Vec<_>>>()?;
        let mut report = JointElectricResponse {
            tensor: [[Complex::ZERO; N]; N],
            frequency,
            relative_residuals: [0.0; N],
            iterations: [0; N],
            reference_residual,
        };
        if operator.dimension() != 0 {
            for (b, vertex) in vertices.iter().enumerate() {
                let rhs: Vec<_> = vertex.iter().map(|v| -*v).collect();
                let solved = krylov::solve(&operator, &rhs, None, config)?;
                // The source is minus the conjugate moment. Contract both
                // transition blocks to obtain its response to the field.
                for (a, observable) in vertices.iter().enumerate() {
                    report.tensor[a][b] = -dot(observable, &solved.amplitudes);
                }
                report.relative_residuals[b] = solved.relative_residual;
                report.iterations[b] = solved.iterations;
            }
        }
        Ok(report)
    }
}
