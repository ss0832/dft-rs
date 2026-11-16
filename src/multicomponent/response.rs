// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared electronic/quantum-particle orbital response, including mixed XC.

use super::correlation::CorrelationModel;
use super::{
    add, append_basis, block, orbital_density, CorrelationGrid, MulticomponentResult,
    QuantumParticle,
};
use crate::basis::MolBasis;
use crate::grid::{eval::eval_aos, MolecularGrid};
use crate::integrals::{dipole_matrices, CoulombKernel};
use crate::linalg::{Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::response::dynamic::Frequency;
use crate::response::krylov::{self, LinearOperator, OperatorStructure};
use crate::response::{ResponseSolver, ResponseSolverConfig};
use crate::scf::{JkEngine, ScfOptions};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

mod electric_quadrupole;
pub use electric_quadrupole::CoupledQuadrupoleResponse;

pub(super) struct Frame {
    pub(super) coefficients: Matrix,
    pub(super) energies: Vec<f64>,
    pub(super) occupied: usize,
}

/// First-order physical AO densities and the fresh coupled equation residual.
#[derive(Clone, Debug)]
pub struct CoupledDensityResponse {
    pub electronic_density: [Matrix; 2],
    pub quantum_density: Vec<Matrix>,
    pub amplitudes: Vec<f64>,
    pub relative_residual: f64,
    pub reference_residual: f64,
    pub iterations: usize,
    pub converged: bool,
    pub position_multipliers: Vec<[f64; 3]>,
    pub position_residual: f64,
}

#[derive(Clone, Debug)]
pub struct CoupledPolarizability {
    pub tensor: [[f64; 3]; 3],
    pub raw_asymmetry: f64,
    pub responses: Vec<CoupledDensityResponse>,
}

/// Cartesian dipole response for one signed frequency in atomic units.
#[derive(Clone, Debug)]
pub struct CoupledDynamicPolarizability {
    pub tensor: [[Complex; 3]; 3],
    pub residuals: [f64; 3],
    pub position_residuals: [f64; 3],
    pub iterations: [usize; 3],
    pub reference_residual: f64,
}

/// Variational orbital Hessian of every connected particle space. Restricted
/// electronic amplitudes use sqrt(2) normalization, making their coupling to
/// singly occupied quantum particles Hermitian without altering physical gaps.
pub struct CoupledResponseSolver<'a> {
    parent: ResponseSolver<'a>,
    electronic_basis: &'a MolBasis,
    particles: &'a [QuantumParticle],
    correlation: &'a CorrelationModel,
    state: &'a MulticomponentResult,
    combined: MolBasis,
    engine: JkEngine,
    offsets: Vec<usize>,
    pub(super) frames: Vec<Frame>,
    slots: Vec<(usize, usize, usize)>,
    pub(super) electronic_blocks: usize,
    grid: Option<CorrelationGrid>,
    base_density: Vec<Vec<f64>>,
}

fn density_variation(frame: &Frame, amplitude: &Matrix, transpose_sign: f64) -> Matrix {
    let c = &frame.coefficients;
    let mut rotation = Matrix::zeros(c.cols, c.cols);
    for a in frame.occupied..c.cols {
        for i in 0..frame.occupied {
            rotation[(a, i)] = amplitude[(a, i)];
            rotation[(i, a)] = transpose_sign * amplitude[(a, i)];
        }
    }
    c.matmul_seq(&rotation).matmul_transpose_seq(c)
}

impl<'a> CoupledResponseSolver<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        molecule: &Molecule,
        electronic_basis: &'a MolBasis,
        options: &ScfOptions,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        particles: &'a [QuantumParticle],
        correlation: &'a CorrelationModel,
        state: &'a MulticomponentResult,
    ) -> Result<Self> {
        if !state.converged
            || particles.len() != state.quantum.len()
            || electronic_basis.nao_sph != state.electronic_reference.overlap.rows
        {
            return Err(DftError::InvalidInput("coupled response requires its converged matching electronic/quantum-particle reference".into()));
        }
        let e = &state.electronic_reference;
        let parent = ResponseSolver::new(molecule, electronic_basis, functional, grid, e, options)?;
        let electronic_blocks = if e.unrestricted { 2 } else { 1 };
        let mut frames: Vec<_> = (0..electronic_blocks)
            .map(|s| Frame {
                coefficients: e.mo_coeff[s].clone(),
                energies: e.mo_energy[s].clone(),
                occupied: [e.n_alpha, e.n_beta][s],
            })
            .collect();
        let mut combined = electronic_basis.clone();
        let mut offsets = vec![0];
        for (p, q) in particles.iter().zip(&state.quantum) {
            if p.basis.nao_sph != q.overlap.rows {
                return Err(DftError::InvalidInput(
                    "coupled response quantum basis dimensions disagree".into(),
                ));
            }
            let orth = Orthogonalizer::canonical(&q.overlap, options.orth_threshold)?;
            let (energies, coefficients) = orth.solve_roothaan(&q.fock)?;
            frames.push(Frame {
                coefficients,
                energies,
                occupied: 1,
            });
            offsets.push(combined.nao_sph);
            append_basis(&mut combined, &p.basis);
        }
        let engine = JkEngine::new(
            &combined,
            options.effective_eri_mode(combined.nao_sph),
            CoulombKernel::Coulomb,
        )?;
        let slots = frames
            .iter()
            .enumerate()
            .flat_map(|(s, f)| {
                (f.occupied..f.coefficients.cols)
                    .flat_map(move |a| (0..f.occupied).map(move |i| (s, a, i)))
            })
            .collect();
        let correlation_grid = if correlation.is_zero() {
            None
        } else {
            let grid = grid.ok_or_else(|| {
                DftError::InvalidInput(
                    "coupled interspecies XC response requires its reference grid".into(),
                )
            })?;
            Some(CorrelationGrid {
                weights: grid.weights.clone(),
                phi: std::iter::once(electronic_basis)
                    .chain(particles.iter().map(|p| &p.basis))
                    .map(|b| eval_aos(b, &grid.points, false).phi)
                    .collect(),
            })
        };
        let mut base_density = Vec::new();
        if let Some(g) = &correlation_grid {
            let mut density = orbital_density(&g.phi[0], &e.mo_coeff[0], &e.occupations[0]);
            for (v, b) in density.iter_mut().zip(orbital_density(
                &g.phi[0],
                &e.mo_coeff[1],
                &e.occupations[1],
            )) {
                *v += b;
            }
            base_density.push(density);
            for i in 0..particles.len() {
                base_density.push(orbital_density(
                    &g.phi[i + 1],
                    &frames[electronic_blocks + i].coefficients,
                    &[1.0],
                ));
            }
        }
        Ok(Self {
            parent,
            electronic_basis,
            particles,
            correlation,
            state,
            combined,
            engine,
            offsets,
            frames,
            slots,
            electronic_blocks,
            grid: correlation_grid,
            base_density,
        })
    }

    pub fn dimension(&self) -> usize {
        self.slots.len()
    }

    /// Orbital-energy gaps of the rotation slots (diagonal of the uncoupled
    /// Hessian).
    pub fn orbital_gaps(&self) -> Vec<f64> {
        self.slots
            .iter()
            .map(|&(s, a, i)| self.frames[s].energies[a] - self.frames[s].energies[i])
            .collect()
    }

    /// Occupied orbitals of every species rotated by `step` times the
    /// normalized amplitudes `mode` (restricted electronic amplitudes carry
    /// the factor sqrt(2) of the Hessian normalization) and orthonormalized:
    /// `C_o' = (C_o + step C_v K)(1 + step^2 K^T K)^(-1/2)`.
    pub fn rotated_state(&self, mode: &[f64], step: f64) -> Result<super::InitialState> {
        if mode.len() != self.dimension() || !step.is_finite() {
            return Err(DftError::InvalidInput(
                "a coupled rotation needs one finite amplitude per rotation slot".into(),
            ));
        }
        let mut kappa: Vec<Matrix> = self
            .frames
            .iter()
            .map(|f| Matrix::zeros(f.coefficients.cols - f.occupied, f.occupied))
            .collect();
        for (&(s, a, i), &x) in self.slots.iter().zip(mode) {
            let scale = if s < self.electronic_blocks && self.electronic_blocks == 1 {
                1.0 / 2.0_f64.sqrt()
            } else {
                1.0
            };
            kappa[s][(a - self.frames[s].occupied, i)] = step * scale * x;
        }
        let occupied: Vec<Matrix> = self
            .frames
            .iter()
            .zip(&kappa)
            .map(|(f, k)| -> Result<Matrix> {
                let c = &f.coefficients;
                let (no, nv) = (f.occupied, c.cols - f.occupied);
                let mut out = Matrix::zeros(c.rows, no);
                for m in 0..c.rows {
                    for i in 0..no {
                        let mut value = c[(m, i)];
                        for a in 0..nv {
                            value += c[(m, no + a)] * k[(a, i)];
                        }
                        out[(m, i)] = value;
                    }
                }
                let mut metric = k.transpose_matmul_seq(k);
                for i in 0..no {
                    metric[(i, i)] += 1.0;
                }
                let (values, vectors) = crate::linalg::symmetric_eigen(&metric)?;
                let mut inverse_root = Matrix::zeros(no, no);
                for (q, value) in values.iter().enumerate() {
                    for i in 0..no {
                        for j in 0..no {
                            inverse_root[(i, j)] +=
                                vectors[(i, q)] * vectors[(j, q)] / value.sqrt();
                        }
                    }
                }
                Ok(out.matmul_seq(&inverse_root))
            })
            .collect::<Result<_>>()?;
        let projector = |c: &Matrix| c.matmul_transpose_seq(c);
        let electronic_density = if self.electronic_blocks == 1 {
            let d = projector(&occupied[0]);
            [d.clone(), d]
        } else {
            [projector(&occupied[0]), projector(&occupied[1])]
        };
        let quantum = occupied[self.electronic_blocks..]
            .iter()
            .map(|c| (0..c.rows).map(|m| c[(m, 0)]).collect())
            .collect();
        Ok(super::InitialState {
            electronic_density,
            quantum,
        })
    }

    fn densities(&self, amplitudes: &[f64]) -> ([Matrix; 2], Vec<Matrix>) {
        let mut delta = self.density_blocks(amplitudes, self.frames.len(), 1.0);
        let electronic = self.electronic_densities(&delta);
        (electronic, delta.split_off(self.electronic_blocks))
    }

    fn density_blocks(
        &self,
        amplitudes: &[f64],
        blocks: usize,
        transpose_sign: f64,
    ) -> Vec<Matrix> {
        let mut rotations: Vec<_> = self
            .frames
            .iter()
            .take(blocks)
            .map(|f| Matrix::zeros(f.coefficients.cols, f.coefficients.cols))
            .collect();
        for (&(s, a, i), x) in self.slots.iter().zip(amplitudes) {
            if let Some(rotation) = rotations.get_mut(s) {
                rotation[(a, i)] = *x;
            }
        }
        self.frames
            .iter()
            .zip(rotations)
            .map(|(f, a)| density_variation(f, &a, transpose_sign))
            .collect()
    }

    fn electronic_densities(&self, delta: &[Matrix]) -> [Matrix; 2] {
        if self.electronic_blocks == 1 {
            let mut d = delta[0].clone();
            for v in d.as_mut_slice() {
                *v /= 2.0_f64.sqrt();
            }
            [d.clone(), d]
        } else {
            [delta[0].clone(), delta[1].clone()]
        }
    }

    fn coulomb(&self, p: &Matrix, offset: usize) -> Result<Matrix> {
        let n = self.combined.nao_sph;
        let mut padded = Matrix::zeros(n, n);
        for i in 0..p.rows {
            for j in 0..p.cols {
                padded[(offset + i, offset + j)] = p[(i, j)];
            }
        }
        let zero = Matrix::zeros(n, n);
        Ok(self
            .engine
            .build(&self.combined, &padded, [&zero, &zero], false)?
            .0)
    }

    fn induced_potentials(
        &self,
        e: &[Matrix; 2],
        q: &[Matrix],
        amplitudes: &[f64],
    ) -> Result<([Matrix; 2], Vec<Matrix>)> {
        let mut fe = self.parent.fock_response(e)?;
        let mut total = e[0].clone();
        add(&mut total, &e[1], 1.0);
        let je = self.coulomb(&total, 0)?;
        let jq = q
            .iter()
            .enumerate()
            .map(|(i, p)| self.coulomb(p, self.offsets[i + 1]))
            .collect::<Result<Vec<_>>>()?;
        let mut fq = Vec::new();
        for i in 0..q.len() {
            let dim = q[i].rows;
            let mut f = block(&je, self.offsets[i + 1], dim);
            for v in f.as_mut_slice() {
                *v *= -self.particles[i].charge;
            }
            for j in 0..q.len() {
                if i != j {
                    add(
                        &mut f,
                        &block(&jq[j], self.offsets[i + 1], dim),
                        self.particles[i].charge * self.particles[j].charge,
                    );
                }
            }
            let v = block(&jq[i], 0, self.electronic_basis.nao_sph);
            for f in &mut fe {
                add(f, &v, -self.particles[i].charge);
            }
            fq.push(f);
        }
        if let Some(grid) = &self.grid {
            // d rho = 2 psi d psi preserves an exact zero tangent at an
            // orbital node. Contracting an indefinite AO density instead
            // can create a spurious nonzero tangent by cancellation.
            let mut variations = vec![vec![0.0; grid.weights.len()]; grid.phi.len()];
            let waves: Vec<_> = self
                .frames
                .iter()
                .enumerate()
                .map(|(s, f)| {
                    grid.phi[if s < self.electronic_blocks {
                        0
                    } else {
                        s - self.electronic_blocks + 1
                    }]
                    .matmul_seq(&f.coefficients)
                })
                .collect();
            for (&(s, a, i), &x) in self.slots.iter().zip(amplitudes) {
                let block = if s < self.electronic_blocks {
                    0
                } else {
                    s - self.electronic_blocks + 1
                };
                let scale = if s < self.electronic_blocks && self.electronic_blocks == 1 {
                    2.0_f64.sqrt()
                } else {
                    1.0
                };
                for g in 0..grid.weights.len() {
                    variations[block][g] += 2.0 * scale * x * waves[s][(g, i)] * waves[s][(g, a)];
                }
            }
            let de = &variations[0];
            let channels = self.correlation.len();
            let mut p = vec![vec![0.0; de.len()]; channels];
            let mut dp = p.clone();
            for i in 0..q.len() {
                if let Some(c) = self.particles[i].correlation_channel {
                    let d = &variations[i + 1];
                    for g in 0..de.len() {
                        p[c][g] += self.base_density[i + 1][g];
                        dp[c][g] += d[g];
                    }
                }
            }
            // Kernel actions on the electrons (summed over channels) and on
            // every channel.
            let mut on_electrons = vec![0.0; de.len()];
            let mut on_channel = vec![vec![0.0; de.len()]; channels];
            for (c, provider) in self.correlation.channels().iter().enumerate() {
                if provider.is_zero() {
                    continue;
                }
                let action = (0..de.len())
                    .map(|g| {
                        provider.kernel_action(
                            self.base_density[0][g],
                            p[c][g],
                            super::correlation::representable_tangent(
                                [self.base_density[0][g], p[c][g]],
                                [de[g], dp[c][g]],
                            ),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                for (g, a) in action.iter().enumerate() {
                    on_electrons[g] += a[0];
                    on_channel[c][g] = a[1];
                }
            }
            for s in 0..grid.phi.len() {
                let values = if s == 0 {
                    &on_electrons
                } else if let Some(c) = self.particles[s - 1].correlation_channel {
                    &on_channel[c]
                } else {
                    continue;
                };
                let phi = &grid.phi[s];
                let mut potential = Matrix::zeros(phi.cols, phi.cols);
                for g in 0..de.len() {
                    let v = grid.weights[g] * values[g];
                    for i in 0..phi.cols {
                        for j in 0..phi.cols {
                            potential[(i, j)] += v * phi[(g, i)] * phi[(g, j)];
                        }
                    }
                }
                if s == 0 {
                    for f in &mut fe {
                        add(f, &potential, 1.0);
                    }
                } else {
                    add(&mut fq[s - 1], &potential, 1.0);
                }
            }
        }
        Ok((fe, fq))
    }

    pub(super) fn project(&self, e: &[Matrix; 2], q: &[Matrix]) -> Vec<f64> {
        let f: Vec<_> = self
            .frames
            .iter()
            .enumerate()
            .map(|(s, frame)| {
                let ao = if s < self.electronic_blocks {
                    &e[s]
                } else {
                    &q[s - self.electronic_blocks]
                };
                frame
                    .coefficients
                    .transpose_matmul_seq(ao)
                    .matmul_seq(&frame.coefficients)
            })
            .collect();
        self.slots
            .iter()
            .map(|&(s, a, i)| {
                f[s][(a, i)]
                    * if s < self.electronic_blocks && self.electronic_blocks == 1 {
                        2.0_f64.sqrt()
                    } else {
                        1.0
                    }
            })
            .collect()
    }

    /// Action of the complete coupled orbital Hessian, without matrix assembly.
    pub fn apply_real(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.dimension() || x.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "coupled orbital amplitude dimensions or values are invalid".into(),
            ));
        }
        let (e, q) = self.densities(x);
        let (fe, fq) = self.induced_potentials(&e, &q, x)?;
        let mut result = self.project(&fe, &fq);
        for (n, &(s, a, i)) in self.slots.iter().enumerate() {
            result[n] += (self.frames[s].energies[a] - self.frames[s].energies[i]) * x[n];
        }
        Ok(result)
    }

    /// Solve the response to explicit physical first-order AO Fock sources.
    pub fn solve(
        &self,
        electronic_source: &[Matrix; 2],
        quantum_source: &[Matrix],
        config: &ResponseSolverConfig,
    ) -> Result<CoupledDensityResponse> {
        self.solve_position_sources(
            electronic_source,
            quantum_source,
            &vec![0.0; 3 * self.particles.len()],
            config,
        )
    }

    pub(super) fn solve_position_sources(
        &self,
        electronic_source: &[Matrix; 2],
        quantum_source: &[Matrix],
        position_source: &[f64],
        config: &ResponseSolverConfig,
    ) -> Result<CoupledDensityResponse> {
        if self.state.position_constraints.is_none() {
            return self.solve_unconstrained(electronic_source, quantum_source, config);
        }
        let reference_residual = self.validate_sources(electronic_source, quantum_source)?;
        let n = self.dimension();
        let columns = self.position_columns();
        if position_source.len() != columns.len() || position_source.iter().any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput("position response requires one finite explicit constraint derivative per target coordinate".into()));
        }
        let operator = PositionOperator {
            orbital: self,
            columns: &columns,
        };
        let mut rhs: Vec<_> = self
            .project(electronic_source, quantum_source)
            .into_iter()
            .map(|x| Complex::new(-x, 0.0))
            .collect();
        rhs.extend(position_source.iter().map(|x| Complex::new(-0.5 * x, 0.0)));
        let solution = krylov::solve(&operator, &rhs, None, config)?;
        let amplitudes: Vec<_> = solution.amplitudes[..n].iter().map(|z| z.re).collect();
        let (e, q) = self.densities(&amplitudes);
        let position_residual = columns
            .iter()
            .zip(position_source)
            .map(|(r, c)| {
                (2.0 * r.iter().zip(&amplitudes).map(|(a, b)| a * b).sum::<f64>() + c).abs()
            })
            .fold(0.0_f64, f64::max);
        let constraint_reference = self
            .state
            .position_constraints
            .as_ref()
            .map_or(0.0, |c| c.residual);
        Ok(CoupledDensityResponse {
            electronic_density: e,
            quantum_density: q,
            amplitudes,
            position_multipliers: solution.amplitudes[n..]
                .chunks_exact(3)
                .map(|c| [c[0].re, c[1].re, c[2].re])
                .collect(),
            position_residual,
            relative_residual: solution.relative_residual,
            reference_residual: reference_residual.max(constraint_reference),
            iterations: solution.iterations,
            converged: reference_residual.max(constraint_reference) <= config.acceptance()
                && solution.relative_residual <= config.acceptance()
                && position_residual <= config.acceptance(),
        })
    }

    fn position_columns(&self) -> Vec<Vec<f64>> {
        if self.state.position_constraints.is_none() {
            return Vec::new();
        }
        let zero = [
            Matrix::zeros(self.electronic_basis.nao_sph, self.electronic_basis.nao_sph),
            Matrix::zeros(self.electronic_basis.nao_sph, self.electronic_basis.nao_sph),
        ];
        let mut columns = Vec::new();
        for p in 0..self.particles.len() {
            let moments = dipole_matrices(&self.particles[p].basis, Vec3::zero());
            for moment in moments {
                let mut q: Vec<_> = self
                    .particles
                    .iter()
                    .map(|p| Matrix::zeros(p.basis.nao_sph, p.basis.nao_sph))
                    .collect();
                q[p] = moment;
                columns.push(self.project(&zero, &q));
            }
        }
        columns
    }

    fn validate_sources(
        &self,
        electronic_source: &[Matrix; 2],
        quantum_source: &[Matrix],
    ) -> Result<f64> {
        if quantum_source.len() != self.particles.len()
            || electronic_source
                .iter()
                .any(|m| m.rows != self.electronic_basis.nao_sph || m.cols != m.rows)
            || quantum_source
                .iter()
                .zip(self.particles)
                .any(|(m, p)| m.rows != p.basis.nao_sph || m.cols != m.rows)
            || electronic_source.iter().chain(quantum_source).any(|m| {
                m.as_slice().iter().any(|v| !v.is_finite())
                    || m.max_abs_diff(&m.transpose()) > 1e-12
            })
        {
            return Err(DftError::InvalidInput(
                "coupled first-order sources require matching finite symmetric AO matrices for every species"
                    .into(),
            ));
        }
        Ok(self.state.quantum.iter().fold(
            self.state
                .electronic_orbital_residual
                .max(self.state.electronic_density_residual),
            |m, q| m.max(q.orbital_residual).max(q.density_residual),
        ))
    }

    /// Fixed-multiplier orbital response before optional position augmentation.
    pub(super) fn solve_unconstrained(
        &self,
        electronic_source: &[Matrix; 2],
        quantum_source: &[Matrix],
        config: &ResponseSolverConfig,
    ) -> Result<CoupledDensityResponse> {
        let reference_residual = self.validate_sources(electronic_source, quantum_source)?;
        let rhs: Vec<_> = self
            .project(electronic_source, quantum_source)
            .into_iter()
            .map(|x| Complex::new(-x, 0.0))
            .collect();
        let solution = krylov::solve(self, &rhs, None, config)?;
        let amplitudes: Vec<_> = solution.amplitudes.iter().map(|z| z.re).collect();
        let (e, q) = self.densities(&amplitudes);
        Ok(CoupledDensityResponse {
            electronic_density: e,
            quantum_density: q,
            amplitudes,
            relative_residual: solution.relative_residual,
            reference_residual,
            iterations: solution.iterations,
            converged: reference_residual <= config.acceptance()
                && solution.relative_residual <= config.acceptance(),
            position_multipliers: Vec::new(),
            position_residual: 0.0,
        })
    }

    /// Static total-dipole polarizability, including quantum-charge motion.
    pub fn polarizability(&self, config: &ResponseSolverConfig) -> Result<CoupledPolarizability> {
        let e = dipole_matrices(self.electronic_basis, Vec3::zero());
        let q: Vec<_> = self
            .particles
            .iter()
            .map(|p| dipole_matrices(&p.basis, Vec3::zero()))
            .collect();
        let mut responses = Vec::new();
        let mut tensor = [[0.0; 3]; 3];
        for b in 0..3 {
            let source: [Matrix; 2] = std::array::from_fn(|_| e[b].clone());
            let qs: Vec<_> = q
                .iter()
                .zip(self.particles)
                .map(|(d, p)| {
                    let mut m = d[b].clone();
                    for x in m.as_mut_slice() {
                        *x *= -p.charge;
                    }
                    m
                })
                .collect();
            let response = self.solve(&source, &qs, config)?;
            for a in 0..3 {
                tensor[a][b] = -response
                    .electronic_density
                    .iter()
                    .map(|p| p.frobenius_dot(&e[a]))
                    .sum::<f64>();
                for i in 0..q.len() {
                    tensor[a][b] += self.particles[i].charge
                        * response.quantum_density[i].frobenius_dot(&q[i][a]);
                }
            }
            responses.push(response);
        }
        let raw_asymmetry = (0..3)
            .flat_map(|a| (0..3).map(move |b| (tensor[a][b] - tensor[b][a]).abs()))
            .fold(0.0_f64, f64::max);
        Ok(CoupledPolarizability {
            tensor,
            raw_asymmetry,
            responses,
        })
    }

    /// Signed-frequency total dipole response at the joint stationary state.
    /// The time-even kernel couples the local densities of all species. The
    /// time-odd kernel keeps the electronic exchange of antisymmetric
    /// densities; the distinguishable particles have no exchange kernel.
    pub fn dynamic_polarizability(
        &self,
        frequencies: &[Frequency],
        config: &ResponseSolverConfig,
    ) -> Result<Vec<CoupledDynamicPolarizability>> {
        config.validate()?;
        if frequencies.is_empty() {
            return Err(DftError::InvalidInput(
                "coupled dynamic polarizability requires at least one frequency".into(),
            ));
        }
        let dipoles = dipole_matrices(self.electronic_basis, Vec3::zero());
        let quantum: Vec<_> = self
            .particles
            .iter()
            .map(|p| {
                let mut moments = dipole_matrices(&p.basis, Vec3::zero());
                for matrix in &mut moments {
                    for x in matrix.as_mut_slice() {
                        *x *= -p.charge;
                    }
                }
                moments
            })
            .collect();
        let mut reference_residual = self
            .state
            .position_constraints
            .as_ref()
            .map_or(0.0, |c| c.residual);
        let mut sources = Vec::with_capacity(3);
        for (axis, dipole) in dipoles.iter().enumerate() {
            let e = [dipole.clone(), dipole.clone()];
            let q: Vec<_> = quantum.iter().map(|d| d[axis].clone()).collect();
            reference_residual = reference_residual.max(self.validate_sources(&e, &q)?);
            sources.push(self.project(&e, &q));
        }
        if !reference_residual.is_finite() || reference_residual > config.acceptance() {
            return Err(DftError::InvalidInput(format!(
                "coupled dynamic polarizability reference residual {reference_residual:.6e} exceeds acceptance"
            )));
        }
        let n = self.dimension();
        let columns = self.position_columns();
        let gaps = self.orbital_gaps();
        frequencies
            .iter()
            .map(|frequency| {
                let operator = DynamicOperator {
                    orbital: self,
                    columns: &columns,
                    frequency: frequency.energy()?,
                    gaps: &gaps,
                };
                let mut tensor = [[Complex::ZERO; 3]; 3];
                let mut residuals = [0.0; 3];
                let mut position_residuals = [0.0_f64; 3];
                let mut iterations = [0; 3];
                for (b, source) in sources.iter().enumerate() {
                    let mut rhs = vec![Complex::ZERO; operator.dimension()];
                    for (target, value) in rhs.iter_mut().zip(source) {
                        *target = Complex::from(-*value);
                    }
                    let solution = krylov::solve(&operator, &rhs, None, config)?;
                    let even = &solution.amplitudes[..n];
                    for (a, probe) in sources.iter().enumerate() {
                        // Both off-diagonal halves enter the density derivative.
                        tensor[a][b] = real_projection(probe, even) * -2.0;
                    }
                    for row in &columns {
                        position_residuals[b] =
                            position_residuals[b].max((real_projection(row, even) * 2.0).abs());
                    }
                    if position_residuals[b] > config.acceptance() {
                        return Err(DftError::ResponseNotConverged {
                            iterations: solution.iterations,
                            residual: position_residuals[b],
                        });
                    }
                    residuals[b] = solution.relative_residual;
                    iterations[b] = solution.iterations;
                }
                Ok(CoupledDynamicPolarizability {
                    tensor,
                    residuals,
                    position_residuals,
                    iterations,
                    reference_residual,
                })
            })
            .collect()
    }

    fn time_odd_kernel(&self, amplitudes: &[f64], gaps: &[f64]) -> Result<Vec<f64>> {
        let delta = self.density_blocks(amplitudes, self.electronic_blocks, -1.0);
        let fock = self
            .parent
            .imaginary_fock_response(&self.electronic_densities(&delta))?;
        let projected: Vec<_> = self.frames[..self.electronic_blocks]
            .iter()
            .zip(&fock)
            .map(|(frame, potential)| {
                frame
                    .coefficients
                    .transpose_matmul_seq(potential)
                    .matmul_seq(&frame.coefficients)
            })
            .collect();
        Ok(self
            .slots
            .iter()
            .zip(amplitudes)
            .zip(gaps)
            .map(|((&(s, a, i), x), gap)| {
                let orbital_action = gap * x;
                if let Some(potential) = projected.get(s) {
                    orbital_action
                        + potential[(a, i)]
                            * if self.electronic_blocks == 1 {
                                2.0_f64.sqrt()
                            } else {
                                1.0
                            }
                } else {
                    orbital_action
                }
            })
            .collect())
    }
}

fn real_projection(probe: &[f64], amplitudes: &[Complex]) -> Complex {
    probe
        .iter()
        .zip(amplitudes)
        .fold(Complex::ZERO, |v, (a, b)| v + *b * *a)
}

struct DynamicOperator<'a, 'b> {
    orbital: &'a CoupledResponseSolver<'b>,
    columns: &'a [Vec<f64>],
    frequency: Complex,
    gaps: &'a [f64],
}

impl LinearOperator for DynamicOperator<'_, '_> {
    fn dimension(&self) -> usize {
        2 * self.orbital.dimension() + self.columns.len()
    }

    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }

    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        // P=(X+Y)/2 and Q=(X-Y)/2 obey
        // (A+B)P-zQ+C*l=-V, (A-B)Q-zP=0, C^T P=0.
        let n = self.orbital.dimension();
        let (even, rest) = x.split_at(n);
        let (odd, lambda) = rest.split_at(n);
        let mut image = self.orbital.apply(even)?;
        let image_odd = krylov::apply_real_linear(odd, |values| {
            self.orbital.time_odd_kernel(values, self.gaps)
        })?;
        for (a, b) in image.iter_mut().zip(odd) {
            *a -= self.frequency * *b;
        }
        for (row, multiplier) in self.columns.iter().zip(lambda) {
            for (a, &b) in image.iter_mut().zip(row) {
                *a += *multiplier * b;
            }
        }
        image.extend(
            image_odd
                .into_iter()
                .zip(even)
                .map(|(a, &b)| a - self.frequency * b),
        );
        image.extend(self.columns.iter().map(|row| real_projection(row, even)));
        Ok(image)
    }
}

struct PositionOperator<'a, 'b> {
    orbital: &'a CoupledResponseSolver<'b>,
    columns: &'a [Vec<f64>],
}
impl LinearOperator for PositionOperator<'_, '_> {
    fn dimension(&self) -> usize {
        self.orbital.dimension() + self.columns.len()
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::Hermitian
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        let n = self.orbital.dimension();
        let mut image = self.orbital.apply(&x[..n])?;
        for (r, lambda) in self.columns.iter().zip(&x[n..]) {
            for i in 0..n {
                image[i] += *lambda * r[i];
            }
        }
        image.extend(self.columns.iter().map(|r| {
            r.iter()
                .zip(&x[..n])
                .fold(Complex::ZERO, |v, (a, b)| v + *b * *a)
        }));
        Ok(image)
    }
}

impl LinearOperator for CoupledResponseSolver<'_> {
    fn dimension(&self) -> usize {
        self.slots.len()
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::Hermitian
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| self.apply_real(v))
    }
}
