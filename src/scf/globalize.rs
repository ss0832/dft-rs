// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent-spin adapter of the shared occupied-subspace minimizer.
//! The physical density model owns every energy term and Fock evaluation.

use super::fields::MolecularDensityModel;
use crate::linalg::{symmetric_eigen, Matrix, Orthogonalizer};
use crate::periodic::spinor_scf::minimize::OrbitalMinimizer;
use crate::periodic::BlochMatrix;
use crate::Result;

fn real_blocks(values: &[Matrix; 2], spins: usize) -> Vec<BlochMatrix> {
    values[..spins]
        .iter()
        .map(|m| BlochMatrix {
            re: m.clone(),
            im: Matrix::zeros(m.rows, m.cols),
        })
        .collect()
}

pub(super) struct CollinearGlobalizer {
    optimizer: OrbitalMinimizer,
    counts: [usize; 2],
    spins: usize,
    progress: super::progress::ResidualProgress,
}

pub(super) struct CollinearStep {
    pub density: [Matrix; 2],
    pub coefficients: [Matrix; 2],
    pub density_rms_change: f64,
}

impl CollinearGlobalizer {
    pub fn new(coefficients: &[Matrix; 2], counts: [usize; 2], spins: usize) -> Self {
        let populations: Vec<_> = (0..spins)
            .map(|s| {
                (0..coefficients[s].cols)
                    .map(|i| if i < counts[s] { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        Self {
            optimizer: OrbitalMinimizer::new(real_blocks(coefficients, spins), &populations),
            counts,
            spins,
            progress: super::progress::ResidualProgress::default(),
        }
    }

    pub(super) fn needs_local_recovery(&mut self, residual: f64, window: usize) -> bool {
        self.progress.has_stalled(residual, window)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        model: &MolecularDensityModel<'_>,
        density: &[Matrix; 2],
        fock: &[Matrix; 2],
        energy: f64,
        overlap: &Matrix,
        orth: &Orthogonalizer,
    ) -> Result<Option<CollinearStep>> {
        let blocks = real_blocks(density, self.spins);
        let Some(step) = self.optimizer.step(
            &blocks,
            &real_blocks(fock, self.spins),
            energy,
            if self.spins == 1 { 2.0 } else { 1.0 },
            false,
            |p| {
                let real = [p[0].re.clone(), p[self.spins - 1].re.clone()];
                let fields = model.evaluate(&real)?;
                Ok((
                    fields.electronic_energy,
                    real_blocks(&fields.fock, self.spins),
                ))
            },
        )?
        else {
            return Ok(None);
        };
        let next = [
            step.density[0].re.clone(),
            step.density[self.spins - 1].re.clone(),
        ];
        let sx = overlap.matmul_seq(&orth.x);
        let mut coefficients = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        let mut change = 0.0;
        for spin in 0..self.spins {
            // A natural frame of the new projector keeps complete occupied
            // and virtual spaces, without diagonalizing a stale Fock.
            let projector = sx.transpose_matmul_seq(&next[spin]).matmul_seq(&sx);
            let (_, natural) = symmetric_eigen(&projector)?;
            let n = natural.cols;
            let descending = Matrix::from_row_major(
                n,
                n,
                (0..n)
                    .flat_map(|i| {
                        let natural = &natural;
                        (0..n).map(move |j| natural[(i, n - 1 - j)])
                    })
                    .collect(),
            );
            coefficients[spin] = orth.x.matmul_seq(&descending);
            let check = coefficients[spin].leading_columns_gram(self.counts[spin], 1.0);
            if check.max_abs_diff(&next[spin]) > 1e-9 {
                return Err(crate::DftError::InvalidInput(
                    "orbital globalization lost the integer occupied projector".into(),
                ));
            }
            change += next[spin]
                .as_slice()
                .iter()
                .zip(density[spin].as_slice())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / next[spin].as_slice().len() as f64;
        }
        if self.spins == 1 {
            coefficients[1] = coefficients[0].clone();
        }
        crate::logger::log(&format!(
            "[scf] orbital quasi-Newton fraction={:.6e} secants={} energy_work={:.6e}",
            step.fraction, step.history_size, step.work
        ));
        Ok(Some(CollinearStep {
            density: next,
            coefficients,
            density_rms_change: (change / self.spins as f64).sqrt(),
        }))
    }
}

/// One accepted physical trust-region step, retaining the finite energy work.
pub(super) struct TrustStep {
    pub density: [Matrix; 2],
    pub coefficients: [Matrix; 2],
    pub energies: [Vec<f64>; 2],
    pub density_rms_change: f64,
    pub energy_work: f64,
}

/// Current determinant data for the shared response kernel. This is internal
/// trial state, never a public converged result or a response acceptance claim.
#[allow(clippy::too_many_arguments)]
pub(super) fn trial_reference(
    density: &[Matrix; 2],
    coefficients: &[Matrix; 2],
    energies: &[Vec<f64>; 2],
    fock: &[Matrix; 2],
    overlap: &Matrix,
    core: &Matrix,
    counts: [usize; 2],
    spins: usize,
) -> super::ScfResult {
    super::ScfResult {
        occupation_tracking: None,
        fixed_occupation_residual: None,
        common_orbital_residual: None,
        energy: 0.0,
        e_nuc: 0.0,
        e_elec: 0.0,
        mo_energy: energies.clone(),
        mo_coeff: coefficients.clone(),
        density: density.clone(),
        fock: fock.clone(),
        overlap: overlap.clone(),
        h_core: core.clone(),
        n_alpha: counts[0],
        n_beta: counts[1],
        unrestricted: spins == 2,
        s_squared: 0.0,
        e_xc: 0.0,
        e_nlc: 0.0,
        e_pcm: 0.0,
        grid_electron_count: None,
        converged: false,
        iterations: 0,
        occupations: std::array::from_fn(|s| {
            (0..coefficients[s].cols)
                .map(|i| if i < counts[s] { 1.0 } else { 0.0 })
                .collect()
        }),
        electronic_temperature: 0.0,
        electronic_entropy: 0.0,
        chemical_potentials: [None; 2],
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn trust_step(
    model: &MolecularDensityModel<'_>,
    reference: &super::ScfResult,
    energy: f64,
    quadratic: bool,
    trust: &mut super::trah::TrustRegion,
) -> Result<Option<TrustStep>> {
    let spins = if reference.unrestricted { 2 } else { 1 };
    let oh = super::soscf::OrbitalHessian {
        c: &reference.mo_coeff,
        eps: &reference.mo_energy,
        n_occ: [reference.n_alpha, reference.n_beta],
        spins,
    };
    if oh.is_empty() {
        return Ok(None);
    }
    let kernel = crate::response::ResponseSolver::kernel_with_engines(
        model.mol,
        model.mb,
        model.functional,
        model.grid,
        reference,
        model.opts,
        model.jk,
        model.jk_lr,
    )?;
    let fock_response = |delta: &[Matrix; 2]| -> Result<[Matrix; 2]> {
        let mut response = kernel.fock_response(delta)?;
        if let Some(extension) = model.extension {
            extension.add_response(delta, &mut response)?;
        }
        Ok(response)
    };
    let gradient = oh.gradient(&reference.fock);
    let hessian = |x: &[f64]| -> Result<Vec<f64>> {
        let delta = oh.delta_density(x);
        let mut image = oh.project(&fock_response(&delta)?);
        for (a, b) in image.iter_mut().zip(oh.fock_commutator(x, &reference.fock)) {
            *a += b;
        }
        Ok(image)
    };
    for rejection in 0..12 {
        let step = super::trah::direction(
            &oh.diagonal(),
            &gradient,
            hessian,
            trust.radius,
            model.opts.soscf_max_cg.max(2),
            1e-3,
        )?;
        if step.predicted_work >= 0.0 {
            return Ok(None);
        }
        let mut coefficients = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        super::soscf::rotate(&oh, &step.rotation, &mut coefficients);
        let counts = [reference.n_alpha, reference.n_beta];
        let density: [Matrix; 2] =
            std::array::from_fn(|s| coefficients[s].leading_columns_gram(counts[s], 1.0));
        let fields = model.evaluate(&density)?;
        let work = if quadratic {
            let delta = oh.finite_density_change(&step.rotation);
            let induced = fock_response(&delta)?;
            oh.quadratic_energy_work(&step.rotation, &reference.fock, &delta, &induced)
        } else {
            let delta = oh.finite_density_change(&step.rotation);
            let zero = std::array::from_fn(|s| Matrix::zeros(delta[s].rows, delta[s].cols));
            let Some(value) = super::trah::energy_work(energy, fields.electronic_energy, |t| {
                let mut intermediate = reference.density.clone();
                for s in 0..2 {
                    for (p, d) in intermediate[s]
                        .as_mut_slice()
                        .iter_mut()
                        .zip(delta[s].as_slice())
                    {
                        *p += t * d;
                    }
                }
                let f = model.evaluate(&intermediate)?.fock;
                Ok(oh.quadratic_energy_work(&step.rotation, &f, &delta, &zero))
            })?
            else {
                return Ok(None);
            };
            value
        };
        let length = step.rotation.iter().map(|x| x * x).sum::<f64>().sqrt();
        let radius = trust.radius;
        let (accepted, ratio) = trust.assess(step.predicted_work, work, length);
        crate::logger::log(&format!(
            "[scf] TRAH accepted={accepted} rejection={rejection} radius={radius:.12e} step={length:.12e} predicted={:.12e} actual={work:.12e} ratio={ratio:.12e} shift={:.12e} inner_residual={:.12e} products={}",
            step.predicted_work,step.shift,step.residual,step.products));
        if !accepted && length < 0.5 * trust.radius {
            crate::logger::log(
                "[scf] TRAH rejected an interior step; resume the physical Fock iteration",
            );
            return Ok(None);
        }
        if accepted {
            let change = (0..spins)
                .map(|s| {
                    density[s]
                        .as_slice()
                        .iter()
                        .zip(reference.density[s].as_slice())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / density[s].as_slice().len() as f64
                })
                .sum::<f64>();
            let energies = std::array::from_fn(|s| {
                let f = coefficients[s]
                    .transpose_matmul_seq(&fields.fock[s])
                    .matmul_seq(&coefficients[s]);
                (0..f.rows).map(|i| f[(i, i)]).collect()
            });
            return Ok(Some(TrustStep {
                density,
                coefficients,
                energies,
                density_rms_change: (change / spins as f64).sqrt(),
                energy_work: work,
            }));
        }
    }
    Ok(None)
}

#[cfg(test)]
#[path = "trah_tests.rs"]
mod tests;
