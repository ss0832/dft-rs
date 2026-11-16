// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic post-SCF correlation with the parent's full orbital response.
//! The Coulomb tensor is the periodic Ewald tensor, including its explicit
//! background convention. Moving nuclei, strain, Berry electric fields and
//! periodic scalar probes share the same density and pair equations.

use super::lattice::{
    density_series, primitive_atoms, reference, series_reference, Reference, Setup,
};
use super::{matrix_product, LatticePath, NuclearEnergyCoefficient, SeriesReference};
use crate::linalg::Matrix;
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::Cell;
use crate::properties::pt2::Pt2Problem;
use crate::properties::pt2_response::{projector_frames, taylor_response, transform_path};
use crate::response::ResponseSolverConfig;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Factorial-normalized coefficients per primitive cell. Spin-channel
/// energies are unscaled; only `correlation` includes the recipe weights.
#[derive(Clone, Debug)]
pub struct LatticeCorrelatedTaylor {
    pub energy: Vec<f64>,
    pub parent: Vec<f64>,
    pub target: Vec<f64>,
    /// Included in the respective density-functional energy series.
    pub parent_nonlocal: Vec<f64>,
    pub target_nonlocal: Vec<f64>,
    pub same_spin_alpha: Vec<f64>,
    pub opposite_spin: Vec<f64>,
    pub same_spin_beta: Vec<f64>,
    pub correlation: Vec<f64>,
    pub response_residuals: Vec<f64>,
    pub amplitude_residuals: Vec<f64>,
    pub projector_residuals: Vec<f64>,
    pub stationarity_residuals: Vec<f64>,
    pub metric_residuals: Vec<f64>,
    pub number_residuals: Vec<[f64; 2]>,
    pub orbital_frame_residual: f64,
    pub reference: SeriesReference,
}

/// A finite Born-von Karman cell sampled at Gamma. A primitive-cell k mesh
/// is represented by its real supercell, including commensurate finite-q
/// displacement/probe patterns. Converge this cell for the infinite solid.
/// No molecular integral or parent-only response is substituted for PT2.
pub struct LatticeCorrelatedSolver<'a> {
    setup: Setup<'a>,
    reference: Reference,
    target: Option<&'a ResolvedFunctional>,
    target_nonlocal: Option<(f64, crate::periodic::nonlocal::PeriodicVv10Options)>,
    options: &'a PeriodicScfOptions,
    replicas: usize,
    field: [f64; 3],
    coefficients: Pt2Coefficients,
    memory_mb: usize,
    diagnostics: SeriesReference,
}

impl<'a> LatticeCorrelatedSolver<'a> {
    /// Canonical orbitals of the actual series functional. Independent
    /// integral oracles must use these orbitals rather than those of the
    /// separately truncated SCF initializer; their energy difference is
    /// reported by `SeriesReference`.
    pub fn reference_orbitals(&self) -> (&[Matrix; 2], &[Vec<f64>; 2], [usize; 2]) {
        (&self.reference.c, &self.reference.e, self.reference.n_occ)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cell: &'a Cell,
        parent: &'a ResolvedFunctional,
        target: Option<&'a ResolvedFunctional>,
        result: &PeriodicScfResult,
        options: &'a PeriodicScfOptions,
        replicas: usize,
        field: [f64; 3],
        coefficients: Pt2Coefficients,
        memory_mb: usize,
    ) -> Result<Self> {
        primitive_atoms(cell, replicas)?;
        coefficients.validate()?;
        if options.boundary != crate::periodic::features::PeriodicBoundary::FULL_3D {
            return Err(DftError::Unsupported {
                feature: "correlated lattice paths with open axes".into(),
                missing: "dimensional Coulomb and integral paths for the pair response".into(),
            });
        }
        let mut setup = Setup::new(cell, parent, result, options)?;
        setup.set_static_field(field)?;
        // Check resources before constructing even the zeroth-order tensor.
        check_memory(cell.basis().nao_sph, 0, target.is_some(), memory_mb)?;
        let reference = reference(&setup, result, options)?;
        let diagnostics = series_reference(&reference, result, replicas);
        Ok(Self {
            setup,
            reference,
            target,
            target_nonlocal: None,
            options,
            replicas,
            field,
            coefficients,
            memory_mb,
            diagnostics,
        })
    }

    /// Explicit nonlocal part of a distinct target evaluated on the fully
    /// responding parent density. Defaults to the parent's unscaled provider.
    pub fn with_target_nonlocal(
        mut self,
        coefficient: f64,
        settings: crate::periodic::nonlocal::PeriodicVv10Options,
    ) -> Result<Self> {
        settings.validate()?;
        if !coefficient.is_finite() {
            return Err(DftError::InvalidInput(
                "the target nonlocal coefficient must be finite".into(),
            ));
        }
        self.target_nonlocal = Some((coefficient, settings));
        Ok(self)
    }

    /// Actual response through `order`, currently zero through four.
    /// Every parent density coefficient is solved: the parent's reduced
    /// 2n+1 trial density cannot be used for a nonstationary PT2 energy.
    pub fn solve(
        &self,
        line: &LatticePath,
        order: usize,
        config: &ResponseSolverConfig,
    ) -> Result<LatticeCorrelatedTaylor> {
        config.validate()?;
        if order > 4 {
            return Err(DftError::InvalidInput(
                "periodic correlated Taylor order must be between zero and four".into(),
            ));
        }
        let cell = self.setup.cell;
        let nao = cell.basis().nao_sph;
        check_memory(
            nao,
            order,
            self.target.is_some() || self.target_nonlocal.is_some(),
            self.memory_mb,
        )?;
        // Reuse spatial integrals for a pure electric path; only the Berry
        // coupling carries its field series. Moving paths differentiate all
        // Ewald integrals, overlap, quadrature and Pulay contributions.
        let geometry = if line.displacement.is_empty()
            && line.deformation.is_empty()
            && line.potential.is_empty()
        {
            crate::periodic::path_series::validate_charged_path(cell, line)?;
            let mut geometry = self
                .reference
                .geometry
                .clone()
                .extend_fixed_lattice_field(order)?;
            let reduced = self
                .setup
                .reduced_field(line, order)
                .expect("the correlated reference always has a Berry field setup");
            geometry.set_reduced_lattice_field(reduced)?;
            geometry
        } else {
            crate::logger::timed("Correlated lattice parent integral path", || {
                self.setup.geometry(line, order)
            })?
        };
        let path = crate::logger::timed("Correlated lattice density response", || {
            density_series(
                &self.setup,
                &self.reference,
                &geometry,
                order,
                order,
                config,
            )
        })?;
        let mut frame_residual = 0.0_f64;
        let frames: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
            let (u, error) = projector_frames(&path.projectors[s], self.reference.n_occ[s]);
            frame_residual = frame_residual.max(error);
            u
        });
        let orbitals = std::array::from_fn(|s| matrix_product(&path.basis[s], &frames[s], order));
        let g = geometry.coulomb_coefficients();
        let energies = &self.reference.e;
        let occupied = self.reference.n_occ;
        let transformed: Vec<Pt2Problem<f64>> = match order {
            0 => transform_path::<1>(nao, g, &orbitals, energies, occupied)?,
            1 => transform_path::<2>(nao, g, &orbitals, energies, occupied)?,
            2 => transform_path::<3>(nao, g, &orbitals, energies, occupied)?,
            3 => transform_path::<4>(nao, g, &orbitals, energies, occupied)?,
            4 => transform_path::<5>(nao, g, &orbitals, energies, occupied)?,
            _ => unreachable!(),
        };
        let orth_fock: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
            let bt: Vec<_> = path.basis[s].iter().map(Matrix::transpose).collect();
            matrix_product(
                &matrix_product(&bt, &path.evaluation.fock[s], order),
                &path.basis[s],
                order,
            )
        });
        let moving: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
            let ut: Vec<_> = frames[s].iter().map(Matrix::transpose).collect();
            matrix_product(
                &matrix_product(&ut, &orth_fock[s], order),
                &frames[s],
                order,
            )
        });
        let fock: Vec<_> = (1..=order)
            .map(|n| [moving[0][n].clone(), moving[1][n].clone()])
            .collect();
        let integrals: Vec<_> = transformed.iter().skip(1).map(|p| p.ovov.clone()).collect();
        let pair = crate::logger::timed("Correlated lattice pair response", || {
            taylor_response(&transformed[0], &integrals, &fock, self.coefficients)
        })?;
        let scale = 1.0 / self.replicas as f64;
        let parent: Vec<_> = path
            .evaluation
            .energy
            .iter()
            .map(|e| e.total() * scale)
            .collect();
        let parent_nonlocal: Vec<_> = path
            .evaluation
            .energy
            .iter()
            .map(|e| e.nonlocal_correlation * scale)
            .collect();
        let (target, target_nonlocal) = if self.target.is_some() || self.target_nonlocal.is_some() {
            let target = self.target.unwrap_or(self.setup.functional);
            let mut options = self.options.clone();
            if let Some((_, settings)) = self.target_nonlocal {
                options.vv10 = Some(settings);
            }
            let mut setup = Setup::for_cell(cell, target, &options)?;
            setup.set_static_field(self.field)?;
            let mut geometry =
                crate::logger::timed("Correlated lattice target integral path", || {
                    setup.geometry(line, order)
                })?;
            if let Some((coefficient, _)) = self.target_nonlocal {
                geometry.scale_periodic_nonlocal(coefficient)?;
            }
            let evaluation = crate::logger::timed("Correlated lattice target energy", || {
                geometry.evaluate(target, &path.density, order)
            })?;
            (
                evaluation
                    .energy
                    .iter()
                    .map(|e: &NuclearEnergyCoefficient| e.total() * scale)
                    .collect(),
                evaluation
                    .energy
                    .iter()
                    .map(|e| e.nonlocal_correlation * scale)
                    .collect(),
            )
        } else {
            (parent.clone(), parent_nonlocal.clone())
        };
        let scale_series =
            |values: Vec<f64>| values.into_iter().map(|v| v * scale).collect::<Vec<_>>();
        let correlation = scale_series(pair.correlation);
        let mut result = LatticeCorrelatedTaylor {
            energy: target
                .iter()
                .zip(&correlation)
                .map(|(e, c)| e + c)
                .collect(),
            parent,
            target,
            parent_nonlocal,
            target_nonlocal,
            same_spin_alpha: scale_series(pair.same_spin_alpha),
            opposite_spin: scale_series(pair.opposite_spin),
            same_spin_beta: scale_series(pair.same_spin_beta),
            correlation,
            response_residuals: path.residuals,
            amplitude_residuals: pair.residuals,
            projector_residuals: vec![0.0; order + 1],
            stationarity_residuals: vec![0.0; order + 1],
            metric_residuals: vec![0.0; order + 1],
            number_residuals: vec![[0.0; 2]; order + 1],
            orbital_frame_residual: frame_residual,
            reference: self.diagnostics.clone(),
        };
        for s in 0..2 {
            let d = &path.projectors[s];
            let square = matrix_product(d, d, order);
            let fd = matrix_product(&orth_fock[s], d, order);
            let df = matrix_product(d, &orth_fock[s], order);
            let ct: Vec<_> = orbitals[s].iter().map(Matrix::transpose).collect();
            let metric = matrix_product(
                &matrix_product(&ct, &geometry.overlap, order),
                &orbitals[s],
                order,
            );
            let ps = matrix_product(&path.density[s], &geometry.overlap, order);
            for n in 0..=order {
                result.number_residuals[n][s] = (0..ps[n].rows).map(|i| ps[n][(i, i)]).sum();
                for i in 0..d[n].rows {
                    for j in 0..d[n].cols {
                        let identity = if n == 0 && i == j { 1.0 } else { 0.0 };
                        result.projector_residuals[n] = result.projector_residuals[n]
                            .max((square[n][(i, j)] - d[n][(i, j)]).abs());
                        result.stationarity_residuals[n] = result.stationarity_residuals[n]
                            .max((fd[n][(i, j)] - df[n][(i, j)]).abs());
                        result.metric_residuals[n] =
                            result.metric_residuals[n].max((metric[n][(i, j)] - identity).abs());
                    }
                }
                if n == 0 {
                    result.number_residuals[n][s] -= occupied[s] as f64;
                }
            }
        }
        Ok(result)
    }
}

fn check_memory(nao: usize, order: usize, target: bool, limit_mb: usize) -> Result<()> {
    let bytes = nao
        .checked_pow(4)
        .and_then(|n| n.checked_mul(order + 1))
        .and_then(|n| n.checked_mul(8 * (16 + 3 * usize::from(target))))
        .ok_or_else(|| DftError::InvalidInput("periodic PT2 tensor memory size overflow".into()))?;
    let required_mb = bytes.div_ceil(1024 * 1024);
    if required_mb > limit_mb {
        return Err(DftError::ResourceLimit {
            operation: "periodic PT2 integral and orbital paths",
            required_mb,
            limit_mb,
        });
    }
    Ok(())
}
