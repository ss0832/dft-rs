// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Analytic moving-coordinate response of one frozen scalar action.
//!
//! Each full atom follows one raw center by translation, keeping its saved
//! internal offset. Frozen orbitals retain their neutral AO coefficient span
//! and are normalized in the current metric. No displaced target SCF supplies
//! a projector or an energy correction. Rotation/internal deformation beyond
//! this declared affine geometry chart is a separate model.

mod collinear;
mod hessian;
pub use collinear::{CollinearAffineFrozenGeometry, CollinearAffineFrozenGeometryResult};
pub use hessian::{AffineFrozenHessian, AffineFrozenSecondDerivative};

use super::reduction_basis::cross_basis_overlap;
use super::reduction_reference::{
    ActiveSpaceScf, FrozenDensityEvaluator, FrozenReferenceScf, MappedFrozenAction,
};
use super::reduction_support::{metric_power_direction, positive_power, OrthogonalReducedChart};
use crate::basis::MolBasis;
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::deriv_mat::overlap_derivative_matrices;
use crate::integrals::{kinetic_matrix, nuclear_matrix, overlap_matrix};
use crate::linalg::{AccurateSum, Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

fn invalid<T>(text: &str) -> Result<T> {
    Err(DftError::InvalidInput(text.into()))
}
fn add(a: &mut Matrix, b: &Matrix, factor: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += factor * y;
    }
}
fn moved_basis(basis: &MolBasis, molecule: &Molecule) -> Result<MolBasis> {
    let mut result = basis.clone();
    for shell in &mut result.shells {
        shell.center = molecule
            .atoms
            .get(shell.atom)
            .ok_or_else(|| DftError::InvalidInput("affine scalar AO atom is out of range".into()))?
            .position;
    }
    Ok(result)
}
fn block(matrix: &Matrix, row: usize, column: usize, n: usize, m: usize) -> Matrix {
    let mut out = Matrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            out[(i, j)] = matrix[(row + i, column + j)];
        }
    }
    out
}

/// Explicit one-owner affine chart. It is immutable construction data, not
/// a function of a converged product density. The raw molecule's charge is
/// the fixed active-sector charge, separate from full endpoint accounting.
pub struct AffineFrozenGeometry<'a> {
    pub full_reference: &'a Molecule,
    pub full_basis: &'a MolBasis,
    pub raw_reference: &'a Molecule,
    pub raw_basis: &'a MolBasis,
    pub frozen_reference: &'a Matrix,
    pub full_atom_owners: &'a [usize],
    pub functional: &'a ResolvedFunctional,
    pub options: &'a ScfOptions,
    pub grid_options: &'a GridOptions,
    pub active_electrons: usize,
    pub reference_energy: f64,
}

pub struct AffineFrozenGeometryResult {
    pub energy: f64,
    /// Every raw Cartesian derivative, in Hartree/Bohr.
    pub gradient: Vec<Vec3>,
    pub active_scf: FrozenReferenceScf,
    pub full_molecule: Molecule,
    pub chart_metric_error: f64,
    pub frozen_overlap: f64,
}

impl AffineFrozenGeometry<'_> {
    /// Optimize the declared affine raw coordinates with the shared analytic
    /// molecular optimizer. This does not unfreeze internal fragment geometry
    /// and does not compute a Hessian, zero-point correction or thermal heat.
    pub fn optimize(
        &self,
        options: &crate::optimizer::OptOptions,
    ) -> Result<crate::optimizer::OptResult<AffineFrozenGeometryResult>> {
        crate::optimizer::optimize_with_evaluator(self.raw_reference, options, |molecule| {
            let coordinates = molecule
                .atoms
                .iter()
                .map(|a| a.position)
                .collect::<Vec<_>>();
            let result = self.evaluate(&coordinates, None)?;
            Ok((result.energy, result.gradient.clone(), result))
        })
    }
    pub(super) fn collinear(&self) -> Result<CollinearAffineFrozenGeometry<'_>> {
        if self.active_electrons == 0 || self.active_electrons % 2 != 0 {
            return invalid(
                "restricted affine scalar requires a nonzero even active electron count",
            );
        }
        Ok(CollinearAffineFrozenGeometry {
            full_reference: self.full_reference,
            full_basis: self.full_basis,
            raw_reference: self.raw_reference,
            raw_basis: self.raw_basis,
            frozen_reference: [self.frozen_reference; 2],
            full_atom_owners: self.full_atom_owners,
            functional: self.functional,
            options: self.options,
            grid_options: self.grid_options,
            active_electrons: [self.active_electrons / 2; 2],
            reference_energy: self.reference_energy,
        })
    }
    pub fn evaluate(
        &self,
        coordinates: &[Vec3],
        initial_density: Option<&Matrix>,
    ) -> Result<AffineFrozenGeometryResult> {
        let half = initial_density.map(|p| {
            let mut half = p.clone();
            for x in half.as_mut_slice() {
                *x *= 0.5;
            }
            half
        });
        let value = self.collinear()?.evaluate_internal(
            coordinates,
            half.as_ref().map(|p| [p, p]),
            true,
        )?;
        let active = value.active_scf;
        let mut density = active.active_density[0].clone();
        add(&mut density, &active.active_density[1], 1.);
        let e = active.evaluation;
        let evaluation = super::reduction_reference::FrozenDensityEvaluation {
            energy: e.energy,
            electronic_energy: e.electronic_energy,
            one_electron_energy: e.one_electron_energy,
            hartree_energy: e.hartree_energy,
            xc_energy: e.xc_energy,
            nuclear_energy: e.nuclear_energy,
            reference_energy: e.reference_energy,
            fock: e.fock[0].clone(),
            active_electrons: e.active_electrons.iter().sum(),
            frozen_electrons: e.frozen_electrons.iter().sum(),
            grid_total_electrons: e.grid_total_electrons,
        };
        Ok(AffineFrozenGeometryResult {
            energy: value.energy,
            gradient: value.gradient,
            full_molecule: value.full_molecule,
            chart_metric_error: value.chart_metric_error[0],
            frozen_overlap: value.frozen_overlap[0],
            active_scf: FrozenReferenceScf {
                active_density: density,
                active_orbitals: active.active_orbitals[0].clone(),
                orbital_energies: active.orbital_energies[0].clone(),
                evaluation,
                history: active.history,
                projected_orbital_residual: active.projected_orbital_residual,
                active_idempotency_error: 4. * active.active_idempotency_error[0],
                active_frozen_overlap: active.active_frozen_overlap[0],
                particle_number_error: 2. * active.particle_number_error[0],
            },
        })
    }
}
