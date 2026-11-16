// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Immutable molecular conditions and checked SCF capabilities. This facade
//! reuses the common collinear method, gradient and Hessian implementations.
//! Existing low-level research interfaces remain available; their raw results
//! cannot be imported as a checked solution here.
//!
//! ```compile_fail
//! use dft_rs::driver::typed::{Iterated, ScfSolution};
//! fn gradient_of_trial(trial: &ScfSolution<Iterated>) {
//!     let _ = trial.nuclear_gradient();
//! }
//! ```
//! ```no_run
//! use dft_rs::driver::typed::{Converged, ScfSolution};
//! fn gradient_of_solution(solution: &ScfSolution<Converged>) {
//!     let forces = solution.nuclear_gradient().unwrap().into_force();
//!     let _ = forces.into_ev_per_angstrom().unwrap();
//! }
//! ```

use super::method::{
    iterate_molecular_reference, prepare_molecular, MolecularMethod, MolecularReference,
    MolecularSetup,
};
use crate::indices::{self, AoId, AtomId, IndexSpace, ShellId, SpinComponent};
use crate::quantities::{
    Bohr, CartesianHessian, Coordinates, Energy, EnergyGradient, Hartree, PerMolecule,
};
use crate::response::diagnostics::ReferenceDiagnostics;
use crate::{DftError, Matrix, Result};
use std::marker::PhantomData;
use std::sync::Arc;

/// A complete fixed molecular problem. No mutable setup, basis, functional,
/// charge, field or convergence options escape this boundary.
#[derive(Clone)]
pub struct MolecularProblem {
    data: Arc<ProblemData>,
}

struct ProblemData {
    setup: MolecularSetup,
    atoms: IndexSpace<indices::Atom>,
    aos: IndexSpace<indices::AtomicOrbital>,
    shells: IndexSpace<indices::Shell>,
}

impl MolecularProblem {
    pub fn new(
        numbers: &[u8],
        positions: Coordinates<Bohr>,
        method: &MolecularMethod,
    ) -> Result<Self> {
        let setup = prepare_molecular(numbers, positions.as_slice(), method)?;
        Ok(Self {
            data: Arc::new(ProblemData {
                atoms: IndexSpace::new(setup.molecule.len())?,
                aos: IndexSpace::new(setup.basis.nao_sph)?,
                shells: IndexSpace::new(setup.basis.shells.len())?,
                setup,
            }),
        })
    }

    /// Changing positions constructs a new owner/generation. Previously
    /// converged states continue to describe their original immutable problem.
    pub fn with_coordinates(&self, positions: Coordinates<Bohr>) -> Result<Self> {
        let numbers: Vec<_> = self.data.setup.molecule.atoms.iter().map(|a| a.z).collect();
        Self::new(&numbers, positions, &self.data.setup.method)
    }
    pub fn method(&self) -> &MolecularMethod {
        &self.data.setup.method
    }
    pub fn atom_ids(&self) -> &IndexSpace<indices::Atom> {
        &self.data.atoms
    }
    pub fn ao_ids(&self) -> &IndexSpace<indices::AtomicOrbital> {
        &self.data.aos
    }
    pub fn shell_ids(&self) -> &IndexSpace<indices::Shell> {
        &self.data.shells
    }
    pub fn position(&self, atom: AtomId) -> Result<crate::quantities::Position<Bohr>> {
        crate::quantities::Position::new(
            self.data.setup.molecule.atoms[self.data.atoms.resolve(atom)?]
                .position
                .to_array(),
        )
    }
    pub fn shell(&self, shell: ShellId) -> Result<&crate::basis::MolShell> {
        Ok(&self.data.setup.basis.shells[self.data.shells.resolve(shell)?])
    }

    /// Execute the ordinary SCF iteration. A final trial energy remains
    /// observable even when its stopping tests fail. This does not authorize
    /// a stationary-state derivative until into_converged succeeds.
    pub fn iterate(&self) -> Result<ScfSolution<Iterated>> {
        let reference = iterate_molecular_reference(&self.data.setup)?;
        let diagnostics = crate::response::diagnostics::density_diagnostics(&reference.scf)?;
        Ok(ScfSolution {
            problem: self.clone(),
            reference,
            diagnostics,
            state: PhantomData,
        })
    }
    pub fn run(&self) -> Result<ScfSolution<Converged>> {
        self.iterate()?.into_converged()
    }
}

/// An SCF calculation has run, but stationary-state use is not yet certified.
pub struct Iterated;
/// The solver's stopping tests and a fresh finite orbital residual passed
/// for the captured problem and tolerance. This does not assert stability,
/// the global ground state, grid convergence or any response accuracy.
pub struct Converged;

pub struct ScfSolution<S> {
    problem: MolecularProblem,
    reference: MolecularReference,
    diagnostics: ReferenceDiagnostics,
    state: PhantomData<S>,
}
impl<S> ScfSolution<S> {
    pub fn problem(&self) -> &MolecularProblem {
        &self.problem
    }
    pub fn diagnostics(&self) -> &ReferenceDiagnostics {
        &self.diagnostics
    }
    pub fn trial_energy(&self) -> Result<Energy<Hartree, PerMolecule>> {
        Energy::new(self.reference.total_energy())
    }
    pub fn density_element(&self, spin: SpinComponent, row: AoId, column: AoId) -> Result<f64> {
        let row = self.problem.data.aos.resolve(row)?;
        let column = self.problem.data.aos.resolve(column)?;
        let spin = match spin {
            SpinComponent::Alpha => 0,
            SpinComponent::Beta => 1,
        };
        Ok(self.reference.scf.density[spin][(row, column)])
    }
    /// Export an initial guess, never a certificate for another Hamiltonian.
    /// The existing seeded SCF checks dimensions, new-overlap idempotency and
    /// particle number. Reprojection to a changed AO metric is not implicit.
    pub fn initial_guess(&self) -> InitialGuess {
        InitialGuess {
            density: self.reference.scf.density.clone(),
        }
    }
}
impl ScfSolution<Iterated> {
    pub fn into_converged(self) -> Result<ScfSolution<Converged>> {
        let scf = &self.reference.scf;
        let residual = self.diagnostics.orbital_residual;
        let tolerance = self.problem.data.setup.options.conv_tol_grad;
        if !scf.converged
            || !self.reference.total_energy().is_finite()
            || !residual.is_finite()
            || residual > tolerance
        {
            return Err(DftError::ScfNotConverged {
                iterations: scf.iterations,
                error: residual,
            });
        }
        Ok(ScfSolution {
            problem: self.problem,
            reference: self.reference,
            diagnostics: self.diagnostics,
            state: PhantomData,
        })
    }
}
impl ScfSolution<Converged> {
    pub fn energy(&self) -> Energy<Hartree, PerMolecule> {
        // The private transition checked finiteness, and the snapshot cannot mutate.
        Energy::new(self.reference.total_energy()).expect("checked finite molecular energy")
    }
    pub fn nuclear_gradient(&self) -> Result<EnergyGradient> {
        let values = super::molecular::total_gradient(&self.problem.data.setup, &self.reference)?;
        EnergyGradient::new(values.into_iter().map(|v| v.to_array()).collect())
    }
    pub fn cartesian_hessian(&self) -> Result<CartesianHessian> {
        let values =
            super::vibrational::reference_hessian(&self.problem.data.setup, &self.reference)?;
        CartesianHessian::new(values.hessian)
    }
}

/// An explicitly demoted density, usable with the existing seeded SCF API.
/// This type has no energy, gradient, Hessian or convergence capability.
#[derive(Clone)]
pub struct InitialGuess {
    density: [Matrix; 2],
}
impl InitialGuess {
    pub fn as_density(&self) -> &[Matrix; 2] {
        &self.density
    }
}
