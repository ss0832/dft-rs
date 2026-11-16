// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent state with a hole in one chosen orbital.
//!
//! One electron is removed from an occupied orbital of one spin of a
//! converged reference, and the unrestricted Kohn-Sham equations are solved
//! again. In every iteration the occupied orbitals are those with the
//! largest weight in the initial occupied space without the hole, so that
//! the hole does not fall back into the lowest orbital. The difference to
//! the reference energy is the ionization energy of that orbital with
//! relaxed orbitals.

use super::occupations::{run_scf_with_occupations, DeterminantOccupation, OverlapReference};
use super::{Reference, ScfOptions, ScfResult};
use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

/// Relaxed ion with the selected hole. Empty determinants have unit overlap
/// by convention; all nonempty projector weights remain unmodified.
#[derive(Clone, Debug)]
pub struct CoreHoleState {
    pub energy: f64,
    pub converged: bool,
    pub iterations: usize,
    pub smallest_overlap: f64,
    pub density: [Matrix; 2],
}

/// Remove one occupied spin orbital and relax with the common projector-IMOM
/// SCF driver. Functional, exchange, grid, solvent and convergence terms are
/// identical to the ordinary SCF path; the hole changes only the occupation.
pub fn run_core_hole(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    spin: usize,
    orbital: usize,
) -> Result<CoreHoleState> {
    let state = core_hole_reference(mol, mb, opts, functional, grid, reference, spin, orbital)?;
    let tracking = state
        .occupation_tracking
        .as_ref()
        .expect("tracked SCF supplies occupation diagnostics");
    let smallest_overlap = tracking
        .overlap_weights
        .iter()
        .flatten()
        .copied()
        .reduce(f64::min)
        .unwrap_or(1.0);
    Ok(CoreHoleState {
        energy: state.energy,
        converged: state.converged,
        iterations: state.iterations,
        smallest_overlap,
        density: state.density,
    })
}

/// Same relaxed hole reference, retaining the final occupied and empty
/// orbitals in occupation order for fixed-geometry final-state spectroscopy.
#[allow(clippy::too_many_arguments)]
pub fn core_hole_reference(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    spin: usize,
    orbital: usize,
) -> Result<ScfResult> {
    let counts = [reference.n_alpha, reference.n_beta];
    if !reference.converged
        || opts.electronic_temperature != 0.0
        || opts.fixed_occupations.is_some()
        || !reference.has_integer_occupations()
        || spin > 1
        || orbital >= counts[spin.min(1)]
    {
        return Err(DftError::InvalidInput("a core-hole state needs a converged integer reference and an occupied orbital of spin 0 or 1".into()));
    }
    let occupied: [Vec<usize>; 2] = std::array::from_fn(|s| {
        (0..counts[s])
            .filter(|&i| !(s == spin && i == orbital))
            .collect()
    });
    let mut ion = mol.clone();
    ion.charge = ion
        .charge
        .checked_add(1)
        .ok_or_else(|| DftError::InvalidInput("core-hole charge overflows".into()))?;
    ion.multiplicity = occupied[0].len().abs_diff(occupied[1].len()) + 1;
    let mut options = opts.clone();
    options.reference = Reference::Uhf;
    options.occupation = None;
    run_scf_with_occupations(
        &ion,
        mb,
        &options,
        functional,
        grid,
        reference,
        &DeterminantOccupation {
            occupied,
            reference: OverlapReference::Initial,
        },
    )
}
