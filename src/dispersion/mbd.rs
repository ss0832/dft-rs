// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! The finite-molecule MBD@TS variant of coupled-oscillator dispersion.
//! Ambrosetti et al., J. Chem. Phys. 140, 18A508 (2014), Eq. (14),
//! Section V and Appendix. The oscillators use unscreened Hirshfeld/TS
//! parameters and couple through the long-range dipole tensor.

use super::ts::{atom_environments, volume_ratios, AtomEnvironment, FreeAtom};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::system::Molecule;
use crate::{DftError, Result};

mod screening;
pub use screening::{rsscs_energy_from_volumes, rsscs_energy_on_density, MbdRsScsEnergy};

/// Molecular MBD@TS contribution and the uncoupled/coupled spectra.
#[derive(Clone, Debug)]
pub struct MbdEnergy {
    /// Coupled minus uncoupled zero-point energy (Hartree).
    pub energy: f64,
    pub atoms: Vec<AtomEnvironment>,
    /// One uncoupled oscillator frequency per atom (Hartree).
    pub frequencies: Vec<f64>,
    /// Frequencies of the 3N coupled oscillator modes (Hartree), ascending.
    pub coupled_frequencies: Vec<f64>,
    /// Lowest eigenvalue of the coupled oscillator matrix (Hartree²).
    pub minimum_squared_frequency: f64,
}

/// Evaluate MBD@TS with externally supplied Hirshfeld volume ratios and
/// neutral atomic parameters, in molecular atom order.
pub fn ts_energy_from_volumes(
    molecule: &Molecule,
    ratios: &[f64],
    references: &[FreeAtom],
    beta: f64,
) -> Result<MbdEnergy> {
    if !beta.is_finite() || beta <= 0.0 || molecule.len() != references.len() {
        return Err(DftError::InvalidInput(
            "MBD@TS requires a positive finite beta and one free-atom reference per atom".into(),
        ));
    }
    let atoms = atom_environments(ratios, references)?;
    coupled_energy(molecule, atoms, beta, "MBD@TS")
}

fn coupled_energy(
    molecule: &Molecule,
    atoms: Vec<AtomEnvironment>,
    beta: f64,
    model: &str,
) -> Result<MbdEnergy> {
    let dimension = atoms
        .len()
        .checked_mul(3)
        .ok_or_else(|| DftError::InvalidInput("MBD oscillator matrix dimension overflow".into()))?;
    let frequencies: Vec<_> = atoms
        .iter()
        .map(|atom| 4.0 * atom.c6 / (3.0 * atom.polarizability.powi(2)))
        .collect();
    let mut coupling = Matrix::zeros(dimension, dimension);
    for (i, a) in atoms.iter().enumerate() {
        for k in 0..3 {
            coupling[(3 * i + k, 3 * i + k)] = frequencies[i].powi(2);
        }
        for (j, b) in atoms[..i].iter().enumerate() {
            let separation = molecule.atoms[i].position - molecule.atoms[j].position;
            let r = separation.norm();
            if !r.is_finite() || r <= 0.0 {
                return Err(DftError::InvalidInput(format!(
                    "atoms {i} and {j} have an invalid MBD separation"
                )));
            }
            let direction = (separation * r.recip()).to_array();
            let damping = super::logistic_damping(r, beta * (a.radius + b.radius), 6.0);
            let scale = frequencies[i]
                * frequencies[j]
                * (a.polarizability * b.polarizability).sqrt()
                * damping
                / r.powi(3);
            for x in 0..3 {
                for y in 0..3 {
                    let v = scale * (f64::from(x == y) - 3.0 * direction[x] * direction[y]);
                    coupling[(3 * i + x, 3 * j + y)] = v;
                    coupling[(3 * j + y, 3 * i + x)] = v;
                }
            }
        }
    }
    if coupling.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "MBD parameters give a nonfinite oscillator matrix".into(),
        ));
    }
    let (squared_frequencies, _) = symmetric_eigen(&coupling)?;
    let minimum_squared_frequency = squared_frequencies[0];
    if minimum_squared_frequency <= 0.0 {
        return Err(DftError::InvalidInput(format!("unstable {model} oscillator matrix (minimum squared frequency {minimum_squared_frequency:e} Hartree^2)")));
    }
    let coupled_frequencies: Vec<_> = squared_frequencies.into_iter().map(f64::sqrt).collect();
    let mut uncoupled = frequencies.clone();
    uncoupled.sort_by(f64::total_cmp);
    // Subtract the sorted uncoupled modes before summing the zero-point
    // shift, keeping the two large absolute zero-point energies separate.
    let energy = 0.5
        * coupled_frequencies
            .iter()
            .enumerate()
            .map(|(mode, coupled)| coupled - uncoupled[mode / 3])
            .sum::<f64>();
    Ok(MbdEnergy {
        energy,
        atoms,
        frequencies,
        coupled_frequencies,
        minimum_squared_frequency,
    })
}

/// Evaluate molecular MBD@TS on a fixed total AO density and the same
/// neutral spherical all-electron SAD references as pairwise TS.
pub fn ts_energy_on_density(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: &MolecularGrid,
    density: &Matrix,
    references: &[FreeAtom],
    beta: f64,
) -> Result<MbdEnergy> {
    let ratios = volume_ratios(molecule, basis, grid, density)?;
    ts_energy_from_volumes(molecule, &ratios, references, beta)
}
