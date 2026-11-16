// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Tkatchenko–Scheffler molecular pair dispersion on a supplied AO density.
//! The volume scaling and pair energy follow Phys. Rev. Lett. 102, 073005
//! (2009), equations (6)–(12). Free-atom constants are supplied by the caller.

use super::hirshfeld::Promolecule;
use crate::basis::MolBasis;
use crate::grid::{eval::eval_aos_full, MolecularGrid};
use crate::linalg::Matrix;
use crate::system::Molecule;
use crate::{DftError, Result};

/// Neutral free-atom parameters, in input atom order.
#[derive(Clone, Copy, Debug)]
pub struct FreeAtom {
    /// Static dipole polarizability (Bohr³).
    pub polarizability: f64,
    /// Homonuclear dispersion coefficient (Hartree Bohr⁶).
    pub c6: f64,
    /// Free-atom van der Waals radius (Bohr).
    pub radius: f64,
}

/// Effective in-molecule atom parameters and their Hirshfeld volume ratio.
#[derive(Clone, Copy, Debug)]
pub struct AtomEnvironment {
    /// Effective volume divided by the free-atom volume.
    pub volume_ratio: f64,
    /// Effective static dipole polarizability (Bohr³).
    pub polarizability: f64,
    /// Effective homonuclear coefficient (Hartree Bohr⁶).
    pub c6: f64,
    /// Effective van der Waals radius (Bohr).
    pub radius: f64,
}

/// Tkatchenko–Scheffler energy and diagnostic atomic properties.
#[derive(Clone, Debug)]
pub struct TsEnergy {
    /// Sum of the pair corrections (Hartree).
    pub energy: f64,
    pub atoms: Vec<AtomEnvironment>,
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Hirshfeld volume ratios from the total AO density and the neutral
/// spherical, all-electron SAD promolecule in the supplied AO basis.
pub(super) fn volume_ratios(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: &MolecularGrid,
    density: &Matrix,
) -> Result<Vec<f64>> {
    if basis.n_core_electrons != 0 || molecule.atoms.iter().any(|atom| atom.z == 0) {
        return Err(DftError::Unsupported {
            feature: "Hirshfeld dispersion with an ECP or ghost centre".into(),
            missing: "an all-electron reconstruction of each participating atomic density".into(),
        });
    }
    if density.rows != basis.nao_sph
        || density.cols != basis.nao_sph
        || density.as_slice().iter().any(|v| !v.is_finite())
        || grid.points.len() != grid.weights.len()
        || grid.is_empty()
        || grid.weights.iter().any(|v| !v.is_finite())
        || grid.points.iter().any(|v| !v.norm2().is_finite())
    {
        return Err(DftError::InvalidInput(
            "Hirshfeld dispersion requires a finite total AO density and molecular quadrature"
                .into(),
        ));
    }
    let promolecule = Promolecule::new(molecule, basis)?;
    let n = molecule.len();
    let mut v_free = vec![0.0; n];
    let mut v_eff = vec![0.0; n];
    const BATCH: usize = 300;
    for (points, weights) in grid.points.chunks(BATCH).zip(grid.weights.chunks(BATCH)) {
        let ao = eval_aos_full(basis, points, false, false).phi;
        let phip = ao.matmul_seq(density);
        let proatoms = promolecule.collocate(&ao);
        for (i, (&point, &weight)) in points.iter().zip(weights).enumerate() {
            let promol: f64 = proatoms.iter().map(|rho| rho[i].max(0.0)).sum();
            if promol <= 1e-300 {
                continue;
            }
            let rho: f64 = (0..basis.nao_sph)
                .map(|mu| phip[(i, mu)] * ao[(i, mu)])
                .sum();
            for atom in 0..n {
                let r3 = (point - molecule.atoms[atom].position).norm().powi(3);
                let free = weight * r3 * proatoms[atom][i].max(0.0);
                v_free[atom] += free;
                v_eff[atom] += free * (rho.max(0.0) / promol);
            }
        }
    }
    v_eff
        .into_iter()
        .zip(v_free)
        .enumerate()
        .map(|(atom, (eff, free))| {
            let ratio = eff / free;
            if positive_finite(free) && positive_finite(ratio) {
                Ok(ratio)
            } else {
                Err(DftError::InvalidInput(format!(
                    "atom {atom} has an invalid Hirshfeld volume"
                )))
            }
        })
        .collect()
}

/// Combine volume ratios with the supplied neutral atomic data. This is
/// also the unscreened single-oscillator input for many-body dispersion.
pub(super) fn atom_environments(
    volume_ratios: &[f64],
    references: &[FreeAtom],
) -> Result<Vec<AtomEnvironment>> {
    if volume_ratios.len() != references.len() || references.is_empty() {
        return Err(DftError::InvalidInput(
            "TS requires one free-atom reference for every volume ratio".into(),
        ));
    }
    volume_ratios
        .iter()
        .zip(references)
        .enumerate()
        .map(|(atom, (&volume_ratio, free))| {
            if ![volume_ratio, free.polarizability, free.c6, free.radius]
                .into_iter()
                .all(positive_finite)
            {
                return Err(DftError::InvalidInput(format!(
                    "atom {atom} needs positive finite TS reference parameters and volume ratio"
                )));
            }
            let result = AtomEnvironment {
                volume_ratio,
                polarizability: volume_ratio * free.polarizability,
                c6: volume_ratio.powi(2) * free.c6,
                radius: volume_ratio.cbrt() * free.radius,
            };
            if ![result.polarizability, result.c6, result.radius]
                .into_iter()
                .all(positive_finite)
            {
                return Err(DftError::InvalidInput(format!(
                    "atom {atom} has invalid effective dispersion parameters"
                )));
            }
            Ok(result)
        })
        .collect()
}

/// Pair dispersion evaluated with a caller-supplied set of Hirshfeld volume
/// ratios, in the same atom order as the molecule and free-atom data.
pub fn energy_from_volumes(
    molecule: &Molecule,
    volume_ratios: &[f64],
    references: &[FreeAtom],
    radius_scale: f64,
) -> Result<TsEnergy> {
    if !positive_finite(radius_scale) || molecule.len() != references.len() {
        return Err(DftError::InvalidInput(
            "TS requires one free-atom reference per atom and a positive finite radius scale"
                .into(),
        ));
    }
    let atoms = atom_environments(volume_ratios, references)?;
    let mut energy = 0.0;
    for (i, a) in atoms.iter().enumerate() {
        for (j, b) in atoms[..i].iter().enumerate() {
            let r = (molecule.atoms[i].position - molecule.atoms[j].position).norm();
            if !positive_finite(r) {
                return Err(DftError::InvalidInput(format!(
                    "atoms {i} and {j} have an invalid TS separation"
                )));
            }
            let alpha_ratio = a.polarizability / b.polarizability;
            let c6 = 2.0 / (alpha_ratio / a.c6 + alpha_ratio.recip() / b.c6);
            let radius = radius_scale * (a.radius + b.radius);
            if !positive_finite(c6) || !positive_finite(radius) {
                return Err(DftError::InvalidInput(format!(
                    "atoms {i} and {j} have invalid TS pair parameters"
                )));
            }
            energy -= super::d2_radial_g(r, radius) * c6;
        }
    }
    if !energy.is_finite() {
        return Err(DftError::InvalidInput(
            "TS parameters give a nonfinite pair energy".into(),
        ));
    }
    Ok(TsEnergy { energy, atoms })
}

/// Evaluate post-SCF TS pair dispersion at a fixed total AO density. The
/// neutral proatom densities are spherical all-electron SAD densities in
/// the same Gaussian basis and quadrature as the supplied density.
pub fn energy_on_density(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: &MolecularGrid,
    density: &Matrix,
    references: &[FreeAtom],
    radius_scale: f64,
) -> Result<TsEnergy> {
    let ratios = volume_ratios(molecule, basis, grid, density)?;
    energy_from_volumes(molecule, &ratios, references, radius_scale)
}
