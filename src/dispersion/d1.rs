// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Grimme's original DFT-D pair correction, J. Comput. Chem. 25, 1463–1473
//! (2004), with the D1 harmonic mean and damping steepness 23.

use crate::dual::Scalar;
use crate::system::Molecule;
use crate::{DftError, Result};

/// Atomic D1 reference parameters in molecular atom order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomParameters {
    /// Atomic dispersion coefficient (Hartree Bohr⁶).
    pub c6: f64,
    /// Atomic van der Waals radius (Bohr).
    pub radius: f64,
}

/// Molecular D1 pair energy in Hartree. The atomic parameters and global
/// functional scale `s6` come from the caller's chosen D1 parametrization.
pub fn energy(molecule: &Molecule, atoms: &[AtomParameters], s6: f64) -> Result<f64> {
    let positions: Vec<_> = molecule
        .atoms
        .iter()
        .map(|atom| atom.position.to_array())
        .collect();
    energy_geometry(&positions, atoms, s6)
}

/// D1 energy on Cartesian coordinates carrying analytic derivative seeds.
pub fn energy_geometry<S: Scalar>(
    positions: &[[S; 3]],
    atoms: &[AtomParameters],
    s6: f64,
) -> Result<S> {
    if positions.len() != atoms.len() {
        return Err(DftError::InvalidInput(
            "D1 needs one reference per atom".into(),
        ));
    }
    validate_parameters(atoms, s6)?;
    let mut energy = S::from_f64(0.0);
    for (i, a) in atoms.iter().enumerate() {
        for (j, b) in atoms[..i].iter().enumerate() {
            let [x, y, z] = std::array::from_fn(|axis| positions[i][axis] - positions[j][axis]);
            let r2 = x * x + y * y + z * z;
            let (c6, radius) = pair_parameters(*a, *b);
            if !r2.value().is_finite()
                || r2.value() <= 0.0
                || !radius.is_finite()
                || !c6.is_finite()
            {
                return Err(DftError::InvalidInput(format!(
                    "atoms {i} and {j} have an invalid D1 separation or reference"
                )));
            }
            energy = energy - radial(r2.sqrt(), radius).scale(s6 * c6);
        }
    }
    if !energy.value().is_finite() {
        return Err(DftError::InvalidInput(
            "D1 parameters give a nonfinite pair energy".into(),
        ));
    }
    Ok(energy)
}

pub(crate) fn validate_parameters(atoms: &[AtomParameters], s6: f64) -> Result<()> {
    let positive_finite = |v: f64| v.is_finite() && v > 0.0;
    if atoms.is_empty()
        || !positive_finite(s6)
        || atoms
            .iter()
            .any(|atom| !positive_finite(atom.c6) || !positive_finite(atom.radius))
    {
        return Err(DftError::InvalidInput("D1 needs positive finite atom coefficients and radii, with one positive functional scale s6".into()));
    }
    Ok(())
}

pub(crate) fn pair_parameters(a: AtomParameters, b: AtomParameters) -> (f64, f64) {
    (2.0 / (a.c6.recip() + b.c6.recip()), a.radius + b.radius)
}

/// Damped inverse sixth power; the coefficient and energy sign are separate.
pub(crate) fn radial<S: Scalar>(r: S, radius: f64) -> S {
    super::logistic_damping(r, radius, 23.0) * r.powi(-6)
}
