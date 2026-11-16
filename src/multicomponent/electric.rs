// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared electric multipole operators for stationary and response fields.

use super::add;
use crate::basis::MolBasis;
use crate::integrals::{dipole_matrices, moment_matrices};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::electric_tensor::{QUADRUPOLE_AXES, quadrupole_modes, quadrupole_weights};
use crate::system::Molecule;
use crate::{DftError, Result};

/// Finite electric field gradient in the five orthonormal traceless modes.
/// The potential is `-(q/2) sum_g amplitude_g r^T mode_g r`
/// with the displacement measured from the supplied origin.
#[derive(Clone, Copy, Debug)]
pub struct QuadrupoleField {
    pub amplitudes: [f64; 5],
    pub origin: Vec3,
}

impl QuadrupoleField {
    pub(super) fn validate(&self) -> Result<()> {
        if self
            .amplitudes
            .iter()
            .chain(&self.origin.to_array())
            .any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "finite multicomponent quadrupole field needs finite amplitudes and origin".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn add_potential(&self, operator: &mut Matrix, basis: &MolBasis, charge: f64) {
        for (matrix, amplitude) in quadrupole_operators(basis, self.origin)
            .iter()
            .zip(self.amplitudes)
        {
            add(operator, matrix, -charge * amplitude);
        }
    }

    pub(super) fn classical_energy(&self, molecule: &Molecule, charges: &[f64]) -> f64 {
        let moments = classical_moments(molecule, charges, self.origin);
        -moments[3..]
            .iter()
            .zip(self.amplitudes)
            .map(|(moment, field)| moment * field)
            .sum::<f64>()
    }
}

fn quadrupole_operators(basis: &MolBasis, origin: Vec3) -> [Matrix; 5] {
    let moments = moment_matrices(basis, origin, 2);
    let monomials = QUADRUPOLE_AXES.map(|(a, b)| {
        let mut power = [0; 3];
        power[a] += 1;
        power[b] += 1;
        let key = (power[0], power[1], power[2]);
        &moments
            .iter()
            .find(|(p, _)| *p == key)
            .expect("Cartesian degree two")
            .1
    });
    quadrupole_modes().map(|mode| {
        let mut operator = Matrix::zeros(basis.nao_sph, basis.nao_sph);
        for (moment, coefficient) in monomials.iter().zip(quadrupole_weights(mode)) {
            add(&mut operator, moment, coefficient);
        }
        operator
    })
}

pub(super) fn field_operators(basis: &MolBasis, origin: Vec3) -> [Matrix; 8] {
    let [x, y, z] = dipole_matrices(basis, origin);
    let [q0, q1, q2, q3, q4] = quadrupole_operators(basis, origin);
    [x, y, z, q0, q1, q2, q3, q4]
}

pub(super) fn classical_moments(molecule: &Molecule, charges: &[f64], origin: Vec3) -> [f64; 8] {
    let modes = quadrupole_modes();
    let mut moments = [0.0; 8];
    for (atom, &charge) in molecule.atoms.iter().zip(charges) {
        let r = (atom.position - origin).to_array();
        for (value, coordinate) in moments[..3].iter_mut().zip(r) {
            *value += charge * coordinate;
        }
        for (value, mode) in moments[3..].iter_mut().zip(modes) {
            for ((a, b), weight) in QUADRUPOLE_AXES.into_iter().zip(quadrupole_weights(mode)) {
                *value += charge * weight * r[a] * r[b];
            }
        }
    }
    moments
}
