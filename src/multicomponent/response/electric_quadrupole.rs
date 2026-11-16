// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Joint dipole and field-gradient response of a multicomponent reference.

use super::{CoupledDensityResponse, CoupledResponseSolver};
use crate::math::Vec3;
use crate::multicomponent::electric::{classical_moments, field_operators};
use crate::response::ResponseSolverConfig;
use crate::system::Molecule;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct CoupledQuadrupoleResponse {
    /// Three dipole and five field-conjugate quadrupole moments.
    pub moments: [f64; 8],
    /// Observed moment and driven field, in the same normalization.
    pub tensor: [[f64; 8]; 8],
    pub responses: Vec<CoupledDensityResponse>,
}

impl CoupledResponseSolver<'_> {
    /// Static response to uniform fields and the five orthonormal traceless
    /// gradients. Each field leg is solved before contractions. The
    /// field-conjugate gradient moments can be multiplied by three for
    /// the Buckingham convention.
    pub fn electric_quadrupole(
        &self,
        molecule: &Molecule,
        origin: Vec3,
        config: &ResponseSolverConfig,
    ) -> Result<CoupledQuadrupoleResponse> {
        if origin.to_array().iter().any(|x| !x.is_finite()) {
            return Err(DftError::InvalidInput(
                "non-finite joint quadrupole origin".into(),
            ));
        }
        let electronic = field_operators(self.electronic_basis, origin);
        let quantum: Vec<_> = self
            .particles
            .iter()
            .map(|p| field_operators(&p.basis, origin))
            .collect();
        let mut classical_charges = self.electronic_basis.effective_charges(molecule);
        for index in self.particles.iter().filter_map(|p| p.replaced_atom) {
            classical_charges[index] = 0.0;
        }
        let mut moments = classical_moments(molecule, &classical_charges, origin);
        let mut responses = Vec::with_capacity(8);
        let mut probes = Vec::with_capacity(8);
        for (mode, operator) in electronic.iter().enumerate() {
            let electronic = [operator.clone(), operator.clone()];
            let mut q: Vec<_> = quantum
                .iter()
                .map(|operators| operators[mode].clone())
                .collect();
            moments[mode] -= self
                .state
                .electronic_density
                .iter()
                .map(|p| p.frobenius_dot(operator))
                .sum::<f64>();
            for ((m, p), state) in q.iter_mut().zip(self.particles).zip(&self.state.quantum) {
                moments[mode] += p.charge * state.density.frobenius_dot(m);
                for x in m.as_mut_slice() {
                    *x *= -p.charge;
                }
            }
            probes.push(self.project(&electronic, &q));
            responses.push(self.solve(&electronic, &q, config)?);
        }
        let tensor = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                -2.0 * probes[a]
                    .iter()
                    .zip(&responses[b].amplitudes)
                    .map(|(u, v)| u * v)
                    .sum::<f64>()
            })
        });
        Ok(CoupledQuadrupoleResponse {
            moments,
            tensor,
            responses,
        })
    }
}
