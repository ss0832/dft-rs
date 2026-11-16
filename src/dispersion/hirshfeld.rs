// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Atomic promolecule densities for Hirshfeld partitions.

use crate::basis::MolBasis;
use crate::linalg::Matrix;
use crate::system::Molecule;
use crate::Result;

pub(crate) type AtomicScalars = Vec<Vec<f64>>;

/// Neutral, spherically averaged atomic densities carried in the molecular
/// AO basis, with one block per atom.
pub(crate) struct Promolecule {
    density: Matrix,
    atom_of: Vec<usize>,
    atom_count: usize,
}

impl Promolecule {
    pub(crate) fn new(molecule: &Molecule, basis: &MolBasis) -> Result<Self> {
        Ok(Self {
            density: crate::scf::sad::sad_density(molecule, basis)?,
            atom_of: basis.sph_ao_atoms(),
            atom_count: molecule.len(),
        })
    }

    /// Free-atom densities in atom-major, grid-point-minor order. A single
    /// matrix product suffices since only same-atom AO pairs contribute.
    pub(crate) fn collocate(&self, ao: &Matrix) -> AtomicScalars {
        let product = ao.matmul_seq(&self.density);
        self.contract(ao, &product)
    }

    /// Values and spatial gradients of the same neutral proatom densities.
    pub(crate) fn collocate_gradients(
        &self,
        ao: &Matrix,
        grad: &[Matrix; 3],
    ) -> (AtomicScalars, [AtomicScalars; 3]) {
        let product = ao.matmul_seq(&self.density);
        let values = self.contract(ao, &product);
        // The density matrix is symmetric, so both differentiated AO factors agree.
        let gradients = grad.each_ref().map(|d| {
            let mut values = self.contract(d, &product);
            for value in values.iter_mut().flatten() {
                *value *= 2.0;
            }
            values
        });
        (values, gradients)
    }

    fn contract(&self, ao: &Matrix, product: &Matrix) -> AtomicScalars {
        let mut densities = vec![vec![0.0; ao.rows]; self.atom_count];
        for point in 0..ao.rows {
            for (mu, &atom) in self.atom_of.iter().enumerate() {
                densities[atom][point] += product[(point, mu)] * ao[(point, mu)];
            }
        }
        densities
    }
}
