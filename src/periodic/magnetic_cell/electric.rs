// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Berry electric enthalpy on continuum magnetic-translation orbitals.
use super::*;
use crate::linalg::Matrix;
use crate::response::berry_field::BerryField;

pub(super) fn geometry(
    cell: &MagneticCell,
    basis: &MolBasis,
    cutoff: f64,
    sources: &[ElectrostaticSource],
    background_dipole: Vec3,
    field: [f64; 3],
    electrons: f64,
    directions: &[[f64; 3]],
    order: usize,
) -> Result<Option<BerryField>> {
    if field.iter().all(|x| *x == 0.0) {
        return Ok(None);
    }
    if cell.geometry.dimension() != 3 || field.iter().any(|x| !x.is_finite()) {
        return Err(DftError::Unsupported {
            feature: "electric enthalpy on magnetic orbitals".into(),
            missing:
                "a three-dimensional finite insulating magnetic cell with finite electric field"
                    .into(),
        });
    }
    let overlap = cell.nuclear_operator_taylor(
        basis,
        cutoff,
        Vec3::zero(),
        MagneticPairOperator::Fourier,
        directions,
        order,
    )?;
    let n = 2 * basis.nao_sph;
    let mut polar = std::array::from_fn(|_| Vec::new());
    let mut ionic = std::array::from_fn(|_| vec![0.0; order + 1]);
    let mut reduced = ionic.clone();
    let mut phase_reference = [0.0; 3];
    let nuclear_charge = sources
        .iter()
        .take(directions.len())
        .map(|s| s.charge)
        .sum::<f64>();
    for a in 0..3 {
        let b = cell.geometry.lattice().reciprocal()[a];
        let q = cell.nuclear_operator_taylor(
            basis,
            cutoff,
            b,
            MagneticPairOperator::Fourier,
            directions,
            order,
        )?;
        polar[a] = q
            .iter()
            .zip(&overlap)
            .map(|(q, s)| {
                let mut q = spin_diagonal(q);
                q.add_scaled(&spin_diagonal(s), -1.0);
                (q.re, q.im)
            })
            .collect();
        ionic[a][0] = b.dot(background_dipole);
        if nuclear_charge != 0.0 {
            phase_reference[a] = electrons / nuclear_charge
                * sources
                    .iter()
                    .take(directions.len())
                    .map(|s| s.charge * b.dot(s.position))
                    .sum::<f64>();
        }
        for (i, source) in sources.iter().enumerate() {
            ionic[a][0] += source.charge * b.dot(source.position);
            if order > 0 {
                let d = directions.get(i).copied().unwrap_or([0.0; 3]);
                ionic[a][1] += source.charge * (b.x * d[0] + b.y * d[1] + b.z * d[2]);
            }
        }
        let lattice = cell.geometry.lattice().a[a];
        reduced[a][0] =
            (field[0] * lattice.x + field[1] * lattice.y + field[2] * lattice.z) / (2.0 * PI);
        // Retain the full spinor dimension also on exactly zero operators.
        debug_assert!(polar[a].iter().all(|m: &(Matrix, Matrix)| m.0.rows == n));
    }
    Ok(Some(BerryField {
        phase_reference: Some(phase_reference),
        polar,
        ionic,
        reduced,
    }))
}

impl MagneticDftBackend<'_> {
    /// Dipole per magnetic cell on the local single-point Berry branch.
    /// Uniform compensation is centred at half the sum of lattice vectors;
    /// an explicit Gaussian countercharge contributes its actual centroid.
    pub fn electric_dipole(&self, density: &[BlochMatrix]) -> Result<Option<[f64; 3]>> {
        let Some(field) = &self.electric else {
            return Ok(None);
        };
        if density.len() != 1 {
            return Err(DftError::InvalidInput(
                "finite-cell Berry dipole needs one magnetic density sector".into(),
            ));
        }
        let phases = field.phases(&[(density[0].re.clone(), density[0].im.clone())])?;
        let mut dipole = [0.0; 3];
        for (a, phase) in phases.iter().enumerate() {
            let vector = self.cell.geometry.lattice().a[a].to_array();
            for i in 0..3 {
                dipole[i] -= phase * vector[i] / (2.0 * PI);
            }
        }
        Ok(Some(dipole))
    }
}
