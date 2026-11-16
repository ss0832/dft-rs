// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Dimensional electronic basis operators and open-direction XC quadrature.
//! This driver builds genuine Gaussian Bloch operators; it does not substitute
//! a prescribed-charge calculation for an electronic SCF.

use super::bloch::BlochMatrix;
use super::grid::{dimensional_weight_gradient, PeriodicGrid};
use super::integrals::{
    bloch, bloch_derivative, dimensional_image_blocks, dimensional_image_derivative_blocks,
    shifted, Operator,
};
use super::PeriodicGeometry;
use crate::basis::{n_sph, MolBasis};
use crate::grid::{eval::eval_aos, GridOptions};
use crate::linalg::symmetric_eigen;
use crate::math::Vec3;
use crate::system::Molecule;
use crate::{DftError, Result};
use rayon::prelude::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct PeriodicBasisOptions {
    pub k_mesh: [usize; 3],
    /// Shell-pair image cutoff in Bohr; never an electrostatic truncation.
    pub basis_cutoff: f64,
    /// Independent physical radial/Becke cutoff in Bohr.
    pub grid_cutoff: f64,
    pub grid: Option<GridOptions>,
    pub grid_derivatives: bool,
}
impl Default for PeriodicBasisOptions {
    fn default() -> Self {
        Self {
            k_mesh: [1; 3],
            basis_cutoff: 12.0,
            grid_cutoff: 12.0,
            grid: None,
            grid_derivatives: false,
        }
    }
}

/// Bloch AO values on arbitrary Cartesian points, [point, AO].
/// The image selection is local to each point and shell. Open directions are
/// not wrapped or repeated. This is usable independently of SCF occupations.
pub fn dimensional_bloch_aos(
    mb: &MolBasis,
    geometry: &PeriodicGeometry,
    points: &[Vec3],
    k: Vec3,
    cutoff: f64,
) -> Result<BlochMatrix> {
    if !k.norm().is_finite() || !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(DftError::InvalidInput(
            "Bloch AO wave vector and positive cutoff must be finite".into(),
        ));
    }
    let values: Vec<(Vec<f64>, Vec<f64>)> = points
        .par_iter()
        .map(|&point| -> Result<_> {
            let mut indices = BTreeSet::new();
            for shell in &mb.shells {
                indices.extend(geometry.nearby_translation_indices(shell.center - point, cutoff)?);
            }
            let mut re = vec![0.0; mb.nao_sph];
            let mut im = vec![0.0; mb.nao_sph];
            for index in indices {
                let translation = geometry.lattice().translation(index);
                let basis = shifted(mb, translation);
                let values = eval_aos(&basis, &[point], false);
                let phase = k.dot(translation);
                for shell in &basis.shells {
                    if (shell.center - point).norm() > cutoff + 64.0 * f64::EPSILON * (1.0 + cutoff)
                    {
                        continue;
                    }
                    for ao in shell.sph_off..shell.sph_off + n_sph(shell.l) {
                        re[ao] += phase.cos() * values.phi[(0, ao)];
                        im[ao] += phase.sin() * values.phi[(0, ao)];
                    }
                }
            }
            Ok((re, im))
        })
        .collect::<Result<_>>()?;
    let mut out = BlochMatrix::zeros(points.len(), mb.nao_sph);
    for (point, (re, im)) in values.into_iter().enumerate() {
        for ao in 0..mb.nao_sph {
            out.re[(point, ao)] = re[ao];
            out.im[(point, ao)] = im[ao];
        }
    }
    Ok(out)
}

pub struct PeriodicBasisReport {
    pub k_points: Vec<Vec3>,
    pub overlap: Vec<BlochMatrix>,
    pub kinetic: Vec<BlochMatrix>,
    /// [k][atom][Cartesian axis], fixed-cell derivatives.
    pub overlap_derivatives: Vec<Vec<[BlochMatrix; 3]>>,
    pub kinetic_derivatives: Vec<Vec<[BlochMatrix; 3]>>,
    pub minimum_overlap_eigenvalue: f64,
    pub hermiticity_defect: f64,
    pub translation_defect: f64,
    pub grid: Option<PeriodicGrid>,
    /// [grid point][atom], derivatives including motion with the owner atom.
    pub moving_weight_derivatives: Option<Vec<Vec<Vec3>>>,
    pub grid_overlap_defect: Option<f64>,
    pub warnings: Vec<String>,
    pub model: &'static str,
    pub units: &'static str,
}

/// Common Rust/CLI/Python/ASE electronic basis foundation driver.
/// Charge and functional class do not change S, T or the integration measure.
pub fn periodic_basis(
    mol: &Molecule,
    mb: &MolBasis,
    geometry: &PeriodicGeometry,
    options: &PeriodicBasisOptions,
) -> Result<PeriodicBasisReport> {
    if mol.is_empty()
        || mb.nao_sph == 0
        || !options.basis_cutoff.is_finite()
        || options.basis_cutoff <= 0.0
        || !options.grid_cutoff.is_finite()
        || options.grid_cutoff <= 0.0
    {
        return Err(DftError::InvalidInput(
            "periodic basis requires atoms, AOs and finite positive cutoffs".into(),
        ));
    }
    if options.grid_derivatives && options.grid.is_none() {
        return Err(DftError::InvalidInput(
            "grid derivatives require a dimensional grid".into(),
        ));
    }
    for (i, atom) in mol.atoms.iter().enumerate() {
        if !(1..=118).contains(&atom.z) || !atom.position.norm().is_finite() {
            return Err(DftError::InvalidInput(
                "periodic basis atom numbers and positions must be valid".into(),
            ));
        }
        for other in &mol.atoms[..i] {
            if geometry
                .minimum_image(atom.position - other.position)?
                .norm()
                < 1e-10
            {
                return Err(DftError::InvalidInput(
                    "distinct atoms must not coincide in periodic images".into(),
                ));
            }
        }
    }
    if mb.shells.iter().any(|s| {
        s.atom >= mol.len()
            || !s.center.norm().is_finite()
            || (s.center - mol.atoms[s.atom].position).norm() > 1e-10
    }) {
        return Err(DftError::InvalidInput(
            "periodic basis shells must match their physical atom centres".into(),
        ));
    }
    let k_points = geometry.k_points(options.k_mesh)?;
    let mut operators = Vec::new();
    let mut derivatives = Vec::new();
    let mut hermiticity_defect = 0.0f64;
    let mut translation_defect = 0.0f64;
    for operator in [Operator::Overlap, Operator::Kinetic] {
        let images = dimensional_image_blocks(mb, geometry, options.basis_cutoff, operator)?;
        let d_images =
            dimensional_image_derivative_blocks(mb, geometry, options.basis_cutoff, operator)?;
        let mut matrices = Vec::new();
        let mut d_matrices = Vec::new();
        for &k in &k_points {
            let matrix = BlochMatrix::from_pair(bloch(&images, k));
            hermiticity_defect = hermiticity_defect.max(matrix.hermiticity_error());
            let d: Vec<[BlochMatrix; 3]> = bloch_derivative(&d_images, k, mol.len(), mb.nao_sph)
                .into_iter()
                .map(|axes| axes.map(BlochMatrix::from_pair))
                .collect();
            for axis in 0..3 {
                let mut sum = BlochMatrix::zeros(mb.nao_sph, mb.nao_sph);
                for atom in &d {
                    hermiticity_defect = hermiticity_defect.max(atom[axis].hermiticity_error());
                    sum.add_scaled(&atom[axis], 1.0);
                }
                translation_defect = translation_defect.max(max_component(&sum));
            }
            matrices.push(matrix);
            d_matrices.push(d);
        }
        operators.push(matrices);
        derivatives.push(d_matrices);
    }
    let kinetic = operators.pop().unwrap();
    let overlap = operators.pop().unwrap();
    let kinetic_derivatives = derivatives.pop().unwrap();
    let overlap_derivatives = derivatives.pop().unwrap();
    let mut minimum_overlap_eigenvalue = f64::INFINITY;
    for matrix in &overlap {
        minimum_overlap_eigenvalue =
            minimum_overlap_eigenvalue.min(symmetric_eigen(&matrix.real_embedding())?.0[0]);
    }
    let mut warnings = Vec::new();
    if minimum_overlap_eigenvalue <= 0.0 {
        warnings.push("overlap is not positive definite at the requested image cutoff; refine the cutoff or basis".into());
    }
    let (grid, moving_weight_derivatives, grid_overlap_defect) = if let Some(grid_options) =
        &options.grid
    {
        let grid =
            PeriodicGrid::build_dimensional(mol, geometry, grid_options, options.grid_cutoff)?;
        let moving = if options.grid_derivatives {
            Some(
                grid.points
                    .par_iter()
                    .zip(&grid.atom_of)
                    .zip(&grid.raw_weights)
                    .map(|((&point, &owner), &raw)| -> Result<_> {
                        let (_, mut gradients) = dimensional_weight_gradient(
                            mol,
                            geometry,
                            point,
                            owner,
                            options.grid_cutoff,
                        )?;
                        let point_gradient = gradients.iter().fold(Vec3::zero(), |s, &g| s - g);
                        gradients[owner] += point_gradient;
                        for g in &mut gradients {
                            *g = *g * raw;
                        }
                        Ok(gradients)
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let mut defect = 0.0f64;
        for (index, &k) in k_points.iter().enumerate() {
            let ao = dimensional_bloch_aos(mb, geometry, &grid.points, k, options.basis_cutoff)?;
            let mut quadrature = weighted_bloch_overlap(&ao, &grid.weights);
            quadrature.add_scaled(&overlap[index], -1.0);
            defect = defect.max(max_component(&quadrature));
        }
        warnings.push("grid overlap defect measures the combined quadrature, radial-tail and AO-image truncations; refine these independently".into());
        (Some(grid), moving, Some(defect))
    } else {
        (None, None, None)
    };
    Ok(PeriodicBasisReport { k_points, overlap, kinetic, overlap_derivatives, kinetic_derivatives,
        minimum_overlap_eigenvalue, hermiticity_defect, translation_defect, grid,
        moving_weight_derivatives, grid_overlap_defect, warnings,
        model: "Gaussian Bloch overlap/kinetic and dimensional atom-centred quadrature at fixed lattice",
        units: "overlap dimensionless; kinetic Hartree; derivatives per Bohr; grid weights Bohr^3",
    })
}

pub(crate) fn weighted_bloch_overlap(ao: &BlochMatrix, weights: &[f64]) -> BlochMatrix {
    debug_assert_eq!(ao.rows(), weights.len());
    let mut quadrature = BlochMatrix::zeros(ao.cols(), ao.cols());
    for (point, &w) in weights.iter().enumerate() {
        for i in 0..ao.cols() {
            for j in 0..ao.cols() {
                // chi_i^* chi_j uses the same positive Bloch phase as S(k).
                quadrature.re[(i, j)] += w
                    * (ao.re[(point, i)] * ao.re[(point, j)]
                        + ao.im[(point, i)] * ao.im[(point, j)]);
                quadrature.im[(i, j)] += w
                    * (ao.re[(point, i)] * ao.im[(point, j)]
                        - ao.im[(point, i)] * ao.re[(point, j)]);
            }
        }
    }
    quadrature
}

fn max_component(m: &BlochMatrix) -> f64 {
    m.re.as_slice()
        .iter()
        .chain(m.im.as_slice())
        .fold(0.0, |max, v| max.max(v.abs()))
}
