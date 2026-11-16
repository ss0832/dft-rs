// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic cubic/quartic tensors on caller patterns or Cartesian coordinates.
//! The nuclear response equations determine each independent component. Tensor
//! permutations are filled only after explicit reverse-order comparisons.

use super::mixed::{polarized_tensor, tensor_indices as indices};
use super::nuclear::nuclear_taylor;
use super::{ResponseReport, ResponseSolverConfig};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct AnalyticForceConstants {
    pub order: usize,
    pub patterns: Vec<Vec<[f64; 3]>>,
    /// Row-major pattern indices; actual derivatives without factorial scaling.
    pub tensor: Vec<f64>,
    pub accumulation_roundoff: Vec<f64>,
    pub permutation_defect: f64,
    pub permutation_comparisons: usize,
    /// Present when a Cartesian block contains every coordinate of every atom.
    pub translation_residual: Option<f64>,
    pub translation_comparisons: usize,
    pub independent_components: usize,
    pub directional_evaluations: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn analytic_force_constants(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    patterns: &[Vec<[f64; 3]>],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<AnalyticForceConstants>> {
    analytic_force_constants_with(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        patterns,
        order,
        config,
        |_, _| Ok(0.0),
    )
}

/// Add a geometry-only Taylor coefficient to each relaxed electronic path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn analytic_force_constants_with(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    patterns: &[Vec<[f64; 3]>],
    order: usize,
    config: &ResponseSolverConfig,
    mut correction: impl FnMut(&[[f64; 3]], usize) -> Result<f64>,
) -> Result<ResponseReport<AnalyticForceConstants>> {
    config.validate()?;
    if !(3..=4).contains(&order)
        || patterns.is_empty()
        || patterns
            .iter()
            .any(|p| p.len() != mol.len() || p.iter().flatten().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "force constants require order=3 or 4 and nonempty finite atom-by-Cartesian patterns"
                .into(),
        ));
    }
    let dimension = patterns.len();
    let vectors: Vec<Vec<f64>> = patterns
        .iter()
        .map(|p| p.iter().flatten().copied().collect())
        .collect();
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    let assembled = polarized_tensor(&vectors, order, 1, opts.incore_mb, true, |d, n| {
        let direction: Vec<_> = d.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
        let report = nuclear_taylor(mol, mb, functional, grid, scf, opts, &direction, n, config)?;
        converged &= report.converged;
        residuals.extend(report.residuals);
        iterations.extend(report.iterations);
        warnings.extend(report.warnings);
        Ok(vec![
            report.value.energy_coefficients[n] + correction(&direction, n)?,
        ])
    })?;
    let tensor = assembled
        .values
        .into_iter()
        .next()
        .expect("one energy channel");
    let accumulation_roundoff = assembled
        .roundoff
        .into_iter()
        .next()
        .expect("one roundoff channel");
    let permutation_defect = assembled.permutation_defects[0];
    let permutation_comparisons = assembled.permutation_comparisons;
    let directional_evaluations = assembled.directional_evaluations;
    if tensor
        .iter()
        .zip(&accumulation_roundoff)
        .any(|(v, r)| *r > config.tolerance * v.abs().max(1.0))
    {
        converged = false;
        warnings.push(
            "mixed nuclear polarization accumulation exceeds the requested roundoff scale".into(),
        );
    }
    let scale = tensor.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    if permutation_defect > config.tolerance * scale {
        converged = false;
        warnings.push("independently evaluated reverse-order force constants violate the requested interchange tolerance".into());
    }
    warnings.sort();
    warnings.dedup();
    Ok(ResponseReport {
        value:AnalyticForceConstants {order,patterns:patterns.to_vec(),tensor,accumulation_roundoff,
            permutation_defect,permutation_comparisons,translation_residual:None,translation_comparisons:0,
            independent_components:assembled.independent_components,directional_evaluations},
        shape:vec![dimension;order],units:"Hartree; actual mixed derivatives per supplied Bohr displacement patterns",
        functional:functional.description.clone(),reference:if scf.electronic_temperature > 0.0 {
            super::diagnostics::density_diagnostics(scf)?
        } else {
            super::diagnostics::reference_diagnostics(scf)?
        },
        model:if scf.electronic_temperature > 0.0 {
            "static fixed-temperature canonical Mermin free-energy tensor; analytic moving basis/grid; exact polarization contraction"
        } else {
            "static integer-occupation all-electron KS/HF; analytic moving basis/grid; exact polarization contraction"
        },
        axis_convention:"row-major pattern indices; cubic/quartic derivatives without factorial \
            normalization; reverse-order symmetry measured before permutation \
            filling",
        converged,residuals,iterations,warnings,
    })
}

/// Cartesian coordinate IDs are 3*atom+axis. Their order sets the tensor axes.
#[allow(clippy::too_many_arguments)]
pub fn cartesian_force_constant_block(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    coordinates: &[usize],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<AnalyticForceConstants>> {
    let mut sorted = coordinates.to_vec();
    sorted.sort_unstable();
    if sorted.is_empty()
        || sorted.iter().any(|&i| i >= 3 * mol.len())
        || sorted.windows(2).any(|w| w[0] == w[1])
    {
        return Err(DftError::InvalidInput("force-constant coordinates must be unique nonempty Cartesian IDs 3*atom+axis within the molecule".into()));
    }
    let patterns: Vec<_> = coordinates
        .iter()
        .map(|&i| {
            let mut path = vec![[0.0; 3]; mol.len()];
            path[i / 3][i % 3] = 1.0;
            path
        })
        .collect();
    let mut report = analytic_force_constants(
        mol, mb, functional, grid, scf, opts, &patterns, order, config,
    )?;
    apply_cartesian_diagnostics(&mut report, coordinates, mol.len(), config)?;
    Ok(report)
}

pub(crate) fn apply_cartesian_diagnostics(
    report: &mut ResponseReport<AnalyticForceConstants>,
    coordinates: &[usize],
    atoms: usize,
    config: &ResponseSolverConfig,
) -> Result<()> {
    report.units = "Hartree/Bohr^order; actual Cartesian cubic/quartic force constants";
    if let Some((max_residual, comparisons)) = cartesian_translation_residual(
        &report.value.tensor,
        coordinates,
        atoms,
        report.value.order,
    )? {
        report.value.translation_residual = Some(max_residual);
        report.value.translation_comparisons = comparisons;
        let scale = report
            .value
            .tensor
            .iter()
            .fold(1.0_f64, |m, v| m.max(v.abs()));
        if max_residual > config.tolerance * scale {
            report.converged = false;
            report.warnings.push("raw Cartesian force constants violate the acoustic translation sum rule at the requested tolerance".into());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn cartesian_force_constants(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<AnalyticForceConstants>> {
    cartesian_force_constant_block(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &(0..3 * mol.len()).collect::<Vec<_>>(),
        order,
        config,
    )
}

/// Raw acoustic sum over every leg and Cartesian axis; never projects a tensor.
/// A coordinate block can measure this only if every atomic coordinate is present.
pub(crate) fn cartesian_translation_residual(
    tensor: &[f64],
    coordinates: &[usize],
    atoms: usize,
    order: usize,
) -> Result<Option<(f64, usize)>> {
    let dimension = coordinates.len();
    let mut sorted = coordinates.to_vec();
    sorted.sort_unstable();
    if dimension == 0
        || !(3..=4).contains(&order)
        || dimension.checked_pow(order as u32) != Some(tensor.len())
        || sorted.iter().any(|&i| i >= 3 * atoms)
        || sorted.windows(2).any(|w| w[0] == w[1])
        || tensor.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "incompatible Cartesian force-constant translation inputs".into(),
        ));
    }
    if dimension != 3 * atoms {
        return Ok(None);
    }
    let mut lookup = vec![0; dimension];
    for (slot, &coordinate) in coordinates.iter().enumerate() {
        lookup[coordinate] = slot;
    }
    let mut max_residual = 0.0_f64;
    let mut comparisons = 0;
    for flat in 0..dimension.pow((order - 1) as u32) {
        let others = indices(flat, dimension, order - 1);
        for leg in 0..order {
            for axis in 0..3 {
                let mut sum = 0.0;
                for atom in 0..atoms {
                    let mut tuple = others.clone();
                    tuple.insert(leg, lookup[3 * atom + axis]);
                    let index = tuple.into_iter().fold(0, |ix, i| ix * dimension + i);
                    sum += tensor[index];
                }
                max_residual = max_residual.max(sum.abs());
                comparisons += 1;
            }
        }
    }
    Ok(Some((max_residual, comparisons)))
}
