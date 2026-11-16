// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Mixed magnetic energy derivatives from the common analytic field path.

use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::math::Vec3;
use crate::response::field_path::field_path_taylor;
use crate::response::mixed::polarized_mixed;
use crate::response::ResponseSolverConfig;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct MagneticMixedEnergy {
    pub derivative: f64,
    pub accumulation_roundoff: f64,
    pub directional_evaluations: usize,
    pub response_residual: f64,
    pub projector_defect: f64,
    pub stationarity_defect: f64,
    pub reference_defect: f64,
}

/// d^n E/(dt_1 ... dt_n), B=sum_j t_j*directions[j], n=1..4.
/// Parameters t_j are dimensionless. Repeated or non-orthogonal directions
/// are valid. This is an analytic polarization identity, not finite fields.
#[allow(clippy::too_many_arguments)]
pub fn mixed_magnetic_energy(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    origin: Vec3,
    london: bool,
    directions: &[[f64; 3]],
    config: &ResponseSolverConfig,
) -> Result<MagneticMixedEnergy> {
    mixed_electromagnetic_energy(
        molecule,
        basis,
        functional,
        grid,
        reference,
        options,
        origin,
        london,
        &vec![[0.0; 3]; directions.len()],
        directions,
        config,
    )
}

/// Actual mixed energy derivative for E=sum t_j e_j and B=sum t_j b_j.
/// Each parameter can contain electric and magnetic parts simultaneously.
/// The analytic polarization identity uses the same six-dimensional source
/// and moving-metric response; it never forms finite-field differences.
#[allow(clippy::too_many_arguments)]
pub fn mixed_electromagnetic_energy(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    origin: Vec3,
    london: bool,
    electric_directions: &[[f64; 3]],
    directions: &[[f64; 3]],
    config: &ResponseSolverConfig,
) -> Result<MagneticMixedEnergy> {
    if directions.is_empty() || directions.len() > 4 {
        return Err(DftError::InvalidInput(
            "mixed magnetic energy needs one through four field directions".into(),
        ));
    }
    if electric_directions.len() != directions.len() {
        return Err(DftError::InvalidInput(
            "electric and magnetic mixed direction lists must have equal lengths".into(),
        ));
    }
    let vectors: Vec<Vec<f64>> = electric_directions
        .iter()
        .zip(directions)
        .map(|(e, b)| e.iter().chain(b).copied().collect())
        .collect();
    let mut response_residual = 0.0_f64;
    let mut projector_defect = 0.0_f64;
    let mut stationarity_defect = 0.0_f64;
    let mut reference_defect = 0.0_f64;
    let mixed = polarized_mixed(6, &vectors, |direction, n| {
        let path = field_path_taylor(
            molecule,
            basis,
            functional,
            grid,
            reference,
            options,
            [direction[0], direction[1], direction[2]],
            [direction[3], direction[4], direction[5]],
            origin,
            london,
            n,
            config,
        )?;
        response_residual = path
            .residuals
            .iter()
            .copied()
            .fold(response_residual, f64::max);
        projector_defect = path
            .projector_defects
            .iter()
            .copied()
            .fold(projector_defect, f64::max);
        stationarity_defect = path
            .stationarity_defects
            .iter()
            .copied()
            .fold(stationarity_defect, f64::max);
        reference_defect = reference_defect.max(path.reference_defect);
        Ok(vec![path.energy[n]])
    })?;
    Ok(MagneticMixedEnergy {
        derivative: mixed.value[0],
        accumulation_roundoff: mixed.accumulation_roundoff[0],
        directional_evaluations: mixed.directional_evaluations,
        response_residual,
        projector_defect,
        stationarity_defect,
        reference_defect,
    })
}
