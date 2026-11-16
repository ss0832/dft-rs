// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed common one-center AO alternatives, independent of electron accounting.
//! All templates keep the ionic center's H label and +1 ionic convention.
//! Carbon alternatives use fixed bundled contractions. Keeping a core-s basis
//! function does not unfreeze electrons or change any operator coefficient.

use crate::basis::{data, ElementBasis, MolBasis};
use crate::integrals::overlap_matrix;
use crate::linalg::{solve_linear_matrix, symmetric_eigen, Matrix};
use crate::system::Molecule;
use crate::{DftError, Result};

pub const ELECTRONIC_OPERATOR_COEFFICIENTS: usize = 39;
pub const ELECTROSTATIC_COEFFICIENTS: usize = 12;
pub const H1S_TEMPLATE_ID: &str = "one-center-h1s-local3-projector8-charge3-v1";
pub const VALENCE_SP_TEMPLATE_ID: &str = "one-center-valence-sp-local3-projector8-charge3-v1";
pub const CORE_VALENCE_SP_TEMPLATE_ID: &str =
    "one-center-core-valence-sp-local3-projector8-charge3-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneCenterBasisTemplate {
    Hydrogen1s,
    CarbonValenceSp,
    CarbonCoreValenceSp,
}

impl OneCenterBasisTemplate {
    pub fn id(self) -> &'static str {
        match self {
            Self::Hydrogen1s => H1S_TEMPLATE_ID,
            Self::CarbonValenceSp => VALENCE_SP_TEMPLATE_ID,
            Self::CarbonCoreValenceSp => CORE_VALENCE_SP_TEMPLATE_ID,
        }
    }

    pub fn from_id(id: &str) -> Result<Self> {
        match id {
            H1S_TEMPLATE_ID => Ok(Self::Hydrogen1s),
            VALENCE_SP_TEMPLATE_ID => Ok(Self::CarbonValenceSp),
            CORE_VALENCE_SP_TEMPLATE_ID => Ok(Self::CarbonCoreValenceSp),
            _ => invalid("unknown fixed one-center AO template"),
        }
    }

    pub fn site_ao_count(self) -> usize {
        match self {
            Self::Hydrogen1s => 1,
            Self::CarbonValenceSp => 4,
            Self::CarbonCoreValenceSp => 5,
        }
    }
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

/// A fixed raw contraction definition. Its z=1 is the retained role label,
/// not a request to change physical ionic charge or infer an electron count.
pub fn fixed_site_basis(template: OneCenterBasisTemplate) -> Result<ElementBasis> {
    let common = data::load("sto-3g")?;
    let shells = match template {
        OneCenterBasisTemplate::Hydrogen1s => {
            let h = common.element(1)?;
            if h.shells.len() != 1 || h.shells[0].l != 0 {
                return invalid("bundled H template no longer has the declared single s shell");
            }
            h.shells.clone()
        }
        OneCenterBasisTemplate::CarbonValenceSp | OneCenterBasisTemplate::CarbonCoreValenceSp => {
            let c = common.element(6)?;
            if c.shells.len() != 3
                || c.shells.iter().map(|s| s.l).collect::<Vec<_>>() != [0, 0, 1]
                || c.shells[1].exps != c.shells[2].exps
            {
                return invalid("bundled carbon template no longer has the declared core-s, valence-s, valence-p ordering");
            }
            if template == OneCenterBasisTemplate::CarbonCoreValenceSp {
                // A basis direction, not an additional occupied core or ion.
                // Frozen-electron accounting remains a separate contract.
                c.shells.clone()
            } else {
                vec![c.shells[1].clone(), c.shells[2].clone()]
            }
        }
    };
    Ok(ElementBasis { z: 1, shells })
}

/// Retained atoms use their ordinary bundled STO-3G basis in all alternatives.
/// No per-fragment width, contraction, AO count or host basis is fitted here.
pub fn build_reduction_basis(
    molecule: &Molecule,
    electronic_center: usize,
    template: OneCenterBasisTemplate,
) -> Result<MolBasis> {
    if electronic_center >= molecule.len() || molecule.atoms[electronic_center].z != 1 {
        return invalid("the fixed reduced electronic center must be an explicitly supplied H-labelled ionic role point");
    }
    let common = data::load("sto-3g")?;
    let basis = MolBasis::build_with_overrides(
        molecule,
        &common,
        &[(electronic_center, fixed_site_basis(template)?)],
    )?;
    let count: usize = basis
        .shells
        .iter()
        .filter(|s| s.atom == electronic_center)
        .map(|s| 2 * usize::from(s.l) + 1)
        .sum();
    if count != template.site_ao_count() {
        return invalid("one-center AO count differs from its fixed template identifier");
    }
    Ok(basis)
}

pub struct OverlapCondition {
    pub ao_count: usize,
    pub minimum_eigenvalue: f64,
    pub maximum_eigenvalue: f64,
    pub condition_number: f64,
}

pub fn overlap_condition(basis: &MolBasis, relative_tolerance: f64) -> Result<OverlapCondition> {
    if !relative_tolerance.is_finite()
        || relative_tolerance <= 0.0
        || relative_tolerance >= 1.0
        || basis.nao_sph == 0
    {
        return invalid(
            "overlap conditioning requires a nonempty basis and a positive relative threshold",
        );
    }
    let (values, _) = symmetric_eigen(&overlap_matrix(basis))?;
    let minimum = values[0];
    let maximum = values[values.len() - 1];
    if !minimum.is_finite() || !maximum.is_finite() || minimum <= relative_tolerance * maximum {
        return invalid(
            "the reduced AO overlap is nonpositive or rank-deficient at the declared threshold",
        );
    }
    Ok(OverlapCondition {
        ao_count: basis.nao_sph,
        minimum_eigenvalue: minimum,
        maximum_eigenvalue: maximum,
        condition_number: maximum / minimum,
    })
}

/// Reuse the common overlap engine on an AO union. This union is used only
/// for S, never as a molecule, electron-count adapter or SCF basis.
pub fn cross_basis_overlap(left: &MolBasis, right: &MolBasis) -> Result<Matrix> {
    let nsph = left
        .nao_sph
        .checked_add(right.nao_sph)
        .ok_or_else(|| DftError::InvalidInput("cross overlap AO count overflow".into()))?;
    let ncart = left
        .nao_cart
        .checked_add(right.nao_cart)
        .ok_or_else(|| DftError::InvalidInput("cross overlap Cartesian count overflow".into()))?;
    let mut shells = left.shells.clone();
    for shell in &right.shells {
        let mut placed = shell.clone();
        placed.sph_off += left.nao_sph;
        placed.cart_off += left.nao_cart;
        shells.push(placed);
    }
    let union = MolBasis {
        shells,
        nao_sph: nsph,
        nao_cart: ncart,
        max_l: left.max_l.max(right.max_l),
        ecps: Vec::new(),
        n_core_electrons: 0,
    };
    let overlap = overlap_matrix(&union);
    let mut cross = Matrix::zeros(left.nao_sph, right.nao_sph);
    for i in 0..left.nao_sph {
        for j in 0..right.nao_sph {
            cross[(i, j)] = overlap[(i, left.nao_sph + j)];
        }
    }
    Ok(cross)
}

pub struct OccupiedCapture {
    pub spatial_orbitals: usize,
    pub active_electrons: usize,
    /// Raw squared principal captures; no clipping or renormalization.
    pub squared_principal_captures: Vec<f64>,
    pub minimum_squared_capture: f64,
    pub captured_electrons: f64,
    pub uncaptured_electrons: f64,
    pub reference_metric_defect: f64,
    pub projected_metric: Matrix,
    /// Unrenormalized AO projections, not a valid determinant by assumption.
    pub projected_orbitals: Matrix,
    pub projection_relative_residual: f64,
    pub overlap: OverlapCondition,
}

/// Project a fixed metric-orthonormal closed-shell occupied subspace into a
/// reduced AO span. Traces count two electrons per spatial orbital. A loss
/// is reported rather than hidden by renormalizing the projected orbitals.
pub fn closed_shell_occupied_capture(
    reference_basis: &MolBasis,
    occupied: &Matrix,
    reduced_basis: &MolBasis,
    relative_tolerance: f64,
) -> Result<OccupiedCapture> {
    if occupied.rows != reference_basis.nao_sph
        || occupied.cols == 0
        || occupied.as_slice().iter().any(|v| !v.is_finite())
    {
        return invalid(
            "capture requires finite reference occupied columns in the complete reference AO order",
        );
    }
    let overlap = overlap_condition(reduced_basis, relative_tolerance)?;
    let reference_s = overlap_matrix(reference_basis);
    let metric = occupied.transpose_matmul_seq(&reference_s.matmul_seq(occupied));
    let reference_defect = metric.max_abs_diff(&Matrix::identity(occupied.cols));
    if reference_defect > 1e-9 {
        return invalid("reference occupied columns are not metric-orthonormal");
    }
    let cross = cross_basis_overlap(reduced_basis, reference_basis)?;
    let source = cross.matmul_seq(occupied);
    let reduced_s = overlap_matrix(reduced_basis);
    let projected = solve_linear_matrix(&reduced_s, &source)?;
    let mut residual = reduced_s.matmul_seq(&projected);
    for (v, &r) in residual.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *v -= r;
    }
    let source_norm = source.as_slice().iter().fold(0.0_f64, |n, &v| n.hypot(v));
    let residual_norm = residual.as_slice().iter().fold(0.0_f64, |n, &v| n.hypot(v));
    let projected_metric = source.transpose_matmul_seq(&projected);
    let (values, _) = symmetric_eigen(&projected_metric)?;
    if values
        .iter()
        .any(|v| !v.is_finite() || *v < -1e-9 || *v > 1.0 + 1e-9)
    {
        return invalid("occupied capture is outside physical projection bounds");
    }
    let electrons = occupied
        .cols
        .checked_mul(2)
        .ok_or_else(|| DftError::InvalidInput("active electron count overflow".into()))?;
    let captured = 2.0
        * (0..occupied.cols)
            .map(|i| projected_metric[(i, i)])
            .sum::<f64>();
    Ok(OccupiedCapture {
        spatial_orbitals: occupied.cols,
        active_electrons: electrons,
        minimum_squared_capture: values[0],
        squared_principal_captures: values,
        captured_electrons: captured,
        uncaptured_electrons: electrons as f64 - captured,
        reference_metric_defect: reference_defect,
        projected_metric,
        projected_orbitals: projected,
        projection_relative_residual: if source_norm == 0.0 {
            residual_norm
        } else {
            residual_norm / source_norm
        },
        overlap,
    })
}
