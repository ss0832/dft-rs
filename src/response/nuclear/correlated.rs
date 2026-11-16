// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Correlated molecular nuclear paths reusing complete parent projector and
//! metric transport, moving Gaussian integrals and noncanonical PT2 equations.
use super::{matrix_product, molecular_taylor_parts, MolecularDirection};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::properties::pt2_response::{
    projector_frames, taylor_response, transform_path, transform_path_retained,
};
use crate::response::field_path::FieldEnergyModel;
use crate::response::ResponseSolverConfig;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Factorial-normalized nuclear coefficients. Individual spin channels are
/// unscaled; `correlation` and `energy` contain the supplied SS/OS weights.
pub struct MolecularCorrelatedTaylor {
    /// Complete AO orbital Taylor frames, including metric transport and
    /// within-subspace response, for wavefunction observables such as VCD.
    pub orbitals: [Vec<Matrix>; 2],
    /// Retained low parts, also used by the AO-to-MO PT2 contractions.
    pub orbital_corrections: [Vec<Matrix>; 2],
    pub occupied: [usize; 2],
    /// Unscaled [order][SS-alpha,OS,SS-beta][i,a,j,b] amplitudes.
    pub amplitudes: Vec<[Vec<f64>; 3]>,
    pub energy: Vec<f64>,
    pub parent: Vec<f64>,
    pub target: Vec<f64>,
    pub parent_nonlocal: Vec<f64>,
    pub target_nonlocal: Vec<f64>,
    pub same_spin_alpha: Vec<f64>,
    pub opposite_spin: Vec<f64>,
    pub same_spin_beta: Vec<f64>,
    pub correlation: Vec<f64>,
    pub response_residuals: Vec<f64>,
    pub amplitude_residuals: Vec<f64>,
    /// Fresh physical amplitude-equation residuals normalized by the source.
    pub amplitude_relative_residuals: Vec<f64>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub metric_defects: Vec<f64>,
    pub number_defects: Vec<[f64; 2]>,
    pub orbital_frame_defect: f64,
    pub reference_energy_defect: f64,
    pub reference_fock_defect: f64,
    pub coefficient_equations: Option<Vec<super::CoefficientEquationDiagnostics>>,
}

/// Full parent orbital response, all AO integral endpoints and PT2 amplitude
/// response through fourth order. A post-SCF target is nonstationary in the
/// parent orbitals, so no parent 2n+1 truncation is applied to its density.
pub fn correlated_molecular_taylor(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<&ResolvedFunctional>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    direction: &MolecularDirection,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<MolecularCorrelatedTaylor> {
    let target = target
        .map(|f| {
            let model = FieldEnergyModel::new(f);
            f.required_vv10()
                .map_or(Ok(model.clone()), |p| model.with_vv10(1.0, p))
        })
        .transpose()?;
    correlated_molecular_taylor_for_model(
        molecule,
        basis,
        parent,
        target,
        grid,
        reference,
        options,
        coefficients,
        direction,
        order,
        config,
    )
}

/// Explicit target energy composition shared with the complex-field driver.
/// Every auxiliary term is evaluated on the fully responding parent density.
#[allow(clippy::too_many_arguments)]
pub fn correlated_molecular_taylor_for_model(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    direction: &MolecularDirection,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<MolecularCorrelatedTaylor> {
    coefficients.validate()?;
    if order > 4
        || reference.electronic_temperature != 0.0
        || reference.fixed_occupation_residual.is_some()
    {
        return Err(DftError::InvalidInput("correlated nuclear paths require integer zero-temperature populations and order zero through four".into()));
    }
    if target.as_ref().is_some_and(|model| {
        model
            .functional()
            .required_nonlocal_kind()
            .is_some_and(|kind| {
                kind != crate::xc::RequiredNonlocalTerm::Vv10 || model.nonlocal().is_none()
            })
    }) {
        return Err(DftError::Unsupported {
            feature: "the requested correlated nuclear target composition".into(),
            missing: "an explicit moving provider for the target nonlocal term".into(),
        });
    }
    let (report, path) = molecular_taylor_parts(
        molecule, basis, parent, grid, reference, options, direction, order, config,
    )?;
    let mut frame_defect = 0.0_f64;
    let retained_frames = path.retained.as_ref().map(|value| {
        std::array::from_fn::<_, 2, _>(|s| {
            let (frame, error) = super::precision::frames(&value.d[s], path.n_occ[s]);
            frame_defect = frame_defect.max(error);
            frame
        })
    });
    let frames: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        if let Some(frame) = &retained_frames {
            return super::precision::high(&frame[s]);
        }
        let (frame, error) = projector_frames(&path.d[s], path.n_occ[s]);
        frame_defect = frame_defect.max(error);
        frame
    });
    let retained_orbitals =
        path.retained
            .as_ref()
            .zip(retained_frames.as_ref())
            .map(|(value, frames)| {
                std::array::from_fn::<_, 2, _>(|s| {
                    super::precision::product(&value.b[s], &frames[s], order)
                })
            });
    let orbitals = std::array::from_fn(|s| match &retained_orbitals {
        Some(value) => super::precision::high(&value[s]),
        None => matrix_product(&path.b[s], &frames[s], order),
    });
    let integrals = path.geometry.coulomb_coefficients();
    let occupied = path.n_occ;
    let transformed = if let Some(orbitals) = &retained_orbitals {
        transform_path_retained(
            basis.nao_sph,
            integrals,
            orbitals,
            &reference.mo_energy,
            occupied,
        )?
    } else {
        match order {
            0 => transform_path::<1>(
                basis.nao_sph,
                integrals,
                &orbitals,
                &reference.mo_energy,
                occupied,
            )?,
            1 => transform_path::<2>(
                basis.nao_sph,
                integrals,
                &orbitals,
                &reference.mo_energy,
                occupied,
            )?,
            2 => transform_path::<3>(
                basis.nao_sph,
                integrals,
                &orbitals,
                &reference.mo_energy,
                occupied,
            )?,
            3 => transform_path::<4>(
                basis.nao_sph,
                integrals,
                &orbitals,
                &reference.mo_energy,
                occupied,
            )?,
            4 => transform_path::<5>(
                basis.nao_sph,
                integrals,
                &orbitals,
                &reference.mo_energy,
                occupied,
            )?,
            _ => unreachable!(),
        }
    };
    let moving: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        if let Some((value, frames)) = path.retained.as_ref().zip(retained_frames.as_ref()) {
            return super::precision::high(&super::precision::congruence(
                &frames[s],
                &value.f[s],
                order,
            ));
        }
        let transpose: Vec<_> = frames[s].iter().map(Matrix::transpose).collect();
        matrix_product(
            &matrix_product(&transpose, &path.f[s], order),
            &frames[s],
            order,
        )
    });
    let fock: Vec<_> = (1..=order)
        .map(|k| [moving[0][k].clone(), moving[1][k].clone()])
        .collect();
    let changes: Vec<_> = transformed.iter().skip(1).map(|p| p.ovov.clone()).collect();
    let pair = taylor_response(&transformed[0], &changes, &fock, coefficients)?;
    let parent_nonlocal: Vec<_> = report
        .value
        .energy_components
        .iter()
        .map(|e| e.nonlocal_correlation)
        .collect();
    let (target_energy, target_nonlocal) = if let Some(target) = target {
        let functional = target.functional();
        let mut geometry = super::geometry::GeometryTaylor::new(
            molecule,
            basis,
            grid,
            functional,
            target.learned(),
            &direction.nuclear,
            options.external_field.map_or([0.0; 3], |v| v.to_array()),
            direction.electric,
            order,
            options.incore_mb,
        )?;
        if let Some((coefficient, parameters)) = target.nonlocal() {
            let grid = options.nonlocal_grid(molecule)?;
            geometry.attach_molecular_nonlocal(
                molecule,
                basis,
                &grid,
                parameters,
                coefficient,
                &direction.nuclear,
                order,
                options.incore_mb,
            )?;
        }
        let evaluation = match &path.retained {
            Some(value) => geometry.evaluate_retained(functional, &value.p, order)?.0,
            None => geometry.evaluate(functional, &path.p, order)?,
        };
        (
            evaluation.energy.iter().map(|e| e.total()).collect(),
            evaluation
                .energy
                .iter()
                .map(|e| e.nonlocal_correlation)
                .collect(),
        )
    } else {
        (path.energy.clone(), parent_nonlocal.clone())
    };
    Ok(MolecularCorrelatedTaylor {
        orbital_corrections: std::array::from_fn(|s| match &retained_orbitals {
            Some(value) => super::precision::low(&value[s]),
            None => orbitals[s]
                .iter()
                .map(|c| Matrix::zeros(c.rows, c.cols))
                .collect(),
        }),
        orbitals,
        occupied,
        amplitudes: pair.amplitude_coefficients,
        energy: target_energy
            .iter()
            .zip(&pair.correlation)
            .map(|(e, c)| e + c)
            .collect(),
        parent: path.energy,
        target: target_energy,
        parent_nonlocal,
        target_nonlocal,
        same_spin_alpha: pair.same_spin_alpha,
        opposite_spin: pair.opposite_spin,
        same_spin_beta: pair.same_spin_beta,
        correlation: pair.correlation,
        response_residuals: report.residuals,
        coefficient_equations: report.value.coefficient_equations,
        amplitude_residuals: pair.residuals,
        amplitude_relative_residuals: pair.relative_residuals,
        projector_defects: report.value.projector_defects,
        stationarity_defects: report.value.stationarity_defects,
        metric_defects: report.value.metric_defects,
        number_defects: report.value.particle_number_coefficients,
        orbital_frame_defect: frame_defect,
        reference_energy_defect: report.value.reference_energy_error,
        reference_fock_defect: report.value.reference_fock_error,
    })
}
