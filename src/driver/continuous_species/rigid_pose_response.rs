// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Orbital-relaxed analytic Hessian of the static center/Euler chart.
//! The common retained moving-basis/grid solver supplies electronic response.
//! This adapter supplies the complete extra operator and classical ledger.
//! No displaced SCF, finite-difference production derivative or optimization
//! is used. It is separate from a Hessian of the fitted model coefficients.

use super::rigid_pose::{self, RigidFragment};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{AccurateSum, Matrix};
use crate::response::nuclear::extra_core::{
    molecular_taylor_with_extra_core, second_order_operator_coefficients, ExternalClassicalTaylor,
    ExtraCoreCoordinates, ExtraCoreFieldConvention, ExtraCoreIdentity, ExtraCoreTaylor,
};
use crate::response::nuclear::{MolecularDirection, MolecularTaylor};
use crate::response::{ResponseReport, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};
use rayon::prelude::*;

pub struct RigidPoseDirectional {
    /// E(t)=E0+t*E1+t^2*E2; E2 is half the actual directional curvature.
    pub energy_coefficients: [f64; 3],
    pub response: ResponseReport<MolecularTaylor>,
    pub source_fingerprint: u64,
    /// Already included once in energy_coefficients, not an additive result.
    pub external_classical_coefficients: Vec<f64>,
    pub internal_energy_reference: f64,
}

pub struct RigidPoseHessian {
    /// Cartesian centers in bohr followed by Euler angles in radians.
    pub gradient: Vec<f64>,
    pub hessian: Matrix,
    pub total_energy: f64,
    /// Unit directions followed by pair sums. Ordered finite-difference
    /// verification must still check every Hessian element independently.
    pub directional: Vec<RigidPoseDirectional>,
    // Matrix data are reserved by Matrix itself. Keep the retained report
    // headers charged for the lifetime of this result as well.
    _report_headers: crate::resources::MemoryReservation,
}

/// Check a lower bound on retained tensor/report storage before a costly SCF.
/// Runtime matrix/workspace reservations remain authoritative. This is not
/// a process RSS limit and does not assume a particular quadrature size.
pub fn preflight_storage(atoms: usize, fragments: usize, nao: usize) -> Result<usize> {
    if atoms == 0 || fragments == 0 || nao == 0 {
        return Err(DftError::InvalidInput(
            "rigid-pose storage requires nonempty center, fragment and AO sets".into(),
        ));
    }
    let n = atoms
        .checked_add(fragments)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| DftError::InvalidInput("rigid-pose coordinate count overflow".into()))?;
    let paths = n
        .checked_add(1)
        .and_then(|v| v.checked_mul(n))
        .map(|v| v / 2)
        .ok_or_else(|| DftError::InvalidInput("rigid-pose direction count overflow".into()))?;
    // Three orders of two spin densities and their low parts, plus overlap:
    // fifteen nao^2 matrices are retained even if the MO frame loses rank.
    let matrices = crate::resources::checked_bytes(
        &[paths, 15, nao, nao, std::mem::size_of::<f64>()],
        "rigid-pose retained response lower bound",
    )?;
    let tensor = crate::resources::checked_bytes(
        &[n, n, std::mem::size_of::<f64>()],
        "rigid-pose Hessian storage",
    )?;
    let headers = crate::resources::checked_bytes(
        &[paths, std::mem::size_of::<RigidPoseDirectional>()],
        "rigid-pose retained report headers",
    )?;
    let bytes = matrices
        .checked_add(tensor)
        .and_then(|v| v.checked_add(headers))
        .ok_or_else(|| DftError::InvalidInput("rigid-pose retained storage overflow".into()))?;
    let _reservation =
        crate::resources::reserve(bytes, "rigid-pose retained response lower bound")?;
    Ok(n)
}

#[allow(clippy::too_many_arguments)]
pub fn directional(
    molecule: &Molecule,
    basis: &MolBasis,
    fragments: &[RigidFragment],
    functional: &ResolvedFunctional,
    grid: &MolecularGrid,
    reference: &ScfResult,
    options: &ScfOptions,
    direction: &[f64],
    config: &ResponseSolverConfig,
) -> Result<RigidPoseDirectional> {
    config.validate()?;
    let pair = rigid_pose::pair(molecule, basis, fragments, direction, direction)?;
    let n = molecule.len();
    let nuclear = MolecularDirection::nuclear(
        direction[..3 * n]
            .chunks_exact(3)
            .map(|d| [d[0], d[1], d[2]])
            .collect(),
    );
    let angles = &direction[3 * n..];
    let coefficients = second_order_operator_coefficients(
        pair.operator[0].clone(),
        pair.operator[1].clone(),
        pair.operator[3].clone(),
    )?;
    let classical = ExternalClassicalTaylor {
        interaction: vec![
            pair.classical_correction[0],
            pair.classical_correction[1],
            0.5 * pair.classical_correction[3],
        ],
        internal_energy_reference: pair.internal_energy_reference,
        permanent_dipole: vec![
            std::array::from_fn(|i| pair.permanent_dipole[i][0]),
            std::array::from_fn(|i| pair.permanent_dipole[i][1]),
            std::array::from_fn(|i| 0.5 * pair.permanent_dipole[i][3]),
        ],
    };
    // The leaf reference stores the bare point-ion scalar. The bound response
    // context stores the COMPLETE classical scalar, including E0 exactly once.
    // Electronic density, orbitals, Fock and electronic energy are unchanged.
    let mut complete_reference = reference.clone();
    let scalar = pair.classical_correction[0] + pair.internal_energy_reference;
    complete_reference.energy += scalar;
    complete_reference.e_nuc += scalar;
    let mut identity = super::Fingerprint::new();
    identity.string("primitive-eight-projector-rigid-pose-response-v1");
    identity.bytes(&pair.coordinate_map.to_le_bytes());
    for fragment in fragments {
        identity.number(fragment.atom);
        for &x in fragment
            .parameters
            .charge_exponents
            .iter()
            .chain(&fragment.parameters.charge_coefficients)
            .chain(&fragment.parameters.projector_s_exponents)
            .chain(&fragment.parameters.projector_p_exponents)
            .chain(&fragment.parameters.nonlocal_coefficients)
            .chain(&fragment.local_exponents)
            .chain(&fragment.local_amplitudes)
        {
            identity.float(x);
        }
        for p in fragment.parameters.body_dipoles {
            for x in p.to_array() {
                identity.float(x);
            }
        }
        identity.float(fragment.internal_energy_reference);
    }
    let identity = ExtraCoreIdentity {
        chart: identity.0,
        operator_layout: identity.0,
        grid_recipe: identity.0,
        coordinate_map: pair.coordinate_map,
    };
    // Centers follow independent straight Cartesian paths in GeometryTaylor.
    // Euler-angle curvature belongs solely to the supplied operator source.
    // No nuclear-coordinate lifting or second frame pullback occurs here.
    let source = ExtraCoreTaylor::new(
        molecule,
        basis,
        functional,
        Some(grid),
        &complete_reference,
        options,
        &nuclear,
        angles,
        identity.clone(),
        ExtraCoreCoordinates::IndependentRoles,
        [coefficients.clone(), coefficients],
        classical,
        [0.; 3],
        ExtraCoreFieldConvention::ProvidedCompleteDeltaFromOrdinaryOriginZero,
        2,
    )?;
    let geometry = source.geometry(
        molecule,
        basis,
        functional,
        Some(grid),
        &complete_reference,
        options,
        &nuclear,
        angles,
        &identity,
    )?;
    let report = molecular_taylor_with_extra_core(
        molecule,
        basis,
        functional,
        Some(grid),
        &complete_reference,
        options,
        &nuclear,
        angles,
        &identity,
        &source,
        2,
        config,
        geometry,
    )?;
    if !report.response.converged || report.response.value.energy_coefficients.len() != 3 {
        return Err(DftError::ResponseNotConverged {
            iterations: report.response.iterations.iter().sum(),
            residual: report.response.residuals.iter().copied().fold(0., f64::max),
        });
    }
    if report.coordinates != ExtraCoreCoordinates::IndependentRoles {
        return Err(DftError::InvalidInput(
            "rigid-pose response unexpectedly changed coordinate convention".into(),
        ));
    }
    Ok(RigidPoseDirectional {
        energy_coefficients: std::array::from_fn(|i| report.response.value.energy_coefficients[i]),
        source_fingerprint: report.source_fingerprint,
        external_classical_coefficients: report.external_classical_coefficients,
        internal_energy_reference: report.internal_energy_reference,
        response: report.response,
    })
}

/// Complete center/Euler Hessian, including center/orientation mixed blocks.
/// Polarization reconstructs a symmetric tensor from an unisolvent set of
/// analytic paths. It does not symmetrize a nonsymmetric numerical Hessian.
#[allow(clippy::too_many_arguments)]
pub fn hessian(
    molecule: &Molecule,
    basis: &MolBasis,
    fragments: &[RigidFragment],
    functional: &ResolvedFunctional,
    grid: &MolecularGrid,
    reference: &ScfResult,
    options: &ScfOptions,
    config: &ResponseSolverConfig,
) -> Result<RigidPoseHessian> {
    preflight_storage(molecule.len(), fragments.len(), basis.nao_sph)?;
    let first = rigid_pose::assemble_gradient(
        molecule, basis, fragments, functional, grid, reference, options,
    )?;
    let n = first.gradient.len();
    let pair_count = n
        .checked_mul(n.saturating_sub(1))
        .map(|v| v / 2)
        .ok_or_else(|| DftError::InvalidInput("rigid-pose direction count overflow".into()))?;
    let count = n
        .checked_add(pair_count)
        .ok_or_else(|| DftError::InvalidInput("rigid-pose report count overflow".into()))?;
    let report_headers = crate::resources::reserve(
        crate::resources::checked_bytes(
            &[count, std::mem::size_of::<RigidPoseDirectional>()],
            "rigid-pose report headers",
        )?,
        "rigid-pose report headers",
    )?;
    let mut tensor = Matrix::zeros(n, n);
    let mut reports = Vec::new();
    reports.try_reserve_exact(count).map_err(|_| {
        DftError::InvalidInput("cannot allocate rigid-pose response reports".into())
    })?;
    let mut second = Vec::with_capacity(n);
    // Independent paths use the existing request pool. Its worker context
    // carries the same memory ledger and run-local logger; no pool or global
    // setting is created here. Indexed collection preserves report order.
    let unit_reports: Result<Vec<_>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut d = vec![0.; n];
            d[i] = 1.;
            crate::logger::log(&format!("[rigid-pose] unit direction {i} started"));
            let result = directional(
                molecule, basis, fragments, functional, grid, reference, options, &d, config,
            );
            match &result {
                Ok(_) => crate::logger::log(&format!("[rigid-pose] unit direction {i} completed")),
                Err(error) => {
                    crate::logger::warn(&format!("[rigid-pose] unit direction {i} failed: {error}"))
                }
            }
            result
        })
        .collect();
    for (i, report) in unit_reports?.into_iter().enumerate() {
        if (report.energy_coefficients[1] - first.gradient[i]).abs() > config.acceptance() {
            return Err(DftError::InvalidInput("rigid-pose Taylor first coefficient disagrees with the independent complete analytic gradient".into()));
        }
        second.push(report.energy_coefficients[2]);
        tensor[(i, i)] = 2. * report.energy_coefficients[2];
        reports.push(report);
    }
    let _pair_headers = crate::resources::reserve(
        crate::resources::checked_bytes(
            &[pair_count, std::mem::size_of::<(usize, usize)>()],
            "rigid-pose direction pairs",
        )?,
        "rigid-pose direction pairs",
    )?;
    let mut pairs = Vec::new();
    pairs
        .try_reserve_exact(pair_count)
        .map_err(|_| DftError::InvalidInput("cannot allocate rigid-pose direction pairs".into()))?;
    for i in 0..n {
        for j in 0..i {
            pairs.push((i, j));
        }
    }
    let mixed_reports: Result<Vec<_>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut d = vec![0.; n];
            d[i] = 1.;
            d[j] = 1.;
            crate::logger::log(&format!("[rigid-pose] mixed direction {i},{j} started"));
            let result = directional(
                molecule, basis, fragments, functional, grid, reference, options, &d, config,
            );
            match &result {
                Ok(_) => {
                    crate::logger::log(&format!("[rigid-pose] mixed direction {i},{j} completed"))
                }
                Err(error) => crate::logger::warn(&format!(
                    "[rigid-pose] mixed direction {i},{j} failed: {error}"
                )),
            }
            result
        })
        .collect();
    for ((i, j), report) in pairs.into_iter().zip(mixed_reports?) {
        let mut sum = AccurateSum::default();
        sum.add_product(report.energy_coefficients[2], 1.);
        sum.add_product(second[i], -1.);
        sum.add_product(second[j], -1.);
        tensor[(i, j)] = sum.value();
        tensor[(j, i)] = sum.value();
        reports.push(report);
    }
    Ok(RigidPoseHessian {
        gradient: first.gradient,
        hessian: tensor,
        total_energy: first.total_energy,
        directional: reports,
        _report_headers: report_headers,
    })
}

/// Exact second-order pullback from the center/Euler chart to a declared
/// anchor chart: J^T H J + sum_i g_i * d2x_i. The second-map tensors are
/// mandatory, including for a nonlinear local frame; a first-only force
/// pullback is insufficient for a nuclear Hessian away from stationarity.
pub fn pullback(
    gradient: &[f64],
    hessian: &Matrix,
    jacobian: &Matrix,
    map_hessians: &[Matrix],
) -> Result<(Vec<f64>, Matrix)> {
    let n = gradient.len();
    let m = jacobian.cols;
    if n == 0
        || m == 0
        || hessian.rows != n
        || hessian.cols != n
        || jacobian.rows != n
        || map_hessians.len() != n
        || map_hessians.iter().any(|h| h.rows != m || h.cols != m)
        || gradient
            .iter()
            .chain(hessian.as_slice())
            .chain(jacobian.as_slice())
            .chain(map_hessians.iter().flat_map(|h| h.as_slice()))
            .any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput("second-order pose pullback requires finite complete gradient, Hessian, Jacobian and every map Hessian".into()));
    }
    let g = (0..m)
        .map(|a| (0..n).map(|i| jacobian[(i, a)] * gradient[i]).sum())
        .collect();
    let mut h = jacobian.transpose_matmul_seq(&hessian.matmul_seq(jacobian));
    for i in 0..n {
        for a in 0..m {
            for b in 0..m {
                h[(a, b)] += gradient[i] * map_hessians[i][(a, b)];
            }
        }
    }
    Ok((g, h))
}

#[cfg(test)]
mod storage_contract {
    use super::preflight_storage;

    #[test]
    fn dimensions_and_overflow_are_checked_without_allocating_a_tensor() {
        assert_eq!(preflight_storage(2, 2, 10).unwrap(), 12);
        assert!(preflight_storage(0, 1, 5).is_err());
        assert!(preflight_storage(1, 0, 5).is_err());
        assert!(preflight_storage(1, 1, 0).is_err());
        assert!(preflight_storage(usize::MAX, 1, 1).is_err());
        assert!(preflight_storage(1, 1, usize::MAX).is_err());
    }

    #[test]
    fn insufficient_managed_budget_is_rejected_before_any_scf() {
        let result = crate::resources::ResourceLimits::new(Some(1), Some(1))
            .unwrap()
            .run(|| preflight_storage(3, 3, 15));
        assert!(matches!(result, Err(crate::DftError::ResourceLimit { .. })));
    }

    #[test]
    fn successful_preflight_does_not_hold_a_second_matrix_reservation() {
        crate::resources::ResourceLimits::new(Some(1), Some(8))
            .unwrap()
            .run(|| {
                let available = crate::resources::remaining_bytes();
                assert_eq!(preflight_storage(1, 1, 5)?, 6);
                assert_eq!(crate::resources::remaining_bytes(), available);
                Ok(())
            })
            .unwrap();
    }
}
