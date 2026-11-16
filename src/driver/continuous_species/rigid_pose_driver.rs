// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit static primitive center/Euler mode of the common driver.
//! Global logging, units and resource envelopes remain owned by driver::run.

use super::fragment_driver::object_control;
use super::rigid_pose;
use super::rigid_pose_record::{RigidPoseRecord, RECORD_VERSION};
use super::rigid_pose_response;
use crate::driver::method::{prepare_molecular, MolecularMethod};
use crate::driver::{Controls, System, Value};
use crate::linalg::Matrix;
use crate::{DftError, Result};

pub const MODE: &str = "rigid-evaluate";

fn unsupported<T>(message: &str) -> Result<T> {
    Err(DftError::Unsupported {
        feature: message.into(),
        missing: "an implemented and independently checked connection for this static primitive pose mode".into(),
    })
}

fn rows(values: &[f64]) -> Vec<Vec<f64>> {
    values.chunks_exact(3).map(<[f64]>::to_vec).collect()
}

fn block(
    matrix: &Matrix,
    first: std::ops::Range<usize>,
    second: std::ops::Range<usize>,
) -> Vec<Vec<f64>> {
    first
        .map(|i| second.clone().map(|j| matrix[(i, j)]).collect())
        .collect()
}

pub fn run(system: &System, controls: &Controls) -> Result<Value> {
    if system.is_periodic() {
        return unsupported("periodic static primitive pose derivatives");
    }
    // Refuse a known unsupported reference before any SCF setup can replace it.
    if let Some(reference) = controls.peek_str("reference")? {
        match reference.to_ascii_lowercase().as_str() {
            "auto" | "rks" | "rhf" => {}
            "uks" | "uhf" | "roks" | "rohf" | "gks" | "ghf" => {
                return unsupported("non-restricted static primitive pose reference")
            }
            _ => {
                return Err(DftError::InvalidInput(
                    "unknown static primitive pose reference".into(),
                ))
            }
        }
    }
    let _ = controls.opt_str("reference")?;
    if controls.str("fragment_mode", "")? != MODE {
        return Err(DftError::InvalidInput(
            "static primitive pose mode requires fragment_mode=rigid-evaluate".into(),
        ));
    }
    let record = RigidPoseRecord::from_value(&object_control(controls, "fragment_model_json")?)?;
    let order = controls.usize("derivative_order", 1)?;
    if order > 2 {
        return unsupported("static primitive pose derivatives above order two");
    }
    let mut method = MolecularMethod::from_controls(controls)?;
    // These defaults belong to this new mode only, not to existing references.
    if !controls.contains("xc") && !controls.contains("x") && !controls.contains("c") {
        method.xc = Some("pbe".into());
    }
    if !controls.contains("grid_level") {
        method.grid_level = 1;
    }
    if !controls.contains("grid_weight_cutoff") {
        method.grid_weight_cutoff = 0.;
    }
    if !controls.contains("max_iter") {
        method.max_iter = 300;
    }
    if !controls.contains("conv_tol") {
        method.conv_tol = 1e-13;
    }
    if !controls.contains("conv_tol_dp") {
        method.conv_tol_dp = 1e-12;
    }
    if !controls.contains("conv_tol_grad") {
        method.conv_tol_grad = Some(1e-11);
    }
    if !controls.contains("guess") {
        method.guess = Some("core".into());
    }
    if !method
        .xc
        .as_deref()
        .is_some_and(|name| name.eq_ignore_ascii_case("pbe"))
        || method.x.is_some()
        || method.c.is_some()
        || !method.xc_parameters.is_empty()
        || method.omega.is_some()
    {
        return unsupported(
            "this public primitive pose recipe currently exposes the bounded PBE reference",
        );
    }
    if method.unrestricted
        || method.multiplicity != 1
        || method.ecp.is_some()
        || method.density_fit.is_some()
        || method.skala_model.is_some()
        || method.solvent.is_some()
        || method.eps.is_some()
        || method.field.is_some_and(|v| v.norm() != 0.)
        || method.electronic_temperature != 0.
        || method.electron_offset != [0.; 2]
        || method.occupation.is_some()
        || method.fixed_occupations.is_some()
        || method.stabilize
        || method.grid_weight_cutoff != 0.
        || !matches!(
            method.dispersion.to_ascii_lowercase().as_str(),
            "none" | "auto"
        )
    {
        return unsupported("static primitive pose environment, population constraint or pruned quadrature combination");
    }
    let response_config = if order == 2 {
        Some(crate::driver::vibrational::response_config(controls)?)
    } else {
        None
    };
    controls.finish("continuous-species rigid-evaluate")?;
    let mut setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    setup.basis = record.build_basis(&setup.molecule, &setup.basis_set)?;
    if order == 2 {
        rigid_pose_response::preflight_storage(
            setup.molecule.len(),
            record.fragments().len(),
            setup.basis.nao_sph,
        )?;
    }
    let count = 3 * (setup.molecule.len() + record.fragments().len());
    let zero = vec![0.; count];
    let extra = rigid_pose::pair(
        &setup.molecule,
        &setup.basis,
        record.fragments(),
        &zero,
        &zero,
    )?;
    setup.options.extra_hcore = Some(extra.operator[0].clone());
    let grid = setup.grid.as_ref().ok_or_else(|| {
        DftError::InvalidInput("static primitive PBE requires its molecular quadrature".into())
    })?;
    let state = crate::scf::run_scf(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        Some(grid),
    )?;
    let diagnostics = crate::response::diagnostics::reference_diagnostics(&state)?;
    if !state.converged || diagnostics.orbital_residual > setup.options.conv_tol_grad {
        return Err(DftError::ScfNotConverged {
            iterations: state.iterations,
            error: diagnostics.orbital_residual,
        });
    }
    let classical = state.e_nuc + extra.classical_correction[0] + extra.internal_energy_reference;
    let canonical_record = record.to_value();
    let core_templates = canonical_record
        .get("fragments")
        .and_then(Value::as_array)
        .expect("validated primitive fragments")
        .iter()
        .map(|fragment| {
            fragment
                .get("basis_template")
                .expect("validated primitive template")
                .clone()
        })
        .collect::<Vec<_>>();
    let mut report = Value::object()
        .with("feature", "continuous-species").with("fragment_mode", MODE)
        .with("model_version", RECORD_VERSION).with("experimental", true)
        .with("chemical_accuracy", "not_established").with("reference_type", "rks")
        .with("application_functional", setup.functional.description.clone())
        .with("application_basis", method.basis.clone())
        .with("application_core_templates", Value::Array(core_templates))
        .with("generation", record.to_value().get("generation").expect("validated generation provenance").clone())
        .with("coordinate_convention", "atom-major retained Cartesian centers, then fragment-major Rz(gamma)Ry(beta)Rx(alpha) Euler angles")
        .with("derivative_order", order).with("converged", true)
        .with("energy", state.e_elec + classical).with("electronic_energy", state.e_elec)
        .with("classical_energy", classical).with("internal_energy_reference", extra.internal_energy_reference)
        .with("active_electrons", record.ledger().active_electrons)
        .with("frozen_electrons", record.ledger().frozen_electrons)
        .with("full_electrons", record.ledger().full_electrons)
        .with("ao_count", setup.basis.nao_sph).with("scf_iterations", state.iterations)
        .with("scf_orbital_residual", diagnostics.orbital_residual)
        .with("particle_number_error", diagnostics.particle_number_error)
        .with("internal_coordinates", "frozen; no internal C-H deformation")
        .with("warnings", vec!["Experimental static recipe. Functional/basis transfer and chemical accuracy are not established. E0 is already included once in classical_energy."]);
    let mut units = Value::object()
        .with("energy", "hartree")
        .with("electronic_energy", "hartree")
        .with("classical_energy", "hartree")
        .with("internal_energy_reference", "hartree")
        .with("scf_orbital_residual", "hartree")
        .with("particle_number_error", "dimensionless");
    if order > 0 {
        let center = 3 * setup.molecule.len();
        let (gradient, hessian) = if let Some(config) = response_config {
            let value = rigid_pose_response::hessian(
                &setup.molecule,
                &setup.basis,
                record.fragments(),
                &setup.functional,
                grid,
                &state,
                &setup.options,
                &config,
            )?;
            let residuals: Vec<f64> = value
                .directional
                .iter()
                .flat_map(|r| r.response.residuals.iter().copied())
                .collect();
            report.set("response_residuals", residuals);
            report.set("response_paths", value.directional.len());
            report.set("tested_derivative_scope", "Static PBE two-center, two-active-electron, 10-AO fixture: all144 entries have independent four-step witnesses.");
            report.set("derivative_validation_limit", "Two ordered gradient-difference checks remain unresolved; independent conditioned mixed-energy witnesses passed. This is not per-input chemical validation.");
            units.set("response_residuals", "dimensionless");
            (value.gradient, Some(value.hessian))
        } else {
            let value = rigid_pose::assemble_gradient(
                &setup.molecule,
                &setup.basis,
                record.fragments(),
                &setup.functional,
                grid,
                &state,
                &setup.options,
            )?;
            (value.gradient, None)
        };
        report.set("center_gradient", rows(&gradient[..center]));
        report.set(
            "forces",
            rows(&gradient[..center])
                .into_iter()
                .map(|r| r.into_iter().map(|x| -x).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        );
        report.set("euler_gradient", rows(&gradient[center..]));
        report.set(
            "derivative_method",
            if order == 2 {
                "analytic-moving-basis-grid-static-operators-and-coupled-orbital-response"
            } else {
                "analytic-stationary-gradient-moving-basis-grid-and-static-operators"
            },
        );
        units.set("center_gradient", "hartree/bohr");
        units.set("forces", "hartree/bohr");
        units.set("euler_gradient", "hartree/radian");
        if let Some(hessian) = hessian {
            report.set("center_hessian", block(&hessian, 0..center, 0..center));
            report.set(
                "center_angle_hessian",
                block(&hessian, 0..center, center..count),
            );
            report.set(
                "angle_hessian",
                block(&hessian, center..count, center..count),
            );
            units.set("center_hessian", "hartree/bohr^2");
            units.set("center_angle_hessian", "hartree/bohr/radian");
            units.set("angle_hessian", "hartree/radian^2");
        }
    }
    report.set("units", units);
    Ok(report)
}
