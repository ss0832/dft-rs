// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit candidate generation and saved-model evaluation. Model JSON is
//! opaque native-unit text: report-unit conversion must not rewrite its arrays.

use super::reduced::evaluate_candidate;
use crate::driver::{Controls, System, Value};
use crate::math::Vec3;
use crate::pseudo_sub::reduction::{
    generate_methyl, GenerationOptions, ProjectionRule, ReducedFragment,
};
use crate::pseudo_sub::reduction_basis::{OneCenterBasisTemplate, H1S_TEMPLATE_ID};
use crate::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec};
use crate::system::{Atom, Molecule};
use crate::{DftError, Result};

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

pub(super) fn object_control(controls: &Controls, name: &str) -> Result<Value> {
    match controls.take(name) {
        Some(Value::Str(text)) => {
            let value = Value::parse_json(text)?;
            if !matches!(value, Value::Object(_)) {
                return invalid(&format!("{name} must contain a JSON object"));
            }
            Ok(value)
        }
        Some(value @ Value::Object(_)) => Ok(value.clone()),
        _ => invalid(&format!("{name} requires an object or JSON object text")),
    }
}

// The canonical loader validates values/dimensions. Also refuse surplus or
// duplicate saved fields recursively instead of silently ignoring metadata.
fn strict_model_keys(input: &Value, canonical: &Value) -> Result<()> {
    strict_model_keys_at(input, canonical, "")
}

fn strict_model_keys_at(input: &Value, canonical: &Value, path: &str) -> Result<()> {
    match (input, canonical) {
        (Value::Object(fields), Value::Object(expected)) => {
            // Only the original default rule may be absent in old saved
            // options. This exception does not apply to other object paths.
            let legacy_rule = usize::from(
                path == "options"
                    && input.get("projection_rule").is_none()
                    && canonical.get("projection_rule").and_then(Value::as_str)
                        == Some("full-ao-metric-v1"),
            );
            if fields.len() + legacy_rule != expected.len() {
                return invalid("saved fragment model has unknown, duplicate or missing fields");
            }
            for (name, value) in fields {
                if fields.iter().filter(|(key, _)| key == name).count() != 1 {
                    return invalid("saved fragment model has duplicate fields");
                }
                let Some((_, reference)) = expected.iter().find(|(key, _)| key == name) else {
                    return invalid("saved fragment model has unknown fields");
                };
                let child_path = if path.is_empty() {
                    name.clone()
                } else {
                    format!("{path}.{name}")
                };
                strict_model_keys_at(value, reference, &child_path)?;
            }
        }
        (Value::Array(items), Value::Array(expected)) => {
            for (value, reference) in items.iter().zip(expected) {
                strict_model_keys_at(value, reference, &format!("{path}[]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn indices(value: Option<&Value>, name: &str) -> Result<Vec<usize>> {
    let Some(Value::Array(items)) = value else {
        return invalid(&format!("fragment_spec.{name} requires an integer array"));
    };
    items
        .iter()
        .map(|v| {
            v.as_i64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| {
                    DftError::InvalidInput(format!(
                        "fragment_spec.{name} requires nonnegative indices"
                    ))
                })
        })
        .collect()
}

pub(super) fn fragment_spec(value: &Value) -> Result<FragmentSpec> {
    let spec = Controls::from_value(value)?;
    let explicit_atoms = indices(spec.take("explicit_atoms"), "explicit_atoms")?;
    let retained_atoms = indices(spec.take("retained_atoms"), "retained_atoms")?;
    let port = indices(spec.take("port"), "port")?;
    if port.len() != 3 {
        return invalid("fragment_spec.port requires [fragment_atom,host_atom,reference_atom]");
    }
    let Some(Value::Array(bonds)) = spec.take("single_bonds") else {
        return invalid("fragment_spec.single_bonds requires a list of index pairs");
    };
    let single_bonds = bonds
        .iter()
        .map(|v| {
            let pair = indices(Some(v), "single_bonds")?;
            pair.try_into().map_err(|_| {
                DftError::InvalidInput("fragment_spec.single_bonds requires pairs".into())
            })
        })
        .collect::<Result<Vec<[usize; 2]>>>()?;
    let electronic_centers = spec.usize("electronic_centers", 1)?;
    spec.finish("continuous-species fragment_spec")?;
    Ok(FragmentSpec {
        explicit_atoms,
        retained_atoms,
        port: AttachmentPort {
            fragment_atom: port[0],
            host_atom: port[1],
            reference_atom: port[2],
        },
        single_bonds,
        electronic_centers,
    })
}

fn sector(controls: &Controls, default_charge: i32) -> Result<i32> {
    let charge = controls
        .opt_i64("charge")?
        .unwrap_or(i64::from(default_charge));
    let charge = i32::try_from(charge).map_err(|_| {
        DftError::InvalidInput("fragment charge must fit a signed 32-bit integer".into())
    })?;
    if controls.usize("multiplicity", 1)? != 1 {
        return invalid(
            "the saved static methyl template supports only closed-shell multiplicity one",
        );
    }
    Ok(charge)
}

fn positions(system: &System) -> Result<Vec<Vec3>> {
    if system.cell.is_some() || system.pbc.iter().any(|&p| p) {
        return invalid("saved reduced-fragment models do not yet support a cell or periodic axes");
    }
    if system.numbers.is_empty()
        || system.numbers.len() != system.positions.len()
        || system.numbers.iter().any(|&z| z == 0 || z > 118)
        || system.positions.iter().flatten().any(|x| !x.is_finite())
    {
        return invalid(
            "fragment calculations require matching real-element labels and finite Bohr positions",
        );
    }
    Ok(system
        .positions
        .iter()
        .map(|r| Vec3::new(r[0], r[1], r[2]))
        .collect())
}

fn counts(model: &ReducedFragment) -> Value {
    Value::object()
        .with("full_electrons", model.electron_ledger.full_electron_count)
        .with(
            "frozen_electrons",
            model.electron_ledger.frozen_electron_count,
        )
        .with(
            "active_electrons",
            model.electron_ledger.active_electron_count,
        )
        .with("full_ao_count", model.diagnostics.full_ao_count)
        .with("reduced_ao_count", model.diagnostics.reduced_ao_count)
        .with("full_atoms", model.source.len())
        .with("reduced_atoms", model.spec.retained_atoms.len() + 1)
        .with("electronic_centers", 1_usize)
        .with("projector_count", 8_usize)
}

fn report(model: &ReducedFragment, mode: &str) -> Value {
    Value::object()
        .with("feature", "continuous-species")
        .with("model", "fixed generated static methyl candidate")
        .with("fragment_mode", mode)
        .with("candidate_status", model.candidate_status())
        .with("shape_status", model.shape_status())
        .with("response_status", model.response_status())
        .with("model_version", model.model_version.clone())
        .with("template_id", model.template_id.clone())
        .with("fingerprint", model.construction_fingerprint.clone())
        .with("counts", counts(model))
        .with("warnings", vec!["Candidate operator projection; chemical accuracy and internal polarization are not established."])
}

/// Alternate mode inside the existing continuous-species common driver.
/// The common resource envelope consumes CPU/memory controls before dispatch;
/// direct callers establish a ResourceLimits scope explicitly if required.
pub fn run(system: &System, controls: &Controls) -> Result<Value> {
    let coordinates = positions(system)?;
    match controls.str("fragment_mode", "")?.as_str() {
        "few-evaluate" => super::few_center_driver::run(system, controls),
        "rigid-evaluate" => super::rigid_pose_driver::run(system, controls),
        "scalar-evaluate"
        | "scalar-optimize"
        | "scalar-gradient"
        | "scalar-geometry-optimize"
        | "scalar-hessian" => super::scalar::run(system, controls),
        "generate" => {
            let spec = fragment_spec(&object_control(controls, "fragment_spec")?)?;
            let charge = sector(controls, 0)?;
            let mut options = GenerationOptions {
                projection_rule: ProjectionRule::from_id(
                    &controls.str("fragment_projection_rule", "full-ao-metric-v1")?,
                )?,
                basis_template: OneCenterBasisTemplate::from_id(
                    &controls.str("fragment_basis_template", H1S_TEMPLATE_ID)?,
                )?,
                functional: controls.str("xc", "pbe")?.to_ascii_lowercase(),
                grid_level: controls.usize("grid_level", 1)?,
                electrostatic_weight: controls.f64("fragment_esp_weight", 1.0)?,
                ..GenerationOptions::default()
            };
            if let Some(rows) = controls.opt_rows("fragment_esp_points")? {
                for row in rows {
                    if row.len() != 3 {
                        return invalid("fragment_esp_points requires an N by 3 array in Bohr");
                    }
                    options
                        .electrostatic_points
                        .push(Vec3::new(row[0], row[1], row[2]));
                }
            }
            controls.finish("continuous-species")?;
            let source = Molecule::new(
                system
                    .numbers
                    .iter()
                    .zip(coordinates)
                    .map(|(&z, position)| Atom { z, position })
                    .collect(),
            )
            .with_charge(charge)
            .with_multiplicity(1);
            let model = generate_methyl(&source, spec, options)?;
            let d = &model.diagnostics;
            Ok(report(&model, "generate")
                .with("model_json", model.to_json()?)
                .with("converged", true)
                .with("generation_status", "generated-candidate-not-chemical-acceptance")
                .with("reference_energy", d.full_reference_energy)
                .with("reference_reconstruction_error", d.reference_energy_reconstruction_error)
                .with("operator_projection_residual_squared", d.electronic_projection.weighted_residual_squared)
                .with("electrostatic_projection_residual_squared", d.sigma_projection.weighted_residual_squared)
                .with("sigma_charge_defect", d.sigma_charge_defect)
                .with("sigma_dipole_defect", d.sigma_dipole_defect)
                .with("operator_data_rank", d.electronic_projection.data_rank)
                .with("electrostatic_data_rank", d.sigma_projection.data_rank)
                .with("active_density_trace", d.active_trace)
                .with("frozen_density_trace", d.frozen_trace)
                .with("units", Value::object()
                    .with("model_json", "text opaque native-unit model")
                    .with("reference_energy", "hartree")
                    .with("reference_reconstruction_error", "hartree")
                    .with("operator_projection_residual_squared", "hartree^2")
                    .with("electrostatic_projection_residual_squared", "atomic units weighted charge/dipole/electrostatic-potential projection objective")
                    .with("sigma_charge_defect", "e")
                    .with("sigma_dipole_defect", "e*bohr")
                    .with("active_density_trace", "dimensionless")
                    .with("frozen_density_trace", "dimensionless")))
        }
        "evaluate" => {
            let saved = object_control(controls, "fragment_model_json")?;
            let model = ReducedFragment::from_value(&saved)?;
            strict_model_keys(&saved, &model.to_value()?)?;
            let charge = sector(controls, model.source.charge)?;
            if charge != model.source.charge {
                return invalid("evaluation charge differs from the saved model electron sector");
            }
            let labels: Vec<_> = model
                .spec
                .retained_atoms
                .iter()
                .map(|&i| model.source.atoms[i].z)
                .chain([1])
                .collect();
            if system.numbers != labels {
                return invalid("reduced System labels must exactly match saved retained atoms followed by the H-like center label 1");
            }
            let field = controls.opt_vec3("field")?.unwrap_or_else(Vec3::zero);
            let origin = controls
                .opt_vec3("charge_origin")?
                .unwrap_or_else(Vec3::zero);
            controls.finish("continuous-species")?;
            let value = evaluate_candidate(&model, &coordinates, field, origin)?;
            Ok(report(&model, "evaluate")
                .with("converged", true)
                .with("energy", value.energy)
                .with("electronic_energy", value.electronic_energy)
                .with("classical_energy", value.classical_energy)
                .with(
                    "energy_without_internal_reference",
                    value.energy_without_internal_reference,
                )
                .with(
                    "classical_energy_without_internal_reference",
                    value.classical_energy_without_internal_reference,
                )
                .with("fragment_internal_energy", value.fragment_internal_energy)
                .with(
                    "scf_energy_convention",
                    super::FRAGMENT_SCF_ENERGY_CONVENTION,
                )
                .with("gradient", value.gradient)
                .with("forces", value.forces)
                .with("dipole", value.model_dipole)
                .with("permanent_ionic_dipole", value.permanent_ionic_dipole)
                .with("active_electronic_dipole", value.active_electronic_dipole)
                .with("physical_total_charge", value.physical_total_charge)
                .with("charge_origin", origin)
                .with("field", field)
                .with("active_density_trace", value.active_density_trace)
                .with("scf_iterations", value.scf_iterations)
                .with("scf_orbital_residual", value.scf_orbital_residual)
                .with("frontier_gap", value.frontier_gap)
                .with("grid_points", value.grid_points)
                .with("occupied_spatial_orbitals", value.occupied_spatial_orbitals)
                .with(
                    "derivative_status",
                    "analytic-fixed-generated-model-rigid-expansion",
                )
                .with(
                    "units",
                    Value::object()
                        .with("energy", "hartree")
                        .with("electronic_energy", "hartree")
                        .with("classical_energy", "hartree")
                        .with("energy_without_internal_reference", "hartree")
                        .with("classical_energy_without_internal_reference", "hartree")
                        .with("fragment_internal_energy", "hartree")
                        .with("gradient", "hartree/bohr")
                        .with("forces", "hartree/bohr")
                        .with("dipole", "e*bohr")
                        .with("permanent_ionic_dipole", "e*bohr")
                        .with("active_electronic_dipole", "e*bohr")
                        .with("physical_total_charge", "e")
                        .with("charge_origin", "bohr")
                        .with("field", "atomic units electric field")
                        .with("active_density_trace", "dimensionless")
                        .with("scf_orbital_residual", "hartree")
                        .with("frontier_gap", "hartree"),
                ))
        }
        _ => invalid("fragment_mode must be generate, evaluate, few-evaluate or rigid-evaluate"),
    }
}
