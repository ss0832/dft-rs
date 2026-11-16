// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common molecular static electric response on every public surface.

use super::controls::Controls;
use super::method::{run_molecular_reference, MolecularReference, MolecularSetup};
use super::molecular::{reference_report, setup, units};
use super::{System, Value};
use crate::response::ensemble::FixedPopulationDiagnostics;
use crate::response::{ResponseReport, StaticElectricTensor};
use crate::{DftError, Result};

fn population(d: &FixedPopulationDiagnostics) -> Value {
    Value::object()
        .with("reference_population_defect", d.reference_population_defect)
        .with("reference_density_defect", d.reference_density_defect)
        .with("population_defects", d.population_defects.clone())
        .with("orthogonality_defects", d.orthogonality_defects.clone())
        .with(
            "relative_stationarity_defects",
            d.relative_stationarity_defects.clone(),
        )
}

pub(super) fn metadata<T>(
    s: &MolecularSetup,
    r: &MolecularReference,
    response: &ResponseReport<T>,
) -> Value {
    let d = &response.reference;
    let mut report = reference_report(s, r, response.model)
        .with("shape", response.shape.clone())
        .with("functional", response.functional.clone())
        .with("axis_convention", response.axis_convention)
        .with("response_units", response.units)
        .with("converged", response.converged && d.converged)
        .with("residuals", response.residuals.clone())
        .with("iterations", response.iterations.clone())
        .with(
            "reference_field",
            s.options.external_field.map_or([0.0; 3], |v| v.to_array()),
        )
        .with(
            "reference",
            Value::object()
                .with("converged", d.converged)
                .with("iterations", d.iterations)
                .with("orbital_residual", d.orbital_residual)
                .with("eigen_residual", d.eigen_residual)
                .with("orthonormality_defect", d.orthonormality_defect)
                .with("projector_defect", d.projector_defect)
                .with("particle_number_error", d.particle_number_error),
        )
        .with("provenance", "analytic");
    let mut warnings = report
        .get("warnings")
        .and_then(Value::as_array)
        .unwrap_or(&[])
        .to_vec();
    warnings.extend(response.warnings.iter().cloned().map(Value::Str));
    report.set("warnings", Value::Array(warnings));
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            ("reference_field", "hartree/e/bohr"),
            ("residuals", "dimensionless"),
            ("reference.orbital_residual", "hartree"),
            ("reference.eigen_residual", "dimensionless"),
            ("reference.orthonormality_defect", "dimensionless"),
            ("reference.projector_defect", "dimensionless"),
            ("reference.particle_number_error", "electrons"),
        ]),
    );
    report
}

fn labels(report: &mut Value, extra: &[(&str, &str)]) {
    let mut u = report.get("units").cloned().unwrap_or_else(Value::object);
    for (key, label) in extra {
        u.set(key, *label);
    }
    report.set("units", u);
}

pub(super) fn tensor(
    s: &MolecularSetup,
    r: &MolecularReference,
    response: ResponseReport<StaticElectricTensor>,
    assembly: &str,
) -> Value {
    let mut report = metadata(s, r, &response);
    let t = response.value;
    report.set("order", t.order);
    report.set("tensor", t.tensor);
    report.set("assembly", assembly);
    report.set("accumulation_roundoff", t.accumulation_roundoff);
    report.set(
        "intrinsic_permutation_defect",
        t.intrinsic_permutation_defect,
    );
    report.set(
        "intrinsic_permutation_comparisons",
        t.intrinsic_permutation_comparisons,
    );
    report.set("energy_permutation_defect", t.energy_permutation_defect);
    report.set(
        "energy_permutation_comparisons",
        t.energy_permutation_comparisons,
    );
    report.set("frobenius_norm", t.frobenius_norm);
    report.set("beta_vector", t.beta_vector);
    report.set("beta_vector_norm", t.beta_vector_norm);
    report.set("gamma_isotropic", t.gamma_isotropic);
    report.set("ordered_field_evaluations", t.ordered_field_evaluations);
    report.set("directional_evaluations", t.directional_evaluations);
    report.set("projector_defect", t.projector_defect);
    report.set("projector_constraint", t.projector_constraint);
    report.set("fixed_population_defect", t.fixed_population_defect);
    if let Some(defect) = t.fermi_density_defect {
        report.set("fermi_density_defect", defect);
        labels(&mut report, &[("fermi_density_defect", "dimensionless")]);
    }
    report.set("stationarity_defect", t.stationarity_defect);
    report.set("particle_number_defect", t.particle_number_defect);
    labels(
        &mut report,
        &[
            ("tensor", "e^(order+1)*bohr^(order+1)/hartree^order"),
            (
                "accumulation_roundoff",
                "e^(order+1)*bohr^(order+1)/hartree^order",
            ),
            (
                "intrinsic_permutation_defect",
                "e^(order+1)*bohr^(order+1)/hartree^order",
            ),
            (
                "energy_permutation_defect",
                "e^(order+1)*bohr^(order+1)/hartree^order",
            ),
            ("frobenius_norm", "e^(order+1)*bohr^(order+1)/hartree^order"),
            ("beta_vector", "e^3*bohr^3/hartree^2"),
            ("beta_vector_norm", "e^3*bohr^3/hartree^2"),
            ("gamma_isotropic", "e^4*bohr^4/hartree^3"),
            ("projector_defect", "atomic units of density coefficients"),
            (
                "fixed_population_defect",
                "atomic units of density coefficients",
            ),
            ("stationarity_defect", "hartree"),
            ("particle_number_defect", "electrons"),
        ],
    );
    report
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    if super::double_hybrid::requested(controls)? {
        if matches!(id, "beta" | "gamma") {
            return super::correlated_high_order::electric(id, system, controls);
        }
        return Err(DftError::Unsupported {
            feature: "correlated molecular dipole-series export".into(),
            missing: "the dipole-series/mixed export of the shared correlated energy provider; static beta/gamma use its full energy tensors"
                .into(),
        });
    }
    let config = super::vibrational::response_config(controls)?;
    let order = if id == "field-taylor" {
        controls.usize("response_order", 3)?
    } else {
        0
    };
    let direction = if id == "field-taylor" {
        controls
            .opt_vec3("field_direction")?
            .map_or([0.0, 0.0, 1.0], |v| v.to_array())
    } else {
        [0.0; 3]
    };
    let directions = if id == "field-mixed" {
        let rows = controls.opt_rows("field_directions")?.ok_or_else(|| {
            DftError::InvalidInput("field-mixed requires field_directions".into())
        })?;
        if rows.is_empty() || rows.iter().any(|v| v.len() != 3) {
            return Err(DftError::InvalidInput(
                "field_directions must be nonempty Cartesian triples".into(),
            ));
        }
        rows.iter().map(|v| [v[0], v[1], v[2]]).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    if controls.contains("tensor_assembly") && controls.contains("electric_assembly") {
        return Err(DftError::InvalidInput(
            "use either tensor_assembly or electric_assembly".into(),
        ));
    }
    let assembly = if matches!(id, "beta" | "gamma") {
        if controls.contains("electric_assembly") {
            controls.str("electric_assembly", "projector")?
        } else {
            controls.str("tensor_assembly", "projector")?
        }
    } else {
        "projector".into()
    };
    if !matches!(assembly.as_str(), "projector" | "variational" | "energy") {
        return Err(DftError::InvalidInput(
            "unknown tensor_assembly; recognized selectors are projector, variational and energy"
                .into(),
        ));
    }
    if assembly == "energy" || id == "gamma" && assembly == "variational" {
        return Err(DftError::Unsupported {
            feature: format!("{id} with tensor_assembly={assembly}"),
            missing: "this assembly on the selected ordinary reference; use projector, or variational for ordinary beta".into(),
        });
    }
    let s = setup(system, controls)?;
    controls.finish(id)?;
    let r = run_molecular_reference(&s)?;
    match id {
        "field-taylor" => {
            let response = crate::response::electric_field_taylor(
                &s.molecule,
                &s.basis,
                &s.functional,
                s.grid.as_ref(),
                &r.scf,
                &s.options,
                direction,
                order,
                &config,
            )?;
            let mut report = metadata(&s, &r, &response);
            let v = response.value;
            report.set("direction", direction);
            report.set("response_order", order);
            report.set("dipole_coefficients", v.dipole_coefficients);
            report.set("projector_defects", v.projector_defects);
            report.set(
                "projector_constraint",
                v.fixed_population.is_none() && v.thermal_population.is_none(),
            );
            report.set(
                "fixed_population",
                v.fixed_population.as_ref().map(population),
            );
            report.set("stationarity_defects", v.stationarity_defects);
            report.set("particle_number_derivatives", v.particle_number_derivatives);
            if let Some(thermal) = v.thermal_population {
                report.set("chemical_potential_coefficients", thermal.chemical_potentials);
                report.set("fermi_density_defects", thermal.density_defects);
                labels(
                    &mut report,
                    &[
                        ("chemical_potential_coefficients", "hartree"),
                        ("fermi_density_defects", "dimensionless"),
                    ],
                );
            }
            labels(
                &mut report,
                &[
                    ("direction", "hartree/e/bohr"),
                    ("dipole_coefficients", "e*bohr"),
                    ("projector_defects", "atomic units of density coefficients"),
                    ("fixed_population", "atomic units of density coefficients"),
                    ("stationarity_defects", "hartree"),
                    ("particle_number_derivatives", "electrons"),
                ],
            );
            Ok(report)
        }
        "field-mixed" => {
            let response = crate::response::mixed_electric_field_response(
                &s.molecule,
                &s.basis,
                &s.functional,
                s.grid.as_ref(),
                &r.scf,
                &s.options,
                &directions,
                &config,
            )?;
            let mut report = metadata(&s, &r, &response);
            let v = response.value;
            report.set("directions", v.directions);
            report.set("dipole_derivative", v.dipole_derivative);
            report.set("accumulation_roundoff", v.accumulation_roundoff);
            report.set("directional_evaluations", v.directional_evaluations);
            report.set("fixed_population_defect", v.fixed_population_defect);
            labels(
                &mut report,
                &[
                    ("directions", "hartree/e/bohr"),
                    ("dipole_derivative", "e*bohr"),
                    ("accumulation_roundoff", "e*bohr"),
                    (
                        "fixed_population_defect",
                        "atomic units of density coefficients",
                    ),
                ],
            );
            Ok(report)
        }
        "beta" | "gamma" => {
            let response = if assembly == "variational" {
                crate::response::static_beta_variational(
                    &s.molecule,
                    &s.basis,
                    &s.functional,
                    s.grid.as_ref(),
                    &r.scf,
                    &s.options,
                    &config,
                )?
            } else {
                crate::response::static_electric_tensor(
                    &s.molecule,
                    &s.basis,
                    &s.functional,
                    s.grid.as_ref(),
                    &r.scf,
                    &s.options,
                    if id == "beta" { 2 } else { 3 },
                    &config,
                )?
            };
            let assembly = if r.scf.electronic_temperature > 0.0 {
                "canonical-fermi"
            } else {
                &assembly
            };
            Ok(tensor(&s, &r, response, assembly))
        }
        _ => Err(DftError::InvalidInput(format!(
            "unknown static electric request `{id}`"
        ))),
    }
}
