// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Reference-independent harmonic and RRHO postprocessing of analytic Hessians.
//! The existing common driver supplies the constrained electronic response.

use super::{Controls, System, Value};
use crate::constants::{atomic_mass_au, AMU_TO_AU_MASS};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::thermochemistry::{thermochemistry_from_mermin, ThermochemistryOptions};
use crate::properties::vibrational::vibrational_analysis_with_projection;
use crate::system::{Atom, Molecule};
use crate::{DftError, Result};

const POSTPROCESSING_CONTROLS: &[&str] = &[
    "temperature",
    "pressure",
    "symmetry_number",
    "electronic_degeneracy",
    "vibrational_projection",
];

fn matrix(report: &Value, key: &str, rows: usize, cols: usize) -> Result<Matrix> {
    let (shape, values) = report
        .get(key)
        .and_then(Value::numeric_array)
        .ok_or_else(|| {
            DftError::InvalidInput(format!("the reference report has no numeric {key}"))
        })?;
    if shape != [rows, cols] || values.iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(format!(
            "the reference {key} must be finite with shape [{rows}, {cols}]"
        )));
    }
    Ok(Matrix::from_row_major(rows, cols, values))
}

fn laboratory_source(controls: &Controls) -> Result<bool> {
    for key in ["field", "reference_field", "bfield"] {
        if controls
            .opt_vec3(key)?
            .is_some_and(|v| v.to_array().iter().any(|&x| x != 0.0))
        {
            return Ok(true);
        }
    }
    for key in ["gks_operator_real", "gks_operator_imaginary"] {
        if controls
            .opt_rows(key)?
            .is_some_and(|m| m.iter().flatten().any(|&x| x != 0.0))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let te = controls.f64("electronic_temperature", 0.0)?;
    let temperature = controls.f64("temperature", if te > 0.0 { te } else { 298.15 })?;
    let pressure = controls.f64("pressure", 101_325.0)?;
    let symmetry_number = controls.usize("symmetry_number", 1)?;
    let reference = controls.str("reference", "auto")?.to_ascii_lowercase();
    if matches!(reference.as_str(), "roks" | "rohf" | "gks" | "ghf")
        && super::double_hybrid::requested(controls)?
    {
        return Err(DftError::Unsupported {
            feature: "constrained-reference double-hybrid vibrational frequencies".into(),
            missing: "the correlated amplitude and orbital nuclear response for this reference"
                .into(),
        });
    }
    let multiplicity = controls.usize("multiplicity", 1)?;
    let degeneracy = controls.usize(
        "electronic_degeneracy",
        if matches!(reference.as_str(), "gks" | "ghf") {
            1
        } else {
            multiplicity
        },
    )?;
    if !temperature.is_finite()
        || temperature < 0.0
        || !pressure.is_finite()
        || pressure <= 0.0
        || symmetry_number == 0
        || degeneracy == 0
    {
        return Err(DftError::InvalidInput(
            "harmonic analysis requires nonnegative finite temperature, positive finite pressure and positive symmetry number/electronic degeneracy".into(),
        ));
    }
    let has_source = laboratory_source(controls)?;
    let projection = controls.str("vibrational_projection", "auto")?;
    let project = match projection.as_str() {
        "auto" => !has_source,
        "eckart" => true,
        "cartesian" => false,
        _ => {
            return Err(DftError::InvalidInput(
                "vibrational_projection must be auto, eckart or cartesian".into(),
            ))
        }
    };
    let mut child = controls.to_value();
    if let Value::Object(entries) = &mut child {
        entries.retain(|(key, _)| !POSTPROCESSING_CONTROLS.contains(&key.as_str()));
    }
    let mut report = super::molecular::run("hessian", system, &Controls::from_value(&child)?)?;
    append(
        id,
        system,
        &mut report,
        project,
        has_source,
        te,
        ThermochemistryOptions {
            temperature_kelvin: temperature,
            pressure_pascal: pressure,
            symmetry_number,
            electronic_degeneracy: degeneracy,
        },
    )?;
    Ok(report)
}

fn append(
    id: &str,
    system: &System,
    report: &mut Value,
    project: bool,
    has_source: bool,
    electronic_temperature: f64,
    options: ThermochemistryOptions,
) -> Result<()> {
    let molecule = Molecule::new(
        system
            .numbers
            .iter()
            .zip(&system.positions)
            .map(|(&z, &r)| Atom {
                z,
                position: Vec3::new(r[0], r[1], r[2]),
            })
            .collect(),
    );
    let dim = 3 * molecule.len();
    let hessian = matrix(report, "hessian", dim, dim)?;
    let apt = match report.get("dipole_derivatives") {
        Some(Value::Null) | None => None,
        Some(_) => Some(matrix(report, "dipole_derivatives", 3, dim)?),
    };
    let modes = vibrational_analysis_with_projection(&molecule, &hessian, apt.as_ref(), project)?;
    let patterns = (0..modes.modes.cols)
        .map(|k| {
            molecule
                .atoms
                .iter()
                .enumerate()
                .map(|(a, atom)| {
                    let mass = (atomic_mass_au(atom.z) / AMU_TO_AU_MASS).sqrt();
                    std::array::from_fn::<_, 3, _>(|axis| modes.modes[(3 * a + axis, k)] / mass)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut warnings = match report.get("warnings").cloned() {
        Some(Value::Array(values)) => values,
        _ => Vec::new(),
    };
    let reason = if has_source {
        Some("ideal-gas RRHO is withheld for a fixed laboratory field or additional AO operator")
    } else if !project {
        Some("ideal-gas RRHO requires Eckart-projected internal vibrations")
    } else if options.temperature_kelvin == 0.0 {
        Some("ideal-gas RRHO thermochemistry requires a positive temperature")
    } else if modes.frequencies_cm.iter().any(|&f| f <= 0.0) {
        Some("ideal-gas RRHO requires strictly positive internal vibrational frequencies")
    } else if report.get("converged").and_then(Value::as_bool) != Some(true) {
        Some("ideal-gas RRHO requires a converged reference and nuclear response")
    } else {
        None
    };
    if let Some(reason) = reason {
        if id == "gibbs-free-energy" {
            return Err(DftError::InvalidInput(reason.into()));
        }
        warnings.push(Value::Str(reason.into()));
        report.set("thermochemistry", Value::Null);
    } else {
        let energy = report
            .get("energy")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                DftError::InvalidInput(
                    "the reference report has no finite electronic energy".into(),
                )
            })?;
        let entropy = if electronic_temperature > 0.0 {
            let term = report
                .get("electronic_entropy_term")
                .and_then(Value::as_f64)
                .ok_or_else(|| DftError::Unsupported {
                    feature: "finite-temperature constrained-reference Gibbs thermochemistry"
                        .into(),
                    missing: "the reference electronic entropy term".into(),
                })?;
            term / (crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K * electronic_temperature)
        } else {
            0.0
        };
        let thermo = thermochemistry_from_mermin(
            &molecule,
            &modes.frequencies_cm,
            energy,
            electronic_temperature,
            entropy,
            options,
        )?;
        report.set("gibbs_free_energy", thermo.gibbs_free_energy);
        report.set(
            "free_energy_kind",
            "gibbs; ideal-gas RRHO, including pV = k_B T per molecule",
        );
        report.set(
            "thermochemistry",
            super::vibrational::thermochemistry_value(&thermo),
        );
    }
    report.set("frequencies_cm1", modes.frequencies_cm);
    report.set("normal_modes_cartesian", patterns);
    if apt.is_some() {
        report.set("ir_intensities_km_mol", modes.ir_intensities);
    } else {
        report.set("ir_intensities_km_mol", Value::Null);
        warnings.push(Value::Str("IR intensities require an atomic polar tensor, which is not available in this reference report".into()));
    }
    report.set("n_projected", modes.n_projected);
    report.set("n_imaginary", modes.n_imaginary);
    report.set(
        "vibrational_projection",
        if project { "eckart" } else { "cartesian" },
    );
    report.set("electronic_degeneracy", options.electronic_degeneracy);
    report.set("vibrational_model", "harmonic analysis of the same analytic reference Hessian; Eckart projection for a free molecule or the complete unprojected Cartesian spectrum; unavailable atomic polar tensors give null IR intensities; fixed-spinor electronic degeneracy is supplied explicitly, default one");
    report.set("warnings", warnings);
    let mut units = report.get("units").cloned().unwrap_or_else(Value::object);
    for (key, label) in [
        ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
        ("normal_modes_cartesian", "1/sqrt(amu)"),
        ("ir_intensities_km_mol", "km/mol"),
        ("n_projected", "integer"),
        ("n_imaginary", "integer"),
        ("electronic_degeneracy", "integer"),
        ("gibbs_free_energy", "hartree per molecule"),
    ] {
        units.set(key, label);
    }
    for (key, label) in super::vibrational::thermochemistry_units() {
        units.set(key, label);
    }
    report.set("units", units);
    Ok(())
}
