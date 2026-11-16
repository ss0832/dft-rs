// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Thin public adapter for the independently checked one-electron Pauli model.
//! Static g tensors retain their separate reference and gauge capabilities.

use super::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
#[cfg(test)]
use super::value::ToValue;
use super::{Controls, System, Value};
use crate::math::Vec3;
use crate::properties::g_nuclear::{
    PauliGCartesianDerivatives, PauliGDerivative, PauliGNuclearContext,
};
use crate::response::field_path::FieldPathTaylor;
use crate::{DftError, Result};

fn unsupported(missing: &str) -> DftError {
    DftError::Unsupported {
        feature: "g-tensor nuclear derivatives".into(),
        missing: missing.into(),
    }
}

/// Read this before static reference dispatch consumes a selector. Zero leaves
/// that dispatch unchanged, including its common-origin and RO/spinor routes.
pub(super) fn order(controls: &Controls) -> Result<usize> {
    match controls.take("nuclear_order") {
        None | Some(Value::Null) => Ok(0),
        Some(Value::Int(value)) if (0..=2).contains(value) => Ok(*value as usize),
        Some(Value::Int(value)) if *value > 2 => {
            Err(unsupported("validated nuclear orders above two"))
        }
        _ => Err(DftError::InvalidInput(
            "nuclear_order must be an integer from zero to two".into(),
        )),
    }
}

fn reference(controls: &Controls) -> Result<()> {
    match controls.take("reference") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Str(name)) => match name.to_ascii_lowercase().as_str() {
            "auto" | "uhf" | "uks" => Ok(()),
            "rhf" | "rks" | "rohf" | "roks" | "ghf" | "gks" => Err(unsupported(
                "validated nuclear response for the requested reference; static EPR support does not imply nuclear derivative support")),
            _ => Err(DftError::InvalidInput(format!("unknown reference `{name}`"))),
        },
        _ => Err(DftError::InvalidInput("reference must be a string selector".into())),
    }
}

fn validate_method(method: &MolecularMethod) -> Result<()> {
    if method.stabilize {
        return Err(unsupported(
            "nuclear response of a stability-followed or replaced reference",
        ));
    }
    if method
        .field
        .is_some_and(|v| v.to_array().iter().any(|&x| x != 0.0))
    {
        return Err(unsupported(
            "nuclear g response in a nonzero electric field",
        ));
    }
    if method.electronic_temperature != 0.0
        || method.electron_offset != [0.0; 2]
        || method.fixed_occupations.is_some()
        || method.occupation.is_some()
        || method.solvent.is_some()
        || method.eps.is_some()
        || method.density_fit.is_some()
    {
        return Err(unsupported(
            "validated ensemble, tracked-occupation, solvent or fitted-integral nuclear g response",
        ));
    }
    let functional = crate::xc::resolve(&method.functional_spec()?)?;
    if !functional.parts.is_empty()
        || functional.hybrid != crate::xc::HybridParams::HF
        || functional.required_nonlocal_term().is_some()
    {
        return Err(unsupported("accepted DFT nuclear g response; this increment uses collinear HF/UHF and a one-electron Pauli operator"));
    }
    Ok(())
}

fn workspace_bytes(atoms: usize, order: usize) -> Result<usize> {
    let n = atoms
        .checked_mul(3)
        .ok_or_else(|| DftError::InvalidInput("nuclear g coordinate count overflow".into()))?;
    let paths = n
        .checked_mul(
            n.checked_add(1)
                .ok_or_else(|| DftError::InvalidInput("nuclear g path count overflow".into()))?,
        )
        .and_then(|v| v.checked_div(2))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| DftError::InvalidInput("nuclear g path count overflow".into()))?;
    // Retained Value trees and native path diagnostics; the backend separately
    // reserves its AO response workspaces. This is not a process RSS limit.
    crate::resources::checked_bytes(
        &[paths, order + 2, 64, 32, 8],
        "nuclear g report and diagnostics",
    )
}

pub(super) fn run(system: &System, controls: &Controls, order: usize) -> Result<Value> {
    if !(1..=2).contains(&order) {
        return Err(DftError::InvalidInput(
            "nuclear g adapter requires order one or two".into(),
        ));
    }
    reference(controls)?;
    if super::double_hybrid::requested(controls)? {
        return Err(unsupported(
            "a correlated nuclear g response; an extracted SCF parent is not the requested method",
        ));
    }
    let gauge = controls.str("gauge", "london")?;
    match gauge.as_str() {
        "london" => (),
        "common-origin" => {
            return Err(unsupported(
                "validated common-origin nuclear g response; use gauge=london",
            ))
        }
        _ => {
            return Err(DftError::InvalidInput(format!(
                "unknown gauge `{gauge}` (london or common-origin)"
            )))
        }
    }
    let explicit_origin = controls.opt_vec3("gauge_origin")?;
    let explicit_charges = controls.opt_f64_list("soc_charges")?;
    let mut method = MolecularMethod::from_controls(controls)?;
    validate_method(&method)?;
    if method.multiplicity < 2 {
        return Err(DftError::InvalidInput(
            "nuclear g requires an open-shell multiplicity above one".into(),
        ));
    }
    if controls.contains("unrestricted") && !method.unrestricted {
        return Err(unsupported(
            "a restricted nuclear g reference; use unrestricted=true",
        ));
    }
    method.unrestricted = true;
    let config = super::vibrational::response_config(controls)?;
    controls.finish("g-tensor nuclear derivatives")?;
    if system.numbers.is_empty() || system.numbers.len() > 12 {
        return Err(unsupported(
            "validated nuclear g systems with one through twelve nuclei",
        ));
    }
    let charges =
        explicit_charges.unwrap_or_else(|| system.numbers.iter().map(|&z| f64::from(z)).collect());
    if charges.len() != system.numbers.len() || charges.iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "soc_charges requires one finite effective charge per atom".into(),
        ));
    }
    let report_workspace_bytes = workspace_bytes(system.numbers.len(), order)?;
    let _report_memory =
        crate::resources::reserve(report_workspace_bytes, "nuclear g report and diagnostics")?;
    let setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    // `ecp=none` is a valid all-electron choice. Test actual prepared operators.
    if !setup.basis.ecps.is_empty() {
        return Err(unsupported(
            "all-electron nuclear g operators for ECP references",
        ));
    }
    let origin = match explicit_origin {
        Some(origin) => origin,
        None => {
            let mut sum = Vec3::zero();
            let mut total = 0.0;
            for atom in &setup.molecule.atoms {
                let z = f64::from(atom.z);
                sum += atom.position * z;
                total += z;
            }
            if total <= 0.0 {
                return Err(DftError::InvalidInput(
                    "g-tensor needs gauge_origin for a structure without nuclear charge".into(),
                ));
            }
            sum * (1.0 / total)
        }
    };
    let prepared = run_molecular_reference(&setup)?;
    if prepared.following.is_some() {
        return Err(unsupported("a followed reference returned by common setup"));
    }
    let context = PauliGNuclearContext::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &prepared.scf,
        &setup.options,
        &config,
        origin,
        &charges,
    )?;
    let result = context.cartesian_derivatives(order)?;
    let mut response_rows = Vec::with_capacity(result.paths.len());
    let diagnostics = (|| {
        let pure = context
            .pure_field()
            .iter()
            .enumerate()
            .map(|(axis, path)| {
                checked_field_path(&path.report, config.acceptance()).map(|value| {
                    Value::object()
                        .with("field_axis", axis)
                        .with("report", value)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for (index, path) in result.paths.iter().enumerate() {
            let real = &path.real_response;
            if !real.converged
                || !real.reference.converged
                || real
                    .residuals
                    .iter()
                    .any(|v| !v.is_finite() || *v > config.acceptance())
            {
                return Err(response_failure(&real.iterations, &real.residuals));
            }
            let reference = &real.reference;
            if [
                reference.orbital_residual,
                reference.eigen_residual,
                reference.orthonormality_defect,
                reference.projector_defect,
                reference.particle_number_error,
            ]
            .iter()
            .any(|v| !v.is_finite())
            {
                return Err(DftError::InvalidInput(
                    "nonfinite nuclear g reference diagnostics".into(),
                ));
            }
            let value = &real.value;
            if value
                .projector_defects
                .iter()
                .chain(&value.stationarity_defects)
                .chain(&value.metric_defects)
                .chain(value.particle_number_coefficients.iter().flatten())
                .any(|v| !v.is_finite())
                || !value.reference_energy_error.is_finite()
                || !value.reference_fock_error.is_finite()
            {
                return Err(DftError::InvalidInput(
                    "nonfinite nuclear g real-response diagnostics".into(),
                ));
            }
            let equations = value
                .coefficient_equations
                .as_ref()
                .map(|rows| {
                    rows.iter()
                        .enumerate()
                        .map(|(index, e)| {
                            if [
                                e.source_norm,
                                e.reconstructed_norm,
                                e.linear_relative_residual,
                                e.inner_linear_relative_residual,
                            ]
                            .iter()
                            .any(|v| !v.is_finite())
                                || (!e.reconstructed_relative_residual.is_finite()
                                    && e.source_norm != 0.0)
                            {
                                return Err(DftError::InvalidInput(
                                    "nonfinite nuclear g coefficient-equation diagnostics".into(),
                                ));
                            }
                            Ok(Value::object()
                                .with("order", index + 1)
                                .with("source_norm", e.source_norm)
                                .with("reconstructed_norm", e.reconstructed_norm)
                                .with(
                                    "reconstructed_relative_residual",
                                    e.reconstructed_relative_residual
                                        .is_finite()
                                        .then_some(e.reconstructed_relative_residual),
                                )
                                .with("linear_relative_residual", e.linear_relative_residual)
                                .with(
                                    "inner_linear_relative_residual",
                                    e.inner_linear_relative_residual,
                                ))
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            if path.mixed_reports.len() != path.mixed_report_directions.len() {
                return Err(DftError::InvalidInput(
                    "nuclear g mixed report/direction count mismatch".into(),
                ));
            }
            let mixed = path
                .mixed_reports
                .iter()
                .zip(&path.mixed_report_directions)
                .map(|(report, &(axis, sign))| {
                    checked_field_path(report, config.acceptance()).map(|value| {
                        Value::object()
                            .with("field_axis", axis)
                            .with("nuclear_sign", sign)
                            .with("report", value)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            response_rows.push(
                Value::object()
                    .with("path_index", index)
                    .with("direction", &value.direction)
                    .with("real_converged", real.converged)
                    .with("real_iterations", &real.iterations)
                    .with("real_residuals", &real.residuals)
                    .with("real_reference_energy_error", value.reference_energy_error)
                    .with("real_reference_fock_error", value.reference_fock_error)
                    .with(
                        "real_reference",
                        Value::object()
                            .with("converged", reference.converged)
                            .with("iterations", reference.iterations)
                            .with("orbital_residual", reference.orbital_residual)
                            .with("eigen_residual", reference.eigen_residual)
                            .with("orthonormality_defect", reference.orthonormality_defect)
                            .with("projector_defect", reference.projector_defect)
                            .with("particle_number_error", reference.particle_number_error),
                    )
                    .with("real_projector_defects", &value.projector_defects)
                    .with("real_stationarity_defects", &value.stationarity_defects)
                    .with("real_metric_defects", &value.metric_defects)
                    .with("real_coefficient_equations", equations)
                    .with(
                        "real_particle_number_coefficients",
                        &value.particle_number_coefficients,
                    )
                    .with(
                        "response_degrees",
                        (1..=value.projector_defects.len()).collect::<Vec<_>>(),
                    )
                    .with("mixed", Value::Array(mixed))
                    .with("accumulation_bounds", &path.accumulation_bounds)
                    .with(
                        "explicit_pair_parity_defect",
                        path.explicit_pair_parity_defect,
                    )
                    .with("warnings", &real.warnings),
            );
        }
        Ok(Value::Array(pure))
    })();
    let pure = match diagnostics {
        Ok(value) => value,
        Err(error) => {
            if crate::logger::is_active() {
                for path in &result.paths {
                    crate::logger::warn(&format!(
                        "nuclear g real response: {:?}; mixed response: {:?}",
                        path.real_response, path.mixed_reports
                    ));
                }
            }
            return Err(error);
        }
    };
    let mut report = super::molecular::reference_report(&setup, &prepared,
        "analytic nuclear g derivatives of the integer collinear HF/UHF one-electron Pauli London model; no two-electron SOC or picture change");
    let fields = derivative_report(&result)?;
    let Value::Object(entries) = &mut report else {
        unreachable!("reference report object")
    };
    let Value::Object(fields) = fields else {
        unreachable!("derivative report object")
    };
    entries.extend(fields);
    report.set("gauge", "london");
    report.set("gauge_origin", origin);
    report.set("soc_charges", charges);
    report.set("reference", "uhf");
    report.set(
        "spin",
        0.5 * (prepared.scf.n_alpha - prepared.scf.n_beta) as f64,
    );
    report.set("report_workspace_bytes", report_workspace_bytes);
    report.set("response_converged", true);
    report.set("raw_paths", Value::Array(response_rows));
    report.set("pure_field_paths", pure);
    let mut warnings = report
        .get("warnings")
        .and_then(Value::as_array)
        .unwrap_or(&[])
        .to_vec();
    warnings.push(Value::Str("One-electron Pauli London nuclear g model: no two-electron SOC, picture change, vibrational averaging or experimental-spectrum validation".into()));
    report.set("warnings", Value::Array(warnings));
    Ok(report)
}

fn response_failure(iterations: &[usize], residuals: &[f64]) -> DftError {
    DftError::ResponseNotConverged {
        iterations: iterations.iter().copied().max().unwrap_or(0),
        residual: residuals.iter().copied().fold(0.0_f64, |a, b| {
            if b.is_finite() {
                a.max(b)
            } else {
                f64::INFINITY
            }
        }),
    }
}

fn checked_field_path(path: &FieldPathTaylor, tolerance: f64) -> Result<Value> {
    if path
        .energy
        .iter()
        .chain(path.dipole.iter().flatten())
        .chain(&path.projector_defects)
        .chain(&path.stationarity_defects)
        .any(|v| !v.is_finite())
        || !path.reference_defect.is_finite()
        || path
            .residuals
            .iter()
            .any(|v| !v.is_finite() || *v > tolerance)
    {
        return Err(response_failure(&[], &path.residuals));
    }
    Ok(Value::object()
        .with("energy_coefficients", &path.energy)
        .with("dipole_coefficients", &path.dipole)
        .with("reference_defect", path.reference_defect)
        .with("residuals", &path.residuals)
        .with("projector_defects", &path.projector_defects)
        .with("stationarity_defects", &path.stationarity_defects)
        .with(
            "response_degrees",
            (1..=path.residuals.len()).collect::<Vec<_>>(),
        ))
}

fn derivative_report(result: &PauliGCartesianDerivatives) -> Result<Value> {
    let tensor = |v: &PauliGDerivative| v.tensor;
    let mass = |v: &PauliGDerivative| v.mass_correction;
    let gauge = |v: &PauliGDerivative| v.gauge_correction;
    let spin_orbit = |v: &PauliGDerivative| v.spin_orbit;
    let mut out = Value::object()
        .with("nuclear_order", result.order)
        .with("g_tensor", result.value.tensor)
        .with("mass_correction", result.value.mass_correction)
        .with("gauge_correction", result.value.gauge_correction)
        .with("spin_orbit", result.value.spin_orbit)
        .with(
            "derivative_convention",
            "actual Cartesian derivatives, not derivative/k!; coordinate=3*atom+axis",
        )
        .with("provenance", "analytic")
        .with("two_electron_soc_included", false)
        .with("picture_change_included", false);
    let to_ppm = |tensor: [[f64; 3]; 3]| tensor.map(|row| row.map(|v| 1e6 * v));
    out.set(
        "g_isotropic",
        (result.value.tensor[0][0] + result.value.tensor[1][1] + result.value.tensor[2][2]) / 3.0,
    );
    out.set(
        "g_shift_ppm",
        to_ppm(std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                result.value.tensor[a][b]
                    - if a == b {
                        crate::constants::ELECTRON_G_FACTOR
                    } else {
                        0.0
                    }
            })
        })),
    );
    out.set("mass_correction_ppm", 1e6 * result.value.mass_correction);
    out.set(
        "gauge_correction_ppm",
        to_ppm(result.value.gauge_correction),
    );
    out.set("spin_orbit_ppm", to_ppm(result.value.spin_orbit));
    for (key, projection) in [
        ("g", tensor as fn(&PauliGDerivative) -> [[f64; 3]; 3]),
        ("gauge_correction", gauge),
        ("spin_orbit", spin_orbit),
    ] {
        out.set(
            &format!("{key}_gradient"),
            result
                .gradient
                .as_ref()
                .map(|rows| rows.iter().map(projection).collect::<Vec<_>>()),
        );
        out.set(
            &format!("{key}_hessian"),
            result.hessian.as_ref().map(|rows| {
                rows.iter()
                    .map(|row| row.iter().map(projection).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            }),
        );
    }
    out.set(
        "mass_gradient",
        result
            .gradient
            .as_ref()
            .map(|rows| rows.iter().map(mass).collect::<Vec<_>>()),
    );
    out.set(
        "mass_hessian",
        result.hessian.as_ref().map(|rows| {
            rows.iter()
                .map(|row| row.iter().map(mass).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        }),
    );
    let mut units = super::molecular::units(&[
        ("gauge_origin", "bohr"),
        ("soc_charges", "e"),
        ("g_tensor", "dimensionless"),
        ("spin", "hbar"),
        ("g_isotropic", "dimensionless"),
        ("g_shift_ppm", "ppm"),
        ("mass_correction_ppm", "ppm"),
        ("gauge_correction_ppm", "ppm"),
        ("spin_orbit_ppm", "ppm"),
        ("mass_correction", "dimensionless"),
        ("gauge_correction", "dimensionless"),
        ("spin_orbit", "dimensionless"),
        (
            "raw_paths",
            "atomic units, raw factorial-normalized nuclear and magnetic path diagnostics",
        ),
        (
            "pure_field_paths",
            "atomic units, raw factorial-normalized magnetic path diagnostics",
        ),
    ]);
    for prefix in ["g", "mass", "gauge_correction", "spin_orbit"] {
        units.set(&format!("{prefix}_gradient"), "1/bohr");
        units.set(&format!("{prefix}_hessian"), "1/bohr^2");
    }
    out.set("units", units);
    // Reject nonfinite physical output as well as invalid response diagnostics.
    fn finite(value: &Value) -> bool {
        match value {
            Value::Float(v) => v.is_finite(),
            Value::Array(values) => values.iter().all(finite),
            Value::Object(entries) => entries.iter().all(|(_, value)| finite(value)),
            _ => true,
        }
    }
    if !finite(&out) {
        return Err(DftError::InvalidInput(
            "nonfinite nuclear g tensor or derivative".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_is_strict_and_does_not_replace_static_references() {
        for n in 0..=2 {
            let controls = Controls::from_value(
                &Value::object()
                    .with("nuclear_order", n)
                    .with("reference", "gks"),
            )
            .unwrap();
            assert_eq!(order(&controls).unwrap(), n as usize);
            assert!(controls.finish("unconsumed static reference").is_err());
        }
        for invalid in [
            Value::Bool(true),
            Value::Float(1.0),
            Value::Int(-1),
            "1".to_value(),
        ] {
            assert!(matches!(
                order(
                    &Controls::from_value(&Value::object().with("nuclear_order", invalid)).unwrap()
                ),
                Err(DftError::InvalidInput(_))
            ));
        }
        assert!(matches!(
            order(&Controls::from_value(&Value::object().with("nuclear_order", 3)).unwrap()),
            Err(DftError::Unsupported { .. })
        ));
    }

    #[test]
    fn unsupported_references_fail_before_basis_or_scf() {
        for name in ["rks", "rhf", "roks", "rohf", "gks", "ghf"] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("reference", name)
                    .with("basis", "absent-basis"),
            )
            .unwrap();
            assert!(matches!(
                run(&System::default(), &controls, 1),
                Err(DftError::Unsupported { .. })
            ));
        }
    }

    #[test]
    fn field_zero_is_supported_and_following_is_rejected() {
        let mut method = MolecularMethod {
            xc: Some("hf".into()),
            ..MolecularMethod::default()
        };
        validate_method(&method).unwrap();
        method.field = Some(Vec3::zero());
        validate_method(&method).unwrap();
        method.field = Some(Vec3::new(0.0, 1e-7, 0.0));
        assert!(matches!(
            validate_method(&method),
            Err(DftError::Unsupported { .. })
        ));
        method.field = None;
        method.stabilize = true;
        assert!(matches!(
            validate_method(&method),
            Err(DftError::Unsupported { .. })
        ));
    }

    #[test]
    fn gauge_and_dft_fail_before_basis_or_scf() {
        for value in [
            Value::object().with("gauge", "common-origin"),
            Value::object().with("xc", "pbe"),
            Value::object().with("stabilize", true),
            Value::object().with("field", [0.0, 0.0, 1e-6]),
        ] {
            let controls = Controls::from_value(&value.with("basis", "absent-basis")).unwrap();
            assert!(matches!(
                run(&System::default(), &controls, 1),
                Err(DftError::Unsupported { .. })
            ));
        }
    }

    #[test]
    fn mixed_response_nonconvergence_is_an_error_and_degrees_start_at_one() {
        let mut path = FieldPathTaylor {
            energy: vec![1.0, 2.0, 3.0],
            dipole: vec![[0.0; 3]; 3],
            reference_defect: 1e-14,
            projector_defects: vec![1e-14; 2],
            stationarity_defects: vec![1e-14; 2],
            residuals: vec![1e-14; 2],
        };
        let report = checked_field_path(&path, 1e-12).unwrap();
        assert_eq!(
            report.get("response_degrees"),
            Some(&vec![1usize, 2].to_value())
        );
        path.residuals[1] = 1e-4;
        assert!(matches!(
            checked_field_path(&path, 1e-12),
            Err(DftError::ResponseNotConverged { .. })
        ));
        path.residuals[1] = f64::NAN;
        assert!(checked_field_path(&path, 1e-12).is_err());
    }

    #[test]
    fn workspace_overflow_is_explicit() {
        assert!(workspace_bytes(usize::MAX, 2).is_err());
        assert!(workspace_bytes(12, 2).unwrap() > workspace_bytes(2, 1).unwrap());
    }
}
