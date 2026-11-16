// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common-report adapter for the validated collinear HF hyperfine response.
//! This module adds no response equations and never launches a second SCF.

use super::controls::Controls;
use super::method::MolecularSetup;
use super::value::{ToValue, Value};
use crate::properties::hyperfine::nuclear::{
    hyperfine_nuclear_derivatives, HyperfineCoefficient, HyperfineNuclearDerivatives,
    HyperfinePathDiagnostics,
};
use crate::response::ResponseSolverConfig;
use crate::scf::ScfResult;
use crate::{DftError, Result};

pub(super) struct NuclearOptions {
    pub order: usize,
    pub nuclei: Vec<usize>,
}

fn unsupported(missing: &str) -> DftError {
    DftError::Unsupported {
        feature: "hyperfine nuclear derivatives".into(),
        missing: missing.into(),
    }
}

impl NuclearOptions {
    /// Call before reference dispatch can strip or translate a selector.
    pub fn parse(controls: &Controls, atoms: usize) -> Result<Self> {
        let order = match controls.take("nuclear_order") {
            None | Some(Value::Null) => 0,
            Some(Value::Int(value)) if *value >= 0 => usize::try_from(*value)
                .map_err(|_| DftError::InvalidInput("nuclear_order overflows usize".into()))?,
            _ => {
                return Err(DftError::InvalidInput(
                    "nuclear_order must be an integer >= 0".into(),
                ))
            }
        };
        if order > 2 {
            return Err(unsupported("analytic nuclear orders above two"));
        }
        let selection = controls.take("derivative_nuclei");
        let nuclei = match selection {
            None | Some(Value::Null) => (0..atoms).collect(),
            Some(Value::Array(values)) if order > 0 => {
                let mut result = Vec::with_capacity(values.len());
                for value in values {
                    let index = match value {
                        Value::Int(index) if *index >= 0 => usize::try_from(*index).ok(),
                        _ => None,
                    }
                    .filter(|&index| index < atoms)
                    .ok_or_else(|| {
                        DftError::InvalidInput(
                            "derivative_nuclei requires valid integer atom indices".into(),
                        )
                    })?;
                    if result.contains(&index) {
                        return Err(DftError::InvalidInput(
                            "derivative_nuclei contains duplicate indices".into(),
                        ));
                    }
                    result.push(index);
                }
                if result.is_empty() {
                    return Err(DftError::InvalidInput(
                        "derivative_nuclei must not be empty".into(),
                    ));
                }
                result
            }
            _ => {
                return Err(DftError::InvalidInput(
                    "derivative_nuclei is an integer array for nuclear_order 1 or 2".into(),
                ))
            }
        };
        if order > 0 {
            match controls.take("reference") {
                None | Some(Value::Null) => (),
                Some(Value::Str(name)) => match name.to_ascii_lowercase().as_str() {
                    "auto" | "uhf" | "uks" => (),
                    "rhf" | "rks" | "rohf" | "roks" | "ghf" | "gks" =>
                        return Err(unsupported("validated nuclear response for the requested reference; static EPR support does not imply nuclear derivative support")),
                    _ => return Err(DftError::InvalidInput(format!("unknown reference `{name}`"))),
                },
                _ => return Err(DftError::InvalidInput("reference must be a string selector".into())),
            }
        }
        Ok(Self { order, nuclei })
    }

    /// A correlated recipe must additionally be refused before its SCF parent
    /// is extracted. The common handler owns that recipe-level check.
    pub fn validate_setup(&self, setup: &MolecularSetup) -> Result<()> {
        if self.order == 0 {
            return Ok(());
        }
        if !setup.functional.parts.is_empty()
            || setup.functional.hybrid != crate::xc::HybridParams::HF
            || setup.functional.required_nonlocal_term().is_some()
        {
            return Err(unsupported("complete molecular DFT nuclear-A acceptance; the current public increment is HF and the full PBE validation is still running"));
        }
        if !setup.basis.ecps.is_empty() {
            return Err(unsupported(
                "all-electron nuclear-A operators for ECP references",
            ));
        }
        let method = &setup.method;
        if method.electronic_temperature != 0.0
            || method.electron_offset != [0.0; 2]
            || method.fixed_occupations.is_some()
            || method.occupation.is_some()
        {
            return Err(unsupported(
                "validated thermal, fractional or constrained-occupation nuclear-A response",
            ));
        }
        if method.solvent.is_some() || method.eps.is_some() || method.density_fit.is_some() {
            return Err(unsupported(
                "validated solvent or fitted-integral nuclear-A response",
            ));
        }
        Ok(())
    }

    /// Augment the existing static report after its single converged SCF.
    pub fn append(
        &self,
        report: &mut Value,
        setup: &MolecularSetup,
        scf: &ScfResult,
        config: &ResponseSolverConfig,
        nuclear_g: Option<&[f64]>,
    ) -> Result<()> {
        if self.order == 0 {
            return Ok(());
        }
        self.validate_setup(setup)?;
        validate_nuclear_g(nuclear_g, setup.molecule.len())?;
        let result = hyperfine_nuclear_derivatives(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            &setup.options,
            &self.nuclei,
            self.order,
            config,
        )?;
        let mut nuclear = match nuclear_report(&result, nuclear_g, setup.molecule.len()) {
            Ok(value) => value,
            Err(error) => {
                if crate::logger::is_active() {
                    for path in &result.paths {
                        crate::logger::warn(&format!(
                            "hyperfine nuclear raw diagnostics: {path:?}"
                        ));
                    }
                }
                return Err(error);
            }
        };
        // The common unit registry and ASE walker use a single root table of
        // dotted paths. Move the labels there without cloning the tensor tree.
        let Value::Object(child) = &mut nuclear else {
            unreachable!("nuclear report object")
        };
        let index = child
            .iter()
            .position(|(key, _)| key == "units")
            .expect("report labels");
        let (_, Value::Object(labels)) = child.remove(index) else {
            unreachable!("report unit object")
        };
        let Value::Object(parent) = report else {
            unreachable!("static report object")
        };
        if !parent.iter().any(|(key, _)| key == "units") {
            parent.push(("units".into(), Value::object()));
        }
        let (_, units) = parent.iter_mut().find(|(key, _)| key == "units").unwrap();
        for (key, label) in labels {
            owned_set(units, &format!("hyperfine_nuclear.{key}"), label);
        }
        owned_set(report, "hyperfine_nuclear", nuclear);
        Ok(())
    }
}

fn validate_nuclear_g(values: Option<&[f64]>, atoms: usize) -> Result<()> {
    if values.is_some_and(|values| values.len() != atoms || values.iter().any(|v| !v.is_finite())) {
        return Err(DftError::InvalidInput(
            "nuclear_g requires one finite signed factor per input atom".into(),
        ));
    }
    Ok(())
}

fn finite_path(path: &HyperfinePathDiagnostics) -> bool {
    path.residuals.iter().chain(&path.stationarity_defects).chain(&path.metric_defects)
        .chain(&path.projector_defects).chain(path.particle_number_coefficients.iter().flatten())
        .chain(path.accumulation_bounds.iter().flatten().flatten())
        .chain(path.operator_poisson_residuals.iter().flatten())
        .chain(path.operator_hermiticity_residuals.iter().flatten())
        .all(|v| v.is_finite())
        && path.reference_energy_error.is_finite() && path.reference_fock_error.is_finite()
        && path.coefficient_equations.as_ref().is_none_or(|equations| equations.iter().all(|e|
            e.source_norm.is_finite() && e.reconstructed_norm.is_finite()
            && e.linear_relative_residual.is_finite() && e.inner_linear_relative_residual.is_finite()
            // A reconstructed/source ratio is undefined at an exactly zero source.
            && (e.reconstructed_relative_residual.is_finite() || e.source_norm == 0.0)))
}

fn field(value: &HyperfineCoefficient, component: &str, scale: f64) -> Value {
    match component {
        "contact" => (scale * value.contact).to_value(),
        "dipolar" => value.dipolar.map(|row| row.map(|v| scale * v)).to_value(),
        "total" => value.total.map(|row| row.map(|v| scale * v)).to_value(),
        _ => unreachable!("internal tensor component"),
    }
}

fn owned_set(object: &mut Value, key: &str, value: Value) {
    // Value::set clones an already-built tree through ToValue. Transfer large
    // tensor buffers directly instead of transiently duplicating them.
    let Value::Object(entries) = object else {
        unreachable!("internal object")
    };
    if let Some((_, old)) = entries.iter_mut().find(|(name, _)| name == key) {
        *old = value;
    } else {
        entries.push((key.into(), value));
    }
}

fn raw_path(path: &HyperfinePathDiagnostics, order: usize) -> Value {
    let mut out = Value::object()
        .with("direction", &path.direction)
        .with("converged", path.converged)
        .with("iterations", &path.iterations)
        .with("residuals", &path.residuals)
        .with("warnings", &path.warnings)
        .with("reference_energy_error", path.reference_energy_error)
        .with("reference_fock_error", path.reference_fock_error);
    let orders = (0..=order)
        .map(|degree| {
            // Shared response defects start at order one. Operator and
            // accumulation arrays below also contain their zeroth coefficient.
            let response_index = degree.checked_sub(1);
            let mut row = Value::object()
                .with("order", degree)
                .with(
                    "stationarity_defect",
                    response_index
                        .and_then(|i| path.stationarity_defects.get(i))
                        .copied(),
                )
                .with(
                    "metric_defect",
                    response_index
                        .and_then(|i| path.metric_defects.get(i))
                        .copied(),
                )
                .with(
                    "projector_defect",
                    response_index
                        .and_then(|i| path.projector_defects.get(i))
                        .copied(),
                )
                .with(
                    "particle_number_coefficient",
                    response_index
                        .and_then(|i| path.particle_number_coefficients.get(i))
                        .copied(),
                )
                .with(
                    "accumulation_bounds",
                    path.accumulation_bounds
                        .iter()
                        .map(|p| p.get(degree).copied())
                        .collect::<Vec<_>>(),
                )
                .with(
                    "operator_poisson_residuals",
                    path.operator_poisson_residuals
                        .iter()
                        .map(|p| p.get(degree).copied())
                        .collect::<Vec<_>>(),
                )
                .with(
                    "operator_hermiticity_residuals",
                    path.operator_hermiticity_residuals
                        .iter()
                        .map(|p| p.get(degree).copied())
                        .collect::<Vec<_>>(),
                );
            let equation = degree
                .checked_sub(1)
                .and_then(|i| path.coefficient_equations.as_ref()?.get(i));
            if let Some(e) = equation {
                row.set(
                    "coefficient_equation",
                    Value::object()
                        .with("order", degree)
                        .with("source_norm", e.source_norm)
                        .with("reconstructed_norm", e.reconstructed_norm)
                        .with(
                            "reconstructed_relative_residual",
                            e.reconstructed_relative_residual,
                        )
                        .with("linear_relative_residual", e.linear_relative_residual)
                        .with(
                            "inner_linear_relative_residual",
                            e.inner_linear_relative_residual,
                        ),
                );
            } else {
                row.set("coefficient_equation", Value::Null);
            }
            row
        })
        .collect();
    owned_set(&mut out, "coefficient_diagnostics", Value::Array(orders));
    out
}

pub(super) fn nuclear_report(
    result: &HyperfineNuclearDerivatives,
    nuclear_g: Option<&[f64]>,
    atoms: usize,
) -> Result<Value> {
    validate_nuclear_g(nuclear_g, atoms)?;
    if result.nuclei.iter().any(|&atom| atom >= atoms) {
        return Err(DftError::InvalidInput(
            "hyperfine result has an invalid target nucleus".into(),
        ));
    }
    if !result.converged || result.paths.iter().any(|p| !p.converged || !finite_path(p)) {
        let iterations = result
            .paths
            .iter()
            .flat_map(|p| &p.iterations)
            .copied()
            .max()
            .unwrap_or(0);
        let residual = result
            .paths
            .iter()
            .flat_map(|p| &p.residuals)
            .fold(0.0_f64, |a, &b| {
                if b.is_finite() {
                    a.max(b)
                } else {
                    f64::INFINITY
                }
            });
        return Err(DftError::ResponseNotConverged {
            iterations,
            residual,
        });
    }
    for (target, gradient) in result.gradient.iter().enumerate() {
        let factor = nuclear_g.map_or(1.0, |g| g[result.nuclei[target]]);
        let hessian = result
            .hessian
            .as_ref()
            .into_iter()
            .flat_map(|h| h[target].iter().flatten());
        for value in gradient.iter().chain(hessian) {
            if std::iter::once(&value.contact)
                .chain(value.dipolar.iter().flatten())
                .chain(value.total.iter().flatten())
                .any(|&v| !v.is_finite() || !(v * factor).is_finite())
            {
                return Err(DftError::InvalidInput(
                    "nonfinite hyperfine derivative or overflowing nuclear_g scaling".into(),
                ));
            }
        }
    }
    // This bounds report construction while the analytic output lease remains
    // alive. Caller-owned Value trees, JSON strings and Python copies after
    // return are outside this temporary reservation, not a claimed RSS cap.
    let dimension = atoms
        .checked_mul(3)
        .ok_or_else(|| DftError::InvalidInput("nuclear dimension overflow".into()))?;
    let count = dimension
        .checked_mul(dimension)
        .and_then(|h| h.checked_add(dimension.checked_add(1)?))
        .and_then(|n| n.checked_mul(result.nuclei.len()))
        .and_then(|n| {
            n.checked_add(
                result
                    .paths
                    .len()
                    .checked_mul(atoms.checked_mul(3)?.checked_add(512)?)?,
            )
        })
        .ok_or_else(|| DftError::InvalidInput("hyperfine report size overflow".into()))?;
    let bytes = crate::resources::checked_bytes(
        &[count, 64, std::mem::size_of::<Value>()],
        "hyperfine report construction",
    )?;
    let _workspace = crate::resources::reserve(bytes, "hyperfine report construction")?;
    let mut out = Value::object().with("order", result.order).with("derivative_nuclei", &result.nuclei)
        .with("coordinate_order", "all input atoms, atom-major xyz; coordinate=3*atom+axis")
        .with("converged", true).with("provenance", "analytic moving-basis/operator and self-consistent nuclear response")
        .with("model", "all-electron integer collinear HF; no isotope inference or vibrational correction")
        .with("independent_acceptance", "C1 methyl HF/STO-3G: all6840 strict four-step component fits; no DFT nuclear-A acceptance yet")
        .with("factorial_convention", "Cartesian derivatives; no extra factorial conversion. Raw coefficient diagnostics retain derivative/k!.");
    let mut units = Value::object().with("order", "dimensionless").with(
        "derivative_nuclei",
        "dimensionless, zero-based input atom indices",
    );
    for (key, label) in [
        ("direction", "dimensionless"),
        ("iterations", "dimensionless"),
        ("residuals", "dimensionless"),
        ("reference_energy_error", "hartree"),
        ("reference_fock_error", "hartree"),
        ("coefficient_diagnostics.order", "dimensionless"),
        (
            "coefficient_diagnostics.stationarity_defect",
            "hartree/bohr^order",
        ),
        ("coefficient_diagnostics.metric_defect", "1/bohr^order"),
        ("coefficient_diagnostics.projector_defect", "1/bohr^order"),
        (
            "coefficient_diagnostics.particle_number_coefficient",
            "1/bohr^order",
        ),
        (
            "coefficient_diagnostics.accumulation_bounds",
            "1/bohr^(order+3)",
        ),
        (
            "coefficient_diagnostics.operator_poisson_residuals",
            "1/bohr^(order+3)",
        ),
        (
            "coefficient_diagnostics.operator_hermiticity_residuals",
            "1/bohr^(order+3)",
        ),
        (
            "coefficient_diagnostics.coefficient_equation.order",
            "dimensionless",
        ),
        (
            "coefficient_diagnostics.coefficient_equation.source_norm",
            "hartree/bohr^order",
        ),
        (
            "coefficient_diagnostics.coefficient_equation.reconstructed_norm",
            "hartree/bohr^order",
        ),
        (
            "coefficient_diagnostics.coefficient_equation.reconstructed_relative_residual",
            "dimensionless",
        ),
        (
            "coefficient_diagnostics.coefficient_equation.linear_relative_residual",
            "dimensionless",
        ),
        (
            "coefficient_diagnostics.coefficient_equation.inner_linear_relative_residual",
            "dimensionless",
        ),
    ] {
        units.set(&format!("raw_paths.{key}"), label);
    }
    for scaled in [false, true] {
        if scaled && nuclear_g.is_none() {
            continue;
        }
        let suffix = if scaled { "" } else { "_per_g" };
        for component in ["contact", "dipolar", "total"] {
            let gradient = result
                .gradient
                .iter()
                .enumerate()
                .map(|(target, row)| {
                    let factor = if scaled {
                        nuclear_g.unwrap()[result.nuclei[target]]
                    } else {
                        1.0
                    };
                    Value::Array(
                        row.iter()
                            .map(|value| field(value, component, factor))
                            .collect(),
                    )
                })
                .collect();
            let key = format!("{component}_gradient{suffix}");
            owned_set(&mut out, &key, Value::Array(gradient));
            units.set(
                &key,
                if scaled {
                    "MHz/bohr"
                } else {
                    "MHz/bohr, per nuclear g"
                },
            );
            let hessian = result
                .hessian
                .as_ref()
                .map(|tensor| {
                    Value::Array(
                        tensor
                            .iter()
                            .enumerate()
                            .map(|(target, matrix)| {
                                let factor = if scaled {
                                    nuclear_g.unwrap()[result.nuclei[target]]
                                } else {
                                    1.0
                                };
                                Value::Array(
                                    matrix
                                        .iter()
                                        .map(|row| {
                                            Value::Array(
                                                row.iter()
                                                    .map(|value| field(value, component, factor))
                                                    .collect(),
                                            )
                                        })
                                        .collect(),
                                )
                            })
                            .collect(),
                    )
                })
                .unwrap_or(Value::Null);
            let key = format!("{component}_hessian{suffix}");
            owned_set(&mut out, &key, hessian);
            units.set(
                &key,
                if scaled {
                    "MHz/bohr^2"
                } else {
                    "MHz/bohr^2, per nuclear g"
                },
            );
        }
    }
    owned_set(
        &mut out,
        "raw_paths",
        Value::Array(
            result
                .paths
                .iter()
                .map(|p| raw_path(p, result.order))
                .collect(),
        ),
    );
    owned_set(&mut out, "units", units);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::nuclear::CoefficientEquationDiagnostics;

    #[test]
    fn diagnostic_orders_preserve_exact_raw_values() {
        for order in [1, 2] {
            let path = HyperfinePathDiagnostics {
                direction: vec![[1.0, 0.0, 0.0]],
                converged: true,
                iterations: (0..order).map(|i| 10 + i).collect(),
                residuals: vec![0.01; order],
                stationarity_defects: (0..order).map(|i| 11.0 + i as f64).collect(),
                metric_defects: (0..order).map(|i| 21.0 + i as f64).collect(),
                projector_defects: (0..order).map(|i| 31.0 + i as f64).collect(),
                particle_number_coefficients: (0..order)
                    .map(|i| [41.0 + i as f64, 51.0 + i as f64])
                    .collect(),
                reference_energy_error: 6.0,
                reference_fock_error: 7.0,
                coefficient_equations: Some(
                    (0..order)
                        .map(|i| CoefficientEquationDiagnostics {
                            source_norm: 61.0 + i as f64,
                            reconstructed_norm: 71.0 + i as f64,
                            reconstructed_relative_residual: 0.1,
                            linear_relative_residual: 0.2,
                            inner_linear_relative_residual: 0.3,
                        })
                        .collect(),
                ),
                accumulation_bounds: vec![(0..=order).map(|i| [101.0 + i as f64; 10]).collect()],
                operator_poisson_residuals: vec![(0..=order).map(|i| 201.0 + i as f64).collect()],
                operator_hermiticity_residuals: vec![(0..=order)
                    .map(|i| 301.0 + i as f64)
                    .collect()],
                warnings: Vec::new(),
            };
            let report = raw_path(&path, order);
            let Value::Array(rows) = report.get("coefficient_diagnostics").unwrap() else {
                panic!("per-order diagnostic rows");
            };
            assert_eq!(rows.len(), order + 1);
            for (degree, row) in rows.iter().enumerate() {
                assert_eq!(row.get("order"), Some(&degree.to_value()));
                if degree == 0 {
                    for key in [
                        "stationarity_defect",
                        "metric_defect",
                        "projector_defect",
                        "particle_number_coefficient",
                        "coefficient_equation",
                    ] {
                        assert_eq!(row.get(key), Some(&Value::Null), "unmeasured zeroth {key}");
                    }
                } else {
                    assert_eq!(
                        row.get("stationarity_defect"),
                        Some(&(10.0 + degree as f64).to_value())
                    );
                    assert_eq!(
                        row.get("metric_defect"),
                        Some(&(20.0 + degree as f64).to_value())
                    );
                    assert_eq!(
                        row.get("projector_defect"),
                        Some(&(30.0 + degree as f64).to_value())
                    );
                    assert_eq!(
                        row.get("particle_number_coefficient"),
                        Some(&[40.0 + degree as f64, 50.0 + degree as f64].to_value())
                    );
                    let equation = row.get("coefficient_equation").unwrap();
                    assert_eq!(
                        equation.get("source_norm"),
                        Some(&(60.0 + degree as f64).to_value())
                    );
                    assert_eq!(
                        equation.get("reconstructed_norm"),
                        Some(&(70.0 + degree as f64).to_value())
                    );
                }
                assert_eq!(
                    row.get("accumulation_bounds"),
                    Some(&vec![[101.0 + degree as f64; 10]].to_value())
                );
                assert_eq!(
                    row.get("operator_poisson_residuals"),
                    Some(&vec![201.0 + degree as f64].to_value())
                );
                assert_eq!(
                    row.get("operator_hermiticity_residuals"),
                    Some(&vec![301.0 + degree as f64].to_value())
                );
            }
            assert_eq!(
                report.get("reference_energy_error"),
                Some(&Value::Float(6.0))
            );
            assert_eq!(report.get("reference_fock_error"), Some(&Value::Float(7.0)));
        }
    }
}
