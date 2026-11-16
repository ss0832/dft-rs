// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Public adapter for the direct spin-spin contribution to H_D = S.D.S.
//! The accepted nuclear response is all-electron integer UHF/HF only.
//! SOC, spin purification and a complete zero-field-splitting model are absent.

use super::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use super::value::ToValue;
use super::{Controls, System, Value};
use crate::properties::zero_field_splitting::{
    spin_spin_nuclear_derivatives, SpinSpinNuclearDerivatives, SpinSpinPathDiagnostics,
};
use crate::{DftError, Result};

fn unsupported(missing: &str) -> DftError {
    DftError::Unsupported {
        feature: "direct spin-spin zero-field splitting".into(),
        missing: missing.into(),
    }
}

fn parse_order(controls: &Controls) -> Result<usize> {
    match controls.take("nuclear_order") {
        None | Some(Value::Null) => Ok(0),
        Some(Value::Int(order)) if (0..=2).contains(order) => Ok(*order as usize),
        Some(Value::Int(order)) if *order > 2 => Err(unsupported(
            "validated spin-spin nuclear derivatives above order two",
        )),
        _ => Err(DftError::InvalidInput(
            "nuclear_order must be an integer from zero to two".into(),
        )),
    }
}

fn check_reference(controls: &Controls) -> Result<()> {
    match controls.take("reference") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Str(name)) if name.eq_ignore_ascii_case("uhf") => Ok(()),
        Some(Value::Str(name))
            if matches!(name.to_ascii_lowercase().as_str(),
                "auto" | "rhf" | "rks" | "uks" | "rohf" | "roks" | "ghf" | "gks") =>
        {
            Err(unsupported("the requested reference; this dedicated route uses reference=uhf and does not substitute another spin model"))
        }
        Some(Value::Str(name)) => Err(DftError::InvalidInput(format!(
            "unknown reference `{name}` for spin-spin-zfs"
        ))),
        _ => Err(DftError::InvalidInput("reference must be a string selector".into())),
    }
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let order = parse_order(controls)?;
    check_reference(controls)?;
    if super::double_hybrid::requested(controls)? {
        return Err(unsupported(
            "a correlated spin-spin response; an SCF parent is not the requested property",
        ));
    }
    let mut method = MolecularMethod::from_controls(controls)?;
    if method.multiplicity < 3 {
        return Err(DftError::InvalidInput(
            "spin-spin-zfs requires multiplicity >= 3 with M_S=S>=1".into(),
        ));
    }
    if controls.contains("unrestricted") && !method.unrestricted {
        return Err(unsupported(
            "a restricted reference; spin-spin-zfs uses unrestricted=true",
        ));
    }
    method.unrestricted = true;
    if !controls.contains("conv_tol") {
        method.conv_tol = 1e-14;
    }
    if !controls.contains("conv_tol_dp") {
        method.conv_tol_dp = 1e-13;
    }
    if !controls.contains("conv_tol_grad") {
        method.conv_tol_grad = Some(1e-12);
    }
    if method.ecp.is_some()
        || method.electronic_temperature != 0.0
        || method.electron_offset != [0.0; 2]
        || method.fixed_occupations.is_some()
        || method.occupation.is_some()
        || method.solvent.is_some()
        || method.eps.is_some()
        || method.density_fit.is_some()
        || method.stabilize
        || method
            .field
            .is_some_and(|v| v.to_array().iter().any(|&x| x != 0.0))
        || !matches!(
            method.dispersion.to_ascii_lowercase().as_str(),
            "none" | "auto"
        )
        || method.skala_model.is_some()
    {
        return Err(unsupported("validated ECP, ensemble, tracked-occupation, solvent, fitted-integral, stability-followed, biased or dispersion-corrected spin-spin response"));
    }
    let functional = crate::xc::resolve(&method.functional_spec()?)?;
    if !functional.parts.is_empty()
        || functional.hybrid != crate::xc::HybridParams::HF
        || functional.required_nonlocal_term().is_some()
        || method.omega.is_some()
        || !method.xc_parameters.is_empty()
    {
        return Err(unsupported("complete DFT spin-spin nuclear acceptance; the public route is HF only, including nuclear_order=0"));
    }
    let mut config = super::vibrational::response_config(controls)?;
    if !controls.contains("response_tolerance") && !controls.contains("response_tol") {
        config.tolerance = 1e-12;
    }
    if !controls.contains("response_max_iterations") && !controls.contains("response_max_iter") {
        config.max_iterations = 1000;
    }
    controls.finish("spin-spin-zfs")?;
    let workspace_bytes = report_workspace_bytes(system.numbers.len(), order)?;
    // Reserve the retained report before starting the SCF/response, so a tiny
    // managed budget fails before expensive work or unsupported allocations.
    let _report_memory =
        crate::resources::reserve(workspace_bytes, "spin-spin report construction")?;
    let setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let raw_reference = crate::response::diagnostics::reference_diagnostics(scf)?;
    if [
        scf.energy,
        scf.s_squared,
        raw_reference.orbital_residual,
        raw_reference.eigen_residual,
        raw_reference.orthonormality_defect,
        raw_reference.projector_defect,
        raw_reference.particle_number_error,
    ]
    .iter()
    .any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "nonfinite spin-spin SCF reference diagnostics".into(),
        ));
    }
    let result = spin_spin_nuclear_derivatives(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        order,
        &config,
    )?;
    if let Err(error) = validate_paths(&result.paths, result.converged) {
        if crate::logger::is_active() {
            for path in &result.paths {
                crate::logger::warn(&format!("spin-spin-zfs raw response diagnostics: {path:?}"));
            }
        }
        return Err(error);
    }
    let mut report = report(&result, workspace_bytes)?;
    report.set("energy", scf.energy);
    report.set("scf_iterations", scf.iterations);
    report.set("spin", 0.5 * (scf.n_alpha - scf.n_beta) as f64);
    report.set("s_squared", scf.s_squared);
    report.set(
        "reference_diagnostics",
        Value::object()
            .with("converged", raw_reference.converged)
            .with("orbital_residual", raw_reference.orbital_residual)
            .with("eigen_residual", raw_reference.eigen_residual)
            .with("orthonormality_defect", raw_reference.orthonormality_defect)
            .with("projector_defect", raw_reference.projector_defect)
            .with("particle_number_error", raw_reference.particle_number_error),
    );
    Ok(report)
}

fn finite_path(path: &SpinSpinPathDiagnostics) -> bool {
    let Some(order) = path.accumulation_bounds.len().checked_sub(1) else {
        return false;
    };
    if [
        path.iterations.len(),
        path.residuals.len(),
        path.stationarity_defects.len(),
        path.metric_defects.len(),
        path.projector_defects.len(),
        path.particle_number_coefficients.len(),
    ]
    .iter()
    .any(|&count| count != order)
        || path.trace_residuals.len() != order + 1
        || path.antisymmetry_residuals.len() != order + 1
        || (order > 0
            && path
                .coefficient_equations
                .as_ref()
                .is_none_or(|rows| rows.len() != order))
    {
        return false;
    }
    path.direction
        .iter()
        .flatten()
        .chain(&path.residuals)
        .chain(&path.stationarity_defects)
        .chain(&path.metric_defects)
        .chain(&path.projector_defects)
        .chain(path.particle_number_coefficients.iter().flatten())
        .chain(path.accumulation_bounds.iter().flatten().flatten())
        .chain(&path.trace_residuals)
        .chain(&path.antisymmetry_residuals)
        .all(|v| v.is_finite())
        && path.reference_energy_error.is_finite()
        && path.reference_fock_error.is_finite()
        && path.coefficient_equations.as_ref().is_none_or(|rows| {
            rows.iter().all(|e| {
                e.source_norm.is_finite()
                    && e.reconstructed_norm.is_finite()
                    && e.linear_relative_residual.is_finite()
                    && e.inner_linear_relative_residual.is_finite()
                    && (e.reconstructed_relative_residual.is_finite() || e.source_norm == 0.0)
            })
        })
}

fn validate_paths(paths: &[SpinSpinPathDiagnostics], converged: bool) -> Result<()> {
    if paths.is_empty() {
        return Err(DftError::InvalidInput(
            "spin-spin-zfs response diagnostics are missing".into(),
        ));
    }
    if !converged
        || paths
            .iter()
            .any(|path| !path.converged || !finite_path(path))
    {
        return Err(DftError::ResponseNotConverged {
            iterations: paths
                .iter()
                .flat_map(|p| &p.iterations)
                .copied()
                .max()
                .unwrap_or(0),
            residual: paths
                .iter()
                .flat_map(|p| &p.residuals)
                .fold(0.0_f64, |a, &b| {
                    if b.is_finite() {
                        a.max(b)
                    } else {
                        f64::INFINITY
                    }
                }),
        });
    }
    Ok(())
}

fn raw_path(path: &SpinSpinPathDiagnostics) -> Value {
    let rows = (0..path.accumulation_bounds.len())
        .map(|order| {
            // Response arrays begin at degree one; operator arrays include zero.
            let index = order.checked_sub(1);
            let equation = index.and_then(|i| path.coefficient_equations.as_ref()?.get(i));
            Value::object()
                .with("order", order)
                .with(
                    "stationarity_defect",
                    index
                        .and_then(|i| path.stationarity_defects.get(i))
                        .copied(),
                )
                .with(
                    "metric_defect",
                    index.and_then(|i| path.metric_defects.get(i)).copied(),
                )
                .with(
                    "projector_defect",
                    index.and_then(|i| path.projector_defects.get(i)).copied(),
                )
                .with(
                    "particle_number_coefficient",
                    index
                        .and_then(|i| path.particle_number_coefficients.get(i))
                        .copied(),
                )
                .with("accumulation_bounds", path.accumulation_bounds[order])
                .with("trace_residual", path.trace_residuals.get(order).copied())
                .with(
                    "antisymmetry_residual",
                    path.antisymmetry_residuals.get(order).copied(),
                )
                .with(
                    "coefficient_equation",
                    equation.map(|e| {
                        Value::object()
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
                            )
                    }),
                )
        })
        .collect();
    let mut out = Value::object()
        .with("direction", &path.direction)
        .with("converged", path.converged)
        .with("iterations", &path.iterations)
        .with("residuals", &path.residuals)
        .with("reference_energy_error", path.reference_energy_error)
        .with("reference_fock_error", path.reference_fock_error)
        .with("warnings", &path.warnings);
    put(&mut out, "coefficient_diagnostics", Value::Array(rows));
    out
}

fn put(object: &mut Value, key: &str, value: Value) {
    // Transfer large report trees rather than cloning them through ToValue.
    let Value::Object(entries) = object else {
        unreachable!("internal report object")
    };
    entries.push((key.into(), value));
}

fn report_workspace_bytes(atoms: usize, order: usize) -> Result<usize> {
    let dimension = atoms
        .checked_mul(3)
        .ok_or_else(|| DftError::InvalidInput("spin-spin report dimension overflow".into()))?;
    let paths =
        match order {
            0 => Some(1),
            1 => dimension.checked_add(1),
            2 => dimension
                .checked_mul(dimension.checked_add(1).ok_or_else(|| {
                    DftError::InvalidInput("spin-spin path count overflow".into())
                })?)
                .and_then(|n| (n / 2).checked_add(1)),
            _ => return Err(unsupported("analytic orders above two")),
        }
        .ok_or_else(|| DftError::InvalidInput("spin-spin path count overflow".into()))?;
    let tensors = 1usize
        .checked_add(if order > 0 { dimension } else { 0 })
        .and_then(|n| {
            n.checked_add(if order == 2 {
                dimension.checked_mul(dimension)?
            } else {
                0
            })
        });
    let numbers = tensors
        .and_then(|n| n.checked_mul(72))
        .and_then(|n| n.checked_add(paths.checked_mul(dimension.checked_add(768)?)?))
        .ok_or_else(|| DftError::InvalidInput("spin-spin report size overflow".into()))?;
    crate::resources::checked_bytes(
        &[numbers, std::mem::size_of::<Value>()],
        "spin-spin report construction",
    )
}

fn report(result: &SpinSpinNuclearDerivatives, bytes: usize) -> Result<Value> {
    if result
        .value
        .iter()
        .flatten()
        .chain(result.gradient.iter().flatten().flatten())
        .chain(
            result
                .hessian
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .flatten(),
        )
        .any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "nonfinite spin-spin tensor or derivative".into(),
        ));
    }
    let mut out = Value::object().with("model", "direct electron spin-spin only; all-electron integer UHF/HF")
        .with("hamiltonian_convention", "H_D=S.D.S; spin operators in units of hbar")
        .with("soc_included", false).with("spin_projection_included", false)
        .with("is_total_zfs", false).with("reference", "uhf").with("functional", "hf")
        .with("nuclear_order", result.order).with("converged", true)
        .with("coordinate_order", "all input atoms, atom-major xyz; coordinate=3*atom+axis")
        .with("provenance", "analytic moving-AO spin-spin operator and self-consistent orbital response")
        .with("report_workspace_bytes", bytes)
        .with("memory_accounting", "managed response/output/report reservations; returned Value, Python and JSON copies are not a process RSS cap")
        .with("warnings", vec!["Direct spin-spin contribution only. No SOC or spin purification; a spin-contaminated determinant is not a pure multiplet."]);
    put(&mut out, "d_tensor", result.value.to_value());
    out.set("d_trace_residual", result.paths[0].trace_residuals[0]);
    out.set(
        "d_antisymmetry_residual",
        result.paths[0].antisymmetry_residuals[0],
    );
    out.set("hessian_assembly", "analytic path polarization; transposed nuclear-coordinate pairs share the mixed path; raw spin tensors are not trace-subtracted or symmetrized");
    put(
        &mut out,
        "d_gradient",
        if result.order > 0 {
            result.gradient.to_value()
        } else {
            Value::Null
        },
    );
    put(&mut out, "d_hessian", result.hessian.to_value());
    put(
        &mut out,
        "raw_paths",
        Value::Array(result.paths.iter().map(raw_path).collect()),
    );
    let mut labels = super::molecular::units(&[
        ("energy", "hartree"),
        ("spin", "hbar"),
        ("s_squared", "hbar^2"),
        ("d_tensor", "hartree"),
        ("d_gradient", "hartree/bohr"),
        ("d_hessian", "hartree/bohr^2"),
        ("d_trace_residual", "hartree"),
        ("d_antisymmetry_residual", "hartree"),
        ("report_workspace_bytes", "byte"),
        ("reference_diagnostics.orbital_residual", "hartree"),
        ("reference_diagnostics.eigen_residual", "dimensionless"),
        (
            "reference_diagnostics.orthonormality_defect",
            "dimensionless",
        ),
        ("reference_diagnostics.projector_defect", "dimensionless"),
        ("reference_diagnostics.particle_number_error", "e"),
        ("raw_paths.direction", "dimensionless"),
        ("raw_paths.residuals", "dimensionless"),
        ("raw_paths.reference_energy_error", "hartree"),
        ("raw_paths.reference_fock_error", "hartree"),
        (
            "raw_paths.coefficient_diagnostics.stationarity_defect",
            "hartree/bohr^order",
        ),
        (
            "raw_paths.coefficient_diagnostics.metric_defect",
            "1/bohr^order",
        ),
        (
            "raw_paths.coefficient_diagnostics.projector_defect",
            "1/bohr^order",
        ),
        (
            "raw_paths.coefficient_diagnostics.particle_number_coefficient",
            "1/bohr^order",
        ),
        (
            "raw_paths.coefficient_diagnostics.accumulation_bounds",
            "hartree/bohr^order",
        ),
        (
            "raw_paths.coefficient_diagnostics.trace_residual",
            "hartree/bohr^order",
        ),
        (
            "raw_paths.coefficient_diagnostics.antisymmetry_residual",
            "hartree/bohr^order",
        ),
    ]);
    for key in [
        "source_norm",
        "reconstructed_norm",
        "reconstructed_relative_residual",
        "linear_relative_residual",
        "inner_linear_relative_residual",
    ] {
        labels.set(
            &format!("raw_paths.coefficient_diagnostics.coefficient_equation.{key}"),
            if key.ends_with("norm") {
                "hartree/bohr^order"
            } else {
                "dimensionless"
            },
        );
    }
    put(&mut out, "units", labels);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::nuclear::CoefficientEquationDiagnostics;

    fn path(order: usize) -> SpinSpinPathDiagnostics {
        SpinSpinPathDiagnostics {
            direction: vec![[1.0, 0.0, 0.0]],
            converged: true,
            iterations: vec![1; order],
            residuals: vec![1e-14; order],
            stationarity_defects: (0..order).map(|i| 11.0 + i as f64).collect(),
            metric_defects: (0..order).map(|i| 21.0 + i as f64).collect(),
            projector_defects: vec![1e-15; order],
            particle_number_coefficients: vec![[1e-15; 2]; order],
            reference_energy_error: 2e-15,
            reference_fock_error: 3e-15,
            coefficient_equations: Some(
                (0..order)
                    .map(|_| CoefficientEquationDiagnostics {
                        source_norm: 1.0,
                        reconstructed_norm: 1e-14,
                        reconstructed_relative_residual: 1e-14,
                        linear_relative_residual: 1e-14,
                        inner_linear_relative_residual: 1e-14,
                    })
                    .collect(),
            ),
            accumulation_bounds: vec![[[1e-16; 3]; 3]; order + 1],
            trace_residuals: vec![1e-16; order + 1],
            antisymmetry_residuals: vec![2e-16; order + 1],
            warnings: Vec::new(),
        }
    }

    #[test]
    fn nuclear_order_is_strict_and_preserves_unsupported_orders() {
        for order in 0..=2 {
            let value = Value::object().with("nuclear_order", order);
            assert_eq!(
                parse_order(&Controls::from_value(&value).unwrap()).unwrap(),
                order as usize
            );
        }
        for value in [
            Value::Bool(true),
            Value::Float(2.0),
            Value::Int(-1),
            "2".to_value(),
        ] {
            let controls =
                Controls::from_value(&Value::object().with("nuclear_order", value)).unwrap();
            assert!(matches!(
                parse_order(&controls),
                Err(DftError::InvalidInput(_))
            ));
        }
        let controls = Controls::from_value(&Value::object().with("nuclear_order", 3)).unwrap();
        assert!(matches!(
            parse_order(&controls),
            Err(DftError::Unsupported { .. })
        ));
    }

    #[test]
    fn reference_is_never_replaced_by_a_different_spin_model() {
        for name in ["auto", "rhf", "rks", "uks", "rohf", "roks", "ghf", "gks"] {
            let controls = Controls::from_value(&Value::object().with("reference", name)).unwrap();
            assert!(matches!(
                check_reference(&controls),
                Err(DftError::Unsupported { .. })
            ));
        }
        for name in ["uhf", "UHF"] {
            let controls = Controls::from_value(&Value::object().with("reference", name)).unwrap();
            check_reference(&controls).unwrap();
        }
        check_reference(&Controls::new()).unwrap();
        let controls = Controls::from_value(&Value::object().with("reference", "uhff")).unwrap();
        assert!(matches!(
            check_reference(&controls),
            Err(DftError::InvalidInput(_))
        ));
    }

    #[test]
    fn unsupported_functional_fails_before_basis_or_scf() {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "pbe")
                .with("multiplicity", 3)
                .with("basis", "missing-spin-spin-basis"),
        )
        .unwrap();
        assert!(matches!(
            run(&System::default(), &controls),
            Err(DftError::Unsupported { .. })
        ));
    }

    #[test]
    fn diagnostic_degrees_keep_their_actual_index() {
        let path = path(2);
        let report = raw_path(&path);
        let rows = report
            .get("coefficient_diagnostics")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get("stationarity_defect"), Some(&Value::Null));
        assert_eq!(rows[0].get("coefficient_equation"), Some(&Value::Null));
        assert_eq!(rows[1].get("stationarity_defect"), Some(&11.0.to_value()));
        assert_eq!(rows[2].get("stationarity_defect"), Some(&12.0.to_value()));
        assert_eq!(rows[2].get("metric_defect"), Some(&22.0.to_value()));
    }

    #[test]
    fn every_path_and_raw_diagnostic_must_be_valid() {
        let good = path(2);
        validate_paths(&[path(0), good.clone()], true).unwrap();
        let mut bad = good.clone();
        bad.converged = false;
        assert!(matches!(
            validate_paths(&[good.clone(), bad], true),
            Err(DftError::ResponseNotConverged { .. })
        ));
        let mut bad = good.clone();
        bad.projector_defects[1] = f64::NAN;
        assert!(validate_paths(&[good.clone(), bad], true).is_err());
        let mut bad = good.clone();
        bad.coefficient_equations = None;
        assert!(validate_paths(&[bad], true).is_err());
        assert!(validate_paths(&[good], false).is_err());
        assert!(validate_paths(&[], true).is_err());
    }

    #[test]
    fn report_workspace_is_checked_and_scales_with_requested_order() {
        let sizes = [0, 1, 2].map(|order| report_workspace_bytes(4, order).unwrap());
        assert!(sizes[0] < sizes[1] && sizes[1] < sizes[2]);
        assert!(report_workspace_bytes(usize::MAX, 2).is_err());
        let controls = Controls::from_value(
            &Value::object()
                .with("multiplicity", 3)
                .with("nuclear_order", 2)
                .with("basis", "missing-budget-preflight-basis"),
        )
        .unwrap();
        let system = System {
            numbers: vec![1; 4],
            positions: vec![[0.0; 3]; 4],
            ..System::default()
        };
        crate::resources::ResourceLimits::new(Some(1), Some(1))
            .unwrap()
            .run(move || {
                let error = run(&system, &controls).unwrap_err();
                assert!(error.to_string().contains("spin-spin report construction"));
                Ok(())
            })
            .unwrap();
    }
}
