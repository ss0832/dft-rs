// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit periodic common-orbital reference dispatch; no unrestricted fallback.

use super::band_properties::{roks_semicanonical_bands, stationary_mesh_dos, DosOptions};
use super::controls::Controls;
use super::molecular::units;
use super::periodic::{
    append_dimensional_geometry, prepare_periodic, report_dispersion, PeriodicMethod,
};
use super::value::Value;
use super::System;
use crate::linalg::Matrix;
use crate::periodic::dispersion::{cell_dispersion, PeriodicDispersionResult};
use crate::periodic::roks::{GammaRoksProblem, MeshRoksProblem, MeshRoksResult};
use crate::periodic::scf_boundary::{DimensionalScfBackend, DimensionalScfOptions};
use crate::periodic::PeriodicGeometry;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

fn matrix(value: &Matrix) -> Vec<Vec<f64>> {
    (0..value.rows)
        .map(|i| (0..value.cols).map(|j| value[(i, j)]).collect())
        .collect()
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let reference = controls
        .str("reference", "roks")?
        .trim()
        .to_ascii_lowercase();
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: format!("correlated periodic {reference} reference"),
            missing: "the constrained common-orbital PT2 amplitudes and variational orbital/pair response".into(),
        });
    }
    let with_forces = controls.bool("forces", matches!(id, "stress" | "opt-fixed-cell"))?;
    let with_stress = controls.bool("stress", false)? || id == "stress";
    if !matches!(id, "periodic-energy" | "stress" | "opt-fixed-cell" | "dos") {
        return Err(DftError::Unsupported {
            feature: format!("periodic {reference} {id} response"),
            missing: "the corresponding constrained common-orbital coupled response".into(),
        });
    }
    if system.pbc != [true; 3] {
        if id == "opt-fixed-cell" {
            return Err(DftError::Unsupported {
                feature: "lower-dimensional common-orbital fixed-cell optimization".into(),
                missing: "the dimensional optimizer integration".into(),
            });
        }
        return super::periodic::periodic_field_scf_impl(system, controls, false, Some(id));
    }
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let dispersion = method.dispersion_model_for(&system.numbers)?;
    validate_method(controls, &method)?;
    let dos = (id == "dos")
        .then(|| DosOptions::parse(controls))
        .transpose()?;
    if let Some(dos) = &dos {
        dos.require_reference_mesh(method.k_mesh)?;
        if with_forces || with_stress {
            return Err(DftError::InvalidInput(
                "dos accepts a stationary reference without geometry derivatives".into(),
            ));
        }
    }
    let optimization_options = if id == "opt-fixed-cell" {
        let defaults = crate::periodic::optimizer::PeriodicOptOptions::default();
        Some(crate::periodic::optimizer::PeriodicOptOptions {
            max_steps: controls.usize("max_steps", defaults.max_steps)?,
            gtol: controls.f64("gtol", defaults.gtol)?,
            step_max: controls.f64("step_max", defaults.step_max)?,
            history: controls.usize("history", defaults.history)?,
            trajectory: None,
            dispersion: dispersion.clone(),
        })
    } else {
        None
    };
    controls.finish(id)?;
    let setup = prepare_periodic(system, &method)?;
    validate_functional(&reference, &setup.functional, setup.options.vv10.is_some())?;
    if let Some(options) = optimization_options {
        return fixed_cell(setup, &reference, &options, with_stress);
    }
    let dispersion = cell_dispersion(&setup.cell, &dispersion)?;
    if setup.options.k_mesh != [1, 1, 1] {
        let problem = MeshRoksProblem::new(&setup.cell, &setup.functional, &setup.options)?;
        let (mut report, result) = mesh_reference_report(
            &reference,
            &setup.functional,
            &setup.options,
            setup.method.overlap_cutoff,
            &problem,
        )?;
        if let Some(dos) = &dos {
            report_dispersion(&mut report, &method, dispersion.as_ref());
            return mesh_dos(
                report,
                setup.cell.basis(),
                setup.cell.molecule().len(),
                &problem,
                &result,
                dos,
            );
        }
        if with_forces || with_stress {
            let (gradient, stress) = problem.bulk_geometry(&result, with_forces, with_stress)?;
            let mut labels = Vec::new();
            append_bulk_geometry(
                &mut report,
                &mut labels,
                gradient,
                stress,
                dispersion.as_ref(),
            );
            let mut original = match report.get("units") {
                Some(Value::Object(labels)) => labels.clone(),
                _ => Vec::new(),
            };
            original.extend(
                labels
                    .into_iter()
                    .map(|(key, unit)| (key.into(), Value::Str(unit.into()))),
            );
            report.set("units", Value::Object(original));
        }
        report_dispersion(&mut report, &method, dispersion.as_ref());
        return Ok(report);
    }
    let problem = GammaRoksProblem::new(&setup.cell, &setup.functional, &setup.options)?;
    if (with_forces || with_stress) && problem.max_nuclear_response_order() == 0 {
        return Err(DftError::Unsupported {
            feature: "periodic ROKS nuclear/strain derivative with truncated overlap rank".into(),
            missing: "the geometry-dependent discarded-space projector derivative".into(),
        });
    }
    let result = problem.run()?;
    if let Some(dos) = &dos {
        let mut labels = Vec::new();
        let mut report = base_report(
            &reference,
            &setup.functional,
            setup
                .options
                .xc_ao_cutoff
                .unwrap_or(setup.method.overlap_cutoff),
            &problem,
            &result,
            &mut labels,
            setup.options.vv10,
        );
        report.set("units", units(&labels));
        report_dispersion(&mut report, &method, dispersion.as_ref());
        return gamma_dos(
            report,
            setup.cell.basis(),
            setup.cell.molecule().len(),
            &problem,
            &result,
            dos,
        );
    }
    reference_report(
        &setup,
        &reference,
        &problem,
        &result,
        with_forces,
        with_stress,
        dispersion.as_ref(),
    )
}

pub(super) fn validate_method(controls: &Controls, method: &PeriodicMethod) -> Result<()> {
    if method.unrestricted
        || !method.guess.eq_ignore_ascii_case("core")
        || method.level_shift != 0.0
        || (controls.contains("damping") && method.damping != 0.0)
        || (controls.contains("no_adiis") && method.adiis)
        || controls.contains("adiis_until_error")
    {
        return Err(DftError::Unsupported {
            feature: "periodic common-orbital reference with unrestricted/damped/shifted or ADIIS initialization".into(),
            missing: "the common-orbital solver currently uses common core orbitals and centered CDIIS".into(),
        });
    }
    Ok(())
}

fn validate_functional(reference: &str, functional: &ResolvedFunctional, vv10: bool) -> Result<()> {
    if reference == "rohf"
        && (!functional.parts.is_empty()
            || functional.hybrid.cx != 1.0
            || functional.hybrid.clr != 0.0
            || vv10)
    {
        return Err(DftError::Unsupported { feature: "periodic ROHF with a non-HF provider".into(),
            missing: "an exact-exchange-only HF energy provider; use ROKS for a semilocal/hybrid provider".into() });
    }
    Ok(())
}

fn reference_report(
    setup: &super::periodic::PeriodicSetup,
    reference: &str,
    problem: &GammaRoksProblem<'_>,
    result: &crate::periodic::roks::GammaRoksResult,
    with_forces: bool,
    with_stress: bool,
    dispersion: Option<&PeriodicDispersionResult>,
) -> Result<Value> {
    let mut labels = Vec::new();
    let mut report = base_report(
        reference,
        &setup.functional,
        setup
            .options
            .xc_ao_cutoff
            .unwrap_or(setup.method.overlap_cutoff),
        problem,
        result,
        &mut labels,
        setup.options.vv10,
    );
    let gradient = with_forces
        .then(|| problem.nuclear_gradient(result))
        .transpose()?;
    let stress = with_stress.then(|| problem.stress(result)).transpose()?;
    append_bulk_geometry(&mut report, &mut labels, gradient, stress, dispersion);
    report.set("units", units(&labels));
    report_dispersion(&mut report, &setup.method, dispersion);
    Ok(report)
}

fn append_bulk_geometry(
    report: &mut Value,
    labels: &mut Vec<(&'static str, &'static str)>,
    mut gradient: Option<Vec<crate::math::Vec3>>,
    mut stress: Option<crate::periodic::stress::PeriodicStress>,
    dispersion: Option<&PeriodicDispersionResult>,
) {
    if let Some(dispersion) = dispersion {
        if let Some(gradient) = &mut gradient {
            dispersion.add_to_gradient(gradient);
        }
        if let Some(stress) = &mut stress {
            crate::periodic::dispersion::add_to_stress(stress, &dispersion.strain_derivative);
        }
    }
    if let Some(gradient) = gradient {
        let sum = gradient
            .iter()
            .fold(crate::math::Vec3::zero(), |a, b| a + *b);
        report.set(
            "gradient",
            gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>(),
        );
        report.set(
            "forces",
            gradient
                .iter()
                .map(|g| (-*g).to_array())
                .collect::<Vec<_>>(),
        );
        report.set("translation_residual", sum.norm());
        report.set("gradient_provenance","analytic constrained common-orbital Lagrangian; full moving basis, overlap, J/K and XC grid terms");
        labels.extend([
            ("gradient", "hartree/bohr"),
            ("forces", "hartree/bohr"),
            ("translation_residual", "hartree/bohr"),
        ]);
    }
    if let Some(stress) = stress {
        super::periodic::stress_fields(report, &stress);
        report.set("volume", stress.volume);
        report.set(
            "stress_provenance",
            "analytic constrained common-orbital Lagrangian; fixed fractional coordinates",
        );
        labels.extend(super::periodic::STRESS_UNITS);
        labels.push(("volume", "bohr^3"));
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dimensional_report(
    backend: &DimensionalScfBackend<'_>,
    options: &DimensionalScfOptions,
    geometry: &PeriodicGeometry,
    reference: &str,
    functional: &ResolvedFunctional,
    [with_forces, with_stress]: [bool; 2],
    dos: Option<&DosOptions>,
    dispersion: Option<&PeriodicDispersionResult>,
) -> Result<Value> {
    validate_functional(reference, functional, options.scf.vv10.is_some())?;
    if options.scf.k_mesh != [1, 1, 1] {
        let problem = backend.mesh_roks_problem()?;
        let (mut report, result) = mesh_reference_report(
            reference,
            functional,
            &options.scf,
            options.basis_cutoff,
            &problem,
        )?;
        report.set("dimension", geometry.dimension());
        report.set("pbc", geometry.boundary().axes);
        if let Some(dos) = dos {
            let frame = backend.field_free_nuclear_frame()?;
            return mesh_dos(
                report,
                frame.basis,
                frame.molecule.len(),
                &problem,
                &result,
                dos,
            );
        }
        if with_forces || with_stress {
            let derivatives = problem.dimensional_derivatives(&result, with_forces, with_stress)?;
            append_dimensional_geometry(
                &mut report,
                geometry,
                derivatives,
                dispersion,
                "constrained common-orbital",
            )?;
        }
        return Ok(report);
    }
    let problem = backend.gamma_roks_problem()?;
    let result = problem.run()?;
    let mut labels = Vec::new();
    let mut report = base_report(
        reference,
        functional,
        options.scf.xc_ao_cutoff.unwrap_or(options.basis_cutoff),
        &problem,
        &result,
        &mut labels,
        options.scf.vv10,
    );
    report.set(
        "model",
        "Dimensional Gamma constrained common-spatial-orbital open-shell reference",
    );
    report.set("dimension", geometry.dimension());
    report.set("pbc", geometry.boundary().axes);
    labels.push(("dimension", "dimensionless"));
    report.set("units", units(&labels));
    if let Some(dos) = dos {
        let frame = backend.field_free_nuclear_frame()?;
        return gamma_dos(
            report,
            frame.basis,
            frame.molecule.len(),
            &problem,
            &result,
            dos,
        );
    }
    if with_forces || with_stress {
        let derivatives = problem.dimensional_derivatives(&result, with_forces, with_stress)?;
        append_dimensional_geometry(
            &mut report,
            geometry,
            derivatives,
            dispersion,
            "constrained common-orbital",
        )?;
    }
    Ok(report)
}

fn gamma_dos(
    mut report: Value,
    basis: &crate::basis::MolBasis,
    natoms: usize,
    problem: &GammaRoksProblem<'_>,
    result: &crate::periodic::roks::GammaRoksResult,
    options: &DosOptions,
) -> Result<Value> {
    use crate::periodic::BlochMatrix;
    let real = |m: &Matrix| BlochMatrix::from_pair((m.clone(), Matrix::zeros(m.rows, m.cols)));
    let coefficients = [real(&result.coefficients)];
    let fock = result.state.fock.each_ref().map(|f| vec![real(f)]);
    let overlap = [real(&problem.overlap)];
    let bands = roks_semicanonical_bands(&coefficients, &fock, [result.n_alpha, result.n_beta])?;
    report.set(
        "spectral_model",
        "spin-resolved semicanonical closed, open and virtual ROKS blocks",
    );
    stationary_mesh_dos(report, basis, natoms, &overlap, &bands, 1.0, options)
}

fn mesh_dos(
    mut report: Value,
    basis: &crate::basis::MolBasis,
    natoms: usize,
    problem: &MeshRoksProblem<'_>,
    result: &MeshRoksResult,
    options: &DosOptions,
) -> Result<Value> {
    let coefficients: Vec<_> = result
        .orbitals
        .iter()
        .map(|band| band.coefficients.clone())
        .collect();
    let bands = roks_semicanonical_bands(&coefficients, &result.state.fock, problem.occupations)?;
    report.set(
        "spectral_model",
        "spin-resolved semicanonical closed, open and virtual ROKS blocks",
    );
    stationary_mesh_dos(
        report,
        basis,
        natoms,
        &problem.overlap,
        &bands,
        1.0,
        options,
    )
}

fn mesh_reference_report(
    reference: &str,
    functional: &ResolvedFunctional,
    options: &crate::periodic::scf::PeriodicScfOptions,
    overlap_cutoff: f64,
    problem: &MeshRoksProblem<'_>,
) -> Result<(Value, MeshRoksResult)> {
    use crate::periodic::BlochMatrix;
    let result = problem.run()?;
    let state = &result.state;
    let coefficients: Vec<_> = result.orbitals.iter().map(|b| &b.coefficients).collect();
    let parts = |mesh: &[BlochMatrix], imaginary: bool| {
        mesh.iter()
            .map(|m| matrix(if imaginary { &m.im } else { &m.re }))
            .collect::<Vec<_>>()
    };
    let mut labels = vec![
        ("energy", "hartree per cell"),
        ("free_energy", "hartree per cell"),
        ("electronic_energy", "hartree per cell"),
        ("nuclear_energy", "hartree per cell"),
        ("xc_energy", "hartree per cell"),
        ("n_alpha", "dimensionless"),
        ("n_beta", "dimensionless"),
        ("scf_iterations", "dimensionless"),
        ("nuclear_response_order", "dimensionless"),
        ("orbital_residual", "hartree"),
        ("density_change", "dimensionless"),
        ("energy_change", "hartree"),
        ("metric_defect", "dimensionless"),
        ("particle_number_errors", "electrons"),
        ("idempotency_defects", "dimensionless"),
        ("closed_subspace_defect", "dimensionless"),
        ("density_conjugation_defect", "dimensionless"),
        ("fock_conjugation_defect", "hartree"),
        ("k_points", "1/bohr"),
        ("k_mesh", "dimensionless"),
        ("xc_ao_cutoff", "bohr"),
        ("dimension", "dimensionless"),
    ];
    let mut report = Value::object()
        .with("model", "Complex periodic mesh constrained common-spatial-orbital reference")
        .with("reference_type", reference)
        .with("functional", functional.description.as_str())
        .with("common_orbital_constraint", "fixed integer alpha and beta ranks at every k; closed beta subspace contained in alpha")
        .with("converged", true)
        .with("unrestricted", false)
        .with("nuclear_response_available", problem.supports_geometry())
        .with("nuclear_response_order", usize::from(problem.supports_geometry()))
        .with("nuclear_hessian_available", false)
        .with("energy", state.energy)
        .with("free_energy", state.energy)
        .with("electronic_energy", state.electronic_energy)
        .with("nuclear_energy", state.energy - state.electronic_energy)
        .with("xc_energy", state.xc_energy)
        .with("n_alpha", problem.occupations[0])
        .with("n_beta", problem.occupations[1])
        .with("scf_iterations", result.iterations)
        .with("orbital_residual", state.orbital_residual)
        .with("density_change", result.density_change)
        .with("energy_change", result.energy_change)
        .with("metric_defect", state.metric_defect)
        .with("particle_number_errors", result.particle_number_error)
        .with("idempotency_defects", result.idempotency_defect)
        .with("closed_subspace_defect", result.closed_subspace_defect)
        .with("density_conjugation_defect", result.conjugation_defect[0])
        .with("fock_conjugation_defect", result.conjugation_defect[1])
        .with("k_points", problem.points.iter().map(|v| v.to_array()).collect::<Vec<_>>())
        .with("k_mesh", options.k_mesh)
        .with("dimension", options.boundary.axes.iter().filter(|&&axis| axis).count())
        .with("xc_ao_cutoff", options.xc_ao_cutoff.unwrap_or(overlap_cutoff));
    for (imaginary, orbital, density, fock, gradient) in [
        (
            false,
            "common_orbitals_real",
            "density_real",
            "fock_real",
            "orbital_gradient_real",
        ),
        (
            true,
            "common_orbitals_imag",
            "density_imag",
            "fock_imag",
            "orbital_gradient_imag",
        ),
    ] {
        report.set(
            orbital,
            coefficients
                .iter()
                .map(|m| matrix(if imaginary { &m.im } else { &m.re }))
                .collect::<Vec<_>>(),
        );
        report.set(
            density,
            state
                .density
                .each_ref()
                .map(|mesh| parts(mesh, imaginary))
                .to_vec(),
        );
        report.set(
            fock,
            state
                .fock
                .each_ref()
                .map(|mesh| parts(mesh, imaginary))
                .to_vec(),
        );
        report.set(gradient, parts(&state.orbital_gradient, imaginary));
        labels.extend([
            (orbital, "dimensionless"),
            (density, "dimensionless"),
            (fock, "hartree"),
            (gradient, "hartree"),
        ]);
    }
    if let Some(settings) = options.vv10 {
        append_nonlocal(
            &mut report,
            &mut labels,
            state.nonlocal_correlation_energy,
            settings,
        );
    }
    report.set("units", units(&labels));
    Ok((report, result))
}

fn base_report(
    reference: &str,
    functional: &ResolvedFunctional,
    xc_ao_cutoff: f64,
    problem: &GammaRoksProblem<'_>,
    result: &crate::periodic::roks::GammaRoksResult,
    labels: &mut Vec<(&'static str, &'static str)>,
    vv10: Option<crate::periodic::nonlocal::PeriodicVv10Options>,
) -> Value {
    let p = &result.state.density;
    let diagnostics = problem.diagnostics(&result.state);
    labels.extend([
        ("energy", "hartree per cell"),
        ("free_energy", "hartree per cell"),
        ("electronic_energy", "hartree per cell"),
        ("nuclear_energy", "hartree per cell"),
        ("xc_energy", "hartree per cell"),
        ("n_alpha", "dimensionless"),
        ("n_beta", "dimensionless"),
        ("scf_iterations", "dimensionless"),
        ("nuclear_response_order", "dimensionless"),
        ("orbital_residual", "hartree"),
        ("density_change", "dimensionless"),
        ("energy_change", "hartree"),
        ("metric_defect", "dimensionless"),
        ("particle_number_errors", "dimensionless"),
        ("idempotency_defects", "dimensionless"),
        ("closed_subspace_defect", "dimensionless"),
        ("common_orbitals", "dimensionless"),
        ("density_alpha", "dimensionless"),
        ("density_beta", "dimensionless"),
        ("fock_alpha", "hartree"),
        ("fock_beta", "hartree"),
        ("orbital_gradient", "hartree"),
        ("k_mesh", "dimensionless"),
        ("xc_ao_cutoff", "bohr"),
    ]);
    let mut report = Value::object()
        .with(
            "model",
            "Gamma periodic constrained common-spatial-orbital open-shell reference",
        )
        .with("reference_type", reference)
        .with("functional", functional.description.as_str())
        .with(
            "common_orbital_constraint",
            "closed beta subspace contained in alpha; common metric-orthonormal spatial orbitals",
        )
        .with(
            "algorithm",
            "core common orbitals and centered CDIIS; fresh physical final Focks",
        )
        .with("converged", true)
        .with("unrestricted", false)
        .with(
            "nuclear_response_available",
            problem.max_nuclear_response_order() > 0,
        )
        .with(
            "nuclear_response_order",
            problem.max_nuclear_response_order(),
        )
        .with("nuclear_hessian_available", false)
        .with("energy", result.state.energy)
        .with("free_energy", result.state.energy)
        .with("electronic_energy", result.state.electronic_energy)
        .with(
            "nuclear_energy",
            result.state.energy - result.state.electronic_energy,
        )
        .with("xc_energy", result.state.xc_energy)
        .with("n_alpha", result.n_alpha)
        .with("n_beta", result.n_beta)
        .with("scf_iterations", result.iterations)
        .with("orbital_residual", result.state.orbital_residual)
        .with("density_change", result.density_change)
        .with("energy_change", result.energy_change)
        .with("metric_defect", result.state.metric_defect)
        .with("particle_number_errors", diagnostics.particle_errors)
        .with("idempotency_defects", diagnostics.idempotency)
        .with("closed_subspace_defect", diagnostics.closed_subspace)
        .with("common_orbitals", matrix(&result.coefficients))
        .with("density_alpha", matrix(&p[0]))
        .with("density_beta", matrix(&p[1]))
        .with("fock_alpha", matrix(&result.state.fock[0]))
        .with("fock_beta", matrix(&result.state.fock[1]))
        .with("orbital_gradient", matrix(&result.state.orbital_gradient))
        .with("xc_ao_cutoff", xc_ao_cutoff)
        .with("k_mesh", vec![1usize; 3]);
    if let Some(settings) = vv10 {
        append_nonlocal(
            &mut report,
            labels,
            result.state.nonlocal_correlation_energy,
            settings,
        );
    }
    report
}

fn append_nonlocal(
    report: &mut Value,
    labels: &mut Vec<(&'static str, &'static str)>,
    energy: f64,
    settings: crate::periodic::nonlocal::PeriodicVv10Options,
) {
    report.set("nonlocal_correlation_energy", energy);
    labels.push(("nonlocal_correlation_energy", "hartree per cell"));
    super::periodic::append_vv10_metadata(report, settings);
}

fn fixed_cell(
    setup: super::periodic::PeriodicSetup,
    reference: &str,
    options: &crate::periodic::optimizer::PeriodicOptOptions,
    with_stress: bool,
) -> Result<Value> {
    let basis = super::method::build_basis(&setup.method.basis, setup.method.ecp.as_deref())?;
    let result = crate::periodic::optimizer::optimize_fixed_cell_roks(
        setup.cell.molecule(),
        setup.cell.lattice(),
        &basis,
        setup.method.overlap_cutoff,
        &setup.functional,
        &setup.options,
        options,
    )?;
    let final_setup = super::periodic::PeriodicSetup {
        method: setup.method,
        cell: result.cell,
        functional: setup.functional,
        options: setup.options,
    };
    let problem = GammaRoksProblem::new(
        &final_setup.cell,
        &final_setup.functional,
        &final_setup.options,
    )?;
    let dispersion = cell_dispersion(&final_setup.cell, &options.dispersion)?;
    let mut report = reference_report(
        &final_setup,
        reference,
        &problem,
        &result.scf,
        true,
        with_stress,
        dispersion.as_ref(),
    )?;
    report.set("objective", result.objective);
    report.set("optimization_converged", result.converged);
    report.set("iterations", result.iterations);
    report.set(
        "numbers",
        result
            .molecule
            .atoms
            .iter()
            .map(|a| a.z as usize)
            .collect::<Vec<_>>(),
    );
    report.set(
        "positions",
        result
            .molecule
            .atoms
            .iter()
            .map(|a| a.position.to_array())
            .collect::<Vec<_>>(),
    );
    report.set(
        "trajectory",
        Value::Array(
            result
                .trajectory
                .iter()
                .map(|frame| {
                    Value::object()
                        .with("step", frame.step)
                        .with("objective", frame.objective)
                        .with("grad_max", frame.grad_max)
                })
                .collect(),
        ),
    );
    if !result.converged {
        report.set(
            "warnings",
            vec![format!(
                "WARNING [opt-fixed-cell] not converged after {} steps (gtol {} hartree/bohr)",
                result.iterations, options.gtol,
            )],
        );
    }
    let labels = report.get("units").cloned().ok_or_else(|| {
        DftError::LinearAlgebra("common-orbital optimizer lost its reference unit report".into())
    })?;
    report.set(
        "units",
        labels
            .with("objective", "hartree per cell")
            .with("positions", "bohr")
            .with("iterations", "dimensionless")
            .with("trajectory.step", "dimensionless")
            .with("trajectory.objective", "hartree per cell")
            .with("trajectory.grad_max", "hartree/bohr"),
    );
    Ok(report)
}
