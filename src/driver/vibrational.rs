// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular second-derivative observables routed through the common driver:
//! the analytic Hessian, harmonic frequencies with RRHO thermochemistry,
//! analytic IR and Raman intensities, the static polarizability and analytic
//! multipole gradients.
//!
//! Every derivative here is analytic. A method combination whose analytic
//! second derivative is not implemented is refused with the missing term
//! named; no finite-difference fallback is used on this route.

use super::controls::Controls;
use super::method::{run_molecular_reference, MolecularReference, MolecularSetup};
use super::molecular::{moment_origin, raw_moments, reference_report, setup, units};
use super::value::Value;
use super::System;
use crate::constants::{atomic_mass_au, BOHR_TO_ANGSTROM};
use crate::hessian::analytic::{analytic_hessian_full, AnalyticHessianResult};
use crate::linalg::Matrix;
use crate::response::krylov::ResponseSolverConfig;
use crate::{DftError, Result};

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    if controls.contains("bfield")
        && matches!(id, "hessian" | "cubic-fc-along" | "quartic-fc-along")
    {
        return super::magnetizability::nuclear_surface(id, system, controls);
    }
    match id {
        "hessian" => hessian(system, controls),
        "frequencies" | "gibbs-free-energy" => frequencies(id, system, controls, false),
        "ir-raman" => frequencies(id, system, controls, true),
        "polarizability" => polarizability(system, controls),
        "multipole-gradient" => multipole_gradient(system, controls),
        "response-save" => response_save(system, controls),
        "third-derivative-2n1" => cubic_reduced(system, controls),
        "cubic-fc-along" | "quartic-fc-along" => force_constants_along(id, system, controls),
        other => Err(DftError::Unsupported {
            feature: format!("vibrational calculation `{other}`"),
            missing: "a common-driver implementation".into(),
        }),
    }
}

/// Response solver controls shared by every response calculation.
pub(super) fn response_config(controls: &Controls) -> Result<ResponseSolverConfig> {
    let d = ResponseSolverConfig::default();
    for (canonical, legacy) in [
        ("response_tolerance", "response_tol"),
        ("response_max_iterations", "response_max_iter"),
    ] {
        if controls.contains(canonical) && controls.contains(legacy) {
            return Err(DftError::InvalidInput(format!(
                "use either {canonical} or its compatibility alias {legacy}"
            )));
        }
    }
    let config = ResponseSolverConfig {
        tolerance: if controls.contains("response_tol") {
            controls.f64("response_tol", d.tolerance)?
        } else {
            controls.f64("response_tolerance", d.tolerance)?
        },
        max_iterations: if controls.contains("response_max_iter") {
            controls.usize("response_max_iter", d.max_iterations)?
        } else {
            controls.usize("response_max_iterations", d.max_iterations)?
        },
        gmres_restart: controls.usize("response_restart", d.gmres_restart)?,
        preconditioner_floor: controls
            .f64("response_preconditioner_floor", d.preconditioner_floor)?,
        accept_residual: controls.opt_f64("response_accept_residual")?,
        divergence_factor: controls.f64("response_divergence_factor", d.divergence_factor)?,
        min_denominator: controls.f64("response_min_denominator", d.min_denominator)?,
    };
    config.validate()?;
    Ok(config)
}

/// Refuse an orbital response whose smallest occupied-virtual gap is below
/// `response_min_denominator`, for the CPKS routes that do not pass through
/// the Krylov preconditioner check.
pub(super) fn check_orbital_denominators(
    scf: &crate::scf::ScfResult,
    config: &ResponseSolverConfig,
) -> Result<()> {
    if config.min_denominator == 0.0 {
        return Ok(());
    }
    let spins = if scf.unrestricted { 2 } else { 1 };
    for spin in 0..spins {
        let n_occ = if spin == 0 { scf.n_alpha } else { scf.n_beta };
        let energies = &scf.mo_energy[spin];
        if let (Some(&homo), Some(&lumo)) = (
            n_occ.checked_sub(1).and_then(|i| energies.get(i)),
            energies.get(n_occ),
        ) {
            let gap = lumo - homo;
            if gap < config.min_denominator {
                return Err(DftError::Unsupported {
                    feature: format!(
                        "orbital response with an occupied-virtual gap of {gap:.3e} Hartree"
                    ),
                    missing: format!(
                        "a resolved response; the gap is below response_min_denominator = {:.3e}",
                        config.min_denominator
                    ),
                });
            }
        }
    }
    Ok(())
}

pub(super) struct HessianData {
    pub(super) setup: MolecularSetup,
    pub(super) reference: MolecularReference,
    pub(super) analytic: AnalyticHessianResult,
    /// Electronic plus dispersion Hessian, Hartree/Bohr^2.
    pub(super) hessian: Matrix,
}

pub(super) fn compute_hessian(system: &System, controls: &Controls) -> Result<HessianData> {
    let setup = setup(system, controls)?;
    let reference = run_molecular_reference(&setup)?;
    let analytic = reference_hessian(&setup, &reference)?;
    let hessian = analytic.hessian.clone();
    Ok(HessianData {
        setup,
        reference,
        analytic,
        hessian,
    })
}

/// Reuse a converged molecular reference, including fractional occupations.
pub(super) fn reference_hessian(
    setup: &MolecularSetup,
    reference: &MolecularReference,
) -> Result<AnalyticHessianResult> {
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "analytic Hessian with the SMD non-electrostatic term".into(),
            missing: "the analytic CDS surface-tension second derivative".into(),
        });
    }
    if !reference.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: reference.scf.iterations,
            error: f64::NAN,
        });
    }
    let mut analytic = analytic_hessian_full(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        &reference.scf,
        setup.grid.as_ref(),
    )?;
    if setup.dispersion != crate::dispersion::Dispersion::None {
        let correction = crate::dispersion::dispersion_hessian(
            &setup.molecule,
            &setup.dispersion,
            &setup.method.dispersion_functional_name(),
        )?;
        for (total, delta) in analytic
            .hessian
            .as_mut_slice()
            .iter_mut()
            .zip(correction.as_slice())
        {
            *total += *delta;
        }
    }
    Ok(analytic)
}

/// Acoustic sum rule: in the absence of an external field, translating the
/// whole molecule leaves the gradient unchanged, so every row of the
/// Hessian sums to zero over atoms for each Cartesian axis.
fn translation_residual(hessian: &Matrix, natoms: usize) -> f64 {
    let mut worst = 0.0_f64;
    for row in 0..3 * natoms {
        for axis in 0..3 {
            let sum: f64 = (0..natoms).map(|b| hessian[(row, 3 * b + axis)]).sum();
            worst = worst.max(sum.abs());
        }
    }
    worst
}

fn matrix_rows(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|r| (0..m.cols).map(|c| m[(r, c)]).collect())
        .collect()
}

fn hessian(system: &System, controls: &Controls) -> Result<Value> {
    let data = compute_hessian(system, controls)?;
    controls.finish("hessian")?;
    let natoms = data.setup.molecule.len();
    let field = data.setup.options.external_field.is_some();
    let mut report = reference_report(
        &data.setup,
        &data.reference,
        "analytic Cartesian Hessian of the total energy (CPKS)",
    )
    .with("hessian", matrix_rows(&data.hessian))
    .with(
        "dipole_derivatives",
        matrix_rows(&data.analytic.dipole_derivatives),
    )
    .with("raw_asymmetry", data.analytic.raw_asymmetry)
    .with(
        "orbital_response_residuals",
        data.analytic.orbital_response_residuals,
    )
    .with(
        "ensemble_response_residuals",
        data.analytic.ensemble_response_residuals,
    )
    .with(
        "ensemble_number_defects",
        data.analytic.ensemble_number_defects,
    )
    .with(
        "translation_residual",
        if field {
            None
        } else {
            Some(translation_residual(&data.hessian, natoms))
        },
    )
    .with("provenance", "analytic")
    .with("symmetrized", true)
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("hessian", "hartree/bohr^2"),
            (
                "dipole_derivatives",
                "e (d mu_x / d R, rows x, columns 3*atom+axis)",
            ),
            ("raw_asymmetry", "hartree/bohr^2"),
            ("translation_residual", "hartree/bohr^2"),
        ]),
    );
    report.set(
        "axis_convention",
        "rows and columns 3*atom + axis; dipole_derivatives rows are dipole axes",
    );
    Ok(report)
}

/// One common term-by-term RRHO report on every public route.
pub(super) fn thermochemistry_value(
    t: &crate::properties::thermochemistry::Thermochemistry,
) -> Value {
    Value::object()
        .with("temperature_kelvin", t.temperature_kelvin)
        .with("pressure_pascal", t.pressure_pascal)
        .with("symmetry_number", t.symmetry_number)
        .with("electronic_degeneracy", t.electronic_degeneracy)
        .with("linear", t.is_linear)
        .with("zero_point_energy", t.zero_point_energy)
        .with("vibrational_thermal_energy", t.vibrational_thermal_energy)
        .with("translational_energy", t.translational_energy)
        .with("rotational_energy", t.rotational_energy)
        .with("internal_energy_correction", t.internal_energy_correction)
        .with("enthalpy_correction", t.enthalpy_correction)
        .with("gibbs_correction", t.gibbs_correction)
        .with("entropy", t.entropy)
        .with("entropy_translational", t.entropy_translational)
        .with("entropy_rotational", t.entropy_rotational)
        .with("entropy_vibrational", t.entropy_vibrational)
        .with("entropy_electronic", t.entropy_electronic)
        .with("internal_energy", t.internal_energy)
        .with("enthalpy", t.enthalpy)
        .with("gibbs_free_energy", t.gibbs_free_energy)
}

pub(super) fn thermochemistry_units() -> Vec<(&'static str, &'static str)> {
    vec![
        ("thermochemistry.temperature_kelvin", "K"),
        ("thermochemistry.pressure_pascal", "Pa"),
        ("thermochemistry.symmetry_number", "dimensionless"),
        ("thermochemistry.electronic_degeneracy", "dimensionless"),
        ("thermochemistry.zero_point_energy", "hartree per molecule"),
        (
            "thermochemistry.internal_energy_correction",
            "hartree per molecule",
        ),
        (
            "thermochemistry.enthalpy_correction",
            "hartree per molecule",
        ),
        ("thermochemistry.gibbs_correction", "hartree per molecule"),
        ("thermochemistry.entropy", "hartree/K per molecule"),
        ("thermochemistry.internal_energy", "hartree per molecule"),
        ("thermochemistry.enthalpy", "hartree per molecule"),
        ("thermochemistry.gibbs_free_energy", "hartree per molecule"),
        (
            "thermochemistry.vibrational_thermal_energy",
            "hartree per molecule",
        ),
        (
            "thermochemistry.translational_energy",
            "hartree per molecule",
        ),
        ("thermochemistry.rotational_energy", "hartree per molecule"),
        (
            "thermochemistry.entropy_translational",
            "hartree/K per molecule",
        ),
        (
            "thermochemistry.entropy_rotational",
            "hartree/K per molecule",
        ),
        (
            "thermochemistry.entropy_vibrational",
            "hartree/K per molecule",
        ),
        (
            "thermochemistry.entropy_electronic",
            "hartree/K per molecule",
        ),
    ]
}

fn frequencies(id: &str, system: &System, controls: &Controls, raman: bool) -> Result<Value> {
    let config = if raman {
        Some(response_config(controls)?)
    } else {
        None
    };
    let temperature = controls.opt_f64("temperature")?;
    let pressure = controls.f64("pressure", 101_325.0)?;
    let symmetry_number = controls.usize("symmetry_number", 1)?;
    let raman_frequency = if raman {
        controls.opt_f64("raman_frequency")?
    } else {
        None
    };
    let data = compute_hessian(system, controls)?;
    controls.finish(id)?;
    let temperature = temperature.unwrap_or(if data.setup.method.electronic_temperature > 0.0 {
        data.setup.method.electronic_temperature
    } else {
        298.15
    });
    let molecule = &data.setup.molecule;
    let modes = crate::properties::vibrational::vibrational_analysis(
        molecule,
        &data.hessian,
        Some(&data.analytic.dipole_derivatives),
    )?;
    let natoms = molecule.len();
    // Cartesian displacement patterns (Bohr per sqrt(amu)*Bohr of Q).
    let sqrt_m_amu: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| (atomic_mass_au(a.z) / crate::constants::AMU_TO_AU_MASS).sqrt())
        .collect();
    let mut cartesian = Vec::with_capacity(modes.modes.cols);
    for k in 0..modes.modes.cols {
        let pattern: Vec<[f64; 3]> = (0..natoms)
            .map(|a| std::array::from_fn(|ax| modes.modes[(3 * a + ax, k)] / sqrt_m_amu[a]))
            .collect();
        cartesian.push(pattern);
    }
    let mut warnings = Vec::new();
    let thermo = if data.reference.scf.fixed_occupation_residual.is_some() {
        if id == "gibbs-free-energy" {
            return Err(DftError::InvalidInput(
                "equilibrium Gibbs thermochemistry requires thermal equilibrium populations; prescribed nonthermal populations define a constrained energy surface".into(),
            ));
        }
        warnings.push("WARNING [frequencies] fixed nonthermal orbital populations define a constrained energy surface; equilibrium RRHO thermochemistry is withheld".into());
        None
    } else if modes.n_imaginary == 0 {
        let electronic_degeneracy = if data.reference.scf.unrestricted {
            data.reference.scf.n_alpha - data.reference.scf.n_beta + 1
        } else {
            1
        };
        Some(
            crate::properties::thermochemistry::thermochemistry_from_mermin(
                molecule,
                &modes.frequencies_cm,
                data.reference.total_energy(),
                data.setup.method.electronic_temperature,
                data.reference.scf.electronic_entropy,
                crate::properties::thermochemistry::ThermochemistryOptions {
                    temperature_kelvin: temperature,
                    pressure_pascal: pressure,
                    symmetry_number,
                    electronic_degeneracy,
                },
            )?,
        )
    } else {
        if id == "gibbs-free-energy" {
            return Err(DftError::InvalidInput(format!(
                "Gibbs RRHO thermochemistry requires a minimum with real vibrational frequencies; {} imaginary mode(s) remain",
                modes.n_imaginary
            )));
        }
        warnings.push(format!(
            "WARNING [frequencies] {} imaginary mode(s); RRHO thermochemistry is withheld",
            modes.n_imaginary
        ));
        None
    };
    let mut report = reference_report(
        &data.setup,
        &data.reference,
        "harmonic analysis of the analytic Hessian with Eckart projection",
    )
    .with("frequencies_cm1", modes.frequencies_cm.clone())
    .with("ir_intensities_km_mol", modes.ir_intensities.clone())
    .with("n_imaginary", modes.n_imaginary)
    .with("n_projected", modes.n_projected)
    .with("normal_modes_cartesian", cartesian.clone())
    .with("raw_asymmetry", data.analytic.raw_asymmetry)
    .with(
        "orbital_response_residuals",
        data.analytic.orbital_response_residuals,
    )
    .with(
        "ensemble_response_residuals",
        data.analytic.ensemble_response_residuals,
    )
    .with(
        "ensemble_number_defects",
        data.analytic.ensemble_number_defects,
    )
    .with("provenance", "analytic");
    if let Some(t) = &thermo {
        report.set("gibbs_free_energy", t.gibbs_free_energy);
        report.set(
            "free_energy_kind",
            "gibbs; ideal-gas RRHO, G = H - T S, including pV = k_B T per molecule",
        );
        report.set("thermochemistry", thermochemistry_value(t));
    } else {
        report.set("thermochemistry", Value::Null);
    }
    let mut unit_pairs = vec![
        ("energy", "hartree"),
        ("gibbs_free_energy", "hartree per molecule"),
        ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
        ("raw_asymmetry", "hartree/bohr^2"),
        ("ir_intensities_km_mol", "km/mol"),
        (
            "normal_modes_cartesian",
            "1/sqrt(amu), Cartesian displacement per unit mass-weighted normal coordinate",
        ),
    ];
    unit_pairs.extend(thermochemistry_units());
    if raman {
        let mut activities = Vec::with_capacity(cartesian.len());
        let mut depolarization = Vec::with_capacity(cartesian.len());
        let mut derivatives = Vec::with_capacity(cartesian.len());
        let mut permutation_defect = 0.0_f64;
        let mut converged = true;
        for pattern in &cartesian {
            let tensor = crate::response::nuclear::static_spectroscopic_derivative(
                molecule,
                &data.setup.basis,
                &data.setup.functional,
                data.setup.grid.as_ref(),
                &data.reference.scf,
                &data.setup.options,
                pattern,
                2,
                config.as_ref().expect("Raman response controls"),
            )?;
            converged &= tensor.converged;
            warnings.extend(tensor.warnings.iter().cloned());
            permutation_defect = permutation_defect.max(tensor.value.permutation_defect);
            let t = &tensor.value.tensor;
            // d alpha / dQ in Angstrom^2 / sqrt(amu).
            let conv = BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM;
            let d = |a: usize, b: usize| 0.5 * (t[3 * a + b] + t[3 * b + a]) * conv;
            let mean = (d(0, 0) + d(1, 1) + d(2, 2)) / 3.0;
            let gamma2 = 0.5
                * ((d(0, 0) - d(1, 1)).powi(2)
                    + (d(1, 1) - d(2, 2)).powi(2)
                    + (d(2, 2) - d(0, 0)).powi(2))
                + 3.0 * (d(0, 1).powi(2) + d(1, 2).powi(2) + d(0, 2).powi(2));
            let activity = 45.0 * mean * mean + 7.0 * gamma2;
            activities.push(activity);
            let denominator = 45.0 * mean * mean + 4.0 * gamma2;
            depolarization.push(if denominator > 0.0 {
                Some(3.0 * gamma2 / denominator)
            } else {
                None
            });
            derivatives.push(t.iter().map(|v| v * conv).collect::<Vec<_>>());
        }
        report.set("raman_activities", activities);
        report.set("depolarization_ratios", depolarization);
        report.set("polarizability_derivatives", derivatives);
        report.set("raman_permutation_defect", permutation_defect);
        report.set("response_converged", converged);
        unit_pairs.push(("raman_activities", "angstrom^4/amu"));
        unit_pairs.push((
            "polarizability_derivatives",
            "angstrom^2/sqrt(amu), flat [a][b] with d alpha_ab/dQ",
        ));
        // The same activities with the polarizability at a photon energy:
        // first Taylor coefficient of the dynamic polarizability along
        // every normal coordinate.
        if let Some(omega) = raman_frequency {
            if !(omega.is_finite() && omega > 0.0) {
                return Err(DftError::InvalidInput(
                    "raman_frequency is a positive photon energy in hartree".into(),
                ));
            }
            let mut dynamic_activities = Vec::with_capacity(cartesian.len());
            let mut dynamic_depolarization = Vec::with_capacity(cartesian.len());
            let mut dynamic_derivatives = Vec::with_capacity(cartesian.len());
            let mut dynamic_residual = 0.0_f64;
            let mut dynamic_converged = true;
            for pattern in &cartesian {
                let path = crate::response::nuclear::optical_tensor_path(
                    molecule,
                    &data.setup.basis,
                    &data.setup.functional,
                    data.setup.grid.as_ref(),
                    &data.reference.scf,
                    &data.setup.options,
                    pattern,
                    1,
                    omega,
                    crate::math::Vec3::zero(),
                    false,
                    config.as_ref().expect("Raman response controls"),
                )?;
                dynamic_converged &= path.converged;
                dynamic_residual = path
                    .residuals
                    .iter()
                    .copied()
                    .fold(dynamic_residual, f64::max);
                for warning in &path.warnings {
                    if !warnings.contains(warning) {
                        warnings.push(warning.clone());
                    }
                }
                let t = &path.polarizability[1];
                let conv = BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM;
                let d = |a: usize, b: usize| 0.5 * (t[a][b] + t[b][a]) * conv;
                let mean = (d(0, 0) + d(1, 1) + d(2, 2)) / 3.0;
                let gamma2 = 0.5
                    * ((d(0, 0) - d(1, 1)).powi(2)
                        + (d(1, 1) - d(2, 2)).powi(2)
                        + (d(2, 2) - d(0, 0)).powi(2))
                    + 3.0 * (d(0, 1).powi(2) + d(1, 2).powi(2) + d(0, 2).powi(2));
                dynamic_activities.push(45.0 * mean * mean + 7.0 * gamma2);
                let denominator = 45.0 * mean * mean + 4.0 * gamma2;
                dynamic_depolarization.push(if denominator > 0.0 {
                    Some(3.0 * gamma2 / denominator)
                } else {
                    None
                });
                dynamic_derivatives.push(t.iter().flatten().map(|v| v * conv).collect::<Vec<_>>());
            }
            report.set("raman_frequency", omega);
            report.set("dynamic_raman_activities", dynamic_activities);
            report.set("dynamic_depolarization_ratios", dynamic_depolarization);
            report.set("dynamic_polarizability_derivatives", dynamic_derivatives);
            report.set("dynamic_response_residual", dynamic_residual);
            report.set("dynamic_response_converged", dynamic_converged);
            unit_pairs.push(("raman_frequency", "hartree"));
            unit_pairs.push(("dynamic_raman_activities", "angstrom^4/amu"));
            unit_pairs.push(("dynamic_depolarization_ratios", "dimensionless"));
            unit_pairs.push((
                "dynamic_polarizability_derivatives",
                "angstrom^2/sqrt(amu), flat [a][b] with d alpha_ab(omega)/dQ",
            ));
            unit_pairs.push(("dynamic_response_residual", "dimensionless"));
        }
    }
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        all.extend(warnings.into_iter().map(Value::Str));
        report.set("warnings", Value::Array(all));
    }
    report.set("units", units(&unit_pairs));
    Ok(report)
}

fn polarizability(system: &System, controls: &Controls) -> Result<Value> {
    if super::double_hybrid::requested(controls)? {
        return super::double_hybrid::polarizability(system, controls);
    }
    let config = response_config(controls)?;
    let setup = setup(system, controls)?;
    controls.finish("polarizability")?;
    let reference = run_molecular_reference(&setup)?;
    if reference.scf.fixed_occupation_residual.is_some()
        || reference.scf.electronic_temperature > 0.0
    {
        let solver = crate::response::ensemble::EnsembleSolver::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
        )?;
        let response = solver.polarizability(&config)?;
        let mut report = polarizability_report_fields(
            reference_report(&setup, &reference,
                "static electronic dipole polarizability from the full spectral population response"),
            response.tensor,
        );
        report.set("response_population_model", solver.population_model());
        report.set(
            "response_residuals",
            response
                .responses
                .iter()
                .map(|r| r.relative_residual)
                .collect::<Vec<_>>(),
        );
        report.set(
            "response_iterations",
            response
                .responses
                .iter()
                .map(|r| r.iterations)
                .collect::<Vec<_>>(),
        );
        report.set(
            "number_defects",
            response
                .responses
                .iter()
                .map(|r| r.number_defects)
                .collect::<Vec<_>>(),
        );
        let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
        labels.set("response_residuals", "dimensionless");
        labels.set(
            "number_defects",
            "e*bohr/hartree, electron-count derivative per electric field",
        );
        report.set("units", labels);
        return Ok(report);
    }
    check_orbital_denominators(&reference.scf, &config)?;
    // alpha_xy = -Tr[dP_y D_x] for H' = +F_y D_y. The CPKS residual bound is
    // the response tolerance (absolute, per amplitude).
    let mut solver = crate::response::ResponseSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
    )?;
    solver.tol = config.tolerance.min(config.acceptance());
    solver.max_iter = config.max_iterations;
    let dipoles = crate::integrals::dipole_matrices(&setup.basis, crate::math::Vec3::zero());
    let responses = solver.solve_ao_block_report(&dipoles)?;
    let residuals: Vec<_> = responses.iter().map(|(_, residual)| *residual).collect();
    let largest = residuals.iter().copied().fold(0.0, f64::max);
    if largest > config.acceptance() {
        return Err(DftError::LinearAlgebra(format!(
            "static dipole CPKS fails the fresh orbital-equation check: residual {largest:.6e} exceeds {:.6e}",
            config.acceptance()
        )));
    }
    let mut alpha = [[0.0; 3]; 3];
    for (y, (dp, _)) in responses.iter().enumerate() {
        for (x, dipole) in dipoles.iter().enumerate() {
            alpha[x][y] = -(dp[0].frobenius_dot(dipole) + dp[1].frobenius_dot(dipole));
        }
    }
    let mut report = polarizability_report_fields(
        reference_report(
            &setup,
            &reference,
            "static electronic dipole polarizability from analytic CPKS",
        ),
        alpha,
    );
    report.set("response_absolute_residuals", residuals);
    let mut labels = report
        .get("units")
        .cloned()
        .expect("static polarizability units");
    labels.set("response_absolute_residuals", "e*bohr");
    report.set("units", labels);
    Ok(report)
}

pub(super) fn polarizability_report_fields(report: Value, alpha: [[f64; 3]; 3]) -> Value {
    let isotropic = (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0;
    let mut asymmetry = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            asymmetry = asymmetry.max((alpha[a][b] - alpha[b][a]).abs());
        }
    }
    let s = |a: usize, b: usize| 0.5 * (alpha[a][b] + alpha[b][a]);
    let anisotropy = (0.5
        * ((s(0, 0) - s(1, 1)).powi(2)
            + (s(1, 1) - s(2, 2)).powi(2)
            + (s(2, 2) - s(0, 0)).powi(2))
        + 3.0 * (s(0, 1).powi(2) + s(1, 2).powi(2) + s(0, 2).powi(2)))
    .sqrt();
    let to_a3 = BOHR_TO_ANGSTROM.powi(3);
    let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
    for (key, label) in [
        ("energy", "hartree"),
        (
            "polarizability",
            "e^2*bohr^2/hartree, alpha_ab = d mu_a / d F_b",
        ),
        ("polarizability_angstrom3", "angstrom^3"),
        ("raw_asymmetry", "e^2*bohr^2/hartree"),
    ] {
        labels.set(key, label);
    }
    report
        .with("polarizability", alpha.to_vec())
        .with("isotropic", isotropic)
        .with("anisotropy", anisotropy)
        .with("raw_asymmetry", asymmetry)
        .with(
            "polarizability_angstrom3",
            alpha
                .iter()
                .map(|r| r.map(|v| v * to_a3))
                .collect::<Vec<_>>(),
        )
        .with("isotropic_angstrom3", isotropic * to_a3)
        .with("provenance", "analytic")
        .with("units", labels)
}

/// Analytic `dM/dR` of every Cartesian multipole moment of the requested
/// ranks: nuclear term, relaxed CPKS density derivative contracted with the
/// moment integrals, moving-basis integral derivative and, for a
/// mass/charge-weighted origin, the origin motion.
fn multipole_gradient(system: &System, controls: &Controls) -> Result<Value> {
    let max_rank = controls.usize("max_rank", 2)?;
    if max_rank > 8 {
        return Err(DftError::InvalidInput(
            "max_rank must be between 0 and 8".into(),
        ));
    }
    let base = setup(system, controls)?;
    let (origin, origin_name) = moment_origin(controls, &base)?;
    controls.finish("multipole-gradient")?;
    drop(base);
    let data = compute_hessian(system, controls)?;
    let setup = &data.setup;
    let molecule = &setup.molecule;
    let natoms = molecule.len();
    let charges = setup.basis.effective_charges(molecule);
    let p = data.reference.scf.p_total();
    // d origin / d R_(atom, axis) weight for the chosen convention.
    let origin_weights: Vec<f64> = match origin_name.as_str() {
        "center_of_mass" => {
            let m: Vec<f64> = molecule.atoms.iter().map(|a| atomic_mass_au(a.z)).collect();
            let total: f64 = m.iter().sum();
            m.iter().map(|v| v / total).collect()
        }
        "center_of_nuclear_charge" => {
            let z: Vec<f64> = molecule.atoms.iter().map(|a| a.z as f64).collect();
            let total: f64 = z.iter().sum();
            z.iter().map(|v| v / total).collect()
        }
        _ => vec![0.0; natoms],
    };
    let mut ranks = Vec::new();
    let mut lower_moments: Vec<((i32, i32, i32), f64)> = Vec::new();
    for n in 0..=max_rank {
        let moments = raw_moments(setup, &data.reference, origin, n as u8);
        let lookup = |e: (i32, i32, i32)| -> f64 {
            if e.0 < 0 || e.1 < 0 || e.2 < 0 {
                return 0.0;
            }
            lower_moments
                .iter()
                .find(|(k, _)| *k == e)
                .map(|(_, v)| *v)
                .unwrap_or(0.0)
        };
        let mut gradient = vec![vec![0.0; 3 * natoms]; moments.len()];
        let moment_matrices =
            crate::integrals::one_electron::moment_matrices(&setup.basis, origin, n as u8);
        for (c, ((e, f, g), _)) in moments.iter().enumerate() {
            let power = (*e, *f, *g);
            let integral_derivatives =
                crate::integrals::deriv_mat::moving_center_derivative_matrices(
                    &setup.basis,
                    natoms,
                    |a, b| crate::integrals::one_electron::moment_pair_block(a, b, origin, power),
                );
            let moment_matrix = &moment_matrices[c].1;
            debug_assert_eq!(moment_matrices[c].0, power);
            for atom in 0..natoms {
                let r = molecule.atoms[atom].position - origin;
                let rv = [r.x, r.y, r.z];
                let exps = [*e, *f, *g];
                for axis in 0..3 {
                    let x = 3 * atom + axis;
                    // Nuclear term d/dR_(atom,axis) of Z (R-O)^power.
                    let mut nuclear = 0.0;
                    if exps[axis] > 0 {
                        let mut value = charges[atom] * exps[axis] as f64;
                        for k in 0..3 {
                            let power_k = exps[k] - if k == axis { 1 } else { 0 };
                            value *= rv[k].powi(power_k);
                        }
                        nuclear = value;
                    }
                    let dp = &data.analytic.density_derivatives[x];
                    let mut response = 0.0;
                    for (pa, pb, m) in dp[0]
                        .as_slice()
                        .iter()
                        .zip(dp[1].as_slice())
                        .zip(moment_matrix.as_slice())
                        .map(|((a, b), m)| (a, b, m))
                    {
                        response += (pa + pb) * m;
                    }
                    let moving = p.frobenius_dot(&integral_derivatives[atom][axis]);
                    // Origin motion: dM/dO_axis = -e_axis M(power - 1_axis).
                    let mut lowered = power;
                    match axis {
                        0 => lowered.0 -= 1,
                        1 => lowered.1 -= 1,
                        _ => lowered.2 -= 1,
                    }
                    let origin_term = if origin_weights[atom] != 0.0 && exps[axis] > 0 {
                        -(exps[axis] as f64) * lookup(lowered) * origin_weights[atom]
                    } else {
                        0.0
                    };
                    gradient[c][x] = nuclear - response - moving + origin_term;
                }
            }
        }
        ranks.push(
            Value::object()
                .with("rank", n)
                .with(
                    "monomials",
                    moments
                        .iter()
                        .map(|((e, f, g), _)| [*e, *f, *g])
                        .collect::<Vec<_>>(),
                )
                .with(
                    "cartesian",
                    moments.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                )
                .with("gradient", gradient),
        );
        lower_moments = moments;
    }
    Ok(reference_report(
        setup,
        &data.reference,
        "analytic nuclear derivatives of molecular multipole moments (relaxed CPKS density)",
    )
    .with("origin", origin.to_array())
    .with("origin_convention", origin_name)
    .with("ranks", ranks)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("ranks.cartesian", "e*bohr^rank"),
            (
                "ranks.gradient",
                "e*bohr^(rank-1); rows monomials, columns 3*atom+axis",
            ),
            ("origin", "bohr"),
        ]),
    ))
}

/// All Cartesian third derivatives of the energy from first-order responses
/// by the `2n + 1` rule: no second-order response equation is solved.
fn cubic_reduced(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let setup = setup(system, controls)?;
    controls.finish("third-derivative-2n1")?;
    let reference = run_molecular_reference(&setup)?;
    if !reference.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: reference.scf.iterations,
            error: f64::NAN,
        });
    }
    let result = crate::response::nuclear::cubic_force_constants_reduced(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        &config,
    )?;
    let n = result.dimension;
    let nested: Vec<Vec<Vec<f64>>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).map(|k| result.tensor[(i * n + j) * n + k]).collect())
                .collect()
        })
        .collect();
    // Translating the molecule leaves the Hessian unchanged, so the tensor
    // summed over the atoms of one index vanishes without an external field.
    let mut translation = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            for axis in 0..3 {
                let sum: f64 = (0..n / 3)
                    .map(|atom| result.tensor[(i * n + j) * n + 3 * atom + axis])
                    .sum();
                translation = translation.max(sum.abs());
            }
        }
    }
    let mut report = reference_report(
        &setup,
        &reference,
        "Cartesian third derivatives of the energy from first-order responses (2n + 1 rule)",
    )
    .with("cubic_force_constants", nested)
    .with("path_evaluations", result.paths)
    .with("accumulation_roundoff", result.accumulation_roundoff)
    .with(
        "translation_residual",
        if setup.options.external_field.is_some() {
            None
        } else {
            Some(translation)
        },
    )
    .with("response_converged", result.converged)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            (
                "cubic_force_constants",
                "hartree/bohr^3, [i][j][k] with coordinates 3*atom + axis",
            ),
            ("accumulation_roundoff", "hartree/bohr^3"),
            ("translation_residual", "hartree/bohr^3"),
        ]),
    );
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        all.extend(result.warnings.into_iter().map(Value::Str));
        report.set("warnings", Value::Array(all));
    }
    Ok(report)
}

/// Cubic or quartic force constants contracted with caller displacement
/// patterns on their leading indices: the derivative of the Hessian along
/// `probe`, or its second derivative along `probe` and `second_probe`. No
/// full tensor is formed; a normal mode as the probe gives the coupling of
/// that mode to every pair of coordinates.
fn force_constants_along(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    if super::double_hybrid::requested(controls)? {
        return super::correlated_high_order::nuclear(id, system, controls);
    }
    let quartic = id == "quartic-fc-along";
    let config = response_config(controls)?;
    let probe = controls.opt_f64_list("probe")?.ok_or_else(|| {
        DftError::InvalidInput(format!(
            "{id} needs probe: one displacement value per Cartesian coordinate (3*atom + axis)"
        ))
    })?;
    let second = if quartic {
        controls.opt_f64_list("second_probe")?
    } else {
        None
    };
    let setup = setup(system, controls)?;
    controls.finish(id)?;
    let reference = run_molecular_reference(&setup)?;
    if !reference.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: reference.scf.iterations,
            error: f64::NAN,
        });
    }
    let result = if quartic {
        let second = second.unwrap_or_else(|| probe.clone());
        crate::response::nuclear::quartic_force_constants_along(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            &probe,
            &second,
            &config,
        )?
    } else {
        crate::response::nuclear::cubic_force_constants_along(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            &probe,
            &config,
        )?
    };
    let mut report = reference_report(
        &setup,
        &reference,
        if quartic {
            "quartic force constants contracted with two displacement patterns, from responses \
             through second order"
        } else {
            "cubic force constants contracted with one displacement pattern, from first-order \
             responses (2n + 1 rule)"
        },
    )
    .with("probe", probe.clone())
    .with("contracted_force_constants", matrix_rows(&result.matrix))
    .with("response_residuals", result.response_residuals.clone())
    .with("path_evaluations", result.paths)
    .with("accumulation_roundoff", result.accumulation_roundoff)
    .with("response_converged", result.converged)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("probe", "dimensionless"),
            ("response_residuals", "dimensionless"),
            (
                "contracted_force_constants",
                if quartic {
                    "hartree/bohr^4 per unit of both patterns, rows and columns 3*atom + axis"
                } else {
                    "hartree/bohr^3 per unit pattern, rows and columns 3*atom + axis"
                },
            ),
            (
                "accumulation_roundoff",
                if quartic {
                    "hartree/bohr^4"
                } else {
                    "hartree/bohr^3"
                },
            ),
        ]),
    );
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        all.extend(result.warnings.into_iter().map(Value::Str));
        report.set("warnings", Value::Array(all));
    }
    Ok(report)
}

/// Static first-order response quantities of the reference for reuse:
/// orbital rotations `U^F` and relaxed densities `dP/dF` for a uniform field
/// along each axis (`H' = +F r`), and relaxed nuclear density derivatives
/// `dP/dR` from the analytic Hessian's CPKS solves.
fn response_save(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let path = controls.opt_str("path")?;
    let nuclear = controls.bool("nuclear", true)?;
    let frequencies = controls.opt_f64_list("frequencies")?;
    let damping = controls.f64("frequency_damping", 0.0)?;
    let data = if nuclear {
        Some(compute_hessian(system, controls)?)
    } else {
        None
    };
    let owned = match data {
        Some(_) => None,
        None => {
            let s = setup(system, controls)?;
            let r = run_molecular_reference(&s)?;
            Some((s, r))
        }
    };
    controls.finish("response-save")?;
    let (setup, reference) = match (&data, &owned) {
        (Some(d), _) => (&d.setup, &d.reference),
        (None, Some((s, r))) => (s, r),
        (None, None) => unreachable!("one reference is always built"),
    };
    check_orbital_denominators(&reference.scf, &config)?;
    let mut solver = crate::response::ResponseSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
    )?;
    solver.tol = config.tolerance;
    solver.max_iter = config.max_iterations;
    let (n_occ, n_vir) = solver.ov_dims();
    let spins = if reference.scf.unrestricted { 2 } else { 1 };
    let length: usize = (0..spins).map(|s| n_occ[s] * n_vir[s]).sum();
    let dipoles = crate::integrals::dipole_matrices(&setup.basis, crate::math::Vec3::zero());
    let mut field = Vec::new();
    let mut alpha = [[0.0; 3]; 3];
    for (axis, dipole) in dipoles.iter().enumerate() {
        let mut b = vec![0.0; length];
        solver.project_ov([dipole, dipole], &mut b);
        let u = solver.solve_ov(&b)?;
        let dp = solver.delta_density(&u);
        for (x, d) in dipoles.iter().enumerate() {
            alpha[x][axis] = -(dp[0].frobenius_dot(d) + dp[1].frobenius_dot(d));
        }
        let mut rotations = Vec::new();
        let mut offset = 0;
        for spin in 0..spins {
            let block: Vec<Vec<f64>> = (0..n_vir[spin])
                .map(|a| {
                    (0..n_occ[spin])
                        .map(|i| u[offset + a * n_occ[spin] + i])
                        .collect()
                })
                .collect();
            rotations.push(block);
            offset += n_occ[spin] * n_vir[spin];
        }
        field.push(
            Value::object()
                .with("axis", ["x", "y", "z"][axis])
                .with("orbital_rotations", rotations)
                .with("density_derivative", vec![dp[0].clone(), dp[1].clone()]),
        );
    }
    let mut report = reference_report(setup, reference, "first-order static response derivatives of the reference (CPKS)")
        .with("field", field)
        .with("polarizability", alpha.to_vec())
        .with(
            "conventions",
            Value::object()
                .with("field", "H' = +F r; dP/dF per spin [alpha, beta]; U^F[spin][virtual][occupied] with dP = C_v U C_o^T + transpose")
                .with("nuclear", "dP/dR per Cartesian coordinate 3*atom+axis, per spin, AO functions moving with their atoms"),
        );
    if let Some(d) = &data {
        report.set(
            "nuclear_density_derivatives",
            d.analytic
                .density_derivatives
                .iter()
                .map(|p| vec![p[0].clone(), p[1].clone()])
                .collect::<Vec<_>>(),
        );
    }
    // Amplitudes of the frequency-dependent field response.
    if let Some(list) = &frequencies {
        if list.iter().any(|w| !w.is_finite()) || !(damping.is_finite() && damping >= 0.0) {
            return Err(DftError::InvalidInput(
                "frequencies are finite photon energies and frequency_damping is not negative"
                    .into(),
            ));
        }
        let blocks =
            |values: &[crate::response::complex::Complex], imaginary: bool| -> Vec<Vec<Vec<f64>>> {
                let mut out = Vec::new();
                let mut offset = 0;
                for spin in 0..spins {
                    let block: Vec<Vec<f64>> = (0..n_vir[spin])
                        .map(|a| {
                            (0..n_occ[spin])
                                .map(|i| {
                                    let z = values[offset + a * n_occ[spin] + i];
                                    if imaginary {
                                        z.im
                                    } else {
                                        z.re
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    out.push(block);
                    offset += n_occ[spin] * n_vir[spin];
                }
                out
            };
        let mut dynamic = Vec::with_capacity(list.len());
        for &omega in list {
            let result = crate::response::dynamic::dynamic_amplitudes(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                &setup.options,
                crate::response::dynamic::Frequency::Retarded { omega, damping },
                &config,
            )?;
            let fields: Vec<Value> = (0..3)
                .map(|axis| {
                    Value::object()
                        .with("axis", ["x", "y", "z"][axis])
                        .with("x_plus_y_real", blocks(&result.sum[axis], false))
                        .with("x_plus_y_imag", blocks(&result.sum[axis], true))
                        .with("x_minus_y_real", blocks(&result.difference[axis], false))
                        .with("x_minus_y_imag", blocks(&result.difference[axis], true))
                })
                .collect();
            dynamic.push(
                Value::object()
                    .with("frequency", omega)
                    .with("damping", damping)
                    .with("field", fields)
                    .with(
                        "polarizability_real",
                        result.alpha.map(|row| row.map(|z| z.re)).to_vec(),
                    )
                    .with(
                        "polarizability_imag",
                        result.alpha.map(|row| row.map(|z| z.im)).to_vec(),
                    )
                    .with("residuals", result.residuals.to_vec()),
            );
        }
        report.set("dynamic", dynamic);
        report.set(
            "dynamic_conventions",
            "H' = +F r exp(-i (omega + i damping) t); amplitudes [spin][virtual][occupied] of \
             X + Y and X - Y, one spin block for a restricted reference; at zero frequency \
             X + Y is twice the static orbital rotation and X - Y vanishes",
        );
    }
    let mut unit_pairs = vec![
        ("energy", "hartree"),
        (
            "field.density_derivative",
            "electrons per AO element per a.u. field",
        ),
        (
            "nuclear_density_derivatives",
            "electrons/bohr per AO element",
        ),
        ("polarizability", "bohr^3"),
    ];
    if frequencies.is_some() {
        let amplitude = "per a.u. field, [spin][virtual][occupied]";
        unit_pairs.extend([
            ("dynamic.frequency", "hartree"),
            ("dynamic.damping", "hartree"),
            ("dynamic.field.x_plus_y_real", amplitude),
            ("dynamic.field.x_plus_y_imag", amplitude),
            ("dynamic.field.x_minus_y_real", amplitude),
            ("dynamic.field.x_minus_y_imag", amplitude),
            ("dynamic.polarizability_real", "bohr^3"),
            ("dynamic.polarizability_imag", "bohr^3"),
            ("dynamic.residuals", "dimensionless"),
        ]);
    }
    report.set("units", units(&unit_pairs));
    if let Some(path) = path {
        let mut written = report.clone();
        if let Value::Object(entries) = &mut written {
            entries.insert(0, ("feature".into(), Value::Str("response-save".into())));
            entries.insert(
                1,
                (
                    "version".into(),
                    Value::Str(env!("CARGO_PKG_VERSION").into()),
                ),
            );
        }
        std::fs::write(&path, written.to_json_pretty())?;
        report.set("path", path);
    }
    Ok(report)
}
