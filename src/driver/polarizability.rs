// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common frequency-resolved molecular susceptibility and reference selection.
use super::method::{run_molecular_reference, MolecularReference};
use super::molecular::{reference_report, setup};
use super::{Controls, System, Value};
use crate::response::complex::Complex;
use crate::response::spinor::SpinorPolarizability;
use crate::response::Frequency;
use crate::{DftError, Result};

const DIPOLE_RESPONSE_UNIT: &str = "e^2*bohr^2/hartree";

pub(super) fn frequency(controls: &Controls) -> Result<Frequency> {
    for (key, alias) in [
        ("frequency", "response_omega"),
        ("frequency_damping", "response_damping"),
        ("imaginary", "imaginary_frequency"),
    ] {
        if controls.contains(key) && controls.contains(alias) {
            return Err(DftError::InvalidInput(format!(
                "use either {key} or {alias}"
            )));
        }
    }
    let omega = if controls.contains("response_omega") {
        controls.f64("response_omega", 0.0)?
    } else {
        controls.f64("frequency", 0.0)?
    };
    let damping = if controls.contains("response_damping") {
        controls.f64("response_damping", 0.0)?
    } else {
        controls.f64("frequency_damping", 0.0)?
    };
    let imaginary = if controls.contains("imaginary_frequency") {
        controls.bool("imaginary_frequency", false)?
    } else {
        controls.bool("imaginary", false)?
    };
    let frequency = if imaginary {
        if damping != 0.0 {
            return Err(DftError::InvalidInput(
                "damping is not an imaginary-axis parameter".into(),
            ));
        }
        Frequency::Imaginary { frequency: omega }
    } else if omega == 0.0 && damping == 0.0 {
        Frequency::Static
    } else {
        Frequency::Retarded { omega, damping }
    };
    frequency.validate()?;
    Ok(frequency)
}

pub(super) fn tensor_report(report: &mut Value, value: &SpinorPolarizability) {
    tensor_values(
        report,
        value.alpha,
        value.frequency,
        &value.residuals,
        &value.iterations,
    );
}

pub(super) fn tensor_values(
    report: &mut Value,
    alpha: [[Complex; 3]; 3],
    frequency: Complex,
    residuals: &[f64],
    iterations: &[usize],
) {
    let real = alpha.map(|r| r.map(|z| z.re));
    let imag = alpha.map(|r| r.map(|z| z.im));
    let isotropic = (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0;
    let reciprocity = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| (alpha[a][b] - alpha[b][a]).abs())
        .fold(0.0_f64, f64::max);
    report.set("alpha_real", real.to_vec());
    report.set("alpha_imag", imag.to_vec());
    report.set("polarizability_real", real.to_vec());
    report.set("polarizability_imag", imag.to_vec());
    if frequency.abs() == 0.0 {
        report.set("polarizability", real.to_vec());
        report.set("isotropic", isotropic.re);
    }
    report.set("isotropic_real", isotropic.re);
    report.set("isotropic_imag", isotropic.im);
    report.set("frequency_real", frequency.re);
    report.set("frequency_imag", frequency.im);
    report.set("reciprocity_defect", reciprocity);
    report.set("response_residuals", residuals.to_vec());
    report.set("response_iterations", iterations.to_vec());
    report.set("residuals", residuals.to_vec());
    report.set("iterations", iterations.to_vec());
    report.set("shape", vec![3usize, 3]);
    report.set("axis_convention", "alpha[observable Cartesian axis][electric-field Cartesian axis]; H'=+E.r; exp(-i z t); all nine contractions retained");
    report.set("provenance", "analytic");
    let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
    for key in [
        "alpha_real",
        "alpha_imag",
        "polarizability_real",
        "polarizability_imag",
        "polarizability",
        "isotropic",
        "isotropic_real",
        "isotropic_imag",
        "reciprocity_defect",
    ] {
        labels.set(key, DIPOLE_RESPONSE_UNIT);
    }
    for key in ["frequency_real", "frequency_imag"] {
        labels.set(key, "hartree");
    }
    labels.set("response_residuals", "dimensionless");
    labels.set("response_iterations", "integer");
    labels.set("residuals", "dimensionless");
    labels.set("iterations", "integer");
    labels.set("shape", "integer");
    report.set("units", labels);
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let frequency = frequency(controls)?;
    let thermal_response = controls.str("thermal_response", "integer")?;
    if !matches!(thermal_response.as_str(), "integer" | "collisionless") {
        return Err(DftError::InvalidInput(
            "thermal_response must be integer or collisionless".into(),
        ));
    }
    let collisionless = thermal_response == "collisionless";
    if super::double_hybrid::requested(controls)? {
        if collisionless {
            return Err(DftError::Unsupported {
                feature: "collisionless thermal correlated susceptibility".into(),
                missing: "the population-weighted causal correlation kernel".into(),
            });
        }
        let mut translated = controls.to_value();
        if let Value::Object(entries) = &mut translated {
            entries.retain(|(k, _)| {
                ![
                    "frequency",
                    "frequency_damping",
                    "imaginary",
                    "response_omega",
                    "response_damping",
                    "imaginary_frequency",
                    "thermal_response",
                ]
                .contains(&k.as_str())
            });
        }
        if frequency.energy()?.abs() != 0.0 {
            return correlated_frequency(system, controls, translated, frequency.energy()?);
        }
        let mut report = super::molecular::run(
            "polarizability",
            system,
            &Controls::from_value(&translated)?,
        )?;
        let alpha = report
            .get("polarizability")
            .cloned()
            .expect("correlated tensor");
        let (_, entries) = alpha.numeric_array().expect("Cartesian correlated tensor");
        report.set("alpha_real", alpha);
        report.set("alpha_imag", vec![[0.0; 3]; 3]);
        report.set(
            "isotropic_real",
            (entries[0] + entries[4] + entries[8]) / 3.0,
        );
        report.set("isotropic_imag", 0.0);
        report.set("frequency_real", 0.0);
        report.set("frequency_imag", 0.0);
        report.set("shape", vec![3usize, 3]);
        report.set(
            "residuals",
            report
                .get("response_residuals")
                .cloned()
                .expect("correlated residuals"),
        );
        let mut labels = report.get("units").cloned().expect("correlated units");
        for key in [
            "alpha_real",
            "alpha_imag",
            "isotropic_real",
            "isotropic_imag",
        ] {
            labels.set(key, "e^2*bohr^2/hartree");
        }
        for key in ["frequency_real", "frequency_imag"] {
            labels.set(key, "hartree");
        }
        labels.set("shape", "integer");
        labels.set("residuals", "dimensionless");
        report.set("units", labels);
        return Ok(report);
    }
    let reference_kind = controls.str("reference", "auto")?.to_ascii_lowercase();
    if collisionless && matches!(reference_kind.as_str(), "gks" | "ghf" | "roks" | "rohf") {
        return Err(DftError::Unsupported {
            feature: "collisionless thermal response of this reference".into(),
            missing: "a validated population-weighted spinor or constrained common-orbital kernel"
                .into(),
        });
    }
    if matches!(reference_kind.as_str(), "gks" | "ghf") {
        return super::gks::run("dynamic-polarizability", system, controls);
    }
    super::method::CollinearPropertyReference::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let setup = setup(system, controls)?;
    controls.finish("dynamic-polarizability")?;
    if matches!(reference_kind.as_str(), "roks" | "rohf") {
        if setup.method.stabilize {
            return Err(DftError::Unsupported {
                feature: "constrained response with stability following".into(),
                missing: "the corresponding common-orbital reference provider".into(),
            });
        }
        let (scf, orbitals) = crate::scf::roks::run_roks_with_orbitals(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )?;
        let value = crate::response::nuclear::open_shell_dynamic_polarizability(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &scf,
            &orbitals,
            &setup.options,
            frequency,
            &config,
        )?;
        let reference = MolecularReference {
            scf,
            e_dispersion: crate::dispersion::dispersion_energy(
                &setup.molecule,
                &setup.dispersion,
                &setup.method.dispersion_functional_name(),
            )?,
            e_smd_cds: match setup.smd {
                Some(solvent) => crate::pcm::smd_cds_energy(&setup.molecule, solvent, 2048)?,
                None => 0.0,
            },
            following: None,
        };
        let mut report = reference_report(&setup, &reference, "adiabatic time-dependent constrained common-orbital ROKS/ROHF response; paired time-even/time-odd Hessians and occupation-difference Berry metric; no UKS substitution")
            .with("reference_type", "roks").with("reference_stationarity", value.reference_stationarity).with("metric_defect", value.metric_defect);
        tensor_report(&mut report, &value.response);
        if value.response.frequency.abs() == 0.0 {
            let dipole: Vec<_> =
                super::molecular::raw_moments(&setup, &reference, crate::math::Vec3::zero(), 1)
                    .into_iter()
                    .map(|(_, value)| value)
                    .collect();
            report.set("dipole", dipole);
            let a = value.response.alpha.map(|r| r.map(|z| z.re));
            let anisotropy = (0.5
                * ((a[0][0] - a[1][1]).powi(2)
                    + (a[1][1] - a[2][2]).powi(2)
                    + (a[2][2] - a[0][0]).powi(2))
                + 3.0 * (a[0][1].powi(2) + a[0][2].powi(2) + a[1][2].powi(2)))
            .sqrt();
            report.set("anisotropy", anisotropy);
        }
        report.set(
            "response_residual",
            value.response.residuals.into_iter().fold(0.0_f64, f64::max),
        );
        report.set("flag_defect", value.metric_defect);
        report.set("stationarity_defect", value.reference_stationarity);
        report.set("response_converged", true);
        report.set("path_evaluations", 3usize);
        let mut labels = report.get("units").cloned().expect("tensor labels");
        labels.set("reference_stationarity", "hartree");
        labels.set("metric_defect", "dimensionless");
        labels.set("dipole", "e*bohr");
        labels.set("anisotropy", "e^2*bohr^2/hartree");
        labels.set("response_residual", "dimensionless");
        labels.set("flag_defect", "dimensionless");
        labels.set("stationarity_defect", "hartree");
        labels.set("path_evaluations", "integer");
        report.set("units", labels);
        return Ok(report);
    }
    let reference = run_molecular_reference(&setup)?;
    if collisionless {
        let thermal = crate::response::thermal_dynamic::polarizability(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            frequency,
            &config,
        )?;
        let result = thermal.response;
        let mut report = reference_report(&setup, &reference, result.model);
        tensor_report(
            &mut report,
            &SpinorPolarizability {
                frequency: frequency.energy()?,
                alpha: result.value.alpha,
                residuals: result.residuals.try_into().expect("three Cartesian solves"),
                iterations: result
                    .iterations
                    .try_into()
                    .expect("three Cartesian solves"),
            },
        );
        report.set("thermal_response", "collisionless");
        report.set("population_constraint", "fixed reference Fermi populations per spin; no diagonal occupation relaxation or chemical-potential response");
        report.set(
            "static_limit",
            "collisionless fixed populations; distinct from isothermal Mermin response",
        );
        report.set("electronic_temperature", thermal.temperature);
        report.set("chemical_potentials", thermal.chemical_potentials.to_vec());
        report.set("occupations", reference.scf.occupations.to_vec());
        report.set("electronic_entropy", reference.scf.electronic_entropy);
        report.set(
            "energy_convention",
            "canonical Mermin free energy including -T*S",
        );
        report.set("population_defect", thermal.population_defect);
        report.set(
            "response_particle_number_defects",
            thermal.number_defects.to_vec(),
        );
        report.set(
            "reference_orbital_residual",
            result.reference.orbital_residual,
        );
        report.set("reference_eigen_residual", result.reference.eigen_residual);
        report.set(
            "reference_metric_defect",
            result.reference.orthonormality_defect,
        );
        report.set(
            "reference_projector_defect",
            result.reference.projector_defect,
        );
        report.set(
            "reference_projector_interpretation",
            "physical thermal non-idempotency; not an error gate",
        );
        report.set(
            "reference_particle_number_error",
            result.reference.particle_number_error,
        );
        report.set("response_converged", result.converged);
        report.set("warnings", result.warnings);
        let mut labels = report.get("units").cloned().expect("tensor labels");
        labels.set("electronic_temperature", "hartree");
        labels.set("chemical_potentials", "hartree");
        labels.set("occupations", "electrons per spin orbital");
        labels.set("electronic_entropy", "k_B");
        labels.set("population_defect", "dimensionless");
        labels.set(
            "response_particle_number_defects",
            "electrons*e*bohr/hartree",
        );
        labels.set("reference_orbital_residual", "hartree");
        labels.set("reference_eigen_residual", "dimensionless");
        labels.set("reference_metric_defect", "dimensionless");
        labels.set("reference_projector_defect", "dimensionless");
        labels.set("reference_particle_number_error", "electrons");
        report.set("units", labels);
        return Ok(report);
    }
    let result = crate::response::dynamic_polarizability(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        frequency,
        &config,
    )?;
    let mut report = reference_report(&setup, &reference, result.model);
    report.set(
        "reference_orbital_residual",
        result.reference.orbital_residual,
    );
    report.set("reference_eigen_residual", result.reference.eigen_residual);
    report.set(
        "reference_metric_defect",
        result.reference.orthonormality_defect,
    );
    report.set(
        "reference_projector_defect",
        result.reference.projector_defect,
    );
    report.set(
        "reference_particle_number_error",
        result.reference.particle_number_error,
    );
    tensor_report(
        &mut report,
        &SpinorPolarizability {
            frequency: frequency.energy()?,
            alpha: result.value.alpha,
            residuals: result.residuals.try_into().expect("three Cartesian solves"),
            iterations: result
                .iterations
                .try_into()
                .expect("three Cartesian solves"),
        },
    );
    let mut labels = report.get("units").cloned().expect("tensor labels");
    labels.set("reference_orbital_residual", "hartree");
    labels.set("reference_eigen_residual", "dimensionless");
    labels.set("reference_metric_defect", "dimensionless");
    labels.set("reference_projector_defect", "dimensionless");
    labels.set("reference_particle_number_error", "electrons");
    report.set("units", labels);
    Ok(report)
}

fn correlated_frequency(
    system: &System,
    controls: &Controls,
    mut translated: Value,
    z: Complex,
) -> Result<Value> {
    let reference = controls.str("reference", "auto")?.to_ascii_lowercase();
    if matches!(reference.as_str(), "gks" | "ghf" | "roks" | "rohf") {
        return Err(DftError::Unsupported {
            feature: format!(
                "frequency-dependent correlated susceptibility with reference={reference}"
            ),
            missing: "the corresponding spinor or constrained pair quasi-energy action".into(),
        });
    }
    let kind = super::method::CollinearPropertyReference::from_controls(controls)?;
    if let Value::Object(entries) = &mut translated {
        entries.retain(|(key, _)| key != "reference");
    }
    if kind != super::method::CollinearPropertyReference::Auto {
        let unrestricted = kind == super::method::CollinearPropertyReference::Uks;
        if !unrestricted && controls.usize("multiplicity", 1)? != 1 {
            return Err(DftError::InvalidInput(
                "a restricted closed-shell reference requires multiplicity=1".into(),
            ));
        }
        translated.set("unrestricted", unrestricted);
    }
    let input = Controls::from_value(&translated)?;
    let prepared = super::double_hybrid::prepare(system, &input, "dynamic-polarizability")?;
    let sources =
        crate::integrals::dipole_matrices(&prepared.setup.basis, crate::math::Vec3::zero());
    let config = prepared
        .config
        .as_ref()
        .expect("correlated response configuration");
    let solved = super::correlated_electric::frequency::solve(&prepared, &sources, z, config)?;
    let alpha = std::array::from_fn(|a| std::array::from_fn(|b| solved.response.tensor[a][b]));
    let mut report = Value::object();
    tensor_values(
        &mut report,
        alpha,
        z,
        &solved.response.residuals,
        &solved.response.iterations,
    );
    solved.append_channels(&mut report, "polarizability", 1.0, 1, DIPOLE_RESPONSE_UNIT);
    super::double_hybrid::append_prepared_energy(prepared, report)
}
