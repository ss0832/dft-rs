// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular orbital self-interaction correction through the common driver.

use super::method::MolecularSetup;
use super::molecular::{setup, units};
use super::{Controls, System, Value};
use crate::math::Vec3;
use crate::scf::sic;
use crate::{DftError, Result};

fn prepare(system: &System, controls: &Controls, model: &str) -> Result<(MolecularSetup, String)> {
    let reference = controls.str("reference", "uks")?.to_ascii_lowercase();
    if !matches!(reference.as_str(), "auto" | "uks" | "uhf") {
        return Err(DftError::Unsupported {
            feature: format!("{model} with reference={reference}"),
            missing: "a variational self-interaction correction on the requested orbital manifold"
                .into(),
        });
    }
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: format!("{model} with a double hybrid"),
            missing: "a variational correlated orbital correction".into(),
        });
    }
    let setup = setup(system, controls)?;
    if setup.method.stabilize
        || setup.smd.is_some()
        || setup.options.reference == crate::scf::Reference::Rhf
    {
        return Err(DftError::Unsupported {
            feature: format!("{model} with stability following, SMD or restricted orbitals"),
            missing: "the corresponding corrected orbital Hessian or orbital functional".into(),
        });
    }
    Ok((setup, reference))
}

fn self_energies(terms: &[Vec<sic::SelfEnergy>; 2]) -> Vec<Vec<Value>> {
    terms
        .iter()
        .map(|terms| {
            terms
                .iter()
                .map(|value| {
                    Value::object()
                        .with("hartree", value.hartree)
                        .with("exact_exchange", value.exchange)
                        .with("long_range_exchange", value.long_range_exchange)
                        .with("semilocal_exchange_correlation", value.semilocal)
                        .with("nonlocal_correlation", value.nonlocal)
                        .with("total", value.total())
                })
                .collect()
        })
        .collect()
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let (setup, reference) = prepare(system, controls, "PZ-SIC")?;
    controls.finish("pz-sic")?;
    let result = sic::run_pz(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )?;
    let dispersion = crate::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )?;
    Ok(Value::object()
        .with(
            "model",
            "orbital-stationary Perdew-Zunger SIC with real integer unrestricted orbitals",
        )
        .with("reference_type", "uks")
        .with("requested_reference", reference)
        .with("functional", setup.functional.description.as_str())
        .with("basis", setup.method.basis.as_str())
        .with("energy", result.energy + dispersion)
        .with("parent_energy", result.parent_energy + dispersion)
        .with(
            "initial_scf_energy",
            result.initial_parent_energy + dispersion,
        )
        .with(
            "self_interaction_correction",
            result.energy - result.parent_energy,
        )
        .with("dispersion_energy", dispersion)
        .with("orbital_self_energies", self_energies(&result.terms))
        .with("scf_orbital_residual", result.orbital_gradient)
        .with("scf_iterations", result.iterations)
        .with("converged", true)
        .with(
            "units",
            units(&[
                ("energy", "hartree"),
                ("parent_energy", "hartree"),
                ("initial_scf_energy", "hartree"),
                ("self_interaction_correction", "hartree"),
                ("dispersion_energy", "hartree"),
                ("orbital_self_energies", "hartree"),
                ("scf_orbital_residual", "hartree"),
            ]),
        ))
}

fn read_descriptors(controls: &Controls, key: &str) -> Result<Vec<Vec3>> {
    controls
        .opt_rows(key)?
        .unwrap_or_default()
        .iter()
        .map(|row| {
            let [x, y, z] = row.as_slice() else {
                return Err(DftError::InvalidInput(format!(
                    "{key} must contain Cartesian triples in Bohr"
                )));
            };
            Ok(Vec3::new(*x, *y, *z))
        })
        .collect()
}

pub(super) fn run_fixed_flo(system: &System, controls: &Controls) -> Result<Value> {
    let (setup, reference) = prepare(system, controls, "Fixed-descriptor FLO-SIC")?;
    let alpha = read_descriptors(controls, "fod_alpha")?;
    let beta = read_descriptors(controls, "fod_beta")?;
    let complex =
        controls.contains("fod_alpha_imaginary") || controls.contains("fod_beta_imaginary");
    let complex_descriptors = if complex {
        let pair = |real: &[Vec3], key: &str| -> Result<Vec<sic::flo_sic::ComplexDescriptor>> {
            let imaginary = if controls.contains(key) {
                read_descriptors(controls, key)?
            } else {
                vec![Vec3::zero(); real.len()]
            };
            if real.len() != imaginary.len() {
                return Err(DftError::InvalidInput(format!(
                    "{key} must match the number of real descriptors"
                )));
            }
            Ok(real
                .iter()
                .zip(imaginary)
                .map(|(&real, imaginary)| sic::flo_sic::ComplexDescriptor { real, imaginary })
                .collect())
        };
        Some([
            pair(&alpha, "fod_alpha_imaginary")?,
            pair(&beta, "fod_beta_imaginary")?,
        ])
    } else {
        None
    };
    let self_consistent = controls.bool("flo_self_consistent", false)?;
    let optimize = controls.bool("fod_optimize", false)?;
    let gradient = controls.bool("fod_gradient", false)?;
    let mode = if self_consistent {
        sic::flo_sic::DescriptorMode::SelfConsistent(sic::flo_sic::StationaryFloOptions {
            orbital_tolerance: controls.f64("flo_orbital_tolerance", 1e-6)?,
            descriptor_tolerance: controls.f64("fod_gradient_tolerance", 1e-6)?,
            max_iterations: controls.usize("fod_max_iterations", 200)?,
        })
    } else if optimize {
        sic::flo_sic::DescriptorMode::Optimize {
            gradient_tolerance: controls.f64("fod_gradient_tolerance", 1e-6)?,
            max_iterations: controls.usize("fod_max_iterations", 100)?,
        }
    } else if gradient {
        sic::flo_sic::DescriptorMode::Gradient
    } else {
        sic::flo_sic::DescriptorMode::Fixed
    };
    controls.finish("flo-sic-fixed")?;
    let dispersion = crate::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )?;
    if let Some(d) = complex_descriptors {
        let result = sic::flo_sic::run_complex_descriptors(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
            [&d[0], &d[1]],
            mode,
        )?;
        return Ok(complex_flo_report(&result, &setup, &reference, dispersion));
    }
    let run = sic::flo_sic::run_descriptors(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
        [&alpha, &beta],
        mode,
    )?;
    let result = run.state;
    let mut report = Value::object()
        .with(
            "model",
            if self_consistent {
                "self-consistent variational real Fermi-Lowdin SIC on an integer UKS subspace"
            } else if optimize {
                "descriptor-optimized Fermi-Lowdin SIC on a frozen integer UKS occupied subspace"
            } else {
                "fixed-descriptor Fermi-Lowdin SIC on a frozen integer UKS determinant"
            },
        )
        .with("reference_type", "uks")
        .with("requested_reference", reference)
        .with("functional", setup.functional.description.as_str())
        .with("basis", setup.method.basis.as_str())
        .with("energy", result.energy + dispersion)
        .with("parent_energy", result.parent_energy + dispersion)
        .with("self_interaction_correction", result.correction)
        .with("dispersion_energy", dispersion)
        .with("orbital_self_energies", self_energies(&result.terms))
        .with(
            "fermi_descriptor_densities",
            result
                .orbitals
                .each_ref()
                .map(|o| o.descriptor_densities.clone()),
        )
        .with(
            "fermi_overlap_min_eigenvalues",
            result
                .orbitals
                .each_ref()
                .map(|o| o.min_fermi_overlap_eigenvalue),
        )
        .with(
            "localized_orthogonality_residuals",
            result.orbitals.each_ref().map(|o| o.orthogonality_residual),
        )
        .with(
            "occupied_orthogonality_residuals",
            result
                .orbitals
                .each_ref()
                .map(|o| o.input_orthogonality_residual),
        )
        .with("localized_density_residual", result.density_residual)
        .with("parent_orbital_residual", result.parent_orbital_residual)
        .with("scf_iterations", result.parent_iterations)
        .with(
            "converged",
            run.optimization.as_ref().is_none_or(|r| r.converged)
                && run.stationarity.is_none_or(|r| r.converged),
        );
    let mut units = units(&[
        ("energy", "hartree"),
        ("parent_energy", "hartree"),
        ("self_interaction_correction", "hartree"),
        ("dispersion_energy", "hartree"),
        ("orbital_self_energies", "hartree"),
        ("fermi_descriptor_densities", "electrons/bohr^3"),
        ("fermi_overlap_min_eigenvalues", "dimensionless"),
        ("localized_orthogonality_residuals", "dimensionless"),
        ("occupied_orthogonality_residuals", "dimensionless"),
        ("localized_density_residual", "dimensionless"),
        ("parent_orbital_residual", "hartree"),
    ]);
    if let Some(gradient) = run.descriptor_gradient {
        let descriptors = run
            .descriptors
            .map(|d| d.into_iter().map(|p| p.to_array()).collect::<Vec<_>>());
        report.set("fod_alpha", &descriptors[0]);
        report.set("fod_beta", &descriptors[1]);
        report.set("fod_gradient_alpha", &gradient[0]);
        report.set("fod_gradient_beta", &gradient[1]);
        units.set("fod_alpha", "bohr");
        units.set("fod_beta", "bohr");
        units.set("fod_gradient_alpha", "hartree/bohr");
        units.set("fod_gradient_beta", "hartree/bohr");
    }
    if let Some(optimization) = run.optimization {
        report.set("fod_iterations", optimization.iterations);
        report.set("fod_evaluations", optimization.evaluations);
        report.set("fod_gradient_max", optimization.projected_gradient_max);
        units.set("fod_gradient_max", "hartree/bohr");
    }
    if let Some(stationarity) = run.stationarity {
        report.set("fod_iterations", stationarity.iterations);
        report.set("fod_evaluations", stationarity.evaluations);
        report.set("fod_gradient_max", stationarity.descriptor_residual);
        report.set("flo_orbital_residual", stationarity.orbital_residual);
        units.set("fod_gradient_max", "hartree/bohr");
        units.set("flo_orbital_residual", "hartree");
    }
    report.set("units", units);
    Ok(report)
}

fn complex_flo_report(
    result: &sic::flo_sic::ComplexFloRun,
    setup: &MolecularSetup,
    reference: &str,
    dispersion: f64,
) -> Value {
    let mut report = Value::object()
        .with(
            "model",
            "complex Fermi-Lowdin SIC on a frozen real integer UKS occupied subspace",
        )
        .with("reference_type", "uks")
        .with("requested_reference", reference)
        .with("functional", setup.functional.description.as_str())
        .with("basis", setup.method.basis.as_str())
        .with("energy", result.energy + dispersion)
        .with("parent_energy", result.parent_energy + dispersion)
        .with("self_interaction_correction", result.correction)
        .with("dispersion_energy", dispersion)
        .with("orbital_self_energies", self_energies(&result.terms))
        .with(
            "log_fermi_descriptor_densities",
            &result.log_descriptor_densities,
        )
        .with(
            "localized_orthogonality_residuals",
            result.orthogonality_residuals,
        )
        .with("localized_embedding_residuals", result.embedding_residuals)
        .with("localized_density_residual", result.density_residual)
        .with("parent_orbital_residual", result.parent_orbital_residual)
        .with("scf_iterations", result.parent_iterations)
        .with(
            "flo_coefficients_real",
            result.coefficients.each_ref().map(|c| c.re.clone()),
        )
        .with(
            "flo_coefficients_imaginary",
            result.coefficients.each_ref().map(|c| c.im.clone()),
        )
        .with(
            "converged",
            result.optimization.as_ref().is_none_or(|r| r.converged),
        );
    let mut units = units(&[
        ("energy", "hartree"),
        ("parent_energy", "hartree"),
        ("self_interaction_correction", "hartree"),
        ("dispersion_energy", "hartree"),
        ("orbital_self_energies", "hartree"),
        ("log_fermi_descriptor_densities", "dimensionless"),
        ("localized_orthogonality_residuals", "dimensionless"),
        ("localized_embedding_residuals", "dimensionless"),
        ("localized_density_residual", "dimensionless"),
        ("parent_orbital_residual", "hartree"),
        ("flo_coefficients_real", "dimensionless"),
        ("flo_coefficients_imaginary", "dimensionless"),
    ]);
    for (spin, label) in ["alpha", "beta"].into_iter().enumerate() {
        let real = format!("fod_{label}");
        let imaginary = format!("fod_{label}_imaginary");
        report.set(
            &real,
            result.descriptors[spin]
                .iter()
                .map(|p| p.real.to_array())
                .collect::<Vec<_>>(),
        );
        report.set(
            &imaginary,
            result.descriptors[spin]
                .iter()
                .map(|p| p.imaginary.to_array())
                .collect::<Vec<_>>(),
        );
        units.set(&real, "bohr");
        units.set(&imaginary, "bohr");
        if let Some(gradients) = &result.gradients {
            let key = format!("fod_gradient_{label}");
            report.set(&key, &gradients[spin]);
            units.set(&key, "hartree/bohr");
        }
    }
    if let Some(optimization) = &result.optimization {
        report.set("fod_iterations", optimization.iterations);
        report.set("fod_evaluations", optimization.evaluations);
        report.set("fod_gradient_max", optimization.projected_gradient_max);
        units.set("fod_gradient_max", "hartree/bohr");
    }
    report.set("units", units);
    report
}
