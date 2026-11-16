// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Restricted open-shell Kohn-Sham energy and analytic gradient routed
//! through the common driver.

use super::controls::Controls;
use super::method::MolecularReference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::System;
use crate::math::Vec3;
use crate::{DftError, Result};

pub(super) fn roks(system: &System, controls: &Controls) -> Result<Value> {
    run(system, controls, false)
}

pub(super) fn mermin(system: &System, controls: &Controls) -> Result<Value> {
    run(system, controls, true)
}

fn run(system: &System, controls: &Controls, mermin: bool) -> Result<Value> {
    let with_gradient = controls.bool("gradient", !mermin)?;
    if mermin && controls.bool("gibbs", false)? {
        return Err(DftError::Unsupported {
            feature: "Gibbs thermochemistry of a common-orbital Mermin reference".into(),
            missing: "the thermal shared-orbital molecular Hessian".into(),
        });
    }
    if super::double_hybrid::requested(controls)? {
        if mermin || controls.f64("electronic_temperature", 0.0)? > 0.0 {
            return Err(DftError::Unsupported {
                feature: "common-orbital Mermin free energy with PT2".into(),
                missing: "a finite-temperature correlated energy for the common-orbital ensemble"
                    .into(),
            });
        }
        if with_gradient {
            return Err(DftError::Unsupported {
                feature: "common-orbital double-hybrid nuclear gradient".into(),
                missing: "the constrained common-orbital PT2 response Lagrangian; total energies are available with gradient=false".into(),
            });
        }
        let mut selected = controls.to_value();
        if let Value::Object(entries) = &mut selected {
            entries.retain(|(key, _)| key != "gradient");
        }
        selected.set("reference", "roks");
        return super::double_hybrid::energy(system, &Controls::from_value(&selected)?);
    }
    // Shared reference requests may carry the same response controls for
    // gradient and Hessian calls. Validate them even when the first energy
    // derivative needs no orbital response equation.
    super::vibrational::response_config(controls)?;
    let (setup, density_dispersion) = if with_gradient {
        (setup(system, controls)?, None)
    } else {
        super::density_dispersion::prepare_energy(system, controls)?
    };
    controls.finish(if mermin { "mermin-free-energy" } else { "roks" })?;
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "restricted open-shell SCF with the SMD non-electrostatic term".into(),
            missing: "the continuum terms of the restricted open-shell Fock build".into(),
        });
    }
    let thermal = setup.options.electronic_temperature > 0.0;
    let solver = if thermal {
        crate::scf::roks::run_thermal_roks
    } else {
        crate::scf::roks::run_roks
    };
    let scf = solver(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )?;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let e_dispersion = crate::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )?;
    let reference = MolecularReference {
        scf,
        e_dispersion,
        e_smd_cds: 0.0,
        following: None,
    };
    let mut report = reference_report(
        &setup,
        &reference,
        if thermal {
            "Mermin ensemble of shared spatial orbitals at fixed alpha and beta electron counts"
        } else {
            "restricted open-shell Kohn-Sham: one set of spatial orbitals with closed and open \
             shells, Roothaan effective Fock matrix"
        },
    )
    .with("reference_type", "restricted open-shell")
    .with("provenance", "analytic");
    let mut unit_pairs = vec![("energy", "hartree"), ("energy_components", "hartree")];
    if mermin || thermal {
        let scf = &reference.scf;
        let kt = scf.electronic_temperature;
        let entropy_term = -kt * scf.electronic_entropy;
        report.set("mermin_free_energy", reference.total_energy());
        report.set("free_energy", reference.total_energy());
        report.set("internal_energy", reference.total_energy() - entropy_term);
        report.set("entropy", scf.electronic_entropy);
        report.set("entropy_term", entropy_term);
        report.set("kt", kt);
        report.set(
            "electronic_temperature_kelvin",
            setup.method.electronic_temperature,
        );
        report.set("occupations", scf.occupations.to_vec());
        report.set("orbital_levels", scf.mo_energy.to_vec());
        report.set("chemical_potentials", scf.chemical_potentials.to_vec());
        report.set(
            "orbital_residual",
            super::molecular::reference_orbital_residual(scf),
        );
        report.set(
            "ensemble",
            "fixed alpha and beta electron counts in one common spatial frame",
        );
        if thermal {
            let c = &scf.mo_coeff[0];
            let metric = c.transpose_matmul(&scf.overlap.matmul(c));
            let defect = metric.max_abs_diff(&crate::linalg::Matrix::identity(c.cols));
            let population_defect = (0..2)
                .filter_map(|s| {
                    scf.chemical_potentials[s].map(|mu| {
                        scf.mo_energy[s]
                            .iter()
                            .zip(&scf.occupations[s])
                            .map(|(energy, occupation)| {
                                let (fermi, _) = crate::periodic::occupations::fermi_and_hole(
                                    (energy - mu) / kt,
                                );
                                (fermi - occupation).abs()
                            })
                            .fold(0.0, f64::max)
                    })
                })
                .fold(0.0, f64::max);
            report.set("common_orbital_metric_defect", defect);
            report.set("thermal_population_defect", population_defect);
            unit_pairs.extend([
                ("common_orbital_metric_defect", "dimensionless"),
                ("thermal_population_defect", "electrons"),
            ]);
        }
        unit_pairs.extend([
            ("free_energy", "hartree"),
            ("mermin_free_energy", "hartree"),
            ("internal_energy", "hartree"),
            ("entropy", "k_B (dimensionless)"),
            ("entropy_term", "hartree"),
            ("kt", "hartree"),
            ("electronic_temperature_kelvin", "K"),
            ("occupations", "electrons per orbital"),
            ("orbital_levels", "hartree"),
            ("chemical_potentials", "hartree"),
            ("orbital_residual", "hartree"),
        ]);
    }
    if with_gradient {
        let ordinary = || {
            crate::gradient::assemble_gradient(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                &setup.options,
            )
        };
        let mut gradient = if thermal && reference.scf.n_alpha != reference.scf.n_beta {
            let scf = &reference.scf;
            let c = &scf.mo_coeff[0];
            // The multiplier of C^T S C = I is sym(sum_s f_s F_s).
            // It retains the off-diagonal common-frame Pulay contribution.
            let mut multiplier = crate::linalg::Matrix::zeros(c.cols, c.cols);
            for spin in 0..2 {
                let fock = c.transpose_matmul(&scf.fock[spin].matmul(c));
                for i in 0..c.cols {
                    for j in 0..c.cols {
                        multiplier[(i, j)] += 0.5
                            * (scf.occupations[spin][i] + scf.occupations[spin][j])
                            * fock[(i, j)];
                    }
                }
            }
            let weighted = c.matmul(&multiplier).matmul_transpose_seq(c);
            crate::gradient::assemble_gradient_weighted(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                &weighted,
            )?
        } else {
            ordinary()?
        };
        if setup.dispersion != crate::dispersion::Dispersion::None {
            let correction = crate::dispersion::dispersion_gradient(
                &setup.molecule,
                &setup.dispersion,
                &setup.method.dispersion_functional_name(),
            )?;
            for (total, delta) in gradient.iter_mut().zip(correction) {
                *total += delta;
            }
        }
        let mut sum = Vec3::zero();
        for g in &gradient {
            sum += *g;
        }
        report.set(
            "gradient",
            gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>(),
        );
        report.set("translation_residual", sum.norm());
        unit_pairs.push(("gradient", "hartree/bohr"));
        unit_pairs.push(("translation_residual", "hartree/bohr"));
    }
    report.set("units", units(&unit_pairs));
    if let Some(request) = density_dispersion {
        request.evaluate_collinear(&setup, &reference.scf, &mut report)?;
    }
    Ok(report)
}

/// Analytic Hessian of the restricted open-shell energy from the Taylor
/// series of the energy along nuclear paths.
pub(super) fn roks_hessian(system: &System, controls: &Controls) -> Result<Value> {
    let config = super::vibrational::response_config(controls)?;
    let setup = setup(system, controls)?;
    controls.finish("roks-hessian")?;
    if setup.options.electronic_temperature > 0.0 {
        return thermal_geometry(system, &setup, &config, None);
    }
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "restricted open-shell SCF with the SMD non-electrostatic term".into(),
            missing: "the continuum terms of the restricted open-shell Fock build".into(),
        });
    }
    let (scf, orbitals) = crate::scf::roks::run_roks_with_orbitals(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )?;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let result = crate::response::nuclear::open_shell_hessian(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &scf,
        &orbitals,
        &setup.options,
        &config,
    )?;
    let name = setup.method.dispersion_functional_name();
    let mut hessian = result.hessian.clone();
    let mut gradient = result.gradient.clone();
    if setup.dispersion != crate::dispersion::Dispersion::None {
        let correction =
            crate::dispersion::dispersion_hessian(&setup.molecule, &setup.dispersion, &name)?;
        for (total, delta) in hessian.as_mut_slice().iter_mut().zip(correction.as_slice()) {
            *total += *delta;
        }
        let forces =
            crate::dispersion::dispersion_gradient(&setup.molecule, &setup.dispersion, &name)?;
        for (atom, delta) in forces.iter().enumerate() {
            let delta = delta.to_array();
            for axis in 0..3 {
                gradient[3 * atom + axis] += delta[axis];
            }
        }
    }
    let natoms = setup.molecule.len();
    let mut translation = 0.0_f64;
    for row in 0..3 * natoms {
        for axis in 0..3 {
            let sum: f64 = (0..natoms).map(|b| hessian[(row, 3 * b + axis)]).sum();
            translation = translation.max(sum.abs());
        }
    }
    let electric = crate::response::nuclear::open_shell_electric(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &scf,
        &orbitals,
        &setup.options,
        Some(&result.second),
        &config,
    )?;
    let modes = crate::properties::vibrational::vibrational_analysis(
        &setup.molecule,
        &hessian,
        electric.dipole_derivatives.as_ref(),
    )?;
    let dipole_derivatives: Vec<Vec<f64>> =
        electric
            .dipole_derivatives
            .as_ref()
            .map_or_else(Vec::new, |m| {
                (0..m.rows)
                    .map(|r| (0..m.cols).map(|c| m[(r, c)]).collect())
                    .collect()
            });
    let e_dispersion =
        crate::dispersion::dispersion_energy(&setup.molecule, &setup.dispersion, &name)?;
    let reference = MolecularReference {
        scf,
        e_dispersion,
        e_smd_cds: 0.0,
        following: None,
    };
    let rows: Vec<Vec<f64>> = (0..hessian.rows)
        .map(|r| (0..hessian.cols).map(|c| hessian[(r, c)]).collect())
        .collect();
    let mut report = reference_report(
        &setup,
        &reference,
        "analytic Hessian of the restricted open-shell energy: second-order Taylor \
         coefficients of the energy along nuclear paths, with the closed-open, closed-virtual \
         and open-virtual rotations solved at first order on the orbital Hessian of the \
         reference",
    )
    .with("reference_type", "restricted open-shell")
    .with(
        "gradient",
        gradient
            .chunks_exact(3)
            .map(|g| [g[0], g[1], g[2]])
            .collect::<Vec<_>>(),
    )
    .with(
        "forces",
        gradient
            .chunks_exact(3)
            .map(|g| [-g[0], -g[1], -g[2]])
            .collect::<Vec<_>>(),
    )
    .with("hessian", rows)
    .with("frequencies_cm1", modes.frequencies_cm.clone())
    .with("ir_intensities_km_mol", modes.ir_intensities.clone())
    .with("dipole_derivatives", dipole_derivatives)
    .with("dipole", electric.dipole)
    .with("polarizability", electric.polarizability.to_vec())
    .with("n_imaginary", modes.n_imaginary)
    .with("n_projected", modes.n_projected)
    .with(
        "translation_residual",
        if setup.options.external_field.is_some() {
            None
        } else {
            Some(translation)
        },
    )
    .with("response_residual", result.residual)
    .with("flag_defect", result.flag_defect)
    .with("stationarity_defect", result.stationarity_defect)
    .with("reference_stationarity", result.reference_stationarity)
    .with("reference_energy_error", result.reference_energy_error)
    .with("response_converged", result.converged && electric.converged)
    .with(
        "converged",
        reference.scf.converged && result.converged && electric.converged,
    )
    .with("path_evaluations", result.paths + electric.paths)
    .with("symmetrized", true)
    .with("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("gradient", "hartree/bohr"),
            ("forces", "hartree/bohr"),
            ("hessian", "hartree/bohr^2"),
            ("frequencies_cm1", "cm^-1"),
            ("ir_intensities_km_mol", "km/mol"),
            (
                "dipole_derivatives",
                "e (d mu_x / d R, rows x, columns 3*atom+axis)",
            ),
            ("dipole", "e*bohr"),
            (
                "polarizability",
                "e^2*bohr^2/hartree (atomic units of polarizability), alpha_ab = d mu_a / d F_b",
            ),
            ("translation_residual", "hartree/bohr^2"),
            ("response_residual", "dimensionless"),
            ("flag_defect", "dimensionless"),
            ("stationarity_defect", "hartree"),
            ("reference_stationarity", "hartree"),
            ("reference_energy_error", "hartree"),
        ]),
    );
    report.set(
        "axis_convention",
        "rows and columns of the Hessian are 3*atom + axis",
    );
    Ok(report)
}

/// Static limit of the same constrained paired response used at finite frequency.
pub(super) fn roks_polarizability(system: &System, controls: &Controls) -> Result<Value> {
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "common-orbital double-hybrid dipole polarizability".into(),
            missing: "the constrained common-orbital PT2 energy Hessian".into(),
        });
    }
    let mut selected = controls.to_value();
    selected.set("reference", "roks");
    super::polarizability::run(system, &Controls::from_value(&selected)?)
}

/// Complete nuclear tensors on the stationary common-orbital flag manifold.
pub(super) fn force_constants(system: &System, controls: &Controls) -> Result<Value> {
    super::force_constants::require_variational_reference(controls)?;
    let selection = super::force_constants::Selection::new(system, controls)?;
    let config = super::vibrational::response_config(controls)?;
    let setup = setup(system, controls)?;
    selection.check_report_storage(3, setup.options.incore_mb)?;
    controls.finish("force-constants")?;
    if setup.smd.is_some() || setup.method.stabilize {
        return Err(DftError::Unsupported {
            feature: "common-orbital complete nuclear tensors with SMD or stability following"
                .into(),
            missing: "the matching constrained reference and non-electrostatic geometry provider"
                .into(),
        });
    }
    if setup.options.electronic_temperature > 0.0 {
        return thermal_geometry(system, &setup, &config, Some(&selection));
    }
    let (scf, orbitals) = crate::scf::roks::run_roks_with_orbitals(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )?;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let mut diagnostics = super::nuclear_tensor::NuclearDiagnostics::default();
    let (tensors, evaluations) = super::force_constants::assemble_energy_paths(
        &selection,
        3,
        setup.options.incore_mb,
        |d, order| {
            let direction: Vec<_> = d.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
            let value = crate::response::nuclear::open_shell_nuclear_energy_taylor(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &scf,
                &orbitals,
                &setup.options,
                &direction,
                order,
                &config,
            )?;
            diagnostics.response.extend(value.residuals);
            diagnostics.projector.extend(value.flag_defects);
            diagnostics.stationarity.extend(value.stationarity_defects);
            diagnostics
                .reference
                .extend([value.reference_stationarity, value.reference_energy_error]);
            let dispersion =
                super::nuclear_tensor::dispersion_path(&setup, system, &direction, order, None)?
                    [order];
            Ok(vec![
                value.energy[order] + dispersion,
                value.energy[order],
                dispersion,
            ])
        },
    )?;
    let e_dispersion = crate::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )?;
    let reference = MolecularReference {
        scf,
        e_dispersion,
        e_smd_cds: 0.0,
        following: None,
    };
    let mut report = reference_report(&setup, &reference,
        "complete common-orbital ROKS/ROHF nuclear tensors; analytic moving basis/grid and metric transport; closed/open projector flag constraints; stationary 2n+1 energy through fourth order; intrinsic permutation reuse")
        .with("reference_type", "restricted open-shell")
        .with("reference_field", setup.options.external_field.map_or([0.0;3], |v| v.to_array()))
        .with("nuclear_path_evaluations", evaluations);
    diagnostics.append(&mut report, &config);
    super::force_constants::append_energy_tensors(
        &mut report,
        &selection,
        tensors,
        evaluations,
        &["", "parent_", "dispersion_"],
        &config,
    )?;
    let mut labels = report.get("units").cloned().expect("force tensor units");
    labels.set("reference_field", "hartree/e/bohr");
    report.set("units", labels);
    Ok(report)
}

fn thermal_geometry(
    system: &System,
    setup: &super::method::MolecularSetup,
    config: &crate::response::ResponseSolverConfig,
    selection: Option<&super::force_constants::Selection>,
) -> Result<Value> {
    if setup.smd.is_some() || setup.method.stabilize {
        return Err(DftError::Unsupported {
            feature: "common-orbital Mermin geometry with SMD or stability following".into(),
            missing: "the matching constrained reference and non-electrostatic geometry provider"
                .into(),
        });
    }
    let scf = crate::scf::roks::run_thermal_roks(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )?;
    let reference = MolecularReference {
        e_dispersion: crate::dispersion::dispersion_energy(
            &setup.molecule,
            &setup.dispersion,
            &setup.method.dispersion_functional_name(),
        )?,
        scf,
        e_smd_cds: 0.0,
        following: None,
    };
    let context = crate::response::nuclear::ThermalRoksNuclearContext::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
    )?;
    let mut diagnostics = super::nuclear_tensor::NuclearDiagnostics::default();
    let mut path = |direction: &[[f64; 3]], order: usize| -> Result<(Vec<f64>, Vec<f64>)> {
        let value = context.energy_taylor(direction, order, config)?;
        diagnostics.response.extend(value.residuals);
        diagnostics.stationarity.extend(value.stationarity);
        diagnostics.thermal_number.extend(value.number_defects);
        diagnostics.thermal_chemical.extend(value.chemical);
        diagnostics
            .reference
            .push(value.energy[0] - reference.scf.energy);
        let dispersion =
            super::nuclear_tensor::dispersion_path(setup, system, direction, order, None)?;
        Ok((value.energy, dispersion))
    };
    let mut report = reference_report(
        setup,
        &reference,
        "analytic nuclear Taylor derivatives of stationary shared-orbital Mermin free energy",
    )
    .with("reference_type", "restricted open-shell")
    .with("mermin_free_energy", reference.total_energy())
    .with(
        "energy_derivative_potential",
        "electronic Mermin free energy at fixed temperature and spin electron counts",
    )
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("mermin_free_energy", "hartree"),
        ]),
    );
    if let Some(selection) = selection {
        let (tensors, evaluations) = super::force_constants::assemble_energy_paths(
            selection,
            3,
            setup.options.incore_mb,
            |d, order| {
                let direction = d
                    .chunks_exact(3)
                    .map(|v| [v[0], v[1], v[2]])
                    .collect::<Vec<_>>();
                let (parent, dispersion) = path(&direction, order)?;
                Ok(vec![
                    parent[order] + dispersion[order],
                    parent[order],
                    dispersion[order],
                ])
            },
        )?;
        super::force_constants::append_energy_tensors(
            &mut report,
            selection,
            tensors,
            evaluations,
            &["", "parent_", "dispersion_"],
            config,
        )?;
    } else {
        let tensors = super::nuclear_tensor::assemble(setup.molecule.len(), 2, 1, |direction| {
            let (parent, dispersion) = path(&direction, 2)?;
            Ok(vec![parent
                .iter()
                .zip(dispersion)
                .map(|(a, b)| a + b)
                .collect()])
        })?;
        let gradient = tensors.gradients[0]
            .chunks_exact(3)
            .map(|g| g.to_vec())
            .collect::<Vec<_>>();
        let forces = gradient
            .iter()
            .map(|row| row.iter().map(|v| -v).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        report.set("gradient", gradient);
        report.set("forces", forces);
        report.set("hessian", tensors.hessians[0].clone());
        report.set("nuclear_path_evaluations", tensors.paths);
        let mut labels = report.get("units").cloned().expect("Mermin geometry units");
        labels.set("gradient", "hartree/bohr");
        labels.set("forces", "hartree/bohr");
        labels.set("hessian", "hartree/bohr^2");
        report.set("units", labels);
    }
    diagnostics.append(&mut report, config);
    Ok(report)
}
/// Static electric and mixed nuclear/electric tensors of the same ROHF/ROKS
/// flag-projector reference used for its nuclear energy derivatives.
pub(super) fn electric_tensor(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let request = super::reference_electric::Request::new(id, system, controls)?;
    let config = super::vibrational::response_config(controls)?;
    let setup = setup(system, controls)?;
    controls.finish(id)?;
    if setup.smd.is_some() || setup.method.stabilize {
        return Err(DftError::Unsupported {
            feature: "constrained electric response with SMD or stability following".into(),
            missing: "the matching common-orbital reference/source provider".into(),
        });
    }
    let (scf, orbitals) = crate::scf::roks::run_roks_with_orbitals(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )?;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let reference = MolecularReference {
        e_dispersion: crate::dispersion::dispersion_energy(
            &setup.molecule,
            &setup.dispersion,
            &setup.method.dispersion_functional_name(),
        )?,
        scf,
        e_smd_cds: 0.0,
        following: None,
    };
    let mut report=reference_report(&setup,&reference,
        "stationary ROHF/ROKS electric and mixed nuclear/electric energy derivatives; closed/open/virtual flag-projector constraints; analytic moving basis/grid and metric; scalar-energy polarization; field-independent empirical dispersion has exactly zero electric derivatives")
        .with("reference_type","restricted open-shell")
        .with("reference_field",setup.options.external_field.map_or([0.0;3],|v|v.to_array()));
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("reference_field", "hartree/e/bohr"),
            ("residuals", "dimensionless"),
        ]),
    );
    let mut diagnostics = super::nuclear_tensor::NuclearDiagnostics::default();
    request.apply_retained(&mut report, &config, |direction, order| {
        let (value, remainders) =
            crate::response::nuclear::open_shell_molecular_energy_taylor_retained(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                &orbitals,
                &setup.options,
                direction,
                order,
                &config,
            )?;
        diagnostics.response.extend(value.residuals);
        diagnostics.projector.extend(value.flag_defects);
        diagnostics.stationarity.extend(value.stationarity_defects);
        diagnostics
            .reference
            .extend([value.reference_stationarity, value.reference_energy_error]);
        Ok([value.energy, remainders])
    })?;
    report.set("residuals", diagnostics.response.clone());
    diagnostics.append(&mut report, &config);
    Ok(report)
}
