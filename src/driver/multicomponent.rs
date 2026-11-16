// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Experimental simultaneous electronic/quantum-nuclear SCF on every surface.

use super::{Controls, System, Value};
use crate::grid::MolecularGrid;
use crate::math::Vec3;
use crate::multicomponent::annihilation::{Enhancement, annihilation};
use crate::multicomponent::correlation::CorrelationModel;
use crate::multicomponent::{
    MulticomponentOptions, QuadrupoleField, QuantumParticle, gaussian_particle_basis, solve,
};
use crate::multicomponent::{constraints, derivatives, response::CoupledResponseSolver};
use crate::response::dynamic::Frequency;
use crate::{DftError, Result};

mod electric;
mod relativistic;

fn quantum_basis_options(controls: &Controls) -> Result<(Vec<f64>, usize)> {
    let exponents = controls
        .opt_f64_list("quantum_exponents")?
        .unwrap_or(vec![4.0, 8.0, 16.0]);
    let lmax = controls.usize("quantum_lmax", 1)?;
    if lmax > 3 {
        return Err(DftError::Unsupported {
            feature: format!("quantum-particle basis with quantum_lmax={lmax}"),
            missing: "a common-driver basis connection above angular momentum 3".into(),
        });
    }
    Ok((exponents, lmax))
}

fn species_options(controls: &Controls) -> Result<MulticomponentOptions> {
    Ok(MulticomponentOptions {
        tolerance: controls.f64("species_tolerance", 1e-10)?,
        max_iterations: controls.usize("species_max_iterations", 500)?,
        mixing: controls.f64("species_mixing", 0.5)?,
    })
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    match controls
        .str("species_electronic_model", "nonrelativistic")?
        .as_str()
    {
        "nonrelativistic" => {}
        "dirac-coulomb" => return relativistic::run(system, controls),
        _ => {
            return Err(DftError::InvalidInput(
                "species_electronic_model must be nonrelativistic or dirac-coulomb".into(),
            ));
        }
    }
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "multicomponent double-hybrid energy".into(),
            missing:
                "the jointly stationary electronic and interspecies PT2 density/amplitude provider"
                    .into(),
        });
    }
    let (exponents, lmax) = quantum_basis_options(controls)?;
    let proton_correlation = controls.str("interspecies_correlation", "epc17")?;
    let positron_correlation = controls.str("positron_correlation", "none")?;
    let enhancement_name = controls.str("annihilation_enhancement", "drummond-2011")?;
    let enhancement = Enhancement::by_name(&enhancement_name)?;
    let gradient_alpha = controls.opt_f64("annihilation_gradient_alpha")?;
    let options = species_options(controls)?;
    let constrain_positions = controls.bool("quantum_position_constraint", false)?;
    let stability_mode = controls.str("species_stability", "none")?;
    let stability_roots = controls.usize("species_stability_roots", 3)?;
    if !matches!(stability_mode.as_str(), "none" | "report" | "follow") || stability_roots == 0 {
        return Err(DftError::InvalidInput(
            "species_stability is none, report or follow, with at least one root".into(),
        ));
    }
    if stability_mode == "follow" && constrain_positions {
        return Err(DftError::Unsupported {
            feature: "stability following of a position-constrained quantum reference".into(),
            missing: "the constrained coupled Hessian of the Lagrangian".into(),
        });
    }
    let targets = controls.opt_rows("quantum_position_targets")?;
    let derivative_order = controls.usize("species_derivative_order", 0)?;
    let direction = controls.opt_rows("species_direction")?;
    let polarizability = controls.bool("species_polarizability", false)?;
    let quadrupole_response = controls.bool("species_quadrupole_polarizability", false)?;
    let gradient_amplitudes: Option<[f64; 5]> = controls
        .opt_f64_list("species_field_gradient")?
        .map(|values| {
            values.try_into().map_err(|_| {
                DftError::InvalidInput(
                    "species_field_gradient requires five mode amplitudes".into(),
                )
            })
        })
        .transpose()?;
    let frequencies = controls.opt_f64_list("frequencies")?;
    let damping = if frequencies.is_some() {
        controls.f64("frequency_damping", 0.0)?
    } else {
        0.0
    };
    let frequency_points = frequencies.as_ref().map(|values| {
        values
            .iter()
            .map(|&omega| Frequency::Retarded { omega, damping })
            .collect::<Vec<_>>()
    });
    if let Some(frequency_points) = &frequency_points {
        if frequency_points.is_empty() {
            return Err(DftError::InvalidInput(
                "multicomponent frequencies cannot be empty".into(),
            ));
        }
        for frequency in frequency_points {
            frequency.validate()?;
        }
    }
    let quantum_mass_response = controls.bool("quantum_mass_response", false)?;
    let response_config = super::vibrational::response_config(controls)?;
    if direction.is_some() && derivative_order == 0 || targets.is_some() && !constrain_positions {
        return Err(DftError::InvalidInput("species_direction requires derivatives, and quantum_position_targets requires quantum_position_constraint=true".into()));
    }
    if derivative_order > 2 {
        return Err(DftError::Unsupported {
            feature: format!("multicomponent derivative order {derivative_order}"),
            missing: "coupled species response above second order".into(),
        });
    }
    if gradient_amplitudes.is_some() && (derivative_order != 0 || stability_mode == "follow") {
        return Err(DftError::Unsupported {
            feature: "finite multicomponent electric field gradient with geometry derivatives or stability following".into(),
            missing: "moving quadrupole field integrals and field-preserving restart".into(),
        });
    }
    let setup = super::molecular::setup(system, controls)?;
    let moment_origin = if quadrupole_response || gradient_amplitudes.is_some() {
        Some(super::molecular::molecular_moment_origin(
            controls,
            &setup.molecule,
            "center_of_nuclear_charge",
        )?)
    } else {
        None
    };
    let applied_gradient =
        gradient_amplitudes
            .zip(moment_origin.as_ref())
            .map(|(amplitudes, (origin, _))| QuadrupoleField {
                amplitudes,
                origin: *origin,
            });
    let quadrupole_origin = moment_origin.filter(|_| quadrupole_response);
    let requests = particle_requests(
        controls,
        &setup.molecule,
        &exponents,
        lmax,
        &proton_correlation,
        &positron_correlation,
    )?;
    controls.finish("multicomponent")?;
    if setup.dispersion != crate::dispersion::Dispersion::None
        || setup.smd.is_some()
        || setup.method.stabilize
    {
        return Err(DftError::Unsupported {feature:"multicomponent point-nucleus dispersion, solvent or electronic-only stability following".into(),missing:"the quantum-density coupling of those terms; use dispersion=none and a jointly stationary species model".into()});
    }
    let (particles, correlation) = particles_and_model(&requests)?;
    let atoms: Vec<usize> = particles.iter().filter_map(|p| p.replaced_atom).collect();
    let extra_grid;
    let grid = match setup.grid.as_ref() {
        Some(g) => Some(g),
        None if !correlation.is_zero() => {
            extra_grid = MolecularGrid::build(&setup.molecule, &setup.grid_options)?;
            Some(&extra_grid)
        }
        None => None,
    };
    let targets: Vec<[f64; 3]> = if let Some(targets) = targets {
        targets
            .iter()
            .map(|r| {
                if r.len() == 3 {
                    Ok([r[0], r[1], r[2]])
                } else {
                    Err(DftError::InvalidInput(
                        "quantum_position_targets requires three Bohr coordinates per particle"
                            .into(),
                    ))
                }
            })
            .collect::<Result<_>>()?
    } else {
        particles
            .iter()
            .map(|p| p.basis.shells[0].center.to_array())
            .collect()
    };
    let result = match (constrain_positions, applied_gradient.as_ref()) {
        (true, Some(field)) => constraints::solve_in_field_gradient(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            grid,
            &particles,
            &correlation,
            &options,
            &targets,
            &response_config,
            field,
        )?,
        (false, Some(field)) => crate::multicomponent::solve_in_field_gradient(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            grid,
            &particles,
            &correlation,
            &options,
            field,
        )?,
        (true, None) => constraints::solve(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            grid,
            &particles,
            &correlation,
            &options,
            &targets,
            &response_config,
        )?,
        (false, None) => solve(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            grid,
            &particles,
            &correlation,
            &options,
        )?,
    };
    let (result, stability) = if stability_mode == "none" {
        (result, Value::Null)
    } else {
        let followed = crate::multicomponent::stability::follow(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            grid,
            &particles,
            &correlation,
            &options,
            result,
            stability_roots,
            1e-6,
            stability_mode == "follow",
            8,
        )?;
        let report = Value::object()
            .with("mode", stability_mode.as_str())
            .with("eigenvalues", followed.stability.eigenvalues.clone())
            .with("residuals", followed.stability.residuals.clone())
            .with("lowest_coupled_eigenvalue", followed.stability.eigenvalues[0])
            .with("lowest_eigenvalue_history", followed.history.clone())
            .with("followed_energy_lowering", followed.energy_lowering)
            .with("stability_iterations", followed.steps)
            .with("davidson_iterations", followed.stability.iterations)
            .with("dimension", followed.stability.dimension)
            .with(
                "normalization",
                "eigenvalues of one half of the coupled variational energy Hessian; restricted electronic amplitudes carry a sqrt(2) factor",
            );
        (followed.result, report)
    };
    let mut derivative = Value::Null;
    let mut response_converged = true;
    if derivative_order > 0 {
        if let Some(direction) = direction {
            let direction: Vec<[f64; 3]> = direction
                .iter()
                .map(|r| {
                    if r.len() == 3 {
                        Ok([r[0], r[1], r[2]])
                    } else {
                        Err(DftError::InvalidInput(
                            "species_direction requires a Cartesian triple per requested site"
                                .into(),
                        ))
                    }
                })
                .collect::<Result<_>>()?;
            let r = derivatives::directional(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                grid,
                &particles,
                &correlation,
                &result,
                &direction,
                &response_config,
            )?;
            response_converged &= r.converged;
            derivative = Value::object()
                .with("order", derivative_order)
                .with("direction", direction)
                .with("gradient", r.gradient)
                .with("response_residual", r.response_residual)
                .with("position_response_residual", r.position_response_residual)
                .with("reference_energy_error", r.reference_energy_error)
                .with("reference_residual", r.reference_residual)
                .with("iterations", r.iterations)
                .with("converged", r.converged);
            if derivative_order == 2 {
                derivative = derivative
                    .with("hessian", r.hessian)
                    .with("explicit_hessian", r.explicit_hessian)
                    .with("orbital_relaxation", r.orbital_relaxation);
            }
        } else {
            let r = derivatives::cartesian(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                grid,
                &particles,
                &correlation,
                &result,
                derivative_order,
                &response_config,
            )?;
            response_converged &= r.converged;
            derivative = Value::object()
                .with("order", derivative_order)
                .with("gradient", r.gradient)
                .with("directional_evaluations", r.directional_evaluations)
                .with("response_residuals", r.response_residuals)
                .with("position_response_residuals", r.position_response_residuals)
                .with("reference_energy_errors", r.reference_energy_errors)
                .with("converged", r.converged);
            if let Some(hessian) = r.hessian {
                derivative = derivative
                    .with("hessian", hessian.as_slice().to_vec())
                    .with("hessian_shape", vec![hessian.rows, hessian.cols]);
            }
        }
        derivative = derivative
            .with("coordinates", if constrain_positions {
                "nuclear sites then added-particle Gaussian sites; each constrained expectation target follows its basis site"
            } else {
                "nuclear sites then added-particle Gaussian sites; Gaussian basis parameters move with their sites"
            })
            .with("added_particle_coordinate_order", particles.iter().enumerate()
                .filter_map(|(index, p)| p.replaced_atom.is_none().then_some(index))
                .collect::<Vec<_>>());
    }
    let mut electric = Value::Null;
    let mut dynamic_electric = Value::Null;
    let mut quadrupole = Value::Null;
    if polarizability || frequency_points.is_some() || quadrupole_origin.is_some() {
        let solver = CoupledResponseSolver::new(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            grid,
            &particles,
            &correlation,
            &result,
        )?;
        if let Some((origin, convention)) = quadrupole_origin {
            let response = solver.electric_quadrupole(&setup.molecule, origin, &response_config)?;
            let (report, converged) = electric::quadrupole_report(&response, origin, &convention);
            quadrupole = report;
            response_converged &= converged;
        }
        if let Some(points) = &frequency_points {
            let reports = solver.dynamic_polarizability(points, &response_config)?;
            dynamic_electric = Value::object()
                .with("frequencies", frequencies.unwrap_or_default())
                .with("frequency_damping", damping)
                .with("tensor_real", reports.iter().map(|r| r.tensor.map(|row| row.map(|z| z.re).to_vec()).to_vec()).collect::<Vec<_>>())
                .with("tensor_imaginary", reports.iter().map(|r| r.tensor.map(|row| row.map(|z| z.im).to_vec()).to_vec()).collect::<Vec<_>>())
                .with("response_residuals", reports.iter().map(|r| r.residuals.to_vec()).collect::<Vec<_>>())
                .with("position_response_residuals", reports.iter().map(|r| r.position_residuals.to_vec()).collect::<Vec<_>>())
                .with("reference_residual", reports[0].reference_residual)
                .with("iterations", reports.iter().map(|r| r.iterations.to_vec()).collect::<Vec<_>>())
                .with("axis_convention", "frequency, observed Cartesian dipole, driven Cartesian field; exp(-i omega t) with z=omega+i eta")
                .with("converged", true);
        }
        if polarizability {
            let a = solver.polarizability(&response_config)?;
            let converged = a.responses.iter().all(|r| r.converged);
            response_converged &= converged;
            electric = Value::object()
                .with("tensor", a.tensor.to_vec())
                .with("raw_asymmetry", a.raw_asymmetry)
                .with(
                    "response_residuals",
                    a.responses
                        .iter()
                        .map(|r| r.relative_residual)
                        .collect::<Vec<_>>(),
                )
                .with(
                    "position_response_residuals",
                    a.responses
                        .iter()
                        .map(|r| r.position_residual)
                        .collect::<Vec<_>>(),
                )
                .with(
                    "iterations",
                    a.responses.iter().map(|r| r.iterations).collect::<Vec<_>>(),
                )
                .with("converged", converged);
        }
    }
    let mut mass_response = Value::Null;
    if quantum_mass_response {
        let r = crate::multicomponent::mass::response(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            grid,
            &particles,
            &correlation,
            &result,
            &response_config,
        )?;
        let converged = r.responses.iter().all(|v| v.converged);
        response_converged &= converged;
        mass_response = Value::object()
            .with("gradient", r.gradient)
            .with("hessian", r.hessian.as_slice().to_vec())
            .with("hessian_shape", vec![r.hessian.rows, r.hessian.cols])
            .with("raw_asymmetry", r.raw_asymmetry)
            .with("reference_number_defects", r.reference_number_defects)
            .with("response_number_defects", r.response_number_defects)
            .with("kinetic_energy_defects", r.kinetic_energy_defects)
            .with(
                "response_residuals",
                r.responses.iter().map(|v| v.relative_residual).collect::<Vec<_>>(),
            )
            .with(
                "reference_residuals",
                r.responses.iter().map(|v| v.reference_residual).collect::<Vec<_>>(),
            )
            .with(
                "position_response_residuals",
                r.responses.iter().map(|v| v.position_residual).collect::<Vec<_>>(),
            )
            .with(
                "iterations",
                r.responses.iter().map(|v| v.iterations).collect::<Vec<_>>(),
            )
            .with("converged", converged)
            .with("convention", "laboratory-frame mass derivatives in quantum-particle order; fixed Gaussian bases, charges, correlation parameters and expectation-position targets; masses in electron masses; no center-of-mass subtraction or particle exchange");
    }
    let position_report = result
        .position_constraints
        .as_ref()
        .map(|c| {
            Value::object()
                .with("targets", c.targets.clone())
                .with("multipliers", c.multipliers.clone())
                .with("residual", c.residual)
                .with("iterations", c.iterations)
                .with("residual_history", c.residual_history.clone())
                .with("convention", "L = E + lambda dot (<r> - target)")
        })
        .unwrap_or(Value::Null);
    let mut warnings=vec!["Experimental laboratory-frame model: distinguishable nuclear/added-particle Hartree orbitals; no exchange between these particles. Electronic and nuclear orbitals are optimized simultaneously; no Born-Oppenheimer electronic surface is constructed.".to_string()];
    if derivative_order > 0 && !constrain_positions {
        warnings.push("Quantum-center derivatives measure finite-basis dependence; use quantum_position_constraint for derivatives with respect to nuclear expectation positions.".into());
    }
    if atoms.len() == setup.molecule.len() {
        warnings.push("All nuclei are quantum particles. The total laboratory kinetic energy includes center-of-mass motion; this is not a translation/rotation-free internal molecular energy.".into());
    } else {
        warnings.push("Unselected nuclei remain clamped classical Coulomb sources; the quantum treatment removes the Born-Oppenheimer separation only inside the selected electronic/nuclear subsystem.".into());
    }
    if correlation.is_zero() {
        warnings.push("Interspecies dynamical correlation is explicitly zero; mean-field proton densities can be excessively localized.".into());
    }
    if requests
        .iter()
        .any(|r| r.kind == "positron" && r.correlation == "none")
    {
        warnings.push("A positron has no electron-positron correlation (positron_correlation or particles.correlation = none); its density and annihilation rate are those of the mean-field model.".into());
    }
    if requests.iter().filter(|r| r.kind != "nucleus").count() > 1 {
        warnings.push("Added particles are distinguishable Hartree orbitals; identical fermions of the same spin are not antisymmetrized.".into());
    }
    if !result.converged {
        warnings.push("The jointly evaluated electronic/nuclear orbital and density-map residuals exceed the requested tolerance.".into());
    }
    let quantum: Vec<_> = result
        .quantum
        .iter()
        .zip(&particles)
        .map(|(q, p)| {
            Value::object()
                .with("label", p.label.as_str())
                .with("charge", p.charge)
                .with("mass", p.mass)
                .with("replaced_atom", p.replaced_atom.map(|a| a as i64))
                .with(
                    "correlation_channel",
                    p.correlation_channel.map(|c| c as i64),
                )
                .with("particle_number", q.particle_number)
                .with("position", q.position.to_vec())
                .with("position_variance", q.position_variance.to_vec())
                .with("kinetic_energy", q.kinetic_energy)
                .with("classical_energy", q.classical_energy)
                .with("orbital_residual", q.orbital_residual)
                .with("density_residual", q.density_residual)
                .with("coefficients", q.coefficient.clone())
                .with("orbital_energies", q.orbital_energies.clone())
                .with("density", q.density.as_slice().to_vec())
                .with("overlap", q.overlap.as_slice().to_vec())
                .with("fock", q.fock.as_slice().to_vec())
                .with("dimension", q.density.rows)
        })
        .collect();
    let components = Value::object()
        .with("electronic_one_body", result.electronic_one_body)
        .with("electronic_coulomb", result.electronic_coulomb)
        .with(
            "electronic_exchange_correlation",
            result.electronic_exchange_correlation,
        )
        .with(
            "electronic_nonlocal_correlation",
            result.electronic_nonlocal_correlation,
        )
        .with("electron_quantum_coulomb", result.electron_quantum_coulomb)
        .with("quantum_pair_coulomb", result.quantum_pair_coulomb)
        .with(
            "quantum_kinetic",
            result.quantum.iter().map(|q| q.kinetic_energy).sum::<f64>(),
        )
        .with(
            "quantum_classical",
            result
                .quantum
                .iter()
                .map(|q| q.classical_energy)
                .sum::<f64>(),
        )
        .with("interspecies_correlation", result.interspecies_correlation)
        .with("classical", result.classical_energy);
    let annihilation_grid;
    let annihilation_grid = match grid {
        Some(g) => g,
        None => {
            annihilation_grid = MolecularGrid::build(&setup.molecule, &setup.grid_options)?;
            &annihilation_grid
        }
    };
    let mut positron_rates = Vec::new();
    for ((request, particle), state) in requests.iter().zip(&particles).zip(&result.quantum) {
        if request.kind != "positron" {
            continue;
        }
        let a = annihilation(
            &setup.basis,
            &result.electronic_density,
            &particle.basis,
            &state.density,
            annihilation_grid,
            enhancement,
            gradient_alpha,
        )?;
        positron_rates.push(
            Value::object()
                .with("label", particle.label.as_str())
                .with("contact_density", a.contact_density)
                .with("enhanced_contact_density", a.enhanced_contact_density)
                .with("rate", a.rate)
                .with("rate_per_ns", a.rate_per_ns())
                .with("lifetime_ps", 1e3 / a.rate_per_ns()),
        );
    }
    let annihilation_report = if positron_rates.is_empty() {
        Value::Null
    } else {
        Value::object()
            .with("enhancement", enhancement_name.as_str())
            .with("gradient_alpha", gradient_alpha)
            .with(
                "convention",
                "lambda = pi r_e^2 c int n_+ n_- gamma(r_s(n_-)) per positron, spin averaged",
            )
            .with("positrons", Value::Array(positron_rates))
    };
    let mut units = super::molecular::units(&[
        ("annihilation.positrons.contact_density", "1/bohr^3"),
        (
            "annihilation.positrons.enhanced_contact_density",
            "1/bohr^3",
        ),
        (
            "annihilation.positrons.rate",
            "atomic units of inverse time (hbar/E_h)^-1",
        ),
        ("annihilation.positrons.rate_per_ns", "1/ns"),
        ("annihilation.positrons.lifetime_ps", "ps"),
        ("annihilation.gradient_alpha", "dimensionless"),
        ("energy", "hartree"),
        ("energy_components", "hartree"),
        ("electronic_kinetic", "hartree"),
        ("energy_history", "hartree"),
        (
            "residual_history",
            "dimensionless (atomic-unit maximum of orbital and density residuals)",
        ),
        ("electronic_orbital_residual", "hartree"),
        ("electronic_density_residual", "dimensionless"),
        ("electronic_density", "dimensionless"),
        ("quantum.charge", "e"),
        ("quantum.mass", "atomic units of mass"),
        (
            "mass_response.gradient",
            "hartree/m_e (m_e is the electron mass)",
        ),
        ("mass_response.hessian", "hartree/m_e^2"),
        ("mass_response.raw_asymmetry", "hartree/m_e^2"),
        ("mass_response.reference_number_defects", "dimensionless"),
        ("mass_response.response_number_defects", "1/m_e"),
        ("mass_response.kinetic_energy_defects", "hartree"),
        ("mass_response.response_residuals", "dimensionless"),
        (
            "mass_response.reference_residuals",
            "dimensionless (atomic-unit orbital and density residuals)",
        ),
        (
            "mass_response.position_response_residuals",
            "bohr/m_e (linearized expectation-position derivative residual)",
        ),
        ("quantum.particle_number", "dimensionless"),
        ("quantum.position", "bohr"),
        ("quantum.position_variance", "bohr^2"),
        ("quantum.kinetic_energy", "hartree"),
        ("quantum.classical_energy", "hartree"),
        ("quantum.orbital_residual", "hartree"),
        ("quantum.density_residual", "dimensionless"),
        ("quantum.coefficients", "dimensionless"),
        ("quantum.orbital_energies", "hartree"),
        ("quantum.density", "dimensionless"),
        ("quantum.overlap", "dimensionless"),
        ("quantum.fock", "hartree"),
        ("geometry_derivatives.direction", "dimensionless"),
        ("geometry_derivatives.gradient", "hartree/bohr"),
        ("geometry_derivatives.hessian", "hartree/bohr^2"),
        ("geometry_derivatives.explicit_hessian", "hartree/bohr^2"),
        ("geometry_derivatives.orbital_relaxation", "hartree/bohr^2"),
        ("geometry_derivatives.response_residual", "dimensionless"),
        ("geometry_derivatives.response_residuals", "dimensionless"),
        (
            "geometry_derivatives.position_response_residuals",
            "dimensionless",
        ),
        (
            "geometry_derivatives.position_response_residual",
            "dimensionless",
        ),
        ("geometry_derivatives.reference_energy_error", "hartree"),
        ("geometry_derivatives.reference_energy_errors", "hartree"),
        (
            "geometry_derivatives.reference_residual",
            "dimensionless (atomic-unit orbital and density residuals)",
        ),
        ("polarizability.tensor", "bohr^3"),
        ("polarizability.raw_asymmetry", "bohr^3"),
        ("polarizability.response_residuals", "dimensionless"),
        ("dynamic_polarizability.frequencies", "hartree"),
        ("dynamic_polarizability.frequency_damping", "hartree"),
        ("dynamic_polarizability.tensor_real", "bohr^3"),
        ("dynamic_polarizability.tensor_imaginary", "bohr^3"),
        ("dynamic_polarizability.response_residuals", "dimensionless"),
        ("dynamic_polarizability.reference_residual", "dimensionless"),
        (
            "dynamic_polarizability.position_response_residuals",
            "e*bohr^2/hartree",
        ),
        (
            "polarizability.position_response_residuals",
            "e*bohr^2/hartree (linearized expectation-position derivative residual)",
        ),
        ("quadrupole.moment", "e*bohr^2"),
        ("quadrupole.origin", "bohr"),
        ("quadrupole.field_gradient_modes", "dimensionless"),
        ("quadrupole.polarizability", "e^2*bohr^4/hartree"),
        ("quadrupole.dipole_moment", "e*bohr"),
        (
            "quadrupole.dipole_from_field_gradient",
            "e^2*bohr^3/hartree",
        ),
        (
            "quadrupole.field_conjugate_moment_from_field",
            "e^2*bohr^3/hartree",
        ),
        ("quadrupole.raw_asymmetry", "e^2*bohr^4/hartree"),
        ("quadrupole.cross_asymmetry", "e^2*bohr^3/hartree"),
        ("quadrupole.response_residuals", "dimensionless"),
        ("quadrupole.reference_residuals", "dimensionless"),
        (
            "quadrupole.dipole_position_response_residuals",
            "e*bohr^2/hartree",
        ),
        (
            "quadrupole.quadrupole_position_response_residuals",
            "e*bohr^3/hartree",
        ),
        ("position_constraints.targets", "bohr"),
        ("position_constraints.multipliers", "hartree/bohr"),
        ("position_constraints.residual", "bohr"),
        ("position_constraints.residual_history", "bohr"),
        ("species_stability.eigenvalues", "hartree"),
        ("species_stability.residuals", "hartree"),
        ("species_stability.lowest_coupled_eigenvalue", "hartree"),
        ("species_stability.lowest_eigenvalue_history", "hartree"),
        ("species_stability.followed_energy_lowering", "hartree"),
    ]);
    let mut report = Value::object().with("model","experimental simultaneous nonrelativistic electronic KS/HF and distinguishable quantum-particle Hartree orbitals; Coulomb interaction and interspecies correlation channels; laboratory frame")
        .with("functional",setup.functional.description.as_str()).with("interspecies_correlation",correlation.descriptions())
        .with("annihilation",annihilation_report)
        .with("energy",result.energy).with("energy_components",components).with("electronic_kinetic",result.electronic_kinetic)
        .with("electronic_density",result.electronic_density.iter().map(|p|p.as_slice().to_vec()).collect::<Vec<_>>())
        .with("electronic_dimension",setup.basis.nao_sph).with("quantum",Value::Array(quantum))
        .with("electronic_orbital_residual",result.electronic_orbital_residual).with("electronic_density_residual",result.electronic_density_residual)
        .with("converged",result.converged && response_converged).with("iterations",result.iterations).with("energy_history",result.energy_history)
        .with("residual_history",result.residual_history).with("experimental",true).with("center_of_mass_removed",false)
        .with("geometry_derivatives",derivative).with("species_stability",stability).with("polarizability",electric).with("position_constraints",position_report)
        .with("mass_response",mass_response).with("dynamic_polarizability",dynamic_electric).with("quadrupole",quadrupole)
        .with("quantum_atoms",atoms).with("warnings",warnings);
    if let Some(field) = applied_gradient {
        report.set("field_gradient", field.amplitudes);
        report.set("field_gradient_origin", field.origin.to_array());
        units.set("field_gradient", "hartree/e/bohr^2");
        units.set("field_gradient_origin", "bohr");
    }
    Ok(report.with("units", units))
}

/// One requested quantum particle.
struct Request {
    kind: &'static str,
    label: String,
    charge: f64,
    mass: f64,
    replaced_atom: Option<usize>,
    center: Vec3,
    exponents: Vec<f64>,
    lmax: usize,
    correlation: String,
}

fn object_f64(entry: &Value, key: &str) -> Result<Option<f64>> {
    match entry.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| DftError::InvalidInput(format!("particles.{key} must be a number"))),
    }
}

fn object_vector(entry: &Value, key: &str) -> Result<Option<Vec<f64>>> {
    match entry.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_array()
            .and_then(|a| a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
            .map(Some)
            .ok_or_else(|| {
                DftError::InvalidInput(format!("particles.{key} must be a list of numbers"))
            }),
    }
}

/// Keys of a `particles` entry.
const PARTICLE_KEYS: [&str; 9] = [
    "kind",
    "atom",
    "position",
    "charge",
    "mass",
    "exponents",
    "lmax",
    "correlation",
    "label",
];

/// The quantum particles requested by `particles`, or by the earlier
/// `quantum_atoms`, `quantum_masses` and `positron_positions` controls.
///
/// A `particles` entry is an object with a `kind`:
/// * `nucleus`: the nucleus of `atom` (charge Z; mass of the bare nucleus,
///   the proton mass for hydrogen), replacing its point charge;
/// * `positron` (charge +1, mass 1), `muon` (positive muon, charge +1) and
///   `negative-muon` (charge -1), added at `position` (Bohr) or at the
///   position of `atom`;
/// * `particle`: an added particle with the given `charge` and `mass`.
///
/// `mass` overrides the default mass of any kind; `exponents` and `lmax`
/// override `quantum_exponents` and `quantum_lmax`; `correlation` names the
/// interspecies functional of the particle (default: the
/// `interspecies_correlation` control for a hydrogen nucleus, the
/// `positron_correlation` control for a positron, `none` otherwise).
fn particle_requests(
    controls: &Controls,
    molecule: &crate::system::Molecule,
    exponents: &[f64],
    lmax: usize,
    proton_correlation: &str,
    positron_correlation: &str,
) -> Result<Vec<Request>> {
    let atoms = controls.opt_usize_list("quantum_atoms")?;
    let masses = controls.opt_f64_list("quantum_masses")?;
    let positrons = controls.opt_rows("positron_positions")?;
    let structured = controls.take("particles");
    if structured.is_some() && (atoms.is_some() || masses.is_some() || positrons.is_some()) {
        return Err(DftError::InvalidInput(
            "give quantum particles either as `particles` or as quantum_atoms/quantum_masses/\
             positron_positions, not both"
                .into(),
        ));
    }
    let nucleus_mass = |z: u8| {
        if z == 1 {
            crate::constants::PROTON_ELECTRON_MASS_RATIO
        } else {
            crate::constants::atomic_mass_au(z) - f64::from(z)
        }
    };
    let atom_of = |atom: usize| -> Result<&crate::system::Atom> {
        molecule.atoms.get(atom).ok_or_else(|| {
            DftError::InvalidInput("a quantum particle names an atom that does not exist".into())
        })
    };
    let mut requests = Vec::new();
    let Some(structured) = structured else {
        let atoms = atoms.unwrap_or_default();
        if masses.as_ref().is_some_and(|m| m.len() != atoms.len()) {
            return Err(DftError::InvalidInput(
                "quantum_masses must match the number of quantum_atoms".into(),
            ));
        }
        for (i, &atom) in atoms.iter().enumerate() {
            let a = atom_of(atom)?;
            requests.push(Request {
                kind: "nucleus",
                label: format!("quantum nucleus {atom}"),
                charge: f64::from(a.z),
                mass: masses.as_ref().map_or_else(|| nucleus_mass(a.z), |m| m[i]),
                replaced_atom: Some(atom),
                center: a.position,
                exponents: exponents.to_vec(),
                lmax,
                correlation: if a.z == 1 { proton_correlation } else { "none" }.to_string(),
            });
        }
        for (i, position) in positrons.unwrap_or_default().iter().enumerate() {
            if position.len() != 3 {
                return Err(DftError::InvalidInput(
                    "positron_positions requires three Cartesian Bohr coordinates per particle"
                        .into(),
                ));
            }
            requests.push(Request {
                kind: "positron",
                label: format!("positron {i}"),
                charge: 1.0,
                mass: 1.0,
                replaced_atom: None,
                center: Vec3::new(position[0], position[1], position[2]),
                exponents: exponents.to_vec(),
                lmax,
                correlation: positron_correlation.to_string(),
            });
        }
        return Ok(requests);
    };
    let entries = structured.as_array().ok_or_else(|| {
        DftError::InvalidInput("particles must be a list of particle objects".into())
    })?;
    for (i, entry) in entries.iter().enumerate() {
        let Value::Object(fields) = entry else {
            return Err(DftError::InvalidInput(
                "every particles entry must be an object".into(),
            ));
        };
        if let Some((key, _)) = fields
            .iter()
            .find(|(k, _)| !PARTICLE_KEYS.contains(&k.as_str()))
        {
            return Err(DftError::InvalidInput(format!(
                "unknown particles key `{key}`; known keys: {}",
                PARTICLE_KEYS.join(", ")
            )));
        }
        let kind = entry
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| DftError::InvalidInput("every particles entry needs a `kind`".into()))?;
        let atom = match entry.get("atom") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_i64().filter(|a| *a >= 0).ok_or_else(|| {
                DftError::InvalidInput("particles.atom must be a zero-based atom index".into())
            })? as usize),
        };
        let position = object_vector(entry, "position")?;
        let mass = object_f64(entry, "mass")?;
        let charge = object_f64(entry, "charge")?;
        let center = match (&position, atom) {
            (Some(r), _) if r.len() == 3 => Vec3::new(r[0], r[1], r[2]),
            (Some(_), _) => {
                return Err(DftError::InvalidInput(
                    "particles.position needs three Cartesian Bohr coordinates".into(),
                ));
            }
            (None, Some(atom)) => atom_of(atom)?.position,
            (None, None) => {
                return Err(DftError::InvalidInput(format!(
                    "particle {i} needs a `position` or an `atom`"
                )));
            }
        };
        let (kind, default_charge, default_mass, replaced_atom, default_correlation) = match kind {
            "nucleus" => {
                let atom = atom.ok_or_else(|| {
                    DftError::InvalidInput("a nucleus particle needs its `atom`".into())
                })?;
                if position.is_some() {
                    return Err(DftError::InvalidInput(
                        "a nucleus particle is centred on its atom; omit `position`".into(),
                    ));
                }
                let z = atom_of(atom)?.z;
                let correlation = if z == 1 { proton_correlation } else { "none" };
                (
                    "nucleus",
                    f64::from(z),
                    nucleus_mass(z),
                    Some(atom),
                    correlation,
                )
            }
            "positron" => ("positron", 1.0, 1.0, None, positron_correlation),
            "muon" => (
                "muon",
                1.0,
                crate::constants::MUON_ELECTRON_MASS_RATIO,
                None,
                "none",
            ),
            "negative-muon" => (
                "negative-muon",
                -1.0,
                crate::constants::MUON_ELECTRON_MASS_RATIO,
                None,
                "none",
            ),
            "particle" => {
                if charge.is_none() || mass.is_none() {
                    return Err(DftError::InvalidInput(
                        "a `particle` entry needs its `charge` and `mass`".into(),
                    ));
                }
                ("particle", 0.0, 0.0, None, "none")
            }
            other => {
                return Err(DftError::InvalidInput(format!(
                    "unknown particle kind `{other}`; use nucleus, positron, muon, \
                     negative-muon or particle"
                )));
            }
        };
        if charge.is_some() && kind != "particle" {
            return Err(DftError::InvalidInput(format!(
                "the charge of a {kind} is fixed; use kind `particle` for another charge"
            )));
        }
        let label = match entry.get("label") {
            None | Some(Value::Null) => format!("{kind} {i}"),
            Some(v) => v
                .as_str()
                .ok_or_else(|| DftError::InvalidInput("particles.label must be text".into()))?
                .to_string(),
        };
        let lmax = match entry.get("lmax") {
            None | Some(Value::Null) => lmax,
            Some(v) => v.as_i64().filter(|l| (0..=3).contains(l)).ok_or_else(|| {
                DftError::InvalidInput("particles.lmax must be an integer from 0 to 3".into())
            })? as usize,
        };
        let correlation = match entry.get("correlation") {
            None | Some(Value::Null) => default_correlation.to_string(),
            Some(v) => v
                .as_str()
                .ok_or_else(|| {
                    DftError::InvalidInput("particles.correlation must be a functional name".into())
                })?
                .to_string(),
        };
        requests.push(Request {
            kind,
            label,
            charge: charge.unwrap_or(default_charge),
            mass: mass.unwrap_or(default_mass),
            replaced_atom,
            center,
            exponents: object_vector(entry, "exponents")?.unwrap_or_else(|| exponents.to_vec()),
            lmax,
            correlation,
        });
    }
    Ok(requests)
}

// Canonical provider identity, not description text: EPC17 variants differ.
fn correlation_key(name: &str) -> Result<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        "epc17" | "epc17-1" => Ok("epc17"),
        "epc17-2" => Ok("epc17-2"),
        "drummond-2011" | "qmc-2011" => Ok("drummond-2011"),
        "none" => Ok("none"),
        _ => Err(DftError::InvalidInput(format!(
            "unknown interspecies correlation `{name}`; use epc17 (epc17-1), epc17-2, drummond-2011 (qmc-2011) or none"
        ))),
    }
}

/// The particles and the correlation model of the requests: one channel
/// per distinct correlation functional other than `none`.
fn particles_and_model(requests: &[Request]) -> Result<(Vec<QuantumParticle>, CorrelationModel)> {
    let mut names: Vec<String> = Vec::new();
    let mut particles = Vec::new();
    for request in requests {
        let key = correlation_key(&request.correlation)?;
        let channel = if key == "none" {
            None
        } else if let Some(c) = names.iter().position(|n| n == key) {
            Some(c)
        } else {
            names.push(key.to_string());
            Some(names.len() - 1)
        };
        particles.push(QuantumParticle {
            label: request.label.clone(),
            charge: request.charge,
            mass: request.mass,
            basis: gaussian_particle_basis(request.center, &request.exponents, request.lmax as u8)?,
            replaced_atom: request.replaced_atom,
            correlation_channel: channel,
        });
    }
    let channels = names
        .iter()
        .map(|name| crate::multicomponent::correlation::by_name(name))
        .collect::<Result<Vec<_>>>()?;
    Ok((particles, CorrelationModel::new(channels)))
}
