// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common Car-Parrinello molecular trajectory driver.

use super::molecular::{setup, units};
use super::{Controls, System, Value};
use crate::math::Vec3;
use crate::periodic::spinor_scf::run_spinor_scf;
use crate::periodic::BlochMatrix;
use crate::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use crate::scf::{self, car_parrinello as cp};
use crate::{DftError, Result};

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let feature = "car-parrinello-md";
    let reference = controls.str("reference", "auto")?.to_ascii_lowercase();
    let roks = matches!(reference.as_str(), "roks" | "rohf");
    let gks = matches!(reference.as_str(), "gks" | "ghf");
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: format!("{feature} with reference={reference}"),
            missing:
                "a correlated off-stationary energy and orbital/nuclear forces in the moving frame"
                    .into(),
        });
    }
    let setup = setup(system, controls)?;
    let options = &setup.options;
    if setup.method.stabilize
        || setup.smd.is_some()
        || !setup.basis.ecps.is_empty()
        || options.density_fit.is_some()
        || options.asymptotic_correction.is_some()
        || options.fixed_occupations.is_some()
        || options.occupation.is_some()
        || options.electronic_temperature != 0.0
        || options.electron_offset != [0.0; 2]
        || gks && (options.skala.is_some() || options.pcm.is_some())
    {
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "the variational all-electron integer-orbital model with analytic off-shell nuclear forces".into(),
        });
    }
    let steps = controls.usize("md_steps", 10)?;
    let dt = controls.f64("md_timestep", 2.0)?;
    let fictitious_mass = controls.f64("fictitious_mass", 400.0)?;
    let tolerance = controls.f64("md_constraint_tolerance", 1e-12)?;
    let n = setup.molecule.len();
    let masses = controls.opt_f64_list("nuclear_masses")?.unwrap_or_else(|| {
        setup
            .molecule
            .atoms
            .iter()
            .map(|atom| crate::constants::atomic_mass_au(atom.z))
            .collect()
    });
    let velocities = controls
        .opt_rows("nuclear_velocities")?
        .unwrap_or_else(|| vec![vec![0.0; 3]; n]);
    let max_rows = options.incore_mb.saturating_mul(1024 * 1024)
        / n.saturating_mul(18 * std::mem::size_of::<f64>());
    if steps == 0
        || steps >= max_rows
        || !dt.is_finite()
        || dt <= 0.0
        || !fictitious_mass.is_finite()
        || fictitious_mass <= 0.0
        || !tolerance.is_finite()
        || !(1e-14..=1e-8).contains(&tolerance)
        || masses.len() != n
        || masses.iter().any(|m| !m.is_finite() || *m <= 0.0)
        || velocities.len() != n
        || velocities
            .iter()
            .any(|v| v.len() != 3 || v.iter().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "Car-Parrinello dynamics needs positive finite masses and timestep, a trajectory within incore_mb, one finite Cartesian velocity per nucleus, and md_constraint_tolerance in [1e-14,1e-8]".into(),
        ));
    }
    let spin_axis = if gks {
        controls
            .opt_vec3("spin_axis")?
            .map_or([0.0, 0.0, 1.0], Vec3::to_array)
    } else {
        [0.0, 0.0, 1.0]
    };
    let xc_tolerance = if gks {
        controls.f64("spinor_xc_tolerance", 1e-10)?
    } else {
        1e-10
    };
    controls.finish(feature)?;
    let (scf, common_orbitals) = if roks {
        let (scf, orbitals) = scf::roks::run_roks_with_orbitals(
            &setup.molecule,
            &setup.basis,
            options,
            &setup.functional,
            setup.grid.as_ref(),
        )?;
        (scf, Some(orbitals))
    } else {
        (
            scf::run_scf(
                &setup.molecule,
                &setup.basis,
                options,
                &setup.functional,
                setup.grid.as_ref(),
            )?,
            None,
        )
    };
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: scf.common_orbital_residual.unwrap_or(f64::INFINITY),
        });
    }
    let common = roks || !scf.unrestricted;
    let coefficients = common_orbitals
        .as_ref()
        .map_or([&scf.mo_coeff[0], &scf.mo_coeff[1]], |c| [c, c]);
    let spins = [scf.n_alpha, scf.n_beta];
    let dispersion_functional = setup.method.dispersion_functional_name();
    let model = cp::Model {
        basis_set: &setup.basis_set,
        functional: &setup.functional,
        grid_options: &setup.grid_options,
        options,
        dispersion: &setup.dispersion,
        dispersion_functional: &dispersion_functional,
        spins,
        common,
    };
    let velocities = velocities
        .iter()
        .map(|v| Vec3::new(v[0], v[1], v[2]))
        .collect();
    let (trajectory, scf_iterations, scf_residual) = if gks {
        let dimension = 2 * setup.basis.nao_sph;
        let backend = MolecularSpinorBackend::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &scf,
            options,
            BlochMatrix::zeros(dimension, dimension),
            spin_axis,
            MolecularSpinorFunctionalModel::SpinCovariant,
            xc_tolerance,
        )?;
        let state = run_spinor_scf(&backend, &setup.spinor_scf_config())?;
        if !state.converged {
            return Err(DftError::ScfNotConverged {
                iterations: state.iterations,
                error: state.orbital_residual,
            });
        }
        let coordinates = cp::initial_spinor_coordinates(
            &setup.basis,
            options.orth_threshold,
            &state.bands[0].coefficients,
            spins.iter().sum(),
        )?;
        let frames = cp::propagate_spinor(
            &model,
            setup.molecule.clone(),
            coordinates,
            velocities,
            &masses,
            fictitious_mass,
            dt,
            steps,
            tolerance,
            xc_tolerance,
        )?;
        (frames, state.iterations, state.orbital_residual)
    } else {
        let coordinates = cp::initial_coordinates(
            &setup.basis,
            options.orth_threshold,
            coefficients,
            spins,
            common,
        )?;
        let frames = cp::propagate(
            &model,
            setup.molecule.clone(),
            coordinates,
            velocities,
            &masses,
            fictitious_mass,
            dt,
            steps,
            tolerance,
        )?;
        let diagnostics = crate::response::diagnostics::reference_diagnostics(&scf)?;
        (
            frames,
            scf.iterations,
            scf.common_orbital_residual
                .unwrap_or(diagnostics.orbital_residual),
        )
    };
    let initial = &trajectory[0];
    let initial_energy = initial.potential_energy + initial.nuclear_kinetic;
    let mut maximum_drift = 0.0_f64;
    let frames: Vec<_> = trajectory
        .into_iter()
        .enumerate()
        .map(|(i, frame)| {
            let extended =
                frame.potential_energy + frame.nuclear_kinetic + frame.electronic_kinetic;
            let drift = extended - initial_energy;
            maximum_drift = maximum_drift.max(drift.abs());
            Value::object()
                .with("step", i)
                .with("time", i as f64 * dt)
                .with("positions", frame.positions)
                .with("velocities", frame.velocities)
                .with("potential_energy", frame.potential_energy)
                .with("nuclear_kinetic_energy", frame.nuclear_kinetic)
                .with("electronic_kinetic_energy", frame.electronic_kinetic)
                .with("extended_energy", extended)
                .with("energy_drift", drift)
                .with(
                    "forces",
                    frame
                        .nuclear_gradient
                        .iter()
                        .map(|g| -*g)
                        .collect::<Vec<_>>(),
                )
                .with("orbital_orthogonality_defect", frame.orthogonality_defect)
                .with("orbital_tangency_defect", frame.tangency_defect)
                .with("minimum_overlap_eigenvalue", frame.minimum_overlap)
        })
        .collect();
    Ok(Value::object()
        .with("model","microcanonical Car-Parrinello extended Lagrangian in the instantaneous symmetric Lowdin AO frame; Euclidean electronic kinetic metric in that frame; Hermitian orbital constraints by RATTLE; initial SCF only")
        .with("reference_type",if gks {"gks"} else if roks {"roks"} else if scf.unrestricted {"uks"} else {"rks"})
        .with("requested_reference",reference).with("functional",setup.functional.description.as_str())
        .with("converged",true).with("scf_iterations",scf_iterations)
        .with("scf_orbital_residual",scf_residual)
        .with("md_timestep",dt).with("fictitious_mass",fictitious_mass)
        .with("nuclear_masses",masses).with("max_extended_energy_drift",maximum_drift)
        .with("frames",frames).with("warnings",Vec::<String>::new())
        .with("units",units(&[
            ("scf_orbital_residual","hartree"),("md_timestep","atomic units of time"),
            ("fictitious_mass","atomic units of mass"),("nuclear_masses","atomic units of mass"),
            ("max_extended_energy_drift","hartree"),("frames.time","atomic units of time"),
            ("frames.positions","bohr"),("frames.velocities","atomic units of velocity"),
            ("frames.potential_energy","hartree"),("frames.nuclear_kinetic_energy","hartree"),
            ("frames.electronic_kinetic_energy","hartree"),("frames.extended_energy","hartree"),
            ("frames.energy_drift","hartree"),("frames.forces","hartree/bohr"),
            ("frames.orbital_orthogonality_defect","dimensionless"),("frames.orbital_tangency_defect","atomic units of inverse time"),
            ("frames.minimum_overlap_eigenvalue","dimensionless"),
        ])))
}
