// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common native/CLI/ASE collinear correlated excitation nuclear surfaces.
use super::{refuse_fractional_reference, spin_name, Request};
use crate::driver::molecular::{reference_report, units};
use crate::driver::{Controls, System, Value};
use crate::response::correlated_tda::correlated_tda_states;
use crate::response::nuclear::correlated_excitation_path;
use crate::response::spinor_tda::SpinorTdaConfig;
use crate::response::tddft::{TdaSolver, TdaSpin};
use crate::{DftError, Result};

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let root = controls.usize("root", 1)?;
    let request = Request::from_controls(controls, root.max(1))?;
    let memory_mb = controls.usize("td_memory_mb", 256)?;
    let minimum_root_gap = controls.f64("td_minimum_root_gap", 1e-6)?;
    if id == "td-state-gradient" && controls.str("td_gradient_route", "path")? != "path" {
        return Err(DftError::Unsupported {
            feature: "correlated excited-state Z-vector gradient".into(),
            missing: "select the analytic td_gradient_route=path model".into(),
        });
    }
    if root == 0
        || request.nstates < root
        || !minimum_root_gap.is_finite()
        || minimum_root_gap <= 0.0
    {
        return Err(DftError::InvalidInput("correlated excitation paths need a positive selected root, sufficient states and a positive root gap".into()));
    }
    if !request.tamm_dancoff {
        return Err(DftError::Unsupported {
            feature: "correlated RPA nuclear derivatives".into(),
            missing: "the paired non-TDA propagator model".into(),
        });
    }
    let prepared = crate::driver::double_hybrid::prepare(system, controls, id)?;
    let setup = &prepared.setup;
    refuse_fractional_reference(setup)?;
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "correlated excited-state derivatives with SMD surface tension".into(),
            missing: "the matching surface-tension nuclear coefficients".into(),
        });
    }
    prepared.check_path_memory(if id == "td-hessian" { 2 } else { 1 })?;
    let config = prepared
        .config
        .as_ref()
        .expect("correlated excitation response configuration");
    let spin = request
        .spin
        .unwrap_or(if prepared.reference.scf.unrestricted {
            TdaSpin::SpinResolved
        } else {
            TdaSpin::Singlet
        });
    let tda = SpinorTdaConfig {
        states: request.nstates,
        memory_mb: memory_mb.min(prepared.pt2_memory_mb),
        tolerance: request.tolerance,
        minimum_root_gap,
    };
    let mut solver = TdaSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &prepared.reference.scf,
        &setup.options,
        spin,
    )?;
    solver.tol = request.tolerance;
    solver.max_iter = request.max_iterations;
    let static_result = correlated_tda_states(
        &solver,
        &setup.basis,
        &prepared.reference.scf,
        &prepared.eri,
        &prepared.problem,
        &prepared.pt2,
        prepared.coefficients,
        root,
        tda.memory_mb,
    )?;
    let state = static_result.states.get(root - 1).ok_or_else(|| {
        DftError::InvalidInput("correlated root exceeds the parent excitation space".into())
    })?;
    let order = if id == "td-hessian" { 2 } else { 1 };
    let mut relative = 0.0_f64;
    let mut absolute = 0.0_f64;
    let mut frame = 0.0_f64;
    let mut parent_gap = f64::INFINITY;
    let mut double_gap = f64::INFINITY;
    let tensor =
        crate::driver::nuclear_tensor::assemble(setup.molecule.len(), order, 5, |direction| {
            let path = correlated_excitation_path(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                prepared.field_target()?,
                setup.grid.as_ref(),
                &prepared.reference.scf,
                &setup.options,
                prepared.coefficients,
                spin,
                root - 1,
                &direction,
                order,
                &tda,
                config,
            )?;
            relative = path
                .reference_residuals
                .iter()
                .copied()
                .fold(relative, f64::max);
            absolute = absolute
                .max(path.reference_operator_defect)
                .max(path.coupling_duality_defect);
            for value in path
                .parent_residuals
                .iter()
                .chain(&path.double_residuals)
                .chain(&path.amplitude_residuals)
            {
                absolute = absolute.max(*value);
            }
            frame = frame.max(path.orbital_frame_defect);
            parent_gap = parent_gap.min(path.parent_root_gap);
            double_gap = double_gap.min(path.smallest_double_gap);
            let dispersion = crate::driver::correlated_nuclear::dispersion_path(
                &prepared, system, &direction, order,
            )?;
            let mut total = path.total;
            let mut ground = path.ground;
            for k in 0..=order {
                total[k] += dispersion[k];
                ground[k] += dispersion[k];
            }
            Ok(vec![
                total,
                ground,
                path.parent_excitation,
                path.correlation_shift,
                path.excitation,
            ])
        })?;
    let pole_defect = (tensor.energies[4] - state.energy).abs();
    if pole_defect > tda.tolerance {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: pole_defect,
        });
    }
    let mut report=reference_report(setup,&prepared.reference,"analytic nuclear derivatives of the existing normal-ordered full-Coulomb second-order TDA pole and its complete ground energy");
    report.set("td_method", "tda");
    report.set("td_spin", spin_name(spin));
    report.set("td_gradient_route", "path");
    report.set("root", root);
    report.set("correlated_excitation_model", static_result.model);
    report.set("provenance", "analytic");
    report.set("excited_state_energy", tensor.energies[0]);
    report.set("excitation_energy", tensor.energies[4]);
    report.set("parent_excitation_energy", tensor.energies[2]);
    report.set("excitation_correlation_shift", tensor.energies[3]);
    let vectors = |v: &[f64]| {
        v.chunks_exact(3)
            .map(|a| [a[0], a[1], a[2]])
            .collect::<Vec<_>>()
    };
    report.set("gradient", vectors(&tensor.gradients[0]));
    report.set(
        "forces",
        vectors(&tensor.gradients[0].iter().map(|g| -g).collect::<Vec<_>>()),
    );
    report.set("excitation_gradient", vectors(&tensor.gradients[4]));
    report.set(
        "excitation_correlation_gradient",
        vectors(&tensor.gradients[3]),
    );
    report.set("oscillator_strength", state.oscillator_strength);
    report.set("response_residual", relative);
    report.set("propagator_equation_residual", absolute);
    report.set("orbital_frame_defect", frame);
    report.set("parent_pole_defect", pole_defect);
    report.set(
        "minimum_parent_gap",
        if parent_gap.is_finite() {
            Some(parent_gap)
        } else {
            None
        },
    );
    report.set(
        "minimum_double_gap",
        if double_gap.is_finite() {
            Some(double_gap)
        } else {
            None
        },
    );
    report.set("path_evaluations", tensor.paths);
    report.set(
        "response_converged",
        relative <= config.acceptance()
            && frame <= config.acceptance()
            && absolute <= tda.tolerance,
    );
    let mut labels = units(&[
        ("energy", "hartree (ground state)"),
        ("excited_state_energy", "hartree"),
        ("excitation_energy", "hartree"),
        ("parent_excitation_energy", "hartree"),
        ("excitation_correlation_shift", "hartree"),
        ("gradient", "hartree/bohr"),
        ("forces", "hartree/bohr"),
        ("excitation_gradient", "hartree/bohr"),
        ("excitation_correlation_gradient", "hartree/bohr"),
        ("oscillator_strength", "dimensionless"),
        ("response_residual", "dimensionless"),
        ("propagator_equation_residual", "hartree"),
        ("orbital_frame_defect", "dimensionless"),
        ("parent_pole_defect", "hartree"),
        ("minimum_parent_gap", "hartree"),
        ("minimum_double_gap", "hartree"),
    ]);
    if order == 2 {
        report.set("hessian", &tensor.hessians[0]);
        report.set("excitation_hessian", &tensor.hessians[4]);
        report.set("excitation_correlation_hessian", &tensor.hessians[3]);
        for key in [
            "hessian",
            "excitation_hessian",
            "excitation_correlation_hessian",
        ] {
            labels.set(key, "hartree/bohr^2");
        }
        let modes = crate::properties::vibrational::vibrational_analysis(
            &setup.molecule,
            &tensor.hessians[0],
            None,
        )?;
        report.set("frequencies_cm1", modes.frequencies_cm);
        report.set("n_imaginary", modes.n_imaginary);
        report.set("n_projected", modes.n_projected);
        labels.set("frequencies_cm1", "cm^-1 (negative = imaginary)");
    }
    report.set("units", labels);
    crate::driver::double_hybrid::append_prepared_energy(prepared, report)
}
