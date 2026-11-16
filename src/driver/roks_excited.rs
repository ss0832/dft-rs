// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common-orbital flag excitation space on the shared calculation surface.
use super::method::MolecularReference;
use super::molecular::{reference_report, setup};
use super::{Controls, System, Value};
use crate::linalg::Matrix;
use crate::{DftError, Result};

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    if controls.str("td_method", "tda")? != "tda" {
        return Err(DftError::Unsupported {
            feature: "common-orbital flag RPA spectrum".into(),
            missing: "the paired flag eigenproblem and transition normalization".into(),
        });
    }
    let spin = controls.str("td_spin", "auto")?;
    if !matches!(spin.as_str(), "auto" | "reference-spin") {
        return Err(DftError::Unsupported {
            feature: format!("common-orbital excitation sector {spin}"),
            missing: "an explicitly spin-changing configuration space; use td_spin=reference-spin for the spin-conserving flag model".into(),
        });
    }
    let root = if id == "excitations" {
        1
    } else {
        controls.usize("root", 1)?
    };
    let nstates = controls.usize("nstates", 5usize.max(root))?;
    let tolerance = controls.f64("td_tolerance", 1e-9)?;
    let memory = controls.usize("td_memory_mb", 256)?;
    let amplitudes = controls.bool("amplitudes", false)?;
    let orbital_analysis = if id == "excitations" {
        super::nto::Request::from_controls(controls)?
    } else {
        Default::default()
    };
    if orbital_analysis.needs_difference() {
        return Err(DftError::Unsupported {
            feature: "common-orbital flag difference density".into(),
            missing: "the excited-state one-particle density for the common-orbital flag model".into(),
        });
    }
    let minimum_root_gap = controls.f64("td_minimum_root_gap", 1e-6)?;
    if id == "td-state-gradient" && controls.str("td_gradient_route", "path")? != "path" {
        return Err(DftError::Unsupported {
            feature: "common-orbital singlet Z-vector".into(),
            missing: "use the analytic flag td_gradient_route=path".into(),
        });
    }
    if root == 0
        || nstates < root
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || !minimum_root_gap.is_finite()
        || minimum_root_gap <= 0.0
    {
        return Err(DftError::InvalidInput(
            "flag TDA requires positive states and tolerance".into(),
        ));
    }
    let mut config = super::vibrational::response_config(controls)?;
    config.accept_residual = Some(config.acceptance().min(tolerance));
    let setup = setup(system, controls)?;
    controls.finish(id)?;
    if setup.smd.is_some() || setup.method.stabilize {
        return Err(DftError::Unsupported {
            feature: "common-orbital flag excitations with SMD or unrestricted stability following"
                .into(),
            missing: "a compatible common-orbital reference model".into(),
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
    let spectrum = crate::response::nuclear::open_shell_tda_states(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &scf,
        &orbitals,
        &setup.options,
        nstates,
        memory,
        &config,
    )?;
    let nc = scf.n_beta;
    let no = scf.n_alpha - scf.n_beta;
    let nv = orbitals.cols - nc - no;
    let dipoles = crate::integrals::dipole_matrices(&setup.basis, crate::math::Vec3::zero());
    let states = spectrum
        .states
        .iter()
        .map(|state| {
            // The lower interclass block of (delta P_alpha+delta P_beta)
            // is the non-Hermitian transition, with the same Berry normalization.
            let mut transition = Matrix::zeros(orbitals.cols, orbitals.cols);
            let mut k = 0;
            for i in 0..nc {
                for q in 0..no {
                    transition[(nc + q, i)] = state.rotations[k];
                    k += 1;
                }
            }
            for i in 0..nc {
                for a in 0..nv {
                    transition[(nc + no + a, i)] = 2.0 * state.rotations[k];
                    k += 1;
                }
            }
            for q in 0..no {
                for a in 0..nv {
                    transition[(nc + no + a, nc + q)] = state.rotations[k];
                    k += 1;
                }
            }
            let ao = orbitals
                .matmul_seq(&transition)
                .matmul_seq(&orbitals.transpose());
            let dipole: [f64; 3] = std::array::from_fn(|axis| {
                -(0..ao.rows)
                    .flat_map(|i| (0..ao.cols).map(move |j| (i, j)))
                    .map(|(i, j)| dipoles[axis][(i, j)] * ao[(j, i)])
                    .sum::<f64>()
            });
            let strength = if state.energy > 0.0 {
                Some((2.0 / 3.0) * state.energy * dipole.iter().map(|d| d * d).sum::<f64>())
            } else {
                None
            };
            let mut value = Value::object()
                .with("energy", state.energy)
                .with("energy_ev", state.energy * crate::constants::HARTREE_TO_EV)
                .with("transition_dipole", dipole)
                .with("oscillator_strength", strength)
                .with("residual", state.residual)
                .with(
                    "nearest_gap",
                    if state.nearest_gap.is_finite() {
                        Some(state.nearest_gap)
                    } else {
                        None
                    },
                );
            if amplitudes {
                value.set("flag_amplitudes", state.rotations.clone());
            }
            if orbital_analysis.enabled() {
                use crate::properties::natural_transition_orbitals::real;
                let frame = real(orbitals.clone());
                orbital_analysis.attach(&mut value, vec![super::nto::channel(
                    &real(transition), &frame, &frame, "charge", 1.0, orbital_analysis,
                )?]);
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>>>()?;
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
    let mut report=reference_report(&setup,&reference,"common-orbital spin-conserving flag TDA")
        .with("reference_type","restricted open-shell").with("td_method","tda")
        .with("td_spin","reference-spin").with("states",states)
        .with("excited_state_model","common closed/open/virtual flag; Berry weights (1,2,1); spin-changing sectors require a separate excitation space")
        .with("berry_weights",spectrum.berry_weights)
        .with("reference_stationarity",spectrum.reference_stationarity)
        .with("reference_metric_defect",spectrum.metric_defect);
    let mut units =
        super::molecular::units(&[("energy", "hartree"), ("energy_components", "hartree")]);
    for (key, unit) in [
        ("states.energy", "hartree"),
        ("states.energy_ev", "eV"),
        ("states.transition_dipole", "e*bohr"),
        ("states.oscillator_strength", "dimensionless"),
        ("states.residual", "hartree"),
        ("states.nearest_gap", "hartree"),
        ("states.flag_amplitudes", "dimensionless"),
        ("berry_weights", "dimensionless"),
        ("reference_stationarity", "hartree"),
        ("reference_metric_defect", "dimensionless"),
    ] {
        units.set(key, unit);
    }
    super::nto::add_units(&mut units, orbital_analysis);
    report.set("units", units);
    if id != "excitations" {
        let tda = crate::response::spinor_tda::SpinorTdaConfig {
            states: nstates,
            tolerance,
            minimum_root_gap,
            memory_mb: memory,
        };
        append_derivatives(
            id,
            &mut report,
            &setup,
            &reference,
            &orbitals,
            &spectrum.states,
            root - 1,
            &tda,
            &config,
        )?;
    }
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn append_derivatives(
    id: &str,
    report: &mut Value,
    setup: &super::method::MolecularSetup,
    reference: &MolecularReference,
    orbitals: &Matrix,
    states: &[crate::response::nuclear::OpenShellTdaState],
    root: usize,
    tda: &crate::response::spinor_tda::SpinorTdaConfig,
    config: &crate::response::ResponseSolverConfig,
) -> Result<()> {
    let mut labels = report.get("units").cloned().expect("common-orbital units");
    if id != "excitations" {
        let selected = states.get(root).ok_or_else(|| {
            DftError::InvalidInput("root exceeds the complete spinor singles space".into())
        })?;
        let (mut gradient, mut hessian, residual, paths) = if id == "td-hessian" {
            let result = crate::response::nuclear::open_shell_excited_state_hessian(
                &setup.molecule,
                &setup.basis,
                setup.grid.as_ref(),
                &setup.functional,
                &reference.scf,
                orbitals,
                &setup.options,
                root,
                tda,
                config,
            )?;
            report.set("excitation_hessian", &result.excitation_hessian);
            report.set("maximum_stationarity_defect", result.reference_residual);
            (
                result.gradient,
                Some(result.hessian),
                result.residual,
                result.paths,
            )
        } else {
            let mut gradient = vec![0.0; 3 * setup.molecule.len()];
            let mut residual = 0.0_f64;
            for (k, g) in gradient.iter_mut().enumerate() {
                let mut d = vec![[0.0; 3]; setup.molecule.len()];
                d[k / 3][k % 3] = 1.0;
                let p = crate::response::nuclear::open_shell_excitation_path(
                    &setup.molecule,
                    &setup.basis,
                    &setup.functional,
                    setup.grid.as_ref(),
                    &reference.scf,
                    orbitals,
                    &setup.options,
                    &crate::response::nuclear::MolecularDirection::nuclear(d),
                    root,
                    1,
                    tda,
                    config,
                )?;
                *g = p.ground[1] + p.excitation[1];
                residual = residual.max(p.residual);
            }
            let paths = gradient.len();
            (gradient, None, residual, paths)
        };
        if setup.dispersion != crate::dispersion::Dispersion::None {
            let name = setup.method.dispersion_functional_name();
            let delta =
                crate::dispersion::dispersion_gradient(&setup.molecule, &setup.dispersion, &name)?;
            for (row, d) in gradient.chunks_exact_mut(3).zip(delta) {
                for (v, x) in row.iter_mut().zip(d.to_array()) {
                    *v += x;
                }
            }
            if let Some(h) = &mut hessian {
                let delta = crate::dispersion::dispersion_hessian(
                    &setup.molecule,
                    &setup.dispersion,
                    &name,
                )?;
                for (v, x) in h.as_mut_slice().iter_mut().zip(delta.as_slice()) {
                    *v += x;
                }
            }
        }
        report.set(
            "gradient",
            gradient
                .chunks_exact(3)
                .map(|p| p.to_vec())
                .collect::<Vec<_>>(),
        );
        if let Some(h) = hessian {
            report.set("hessian", h);
        }
        report.set("root", root + 1);
        report.set("excitation_energy", selected.energy);
        let ground_energy = report
            .get("energy")
            .and_then(Value::as_f64)
            .expect("common-orbital energy");
        report.set("excited_state_energy", ground_energy + selected.energy);
        report.set("td_gradient_route", "path");
        report.set("provenance", "analytic");
        report.set("response_residual", residual);
        report.set("response_converged", true);
        report.set("path_evaluations", paths);
        for (key, unit) in [
            ("root", "dimensionless"),
            ("gradient", "hartree/bohr"),
            ("hessian", "hartree/bohr^2"),
            ("excitation_hessian", "hartree/bohr^2"),
            ("excitation_energy", "hartree"),
            ("excited_state_energy", "hartree"),
            ("response_residual", "dimensionless"),
            ("path_evaluations", "dimensionless"),
            ("maximum_stationarity_defect", "hartree"),
        ] {
            labels.set(key, unit);
        }
    }
    report.set("units", labels);
    Ok(())
}
