// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common-driver reporting for full complex spinor TDA states and paths.
use super::method::MolecularSetup;
use super::{Controls, Value};
use crate::periodic::{spinor_scf::SpinorScfReport, BlochMatrix};
use crate::response::spinor_tda::SpinorTdaConfig;
use crate::response::ResponseSolverConfig;
use crate::{DftError, Result};

pub(super) struct Request {
    config: SpinorTdaConfig,
    root: usize,
    amplitudes: bool,
    orbital_analysis: super::nto::Request,
}

impl Request {
    pub(super) fn from_controls(id: &str, controls: &Controls) -> Result<Self> {
        if controls.str("td_method", "tda")? != "tda" {
            return Err(DftError::Unsupported {
                feature: "full spinor RPA excitation surface".into(),
                missing: "paired complex spinor pole and nuclear derivative model".into(),
            });
        }
        let spin = controls.str("td_spin", "auto")?;
        if !matches!(spin.as_str(), "auto" | "spinor") {
            return Err(DftError::Unsupported { feature: format!("exact `{spin}` sector of a general complex spinor reference"), missing: "a spin-adapted excitation model; total spin need not commute with the spinor/SOC Hamiltonian".into() });
        }
        if id == "td-state-gradient" && controls.str("td_gradient_route", "path")? != "path" {
            return Err(DftError::Unsupported {
                feature: "spinor singlet Z-vector gradient".into(),
                missing: "the full-spin Lagrangian; select td_gradient_route=path".into(),
            });
        }
        let root = if id == "excitations" {
            1
        } else {
            controls.usize("root", 1)?
        };
        let states = controls.usize("nstates", 5usize.max(root))?;
        if root == 0 || states < root {
            return Err(DftError::InvalidInput(
                "root is one based and nstates must reach it".into(),
            ));
        }
        let tolerance = controls.f64("td_tolerance", 1e-9)?;
        let minimum_root_gap = controls.f64("td_minimum_root_gap", 1e-6)?;
        if !tolerance.is_finite()
            || tolerance <= 0.0
            || !minimum_root_gap.is_finite()
            || minimum_root_gap <= 0.0
            || states == 0
        {
            return Err(DftError::InvalidInput(
                "spinor TDA requires positive states, tolerance and minimum root gap".into(),
            ));
        }
        Ok(Self {
            config: SpinorTdaConfig {
                states,
                tolerance,
                minimum_root_gap,
                memory_mb: controls.usize("td_memory_mb", 256)?,
            },
            root: root - 1,
            amplitudes: controls.bool("amplitudes", false)?,
            orbital_analysis: if id == "excitations" {
                super::nto::Request::from_controls(controls)?
            } else {
                Default::default()
            },
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append(
    id: &str,
    report: &mut Value,
    setup: &MolecularSetup,
    reference: &SpinorScfReport,
    operator: &BlochMatrix,
    request: &Request,
    config: &ResponseSolverConfig,
) -> Result<()> {
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "spinor excitation surface with SMD".into(),
            missing: "the excited-state solvent geometry model".into(),
        });
    }
    let spectrum = crate::response::nuclear::spinor_excitations(
        &setup.molecule,
        &setup.basis,
        setup.grid.as_ref(),
        &setup.functional,
        reference,
        &setup.options,
        operator,
        &request.config,
        config,
    )?;
    let dipoles = crate::integrals::dipole_matrices(&setup.basis, crate::math::Vec3::zero());
    let states = spectrum
        .states
        .iter()
        .map(|state| {
            let n = setup.basis.nao_sph;
            let mut real = [0.0; 3];
            let mut imaginary = [0.0; 3];
            for axis in 0..3 {
                for spin in 0..2 {
                    for i in 0..n {
                        for j in 0..n {
                            real[axis] -= dipoles[axis][(i, j)]
                                * state.transition_density.re[(spin * n + j, spin * n + i)];
                            imaginary[axis] -= dipoles[axis][(i, j)]
                                * state.transition_density.im[(spin * n + j, spin * n + i)];
                        }
                    }
                }
            }
            let strength = if state.energy > 0.0 {
                Some(
                    (2.0 / 3.0)
                        * state.energy
                        * (0..3)
                            .map(|a| real[a] * real[a] + imaginary[a] * imaginary[a])
                            .sum::<f64>(),
                )
            } else {
                None
            };
            let mut value = Value::object()
                .with("energy", state.energy)
                .with("energy_ev", state.energy * crate::constants::HARTREE_TO_EV)
                .with("transition_dipole_real", real)
                .with("transition_dipole_imaginary", imaginary)
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
            if request.amplitudes {
                value.set("amplitudes_real", &state.amplitudes.re);
                value.set("amplitudes_imaginary", &state.amplitudes.im);
            }
            if request.orbital_analysis.enabled() {
                use crate::properties::natural_transition_orbitals::columns;
                let frame = &reference.bands[0].coefficients;
                let no = state.amplitudes.cols();
                let nv = state.amplitudes.rows();
                request.orbital_analysis.attach(&mut value, vec![super::nto::channel(
                    &state.amplitudes,
                    &columns(frame, no, nv),
                    &columns(frame, 0, no),
                    "spinor",
                    1.0,
                    request.orbital_analysis,
                )?]);
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>>>()?;
    report.set("states", states);
    report.set("td_method", "tda");
    report.set("td_spin", "spinor");
    report.set("excited_state_model","full complex occupied-empty spinor TDA; transverse channels retained; no exact total-spin multiplicity assigned");
    report.set("reference_fock_defect", spectrum.reference_fock_defect);
    report.set("excited_xc_angular_history", spectrum.angular_history);
    report.set("excited_xc_cache_history", spectrum.cache_bound_history);
    let mut labels = report.get("units").cloned().expect("spinor units");
    super::nto::add_units(&mut labels, request.orbital_analysis);
    for (key, unit) in [
        ("states.energy", "hartree"),
        ("states.energy_ev", "eV"),
        ("states.transition_dipole_real", "e*bohr"),
        ("states.transition_dipole_imaginary", "e*bohr"),
        ("states.oscillator_strength", "dimensionless"),
        ("states.residual", "hartree"),
        ("states.nearest_gap", "hartree"),
        ("states.amplitudes_real", "dimensionless"),
        ("states.amplitudes_imaginary", "dimensionless"),
        ("reference_fock_defect", "hartree"),
        (
            "excited_xc_angular_history",
            "[dimensionless, hartree, hartree]",
        ),
        (
            "excited_xc_cache_history",
            "[dimensionless, hartree, hartree]",
        ),
    ] {
        labels.set(key, unit);
    }
    if id != "excitations" {
        let selected = spectrum.states.get(request.root).ok_or_else(|| {
            DftError::InvalidInput("root exceeds the complete spinor singles space".into())
        })?;
        let (mut gradient, mut hessian, residual, paths) = if id == "td-hessian" {
            let result = crate::response::nuclear::spinor_excited_state_hessian(
                &setup.molecule,
                &setup.basis,
                setup.grid.as_ref(),
                &setup.functional,
                reference,
                &setup.options,
                operator,
                request.root,
                &request.config,
                config,
            )?;
            report.set("excitation_hessian", &result.excitation_hessian);
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
                let p = crate::response::nuclear::spinor_excitation_path(
                    &setup.molecule,
                    &setup.basis,
                    setup.grid.as_ref(),
                    &setup.functional,
                    reference,
                    &setup.options,
                    operator,
                    &d,
                    request.root,
                    1,
                    &request.config,
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
        report.set("root", request.root + 1);
        report.set("excitation_energy", selected.energy);
        let ground_energy = report
            .get("energy")
            .and_then(Value::as_f64)
            .expect("spinor energy");
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
        ] {
            labels.set(key, unit);
        }
    }
    report.set("units", labels);
    Ok(())
}
