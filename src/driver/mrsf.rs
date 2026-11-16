// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Mixed-reference spin-flip Tamm-Dancoff states routed through the common
//! driver. The states are built on the restricted open-shell triplet of the
//! molecule, whether the input names the singlet or the triplet.

use super::controls::Controls;
use super::method::{prepare_molecular, MolecularMethod, MolecularReference};
use super::molecular::{reference_report, units};
use super::value::Value;
use super::System;
use crate::response::mrsf::{mrsf_tda, MrsfCoupling};
use crate::{DftError, Result};

pub(super) fn mrsf(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = MolecularMethod::from_controls(controls)?;
    let nstates = controls.usize("nstates", 4)?;
    let coupling_name = controls.str("mrsf_coupling", "mrsf")?;
    let kernel_name = controls.str("spin_flip_kernel", "noncollinear")?;
    controls.finish("mrsf")?;
    if nstates == 0 {
        return Err(DftError::InvalidInput("nstates must be at least 1".into()));
    }
    let coupling = match coupling_name.as_str() {
        "mrsf" => MrsfCoupling::Published,
        "ci" => MrsfCoupling::UnionSpace,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown mrsf_coupling `{other}` (mrsf or ci)"
            )))
        }
    };
    let noncollinear = match kernel_name.as_str() {
        "noncollinear" => true,
        "collinear" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown spin_flip_kernel `{other}` (noncollinear or collinear)"
            )))
        }
    };
    let input_multiplicity = method.multiplicity;
    if input_multiplicity != 1 && input_multiplicity != 3 {
        return Err(DftError::Unsupported {
            feature: format!("mixed-reference spin flip from multiplicity {input_multiplicity}"),
            missing: "a triplet reference; the method gives singlets and triplets only".into(),
        });
    }
    method.multiplicity = 3;
    method.unrestricted = true;
    let setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    if setup.options.pcm.is_some() || setup.options.vv10.is_some() || setup.options.skala.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "mixed-reference spin flip with continuum solvation, nonlocal correlation \
                      or a learned functional"
                .into(),
            missing: "those terms in the spin-flip response kernel".into(),
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
    let states = mrsf_tda(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        &setup.options,
        setup.grid.as_ref(),
        &scf,
        &orbitals,
        nstates,
        coupling,
        noncollinear,
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
    let reference_energy = reference.total_energy();
    let describe = |label: &str, list: &[crate::response::mrsf::MrsfState]| -> Vec<Value> {
        list.iter()
            .enumerate()
            .map(|(k, state)| {
                Value::object()
                    .with("label", format!("{label}{k}"))
                    .with("omega", state.omega)
                    .with("energy", reference_energy + state.omega)
                    .with("s_squared", state.s_squared)
                    .with("oscillator_strength", state.oscillator_strength)
                    .with(
                        "transition_dipole",
                        state.transition_dipole.map(|d| d.to_vec()),
                    )
            })
            .collect()
    };
    let exchange = setup.functional.hybrid.cx != 0.0
        || setup.functional.hybrid.clr != 0.0
        || (noncollinear && !setup.functional.parts.is_empty());
    let mut report = reference_report(
        &setup,
        &reference,
        "mixed-reference spin-flip Tamm-Dancoff states on the restricted open-shell triplet; \
         exact exchange and the non-collinear kernel (spin_flip_kernel)",
    )
    .with("input_multiplicity", input_multiplicity)
    .with("mrsf_coupling", coupling_name)
    .with("spin_flip_kernel", kernel_name)
    .with("singlets", describe("S", &states.singlets))
    .with("triplets", describe("T", &states.triplets))
    .with("spin_flip_asymmetry", states.asymmetry)
    .with("closed_orbitals", states.closed)
    .with("virtual_orbitals", states.virtuals)
    .with("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            (
                "singlets.omega",
                "hartree, relative to the reference triplet",
            ),
            ("singlets.energy", "hartree"),
            ("singlets.s_squared", "hbar^2"),
            (
                "singlets.oscillator_strength",
                "dimensionless, length gauge",
            ),
            ("singlets.transition_dipole", "e*bohr"),
            (
                "triplets.omega",
                "hartree, relative to the reference triplet",
            ),
            ("triplets.energy", "hartree"),
            ("triplets.s_squared", "hbar^2"),
            (
                "triplets.oscillator_strength",
                "dimensionless, length gauge",
            ),
            ("triplets.transition_dipole", "e*bohr"),
            ("spin_flip_asymmetry", "hartree"),
        ]),
    );
    if !exchange {
        if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
            let mut all = existing;
            all.push(Value::Str(
                "WARNING [mrsf] no exact exchange and no non-collinear kernel: nothing couples \
                 the spin flips and the states are orbital-energy differences"
                    .into(),
            ));
            report.set("warnings", Value::Array(all));
        }
    }
    Ok(report)
}

/// Label of a state: `S` or `T` followed by its index in the channel.
fn parse_state(label: &str) -> Result<(bool, usize)> {
    let invalid = || {
        DftError::InvalidInput(format!(
            "unknown mrsf_state `{label}`: S or T followed by the index of the state in its \
             channel, such as S0, S1 or T0"
        ))
    };
    let mut characters = label.chars();
    let singlet = match characters.next() {
        Some('S') | Some('s') => true,
        Some('T') | Some('t') => false,
        _ => return Err(invalid()),
    };
    let root = characters
        .as_str()
        .parse::<usize>()
        .map_err(|_| invalid())?;
    Ok((singlet, root))
}

/// Analytic gradient (`mrsf-state`) or gradient and Hessian (`mrsf-hessian`)
/// of the total energy of one mixed-reference spin-flip state.
pub(super) fn mrsf_surface(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    use crate::response::nuclear::{mixed_reference_surface, MixedReferenceState};
    let with_hessian = id == "mrsf-hessian";
    let config = super::vibrational::response_config(controls)?;
    let mut method = MolecularMethod::from_controls(controls)?;
    let label = controls.str("mrsf_state", "S0")?;
    let coupling_name = controls.str("mrsf_coupling", "mrsf")?;
    let kernel_name = controls.str("spin_flip_kernel", "noncollinear")?;
    controls.finish(id)?;
    let (singlet, root) = parse_state(&label)?;
    let coupling = match coupling_name.as_str() {
        "mrsf" => MrsfCoupling::Published,
        "ci" => MrsfCoupling::UnionSpace,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown mrsf_coupling `{other}` (mrsf or ci)"
            )))
        }
    };
    let noncollinear = match kernel_name.as_str() {
        "noncollinear" => true,
        "collinear" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown spin_flip_kernel `{other}` (noncollinear or collinear)"
            )))
        }
    };
    let input_multiplicity = method.multiplicity;
    if input_multiplicity != 1 && input_multiplicity != 3 {
        return Err(DftError::Unsupported {
            feature: format!("mixed-reference spin flip from multiplicity {input_multiplicity}"),
            missing: "a triplet reference; the method gives singlets and triplets only".into(),
        });
    }
    method.multiplicity = 3;
    method.unrestricted = true;
    let setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    if setup.options.pcm.is_some() || setup.options.vv10.is_some() || setup.options.skala.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "mixed-reference spin flip with continuum solvation, nonlocal correlation \
                      or a learned functional"
                .into(),
            missing: "those terms in the spin-flip response kernel".into(),
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
    // The vertical energy of the same state from the direct solution: the
    // reference value of the path series.
    let states = mrsf_tda(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        &setup.options,
        setup.grid.as_ref(),
        &scf,
        &orbitals,
        root + 1,
        coupling,
        noncollinear,
    )?;
    let list = if singlet {
        &states.singlets
    } else {
        &states.triplets
    };
    let vertical = list
        .get(root)
        .ok_or_else(|| {
            DftError::InvalidInput(format!(
                "mrsf_state {label} exceeds the {} states of its channel",
                list.len()
            ))
        })?
        .omega;
    let surface = mixed_reference_surface(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &scf,
        &orbitals,
        &setup.options,
        MixedReferenceState {
            singlet,
            root,
            coupling,
            noncollinear,
        },
        with_hessian,
        &config,
    )?;
    let name = setup.method.dispersion_functional_name();
    let mut gradient = surface.gradient.clone();
    let mut hessian = surface.hessian.clone();
    if setup.dispersion != crate::dispersion::Dispersion::None {
        let forces =
            crate::dispersion::dispersion_gradient(&setup.molecule, &setup.dispersion, &name)?;
        for (atom, delta) in forces.iter().enumerate() {
            let delta = delta.to_array();
            for axis in 0..3 {
                gradient[3 * atom + axis] += delta[axis];
            }
        }
        if let Some(hessian) = hessian.as_mut() {
            let correction =
                crate::dispersion::dispersion_hessian(&setup.molecule, &setup.dispersion, &name)?;
            for (total, delta) in hessian.as_mut_slice().iter_mut().zip(correction.as_slice()) {
                *total += *delta;
            }
        }
    }
    let e_dispersion =
        crate::dispersion::dispersion_energy(&setup.molecule, &setup.dispersion, &name)?;
    let reference = MolecularReference {
        scf,
        e_dispersion,
        e_smd_cds: 0.0,
        following: None,
    };
    let natoms = setup.molecule.len();
    let mut report = reference_report(
        &setup,
        &reference,
        "analytic nuclear derivatives of the total energy of one mixed-reference spin-flip \
         state: Taylor series of the restricted open-shell triplet and of the eigenvalue of \
         the channel matrix along nuclear paths, with the orbital rotations of the reference \
         solved at every order",
    )
    .with("input_multiplicity", input_multiplicity)
    .with("mrsf_state", label)
    .with("mrsf_coupling", coupling_name)
    .with("spin_flip_kernel", kernel_name)
    .with("omega", surface.omega)
    .with("state_energy", reference.total_energy() + surface.omega)
    .with("omega_reference_error", (surface.omega - vertical).abs())
    .with(
        "gradient",
        gradient
            .chunks_exact(3)
            .map(|g| [g[0], g[1], g[2]])
            .collect::<Vec<_>>(),
    )
    .with("root_gap", surface.gap)
    .with("response_residual", surface.residual)
    .with("flag_defect", surface.flag_defect)
    .with("stationarity_defect", surface.stationarity_defect)
    .with("response_converged", surface.converged)
    .with("path_evaluations", surface.paths)
    .with("provenance", "analytic");
    let mut unit_pairs = vec![
        ("energy", "hartree"),
        ("omega", "hartree"),
        ("state_energy", "hartree"),
        ("omega_reference_error", "hartree"),
        ("gradient", "hartree/bohr"),
        ("root_gap", "hartree"),
        ("response_residual", "dimensionless"),
        ("flag_defect", "dimensionless"),
        ("stationarity_defect", "hartree"),
    ];
    if let Some(hessian) = &hessian {
        let mut translation = 0.0_f64;
        for row in 0..3 * natoms {
            for axis in 0..3 {
                let sum: f64 = (0..natoms).map(|b| hessian[(row, 3 * b + axis)]).sum();
                translation = translation.max(sum.abs());
            }
        }
        let modes =
            crate::properties::vibrational::vibrational_analysis(&setup.molecule, hessian, None)?;
        let rows: Vec<Vec<f64>> = (0..hessian.rows)
            .map(|r| (0..hessian.cols).map(|c| hessian[(r, c)]).collect())
            .collect();
        report.set("hessian", rows);
        report.set("frequencies_cm1", modes.frequencies_cm.clone());
        report.set("n_imaginary", modes.n_imaginary);
        report.set("n_projected", modes.n_projected);
        report.set("translation_residual", translation);
        report.set("symmetrized", true);
        unit_pairs.push(("hessian", "hartree/bohr^2"));
        unit_pairs.push(("frequencies_cm1", "cm^-1"));
        unit_pairs.push(("translation_residual", "hartree/bohr^2"));
    }
    report.set("units", units(&unit_pairs));
    report.set(
        "axis_convention",
        "gradient rows are atoms; rows and columns of the Hessian are 3*atom + axis",
    );
    Ok(report)
}
