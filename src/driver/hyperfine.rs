// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Contact and spin-dipolar hyperfine tensors of an open-shell molecule
//! routed through the common driver.
//!
//! For a nucleus A with nuclear g factor g_A the isotropic coupling is
//!
//! `A_iso [Hartree] = (2 pi / 3) alpha^2 g_e g_A rho_s(R_A) / (m_p two_s)`,
//!
//! where `rho_s(R_A)` is the spin density (alpha minus beta) at the point
//! nucleus in electrons/bohr^3, `two_s = N_alpha - N_beta`, `alpha` the fine
//! structure constant, `g_e` the free-electron g factor and `m_p` the proton
//! mass in electron masses. Nuclei are point charges in the non-relativistic
//! limit. The anisotropic part is the traceless spin-dipolar tensor
//!
//! `A_dip,ab [Hartree] = alpha^2 g_e g_A <rho_s| (3 r_a r_b - delta_ab r^2) / r^5 > / (4 m_p two_s)`
//!
//! with `r` measured from the nucleus: the second derivative of the nuclear
//! attraction integral with respect to the nucleus gives its principal
//! value minus `(4 pi / 3) delta_ab delta(r)`, and the contact density
//! restores the traceless operator.

use super::controls::Controls;
use super::method::run_molecular_reference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::System;
use crate::constants::{
    BOHR_TO_METER, ELECTRON_G_FACTOR, ELEMENTARY_CHARGE_C, HARTREE_TO_JOULE, HBAR_J_S,
};
use crate::math::Vec3;
use crate::{DftError, Result};

pub(super) fn hyperfine(system: &System, controls: &Controls) -> Result<Value> {
    let nuclear = super::hyperfine_nuclear::NuclearOptions::parse(controls, system.numbers.len())?;
    if nuclear.order > 0 && super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "correlated hyperfine nuclear derivatives".into(),
            missing:
                "a correlated nuclear-A response; an SCF parent cannot replace the requested recipe"
                    .into(),
        });
    }
    let response_config = if nuclear.order > 0 {
        let mut config = super::vibrational::response_config(controls)?;
        if !controls.contains("response_tolerance") && !controls.contains("response_tol") {
            config.tolerance = 1e-12;
        }
        if !controls.contains("response_max_iterations") && !controls.contains("response_max_iter")
        {
            config.max_iterations = 1000;
        }
        Some(config)
    } else {
        None
    };
    let setup = setup(system, controls)?;
    nuclear.validate_setup(&setup)?;
    let nuclear_g = controls.opt_f64_list("nuclear_g")?;
    controls.finish("hyperfine")?;
    let natoms = setup.molecule.len();
    if let Some(g) = &nuclear_g {
        if g.len() != natoms {
            return Err(DftError::InvalidInput(format!(
                "nuclear_g has {} values for {natoms} atoms; give one nuclear g factor per atom",
                g.len()
            )));
        }
    }
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    if scf.n_alpha == scf.n_beta {
        return Err(DftError::InvalidInput(
            "hyperfine needs an open-shell reference: set multiplicity above 1 for a molecule \
             with unpaired electrons"
                .into(),
        ));
    }
    let two_s = scf.n_alpha as f64 - scf.n_beta as f64;

    // Basis functions at the nuclei, then sum_mn phi_m phi_n (P_alpha +- P_beta)_mn.
    let points: Vec<Vec3> = setup
        .molecule
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect();
    let batch = crate::grid::eval::eval_aos(&setup.basis, &points, false);
    let nao = setup.basis.nao_sph;
    let density_at = |atom: usize| -> (f64, f64) {
        let mut total = 0.0;
        let mut spin = 0.0;
        for m in 0..nao {
            for n in 0..nao {
                let weight = batch.phi[(atom, m)] * batch.phi[(atom, n)];
                let pa = scf.density[0][(m, n)];
                let pb = scf.density[1][(m, n)];
                total += weight * (pa + pb);
                spin += weight * (pa - pb);
            }
        }
        (total, spin)
    };
    let (total, spin): (Vec<f64>, Vec<f64>) = (0..natoms).map(density_at).unzip();

    // An atom with an effective core potential has no all-electron contact
    // density; its outputs are reported as null.
    let charges = setup.basis.effective_charges(&setup.molecule);
    let all_electron: Vec<bool> = setup
        .molecule
        .atoms
        .iter()
        .zip(&charges)
        .map(|(atom, z)| *z == atom.z as f64)
        .collect();
    let masked = |values: &[f64]| -> Vec<Option<f64>> {
        values
            .iter()
            .zip(&all_electron)
            .map(|(v, &ae)| ae.then_some(*v))
            .collect()
    };

    // Constants of the Fermi-contact formula and its unit conversions.
    let pi = std::f64::consts::PI;
    let hartree_to_mhz = HARTREE_TO_JOULE / (2.0 * pi * HBAR_J_S) * 1.0e-6;
    let field_gauss = HBAR_J_S / (ELEMENTARY_CHARGE_C * BOHR_TO_METER * BOHR_TO_METER) * 1.0e4;
    // One functional-independent tensor kernel supplies the Rust and all
    // common-driver surfaces. ECP sites retain the existing null convention.
    let mut spin_matrix = scf.density[0].clone();
    for (value, beta) in spin_matrix
        .as_mut_slice()
        .iter_mut()
        .zip(scf.density[1].as_slice())
    {
        *value -= beta;
    }
    let mut tensors = Vec::with_capacity(natoms);
    for (atom, site) in setup.molecule.atoms.iter().enumerate() {
        if !all_electron[atom] {
            tensors.push(None);
            continue;
        }
        tensors.push(Some(crate::properties::hyperfine::spin_hyperfine_tensor(
            &setup.basis,
            &spin_matrix,
            site.position,
            two_s,
        )?));
    }
    // Derive the compatibility scalar from the same tensor kernel. ECP
    // entries are masked below; their pseudo-density is never converted.
    let per_g_mhz: Vec<f64> = tensors
        .iter()
        .map(|t| t.as_ref().map_or(0.0, |t| t.contact))
        .collect();
    let per_g_hartree: Vec<f64> = per_g_mhz.iter().map(|a| a / hartree_to_mhz).collect();
    let dipolar_per_g_mhz: Vec<Option<Vec<Vec<f64>>>> = tensors
        .iter()
        .map(|tensor| {
            tensor
                .as_ref()
                .map(|t| t.dipolar.iter().map(|row| row.to_vec()).collect())
        })
        .collect();

    let ecp_atoms: Vec<usize> = all_electron
        .iter()
        .enumerate()
        .filter(|(_, &ae)| !ae)
        .map(|(atom, _)| atom)
        .collect();

    let mut report = reference_report(
        &setup,
        &reference,
        "isotropic Fermi-contact hyperfine couplings from the spin density at the nuclei and \
         spin-dipolar tensors of an open-shell reference",
    );
    report.set("two_s", two_s);
    report.set("density_at_nuclei", masked(&total));
    report.set("spin_density_at_nuclei", masked(&spin));
    report.set("isotropic_coupling_per_g_mhz", masked(&per_g_mhz));
    report.set(
        "total_coupling_per_g_mhz",
        tensors
            .iter()
            .map(|t| t.as_ref().map(|t| t.total))
            .collect::<Vec<_>>(),
    );
    report.set(
        "principal_couplings_per_g_mhz",
        tensors
            .iter()
            .map(|t| t.as_ref().map(|t| t.principal_values))
            .collect::<Vec<_>>(),
    );
    report.set(
        "principal_axes_per_g",
        tensors
            .iter()
            .map(|t| t.as_ref().map(|t| t.principal_axes))
            .collect::<Vec<_>>(),
    );
    report.set(
        "dipolar_trace_per_g_mhz",
        tensors
            .iter()
            .map(|t| t.as_ref().map(|t| t.dipolar_trace))
            .collect::<Vec<_>>(),
    );
    report.set(
        "tensor_asymmetry_per_g_mhz",
        tensors
            .iter()
            .map(|t| t.as_ref().map(|t| t.symmetry_residual))
            .collect::<Vec<_>>(),
    );
    report.set(
        "tensor_eigen_residual_per_g_mhz",
        tensors
            .iter()
            .map(|t| t.as_ref().map(|t| t.eigen_residual))
            .collect::<Vec<_>>(),
    );
    report.set(
        "tensor_axes_orthogonality_residual",
        tensors
            .iter()
            .map(|t| t.as_ref().map(|t| t.axes_orthogonality_residual))
            .collect::<Vec<_>>(),
    );
    report.set(
        "tensor_relative_residual",
        tensors
            .iter()
            .map(|t| t.as_ref().map(|t| t.relative_residual()))
            .collect::<Vec<_>>(),
    );
    let tensor_tolerance = 1e-10;
    let verdicts: Vec<_> = tensors
        .iter()
        .map(|t| {
            t.as_ref().map(|t| {
                t.relative_residual() <= tensor_tolerance
                    && t.axes_orthogonality_residual <= tensor_tolerance
            })
        })
        .collect();
    report.set("tensor_converged", verdicts.clone());
    report.set("tensor_residual_tolerance", tensor_tolerance);
    report.set("tensor_axis_convention","Cartesian tensor [nucleus][a][b]; principal axes [nucleus][Cartesian][principal] are columns with ascending principal values; axis signs and bases inside degenerate eigenspaces are not unique");
    report.set("reference_tensor_convention","Nonrelativistic point-nucleus tensor of the converged collinear spin density normalized by 2 M_S; no averaging over degenerate electronic states; compare rotations with consistent electronic-state selection");
    if verdicts.contains(&Some(false)) {
        report.set("converged", false);
        push_warning(&mut report,"WARNING [hyperfine] a raw tensor or eigensystem residual exceeds the reported tolerance");
    }
    let mut pairs = vec![
        ("two_s", "dimensionless, N_alpha minus N_beta"),
        (
            "density_at_nuclei",
            "electrons/bohr^3, alpha plus beta density at each nucleus; null for effective core potential atoms",
        ),
        (
            "spin_density_at_nuclei",
            "electrons/bohr^3, alpha minus beta density at each nucleus; null for effective core potential atoms",
        ),
        (
            "isotropic_coupling_per_g_mhz",
            "MHz, A_iso/g_A (independent of the nuclear g factor); null for effective core potential atoms",
        ),
    ];
    report.set("dipolar_coupling_per_g_mhz", dipolar_per_g_mhz.clone());
    pairs.push((
        "dipolar_coupling_per_g_mhz",
        "MHz, traceless spin-dipolar tensor A_dip/g_A as [nucleus][a][b]; null for effective core potential atoms",
    ));
    pairs.extend([
        ("total_coupling_per_g_mhz","MHz, contact times identity plus spin-dipolar tensor per nuclear g; null for ECP nuclei"),
        ("principal_couplings_per_g_mhz","MHz, ascending principal values per nuclear g"),
        ("principal_axes_per_g","dimensionless, columns match principal_couplings_per_g_mhz"),
        ("dipolar_trace_per_g_mhz","MHz, signed raw spin-dipolar trace before any projection"),
        ("tensor_asymmetry_per_g_mhz","MHz, raw maximum A_ab minus A_ba magnitude"),
        ("tensor_eigen_residual_per_g_mhz","MHz, raw maximum A V minus V diag(values) magnitude"),
        ("tensor_axes_orthogonality_residual","dimensionless, raw maximum V^T V minus identity magnitude"),
        ("tensor_relative_residual","dimensionless, trace and operator residuals divided by the actual full tensor norm"),
        ("tensor_residual_tolerance","dimensionless, numerical tensor and orthogonality acceptance tolerance"),
    ]);
    if let Some(g) = &nuclear_g {
        let coupling: Vec<f64> = per_g_mhz
            .iter()
            .zip(g)
            .map(|(a, factor)| a * factor)
            .collect();
        // Gauss value: A_iso / (g_e/2) in units of hbar/(e a0^2), in gauss.
        let gauss: Vec<f64> = per_g_hartree
            .iter()
            .zip(g)
            .map(|(a, factor)| a * factor / (0.5 * ELECTRON_G_FACTOR) * field_gauss)
            .collect();
        report.set("nuclear_g", g.clone());
        report.set("isotropic_coupling_mhz", masked(&coupling));
        report.set("isotropic_coupling_gauss", masked(&gauss));
        let dipolar: Vec<Option<Vec<Vec<f64>>>> = dipolar_per_g_mhz
            .iter()
            .zip(g)
            .map(|(tensor, factor)| {
                tensor.as_ref().map(|rows| {
                    rows.iter()
                        .map(|row| row.iter().map(|v| v * factor).collect())
                        .collect()
                })
            })
            .collect();
        report.set("dipolar_coupling_mhz", dipolar);
        let absolute: Vec<_> = tensors
            .iter()
            .zip(g)
            .map(|(tensor, g)| tensor.as_ref().map(|t| t.scaled(*g)).transpose())
            .collect::<Result<_>>()?;
        report.set(
            "total_coupling_mhz",
            absolute
                .iter()
                .map(|t| t.as_ref().map(|t| t.total))
                .collect::<Vec<_>>(),
        );
        report.set(
            "principal_couplings_mhz",
            absolute
                .iter()
                .map(|t| t.as_ref().map(|t| t.principal_values))
                .collect::<Vec<_>>(),
        );
        report.set(
            "principal_axes",
            absolute
                .iter()
                .map(|t| t.as_ref().map(|t| t.principal_axes))
                .collect::<Vec<_>>(),
        );
        pairs.extend([
            (
                "total_coupling_mhz",
                "MHz, physical contact plus spin-dipolar tensor with supplied nuclear g",
            ),
            (
                "principal_couplings_mhz",
                "MHz, ascending physical principal values including signed nuclear g",
            ),
            (
                "principal_axes",
                "dimensionless, columns match principal_couplings_mhz including negative nuclear g",
            ),
        ]);
        pairs.push((
            "dipolar_coupling_mhz",
            "MHz, A_dip = g_A times dipolar_coupling_per_g_mhz; null for effective core potential atoms",
        ));
        pairs.extend([
            ("nuclear_g", "dimensionless, nuclear g factor per atom as given"),
            (
                "isotropic_coupling_mhz",
                "MHz, A_iso = g_A times isotropic_coupling_per_g_mhz; null for effective core potential atoms",
            ),
            (
                "isotropic_coupling_gauss",
                "gauss, A_iso/(g_e/2) expressed as a field in units of hbar/(e a0^2); null for effective core potential atoms",
            ),
        ]);
    }
    // At a nucleus with an effective core potential only the valence
    // pseudo-orbitals exist. Their spin density there is reported under its
    // own key: it is not the contact density of the all-electron atom, so no
    // coupling is derived from it.
    if !ecp_atoms.is_empty() {
        let pseudo: Vec<Option<f64>> = spin
            .iter()
            .zip(&all_electron)
            .map(|(v, &ae)| (!ae).then_some(*v))
            .collect();
        report.set("pseudo_valence_spin_density_at_nuclei", pseudo);
        pairs.push((
            "pseudo_valence_spin_density_at_nuclei",
            "electrons/bohr^3, alpha minus beta of the valence pseudo-orbitals at nuclei with an effective core potential (null elsewhere); not an all-electron contact density",
        ));
    }
    if !ecp_atoms.is_empty() {
        push_warning(
            &mut report,
            &format!(
                "WARNING [hyperfine] atoms {ecp_atoms:?} carry effective core potentials; their \
                 contact densities and couplings are reported as null; the spin density of the valence pseudo-orbitals is reported separately"
            ),
        );
    }
    report.set("units", units(&pairs));
    if let Some(config) = response_config.as_ref() {
        nuclear.append(&mut report, &setup, scf, config, nuclear_g.as_deref())?;
    }
    Ok(report)
}

fn push_warning(report: &mut Value, text: &str) {
    if let Value::Object(entries) = report {
        if let Some((_, Value::Array(warnings))) = entries.iter_mut().find(|(k, _)| k == "warnings")
        {
            warnings.push(Value::Str(text.into()));
        }
    }
}
