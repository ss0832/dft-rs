// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Vibrational circular dichroism routed through the common driver.
//!
//! The rotational strength of the fundamental of mode `k` is
//! `R_k = P_k . M_k` (atomic units), with `P_k` the derivative of the
//! electric dipole along the mode (from the atomic polar tensors) and `M_k`
//! the imaginary part of the atomic axial tensors contracted with the mode.
//! The electronic axial tensor of the nuclear coordinate `x` and field axis
//! `b` is the overlap of the two derivative wavefunctions,
//!
//! ```text
//! I_xb = sum_spin sum_ai U^B_ai (U^R_ai + sum_mn C_mi <d chi_m / dR_x | chi_n> C_na)
//! ```
//!
//! with the virtual-occupied amplitudes `U^R` of the nuclear and `U^B` of
//! the magnetic coupled-perturbed response (the factor `i` of the magnetic
//! response taken out), and the nuclear part is
//! `(Z / 4) sum_c epsilon_abc (R - O)_c`. All basis functions share the
//! gauge origin `O`, so the tensors depend on it in a finite basis.

use super::controls::Controls;
use super::molecular::{reference_report, units};
use super::value::Value;
use super::vibrational::{compute_hessian, response_config};
use super::System;
use crate::constants::{atomic_mass_au, HARTREE_TO_CM1};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::magnetic::{magnetic_density_response, Gauge};
use crate::properties::nmr::gradient_operator;
use crate::{DftError, Result};

/// One atomic unit of rotational strength, `e a0` times `e hbar / m_e`, in
/// 1e-44 esu^2 cm^2.
const ATOMIC_UNIT_IN_1E44_CGS: f64 = 4.71444e6;

pub(super) fn vcd(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let gauge = controls.str("gauge", "distributed-origin")?;
    let correlated = if super::double_hybrid::requested(controls)? {
        Some(super::double_hybrid::prepare(system, controls, "vcd")?)
    } else {
        None
    };
    let ordinary = if correlated.is_none() {
        Some(compute_hessian(system, controls)?)
    } else {
        None
    };
    let correlated_vibrational = correlated
        .as_ref()
        .map(|p| super::correlated_nuclear::vibrational(p, system))
        .transpose()?;
    controls.finish("vcd")?;
    // With distributed origins the tensor of every atom has its gauge origin
    // at that atom, which makes the rotational strengths independent of the
    // common origin.
    let distributed = match gauge.as_str() {
        "distributed-origin" => true,
        "common-origin" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown gauge `{other}` for vcd (distributed-origin or common-origin)"
            )))
        }
    };
    let (setup, reference) = if let Some(data) = &correlated {
        (&data.setup, &data.reference)
    } else {
        let data = ordinary.as_ref().expect("ordinary Hessian");
        (&data.setup, &data.reference)
    };
    let (hessian, polar) = if let Some(data) = &correlated_vibrational {
        (&data.hessian, &data.polar)
    } else {
        let data = ordinary.as_ref().expect("ordinary Hessian");
        (&data.hessian, &data.analytic.dipole_derivatives)
    };
    let scf = &reference.scf;
    let molecule = &setup.molecule;
    let natoms = molecule.len();
    let dim = 3 * natoms;
    let origin = match explicit {
        Some(origin) => origin,
        None => {
            // Centre of nuclear charge.
            let mut sum = Vec3::zero();
            let mut charge = 0.0;
            for atom in &molecule.atoms {
                let z = atom.z as f64;
                sum += atom.position * z;
                charge += z;
            }
            if charge <= 0.0 {
                return Err(DftError::InvalidInput(
                    "vcd needs gauge_origin for a structure without nuclear charge".into(),
                ));
            }
            sum * (1.0 / charge)
        }
    };
    let correlated_axial = correlated
        .as_ref()
        .map(|prepared| {
            crate::properties::correlated_axial::correlated_molecular_axial_response(
                molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                prepared.coefficients,
                origin,
                distributed,
                controls.usize("pt2_memory_mb", 2048)?,
                &config,
            )
        })
        .transpose()?;
    let response = if let Some(value) = &correlated_axial {
        AxialData {
            value: value.response.value.clone(),
            converged: value.response.converged,
            residuals: value.response.residuals.clone(),
            warnings: value.response.warnings.clone(),
        }
    } else {
        ordinary_axial(
            ordinary.as_ref().expect("ordinary reference"),
            origin,
            distributed,
            &config,
        )?
    };
    let electronic = response.value.clone();

    // Nuclear part and total.
    let mut nuclear_part = vec![[0.0_f64; 3]; dim];
    let mut total = vec![[0.0_f64; 3]; dim];
    for (atom_index, atom) in molecule.atoms.iter().enumerate() {
        let offset = atom.position - origin;
        let z = atom.z as f64;
        for a in 0..3 {
            for b in 0..3 {
                if a == b {
                    continue;
                }
                let c = 3 - a - b;
                let sign = if (a + 1) % 3 == b { 1.0 } else { -1.0 };
                nuclear_part[3 * atom_index + a][b] = 0.25 * z * sign * offset.get(c);
            }
        }
    }
    if distributed {
        // The tensors with the origin at their own atom are moved to the
        // common origin with the atomic polar tensors, which also carry the
        // nuclear charges: (1/4) sum_cd epsilon_bcd (R - O)_c dmu_d/dR_x.
        for (x, row) in nuclear_part.iter_mut().enumerate() {
            let s = molecule.atoms[x / 3].position - origin;
            let shift = [s.x, s.y, s.z];
            for (b, value) in row.iter_mut().enumerate() {
                let (j, k) = ((b + 1) % 3, (b + 2) % 3);
                *value = 0.25 * (shift[j] * polar[(k, x)] - shift[k] * polar[(j, x)]);
            }
        }
    }
    for x in 0..dim {
        for b in 0..3 {
            total[x][b] = electronic[x][b] + nuclear_part[x][b];
        }
    }

    let modes =
        crate::properties::vibrational::vibrational_analysis(molecule, hessian, Some(polar))?;
    let nmodes = modes.modes.cols;
    let mut dipole_strengths: Vec<Option<f64>> = Vec::with_capacity(nmodes);
    let mut rotational_strengths = Vec::with_capacity(nmodes);
    let mut rotational_cgs = Vec::with_capacity(nmodes);
    for k in 0..nmodes {
        let mut electric = [0.0_f64; 3];
        let mut magnetic = [0.0_f64; 3];
        for x in 0..dim {
            let pattern = modes.modes[(x, k)] / atomic_mass_au(molecule.atoms[x / 3].z).sqrt();
            for b in 0..3 {
                electric[b] += polar[(b, x)] * pattern;
                magnetic[b] += total[x][b] * pattern;
            }
        }
        let omega = modes.frequencies_cm[k] / HARTREE_TO_CM1;
        let strength: f64 = (0..3).map(|b| electric[b] * magnetic[b]).sum();
        dipole_strengths.push(if omega > 0.0 {
            Some((0..3).map(|b| electric[b] * electric[b]).sum::<f64>() / (2.0 * omega))
        } else {
            None
        });
        rotational_strengths.push(strength);
        rotational_cgs.push(strength * ATOMIC_UNIT_IN_1E44_CGS);
    }
    let mut report = reference_report(
        setup,
        reference,
        "vibrational circular dichroism from atomic polar and atomic axial tensors with a common \
         gauge origin",
    )
    .with("gauge", gauge.clone())
    .with("gauge_origin", origin)
    .with("hessian", hessian)
    .with("dipole_derivatives", polar)
    .with("frequencies_cm1", modes.frequencies_cm.clone())
    .with("n_imaginary", modes.n_imaginary)
    .with("atomic_axial_tensors", total)
    .with("atomic_axial_tensors_electronic", electronic)
    .with("atomic_axial_tensors_nuclear", nuclear_part)
    .with("dipole_strengths", dipole_strengths)
    .with("rotational_strengths", rotational_strengths)
    .with("rotational_strengths_1e44_cgs", rotational_cgs)
    .with("response_converged", response.converged
        && correlated_vibrational.as_ref().is_none_or(|v| v.converged))
    .with("converged", scf.converged && response.converged
        && correlated_vibrational.as_ref().is_none_or(|v| v.converged))
    .with("response_residuals", response.residuals.clone())
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("gauge_origin", "bohr"),
            ("hessian", "hartree/bohr^2"),
            ("dipole_derivatives", "e, [dipole axis][3*nucleus + nuclear axis]"),
            ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
            (
                "atomic_axial_tensors",
                "e*bohr/hbar, [3*atom + axis][field axis], imaginary part",
            ),
            ("atomic_axial_tensors_electronic", "e*bohr/hbar"),
            (
                "atomic_axial_tensors_nuclear",
                "e*bohr/hbar, nuclear part (common origin) or polar-tensor origin term (distributed origins)",
            ),
            (
                "dipole_strengths",
                "e^2*bohr^2, null for a mode without a positive frequency",
            ),
            ("rotational_strengths", "e^2*bohr*hbar/m_e"),
            ("rotational_strengths_1e44_cgs", "1e-44 esu^2*cm^2"),
            ("response_residuals", "dimensionless"),
        ]),
    );
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        all.extend(response.warnings.iter().cloned().map(Value::Str));
        if !distributed {
            all.push(Value::Str(
                "WARNING [vcd] common gauge origin: the atomic axial tensors depend on \
                 gauge_origin in a finite basis"
                    .into(),
            ));
        }
        report.set("warnings", Value::Array(all));
    }
    if let Some(data) = correlated_axial {
        report.set("model", data.response.model);
        report.set("parent_atomic_axial_tensors_electronic", data.parent);
        report.set("pt2_axial_channel_bilinears", data.channel_bilinears);
        report.set("pt2_wavefunction_norm", data.norm);
        report.set("pt2_wavefunction_norm_derivatives", data.norm_derivatives);
        report.set("pt2_magnetic_connections", data.magnetic_connections);
        report.set("axial_connection_defects", data.connection_defects);
        report.set("axial_cancellation_defects", data.cancellation_defects);
        report.set("amplitude_response_residuals", data.amplitude_residuals);
        report.set("orbital_frame_defects", data.orbital_frame_defects);
        report.set(
            "vibrational_response_residuals",
            correlated_vibrational
                .as_ref()
                .expect("correlated force field")
                .residuals
                .clone(),
        );
        let mut labels = report.get("units").cloned().expect("VCD units");
        for (key, unit) in [
            ("parent_atomic_axial_tensors_electronic", "e*bohr/hbar"),
            ("pt2_axial_channel_bilinears", "e*bohr/hbar"),
            ("pt2_wavefunction_norm", "dimensionless"),
            ("pt2_wavefunction_norm_derivatives", "1/bohr"),
            ("pt2_magnetic_connections", "e*bohr^2/hbar"),
            ("axial_connection_defects", "dimensionless"),
            ("axial_cancellation_defects", "dimensionless"),
            ("amplitude_response_residuals", "hartree"),
            ("orbital_frame_defects", "dimensionless"),
            ("vibrational_response_residuals", "dimensionless"),
        ] {
            labels.set(key, unit);
        }
        report.set("units", labels);
    }
    if let Some(prepared) = correlated {
        report = super::double_hybrid::append_prepared_energy(prepared, report)?;
    }
    Ok(report)
}

struct AxialData {
    value: Vec<[f64; 3]>,
    converged: bool,
    residuals: Vec<f64>,
    warnings: Vec<String>,
}

fn ordinary_axial(
    data: &super::vibrational::HessianData,
    origin: Vec3,
    distributed: bool,
    config: &crate::response::ResponseSolverConfig,
) -> Result<AxialData> {
    let setup = &data.setup;
    let scf = &data.reference.scf;
    let molecule = &setup.molecule;
    let dim = 3 * molecule.len();
    let response = magnetic_density_response(
        molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        Gauge::CommonOrigin,
        origin,
        config,
    )?;
    // Response to the momentum operator from two more origins: the field
    // perturbation with the origin at O' is the one at O plus
    // (1/2) (O' - O) x grad, and the response is linear in it.
    let momentum: Option<[[Matrix; 3]; 2]> = if distributed {
        let moved = |axis: usize| -> Result<[[Matrix; 3]; 2]> {
            let mut other = origin;
            if axis == 0 {
                other.x += 1.0;
            } else {
                other.y += 1.0;
            }
            Ok(magnetic_density_response(
                molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                Gauge::CommonOrigin,
                other,
                config,
            )?
            .value
            .density_hat)
        };
        let (along_x, along_y) = (moved(0)?, moved(1)?);
        let base = &response.value.density_hat;
        let difference = |a: &Matrix, b: &Matrix, scale: f64| -> Matrix {
            let mut out = a.clone();
            for (t, s) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
                *t = scale * (*t - *s);
            }
            out
        };
        Some(std::array::from_fn(|spin| {
            [
                difference(&along_y[spin][2], &base[spin][2], -2.0),
                difference(&along_x[spin][2], &base[spin][2], 2.0),
                difference(&along_x[spin][1], &base[spin][1], -2.0),
            ]
        }))
    } else {
        None
    };
    let ao_atom = setup.basis.sph_ao_atoms();
    let gradient = gradient_operator(&setup.basis);
    let nao = scf.overlap.rows;

    // Electronic atomic axial tensors, [3 atom + axis][field axis].
    let mut electronic = vec![[0.0_f64; 3]; dim];
    for spin in 0..2 {
        let c = &scf.mo_coeff[spin];
        let no = if spin == 0 { scf.n_alpha } else { scf.n_beta };
        let nv = c.cols - no;
        if no == 0 || nv == 0 {
            continue;
        }
        let sc = scf.overlap.matmul(c);
        // Magnetic amplitudes U^B_ai at [(no + a, i)] for the three field
        // axes, with the gauge origin at the common origin or at one atom.
        let to_mo = |density: &Matrix| sc.transpose_matmul(density).matmul(&sc);
        let common: Vec<Matrix> = (0..3)
            .map(|b| to_mo(&response.value.density_hat[spin][b]))
            .collect();
        let momentum_mo: Option<Vec<Matrix>> = momentum
            .as_ref()
            .map(|m| (0..3).map(|c| to_mo(&m[spin][c])).collect());
        let mut magnetic = common.clone();
        let mut current_atom = usize::MAX;
        for (x, target) in electronic.iter_mut().enumerate() {
            let (atom, axis) = (x / 3, x % 3);
            if let (Some(p), true) = (&momentum_mo, atom != current_atom) {
                // U^B(R_atom) = U^B(O) + (1/2) (R_atom - O) x U^grad.
                let s = molecule.atoms[atom].position - origin;
                let shift = [s.x, s.y, s.z];
                for (b, matrix) in magnetic.iter_mut().enumerate() {
                    let (j, k) = ((b + 1) % 3, (b + 2) % 3);
                    *matrix = common[b].clone();
                    for (value, (pk, pj)) in matrix
                        .as_mut_slice()
                        .iter_mut()
                        .zip(p[k].as_slice().iter().zip(p[j].as_slice()))
                    {
                        *value += 0.5 * (shift[j] * pk - shift[k] * pj);
                    }
                }
                current_atom = atom;
            }
            // Nuclear amplitudes U^R_ai at [(no + a, i)].
            let nuclear = sc
                .transpose_matmul(&data.analytic.density_derivatives[x][spin])
                .matmul(&sc);
            // <d chi_m / dR_x | chi_n> = <chi_m | d/dr_axis | chi_n> for the
            // basis functions m of the displaced atom.
            let mut moved = Matrix::zeros(nao, nao);
            for mu in 0..nao {
                if ao_atom[mu] != atom {
                    continue;
                }
                for nu in 0..nao {
                    moved[(mu, nu)] = gradient[axis][(mu, nu)];
                }
            }
            let half = c.transpose_matmul(&moved).matmul(c);
            for (b, value) in target.iter_mut().enumerate() {
                let mut sum = 0.0;
                for a in 0..nv {
                    for i in 0..no {
                        sum +=
                            magnetic[b][(no + a, i)] * (nuclear[(no + a, i)] + half[(i, no + a)]);
                    }
                }
                *value += sum;
            }
        }
    }

    Ok(AxialData {
        value: electronic,
        converged: response.converged,
        residuals: response.residuals,
        warnings: response.warnings,
    })
}
