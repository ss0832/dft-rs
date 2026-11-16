// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electronic g tensor of an open-shell molecule, routed through the common
//! driver.
//!
//! For the effective spin Hamiltonian `mu_B B . g . S` of the high-spin
//! state `M_S = S` with spin density `P_s = P_alpha - P_beta`:
//!
//! ```text
//! g_kl = g_e delta_kl + dg_RMC delta_kl + dg_GC_kl + dg_SO_kl
//! dg_RMC   = -(alpha^2 g_e / 2S) Tr[P_s T]
//! dg_GC_kl =  (alpha^2 / 2S) sum_A Z_A Tr[P_s <(1/2)(delta_kl r_O.r_A - r_A,k r_O,l) / r_A^3>]
//! dg_SO_kl =  (alpha^2 / 2S) sum_A Z_A Tr[(dP_alpha/dB_k - dP_beta/dB_k) (l_A / r_A^3)_l]
//! ```
//!
//! with the kinetic energy `T`, the first-order densities `dP/dB` of the
//! coupled-perturbed magnetic response, and the spin-orbit operator in its
//! one-electron form `(alpha^2 / 2) sum_A Z_A (l_A . s) / r_A^3`. `Z_A` are
//! the nuclear charges unless `soc_charges` gives effective ones; the
//! two-electron spin-orbit terms are not included. With `gauge = london`
//! (default) the basis functions carry London phases: the response density
//! is the one of the London magnetic response and the gauge-correction
//! operator holds the phase derivatives of the spin-orbit operator, so the
//! tensor does not depend on `O`. With `gauge = common-origin` all basis
//! functions share the origin `O` and the tensor depends on it in a finite
//! basis.

use super::controls::Controls;
use super::method::run_molecular_reference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::vibrational::response_config;
use super::System;
use crate::constants::{ELECTRON_G_FACTOR, FINE_STRUCTURE};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::magnetic::{magnetic_density_response, Gauge};
use crate::properties::nmr::{paramagnetic_operator, shielding_mixed_operator};
use crate::{DftError, Result};

/// `sum_ij a_ij b_ji`.
fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.rows {
        for j in 0..a.cols {
            sum += a[(i, j)] * b[(j, i)];
        }
    }
    sum
}

pub(super) fn g_tensor(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let soc_charges = controls.opt_f64_list("soc_charges")?;
    let gauge = controls.str("gauge", "london")?;
    let setup = setup(system, controls)?;
    controls.finish("g-tensor")?;
    let london = match gauge.as_str() {
        "london" => true,
        "common-origin" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown gauge `{other}` for g-tensor (london or common-origin)"
            )))
        }
    };
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    if !scf.unrestricted || scf.n_alpha <= scf.n_beta {
        return Err(DftError::InvalidInput(
            "g-tensor needs an unrestricted open-shell reference: set unrestricted = true and a \
             multiplicity above 1"
                .into(),
        ));
    }
    let natoms = setup.molecule.len();
    let charges: Vec<f64> = match soc_charges {
        Some(values) if values.len() == natoms && values.iter().all(|v| v.is_finite()) => values,
        Some(values) => {
            return Err(DftError::InvalidInput(format!(
                "soc_charges has {} values for {natoms} atoms; give one finite effective charge \
                 per atom",
                values.len()
            )))
        }
        None => setup.molecule.atoms.iter().map(|a| a.z as f64).collect(),
    };
    let origin = match explicit {
        Some(origin) => origin,
        None => {
            // Centre of nuclear charge.
            let mut sum = Vec3::zero();
            let mut total = 0.0;
            for atom in &setup.molecule.atoms {
                let z = atom.z as f64;
                sum += atom.position * z;
                total += z;
            }
            if total <= 0.0 {
                return Err(DftError::InvalidInput(
                    "g-tensor needs gauge_origin for a structure without nuclear charge".into(),
                ));
            }
            sum * (1.0 / total)
        }
    };
    let response = magnetic_density_response(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        if london {
            Gauge::Giao
        } else {
            Gauge::CommonOrigin
        },
        origin,
        &config,
    )?;
    let first = &response.value.density_hat;
    let spin = 0.5 * (scf.n_alpha - scf.n_beta) as f64;
    let n = scf.overlap.rows;
    let mut spin_density = Matrix::zeros(n, n);
    for (target, (a, b)) in spin_density.as_mut_slice().iter_mut().zip(
        scf.density[0]
            .as_slice()
            .iter()
            .zip(scf.density[1].as_slice()),
    ) {
        *target = a - b;
    }
    let alpha2 = FINE_STRUCTURE * FINE_STRUCTURE;
    let prefactor = alpha2 / (2.0 * spin);
    let kinetic = crate::integrals::kinetic_matrix(&setup.basis);
    let mass_correction = -prefactor * ELECTRON_G_FACTOR * trace_product(&spin_density, &kinetic);
    let mut gauge_correction = [[0.0_f64; 3]; 3];
    let mut spin_orbit = [[0.0_f64; 3]; 3];
    for (atom, charge) in setup.molecule.atoms.iter().zip(&charges) {
        if *charge == 0.0 {
            continue;
        }
        // With London orbitals this holds the phase derivatives of the
        // spin-orbit operator next to the gauge correction.
        let mixed = shielding_mixed_operator(&setup.basis, atom.position, origin, london);
        let orbital = paramagnetic_operator(&setup.basis, atom.position);
        for k in 0..3 {
            for l in 0..3 {
                gauge_correction[k][l] +=
                    prefactor * charge * trace_product(&spin_density, &mixed[k][l]);
                spin_orbit[k][l] += prefactor
                    * charge
                    * (trace_product(&first[0][k], &orbital[l])
                        - trace_product(&first[1][k], &orbital[l]));
            }
        }
    }
    let mut tensor = [[0.0_f64; 3]; 3];
    let mut shift = [[0.0_f64; 3]; 3];
    for k in 0..3 {
        for l in 0..3 {
            let diagonal = if k == l { 1.0 } else { 0.0 };
            shift[k][l] = diagonal * mass_correction + gauge_correction[k][l] + spin_orbit[k][l];
            tensor[k][l] = diagonal * ELECTRON_G_FACTOR + shift[k][l];
        }
    }
    let isotropic = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    let to_ppm = |t: &[[f64; 3]; 3]| -> [[f64; 3]; 3] { t.map(|row| row.map(|v| 1e6 * v)) };
    let mut report = reference_report(
        &setup,
        &reference,
        "electronic g tensor: relativistic mass correction, gauge correction and the orbital \
         Zeeman / one-electron spin-orbit cross term with a common gauge origin",
    )
    .with("gauge", gauge.clone())
    .with("gauge_origin", origin)
    .with("spin", spin)
    .with("soc_charges", charges)
    .with("g_tensor", tensor)
    .with("g_isotropic", isotropic)
    .with("g_shift_ppm", to_ppm(&shift))
    .with("mass_correction_ppm", 1e6 * mass_correction)
    .with("gauge_correction_ppm", to_ppm(&gauge_correction))
    .with("spin_orbit_ppm", to_ppm(&spin_orbit))
    .with("response_converged", response.converged)
    .with("response_residuals", response.residuals.clone())
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("gauge_origin", "bohr"),
            ("spin", "hbar"),
            ("soc_charges", "e"),
            (
                "g_tensor",
                "dimensionless, [field axis][spin axis] of mu_B B . g . S",
            ),
            ("g_isotropic", "dimensionless"),
            ("g_shift_ppm", "ppm, g - g_e"),
            ("mass_correction_ppm", "ppm"),
            ("gauge_correction_ppm", "ppm"),
            ("spin_orbit_ppm", "ppm"),
            ("response_residuals", "dimensionless"),
        ]),
    );
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        all.extend(response.warnings.iter().cloned().map(Value::Str));
        all.push(Value::Str(
            "WARNING [g-tensor] one-electron spin-orbit operator: the two-electron spin-orbit \
             terms are absent"
                .into(),
        ));
        if !london {
            all.push(Value::Str(
                "WARNING [g-tensor] common gauge origin: the tensor depends on gauge_origin in \
                 a finite basis"
                    .into(),
            ));
        }
        report.set("warnings", Value::Array(all));
    }
    Ok(report)
}
