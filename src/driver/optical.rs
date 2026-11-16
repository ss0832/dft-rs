// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Optical rotation routed through the common driver.
//!
//! For every photon energy the electric dipole - magnetic dipole
//! polarizability `G'(omega)` of the paired Casida response is evaluated in
//! the length gauge about `gauge_origin`, with
//! `beta = -Tr G' / (3 omega)` and the specific rotation
//!
//! ```text
//! [alpha] = 28800 pi^2 N_A a0^4 nu^2 beta / M
//! ```
//!
//! (`nu` in cm^-1, `M` in g/mol, `beta` in atomic units, the result in
//! deg dm^-1 (g/cm^3)^-1). The length gauge without gauge-including orbitals
//! depends on the origin in a finite basis.

use super::controls::Controls;
use super::method::run_molecular_reference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::vibrational::response_config;
use super::System;
use crate::constants::{atomic_mass_au, AMU_TO_AU_MASS, FINE_STRUCTURE, HARTREE_TO_CM1};
use crate::math::Vec3;
use crate::response::dynamic::{
    dynamic_optical_rotation, dynamic_optical_rotation_velocity, Frequency,
};
use crate::{DftError, Result};

/// `28800 pi^2 N_A a0^4` with the Bohr radius in cm.
const SPECIFIC_ROTATION_FACTOR: f64 = 1.34229e-4;

pub(super) fn optical_activity(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let frequencies = controls.opt_f64_list("frequencies")?.ok_or_else(|| {
        DftError::InvalidInput(
            "optical-activity needs frequencies: the photon energies of the rotation".into(),
        )
    })?;
    let damping = controls.f64("frequency_damping", 0.0)?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let gauge = controls.str("gauge", "velocity")?;
    let setup = setup(system, controls)?;
    controls.finish("optical-activity")?;
    // The modified velocity gauge gives an origin-independent rotation.
    let velocity = match gauge.as_str() {
        "velocity" => true,
        "length" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown gauge `{other}` for optical-activity (velocity or length)"
            )))
        }
    };
    if frequencies.is_empty()
        || frequencies.iter().any(|w| !(w.is_finite() && *w > 0.0))
        || !(damping.is_finite() && damping >= 0.0)
    {
        return Err(DftError::InvalidInput(
            "frequencies are positive photon energies and frequency_damping is not negative".into(),
        ));
    }
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let mut charge_sum = Vec3::zero();
    let mut charge = 0.0;
    let mut molar_mass = 0.0;
    for atom in &setup.molecule.atoms {
        let z = atom.z as f64;
        charge_sum += atom.position * z;
        charge += z;
        molar_mass += atomic_mass_au(atom.z) / AMU_TO_AU_MASS;
    }
    let origin = match explicit {
        Some(origin) => origin,
        None if charge > 0.0 => charge_sum * (1.0 / charge),
        None => {
            return Err(DftError::InvalidInput(
                "optical-activity needs gauge_origin for a structure without nuclear charge".into(),
            ))
        }
    };
    let mut tensors = Vec::with_capacity(frequencies.len());
    let mut tensors_imaginary = Vec::with_capacity(frequencies.len());
    let mut betas = Vec::with_capacity(frequencies.len());
    let mut betas_imaginary = Vec::with_capacity(frequencies.len());
    let mut rotations = Vec::with_capacity(frequencies.len());
    let mut residuals = Vec::with_capacity(frequencies.len());
    for &omega in &frequencies {
        let result = if velocity {
            dynamic_optical_rotation_velocity(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                Frequency::Retarded { omega, damping },
                origin,
                &config,
            )?
        } else {
            dynamic_optical_rotation(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                Frequency::Retarded { omega, damping },
                origin,
                &config,
            )?
        };
        let g = &result.value.tensor;
        tensors.push(g.map(|row| row.map(|value| value.re)));
        tensors_imaginary.push(g.map(|row| row.map(|value| value.im)));
        betas.push(result.value.beta.re);
        betas_imaginary.push(result.value.beta.im);
        let wavenumber = omega * HARTREE_TO_CM1;
        rotations.push(if molar_mass > 0.0 {
            Some(
                SPECIFIC_ROTATION_FACTOR * wavenumber * wavenumber * result.value.beta.re
                    / molar_mass,
            )
        } else {
            None
        });
        residuals.push(result.residuals.iter().copied().fold(0.0_f64, f64::max));
    }
    let mut report = reference_report(
        &setup,
        &reference,
        if velocity {
            "optical rotation from the paired Casida response in the modified velocity gauge with the zero-frequency subtraction"
        } else {
            "optical rotation from the electric dipole - magnetic dipole polarizability of the paired Casida response in the length gauge"
        },
    )
    .with("gauge", gauge.clone())
    .with("gauge_origin", origin)
    .with("frequencies", frequencies.clone())
    .with("frequency_damping", damping)
    .with("rotation_tensors", tensors)
    .with("rotation_tensors_imaginary", tensors_imaginary)
    .with("beta", betas)
    .with("beta_imaginary", betas_imaginary)
    .with("molar_mass", molar_mass)
    .with("specific_rotation", rotations)
    .with("response_residuals", residuals)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("gauge_origin", "bohr"),
            ("frequencies", "hartree"),
            ("frequency_damping", "hartree"),
            (
                "rotation_tensors",
                "e^2*bohr^3/hbar, [frequency][electric-field axis][magnetic-dipole axis], real part",
            ),
            ("rotation_tensors_imaginary", "e^2*bohr^3/hbar"),
            ("beta", "e^2*bohr^3/hartree, -Tr G' / (3 omega), real part"),
            ("beta_imaginary", "e^2*bohr^3/hartree"),
            ("molar_mass", "g/mol"),
            ("specific_rotation", "deg/(dm*g/cm^3)"),
            ("response_residuals", "dimensionless"),
        ]),
    );
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        if !velocity {
            all.push(Value::Str(
                "WARNING [optical-activity] length gauge with a common origin: the rotation \
                 depends on gauge_origin in a finite basis"
                    .into(),
            ));
        }
        report.set("warnings", Value::Array(all));
    }
    Ok(report)
}

/// Invariants of Raman optical activity from the derivatives of the three
/// optical tensors along one normal mode: `alpha^2`, `beta(alpha)^2`,
/// `alpha G'`, `beta(G')^2` and `beta(A)^2`.
fn roa_invariants(
    alpha: &[[f64; 3]; 3],
    rotation: &[[f64; 3]; 3],
    quadrupole: &[[[f64; 3]; 3]; 3],
    omega: f64,
) -> [f64; 5] {
    let s = |i: usize, j: usize| 0.5 * (alpha[i][j] + alpha[j][i]);
    let trace_alpha = s(0, 0) + s(1, 1) + s(2, 2);
    let trace_rotation = rotation[0][0] + rotation[1][1] + rotation[2][2];
    let mut alpha_alpha = 0.0;
    let mut alpha_rotation = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            alpha_alpha += s(i, j) * s(i, j);
            alpha_rotation += s(i, j) * rotation[i][j];
        }
    }
    // beta(A)^2 = (omega / 2) alpha_ab eps_acd A_c,db.
    let mut mixed = 0.0;
    for (a, c, d, sign) in [
        (0, 1, 2, 1.0),
        (1, 2, 0, 1.0),
        (2, 0, 1, 1.0),
        (0, 2, 1, -1.0),
        (2, 1, 0, -1.0),
        (1, 0, 2, -1.0),
    ] {
        for b in 0..3 {
            mixed += sign * s(a, b) * quadrupole[c][d][b];
        }
    }
    [
        trace_alpha * trace_alpha / 9.0,
        0.5 * (3.0 * alpha_alpha - trace_alpha * trace_alpha),
        trace_alpha * trace_rotation / 9.0,
        0.5 * (3.0 * alpha_rotation - trace_alpha * trace_rotation),
        0.5 * omega * mixed,
    ]
}

/// Raman optical activity routed through the common driver.
///
/// For every photon energy and every normal mode the derivatives of the
/// electric dipole polarizability, of the optical rotation tensor `G'` and of
/// the dipole-quadrupole polarizability `A` along the mode are the first
/// Taylor coefficients of the paired response along that nuclear path. The
/// dimensionless circular intensity differences
/// `(I_R - I_L) / (I_R + I_L)` of the far-from-resonance theory follow from
/// the invariants, with `1 / c` the fine-structure constant in atomic units:
///
/// ```text
/// forward        4 [45 aG' + b(G')^2 - b(A)^2] / (c [45 a^2 + 7 b(a)^2])
/// backward       24 [b(G')^2 + b(A)^2 / 3] / (c [45 a^2 + 7 b(a)^2])
/// right angle, polarized    2 [45 aG' + 7 b(G')^2 + b(A)^2] / (c [45 a^2 + 7 b(a)^2])
/// right angle, depolarized  12 [b(G')^2 - b(A)^2 / 3] / (6 c b(a)^2)
/// ```
///
/// `gauge` selects the velocity gauge (default), whose invariants do not
/// depend on `gauge_origin`, or the length gauge about that origin, which
/// depends on it in a finite basis. The origin stays fixed in space.
pub(super) fn roa(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let frequencies = controls.opt_f64_list("frequencies")?.ok_or_else(|| {
        DftError::InvalidInput(
            "roa needs frequencies: the photon energies of the incident light".into(),
        )
    })?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let gauge = controls.str("gauge", "velocity")?;
    let data = super::vibrational::compute_hessian(system, controls)?;
    controls.finish("roa")?;
    // The velocity gauge gives invariants that do not depend on the origin.
    let velocity = match gauge.as_str() {
        "velocity" => true,
        "length" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown gauge `{other}` for roa (velocity or length)"
            )))
        }
    };
    if frequencies.is_empty() || frequencies.iter().any(|w| !(w.is_finite() && *w > 0.0)) {
        return Err(DftError::InvalidInput(
            "frequencies are positive photon energies".into(),
        ));
    }
    let molecule = &data.setup.molecule;
    let modes = crate::properties::vibrational::vibrational_analysis(
        molecule,
        &data.hessian,
        Some(&data.analytic.dipole_derivatives),
    )?;
    let mut charge_sum = Vec3::zero();
    let mut charge = 0.0;
    for atom in &molecule.atoms {
        let z = atom.z as f64;
        charge_sum += atom.position * z;
        charge += z;
    }
    let origin = match explicit {
        Some(origin) => origin,
        None if charge > 0.0 => charge_sum * (1.0 / charge),
        None => {
            return Err(DftError::InvalidInput(
                "roa needs gauge_origin for a structure without nuclear charge".into(),
            ))
        }
    };
    let natoms = molecule.len();
    let sqrt_m_amu: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| (atomic_mass_au(a.z) / AMU_TO_AU_MASS).sqrt())
        .collect();
    let patterns: Vec<Vec<[f64; 3]>> = (0..modes.modes.cols)
        .map(|k| {
            (0..natoms)
                .map(|a| std::array::from_fn(|ax| modes.modes[(3 * a + ax, k)] / sqrt_m_amu[a]))
                .collect()
        })
        .collect();
    let ratio = |numerator: f64, denominator: f64| -> Option<f64> {
        if denominator > 0.0 {
            Some(FINE_STRUCTURE * numerator / denominator)
        } else {
            None
        }
    };
    let mut alpha_derivatives = Vec::with_capacity(frequencies.len());
    let mut rotation_derivatives = Vec::with_capacity(frequencies.len());
    let mut quadrupole_derivatives = Vec::with_capacity(frequencies.len());
    let mut invariants: [Vec<Vec<f64>>; 5] = Default::default();
    let mut differences: [Vec<Vec<Option<f64>>>; 4] = Default::default();
    let mut residual = 0.0_f64;
    let mut converged = true;
    let mut warnings: Vec<String> = Vec::new();
    for &omega in &frequencies {
        let mut d_alpha = Vec::with_capacity(patterns.len());
        let mut d_rotation = Vec::with_capacity(patterns.len());
        let mut d_quadrupole = Vec::with_capacity(patterns.len());
        let mut values: [Vec<f64>; 5] = Default::default();
        let mut cid: [Vec<Option<f64>>; 4] = Default::default();
        for pattern in &patterns {
            let path = crate::response::nuclear::optical_tensor_path(
                molecule,
                &data.setup.basis,
                &data.setup.functional,
                data.setup.grid.as_ref(),
                &data.reference.scf,
                &data.setup.options,
                pattern,
                1,
                omega,
                origin,
                velocity,
                &config,
            )?;
            converged &= path.converged;
            residual = path.residuals.iter().copied().fold(residual, f64::max);
            for warning in &path.warnings {
                if !warnings.contains(warning) {
                    warnings.push(warning.clone());
                }
            }
            let [a2, b2, ag, bg, ba] = roa_invariants(
                &path.polarizability[1],
                &path.rotation[1],
                &path.quadrupole[1],
                omega,
            );
            for (list, value) in values.iter_mut().zip([a2, b2, ag, bg, ba]) {
                list.push(value);
            }
            let sum = 45.0 * a2 + 7.0 * b2;
            cid[0].push(ratio(4.0 * (45.0 * ag + bg - ba), sum));
            cid[1].push(ratio(24.0 * (bg + ba / 3.0), sum));
            cid[2].push(ratio(2.0 * (45.0 * ag + 7.0 * bg + ba), sum));
            cid[3].push(ratio(12.0 * (bg - ba / 3.0), 6.0 * b2));
            d_alpha.push(path.polarizability[1]);
            d_rotation.push(path.rotation[1]);
            d_quadrupole.push(path.quadrupole[1]);
        }
        alpha_derivatives.push(d_alpha);
        rotation_derivatives.push(d_rotation);
        quadrupole_derivatives.push(d_quadrupole);
        for (all, list) in invariants.iter_mut().zip(values) {
            all.push(list);
        }
        for (all, list) in differences.iter_mut().zip(cid) {
            all.push(list);
        }
    }
    let [isotropic, anisotropic, isotropic_rotation, anisotropic_rotation, anisotropic_quadrupole] =
        invariants;
    let [forward, backward, polarized, depolarized] = differences;
    let mut report = reference_report(
        &data.setup,
        &data.reference,
        "Raman optical activity from the normal-mode derivatives of the optical tensors of the \
         paired response",
    )
    .with("gauge", gauge.clone())
    .with("gauge_origin", origin)
    .with("frequencies", frequencies.clone())
    .with("frequencies_cm1", modes.frequencies_cm.clone())
    .with("n_imaginary", modes.n_imaginary)
    .with("polarizability_derivatives", alpha_derivatives)
    .with("rotation_derivatives", rotation_derivatives)
    .with("quadrupole_derivatives", quadrupole_derivatives)
    .with("isotropic_invariants", isotropic)
    .with("anisotropic_invariants", anisotropic)
    .with("isotropic_rotation_invariants", isotropic_rotation)
    .with("anisotropic_rotation_invariants", anisotropic_rotation)
    .with("anisotropic_quadrupole_invariants", anisotropic_quadrupole)
    .with("cid_forward", forward)
    .with("cid_backward", backward)
    .with("cid_right_angle_polarized", polarized)
    .with("cid_right_angle_depolarized", depolarized)
    .with("response_residual", residual)
    .with("response_converged", converged)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("gauge_origin", "bohr"),
            ("frequencies", "hartree"),
            ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
            (
                "polarizability_derivatives",
                "atomic units of polarizability per bohr*sqrt(amu), \
                 [frequency][mode][observable axis][field axis]",
            ),
            (
                "rotation_derivatives",
                "atomic units of G' per bohr*sqrt(amu), \
                 [frequency][mode][field axis][magnetic-dipole axis]",
            ),
            (
                "quadrupole_derivatives",
                "atomic units of A per bohr*sqrt(amu), \
                 [frequency][mode][field axis][quadrupole component]",
            ),
            (
                "isotropic_invariants",
                "squared atomic units per bohr^2*amu, alpha^2",
            ),
            (
                "anisotropic_invariants",
                "squared atomic units per bohr^2*amu, beta(alpha)^2",
            ),
            (
                "isotropic_rotation_invariants",
                "squared atomic units per bohr^2*amu, alpha G'",
            ),
            (
                "anisotropic_rotation_invariants",
                "squared atomic units per bohr^2*amu, beta(G')^2",
            ),
            (
                "anisotropic_quadrupole_invariants",
                "squared atomic units per bohr^2*amu, beta(A)^2",
            ),
            ("cid_forward", "dimensionless"),
            ("cid_backward", "dimensionless"),
            ("cid_right_angle_polarized", "dimensionless"),
            ("cid_right_angle_depolarized", "dimensionless"),
            ("response_residual", "dimensionless"),
        ]),
    );
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        if !velocity {
            all.push(Value::Str(
                "WARNING [roa] length gauge with a common origin: the rotation and quadrupole \
                 tensors depend on gauge_origin in a finite basis"
                    .into(),
            ));
        }
        all.extend(warnings.into_iter().map(Value::Str));
        report.set("warnings", Value::Array(all));
    }
    Ok(report)
}
