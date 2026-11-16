// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular magnetizability with a common gauge origin, routed through the
//! common driver.
//!
//! With `h(B) = h + (1/2) B . L_O + (1/8) (B x r_O)^2` in atomic units, the
//! magnetizability `xi_ab = -d2E/(dB_a dB_b)` is
//!
//! ```text
//! xi_ab = -(1/4) Tr[P (r_O^2 delta_ab - r_a r_b)] - (1/2) Tr[dP/dB_a L_b]
//! ```
//!
//! with `dP/dB` from the coupled-perturbed magnetic response. All basis
//! functions share the gauge origin `O`, so the two parts and, in a finite
//! basis, their sum depend on it; the default origin is the centre of
//! nuclear charge.

use super::controls::Controls;
use super::method::{run_molecular_reference, MolecularReference, MolecularSetup};
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::vibrational::response_config;
use super::System;
use crate::constants::{BOLTZMANN_J_PER_K, HARTREE_TO_JOULE};
use crate::integrals::one_electron::moment_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::magnetic::{magnetic_density_response, Gauge};
use crate::properties::nmr::angular_momentum_operator;
use crate::system::Molecule;
use crate::{DftError, Result};

pub(super) mod spinor;

/// One atomic unit of magnetizability, `e^2 a0^2 / m_e`, as a molar
/// susceptibility in 1e-6 cm^3/mol (CODATA 2018: 7.8910366e-29 J/T^2 times
/// the Avogadro constant and mu0 / 4 pi).
pub(super) const ATOMIC_UNIT_IN_CGS_PPM: f64 = 4.7520933;

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

fn nuclear_charge_origin(molecule: &Molecule) -> Result<Vec3> {
    let mut sum = Vec3::zero();
    let mut charge = 0.0;
    for atom in &molecule.atoms {
        let z = atom.z as f64;
        sum += atom.position * z;
        charge += z;
    }
    if charge <= 0.0 {
        return Err(DftError::InvalidInput(
            "magnetizability needs gauge_origin for a structure without nuclear charge".into(),
        ));
    }
    Ok(sum * (1.0 / charge))
}

/// Fixed-density curvature of the orbital diamagnetic Hamiltonian.
fn diamagnetic_tensor(
    basis: &crate::basis::MolBasis,
    origin: Vec3,
    density: &Matrix,
) -> [[f64; 3]; 3] {
    let mut moments = [[0.0_f64; 3]; 3];
    for (power, matrix) in moment_matrices(basis, origin, 2) {
        let value = trace_product(density, &matrix);
        let powers = [power.0, power.1, power.2];
        let axes: Vec<usize> = (0..3)
            .flat_map(|axis| std::iter::repeat_n(axis, powers[axis] as usize))
            .collect();
        moments[axes[0]][axes[1]] = value;
        moments[axes[1]][axes[0]] = value;
    }
    let r2 = moments[0][0] + moments[1][1] + moments[2][2];
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let delta = if a == b { r2 } else { 0.0 };
            -0.25 * (delta - moments[a][b])
        })
    })
}

pub(super) fn magnetizability(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let gauge = controls.str("gauge", "common-origin")?;
    let higher_order = controls.opt_usize("response_order")?;
    let magnetic_directions = controls
        .opt_rows("magnetic_directions")?
        .map(|rows| {
            if rows.is_empty() || rows.len() > 4 || rows.iter().any(|r| r.len() != 3) {
                return Err(DftError::InvalidInput(
                    "magnetic_directions must contain one through four Cartesian field directions"
                        .into(),
                ));
            }
            Ok(rows.iter().map(|r| [r[0], r[1], r[2]]).collect::<Vec<_>>())
        })
        .transpose()?;
    let magnetic_direction = controls
        .opt_vec3("magnetic_direction")?
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let electric_directions = controls
        .opt_rows("electric_directions")?
        .map(|rows| {
            if magnetic_directions.as_ref().map(Vec::len) != Some(rows.len())
                || rows.iter().any(|r| r.len() != 3)
            {
                return Err(DftError::InvalidInput(
                    "electric_directions must match the magnetic_directions list".into(),
                ));
            }
            Ok(rows.iter().map(|r| [r[0], r[1], r[2]]).collect::<Vec<_>>())
        })
        .transpose()?;
    if higher_order.is_some_and(|n| n == 0 || n > 4) {
        return Err(DftError::InvalidInput(
            "magnetizability response_order must be between one and four".into(),
        ));
    }
    let prepared = if super::double_hybrid::requested(controls)? {
        if higher_order.is_some() || magnetic_directions.is_some() || electric_directions.is_some()
        {
            return Err(DftError::Unsupported {
                feature: "correlated contracted high-order magnetizability".into(),
                missing: "public contraction of the correlated higher-order field paths".into(),
            });
        }
        Some(super::double_hybrid::prepare(
            system,
            controls,
            "magnetizability",
        )?)
    } else {
        None
    };
    let ordinary = if prepared.is_none() {
        let setup = setup(system, controls)?;
        controls.finish("magnetizability")?;
        let reference = run_molecular_reference(&setup)?;
        Some((setup, reference))
    } else {
        None
    };
    let (setup, reference) = match (&prepared, &ordinary) {
        (Some(data), _) => (&data.setup, &data.reference),
        (_, Some((setup, reference))) => (setup, reference),
        _ => unreachable!("one magnetic reference was prepared"),
    };
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let origin = match explicit {
        Some(origin) => origin,
        None => nuclear_charge_origin(&setup.molecule)?,
    };
    if let Some(data) = prepared {
        return correlated_report(data, origin, &gauge, &config);
    }
    if gauge == "london" {
        // Second-order energy of the field path with London orbitals: the
        // tensor does not depend on the gauge origin.
        let (tensor, residual) = crate::response::field_path::field_magnetizability(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            &setup.options,
            origin,
            true,
            &config,
        )?;
        let isotropic = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
        let report = reference_report(
            setup,
            reference,
            "magnetizability from the second-order energy of the field path response with \
             London orbitals",
        )
        .with("gauge", "london")
        .with("magnetizability", tensor)
        .with("isotropic", isotropic)
        .with("isotropic_cgs_ppm", isotropic * ATOMIC_UNIT_IN_CGS_PPM)
        .with("response_residual", residual)
        .with("provenance", "analytic")
        .with(
            "units",
            units(&[
                ("energy", "hartree"),
                (
                    "magnetizability",
                    "e^2*bohr^2/m_e, xi_ab = -d2E/(dB_a dB_b)",
                ),
                ("isotropic", "e^2*bohr^2/m_e"),
                ("isotropic_cgs_ppm", "1e-6 cm^3/mol"),
                ("response_residual", "dimensionless"),
            ]),
        );
        return higher_magnetic_report(
            report,
            setup,
            reference,
            origin,
            true,
            higher_order,
            magnetic_direction,
            magnetic_directions.as_deref(),
            electric_directions.as_deref(),
            &config,
        );
    } else if gauge != "common-origin" {
        return Err(DftError::InvalidInput(format!(
            "unknown gauge `{gauge}` for magnetizability (common-origin or london)"
        )));
    }
    let response = magnetic_density_response(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        Gauge::CommonOrigin,
        origin,
        &config,
    )?;
    // dP/dB of both spins, with the factor i taken out.
    let mut first = response.value.density_hat[0].clone();
    for (target, source) in first.iter_mut().zip(&response.value.density_hat[1]) {
        for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
            *t += *s;
        }
    }
    let p_total = scf.p_total();
    let angular = angular_momentum_operator(&setup.basis, origin);
    let diamagnetic = diamagnetic_tensor(&setup.basis, origin, &p_total);
    let mut paramagnetic = [[0.0_f64; 3]; 3];
    let mut tensor = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            paramagnetic[a][b] = -0.5 * trace_product(&first[a], &angular[b]);
            tensor[a][b] = diamagnetic[a][b] + paramagnetic[a][b];
        }
    }
    let isotropic = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    let asymmetry = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| (tensor[a][b] - tensor[b][a]).abs())
        .fold(0.0_f64, f64::max);
    let mut report = reference_report(
        setup,
        reference,
        "magnetizability from the coupled-perturbed magnetic response with a common gauge origin",
    )
    .with("gauge", "common-origin")
    .with("gauge_origin", origin)
    .with("magnetizability", tensor)
    .with("diamagnetic", diamagnetic)
    .with("paramagnetic", paramagnetic)
    .with("isotropic", isotropic)
    .with("isotropic_cgs_ppm", isotropic * ATOMIC_UNIT_IN_CGS_PPM)
    .with("raw_asymmetry", asymmetry)
    .with("response_converged", response.converged)
    .with("response_residuals", response.residuals.clone())
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("gauge_origin", "bohr"),
            (
                "magnetizability",
                "e^2*bohr^2/m_e, xi_ab = -d2E/(dB_a dB_b)",
            ),
            ("diamagnetic", "e^2*bohr^2/m_e"),
            ("paramagnetic", "e^2*bohr^2/m_e"),
            ("isotropic", "e^2*bohr^2/m_e"),
            ("isotropic_cgs_ppm", "1e-6 cm^3/mol"),
            ("raw_asymmetry", "e^2*bohr^2/m_e"),
            ("response_residuals", "dimensionless"),
        ]),
    );
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        all.extend(response.warnings.iter().cloned().map(Value::Str));
        all.push(Value::Str(
            "WARNING [magnetizability] common gauge origin: the tensor depends on gauge_origin in \
             a finite basis"
                .into(),
        ));
        report.set("warnings", Value::Array(all));
    }
    higher_magnetic_report(
        report,
        setup,
        reference,
        origin,
        false,
        higher_order,
        magnetic_direction,
        magnetic_directions.as_deref(),
        electric_directions.as_deref(),
        &config,
    )
}

/// Contracted orbital magnetic energy derivatives use the existing metric
/// and London-integral path, shared by all supported functional providers.
fn higher_magnetic_report(
    mut report: Value,
    setup: &MolecularSetup,
    reference: &MolecularReference,
    origin: Vec3,
    london: bool,
    order: Option<usize>,
    direction: Vec3,
    mixed_directions: Option<&[[f64; 3]]>,
    electric_directions: Option<&[[f64; 3]]>,
    config: &crate::response::ResponseSolverConfig,
) -> Result<Value> {
    if let Some(directions) = mixed_directions {
        let electric =
            electric_directions.map_or_else(|| vec![[0.0; 3]; directions.len()], |x| x.to_vec());
        let mixed = crate::response::magnetic_mixed::mixed_electromagnetic_energy(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            origin,
            london,
            &electric,
            directions,
            config,
        )?;
        report.set("magnetic_directions", directions.to_vec());
        if electric_directions.is_some() {
            report.set("electric_directions", electric);
        }
        report.set("magnetic_mixed_order", directions.len());
        report.set("magnetic_mixed_derivative", mixed.derivative);
        report.set("magnetic_mixed_roundoff", mixed.accumulation_roundoff);
        report.set("magnetic_mixed_evaluations", mixed.directional_evaluations);
        report.set("magnetic_mixed_response_residual", mixed.response_residual);
        report.set("magnetic_mixed_projector_defect", mixed.projector_defect);
        report.set(
            "magnetic_mixed_stationarity_defect",
            mixed.stationarity_defect,
        );
        report.set("magnetic_mixed_reference_defect", mixed.reference_defect);
        report.set("magnetic_mixed_convention","actual joint mixed derivative d^n E/(dt1...dtn), E=sum_j tj*electric_directions[j] (zero when omitted), B=sum_j tj*magnetic_directions[j]; dimensionless parameters; orbital collinear zero-field reference, spin Zeeman and current-dependent correlation excluded");
        let mut labels = report.get("units").unwrap().clone();
        labels.set("magnetic_directions", "atomic units of magnetic field");
        if electric_directions.is_some() {
            labels.set("electric_directions", "hartree/e/bohr");
        }
        for key in [
            "magnetic_mixed_derivative",
            "magnetic_mixed_roundoff",
            "magnetic_mixed_stationarity_defect",
            "magnetic_mixed_reference_defect",
        ] {
            labels.set(key, "hartree along dimensionless field parameters");
        }
        for key in [
            "magnetic_mixed_response_residual",
            "magnetic_mixed_projector_defect",
        ] {
            labels.set(key, "dimensionless");
        }
        report.set("units", labels);
    }
    let Some(order) = order else {
        return Ok(report);
    };
    let value = crate::response::field_path::field_path_taylor(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        [0.0; 3],
        direction.to_array(),
        origin,
        london,
        order,
        config,
    )?;
    let mut factorial = 1.0;
    let derivatives: Vec<_> = value
        .energy
        .iter()
        .enumerate()
        .map(|(n, v)| {
            if n > 0 {
                factorial *= n as f64;
            }
            factorial * v
        })
        .collect();
    report.set("magnetic_direction", direction);
    report.set("magnetic_response_order", order);
    report.set("magnetic_energy_coefficients", value.energy);
    report.set("magnetic_energy_derivatives", derivatives);
    report.set("magnetic_path_residuals", value.residuals);
    report.set("magnetic_projector_defects", value.projector_defects);
    report.set("magnetic_stationarity_defects", value.stationarity_defects);
    report.set("magnetic_path_reference_defect", value.reference_defect);
    report.set("magnetic_path_model","static orbital energy at B(t)=t*magnetic_direction; zero-field integer-occupation collinear reference; spin Zeeman and current-dependent correlation are excluded");
    let mut labels = report
        .get("units")
        .cloned()
        .unwrap_or_else(|| Value::Object(Vec::new()));
    for (key, label) in [
        ("magnetic_direction", "atomic units of magnetic field"),
        (
            "magnetic_energy_coefficients",
            "hartree along dimensionless paths",
        ),
        (
            "magnetic_energy_derivatives",
            "hartree along dimensionless paths",
        ),
        ("magnetic_path_residuals", "dimensionless"),
        ("magnetic_projector_defects", "dimensionless"),
        (
            "magnetic_stationarity_defects",
            "hartree along dimensionless paths",
        ),
        ("magnetic_path_reference_defect", "hartree"),
    ] {
        labels.set(key, label);
    }
    report.set("units", labels);
    Ok(report)
}

/// Centre of nuclear charge, the default gauge origin.
fn charge_centre(molecule: &crate::system::Molecule, feature: &str) -> Result<Vec3> {
    let mut sum = Vec3::zero();
    let mut charge = 0.0;
    for atom in &molecule.atoms {
        let z = atom.z as f64;
        sum += atom.position * z;
        charge += z;
    }
    if charge <= 0.0 {
        return Err(DftError::InvalidInput(format!(
            "{feature} needs gauge_origin for a structure without nuclear charge"
        )));
    }
    Ok(sum * (1.0 / charge))
}

/// Static hypermagnetizability `eta_ab,cd = d2 alpha_ab / (dB_c dB_d)` and,
/// for `cotton-mouton`, the Cotton-Mouton constant per molecule
///
/// ```text
/// C = (2 pi / 27) [ D_eta + 2 (alpha_ab xi_ab - 3 alpha xi) / (15 k T) ],
/// D_eta = (eta_ab,ab - eta_aa,bb / 3) / 5,
/// ```
///
/// with the static polarizability `alpha`, the magnetizability `xi` and
/// their isotropic parts. All three tensors come from the density-projector
/// response to combined electric and magnetic fields with a common gauge
/// origin; `eta` is the zero-frequency limit.
pub(super) fn hypermagnetizability(
    id: &str,
    system: &System,
    controls: &Controls,
) -> Result<Value> {
    let config = response_config(controls)?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let gauge = controls.str("gauge", "london")?;
    let frequency = controls.f64("frequency", 0.0)?;
    let temperature = if id == "cotton-mouton" {
        Some(controls.f64("temperature", 298.15)?)
    } else {
        None
    };
    let prepared = if super::double_hybrid::requested(controls)? {
        if frequency != 0.0 {
            return Err(DftError::Unsupported {
                feature: "frequency-dependent correlated hypermagnetizability".into(),
                missing:
                    "mixed magnetic derivatives of the causal parent/target/PT2 susceptibility"
                        .into(),
            });
        }
        Some(super::double_hybrid::prepare(system, controls, id)?)
    } else {
        None
    };
    let ordinary = if prepared.is_none() {
        let setup = setup(system, controls)?;
        controls.finish(id)?;
        let reference = run_molecular_reference(&setup)?;
        Some((setup, reference))
    } else {
        None
    };
    let (setup, reference) = match (&prepared, &ordinary) {
        (Some(data), _) => (&data.setup, &data.reference),
        (_, Some((setup, reference))) => (setup, reference),
        _ => unreachable!("one electromagnetic reference was prepared"),
    };
    let london = match gauge.as_str() {
        "london" => true,
        "common-origin" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown gauge `{other}` for {id} (london or common-origin)"
            )))
        }
    };
    if temperature.is_some_and(|t| !(t.is_finite() && t > 0.0)) {
        return Err(DftError::InvalidInput(
            "temperature must be positive".into(),
        ));
    }
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let origin = match explicit {
        Some(origin) => origin,
        None => charge_centre(&setup.molecule, id)?,
    };
    // London orbitals make the tensors independent of the gauge origin; the
    // common-origin recursion works on the production integral engines.
    if !(frequency.is_finite() && frequency >= 0.0) {
        return Err(DftError::InvalidInput(
            "frequency is a non-negative photon energy".into(),
        ));
    }
    // At a photon energy the polarizability and its field derivatives come
    // from the paired response along magnetic paths.
    let correlated = prepared
        .as_ref()
        .map(|data| {
            data.check_path_memory(4)?;
            crate::properties::correlated_magnetic::correlated_hypermagnetizability_for_model(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                data.field_target()?,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                data.coefficients,
                if london {
                    Gauge::Giao
                } else {
                    Gauge::CommonOrigin
                },
                origin,
                data.pt2_memory_mb,
                &config,
            )
        })
        .transpose()?;
    let mut faraday = None;
    let result = if let Some(value) = &correlated {
        crate::response::magnetic_taylor::Hypermagnetizability {
            origin: origin.to_array(),
            polarizability: value.polarizability[0],
            magnetizability: value.magnetic.total,
            tensor: value.tensor[0],
            accumulation_roundoff: value.accumulation_roundoff[0],
            directional_evaluations: value.directional_evaluations,
            residual: value
                .magnetic
                .response_residuals
                .iter()
                .copied()
                .fold(0.0, f64::max),
        }
    } else if frequency > 0.0 {
        let dynamic = crate::response::field_path::dynamic_hypermagnetizability(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            &setup.options,
            origin,
            london,
            frequency,
            &config,
        )?;
        let (static_xi, xi_residual) = crate::response::field_path::field_magnetizability(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            &setup.options,
            origin,
            london,
            &config,
        )?;
        faraday = Some(dynamic.faraday);
        crate::response::magnetic_taylor::Hypermagnetizability {
            origin: origin.to_array(),
            polarizability: dynamic.polarizability,
            magnetizability: static_xi,
            tensor: dynamic.tensor,
            accumulation_roundoff: 0.0,
            directional_evaluations: dynamic.directional_evaluations,
            residual: dynamic.residual.max(xi_residual),
        }
    } else if london {
        crate::response::field_path::field_hypermagnetizability(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            &setup.options,
            origin,
            true,
            &config,
        )?
    } else {
        crate::response::magnetic_taylor::hypermagnetizability(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            &setup.options,
            origin,
            &config,
        )?
    };
    let eta = &result.tensor;
    let alpha = &result.polarizability;
    let xi = &result.magnetizability;
    let mut anisotropy = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            anisotropy += eta[a][b][a][b] - eta[a][a][b][b] / 3.0;
        }
    }
    anisotropy /= 5.0;
    let mut report = reference_report(
        setup,
        reference,
        if correlated.is_some() {
            "static orbital hypermagnetizability from the full target plus noncanonical PT2 energy; complete mixed electric/magnetic derivatives on an integer collinear parent"
        } else {
            "hypermagnetizability from the mixed electric/magnetic response of the stationary parent"
        },
    )
    .with("gauge", gauge.clone())
    .with("gauge_origin", origin)
    .with("polarizability", *alpha)
    .with("magnetizability", *xi)
    .with("hypermagnetizability", *eta)
    .with("hypermagnetizability_anisotropy", anisotropy)
    .with("accumulation_roundoff", result.accumulation_roundoff)
    .with("directional_evaluations", result.directional_evaluations)
    .with("response_residual", result.residual)
    .with("provenance", "analytic");
    let mut unit_pairs = vec![
        ("energy", "hartree"),
        ("gauge_origin", "bohr"),
        (
            "polarizability",
            "e^2*bohr^2/hartree, alpha_ab = d mu_a / d E_b",
        ),
        (
            "magnetizability",
            "e^2*bohr^2/m_e, xi_ab = -d2E/(dB_a dB_b)",
        ),
        (
            "hypermagnetizability",
            "atomic units, eta[a][b][c][d] = d2 alpha_ab / (dB_c dB_d)",
        ),
        (
            "hypermagnetizability_anisotropy",
            "atomic units, (eta_ab,ab - eta_aa,bb / 3) / 5",
        ),
        ("accumulation_roundoff", "atomic units"),
        ("response_residual", "dimensionless"),
    ];
    if let Some(temperature) = temperature {
        let mut product = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                product += 0.25 * (alpha[a][b] + alpha[b][a]) * (xi[a][b] + xi[b][a]);
            }
        }
        let trace_alpha = alpha[0][0] + alpha[1][1] + alpha[2][2];
        let trace_xi = xi[0][0] + xi[1][1] + xi[2][2];
        let thermal = temperature * BOLTZMANN_J_PER_K / HARTREE_TO_JOULE;
        let orientational = 2.0 * (product - trace_alpha * trace_xi / 3.0) / (15.0 * thermal);
        report.set("temperature", temperature);
        report.set("orientational_term", orientational);
        report.set(
            "cotton_mouton_constant",
            2.0 * std::f64::consts::PI / 27.0 * (anisotropy + orientational),
        );
        unit_pairs.push(("temperature", "K"));
        unit_pairs.push((
            "orientational_term",
            "atomic units, 2 (alpha_ab xi_ab - 3 alpha xi) / (15 k T)",
        ));
        unit_pairs.push((
            "cotton_mouton_constant",
            "atomic units per molecule, (2 pi / 27) (anisotropy + orientational term)",
        ));
    }
    report.set("frequency", frequency);
    unit_pairs.push(("frequency", "hartree"));
    if let Some(faraday) = faraday {
        let mut invariant = 0.0;
        for (a, b, c, sign) in [
            (0, 1, 2, 1.0),
            (1, 2, 0, 1.0),
            (2, 0, 1, 1.0),
            (0, 2, 1, -1.0),
            (2, 1, 0, -1.0),
            (1, 0, 2, -1.0),
        ] {
            invariant += sign * faraday[a][b][c] / 6.0;
        }
        report.set("faraday_tensor", faraday);
        report.set("faraday_invariant", invariant);
        unit_pairs.push(("faraday_tensor", "atomic units, Im d alpha_ab / dB_c"));
        unit_pairs.push((
            "faraday_invariant",
            "atomic units, eps_abc Im d alpha_ab / dB_c / 6",
        ));
    }
    report.set("units", units(&unit_pairs));
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        if !london {
            all.push(Value::Str(format!(
                "WARNING [{id}] common gauge origin: the tensors depend on gauge_origin in a \
                 finite basis"
            )));
        }
        report.set("warnings", Value::Array(all));
    }
    if let Some(value) = correlated {
        let mut labels = report.get("units").unwrap().clone();
        let names = [
            "total",
            "parent",
            "target",
            "pt2",
            "same_spin_alpha_unscaled",
            "opposite_spin_unscaled",
            "same_spin_beta_unscaled",
            "parent_nonlocal",
            "target_nonlocal",
        ];
        let mut terms = Value::object();
        for (channel, name) in names.iter().enumerate() {
            let xi = [
                &value.magnetic.total,
                &value.magnetic.parent,
                &value.magnetic.target,
                &value.magnetic.correlation,
                &value.magnetic.same_spin_alpha,
                &value.magnetic.opposite_spin,
                &value.magnetic.same_spin_beta,
                &value.magnetic.parent_nonlocal,
                &value.magnetic.target_nonlocal,
            ][channel];
            terms.set(
                name,
                Value::object()
                    .with("polarizability", value.polarizability[channel])
                    .with("magnetizability", *xi)
                    .with("hypermagnetizability", value.tensor[channel])
                    .with(
                        "accumulation_roundoff",
                        value.accumulation_roundoff[channel],
                    ),
            );
            for (field, unit) in [
                ("polarizability", "e^2*bohr^2/hartree"),
                ("magnetizability", "e^2*bohr^2/m_e"),
                (
                    "hypermagnetizability",
                    "atomic units, -d4E/(dE_a dE_b dB_c dB_d)",
                ),
                ("accumulation_roundoff", "atomic units"),
            ] {
                labels.set(&format!("correlated_terms.{name}.{field}"), unit);
            }
        }
        report.set("correlated_terms", terms);
        append_correlated_magnetic_diagnostics(&mut report, &mut labels, value.magnetic);
        report.set("units", labels);
    }
    if let Some(data) = prepared {
        super::double_hybrid::append_prepared_energy(data, report)
    } else {
        Ok(report)
    }
}

/// Faraday rotation tensor `Im d alpha_ab(w) / dB_c` of the paired response
/// along magnetic paths, with London orbitals (default) or a common gauge
/// origin, and its rotational invariant `eps_abc Im d alpha_ab / dB_c / 6`.
pub(super) fn magneto_optical(system: &System, controls: &Controls) -> Result<Value> {
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "correlated magneto-optical response".into(),
            missing: "the causal magnetic derivative of the PT2 amplitude and orbital response"
                .into(),
        });
    }
    let config = response_config(controls)?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let gauge = controls.str("gauge", "london")?;
    let source = controls.str("magneto_optical_source", "auto")?;
    let soc_charges = controls.opt_f64_list("soc_charges")?;
    let frequency = super::polarizability::frequency(controls)?;
    let crate::response::Frequency::Retarded { omega, damping } = frequency else {
        return Err(DftError::InvalidInput(
            "magneto-optical needs a positive real photon energy".into(),
        ));
    };
    if omega <= 0.0 {
        return Err(DftError::InvalidInput(
            "magneto-optical needs a positive real photon energy".into(),
        ));
    }
    let setup = setup(system, controls)?;
    controls.finish("magneto-optical")?;
    let london = match gauge.as_str() {
        "london" => true,
        "common-origin" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown gauge `{other}` for magneto-optical (london or common-origin)"
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
    let origin = match explicit {
        Some(origin) => origin,
        None => charge_centre(&setup.molecule, "magneto-optical")?,
    };
    // An open-shell reference carries a magnetization: its magneto-optical
    // tensor is the first-order spin-orbit change of the polarizability.
    let spin_orbit = match source.as_str() {
        "spin-orbit" => true,
        "field" => false,
        "auto" => scf.unrestricted && scf.n_alpha != scf.n_beta,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown magneto_optical_source `{other}` (auto, spin-orbit or field)"
            )))
        }
    };
    if spin_orbit {
        return spin_orbit_magneto_optical(&setup, &reference, soc_charges, frequency, &config);
    }
    let result = crate::response::field_path::causal_faraday_tensor(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        origin,
        london,
        frequency,
        &config,
    )?;
    let faraday = result
        .derivative
        .map(|row| row.map(|entry| entry.map(|z| z.im)));
    let faraday_real = result
        .derivative
        .map(|row| row.map(|entry| entry.map(|z| z.re)));
    let real = result.polarizability.map(|row| row.map(|z| z.re));
    let imaginary = result.polarizability.map(|row| row.map(|z| z.im));
    let mut invariant = 0.0;
    for (a, b, c, sign) in [
        (0, 1, 2, 1.0),
        (1, 2, 0, 1.0),
        (2, 0, 1, 1.0),
        (0, 2, 1, -1.0),
        (2, 1, 0, -1.0),
        (1, 0, 2, -1.0),
    ] {
        invariant += sign * faraday[a][b][c] / 6.0;
    }
    Ok(reference_report(
        &setup,
        &reference,
        "Faraday rotation tensor: first magnetic-field derivative of the frequency-dependent \
         polarizability of the paired response",
    )
    .with("gauge", gauge.clone())
    .with("gauge_origin", origin)
    .with("frequency", omega)
    .with("frequency_damping", damping)
    .with("magneto_optical_source", "field")
    .with("polarizability", real)
    .with("polarizability_real", real)
    .with("polarizability_imag", imaginary)
    .with("faraday_tensor", faraday)
    .with("faraday_tensor_real", faraday_real)
    .with("faraday_tensor_imag", faraday)
    .with("faraday_invariant", invariant)
    .with("response_residual", result.residual)
    .with("response_reference_fock_defect", result.reference_defect)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("gauge_origin", "bohr"),
            ("frequency", "hartree"),
            ("frequency_damping", "hartree"),
            ("polarizability_real", "e^2*bohr^2/hartree"),
            ("polarizability_imag", "e^2*bohr^2/hartree"),
            ("response_reference_fock_defect", "hartree"),
            (
                "polarizability",
                "e^2*bohr^2/hartree, alpha_ab at the frequency",
            ),
            ("faraday_tensor", "atomic units, Im d alpha_ab / dB_c"),
            ("faraday_tensor_real", "atomic units, Re d alpha_ab / dB_c"),
            ("faraday_tensor_imag", "atomic units, Im d alpha_ab / dB_c"),
            (
                "faraday_invariant",
                "atomic units, eps_abc Im d alpha_ab / dB_c / 6",
            ),
            ("response_residual", "dimensionless"),
        ]),
    ))
}

/// Magneto-optical tensor of an open-shell reference to first order in the
/// one-electron spin-orbit coupling `(alpha^2 / 2) sum_A Z_A l_A . s / r_A^3`
/// (`soc_charges` replaces the nuclear charges): for the spins along each
/// axis `k`, the antisymmetric imaginary part of the first-order change of
/// the frequency-dependent polarizability.
fn spin_orbit_magneto_optical(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    soc_charges: Option<Vec<f64>>,
    frequency: crate::response::Frequency,
    config: &crate::response::krylov::ResponseSolverConfig,
) -> Result<Value> {
    let scf = &reference.scf;
    let natoms = setup.molecule.len();
    let charges: Vec<f64> = match soc_charges {
        Some(list) if list.len() == natoms && list.iter().all(|z| z.is_finite()) => list,
        Some(_) => {
            return Err(DftError::InvalidInput(format!(
                "soc_charges holds one finite effective charge for each of the {natoms} atoms"
            )))
        }
        None => setup.molecule.atoms.iter().map(|a| a.z as f64).collect(),
    };
    // (alpha^2 / 2) sum_A Z_A (r_A x grad) / r_A^3.
    let n = scf.overlap.rows;
    let mut orbital: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    let scale = 0.5 * crate::constants::FINE_STRUCTURE * crate::constants::FINE_STRUCTURE;
    for (atom, charge) in setup.molecule.atoms.iter().zip(&charges) {
        if *charge == 0.0 {
            continue;
        }
        let operator = crate::properties::nmr::paramagnetic_operator(&setup.basis, atom.position);
        for (target, source) in orbital.iter_mut().zip(&operator) {
            for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
                *t += scale * charge * s;
            }
        }
    }
    let result = crate::response::field_path::causal_spin_orbit_polarizability(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        &orbital,
        frequency,
        config,
    )?;
    let energy = frequency.energy()?;
    let real = result.polarizability.map(|row| row.map(|z| z.re));
    let imaginary = result.polarizability.map(|row| row.map(|z| z.im));
    let spin_orbit_real = result
        .derivative
        .map(|row| row.map(|entry| entry.map(|z| z.re)));
    let spin_orbit_imaginary = result
        .derivative
        .map(|row| row.map(|entry| entry.map(|z| z.im)));
    // Axial vector of the antisymmetric part for every spin axis, and what
    // is left outside it.
    let mut axial = [[0.0; 3]; 3];
    let mut remainder = 0.0_f64;
    for k in 0..3 {
        for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            axial[c][k] = 0.5 * (spin_orbit_imaginary[a][b][k] - spin_orbit_imaginary[b][a][k]);
        }
        for a in 0..3 {
            for b in 0..3 {
                remainder =
                    remainder.max((result.derivative[a][b][k] + result.derivative[b][a][k]).abs());
                if energy.im == 0.0 {
                    remainder = remainder.max(spin_orbit_real[a][b][k].abs());
                }
            }
        }
    }
    Ok(reference_report(
        setup,
        reference,
        "magneto-optical tensor of an open-shell reference: first-order change of the \
         frequency-dependent polarizability under one-electron spin-orbit coupling",
    )
    .with("magneto_optical_source", "spin-orbit")
    .with("soc_charges", charges)
    .with("frequency", energy.re)
    .with("frequency_damping", energy.im)
    .with("polarizability", real)
    .with("polarizability_real", real)
    .with("polarizability_imag", imaginary)
    .with("magneto_optical_tensor", spin_orbit_imaginary)
    .with("magneto_optical_tensor_real", spin_orbit_real)
    .with("magneto_optical_tensor_imag", spin_orbit_imaginary)
    .with("magneto_optical_axial", axial)
    .with("symmetric_remainder", remainder)
    .with("response_residual", result.residual)
    .with("response_reference_fock_defect", result.reference_defect)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("soc_charges", "e"),
            ("frequency", "hartree"),
            ("frequency_damping", "hartree"),
            ("polarizability_real", "e^2*bohr^2/hartree"),
            ("polarizability_imag", "e^2*bohr^2/hartree"),
            ("response_reference_fock_defect", "hartree"),
            (
                "polarizability",
                "e^2*bohr^2/hartree, alpha_ab at the frequency without spin-orbit coupling",
            ),
            (
                "magneto_optical_tensor",
                "atomic units, Im d alpha_ab / d lambda as [a][b][spin axis]",
            ),
            (
                "magneto_optical_tensor_real",
                "atomic units, Re d alpha_ab / d lambda",
            ),
            (
                "magneto_optical_tensor_imag",
                "atomic units, Im d alpha_ab / d lambda",
            ),
            (
                "magneto_optical_axial",
                "atomic units, eps_abc Im d alpha_ab / d lambda / 2 as [c][spin axis]",
            ),
            ("symmetric_remainder", "atomic units"),
            ("response_residual", "dimensionless"),
        ]),
    ))
}

struct FiniteMagneticState {
    setup: MolecularSetup,
    reference: Option<MolecularReference>,
    options: crate::scf::magnetic::MagneticFieldOptions,
    result: crate::scf::magnetic::MagneticFieldResult,
    seed_failure: Option<String>,
}

/// Hessian and contracted cubic/quartic nuclear tensors at a finite London
/// magnetic field and an optional simultaneous electric field. Every surface
/// enters here through the existing observable identifiers and controls.
pub(super) fn nuclear_surface(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported { feature:"correlated finite-field nuclear response".into(), missing:"the complex PT2 amplitude and orbital nuclear response on the finite-field reference".into() });
    }
    let config = response_config(controls)?;
    let order = match id {
        "hessian" => 2,
        "cubic-fc-along" => 3,
        "quartic-fc-along" => 4,
        _ => {
            return Err(DftError::InvalidInput(
                "unknown finite-field nuclear tensor".into(),
            ))
        }
    };
    let dimension = 3 * system.numbers.len();
    let probe = if order > 2 {
        Some(controls.opt_f64_list("probe")?.ok_or_else(|| {
            DftError::InvalidInput("a contracted finite-field nuclear tensor requires probe".into())
        })?)
    } else {
        None
    };
    let second = if order == 4 {
        Some(
            controls
                .opt_f64_list("second_probe")?
                .unwrap_or_else(|| probe.as_ref().expect("quartic probe").clone()),
        )
    } else {
        None
    };
    if probe
        .iter()
        .chain(second.iter())
        .any(|v| v.len() != dimension || v.iter().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "finite-field nuclear probes must have one finite entry per Cartesian coordinate"
                .into(),
        ));
    }
    let state = finite_magnetic_state(id, system, controls, Some(order))?;
    if !state.result.converged {
        return Err(DftError::ScfNotConverged {
            iterations: state.result.iterations,
            error: state.result.residual,
        });
    }
    let mut cache: Vec<(Vec<f64>, usize, Vec<f64>)> = Vec::new();
    let mut residuals = Vec::new();
    let mut projector_defects = Vec::new();
    let mut stationarity_defects = Vec::new();
    let mut reference_defect = 0.0_f64;
    let mut run = |direction: &[f64], degree: usize| -> Result<Vec<f64>> {
        if let Some((_, _, e)) = cache
            .iter()
            .find(|(v, n, _)| v.as_slice() == direction && *n >= degree)
        {
            return Ok(vec![e[degree]]);
        }
        let direction3: Vec<[f64; 3]> = direction
            .chunks_exact(3)
            .map(|d| [d[0], d[1], d[2]])
            .collect();
        let result = crate::response::nuclear::magnetic_nuclear_taylor(
            &state.setup.molecule,
            &state.setup.basis,
            &state.setup.functional,
            state.setup.grid.as_ref(),
            &state.setup.options,
            &state.options,
            &state.result,
            &direction3,
            degree,
            &config,
        )?;
        reference_defect = reference_defect.max(result.reference_defect);
        residuals.extend(result.residuals);
        projector_defects.extend(result.projector_defects);
        stationarity_defects.extend(result.stationarity_defects);
        let mut energy = result.energy;
        if state.setup.dispersion != crate::dispersion::Dispersion::None {
            use crate::dual::Taylor;
            let positions: Vec<[Taylor<5>; 3]> = system
                .positions
                .iter()
                .zip(&direction3)
                .map(|(r, d)| std::array::from_fn(|a| Taylor::from_coefficients(&[r[a], d[a]])))
                .collect();
            let dispersion = crate::dispersion::dispersion_energy_g(
                &system.numbers,
                &positions,
                &state.setup.dispersion,
                &state.setup.method.dispersion_functional_name(),
            )?;
            for n in 0..=degree {
                energy[n] += dispersion.c[n];
            }
        }
        let value = energy[degree];
        cache.push((direction.to_vec(), degree, energy));
        Ok(vec![value])
    };
    let leading: Vec<_> = probe.iter().chain(second.iter()).cloned().collect();
    let (matrix, accumulation_roundoff) =
        crate::response::mixed::polarized_matrix(dimension, &leading, &mut run)?;
    let unit = match order {
        2 => "hartree/bohr^2",
        3 => "hartree/bohr^3",
        _ => "hartree/bohr^4",
    };
    let key = if order == 2 {
        "hessian"
    } else {
        "contracted_force_constants"
    };
    let dispersion = crate::dispersion::dispersion_energy(
        &state.setup.molecule,
        &state.setup.dispersion,
        &state.setup.method.dispersion_functional_name(),
    )?;
    let mut report=Value::object()
        .with("model","analytic nuclear response about the self-consistent finite London-field state; fixed Cartesian electric/magnetic fields, moving AO metric and atom-owned quadrature; integer occupations")
        .with("functional",state.setup.functional.description.as_str())
        .with("basis",state.setup.method.basis.as_str())
        .with("charge",state.setup.molecule.charge)
        .with("multiplicity",state.setup.molecule.multiplicity)
        .with("unrestricted",state.result.unrestricted)
        .with("n_alpha",state.result.n_alpha)
        .with("n_beta",state.result.n_beta)
        .with("spin_quantization_convention","fixed collinear spin populations; alpha/beta Zeeman offsets are +/- g_e |B|/4")
        .with("energy",state.result.energy+dispersion)
        .with("bfield",state.options.field)
        .with("electric_field",state.result.electric_field)
        .with("spin_zeeman",state.options.spin_zeeman)
        .with("converged",true)
        .with("response_converged",residuals.iter().all(|r| *r<=config.acceptance()))
        .with("field_iterations",state.result.iterations)
        .with("field_residual",state.result.residual)
        .with("series_order",state.result.series_order)
        .with("series_tail",state.result.series_tail)
        .with("reference_defect",reference_defect)
        .with("response_residuals",residuals)
        .with("projector_defects",projector_defects)
        .with("stationarity_defects",stationarity_defects)
        .with("path_evaluations",cache.len())
        .with("accumulation_roundoff",accumulation_roundoff)
        .with("symmetrized",true)
        .with("symmetry_convention","intrinsic permutation used in production; independent ordered asymmetry is not evaluated by this driver")
        .with("warnings",state.seed_failure.iter().cloned().map(Value::Str).collect::<Vec<_>>())
        .with("provenance","analytic")
        .with(key,matrix);
    if let Some(probe) = probe {
        report.set("probe", probe);
    }
    if let Some(second) = second {
        report.set("second_probe", second);
    }
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            (key, unit),
            ("bfield", "atomic units of magnetic flux density"),
            ("electric_field", "hartree/e/bohr"),
            ("field_residual", "hartree"),
            ("series_tail", "hartree"),
            ("reference_defect", "hartree"),
            ("response_residuals", "dimensionless"),
            ("projector_defects", "dimensionless"),
            ("stationarity_defects", "hartree"),
            ("accumulation_roundoff", unit),
            ("probe", "dimensionless"),
            ("second_probe", "dimensionless"),
        ]),
    );
    Ok(report)
}

fn finite_magnetic_state(
    id: &str,
    system: &System,
    controls: &Controls,
    nuclear_order: Option<usize>,
) -> Result<FiniteMagneticState> {
    let field = controls.opt_vec3("bfield")?.ok_or_else(|| {
        DftError::InvalidInput(format!(
            "{id} needs bfield: the uniform magnetic field in atomic units \
             (1 a.u. = 2.35052e5 T)"
        ))
    })?;
    let spin_zeeman = controls.bool("spin_zeeman", true)?;
    let mut series_order = controls.opt_usize("bfield_series_order")?;
    let initial_guess = controls.str("bfield_initial_guess", "auto")?;
    if !matches!(initial_guess.as_str(), "auto" | "core" | "zero-field") {
        return Err(DftError::InvalidInput(
            "bfield_initial_guess must be auto, core or zero-field".into(),
        ));
    }
    let setup = setup(system, controls)?;
    controls.finish(id)?;
    if initial_guess == "core" && setup.method.stabilize {
        return Err(DftError::InvalidInput(
            "bfield_initial_guess=core cannot bypass requested zero-field stability following"
                .into(),
        ));
    }
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "finite magnetic field SCF with the SMD non-electrostatic term".into(),
            missing: "the continuum terms over London orbitals".into(),
        });
    }
    let mut seed_failure = None;
    let reference = if initial_guess == "core" {
        None
    } else {
        match run_molecular_reference(&setup) {
            Ok(reference) if reference.scf.converged => Some(reference),
            Ok(reference) if initial_guess == "auto" && !setup.method.stabilize => {
                seed_failure = Some(format!(
                    "zero-field seed did not converge in {} iterations",
                    reference.scf.iterations
                ));
                None
            }
            Err(DftError::ScfNotConverged { iterations, error })
                if initial_guess == "auto" && !setup.method.stabilize =>
            {
                seed_failure = Some(format!("zero-field seed did not converge in {iterations} iterations (reported error {error}); finite-field core initialization used"));
                None
            }
            Ok(reference) => {
                return Err(DftError::ScfNotConverged {
                    iterations: reference.scf.iterations,
                    error: f64::NAN,
                })
            }
            Err(error) => return Err(error),
        }
    };
    if let Some(order) = nuclear_order {
        // Nuclear derivatives can generate phase terms that vanish in the
        // reference geometry. Retain their field degree before building SCF.
        let estimate = series_order.unwrap_or_else(|| {
            crate::scf::magnetic::recommended_series_order(
                &setup.molecule,
                &setup.basis,
                [field.x, field.y, field.z],
                1e-12,
            )
            .saturating_add(order + 2)
        });
        series_order = Some(estimate.max(order + 2).min(24));
    }
    let options = crate::scf::magnetic::MagneticFieldOptions {
        field: [field.x, field.y, field.z],
        spin_zeeman,
        series_order,
        series_tolerance: 1e-12,
    };
    let result = if let Some(reference) = &reference {
        crate::scf::magnetic::magnetic_field_scf(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            &options,
        )?
    } else {
        crate::scf::magnetic::magnetic_field_scf_from_core(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &setup.options,
            &options,
        )?
    };
    Ok(FiniteMagneticState {
        setup,
        reference,
        options,
        result,
        seed_failure,
    })
}

/// Self-consistent field in a finite uniform magnetic field with London
/// orbitals, started from the zero-field reference; `bfield-gradient` adds
/// the analytic gradient of the energy in the field.
pub(super) fn finite_field(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let with_gradient = id == "bfield-gradient";
    let FiniteMagneticState {
        setup,
        reference,
        options,
        result,
        seed_failure,
    } = finite_magnetic_state(id, system, controls, None)?;
    let field = Vec3::new(options.field[0], options.field[1], options.field[2]);
    let spin_zeeman = options.spin_zeeman;
    let dispersion = match &reference {
        Some(reference) => reference.e_dispersion,
        None => crate::dispersion::dispersion_energy(
            &setup.molecule,
            &setup.dispersion,
            &setup.method.dispersion_functional_name(),
        )?,
    };
    const ATOMIC_UNIT_IN_TESLA: f64 = 2.350_517_567_58e5;
    let tesla = [field.x, field.y, field.z].map(|v| v * ATOMIC_UNIT_IN_TESLA);
    let model = "self-consistent field in a finite uniform magnetic field with London orbitals: convergent integral series, complex Hermitian Kohn-Sham matrices, gauge-invariant kinetic energy density; core or converged zero-field initialization";
    let base = if let Some(reference) = &reference {
        reference_report(&setup, reference, model)
    } else {
        Value::object()
            .with("model", model)
            .with("functional", setup.functional.description.as_str())
            .with("basis", setup.method.basis.as_str())
            .with("energy", Option::<f64>::None)
            .with("n_alpha", result.n_alpha)
            .with("n_beta", result.n_beta)
            .with("unrestricted", result.unrestricted)
            .with("scf_iterations", result.iterations)
    };
    let mut warnings = match base.get("warnings") {
        Some(Value::Array(warnings)) => warnings.clone(),
        _ => Vec::new(),
    };
    warnings.extend(seed_failure.iter().cloned().map(Value::Str));
    let mut report = base
        .with("bfield", [field.x, field.y, field.z])
        .with("electric_field", result.electric_field)
        .with("bfield_tesla", tesla)
        .with("spin_zeeman", spin_zeeman)
        .with("field_energy", result.energy + dispersion)
        .with("converged", result.converged)
        .with(
            "initial_guess",
            if reference.is_some() {
                "zero-field"
            } else {
                "core"
            },
        )
        .with(
            "zero_field_reference_converged",
            reference
                .as_ref()
                .map(|r| r.scf.converged)
                .or(seed_failure.as_ref().map(|_| false)),
        )
        .with("zero_field_seed_failure", seed_failure.clone())
        .with("warnings", warnings)
        .with("zeeman_energy", result.zeeman_energy)
        .with("field_nonlocal_correlation", result.e_nlc)
        .with("field_orbital_energies", result.orbital_energies.to_vec())
        .with("field_converged", result.converged)
        .with("field_iterations", result.iterations)
        .with("field_residual", result.residual)
        .with("series_order", result.series_order)
        .with("series_tail", result.series_tail)
        .with("provenance", "analytic");
    let mut list = vec![
        ("energy", "hartree, zero-field reference"),
        ("electric_field", "hartree/e/bohr"),
        ("bfield", "atomic units of magnetic flux density"),
        ("bfield_tesla", "tesla"),
        ("field_energy", "hartree, total energy in the field"),
        ("zeeman_energy", "hartree"),
        ("field_nonlocal_correlation", "hartree"),
        ("field_orbital_energies", "hartree"),
        ("field_residual", "hartree"),
        ("series_tail", "hartree"),
    ];
    if with_gradient {
        let mut gradient = crate::scf::magnetic::magnetic_field_gradient(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &options,
            &result,
        )?;
        if setup.dispersion != crate::dispersion::Dispersion::None {
            let correction = crate::dispersion::dispersion_gradient(
                &setup.molecule,
                &setup.dispersion,
                &setup.method.dispersion_functional_name(),
            )?;
            for (total, delta) in gradient.iter_mut().zip(correction) {
                let delta = delta.to_array();
                for axis in 0..3 {
                    total[axis] += delta[axis];
                }
            }
        }
        let mut sum = [0.0; 3];
        for g in &gradient {
            for axis in 0..3 {
                sum[axis] += g[axis];
            }
        }
        let target = result
            .electric_field
            .map(|x| -(setup.molecule.charge as f64) * x);
        let residual = (0..3)
            .map(|a| (sum[a] - target[a]).powi(2))
            .sum::<f64>()
            .sqrt();
        report.set("gradient_translation_target", target);
        report.set("gradient", gradient);
        report.set("translation_residual", residual);
        list.push(("gradient", "hartree/bohr"));
        list.push(("translation_residual", "hartree/bohr"));
        list.push(("gradient_translation_target", "hartree/bohr"));
    }
    report.set("units", units(&list));
    Ok(report)
}

/// Linear magnetoelectric response of an open-shell reference to first
/// order in the one-electron spin-orbit coupling
/// `(alpha^2 / 2) sum_A Z_A l_A . s / r_A^3` (`soc_charges` replaces the
/// nuclear charges): the electric dipole a magnetic field induces, for the
/// spins along each axis.
pub(super) fn magnetoelectric(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let gauge = controls.str("gauge", "london")?;
    let soc_charges = controls.opt_f64_list("soc_charges")?;
    let setup = setup(system, controls)?;
    controls.finish("magnetoelectric")?;
    let london = match gauge.as_str() {
        "london" => true,
        "common-origin" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown gauge `{other}` for magnetoelectric (london or common-origin)"
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
    let natoms = setup.molecule.len();
    let charges: Vec<f64> = match soc_charges {
        Some(list) if list.len() == natoms && list.iter().all(|z| z.is_finite()) => list,
        Some(_) => {
            return Err(DftError::InvalidInput(format!(
                "soc_charges holds one finite effective charge for each of the {natoms} atoms"
            )))
        }
        None => setup.molecule.atoms.iter().map(|a| a.z as f64).collect(),
    };
    // (alpha^2 / 2) sum_A Z_A (r_A x grad) / r_A^3.
    let n = scf.overlap.rows;
    let mut orbital: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    let scale = 0.5 * crate::constants::FINE_STRUCTURE * crate::constants::FINE_STRUCTURE;
    for (atom, charge) in setup.molecule.atoms.iter().zip(&charges) {
        if *charge == 0.0 {
            continue;
        }
        let operator = crate::properties::nmr::paramagnetic_operator(&setup.basis, atom.position);
        for (target, source) in orbital.iter_mut().zip(&operator) {
            for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
                *t += scale * charge * s;
            }
        }
    }
    let origin = match explicit {
        Some(origin) => origin,
        None => charge_centre(&setup.molecule, "magnetoelectric")?,
    };
    let result = crate::response::field_path::spin_orbit_magnetoelectric(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        &orbital,
        origin,
        london,
        &config,
    )?;
    Ok(reference_report(
        &setup,
        &reference,
        "linear magnetoelectric response of an open-shell reference: electric dipole induced \
         by a magnetic field to first order in the one-electron spin-orbit coupling, from mixed \
         third derivatives of the energy along field paths",
    )
    .with("gauge", gauge.as_str())
    .with("gauge_origin", origin.to_array())
    .with("soc_charges", charges)
    .with("magnetoelectric_tensor", result.tensor)
    .with("path_evaluations", result.evaluations)
    .with("accumulation_roundoff", result.accumulation_roundoff)
    .with("response_residual", result.residual)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("gauge", "dimensionless"),
            ("gauge_origin", "bohr"),
            ("soc_charges", "e"),
            (
                "magnetoelectric_tensor",
                "atomic units, d mu_a/dB_b per unit spin-orbit strength as [a][b][spin axis]",
            ),
            ("accumulation_roundoff", "atomic units"),
            ("response_residual", "dimensionless"),
        ]),
    ))
}

/// Complete static orbital tensor of a post-SCF target on its stationary parent.
fn correlated_report(
    prepared: super::double_hybrid::Prepared,
    origin: Vec3,
    gauge: &str,
    config: &crate::response::ResponseSolverConfig,
) -> Result<Value> {
    let gauge_kind = match gauge {
        "common-origin" => Gauge::CommonOrigin,
        "london" => Gauge::Giao,
        _ => {
            return Err(DftError::InvalidInput(
                "magnetizability gauge must be common-origin or london".into(),
            ))
        }
    };
    let setup = &prepared.setup;
    let value = crate::properties::correlated_magnetic::correlated_magnetizability_for_model(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        prepared.field_target()?,
        setup.grid.as_ref(),
        &prepared.reference.scf,
        &setup.options,
        prepared.coefficients,
        gauge_kind,
        origin,
        prepared.pt2_memory_mb,
        config,
    )?;
    let isotropic = (value.total[0][0] + value.total[1][1] + value.total[2][2]) / 3.0;
    let mut report = reference_report(setup, &prepared.reference,
        "static orbital magnetizability of the full target plus noncanonical PT2 energy; integer collinear stationary parent; spin Zeeman and current-dependent correlation excluded")
        .with("gauge", gauge)
        .with("gauge_origin", origin)
        .with("provenance", "analytic")
        .with("isotropic", isotropic)
        .with("isotropic_cgs_ppm", isotropic * ATOMIC_UNIT_IN_CGS_PPM);
    let mut labels = units(&[
        ("energy", "hartree"),
        ("gauge_origin", "bohr"),
        ("isotropic", "e^2*bohr^2/m_e"),
        ("isotropic_cgs_ppm", "1e-6 cm^3/mol"),
    ]);
    for (key, tensor) in [
        ("magnetizability", value.total),
        ("parent_magnetizability", value.parent),
        ("target_magnetizability", value.target),
        ("pt2_magnetizability", value.correlation),
        (
            "same_spin_alpha_magnetizability_unscaled",
            value.same_spin_alpha,
        ),
        (
            "opposite_spin_magnetizability_unscaled",
            value.opposite_spin,
        ),
        (
            "same_spin_beta_magnetizability_unscaled",
            value.same_spin_beta,
        ),
        ("parent_nonlocal_magnetizability", value.parent_nonlocal),
        ("target_nonlocal_magnetizability", value.target_nonlocal),
    ] {
        report.set(key, tensor);
        labels.set(key, "e^2*bohr^2/m_e, xi_ab = -d2E/(dB_a dB_b)");
    }
    append_correlated_magnetic_diagnostics(&mut report, &mut labels, value);
    report.set("units", labels);
    super::double_hybrid::append_prepared_energy(prepared, report)
}

/// Preserve the dimensional raw residuals for both xi and mixed E/E/B/B eta.
fn append_correlated_magnetic_diagnostics(
    report: &mut Value,
    labels: &mut Value,
    value: crate::properties::correlated_magnetic::CorrelatedMagnetizability,
) {
    for (key, values, unit) in [
        (
            "response_residuals",
            value.response_residuals,
            "dimensionless",
        ),
        (
            "amplitude_response_residuals",
            value.amplitude_residuals,
            "hartree",
        ),
        (
            "amplitude_relative_residuals",
            value.amplitude_relative_residuals,
            "dimensionless",
        ),
        (
            "projector_defects",
            value.projector_defects,
            "dimensionless",
        ),
        (
            "stationarity_defects",
            value.stationarity_defects,
            "hartree",
        ),
        ("reference_defects", value.reference_defects, "hartree"),
        (
            "orbital_frame_defects",
            value.orbital_frame_defects,
            "dimensionless",
        ),
        (
            "imaginary_energy_defects",
            value.imaginary_energy_defects,
            "hartree",
        ),
    ] {
        report.set(key, values);
        labels.set(key, unit);
    }
}
