// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Pairing and optional phonon mass on supplied normal-band and interaction data.

use super::{Controls, Value};
use crate::dual::Taylor;
use crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
use crate::properties::pairing::{Lm2005OffDiagonal, PairingInput};
use crate::{DftError, Result};

pub(super) fn required(c: &Controls, key: &str) -> Result<Vec<f64>> {
    c.opt_f64_list(key)?
        .ok_or_else(|| DftError::InvalidInput(format!("{key} is required")))
}
pub(super) fn rows(c: &Controls, key: &str, n: usize) -> Result<Vec<f64>> {
    let data = c
        .opt_rows(key)?
        .ok_or_else(|| DftError::InvalidInput(format!("{key} is required")))?;
    if data.len() != n || data.iter().any(|r| r.len() != n) {
        return Err(DftError::InvalidInput(format!(
            "{key} must be an n by n matrix"
        )));
    }
    Ok(data.into_iter().flatten().collect())
}
pub(super) fn append(
    parameters: &mut Vec<Taylor<5>>,
    values: &[f64],
    direction: &[f64],
) -> Result<()> {
    if values.len() != direction.len() {
        return Err(DftError::InvalidInput(
            "superconducting parameter direction has the wrong shape".into(),
        ));
    }
    parameters.extend(
        values
            .iter()
            .zip(direction)
            .map(|(&x, &d)| Taylor::from_coefficients(&[x, d])),
    );
    Ok(())
}
pub(super) fn direction(c: &Controls, key: &str, n: usize) -> Result<Vec<f64>> {
    let data = c.opt_f64_list(key)?.unwrap_or_else(|| vec![0.0; n]);
    if data.len() != n {
        return Err(DftError::InvalidInput(format!("{key} needs {n} entries")));
    }
    Ok(data)
}
pub(super) fn matrix_direction(c: &Controls, key: &str, n: usize) -> Result<Vec<f64>> {
    match c.opt_rows(key)? {
        None => Ok(vec![0.0; n * n]),
        Some(data) if data.len() == n && data.iter().all(|r| r.len() == n) => {
            Ok(data.into_iter().flatten().collect())
        }
        _ => Err(DftError::InvalidInput(format!(
            "{key} must be an n by n matrix"
        ))),
    }
}

pub(super) fn run(c: &Controls) -> Result<Value> {
    let energies = required(c, "band_energies")?;
    let n = energies.len();
    if n == 0 || n.checked_mul(n).is_none() {
        return Err(DftError::InvalidInput(
            "pairing needs a nonempty finite band dimension".into(),
        ));
    }
    let mu = c
        .opt_f64("chemical_potential")?
        .ok_or_else(|| DftError::InvalidInput("chemical_potential is required".into()))?;
    let temperature = c.opt_f64("electronic_temperature")?.ok_or_else(|| {
        DftError::InvalidInput("electronic_temperature is required in Kelvin".into())
    })?;
    let energy_direction = direction(c, "band_energy_direction", n)?;
    let phonons = required(c, "phonon_energies")?;
    let modes = phonons.len();
    let vertices = c
        .opt_rows("pair_couplings_squared")?
        .ok_or_else(|| DftError::InvalidInput("pair_couplings_squared is required".into()))?;
    let vertex_direction = c
        .opt_rows("pair_couplings_squared_direction")?
        .unwrap_or_else(|| vec![vec![0.0; n * n]; modes]);
    if vertices.len() != modes
        || vertex_direction.len() != modes
        || vertices
            .iter()
            .chain(&vertex_direction)
            .any(|r| r.len() != n * n)
    {
        return Err(DftError::InvalidInput(
            "squared phonon vertices and directions need m flattened n*n rows".into(),
        ));
    }
    let coulomb = rows(c, "screened_pair_coulomb", n)?;
    let jets = |values: &[f64], directions: &[f64]| -> Result<Vec<Taylor<5>>> {
        let mut result = Vec::new();
        append(&mut result, values, directions)?;
        Ok(result)
    };
    let input = PairingInput {
        energies: jets(&energies, &energy_direction)?,
        chemical_potential: Taylor::from_coefficients(&[
            mu,
            c.f64("chemical_potential_direction", 0.0)?,
        ]),
        temperature_energy: Taylor::from_coefficients(&[
            temperature * BOLTZMANN_HARTREE_PER_K,
            c.f64("temperature_direction", 0.0)? * BOLTZMANN_HARTREE_PER_K,
        ]),
        phonon_energies: jets(&phonons, &direction(c, "phonon_energy_direction", modes)?)?,
        coupling_squared: vertices
            .iter()
            .zip(&vertex_direction)
            .map(|(v, d)| jets(v, d))
            .collect::<Result<Vec<_>>>()?,
        screened_coulomb: jets(
            &coulomb,
            &matrix_direction(c, "screened_pair_coulomb_direction", n)?,
        )?,
    };
    let weights = c.opt_f64_list("band_weights")?;
    let weight_jets = match &weights {
        Some(values) => Some(jets(values, &direction(c, "band_weight_direction", n)?)?),
        None => None,
    };
    let order = c.usize("response_order", 0)?;
    if order > 4 {
        return Err(DftError::InvalidInput(
            "pairing response_order must be between zero and four".into(),
        ));
    }
    c.finish("scdft-pairing-kernel")?;
    let rule = Lm2005OffDiagonal::default();
    let alternative = Lm2005OffDiagonal::new(16)?;
    let result = rule.evaluate(&input)?;
    let check = alternative.evaluate(&input)?;
    if result
        .total
        .iter()
        .chain(&result.phonon)
        .chain(&result.coulomb)
        .chain(&check.total)
        .any(|x| x.c[..=order].iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::LinearAlgebra(
            "pairing path coefficient overflow".into(),
        ));
    }
    let matrices = |data: &[Taylor<5>], k: usize| -> Vec<Vec<f64>> {
        data.chunks(n)
            .map(|r| r.iter().map(|x| x.c[k]).collect())
            .collect()
    };
    let mut absolute_error = vec![0.0_f64; order + 1];
    let mut relative_error = vec![0.0_f64; order + 1];
    let mut asymmetry = vec![0.0_f64; order + 1];
    for k in 0..=order {
        let mut scale = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let value = result.total[i * n + j].c[k];
                scale = scale.max(value.abs());
                absolute_error[k] =
                    absolute_error[k].max((value - check.total[i * n + j].c[k]).abs());
                asymmetry[k] = asymmetry[k].max((value - result.total[j * n + i].c[k]).abs());
            }
        }
        relative_error[k] = if scale == 0.0 {
            absolute_error[k]
        } else {
            absolute_error[k] / scale
        };
    }
    let mut converged = relative_error.iter().all(|&x| x <= 1e-10);
    let mut report=Value::object()
        .with("model","LM2005 static singlet phonon K plus supplied screened Coulomb; optional generated particle-hole-symmetric phonon Z when band_weights are supplied; no normal-density backreaction or causal external-frequency kernel")
        .with("band_energies",energies).with("chemical_potential",mu).with("electronic_temperature",temperature)
        .with("phonon_energies",phonons).with("band_energy_direction",energy_direction).with("response_order",order)
        .with("phonon_pairing_kernel",matrices(&result.phonon,0)).with("coulomb_pairing_kernel",matrices(&result.coulomb,0))
        .with("pairing_kernel",matrices(&result.total,0))
        .with("pairing_kernel_coefficients",(0..=order).map(|k|matrices(&result.total,k)).collect::<Vec<_>>())
        .with("quadrature_absolute_errors",absolute_error).with("quadrature_relative_errors",relative_error).with("pairing_asymmetry",asymmetry)
        .with("coefficient_convention","factorial-normalized Taylor coefficients along supplied parameter directions; t dimensionless; mode/q weights in squared vertices; K excludes band weights; Z includes one final-band weight");
    let mut units = Value::object();
    for key in [
        "band_energies",
        "chemical_potential",
        "phonon_energies",
        "band_energy_direction",
        "phonon_pairing_kernel",
        "coulomb_pairing_kernel",
        "pairing_kernel",
        "pairing_kernel_coefficients",
        "quadrature_absolute_errors",
        "pairing_asymmetry",
    ] {
        units.set(key, "hartree");
    }
    units.set("electronic_temperature", "K");
    units.set("quadrature_relative_errors", "dimensionless");
    if let Some(weight_jets) = weight_jets {
        let mass = rule.mass(&input, &weight_jets)?;
        let check = alternative.mass(&input, &weight_jets)?;
        if mass
            .iter()
            .chain(&check)
            .any(|x| x.c[..=order].iter().any(|v| !v.is_finite()))
        {
            return Err(DftError::LinearAlgebra(
                "phonon mass path coefficient overflow".into(),
            ));
        }
        let mut absolute = Vec::new();
        let mut relative = Vec::new();
        for k in 0..=order {
            let error = mass
                .iter()
                .zip(&check)
                .map(|(x, y)| (x.c[k] - y.c[k]).abs())
                .fold(0.0_f64, f64::max);
            let scale = mass.iter().map(|x| x.c[k].abs()).fold(0.0_f64, f64::max);
            absolute.push(error);
            relative.push(if scale == 0.0 { error } else { error / scale });
        }
        converged &= relative.iter().all(|&x| x <= 1e-10);
        report.set("band_weights", weights.unwrap());
        report.set(
            "mass_renormalization",
            mass.iter().map(|x| x.c[0]).collect::<Vec<_>>(),
        );
        report.set(
            "mass_renormalization_coefficients",
            (0..=order)
                .map(|k| mass.iter().map(|x| x.c[k]).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        );
        report.set("mass_quadrature_absolute_errors", absolute);
        report.set("mass_quadrature_relative_errors", relative);
        report.set("mass_model","LM2005 phonon Z with the sign convention of PRB 88, 014514 Eq. (10), confluent Fermi derivatives and particle-hole-symmetric mass approximation; no Coulomb mass");
        for key in [
            "band_weights",
            "mass_renormalization",
            "mass_renormalization_coefficients",
            "mass_quadrature_absolute_errors",
            "mass_quadrature_relative_errors",
        ] {
            units.set(key, "dimensionless");
        }
    }
    report.set("converged", converged);
    report.set("warnings",if converged {Vec::<String>::new()} else {vec!["LM analytic-limit quadrature did not agree between orders 16 and 32 to relative tolerance 1e-10".into()]});
    report.set("units", units);
    Ok(report)
}
