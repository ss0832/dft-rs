// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static APT, Raman and hyper-Raman from shared mixed response providers.
use super::correlated_high_order::Diagnostics;
use super::double_hybrid::Prepared;
use super::{Controls, System, Value};
use crate::response::mixed::{polarized_mixed, DirectionalCoefficients};
use crate::response::nuclear::{
    static_spectroscopic_derivative, static_spectroscopic_from_mixed, MixedMolecularResponse,
    MolecularDirection, StaticSpectroscopicDerivative,
};
use crate::response::ResponseReport;
use crate::{DftError, Result};

pub(super) fn nuclear_direction(system: &System, controls: &Controls) -> Result<Vec<[f64; 3]>> {
    let raw = controls.take("nuclear_direction").ok_or_else(|| {
        DftError::InvalidInput("static-spectroscopy requires nuclear_direction".into())
    })?;
    let (shape, values) = raw.numeric_array().ok_or_else(|| {
        DftError::InvalidInput("nuclear_direction must be a finite [atom][xyz] array".into())
    })?;
    let atoms = system.numbers.len();
    if atoms == 0
        || !(shape == [atoms, 3] || atoms == 1 && shape == [3])
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "nuclear_direction must contain one finite Cartesian triple per atom".into(),
        ));
    }
    Ok(values.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect())
}

fn append_tensor(report: &mut Value, response: ResponseReport<StaticSpectroscopicDerivative>) {
    append_values(report, response.value);
}

pub(super) fn append_values(report: &mut Value, value: StaticSpectroscopicDerivative) {
    report.set("nuclear_direction", value.nuclear_direction);
    report.set("electric_order", value.electric_order);
    report.set("tensor", value.tensor);
    report.set("accumulation_roundoff", value.accumulation_roundoff);
    report.set("permutation_defect", value.permutation_defect);
    report.set("permutation_comparisons", value.permutation_comparisons);
    report.set("directional_evaluations", value.directional_evaluations);
    let unit = match value.electric_order {
        1 => "e*bohr",
        2 => "e^2*bohr^2/hartree",
        _ => "e^3*bohr^3/hartree^2",
    };
    let mut labels = report.get("units").cloned().expect("response units");
    // The supplied nuclear path is in Bohr and its parameter is dimensionless.
    // An axial unit displacement consequently returns a derivative per Bohr
    // numerically; ASE converts the entire path, not a hidden tensor index.
    for (key, unit) in [
        ("nuclear_direction", "bohr"),
        ("tensor", unit),
        ("accumulation_roundoff", unit),
        ("permutation_defect", unit),
    ] {
        labels.set(key, unit);
    }
    report.set("units", labels);
}

/// Differentiate the complete nonstationary target energy, including the
/// orbital and PT2-amplitude coefficients through the required total order.
/// The dipole is minus one additional electric derivative of that energy.
fn correlated_mixed(
    prepared: &Prepared,
    system: &System,
    directions: &[MolecularDirection],
    diagnostics: &mut Diagnostics,
    cache: &mut DirectionalCoefficients,
) -> Result<ResponseReport<MixedMolecularResponse>> {
    let atoms = prepared.setup.molecule.len();
    let dimension = 3 * atoms + 3;
    let config = prepared
        .config
        .as_ref()
        .expect("spectroscopic response configuration");
    let vectors: Vec<Vec<f64>> = directions
        .iter()
        .map(|d| {
            d.nuclear
                .iter()
                .flatten()
                .chain(&d.electric)
                .copied()
                .collect()
        })
        .collect();
    let initial_evaluations = cache.evaluations;
    let mut residuals = Vec::new();
    let mut coefficient = |d: &[f64], order: usize| {
        cache.get(d, order, 1, |d, order| {
            let direction = MolecularDirection {
                nuclear: d[..3 * atoms]
                    .chunks_exact(3)
                    .map(|v| [v[0], v[1], v[2]])
                    .collect(),
                electric: [d[3 * atoms], d[3 * atoms + 1], d[3 * atoms + 2]],
            };
            let value = super::correlated_nuclear::nuclear_path(prepared, &direction, order)?;
            diagnostics.nuclear(&value);
            residuals.extend(&value.response_residuals);
            Ok(vec![value.energy[order]])
        })
    };
    let mut energy = polarized_mixed(dimension, &vectors, &mut coefficient)?;
    let mut dipole = [0.0; 3];
    let mut roundoff = [0.0; 3];
    for a in 0..3 {
        let mut legs = vectors.clone();
        let mut field = vec![0.0; dimension];
        field[3 * atoms + a] = 1.0;
        legs.push(field);
        let value = polarized_mixed(dimension, &legs, &mut coefficient)?;
        dipole[a] = -value.value[0];
        roundoff[a] = value.accumulation_roundoff[0];
    }
    // Dispersion has no explicit electric dependence in the supported
    // energy composition. Its mixed electric derivatives vanish analytically.
    if directions.iter().all(|d| d.electric == [0.0; 3]) {
        let nuclear: Vec<Vec<f64>> = directions
            .iter()
            .map(|d| d.nuclear.iter().flatten().copied().collect())
            .collect();
        let dispersion = polarized_mixed(3 * atoms, &nuclear, |d, order| {
            let d: Vec<_> = d.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
            Ok(vec![
                super::correlated_nuclear::dispersion_path(prepared, system, &d, order)?[order],
            ])
        })?;
        energy.value[0] += dispersion.value[0];
        energy.accumulation_roundoff[0] += dispersion.accumulation_roundoff[0];
    }
    let roundoff_ok = energy.accumulation_roundoff[0]
        <= config.tolerance * energy.value[0].abs().max(1.0)
        && (0..3).all(|a| roundoff[a] <= config.tolerance * dipole[a].abs().max(1.0));
    Ok(ResponseReport {
        value: MixedMolecularResponse {
            directions: directions.to_vec(), energy_derivative: energy.value[0], dipole_derivative: dipole,
            energy_accumulation_roundoff: energy.accumulation_roundoff[0], dipole_accumulation_roundoff: roundoff,
            directional_evaluations: cache.evaluations - initial_evaluations,
        },
        shape: vec![4], functional: prepared.model.clone(),
        reference: crate::response::diagnostics::reference_diagnostics(&prepared.reference.scf)?,
        units: "atomic units; unnormalized mixed derivatives of complete energy and its electric dipole",
        model: "static integer RKS/UKS; full parent/target orbital and noncanonical PT2 amplitude response; geometry-only dispersion",
        axis_convention: "energy scalar then Cartesian dipole vector; Bohr nuclear paths and atomic electric fields",
        converged: diagnostics.converged(config) && roundoff_ok, residuals, iterations: Vec::new(),
        warnings: if roundoff_ok { Vec::new() } else { vec!["mixed spectroscopic polarization roundoff exceeds the requested tolerance".into()] },
    })
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let direction = nuclear_direction(system, controls)?;
    let order = controls.usize("electric_order", 2)?;
    if !(1..=3).contains(&order) {
        return Err(DftError::InvalidInput(
            "electric_order must be one, two or three".into(),
        ));
    }
    super::correlated_high_order::collinear_reference(controls)?;
    if super::double_hybrid::requested(controls)? {
        let prepared = super::double_hybrid::prepare(system, controls, "static-spectroscopy")?;
        prepared.check_path_memory(order + 1)?;
        let config = prepared
            .config
            .as_ref()
            .expect("spectroscopic response configuration");
        let mut diagnostics = Diagnostics::default();
        // Reuse identical normalized scalar paths across ordered electric
        // slots; every tensor slot still evaluates its polarization sum.
        // Only scalar coefficients are retained, not the large AO paths.
        let mut cache = DirectionalCoefficients::default();
        let response = static_spectroscopic_from_mixed(
            prepared.setup.molecule.len(), &direction, order, config,
            "static integer RKS/UKS APT/Raman/hyper-Raman from complete post-SCF energy; full target orbital and noncanonical PT2 amplitude response; ordered electric slots",
            |directions| correlated_mixed(&prepared, system, directions, &mut diagnostics, &mut cache),
        )?;
        let mut report = super::electric::metadata(&prepared.setup, &prepared.reference, &response);
        append_tensor(&mut report, response);
        diagnostics.append(&mut report, config);
        super::double_hybrid::append_prepared_energy(prepared, report)
    } else {
        let config = super::vibrational::response_config(controls)?;
        let setup = super::molecular::setup(system, controls)?;
        controls.finish("static-spectroscopy")?;
        let reference = super::method::run_molecular_reference(&setup)?;
        let response = static_spectroscopic_derivative(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            &direction,
            order,
            &config,
        )?;
        let mut report = super::electric::metadata(&setup, &reference, &response);
        append_tensor(&mut report, response);
        Ok(report)
    }
}
