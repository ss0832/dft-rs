// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear derivatives of the generalized double-hybrid energy.

use crate::driver::double_hybrid::{append_spinor_energy, SpinorModel};
use crate::driver::method::MolecularSetup;
use crate::driver::nuclear_tensor::{assemble, dispersion_path, NuclearDiagnostics};
use crate::driver::{System, Value};
use crate::periodic::{spinor_scf::SpinorScfReport, BlochMatrix};
use crate::response::{nuclear::SpinorNuclearContext, ResponseSolverConfig};
use crate::Result;

pub(in crate::driver) fn append_spinor_gradient(
    report: &mut Value,
    system: &System,
    setup: &MolecularSetup,
    reference: &SpinorScfReport,
    operator: &BlochMatrix,
    model: &SpinorModel,
    config: &ResponseSolverConfig,
) -> Result<()> {
    let parent = SpinorNuclearContext::new(
        &setup.molecule,
        &setup.basis,
        setup.grid.as_ref(),
        &setup.functional,
        reference,
        &setup.options,
        operator,
        2,
        config,
    )?;
    let mut target_options = setup.options.clone();
    target_options.vv10 = model.target_vv10.map(|(_, parameters)| parameters);
    let target_scale = model.target_vv10.map_or(1.0, |(scale, _)| scale);
    let target = model
        .target
        .as_ref()
        .map(|functional| {
            SpinorNuclearContext::new(
                &setup.molecule,
                &setup.basis,
                setup.grid.as_ref(),
                functional,
                reference,
                &target_options,
                operator,
                1,
                config,
            )
        })
        .transpose()?;
    let mut diagnostics = NuclearDiagnostics::default();
    let mut pair_diagnostics = Vec::new();
    let names = [
        "gradient",
        "parent_gradient",
        "generalized_pt2_gradient",
        "pt2_gradient",
        "target_gradient",
        "dispersion_gradient",
    ];
    let tensors = assemble(setup.molecule.len(), 1, names.len(), |direction| {
        let (value, correction) = parent.correlated_gradient_path(
            target.as_ref().map(|target| (target, target_scale)),
            &direction,
            model.coefficient,
            model.memory_mb,
        )?;
        let pair: Vec<_> = value.pairs.energy.iter().map(|v| v.re).collect();
        let weighted = pair.iter().map(|v| model.coefficient * v).collect();
        let correction = correction.unwrap_or_else(|| vec![0.0; 2]);
        let dispersion =
            dispersion_path(setup, system, &direction, 1, model.dispersion_parameters)?;
        let total = value
            .energy
            .iter()
            .zip(&correction)
            .zip(&dispersion)
            .map(|((energy, target), dispersion)| energy + target + dispersion)
            .collect();
        let path = value.parent.parent;
        diagnostics.response.extend(path.residuals);
        diagnostics.projector.extend(path.projector_defects);
        diagnostics
            .projector
            .extend(&value.parent.particle_number_defects);
        diagnostics.projector.push(value.parent.frame_defect);
        diagnostics.stationarity.extend(path.stationarity_defects);
        diagnostics.reference.extend([
            path.reference_defect,
            path.energy[0] - reference.fields.energy,
        ]);
        diagnostics
            .angular_history
            .extend(value.parent.angular_history);
        pair_diagnostics.push(
            Value::object()
                .with("pair_residuals", value.pairs.residuals)
                .with("pair_relative_residuals", value.pairs.relative_residuals)
                .with(
                    "pair_imaginary_energy",
                    value.pairs.energy.iter().map(|v| v.im).collect::<Vec<_>>(),
                )
                .with(
                    "particle_number_defects",
                    value.parent.particle_number_defects,
                )
                .with("orbital_frame_defect", value.parent.frame_defect),
        );
        Ok(vec![
            total,
            path.energy,
            pair,
            weighted,
            correction,
            dispersion,
        ])
    })?;

    append_spinor_energy(report, model, tensors.energies[2], tensors.energies[4]);
    let mut labels = report.get("units").cloned().expect("spinor energy units");
    for (&name, gradient) in names.iter().zip(&tensors.gradients) {
        report.set(
            name,
            gradient
                .chunks_exact(3)
                .map(|row| row.to_vec())
                .collect::<Vec<_>>(),
        );
        labels.set(name, "hartree/bohr");
    }
    report.set(
        "forces",
        tensors.gradients[0]
            .chunks_exact(3)
            .map(|row| row.iter().map(|v| -v).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );
    report.set("nuclear_path_evaluations", tensors.paths);
    report.set("spinor_nuclear_pair_diagnostics", pair_diagnostics);
    report.set("provenance", "analytic");
    report.set("nuclear_model", "integer full-spin parent orbital response with moving AO overlap, integrals and grid; generalized PT2 pair response with a common coefficient, optional relaxed parent density in a distinct target with its own semilocal and VV10 terms, and analytic empirical dispersion");
    labels.set("forces", "hartree/bohr");
    for (name, unit) in [
        ("pair_residuals", "[hartree, hartree/bohr]"),
        ("pair_relative_residuals", "dimensionless"),
        ("pair_imaginary_energy", "[hartree, hartree/bohr]"),
        ("particle_number_defects", "[electrons, electrons/bohr]"),
        ("orbital_frame_defect", "atomic units by Taylor order"),
    ] {
        labels.set(&format!("spinor_nuclear_pair_diagnostics.{name}"), unit);
    }
    report.set("units", labels);
    diagnostics.append(report, config);
    Ok(())
}
