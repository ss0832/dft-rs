// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete molecular double-hybrid nuclear tensors on the common driver.
use super::{Controls, System, Value};
use crate::response::nuclear::{correlated_molecular_taylor_for_model, MolecularDirection};
use crate::Result;

mod spinor;
pub(super) use spinor::append_spinor_gradient;

pub(super) const ENERGY_PREFIXES: [&str; 10] = [
    "",
    "parent_",
    "target_",
    "pt2_",
    "pt2_same_spin_alpha_",
    "pt2_opposite_spin_",
    "pt2_same_spin_beta_",
    "dispersion_",
    "parent_nonlocal_",
    "target_nonlocal_",
];

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let prepared = super::double_hybrid::prepare(system, controls, id)?;
    let setup = &prepared.setup;
    let order = if id == "hessian" { 2 } else { 1 };
    let mut residuals = Vec::new();
    let mut amplitudes = Vec::new();
    let mut projector = Vec::new();
    let mut stationarity = Vec::new();
    let mut metric = Vec::new();
    let mut numbers = Vec::new();
    let mut frames = Vec::new();
    let mut energy_defects = Vec::new();
    let mut fock_defects = Vec::new();
    let mut coefficient_equations = Some(Vec::new());
    let config = prepared
        .config
        .as_ref()
        .expect("nuclear response configuration");
    let tensors = super::nuclear_tensor::assemble(
        setup.molecule.len(),
        order,
        ENERGY_PREFIXES.len(),
        |direction| {
            let response = nuclear_path(
                &prepared,
                &MolecularDirection {
                    nuclear: direction.clone(),
                    electric: [0.0; 3],
                },
                order,
            )?;
            residuals.extend(response.response_residuals);
            amplitudes.extend(response.amplitude_residuals);
            projector.extend(response.projector_defects);
            stationarity.extend(response.stationarity_defects);
            metric.extend(response.metric_defects);
            numbers.extend(response.number_defects);
            frames.push(response.orbital_frame_defect);
            energy_defects.push(response.reference_energy_defect);
            fock_defects.push(response.reference_fock_defect);
            match (&mut coefficient_equations, response.coefficient_equations) {
                (Some(all), Some(values)) => all.extend(values),
                _ => coefficient_equations = None,
            }
            let dispersion = dispersion_path(&prepared, system, &direction, order)?;
            let total = response
                .energy
                .iter()
                .zip(&dispersion)
                .map(|(e, d)| e + d)
                .collect();
            Ok(vec![
                total,
                response.parent,
                response.target,
                response.correlation,
                response.same_spin_alpha,
                response.opposite_spin,
                response.same_spin_beta,
                dispersion,
                response.parent_nonlocal,
                response.target_nonlocal,
            ])
        },
    )?;
    let tolerance = config.acceptance();
    let nuclear_converged = residuals
        .iter()
        .chain(&amplitudes)
        .chain(&projector)
        .chain(&stationarity)
        .chain(&metric)
        .chain(&frames)
        .chain(&energy_defects)
        .chain(&fock_defects)
        .chain(numbers.iter().flatten())
        .all(|v| v.abs() <= tolerance);
    let mut report = super::double_hybrid::calculate_prepared(prepared, "energy")?;
    if !nuclear_converged {
        report.set("converged", false);
        report.set("warnings",vec!["correlated nuclear response or reference coefficient diagnostics exceed the requested tolerance; inspect all channel defects and tighten reference convergence"]);
    }
    if coefficient_equations.as_ref().is_some_and(|values| {
        values
            .iter()
            .any(|d| d.reconstructed_relative_residual > tolerance)
    }) {
        let mut warnings = report
            .get("warnings")
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |values| values.to_vec());
        warnings.push(Value::Str("a rebuilt coefficient divided by its initial source exceeds the linear-solve tolerance; inspect coefficient_equations for the unfloored source and reconstruction norms. Linear operator residuals and absolute physical coefficient defects are separate diagnostics".into()));
        report.set("warnings", warnings);
    }
    report.set("model", "analytic molecular double-hybrid nuclear tensors; complete moving-basis/grid parent orbital response and noncanonical PT2 amplitude response; full Coulomb without denominator regularization; SS/OS channels unscaled except their explicit weighted sum; integer collinear reference");
    let mut labels = report.get("units").cloned().expect("energy units");
    for (channel, prefix) in ENERGY_PREFIXES.into_iter().enumerate() {
        let name = format!("{prefix}nuclear_energy");
        report.set(&name, tensors.energies[channel]);
        labels.set(&name, "hartree");
        let name = format!("{prefix}gradient");
        report.set(
            &name,
            tensors.gradients[channel]
                .chunks_exact(3)
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        );
        labels.set(&name, "hartree/bohr");
        if order == 2 {
            let name = format!("{prefix}hessian");
            report.set(&name, &tensors.hessians[channel]);
            labels.set(&name, "hartree/bohr^2");
        }
    }
    report.set(
        "forces",
        tensors.gradients[0]
            .chunks_exact(3)
            .map(|v| v.iter().map(|x| -*x).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );
    report.set("response_residuals", residuals);
    report.set("amplitude_response_residuals", amplitudes);
    report.set("projector_defects", projector);
    report.set("nuclear_stationarity_defects", stationarity);
    report.set("metric_defects", metric);
    report.set("response_number_defects", numbers);
    report.set("orbital_frame_defects", frames);
    report.set("reference_energy_defects", energy_defects);
    report.set("reference_fock_defects", fock_defects);
    report.set(
        "coefficient_equations",
        coefficient_equations.map(|values| {
            values
                .iter()
                .map(|d| {
                    Value::object()
                        .with("source_norm", d.source_norm)
                        .with("reconstructed_norm", d.reconstructed_norm)
                        .with(
                            "reconstructed_relative_residual",
                            d.reconstructed_relative_residual,
                        )
                        .with("linear_relative_residual", d.linear_relative_residual)
                        .with(
                            "inner_linear_relative_residual",
                            d.inner_linear_relative_residual,
                        )
                })
                .collect::<Vec<_>>()
        }),
    );
    report.set("residual_convention", "response_residuals are fresh relative residuals of the returned coefficients in their original linear orbital equations; coefficient_equations separately retains inner correction solve residuals and reconstructed physical source/norm/ratio without a normalization floor; projector, metric, number and stationarity defects remain independent absolute acceptance checks");
    report.set("nuclear_path_evaluations", tensors.paths);
    report.set("provenance", "analytic");
    report.set(
        "axis_convention",
        "gradient [atom][xyz], Hessian [3*atom+axis][3*atom+axis]; all Cartesian elements returned",
    );
    for (key, unit) in [
        ("forces", "hartree/bohr"),
        ("response_residuals", "dimensionless"),
        ("amplitude_response_residuals", "hartree"),
        ("projector_defects", "dimensionless"),
        ("nuclear_stationarity_defects", "hartree"),
        ("metric_defects", "dimensionless"),
        ("response_number_defects", "electrons"),
        ("orbital_frame_defects", "dimensionless"),
        ("reference_energy_defects", "hartree"),
        ("reference_fock_defects", "hartree"),
        ("coefficient_equations.source_norm", "hartree"),
        ("coefficient_equations.reconstructed_norm", "hartree"),
        (
            "coefficient_equations.reconstructed_relative_residual",
            "dimensionless",
        ),
        (
            "coefficient_equations.linear_relative_residual",
            "dimensionless",
        ),
        (
            "coefficient_equations.inner_linear_relative_residual",
            "dimensionless",
        ),
    ] {
        labels.set(key, unit);
    }
    report.set("units", labels);
    Ok(report)
}

pub(super) fn nuclear_path(
    prepared: &super::double_hybrid::Prepared,
    direction: &MolecularDirection,
    order: usize,
) -> Result<crate::response::nuclear::MolecularCorrelatedTaylor> {
    let setup = &prepared.setup;
    let target = if prepared.target.is_some() || prepared.target_vv10.is_some() {
        let mut model = crate::response::field_path::FieldEnergyModel::new(
            prepared.target.as_ref().unwrap_or(&setup.functional),
        );
        if let Some((coefficient, parameters)) = prepared.target_vv10 {
            model = model.with_vv10(coefficient, parameters)?;
        }
        Some(model)
    } else {
        None
    };
    correlated_molecular_taylor_for_model(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        target,
        setup.grid.as_ref(),
        &prepared.reference.scf,
        &setup.options,
        prepared.coefficients,
        direction,
        order,
        prepared
            .config
            .as_ref()
            .expect("correlated response configuration"),
    )
}

pub(super) fn dispersion_path(
    prepared: &super::double_hybrid::Prepared,
    system: &System,
    direction: &[[f64; 3]],
    order: usize,
) -> Result<Vec<f64>> {
    super::nuclear_tensor::dispersion_path(
        &prepared.setup,
        system,
        direction,
        order,
        prepared.model_dispersion,
    )
}

/// Complete total-energy harmonic force field and mixed electric/nuclear
/// atomic polar tensors, on the same generic post-SCF energy provider.
pub(super) struct CorrelatedVibrational {
    pub(super) hessian: crate::linalg::Matrix,
    pub(super) polar: crate::linalg::Matrix,
    pub(super) residuals: Vec<f64>,
    pub(super) converged: bool,
}

pub(super) fn vibrational(
    prepared: &super::double_hybrid::Prepared,
    system: &System,
) -> Result<CorrelatedVibrational> {
    let natoms = prepared.setup.molecule.len();
    let dim = 3 * natoms;
    let mut residuals = Vec::new();
    let mut evaluate = |direction: &MolecularDirection| -> Result<Vec<f64>> {
        let value = nuclear_path(prepared, direction, 2)?;
        residuals.extend(&value.response_residuals);
        residuals.extend(&value.amplitude_residuals);
        residuals.extend(&value.projector_defects);
        residuals.extend(&value.stationarity_defects);
        residuals.extend(&value.metric_defects);
        residuals.extend(value.number_defects.iter().flatten());
        residuals.extend([
            value.orbital_frame_defect,
            value.reference_energy_defect,
            value.reference_fock_defect,
        ]);
        Ok(value.energy)
    };
    let tensors = super::nuclear_tensor::assemble(natoms, 2, 1, |nuclear| {
        Ok(vec![evaluate(&MolecularDirection::nuclear(nuclear))?])
    })?;
    let mut polar = crate::linalg::Matrix::zeros(3, dim);
    for b in 0..3 {
        let mut electric = [0.0; 3];
        electric[b] = 1.0;
        let pure_field = evaluate(&MolecularDirection {
            nuclear: vec![[0.0; 3]; natoms],
            electric,
        })?[2];
        for x in 0..dim {
            let mut nuclear = vec![[0.0; 3]; natoms];
            nuclear[x / 3][x % 3] = 1.0;
            let mixed = evaluate(&MolecularDirection { nuclear, electric })?[2];
            polar[(b, x)] = -(mixed - 0.5 * tensors.hessians[0][(x, x)] - pure_field);
        }
    }
    let dispersion = super::nuclear_tensor::assemble(natoms, 2, 1, |direction| {
        Ok(vec![dispersion_path(prepared, system, &direction, 2)?])
    })?;
    let mut hessian = tensors
        .hessians
        .into_iter()
        .next()
        .expect("one energy channel");
    for (v, d) in hessian
        .as_mut_slice()
        .iter_mut()
        .zip(dispersion.hessians[0].as_slice())
    {
        *v += d;
    }
    let tolerance = prepared
        .config
        .as_ref()
        .expect("response configuration")
        .acceptance();
    let converged = residuals
        .iter()
        .all(|r| r.is_finite() && r.abs() <= tolerance);
    Ok(CorrelatedVibrational {
        hessian,
        polar,
        residuals,
        converged,
    })
}
