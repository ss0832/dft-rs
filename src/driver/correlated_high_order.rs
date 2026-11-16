// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! High-order nuclear/electric exports of the shared orbital/PT2 paths.
use super::double_hybrid::Prepared;
use super::{Controls, System, Value};
use crate::integrals::london::FieldPath;
use crate::response::correlated_field::{correlated_field_taylor_for_model, CorrelatedFieldTaylor};
use crate::response::mixed::{polarized_matrices, DirectionalCoefficients as Coefficients};
use crate::response::nuclear::{MolecularCorrelatedTaylor, MolecularDirection};
use crate::response::ResponseSolverConfig;
use crate::{DftError, Result};

#[derive(Default)]
pub(in crate::driver) struct Diagnostics {
    response: Vec<f64>,
    amplitude: Vec<f64>,
    amplitude_relative: Vec<f64>,
    projector: Vec<f64>,
    stationarity: Vec<f64>,
    metric: Vec<f64>,
    number: Vec<f64>,
    frame: Vec<f64>,
    reference: Vec<f64>,
    imaginary: Vec<f64>,
}
fn maximum(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0, f64::max)
}
impl Diagnostics {
    fn field(&mut self, value: &CorrelatedFieldTaylor) {
        let parent = &value.parent.parent;
        self.response.extend(&parent.residuals);
        self.amplitude.extend(&value.amplitude_residuals);
        self.amplitude_relative
            .extend(&value.amplitude_relative_residuals);
        self.projector.extend(&parent.projector_defects);
        self.metric.extend(&value.parent.orbital_metric_defects);
        self.stationarity.extend(&parent.stationarity_defects);
        self.number
            .extend(value.parent.particle_number_coefficients.iter().flatten());
        self.frame.push(value.parent.orbital_frame_defect);
        self.reference.push(parent.reference_defect);
        self.imaginary.push(value.imaginary_energy_defect);
    }
    pub(in crate::driver) fn spinor_field(
        &mut self,
        value: &crate::response::nuclear::SpinorCorrelatedPath,
    ) {
        let parent = &value.parent.parent;
        self.response.extend(&parent.residuals);
        self.amplitude.extend(&value.pairs.residuals);
        self.amplitude_relative
            .extend(&value.pairs.relative_residuals);
        self.projector.extend(&parent.projector_defects);
        self.stationarity.extend(&parent.stationarity_defects);
        self.number.extend(&value.parent.particle_number_defects);
        self.frame.push(value.parent.frame_defect);
        self.reference.push(parent.reference_defect);
        self.imaginary.push(value.imaginary_energy_defect);
    }
    pub(super) fn nuclear(&mut self, value: &MolecularCorrelatedTaylor) {
        self.response.extend(&value.response_residuals);
        self.amplitude.extend(&value.amplitude_residuals);
        self.amplitude_relative
            .extend(&value.amplitude_relative_residuals);
        self.projector.extend(&value.projector_defects);
        self.stationarity.extend(&value.stationarity_defects);
        self.metric.extend(&value.metric_defects);
        self.number.extend(value.number_defects.iter().flatten());
        self.frame.push(value.orbital_frame_defect);
        self.reference
            .extend([value.reference_energy_defect, value.reference_fock_defect]);
    }
    pub(super) fn converged(&self, config: &ResponseSolverConfig) -> bool {
        [
            &self.response,
            &self.amplitude_relative,
            &self.projector,
            &self.stationarity,
            &self.metric,
            &self.number,
            &self.frame,
            &self.reference,
            &self.imaginary,
        ]
        .into_iter()
        .flatten()
        .all(|v| v.is_finite() && v.abs() <= config.acceptance())
            && self.amplitude.iter().all(|v| v.is_finite())
    }
    pub(in crate::driver) fn append(self, report: &mut Value, config: &ResponseSolverConfig) {
        let converged = self.converged(config);
        if !converged {
            let mut failed = Vec::new();
            for (name, values) in [
                ("normalized orbital response", &self.response),
                ("normalized amplitude response", &self.amplitude_relative),
                ("projector", &self.projector),
                ("stationarity", &self.stationarity),
                ("metric", &self.metric),
                ("particle number", &self.number),
                ("orbital frame", &self.frame),
                ("reference", &self.reference),
                ("imaginary energy", &self.imaginary),
            ] {
                if values.iter().any(|v| !v.is_finite()) {
                    failed.push(format!("{name} (nonfinite)"));
                } else if maximum(values) > config.acceptance() {
                    failed.push(format!("{name} (maximum {:.6e})", maximum(values)));
                }
            }
            if self.amplitude.iter().any(|v| !v.is_finite()) {
                failed.push("nonfinite amplitude residual".into());
            }
            let mut warnings = report
                .get("warnings")
                .and_then(Value::as_array)
                .unwrap_or(&[])
                .to_vec();
            warnings.push(Value::Str(format!(
                "correlated response constraints exceed the requested acceptance {:.6e}: {}",
                config.acceptance(),
                failed.join(", ")
            )));
            report.set("warnings", Value::Array(warnings));
        }
        report.set("response_converged", converged);
        report.set(
            "converged",
            converged
                && report
                    .get("converged")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
        );
        let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
        for (key, values, unit) in [
            ("response_residuals", self.response, "dimensionless"),
            ("amplitude_response_residuals", self.amplitude, "hartree"),
            (
                "amplitude_relative_residuals",
                self.amplitude_relative,
                "dimensionless",
            ),
            ("projector_defects", self.projector, "dimensionless"),
            ("nuclear_stationarity_defects", self.stationarity, "hartree"),
            ("metric_defects", self.metric, "dimensionless"),
            ("response_number_defects", self.number, "electrons"),
            ("orbital_frame_defects", self.frame, "dimensionless"),
            ("reference_defects", self.reference, "hartree"),
            ("imaginary_energy_defects", self.imaginary, "hartree"),
        ] {
            report.set(key, values);
            labels.set(key, unit);
        }
        report.set("units", labels);
    }
}

pub(super) fn collinear_reference(controls: &Controls) -> Result<()> {
    if controls.opt_str("reference")?.is_some_and(|r| {
        matches!(
            r.to_ascii_lowercase().as_str(),
            "rohf" | "roks" | "ghf" | "gks"
        )
    }) {
        return Err(DftError::Unsupported {
            feature:
                "correlated high-order response on an open-shell restricted or spinor reference"
                    .into(),
            missing: "the corresponding reference-constrained correlated orbital-response provider"
                .into(),
        });
    }
    Ok(())
}

pub(super) fn nuclear(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    collinear_reference(controls)?;
    let quartic = id == "quartic-fc-along";
    let probe = controls
        .opt_f64_list("probe")?
        .ok_or_else(|| DftError::InvalidInput("contracted force constants require probe".into()))?;
    let second = if quartic {
        controls
            .opt_f64_list("second_probe")?
            .unwrap_or_else(|| probe.clone())
    } else {
        Vec::new()
    };
    let prepared = super::double_hybrid::prepare(system, controls, id)?;
    let dimension = 3 * prepared.setup.molecule.len();
    if probe.len() != dimension || quartic && second.len() != dimension {
        return Err(DftError::InvalidInput(
            "force-constant patterns must have 3*atoms Cartesian entries".into(),
        ));
    }
    let config = prepared
        .config
        .as_ref()
        .expect("high-order response configuration");
    let mut leading = vec![probe.clone()];
    if quartic {
        leading.push(second.clone());
    }
    let order = leading.len() + 2;
    prepared.check_path_memory(order)?;
    let mut diagnostics = Diagnostics::default();
    let mut cache = Coefficients::default();
    let (tensors, roundoff) = polarized_matrices(dimension, &leading, |direction, n| {
        cache.get(direction, n, NUCLEAR_PREFIXES.len(), |d, n| {
            nuclear_coefficients(&prepared, system, d, n, &mut diagnostics)
        })
    })?;
    let converged = diagnostics.converged(config);
    let mut report=super::molecular::reference_report(&prepared.setup,&prepared.reference,
        "analytic contracted nuclear energy derivatives; complete parent/target orbital and noncanonical PT2 amplitude response; exact polarization with intrinsic matrix interchange reuse; unscaled SS/OS channels");
    let unit = if quartic {
        "hartree/bohr^4"
    } else {
        "hartree/bohr^3"
    };
    let mut labels = super::molecular::units(&[
        ("energy", "hartree"),
        ("probe", "dimensionless"),
        ("second_probe", "dimensionless"),
        ("accumulation_roundoff", unit),
    ]);
    for (c, prefix) in NUCLEAR_PREFIXES.iter().enumerate() {
        let key = format!("{prefix}contracted_force_constants");
        report.set(&key, &tensors[c]);
        labels.set(&key, unit);
    }
    report.set("units", labels);
    report.set("probe", probe);
    if quartic {
        report.set("second_probe", second);
    }
    report.set("response_order", order);
    report.set("shape", vec![dimension, dimension]);
    report.set("accumulation_roundoff", roundoff[0]);
    report.set("path_evaluations", cache.evaluations);
    report.set("axis_convention","remaining Cartesian indices 3*atom+axis; leading indices contracted with the supplied displacement patterns; actual derivatives without factorial scaling");
    report.set("provenance", "analytic");
    report.set("response_converged", converged);
    diagnostics.append(&mut report, config);
    super::double_hybrid::append_prepared_energy(prepared, report)
}

fn electric_coefficients(
    prepared: &Prepared,
    d: &[f64],
    order: usize,
    diagnostics: &mut Diagnostics,
) -> Result<Vec<f64>> {
    let s = &prepared.setup;
    let value = correlated_field_taylor_for_model(
        &s.molecule,
        &s.basis,
        &s.functional,
        prepared.field_target()?,
        s.grid.as_ref(),
        &prepared.reference.scf,
        &s.options,
        prepared.coefficients,
        &FieldPath {
            electric: [d[0], d[1], d[2]],
            magnetic: [0.0; 3],
            origin: [0.0; 3],
            london: false,
            order,
        },
        None,
        prepared.pt2_memory_mb,
        prepared
            .config
            .as_ref()
            .expect("electric response configuration"),
    )?;
    diagnostics.field(&value);
    // mu=-dE/dF: every static electric tensor has this common minus sign.
    Ok(value.energy_channels(order).iter().map(|v| -v).collect())
}

pub(super) fn require_electric_assembly(id: &str, controls: &Controls) -> Result<()> {
    if controls.contains("tensor_assembly") && controls.contains("electric_assembly") {
        return Err(DftError::InvalidInput(
            "use either tensor_assembly or electric_assembly".into(),
        ));
    }
    let assembly = if controls.contains("electric_assembly") {
        controls.str("electric_assembly", "projector")?
    } else {
        controls.str("tensor_assembly", "projector")?
    };
    if !matches!(assembly.as_str(), "projector" | "energy" | "variational") {
        return Err(DftError::InvalidInput(
            "unknown tensor_assembly; recognized selectors are projector, energy and variational"
                .into(),
        ));
    }
    if assembly == "variational" {
        return Err(DftError::Unsupported {
            feature: format!("correlated {id} with variational assembly"),
            missing: "the correlated Lagrangian assembly; parent-only 2n+1 assembly cannot replace it, use projector or energy".into(),
        });
    }
    Ok(())
}

pub(in crate::driver) struct ElectricCoefficientsTensor {
    pub tensors: Vec<Vec<f64>>,
    pub roundoff: Vec<Vec<f64>>,
    pub evaluations: usize,
}

/// The same scalar Taylor polarization assembles real and complex-reference
/// energies. Each channel receives the same field perturbation and all
/// ordered Cartesian entries follow from the common Taylor normalization.
pub(in crate::driver) fn electric_coefficients_tensor(
    rank: usize,
    channels: usize,
    mut coefficients: impl FnMut(&[f64], usize) -> Result<Vec<f64>>,
) -> Result<ElectricCoefficientsTensor> {
    let size = 3usize.pow(rank as u32);
    let mut tensors = vec![vec![0.0; size]; channels];
    let mut roundoff = vec![vec![0.0; size]; channels];
    let mut cache = Coefficients::default();
    for flat in 0..3usize.pow((rank - 2) as u32) {
        let mut axes = vec![0; rank - 2];
        let mut index = flat;
        for axis in axes.iter_mut().rev() {
            *axis = index % 3;
            index /= 3;
        }
        let leading: Vec<_> = axes
            .iter()
            .map(|&a| {
                let mut d = vec![0.0; 3];
                d[a] = 1.0;
                d
            })
            .collect();
        let (matrices, error) = polarized_matrices(3, &leading, |d, n| {
            cache.get(d, n, channels, &mut coefficients)
        })?;
        for c in 0..channels {
            for a in 0..3 {
                for b in 0..3 {
                    let index = (flat * 3 + a) * 3 + b;
                    tensors[c][index] = matrices[c][a][b];
                    roundoff[c][index] = error[c];
                }
            }
        }
    }
    Ok(ElectricCoefficientsTensor {
        tensors,
        roundoff,
        evaluations: cache.evaluations,
    })
}

pub(super) fn electric(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    collinear_reference(controls)?;
    require_electric_assembly(id, controls)?;
    let prepared = super::double_hybrid::prepare(system, controls, id)?;
    let config = prepared
        .config
        .as_ref()
        .expect("electric response configuration");
    let rank = if id == "beta" { 3 } else { 4 };
    prepared.check_path_memory(rank)?;
    let mut diagnostics = Diagnostics::default();
    let assembled = electric_coefficients_tensor(rank, 9, |direction, order| {
        electric_coefficients(&prepared, direction, order, &mut diagnostics)
    })?;
    let ElectricCoefficientsTensor {
        tensors,
        roundoff,
        evaluations,
    } = assembled;
    let response=crate::response::electric_tensor::complete_tensor(rank-1,tensors[0].clone(),roundoff[0].clone(),
        3usize.pow((rank-1)as u32),evaluations,maximum(&diagnostics.projector),maximum(&diagnostics.stationarity),
        maximum(&diagnostics.number),None,diagnostics.response.clone(),Vec::new(),Vec::new(),diagnostics.converged(config),
        prepared.target.as_ref().unwrap_or(&prepared.setup.functional),&prepared.reference.scf,config,
        "static clamped-nuclei energy derivatives; full parent/target orbital and noncanonical PT2 amplitude response; unregularized SS/OS; scalar-energy polarization with permutation reuse; integer RKS/UKS")?;
    let mut report = super::electric::tensor(
        &prepared.setup,
        &prepared.reference,
        response,
        "energy-polarization",
    );
    let unit = if rank == 3 {
        "e^3*bohr^3/hartree^2"
    } else {
        "e^4*bohr^4/hartree^3"
    };
    let mut labels = report.get("units").cloned().expect("electric tensor units");
    for (c, prefix) in [
        "",
        "parent_",
        "target_",
        "pt2_",
        "pt2_same_spin_alpha_",
        "pt2_opposite_spin_",
        "pt2_same_spin_beta_",
        "parent_nonlocal_",
        "target_nonlocal_",
    ]
    .iter()
    .enumerate()
    {
        let key = format!("{prefix}tensor");
        report.set(&key, &tensors[c]);
        labels.set(&key, unit);
    }
    // Energy polarization enforces interchange. It is not an independent
    // observed-dipole-leg Maxwell measurement, so do not report a fake zero.
    for key in ["intrinsic_permutation_defect", "energy_permutation_defect"] {
        report.set(key, Value::Null);
    }
    for key in [
        "intrinsic_permutation_comparisons",
        "energy_permutation_comparisons",
    ] {
        report.set(key, 0usize);
    }
    report.set(
        "interchange_diagnostics",
        "scalar-energy permutation reuse; independent finite-field acceptance required",
    );
    labels.set("interchange_diagnostics", "text");
    report.set("axis_convention","row-major dipole Cartesian axis then field Cartesian axes; actual derivatives from scalar energy; intrinsic permutation reuse; observed-leg interchange is unmeasured");
    report.set("units", labels);
    diagnostics.append(&mut report, config);
    super::double_hybrid::append_prepared_energy(prepared, report)
}

use super::correlated_nuclear::ENERGY_PREFIXES as NUCLEAR_PREFIXES;

fn nuclear_coefficients(
    prepared: &Prepared,
    system: &System,
    d: &[f64],
    order: usize,
    diagnostics: &mut Diagnostics,
) -> Result<Vec<f64>> {
    let nuclear: Vec<_> = d.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
    let value = super::correlated_nuclear::nuclear_path(
        prepared,
        &MolecularDirection::nuclear(nuclear.clone()),
        order,
    )?;
    diagnostics.nuclear(&value);
    let dispersion =
        super::correlated_nuclear::dispersion_path(prepared, system, &nuclear, order)?[order];
    Ok(vec![
        value.energy[order] + dispersion,
        value.parent[order],
        value.target[order],
        value.correlation[order],
        value.same_spin_alpha[order],
        value.opposite_spin[order],
        value.same_spin_beta[order],
        dispersion,
        value.parent_nonlocal[order],
        value.target_nonlocal[order],
    ])
}

pub(super) fn force_constants(
    system: &System,
    controls: &Controls,
    selection: &super::force_constants::Selection,
) -> Result<Value> {
    collinear_reference(controls)?;
    let prepared = super::double_hybrid::prepare(system, controls, "force-constants")?;
    selection.check_report_storage(NUCLEAR_PREFIXES.len(), prepared.setup.options.incore_mb)?;
    prepared.check_path_memory(selection.order)?;
    let config = prepared
        .config
        .as_ref()
        .expect("force-constant response configuration");
    let mut diagnostics = Diagnostics::default();
    let (tensors, evaluations) = super::force_constants::assemble_energy_paths(
        selection,
        NUCLEAR_PREFIXES.len(),
        prepared.setup.options.incore_mb,
        |d, n| nuclear_coefficients(&prepared, system, d, n, &mut diagnostics),
    )?;
    let mut report = super::molecular::reference_report(&prepared.setup, &prepared.reference,
        "complete static nuclear tensors; analytic moving basis/grid; full parent/target orbital and noncanonical PT2 amplitude response; scalar-energy polarization with explicit intrinsic permutation reuse");
    report.set(
        "reference_field",
        prepared
            .setup
            .options
            .external_field
            .map_or([0.0; 3], |v| v.to_array()),
    );
    diagnostics.append(&mut report, config);
    super::force_constants::append_energy_tensors(
        &mut report,
        selection,
        tensors,
        evaluations,
        &NUCLEAR_PREFIXES,
        config,
    )?;
    let mut labels = report.get("units").cloned().expect("nuclear tensor units");
    labels.set("reference_field", "hartree/e/bohr");
    report.set("units", labels);
    super::double_hybrid::append_prepared_energy(prepared, report)
}
