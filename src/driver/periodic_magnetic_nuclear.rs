// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! One nuclear response driver for native Rust, CLI, Python and ASE reports.
use super::molecular::units;
use super::{Controls, Value};
use crate::periodic::magnetic_cell::MagneticBlochBackend;
use crate::periodic::spinor_scf::SpinorScfReport;
use crate::response::ResponseSolverConfig;
use crate::{DftError, Result};

pub(super) struct Request {
    order: usize,
    direction: Option<Vec<f64>>,
    leading: Vec<Vec<f64>>,
    dimension: usize,
    config: ResponseSolverConfig,
}
impl Request {
    /// Use the common driver's root-qualified unit convention. Array orders
    /// remain symbolic, so every row is scaled with its own `order` value.
    pub(super) fn attach(parent: &mut Value, nuclear: Value) {
        let Value::Object(mut entries) = nuclear else {
            return;
        };
        let mut labels = match parent.get("units") {
            Some(Value::Object(v)) => v.clone(),
            _ => Vec::new(),
        };
        if let Some((_, Value::Object(units))) = entries.iter().find(|(key, _)| key == "units") {
            labels.extend(
                units
                    .iter()
                    .map(|(key, value)| (format!("nuclear_response.{key}"), value.clone())),
            );
        }
        entries.retain(|(key, _)| key != "units");
        if let Some((_, Value::Array(path))) =
            entries.iter_mut().find(|(key, _)| key == "energy_path")
        {
            for row in path {
                if let Value::Object(row) = row {
                    row.retain(|(key, _)| key != "units");
                }
            }
            for key in ["coefficient", "derivative"] {
                labels.push((
                    format!("nuclear_response.energy_path.{key}"),
                    Value::Str("hartree/bohr^order".into()),
                ));
            }
        }
        parent.set("units", Value::Object(labels));
        parent.set("nuclear_response", Value::Object(entries));
    }

    pub(super) fn parse(controls: &Controls, natoms: usize) -> Result<Option<Self>> {
        let order = controls.opt_usize("nuclear_order")?;
        let direction = controls.opt_f64_list("nuclear_direction")?;
        let Some(order) = order else {
            if direction.is_some() {
                return Err(DftError::InvalidInput(
                    "nuclear_direction requires nuclear_order".into(),
                ));
            }
            return Ok(None);
        };
        if !(1..=4).contains(&order) {
            return Err(DftError::InvalidInput(
                "nuclear_order must be one through four".into(),
            ));
        }
        let dimension = 3 * natoms;
        let probe = controls.opt_f64_list("probe")?;
        let second = controls.opt_f64_list("second_probe")?;
        if direction
            .iter()
            .chain(probe.iter())
            .chain(second.iter())
            .any(|d| d.len() != dimension || d.iter().any(|x| !x.is_finite()))
        {
            return Err(DftError::InvalidInput("nuclear directions/probes require one finite entry per atomic Cartesian coordinate".into()));
        }
        if direction.is_some() && (probe.is_some() || second.is_some())
            || order < 3 && (probe.is_some() || second.is_some())
            || order == 3 && second.is_some()
        {
            return Err(DftError::InvalidInput("nuclear_direction selects a scalar path; tensor probes apply only to cubic/quartic contractions".into()));
        }
        let mut leading = Vec::new();
        if direction.is_none() && order >= 3 {
            let probe=probe.ok_or_else(|| DftError::InvalidInput("cubic/quartic nuclear tensors require probe, or use nuclear_direction for a scalar Taylor path".into()))?;
            if order == 4 {
                leading.push(second.unwrap_or_else(|| probe.clone()));
            }
            leading.insert(0, probe);
        }
        Ok(Some(Self {
            order,
            direction,
            leading,
            dimension,
            config: super::vibrational::response_config(controls)?,
        }))
    }
    pub(super) fn run(
        self,
        backend: &MagneticBlochBackend<'_>,
        reference: &SpinorScfReport,
    ) -> Result<Value> {
        let mut cache: Vec<(Vec<f64>, usize, Vec<f64>)> = Vec::new();
        let mut residuals = Vec::new();
        let mut projector = Vec::new();
        let mut stationarity = Vec::new();
        let mut defect = 0.0_f64;
        let mut energy_defect = 0.0_f64;
        let mut solved_order = 0;
        let mut run = |direction: &[f64], order: usize| -> Result<Vec<f64>> {
            if let Some((_, _, energy)) =
                cache.iter().find(|(d, n, _)| d == direction && *n >= order)
            {
                return Ok(vec![energy[order]]);
            }
            let d: Vec<_> = direction
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect();
            let result = backend.nuclear_energy_taylor(reference, &d, order, &self.config)?;
            solved_order = solved_order.max(result.solved_response_order);
            residuals.extend(result.response_residuals);
            projector.extend(result.projector_defects);
            stationarity.extend(result.stationarity_defects);
            defect = defect.max(result.reference_defect);
            energy_defect = energy_defect.max(result.reference_energy_defect);
            let coefficient = result.energy[order];
            cache.push((direction.to_vec(), order, result.energy));
            Ok(vec![coefficient])
        };
        let mut output=Value::object().with("derivative_order",self.order)
            .with("model","fixed-cell finite-Gamma continuum magnetic bulk/slab; point or Gaussian nuclei, fixed explicit Gaussian compensation, integer spinor occupations, relaxed full/range exchange and generated nuclear SOC; optional 3D fixed Cartesian Berry electric bias; magnetic-BZ/finite-q nuclear response requires intersector providers");
        let mut unit_pairs = vec![
            ("response_residuals", "dimensionless"),
            ("projector_defects", "dimensionless"),
            ("stationarity_defects", "hartree"),
            ("reference_defect", "hartree"),
            ("reference_energy_defect", "hartree"),
            ("direction", "dimensionless"),
            ("probe", "dimensionless"),
            ("second_probe", "dimensionless"),
        ];
        let mut roundoff = 0.0;
        if let Some(direction) = &self.direction {
            run(direction, self.order)?;
            let energy = &cache[0].2;
            let path: Vec<_> = (0..=self.order)
                .map(|n| {
                    let unit = match n {
                        0 => "hartree",
                        1 => "hartree/bohr",
                        2 => "hartree/bohr^2",
                        3 => "hartree/bohr^3",
                        _ => "hartree/bohr^4",
                    };
                    Value::object()
                        .with("order", n)
                        .with("coefficient", energy[n])
                        .with("derivative", energy[n] * (1..=n).product::<usize>() as f64)
                        .with(
                            "units",
                            units(&[("coefficient", unit), ("derivative", unit)]),
                        )
                })
                .collect();
            output.set("energy_path", path);
            output.set("direction", direction.clone());
        } else if self.order == 1 {
            let mut gradient = Vec::with_capacity(self.dimension);
            for axis in 0..self.dimension {
                let mut direction = vec![0.0; self.dimension];
                direction[axis] = 1.0;
                gradient.push(run(&direction, 1)?[0]);
            }
            output.set(
                "gradient",
                gradient
                    .chunks_exact(3)
                    .map(|v| v.to_vec())
                    .collect::<Vec<_>>(),
            );
            unit_pairs.push(("gradient", "hartree/bohr"));
        } else {
            let (matrix, error) =
                crate::response::mixed::polarized_matrix(self.dimension, &self.leading, &mut run)?;
            roundoff = error;
            let key = if self.order == 2 {
                "hessian"
            } else {
                "contracted_force_constants"
            };
            output.set(key, matrix);
            let unit = match self.order {
                2 => "hartree/bohr^2",
                3 => "hartree/bohr^3",
                _ => "hartree/bohr^4",
            };
            unit_pairs.push((key, unit));
            unit_pairs.push(("accumulation_roundoff", unit));
            output.set(
                "symmetry_convention",
                "intrinsic permutation reuse; independent ordered asymmetry not measured",
            );
            output.set(
                "axis_convention",
                "3*atom+axis, axis=x,y,z; probes contract the leading nuclear indices",
            );
            if let Some(probe) = self.leading.first() {
                output.set("probe", probe.clone());
            }
            if let Some(second) = self.leading.get(1) {
                output.set("second_probe", second.clone());
            }
        }
        output.set(
            "converged",
            residuals.iter().all(|r| *r <= self.config.acceptance()),
        );
        output.set("response_residuals", residuals);
        output.set("projector_defects", projector);
        output.set("stationarity_defects", stationarity);
        output.set("reference_defect", defect);
        output.set("reference_energy_defect", energy_defect);
        output.set("path_evaluations", cache.len());
        output.set("solved_response_order", solved_order);
        if self.direction.is_none() && self.order >= 2 {
            output.set("accumulation_roundoff", roundoff);
        }
        output.set("units", units(&unit_pairs));
        Ok(output)
    }
}
