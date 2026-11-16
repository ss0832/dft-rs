// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common complete Cartesian tensor assembly from analytic energy paths.
use crate::linalg::Matrix;
use crate::{DftError, Result};

/// Reference-independent raw diagnostics of stationary nuclear energy paths.
#[derive(Default)]
pub(super) struct NuclearDiagnostics {
    pub response: Vec<f64>,
    pub projector: Vec<f64>,
    pub stationarity: Vec<f64>,
    pub reference: Vec<f64>,
    pub angular_history: Vec<[f64; 3]>,
    pub cache_bounds: Vec<[f64; 3]>,
    pub thermal_stationarity: Vec<f64>,
    pub thermal_number: Vec<f64>,
    pub thermal_chemical: Vec<Vec<f64>>,
}

impl NuclearDiagnostics {
    pub fn append(self, report: &mut super::Value, config: &crate::response::ResponseSolverConfig) {
        let converged = [
            &self.response,
            &self.projector,
            &self.stationarity,
            &self.thermal_stationarity,
            &self.thermal_number,
            &self.reference,
        ]
        .into_iter()
        .flatten()
        .all(|v| v.is_finite() && v.abs() <= config.acceptance());
        let converged = converged
            && report
                .get("response_converged")
                .and_then(super::Value::as_bool)
                .unwrap_or(true);
        report.set("response_converged", converged);
        if !converged {
            report.set("converged", false);
            let mut warnings = report
                .get("warnings")
                .and_then(super::Value::as_array)
                .unwrap_or(&[])
                .to_vec();
            warnings.push(super::Value::Str("nuclear response or reference coefficient diagnostics exceed the requested tolerance; inspect raw defects and tighten reference convergence".into()));
            report.set("warnings", super::Value::Array(warnings));
        }
        let mut labels = report
            .get("units")
            .cloned()
            .unwrap_or_else(super::Value::object);
        for (key, values, unit) in [
            ("response_residuals", self.response, "dimensionless"),
            ("projector_defects", self.projector, "dimensionless"),
            ("nuclear_stationarity_defects", self.stationarity, "hartree"),
            ("reference_defects", self.reference, "hartree"),
        ] {
            report.set(key, values);
            labels.set(key, unit);
        }
        report.set("nuclear_xc_angular_history", self.angular_history);
        report.set("nuclear_xc_cache_bounds", self.cache_bounds);
        if !self.thermal_chemical.is_empty() {
            report.set("thermal_stationarity_defects", self.thermal_stationarity);
            report.set("thermal_number_defects", self.thermal_number);
            report.set("thermal_chemical_potential_paths", self.thermal_chemical);
            labels.set("thermal_stationarity_defects", "dimensionless");
            labels.set("thermal_number_defects", "electrons");
            labels.set("thermal_chemical_potential_paths", "hartree");
        }
        labels.set(
            "nuclear_xc_angular_history",
            "[dimensionless, hartree, hartree]",
        );
        labels.set(
            "nuclear_xc_cache_bounds",
            "[dimensionless, hartree, hartree]",
        );
        report.set("units", labels);
    }
}

/// Analytic empirical-dispersion coefficients shared by all reference spaces.
pub(super) fn dispersion_path(
    setup: &super::method::MolecularSetup,
    system: &super::System,
    direction: &[[f64; 3]],
    order: usize,
    model: Option<crate::dispersion::D3BjParams>,
) -> Result<Vec<f64>> {
    use crate::dual::Scalar;
    use crate::response::series::{evaluate_series_many_result, SeriesScalar};
    if order > 4 || direction.len() != system.positions.len() {
        return Err(DftError::InvalidInput(
            "dispersion paths require matching nuclear directions and order zero through four"
                .into(),
        ));
    }
    let inputs: Vec<_> = system
        .positions
        .iter()
        .zip(direction)
        .flat_map(|(r, d)| {
            (0..3).map(move |a| {
                let mut value = vec![0.0; order + 1];
                value[0] = r[a];
                if order > 0 {
                    value[1] = d[a];
                }
                value
            })
        })
        .collect();
    Ok(evaluate_series_many_result(&inputs, |v| {
        let positions: Vec<[SeriesScalar; 3]> =
            v.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
        let empirical = crate::dispersion::dispersion_energy_g(
            &system.numbers,
            &positions,
            &setup.dispersion,
            &setup.method.dispersion_functional_name(),
        )?;
        let named = model.map_or(SeriesScalar::from_f64(0.0), |p| {
            crate::dispersion::d3_energy_g(&system.numbers, &positions, None, Some(&p))
        });
        Ok(vec![empirical + named])
    })?
    .remove(0))
}

pub struct NuclearTensor {
    pub energies: Vec<f64>,
    pub gradients: Vec<Vec<f64>>,
    pub hessians: Vec<Matrix>,
    pub paths: usize,
}

/// Every Cartesian entry is returned. Symmetry reduces production paths;
/// acceptance differentiates every output entry independently.
pub fn assemble(
    atoms: usize,
    order: usize,
    channels: usize,
    mut evaluate: impl FnMut(Vec<[f64; 3]>) -> Result<Vec<Vec<f64>>>,
) -> Result<NuclearTensor> {
    if atoms == 0 || !(1..=2).contains(&order) || channels == 0 {
        return Err(DftError::InvalidInput(
            "nuclear tensors require atoms, channels and derivative order one or two".into(),
        ));
    }
    let dim = 3 * atoms;
    let mut out = NuclearTensor {
        energies: vec![0.0; channels],
        gradients: vec![vec![0.0; dim]; channels],
        hessians: vec![Matrix::zeros(dim, dim); channels],
        paths: 0,
    };
    let mut diagonal = vec![vec![0.0; dim]; channels];
    let mut path = |direction| {
        let value = evaluate(direction)?;
        if value.len() != channels
            || value
                .iter()
                .any(|v| v.len() <= order || v.iter().any(|v| !v.is_finite()))
        {
            return Err(DftError::InvalidInput("analytic nuclear energy channels have incompatible order or non-finite coefficients".into()));
        }
        Ok(value)
    };
    for a in 0..dim {
        let mut direction = vec![[0.0; 3]; atoms];
        direction[a / 3][a % 3] = 1.0;
        let energy = path(direction)?;
        out.paths += 1;
        for c in 0..channels {
            if a == 0 {
                out.energies[c] = energy[c][0];
            }
            out.gradients[c][a] = energy[c][1];
            if order == 2 {
                diagonal[c][a] = energy[c][2];
                out.hessians[c][(a, a)] = 2.0 * energy[c][2];
            }
        }
    }
    if order == 2 {
        for a in 0..dim {
            for b in 0..a {
                let mut direction = vec![[0.0; 3]; atoms];
                direction[a / 3][a % 3] += 1.0;
                direction[b / 3][b % 3] += 1.0;
                let energy = path(direction)?;
                out.paths += 1;
                for c in 0..channels {
                    let mixed = energy[c][2] - diagonal[c][a] - diagonal[c][b];
                    out.hessians[c][(a, b)] = mixed;
                    out.hessians[c][(b, a)] = mixed;
                }
            }
        }
    }
    Ok(out)
}
