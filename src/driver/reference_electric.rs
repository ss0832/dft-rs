// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Reference-independent electric tensors from stationary energy providers.
//! Only scalar Taylor coefficients are cached. Every ordered tensor slot is
//! assembled, and unavailable post-SCF terms are never inferred from a parent.
use super::{Controls, System, Value};
use crate::response::mixed::{polarized_mixed, tensor_indices, DirectionalCoefficients};
use crate::response::nuclear::{MolecularDirection, StaticSpectroscopicDerivative};
use crate::response::ResponseSolverConfig;
use crate::{DftError, Result};

pub(super) struct Request {
    rank: usize,
    nuclear: Option<Vec<[f64; 3]>>,
    atoms: usize,
}

impl Request {
    pub fn new(id: &str, system: &System, controls: &Controls) -> Result<Self> {
        if super::double_hybrid::requested(controls)? {
            return Err(DftError::Unsupported {
                feature: "reference-constrained correlated electric tensors".into(),
                missing: "the matching constrained PT2 amplitude and orbital provider; parent-only results do not include correlation".into(),
            });
        }
        let nuclear = if id == "static-spectroscopy" {
            Some(super::static_spectroscopy::nuclear_direction(
                system, controls,
            )?)
        } else {
            None
        };
        let rank = match id {
            "static-spectroscopy" => controls.usize("electric_order", 2)?,
            "beta" => 3,
            "gamma" => 4,
            _ => {
                return Err(DftError::InvalidInput(
                    "unknown stationary electric tensor request".into(),
                ))
            }
        };
        if nuclear.is_some() && !(1..=3).contains(&rank) {
            return Err(DftError::InvalidInput(
                "electric_order must be one, two or three".into(),
            ));
        }
        if nuclear.is_none() {
            if controls.contains("tensor_assembly") && controls.contains("electric_assembly") {
                return Err(DftError::InvalidInput(
                    "use either tensor_assembly or electric_assembly".into(),
                ));
            }
            let key = if controls.contains("electric_assembly") {
                "electric_assembly"
            } else {
                "tensor_assembly"
            };
            let assembly = controls.str(key, "energy")?;
            if !matches!(assembly.as_str(), "projector" | "energy") {
                return Err(DftError::InvalidInput(
                    "constrained electric tensors require projector or energy assembly".into(),
                ));
            }
        }
        Ok(Self {
            rank,
            nuclear,
            atoms: system.numbers.len(),
        })
    }

    pub fn apply(
        &self,
        report: &mut Value,
        config: &ResponseSolverConfig,
        mut energy: impl FnMut(&MolecularDirection, usize) -> Result<Vec<f64>>,
    ) -> Result<()> {
        self.apply_coefficients(report, config, false, |d, n| {
            Ok([energy(d, n)?, Vec::new()])
        })
    }

    pub fn apply_retained(
        &self,
        report: &mut Value,
        config: &ResponseSolverConfig,
        energy: impl FnMut(&MolecularDirection, usize) -> Result<[Vec<f64>; 2]>,
    ) -> Result<()> {
        self.apply_coefficients(report, config, true, energy)
    }

    fn apply_coefficients(
        &self,
        report: &mut Value,
        config: &ResponseSolverConfig,
        retained: bool,
        mut energy: impl FnMut(&MolecularDirection, usize) -> Result<[Vec<f64>; 2]>,
    ) -> Result<()> {
        let dimension = 3 * self.atoms + 3;
        let mut cache = DirectionalCoefficients::default();
        let size = 3usize.pow(self.rank as u32);
        let mut values = Vec::with_capacity(size);
        let mut errors = Vec::with_capacity(values.capacity());
        for flat in 0..size {
            let mut legs = Vec::new();
            if let Some(nuclear) = &self.nuclear {
                legs.push(
                    nuclear
                        .iter()
                        .flatten()
                        .copied()
                        .chain([0.0; 3])
                        .collect::<Vec<_>>(),
                );
            }
            for axis in tensor_indices(flat, 3, self.rank) {
                let mut field = vec![0.0; dimension];
                field[3 * self.atoms + axis] = 1.0;
                legs.push(field);
            }
            let result = polarized_mixed(dimension, &legs, |d, n| {
                let evaluate = |d: &[f64], n| {
                    let path = MolecularDirection {
                        nuclear: d[..3 * self.atoms]
                            .chunks_exact(3)
                            .map(|d| [d[0], d[1], d[2]])
                            .collect(),
                        electric: [
                            d[3 * self.atoms],
                            d[3 * self.atoms + 1],
                            d[3 * self.atoms + 2],
                        ],
                    };
                    let [coefficients, remainders] = energy(&path, n)?;
                    if coefficients.len() <= n
                        || coefficients.iter().any(|v| !v.is_finite())
                        || retained
                            && (remainders.len() != coefficients.len()
                                || remainders.iter().any(|v| !v.is_finite()))
                    {
                        return Err(DftError::InvalidInput("stationary energy provider returned incompatible or non-finite coefficients".into()));
                    }
                    Ok(if retained {
                        vec![coefficients[n], remainders[n]]
                    } else {
                        vec![coefficients[n]]
                    })
                };
                if retained {
                    cache.get_retained(d, n, 2, evaluate)
                } else {
                    cache.get(d, n, 1, evaluate)
                }
            })?;
            if retained {
                let mut sum = crate::linalg::AccurateSum::default();
                for &value in &result.value {
                    sum.add_product(value, 1.0);
                }
                values.push(-sum.value());
                errors
                    .push(result.accumulation_roundoff.iter().sum::<f64>() + sum.roundoff_bound());
            } else {
                values.push(-result.value[0]);
                errors.push(result.accumulation_roundoff[0]);
            }
        }
        let c = crate::response::electric_tensor::tensor_contractions(self.rank - 1, &values);
        let defect = c
            .intrinsic_permutation_defect
            .max(c.energy_permutation_defect);
        let scale = values.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        if defect > config.tolerance * scale
            || errors
                .iter()
                .any(|v| !v.is_finite() || *v > config.tolerance * scale)
        {
            report.set("converged", false);
            let mut warnings = report
                .get("warnings")
                .and_then(Value::as_array)
                .unwrap_or(&[])
                .to_vec();
            warnings.push(Value::Str("stationary energy tensor interchange or accumulation exceeds its requested tolerance".into()));
            report.set("warnings", Value::Array(warnings));
        }
        report.set("provenance", "analytic");
        report.set("shape", vec![3usize; self.rank]);
        report.set("assembly", "energy-polarization");
        report.set("interchange_diagnostics","ordered scalar-energy polarization; no tensor-slot copying; scalar coefficients reused; independent finite-field acceptance is required for observed-leg interchange");
        report.set("axis_convention","row-major Cartesian electric indices; actual unnormalized energy derivatives with a minus sign; optional contracted Bohr nuclear path");
        let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
        labels.set("interchange_diagnostics", "text");
        if let Some(nuclear) = &self.nuclear {
            report.set("units", labels);
            super::static_spectroscopy::append_values(
                report,
                StaticSpectroscopicDerivative {
                    nuclear_direction: nuclear.clone(),
                    electric_order: self.rank,
                    tensor: values,
                    accumulation_roundoff: errors,
                    permutation_defect: defect,
                    permutation_comparisons: match self.rank {
                        1 => 0,
                        2 => 6,
                        _ => 36,
                    },
                    directional_evaluations: cache.evaluations,
                },
            );
        } else {
            append_energy_tensor(
                report,
                self.rank,
                values,
                errors,
                cache.evaluations,
                c,
                labels,
            );
        }
        Ok(())
    }
}

/// Common scalar-energy tensor fields. Actual response diagnostics are
/// provided by the orbital backend and finite-field acceptance.
pub(in crate::driver) fn append_energy_tensor(
    report: &mut Value,
    rank: usize,
    values: Vec<f64>,
    errors: Vec<f64>,
    evaluations: usize,
    contractions: crate::response::electric_tensor::TensorContractions,
    mut labels: Value,
) -> &'static str {
    let unit = if rank == 3 {
        "e^3*bohr^3/hartree^2"
    } else {
        "e^4*bohr^4/hartree^3"
    };
    report.set("tensor", values);
    report.set("response_units", unit);
    labels.set("response_units", "text");
    report.set("order", rank - 1);
    report.set("accumulation_roundoff", errors);
    report.set("directional_evaluations", evaluations);
    report.set("ordered_field_evaluations", 3usize.pow((rank - 1) as u32));
    report.set("frobenius_norm", contractions.frobenius_norm);
    report.set("beta_vector", contractions.beta_vector);
    report.set("beta_vector_norm", contractions.beta_vector_norm);
    report.set("gamma_isotropic", contractions.gamma_isotropic);
    for key in ["intrinsic_permutation_defect", "energy_permutation_defect"] {
        report.set(key, Value::Null);
        labels.set(key, unit);
    }
    for key in [
        "intrinsic_permutation_comparisons",
        "energy_permutation_comparisons",
    ] {
        report.set(key, 0usize);
    }
    for key in ["tensor", "accumulation_roundoff", "frobenius_norm"] {
        labels.set(key, unit);
    }
    labels.set("beta_vector", "e^3*bohr^3/hartree^2");
    labels.set("beta_vector_norm", "e^3*bohr^3/hartree^2");
    labels.set("gamma_isotropic", "e^4*bohr^4/hartree^3");
    report.set("units", labels);
    unit
}
