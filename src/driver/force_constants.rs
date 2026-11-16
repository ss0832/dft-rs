// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete molecular cubic/quartic Cartesian and pattern tensors.
use super::{Controls, System, ToValue, Value};
use crate::response::force_constants::{
    analytic_force_constants_with, apply_cartesian_diagnostics,
};
use crate::{DftError, Result};

pub(super) struct Selection {
    pub patterns: Vec<Vec<[f64; 3]>>,
    pub coordinates: Option<Vec<usize>>,
    pub order: usize,
}
impl Selection {
    pub(super) fn new(system: &System, controls: &Controls) -> Result<Self> {
        let order = controls.usize("response_order", 3)?;
        if !(3..=4).contains(&order) || system.numbers.is_empty() {
            return Err(DftError::InvalidInput(
                "force constants require atoms and response_order three or four".into(),
            ));
        }
        for (a, b) in [
            ("coordinates", "force_coordinates"),
            ("patterns", "nuclear_directions"),
        ] {
            if controls.contains(a) && controls.contains(b) {
                return Err(DftError::InvalidInput(format!(
                    "use either {a} or its compatibility alias {b}"
                )));
            }
        }
        let coordinate_key = if controls.contains("force_coordinates") {
            "force_coordinates"
        } else {
            "coordinates"
        };
        let pattern_key = if controls.contains("nuclear_directions") {
            "nuclear_directions"
        } else {
            "patterns"
        };
        let coordinates = controls.opt_usize_list(coordinate_key)?;
        let raw = controls
            .take(pattern_key)
            .filter(|v| !matches!(v, Value::Null));
        if raw.is_some() && coordinates.is_some() {
            return Err(DftError::InvalidInput(
                "force constants accept patterns or coordinates, not both".into(),
            ));
        }
        let atoms = system.numbers.len();
        if let Some(raw) = raw {
            let (shape, values) = raw.numeric_array().ok_or_else(|| {
                DftError::InvalidInput("patterns must be finite [pattern][atom][xyz] arrays".into())
            })?;
            if shape.len() != 3
                || shape[0] == 0
                || shape[1] != atoms
                || shape[2] != 3
                || values.iter().any(|v| !v.is_finite())
            {
                return Err(DftError::InvalidInput(
                    "patterns must be nonempty finite [pattern][atom][xyz] arrays".into(),
                ));
            }
            let patterns = values
                .chunks_exact(3 * atoms)
                .map(|v| v.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect())
                .collect();
            return Ok(Self {
                patterns,
                coordinates: None,
                order,
            });
        }
        let coordinates = coordinates.unwrap_or_else(|| (0..3 * atoms).collect());
        let mut sorted = coordinates.clone();
        sorted.sort_unstable();
        if sorted.is_empty()
            || sorted.iter().any(|&c| c >= 3 * atoms)
            || sorted.windows(2).any(|w| w[0] == w[1])
        {
            return Err(DftError::InvalidInput(
                "force-constant coordinates must be unique nonempty Cartesian IDs 3*atom+axis"
                    .into(),
            ));
        }
        let patterns = coordinates
            .iter()
            .map(|&c| {
                let mut p = vec![[0.0; 3]; atoms];
                p[c / 3][c % 3] = 1.0;
                p
            })
            .collect();
        Ok(Self {
            patterns,
            coordinates: Some(coordinates),
            order,
        })
    }
    pub(super) fn unit(&self) -> &'static str {
        if self.coordinates.is_none() {
            "hartree"
        } else if self.order == 3 {
            "hartree/bohr^3"
        } else {
            "hartree/bohr^4"
        }
    }
    /// Account for both numeric work arrays and labelled report values.
    pub(super) fn check_report_storage(&self, channels: usize, memory_mb: usize) -> Result<()> {
        let bytes = self
            .patterns
            .len()
            .checked_pow(self.order as u32)
            .and_then(|n| n.checked_mul(channels))
            .and_then(|n| {
                n.checked_mul(2 * (std::mem::size_of::<f64>() + std::mem::size_of::<Value>()))
            })
            .ok_or_else(|| {
                DftError::InvalidInput("force-constant report storage overflow".into())
            })?;
        let required_mb = bytes.div_ceil(1024 * 1024);
        if required_mb > memory_mb {
            return Err(DftError::ResourceLimit {
                operation: "complete force-constant numeric and report tensors",
                required_mb,
                limit_mb: memory_mb,
            });
        }
        Ok(())
    }
}

/// The same scalar-energy assembly and metadata serve every reference space.
pub(super) fn assemble_energy_paths(
    selection: &Selection,
    channels: usize,
    memory_mb: usize,
    mut evaluate: impl FnMut(&[f64], usize) -> Result<Vec<f64>>,
) -> Result<(crate::response::mixed::PolarizedTensor, usize)> {
    selection.check_report_storage(channels, memory_mb)?;
    let vectors: Vec<Vec<f64>> = selection
        .patterns
        .iter()
        .map(|p| p.iter().flatten().copied().collect())
        .collect();
    let mut cache = crate::response::mixed::DirectionalCoefficients::default();
    let tensors = crate::response::mixed::polarized_tensor(
        &vectors,
        selection.order,
        channels,
        memory_mb,
        false,
        |d, n| cache.get(d, n, channels, |u, n| evaluate(u, n)),
    )?;
    Ok((tensors, cache.evaluations))
}

pub(super) fn require_variational_reference(controls: &Controls) -> Result<()> {
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "correlated complete nuclear tensors on a constrained or spinor reference"
                .into(),
            missing: "the reference-constrained correlated orbital and amplitude response provider"
                .into(),
        });
    }
    if controls.contains("bfield") {
        return Err(DftError::Unsupported {
            feature: "complete constrained or spinor nuclear tensors in a London magnetic basis"
                .into(),
            missing: "the matching magnetic reference and moving-basis tensor provider".into(),
        });
    }
    Ok(())
}

pub(super) fn append_energy_tensors(
    report: &mut Value,
    selection: &Selection,
    tensors: crate::response::mixed::PolarizedTensor,
    evaluations: usize,
    prefixes: &[&str],
    config: &crate::response::ResponseSolverConfig,
) -> Result<()> {
    let unit = selection.unit();
    let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
    let mut translation_converged = true;
    let atoms = selection.patterns[0].len();
    for (channel, prefix) in prefixes.iter().enumerate() {
        let translation = selection
            .coordinates
            .as_ref()
            .map(|coordinates| {
                crate::response::force_constants::cartesian_translation_residual(
                    &tensors.values[channel],
                    coordinates,
                    atoms,
                    selection.order,
                )
            })
            .transpose()?
            .flatten();
        if let Some((defect, _)) = translation {
            let scale = tensors.values[channel]
                .iter()
                .map(|v| v.abs())
                .fold(1.0, f64::max);
            translation_converged &= defect <= config.tolerance * scale;
        }
        for (name, value) in [
            ("tensor", tensors.values[channel].to_value()),
            (
                "accumulation_roundoff",
                tensors.roundoff[channel].to_value(),
            ),
            ("translation_residual", translation.map(|v| v.0).to_value()),
        ] {
            let key = format!("{prefix}{name}");
            report.set(&key, value);
            labels.set(&key, unit);
        }
        if channel == 0 {
            report.set("translation_comparisons", translation.map_or(0, |v| v.1));
        }
    }
    labels.set("patterns", "bohr");
    labels.set("permutation_defect", unit);
    report.set("units", labels);
    report.set("response_units", unit);
    report.set("order", selection.order);
    report.set("shape", vec![selection.patterns.len(); selection.order]);
    report.set("patterns", selection.patterns.clone());
    report.set("permutation_defect", Value::Null);
    report.set("permutation_comparisons", 0usize);
    report.set("independent_components", tensors.independent_components);
    report.set("directional_evaluations", evaluations);
    report.set("polarization_requests", tensors.directional_evaluations);
    report.set("assembly", "energy-polarization");
    report.set("axis_convention", "row-major selected coordinate or pattern axes; actual derivatives without factorial scaling; scalar-energy permutation reuse; ordered interchange unmeasured");
    report.set("provenance", "analytic");
    if !translation_converged {
        report.set("converged", false);
        report.set("response_converged", false);
        let mut warnings = report
            .get("warnings")
            .and_then(Value::as_array)
            .unwrap_or(&[])
            .to_vec();
        warnings.push(
            "raw complete nuclear tensors violate the acoustic translation sum rule".to_value(),
        );
        report.set("warnings", Value::Array(warnings));
    }
    Ok(())
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let selection = Selection::new(system, controls)?;
    if controls.contains("bfield")
        || controls.opt_str("reference")?.is_some_and(|r| {
            !matches!(
                r.to_ascii_lowercase().as_str(),
                "rks" | "rhf" | "uks" | "uhf" | "auto"
            )
        })
    {
        return Err(DftError::Unsupported {
            feature:
                "complete molecular nuclear tensor on the requested magnetic/constrained reference"
                    .into(),
            missing: "the corresponding complete reference-constrained nuclear energy path".into(),
        });
    }
    if super::double_hybrid::requested(controls)? {
        return super::correlated_high_order::force_constants(system, controls, &selection);
    }
    let config = super::vibrational::response_config(controls)?;
    let setup = super::molecular::setup(system, controls)?;
    selection.check_report_storage(1, setup.options.incore_mb)?;
    let has_d1 = matches!(&setup.dispersion, crate::dispersion::Dispersion::D1 { .. });
    if (!has_d1 && setup.dispersion != crate::dispersion::Dispersion::None) || setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "ordinary complete force tensors with empirical/solvent surface energy".into(),
            missing: "the matching non-electronic energy coefficient provider on this ordinary reference route".into(),
        });
    }
    controls.finish("force-constants")?;
    let reference = super::method::run_molecular_reference(&setup)?;
    let mut response = analytic_force_constants_with(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        &selection.patterns,
        selection.order,
        &config,
        |direction, order| {
            if has_d1 {
                let geometry =
                    super::nuclear_tensor::dispersion_path(&setup, system, direction, order, None)?;
                Ok(geometry[order])
            } else {
                Ok(0.0)
            }
        },
    )?;
    if let Some(coordinates) = &selection.coordinates {
        apply_cartesian_diagnostics(&mut response, coordinates, setup.molecule.len(), &config)?;
    }
    let mut report = super::electric::metadata(&setup, &reference, &response);
    let value = response.value;
    report.set("order", value.order);
    report.set("patterns", value.patterns);
    report.set("tensor", value.tensor);
    report.set("accumulation_roundoff", value.accumulation_roundoff);
    report.set("permutation_defect", value.permutation_defect);
    report.set("permutation_comparisons", value.permutation_comparisons);
    report.set("translation_residual", value.translation_residual);
    report.set("translation_comparisons", value.translation_comparisons);
    report.set("independent_components", value.independent_components);
    report.set("directional_evaluations", value.directional_evaluations);
    let mut labels = report
        .get("units")
        .cloned()
        .expect("force-constant report units");
    labels.set("patterns", "bohr");
    for key in [
        "tensor",
        "accumulation_roundoff",
        "permutation_defect",
        "translation_residual",
    ] {
        labels.set(key, selection.unit());
    }
    report.set("units", labels);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn report_memory_guard_counts_labelled_values_as_well_as_float_arrays() {
        let selection = Selection {
            order: 4,
            coordinates: None,
            patterns: vec![vec![[0.1, 0.2, 0.3]]; 10],
        };
        let numeric = 10usize.pow(4) * 8 * 2 * std::mem::size_of::<f64>();
        assert!(numeric < 2 * 1024 * 1024);
        assert!(matches!(
            selection.check_report_storage(8, 2),
            Err(DftError::ResourceLimit { .. })
        ));
        selection.check_report_storage(8, 10).unwrap();
    }
}
