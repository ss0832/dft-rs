// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Deterministic fitting mechanics for pseudo-substituent parameter candidates.
//!
//! This module deliberately does not create a [`super::SiteOperator`] and does
//! not register any [`super::NamedSubstituent`]. A numerical fit is only a
//! candidate; a name becomes usable after the physical reference evaluator and
//! that substituent's independent acceptance gates have passed.

use crate::error::{DftError, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// Physical observable represented by one normalized residual.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObservableKind {
    Density,
    Esp,
    BondLength,
    BondAngle,
    RepulsionEnergy,
    RepulsionForce,
    TorsionEnergy,
    Other(String),
}

impl ObservableKind {
    pub fn name(&self) -> &str {
        match self {
            Self::Density => "density",
            Self::Esp => "esp",
            Self::BondLength => "bond-length",
            Self::BondAngle => "bond-angle",
            Self::RepulsionEnergy => "repulsion-energy",
            Self::RepulsionForce => "repulsion-force",
            Self::TorsionEnergy => "torsion-energy",
            Self::Other(name) => name,
        }
    }
}

/// Stability diagnostics attached to a physical reference evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Diagnostics {
    pub aux_population: f64,
    pub min_b_eigval: f64,
    pub min_s_eigval: f64,
    pub lowest_mo: f64,
}

impl Diagnostics {
    fn is_finite(self) -> bool {
        self.aux_population.is_finite()
            && self.min_b_eigval.is_finite()
            && self.min_s_eigval.is_finite()
            && self.lowest_mo.is_finite()
    }
}

/// Minimum acceptable values for the four fitter diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DiagnosticLimits {
    pub aux_population: Option<f64>,
    pub min_b_eigval: Option<f64>,
    pub min_s_eigval: Option<f64>,
    pub lowest_mo: Option<f64>,
}

impl DiagnosticLimits {
    fn enabled(self) -> bool {
        self.aux_population.is_some()
            || self.min_b_eigval.is_some()
            || self.min_s_eigval.is_some()
            || self.lowest_mo.is_some()
    }

    fn accepts(self, value: Diagnostics) -> bool {
        self.aux_population
            .is_none_or(|limit| value.aux_population >= limit)
            && self
                .min_b_eigval
                .is_none_or(|limit| value.min_b_eigval >= limit)
            && self
                .min_s_eigval
                .is_none_or(|limit| value.min_s_eigval >= limit)
            && self.lowest_mo.is_none_or(|limit| value.lowest_mo >= limit)
    }
}

/// How a diagnostic failure enters the fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiagnosticPolicy {
    Exclude,
    /// Keep the row and add this explicit, parameter-independent loss.
    Penalize {
        loss: f64,
    },
}

/// One row of a fixed-density/linearized reference problem.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceDatum {
    pub case_id: String,
    /// Split unit. All observables from one molecule/geometry share a group.
    pub group_id: String,
    pub observable: ObservableKind,
    pub target: f64,
    /// Unit normalization for this observable. Must be positive.
    pub scale: f64,
    pub weight: f64,
    pub linear_features: Vec<f64>,
    pub diagnostics: Option<Diagnostics>,
}

/// A provenance-labelled fixed-level reference set.
#[derive(Clone, Debug, PartialEq)]
pub struct FitDataset {
    pub level: String,
    pub dataset_id: String,
    pub parameter_names: Vec<String>,
    pub data: Vec<ReferenceDatum>,
}

/// Packed symmetric parameter block constrained by a spectral lower bound.
///
/// Packing is Euclidean/Frobenius preserving: diagonals are stored directly,
/// off-diagonals as `sqrt(2) * B_ij`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralBlock {
    pub offset: usize,
    pub dimension: usize,
    pub min_eigenvalue: f64,
}

impl SpectralBlock {
    pub fn packed_len(self) -> usize {
        self.dimension * (self.dimension + 1) / 2
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitConfig {
    pub seed: u64,
    pub validation_fraction: f64,
    pub tikhonov: f64,
    pub prior: Vec<f64>,
    pub parameter_scales: Vec<f64>,
    pub spectral_blocks: Vec<SpectralBlock>,
    pub diagnostic_limits: DiagnosticLimits,
    pub diagnostic_policy: DiagnosticPolicy,
    pub max_excluded_fraction: f64,
    pub history: usize,
    pub max_iter: usize,
    pub grad_tol: f64,
    pub min_step: f64,
}

impl FitConfig {
    pub fn for_parameter_count(n: usize) -> Self {
        Self {
            seed: 0x4454_4654_5253,
            validation_fraction: 0.2,
            tikhonov: 1e-8,
            prior: vec![0.0; n],
            parameter_scales: vec![1.0; n],
            spectral_blocks: Vec::new(),
            diagnostic_limits: DiagnosticLimits::default(),
            diagnostic_policy: DiagnosticPolicy::Exclude,
            max_excluded_fraction: 0.1,
            history: 8,
            max_iter: 200,
            grad_tol: 1e-10,
            min_step: 1e-12,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupSplit {
    pub train: Vec<String>,
    pub validation: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerResult {
    pub parameters: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub evaluations: usize,
    pub objective: f64,
    pub projected_gradient_max: f64,
}

/// Candidate-only record. It is never consumed by `NamedSubstituent::build`.
#[derive(Clone, Debug, PartialEq)]
pub struct FitParameterRecord {
    pub level: String,
    pub dataset_id: String,
    pub dataset_fingerprint: u64,
    pub seed: u64,
    pub validation_fraction: f64,
    pub tikhonov: f64,
    pub converged: bool,
    pub iterations: usize,
    pub objective: f64,
    pub train_rmse: f64,
    pub validation_rmse: f64,
    pub excluded: usize,
    pub penalized: usize,
    pub train_groups: Vec<String>,
    pub validation_groups: Vec<String>,
    pub parameters: Vec<(String, f64)>,
}

impl FitParameterRecord {
    pub fn validate_level(&self, requested: &str) -> Result<()> {
        if self.level == requested {
            Ok(())
        } else {
            Err(DftError::InvalidInput(format!(
                "pseudo-substituent parameter level mismatch: record is `{}`, requested `{requested}`",
                self.level
            )))
        }
    }

    /// Deterministic dependency-free candidate record.
    pub fn to_text(&self) -> Result<String> {
        validate_record_token("level", &self.level)?;
        validate_record_token("dataset_id", &self.dataset_id)?;
        for group in self.train_groups.iter().chain(&self.validation_groups) {
            validate_record_token("group id", group)?;
        }
        let mut out = String::new();
        writeln!(&mut out, "dft-rs-pseudo-substituent-fit=1").unwrap();
        writeln!(&mut out, "status=candidate-unaccepted").unwrap();
        writeln!(&mut out, "level={}", self.level).unwrap();
        writeln!(&mut out, "dataset_id={}", self.dataset_id).unwrap();
        writeln!(
            &mut out,
            "dataset_fingerprint={:016x}",
            self.dataset_fingerprint
        )
        .unwrap();
        writeln!(&mut out, "seed={}", self.seed).unwrap();
        writeln!(
            &mut out,
            "validation_fraction={:.17e}",
            self.validation_fraction
        )
        .unwrap();
        writeln!(&mut out, "tikhonov={:.17e}", self.tikhonov).unwrap();
        writeln!(&mut out, "converged={}", self.converged).unwrap();
        writeln!(&mut out, "iterations={}", self.iterations).unwrap();
        writeln!(&mut out, "objective={:.17e}", self.objective).unwrap();
        writeln!(&mut out, "train_rmse={:.17e}", self.train_rmse).unwrap();
        writeln!(&mut out, "validation_rmse={:.17e}", self.validation_rmse).unwrap();
        writeln!(&mut out, "excluded={}", self.excluded).unwrap();
        writeln!(&mut out, "penalized={}", self.penalized).unwrap();
        for group in &self.train_groups {
            writeln!(&mut out, "train_group={group}").unwrap();
        }
        for group in &self.validation_groups {
            writeln!(&mut out, "validation_group={group}").unwrap();
        }
        for (name, value) in &self.parameters {
            validate_record_token("parameter name", name)?;
            if !value.is_finite() {
                return Err(DftError::InvalidInput(format!(
                    "parameter `{name}` is non-finite"
                )));
            }
            writeln!(&mut out, "parameter={name},{value:.17e}").unwrap();
        }
        Ok(out)
    }

    pub fn from_text(text: &str) -> Result<Self> {
        let mut scalar = BTreeMap::<String, String>::new();
        let mut train_groups = Vec::new();
        let mut validation_groups = Vec::new();
        let mut parameters = Vec::new();
        for (index, raw) in text.lines().enumerate() {
            let line_no = index + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line.split_once('=').ok_or_else(|| DftError::Parse {
                line: line_no,
                message: "candidate record line must contain '='".to_string(),
            })?;
            match key {
                "train_group" => train_groups.push(value.to_string()),
                "validation_group" => validation_groups.push(value.to_string()),
                "parameter" => {
                    let (name, raw_value) =
                        value.split_once(',').ok_or_else(|| DftError::Parse {
                            line: line_no,
                            message: "parameter line must be name,value".to_string(),
                        })?;
                    if raw_value.contains(',') {
                        return Err(DftError::Parse {
                            line: line_no,
                            message: "parameter line has too many fields".to_string(),
                        });
                    }
                    parameters.push((name.to_string(), parse_finite(raw_value, line_no)?));
                }
                _ => {
                    if scalar.insert(key.to_string(), value.to_string()).is_some() {
                        return Err(DftError::Parse {
                            line: line_no,
                            message: format!("duplicate `{key}` field"),
                        });
                    }
                }
            }
        }
        if take(&mut scalar, "dft-rs-pseudo-substituent-fit")? != "1" {
            return Err(DftError::InvalidInput(
                "unsupported pseudo-substituent fit record version".to_string(),
            ));
        }
        if take(&mut scalar, "status")? != "candidate-unaccepted" {
            return Err(DftError::InvalidInput(
                "fit record must remain candidate-unaccepted".to_string(),
            ));
        }
        let level = take(&mut scalar, "level")?;
        let dataset_id = take(&mut scalar, "dataset_id")?;
        let fingerprint_text = take(&mut scalar, "dataset_fingerprint")?;
        let dataset_fingerprint = u64::from_str_radix(&fingerprint_text, 16).map_err(|_| {
            DftError::InvalidInput("invalid dataset_fingerprint in fit record".to_string())
        })?;
        let seed = parse_field::<u64>(&mut scalar, "seed")?;
        let validation_fraction = parse_float_field(&mut scalar, "validation_fraction")?;
        let tikhonov = parse_float_field(&mut scalar, "tikhonov")?;
        let converged = parse_field::<bool>(&mut scalar, "converged")?;
        let iterations = parse_field::<usize>(&mut scalar, "iterations")?;
        let objective = parse_float_field(&mut scalar, "objective")?;
        let train_rmse = parse_float_field(&mut scalar, "train_rmse")?;
        let validation_rmse = parse_float_field(&mut scalar, "validation_rmse")?;
        let excluded = parse_field::<usize>(&mut scalar, "excluded")?;
        let penalized = parse_field::<usize>(&mut scalar, "penalized")?;
        if !scalar.is_empty() {
            return Err(DftError::InvalidInput(format!(
                "unknown fit record field(s): {}",
                scalar.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
        Ok(Self {
            level,
            dataset_id,
            dataset_fingerprint,
            seed,
            validation_fraction,
            tikhonov,
            converged,
            iterations,
            objective,
            train_rmse,
            validation_rmse,
            excluded,
            penalized,
            train_groups,
            validation_groups,
            parameters,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitResult {
    pub optimizer: OptimizerResult,
    pub record: FitParameterRecord,
}

/// Deterministic group-wise split. No molecule/geometry leaks across sides.
pub fn deterministic_group_split(
    data: &[ReferenceDatum],
    seed: u64,
    validation_fraction: f64,
) -> Result<GroupSplit> {
    if !(0.0..1.0).contains(&validation_fraction) || !validation_fraction.is_finite() {
        return Err(DftError::InvalidInput(
            "validation_fraction must be finite and in [0,1)".to_string(),
        ));
    }
    let groups: BTreeSet<String> = data.iter().map(|row| row.group_id.clone()).collect();
    if groups.len() < 2 {
        return Err(DftError::InvalidInput(
            "at least two reference groups are required for train/validation splitting".to_string(),
        ));
    }
    let mut ranked: Vec<(u64, String)> = groups
        .into_iter()
        .map(|group| (stable_hash(seed, group.as_bytes()), group))
        .collect();
    ranked.sort();
    let requested = (validation_fraction * ranked.len() as f64).round() as usize;
    let n_validation = requested.max(1).min(ranked.len() - 1);
    let validation: BTreeSet<String> = ranked
        .iter()
        .take(n_validation)
        .map(|(_, group)| group.clone())
        .collect();
    let train = ranked
        .iter()
        .map(|(_, group)| group.clone())
        .filter(|group| !validation.contains(group))
        .collect();
    Ok(GroupSplit {
        train,
        validation: validation.into_iter().collect(),
    })
}

/// Project packed symmetric B blocks onto their configured eigenvalue bounds.
pub fn project_spectral_blocks(parameters: &mut [f64], blocks: &[SpectralBlock]) -> Result<usize> {
    validate_spectral_blocks(parameters.len(), blocks)?;
    let mut active = 0;
    for &block in blocks {
        let n = block.dimension;
        let mut matrix = Matrix::zeros(n, n);
        let mut cursor = block.offset;
        for i in 0..n {
            for j in i..n {
                let value = if i == j {
                    parameters[cursor]
                } else {
                    parameters[cursor] / 2.0_f64.sqrt()
                };
                matrix[(i, j)] = value;
                matrix[(j, i)] = value;
                cursor += 1;
            }
        }
        let (mut eigenvalues, vectors) = symmetric_eigen(&matrix)?;
        for value in &mut eigenvalues {
            if *value < block.min_eigenvalue {
                *value = block.min_eigenvalue;
                active += 1;
            }
        }
        cursor = block.offset;
        for i in 0..n {
            for j in i..n {
                let mut value = 0.0;
                for k in 0..n {
                    value += vectors[(i, k)] * eigenvalues[k] * vectors[(j, k)];
                }
                parameters[cursor] = if i == j {
                    value
                } else {
                    2.0_f64.sqrt() * value
                };
                cursor += 1;
            }
        }
    }
    Ok(active)
}

/// Generic deterministic projected L-BFGS used by both the synthetic gate and
/// later physical nonlinear objectives.
/// An unconverged SCF at a trial point rejects that step and triggers the
/// existing backtracking search. Initial-point and other errors propagate.
pub fn projected_lbfgs<F, P>(
    initial: &[f64],
    history: usize,
    max_iter: usize,
    grad_tol: f64,
    min_step: f64,
    mut evaluate: F,
    mut project: P,
) -> Result<OptimizerResult>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
    P: FnMut(&mut [f64]) -> Result<usize>,
{
    if initial.is_empty() || history == 0 || max_iter == 0 {
        return Err(DftError::InvalidInput(
            "projected L-BFGS requires nonempty parameters, history, and max_iter".to_string(),
        ));
    }
    if !(grad_tol > 0.0 && grad_tol.is_finite() && min_step > 0.0 && min_step.is_finite()) {
        return Err(DftError::InvalidInput(
            "projected L-BFGS tolerances must be finite and positive".to_string(),
        ));
    }
    let mut x = initial.to_vec();
    project(&mut x)?;
    let (mut value, mut gradient) = checked_evaluation(&mut evaluate, &x)?;
    let mut evaluations = 1;
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();

    for iteration in 0..max_iter {
        let projected_gradient_max = gradient_mapping_max(&x, &gradient, &mut project)?;
        if projected_gradient_max < grad_tol {
            return Ok(OptimizerResult {
                parameters: x,
                converged: true,
                iterations: iteration,
                evaluations,
                objective: value,
                projected_gradient_max,
            });
        }
        let mut q = gradient.clone();
        let m = s_history.len();
        let mut alpha = vec![0.0; m];
        for k in (0..m).rev() {
            let sy = dot(&s_history[k], &y_history[k]);
            alpha[k] = dot(&s_history[k], &q) / sy;
            axpy(&mut q, -alpha[k], &y_history[k]);
        }
        if let Some((s, y)) = s_history.last().zip(y_history.last()) {
            let yy = dot(y, y);
            let gamma = if yy > 0.0 { dot(s, y) / yy } else { 1.0 };
            for value in &mut q {
                *value *= gamma;
            }
        }
        for k in 0..m {
            let sy = dot(&s_history[k], &y_history[k]);
            let beta = dot(&y_history[k], &q) / sy;
            axpy(&mut q, alpha[k] - beta, &s_history[k]);
        }
        let mut direction: Vec<f64> = q.iter().map(|value| -value).collect();
        let mut trial = add_scaled(&x, &direction, 1.0);
        project(&mut trial)?;
        let mut displacement: Vec<f64> = trial.iter().zip(&x).map(|(a, b)| a - b).collect();
        if dot(&gradient, &displacement) >= 0.0 {
            s_history.clear();
            y_history.clear();
            direction = gradient.iter().map(|value| -value).collect();
        }

        let mut step = 1.0;
        let mut accepted = None;
        while step >= min_step {
            trial = add_scaled(&x, &direction, step);
            project(&mut trial)?;
            displacement = trial.iter().zip(&x).map(|(a, b)| a - b).collect();
            let slope = dot(&gradient, &displacement);
            if slope < 0.0 {
                let evaluation = checked_evaluation(&mut evaluate, &trial);
                evaluations += 1;
                match evaluation {
                    Ok((trial_value, trial_gradient)) => {
                        if trial_value <= value + 1e-4 * slope {
                            accepted = Some((trial, trial_value, trial_gradient, displacement));
                            break;
                        }
                    }
                    Err(DftError::ScfNotConverged { iterations, error }) => {
                        // A failed electronic solve supplies neither a physical
                        // objective nor a gradient. Keep the last accepted
                        // state and reduce only the proposed parameter step.
                        crate::logger::log(&format!(
                            "[projected L-BFGS] reject unconverged SCF trial step={step:.12e} iterations={iterations} residual={error:.12e}"
                        ));
                    }
                    Err(error) => return Err(error),
                }
            }
            step *= 0.5;
        }
        let Some((new_x, new_value, new_gradient, s)) = accepted else {
            return Ok(OptimizerResult {
                parameters: x,
                converged: false,
                iterations: iteration + 1,
                evaluations,
                objective: value,
                projected_gradient_max,
            });
        };
        let y: Vec<f64> = new_gradient
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a - b)
            .collect();
        let sy = dot(&s, &y);
        let curvature_floor = 1e-12 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt();
        if sy > curvature_floor.max(0.0) {
            s_history.push(s);
            y_history.push(y);
            if s_history.len() > history {
                s_history.remove(0);
                y_history.remove(0);
            }
        }
        x = new_x;
        value = new_value;
        gradient = new_gradient;
    }
    let projected_gradient_max = gradient_mapping_max(&x, &gradient, &mut project)?;
    Ok(OptimizerResult {
        parameters: x,
        converged: projected_gradient_max < grad_tol,
        iterations: max_iter,
        evaluations,
        objective: value,
        projected_gradient_max,
    })
}

/// Fit the linearized reference set, then pass the same objective through the
/// projected nonlinear optimizer. Validation data never influence either step.
pub fn fit_reference_set(dataset: &FitDataset, config: &FitConfig) -> Result<FitResult> {
    validate_dataset(dataset, config)?;
    let split = deterministic_group_split(&dataset.data, config.seed, config.validation_fraction)?;
    let train_groups: BTreeSet<&str> = split.train.iter().map(String::as_str).collect();
    let validation_groups: BTreeSet<&str> = split.validation.iter().map(String::as_str).collect();
    let (active, excluded, penalized, penalty_loss) =
        select_rows(dataset, config, &train_groups, &validation_groups)?;
    let train: Vec<usize> = active
        .iter()
        .copied()
        .filter(|&index| train_groups.contains(dataset.data[index].group_id.as_str()))
        .collect();
    let validation: Vec<usize> = active
        .iter()
        .copied()
        .filter(|&index| validation_groups.contains(dataset.data[index].group_id.as_str()))
        .collect();
    if !train.iter().any(|&index| dataset.data[index].weight > 0.0) {
        return Err(DftError::InvalidInput(
            "no positive-weight training rows remain after diagnostics".to_string(),
        ));
    }
    let initial = weighted_ridge_initial_guess(dataset, config, &train)?;
    let blocks = config.spectral_blocks.clone();
    let mut objective =
        |parameters: &[f64]| quadratic_objective(dataset, config, &train, parameters, penalty_loss);
    let mut projection = |parameters: &mut [f64]| project_spectral_blocks(parameters, &blocks);
    let optimizer = projected_lbfgs(
        &initial,
        config.history,
        config.max_iter,
        config.grad_tol,
        config.min_step,
        &mut objective,
        &mut projection,
    )?;
    let train_rmse = weighted_rmse(dataset, &train, &optimizer.parameters)?;
    let validation_rmse = weighted_rmse(dataset, &validation, &optimizer.parameters)?;
    let record = FitParameterRecord {
        level: dataset.level.clone(),
        dataset_id: dataset.dataset_id.clone(),
        dataset_fingerprint: dataset_fingerprint(dataset),
        seed: config.seed,
        validation_fraction: config.validation_fraction,
        tikhonov: config.tikhonov,
        converged: optimizer.converged,
        iterations: optimizer.iterations,
        objective: optimizer.objective,
        train_rmse,
        validation_rmse,
        excluded,
        penalized,
        train_groups: split.train,
        validation_groups: split.validation,
        parameters: dataset
            .parameter_names
            .iter()
            .cloned()
            .zip(optimizer.parameters.iter().copied())
            .collect(),
    };
    Ok(FitResult { optimizer, record })
}

/// Required self-test: recover known `c` and packed symmetric `B` parameters.
pub fn run_synthetic_recovery_gate() -> Result<FitResult> {
    let truth = [0.35, -0.8, -0.4, 2.0_f64.sqrt() * 0.2, 0.6];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut data = Vec::new();
    for group in 0..20 {
        for row in 0..4 {
            let mut features = Vec::new();
            for _ in 0..truth.len() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                features.push(((state >> 11) as f64 / ((1_u64 << 53) as f64)) * 2.0 - 1.0);
            }
            let target = dot(&features, &truth);
            data.push(ReferenceDatum {
                case_id: format!("g{group}-r{row}"),
                group_id: format!("geometry-{group:02}"),
                observable: ObservableKind::Density,
                target,
                scale: 1.0,
                weight: 1.0,
                linear_features: features,
                diagnostics: Some(Diagnostics {
                    aux_population: 1.0,
                    min_b_eigval: -0.45,
                    min_s_eigval: 0.2,
                    lowest_mo: -0.7,
                }),
            });
        }
    }
    let dataset = FitDataset {
        level: "synthetic/phase4-fitter-gate".to_string(),
        dataset_id: "known-c-and-b-v1".to_string(),
        parameter_names: vec!["c0", "c1", "B00", "sqrt2_B01", "B11"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        data,
    };
    let mut config = FitConfig::for_parameter_count(truth.len());
    config.tikhonov = 0.0;
    config.grad_tol = 1e-12;
    config.spectral_blocks.push(SpectralBlock {
        offset: 2,
        dimension: 2,
        min_eigenvalue: -2.0,
    });
    let result = fit_reference_set(&dataset, &config)?;
    let worst = result
        .optimizer
        .parameters
        .iter()
        .zip(truth)
        .map(|(value, expected)| (value - expected).abs())
        .fold(0.0_f64, f64::max);
    if !result.optimizer.converged || worst >= 1e-6 || result.record.validation_rmse >= 1e-6 {
        return Err(DftError::LinearAlgebra(format!(
            "pseudo-substituent fitter synthetic recovery failed: converged={}, parameter error={worst:.3e}, validation RMSE={:.3e}",
            result.optimizer.converged, result.record.validation_rmse
        )));
    }
    Ok(result)
}

fn validate_dataset(dataset: &FitDataset, config: &FitConfig) -> Result<()> {
    let n = dataset.parameter_names.len();
    if dataset.level.trim().is_empty() || dataset.dataset_id.trim().is_empty() || n == 0 {
        return Err(DftError::InvalidInput(
            "fit dataset needs nonempty level, dataset_id, and parameters".to_string(),
        ));
    }
    if config.prior.len() != n || config.parameter_scales.len() != n {
        return Err(DftError::InvalidInput(
            "fit prior/parameter scale dimension mismatch".to_string(),
        ));
    }
    if !(config.tikhonov >= 0.0
        && config.tikhonov.is_finite()
        && (0.0..=1.0).contains(&config.max_excluded_fraction)
        && config.max_excluded_fraction.is_finite())
    {
        return Err(DftError::InvalidInput(
            "invalid Tikhonov or diagnostic exclusion configuration".to_string(),
        ));
    }
    if config
        .parameter_scales
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || config.prior.iter().any(|value| !value.is_finite())
    {
        return Err(DftError::InvalidInput(
            "fit prior must be finite and parameter scales positive".to_string(),
        ));
    }
    let mut names = BTreeSet::new();
    if dataset
        .parameter_names
        .iter()
        .any(|name| name.trim().is_empty() || !names.insert(name))
    {
        return Err(DftError::InvalidInput(
            "fit parameter names must be nonempty and unique".to_string(),
        ));
    }
    for (index, row) in dataset.data.iter().enumerate() {
        if row.case_id.trim().is_empty()
            || row.group_id.trim().is_empty()
            || row.observable.name().trim().is_empty()
            || !row.target.is_finite()
            || !row.scale.is_finite()
            || row.scale <= 0.0
            || !row.weight.is_finite()
            || row.weight < 0.0
            || row.linear_features.len() != n
            || row.linear_features.iter().any(|value| !value.is_finite())
            || row.diagnostics.is_some_and(|value| !value.is_finite())
        {
            return Err(DftError::InvalidInput(format!(
                "invalid pseudo-substituent fit datum at index {index}"
            )));
        }
    }
    for value in [
        config.diagnostic_limits.aux_population,
        config.diagnostic_limits.min_b_eigval,
        config.diagnostic_limits.min_s_eigval,
        config.diagnostic_limits.lowest_mo,
    ]
    .into_iter()
    .flatten()
    {
        if !value.is_finite() {
            return Err(DftError::InvalidInput(
                "diagnostic limits must be finite".to_string(),
            ));
        }
    }
    if let DiagnosticPolicy::Penalize { loss } = config.diagnostic_policy {
        if !loss.is_finite() || loss < 0.0 {
            return Err(DftError::InvalidInput(
                "diagnostic penalty must be finite and nonnegative".to_string(),
            ));
        }
    }
    validate_spectral_blocks(n, &config.spectral_blocks)
}

fn validate_spectral_blocks(n_parameters: usize, blocks: &[SpectralBlock]) -> Result<()> {
    let mut occupied = vec![false; n_parameters];
    for block in blocks {
        if block.dimension == 0 || !block.min_eigenvalue.is_finite() {
            return Err(DftError::InvalidInput(
                "spectral blocks need positive dimension and finite bound".to_string(),
            ));
        }
        let end = block
            .offset
            .checked_add(block.packed_len())
            .ok_or_else(|| DftError::InvalidInput("spectral block overflow".to_string()))?;
        if end > n_parameters || occupied[block.offset..end].iter().any(|used| *used) {
            return Err(DftError::InvalidInput(
                "spectral block is out of range or overlaps another block".to_string(),
            ));
        }
        occupied[block.offset..end].fill(true);
    }
    Ok(())
}

fn select_rows(
    dataset: &FitDataset,
    config: &FitConfig,
    train_groups: &BTreeSet<&str>,
    validation_groups: &BTreeSet<&str>,
) -> Result<(Vec<usize>, usize, usize, f64)> {
    let mut indices: Vec<usize> = (0..dataset.data.len()).collect();
    indices.sort_by_key(|&index| {
        let row = &dataset.data[index];
        (
            row.group_id.as_str(),
            row.case_id.as_str(),
            row.observable.clone(),
        )
    });
    let mut active = Vec::new();
    let mut excluded = 0;
    let mut penalized = 0;
    let mut penalty_loss = 0.0;
    for index in indices {
        let row = &dataset.data[index];
        if !train_groups.contains(row.group_id.as_str())
            && !validation_groups.contains(row.group_id.as_str())
        {
            continue;
        }
        let passes = if config.diagnostic_limits.enabled() {
            row.diagnostics
                .is_some_and(|value| config.diagnostic_limits.accepts(value))
        } else {
            true
        };
        if passes {
            active.push(index);
        } else {
            match config.diagnostic_policy {
                DiagnosticPolicy::Exclude => excluded += 1,
                DiagnosticPolicy::Penalize { loss } => {
                    penalized += 1;
                    penalty_loss += loss;
                    active.push(index);
                }
            }
        }
    }
    if !dataset.data.is_empty()
        && excluded as f64 / dataset.data.len() as f64 > config.max_excluded_fraction
    {
        return Err(DftError::InvalidInput(format!(
            "diagnostic exclusion fraction {:.3} exceeds configured maximum {:.3}",
            excluded as f64 / dataset.data.len() as f64,
            config.max_excluded_fraction
        )));
    }
    Ok((active, excluded, penalized, penalty_loss))
}

fn weighted_ridge_initial_guess(
    dataset: &FitDataset,
    config: &FitConfig,
    rows: &[usize],
) -> Result<Vec<f64>> {
    let n = dataset.parameter_names.len();
    let mut gram = Matrix::zeros(n, n);
    let mut gram_compensation = Matrix::zeros(n, n);
    let mut rhs = vec![0.0; n];
    let mut rhs_compensation = vec![0.0; n];
    for &index in rows {
        let row = &dataset.data[index];
        let w = row.weight / (row.scale * row.scale);
        if w == 0.0 {
            continue;
        }
        for i in 0..n {
            kahan_add(
                &mut rhs[i],
                &mut rhs_compensation[i],
                w * row.linear_features[i] * row.target,
            );
            for j in i..n {
                let flat = i * n + j;
                kahan_add(
                    &mut gram.as_mut_slice()[flat],
                    &mut gram_compensation.as_mut_slice()[flat],
                    w * row.linear_features[i] * row.linear_features[j],
                );
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            gram[(i, j)] = gram[(j, i)];
        }
        let regularization = config.tikhonov / config.parameter_scales[i].powi(2);
        gram[(i, i)] += regularization;
        rhs[i] += regularization * config.prior[i];
    }
    let (eigenvalues, vectors) = symmetric_eigen(&gram)?;
    let largest = eigenvalues.last().copied().unwrap_or(0.0).abs();
    let threshold = (largest * 1e-13).max(1e-15);
    if eigenvalues.first().copied().unwrap_or(0.0) <= threshold {
        return Err(DftError::LinearAlgebra(format!(
            "pseudo-substituent fitter normal matrix is rank deficient (min={:.3e}, max={largest:.3e})",
            eigenvalues.first().copied().unwrap_or(0.0)
        )));
    }
    let mut modal = vec![0.0; n];
    for k in 0..n {
        modal[k] = (0..n).map(|i| vectors[(i, k)] * rhs[i]).sum::<f64>() / eigenvalues[k];
    }
    let mut solution = vec![0.0; n];
    for i in 0..n {
        solution[i] = (0..n).map(|k| vectors[(i, k)] * modal[k]).sum();
    }
    Ok(solution)
}

fn quadratic_objective(
    dataset: &FitDataset,
    config: &FitConfig,
    rows: &[usize],
    parameters: &[f64],
    penalty_loss: f64,
) -> Result<(f64, Vec<f64>)> {
    if parameters.len() != dataset.parameter_names.len() {
        return Err(DftError::InvalidInput(
            "fit objective parameter dimension mismatch".to_string(),
        ));
    }
    let mut value = penalty_loss;
    let mut gradient = vec![0.0; parameters.len()];
    for &index in rows {
        let row = &dataset.data[index];
        let residual = (dot(&row.linear_features, parameters) - row.target) / row.scale;
        value += 0.5 * row.weight * residual * residual;
        let factor = row.weight * residual / row.scale;
        for (gradient, feature) in gradient.iter_mut().zip(&row.linear_features) {
            *gradient += factor * feature;
        }
    }
    for i in 0..parameters.len() {
        let scaled = (parameters[i] - config.prior[i]) / config.parameter_scales[i];
        value += 0.5 * config.tikhonov * scaled * scaled;
        gradient[i] += config.tikhonov * (parameters[i] - config.prior[i])
            / config.parameter_scales[i].powi(2);
    }
    if !value.is_finite() || gradient.iter().any(|entry| !entry.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "non-finite pseudo-substituent fit objective".to_string(),
        ));
    }
    Ok((value, gradient))
}

fn weighted_rmse(dataset: &FitDataset, rows: &[usize], parameters: &[f64]) -> Result<f64> {
    let mut sum = 0.0;
    let mut weight = 0.0;
    for &index in rows {
        let row = &dataset.data[index];
        let residual = (dot(&row.linear_features, parameters) - row.target) / row.scale;
        sum += row.weight * residual * residual;
        weight += row.weight;
    }
    if weight == 0.0 {
        return Err(DftError::InvalidInput(
            "no positive validation weight remains".to_string(),
        ));
    }
    Ok((sum / weight).sqrt())
}

fn checked_evaluation<F>(evaluate: &mut F, parameters: &[f64]) -> Result<(f64, Vec<f64>)>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
{
    let (value, gradient) = evaluate(parameters)?;
    if gradient.len() != parameters.len()
        || !value.is_finite()
        || gradient.iter().any(|entry| !entry.is_finite())
    {
        return Err(DftError::LinearAlgebra(
            "projected L-BFGS objective returned non-finite or mismatched data".to_string(),
        ));
    }
    Ok((value, gradient))
}

fn gradient_mapping_max<P>(x: &[f64], gradient: &[f64], project: &mut P) -> Result<f64>
where
    P: FnMut(&mut [f64]) -> Result<usize>,
{
    let mut mapped: Vec<f64> = x.iter().zip(gradient).map(|(a, g)| a - g).collect();
    project(&mut mapped)?;
    Ok(x.iter()
        .zip(mapped)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max))
}

fn dataset_fingerprint(dataset: &FitDataset) -> u64 {
    let mut hash = stable_hash(0, dataset.level.as_bytes());
    hash = stable_hash(hash, dataset.dataset_id.as_bytes());
    for name in &dataset.parameter_names {
        hash = stable_hash(hash, name.as_bytes());
    }
    let mut rows: Vec<&ReferenceDatum> = dataset.data.iter().collect();
    rows.sort_by_key(|row| (&row.group_id, &row.case_id, &row.observable));
    for row in rows {
        for text in [&row.group_id, &row.case_id] {
            hash = stable_hash(hash, text.as_bytes());
        }
        hash = stable_hash(hash, row.observable.name().as_bytes());
        for value in [row.target, row.scale, row.weight]
            .into_iter()
            .chain(row.linear_features.iter().copied())
        {
            hash = stable_hash(hash, &value.to_bits().to_le_bytes());
        }
    }
    hash
}

fn stable_hash(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in seed.to_le_bytes().iter().chain(bytes) {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn validate_record_token(what: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() || value.contains(['\n', '\r', ',', '=']) {
        Err(DftError::InvalidInput(format!(
            "fit record {what} is empty or contains a reserved character"
        )))
    } else {
        Ok(())
    }
}

fn take(fields: &mut BTreeMap<String, String>, key: &str) -> Result<String> {
    fields
        .remove(key)
        .ok_or_else(|| DftError::InvalidInput(format!("fit record is missing `{key}`")))
}

fn parse_field<T: std::str::FromStr>(
    fields: &mut BTreeMap<String, String>,
    key: &str,
) -> Result<T> {
    take(fields, key)?.parse().map_err(|_| {
        DftError::InvalidInput(format!("fit record field `{key}` has an invalid value"))
    })
}

fn parse_float_field(fields: &mut BTreeMap<String, String>, key: &str) -> Result<f64> {
    let value = parse_field::<f64>(fields, key)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(DftError::InvalidInput(format!(
            "fit record field `{key}` is non-finite"
        )))
    }
}

fn parse_finite(value: &str, line: usize) -> Result<f64> {
    let parsed = value.parse::<f64>().map_err(|_| DftError::Parse {
        line,
        message: format!("invalid floating-point value `{value}`"),
    })?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(DftError::Parse {
            line,
            message: "non-finite floating-point value".to_string(),
        })
    }
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}

fn add_scaled(x: &[f64], direction: &[f64], step: f64) -> Vec<f64> {
    x.iter()
        .zip(direction)
        .map(|(value, delta)| value + step * delta)
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
    for (target, value) in y.iter_mut().zip(x) {
        *target += alpha * value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_known_parameters_are_recovered_to_one_part_per_million() {
        let fit = run_synthetic_recovery_gate().unwrap();
        assert!(fit.optimizer.converged);
        assert!(fit.record.train_rmse < 1e-6);
        assert!(fit.record.validation_rmse < 1e-6);
        assert_eq!(fit.record.parameters.len(), 5);
    }

    #[test]
    fn group_split_is_order_independent_and_has_no_leakage() {
        let mut rows = Vec::new();
        for group in 0..10 {
            for row in 0..3 {
                rows.push(ReferenceDatum {
                    case_id: format!("{group}-{row}"),
                    group_id: format!("g{group}"),
                    observable: ObservableKind::Esp,
                    target: 0.0,
                    scale: 1.0,
                    weight: 1.0,
                    linear_features: vec![1.0],
                    diagnostics: None,
                });
            }
        }
        let first = deterministic_group_split(&rows, 42, 0.2).unwrap();
        rows.reverse();
        let second = deterministic_group_split(&rows, 42, 0.2).unwrap();
        assert_eq!(first, second);
        let train: BTreeSet<_> = first.train.iter().collect();
        assert!(first.validation.iter().all(|group| !train.contains(group)));
        assert_eq!(first.validation.len(), 2);
    }

    #[test]
    fn spectral_projection_is_bounded_and_idempotent() {
        let block = SpectralBlock {
            offset: 1,
            dimension: 2,
            min_eigenvalue: -2.0,
        };
        let mut parameters = vec![9.0, -5.0, 2.0_f64.sqrt(), 1.0];
        assert!(project_spectral_blocks(&mut parameters, &[block]).unwrap() > 0);
        let once = parameters.clone();
        project_spectral_blocks(&mut parameters, &[block]).unwrap();
        assert!(parameters
            .iter()
            .zip(once)
            .all(|(a, b)| (a - b).abs() < 1e-12));
        let matrix = Matrix::from_row_major(
            2,
            2,
            vec![
                parameters[1],
                parameters[2] / 2.0_f64.sqrt(),
                parameters[2] / 2.0_f64.sqrt(),
                parameters[3],
            ],
        );
        assert!(symmetric_eigen(&matrix).unwrap().0[0] >= -2.0 - 1e-12);
    }

    #[test]
    fn quadratic_gradient_matches_independent_finite_difference() {
        let dataset = FitDataset {
            level: "test".to_string(),
            dataset_id: "gradient".to_string(),
            parameter_names: vec!["a".to_string(), "b".to_string()],
            data: vec![ReferenceDatum {
                case_id: "case".to_string(),
                group_id: "group".to_string(),
                observable: ObservableKind::BondLength,
                target: 0.7,
                scale: 0.4,
                weight: 1.3,
                linear_features: vec![0.2, -1.1],
                diagnostics: None,
            }],
        };
        let mut config = FitConfig::for_parameter_count(2);
        config.tikhonov = 0.17;
        config.prior = vec![0.1, -0.2];
        config.parameter_scales = vec![0.8, 1.7];
        let x = vec![0.4, 0.3];
        let (_, analytic) = quadratic_objective(&dataset, &config, &[0], &x, 0.0).unwrap();
        for axis in 0..2 {
            let h = 1e-6;
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[axis] += h;
            minus[axis] -= h;
            let ep = quadratic_objective(&dataset, &config, &[0], &plus, 0.0)
                .unwrap()
                .0;
            let em = quadratic_objective(&dataset, &config, &[0], &minus, 0.0)
                .unwrap()
                .0;
            assert!((analytic[axis] - (ep - em) / (2.0 * h)).abs() < 1e-9);
        }
    }

    #[test]
    fn projected_lbfgs_reaches_a_known_active_bound() {
        let result = projected_lbfgs(
            &[3.0, -4.0],
            5,
            80,
            1e-11,
            1e-12,
            |x| {
                let dx = x[0] - 2.0;
                let dy = x[1] + 1.0;
                Ok((0.5 * dx * dx + 2.0 * dy * dy, vec![dx, 4.0 * dy]))
            },
            |x| {
                // The unconstrained optimum x=2 is outside x<=0.5; y=-1 is free.
                x[0] = x[0].min(0.5);
                Ok(usize::from(x[0] == 0.5))
            },
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert!((result.parameters[0] - 0.5).abs() < 1e-10);
        assert!((result.parameters[1] + 1.0).abs() < 1e-10);
        assert!(result.projected_gradient_max < 1e-11);
    }

    #[test]
    fn projected_lbfgs_backtracks_after_an_unconverged_trial() {
        let mut failed_trials = 0;
        let result = projected_lbfgs(
            &[1.0],
            5,
            10,
            1e-11,
            1e-12,
            |x| {
                // The first full step reaches -0.5; its electronic solve
                // fails. The existing half step reaches the exact optimum.
                if x[0] < 0.0 {
                    failed_trials += 1;
                    return Err(DftError::ScfNotConverged {
                        iterations: 300,
                        error: 0.1,
                    });
                }
                let delta = x[0] - 0.25;
                Ok((delta * delta, vec![2.0 * delta]))
            },
            |_| Ok(0),
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert_eq!(failed_trials, 1);
        assert_eq!(result.evaluations, 3);
        assert_eq!(result.parameters, vec![0.25]);
        assert_eq!(result.objective, 0.0);
        assert_eq!(result.projected_gradient_max, 0.0);
    }

    #[test]
    fn projected_lbfgs_preserves_initial_and_non_scf_errors() {
        let initial_error = projected_lbfgs(
            &[1.0],
            5,
            10,
            1e-11,
            1e-12,
            |_| {
                Err(DftError::ScfNotConverged {
                    iterations: 300,
                    error: 0.1,
                })
            },
            |_| Ok(0),
        )
        .unwrap_err();
        assert!(matches!(
            initial_error,
            DftError::ScfNotConverged {
                iterations: 300,
                ..
            }
        ));
        let trial_error = projected_lbfgs(
            &[1.0],
            5,
            10,
            1e-11,
            1e-12,
            |x| {
                if x[0] < 0.0 {
                    return Err(DftError::InvalidInput("invalid physical model".into()));
                }
                let delta = x[0] - 0.25;
                Ok((delta * delta, vec![2.0 * delta]))
            },
            |_| Ok(0),
        )
        .unwrap_err();
        assert!(matches!(trial_error, DftError::InvalidInput(_)));
    }

    #[test]
    fn projected_lbfgs_bounds_repeated_unconverged_trials() {
        let mut failed_trials = 0;
        let result = projected_lbfgs(
            &[1.0],
            5,
            10,
            1e-11,
            0.125,
            |x| {
                if x[0] != 1.0 {
                    failed_trials += 1;
                    return Err(DftError::ScfNotConverged {
                        iterations: 300,
                        error: 0.1,
                    });
                }
                Ok((0.5625, vec![1.5]))
            },
            |_| Ok(0),
        )
        .unwrap();
        assert!(!result.converged);
        assert_eq!(failed_trials, 4);
        assert_eq!(result.evaluations, 5);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.parameters, vec![1.0]);
        assert_eq!(result.objective, 0.5625);
    }

    #[test]
    fn diagnostic_exclusion_limit_is_enforced() {
        let mut dataset = FitDataset {
            level: "test".to_string(),
            dataset_id: "diagnostic".to_string(),
            parameter_names: vec!["a".to_string()],
            data: Vec::new(),
        };
        for group in 0..2 {
            dataset.data.push(ReferenceDatum {
                case_id: format!("c{group}"),
                group_id: format!("g{group}"),
                observable: ObservableKind::Density,
                target: 1.0,
                scale: 1.0,
                weight: 1.0,
                linear_features: vec![1.0],
                diagnostics: Some(Diagnostics {
                    aux_population: 0.0,
                    min_b_eigval: 0.0,
                    min_s_eigval: 0.0,
                    lowest_mo: 0.0,
                }),
            });
        }
        let mut config = FitConfig::for_parameter_count(1);
        config.diagnostic_limits.aux_population = Some(0.5);
        config.max_excluded_fraction = 0.4;
        let error = fit_reference_set(&dataset, &config).unwrap_err();
        assert!(error.to_string().contains("exclusion fraction"));
    }

    #[test]
    fn candidate_record_round_trips_and_checks_level() {
        let result = run_synthetic_recovery_gate().unwrap();
        let text = result.record.to_text().unwrap();
        let parsed = FitParameterRecord::from_text(&text).unwrap();
        assert_eq!(parsed, result.record);
        parsed
            .validate_level("synthetic/phase4-fitter-gate")
            .unwrap();
        assert!(parsed.validate_level("different-level").is_err());
        assert!(text.contains("status=candidate-unaccepted"));
    }
}
