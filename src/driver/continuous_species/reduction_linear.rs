// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Accounting and fixed-density projection diagnostics for fragment reduction.
//! These routines generate candidates, not validated chemical substituents.
//! The linear-data representation is shared with `pseudo_sub::fit`; the
//! equality-constrained Gram solve adds no observable optimizer or PSD rule.

use crate::linalg::{solve_linear_matrix, symmetric_eigen, Matrix, Orthogonalizer};
use crate::pseudo_sub::fit::{FitDataset, ObservableKind, ReferenceDatum};
use crate::{DftError, Result};
use std::collections::BTreeSet;

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

fn finite_matrix(matrix: &Matrix) -> bool {
    matrix.as_slice().iter().all(|x| x.is_finite())
}

fn max_abs(matrix: &Matrix) -> f64 {
    matrix
        .as_slice()
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b.abs()))
}

fn norm(matrix: &Matrix) -> f64 {
    matrix.as_slice().iter().fold(0.0_f64, |a, &b| a.hypot(b))
}

fn symmetric(matrix: &Matrix) -> bool {
    if matrix.rows != matrix.cols || !finite_matrix(matrix) {
        return false;
    }
    let threshold = 64.0 * f64::EPSILON * (1.0 + max_abs(matrix));
    (0..matrix.rows).all(|i| (0..i).all(|j| (matrix[(i, j)] - matrix[(j, i)]).abs() <= threshold))
}

fn block(matrix: &Matrix, rows: &[usize], columns: &[usize]) -> Matrix {
    let mut result = Matrix::zeros(rows.len(), columns.len());
    for (i, &row) in rows.iter().enumerate() {
        for (j, &column) in columns.iter().enumerate() {
            result[(i, j)] = matrix[(row, column)];
        }
    }
    result
}

/// The frozen count is an explicit occupied-subspace choice, never inferred
/// from deleted AOs. Atom indices also detect overlap between frozen groups.
#[derive(Clone, Debug)]
pub struct FrozenFragmentSpec {
    pub explicit_atoms: Vec<usize>,
    pub frozen_electrons: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentElectronLedger {
    pub explicit_atoms: Vec<usize>,
    pub nuclear_charge: i64,
    pub frozen_electrons: usize,
    pub effective_ionic_charge: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElectronLedger {
    pub full_nuclear_charge: i64,
    pub retained_nuclear_charge: i64,
    pub full_electron_count: usize,
    pub frozen_electron_count: usize,
    pub active_electron_count: usize,
    pub active_ionic_charge: i64,
    pub full_total_charge: i64,
    pub reduced_total_charge: i64,
    pub fragments: Vec<FragmentElectronLedger>,
}

impl ElectronLedger {
    pub fn new(
        charges: &[u8],
        full_electrons: usize,
        fragments: &[FrozenFragmentSpec],
    ) -> Result<Self> {
        if charges.is_empty() || charges.contains(&0) {
            return invalid("electron accounting requires explicit positive nuclear charges");
        }
        let integer = |n: usize| {
            i64::try_from(n)
                .map_err(|_| DftError::InvalidInput("electron accounting integer overflow".into()))
        };
        let full_electron_integer = integer(full_electrons)?;
        let total = charges
            .iter()
            .try_fold(0i64, |sum, &z| sum.checked_add(i64::from(z)))
            .ok_or_else(|| DftError::InvalidInput("nuclear charge accounting overflow".into()))?;
        let mut used = BTreeSet::new();
        let mut records = Vec::with_capacity(fragments.len());
        let mut frozen = 0usize;
        let mut removed_charge = 0i64;
        for fragment in fragments {
            if fragment.explicit_atoms.is_empty() {
                return invalid("a frozen fragment needs explicit atom indices");
            }
            let mut nuclear = 0i64;
            for &atom in &fragment.explicit_atoms {
                if atom >= charges.len() || !used.insert(atom) {
                    return invalid(
                        "frozen fragment atom indices must be valid, distinct and disjoint",
                    );
                }
                nuclear += i64::from(charges[atom]);
            }
            frozen = frozen
                .checked_add(fragment.frozen_electrons)
                .ok_or_else(|| DftError::InvalidInput("frozen electron count overflow".into()))?;
            removed_charge += nuclear;
            records.push(FragmentElectronLedger {
                explicit_atoms: fragment.explicit_atoms.clone(),
                nuclear_charge: nuclear,
                frozen_electrons: fragment.frozen_electrons,
                effective_ionic_charge: nuclear - integer(fragment.frozen_electrons)?,
            });
        }
        let active = full_electrons.checked_sub(frozen).ok_or_else(|| {
            DftError::InvalidInput("frozen electrons exceed the full-system electron count".into())
        })?;
        let retained = total - removed_charge;
        let ionic = retained
            + records
                .iter()
                .map(|r| r.effective_ionic_charge)
                .sum::<i64>();
        let full_charge = total - full_electron_integer;
        let reduced_charge = ionic - integer(active)?;
        if full_charge != reduced_charge {
            return invalid("inconsistent fragment electron and ionic charge accounting");
        }
        Ok(Self {
            full_nuclear_charge: total,
            retained_nuclear_charge: retained,
            full_electron_count: full_electrons,
            frozen_electron_count: frozen,
            active_electron_count: active,
            active_ionic_charge: ionic,
            full_total_charge: full_charge,
            reduced_total_charge: reduced_charge,
            fragments: records,
        })
    }

    /// Validate only the active trace. No idempotency condition is imposed on
    /// a projected or partial density block from a different orbital space.
    pub fn check_active_trace(
        &self,
        density: &Matrix,
        overlap: &Matrix,
        tolerance: f64,
    ) -> Result<f64> {
        if !symmetric(density)
            || !symmetric(overlap)
            || density.rows != overlap.rows
            || !tolerance.is_finite()
            || tolerance <= 0.0
        {
            return invalid("active density trace requires finite same-size symmetric matrices and a positive tolerance");
        }
        let trace = density.frobenius_dot(overlap);
        if (trace - self.active_electron_count as f64).abs() > tolerance {
            return invalid(format!(
                "active density trace {trace:.16e} differs from fixed electron count {}",
                self.active_electron_count
            ));
        }
        Ok(trace)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemplateDimensions {
    pub centers: usize,
    pub ao_functions: usize,
    pub local_functions: usize,
    pub projector_functions: usize,
    pub geometry_functions: usize,
    pub parameters: usize,
}

#[derive(Clone, Debug)]
pub struct FixedReductionTemplate {
    pub id: String,
    pub version: u32,
    pub dimensions: TemplateDimensions,
}

/// Whole-SCF chart sector. Its AO count includes the retained scaffold and
/// must not be confused with a fragment template's own AO count.
#[derive(Clone, Copy, Debug)]
pub struct FixedElectronicSector {
    pub active_electrons: usize,
    pub spin_electrons: [usize; 2],
    pub total_ao_functions: usize,
}

impl FixedReductionTemplate {
    /// AO and projector counts remain separate; no chemical name changes them.
    pub fn validate(&self, actual: TemplateDimensions) -> Result<()> {
        if self.id.trim().is_empty()
            || self.version == 0
            || !(1..=3).contains(&self.dimensions.centers)
            || self.dimensions.ao_functions == 0
            || actual != self.dimensions
        {
            return invalid(
                "reduced model disagrees with its fixed template dimensions or version",
            );
        }
        Ok(())
    }
}

impl FixedElectronicSector {
    pub fn validate(
        &self,
        active: usize,
        spin: [usize; 2],
        total_ao_functions: usize,
    ) -> Result<()> {
        if active != self.active_electrons
            || spin != self.spin_electrons
            || total_ao_functions == 0
            || total_ao_functions != self.total_ao_functions
            || spin[0].checked_add(spin[1]) != Some(active)
            || spin.iter().any(|&n| n > total_ao_functions)
        {
            return invalid("reduced SCF disagrees with its fixed active electron number, spin sector or total AO template");
        }
        Ok(())
    }
}

/// One target and dictionary evaluated in the same AO metric and geometry.
pub struct OperatorProjectionSample {
    pub case_id: String,
    pub group_id: String,
    pub target: Matrix,
    pub dictionary: Vec<Matrix>,
    /// Symmetric positive-definite W in ||W (V_target-V_model) W||_F.
    pub metric_weight: Matrix,
    pub weight: f64,
}

/// Reuse the existing deterministic fitter's data/provenance format. Packing
/// the upper triangle with sqrt(2) off-diagonal weights preserves the full
/// Frobenius norm. No training/hold-out split is chosen inside this routine.
pub fn operator_projection_dataset(
    samples: &[OperatorProjectionSample],
    parameter_names: Vec<String>,
    level: String,
    dataset_id: String,
) -> Result<FitDataset> {
    let p = parameter_names.len();
    if p == 0
        || samples.is_empty()
        || level.trim().is_empty()
        || dataset_id.trim().is_empty()
        || parameter_names.iter().any(|s| s.trim().is_empty())
        || parameter_names.iter().collect::<BTreeSet<_>>().len() != p
    {
        return invalid(
            "operator projection requires unique named parameters and explicit reference samples",
        );
    }
    let mut data = Vec::new();
    for sample in samples {
        let n = sample.target.rows;
        if n == 0
            || !symmetric(&sample.target)
            || sample.dictionary.len() != p
            || sample
                .dictionary
                .iter()
                .any(|m| m.rows != n || !symmetric(m))
            || sample.metric_weight.rows != n
            || !symmetric(&sample.metric_weight)
            || sample.case_id.trim().is_empty()
            || sample.group_id.trim().is_empty()
            || !sample.weight.is_finite()
            || sample.weight <= 0.0
        {
            return invalid("operator projection samples require compatible finite Hermitian matrices, positive weights and explicit provenance");
        }
        let (metric_values, _) = symmetric_eigen(&sample.metric_weight)?;
        if metric_values.iter().any(|&v| v <= 0.0) {
            return invalid("operator comparison metric weight must be positive definite");
        }
        let transform = |m: &Matrix| {
            sample
                .metric_weight
                .matmul_seq(m)
                .matmul_seq(&sample.metric_weight)
        };
        let target = transform(&sample.target);
        let dictionary: Vec<_> = sample.dictionary.iter().map(transform).collect();
        if !finite_matrix(&target) || dictionary.iter().any(|m| !finite_matrix(m)) {
            return invalid("metric-transformed projection operators overflowed");
        }
        for i in 0..n {
            for j in i..n {
                let factor = if i == j { 1.0 } else { 2.0_f64.sqrt() };
                data.push(ReferenceDatum {
                    case_id: format!("{}[{i},{j}]", sample.case_id),
                    group_id: sample.group_id.clone(),
                    observable: ObservableKind::Other("metric-operator-entry".into()),
                    target: factor * target[(i, j)],
                    scale: 1.0,
                    weight: sample.weight,
                    linear_features: dictionary.iter().map(|m| factor * m[(i, j)]).collect(),
                    diagnostics: None,
                });
            }
        }
    }
    Ok(FitDataset {
        level,
        dataset_id,
        parameter_names,
        data,
    })
}

pub struct ProjectionConstraints {
    pub matrix: Matrix,
    pub values: Vec<f64>,
    pub prior: Vec<f64>,
    pub regularization_operator: Matrix,
    pub regularization_strength: f64,
    pub constraint_absolute_tolerance: f64,
    /// Relative eigenvalue threshold for constraint and reduced Gram ranks.
    pub relative_rank_tolerance: f64,
}

pub struct ConstrainedProjection {
    pub coefficients: Vec<f64>,
    pub gram: Matrix,
    pub constraint_rank: usize,
    pub free_parameters: usize,
    /// Rank of the data Gram restricted to the constraint nullspace.
    pub data_rank: usize,
    pub regularized_rank: usize,
    pub data_unique: bool,
    pub regularized_unique: bool,
    /// Condition numbers of the nullspace-restricted Gram matrices, not of
    /// the unsquared design matrix. None denotes numerical nonuniqueness.
    pub data_condition_number: Option<f64>,
    pub regularized_condition_number: Option<f64>,
    pub weighted_residual_squared: f64,
    pub regularization_penalty: f64,
    pub constraint_max_residual: f64,
    pub projected_stationarity_residual: f64,
}

fn spectral_rank(values: &[f64], tolerance: f64) -> Result<(usize, Option<f64>)> {
    let maximum = values.iter().copied().fold(0.0_f64, f64::max);
    if values
        .iter()
        .any(|v| !v.is_finite() || *v < -tolerance * maximum)
    {
        return invalid("a projection Gram matrix is not numerically positive semidefinite");
    }
    let positive: Vec<_> = values
        .iter()
        .copied()
        .filter(|v| *v > tolerance * maximum)
        .collect();
    let condition = if positive.len() == values.len() {
        if values.is_empty() {
            Some(1.0)
        } else {
            Some(maximum / positive[0])
        }
    } else {
        None
    };
    Ok((positive.len(), condition))
}

/// Exact linear constrained least squares in the existing reference-data
/// convention. Rank-deficient solutions are explicitly minimum-norm and
/// nonunique; a ridge never changes the reported unregularized data rank.
/// Every supplied row is used: callers choose construction states explicitly,
/// and no diagnostic exclusion or training/hold-out split is hidden here.
pub fn constrained_projection(
    dataset: &FitDataset,
    config: &ProjectionConstraints,
) -> Result<ConstrainedProjection> {
    let p = dataset.parameter_names.len();
    let c = &config.matrix;
    let l = &config.regularization_operator;
    let tolerance = config.relative_rank_tolerance;
    if p == 0
        || dataset.data.is_empty()
        || dataset.level.trim().is_empty()
        || dataset.dataset_id.trim().is_empty()
        || dataset
            .parameter_names
            .iter()
            .any(|name| name.trim().is_empty())
        || dataset
            .parameter_names
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != p
        || c.cols != p
        || config.values.len() != c.rows
        || config.prior.len() != p
        || l.cols != p
        || !finite_matrix(c)
        || !finite_matrix(l)
        || config
            .values
            .iter()
            .chain(&config.prior)
            .any(|v| !v.is_finite())
        || !config.regularization_strength.is_finite()
        || config.regularization_strength < 0.0
        || !config.constraint_absolute_tolerance.is_finite()
        || config.constraint_absolute_tolerance <= 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || tolerance >= 1.0
    {
        return invalid(
            "invalid constrained-projection dimensions, regularization or rank tolerance",
        );
    }
    let mut gram = Matrix::zeros(p, p);
    let mut rhs = vec![0.0; p];
    for datum in &dataset.data {
        if datum.linear_features.len() != p
            || datum.linear_features.iter().any(|v| !v.is_finite())
            || !datum.target.is_finite()
            || !datum.scale.is_finite()
            || datum.scale <= 0.0
            || !datum.weight.is_finite()
            || datum.weight <= 0.0
        {
            return invalid(
                "invalid finite linear reference row or normalization in constrained projection",
            );
        }
        let weight = datum.weight / datum.scale.powi(2);
        if !weight.is_finite() || weight <= 0.0 {
            return invalid("reference normalization overflows its finite projection weight");
        }
        for a in 0..p {
            rhs[a] += weight * datum.linear_features[a] * datum.target;
            for b in 0..p {
                gram[(a, b)] += weight * datum.linear_features[a] * datum.linear_features[b];
            }
        }
    }
    let ct = c.transpose();
    let constraint_gram = ct.matmul_seq(c);
    if !finite_matrix(&constraint_gram)
        || !finite_matrix(&gram)
        || rhs.iter().any(|v| !v.is_finite())
    {
        return invalid("projection Gram or right-hand side overflowed");
    }
    let (ce, cv) = symmetric_eigen(&constraint_gram)?;
    let (constraint_rank, _) = spectral_rank(&ce, tolerance)?;
    let maximum = ce.iter().copied().fold(0.0_f64, f64::max);
    let ctd: Vec<_> = (0..p)
        .map(|a| {
            (0..c.rows)
                .map(|i| c[(i, a)] * config.values[i])
                .sum::<f64>()
        })
        .collect();
    let mut particular = vec![0.0; p];
    let mut null_columns = Vec::new();
    for k in 0..p {
        if ce[k] > tolerance * maximum {
            let projection: f64 = (0..p).map(|a| cv[(a, k)] * ctd[a]).sum();
            for a in 0..p {
                particular[a] += cv[(a, k)] * projection / ce[k];
            }
        } else {
            null_columns.push(k);
        }
    }
    let constraint_residual = |theta: &[f64]| {
        (0..c.rows)
            .map(|i| ((0..p).map(|a| c[(i, a)] * theta[a]).sum::<f64>() - config.values[i]).abs())
            .fold(0.0_f64, f64::max)
    };
    if constraint_residual(&particular) > config.constraint_absolute_tolerance {
        return invalid("equality constraints are inconsistent at the declared numerical rank");
    }
    let free = null_columns.len();
    let mut null = Matrix::zeros(p, free);
    for (j, &k) in null_columns.iter().enumerate() {
        for a in 0..p {
            null[(a, j)] = cv[(a, k)];
        }
    }
    let data_reduced = null.transpose_matmul_seq(&gram.matmul_seq(&null));
    let (data_e, _) = symmetric_eigen(&data_reduced)?;
    let (data_rank, data_condition) = spectral_rank(&data_e, tolerance)?;
    let penalty = l.transpose_matmul_seq(l);
    let mut h = gram.clone();
    let mut full_rhs = rhs;
    for a in 0..p {
        full_rhs[a] += config.regularization_strength
            * (0..p)
                .map(|b| penalty[(a, b)] * config.prior[b])
                .sum::<f64>();
        for b in 0..p {
            h[(a, b)] += config.regularization_strength * penalty[(a, b)];
        }
    }
    let reduced = null.transpose_matmul_seq(&h.matmul_seq(&null));
    if !finite_matrix(&reduced) || !finite_matrix(&h) || full_rhs.iter().any(|v| !v.is_finite()) {
        return invalid("regularized projection Gram or right-hand side overflowed");
    }
    let (e, v) = symmetric_eigen(&reduced)?;
    let (regularized_rank, regularized_condition) = spectral_rank(&e, tolerance)?;
    let mut source = vec![0.0; free];
    for j in 0..free {
        source[j] = (0..p)
            .map(|a| {
                null[(a, j)]
                    * (full_rhs[a] - (0..p).map(|b| h[(a, b)] * particular[b]).sum::<f64>())
            })
            .sum();
    }
    let maximum = e.iter().copied().fold(0.0_f64, f64::max);
    let mut z = vec![0.0; free];
    for k in 0..free {
        if e[k] <= tolerance * maximum {
            continue;
        }
        let projection: f64 = (0..free).map(|j| v[(j, k)] * source[j]).sum();
        for j in 0..free {
            z[j] += v[(j, k)] * projection / e[k];
        }
    }
    let coefficients: Vec<_> = (0..p)
        .map(|a| particular[a] + (0..free).map(|j| null[(a, j)] * z[j]).sum::<f64>())
        .collect();
    let stationarity = (0..free)
        .map(|i| ((0..free).map(|j| reduced[(i, j)] * z[j]).sum::<f64>() - source[i]).abs())
        .fold(0.0_f64, f64::max);
    let loss: f64 = dataset
        .data
        .iter()
        .map(|datum| {
            let residual = datum
                .linear_features
                .iter()
                .zip(&coefficients)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                - datum.target;
            datum.weight * (residual / datum.scale).powi(2)
        })
        .sum();
    let regularization_penalty = config.regularization_strength
        * (0..l.rows)
            .map(|i| {
                (0..p)
                    .map(|a| l[(i, a)] * (coefficients[a] - config.prior[a]))
                    .sum::<f64>()
                    .powi(2)
            })
            .sum::<f64>();
    if coefficients.iter().any(|v| !v.is_finite())
        || !finite_matrix(&gram)
        || !stationarity.is_finite()
        || !loss.is_finite()
        || !regularization_penalty.is_finite()
        || constraint_residual(&coefficients) > config.constraint_absolute_tolerance
    {
        return invalid("nonfinite constrained-projection result");
    }
    Ok(ConstrainedProjection {
        constraint_max_residual: constraint_residual(&coefficients),
        coefficients,
        gram,
        constraint_rank,
        free_parameters: free,
        data_rank,
        regularized_rank,
        data_unique: data_rank == free,
        regularized_unique: regularized_rank == free,
        data_condition_number: data_condition,
        regularized_condition_number: regularized_condition,
        weighted_residual_squared: loss,
        regularization_penalty,
        projected_stationarity_residual: stationarity,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct SchurConfig {
    pub overlap_relative_tolerance: f64,
    /// Absolute distance from eliminated generalized eigenvalues, in Hartree.
    pub minimum_pole_distance: f64,
    pub minimum_reciprocal_condition: f64,
}

impl Default for SchurConfig {
    fn default() -> Self {
        Self {
            overlap_relative_tolerance: 1e-12,
            minimum_pole_distance: 1e-8,
            minimum_reciprocal_condition: 1e-12,
        }
    }
}

pub struct SchurModel {
    fock: Matrix,
    overlap: Matrix,
    retained: Vec<usize>,
    eliminated: Vec<usize>,
    poles: Vec<f64>,
    overlap_condition_number: f64,
}

pub struct SchurReport {
    pub energy: f64,
    pub self_energy: Matrix,
    pub self_energy_derivative: Matrix,
    pub reconstruction: Matrix,
    pub reconstruction_derivative: Matrix,
    /// Embedding map in the original AO row ordering.
    pub embedding: Matrix,
    pub effective_metric: Matrix,
    pub pole_distance: Option<f64>,
    pub denominator_condition_number: f64,
    pub overlap_condition_number: f64,
    pub linear_solve_absolute_residual: f64,
    pub linear_solve_relative_residual: f64,
    pub metric_identity_max_residual: f64,
    pub self_energy_symmetry_defect: f64,
}

pub struct SchurEigenpairReport {
    pub occupied: bool,
    pub energy: f64,
    pub reconstructed_coefficients: Vec<f64>,
    pub full_equation_relative_residual: f64,
    pub retained_equation_absolute_residual: f64,
    pub full_metric_norm: f64,
    pub effective_metric_norm: f64,
    pub norm_identity_residual: f64,
    pub elimination: SchurReport,
}

impl SchurModel {
    pub fn new(
        fock: Matrix,
        overlap: Matrix,
        retained: Vec<usize>,
        config: SchurConfig,
    ) -> Result<Self> {
        let n = fock.rows;
        if n == 0
            || !symmetric(&fock)
            || overlap.rows != n
            || !symmetric(&overlap)
            || retained.is_empty()
            || retained.iter().any(|&i| i >= n)
            || retained.iter().collect::<BTreeSet<_>>().len() != retained.len()
            || !config.overlap_relative_tolerance.is_finite()
            || config.overlap_relative_tolerance <= 0.0
            || config.overlap_relative_tolerance >= 1.0
        {
            return invalid("fixed-matrix elimination requires finite symmetric F/S, distinct retained indices and a positive overlap threshold");
        }
        let (e, _) = symmetric_eigen(&overlap)?;
        if e[0] <= config.overlap_relative_tolerance * e[n - 1] {
            return invalid("fixed-matrix elimination requires a full-rank positive AO overlap; no rank truncation is performed");
        }
        let eliminated: Vec<_> = (0..n).filter(|i| !retained.contains(i)).collect();
        let sbb = block(&overlap, &eliminated, &eliminated);
        let fbb = block(&fock, &eliminated, &eliminated);
        let poles = if eliminated.is_empty() {
            Vec::new()
        } else {
            let (se, _) = symmetric_eigen(&sbb)?;
            let orth = Orthogonalizer::canonical(
                &sbb,
                config.overlap_relative_tolerance * se[se.len() - 1],
            )?;
            orth.solve_roothaan(&fbb)?.0
        };
        Ok(Self {
            fock,
            overlap,
            retained,
            eliminated,
            poles,
            overlap_condition_number: e[n - 1] / e[0],
        })
    }

    pub fn eliminated_poles(&self) -> &[f64] {
        &self.poles
    }

    pub fn evaluate(&self, energy: f64, config: SchurConfig) -> Result<SchurReport> {
        if !energy.is_finite()
            || !config.minimum_pole_distance.is_finite()
            || config.minimum_pole_distance < 0.0
            || !config.minimum_reciprocal_condition.is_finite()
            || config.minimum_reciprocal_condition <= 0.0
            || config.minimum_reciprocal_condition >= 1.0
        {
            return invalid(
                "invalid elimination energy, pole-distance guard or reciprocal-condition guard",
            );
        }
        let a = &self.retained;
        let b = &self.eliminated;
        let saa = block(&self.overlap, a, a);
        let sba = block(&self.overlap, b, a);
        let sbb = block(&self.overlap, b, b);
        let fba = block(&self.fock, b, a);
        let fbb = block(&self.fock, b, b);
        let mut denominator = sbb.clone();
        for i in 0..denominator.rows {
            for j in 0..denominator.cols {
                denominator[(i, j)] = energy * sbb[(i, j)] - fbb[(i, j)];
            }
        }
        let mut source = fba;
        for (x, &s) in source.as_mut_slice().iter_mut().zip(sba.as_slice()) {
            *x -= energy * s;
        }
        if !finite_matrix(&denominator) || !finite_matrix(&source) {
            return invalid("fixed-matrix elimination overflowed its finite energy shift");
        }
        let pole_distance = self
            .poles
            .iter()
            .map(|&p| (energy - p).abs())
            .reduce(f64::min);
        let (de, _) = symmetric_eigen(&denominator)?;
        let largest = de.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let smallest = de.iter().fold(f64::INFINITY, |m, &x| m.min(x.abs()));
        let rcond = if b.is_empty() {
            1.0
        } else if largest == 0.0 {
            0.0
        } else {
            smallest / largest
        };
        if pole_distance.is_some_and(|d| d <= config.minimum_pole_distance)
            || rcond < config.minimum_reciprocal_condition
        {
            return invalid(format!("fixed-matrix elimination is too close to a pole: energy={energy:.16e}, pole_distance={pole_distance:?}, reciprocal_condition={rcond:.16e}"));
        }
        let reconstruction = solve_linear_matrix(&denominator, &source)?;
        let mut derivative_source = sbb.matmul_seq(&reconstruction);
        for (v, &s) in derivative_source
            .as_mut_slice()
            .iter_mut()
            .zip(sba.as_slice())
        {
            *v = -*v - s;
        }
        let reconstruction_derivative = solve_linear_matrix(&denominator, &derivative_source)?;
        let self_energy = source.transpose_matmul_seq(&reconstruction);
        let mut self_energy_derivative = source.transpose_matmul_seq(&reconstruction_derivative);
        let explicit = sba.transpose_matmul_seq(&reconstruction);
        for (value, &s) in self_energy_derivative
            .as_mut_slice()
            .iter_mut()
            .zip(explicit.as_slice())
        {
            *value -= s;
        }
        let mut embedding = Matrix::zeros(self.fock.rows, a.len());
        for (i, &row) in a.iter().enumerate() {
            embedding[(row, i)] = 1.0;
        }
        for (i, &row) in b.iter().enumerate() {
            for j in 0..a.len() {
                embedding[(row, j)] = reconstruction[(i, j)];
            }
        }
        let effective_metric = embedding.transpose_matmul_seq(&self.overlap.matmul_seq(&embedding));
        let mut expected_metric = saa;
        for (v, &d) in expected_metric
            .as_mut_slice()
            .iter_mut()
            .zip(self_energy_derivative.as_slice())
        {
            *v -= d;
        }
        let mut residual = denominator.matmul_seq(&reconstruction);
        for (v, &t) in residual.as_mut_slice().iter_mut().zip(source.as_slice()) {
            *v -= t;
        }
        let absolute = norm(&residual);
        let source_norm = norm(&source);
        let relative = if source_norm == 0.0 {
            absolute
        } else {
            absolute / source_norm
        };
        let symmetry = self_energy.max_abs_diff(&self_energy.transpose());
        if !finite_matrix(&self_energy)
            || !finite_matrix(&self_energy_derivative)
            || !finite_matrix(&reconstruction_derivative)
            || !finite_matrix(&effective_metric)
            || !relative.is_finite()
        {
            return invalid(
                "fixed-matrix elimination produced nonfinite reconstruction diagnostics",
            );
        }
        Ok(SchurReport {
            energy,
            self_energy,
            self_energy_derivative,
            reconstruction,
            reconstruction_derivative,
            metric_identity_max_residual: effective_metric.max_abs_diff(&expected_metric),
            embedding,
            effective_metric,
            pole_distance,
            denominator_condition_number: 1.0 / rcond,
            overlap_condition_number: self.overlap_condition_number,
            linear_solve_absolute_residual: absolute,
            linear_solve_relative_residual: relative,
            self_energy_symmetry_defect: symmetry,
        })
    }

    /// Check every occupied and virtual eigenpair of this same fixed F,S.
    /// This validates elimination only, not a density-dependent embedded SCF.
    pub fn check_reference_spectrum(
        &self,
        occupied: usize,
        config: SchurConfig,
    ) -> Result<Vec<SchurEigenpairReport>> {
        if occupied > self.fock.rows
            || !config.overlap_relative_tolerance.is_finite()
            || config.overlap_relative_tolerance <= 0.0
            || config.overlap_relative_tolerance >= 1.0
        {
            return invalid(
                "invalid occupied count or overlap tolerance for fixed reference spectrum",
            );
        }
        let (spectrum, _) = symmetric_eigen(&self.overlap)?;
        if spectrum[0] <= config.overlap_relative_tolerance * spectrum[spectrum.len() - 1] {
            return invalid("reference spectrum must retain the complete positive overlap space");
        }
        let orth = Orthogonalizer::canonical(
            &self.overlap,
            config.overlap_relative_tolerance * spectrum[spectrum.len() - 1],
        )?;
        let (energies, coefficients) = orth.solve_roothaan(&self.fock)?;
        let mut reports = Vec::with_capacity(energies.len());
        for (index, &energy) in energies.iter().enumerate() {
            let elimination = self.evaluate(energy, config)?;
            let ca: Vec<_> = self
                .retained
                .iter()
                .map(|&i| coefficients[(i, index)])
                .collect();
            let full: Vec<_> = (0..self.fock.rows)
                .map(|i| {
                    (0..ca.len())
                        .map(|j| elimination.embedding[(i, j)] * ca[j])
                        .sum::<f64>()
                })
                .collect();
            let fc: Vec<_> = (0..full.len())
                .map(|i| {
                    (0..full.len())
                        .map(|j| self.fock[(i, j)] * full[j])
                        .sum::<f64>()
                })
                .collect();
            let sc: Vec<_> = (0..full.len())
                .map(|i| {
                    (0..full.len())
                        .map(|j| self.overlap[(i, j)] * full[j])
                        .sum::<f64>()
                })
                .collect();
            let residual = fc
                .iter()
                .zip(&sc)
                .fold(0.0_f64, |r, (&f, &s)| r.hypot(f - energy * s));
            let denominator = fc.iter().fold(0.0_f64, |r, &v| r.hypot(v))
                + energy.abs() * sc.iter().fold(0.0_f64, |r, &v| r.hypot(v));
            let faa = block(&self.fock, &self.retained, &self.retained);
            let saa = block(&self.overlap, &self.retained, &self.retained);
            let retained_residual = (0..ca.len()).fold(0.0_f64, |r, i| {
                r.hypot(
                    (0..ca.len())
                        .map(|j| {
                            (faa[(i, j)] + elimination.self_energy[(i, j)] - energy * saa[(i, j)])
                                * ca[j]
                        })
                        .sum::<f64>(),
                )
            });
            let full_norm: f64 = full.iter().zip(&sc).map(|(c, s)| c * s).sum();
            let effective_norm: f64 = (0..ca.len())
                .map(|i| {
                    (0..ca.len())
                        .map(|j| ca[i] * elimination.effective_metric[(i, j)] * ca[j])
                        .sum::<f64>()
                })
                .sum();
            reports.push(SchurEigenpairReport {
                occupied: index < occupied,
                energy,
                reconstructed_coefficients: full,
                full_equation_relative_residual: if denominator == 0.0 {
                    residual
                } else {
                    residual / denominator
                },
                retained_equation_absolute_residual: retained_residual,
                full_metric_norm: full_norm,
                effective_metric_norm: effective_norm,
                norm_identity_residual: (full_norm - effective_norm).abs(),
                elimination,
            });
        }
        Ok(reports)
    }
}
