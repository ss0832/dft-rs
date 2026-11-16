// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent accounting, constrained-Gram and nonorthogonal elimination gates.
//! The source module is isolated until the reduction driver adopts it.
pub use dft_rs::{linalg, pseudo_sub, DftError, Result};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "../src/driver/continuous_species/reduction_linear.rs"]
mod reduction_linear;

use linalg::{solve_linear, Matrix};
use pseudo_sub::fit::{FitDataset, ObservableKind, ReferenceDatum};
use reduction_linear::*;

fn matrix(rows: usize, columns: usize, values: &[f64]) -> Matrix {
    Matrix::from_row_major(rows, columns, values.to_vec())
}

#[test]
fn explicit_electron_ledger_single_double_and_ordinary_limits() {
    let charges = [6, 6, 1, 1, 1, 1, 1, 1];
    let methyl = FrozenFragmentSpec {
        explicit_atoms: vec![1, 5, 6, 7],
        frozen_electrons: 8,
    };
    let ledger = ElectronLedger::new(&charges, 18, std::slice::from_ref(&methyl)).unwrap();
    assert_eq!(ledger.full_nuclear_charge, 18);
    assert_eq!(ledger.retained_nuclear_charge, 9);
    assert_eq!(ledger.full_electron_count, 18);
    assert_eq!(ledger.frozen_electron_count, 8);
    assert_eq!(ledger.active_electron_count, 10);
    assert_eq!(ledger.active_ionic_charge, 10);
    assert_eq!(ledger.full_total_charge, 0);
    assert_eq!(ledger.reduced_total_charge, 0);
    assert_eq!(ledger.fragments[0].nuclear_charge, 9);
    assert_eq!(ledger.fragments[0].effective_ionic_charge, 1);
    assert_eq!(ledger.fragments[0].frozen_electrons, 8);
    assert_eq!(ledger.fragments[0].explicit_atoms, [1, 5, 6, 7]);
    let charged = ElectronLedger::new(&charges, 17, std::slice::from_ref(&methyl)).unwrap();
    assert_eq!(
        (charged.active_electron_count, charged.reduced_total_charge),
        (9, 1)
    );
    let second = FrozenFragmentSpec {
        explicit_atoms: vec![0, 2, 3, 4],
        frozen_electrons: 8,
    };
    let double = ElectronLedger::new(&charges, 18, &[methyl.clone(), second]).unwrap();
    assert_eq!(
        (
            double.retained_nuclear_charge,
            double.active_ionic_charge,
            double.active_electron_count
        ),
        (0, 2, 2)
    );
    let ordinary = ElectronLedger::new(&[17], 18, &[]).unwrap();
    assert_eq!(ordinary.frozen_electron_count, 0);
    assert_eq!(ordinary.active_electron_count, 18);
    assert_eq!(ordinary.full_total_charge, -1);
    assert_eq!(ordinary.full_total_charge, ordinary.reduced_total_charge);
    assert!(ElectronLedger::new(&charges, 18, &[methyl.clone(), methyl]).is_err());
    assert!(ElectronLedger::new(
        &charges,
        18,
        &[FrozenFragmentSpec {
            explicit_atoms: vec![1, 1],
            frozen_electrons: 2
        }]
    )
    .is_err());
    assert!(ElectronLedger::new(
        &charges,
        18,
        &[FrozenFragmentSpec {
            explicit_atoms: vec![8],
            frozen_electrons: 2
        }]
    )
    .is_err());
    assert!(ElectronLedger::new(
        &charges,
        7,
        &[FrozenFragmentSpec {
            explicit_atoms: vec![1, 5, 6, 7],
            frozen_electrons: 8
        }]
    )
    .is_err());
    assert!(ElectronLedger::new(&[0], 0, &[]).is_err());
    // A non-idempotent projected block is allowed: only its metric trace is checked.
    let occupations = [1.5, 1.5, 1.5, 1.5, 2.0, 2.0];
    let scales = [1.0, 1.2, 0.9, 1.1, 1.3, 0.8];
    let mut density = Matrix::zeros(6, 6);
    let mut overlap = Matrix::zeros(6, 6);
    for i in 0..6 {
        density[(i, i)] = occupations[i] / scales[i];
        overlap[(i, i)] = scales[i];
    }
    assert!(
        (ledger
            .check_active_trace(&density, &overlap, 1e-12)
            .unwrap()
            - 10.0)
            .abs()
            < 1e-12
    );
    density[(0, 0)] += 0.1;
    assert!(ledger
        .check_active_trace(&density, &overlap, 1e-12)
        .is_err());
}

#[test]
fn template_dimensions_and_global_electron_sector_are_distinct() {
    let dimensions = TemplateDimensions {
        centers: 1,
        ao_functions: 5,
        local_functions: 2,
        projector_functions: 3,
        geometry_functions: 7,
        parameters: 5,
    };
    let template = FixedReductionTemplate {
        id: "one-center-diagnostic".into(),
        version: 1,
        dimensions,
    };
    template.validate(dimensions).unwrap();
    assert_eq!(
        (
            template.dimensions.local_functions,
            template.dimensions.projector_functions,
            template.dimensions.geometry_functions,
            template.dimensions.parameters
        ),
        (2, 3, 7, 5)
    );
    assert!(template
        .validate(TemplateDimensions {
            projector_functions: 4,
            ..dimensions
        })
        .is_err());
    assert!(FixedReductionTemplate {
        id: "".into(),
        version: 1,
        dimensions
    }
    .validate(dimensions)
    .is_err());
    let sector = FixedElectronicSector {
        active_electrons: 10,
        spin_electrons: [5, 5],
        total_ao_functions: 13,
    };
    sector.validate(10, [5, 5], 13).unwrap();
    assert!(sector.validate(9, [5, 4], 13).is_err());
    assert!(sector.validate(10, [5, 5], 12).is_err());
    assert!(FixedElectronicSector {
        total_ao_functions: 4,
        ..sector
    }
    .validate(10, [5, 5], 4)
    .is_err());
}

fn operator_sample() -> OperatorProjectionSample {
    let dictionary = vec![
        matrix(2, 2, &[1.0, 0.31, 0.31, 0.43]),
        matrix(2, 2, &[0.71, -0.23, -0.23, 1.27]),
        matrix(2, 2, &[-0.21, 0.82, 0.82, 0.64]),
    ];
    let truth = [-0.4, 1.1, 0.3];
    let mut target = Matrix::zeros(2, 2);
    for (coefficient, item) in truth.iter().zip(&dictionary) {
        for (x, &y) in target.as_mut_slice().iter_mut().zip(item.as_slice()) {
            *x += coefficient * y;
        }
    }
    OperatorProjectionSample {
        case_id: "frozen-reference".into(),
        group_id: "construction-host".into(),
        target,
        dictionary,
        metric_weight: matrix(2, 2, &[1.18, 0.17, 0.17, 0.83]),
        weight: 1.7,
    }
}

fn projection_config(p: usize) -> ProjectionConstraints {
    ProjectionConstraints {
        matrix: Matrix::zeros(0, p),
        values: Vec::new(),
        prior: vec![0.0; p],
        regularization_operator: Matrix::identity(p),
        regularization_strength: 0.0,
        constraint_absolute_tolerance: 1e-12,
        relative_rank_tolerance: 1e-10,
    }
}

#[test]
fn constrained_operator_projection_recovers_correlated_dictionary_without_psd_restriction() {
    let sample = operator_sample();
    let dataset = operator_projection_dataset(
        std::slice::from_ref(&sample),
        vec!["c0".into(), "c1".into(), "d01".into()],
        "fixed-reference".into(),
        "gram-recovery".into(),
    )
    .unwrap();
    assert_eq!(dataset.data.len(), 3);
    assert!(dataset
        .data
        .iter()
        .all(|row| row.group_id == "construction-host"));
    let mut config = projection_config(3);
    config.matrix = matrix(2, 3, &[1.0, 1.0, 0.0, 2.0, 2.0, 0.0]);
    config.values = vec![0.7, 1.4];
    let fit = constrained_projection(&dataset, &config).unwrap();
    for (&actual, expected) in fit.coefficients.iter().zip([-0.4, 1.1, 0.3]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(fit.coefficients[0] < 0.0); // General electronic coefficients are not forced positive.
    assert_eq!(
        (
            fit.constraint_rank,
            fit.free_parameters,
            fit.data_rank,
            fit.regularized_rank
        ),
        (1, 2, 2, 2)
    );
    assert!(fit.data_unique && fit.regularized_unique);
    assert!(fit.data_condition_number.unwrap() > 1.0);
    assert!(fit.regularized_condition_number.unwrap() > 1.0);
    assert!(fit.weighted_residual_squared < 1e-24);
    assert_eq!(fit.regularization_penalty, 0.0);
    assert!(fit.constraint_max_residual < 1e-13);
    assert!(fit.projected_stationarity_residual < 1e-13);
    assert!(fit.gram[(0, 1)].abs() > 0.1);
    let naive: Vec<_> = (0..3)
        .map(|a| {
            let rhs = dataset
                .data
                .iter()
                .map(|row| row.weight * row.linear_features[a] * row.target)
                .sum::<f64>();
            rhs / fit.gram[(a, a)]
        })
        .collect();
    assert!(naive
        .iter()
        .zip(&fit.coefficients)
        .any(|(a, b)| (a - b).abs() > 0.1));
    config.values[1] = 1.5;
    assert!(constrained_projection(&dataset, &config).is_err());
}

#[test]
fn metric_projection_loss_and_ridge_match_independent_dense_kkt_equations() {
    let mut sample = operator_sample();
    sample.target[(0, 0)] += 0.09;
    sample.target[(0, 1)] -= 0.04;
    sample.target[(1, 0)] -= 0.04;
    let dataset = operator_projection_dataset(
        std::slice::from_ref(&sample),
        vec!["a".into(), "b".into(), "d".into()],
        "fixed".into(),
        "kkt".into(),
    )
    .unwrap();
    let mut config = projection_config(3);
    config.matrix = matrix(1, 3, &[1.0, 1.0, 0.0]);
    config.values = vec![0.7];
    config.regularization_strength = 0.13;
    config.regularization_operator = matrix(2, 3, &[1.0, 0.17, -0.21, 0.0, 0.83, 0.34]);
    config.prior = vec![-0.2, 0.8, 0.1];
    let fit = constrained_projection(&dataset, &config).unwrap();
    // Solve a full indefinite KKT system independently of the production
    // constraint-nullspace spectral solve.
    let mut kkt = Matrix::zeros(4, 4);
    let mut source = vec![0.0; 4];
    for row in &dataset.data {
        for i in 0..3 {
            source[i] += row.weight * row.linear_features[i] * row.target;
            for j in 0..3 {
                kkt[(i, j)] += row.weight * row.linear_features[i] * row.linear_features[j];
            }
        }
    }
    for r in 0..config.regularization_operator.rows {
        let lp = (0..3)
            .map(|a| config.regularization_operator[(r, a)] * config.prior[a])
            .sum::<f64>();
        for i in 0..3 {
            source[i] +=
                config.regularization_strength * config.regularization_operator[(r, i)] * lp;
            for j in 0..3 {
                kkt[(i, j)] += config.regularization_strength
                    * config.regularization_operator[(r, i)]
                    * config.regularization_operator[(r, j)];
            }
        }
    }
    for i in 0..3 {
        kkt[(i, 3)] = config.matrix[(0, i)];
        kkt[(3, i)] = config.matrix[(0, i)];
    }
    source[3] = 0.7;
    let oracle = solve_linear(&kkt, &source).unwrap();
    for i in 0..3 {
        assert!((oracle[i] - fit.coefficients[i]).abs() < 1e-12);
    }
    let mut residual = sample.target.clone();
    for (coefficient, item) in fit.coefficients.iter().zip(&sample.dictionary) {
        for (x, &y) in residual.as_mut_slice().iter_mut().zip(item.as_slice()) {
            *x -= coefficient * y;
        }
    }
    let weighted = sample
        .metric_weight
        .matmul_seq(&residual)
        .matmul_seq(&sample.metric_weight);
    let loss = sample.weight * weighted.frobenius_dot(&weighted);
    assert!((loss - fit.weighted_residual_squared).abs() < 1e-14);
    assert!(fit.regularization_penalty > 0.0);
    assert!(fit.projected_stationarity_residual < 1e-13);
}

fn dependent_data() -> FitDataset {
    FitDataset {
        level: "fixed-linear".into(),
        dataset_id: "dependent-columns".into(),
        parameter_names: vec![
            "duplicate-a".into(),
            "duplicate-b".into(),
            "independent".into(),
        ],
        data: [
            [1.0, 1.0, 0.3, 0.7],
            [0.4, 0.4, 1.2, -0.2],
            [0.9, 0.9, -0.6, 1.1],
        ]
        .into_iter()
        .enumerate()
        .map(|(i, row)| ReferenceDatum {
            case_id: format!("row{i}"),
            group_id: "construction".into(),
            observable: ObservableKind::Other("operator".into()),
            target: row[3],
            scale: 1.0,
            weight: 1.0,
            linear_features: row[..3].to_vec(),
            diagnostics: None,
        })
        .collect(),
    }
}

#[test]
fn rank_deficiency_and_regularization_are_reported_separately() {
    let dataset = dependent_data();
    let mut config = projection_config(3);
    let unregularized = constrained_projection(&dataset, &config).unwrap();
    assert_eq!(unregularized.data_rank, 2);
    assert_eq!(unregularized.regularized_rank, 2);
    assert!(!unregularized.data_unique && !unregularized.regularized_unique);
    assert!(unregularized.data_condition_number.is_none());
    assert!(unregularized.regularized_condition_number.is_none());
    assert!((unregularized.coefficients[0] - unregularized.coefficients[1]).abs() < 1e-12);
    config.regularization_strength = 0.01;
    config.prior = vec![0.3, -0.1, 0.0];
    let regularized = constrained_projection(&dataset, &config).unwrap();
    assert_eq!(regularized.data_rank, 2);
    assert_eq!(regularized.regularized_rank, 3);
    assert!(!regularized.data_unique && regularized.regularized_unique);
    assert!(regularized
        .regularized_condition_number
        .unwrap()
        .is_finite());
    assert!((regularized.coefficients[0] - regularized.coefficients[1] - 0.4).abs() < 1e-11);
    assert!(regularized.regularization_penalty > 0.0);
    let mut invalid = operator_sample();
    invalid.metric_weight[(0, 0)] = -1.0;
    assert!(operator_projection_dataset(
        &[invalid],
        vec!["a".into(), "b".into(), "c".into()],
        "x".into(),
        "y".into()
    )
    .is_err());
}

fn schur_matrices() -> (Matrix, Matrix) {
    let overlap = matrix(
        5,
        5,
        &[
            1.20, 0.13, -0.09, 0.07, 0.11, 0.13, 1.10, 0.08, -0.12, 0.06, -0.09, 0.08, 0.95, 0.14,
            -0.05, 0.07, -0.12, 0.14, 1.35, 0.09, 0.11, 0.06, -0.05, 0.09, 1.05,
        ],
    );
    let fock = matrix(
        5,
        5,
        &[
            -1.4, 0.17, -0.11, 0.23, 0.09, 0.17, -0.4, 0.12, -0.16, 0.07, -0.11, 0.12, 0.3, 0.19,
            -0.13, 0.23, -0.16, 0.19, 0.9, 0.15, 0.09, 0.07, -0.13, 0.15, 1.6,
        ],
    );
    (fock, overlap)
}

#[test]
fn nonorthogonal_elimination_reconstructs_all_occupied_and_virtual_eigenpairs() {
    let (fock, overlap) = schur_matrices();
    let config = SchurConfig::default();
    let model = SchurModel::new(fock, overlap, vec![0, 3], config).unwrap();
    assert_eq!(model.eliminated_poles().len(), 3);
    let reports = model.check_reference_spectrum(2, config).unwrap();
    assert_eq!(reports.len(), 5);
    assert_eq!(reports.iter().filter(|r| r.occupied).count(), 2);
    assert_eq!(reports.iter().filter(|r| !r.occupied).count(), 3);
    for report in reports {
        println!(
            "fixed spectrum occupied={} energy={:.16e} residual={:.3e} norm={:.16e} pole={:?}",
            report.occupied,
            report.energy,
            report.full_equation_relative_residual,
            report.full_metric_norm,
            report.elimination.pole_distance
        );
        assert!(report.energy.is_finite());
        assert_eq!(report.reconstructed_coefficients.len(), 5);
        assert!(report.full_equation_relative_residual < 1e-12);
        assert!(report.retained_equation_absolute_residual < 1e-12);
        assert!((report.full_metric_norm - 1.0).abs() < 1e-11);
        assert!((report.effective_metric_norm - 1.0).abs() < 1e-11);
        assert!(report.norm_identity_residual < 1e-12);
        assert!(report.elimination.metric_identity_max_residual < 1e-11);
        assert!(report.elimination.self_energy_symmetry_defect < 1e-12);
        assert!(report.elimination.linear_solve_absolute_residual < 1e-12);
        assert!(report.elimination.linear_solve_relative_residual < 1e-12);
        assert!(report.elimination.overlap_condition_number > 1.0);
        assert!(report.elimination.denominator_condition_number >= 1.0);
    }
    assert!(model.evaluate(model.eliminated_poles()[0], config).is_err());
}

#[test]
fn every_self_energy_and_reconstruction_energy_derivative_has_a_four_step_fit() {
    let (fock, overlap) = schur_matrices();
    let config = SchurConfig::default();
    let model = SchurModel::new(fock, overlap, vec![0, 3], config).unwrap();
    let energy = -0.83;
    let exact = model.evaluate(energy, config).unwrap();
    assert_eq!(exact.energy, energy);
    let sigma_pairs = [(0, 0), (0, 1), (1, 1)];
    let reconstruction_pairs: Vec<_> = (0..3).flat_map(|i| (0..2).map(move |j| (i, j))).collect();
    let names: Vec<_> = sigma_pairs
        .iter()
        .map(|(i, j)| format!("self-energy epsilon derivative[{i},{j}]"))
        .chain(
            reconstruction_pairs
                .iter()
                .map(|(i, j)| format!("reconstruction epsilon derivative[{i},{j}]")),
        )
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
        |steps| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (k, &h) in steps.iter().enumerate() {
                let positive = model.evaluate(energy + h, config).unwrap();
                let negative = model.evaluate(energy - h, config).unwrap();
                for (index, &(i, j)) in sigma_pairs.iter().enumerate() {
                    errors[index][k] =
                        ((positive.self_energy[(i, j)] - negative.self_energy[(i, j)]) / (2.0 * h)
                            - exact.self_energy_derivative[(i, j)])
                            .abs();
                }
                for (index, &(i, j)) in reconstruction_pairs.iter().enumerate() {
                    errors[3 + index][k] = ((positive.reconstruction[(i, j)]
                        - negative.reconstruction[(i, j)])
                        / (2.0 * h)
                        - exact.reconstruction_derivative[(i, j)])
                        .abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all independent fixed-matrix energy derivatives", &failures);
}

#[test]
fn complete_retention_identity_and_invalid_metrics() {
    let (fock, overlap) = schur_matrices();
    let config = SchurConfig::default();
    let model =
        SchurModel::new(fock.clone(), overlap.clone(), vec![0, 1, 2, 3, 4], config).unwrap();
    let report = model.evaluate(-0.83, config).unwrap();
    assert!(model.eliminated_poles().is_empty());
    assert!(report.pole_distance.is_none());
    assert!(report.self_energy.as_slice().iter().all(|v| *v == 0.0));
    assert!(report
        .self_energy_derivative
        .as_slice()
        .iter()
        .all(|v| *v == 0.0));
    assert_eq!(report.embedding, Matrix::identity(5));
    assert!(report.effective_metric.max_abs_diff(&overlap) < 1e-14);
    assert_eq!(report.metric_identity_max_residual, 0.0);
    assert!(SchurModel::new(fock.clone(), overlap.clone(), vec![0, 0], config).is_err());
    assert!(SchurModel::new(fock.clone(), overlap.clone(), vec![], config).is_err());
    let mut bad = overlap;
    bad[(0, 0)] = -1.0;
    assert!(SchurModel::new(fock, bad, vec![0, 3], config).is_err());
}
