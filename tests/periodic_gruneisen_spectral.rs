// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::periodic::gruneisen::{isotropic_log_volume_jets, mode_volume_derivatives};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

fn rotation(n: usize) -> Matrix {
    let mut q = Matrix::identity(n);
    for a in 0..n {
        for b in a + 1..n {
            let angle = 0.13 + 0.07 * (a + 2 * b) as f64;
            let (s, c) = angle.sin_cos();
            for i in 0..n {
                let (x, y) = (q[(i, a)], q[(i, b)]);
                q[(i, a)] = c * x - s * y;
                q[(i, b)] = s * x + c * y;
            }
        }
    }
    q
}

fn rotate(a: &Matrix, q: &Matrix) -> Matrix {
    q.matmul(a).matmul(&q.transpose())
}

fn family(degenerate: bool) -> Vec<Matrix> {
    let mut out: Vec<Matrix> = (0..5).map(|_| Matrix::zeros(4, 4)).collect();
    let values = if degenerate {
        [2.0, 2.0, 2.0, 7.0]
    } else {
        [1.1, 2.3, 4.7, 8.2]
    };
    for i in 0..4 {
        out[0][(i, i)] = values[i];
        for j in 0..4 {
            let a = (i.min(j) + 1) as f64;
            let b = (i.max(j) + 1) as f64;
            out[1][(i, j)] = 0.09 * a + 0.031 * b;
            out[2][(i, j)] = 0.12 * a + 0.071 * b;
            out[3][(i, j)] = 0.19 * a + 0.043 * b;
            out[4][(i, j)] = 0.17 * a + 0.023 * b;
        }
    }
    if degenerate {
        for i in 0..3 {
            for j in 0..3 {
                out[1][(i, j)] = if i != j {
                    0.0
                } else if i < 2 {
                    0.3
                } else {
                    -0.4
                };
            }
        }
    }
    let q = rotation(4);
    out.iter().map(|a| rotate(a, &q)).collect()
}

fn evaluate(jets: &[Matrix], x: f64, reference: &Matrix) -> Vec<f64> {
    let mut a = jets[0].clone();
    let mut power = 1.0;
    for (k, jet) in jets.iter().enumerate().skip(1) {
        power *= x / k as f64;
        for i in 0..4 {
            for j in 0..4 {
                a[(i, j)] += power * jet[(i, j)];
            }
        }
    }
    let (values, modes) = symmetric_eigen(&a).unwrap();
    let overlap = reference.transpose().matmul(&modes);
    let indices: Vec<usize> = (0..4)
        .map(|i| {
            (0..4)
                .max_by(|&j, &k| overlap[(i, j)].abs().total_cmp(&overlap[(i, k)].abs()))
                .unwrap()
        })
        .collect();
    for i in 0..4 {
        for j in i + 1..4 {
            assert_ne!(indices[i], indices[j], "mode tracking is ambiguous");
        }
    }
    indices.iter().map(|&i| 0.5 * values[i].ln()).collect()
}

#[test]
fn all_log_frequency_derivatives_have_independent_four_step_fits() {
    for degenerate in [false, true] {
        let jets = family(degenerate);
        let response = mode_volume_derivatives(&jets[0], &jets[1], &jets[2]).unwrap();
        let names: Vec<String> = (0..4)
            .flat_map(|i| {
                [
                    format!("degenerate={degenerate} gamma[{i}]"),
                    format!("degenerate={degenerate} curvature[{i}]"),
                ]
            })
            .collect();
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second)],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 8];
                for (k, &h) in steps.iter().enumerate() {
                    let plus = evaluate(&jets, h, &response.modes);
                    let minus = evaluate(&jets, -h, &response.modes);
                    for i in 0..4 {
                        let branch = &response.branches[i];
                        let fd_first = -stencil.first(plus[i], minus[i], 0.0, 0.0, h);
                        let fd_second =
                            (plus[i] - branch.squared_frequency.ln() + minus[i]) / (h * h);
                        errors[2 * i][k] = (fd_first - branch.first_gruneisen.unwrap()).abs();
                        errors[2 * i + 1][k] =
                            (fd_second - branch.second_log_frequency.unwrap()).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("all branch derivatives", &failures);
    }
}

#[test]
fn analytic_volume_power_and_curvature_conventions() {
    let h: f64 = 0.02;
    let cubic = |x: f64| 0.7 * x + 0.3 * x.powi(3);
    assert!(
        (stencils::Stencil::Fourth.first(cubic(h), cubic(-h), cubic(2.0 * h), cubic(-2.0 * h), h)
            - 0.7)
            .abs()
            < 1e-14
    );
    let mut d = Matrix::zeros(3, 3);
    let mut first = d.clone();
    let mut second = d.clone();
    for (i, (lambda, gamma, curvature)) in [(1.3, 0.7, 0.21), (2.9, -0.3, -0.16), (6.1, 1.2, 0.43)]
        .into_iter()
        .enumerate()
    {
        d[(i, i)] = lambda;
        first[(i, i)] = -2.0 * lambda * gamma;
        second[(i, i)] = lambda * (4.0 * gamma * gamma + 2.0 * curvature);
    }
    let r = mode_volume_derivatives(&d, &first, &second).unwrap();
    for (i, (gamma, curvature)) in [(0.7, 0.21), (-0.3, -0.16), (1.2, 0.43)]
        .into_iter()
        .enumerate()
    {
        assert!((r.branches[i].first_gruneisen.unwrap() - gamma).abs() < 1e-13);
        assert!((r.branches[i].second_log_frequency.unwrap() - curvature).abs() < 1e-13);
    }
}

#[test]
fn basis_invariance_instabilities_and_input_contracts() {
    let jets = family(true);
    let a = mode_volume_derivatives(&jets[0], &jets[1], &jets[2]).unwrap();
    let q = rotation(4).transpose();
    let rotated: Vec<Matrix> = jets.iter().map(|m| rotate(m, &q)).collect();
    let b = mode_volume_derivatives(&rotated[0], &rotated[1], &rotated[2]).unwrap();
    for (x, y) in a.branches.iter().zip(&b.branches) {
        assert!((x.first_squared_frequency - y.first_squared_frequency).abs() < 1e-12);
        assert!((x.second_squared_frequency - y.second_squared_frequency).abs() < 1e-12);
    }
    let mut d = Matrix::zeros(2, 2);
    d[(0, 0)] = -1.0;
    let zero = Matrix::zeros(2, 2);
    let r = mode_volume_derivatives(&d, &zero, &zero).unwrap();
    assert!(r
        .branches
        .iter()
        .all(|b| b.first_gruneisen.is_none() && b.second_log_frequency.is_none()));
    let mut bad = d.clone();
    bad[(0, 1)] = 1e-5;
    assert!(mode_volume_derivatives(&bad, &zero, &zero).is_err());
    assert!(mode_volume_derivatives(&d, &Matrix::zeros(3, 3), &zero).is_err());
    bad[(0, 1)] = f64::NAN;
    assert!(mode_volume_derivatives(&bad, &zero, &zero).is_err());
    assert_eq!(a.reference_eigenvalues.len(), 4);
    assert!(a.symmetry_defects.iter().all(|v| *v < 1e-14));
}

#[test]
fn every_material_to_physical_volume_jet_component_has_a_four_step_fit() {
    let n = 6;
    let mass = [0.37, 0.61];
    let mut coefficients: Vec<Matrix> = (0..5).map(|_| Matrix::zeros(n, n)).collect();
    for i in 0..n {
        for j in 0..n {
            let a = (i.min(j) + 1) as f64;
            let b = (i.max(j) + 1) as f64;
            for (k, m) in coefficients.iter_mut().enumerate() {
                m[(i, j)] = 0.7 + a * (0.08 + 0.013 * k as f64) + b * (0.02 + 0.027 * k as f64);
            }
        }
    }
    let jets =
        isotropic_log_volume_jets(&coefficients[0], &coefficients[1], &coefficients[2], &mass)
            .unwrap();
    let evaluate = |x: f64, i: usize, j: usize| {
        let deformation_squared = (2.0 * x / 3.0).exp();
        let s = 0.5 * (deformation_squared - 1.0);
        let mut power = 1.0;
        let mut value = coefficients[0][(i, j)];
        for (k, m) in coefficients.iter().enumerate().skip(1) {
            power *= s / k as f64;
            value += power * m[(i, j)];
        }
        value * mass[i / 3] * mass[j / 3] / deformation_squared
    };
    let entries: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let names: Vec<String> = entries
        .iter()
        .flat_map(|(i, j)| [format!("D_x[{i},{j}]"), format!("D_xx[{i},{j}]")])
        .collect();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second)],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (index, &(i, j)) in entries.iter().enumerate() {
                for (k, &h) in steps.iter().enumerate() {
                    let plus = evaluate(h, i, j);
                    let minus = evaluate(-h, i, j);
                    let first = stencil.first(plus, minus, 0.0, 0.0, h);
                    let second = (plus - 2.0 * evaluate(0.0, i, j) + minus) / (h * h);
                    errors[2 * index][k] = (first - jets[1][(i, j)]).abs();
                    errors[2 * index + 1][k] = (second - jets[2][(i, j)]).abs();
                    assert_eq!(jets[1][(i, j)], jets[1][(j, i)]);
                    assert_eq!(jets[2][(i, j)], jets[2][(j, i)]);
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all volume jet components", &failures);
    assert!(isotropic_log_volume_jets(
        &coefficients[0],
        &coefficients[1],
        &coefficients[2],
        &[0.0, 0.61]
    )
    .is_err());
    assert!(isotropic_log_volume_jets(
        &coefficients[0],
        &coefficients[1],
        &coefficients[2],
        &[0.37]
    )
    .is_err());
}
