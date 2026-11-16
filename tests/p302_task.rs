// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Original TASK and mTASK exchange, complete local first/second/third partials,
//! spin/kinetic limits and independent numerical kernels.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::{eri_incore, kinetic_matrix, nuclear_matrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::xc::{
    lda, resolve, task, Family, FunctionalSpec, GridVars, HybridParams, ResolvedFunctional,
    XcFunctional, XcOut,
};
use dft_rs::{Atom, Molecule, Vec3};
use std::sync::Arc;

fn providers() -> [(&'static str, Box<dyn XcFunctional>); 2] {
    [("task", task::task_x()), ("mtask", task::mtask_x())]
}

fn evaluate(f: &dyn XcFunctional, p: &[f64], order: usize) -> XcOut {
    let v: Vec<_> = p.iter().map(|&x| [x]).collect();
    let vars = GridVars {
        rho: [&v[0], &v[1]],
        sigma: Some([&v[2], &v[3], &v[4]]),
        tau: Some([&v[5], &v[6]]),
    };
    let mut output = XcOut::default();
    output.prepare(1, order, Family::MetaGga);
    f.eval_add(&vars, order, 1.0, &mut output).unwrap();
    output
}

fn array(v: &Value) -> Vec<f64> {
    v.numeric_array().unwrap().1
}

fn rows(v: &Value) -> &[Value] {
    let Value::Array(values) = v else {
        panic!("array expected")
    };
    values
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn frozen_component_energies_potentials_and_every_active_kernel_entry() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/task.json");
    let reference = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    let Value::Array(points) = reference.get("points").unwrap() else {
        panic!("points")
    };
    for (name, provider) in providers() {
        let Value::Array(values) = reference.get("components").unwrap().get(name).unwrap() else {
            panic!("component values")
        };
        let mut maxima = [0.0_f64; 3];
        for (index, (point, expected)) in points.iter().zip(values).enumerate() {
            let point = array(point);
            let result = evaluate(provider.as_ref(), &point, 2);
            let energy = expected.get("energy").unwrap().as_f64().unwrap();
            maxima[0] = maxima[0].max((result.e[0] - energy).abs());
            assert!(
                (result.e[0] - energy).abs() < 1e-11 * (1.0 + energy.abs()),
                "{name}/{index} energy"
            );
            let (potential, kernel) = result.potential_and_kernel(0, Family::MetaGga).unwrap();
            let expected_potential = array(expected.get("potential").unwrap());
            for i in 0..7 {
                if point[1] == 0.0 && [1, 3, 4, 6].contains(&i) {
                    continue;
                }
                let error = (potential[i] - expected_potential[i]).abs();
                maxima[1] = maxima[1].max(error);
                assert!(
                    error < 1e-10 * (1.0 + expected_potential[i].abs()),
                    "{name}/{index} potential{i}: {error:e}"
                );
            }
            if let Some(Value::Array(rows)) = expected.get("kernel") {
                for i in 0..7 {
                    let row = array(&rows[i]);
                    for j in 0..7 {
                        let error = (kernel[i][j] - row[j]).abs();
                        maxima[2] = maxima[2].max(error);
                        assert!(
                            error < 1e-9 * (1.0 + row[j].abs()),
                            "{name}/{index} kernel{i},{j}: {error:e}"
                        );
                    }
                }
            }
        }
        println!("TASK/mTASK {name} maximum [energy,potential,kernel] errors {maxima:?}");
    }
}

#[test]
fn every_nonzero_component_partial_has_four_step_error_fits() {
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let u = [0.23, -0.17, 0.11, 0.13, -0.19, 0.07, -0.09];
    let denominator = 1.0 + u.iter().sum::<f64>();
    let mut failures = Vec::new();
    for (name, provider) in providers() {
        let output = evaluate(provider.as_ref(), &point, 2);
        let (potential, kernel) = output.potential_and_kernel(0, Family::MetaGga).unwrap();
        let exchange_active = |i| [0, 1, 2, 4, 5, 6].contains(&i);
        let same_spin = |i, j| {
            ([0, 2, 5].contains(&i) && [0, 2, 5].contains(&j))
                || ([1, 4, 6].contains(&i) && [1, 4, 6].contains(&j))
        };
        let mut targets = Vec::new();
        let mut labels = Vec::new();
        for j in 0..7 {
            if exchange_active(j) {
                targets.push((0, j, potential[j]));
                labels.push(format!("TASK/mTASK {name} potential{j}"));
            } else {
                assert_eq!(potential[j], 0.0);
            }
            for i in 0..=j {
                if same_spin(i, j) {
                    targets.push((i + 1, j, kernel[i][j]));
                    labels.push(format!("TASK/mTASK {name} kernel{i},{j}"));
                } else {
                    assert_eq!(kernel[i][j], 0.0);
                    assert_eq!(kernel[j][i], 0.0);
                }
            }
        }
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
                ([0.004, 0.002, 0.001, 0.0005], stencils::Stencil::Second),
                ([0.006, 0.003, 0.0015, 0.00075], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; targets.len()];
                for (step, &h) in steps.iter().enumerate() {
                    let mut directions = [[0.0; 7]; 8];
                    for j in 0..7 {
                        let at = |scale: f64| {
                            let mut displaced = point;
                            for k in 0..7 {
                                displaced[k] += scale * (u[k] + if j == k { 1.0 } else { 0.0 });
                            }
                            assert!(displaced[0] > 0.0 && displaced[1] > 0.0);
                            assert!(
                                displaced[2] >= 0.0
                                    && displaced[4] >= 0.0
                                    && displaced[3].powi(2) < displaced[2] * displaced[4]
                            );
                            for (r, s, t) in [(0, 2, 5), (1, 4, 6)] {
                                assert!(displaced[t] > displaced[s] / (8.0 * displaced[r]));
                            }
                            let value = evaluate(provider.as_ref(), &displaced, 2);
                            let (potential, _) =
                                value.potential_and_kernel(0, Family::MetaGga).unwrap();
                            let mut result = [0.0; 8];
                            result[0] = value.e[0];
                            result[1..].copy_from_slice(&potential);
                            result
                        };
                        let plus = at(h);
                        let minus = at(-h);
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            ([0.0; 8], [0.0; 8])
                        };
                        for i in 0..8 {
                            directions[i][j] =
                                stencil.first(plus[i], minus[i], plus2[i], minus2[i], h);
                        }
                    }
                    for (k, &(i, j, expected)) in targets.iter().enumerate() {
                        let shared = u.iter().zip(directions[i]).map(|(u, d)| u * d).sum::<f64>()
                            / denominator;
                        errors[k][step] = (directions[i][j] - shared - expected).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("TASK/mTASK local component derivatives", &failures);
}

#[test]
fn named_recipes_match_independent_full_energy_potential_and_kernel() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/task.json");
    let reference = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    for name in ["task", "mtask"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let expected_rows = rows(reference.get("combined").unwrap().get(name).unwrap());
        for (point, expected) in rows(reference.get("points").unwrap())
            .iter()
            .zip(expected_rows)
        {
            let point = array(point);
            let v: Vec<_> = point.iter().map(|&x| [x]).collect();
            let vars = GridVars {
                rho: [&v[0], &v[1]],
                sigma: Some([&v[2], &v[3], &v[4]]),
                tau: Some([&v[5], &v[6]]),
            };
            let mut output = XcOut::default();
            functional.eval(&vars, 2, &mut output).unwrap();
            let expected_energy = number(expected, "energy");
            assert!((output.e[0] - expected_energy).abs() < 1e-11 * (1.0 + expected_energy.abs()));
            let (potential, kernel) = output.potential_and_kernel(0, Family::MetaGga).unwrap();
            let expected_potential = array(expected.get("potential").unwrap());
            for i in 0..7 {
                if point[1] == 0.0 && [1, 3, 4, 6].contains(&i) {
                    continue;
                }
                assert!(
                    (potential[i] - expected_potential[i]).abs()
                        < 1e-10 * (1.0 + expected_potential[i].abs())
                );
            }
            if let Some(Value::Array(expected_kernel)) = expected.get("kernel") {
                for i in 0..7 {
                    let row = array(&expected_kernel[i]);
                    for j in 0..7 {
                        assert!(
                            (kernel[i][j] - row[j]).abs() < 1e-9 * (1.0 + row[j].abs()),
                            "{name} combined kernel{i},{j}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn every_parameter_partial_has_four_step_energy_fits() {
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let values: Vec<_> = point.iter().map(|&x| [x]).collect();
    let vars = GridVars {
        rho: [&values[0], &values[1]],
        sigma: Some([&values[2], &values[3], &values[4]]),
        tau: Some([&values[5], &values[6]]),
    };
    let u = [
        0.13, -0.11, 0.09, -0.07, 0.05, 0.06, -0.08, 0.12, -0.04, 0.10, 0.03,
    ];
    let denominator = 1.0 + u.iter().sum::<f64>();
    let mut failures = Vec::new();
    for (name, provider) in providers() {
        let mut gradient = Vec::new();
        provider.param_gradient(&vars, &mut gradient).unwrap();
        assert_eq!(gradient.len(), 11);
        let labels: Vec<_> = provider
            .param_names()
            .iter()
            .map(|p| format!("{name} parameter {p}"))
            .collect();
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.12, 0.06, 0.03, 0.015], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 11];
                for (step, &h) in steps.iter().enumerate() {
                    let mut directional = [0.0; 11];
                    for (axis, derivative) in directional.iter_mut().enumerate() {
                        let at = |scale: f64| {
                            let params: Vec<_> = provider
                                .params()
                                .iter()
                                .enumerate()
                                .map(|(j, p)| {
                                    p + scale * (u[j] + if axis == j { 1.0 } else { 0.0 })
                                })
                                .collect();
                            let shifted = provider.with_params(&params).unwrap();
                            evaluate(shifted.as_ref(), &point, 0).e[0]
                        };
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        *derivative = stencil.first(at(h), at(-h), plus2, minus2, h);
                    }
                    let shared =
                        u.iter().zip(directional).map(|(u, d)| u * d).sum::<f64>() / denominator;
                    for i in 0..11 {
                        errors[i][step] = (directional[i] - shared - gradient[i][0]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("TASK/mTASK all parameter partials", &failures);
}

fn original_pw92_density_partials(point: &[f64]) -> [f64; 15] {
    let v = [point[0], point[1], 0.0, 0.0, 0.0, 0.0, 0.0];
    let output = evaluate(lda::pw92().as_ref(), &v, 3);
    let (potential, kernel) = output.potential_and_kernel(0, Family::MetaGga).unwrap();
    let mut result = [0.0; 15];
    result[0] = output.e[0];
    for i in 0..2 {
        result[1 + i] = potential[i];
        for j in 0..2 {
            result[3 + 2 * i + j] = kernel[i][j];
            for k in 0..2 {
                result[7 + 4 * i + 2 * j + k] = output.v3rho3[i + j + k][0];
            }
        }
    }
    result
}

#[test]
fn original_pw92_matches_independent_recipe_differences() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/task.json");
    let reference = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    let exchange = rows(reference.get("components").unwrap().get("task").unwrap());
    let combined = rows(reference.get("combined").unwrap().get("task").unwrap());
    for (index, point) in rows(reference.get("points").unwrap()).iter().enumerate() {
        let point = array(point);
        let result = original_pw92_density_partials(&point);
        let energy = number(&combined[index], "energy") - number(&exchange[index], "energy");
        assert!((result[0] - energy).abs() < 1e-11 * (1.0 + energy.abs()));
        let cp = array(combined[index].get("potential").unwrap());
        let xp = array(exchange[index].get("potential").unwrap());
        for i in 0..2 {
            if point[i] == 0.0 {
                continue;
            }
            let expected = cp[i] - xp[i];
            assert!((result[1 + i] - expected).abs() < 1e-10 * (1.0 + expected.abs()));
        }
        if let (Some(ck), Some(xk)) = (combined[index].get("kernel"), exchange[index].get("kernel"))
        {
            for i in 0..2 {
                let cr = array(&rows(ck)[i]);
                let xr = array(&rows(xk)[i]);
                for j in 0..2 {
                    let expected = cr[j] - xr[j];
                    assert!(
                        (result[3 + 2 * i + j] - expected).abs() < 1e-9 * (1.0 + expected.abs())
                    );
                }
            }
        }
    }
}

#[test]
fn every_original_pw92_partial_through_three_has_four_step_fits() {
    let point = [0.31, 0.22];
    let u = [0.13, -0.09];
    let analytic = original_pw92_density_partials(&point);
    let targets = [
        (0, 0, 1),
        (0, 1, 2),
        (1, 0, 3),
        (1, 1, 4),
        (2, 1, 6),
        (3, 0, 7),
        (3, 1, 8),
        (4, 1, 10),
        (6, 1, 14),
    ];
    let labels: Vec<_> = (0..targets.len())
        .map(|i| format!("original PW92 partial{i}"))
        .collect();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
            ([0.004, 0.002, 0.001, 0.0005], stencils::Stencil::Second),
            ([0.006, 0.003, 0.0015, 0.00075], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; targets.len()];
            for (step, &h) in steps.iter().enumerate() {
                let mut directions = [[0.0; 2]; 15];
                for axis in 0..2 {
                    let at = |scale: f64| {
                        let v: Vec<_> = (0..2)
                            .map(|i| point[i] + scale * (u[i] + if axis == i { 1.0 } else { 0.0 }))
                            .collect();
                        assert!(v.iter().all(|v| *v > 0.0));
                        original_pw92_density_partials(&v)
                    };
                    let plus = at(h);
                    let minus = at(-h);
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        ([0.0; 15], [0.0; 15])
                    };
                    for i in 0..15 {
                        directions[i][axis] =
                            stencil.first(plus[i], minus[i], plus2[i], minus2[i], h);
                    }
                }
                for (target, &(lower, axis, expected)) in targets.iter().enumerate() {
                    let shared = (u[0] * directions[lower][0] + u[1] * directions[lower][1])
                        / (1.0 + u[0] + u[1]);
                    errors[target][step] =
                        (directions[lower][axis] - shared - analytic[expected]).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("original PW92 full first/second/third partials", &failures);
}

/// Unpack the public derivative blocks independently of production scatter code.
fn symmetric_third(output: &XcOut, i: usize, j: usize, k: usize) -> f64 {
    assert!(i <= j && j <= k && k < 7);
    let pair = |a, b, n| (0..a).map(|p| n - p).sum::<usize>() + b - a;
    let triple = |a, b, c, n| {
        let mut index = 0;
        for p in 0..n {
            for q in p..n {
                for r in q..n {
                    if (p, q, r) == (a, b, c) {
                        return index;
                    }
                    index += 1;
                }
            }
        }
        panic!("invalid symmetric triple")
    };
    let indices = [i, j, k];
    let rho = indices.iter().filter(|&&v| v < 2).count();
    let sigma = indices.iter().filter(|&&v| (2..5).contains(&v)).count();
    match (rho, sigma) {
        (3, 0) => output.v3rho3[i + j + k][0],
        (2, 1) => output.v3rho2sigma[3 * (i + j) + k - 2][0],
        (1, 2) => output.v3rhosigma2[6 * i + pair(j - 2, k - 2, 3)][0],
        (0, 3) => output.v3sigma3[triple(i - 2, j - 2, k - 2, 3)][0],
        (2, 0) => output.v3rho2tau[2 * (i + j) + k - 5][0],
        (1, 1) => output.v3rhosigmatau[6 * i + 2 * (j - 2) + k - 5][0],
        (1, 0) => output.v3rhotau2[3 * i + j + k - 10][0],
        (0, 2) => output.v3sigma2tau[2 * pair(i - 2, j - 2, 3) + k - 5][0],
        (0, 1) => output.v3sigmatau2[3 * (i - 2) + j + k - 10][0],
        (0, 0) => output.v3tau3[i + j + k - 15][0],
        _ => panic!("invalid derivative block"),
    }
}

#[test]
fn every_independent_third_partial_has_four_step_kernel_fits() {
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let u = [0.23, -0.17, 0.11, 0.13, -0.19, 0.07, -0.09];
    let denominator = 1.0 + u.iter().sum::<f64>();
    let mut failures = Vec::new();
    for (name, provider) in providers() {
        let output = evaluate(provider.as_ref(), &point, 3);
        let mut targets = Vec::new();
        let mut labels = Vec::new();
        for i in 0..7 {
            for j in i..7 {
                for k in j..7 {
                    let expected = symmetric_third(&output, i, j, k);
                    let same_spin = [i, j, k].iter().all(|v| [0, 2, 5].contains(v))
                        || [i, j, k].iter().all(|v| [1, 4, 6].contains(v));
                    if !same_spin {
                        assert_eq!(expected, 0.0, "structural zero {i},{j},{k}");
                    } else {
                        targets.push((i, j, k, expected));
                        labels.push(format!("TASK/mTASK {name} third{i},{j},{k}"));
                    }
                }
            }
        }
        assert_eq!(targets.len(), 20);
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
                ([0.004, 0.002, 0.001, 0.0005], stencils::Stencil::Second),
                ([0.006, 0.003, 0.0015, 0.00075], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; targets.len()];
                for (step, &h) in steps.iter().enumerate() {
                    let mut directions = [[[0.0; 7]; 7]; 7];
                    for axis in 0..7 {
                        let at = |scale: f64| {
                            let mut displaced = point;
                            for t in 0..7 {
                                displaced[t] += scale * (u[t] + if axis == t { 1.0 } else { 0.0 });
                            }
                            assert!(displaced[0] > 0.0 && displaced[1] > 0.0);
                            assert!(displaced[2] > 0.0 && displaced[4] > 0.0);
                            assert!(displaced[3].powi(2) < displaced[2] * displaced[4]);
                            for (r, s, t) in [(0, 2, 5), (1, 4, 6)] {
                                assert!(displaced[t] > displaced[s] / (8.0 * displaced[r]));
                            }
                            evaluate(provider.as_ref(), &displaced, 2)
                                .potential_and_kernel(0, Family::MetaGga)
                                .unwrap()
                                .1
                        };
                        let plus = at(h);
                        let minus = at(-h);
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            ([[0.0; 7]; 7], [[0.0; 7]; 7])
                        };
                        for i in 0..7 {
                            for j in 0..7 {
                                directions[i][j][axis] = stencil.first(
                                    plus[i][j],
                                    minus[i][j],
                                    plus2[i][j],
                                    minus2[i][j],
                                    h,
                                );
                            }
                        }
                    }
                    for (target, &(i, j, k, expected)) in targets.iter().enumerate() {
                        let shared = u
                            .iter()
                            .zip(directions[i][j])
                            .map(|(u, d)| u * d)
                            .sum::<f64>()
                            / denominator;
                        errors[target][step] = (directions[i][j][k] - shared - expected).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("TASK/mTASK complete local third derivatives", &failures);
}

#[test]
fn spin_scaling_uniform_exchange_one_electron_and_registry_contracts() {
    for (name, provider) in providers() {
        assert_eq!(provider.max_analytic_order(), 3);
        assert_eq!(provider.params().len(), provider.param_names().len());
        let point = [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21];
        let values: Vec<_> = point.iter().map(|&x| [x]).collect();
        let vars = GridVars {
            rho: [&values[0], &values[1]],
            sigma: Some([&values[2], &values[3], &values[4]]),
            tau: Some([&values[5], &values[6]]),
        };
        let mut unsupported = XcOut::default();
        unsupported.prepare(1, 3, Family::MetaGga);
        assert!(matches!(
            provider.eval_add(&vars, 4, 1.0, &mut unsupported),
            Err(dft_rs::DftError::DerivOrderUnsupported {
                requested: 4,
                max_analytic: 3,
                ..
            })
        ));
        let permutation = [1, 0, 4, 3, 2, 6, 5];
        let flipped: Vec<_> = permutation.iter().map(|&k| point[k]).collect();
        let a = evaluate(provider.as_ref(), &point, 2);
        let b = evaluate(provider.as_ref(), &flipped, 2);
        assert!((a.e[0] - b.e[0]).abs() < 1e-13, "{name}");
        let (pa, ka) = a.potential_and_kernel(0, Family::MetaGga).unwrap();
        let (pb, kb) = b.potential_and_kernel(0, Family::MetaGga).unwrap();
        for i in 0..7 {
            assert!((pa[i] - pb[permutation[i]]).abs() < 1e-12);
            for j in 0..7 {
                assert!((ka[i][j] - kb[permutation[i]][permutation[j]]).abs() < 1e-10);
            }
        }
        assert_eq!(evaluate(provider.as_ref(), &[0.0; 7], 0).e[0], 0.0);
    }
    for (name, provider) in providers() {
        for rho in [0.001_f64, 0.1, 1.0, 10.0] {
            let tau =
                0.3 * (6.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0) * rho.powf(5.0 / 3.0);
            let point = [rho, rho, 0.0, 0.0, 0.0, tau, tau];
            let params = provider.params();
            // Rounded published coefficients do not impose exactly Fx(0,1)=1.
            let h1 = params[3] - params[4] + params[5];
            let interpolation = params[6] - params[8] + params[10];
            let enhancement = params[2] + (1.0 - interpolation) * (h1 - params[2]);
            let expected =
                -1.5 * (6.0 / std::f64::consts::PI).cbrt() * rho.powf(4.0 / 3.0) * enhancement;
            let output = evaluate(provider.as_ref(), &point, 3);
            assert!((output.e[0] - expected).abs() < 1e-12 * (1.0 + expected.abs()));
            assert!(output.v3sigma3.iter().all(|v| v[0].is_finite()));
        }
        let point = [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21];
        let initial = evaluate(provider.as_ref(), &point, 0).e[0];
        for scale in [0.7_f64, 1.3] {
            let scaled: Vec<_> = point
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v * scale.powi(if i < 2 {
                        3
                    } else if i < 5 {
                        8
                    } else {
                        5
                    })
                })
                .collect();
            assert!(
                (evaluate(provider.as_ref(), &scaled, 0).e[0] - initial * scale.powi(4)).abs()
                    < 1e-12
            );
        }
        let combined = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        assert_eq!(combined.family, Family::MetaGga);
        assert_eq!(combined.hybrid.cx, 0.0);
        assert_eq!(combined.hybrid.clr, 0.0);
        assert_eq!(combined.parts.len(), 2);
        assert!(combined.required_nonlocal_kind().is_none());
        assert!(!dft_rs::xc::CORRELATION_NAMES.contains(&name));
        // Identical split names intentionally resolve the complete named recipe.
        let compatible = resolve(&FunctionalSpec::Split {
            x: name.into(),
            c: name.into(),
        })
        .unwrap();
        assert_eq!(compatible.parts.len(), 2);
        assert_eq!(compatible.parts[1].1.name(), "pw92");
        assert!(resolve(&FunctionalSpec::Split {
            x: "slater".into(),
            c: name.into(),
        })
        .is_err());
    }
}

#[test]
fn rks_uks_scf_and_fixed_density_exchange_correlation_terms() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/task.json");
    let reference = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    let set = data::load("sto-3g").unwrap();
    for entry in rows(reference.get("scf").unwrap()) {
        let model = entry.get("functional").unwrap().as_str().unwrap();
        let numbers = array(entry.get("numbers").unwrap());
        let positions: Vec<_> = rows(entry.get("positions").unwrap())
            .iter()
            .map(|r| {
                let r = array(r);
                [r[0], r[1], r[2]]
            })
            .collect();
        let charge = number(entry, "charge") as i32;
        let multiplicity = number(entry, "multiplicity") as usize;
        let mol = Molecule::new(
            numbers
                .iter()
                .zip(&positions)
                .map(|(z, p)| Atom {
                    z: *z as u8,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        )
        .with_charge(charge)
        .with_multiplicity(multiplicity);
        let basis = MolBasis::build(&mol, &set).unwrap();
        let grid = MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 9,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let density: Vec<_> = rows(entry.get("density").unwrap())
            .iter()
            .map(|spin| {
                Matrix::from_row_major(
                    basis.nao_sph,
                    basis.nao_sph,
                    rows(spin).iter().flat_map(array).collect(),
                )
            })
            .collect();
        let mut total = density[0].clone();
        for (value, beta) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
            *value += beta;
        }
        let core = kinetic_matrix(&basis).frobenius_dot(&total)
            + nuclear_matrix(&basis, &mol, None).frobenius_dot(&total);
        let eri = eri_incore(&basis, CoulombKernel::Coulomb, 512).unwrap();
        let mut hartree = 0.0;
        for i in 0..basis.nao_sph {
            for j in 0..basis.nao_sph {
                for k in 0..basis.nao_sph {
                    for l in 0..basis.nao_sph {
                        hartree += 0.5 * total[(i, j)] * total[(k, l)] * eri.get(i, j, k, l);
                    }
                }
            }
        }
        for (name, value) in [("core", core), ("coulomb", hartree)] {
            let error = value - number(entry, name);
            println!("TASK/mTASK multiplicity{multiplicity} fixed-density {name} error {error:e}");
            assert!(error.abs() < 1e-9);
        }
        let mut reconstruction = mol.nuclear_repulsion() + core + hartree;
        let exchange = match model {
            "task" => task::task_x(),
            "mtask" => task::mtask_x(),
            _ => panic!("unknown model"),
        };
        for (name, provider) in [("exchange", exchange), ("correlation", lda::pw92())] {
            let f = ResolvedFunctional::custom(
                name,
                vec![(1.0, Arc::from(provider))],
                HybridParams::PURE,
                Family::MetaGga,
            );
            let actual = dft_rs::ks::xc_contribution(&f, &basis, &grid, [&density[0], &density[1]])
                .unwrap()
                .exc;
            let error = actual - number(entry, name);
            reconstruction += actual;
            println!("TASK/mTASK multiplicity{multiplicity} fixed-density {name} error {error:e}");
            assert!(error.abs() < 5e-7);
        }
        assert!((reconstruction - number(entry, "total")).abs() < 5e-7);
        let system = System::molecule(numbers.iter().map(|z| *z as u8).collect(), positions);
        // Populations belong to the controls of the common public driver.
        let controls = Controls::from_value(&Value::Object(vec![
            ("xc".into(), Value::Str(model.into())),
            ("basis".into(), Value::Str("sto-3g".into())),
            ("charge".into(), Value::Int(charge as i64)),
            ("multiplicity".into(), Value::Int(multiplicity as i64)),
            ("unrestricted".into(), Value::Bool(multiplicity == 2)),
            ("grid_level".into(), Value::Int(9)),
            ("grid_weight_cutoff".into(), Value::Float(0.0)),
            ("conv_tol".into(), Value::Float(1e-12)),
            ("conv_tol_dp".into(), Value::Float(1e-10)),
            ("conv_tol_grad".into(), Value::Float(1e-10)),
        ]))
        .unwrap();
        let result = run("energy", &system, &controls).unwrap();
        let error = number(&result, "energy") - number(entry, "total");
        println!("TASK/mTASK multiplicity{multiplicity} SCF error {error:e}");
        assert!(error.abs() < 5e-7);
    }
}

#[test]
fn every_c1_molecular_gradient_component_has_four_step_fits() {
    let system = System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.17, -0.11, 0.23],
            [1.43, 0.81, -0.62],
            [-0.63, 1.59, -0.43],
            [-1.12, -1.36, -0.71],
        ],
    );
    let mut failures = Vec::new();
    for model in ["task", "mtask"] {
        let name = model;
        let controls = Controls::from_value(&Value::Object(vec![
            ("xc".into(), Value::Str(name.into())),
            ("basis".into(), Value::Str("sto-3g".into())),
            ("grid_level".into(), Value::Int(5)),
            ("grid_weight_cutoff".into(), Value::Float(0.0)),
            ("conv_tol".into(), Value::Float(1e-13)),
            ("conv_tol_grad".into(), Value::Float(1e-11)),
            ("conv_tol_dp".into(), Value::Float(1e-11)),
            ("dispersion".into(), Value::Str("none".into())),
        ]))
        .unwrap();
        let report = run("gradient", &system, &controls).unwrap();
        let gradient = array(report.get("gradient").unwrap());
        assert_eq!(gradient.len(), 12);
        for coordinate in 0..12 {
            stencils::fit_components_staged(
                &[format!("{name} C1 gradient {coordinate}")],
                &[
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    let energy = |h: f64| {
                        let mut s = system.clone();
                        s.positions[coordinate / 3][coordinate % 3] += h;
                        number(&run("energy", &s, &controls).unwrap(), "energy")
                    };
                    let mut errors = [0.0; 4];
                    for (i, &h) in steps.iter().enumerate() {
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (energy(2.0 * h), energy(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        errors[i] = (stencil.first(energy(h), energy(-h), p2, m2, h)
                            - gradient[coordinate])
                            .abs();
                    }
                    vec![errors]
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("TASK/mTASK C1 gradients", &failures);
}
