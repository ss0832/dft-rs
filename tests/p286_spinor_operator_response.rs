// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! General complex one-body response, including every transverse spin block.
//! All nine susceptibilities and all sixteen independent Hermitian density
//! components per source have independent four-step displaced-Hamiltonian
//! fits, for one, two and three occupied spinors. No spatial/physical grid
//! or functional approximation enters this nonsymmetric finite matrix oracle.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::dfpt::orbitals::{band_density, generalized_orbitals};
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;
use dft_rs::response::spinor::constrained_spinor_operator_response_with_kernel;
use dft_rs::response::spinor::spinor_operator_response_with_kernel;
use dft_rs::response::ResponseSolverConfig;

fn descriptors(matrix: &BlochMatrix) -> Vec<f64> {
    let mut out = (0..4).map(|i| matrix.re[(i, i)]).collect::<Vec<_>>();
    for i in 0..4 {
        for j in i + 1..4 {
            out.push(matrix.re[(i, j)]);
            out.push(matrix.im[(i, j)]);
        }
    }
    out
}

#[test]
fn constrained_full_spin_response_has_all_component_displaced_stationary_fits() {
    let mut identity = BlochMatrix::zeros(3, 3);
    let mut h = identity.clone();
    let mut q = identity.clone();
    let mut source = identity.clone();
    for i in 0..3 {
        identity.re[(i, i)] = 1.0;
        h.re[(i, i)] = [-0.9, 0.2, 1.4][i];
    }
    for i in 0..3 {
        for j in i..3 {
            for (matrix, scale, phase) in [(&mut q, 0.27, 0.3), (&mut source, 0.19, 1.1)] {
                matrix.re[(i, j)] = scale * ((i + 3 * j + 1) as f64 + phase).sin();
                matrix.re[(j, i)] = matrix.re[(i, j)];
                if i != j {
                    matrix.im[(i, j)] = scale * ((2 * i + j + 1) as f64 + phase).cos();
                    matrix.im[(j, i)] = -matrix.im[(i, j)];
                }
            }
            if i != j {
                h.re[(i, j)] = 0.053 * (i + j + 1) as f64;
                h.re[(j, i)] = h.re[(i, j)];
                h.im[(i, j)] = 0.037 * (2 * i + j + 1) as f64;
                h.im[(j, i)] = -h.im[(i, j)];
            }
        }
    }
    let orbitals = generalized_orbitals(&h, &identity, 1e-12).unwrap();
    let mut twice = q.clone();
    twice.add_scaled(&q, 1.0);
    let constraints = [q.clone(), twice, BlochMatrix::zeros(3, 3)];
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut failures = Vec::new();
    for occupied in [1, 2] {
        let occupations = (0..3)
            .map(|i| if i < occupied { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let target = q.trace_product(&band_density(&orbitals, &occupations).unwrap());
        let report = constrained_spinor_operator_response_with_kernel(
            &orbitals.coefficients,
            &orbitals.energies,
            occupied,
            &|_| Ok(BlochMatrix::zeros(3, 3)),
            &[source.clone()],
            &constraints,
            &[Complex::ZERO],
            &config,
        )
        .unwrap()
        .remove(0);
        assert_eq!(report.constraint_rank, 1);
        assert!(report.residuals[0] < 1e-11);
        assert!(report.constraint_residuals[0]
            .iter()
            .all(|v| v.abs() < 1e-11));
        let describe = |density: &BlochMatrix, lambda: f64| {
            let mut values = vec![source.trace_product(density), lambda];
            values.extend((0..3).map(|i| density.re[(i, i)]));
            for i in 0..3 {
                for j in i + 1..3 {
                    values.push(density.re[(i, j)]);
                    values.push(density.im[(i, j)]);
                }
            }
            values
        };
        // Two redundant constraints share a unique physical constraining field.
        let lambda =
            report.constraint_multipliers[0][0].re + 2.0 * report.constraint_multipliers[0][1].re;
        let exact = describe(&report.density_responses[0], lambda);
        let names = (0..11)
            .map(|i| format!("constrained occupied={occupied} entry={i}"))
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let at = |t: f64| {
                    let mut lambda = 0.0;
                    for iteration in 0..40 {
                        let mut displaced = h.clone();
                        displaced.add_scaled(&source, t);
                        displaced.add_scaled(&q, lambda);
                        let orbitals = generalized_orbitals(&displaced, &identity, 1e-12).unwrap();
                        let density = band_density(&orbitals, &occupations).unwrap();
                        let residual = q.trace_product(&density) - target;
                        if residual.abs() < 8e-15 {
                            return describe(&density, lambda);
                        }
                        let c = &orbitals.coefficients;
                        let mut derivative = 0.0;
                        for i in 0..occupied {
                            for a in occupied..3 {
                                let mut element = Complex::ZERO;
                                for m in 0..3 {
                                    for n in 0..3 {
                                        element += Complex::new(c.re[(m, a)], c.im[(m, a)]).conj()
                                            * Complex::new(q.re[(m, n)], q.im[(m, n)])
                                            * Complex::new(c.re[(n, i)], c.im[(n, i)]);
                                    }
                                }
                                derivative += 2.0 * element.abs().powi(2)
                                    / (orbitals.energies[i] - orbitals.energies[a]);
                            }
                        }
                        lambda -= residual / derivative;
                        assert!(
                            iteration < 39,
                            "independent constrained stationary solve failed"
                        );
                    }
                    unreachable!()
                };
                let mut errors = vec![[0.0; 4]; 11];
                for (step, &h) in steps.iter().enumerate() {
                    let (p, m) = (at(h), at(-h));
                    let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        (vec![0.0; 11], vec![0.0; 11])
                    };
                    for i in 0..11 {
                        errors[i][step] =
                            (stencil.first(p[i], m[i], p2[i], m2[i], h) - exact[i]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("full constrained spinor response", &failures);
}

#[test]
fn exact_goldstone_modes_are_constrained_without_changing_the_hessian() {
    let mut c = BlochMatrix::zeros(2, 2);
    c.re[(0, 0)] = 1.0;
    c.re[(1, 1)] = 1.0;
    let mut x = BlochMatrix::zeros(2, 2);
    x.re[(0, 1)] = 0.5;
    x.re[(1, 0)] = 0.5;
    let mut y = BlochMatrix::zeros(2, 2);
    y.im[(0, 1)] = -0.5;
    y.im[(1, 0)] = 0.5;
    let operators = [x, y];
    let result = constrained_spinor_operator_response_with_kernel(
        &c,
        &[0.0, 0.0],
        1,
        &|_| Ok(BlochMatrix::zeros(2, 2)),
        &operators,
        &operators,
        &[Complex::ZERO],
        &ResponseSolverConfig::default(),
    )
    .unwrap()
    .remove(0);
    assert_eq!(result.constraint_rank, 2);
    for source in 0..2 {
        assert!(result.residuals[source] < 1e-12);
        assert!(result.density_responses[source]
            .re
            .as_slice()
            .iter()
            .chain(result.density_responses[source].im.as_slice())
            .all(|v| v.abs() < 1e-12));
        for constraint in 0..2 {
            assert!(
                (result.constraint_multipliers[source][constraint]
                    - Complex::from(if source == constraint { -1.0 } else { 0.0 }))
                .abs()
                    < 1e-12
            );
        }
    }
}

#[test]
fn every_source_observable_and_spin_density_component_has_an_independent_fit() {
    let mut overlap = BlochMatrix::zeros(4, 4);
    let mut h = overlap.clone();
    for i in 0..4 {
        overlap.re[(i, i)] = 1.0;
        h.re[(i, i)] = [-1.1, -0.6, 0.7, 1.3][i];
    }
    for i in 0..4 {
        for j in i + 1..4 {
            h.re[(i, j)] = 0.071 * ((i + 2 * j + 1) as f64).sin();
            h.re[(j, i)] = h.re[(i, j)];
            h.im[(i, j)] = 0.053 * ((2 * i + j + 1) as f64).cos();
            h.im[(j, i)] = -h.im[(i, j)];
        }
    }
    let orbitals = generalized_orbitals(&h, &overlap, 1e-12).unwrap();
    let sources: Vec<_> = (0..3)
        .map(|axis| {
            let mut op = BlochMatrix::zeros(4, 4);
            for i in 0..4 {
                for j in i..4 {
                    op.re[(i, j)] = 0.23 * ((i + 2 * j + 3 * axis + 1) as f64).sin()
                        + 0.03 * (i + j + 1) as f64;
                    op.re[(j, i)] = op.re[(i, j)];
                    if i != j {
                        op.im[(i, j)] = 0.19 * ((2 * i + j + 2 * axis + 1) as f64).cos();
                        op.im[(j, i)] = -op.im[(i, j)];
                    }
                }
            }
            op
        })
        .collect();
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut failures = Vec::new();
    for occupied in 1..=3 {
        let report = spinor_operator_response_with_kernel(
            &orbitals.coefficients,
            &orbitals.energies,
            occupied,
            &|_| Ok(BlochMatrix::zeros(4, 4)),
            &sources,
            &[Complex::ZERO],
            &config,
        )
        .unwrap()
        .remove(0);
        for source in 0..3 {
            let mut exact = report
                .susceptibility
                .iter()
                .map(|row| {
                    assert!(row[source].im.abs() < 1e-12);
                    row[source].re
                })
                .collect::<Vec<_>>();
            exact.extend(descriptors(&report.density_responses[source]));
            let names = (0..19)
                .map(|i| {
                    format!("occupied={occupied} source={source} full-spin response entry {i}")
                })
                .collect::<Vec<_>>();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    let mut errors = vec![[0.0; 4]; 19];
                    for (step, &value) in steps.iter().enumerate() {
                        let at = |t: f64| {
                            let mut field = h.clone();
                            field.add_scaled(&sources[source], t);
                            let orbitals = generalized_orbitals(&field, &overlap, 1e-12).unwrap();
                            let occupations = (0..4)
                                .map(|i| if i < occupied { 1.0 } else { 0.0 })
                                .collect::<Vec<_>>();
                            let density = band_density(&orbitals, &occupations).unwrap();
                            let mut result = sources
                                .iter()
                                .map(|a| a.trace_product(&density))
                                .collect::<Vec<_>>();
                            result.extend(descriptors(&density));
                            result
                        };
                        let (p, m) = (at(value), at(-value));
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * value), at(-2.0 * value))
                        } else {
                            (vec![0.0; 19], vec![0.0; 19])
                        };
                        for i in 0..19 {
                            errors[i][step] =
                                (stencil.first(p[i], m[i], p2[i], m2[i], value) - exact[i]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
        assert!(report.residuals.iter().all(|r| *r < 1e-11));
    }
    components::assert_no_failures("full complex one-body spinor response", &failures);
}
