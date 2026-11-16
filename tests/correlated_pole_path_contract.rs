// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex as C;
use dft_rs::response::correlated_tda::{correlated_pole_taylor, PropagatorTaylor};
use dft_rs::response::spinor_tda::{spinor_tda_eigenpair_taylor, SpinorTdaConfig};

fn polynomial<S: Scalar>(x: S, y: S, seed: usize) -> S {
    let s = seed as f64;
    S::from_f64(0.09 + 0.013 * s)
        + x * S::from_f64(0.12 - 0.004 * s)
        + y * S::from_f64(-0.08 + 0.003 * s)
        + x * y * S::from_f64(0.027 + 0.001 * s)
        + x * x * S::from_f64(0.016 - 0.0003 * s)
        + y * y * S::from_f64(-0.021 + 0.0007 * s)
}
fn entries<S: Scalar>(x: S, y: S, block: usize) -> [S; 8] {
    let mut v = std::array::from_fn(|i| polynomial(x, y, block * 8 + i));
    if block != 4 {
        v[4] = v[2];
        v[5] = -v[3];
        v[1] = S::from_f64(0.0);
        v[7] = S::from_f64(0.0);
    }
    if block == 0 {
        v[0] = v[0] + S::from_f64(0.7);
        v[6] = v[6] + S::from_f64(2.0);
    }
    if block == 3 {
        v[0] = v[0] + S::from_f64(3.1);
        v[6] = v[6] + S::from_f64(4.6);
    }
    v
}
fn matrix(values: [f64; 8]) -> BlochMatrix {
    let mut m = BlochMatrix::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            m.re[(i, j)] = values[4 * i + 2 * j];
            m.im[(i, j)] = values[4 * i + 2 * j + 1];
        }
    }
    m
}
fn jets(dx: f64, dy: f64) -> [Vec<BlochMatrix>; 5] {
    let x = Taylor::<3>::from_coefficients(&[0.17, dx]);
    let y = Taylor::<3>::from_coefficients(&[-0.23, dy]);
    std::array::from_fn(|block| {
        let values = entries(x, y, block);
        (0..3).map(|k| matrix(values.map(|v| v.c[k]))).collect()
    })
}
fn borrowed(data: &[Vec<BlochMatrix>; 5]) -> PropagatorTaylor<'_> {
    PropagatorTaylor {
        parent: &data[0],
        singles: &data[1],
        metric: &data[2],
        doubles: &data[3],
        coupling: &data[4],
    }
}
fn get(m: &BlochMatrix, i: usize, j: usize) -> C {
    C::new(m.re[(i, j)], m.im[(i, j)])
}
fn flatten(shift: f64, v: &BlochMatrix) -> [f64; 5] {
    [
        shift,
        v.re[(0, 0)],
        v.im[(0, 0)],
        v.re[(1, 0)],
        v.im[(1, 0)],
    ]
}

// Independent two-by-two eigenvector and direct adjugate inverse. Neither
// the production eigensolver nor its coefficient recurrence is used here.
fn exact(x: f64, y: f64, base: &BlochMatrix) -> [f64; 5] {
    let m: [BlochMatrix; 5] = std::array::from_fn(|b| matrix(entries(x, y, b)));
    let a = m[0].re[(0, 0)];
    let d = m[0].re[(1, 1)];
    let z = get(&m[0], 0, 1);
    let omega = 0.5 * (a + d) - ((0.5 * (a - d)).powi(2) + z.re * z.re + z.im * z.im).sqrt();
    let norm = (z.re * z.re + z.im * z.im + (a - omega).powi(2)).sqrt();
    let mut u = [-z / norm, C::new((a - omega) / norm, 0.0)];
    let overlap = get(base, 0, 0).conj() * u[0] + get(base, 1, 0).conj() * u[1];
    for v in &mut u {
        *v = *v * (overlap.conj() / overlap.abs());
    }
    let mut w = [C::ZERO; 2];
    for (j, v) in w.iter_mut().enumerate() {
        for (i, &ui) in u.iter().enumerate() {
            *v += get(&m[4], i, j).conj() * ui;
        }
    }
    let r00 = C::new(omega, 0.0) - get(&m[3], 0, 0);
    let r11 = C::new(omega, 0.0) - get(&m[3], 1, 1);
    let r01 = -get(&m[3], 0, 1);
    let r10 = -get(&m[3], 1, 0);
    let det = r00 * r11 - r01 * r10;
    let v = [
        (r11 * w[0] - r01 * w[1]) / det,
        (r00 * w[1] - r10 * w[0]) / det,
    ];
    let mut shift = C::ZERO;
    for i in 0..2 {
        for j in 0..2 {
            shift += u[i].conj() * (get(&m[1], i, j) - get(&m[2], i, j) * omega) * u[j];
        }
        shift += w[i].conj() * v[i];
    }
    assert!(shift.im.abs() < 1e-13);
    [shift.re, v[0].re, v[0].im, v[1].re, v[1].im]
}

#[test]
fn all_pole_and_noncanonical_double_gradient_hessian_components_have_four_step_fits() {
    let config = SpinorTdaConfig::default();
    let data = [jets(1.0, 0.0), jets(0.0, 1.0), jets(1.0, 1.0)];
    let paths: Vec<_> = data
        .iter()
        .map(|d| correlated_pole_taylor(&borrowed(d), 0, &config).unwrap())
        .collect();
    let base = spinor_tda_eigenpair_taylor(&data[0][0], 0, &config)
        .unwrap()
        .vectors
        .remove(0);
    for path in &paths {
        for k in 0..3 {
            assert_eq!(
                path.energies[k],
                path.parent_energies[k] + path.corrections[k]
            );
        }
        assert!(path.double_equation_residuals.iter().all(|r| *r < 1e-12));
        assert!(path.parent_equation_residuals.iter().all(|r| *r < 1e-12));
        assert!(path.parent_normalization_defects.iter().all(|r| *r < 1e-12));
        assert!(path.smallest_double_gap > 1.0);
    }
    let analytic0 = flatten(paths[0].corrections[0], &paths[0].double_vectors[0]);
    for (a, b) in analytic0.iter().zip(exact(0.17, -0.23, &base)) {
        assert!((a - b).abs() < 1e-13);
    }
    let first: Vec<_> = paths
        .iter()
        .map(|p| flatten(p.corrections[1], &p.double_vectors[1]))
        .collect();
    let second: Vec<_> = paths
        .iter()
        .map(|p| flatten(p.corrections[2], &p.double_vectors[2]))
        .collect();
    let mut names = Vec::new();
    let mut analytic = Vec::new();
    for axis in 0..2 {
        for c in 0..5 {
            names.push(format!("propagator gradient {axis},{c}"));
            analytic.push(first[axis][c]);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            for c in 0..5 {
                names.push(format!("propagator Hessian {i},{j},{c}"));
                analytic.push(if i == j {
                    2.0 * second[i][c]
                } else {
                    second[2][c] - second[0][c] - second[1][c]
                });
            }
        }
    }
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[
            [0.08, 0.04, 0.02, 0.01],
            [0.04, 0.02, 0.01, 0.005],
            [0.16, 0.08, 0.04, 0.02],
        ],
        |steps| {
            let mut errors = vec![[0.0; 4]; 30];
            let at = |x: f64, y: f64| exact(0.17 + x, -0.23 + y, &base);
            let zero = at(0.0, 0.0);
            for (k, &h) in steps.iter().enumerate() {
                for i in 0..2 {
                    let p = at(if i == 0 { h } else { 0.0 }, if i == 1 { h } else { 0.0 });
                    let m = at(if i == 0 { -h } else { 0.0 }, if i == 1 { -h } else { 0.0 });
                    for c in 0..5 {
                        errors[5 * i + c][k] =
                            ((p[c] - m[c]) / (2.0 * h) - analytic[5 * i + c]).abs();
                    }
                    for j in 0..2 {
                        let fd: [f64; 5] = if i == j {
                            std::array::from_fn(|c| (p[c] + m[c] - 2.0 * zero[c]) / (h * h))
                        } else {
                            let pp = at(h, h);
                            let pm = at(h, -h);
                            let mp = at(-h, h);
                            let mm = at(-h, -h);
                            std::array::from_fn(|c| (pp[c] - pm[c] - mp[c] + mm[c]) / (4.0 * h * h))
                        };
                        for c in 0..5 {
                            let index = 10 + 5 * (2 * i + j) + c;
                            errors[index][k] = (fd[c] - analytic[index]).abs();
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("complete propagator derivatives", &failures);
}

#[test]
fn propagator_rejects_missing_jets_budget_and_resonance() {
    let mut d = jets(1.0, 0.0);
    let config = SpinorTdaConfig::default();
    let mut zero = jets(1.0, 0.0);
    for block in [1, 2, 4] {
        for coefficient in &mut zero[block] {
            *coefficient = BlochMatrix::zeros(2, 2);
        }
    }
    let limit = correlated_pole_taylor(&borrowed(&zero), 0, &config).unwrap();
    assert_eq!(limit.corrections, [0.0; 3]);
    assert_eq!(limit.parent_energies, limit.energies);
    for vector in limit.double_vectors {
        assert!(vector
            .re
            .as_slice()
            .iter()
            .chain(vector.im.as_slice())
            .all(|v| *v == 0.0));
    }
    zero[3] = vec![BlochMatrix::zeros(0, 0); 3];
    zero[4] = vec![BlochMatrix::zeros(2, 0); 3];
    let empty = correlated_pole_taylor(&borrowed(&zero), 0, &config).unwrap();
    assert_eq!(empty.corrections, [0.0; 3]);
    assert_eq!(empty.parent_energies, empty.energies);
    assert_eq!(empty.double_equation_residuals, [0.0; 3]);
    let mut budget = config.clone();
    budget.memory_mb = 0;
    assert!(matches!(
        correlated_pole_taylor(&borrowed(&d), 0, &budget),
        Err(dft_rs::DftError::ResourceLimit { .. })
    ));
    d[4].pop();
    assert!(matches!(
        correlated_pole_taylor(&borrowed(&d), 0, &config),
        Err(dft_rs::DftError::InvalidInput(_))
    ));
    let mut d = jets(1.0, 0.0);
    let parent = spinor_tda_eigenpair_taylor(&d[0], 0, &config).unwrap();
    d[3][0] = BlochMatrix::zeros(2, 2);
    d[3][0].re[(0, 0)] = parent.energies[0];
    d[3][0].re[(1, 1)] = 4.0;
    assert!(matches!(
        correlated_pole_taylor(&borrowed(&d), 0, &config),
        Err(dft_rs::DftError::Unsupported { .. })
    ));
}

#[test]
fn pole_coefficients_are_invariant_under_independent_complex_basis_rotations() {
    use dft_rs::dfpt::orbitals::{adjoint, matmul};
    let data = jets(0.7, -0.4);
    let mut single = BlochMatrix::zeros(2, 2);
    let mut double = BlochMatrix::zeros(2, 2);
    let (s, c) = 0.37_f64.sin_cos();
    single.re[(0, 0)] = c;
    single.re[(1, 1)] = c;
    single.im[(0, 1)] = s;
    single.im[(1, 0)] = s;
    let (s, c) = 0.61_f64.sin_cos();
    double.re[(0, 0)] = c;
    double.re[(1, 1)] = c;
    double.re[(0, 1)] = s;
    double.re[(1, 0)] = -s;
    let changed: [Vec<BlochMatrix>; 5] = std::array::from_fn(|block| {
        let left = if block == 3 { &double } else { &single };
        let right = if block >= 3 { &double } else { &single };
        data[block]
            .iter()
            .map(|a| matmul(&matmul(&adjoint(left).unwrap(), a).unwrap(), right).unwrap())
            .collect()
    });
    let config = SpinorTdaConfig::default();
    let a = correlated_pole_taylor(&borrowed(&data), 0, &config).unwrap();
    let b = correlated_pole_taylor(&borrowed(&changed), 0, &config).unwrap();
    for k in 0..3 {
        assert!((a.corrections[k] - b.corrections[k]).abs() < 1e-12);
        assert!((a.energies[k] - b.energies[k]).abs() < 1e-12);
    }
}
