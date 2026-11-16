// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Full complex Nambu density against independent commuting spectral factors.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::nambu::{nambu_density_taylor, NambuTaylor};
use dft_rs::response::ResponseSolverConfig;
type J = Taylor<5>;
type C = (J, J);
type M = Vec<C>;
fn z() -> C {
    (J::from_f64(0.0), J::from_f64(0.0))
}
fn multiply(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn product(a: &M, b: &M, n: usize) -> M {
    (0..n * n)
        .map(|p| {
            (0..n).fold(z(), |s, k| {
                let v = multiply(a[(p / n) * n + k], b[k * n + p % n]);
                (s.0 + v.0, s.1 + v.1)
            })
        })
        .collect()
}
fn adjoint(a: &M, n: usize) -> M {
    (0..n * n)
        .map(|p| (a[(p % n) * n + p / n].0, -a[(p % n) * n + p / n].1))
        .collect()
}
fn matrices(a: &M, n: usize) -> Vec<BlochMatrix> {
    (0..5)
        .map(|k| {
            let mut out = BlochMatrix::zeros(n, n);
            for (p, v) in a.iter().enumerate() {
                out.re[(p / n, p % n)] = v.0.c[k];
                out.im[(p / n, p % n)] = v.1.c[k];
            }
            out
        })
        .collect()
}
fn zero(n: usize) -> M {
    vec![z(); n * n]
}
fn fermi(x: J, temperature: J) -> J {
    let one = J::from_f64(1.0);
    if x.value() >= 0.0 {
        let e = (-x / temperature).exp();
        e / (one + e)
    } else {
        one / (one + (x / temperature).exp())
    }
}
struct Model {
    normal: Vec<BlochMatrix>,
    pair: Vec<BlochMatrix>,
    overlap: Vec<BlochMatrix>,
    mu: Vec<f64>,
    kt: Vec<f64>,
    exact: M,
    observables: [J; 6],
}
fn model(center: f64, temperature: f64) -> Model {
    let t = J::from_coefficients(&[center, 1.0]);
    let f = J::from_f64;
    let one = f(1.0);
    let a = f(0.3) / (one - t.scale(0.11));
    let b = [
        f(0.07) + t.scale(0.03),
        f(-0.04) / (one + t.scale(0.13)),
        f(0.05) * (one + t.scale(0.2)).powi(2),
    ];
    let delta = (
        f(0.12) / (one - t.scale(0.17)),
        f(0.08) * (one + t.scale(0.15)),
    );
    let kt = f(temperature) / (one - t.scale(0.1));
    let mu = f(0.04) + t.scale(0.01);
    let mut h = zero(2);
    h[0] = (a + mu + b[2], f(0.0));
    h[3] = (a + mu - b[2], f(0.0));
    h[1] = (b[0], -b[1]);
    h[2] = (b[0], b[1]);
    let mut pair = zero(2);
    pair[1] = delta;
    pair[2] = (-delta.0, -delta.1);
    // Independent nonunitary moving frame, including a complex off-diagonal.
    let mut frame = zero(2);
    frame[0] = (one + t.scale(0.1), f(0.0));
    frame[1] = (
        f(0.1) * (one + t.scale(0.16)),
        f(0.2) * (one + t.scale(0.14)),
    );
    frame[3] = (one - t.scale(0.08), f(0.0));
    let mut inverse = zero(2);
    inverse[0] = (one / frame[0].0, f(0.0));
    inverse[3] = (one / frame[3].0, f(0.0));
    inverse[1] = (
        -frame[1].0 / (frame[0].0 * frame[3].0),
        -frame[1].1 / (frame[0].0 * frame[3].0),
    );
    let normal = product(&product(&adjoint(&frame, 2), &h, 2), &frame, 2);
    let overlap = product(&adjoint(&frame, 2), &frame, 2);
    let conjugate = frame.iter().map(|&(r, i)| (r, -i)).collect();
    let transformed_pair = product(&product(&adjoint(&frame, 2), &pair, 2), &conjugate, 2);
    let energy = (a * a + delta.0 * delta.0 + delta.1 * delta.1).sqrt();
    let splitting = b.iter().fold(f(0.0), |s, &v| s + v * v).sqrt();
    let mut q = zero(4);
    let mut spin = q.clone();
    for i in 0..2 {
        q[4 * i + i] = (a, f(0.0));
        q[4 * (i + 2) + i + 2] = (-a, f(0.0));
        for j in 0..2 {
            q[4 * i + j + 2] = pair[2 * i + j];
            q[4 * (j + 2) + i] = (pair[2 * i + j].0, -pair[2 * i + j].1);
            let mut v = h[2 * i + j];
            if i == j {
                v.0 = v.0 - a - mu;
            }
            spin[4 * i + j] = v;
            spin[4 * (i + 2) + j + 2] = (-v.0, v.1);
        }
    }
    let mut density = zero(4);
    for sign in [-1.0, 1.0] {
        for band in [-1.0, 1.0] {
            let mut p = zero(4);
            let mut r = p.clone();
            for i in 0..4 {
                for j in 0..4 {
                    let unit = if i == j { one } else { f(0.0) };
                    p[4 * i + j] = (
                        (unit + q[4 * i + j].0 / energy.scale(sign)).scale(0.5),
                        (q[4 * i + j].1 / energy).scale(0.5 * sign),
                    );
                    r[4 * i + j] = (
                        (unit + spin[4 * i + j].0 / splitting.scale(band)).scale(0.5),
                        (spin[4 * i + j].1 / splitting).scale(0.5 * band),
                    );
                }
            }
            let projector = product(&p, &r, 4);
            let occupation = fermi(energy.scale(sign) + splitting.scale(band), kt);
            for (v, w) in density.iter_mut().zip(projector) {
                v.0 = v.0 + w.0 * occupation;
                v.1 = v.1 + w.1 * occupation;
            }
        }
    }
    let number = density[0].0 + density[5].0;
    let mut transform = zero(4);
    for i in 0..2 {
        for j in 0..2 {
            transform[4 * i + j] = inverse[2 * i + j];
            transform[4 * (i + 2) + j + 2] = (inverse[2 * i + j].0, -inverse[2 * i + j].1);
        }
    }
    let exact = product(
        &product(&transform, &density, 4),
        &adjoint(&transform, 4),
        4,
    );
    let observables = [
        exact[0].0,
        exact[1].0,
        exact[1].1,
        -exact[3].0,
        -exact[3].1,
        number,
    ];
    Model {
        normal: matrices(&normal, 2),
        pair: matrices(&transformed_pair, 2),
        overlap: matrices(&overlap, 2),
        mu: mu.c.to_vec(),
        kt: kt.c.to_vec(),
        exact,
        observables,
    }
}
fn run(m: &Model) -> NambuTaylor {
    nambu_density_taylor(
        &m.normal,
        &m.pair,
        &m.overlap,
        &m.mu,
        &m.kt,
        4,
        1e-12,
        &ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        },
    )
    .unwrap()
}
#[test]
fn general_complex_pair_density_has_moving_metric_four_order_independent_fits() {
    let steps = [0.08, 0.04, 0.02, 0.01];
    for temperature in [0.2, 0.00002] {
        let m = model(0.0, temperature);
        let actual = run(&m);
        for k in 0..5 {
            for p in 0..16 {
                assert!(
                    (actual.thermal.density[k].re[(p / 4, p % 4)] - m.exact[p].0.c[k]).abs()
                        < 1e-11
                );
                assert!(
                    (actual.thermal.density[k].im[(p / 4, p % 4)] - m.exact[p].1.c[k]).abs()
                        < 1e-11
                );
            }
        }
        let observables = (0..5)
            .map(|k| {
                [
                    actual.normal_density[k].re[(0, 0)],
                    actual.normal_density[k].re[(0, 1)],
                    actual.normal_density[k].im[(0, 1)],
                    actual.anomalous_density[k].re[(0, 1)],
                    actual.anomalous_density[k].im[(0, 1)],
                    actual.electron_number[k],
                ]
            })
            .collect::<Vec<_>>();
        println!("Nambu raw kT={temperature} PH={:?} metric={:?} commutator={:?} inverse={:.12e} tail={:.12e} doublings={}",actual.particle_hole_defects,
            actual.thermal.metric_defects,actual.thermal.relative_commutator_defects,actual.thermal.maximum_inverse_residual,
            actual.thermal.exponential_tail_estimate,actual.thermal.doubling_steps);
        for j in 0..6 {
            for k in 1..=4 {
                let target = k as f64 * observables[k][j];
                let estimates = steps
                    .iter()
                    .map(|&h| {
                        (model(h, temperature).observables[j].c[k - 1]
                            - model(-h, temperature).observables[j].c[k - 1])
                            / (2.0 * h)
                    })
                    .collect::<Vec<_>>();
                let errors = estimates
                    .iter()
                    .map(|v| (v - target).abs())
                    .collect::<Vec<_>>();
                let fit = convergence::error_power_fit(&steps, &errors);
                let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
                println!("Nambu density kT={temperature} component={j} order={k} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
                assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999 && rich < 1e-7);
            }
        }
        // A global pairing phase changes only the anomalous-density phase.
        let mut changed = model(0.0, temperature);
        let (c, s) = (0.7_f64.cos(), 0.7_f64.sin());
        for a in &mut changed.pair {
            for i in 0..2 {
                for j in 0..2 {
                    let (r, v) = (a.re[(i, j)], a.im[(i, j)]);
                    a.re[(i, j)] = c * r - s * v;
                    a.im[(i, j)] = s * r + c * v;
                }
            }
        }
        let rotated = run(&changed);
        for k in 0..5 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (rotated.normal_density[k].re[(i, j)]
                            - actual.normal_density[k].re[(i, j)])
                            .abs()
                            < 1e-11
                    );
                    assert!(
                        (rotated.anomalous_density[k].re[(i, j)]
                            - c * actual.anomalous_density[k].re[(i, j)]
                            + s * actual.anomalous_density[k].im[(i, j)])
                            .abs()
                            < 1e-11
                    );
                    assert!(
                        (rotated.anomalous_density[k].im[(i, j)]
                            - s * actual.anomalous_density[k].re[(i, j)]
                            - c * actual.anomalous_density[k].im[(i, j)])
                            .abs()
                            < 1e-11
                    );
                }
            }
        }
    }
}
#[test]
fn normal_state_at_exact_fermi_degeneracy_has_the_closed_matrix_series() {
    let mut normal = vec![BlochMatrix::zeros(2, 2); 2];
    normal[1].re[(0, 0)] = 0.2;
    normal[1].re[(1, 1)] = -0.2;
    normal[1].re[(0, 1)] = 0.1;
    normal[1].re[(1, 0)] = 0.1;
    let mut overlap = BlochMatrix::zeros(2, 2);
    overlap.re[(0, 0)] = 1.0;
    overlap.re[(1, 1)] = 1.0;
    let actual = nambu_density_taylor(
        &normal,
        &[BlochMatrix::zeros(2, 2)],
        &[overlap],
        &[0.0],
        &[0.2],
        4,
        1e-12,
        &ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        },
    )
    .unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (actual.normal_density[0].re[(i, j)] - if i == j { 0.5 } else { 0.0 }).abs()
                    < 1e-12
            );
            assert!(
                (actual.normal_density[1].re[(i, j)] + normal[1].re[(i, j)] / 0.8).abs() < 1e-12
            );
            assert!(
                (actual.normal_density[3].re[(i, j)]
                    - 0.05 * normal[1].re[(i, j)] / (48.0 * 0.2_f64.powi(3)))
                .abs()
                    < 1e-12
            );
        }
    }
    assert!(actual
        .anomalous_density
        .iter()
        .flat_map(|a| a.re.as_slice())
        .all(|v| v.abs() < 1e-12));
    assert!(actual.particle_hole_defects.iter().all(|v| *v < 1e-12));
}
