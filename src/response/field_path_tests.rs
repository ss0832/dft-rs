// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Closed complex, moving-metric mean-field response with coupled spin and
//! real/imaginary orbital rotations. Differences are validation only.
use super::*;
use crate::dual::{Scalar, Taylor};
mod convergence {
    include!("../../tests/support/convergence.rs");
}
type J = Taylor<5>;
type Z = (J, J);
type M = [[Z; 2]; 2];
fn z(x: f64) -> Z {
    (J::from_f64(x), J::from_f64(0.0))
}
fn zmul(a: Z, b: Z) -> Z {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn zadd(a: Z, b: Z) -> Z {
    (a.0 + b.0, a.1 + b.1)
}
fn jproduct(a: M, b: M) -> M {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..2).fold(z(0.0), |s, k| zadd(s, zmul(a[i][k], b[k][j]))))
    })
}
fn jadjoint(a: M) -> M {
    std::array::from_fn(|i| std::array::from_fn(|j| (a[j][i].0, -a[j][i].1)))
}
fn coefficients(a: M) -> Vec<Cm> {
    (0..5)
        .map(|n| {
            let mut out = Cm::zeros(2, 2);
            for i in 0..2 {
                for j in 0..2 {
                    out.re[(i, j)] = a[i][j].0.c[n];
                    out.im[(i, j)] = a[i][j].1.c[n];
                }
            }
            out
        })
        .collect()
}

struct Model {
    s: Vec<Cm>,
    a: Vec<Cm>,
    h: Vec<Cm>,
    g: f64,
}
impl Model {
    fn moments(&self, p: &[Vec<Cm>; 2], order: usize) -> Vec<f64> {
        (0..=order)
            .map(|n| {
                (0..2)
                    .flat_map(|s| (0..=n).map(move |k| trace(&p[s][k], &self.a[n - k])))
                    .sum()
            })
            .collect()
    }
}
impl ComplexPathBackend for Model {
    fn overlap(&self) -> &[Cm] {
        &self.s
    }
    fn fock_coefficient(&self, p: &[Vec<Cm>; 2], n: usize) -> Result<[Cm; 2]> {
        let m = self.moments(p, n);
        let mut f = self.h[n].clone();
        for k in 0..=n {
            cadd(&mut f, &self.a[n - k], self.g * m[k]);
        }
        Ok([f.clone(), f])
    }
    fn linear_kernel(&self, p: &[Cm; 2]) -> Result<[Cm; 2]> {
        let value = self.g * (trace(&p[0], &self.a[0]) + trace(&p[1], &self.a[0]));
        let mut f = Cm::zeros(2, 2);
        cadd(&mut f, &self.a[0], value);
        Ok([f.clone(), f])
    }
    fn energy(&self, p: &[Vec<Cm>; 2], n: usize) -> Result<Vec<f64>> {
        let m = self.moments(p, n);
        Ok((0..=n)
            .map(|i| {
                (0..2)
                    .flat_map(|s| (0..=i).map(move |k| trace(&p[s][k], &self.h[i - k])))
                    .sum::<f64>()
                    + self.g * 0.5 * (0..=i).map(|k| m[k] * m[i - k]).sum::<f64>()
            })
            .collect())
    }
    fn dipole(&self, _p: &[Vec<Cm>; 2], order: usize) -> Vec<[f64; 3]> {
        vec![[0.0; 3]; order + 1]
    }
}

impl RetainedComplexBackend for Model {
    fn retained_fock_coefficient(
        &self,
        high: &[Vec<Cm>; 2],
        low: &[Vec<Cm>; 2],
        order: usize,
    ) -> Result<([Cm; 2], [Cm; 2])> {
        let f = self.fock_coefficient(high, order)?;
        let moments = self.moments(low, order);
        let mut delta = Cm::zeros(2, 2);
        for k in 0..=order {
            cadd(&mut delta, &self.a[order - k], self.g * moments[k]);
        }
        Ok((f, [delta.clone(), delta]))
    }
}

fn data(t: J, g: f64) -> (Model, Vec<Cm>, Vec<Cm>, [J; 4]) {
    let f = J::from_f64;
    let one = f(1.0);
    let zero = f(0.0);
    let u = f(0.4) + t.scale(0.12) + (t * t).scale(0.06);
    let v = f(0.3) + t.scale(0.05) - (t * t).scale(0.02);
    let phase = ((one - v * v) / (one + v * v), v.scale(2.0) / (one + v * v));
    let norm = (one + u * u).sqrt();
    let unitary = [
        [
            (one / norm, zero),
            (-u * phase.0 / norm, u * phase.1 / norm),
        ],
        [(u * phase.0 / norm, u * phase.1 / norm), (one / norm, zero)],
    ];
    let m0 = one + t.scale(0.07);
    let m1 = one - t.scale(0.04);
    let shear = (t.scale(0.05), t.scale(-0.02));
    let moving = [[(m0, zero), shear], [z(0.0), (m1, zero)]];
    let inverse = [
        [
            (one / m0, zero),
            (-shear.0 / (m0 * m1), -shear.1 / (m0 * m1)),
        ],
        [z(0.0), (one / m1, zero)],
    ];
    let c = jproduct(inverse, unitary);
    let p: M =
        std::array::from_fn(|i| std::array::from_fn(|j| zmul(c[i][0], (c[j][0].0, -c[j][0].1))));
    let canonical_a = [
        [(f(1.0), zero), (f(0.2), f(-0.15))],
        [(f(0.2), f(0.15)), (f(-1.0), zero)],
    ];
    let a = jproduct(jproduct(jadjoint(moving), canonical_a), moving);
    let zvalue = (0..2)
        .flat_map(|i| (0..2).map(move |j| zmul(p[i][j], a[j][i]).0))
        .fold(zero, |s, x| s + x);
    let radius = (one + t.scale(0.1) + (t * t).scale(0.03)).sqrt();
    let diagonal = [[(-radius, zero), z(0.0)], [z(0.0), (radius, zero)]];
    let canonical_f = jproduct(jproduct(unitary, diagonal), jadjoint(unitary));
    let desired = jproduct(jproduct(jadjoint(moving), canonical_f), moving);
    let h: M = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (
                desired[i][j].0 - zvalue.scale(2.0 * g) * a[i][j].0,
                desired[i][j].1 - zvalue.scale(2.0 * g) * a[i][j].1,
            )
        })
    });
    let energy = -radius.scale(2.0) - (zvalue * zvalue).scale(2.0 * g);
    (
        Model {
            s: coefficients(jproduct(jadjoint(moving), moving)),
            a: coefficients(a),
            h: coefficients(h),
            g,
        },
        coefficients(c),
        coefficients(p),
        [p[0][0].0, p[0][1].0, p[0][1].1, energy],
    )
}

fn model_reference(g: f64, gauge_beta: bool) -> (Model, ComplexPathReference) {
    let t = J::from_coefficients(&[0.0, 1.0]);
    let (model, c, p, _) = data(t, g);
    let density = [p[0].clone(), p[0].clone()];
    let mut coefficients = [c[0].clone(), c[0].clone()];
    if gauge_beta {
        for i in 0..2 {
            for j in 0..2 {
                let phase = [0.3_f64, -0.7][j];
                let (s, c) = phase.sin_cos();
                let re = coefficients[1].re[(i, j)];
                let im = coefficients[1].im[(i, j)];
                coefficients[1].re[(i, j)] = re * c - im * s;
                coefficients[1].im[(i, j)] = re * s + im * c;
            }
        }
    }
    let fock = model
        .fock_coefficient(&[vec![density[0].clone()], vec![density[1].clone()]], 0)
        .unwrap();
    let reference = ComplexPathReference {
        coefficients,
        energies: [vec![-1.0, 1.0], vec![-1.0, 1.0]],
        occupations: [1, 1],
        density,
        fock,
        unrestricted: gauge_beta,
    };
    (model, reference)
}

fn run_model(g: f64, gauge_beta: bool) -> (FieldPathTaylor, ComplexPathState) {
    let (model, reference) = model_reference(g, gauge_beta);
    stationary_complex_path(
        &model,
        &reference,
        4,
        &ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn reduced_complex_energy_has_closed_four_step_fits_and_full_response_limits() {
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    for gauge_beta in [false, true] {
        let (model, reference) = model_reference(0.13, gauge_beta);
        let (full, _) = stationary_complex_path(&model, &reference, 4, &config).unwrap();
        let oracle = data(J::from_coefficients(&[0.0, 1.0]), 0.13).3[3];
        for order in 1..=4 {
            let reduced =
                stationary_retained_complex_energy_path(&model, &reference, order, &config)
                    .unwrap();
            assert_eq!(reduced.solved_order, order / 2);
            assert_eq!(reduced.residuals.len(), order / 2);
            assert!(
                reduced.reference_defect < 1e-12 && reduced.residuals.iter().all(|r| *r < 1e-12)
            );
            assert!(reduced.projector_defects.iter().all(|r| *r < 1e-11));
            for n in 0..=order {
                assert!((reduced.energy[n] - oracle.c[n]).abs() < 1e-11);
                assert!((reduced.energy[n] - full.energy[n]).abs() < 1e-11);
            }
            let target = order as f64 * reduced.energy[order];
            let estimates: Vec<_> = steps
                .iter()
                .map(|h| {
                    let plus = data(J::from_coefficients(&[*h, 1.0]), 0.13).3[3];
                    let minus = data(J::from_coefficients(&[-*h, 1.0]), 0.13).3[3];
                    (plus.c[order - 1] - minus.c[order - 1]) / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
            println!("complex reduced stationary gauge={gauge_beta} order={order} solved={} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.16e} residuals={:?}",reduced.solved_order,fit.exponent,fit.r_squared,reduced.residuals);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 1e-7);
        }
    }
}

#[test]
fn complex_reference_coupled_response_has_closed_four_order_fits_and_orbital_gauge_covariance() {
    let (report, state) = run_model(0.13, false);
    let (rotated, other) = run_model(0.13, true);
    assert!(report.reference_defect < 1e-12 && report.residuals.iter().all(|r| *r < 1e-12));
    for n in 0..5 {
        for s in 0..2 {
            assert!(state.p[s][n].re.max_abs_diff(&other.p[s][n].re) < 1e-11);
            assert!(state.p[s][n].im.max_abs_diff(&other.p[s][n].im) < 1e-11);
        }
        assert!((report.energy[n] - rotated.energy[n]).abs() < 1e-11);
    }
    let (_, _, _, oracle) = data(J::from_coefficients(&[0.0, 1.0]), 0.13);
    let actual: [Vec<f64>; 4] = [
        state.p[0].iter().map(|p| p.re[(0, 0)]).collect(),
        state.p[0].iter().map(|p| p.re[(0, 1)]).collect(),
        state.p[0].iter().map(|p| p.im[(0, 1)]).collect(),
        report.energy.clone(),
    ];
    let steps = [0.08, 0.04, 0.02, 0.01];
    for component in 0..4 {
        for n in 0..5 {
            assert!((actual[component][n] - oracle[component].c[n]).abs() < 1e-11);
        }
        for order in 1..=4 {
            let target = order as f64 * actual[component][order];
            let estimates: Vec<_> = steps
                .iter()
                .map(|h| {
                    let plus = data(J::from_coefficients(&[*h, 1.0]), 0.13).3[component];
                    let minus = data(J::from_coefficients(&[-*h, 1.0]), 0.13).3[component];
                    (plus.c[order - 1] - minus.c[order - 1]) / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
            println!("complex stationary component={component} order={order} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e} residuals={:?}",fit.exponent,fit.r_squared,report.residuals);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 1e-7);
        }
    }
    let (_, zero) = run_model(0.0, false);
    for n in 0..5 {
        assert!(state.p[0][n].re.max_abs_diff(&zero.p[0][n].re) < 1e-11);
    }
}
