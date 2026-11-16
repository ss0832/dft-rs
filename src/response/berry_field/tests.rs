// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Closed two-dimensional determinant and adjugate oracles, independent of
//! the production matrix inverse/log-determinant response recurrences.
use super::*;
use crate::response::complex::Complex as Z;
mod convergence {
    include!("../../../tests/support/convergence.rs");
}

fn pair(v: [Z; 4]) -> ComplexMatrix {
    let mut m = (Matrix::zeros(2, 2), Matrix::zeros(2, 2));
    for i in 0..4 {
        m.0.as_mut_slice()[i] = v[i].re;
        m.1.as_mut_slice()[i] = v[i].im;
    }
    m
}
fn flat(v: &ComplexMatrix) -> [Z; 4] {
    std::array::from_fn(|i| Z::new(v.0.as_slice()[i], v.1.as_slice()[i]))
}
fn multiply(a: [Z; 4], b: [Z; 4]) -> [Z; 4] {
    std::array::from_fn(|i| (0..2).fold(Z::ZERO, |v, k| v + a[2 * (i / 2) + k] * b[2 * k + i % 2]))
}
fn fixture(gauge: bool) -> ([Z; 4], [Z; 4], [Z; 4], [Z; 4]) {
    let transform = |m: [Z; 4]| {
        if !gauge {
            return m;
        }
        let u = [
            Z::new(0.3_f64.cos(), 0.3_f64.sin()),
            Z::new((-0.6_f64).cos(), (-0.6_f64).sin()),
        ];
        std::array::from_fn(|i| u[i / 2].conj() * m[i] * u[i % 2])
    };
    (
        transform([
            Z::new(0.47, 0.0),
            Z::new(0.06, 0.09),
            Z::new(0.06, -0.09),
            Z::new(0.34, 0.0),
        ]),
        transform([
            Z::new(0.31, 0.0),
            Z::new(0.12, -0.14),
            Z::new(0.12, 0.14),
            Z::new(-0.21, 0.0),
        ]),
        transform([
            Z::new(-0.2, 0.7),
            Z::new(0.14, 0.22),
            Z::new(-0.11, 0.27),
            Z::new(-0.1, 0.5),
        ]),
        transform([
            Z::new(0.08, 0.19),
            Z::new(-0.12, 0.06),
            Z::new(0.07, -0.12),
            Z::new(-0.04, -0.13),
        ]),
    )
}
fn at(gauge: bool, t: f64) -> (f64, f64) {
    let (p, d, b, db) = fixture(gauge);
    let p = std::array::from_fn(|i| p[i] + d[i] * t);
    let b = std::array::from_fn(|i| b[i] + db[i] * t);
    let mut m = multiply(p, b);
    m[0] += Z::from(1.0);
    m[3] += Z::from(1.0);
    let left = multiply(d, b);
    let right = multiply(p, db);
    let dm: [Z; 4] = std::array::from_fn(|i| left[i] + right[i]);
    let det = m[0] * m[3] - m[1] * m[2];
    let derivative = dm[0] * m[3] + m[0] * dm[3] - dm[1] * m[2] - m[1] * dm[2];
    let phase = det.im.atan2(det.re);
    let r = 0.7 + 0.13 * t;
    let ion = 0.2 + 0.1 * t;
    (
        r * (phase - ion),
        0.13 * (phase - ion) + r * ((derivative / det).im - 0.1),
    )
}
fn closed_coefficients(gauge: bool) -> Vec<f64> {
    let (p, d, b, db) = fixture(gauge);
    let mut m = [multiply(p, b), multiply(d, b), multiply(d, db)];
    let extra = multiply(p, db);
    m[0][0] += Z::from(1.0);
    m[0][3] += Z::from(1.0);
    for i in 0..4 {
        m[1][i] += extra[i];
    }
    let mut determinant = [Z::ZERO; 5];
    for j in 0..3 {
        for k in 0..3 {
            determinant[j + k] += m[j][0] * m[k][3] - m[j][1] * m[k][2];
        }
    }
    let mut inverse = [Z::ZERO; 5];
    inverse[0] = Z::from(1.0) / determinant[0];
    let mut phase = vec![determinant[0].im.atan2(determinant[0].re)];
    for j in 1..=4 {
        inverse[j] =
            -(1..=j).fold(Z::ZERO, |v, k| v + determinant[k] * inverse[j - k]) / determinant[0];
        phase.push(
            (1..=j)
                .fold(Z::ZERO, |v, k| {
                    v + determinant[k] * inverse[j - k] * (k as f64)
                })
                .im
                / j as f64,
        );
    }
    (0..5)
        .map(|j| {
            0.7 * phase[j] + if j > 0 { 0.13 * phase[j - 1] } else { 0.0 }
                - match j {
                    0 => 0.14,
                    1 => 0.096,
                    2 => 0.013,
                    _ => 0.0,
                }
        })
        .collect()
}
fn provider(gauge: bool, order: usize) -> (BerryField, Vec<Vec<ComplexMatrix>>) {
    let (p, d, b, db) = fixture(gauge);
    let zero = || (Matrix::zeros(2, 2), Matrix::zeros(2, 2));
    let mut field = BerryField {
        phase_reference: None,
        polar: std::array::from_fn(|_| vec![zero(); order + 1]),
        ionic: std::array::from_fn(|_| vec![0.0; order + 1]),
        reduced: std::array::from_fn(|_| vec![0.0; order + 1]),
    };
    field.polar[0][0] = pair(b);
    field.ionic[0][0] = 0.2;
    field.reduced[0][0] = 0.7;
    let mut path = vec![zero(); order + 1];
    path[0] = pair(p);
    if order > 0 {
        field.polar[0][1] = pair(db);
        field.ionic[0][1] = 0.1;
        field.reduced[0][1] = 0.13;
        path[1] = pair(d);
    }
    (field, vec![path])
}

fn energy_fits(gauge: bool, branch: f64) {
    let steps = [0.08, 0.04, 0.02, 0.01];
    let (mut field, p) = provider(gauge, 4);
    if branch != 0.0 {
        field.phase_reference = Some([
            at(gauge, 0.0).0 / 0.7 + 0.2 + 2.0 * std::f64::consts::PI * branch,
            0.0,
            0.0,
        ]);
    }
    let point = |t: f64| {
        let (e, g) = at(gauge, t);
        let turn = 2.0 * std::f64::consts::PI * branch;
        (e + turn * (0.7 + 0.13 * t), g + turn * 0.13)
    };
    let energy = field.evaluate(&p, 4).unwrap().0;
    let mut closed = closed_coefficients(gauge);
    closed[0] += 2.0 * std::f64::consts::PI * branch * 0.7;
    closed[1] += 2.0 * std::f64::consts::PI * branch * 0.13;
    for j in 0..=4 {
        assert!((energy[j] - closed[j]).abs() < 1e-13);
    }
    let zero = point(0.0);
    let estimates: Vec<_> = steps
        .iter()
        .map(|h| {
            let (a, b, c, d) = (point(*h), point(-h), point(2.0 * h), point(-2.0 * h));
            [
                (a.0 - b.0) / (2.0 * h),
                (a.1 - b.1) / (2.0 * h),
                (a.1 - 2.0 * zero.1 + b.1) / (h * h),
                (c.1 - 2.0 * a.1 + 2.0 * b.1 - d.1) / (2.0 * h * h * h),
            ]
        })
        .collect();
    for order in 1..=4 {
        let target = energy[order] * (1..=order).product::<usize>() as f64;
        let errors: Vec<_> = estimates
            .iter()
            .map(|v| (v[order - 1] - target).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3][order - 1] - estimates[2][order - 1]) / 3.0 - target).abs();
        println!("complex Berry energy gauge={gauge} branch={branch} order={order} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.16e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 2e-6);
    }
}
#[test]
fn complex_berry_energy_orders_one_through_four_have_closed_error_fits() {
    for gauge in [false, true] {
        energy_fits(gauge, 0.0);
    }
}
#[test]
fn continued_berry_branches_have_independent_energy_and_field_derivative_fits() {
    for gauge in [false, true] {
        for branch in [-1.0, 1.0] {
            energy_fits(gauge, branch);
        }
    }
}

fn closed_fock(p: [Z; 4], b: [Z; 4]) -> ComplexMatrix {
    let mut m = multiply(p, b);
    m[0] += Z::from(1.0);
    m[3] += Z::from(1.0);
    let determinant = m[0] * m[3] - m[1] * m[2];
    let inv = [
        m[3] / determinant,
        -m[1] / determinant,
        -m[2] / determinant,
        m[0] / determinant,
    ];
    let k = multiply(b, inv);
    pair(std::array::from_fn(|i| {
        (k[i] - k[2 * (i % 2) + i / 2].conj()) * Z::new(0.0, -0.35)
    }))
}
#[test]
fn complex_berry_fock_and_density_kernel_match_closed_adjugate_and_error_fits() {
    let steps = [0.08, 0.04, 0.02, 0.01];
    for gauge in [false, true] {
        let (field, path) = provider(gauge, 0);
        let (p, d, b, _) = fixture(gauge);
        let fock = field.evaluate(&path, 0).unwrap().1;
        let oracle = closed_fock(p, b);
        assert!(
            fock[0][0].0.max_abs_diff(&oracle.0) < 1e-13
                && fock[0][0].1.max_abs_diff(&oracle.1) < 1e-13
        );
        let kernel = field.linear(&path, &[vec![pair(d)]], 0).unwrap();
        let mut errors = Vec::new();
        for h in steps {
            let plus = flat(&closed_fock(std::array::from_fn(|i| p[i] + d[i] * h), b));
            let minus = flat(&closed_fock(std::array::from_fn(|i| p[i] - d[i] * h), b));
            let exact = flat(&kernel[0][0]);
            let error = (0..4)
                .map(|i| ((plus[i] - minus[i]) / (2.0 * h) - exact[i]).abs().powi(2))
                .sum::<f64>()
                .sqrt();
            errors.push(error);
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        println!("complex Berry density kernel gauge={gauge} steps={steps:?} errors={errors:?} p={:.9} R2={:.12}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999);
    }
}
