// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving-centre finite-field jets against independent complex Gaussian
//! moments and the entire Boys series. Differences are validation only.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::integrals::{
    analytic::{GaussianKernel, Primitive},
    london::{primitive_field_pair, primitive_field_repulsion, ScalarFieldPath},
};
use std::f64::consts::PI;
type J = Taylor<5>;
type C = (J, J);
fn mul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn scale(a: C, s: f64) -> C {
    (a.0.scale(s), a.1.scale(s))
}
fn add(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}
fn exp(a: C) -> C {
    let re = a.0.c[0];
    let im = a.1.c[0];
    let delta = (a.0 - J::from_f64(re), a.1 - J::from_f64(im));
    let mut power = (J::from_f64(1.0), J::from_f64(0.0));
    let mut value = power;
    for n in 1..5 {
        power = scale(mul(power, delta), 1.0 / n as f64);
        value = add(value, power);
    }
    mul(
        (
            J::from_f64(re.exp() * im.cos()),
            J::from_f64(re.exp() * im.sin()),
        ),
        value,
    )
}
fn primitives(t: J, zero_phase: bool) -> [Primitive<J>; 4] {
    let f = J::from_f64;
    let a = if zero_phase { 0.7 } else { -0.2 };
    let centers = [
        [f(a) + t.scale(0.17) + (t * t).scale(0.1), f(0.2), f(0.0)],
        [f(0.7) - t.scale(0.11) + (t * t).scale(0.04), f(0.2), f(0.0)],
        [f(1.4) + t.scale(0.13) - (t * t).scale(0.08), f(0.0), f(0.0)],
        [
            f(-0.5) + t.scale(0.05) + (t * t).scale(0.03),
            f(0.0),
            f(0.0),
        ],
    ];
    std::array::from_fn(|i| Primitive {
        angular: [0; 3],
        exponent: f([0.8, 1.1, 0.7, 1.3][i]),
        center: centers[i],
    })
}
fn field() -> ScalarFieldPath<J> {
    ScalarFieldPath {
        electric: [0.013, -0.007, 0.01].map(J::from_f64),
        magnetic: [0.0, 0.0, 0.04].map(J::from_f64),
        origin: [J::from_f64(0.0); 3],
        london: true,
        phase_order: 12,
        derivative_order: 4,
    }
}
fn zero(x: J) -> bool {
    x.c.iter().all(|v| *v == 0.0)
}

/// Complete the Gaussian square in complex coordinates. No integral engine,
/// phase polynomial or centre-differentiation recurrence enters this oracle.
fn pair_oracle(a: &Primitive<J>, b: &Primitive<J>) -> (C, [C; 3], J) {
    let f = J::from_f64;
    let p = a.exponent + b.exponent;
    let mean: [J; 3] =
        std::array::from_fn(|i| (a.exponent * a.center[i] + b.exponent * b.center[i]) / p);
    let gap: [J; 3] = std::array::from_fn(|i| a.center[i] - b.center[i]);
    let k = [-gap[1].scale(0.02), gap[0].scale(0.02), f(0.0)];
    let r2 = gap.iter().fold(f(0.0), |s, x| s + *x * *x);
    let k2 = k.iter().fold(f(0.0), |s, x| s + *x * *x);
    let phase = (0..3).fold(f(0.0), |s, i| s + k[i] * mean[i]);
    let overlap = scale(
        exp((-a.exponent * b.exponent * r2 / p - k2 / p.scale(4.0), phase)),
        PI.powf(1.5),
    );
    let norm = p.powf(-1.5);
    let overlap = (overlap.0 * norm, overlap.1 * norm);
    (
        overlap,
        std::array::from_fn(|i| (mean[i], k[i] / p.scale(2.0))),
        p,
    )
}
fn core_oracle(a: &Primitive<J>, b: &Primitive<J>) -> [J; 4] {
    let f = J::from_f64;
    let z = f(0.0);
    let (s, mean, p) = pair_oracle(a, b);
    let u: [C; 3] = std::array::from_fn(|i| (mean[i].0 - b.center[i], mean[i].1));
    let u2 = add(add(mul(u[0], u[0]), mul(u[1], u[1])), mul(u[2], u[2]));
    let variance = f(1.5) / p;
    let kinetic = add(
        (b.exponent.scale(3.0), z),
        scale(
            (
                b.exponent * b.exponent * (u2.0 + variance),
                b.exponent * b.exponent * u2.1,
            ),
            -2.0,
        ),
    );
    let magnetic = scale(
        add(add(mul(u[0], u[0]), mul(u[1], u[1])), (f(1.0) / p, z)),
        0.04 * 0.04 / 8.0,
    );
    let electric = (0..3).fold((z, z), |value, i| {
        add(value, scale(mean[i], [0.013, -0.007, 0.01][i]))
    });
    let h = mul(s, add(add(kinetic, magnetic), electric));
    [s.0, s.1, h.0, h.1]
}
fn boys_oracle(t: C) -> C {
    let f = J::from_f64;
    let mut term = (f(1.0), f(0.0));
    let mut boys = term;
    for n in 1..80 {
        term = scale(
            mul(term, t),
            -((2 * n - 1) as f64) / (n * (2 * n + 1)) as f64,
        );
        boys = add(boys, term);
    }
    boys
}
fn repulsion_oracle(p: &[Primitive<J>; 4]) -> [J; 2] {
    let f = J::from_f64;
    let z = f(0.0);
    let (s1, c1, a) = pair_oracle(&p[0], &p[1]);
    let (s2, c2, b) = pair_oracle(&p[2], &p[3]);
    let rho = a * b / (a + b);
    let t = (0..3).fold((z, z), |s, i| {
        let d = (c1[i].0 - c2[i].0, c1[i].1 - c2[i].1);
        add(s, mul(d, d))
    });
    let t = (t.0 * rho, t.1 * rho);
    let boys = boys_oracle(t);
    let value = mul(mul(s1, s2), boys);
    let factor = rho.sqrt().scale(2.0 / PI.sqrt());
    [value.0 * factor, value.1 * factor]
}
fn fits<const N: usize>(label: &str, actual: [J; N], oracle: impl Fn(J) -> [J; N]) {
    let exact = oracle(J::from_coefficients(&[0.0, 1.0]));
    let steps = [0.08, 0.04, 0.02, 0.01];
    for component in 0..N {
        for n in 0..5 {
            assert!(
                (actual[component].c[n] - exact[component].c[n]).abs() < 1e-11,
                "{label} component={component} n={n}"
            );
        }
        for order in 1..=4 {
            let target = order as f64 * actual[component].c[order];
            let estimates: Vec<_> = steps
                .iter()
                .map(|h| {
                    let plus = oracle(J::from_coefficients(&[*h, 1.0]));
                    let minus = oracle(J::from_coefficients(&[-*h, 1.0]));
                    (plus[component].c[order - 1] - minus[component].c[order - 1]) / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
            println!("London nuclear {label} component={component} order={order} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 1e-7);
        }
    }
}

#[test]
fn finite_magnetic_and_electric_pair_jets_have_closed_moment_four_order_fits() {
    for zero_phase in [false, true] {
        let p = primitives(J::from_coefficients(&[0.0, 1.0]), zero_phase);
        let actual = primitive_field_pair(&p[0], &p[1], &[], &field(), zero).unwrap();
        fits(
            if zero_phase {
                "zero-phase"
            } else {
                "finite-phase"
            },
            [actual[0].0, actual[0].1, actual[1].0, actual[1].1],
            |t| {
                let p = primitives(t, zero_phase);
                core_oracle(&p[0], &p[1])
            },
        );
    }
}
#[test]
fn london_quartet_jets_have_independent_entire_boys_four_order_fits() {
    let p = primitives(J::from_coefficients(&[0.0, 1.0]), false);
    let actual = primitive_field_repulsion(
        [&p[0], &p[1], &p[2], &p[3]],
        GaussianKernel::Coulomb,
        &field(),
        zero,
    )
    .unwrap();
    fits("quartet", [actual.0, actual.1], |t| {
        repulsion_oracle(&primitives(t, false))
    });
}

fn nucleus(t: J) -> [J; 3] {
    [
        J::from_f64(0.3) + t.scale(0.12),
        J::from_f64(0.4) - t.scale(0.06),
        J::from_f64(-0.15) + (t * t).scale(0.1),
    ]
}

#[test]
fn moving_nucleus_in_finite_fields_has_independent_complex_boys_fits() {
    let t = J::from_coefficients(&[0.0, 1.0]);
    let p = primitives(t, false);
    let plain = primitive_field_pair(&p[0], &p[1], &[], &field(), zero).unwrap();
    let charged = primitive_field_pair(
        &p[0],
        &p[1],
        &[(nucleus(t), J::from_f64(2.0))],
        &field(),
        zero,
    )
    .unwrap();
    fits(
        "nuclear-attraction",
        [charged[1].0 - plain[1].0, charged[1].1 - plain[1].1],
        |t| {
            let p = primitives(t, false);
            let (s, mean, exponent) = pair_oracle(&p[0], &p[1]);
            let center = nucleus(t);
            let z = J::from_f64(0.0);
            let r2 = (0..3).fold((z, z), |sum, i| {
                let d = (mean[i].0 - center[i], mean[i].1);
                add(sum, mul(d, d))
            });
            let boys = boys_oracle((r2.0 * exponent, r2.1 * exponent));
            let v = mul(s, boys);
            let factor = exponent.sqrt().scale(-4.0 / PI.sqrt());
            [v.0 * factor, v.1 * factor]
        },
    );
}
