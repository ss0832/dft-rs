// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent closed-block oracle for all coefficients of the decoupling.

#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/stencils.rs"]
pub mod stencils;

use dft_rs::dual::{Scalar, Taylor};
use dft_rs::periodic::BlochMatrix;
use dft_rs::properties::two_component::{
    dkh2_taylor, matrix_zora_taylor, x2c_taylor, RelativisticSpin, TwoComponentTaylor,
};
use stencils::{fit_components_staged, Stencil};

type Jet = Taylor<5>;
type Pair = (Jet, Jet);
type Square = [[Pair; 4]; 4];

fn pair(x: f64) -> Pair {
    (Jet::from_f64(x), Jet::from_f64(0.0))
}

fn add(a: Pair, b: Pair) -> Pair {
    (a.0 + b.0, a.1 + b.1)
}

fn mul(a: Pair, b: Pair) -> Pair {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn conjugate(a: Pair) -> Pair {
    (a.0, -a.1)
}

fn scale(a: Pair, x: Jet) -> Pair {
    (a.0 * x, a.1 * x)
}

fn adjoint<const R: usize, const C: usize>(a: &[[Pair; C]; R]) -> [[Pair; R]; C] {
    std::array::from_fn(|j| std::array::from_fn(|i| conjugate(a[i][j])))
}

fn product<const R: usize, const N: usize, const C: usize>(
    a: &[[Pair; N]; R],
    b: &[[Pair; C]; N],
) -> [[Pair; C]; R] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..N).fold(pair(0.0), |v, k| add(v, mul(a[i][k], b[k][j]))))
    })
}

fn rotation(angle: f64, phase: f64) -> [[Pair; 2]; 2] {
    let (s, c) = angle.sin_cos();
    let p = (
        Jet::from_f64(s * phase.cos()),
        Jet::from_f64(s * phase.sin()),
    );
    [
        [pair(c), scale(conjugate(p), Jet::from_f64(-1.0))],
        [p, pair(c)],
    ]
}

fn coefficients<const R: usize, const C: usize>(
    matrix: &[[Pair; C]; R],
    order: usize,
) -> Vec<BlochMatrix> {
    (0..=order)
        .map(|k| {
            let mut result = BlochMatrix::zeros(R, C);
            for (i, row) in matrix.iter().enumerate() {
                for (j, (re, im)) in row.iter().enumerate() {
                    result.re[(i, j)] = re.c[k];
                    result.im[(i, j)] = im.c[k];
                }
            }
            result
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Model {
    X2c,
    Dkh2,
    Zora,
}

impl Model {
    fn name(self) -> &'static str {
        match self {
            Self::X2c => "X2C",
            Self::Dkh2 => "DKH2",
            Self::Zora => "matrix ZORA",
        }
    }
}

struct Blocks {
    h: Square,
    s: Square,
    v: Square,
    property: Square,
    b: [[Pair; 2]; 4],
    core: [[Pair; 2]; 2],
    transformed: [[Pair; 2]; 2],
}

fn fixture(t: Jet, model: Model) -> Blocks {
    let mut h = [[pair(0.0); 4]; 4];
    let mut s = h;
    let mut v = h;
    let mut property = h;
    let mut b = [[pair(0.0); 2]; 4];
    let mut core = [[pair(0.0); 2]; 2];
    let constant = Jet::from_f64;
    // Each block has a closed form in its independently scaled diagonal
    // metric. The complex phases, changing metric, and a fully populated
    // observable exercise both the real and imaginary transformation.
    for i in 0..2 {
        let j = i + 2;
        let x = i as f64;
        let ml = (constant(0.31 + 0.17 * x) + t.scale(0.19 + 0.04 * x)).exp();
        let ms = (constant(-0.43 + 0.10 * x) + t.scale(-0.12 + 0.03 * x)).exp();
        let a = constant(0.37 + 0.14 * x) + t.scale(0.11);
        let d = constant(-2.85 + 0.17 * x) + t.scale(-0.23);
        let u = constant(-0.41 - 0.04 * x) + t.scale(0.27);
        let w = constant(-0.23 + 0.09 * x) + t.scale(-0.31);
        let coupling = (
            constant(0.47 + 0.15 * x) + t.scale(0.13),
            constant(-0.28 + 0.11 * x) + t.scale(0.21),
        );
        let magnitude = (coupling.0 * coupling.0 + coupling.1 * coupling.1).sqrt();
        let phase = scale(conjugate(coupling), magnitude.recip());
        s[i][i].0 = ml;
        s[j][j].0 = ms;
        v[i][i].0 = u * ml;
        v[j][j].0 = w * ms;
        h[i][i].0 = (a + u) * ml;
        h[j][j].0 = (d + w) * ms;
        h[i][j] = scale(coupling, (ml * ms).sqrt());
        h[j][i] = conjugate(h[i][j]);
        let gap = |a: Jet, d: Jet| ((a - d) * (a - d) + (magnitude * magnitude).scale(4.0)).sqrt();
        let (upper, lower, energy) = match model {
            Model::X2c => {
                let delta = gap(a + u, d + w);
                let energy = (a + u + d + w + delta).scale(0.5);
                let upper = (constant(0.5) + (a + u - d - w) / delta.scale(2.0)).sqrt();
                (upper, magnitude / (delta * upper), energy)
            }
            Model::Dkh2 => {
                let delta = gap(a, d);
                let upper = (constant(0.5) + (a - d) / delta.scale(2.0)).sqrt();
                let lower = magnitude / (delta * upper);
                let plus = u * upper * upper + w * lower * lower;
                let minus = u * lower * lower + w * upper * upper;
                let odd = upper * lower * (w - u);
                let first = odd / delta;
                let chart = first + (minus - plus) * first / delta;
                let normalization = (constant(1.0) + chart * chart).sqrt();
                (
                    (upper - lower * chart) / normalization,
                    (lower + upper * chart) / normalization,
                    (a + d + delta).scale(0.5) + plus + first * odd,
                )
            }
            Model::Zora => {
                let chart = -magnitude / (d + w);
                let normalization = (constant(1.0) + chart * chart).sqrt();
                (
                    normalization.recip(),
                    chart / normalization,
                    a + u + magnitude * chart,
                )
            }
        };
        b[i][i].0 = upper / ml.sqrt();
        b[j][i] = scale(phase, lower / ms.sqrt());
        core[i][i].0 = energy;
    }
    for i in 0..4 {
        for j in 0..=i {
            property[i][j] = (
                (constant(0.28 + 0.13 * (i + j) as f64) + t.scale(0.31 + 0.025 * i as f64)).exp(),
                if i == j {
                    constant(0.0)
                } else {
                    (constant(-0.38 + 0.11 * (i + 2 * j) as f64)
                        + t.scale(-0.27 - 0.017 * j as f64))
                    .exp()
                },
            );
            property[j][i] = conjugate(property[i][j]);
        }
    }
    let transformed = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut value = pair(0.0);
            for a in 0..4 {
                for q in 0..4 {
                    value = add(value, mul(conjugate(b[a][i]), mul(property[a][q], b[q][j])));
                }
            }
            value
        })
    });
    // Independent unitary changes of the large and small coordinate frames
    // populate all complex matrix channels while preserving the closed oracle.
    let large = rotation(0.43, 0.47);
    let small = rotation(0.26, 0.19);
    let q: Square = std::array::from_fn(|i| {
        std::array::from_fn(|j| match (i < 2, j < 2) {
            (true, true) => large[i][j],
            (false, false) => small[i - 2][j - 2],
            _ => pair(0.0),
        })
    });
    let transformed_full = |a: &Square| product(&q, &product(a, &adjoint(&q)));
    let transformed_large = |a: &[[Pair; 2]; 2]| product(&large, &product(a, &adjoint(&large)));
    Blocks {
        h: transformed_full(&h),
        s: transformed_full(&s),
        v: transformed_full(&v),
        property: transformed_full(&property),
        b: product(&q, &product(&b, &adjoint(&large))),
        core: transformed_large(&core),
        transformed: transformed_large(&transformed),
    }
}

fn run(f: &Blocks, model: Model, order: usize) -> (TwoComponentTaylor, Vec<BlochMatrix>) {
    let h = coefficients(&f.h, order);
    let s = coefficients(&f.s, order);
    let v = coefficients(&f.v, order);
    let spin = RelativisticSpin::SpinOrbit;
    let result = match model {
        Model::X2c => x2c_taylor(&h, &s, order, -1.0, 1e-12, spin),
        Model::Dkh2 => dkh2_taylor(&h, &s, &v, order, -1.0, 1e-12, spin),
        Model::Zora => matrix_zora_taylor(&h, &s, order, 1e-12, spin),
    }
    .unwrap();
    let property = result.transform(&coefficients(&f.property, order)).unwrap();
    (result, property)
}

fn factorial(order: usize) -> f64 {
    (1..=order).product::<usize>() as f64
}

fn components(
    embedding: &BlochMatrix,
    core: &BlochMatrix,
    property: &BlochMatrix,
) -> Vec<(String, f64)> {
    let mut components = Vec::new();
    for (name, a) in [("B", embedding), ("H", core), ("O", property)] {
        for i in 0..2 {
            components.push((format!("{name}[{i},{i}].re"), a.re[(i, i)]));
        }
        components.push((format!("{name}[1,0].re"), a.re[(1, 0)]));
        components.push((format!("{name}[1,0].im"), a.im[(1, 0)]));
    }
    for i in 2..4 {
        for j in 0..2 {
            components.push((format!("B[{i},{j}].re"), embedding.re[(i, j)]));
            components.push((format!("B[{i},{j}].im"), embedding.im[(i, j)]));
        }
    }
    components
}

#[test]
fn closed_block_decoupling_and_all_independent_coefficient_fits() {
    let mut failures = Vec::new();
    for model in [Model::X2c, Model::Dkh2, Model::Zora] {
        let f = fixture(Jet::from_coefficients(&[0.0, 1.0]), model);
        let (value, property) = run(&f, model, 4);
        for (actual, expected) in [
            (value.embedding(), coefficients(&f.b, 4)),
            (&value.hamiltonian, coefficients(&f.core, 4)),
            (&property, coefficients(&f.transformed, 4)),
        ] {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(actual.re.max_abs_diff(&expected.re) < 3e-12);
                assert!(actual.im.max_abs_diff(&expected.im) < 3e-12);
            }
        }
        assert!(value.metric_defects.iter().all(|r| *r < 3e-12));
        println!(
            "{} metric defects {:?} decoupling residuals {:?}",
            model.name(),
            value.metric_defects,
            value.decoupling_defects
        );
        for order in 1..=4 {
            let analytic = components(
                &value.embedding()[order],
                &value.hamiltonian[order],
                &property[order],
            );
            let names = analytic
                .iter()
                .map(|(name, _)| format!("{} order {order} {name}", model.name()))
                .collect::<Vec<_>>();
            fit_components_staged(
                &names,
                &[
                    ([0.08, 0.06, 0.04, 0.03], Stencil::Second),
                    ([0.04, 0.03, 0.02, 0.015], Stencil::Second),
                    ([0.12, 0.09, 0.07, 0.05], Stencil::Fourth),
                ],
                |steps, stencil| {
                    let mut errors = vec![[0.0; 4]; names.len()];
                    for (j, &h) in steps.iter().enumerate() {
                        // Differentiate the independent closed block values at
                        // each displaced point. Every lower analytic derivative
                        // is checked against the same production recurrence above.
                        let sample = |step| {
                            let point = fixture(Jet::from_coefficients(&[step, 1.0]), model);
                            components(
                                &coefficients(&point.b, order - 1)[order - 1],
                                &coefficients(&point.core, order - 1)[order - 1],
                                &coefficients(&point.transformed, order - 1)[order - 1],
                            )
                            .into_iter()
                            .map(|(_, x)| x * factorial(order - 1))
                            .collect::<Vec<_>>()
                        };
                        let plus = sample(h);
                        let minus = sample(-h);
                        let plus2 = sample(2.0 * h);
                        let minus2 = sample(-2.0 * h);
                        for i in 0..names.len() {
                            let estimate = stencil.first(plus[i], minus[i], plus2[i], minus2[i], h);
                            errors[i][j] = (estimate - analytic[i].1 * factorial(order)).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("two-component Taylor transformations", &failures);
}
