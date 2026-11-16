// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Whole RKB-limit J/K matrices and independent finite-field Coulomb and
//! range-separated exchange energy differences on C1 Gaussian centres.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::grid::gauss_legendre;
use dft_rs::integrals::analytic::GaussianKernel;
use dft_rs::integrals::analytic_basis::AnalyticShell;
use dft_rs::integrals::dirac::{
    electronic_coulomb_exchange, electronic_current_exchange, DiracCurrentInteraction,
    LondonCoulombTaylor, LondonCurrentTaylor,
};
use dft_rs::math::Vec3;
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;
use std::f64::consts::PI;
use stencils::Stencil;

const A: [f64; 4] = [0.935, 0.682, 0.817, 0.751];
const CENTERS: [[f64; 3]; 4] = [
    [0.12, -0.21, 0.15],
    [1.74, 0.18, 0.78],
    [-0.76, 1.27, -0.57],
    [-0.61, -1.52, -0.73],
];
const C: f64 = 13.7;

fn shells() -> Vec<AnalyticShell<f64>> {
    CENTERS
        .iter()
        .zip(A)
        .map(|(&r, a)| AnalyticShell::normalized(0, r, vec![a], vec![1.0]).unwrap())
        .collect()
}

fn density(active: usize) -> BlochMatrix {
    let mut density = BlochMatrix::zeros(16, 16);
    for i in 0..active {
        for j in 0..=i {
            let re = (0.33 * (i + j + 1) as f64).cos() * 0.075;
            let im = if i == j {
                0.0
            } else {
                (0.47 * (i - j) as f64).sin() * 0.042
            };
            density.re[(i, j)] = re;
            density.re[(j, i)] = re;
            density.im[(i, j)] = im;
            density.im[(j, i)] = -im;
        }
    }
    density
}

#[test]
fn london_coulomb_at_zero_field_reproduces_the_full_rkb_fock() {
    let basis = shells();
    let density = density(16);
    let rkb = electronic_coulomb_exchange(&basis, &density, C).unwrap();
    let cache = LondonCoulombTaylor::new(
        &basis,
        C,
        GaussianKernel::Coulomb,
        [0.0; 3],
        [0.0; 3],
        0,
        0,
        5,
    )
    .unwrap();
    let london = cache.fields(0, &density).unwrap();
    let difference =
        |a: &BlochMatrix, b: &BlochMatrix| a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im));
    let error = [
        difference(&rkb.direct, &london.direct),
        difference(&rkb.exchange, &london.exchange),
    ];
    println!(
        "C1 full RKB-limit London J/K errors {error:?}; Hermiticity {:.6e}",
        london.hermiticity_defect
    );
    assert!(error.into_iter().all(|v| v < 2e-13));
    assert!(
        (london.direct_energy + london.exchange_energy - rkb.direct_energy - rkb.exchange_energy)
            .abs()
            < 2e-13
    );
}

#[test]
fn london_current_at_zero_field_reproduces_the_full_rkb_fock() {
    let basis = shells();
    let density = density(16);
    for interaction in [
        DiracCurrentInteraction::Gaunt,
        DiracCurrentInteraction::Breit,
    ] {
        let rkb = electronic_current_exchange(&basis, &density, C, interaction).unwrap();
        let cache =
            LondonCurrentTaylor::new(&basis, C, interaction, [0.0; 3], [0.0; 3], 0, 0, 5).unwrap();
        let london = cache.fields(0, &density).unwrap();
        let error = |a: &BlochMatrix, b: &BlochMatrix| {
            a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
        };
        let j = error(&rkb.direct, &london.direct);
        let k = error(&rkb.exchange, &london.exchange);
        println!("{interaction:?} London vs RKB whole J/K defects {j:.6e} {k:.6e}");
        assert!(j < 2e-13 && k < 2e-13);
        assert!(london.hermiticity_defect < 2e-13);
    }
}

struct GaussianPair {
    exponent: f64,
    center: Vec3,
    wave: Vec3,
    weight: f64,
}

fn gaussian_pair(a: usize, b: usize, magnetic: Vec3) -> GaussianPair {
    let vector = |[x, y, z]: [f64; 3]| Vec3::new(x, y, z);
    let ra = vector(CENTERS[a]);
    let rb = vector(CENTERS[b]);
    let exponent = A[a] + A[b];
    let wave = magnetic.cross(ra - rb) * 0.5;
    let center = (ra * A[a] + rb * A[b]) / exponent;
    let weight = (4.0 * A[a] * A[b] / exponent.powi(2)).powf(0.75)
        * (-A[a] * A[b] / exponent * (ra - rb).norm2()).exp();
    GaussianPair {
        exponent,
        center,
        wave,
        weight,
    }
}

/// Integral of the two unbalanced large components. Completing the
/// Gaussian squares gives a complex Fourier Gaussian. A one-dimensional
/// compactified radial integral supplies the Coulomb or screened kernel.
fn oracle(field: Vec3, omega: Option<f64>, density: &BlochMatrix) -> [f64; 2] {
    let (nodes, weights) = gauss_legendre(24);
    let pairs: [GaussianPair; 16] = std::array::from_fn(|i| gaussian_pair(i / 4, i % 4, field));
    let mut energy = [Complex::ZERO; 2];
    let p = |i, j| Complex::new(density.re[(i, j)], density.im[(i, j)]);
    for (a, left) in pairs.iter().enumerate() {
        for (b, right) in pairs.iter().enumerate() {
            let rho = left.exponent * right.exponent / (left.exponent + right.exponent);
            let limit = omega.map_or(1.0, |w| w / (w * w + rho).sqrt());
            let scaled = left.wave / left.exponent - right.wave / right.exponent;
            let separation = left.center - right.center;
            let real =
                -0.25 * (left.wave.norm2() / left.exponent + right.wave.norm2() / right.exponent);
            let angle = left.wave.dot(left.center) + right.wave.dot(right.center);
            let mut value = Complex::ZERO;
            for (&node, &weight) in nodes.iter().zip(&weights) {
                let u = rho * (0.5 * limit * (node + 1.0)).powi(2);
                let attenuation = (real + u * (0.25 * scaled.norm2() - separation.norm2())).exp();
                let argument = angle - u * separation.dot(scaled);
                value += Complex::new(argument.cos(), argument.sin()) * (weight * attenuation);
            }
            value = value * (left.weight * right.weight * (rho / PI).sqrt() * limit);
            let [a, b, c, d] = [a / 4, a % 4, b / 4, b % 4];
            for s in [0, 4] {
                for t in [0, 4] {
                    energy[0] += p(b + s, a + s) * p(d + t, c + t) * value * 0.5;
                    energy[1] += p(d + t, a + s) * p(b + s, c + t) * value * (-0.5);
                }
            }
        }
    }
    assert!(energy.iter().all(|e| e.im.abs() < 2e-13));
    energy.map(|e| e.re)
}

#[test]
fn all_large_component_coulomb_and_screened_exchange_energy_field_derivatives() {
    c1_geometry::assert_c1(&[7, 1, 1, 1], &CENTERS);
    let basis = shells();
    let density = density(8);
    let field = Vec3::new(0.013, -0.019, 0.011);
    let axes = [
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 1., 0.),
        Vec3::new(0., 0., 1.),
    ];
    let independent = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    for omega in [None, Some(0.32)] {
        let kernel = omega.map_or(GaussianKernel::Coulomb, |omega| GaussianKernel::LongRange {
            omega,
        });
        let coefficient = |dir: Vec3| {
            let cache = LondonCoulombTaylor::new(
                &basis,
                C,
                kernel,
                field.to_array(),
                dir.to_array(),
                2,
                6,
                10,
            )
            .unwrap();
            std::array::from_fn::<_, 3, _>(|order| {
                let data = cache.fields(order, &density).unwrap();
                assert!(data.hermiticity_defect < 2e-12);
                [data.direct_energy, data.exchange_energy]
            })
        };
        let diagonal = axes.map(coefficient);
        let cross = [(0, 1), (0, 2), (1, 2)].map(|(i, j)| coefficient(axes[i] + axes[j]));
        let mut exact = Vec::new();
        let base = oracle(field, omega, &density);
        for term in 0..2 {
            assert!((diagonal[0][0][term] - base[term]).abs() < 2e-12);
            exact.extend(diagonal.iter().map(|d| d[1][term]));
            for (k, (i, j)) in independent.into_iter().enumerate() {
                exact.push(if i == j {
                    2.0 * diagonal[i][2][term]
                } else {
                    cross[k - 3][2][term] - diagonal[i][2][term] - diagonal[j][2][term]
                });
            }
        }
        let names: Vec<_> = (0..18)
            .map(|k| {
                format!(
                    "London large-density omega={omega:?} energy {}, component {}",
                    k / 9,
                    k % 9
                )
            })
            .collect();
        let difference = |k: usize, h: f64, stencil: Stencil| {
            let term = k / 9;
            let index = k % 9;
            let value = |point| oracle(point, omega, &density)[term];
            if index < 3 {
                let v = axes[index] * h;
                let (p2, m2) = if stencil == Stencil::Fourth {
                    (value(field + v * 2.), value(field - v * 2.))
                } else {
                    (0., 0.)
                };
                stencil.first(value(field + v), value(field - v), p2, m2, h)
            } else {
                let (i, j) = independent[index - 3];
                let second = |step: f64| {
                    let a = axes[i] * step;
                    let b = axes[j] * step;
                    if i == j {
                        ((value(field + a) - base[term]) + (value(field - a) - base[term]))
                            / step.powi(2)
                    } else {
                        ((value(field + a + b) - value(field + a - b))
                            - (value(field - a + b) - value(field - a - b)))
                            / (4.0 * step.powi(2))
                    }
                };
                let primary = second(h);
                if stencil == Stencil::Fourth {
                    (4.0 * primary - second(2.0 * h)) / 3.0
                } else {
                    primary
                }
            }
        };
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.10, 0.075, 0.054, 0.039], Stencil::Second),
                ([0.055, 0.041, 0.028, 0.020], Stencil::Second),
                ([0.10, 0.075, 0.054, 0.039], Stencil::Fourth),
            ],
            |steps, stencil| {
                exact
                    .iter()
                    .enumerate()
                    .map(|(k, w)| steps.map(|h| (difference(k, h, stencil) - w).abs()))
                    .collect()
            },
            &mut failures,
        );
        components::assert_no_failures(
            "London large-component fixed-density pair energies",
            &failures,
        );
        let mut error = 0.0_f64;
        for (k, w) in exact.iter().enumerate() {
            error = error.max((difference(k, 0.0038, Stencil::Fourth) - w).abs());
        }
        println!("London energy base {base:?}, omega {omega:?}, derivative components {exact:?}, absolute max {error:.6e}");
        assert!(error < 1e-8);
    }
}
