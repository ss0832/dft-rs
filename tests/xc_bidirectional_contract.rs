// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All mixed potential slots of the generic two-direction Taylor contract.
//! A polynomial reference supplies closed-form third derivatives; this is
//! a provider-interface contract, not acceptance of named fifth derivatives.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::dual::Scalar;
use dft_rs::xc::taylor::XcTaylorInput;

fn expression<S: Scalar>(x: &[S], p: &[S]) -> S {
    let l = (0..7).fold(S::from_f64(0.0), |sum, i| {
        sum + S::from_f64(0.3 + 0.05 * i as f64) * x[i]
    });
    let mut value = l * l * l * l * l + p[0] * x[0] * x[1];
    for (i, &value_i) in x.iter().enumerate() {
        value = value + S::from_f64(0.11 + 0.03 * i as f64) * value_i * value_i * value_i;
    }
    value
}
fn base() -> [f64; 7] {
    [0.7, 0.43, 0.4, 0.1, 0.3, 1.1, 0.9]
}
fn potential(x: [f64; 7], i: usize) -> f64 {
    let l = (0..7).map(|j| (0.3 + 0.05 * j as f64) * x[j]).sum::<f64>();
    5.0 * (0.3 + 0.05 * i as f64) * l.powi(4)
        + 3.0 * (0.11 + 0.03 * i as f64) * x[i] * x[i]
        + if i == 0 {
            0.17 * x[1]
        } else if i == 1 {
            0.17 * x[0]
        } else {
            0.0
        }
}

#[test]
fn every_third_partial_slot_has_an_independent_four_step_fit() {
    let x = base();
    let input = XcTaylorInput {
        variables: x.map(|v| vec![v]),
        parameters: None,
    };
    let zero = std::array::from_fn(|_| vec![0.0]);
    let mut analytic = Vec::new();
    let mut names = Vec::new();
    for a in 0..7 {
        for b in 0..7 {
            let mut first = zero.clone();
            let mut second = zero.clone();
            first[a][0] = 1.0;
            second[b][0] = 1.0;
            let out = input
                .evaluate_bidirectional_potentials(&first, &second, &zero, &[0.17], expression)
                .unwrap();
            assert!((out.value.energy[0] - expression(&x, &[0.17])).abs() < 1e-10);
            for j in 0..7 {
                names.push(format!("bidirectional potential {j}, directions {a},{b}"));
                analytic.push(out.mixed.potentials[j][0]);
                assert!((out.value.potentials[j][0] - potential(x, j)).abs() < 1e-10);
                assert!(out.first.energy[0].is_finite() && out.second.energy[0].is_finite());
            }
        }
    }
    for j in 0..7 {
        names.push(format!("base potential {j}"));
        analytic.push(potential(x, j));
    }
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.16, 0.08, 0.04, 0.02], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 350];
            for (k, &h) in steps.iter().enumerate() {
                for a in 0..7 {
                    for b in 0..7 {
                        for j in 0..7 {
                            let fd = |h: f64| {
                                let at = |da: f64, db: f64| {
                                    let mut point = x;
                                    point[a] += da;
                                    point[b] += db;
                                    potential(point, j)
                                };
                                if a == b {
                                    (at(h, 0.0) - 2.0 * potential(x, j) + at(-h, 0.0)) / (h * h)
                                } else {
                                    (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4.0 * h * h)
                                }
                            };
                            let value = if stencil == stencils::Stencil::Second {
                                fd(h)
                            } else {
                                (4.0 * fd(h) - fd(2.0 * h)) / 3.0
                            };
                            let index = (a * 7 + b) * 7 + j;
                            errors[index][k] = (value - analytic[index]).abs();
                        }
                    }
                }
                for j in 0..7 {
                    let at = |step: f64| {
                        let mut point = x;
                        point[j] += step;
                        expression(&point, &[0.17])
                    };
                    errors[343 + j][k] =
                        (stencil.first(at(h), at(-h), at(2.0 * h), at(-2.0 * h), h)
                            - analytic[343 + j])
                            .abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all generic mixed potential slots", &failures);
}

#[derive(Clone, Copy)]
struct Polynomial([f64; 3]);
impl Polynomial {
    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|k| self.0[k] + other.0[k]))
    }
    fn scale(self, s: f64) -> Self {
        Self(self.0.map(|v| s * v))
    }
    fn mul(self, other: Self) -> Self {
        Self(std::array::from_fn(|k| {
            (0..=k).map(|j| self.0[j] * other.0[k - j]).sum()
        }))
    }
    fn pow(self, n: usize) -> Self {
        (0..n).fold(Self([1.0, 0.0, 0.0]), |p, _| p.mul(self))
    }
}

#[test]
fn mixed_seed_parameter_paths_and_every_normalized_output_coefficient_match_polynomials() {
    let x: [Polynomial; 7] = std::array::from_fn(|i| {
        Polynomial([base()[i], 0.03 * (i + 1) as f64, -0.02 * (i + 2) as f64])
    });
    let a: [Polynomial; 7] = std::array::from_fn(|i| {
        Polynomial([
            0.07 * (i + 1) as f64,
            -0.013 * (i + 2) as f64,
            0.011 * (i + 3) as f64,
        ])
    });
    let b: [Polynomial; 7] = std::array::from_fn(|i| {
        Polynomial([
            -0.04 * (i + 2) as f64,
            0.017 * (i + 1) as f64,
            -0.009 * (i + 4) as f64,
        ])
    });
    let c: [Polynomial; 7] = std::array::from_fn(|i| {
        Polynomial([
            0.019 * (i + 1) as f64,
            -0.012 * (i + 3) as f64,
            0.008 * (i + 2) as f64,
        ])
    });
    let parameter = Polynomial([0.17, 0.03, -0.01]);
    let input = XcTaylorInput {
        variables: x.map(|p| p.0.to_vec()),
        parameters: Some(vec![parameter.0.to_vec()]),
    };
    let out = input
        .evaluate_bidirectional_potentials(
            &a.map(|p| p.0.to_vec()),
            &b.map(|p| p.0.to_vec()),
            &c.map(|p| p.0.to_vec()),
            &[0.17],
            expression,
        )
        .unwrap();
    let weighted = |values: [Polynomial; 7]| {
        (0..7).fold(Polynomial([0.0; 3]), |sum, i| {
            sum.add(values[i].scale(0.3 + 0.05 * i as f64))
        })
    };
    let l = weighted(x);
    let la = weighted(a);
    let lb = weighted(b);
    let lc = weighted(c);
    let mut e0 = l.pow(5).add(parameter.mul(x[0]).mul(x[1]));
    let mut ea = l
        .pow(4)
        .mul(la)
        .scale(5.0)
        .add(parameter.mul(a[0].mul(x[1]).add(x[0].mul(a[1]))));
    let mut eb = l
        .pow(4)
        .mul(lb)
        .scale(5.0)
        .add(parameter.mul(b[0].mul(x[1]).add(x[0].mul(b[1]))));
    let mut ec = l
        .pow(3)
        .mul(la)
        .mul(lb)
        .scale(20.0)
        .add(l.pow(4).mul(lc).scale(5.0))
        .add(
            parameter.mul(
                c[0].mul(x[1])
                    .add(a[0].mul(b[1]))
                    .add(b[0].mul(a[1]))
                    .add(x[0].mul(c[1])),
            ),
        );
    for i in 0..7 {
        let q = 0.11 + 0.03 * i as f64;
        e0 = e0.add(x[i].pow(3).scale(q));
        ea = ea.add(x[i].pow(2).mul(a[i]).scale(3.0 * q));
        eb = eb.add(x[i].pow(2).mul(b[i]).scale(3.0 * q));
        ec = ec.add(
            x[i].mul(a[i])
                .mul(b[i])
                .scale(6.0 * q)
                .add(x[i].pow(2).mul(c[i]).scale(3.0 * q)),
        );
    }
    for (actual, expected) in [&out.value, &out.first, &out.second, &out.mixed]
        .into_iter()
        .zip([e0, ea, eb, ec])
    {
        for (v, w) in actual.energy.iter().zip(expected.0) {
            assert!((v - w).abs() < 1e-10);
        }
    }
    for i in 0..7 {
        let w = 0.3 + 0.05 * i as f64;
        let q = 0.11 + 0.03 * i as f64;
        let mut values = [
            l.pow(4).scale(5.0 * w).add(x[i].pow(2).scale(3.0 * q)),
            l.pow(3)
                .mul(la)
                .scale(20.0 * w)
                .add(x[i].mul(a[i]).scale(6.0 * q)),
            l.pow(3)
                .mul(lb)
                .scale(20.0 * w)
                .add(x[i].mul(b[i]).scale(6.0 * q)),
            l.pow(2)
                .mul(la)
                .mul(lb)
                .scale(60.0 * w)
                .add(l.pow(3).mul(lc).scale(20.0 * w))
                .add(a[i].mul(b[i]).add(x[i].mul(c[i])).scale(6.0 * q)),
        ];
        if i < 2 {
            for (value, tangent) in values
                .iter_mut()
                .zip([x[1 - i], a[1 - i], b[1 - i], c[1 - i]])
            {
                *value = value.add(parameter.mul(tangent));
            }
        }
        for (actual, expected) in [&out.value, &out.first, &out.second, &out.mixed]
            .into_iter()
            .zip(values)
        {
            for (v, w) in actual.potentials[i].iter().zip(expected.0) {
                assert!((v - w).abs() < 1e-10, "slot {i}: {v} != {w}");
            }
        }
    }
}
