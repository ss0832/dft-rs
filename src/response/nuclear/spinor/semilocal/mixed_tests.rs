// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::acceptance::{components, stencils};

fn path(seed: usize) -> Vec<Cm> {
    (0..3)
        .map(|k| {
            let mut value = Cm::zeros(4, 4);
            for i in 0..4 {
                for j in 0..=i {
                    let scale = if k == 0 { 0.009 } else { 0.003 / k as f64 };
                    let re = scale * ((seed + 5 * i + 3 * j + 2 * k) as f64).sin();
                    let im = if i == j {
                        0.0
                    } else {
                        scale * ((seed + 2 * i + 7 * j + k) as f64).cos()
                    };
                    value.re[(i, j)] = re;
                    value.re[(j, i)] = re;
                    value.im[(i, j)] = im;
                    value.im[(j, i)] = -im;
                }
                if k == 0 {
                    value.re[(i, i)] += 0.3;
                }
            }
            value
        })
        .collect()
}

fn components_of(fields: &[Cm], current: bool) -> Vec<f64> {
    let mut out = Vec::new();
    for value in fields {
        let pauli = pauli_components(&BlochMatrix {
            re: value.re.clone(),
            im: value.im.clone(),
        })
        .unwrap();
        for p in pauli {
            out.extend([p.re[(0, 0)], p.re[(1, 0)], p.re[(1, 1)]]);
            if current {
                out.push(p.im[(1, 0)]);
            } else {
                assert_eq!(p.im[(1, 0)], 0.0);
            }
        }
    }
    out
}

#[test]
fn moving_grid_mixed_potential_has_complete_matrix_coefficient_fits() {
    let density = path(1);
    let mut direction = path(11);
    for i in 0..4 {
        direction[0].re[(i, i)] -= 0.3;
    }
    let points = [GridPointTaylor {
        weight: vec![0.8, 0.03, -0.04],
        aos: vec![
            [
                vec![0.7, 0.02, -0.01],
                vec![0.2, -0.01, 0.02],
                vec![0.3, 0.04, -0.02],
                vec![-0.15, 0.02, 0.01],
            ],
            [
                vec![0.4, -0.03, 0.02],
                vec![-0.3, 0.02, 0.01],
                vec![0.1, -0.01, 0.04],
                vec![0.4, 0.03, -0.01],
            ],
        ],
    }];
    for name in ["svwn5", "pbe", "tpss"] {
        let functional =
            crate::xc::resolve(&crate::xc::FunctionalSpec::Combined(name.into())).unwrap();
        let current = name == "tpss";
        let direct = evaluate_linear(&points, &functional, &density, &direction, 2, 4).unwrap();
        assert!(KernelCache::new(&functional, &points, &density, 2, 0, 4)
            .unwrap()
            .is_none());
        let mut cache = KernelCache::new(&functional, &points, &density, 2, 64, 4)
            .unwrap()
            .unwrap();
        assert!(cache.supports(4));
        assert!(cache.supports(8));
        assert!(!cache.supports(16));
        let actual = cache.apply(&direction, 4).unwrap();
        let parity = components_of(&actual, current)
            .into_iter()
            .zip(components_of(&direct, current))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("{name} cached/direct moving coefficient max defect {parity:.16e}");
        assert!(parity < 1e-12);
        let repeated = cache.apply(&direction, 4).unwrap();
        assert_eq!(
            components_of(&actual, current),
            components_of(&repeated, current)
        );
        let finer = cache.apply(&direction, 8).unwrap();
        let finer_direct =
            evaluate_linear(&points, &functional, &density, &direction, 2, 8).unwrap();
        for (a, b) in components_of(&finer, current)
            .into_iter()
            .zip(components_of(&finer_direct, current))
        {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(cache.apply(&direction, 16).is_err());
        let analytic = components_of(&actual, current);
        let names = (0..analytic.len())
            .map(|i| format!("{name} moving grid mixed coefficient {i}"))
            .collect::<Vec<_>>();
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.4, 0.2, 0.1, 0.05], stencils::Stencil::Second),
                ([0.2, 0.1, 0.05, 0.025], stencils::Stencil::Second),
                ([0.8, 0.4, 0.2, 0.1], stencils::Stencil::Fourth),
                ([0.4, 0.2, 0.1, 0.05], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; analytic.len()];
                for (step, &h) in steps.iter().enumerate() {
                    let at = |epsilon: f64| {
                        let mut p = density.clone();
                        for k in 0..3 {
                            for i in 0..4 {
                                for j in 0..4 {
                                    p[k].re[(i, j)] += epsilon * direction[k].re[(i, j)];
                                    p[k].im[(i, j)] += epsilon * direction[k].im[(i, j)];
                                }
                            }
                        }
                        let (_, fock) = super::super::evaluate(
                            &points,
                            &functional,
                            &p,
                            2,
                            4,
                            true,
                            true,
                            None,
                        )
                        .unwrap();
                        components_of(&fock, current)
                    };
                    let plus = at(h);
                    let minus = at(-h);
                    let outer = if stencil == stencils::Stencil::Fourth {
                        Some((at(2.0 * h), at(-2.0 * h)))
                    } else {
                        None
                    };
                    for i in 0..analytic.len() {
                        let (p, m) = outer.as_ref().map_or((0.0, 0.0), |(p, m)| (p[i], m[i]));
                        errors[i][step] =
                            (stencil.first(plus[i], minus[i], p, m, h) - analytic[i]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("moving grid mixed coefficients", &failures);
    }
}
