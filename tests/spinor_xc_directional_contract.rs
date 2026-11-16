// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete descriptor-potential coefficients for one independent mixed
//! density tangent. This is a point contract, not molecular acceptance.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::periodic::spinor_xc::{
    spinor_semilocal_point_directional_taylor, spinor_semilocal_point_taylor,
};
use dft_rs::xc::{resolve, FunctionalSpec};

fn paths() -> ([Vec<f64>; 32], [Vec<f64>; 32]) {
    let mut base: [f64; 32] = std::array::from_fn(|i| 0.012 * ((i * 7 + 2) as f64).sin());
    base[..4].copy_from_slice(&[0.225, 0.03, -0.02, 0.04]);
    base[16..20].copy_from_slice(&[2.0, 0.08, -0.07, 0.11]);
    let x = std::array::from_fn(|i| {
        vec![
            base[i],
            0.018 * ((i + 2) as f64).cos(),
            0.007 * ((i + 3) as f64).sin(),
        ]
    });
    let y = std::array::from_fn(|i| {
        vec![
            0.014 * ((i + 1) as f64).cos(),
            0.008 * ((i + 5) as f64).sin(),
            0.005 * ((i + 7) as f64).cos(),
        ]
    });
    (x, y)
}

#[test]
fn every_projected_potential_coefficient_has_four_step_density_fits() {
    let (x, y) = paths();
    for name in ["svwn5", "pbe", "tpss"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let dimension: usize = match name {
            "svwn5" => 4,
            "pbe" => 16,
            _ => 32,
        };
        // Walsh sign rows are orthogonal. Nonzero base tangent scales keep
        // these density directions linearly independent in every active slot.
        for mode in 0..dimension {
            let y: [Vec<f64>; 32] = std::array::from_fn(|i| {
                y[i].iter()
                    .map(|v| {
                        if ((i & mode).count_ones() % 2) == 0 {
                            *v
                        } else {
                            -*v
                        }
                    })
                    .collect()
            });
            let analytic =
                spinor_semilocal_point_directional_taylor(&functional, &x, &y, 4).unwrap();
            let original = spinor_semilocal_point_taylor(&functional, &x, 4).unwrap();
            for k in 0..3 {
                assert!((analytic.value.energy[k] - original.energy[k]).abs() < 2e-12);
                for i in 0..32 {
                    assert!(
                        (analytic.value.gradient[i][k] - original.gradient[i][k]).abs() < 2e-11
                    );
                }
            }
            let active: Vec<usize> = match name {
                "svwn5" => (0..4).collect(),
                "pbe" => (0..16).collect(),
                _ => (0..32).collect(),
            };
            for i in 0..32 {
                if !active.contains(&i) {
                    assert!(analytic.directional.gradient[i].iter().all(|v| *v == 0.0));
                }
            }
            let mut names = Vec::new();
            for i in &active {
                for k in 0..3 {
                    names.push(format!(
                    "{name} density mode {mode} projected potential {i}, nuclear coefficient {k}"
                ));
                }
            }
            for k in 0..3 {
                names.push(format!(
                    "{name} density mode {mode} projected energy nuclear coefficient {k}"
                ));
            }
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
                    let mut errors = vec![[0.0; 4]; names.len()];
                    for (step, h) in steps.iter().enumerate() {
                        let at = |sign: f64| {
                            let vars = std::array::from_fn(|i| {
                                (0..3).map(|k| x[i][k] + sign * h * y[i][k]).collect()
                            });
                            spinor_semilocal_point_taylor(&functional, &vars, 4).unwrap()
                        };
                        let plus = at(1.0);
                        let minus = at(-1.0);
                        let outer = if stencil == stencils::Stencil::Fourth {
                            Some((at(2.0), at(-2.0)))
                        } else {
                            None
                        };
                        let derivative = |plus: f64, minus: f64, outer: Option<(f64, f64)>| {
                            let (p, m) = outer.unwrap_or((0.0, 0.0));
                            stencil.first(plus, minus, p, m, *h)
                        };
                        for (j, &i) in active.iter().enumerate() {
                            for k in 0..3 {
                                errors[3 * j + k][step] = (derivative(
                                    plus.gradient[i][k],
                                    minus.gradient[i][k],
                                    outer
                                        .as_ref()
                                        .map(|(p, m)| (p.gradient[i][k], m.gradient[i][k])),
                                ) - analytic.directional.gradient[i][k])
                                    .abs();
                            }
                        }
                        for k in 0..3 {
                            errors[active.len() * 3 + k][step] = (derivative(
                                plus.energy[k],
                                minus.energy[k],
                                outer.as_ref().map(|(p, m)| (p.energy[k], m.energy[k])),
                            ) - analytic.directional.energy
                                [k])
                                .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
            components::assert_no_failures("projected tangent entries", &failures);
        }
    }
}
