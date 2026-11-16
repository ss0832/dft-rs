// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Synthetic network contracts, not accuracy of a fitted physical model.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/finite_difference_pair.rs"]
mod finite_difference_pair;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::dual::DualN;
use dft_rs::skala::spinor::{spinor_skala_grid, SpinorSkalaInput};
use dft_rs::skala::{random_tensors, Number, SkalaInput, SkalaModel, SkalaSizes};
use finite_difference_pair::FinitePair;

fn model() -> SkalaModel {
    SkalaModel::from_tensors(&random_tensors(
        7,
        SkalaSizes {
            width: 6,
            channels: 2,
            shells: 2,
            correlation: 2,
            layers: 1,
            post: 3,
        },
    ))
    .unwrap()
}
fn descriptors() -> [[f64; 32]; 2] {
    std::array::from_fn(|p| {
        let mut x = std::array::from_fn(|i| 0.002 * (i + 1) as f64 * (1.0 + 0.1 * p as f64));
        x[..4].copy_from_slice(&[
            0.5 - 0.06 * p as f64,
            0.031 + 0.004 * p as f64,
            -0.021,
            0.014,
        ]);
        x[4] = 0.12;
        x[5] = -0.07;
        x[6] = 0.09;
        x[16] = 0.9 + 0.2 * p as f64;
        x[17] = 0.017;
        x[18] = -0.025;
        x[19] = 0.011;
        x
    })
}
fn evaluate<S: Number>(
    model: &SkalaModel,
    x: &[[S; 32]; 2],
    order: usize,
) -> dft_rs::skala::spinor::SpinorSkalaOutput<S> {
    spinor_skala_grid(
        model,
        &SpinorSkalaInput {
            atom_of: &[0, 0],
            points: &[[0.7, 0.2, -0.1], [-0.3, 0.8, 0.4]],
            centers: &[[0.0; 3]],
            weights: &[S::from_f64(0.4), S::from_f64(0.6)],
            atomic_weights: &[0.5, 0.7],
            descriptors: x,
        },
        order,
    )
    .unwrap()
}
#[test]
fn complete_learned_projected_density_and_transverse_kernel() {
    let model = model();
    let x = descriptors();
    let base = evaluate(&model, &x, 4);
    assert_eq!(base.angular_order, 4);
    let scale: Vec<_> = (0..40)
        .map(|k| x[k / 20][k % 20].abs().max(0.025))
        .collect();
    let mut analytic = Vec::new();
    let mut names = Vec::new();
    for i in 0..40 {
        analytic.push(base.gradient[i / 20][i % 20] * scale[i]);
        names.push(format!("global learned gradient{i}"));
    }
    let mut hessian = vec![vec![0.0; 40]; 40];
    for j in 0..40 {
        let jet = std::array::from_fn(|p| {
            std::array::from_fn(|i| DualN {
                re: x[p][i],
                im: [if j == 20 * p + i { scale[j] } else { 0.0 }],
            })
        });
        let value = evaluate(&model, &jet, 4);
        for i in 0..40 {
            hessian[i][j] = value.gradient[i / 20][i % 20].im[0] * scale[i];
        }
        for p in &value.gradient {
            for v in &p[20..] {
                assert_eq!(v.re, 0.0);
                assert_eq!(v.im[0], 0.0);
            }
        }
    }
    for (i, row) in hessian.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert!((v - hessian[j][i]).abs() < 1e-12 * (1.0 + v.abs()));
            analytic.push(*v);
            names.push(format!("global learned kernel{i},{j}"));
        }
    }
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.16, 0.08, 0.04, 0.02], stencils::Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
            ([0.8, 0.4, 0.2, 0.1], stencils::Stencil::Second),
            ([0.6, 0.4, 0.2, 0.1], stencils::Stencil::Second),
            ([0.8, 0.6, 0.4, 0.3], stencils::Stencil::Second),
            ([0.6, 0.5, 0.4, 0.3], stencils::Stencil::Second),
            ([0.05, 0.025, 0.0125, 0.00625], stencils::Stencil::Second),
            ([0.1, 0.05, 0.025, 0.0125], stencils::Stencil::Second),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 1640];
            if steps[0] == 0.05 || steps[0] == 0.1 {
                println!("Independent finite-pair divided differences: steps {steps:?}");
                for (k, &h) in steps.iter().enumerate() {
                    for j in 0..40 {
                        let pair = std::array::from_fn(|p| {
                            std::array::from_fn(|i| {
                                FinitePair::shifted(
                                    x[p][i],
                                    if j == 20 * p + i { h * scale[j] } else { 0.0 },
                                )
                            })
                        });
                        let value = evaluate(&model, &pair, 4);
                        // Validate both finite endpoints independently against
                        // the ordinary production-scalar network evaluation.
                        let plus = evaluate(&model, &pair.map(|p| p.map(FinitePair::plus)), 4);
                        let minus = evaluate(&model, &pair.map(|p| p.map(FinitePair::minus)), 4);
                        assert!((value.energy.plus() - plus.energy).abs() < 1e-12);
                        assert!((value.energy.minus() - minus.energy).abs() < 1e-12);
                        errors[j][k] = (value.energy.half_difference / h - analytic[j]).abs();
                        for i in 0..40 {
                            let entry = value.gradient[i / 20][i % 20];
                            assert!((entry.plus() - plus.gradient[i / 20][i % 20]).abs() < 1e-12);
                            assert!((entry.minus() - minus.gradient[i / 20][i % 20]).abs() < 1e-12);
                            let slot = 40 + 40 * i + j;
                            errors[slot][k] =
                                (entry.half_difference / h * scale[i] - analytic[slot]).abs();
                        }
                    }
                }
                return errors;
            }

            for (k, &h) in steps.iter().enumerate() {
                for j in 0..40 {
                    let shifted = |t: f64| {
                        let mut y = x;
                        y[j / 20][j % 20] += t * scale[j];
                        evaluate(&model, &y, 4)
                    };
                    let p = shifted(h);
                    let m = shifted(-h);
                    let extra = if stencil == stencils::Stencil::Fourth {
                        Some((shifted(2.0 * h), shifted(-2.0 * h)))
                    } else {
                        None
                    };
                    errors[j][k] = (stencil.first(
                        p.energy,
                        m.energy,
                        extra.as_ref().map_or(0.0, |(p, _)| p.energy),
                        extra.as_ref().map_or(0.0, |(_, m)| m.energy),
                        h,
                    ) - analytic[j])
                        .abs();
                    for i in 0..40 {
                        let at = |q: &dft_rs::skala::spinor::SpinorSkalaOutput<f64>| {
                            q.gradient[i / 20][i % 20] * scale[i]
                        };
                        let slot = 40 + 40 * i + j;
                        errors[slot][k] = (stencil.first(
                            at(&p),
                            at(&m),
                            extra.as_ref().map_or(0.0, |(p, _)| at(p)),
                            extra.as_ref().map_or(0.0, |(_, m)| at(m)),
                            h,
                        ) - analytic[slot])
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("complete global learned grid gradient/kernel", &failures);
}
#[test]
fn learned_global_projection_restores_collinear_model() {
    let model = model();
    let mut x = descriptors();
    for p in &mut x {
        for start in [0, 4, 7, 10, 16] {
            p[start + 1] = 0.0;
            p[start + 2] = 0.0;
            p[start + 3] = 0.06 * p[start];
        }
    }
    // Gradient Pauli vectors are stored by component rather than spin block.
    for p in &mut x {
        for d in 0..3 {
            p[7 + d] = 0.0;
            p[10 + d] = 0.0;
            p[13 + d] = 0.06 * p[4 + d];
        }
    }
    let rho = std::array::from_fn::<_, 2, _>(|s| {
        x.map(|p| 0.5 * (p[0] + if s == 0 { p[3] } else { -p[3] }))
    });
    let grad = std::array::from_fn::<_, 2, _>(|s| {
        x.map(|p| {
            std::array::from_fn(|d| 0.5 * (p[4 + d] + if s == 0 { p[13 + d] } else { -p[13 + d] }))
        })
    });
    let tau = std::array::from_fn::<_, 2, _>(|s| {
        x.map(|p| 0.5 * (p[16] + if s == 0 { p[19] } else { -p[19] }))
    });
    let expected = model
        .evaluate(
            &SkalaInput {
                atom_of: &[0, 0],
                points: &[[0.7, 0.2, -0.1], [-0.3, 0.8, 0.4]],
                centers: &[[0.0; 3]],
                weights: &[0.4, 0.6],
                atomic_weights: &[0.5, 0.7],
                rho: [&rho[0], &rho[1]],
                gradient: [&grad[0], &grad[1]],
                tau: [&tau[0], &tau[1]],
            },
            true,
        )
        .unwrap();
    let a = evaluate(&model, &x, 8);
    let b = evaluate(&model, &x, 16);
    assert!((a.energy - b.energy).abs() < 1e-12);
    assert!((b.energy - expected.energy).abs() < 1e-12);
}

#[test]
fn learned_global_projection_is_spin_rotation_covariant() {
    let model = model();
    let x = descriptors();
    let (sz, cz) = 0.37_f64.sin_cos();
    let (sy, cy) = (-0.21_f64).sin_cos();
    let rotation = [
        [cz * cy, -sz, cz * sy],
        [sz * cy, cz, sz * sy],
        [-sy, 0.0, cy],
    ];
    let groups = [
        [1, 2, 3],
        [7, 10, 13],
        [8, 11, 14],
        [9, 12, 15],
        [17, 18, 19],
    ];
    let mut y = x;
    for p in &mut y {
        for axes in groups {
            let old = axes.map(|i| p[i]);
            for a in 0..3 {
                p[axes[a]] = (0..3).map(|b| rotation[a][b] * old[b]).sum();
            }
        }
    }
    let old = evaluate(&model, &x, 16);
    let new = evaluate(&model, &y, 16);
    let old_fine = evaluate(&model, &x, 32);
    let new_fine = evaluate(&model, &y, 32);
    assert!((old.energy - old_fine.energy).abs() < 1e-12);
    assert!((new.energy - new_fine.energy).abs() < 1e-12);
    assert!((old_fine.energy - new_fine.energy).abs() < 1e-12);
    for p in 0..2 {
        for i in 0..32 {
            assert!((old.gradient[p][i] - old_fine.gradient[p][i]).abs() < 1e-11);
            assert!((new.gradient[p][i] - new_fine.gradient[p][i]).abs() < 1e-11);
        }
    }

    for p in 0..2 {
        let mut back = new_fine.gradient[p];
        for axes in groups {
            let values = axes.map(|i| new_fine.gradient[p][i]);
            for b in 0..3 {
                back[axes[b]] = (0..3).map(|a| rotation[a][b] * values[a]).sum();
            }
        }
        for (i, v) in back.iter().enumerate() {
            assert!(
                (v - old_fine.gradient[p][i]).abs() < 1e-11,
                "rotated gradient {p}/{i}"
            );
        }
    }
}

#[test]
fn finite_pair_keeps_distinct_endpoints_and_elementary_divided_differences() {
    use dft_rs::dual::Scalar;
    for h in [0.08, 0.04, 0.02, 0.01] {
        let x = FinitePair::shifted(0.7, h);
        let functions = [
            x.exp(),
            x.ln(),
            x.sqrt(),
            x.cbrt(),
            x.recip(),
            x.powi(3),
            x.ln_1p(),
        ];
        let expected = [
            ((0.7 + h).exp(), (0.7 - h).exp()),
            ((0.7 + h).ln(), (0.7 - h).ln()),
            ((0.7 + h).sqrt(), (0.7 - h).sqrt()),
            ((0.7 + h).cbrt(), (0.7 - h).cbrt()),
            (1.0 / (0.7 + h), 1.0 / (0.7 - h)),
            ((0.7 + h).powi(3), (0.7 - h).powi(3)),
            ((0.7 + h).ln_1p(), (0.7 - h).ln_1p()),
        ];
        for (value, (plus, minus)) in functions.into_iter().zip(expected) {
            assert!((value.plus() - plus).abs() < 1e-14);
            assert!((value.minus() - minus).abs() < 1e-14);
            assert!((2.0 * value.half_difference - (plus - minus)).abs() < 1e-14);
        }
    }
    let x = FinitePair::shifted(1.0, 1e-20).ln();
    assert!((x.half_difference - 1e-20).abs() < 1e-35);
}
