// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! High-order dispersion derivatives against independently displaced analytic
//! force constants. Finite differences enter validation only. Every step is
//! included in each four-point log-error fit, with no error floors.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::dispersion::Dispersion;
use dft_rs::math::Vec3;
use dft_rs::periodic::dispersion::{
    periodic_d3, periodic_dispersion_force_constant_tensor,
    periodic_dispersion_force_constants_along, periodic_dispersion_hessian, PeriodicDispersion,
    ThreeBody,
};
use dft_rs::periodic::Lattice;

const NUMBERS: [u8; 3] = [6, 7, 8];
const POINTS: [[f64; 3]; 3] = [[0.2, 0.1, 0.3], [3.1, 0.6, 0.2], [0.9, 3.4, 1.6]];
const U: [f64; 9] = [0.3, -0.2, 0.1, -0.3, 0.1, 0.2, 0.0, 0.1, -0.3];
const V: [f64; 9] = [-0.1, 0.3, 0.2, 0.2, -0.1, -0.3, -0.1, -0.2, 0.1];

fn lattice() -> Lattice {
    Lattice::new(
        Vec3::new(10.3, 0.0, 0.0),
        Vec3::new(0.8, 11.1, 0.0),
        Vec3::new(0.4, 0.7, 10.9),
    )
}

fn positions(first: f64, second: f64) -> Vec<Vec3> {
    POINTS
        .iter()
        .enumerate()
        .map(|(atom, r)| {
            let p: [f64; 3] = std::array::from_fn(|axis| {
                r[axis] + first * U[3 * atom + axis] + second * V[3 * atom + axis]
            });
            Vec3::new(p[0], p[1], p[2])
        })
        .collect()
}

fn models() -> Vec<(&'static str, PeriodicDispersion)> {
    // These cutoffs have measured image-distance margins throughout the
    // four-step stencil; the original 13.2 Bohr pair cutoff crossed images.
    let bj = PeriodicDispersion::from_model(&Dispersion::D3Bj, "pbe", 7.0, 12.0)
        .unwrap()
        .unwrap();
    vec![
        ("D3(BJ)", bj.clone()),
        (
            "D3(zero)",
            PeriodicDispersion::from_model(&Dispersion::D3Zero, "pbe", 7.0, 12.0)
                .unwrap()
                .unwrap(),
        ),
        (
            "D3(BJ)+ATM",
            bj.with_three_body(ThreeBody {
                cutoff: 6.0,
                ..ThreeBody::default()
            })
            .unwrap(),
        ),
        (
            "CHG",
            PeriodicDispersion::from_model(&Dispersion::Chg, "wb97xd", 7.0, 12.0)
                .unwrap()
                .unwrap(),
        ),
    ]
}

fn largest(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn contracted_periodic_dispersion_third_and_mixed_fourth_derivatives_have_second_order_error_fits()
{
    let steps = [0.06, 0.03, 0.015, 0.0075];
    for (name, model) in models() {
        let point = positions(0.0, 0.0);
        let lattice = lattice();
        let cubic = periodic_dispersion_force_constants_along(
            &NUMBERS,
            &point,
            &lattice,
            &model,
            &[U.to_vec()],
        )
        .unwrap();
        let quartic = periodic_dispersion_force_constants_along(
            &NUMBERS,
            &point,
            &lattice,
            &model,
            &[U.to_vec(), V.to_vec()],
        )
        .unwrap();
        let reversed = periodic_dispersion_force_constants_along(
            &NUMBERS,
            &point,
            &lattice,
            &model,
            &[V.to_vec(), U.to_vec()],
        )
        .unwrap();
        let energy = periodic_d3(&NUMBERS, &point, &lattice, &model)
            .unwrap()
            .energy;
        assert_eq!(cubic.energy, energy);
        assert_eq!(quartic.energy, energy);
        let hessian = |u: f64, v: f64| {
            periodic_dispersion_hessian(&NUMBERS, &positions(u, v), &lattice, &model).unwrap()
        };
        let mut third = Vec::new();
        let mut fourth = Vec::new();
        for h in steps {
            let (plus, minus) = (hessian(h, 0.0), hessian(-h, 0.0));
            third.push(
                plus.as_slice()
                    .iter()
                    .zip(minus.as_slice())
                    .map(|(a, b)| (a - b) / (2.0 * h))
                    .collect::<Vec<_>>(),
            );
            let corner = [
                hessian(h, h),
                hessian(h, -h),
                hessian(-h, h),
                hessian(-h, -h),
            ];
            fourth.push(
                (0..81)
                    .map(|i| {
                        (corner[0].as_slice()[i]
                            - corner[1].as_slice()[i]
                            - corner[2].as_slice()[i]
                            + corner[3].as_slice()[i])
                            / (4.0 * h * h)
                    })
                    .collect::<Vec<_>>(),
            );
        }
        for (order, exact, differences) in [(3, &cubic, third), (4, &quartic, fourth)] {
            let errors: Vec<_> = differences
                .iter()
                .map(|fd| largest(fd, exact.matrix.as_slice()))
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson: Vec<_> = differences[3]
                .iter()
                .zip(&differences[2])
                .map(|(a, b)| (4.0 * a - b) / 3.0)
                .collect();
            let residual = largest(&richardson, exact.matrix.as_slice());
            println!("{name} order {order}: steps={steps:?} errors={errors:?} p={:.12} R2={:.12} log_rmse={:.6e} Richardson={residual:.12e} acoustic={:.12e}",
                fit.exponent, fit.r_squared, fit.log_rmse, exact.acoustic_residual);
            assert!(
                (fit.exponent - 2.0).abs() < 0.15 && fit.r_squared > 0.999,
                "{name} order {order}"
            );
            assert!(residual < 1e-8, "{name} order {order}: {residual:e}");
            assert!(exact.acoustic_residual < 1e-11);
        }
        let reversal = largest(quartic.matrix.as_slice(), reversed.matrix.as_slice());
        println!("{name}: mixed-probe reversal={reversal:.12e}");
        assert!(reversal < 1e-12);
        let translation = [0.2, -0.3, 0.4].repeat(3);
        let rigid = periodic_dispersion_force_constants_along(
            &NUMBERS,
            &point,
            &lattice,
            &model,
            &[translation.clone(), V.to_vec()],
        )
        .unwrap();
        assert!(rigid.matrix.as_slice().iter().all(|v| v.abs() < 1e-12));
        assert!(
            periodic_dispersion_force_constants_along(&NUMBERS, &point, &lattice, &model, &[])
                .is_err()
        );
    }
}

#[test]
fn complete_dispersion_tensors_match_their_contractions_and_preserve_raw_constraints() {
    let numbers = [6, 8];
    let point = vec![Vec3::new(0.2, 0.1, 0.3), Vec3::new(3.1, 0.6, 0.2)];
    let lattice = lattice();
    let model = models().into_iter().next().unwrap().1;
    let first = vec![0.3, -0.2, 0.1, -0.3, 0.2, -0.1];
    let second = vec![-0.1, 0.3, 0.2, 0.1, -0.3, -0.2];
    let steps = [0.12, 0.06, 0.03, 0.015];
    for order in [3, 4] {
        let tensor =
            periodic_dispersion_force_constant_tensor(&numbers, &point, &lattice, &model, order)
                .unwrap();
        let probes = if order == 3 {
            vec![first.clone()]
        } else {
            vec![first.clone(), second.clone()]
        };
        let exact =
            periodic_dispersion_force_constants_along(&numbers, &point, &lattice, &model, &probes)
                .unwrap();
        let mut contracted = vec![0.0; 36];
        for k in 0..6 {
            for l in 0..6 {
                for i in 0..6 {
                    if order == 3 {
                        contracted[6 * k + l] += first[i] * tensor.tensor[(6 * i + k) * 6 + l];
                    } else {
                        for j in 0..6 {
                            contracted[6 * k + l] +=
                                first[i] * second[j] * tensor.tensor[((6 * i + j) * 6 + k) * 6 + l];
                        }
                    }
                }
            }
        }
        let contraction = largest(&contracted, exact.matrix.as_slice());
        let displaced = |u: f64, v: f64| {
            let p: Vec<_> = point
                .iter()
                .enumerate()
                .map(|(a, r)| {
                    let c: [f64; 3] = std::array::from_fn(|x| {
                        r.get(x) + u * first[3 * a + x] + v * second[3 * a + x]
                    });
                    Vec3::new(c[0], c[1], c[2])
                })
                .collect();
            periodic_dispersion_hessian(&numbers, &p, &lattice, &model).unwrap()
        };
        let errors: Vec<_> = steps
            .iter()
            .map(|&h| {
                let fd: Vec<_> = if order == 3 {
                    let (a, b) = (displaced(h, 0.0), displaced(-h, 0.0));
                    a.as_slice()
                        .iter()
                        .zip(b.as_slice())
                        .map(|(a, b)| (a - b) / (2.0 * h))
                        .collect()
                } else {
                    let corner = [
                        displaced(h, h),
                        displaced(h, -h),
                        displaced(-h, h),
                        displaced(-h, -h),
                    ];
                    (0..36)
                        .map(|i| {
                            (corner[0].as_slice()[i]
                                - corner[1].as_slice()[i]
                                - corner[2].as_slice()[i]
                                + corner[3].as_slice()[i])
                                / (4.0 * h * h)
                        })
                        .collect()
                };
                largest(&fd, &contracted)
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!("complete D3(BJ) order {order}: steps={steps:?} errors={errors:?} p={:.12} R2={:.12} contraction={contraction:.12e} permutation={:.12e} acoustic={:.12e}",fit.exponent,fit.r_squared,tensor.permutation_residual,tensor.acoustic_residual);
        assert!((fit.exponent - 2.0).abs() < 0.15 && fit.r_squared > 0.999);
        assert!(
            contraction < 1e-12
                && tensor.permutation_residual < 1e-11
                && tensor.acoustic_residual < 1e-11
        );
    }
}

/// Choose a cutoff inside a measured gap between image distances before
/// evaluating derivatives. This keeps every finite-step image list fixed.
fn cutoff_in_image_gap(lattice: &Lattice, points: &[Vec3], lower: f64, upper: f64) -> (f64, f64) {
    let mut distances = vec![lower, upper];
    for (i, a) in points.iter().enumerate() {
        for (j, b) in points.iter().enumerate() {
            for x in -3..=3 {
                for y in -3..=3 {
                    for z in -3..=3 {
                        if i == j && [x, y, z] == [0, 0, 0] {
                            continue;
                        }
                        let r = (*b - *a + lattice.translation([x, y, z])).norm();
                        if r > lower && r < upper {
                            distances.push(r);
                        }
                    }
                }
            }
        }
    }
    distances.sort_by(f64::total_cmp);
    let gap = distances
        .windows(2)
        .max_by(|a, b| (a[1] - a[0]).total_cmp(&(b[1] - b[0])))
        .unwrap();
    (0.5 * (gap[0] + gap[1]), 0.5 * (gap[1] - gap[0]))
}

#[test]
fn atm_alias_and_explicit_controls_select_the_same_high_order_energy_model() {
    use dft_rs::driver::{periodic::PeriodicMethod, Controls, Value};
    for (alias, pair) in [("d3bj-abc", "d3bj"), ("d3abc", "d3")] {
        let options = |name, flag| {
            let mut v = Value::object()
                .with("xc", "pbe")
                .with("dispersion", name)
                .with("cn_cutoff", 7.0)
                .with("disp_cutoff", 12.0)
                .with("abc_cutoff", 6.0);
            if let Some(flag) = flag {
                v.set("three_body", flag);
            }
            PeriodicMethod::from_controls_with_dispersion(&Controls::from_value(&v).unwrap())
        };
        let selected = options(alias, None::<bool>).unwrap();
        assert!(selected.three_body);
        let alias_model = selected.dispersion_model().unwrap().unwrap();
        assert_eq!(alias_model.three_body.unwrap().cutoff, 6.0);
        let explicit = options(pair, Some(true))
            .unwrap()
            .dispersion_model()
            .unwrap()
            .unwrap();
        let points = positions(0.0, 0.0);
        for probes in [vec![U.to_vec()], vec![U.to_vec(), V.to_vec()]] {
            let a = periodic_dispersion_force_constants_along(
                &NUMBERS,
                &points,
                &lattice(),
                &alias_model,
                &probes,
            )
            .unwrap();
            let b = periodic_dispersion_force_constants_along(
                &NUMBERS,
                &points,
                &lattice(),
                &explicit,
                &probes,
            )
            .unwrap();
            assert_eq!(a.energy, b.energy);
            assert_eq!(a.matrix.as_slice(), b.matrix.as_slice());
        }
        assert!(options(alias, Some(false)).is_err());
    }
}

#[test]
fn third_and_fourth_dispersion_derivatives_cover_all_seven_crystal_systems() {
    let root96 = 96.0_f64.sqrt();
    let systems = [
        ("cubic", [[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]),
        ("tetragonal", [[10., 0., 0.], [0., 10., 0.], [0., 0., 12.]]),
        (
            "orthorhombic",
            [[10., 0., 0.], [0., 11., 0.], [0., 0., 12.]],
        ),
        (
            "hexagonal",
            [[10., 0., 0.], [-5., 5. * 3.0_f64.sqrt(), 0.], [0., 0., 12.]],
        ),
        (
            "rhombohedral",
            [
                [10., 0., 0.],
                [2., root96, 0.],
                [2., 16. / root96, (96.0_f64 - 256. / 96.).sqrt()],
            ],
        ),
        ("monoclinic", [[10., 0., 0.], [0., 11., 0.], [2., 0., 12.]]),
        (
            "triclinic",
            [[10.3, 0., 0.], [0.8, 11.1, 0.], [0.4, 0.7, 10.9]],
        ),
    ];
    let numbers = [6, 8];
    let points = vec![Vec3::new(0.2, 0.1, 0.3), Vec3::new(3.1, 0.6, 0.2)];
    let u = vec![0.3, -0.2, 0.1, -0.3, 0.2, -0.1];
    let v = vec![-0.1, 0.3, 0.2, 0.1, -0.3, -0.2];
    let steps = [0.04, 0.02, 0.01, 0.005];
    let vector_norm = |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>().sqrt();
    let bound = steps[0]
        * (vector_norm(&u[..3])
            + vector_norm(&u[3..])
            + vector_norm(&v[..3])
            + vector_norm(&v[3..]));
    for (name, rows) in systems {
        let lattice = Lattice::new(
            Vec3::new(rows[0][0], rows[0][1], rows[0][2]),
            Vec3::new(rows[1][0], rows[1][1], rows[1][2]),
            Vec3::new(rows[2][0], rows[2][1], rows[2][2]),
        );
        let (cn, cn_margin) = cutoff_in_image_gap(&lattice, &points, 6., 9.);
        let (disp, disp_margin) = cutoff_in_image_gap(&lattice, &points, 11., 14.);
        assert!(
            cn_margin > bound && disp_margin > bound,
            "{name}: stencil crosses an image cutoff"
        );
        let model = PeriodicDispersion::from_model(&Dispersion::D3Bj, "pbe", cn, disp)
            .unwrap()
            .unwrap();
        let hessian = |a: f64, b: f64| {
            let moved: Vec<_> = points
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let x: [f64; 3] =
                        std::array::from_fn(|k| r.get(k) + a * u[3 * i + k] + b * v[3 * i + k]);
                    Vec3::new(x[0], x[1], x[2])
                })
                .collect();
            periodic_dispersion_hessian(&numbers, &moved, &lattice, &model).unwrap()
        };
        for order in [3, 4] {
            let probes = if order == 3 {
                vec![u.clone()]
            } else {
                vec![u.clone(), v.clone()]
            };
            let exact = periodic_dispersion_force_constants_along(
                &numbers, &points, &lattice, &model, &probes,
            )
            .unwrap();
            let differences: Vec<Vec<f64>> = steps
                .iter()
                .map(|&h| {
                    if order == 3 {
                        let (a, b) = (hessian(h, 0.), hessian(-h, 0.));
                        a.as_slice()
                            .iter()
                            .zip(b.as_slice())
                            .map(|(a, b)| (a - b) / (2. * h))
                            .collect()
                    } else {
                        let c = [
                            hessian(h, h),
                            hessian(h, -h),
                            hessian(-h, h),
                            hessian(-h, -h),
                        ];
                        (0..36)
                            .map(|i| {
                                (c[0].as_slice()[i] - c[1].as_slice()[i] - c[2].as_slice()[i]
                                    + c[3].as_slice()[i])
                                    / (4. * h * h)
                            })
                            .collect()
                    }
                })
                .collect();
            let errors: Vec<_> = differences
                .iter()
                .map(|d| largest(d, exact.matrix.as_slice()))
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich: Vec<_> = differences[3]
                .iter()
                .zip(&differences[2])
                .map(|(a, b)| (4. * a - b) / 3.)
                .collect();
            let richardson = largest(&rich, exact.matrix.as_slice());
            println!("crystal {name} order {order}: cn={cn:.12e} disp={disp:.12e} margins=[{cn_margin:.12e},{disp_margin:.12e}] bound={bound:.12e} steps={steps:?} errors={errors:?} p={:.12} R2={:.12} Richardson={richardson:.12e} acoustic={:.12e}",fit.exponent,fit.r_squared,exact.acoustic_residual);
            assert!(
                (fit.exponent - 2.).abs() < 0.15 && fit.r_squared > 0.999,
                "{name} order {order}"
            );
            assert!(
                richardson < 1e-8 && exact.acoustic_residual < 1e-11,
                "{name} order {order}"
            );
        }
    }
}
