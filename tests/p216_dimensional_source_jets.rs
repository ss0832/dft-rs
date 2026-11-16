// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent moving-source forces, closed zero modes and four-order fits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::math::Vec3;
use dft_rs::periodic::electrostatics::{DimensionalCoulomb, ElectrostaticSource};
use dft_rs::periodic::ewald::EwaldParams;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::{Lattice, PeriodicGeometry};

fn green(d: usize) -> DimensionalCoulomb {
    let geometry = PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.1, 5.4, 0.0),
            Vec3::new(0.3, 0.4, 6.1),
        ),
        PeriodicBoundary::new(std::array::from_fn(|a| a < d)),
    )
    .unwrap();
    DimensionalCoulomb::new(
        geometry,
        EwaldParams {
            alpha: 0.55,
            rcut: 16.0,
            gcut: 8.0,
        },
        128,
    )
    .unwrap()
}
fn positions(sources: &[ElectrostaticSource], direction: &[[f64; 3]], t: f64) -> Vec<[f64; 3]> {
    sources
        .iter()
        .zip(direction)
        .map(|(s, d)| std::array::from_fn(|a| s.position.get(a) + t * d[a]))
        .collect()
}
fn value_and_force(
    g: &DimensionalCoulomb,
    sources: &[ElectrostaticSource],
    d: &[[f64; 3]],
    t: f64,
) -> (f64, f64) {
    let p = positions(sources, d, t);
    let e = g
        .energy_components_scalar(sources, &p, |x| {
            let (s, c) = x.sin_cos();
            (c, s)
        })
        .unwrap()
        .total();
    // Independently contract the explicit analytic first-position gradient.
    let mut slope = 0.0;
    for i in 0..p.len() {
        for j in 0..i {
            let r = Vec3::new(p[i][0] - p[j][0], p[i][1] - p[j][1], p[i][2] - p[j][2]);
            let (_, gradient) = g
                .pair(r, sources[i].width.hypot(sources[j].width), false)
                .unwrap();
            slope += sources[i].charge
                * sources[j].charge
                * (0..3)
                    .map(|a| gradient.get(a) * (d[i][a] - d[j][a]))
                    .sum::<f64>();
        }
    }
    (e, slope)
}
#[test]
fn all_dimensions_mixed_point_gaussian_and_fixed_countercharge_have_four_order_fits() {
    let directions = [[0.1, -0.05, 0.3], [-0.06, 0.04, -0.15], [0.0; 3]];
    let steps = [0.08, 0.04, 0.02, 0.01];
    for d in 1..=3 {
        for finite in [false, true] {
            let sources = [
                ElectrostaticSource {
                    position: Vec3::new(-0.4, 0.2, 0.37),
                    charge: 1.0,
                    width: if finite { 0.27 } else { 0.0 },
                },
                ElectrostaticSource {
                    position: Vec3::new(1.3, -0.7, 0.9),
                    charge: -0.3,
                    width: 0.32,
                },
                ElectrostaticSource {
                    position: Vec3::new(0.3, 0.6, -1.2),
                    charge: -0.7,
                    width: 0.8,
                },
            ];
            let g = green(d);
            let p: Vec<[Taylor<5>; 3]> = sources
                .iter()
                .zip(directions)
                .map(|(s, v)| {
                    std::array::from_fn(|a| Taylor::from_coefficients(&[s.position.get(a), v[a]]))
                })
                .collect();
            let coefficients = g
                .energy_components_scalar(&sources, &p, |x| {
                    let (s, c) = x.sin_cos();
                    (c, s)
                })
                .unwrap()
                .total()
                .c;
            let base = value_and_force(&g, &sources, &directions, 0.0);
            assert!((base.0 - coefficients[0]).abs() < 1e-12);
            assert!((base.1 - coefficients[1]).abs() < 1e-12);
            let mut estimates = Vec::new();
            for h in steps {
                let (p, m, pp, mm) = (
                    value_and_force(&g, &sources, &directions, h),
                    value_and_force(&g, &sources, &directions, -h),
                    value_and_force(&g, &sources, &directions, 2.0 * h),
                    value_and_force(&g, &sources, &directions, -2.0 * h),
                );
                estimates.push([
                    (p.0 - m.0) / (2.0 * h),
                    (p.1 - m.1) / (2.0 * h),
                    (p.1 - 2.0 * base.1 + m.1) / (h * h),
                    (pp.1 - 2.0 * p.1 + 2.0 * m.1 - mm.1) / (2.0 * h * h * h),
                ]);
            }
            for n in 1..=4 {
                let target = coefficients[n] * (1..=n).product::<usize>() as f64;
                let errors: Vec<_> = estimates
                    .iter()
                    .map(|a| (a[n - 1] - target).abs())
                    .collect();
                let fit = convergence::error_power_fit(&steps, &errors);
                let rich = ((4.0 * estimates[3][n - 1] - estimates[2][n - 1]) / 3.0 - target).abs();
                println!("dimensional source d={d} gaussian={finite} order={n} steps={steps:?} target={target:.16e} errors={errors:?} p={:.9} R2={:.12} log_rmse={:.12e} Richardson={rich:.16e}",fit.exponent,fit.r_squared,fit.log_rmse);
                assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 2e-6);
            }
        }
    }
}

#[test]
fn coincident_gaussian_zero_modes_match_closed_quartic_coefficients() {
    let (a, b) = (0.4_f64, 0.17_f64);
    for d in [1, 2] {
        let g = green(d);
        let sources = [
            ElectrostaticSource {
                position: Vec3::zero(),
                charge: 1.0,
                width: 0.3,
            },
            ElectrostaticSource {
                position: Vec3::zero(),
                charge: -1.0,
                width: 0.3,
            },
        ];
        let zero = Taylor::<5>::from_f64(0.0);
        let p = [
            [zero, zero, Taylor::from_coefficients(&[0.0, a, b])],
            [zero; 3],
        ];
        let result = g
            .energy_components_scalar(&sources, &p, |x| {
                let (s, c) = x.sin_cos();
                (c, s)
            })
            .unwrap();
        let measure = g.geometry().measure();
        let (quadratic, quartic) = if d == 1 {
            (
                0.55_f64.powi(2) / measure,
                -0.55_f64.powi(4) / (4.0 * measure),
            )
        } else {
            (
                2.0 * std::f64::consts::PI.sqrt() * 0.55 / measure,
                -std::f64::consts::PI.sqrt() * 0.55_f64.powi(3) / (3.0 * measure),
            )
        };
        let expected = [
            0.0,
            0.0,
            quadratic * a * a,
            2.0 * quadratic * a * b,
            quadratic * b * b + quartic * a.powi(4),
        ];
        for (n, value) in expected.into_iter().enumerate() {
            assert!(
                (result.reciprocal_zero.c[n] - value).abs() < 1e-12,
                "d={d} order={n}"
            );
        }
        assert!(result.total().c.iter().all(|v| v.is_finite()));
    }
}
