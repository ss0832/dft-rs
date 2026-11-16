// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent periodic VV10 energy, derivative and dimensional identities.
#[path = "support/convergence.rs"]
mod convergence;
mod nonlocal_oracle {
    use super::PeriodicVv10Options;
    include!("support/nonlocal_oracle.rs");
}
use dft_rs::dual::Taylor;
use dft_rs::math::Vec3;
use dft_rs::nlc::Vv10Params;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::nonlocal::{vv10_fields, PeriodicVv10Options};
use dft_rs::periodic::{Lattice, PeriodicGeometry};
use nonlocal_oracle::energy as oracle;

const POINTS: [[f64; 3]; 3] = [
    [0.13, -0.21, 0.31],
    [1.17, 0.42, -0.53],
    [-0.72, 1.29, 0.84],
];
const RHO: [f64; 3] = [0.14, 0.21, 0.08];
const SIGMA: [f64; 3] = [0.013, 0.007, 0.019];
const WEIGHTS: [f64; 3] = [0.7, 1.1, 0.9];

fn rows() -> Vec<[[f64; 3]; 3]> {
    let a = 6.7_f64;
    let c = 0.2_f64;
    let s = (1.0 - c * c).sqrt();
    let y = (c - c * c) / s;
    vec![
        [[6.7, 0., 0.], [0., 6.7, 0.], [0., 0., 6.7]],
        [[6.7, 0., 0.], [0., 6.7, 0.], [0., 0., 8.1]],
        [[6.7, 0., 0.], [0., 7.3, 0.], [0., 0., 8.1]],
        [
            [a, 0., 0.],
            [-a / 2.0, a * 3.0_f64.sqrt() / 2.0, 0.],
            [0., 0., 8.1],
        ],
        [
            [a, 0., 0.],
            [a * c, a * s, 0.],
            [a * c, a * y, a * (1.0 - c * c - y * y).sqrt()],
        ],
        [[6.7, 0., 0.], [0., 7.3, 0.], [1.2, 0., 8.1]],
        [[6.7, 0., 0.], [1.1, 7.3, 0.], [-0.8, 0.6, 8.1]],
    ]
}
fn geometry(rows: [[f64; 3]; 3], axes: [bool; 3]) -> PeriodicGeometry {
    PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(rows[0][0], rows[0][1], rows[0][2]),
            Vec3::new(rows[1][0], rows[1][1], rows[1][2]),
            Vec3::new(rows[2][0], rows[2][1], rows[2][2]),
        ),
        PeriodicBoundary::new(axes),
    )
    .unwrap()
}

#[test]
fn nonlocal_lattice_energy_matches_independent_oracle_for_all_crystal_systems_and_dimensions() {
    let settings = PeriodicVv10Options {
        parameters: Vv10Params::ORIGINAL,
        cutoff: 13.7,
    };
    let points: Vec<_> = POINTS.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    for (system, rows) in rows().into_iter().enumerate() {
        for axes in [[true, false, false], [true, true, false], [true; 3]] {
            let g = geometry(rows, axes);
            let value =
                vv10_fields(&g, &points, &POINTS, rows, &WEIGHTS, &RHO, &SIGMA, settings).unwrap();
            let expected = oracle(rows, axes, &POINTS, &RHO, &SIGMA, &WEIGHTS, settings);
            println!("periodic VV10 crystal={system} axes={axes:?} energy={:.15e} oracle_error={:.12e} pairs={}",value.energy,(value.energy-expected).abs(),value.pair_images);
            assert!((value.energy - expected).abs() < 1e-13);
            assert!(value.pair_images > 9);
        }
    }
}

#[test]
fn nonlocal_mixed_density_geometry_weight_derivatives_have_four_step_fits() {
    let rows = rows()[6];
    let axes = [true; 3];
    let g = geometry(rows, axes);
    let points: Vec<_> = POINTS.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    let settings = PeriodicVv10Options {
        parameters: Vv10Params::WB97XV,
        cutoff: 13.7,
    };
    let seed = |v: f64, d: f64| Taylor::<3>::from_coefficients(&[v, d, 0.0]);
    let lattice = std::array::from_fn(|a| {
        std::array::from_fn(|b| seed(rows[a][b], 0.01 * (a + 2 * b + 1) as f64))
    });
    let positions: Vec<_> = POINTS
        .iter()
        .enumerate()
        .map(|(i, p)| std::array::from_fn(|a| seed(p[a], 0.03 * (i + a + 1) as f64)))
        .collect();
    let rho: Vec<_> = RHO
        .iter()
        .enumerate()
        .map(|(i, r)| seed(*r, 0.01 * (i + 1) as f64))
        .collect();
    let sigma: Vec<_> = SIGMA
        .iter()
        .enumerate()
        .map(|(i, r)| seed(*r, 0.002 * (i + 1) as f64))
        .collect();
    let weights: Vec<_> = WEIGHTS
        .iter()
        .enumerate()
        .map(|(i, w)| seed(*w, 0.02 * (i + 1) as f64))
        .collect();
    let exact = vv10_fields(
        &g, &points, &positions, lattice, &weights, &rho, &sigma, settings,
    )
    .unwrap();
    let at = |t: f64| {
        let eval = |v: Taylor<3>| v.c[0] + t * v.c[1];
        let rows = lattice.map(|r| r.map(eval));
        let positions: Vec<_> = positions.iter().map(|p| p.map(eval)).collect();
        oracle(
            rows,
            axes,
            &positions,
            &rho.iter().map(|r| eval(*r)).collect::<Vec<_>>(),
            &sigma.iter().map(|s| eval(*s)).collect::<Vec<_>>(),
            &weights.iter().map(|w| eval(*w)).collect::<Vec<_>>(),
            settings,
        )
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    for order in 1..=2 {
        let value = exact.energy.c[order] * (if order == 2 { 2.0 } else { 1.0 });
        let estimates: Vec<_> = steps
            .iter()
            .map(|h| {
                if order == 1 {
                    (at(*h) - at(-*h)) / (2.0 * h)
                } else {
                    (at(*h) - 2.0 * at(0.0) + at(-*h)) / (h * h)
                }
            })
            .collect();
        let errors: Vec<_> = estimates.iter().map(|e| (e - value).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - value).abs();
        println!("periodic VV10 mixed order={order} value={value:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
        assert!(rich < 1e-10);
    }
}

#[test]
fn nonlocal_vacuum_switch_density_derivatives_have_independent_four_step_fits() {
    let rows = rows()[6];
    let g = geometry(rows, [true; 3]);
    let point = Vec3::zero();
    let settings = PeriodicVv10Options {
        cutoff: 13.7,
        ..Default::default()
    };
    let rho = Taylor::<3>::from_coefficients(&[1.5e-8, 0.2e-8, 0.0]);
    let constant = |v| Taylor::<3>::from_coefficients(&[v]);
    let exact = vv10_fields(
        &g,
        &[point],
        &[[constant(0.0); 3]],
        rows.map(|r| r.map(constant)),
        &[constant(0.9)],
        &[rho],
        &[constant(1e-19)],
        settings,
    )
    .unwrap();
    let at = |t: f64| {
        oracle(
            rows,
            [true; 3],
            &[[0.0; 3]],
            &[rho.c[0] + t * rho.c[1]],
            &[1e-19],
            &[0.9],
            settings,
        )
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    for order in 1..=2 {
        let value = exact.energy.c[order] * if order == 2 { 2.0 } else { 1.0 };
        let estimates: Vec<_> = steps
            .iter()
            .map(|h| {
                if order == 1 {
                    (at(*h) - at(-*h)) / (2.0 * h)
                } else {
                    (at(*h) - 2.0 * at(0.0) + at(-*h)) / (h * h)
                }
            })
            .collect();
        let errors: Vec<_> = estimates.iter().map(|v| (v - value).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - value).abs();
        println!("periodic VV10 smooth vacuum order={order} value={value:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
        assert!(rich < 1e-16);
    }
}
