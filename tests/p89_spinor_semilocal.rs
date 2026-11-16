// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Generic spin projection: independent limits, gauge checks and error fits.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::periodic::spinor_xc::*;
use dft_rs::xc::{resolve, Family, FunctionalSpec, GridVars, XcOut};

fn point() -> SpinorXcPoint {
    SpinorXcPoint {
        density: [0.9, 0.1, 0.12, 0.2],
        gradient: [
            [0.1, 0.02, -0.04],
            [0.01, 0.02, 0.005],
            [0.001, -0.012, 0.002],
            [0.03, 0.01, 0.005],
        ],
        tau: [0.9, 0.02, 0.03, 0.04],
        current: [
            [0.02, -0.01, 0.03],
            [0.005, 0.001, 0.002],
            [-0.003, 0.004, 0.001],
            [0.002, 0.003, -0.005],
        ],
    }
}
fn transform(p: &SpinorXcPoint, boost: [f64; 3]) -> SpinorXcPoint {
    let mut out = p.clone();
    let square = boost.iter().map(|v| v * v).sum::<f64>();
    for a in 0..4 {
        out.tau[a] +=
            (0..3).map(|d| boost[d] * p.current[a][d]).sum::<f64>() + 0.5 * square * p.density[a];
        for d in 0..3 {
            out.current[a][d] += boost[d] * p.density[a];
        }
    }
    out
}

#[test]
fn m06_iso_orbital_gradient_matches_the_physical_one_sided_limit() {
    let functional = resolve(&FunctionalSpec::Combined("m06l".into())).unwrap();
    let mut point = [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.0, 0.0];
    point[5] = point[2] / (8.0 * point[0]);
    point[6] = point[4] / (8.0 * point[1]);
    let eval = |point: [f64; 7], order| {
        let x = point.map(|v| [v]);
        let vars = GridVars {
            rho: [&x[0], &x[1]],
            sigma: Some([&x[2], &x[3], &x[4]]),
            tau: Some([&x[5], &x[6]]),
        };
        let mut result = XcOut::default();
        functional.eval(&vars, order, &mut result).unwrap();
        result
    };
    let reference = eval(point, 1);
    let energy = reference.e[0];
    let analytic = [
        reference.vrho[0][0],
        reference.vrho[1][0],
        reference.vsigma[0][0],
        reference.vsigma[1][0],
        reference.vsigma[2][0],
        reference.vtau[0][0],
        reference.vtau[1][0],
    ];
    let names = ["rho a", "rho b", "sigma aa", "sigma bb", "tau a", "tau b"].map(str::to_string);
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.008, 0.004, 0.002, 0.001]],
        |steps| {
            [0, 1, 2, 4, 5, 6]
                .into_iter()
                .map(|slot| {
                    // The boundary permits larger rho and tau and smaller
                    // like-spin gradient norms at fixed other descriptors.
                    let delta = if slot == 2 || slot == 4 {
                        -point[slot]
                    } else {
                        point[slot]
                    };
                    let displaced = |step| {
                        let mut trial = point;
                        trial[slot] += step * delta;
                        eval(trial, 0).e[0] - energy
                    };
                    steps.map(|h| {
                        let numerical = (4.0 * displaced(h) - displaced(2.0 * h)) / (2.0 * h);
                        (numerical - analytic[slot] * delta).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("one-sided M06 iso-orbital gradient", &failures);
    // The functional is independent of the cross-spin gradient contraction.
    assert_eq!(analytic[3], 0.0);
    point[3] *= 1.008;
    assert_eq!(eval(point, 0).e[0], energy);
}

#[test]
fn all_semilocal_rungs_have_independent_four_step_variational_fits() {
    let direction: [f64; 32] = std::array::from_fn(|i| 0.1 * ((i + 1) as f64 * 0.37).sin());
    let reference = point();
    let values = reference.descriptors();
    let steps = [0.04, 0.02, 0.01, 0.005];
    for name in ["svwn5", "pbe", "tpss", "r2scan", "m06l"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let production = spinor_semilocal_point(&functional, &reference, 16).unwrap();
        let analytic = production
            .gradient
            .iter()
            .zip(direction)
            .map(|(v, d)| v * d)
            .sum::<f64>();
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let displaced = |sign: f64| {
                SpinorXcPoint::from_descriptors(std::array::from_fn(|i| {
                    values[i] + sign * h * direction[i]
                }))
            };
            let plus = spinor_semilocal_point(&functional, &displaced(1.0), 16)
                .unwrap()
                .energy;
            let minus = spinor_semilocal_point(&functional, &displaced(-1.0), 16)
                .unwrap()
                .energy;
            let estimate = (plus - minus) / (2.0 * h);
            estimates.push(estimate);
            errors.push((estimate - analytic).abs());
        }
        let f = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - analytic).abs();
        println!("global spin {name}: steps={steps:?} raw-errors={errors:?} p={:.9} R2={:.10} Richardson={rich:.3e}",f.exponent,f.r_squared);
        assert!((f.exponent - 2.0).abs() < 0.02 && f.r_squared > 0.99999);
        assert!(rich < 1e-8);
        assert!(
            (production.energy - production.component_energies.iter().sum::<f64>()).abs() < 1e-12
        );
        let boost = [0.21, -0.13, 0.17];
        let moved = transform(&reference, boost);
        let gauge = spinor_semilocal_point(&functional, &moved, 16).unwrap();
        let delta = SpinorXcPoint::from_descriptors(direction);
        let moved_direction = transform(&delta, boost).descriptors();
        let gauge_derivative = gauge
            .gradient
            .iter()
            .zip(moved_direction)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        println!(
            "global spin {name}: gauge-energy={:.3e} gauge-derivative={:.3e}",
            (gauge.energy - production.energy).abs(),
            (gauge_derivative - analytic).abs()
        );
        assert!((gauge.energy - production.energy).abs() < 1e-12);
        assert!((gauge_derivative - analytic).abs() < 1e-10);
    }
}

#[test]
fn collinear_limit_matches_native_energy_and_weighted_terms_on_four_angular_rules() {
    let p = SpinorXcPoint {
        density: [0.9, 0.0, 0.0, 0.3],
        gradient: [[0.1, 0.02, -0.04], [0.0; 3], [0.0; 3], [0.03, 0.01, 0.005]],
        tau: [0.9, 0.0, 0.0, 0.04],
        current: [[0.0; 3]; 4],
    };
    let rho = [vec![0.6], vec![0.3]];
    let g: [[f64; 3]; 2] = std::array::from_fn(|a| {
        std::array::from_fn(|d| {
            0.5 * (p.gradient[0][d]
                + if a == 0 {
                    p.gradient[3][d]
                } else {
                    -p.gradient[3][d]
                })
        })
    });
    let sigma = [
        vec![g[0].iter().map(|v| v * v).sum::<f64>()],
        vec![(0..3).map(|d| g[0][d] * g[1][d]).sum::<f64>()],
        vec![g[1].iter().map(|v| v * v).sum::<f64>()],
    ];
    let tau = [vec![0.47], vec![0.43]];
    for name in ["svwn5", "pbe", "tpss", "r2scan", "m06l"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let vars = GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: (functional.family != Family::Lda).then_some([&sigma[0], &sigma[1], &sigma[2]]),
            tau: (functional.family == Family::MetaGga).then_some([&tau[0], &tau[1]]),
        };
        let mut out = XcOut::default();
        functional.eval(&vars, 1, &mut out).unwrap();
        let mut parts = Vec::new();
        for (weight, part) in &functional.parts {
            let mut o = XcOut::default();
            o.prepare(1, 0, functional.family);
            part.eval_add(&vars, 0, *weight, &mut o).unwrap();
            parts.push(o.e[0]);
        }
        let mut defects = Vec::new();
        for order in [4, 8, 16, 32] {
            let result = spinor_semilocal_point(&functional, &p, order).unwrap();
            defects.push((result.energy - out.e[0]).abs());
            println!(
                "global spin {name} angular-order={order}: collinear-energy={:.3e} \
                component-defects={:?}",
                defects.last().unwrap(),
                result
                    .component_energies
                    .iter()
                    .zip(&parts)
                    .map(|(a, b)| (a - b).abs())
                    .collect::<Vec<_>>()
            );
            if order == 32 {
                assert!(*defects.last().unwrap() < 1e-11);
                assert!(result
                    .component_energies
                    .iter()
                    .zip(&parts)
                    .all(|(a, b)| (a - b).abs() < 1e-11));
                assert!(
                    (result.gradient[0] - 0.5 * (out.vrho[0][0] + out.vrho[1][0])).abs() < 1e-10
                );
                assert!(
                    (result.gradient[3] - 0.5 * (out.vrho[0][0] - out.vrho[1][0])).abs() < 1e-10
                );
            }
        }
    }
}
