// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! rSCAN validation in two parts.
//!
//! rSCAN joins its degree-7 polynomial interpolation to the SCAN branch at
//! the regularized indicator 2.5 with a continuous value and first three
//! derivatives; the fourth derivative jumps (exchange -8.19 against +0.62,
//! correlation -18.76 against -0.05). As a field or a nucleus moves,
//! quadrature points cross the junction, so finite differences of molecular
//! energies, dipoles and polarizabilities do not follow a power law: for
//! water the fourth-order stencil of alpha_zy gave p = 2.95 (h = 0.001 to
//! 0.004), against 3.74 with the same code and the infinitely
//! differentiable SCAN interpolation, and removing the
//! `max(tau - tau_W, 0)` floor changed nothing.
//!
//! 1. The functional itself, at points on the polynomial branch close to
//!    the junction from below, on the SCAN branch close to it from above,
//!    inside the polynomial range and on the low and high SCAN branches:
//!    the first, second and third Taylor coefficients of the energy density
//!    along 7, 28 and 84 generic directions (as many as the independent
//!    components of the derivative tensors in the seven variables) against
//!    central differences of the energy, four-step fits of order 2, with
//!    every step kept on the branch of its point.
//! 2. The molecular response code on the same functional with the
//!    switching interval set to `[1, 1]` (`xc_parameters`), which replaces
//!    the polynomial by the SCAN interpolation and keeps every other term:
//!    every component of the gradient, Hessian, polarizability and first
//!    hyperpolarizability (`support/molecular_fits.rs`). The response code
//!    does not depend on the functional's name, so the remaining
//!    difference to rSCAN is the polynomial, validated in part 1. The
//!    measured smooth-interpolation run failed 38 components (34 Hessian,
//!    four alpha/beta); it is not recorded as complete acceptance.
//!
//! Independent test-only branch continuation in xc::branch_freeze passes
//! all 126 molecular components: p=1.918816..2.051137, minimum R2=0.99964605,
//! nuclear h=[0.02,0.01,0.005,0.0025] Bohr and electric
//! h=[0.016,0.008,0.004,0.002] atomic units. No failed step is discarded.
//! Production branch switching is unchanged; finite-step crossings remain
//! distinct from derivatives of the local branch at the reference point.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/molecular_fits.rs"]
mod molecular_fits;
#[path = "support/stencils.rs"]
mod stencils;

use components::{assert_no_failures, fit_components};
use dft_rs::driver::{run, Controls, Value};
use dft_rs::xc::taylor::XcTaylorInput;
use dft_rs::xc::{resolve, FunctionalSpec, GridVars, XcOut};
use stencils::{fit_components_staged, Stencil};

const K: f64 = 4.557_799_872_345_597;
const TAU_R: f64 = 1e-4;
const ALPHA_R: f64 = 1e-3;
const JUNCTION: f64 = 2.5;

/// Regularized indicator `alpha0^3 / (alpha0^2 + alpha_r)`.
fn regularized(alpha0: f64) -> f64 {
    alpha0.powi(3) / (alpha0 * alpha0 + ALPHA_R)
}

/// The three indicators of a point `[rho_a, rho_b, sigma_aa, sigma_ab,
/// sigma_bb, tau_a, tau_b]`: exchange of each spin and correlation.
fn indicators(v: &[f64; 7]) -> [f64; 3] {
    let x = |rho: f64, sig: f64, tau: f64| {
        regularized((tau - sig / (8.0 * rho)) / (K * rho.powf(5.0 / 3.0) + 0.5 * TAU_R))
    };
    let rho = v[0] + v[1];
    let z = (v[0] - v[1]) / rho;
    let ds = ((1.0 + z).powf(5.0 / 3.0) + (1.0 - z).powf(5.0 / 3.0)) / 2.0f64.powf(5.0 / 3.0);
    let sig = v[2] + 2.0 * v[3] + v[4];
    let c = regularized(
        (v[5] + v[6] - sig / (8.0 * rho))
            / ((K * rho.powf(5.0 / 3.0) + 2.0f64.powf(2.0 / 3.0) * TAU_R) * ds),
    );
    [x(v[0], v[2], v[5]), x(v[1], v[4], v[6]), c]
}

/// Point with kinetic energy densities `tau_s = tau_W,s + u_s K rho_s^(5/3)`.
fn point(u: [f64; 2]) -> [f64; 7] {
    let (ra, rb, saa, sab, sbb) = (0.31, 0.22, 0.041, 0.017, 0.029);
    [
        ra,
        rb,
        saa,
        sab,
        sbb,
        saa / (8.0 * ra) + u[0] * K * ra.powf(5.0 / 3.0),
        sbb / (8.0 * rb) + u[1] * K * rb.powf(5.0 / 3.0),
    ]
}

/// Bisection for `u_a` (with `u_b` fixed) or for a common scale of both
/// (`u_b / u_a` from `ub` and further ratios), so that indicator `which`
/// reaches `target` while the other indicators stay at least 0.4 from the
/// junction.
fn place(which: usize, target: f64, ub: f64) -> [f64; 7] {
    for ratio in [ub, 0.4, 0.3, 0.2, 1.5, 2.0, 3.0] {
        let v = place_once(which, target, ratio);
        let all = indicators(&v);
        if (0..3).all(|k| k == which || (all[k] - JUNCTION).abs() > 0.4) {
            return v;
        }
    }
    panic!("no placement of indicator {which} at {target} away from the junction");
}

fn place_once(which: usize, target: f64, ub: f64) -> [f64; 7] {
    let value = |s: f64| {
        let u = if which == 2 { [s, ub * s] } else { [s, ub] };
        indicators(&point(u))[which]
    };
    let (mut lo, mut hi) = (1e-3, 20.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if value(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let s = 0.5 * (lo + hi);
    point(if which == 2 { [s, ub * s] } else { [s, ub] })
}

/// Deterministic generic directions, each variable scaled to one percent
/// of its value.
fn directions(base: &[f64; 7], count: usize, seed: u64) -> Vec<[f64; 7]> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..count)
        .map(|_| std::array::from_fn(|i| 0.01 * base[i] * next()))
        .collect()
}

fn energy(functional: &dft_rs::ResolvedFunctional, v: &[f64; 7]) -> f64 {
    let (ra, rb) = ([v[0]], [v[1]]);
    let (saa, sab, sbb) = ([v[2]], [v[3]], [v[4]]);
    let (ta, tb) = ([v[5]], [v[6]]);
    let vars = GridVars {
        rho: [&ra, &rb],
        sigma: Some([&saa, &sab, &sbb]),
        tau: Some([&ta, &tb]),
    };
    let mut out = XcOut::default();
    functional.eval(&vars, 0, &mut out).unwrap();
    out.e[0]
}

#[test]
fn rscan_branches_pass_four_step_fits_of_every_derivative_component() {
    let functional = resolve(&FunctionalSpec::Combined("rscan".into())).unwrap();
    // (label, indicator to place, target, other spin setting)
    let cases = [
        ("exchange below the junction", 0, 2.2, 1.0),
        ("exchange above the junction", 0, 2.9, 1.0),
        ("correlation below the junction", 2, 2.2, 0.6),
        ("correlation above the junction", 2, 2.9, 0.6),
        ("polynomial range", 0, 1.2, 1.4),
        ("high SCAN branch", 0, 4.0, 3.6),
        ("low indicator", 0, 0.3, 0.5),
    ];
    let mut failures = Vec::new();
    for (label, which, target, other) in cases {
        let base = place(which, target, other);
        let reference = indicators(&base);
        println!("{label}: indicators {reference:?}");
        assert!((reference[which] - target).abs() < 1e-9);
        for (order, count) in [(1usize, 7usize), (2, 28), (3, 84)] {
            let dirs = directions(&base, count, 17 + order as u64);
            // Every evaluation of the fits stays on the branches of the
            // point: the largest step multiple `reach` that keeps every
            // direction on its branches (bisection), and the intervals are
            // scaled into it. The steps keep the energy differences above the
            // round-off of the energy: with steps of 0.025 the second-order
            // errors stopped at 1e-13, and the coarsest interval is for the
            // directions whose coefficients are small.
            let stays = |t: f64| {
                dirs.iter().all(|d| {
                    [-t, t].iter().all(|&s| {
                        let v: [f64; 7] = std::array::from_fn(|i| base[i] + s * d[i]);
                        v[3] * v[3] < v[2] * v[4]
                            && indicators(&v)
                                .iter()
                                .zip(&reference)
                                .all(|(a, b)| (*a > JUNCTION) == (*b > JUNCTION))
                    })
                })
            };
            let (mut lo, mut hi) = (0.0, 8.0);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if stays(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let reach = 0.95 * lo;
            // The stencil of order 3 reaches twice its step.
            let top = if order == 3 { 0.5 * reach } else { reach };
            let largest = top.min(if order == 3 { 3.2 } else { 1.6 });
            let base_top = if order == 3 { 1.6 } else { 0.8 };
            assert!(
                largest >= base_top,
                "{label}: order {order} branch-preserving steps end at {largest}"
            );
            // Base interval, a finer one against higher-order terms, the
            // widest branch-preserving one with ratio sqrt(2), whose finest
            // step stays an order above the round-off, and the fourth-order
            // stencil for the directions whose h^2 and h^4 terms cancel
            // inside the second-order intervals (exchange below the junction,
            // third order: errors 1.73e-11, 6.15e-13, 4.60e-13, 1.38e-13,
            // 3.13e-14 from 1.6 to 0.1).
            let r = std::f64::consts::SQRT_2;
            let fourth_top = (reach / if order == 3 { 3.0 } else { 2.0 }).min(base_top);
            let stages = vec![
                (
                    [base_top, base_top / 2.0, base_top / 4.0, base_top / 8.0],
                    Stencil::Second,
                ),
                (
                    [
                        base_top / 2.0,
                        base_top / 4.0,
                        base_top / 8.0,
                        base_top / 16.0,
                    ],
                    Stencil::Second,
                ),
                (
                    [largest, largest / r, largest / 2.0, largest / (2.0 * r)],
                    Stencil::Second,
                ),
                (
                    [
                        fourth_top,
                        fourth_top / r,
                        fourth_top / 2.0,
                        fourth_top / (2.0 * r),
                    ],
                    Stencil::Fourth,
                ),
            ];
            println!(
                "{label}: order {order} branch-preserving reach {reach:.3}, stages {stages:?}"
            );
            let names: Vec<String> = (0..count)
                .map(|k| format!("rscan {label} order {order} direction {k}"))
                .collect();
            let coefficients: Vec<f64> = dirs
                .iter()
                .map(|d| {
                    let input = XcTaylorInput::linear(base, *d, 3).unwrap();
                    functional.semilocal_taylor_energy(&input).unwrap()[order]
                })
                .collect();
            let errors_at = |steps: &[f64; 4], stencil: Stencil| -> Vec<[f64; 4]> {
                dirs.iter()
                    .zip(&coefficients)
                    .map(|(d, c)| {
                        let at = |t: f64| {
                            let v: [f64; 7] = std::array::from_fn(|i| base[i] + t * d[i]);
                            energy(&functional, &v)
                        };
                        std::array::from_fn(|s| {
                            let h = steps[s];
                            let f = |k: f64| at(k * h);
                            let estimate = match (order, stencil) {
                                (1, Stencil::Second) => (f(1.0) - f(-1.0)) / (2.0 * h),
                                (1, Stencil::Fourth) => {
                                    (-f(2.0) + 8.0 * f(1.0) - 8.0 * f(-1.0) + f(-2.0)) / (12.0 * h)
                                }
                                (2, Stencil::Second) => {
                                    (f(1.0) - 2.0 * f(0.0) + f(-1.0)) / (2.0 * h * h)
                                }
                                (2, Stencil::Fourth) => {
                                    (-f(2.0) + 16.0 * f(1.0) - 30.0 * f(0.0) + 16.0 * f(-1.0)
                                        - f(-2.0))
                                        / (24.0 * h * h)
                                }
                                (_, Stencil::Second) => {
                                    (f(2.0) - 2.0 * f(1.0) + 2.0 * f(-1.0) - f(-2.0))
                                        / (12.0 * h * h * h)
                                }
                                (_, Stencil::Fourth) => {
                                    (-f(3.0) + 8.0 * f(2.0) - 13.0 * f(1.0) + 13.0 * f(-1.0)
                                        - 8.0 * f(-2.0)
                                        + f(-3.0))
                                        / (48.0 * h * h * h)
                                }
                            };
                            (estimate - c).abs()
                        })
                    })
                    .collect()
            };
            if order == 3 {
                fit_components_staged(&names, &stages, errors_at, &mut failures);
            } else {
                // The first and second orders pass on the second-order
                // intervals; the fourth-order stage is not needed there.
                let intervals: Vec<[f64; 4]> = stages[..3].iter().map(|s| s.0).collect();
                fit_components(
                    &names,
                    &intervals,
                    |steps| errors_at(steps, Stencil::Second),
                    &mut failures,
                );
            }
        }
    }
    assert_no_failures("rSCAN branch derivatives", &failures);
}

fn smooth_controls(extra: &[(&str, Value)]) -> Controls {
    let mut all = vec![
        ("xc".to_string(), Value::Str("rscan".into())),
        (
            "xc_parameters".to_string(),
            Value::object()
                .with("switch_lower", 1.0)
                .with("switch_upper", 1.0),
        ),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("grid_level".to_string(), Value::Int(7)),
        ("conv_tol".to_string(), Value::Float(1e-13)),
        ("conv_tol_grad".to_string(), Value::Float(1e-11)),
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        ("dispersion".to_string(), Value::Str("none".into())),
    ];
    for (k, v) in extra {
        all.retain(|(key, _)| key != k);
        all.push((k.to_string(), v.clone()));
    }
    Controls::from_value(&Value::Object(all)).unwrap()
}

/// The switching interval reaches the functional: the published interval
/// leaves the energy unchanged bit for bit, the smooth interval changes it,
/// and a parameter no part has is refused.
#[test]
fn switching_interval_overrides_reach_the_functional() {
    let water = molecular_fits::system(&molecular_fits::WATER);
    let energy = |parameters: Option<Value>| {
        let mut controls = vec![
            ("xc".to_string(), Value::Str("rscan".into())),
            ("basis".to_string(), Value::Str("sto-3g".into())),
            ("conv_tol".to_string(), Value::Float(1e-12)),
        ];
        if let Some(p) = parameters {
            controls.push(("xc_parameters".to_string(), p));
        }
        run(
            "energy",
            &water,
            &Controls::from_value(&Value::Object(controls)).unwrap(),
        )
        .map(|r| r.get("energy").and_then(Value::as_f64).unwrap())
    };
    let published = energy(None).unwrap();
    let explicit = energy(Some(
        Value::object()
            .with("switch_lower", 0.0)
            .with("switch_upper", 2.5),
    ))
    .unwrap();
    let smooth = energy(Some(
        Value::object()
            .with("switch_lower", 1.0)
            .with("switch_upper", 1.0),
    ))
    .unwrap();
    println!("rSCAN {published:.12}, explicit interval {explicit:.12}, smooth {smooth:.12}");
    assert_eq!(published, explicit);
    assert!((smooth - published).abs() > 1e-6);
    assert!(energy(Some(Value::object().with("no_such_parameter", 1.0))).is_err());
}

#[test]
fn smooth_interpolation_gradient_and_hessian_pass_four_step_fits_in_every_component() {
    let mut failures = Vec::new();
    molecular_fits::gradient_and_hessian("rscan [1, 1]", &smooth_controls, &mut failures);
    assert_no_failures("gradient and Hessian", &failures);
}

#[test]
fn smooth_interpolation_polarizability_and_hyperpolarizability_pass_four_step_fits() {
    let mut failures = Vec::new();
    molecular_fits::polarizability_and_hyperpolarizability(
        "rscan [1, 1]",
        &smooth_controls,
        &mut failures,
    );
    assert_no_failures("polarizability and hyperpolarizability", &failures);
}
