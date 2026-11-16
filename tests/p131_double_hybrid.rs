// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent spin-orbital determinant sums and finite-path fits of the
//! generic PT2 amplitude/orbital contractions, including third/fourth order.

#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::properties::pt2::{from_ao, OrbitalSpace};
use dft_rs::xc::double_hybrid::Pt2Coefficients;

const COEFFICIENTS: Pt2Coefficients = Pt2Coefficients {
    same_spin: 0.31,
    opposite_spin: 0.47,
};

fn provider<S: Scalar>(t: S) -> (Vec<S>, [OrbitalSpace<S>; 2]) {
    let n = 4;
    let mut eri = vec![S::from_f64(0.0); n * n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let factor = |p: usize, a: usize, b: usize| {
                        (((a + b + 2) * (p + 1) + (a * b + 1)) as f64).sin() * 0.13
                    };
                    let value: f64 = (0..3).map(|p| factor(p, i, j) * factor(p, k, l)).sum();
                    eri[((i * n + j) * n + k) * n + l] =
                        S::from_f64(value) * (S::from_f64(1.0) + t.scale(0.2) + t * t.scale(0.1));
                }
            }
        }
    }
    let orbitals = std::array::from_fn(|spin| {
        let energies = if spin == 0 {
            [-0.8, -0.4, 0.2, 0.6]
        } else {
            [-0.7, 0.1, 0.5, 0.9]
        };
        let mut c = vec![S::from_f64(0.0); n * n];
        for i in 0..n {
            c[i * n + i] = S::from_f64(1.0);
        }
        let x = t.scale(if spin == 0 { 0.7 } else { -0.4 }) + S::from_f64(0.13);
        let den = S::from_f64(1.0) + x * x;
        let cos = (S::from_f64(1.0) - x * x) / den;
        let sin = x.scale(2.0) / den;
        let (i, a) = if spin == 0 { (1, 2) } else { (0, 3) };
        c[i * n + i] = cos;
        c[a * n + a] = cos;
        c[i * n + a] = -sin;
        c[a * n + i] = sin;
        OrbitalSpace {
            occupied: if spin == 0 { 2 } else { 1 },
            coefficients: c,
            energies: energies
                .into_iter()
                .enumerate()
                .map(|(i, e)| S::from_f64(e) + t.scale(0.03 * (i + 1) as f64))
                .collect(),
        }
    });
    (eri, orbitals)
}

/// Unordered spin-orbital double excitations, with independently transformed
/// Slater-Condon couplings. This uses neither production OVOV blocks nor its
/// ordered-pair multiplicities.
fn determinant_oracle(t: f64) -> [f64; 2] {
    let (eri, o) = provider(t);
    let occupied: Vec<_> = (0..2)
        .flat_map(|s| (0..o[s].occupied).map(move |i| (s, i)))
        .collect();
    let virtuals: Vec<_> = (0..2)
        .flat_map(|s| (o[s].occupied..4).map(move |i| (s, i)))
        .collect();
    let chemist = |p: (usize, usize), r: (usize, usize), q: (usize, usize), s: (usize, usize)| {
        if p.0 != r.0 || q.0 != s.0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        sum += o[p.0].coefficients[4 * i + p.1]
                            * o[r.0].coefficients[4 * j + r.1]
                            * o[q.0].coefficients[4 * k + q.1]
                            * o[s.0].coefficients[4 * l + s.1]
                            * eri[((i * 4 + j) * 4 + k) * 4 + l];
                    }
                }
            }
        }
        sum
    };
    let mut energy = [0.0; 2];
    for i in 0..occupied.len() {
        for j in i + 1..occupied.len() {
            let (p, q) = (occupied[i], occupied[j]);
            for a in 0..virtuals.len() {
                for b in a + 1..virtuals.len() {
                    let (r, s) = (virtuals[a], virtuals[b]);
                    let coupling = chemist(p, r, q, s) - chemist(p, s, q, r);
                    let delta = o[p.0].energies[p.1] + o[q.0].energies[q.1]
                        - o[r.0].energies[r.1]
                        - o[s.0].energies[s.1];
                    energy[usize::from(p.0 != q.0)] += coupling * coupling / delta;
                }
            }
        }
    }
    energy
}

#[test]
fn spin_components_and_stationary_amplitude_functional_match_determinant_oracle() {
    let (eri, o) = provider(0.0);
    let problem = from_ao(4, &eri, &o).unwrap();
    let solution = problem.solve(COEFFICIENTS).unwrap();
    let oracle = determinant_oracle(0.0);
    let ss = (solution.same_spin_alpha + solution.same_spin_beta - oracle[0]).abs();
    let os = (solution.opposite_spin - oracle[1]).abs();
    let lagrangian = (problem
        .lagrangian(&solution.amplitudes, COEFFICIENTS)
        .unwrap()
        - solution.correlation)
        .abs();
    println!("independent SS defect={ss:.12e} OS defect={os:.12e} Lagrangian={lagrangian:.12e} raw amplitude={:.12e}",solution.amplitude_residual);
    assert!(ss < 1e-14 && os < 1e-14 && lagrangian < 1e-14 && solution.amplitude_residual < 1e-14);
    assert!(solution.same_spin_alpha < 0.0 && solution.opposite_spin < 0.0);
    let zero = problem
        .solve(Pt2Coefficients {
            same_spin: 0.0,
            opposite_spin: 0.0,
        })
        .unwrap();
    assert_eq!(zero.correlation, 0.0);
}

#[test]
fn mixed_orbital_integral_and_amplitude_paths_pass_four_step_fits_through_fourth_order() {
    let t = Taylor::<7>::from_coefficients(&[0.0, 1.0]);
    let (eri, o) = provider(t);
    let problem = from_ao(4, &eri, &o).unwrap();
    let solution = problem.solve(COEFFICIENTS).unwrap();
    let lagrangian = problem
        .lagrangian(&solution.amplitudes, COEFFICIENTS)
        .unwrap();
    for order in 0..7 {
        assert!((lagrangian.c[order] - solution.correlation.c[order]).abs() < 1e-12);
    }
    let energy = |x| {
        let e = determinant_oracle(x);
        COEFFICIENTS.same_spin * e[0] + COEFFICIENTS.opposite_spin * e[1]
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    for (order, factorial) in [(1, 1.0), (2, 2.0), (3, 6.0), (4, 24.0)] {
        let analytic = solution.correlation.c[order] * factorial;
        let estimate = |h: f64| match order {
            1 => (energy(h) - energy(-h)) / (2.0 * h),
            2 => (energy(h) - 2.0 * energy(0.0) + energy(-h)) / h.powi(2),
            3 => {
                (energy(2.0 * h) - 2.0 * energy(h) + 2.0 * energy(-h) - energy(-2.0 * h))
                    / (2.0 * h.powi(3))
            }
            4 => {
                (energy(2.0 * h) - 4.0 * energy(h) + 6.0 * energy(0.0) - 4.0 * energy(-h)
                    + energy(-2.0 * h))
                    / h.powi(4)
            }
            _ => unreachable!(),
        };
        let estimates: Vec<_> = steps.iter().map(|h| estimate(*h)).collect();
        let errors: Vec<_> = estimates.iter().map(|v| (v - analytic).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - analytic).abs();
        println!("PT2 mixed path order={order} analytic={analytic:.12e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.2 && fit.r_squared > 0.99);
        assert!(richardson < 1e-5);
    }
}

#[test]
fn singular_physical_pair_denominators_are_rejected_without_a_floor() {
    let (eri, mut o) = provider(0.0);
    o[0].energies.fill(0.0);
    let problem = from_ao(4, &eri, &o).unwrap();
    assert!(problem.solve(COEFFICIENTS).is_err());
}
