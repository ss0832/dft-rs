// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Long-wave force-constant derivatives with independent diatomic-chain
//! Fourier oracles. Both phase conventions and all seven crystal systems.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::periodic::phonon_bands::{supercell_index, ForceConstantModel, PhononPhase};
use dft_rs::periodic::Lattice;

fn systems() -> Vec<Lattice> {
    [
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        [
            [3.0, 0.0, 0.0],
            [-1.5, 2.598076211353316, 0.0],
            [0.0, 0.0, 4.0],
        ],
        [
            [3.0, 0.0, 0.0],
            [1.0, 2.8284271247461903, 0.0],
            [1.0, 0.7071067811865475, 2.7386127875258306],
        ],
        [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 0.0, 4.0]],
        [[3.0, 0.0, 0.0], [0.3, 3.5, 0.0], [0.2, 0.4, 4.2]],
    ]
    .map(|a| {
        let v = |r: [f64; 3]| Vec3::new(r[0], r[1], r[2]);
        Lattice::new(v(a[0]), v(a[1]), v(a[2]))
    })
    .to_vec()
}

fn model(lattice: &Lattice) -> ForceConstantModel {
    let cells = 5;
    let mut hessian = Matrix::zeros(6 * cells, 6 * cells);
    let mut spring = |a: usize, b: usize, k: f64| {
        for axis in 0..3 {
            hessian[(3 * a + axis, 3 * a + axis)] += k;
            hessian[(3 * b + axis, 3 * b + axis)] += k;
            hessian[(3 * a + axis, 3 * b + axis)] -= k;
            hessian[(3 * b + axis, 3 * a + axis)] -= k;
        }
    };
    for n in 0..cells {
        let index = |atom, cell| supercell_index(atom, 2, [cell, 0, 0], [cells, 1, 1]);
        spring(index(0, n), index(1, n), 1.3);
        spring(index(0, n), index(1, (n + cells - 1) % cells), 0.4);
    }
    ForceConstantModel::from_supercell_hessian(
        lattice,
        &[Vec3::zero(), Vec3::new(1.1, 0.2, 0.1)],
        &[2.0, 3.0],
        [cells, 1, 1],
        &hessian,
    )
    .unwrap()
}

fn closed(q: Vec3, d: Vec3, phase: PhononPhase, n: usize) -> (f64, f64) {
    let tau = if phase == PhononPhase::Atomic {
        Vec3::new(1.1, 0.2, 0.1)
    } else {
        Vec3::zero()
    };
    let vectors = [tau, tau - Vec3::new(3.0, 0.0, 0.0)];
    let mut real = 0.0;
    let mut imaginary = 0.0;
    let factorial = (1..=n).product::<usize>() as f64;
    for (v, k) in vectors.into_iter().zip([1.3, 0.4]) {
        let angle = q.dot(v) + n as f64 * std::f64::consts::FRAC_PI_2;
        let prefactor = -k * d.dot(v).powi(n as i32) / factorial / 6.0_f64.sqrt();
        real += prefactor * angle.cos();
        imaginary += prefactor * angle.sin();
    }
    (real, imaginary)
}

#[test]
fn analytic_long_wave_moments_have_four_step_fits_in_all_crystal_systems() {
    let q = Vec3::new(0.17, -0.09, 0.04);
    let d = Vec3::new(0.3, -0.1, 0.2);
    let steps = [0.08, 0.04, 0.02, 0.01];
    for (crystal, lattice) in systems().iter().enumerate() {
        let model = model(lattice);
        assert!(model.acoustic_residual < 1e-15);
        for phase in [PhononPhase::Cell, PhononPhase::Atomic] {
            let exact = model.fourier_taylor(q, d, 4, phase, true).unwrap();
            assert!(exact.hermiticity_defects.iter().all(|v| *v < 1e-13));
            for n in 0..=4 {
                let (a, b) = closed(q, d, phase, n);
                assert!((a - exact.real[n][(0, 3)]).abs() < 1e-13);
                assert!((b - exact.imaginary[n][(0, 3)]).abs() < 1e-13);
            }
            for n in 1..=4 {
                let target = (
                    n as f64 * exact.real[n][(0, 3)],
                    n as f64 * exact.imaginary[n][(0, 3)],
                );
                let estimates: Vec<_> = steps
                    .iter()
                    .map(|&h| {
                        let a = closed(q + d * h, d, phase, n - 1);
                        let b = closed(q - d * h, d, phase, n - 1);
                        ((a.0 - b.0) / (2.0 * h), (a.1 - b.1) / (2.0 * h))
                    })
                    .collect();
                let errors: Vec<_> = estimates
                    .iter()
                    .map(|v| (v.0 - target.0).hypot(v.1 - target.1))
                    .collect();
                let fit = convergence::error_power_fit(&steps, &errors);
                let rich = ((4.0 * estimates[3].0 - estimates[2].0) / 3.0 - target.0)
                    .hypot((4.0 * estimates[3].1 - estimates[2].1) / 3.0 - target.1);
                println!("force moment crystal={crystal} phase={phase:?} order={n} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
                assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
                assert!(rich < 1e-6);
            }
            let unweighted = model.fourier_taylor(q, d, 4, phase, false).unwrap();
            for n in 0..=4 {
                assert!(
                    (unweighted.real[n][(0, 3)] - 6.0_f64.sqrt() * exact.real[n][(0, 3)]).abs()
                        < 1e-13
                );
            }
        }
        let at_gamma = model
            .fourier_taylor(Vec3::zero(), d, 2, PhononPhase::Atomic, false)
            .unwrap();
        for i in 0..6 {
            assert!(
                (0..2)
                    .map(|atom| at_gamma.real[0][(i, 3 * atom + i % 3)])
                    .sum::<f64>()
                    .abs()
                    < 1e-15
            );
        }
    }
}

fn relaxation_closed(h: f64) -> (dft_rs::dual::Taylor<5>, dft_rs::dual::Taylor<5>) {
    use dft_rs::dual::{Scalar, Taylor};
    let d = Vec3::new(0.3, -0.1, 0.2);
    let mut fr = Taylor::<5>::from_f64(0.0);
    let mut fi = fr;
    for n in 0..5 {
        let (r, i) = closed(d * h, d, PhononPhase::Atomic, n);
        fr.c[n] = r * 6.0_f64.sqrt();
        fi.c[n] = i * 6.0_f64.sqrt();
    }
    let den = Taylor::from_f64(0.52 * 1.7) - fr.scale(0.48);
    (-(Taylor::from_f64(1.7) + fr).scale(0.2) / den, -fi / den)
}

#[test]
fn constrained_long_wave_internal_relaxation_and_born_contraction_have_four_step_fits() {
    use dft_rs::response::ResponseSolverConfig;
    let d = Vec3::new(0.3, -0.1, 0.2);
    let steps = [0.08, 0.04, 0.02, 0.01];
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let (sr, si) = relaxation_closed(0.0);
    for (crystal, lattice) in systems().iter().enumerate() {
        let model = model(lattice);
        let exact = model.long_wave_relaxation(d, 4, &config).unwrap();
        let born = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        ];
        let (pr, pi) = exact
            .born_polarization(&born, lattice.volume().abs())
            .unwrap();
        assert!(exact.uniform_force_defect < 1e-12);
        for n in 0..5 {
            let u0 = if n == 0 { 1.0 } else { 0.0 };
            assert!((exact.real[n][(0, 0)] - u0 - 0.6 * sr.c[n]).abs() < 1e-12);
            assert!((exact.imaginary[n][(0, 0)] - 0.6 * si.c[n]).abs() < 1e-12);
            assert!((exact.real[n][(3, 0)] - u0 + 0.4 * sr.c[n]).abs() < 1e-12);
            assert!((pr[n][(0, 0)] - sr.c[n] / lattice.volume().abs()).abs() < 1e-12);
            assert!((pi[n][(0, 0)] - si.c[n] / lattice.volume().abs()).abs() < 1e-12);
            assert!(exact.relative_residuals[n].iter().all(|x| *x < 1e-10));
            assert!(exact.center_of_mass_defects[n].iter().all(|x| *x < 1e-12));
        }
        println!(
            "long-wave relaxation crystal={crystal} residual={:?} COM={:?} Born_mu={:.12e}",
            exact.relative_residuals,
            exact.center_of_mass_defects,
            -pr[2][(0, 0)]
        );
        for n in 1..=4 {
            let target = (
                n as f64 * exact.real[n][(0, 0)],
                n as f64 * exact.imaginary[n][(0, 0)],
            );
            let estimates: Vec<_> = steps
                .iter()
                .map(|&h| {
                    let (p, q) = relaxation_closed(h);
                    let (r, s) = relaxation_closed(-h);
                    (
                        0.6 * (p.c[n - 1] - r.c[n - 1]) / (2.0 * h),
                        0.6 * (q.c[n - 1] - s.c[n - 1]) / (2.0 * h),
                    )
                })
                .collect();
            let errors: Vec<_> = estimates
                .iter()
                .map(|(r, i)| (r - target.0).hypot(i - target.1))
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3].0 - estimates[2].0) / 3.0 - target.0)
                .hypot((4.0 * estimates[3].1 - estimates[2].1) / 3.0 - target.1);
            println!("long-wave relaxation crystal={crystal} order={n} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999 && rich < 1e-6);
        }
    }
}
