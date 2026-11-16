// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Variational complex AO contraction for every semilocal rung.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dfpt::orbitals::{adjoint, matmul};
use dft_rs::periodic::spinor_scf::from_pauli;
use dft_rs::periodic::spinor_xc::spinor_semilocal_grid;
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;
use dft_rs::xc::{resolve, FunctionalSpec};

fn put(m: &mut BlochMatrix, i: usize, j: usize, v: Complex) {
    m.re[(i, j)] = v.re;
    m.im[(i, j)] = v.im;
}
fn orbitals() -> (BlochMatrix, [BlochMatrix; 3]) {
    let points = [[0.2, -0.1, 0.3], [-0.3, 0.15, 0.2], [0.1, 0.25, -0.15]];
    let mut phi = BlochMatrix::zeros(3, 2);
    let mut momentum = std::array::from_fn(|_| BlochMatrix::zeros(3, 2));
    for (g, r) in points.into_iter().enumerate() {
        let square = r.iter().map(|x| x * x).sum::<f64>();
        for (a, exponent) in [0.7, 0.4].into_iter().enumerate() {
            let q = [0.1 * (a + 1) as f64, -0.07, 0.05];
            let phase = (0..3).map(|d| q[d] * r[d]).sum::<f64>();
            let value = Complex::new(phase.cos(), phase.sin()) * (-exponent * square).exp();
            put(&mut phi, g, a, value);
            for d in 0..3 {
                put(
                    &mut momentum[d],
                    g,
                    a,
                    value * Complex::new(q[d], 2.0 * exponent * r[d]),
                );
            }
        }
    }
    (phi, momentum)
}
fn density() -> [BlochMatrix; 4] {
    let mut p = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    p[0].re[(0, 0)] = 0.9;
    p[0].re[(1, 1)] = 0.8;
    p[1].re[(0, 0)] = 0.1;
    p[2].re[(1, 1)] = 0.12;
    p[3].re[(0, 0)] = 0.15;
    p[3].re[(1, 1)] = -0.1;
    put(&mut p[1], 0, 1, Complex::new(0.03, 0.02));
    put(&mut p[1], 1, 0, Complex::new(0.03, -0.02));
    p
}

fn near_stationary_occupied_orbital() -> (BlochMatrix, [BlochMatrix; 3], [BlochMatrix; 4]) {
    let mut phi = BlochMatrix::zeros(1, 2);
    phi.re[(0, 0)] = 1.0;
    put(&mut phi, 0, 1, Complex::new(0.3, -0.1));
    let momentum = std::array::from_fn(|axis| {
        let mut m = BlochMatrix::zeros(1, 2);
        put(
            &mut m,
            0,
            0,
            Complex::new(0.0, [0.0002, -0.0003, 0.0001][axis]),
        );
        put(
            &mut m,
            0,
            1,
            Complex::new([0.01, -0.05, 0.07][axis], [0.12, 0.17, -0.19][axis]),
        );
        m
    });
    let mut p = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    p[0].re[(0, 0)] = 1.0;
    p[3].re[(0, 0)] = 1.0;
    (phi, momentum, p)
}

#[test]
fn iso_orbital_fock_is_invariant_to_a_large_local_phase_gradient() {
    let (phi, momentum, p) = near_stationary_occupied_orbital();
    let functional = resolve(&FunctionalSpec::Combined("m06l".into())).unwrap();
    let evaluate = |p_momentum: &[BlochMatrix; 3]| {
        spinor_semilocal_grid(
            &functional,
            &[1.0],
            &[&phi],
            &[p_momentum],
            std::slice::from_ref(&p),
            8,
        )
        .unwrap()
    };
    let reference = evaluate(&momentum);
    for boost in [[0.21, -0.13, 0.17], [0.39, -0.2, 0.04]] {
        let mut shifted = momentum.clone();
        for d in 0..3 {
            shifted[d].add_scaled(&phi, boost[d]);
        }
        let gauged = evaluate(&shifted);
        let energy_defect = (gauged.0 - reference.0).abs();
        let matrix_defect = gauged.1[0]
            .re
            .max_abs_diff(&reference.1[0].re)
            .max(gauged.1[0].im.max_abs_diff(&reference.1[0].im));
        let scale = reference.1[0]
            .re
            .as_slice()
            .iter()
            .chain(reference.1[0].im.as_slice())
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        println!("iso-orbital boost {boost:?}: energy={energy_defect:.3e} Fock={matrix_defect:.3e} scale={scale:.3e}");
        assert!(energy_defect < 1e-12 && matrix_defect < 1e-10);
    }
}

#[test]
fn iso_orbital_fock_is_covariant_when_ao_kinetic_terms_cancel() {
    let (phi, momentum, density) = near_stationary_occupied_orbital();
    let mut rotation = BlochMatrix::zeros(2, 2);
    rotation.re[(0, 0)] = 0.6;
    rotation.re[(0, 1)] = -0.8;
    rotation.re[(1, 0)] = 0.8;
    rotation.re[(1, 1)] = 0.6;
    let transpose = adjoint(&rotation).unwrap();
    let transform = |m: &BlochMatrix| matmul(&transpose, &matmul(m, &rotation).unwrap()).unwrap();
    let rotated_phi = matmul(&phi, &rotation).unwrap();
    let rotated_momentum = momentum.each_ref().map(|p| matmul(p, &rotation).unwrap());
    let rotated_density = density.each_ref().map(transform);
    let functional = resolve(&FunctionalSpec::Combined("m06l".into())).unwrap();
    let evaluate = |phi: &BlochMatrix, p: &[BlochMatrix; 3], density: &[BlochMatrix; 4]| {
        spinor_semilocal_grid(
            &functional,
            &[1.0],
            &[phi],
            &[p],
            std::slice::from_ref(density),
            8,
        )
        .unwrap()
    };
    let reference = evaluate(&phi, &momentum, &density);
    let rotated = evaluate(&rotated_phi, &rotated_momentum, &rotated_density);
    let mut spin_rotation = BlochMatrix::zeros(4, 4);
    for spin in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                spin_rotation.re[(2 * spin + i, 2 * spin + j)] = rotation.re[(i, j)];
            }
        }
    }
    let expected = matmul(
        &adjoint(&spin_rotation).unwrap(),
        &matmul(&reference.1[0], &spin_rotation).unwrap(),
    )
    .unwrap();
    let energy_defect = (rotated.0 - reference.0).abs();
    let fock_defect = expected
        .re
        .max_abs_diff(&rotated.1[0].re)
        .max(expected.im.max_abs_diff(&rotated.1[0].im));
    println!("near-stationary AO rotation: energy={energy_defect:.3e} Fock={fock_defect:.3e}");
    assert!(energy_defect < 1e-12 && fock_defect < 1e-8);
}

#[test]
fn actual_spinor_fock_has_independent_four_step_energy_fits_and_gauge_covariance() {
    let (phi, momentum) = orbitals();
    let p = density();
    let weights = [0.3, 0.2, 0.4];
    let mut delta = std::array::from_fn::<_, 4, _>(|_| BlochMatrix::zeros(2, 2));
    for a in 0..4 {
        for i in 0..2 {
            delta[a].re[(i, i)] = 0.03 * ((a + i + 1) as f64 * 0.7).sin();
        }
        put(
            &mut delta[a],
            0,
            1,
            Complex::new(0.015 * (a + 1) as f64, 0.008),
        );
        put(
            &mut delta[a],
            1,
            0,
            Complex::new(0.015 * (a + 1) as f64, -0.008),
        );
    }
    let full_delta = from_pauli(&delta).unwrap();
    let boost = [0.21, -0.13, 0.17];
    let mut transformed = momentum.clone();
    for d in 0..3 {
        transformed[d].add_scaled(&phi, boost[d]);
    }
    for name in ["svwn5", "pbe", "tpss", "r2scan", "m06l"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let reference = spinor_semilocal_grid(
            &f,
            &weights,
            &[&phi],
            &[&momentum],
            std::slice::from_ref(&p),
            16,
        )
        .unwrap();
        let exact = full_delta.trace_product(&reference.1[0]);
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let evaluate = |sign: f64| {
                let varied = std::array::from_fn(|a| {
                    let mut value = p[a].clone();
                    value.add_scaled(&delta[a], sign * h);
                    value
                });
                spinor_semilocal_grid(&f, &weights, &[&phi], &[&momentum], &[varied], 16)
                    .unwrap()
                    .0
            };
            let estimate = (evaluate(1.0) - evaluate(-1.0)) / (2.0 * h);
            errors.push((estimate - exact).abs());
            estimates.push(estimate);
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
        println!("spinor AO {name}: steps={steps:?} raw-errors={errors:?} p={:.9} R2={:.10} Richardson={rich:.3e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        assert!(rich < 1e-8);
        assert!(reference.1[0].hermiticity_error() < 1e-12);
        assert!((reference.0 - reference.2.iter().sum::<f64>()).abs() < 1e-12);
        let gauge = spinor_semilocal_grid(
            &f,
            &weights,
            &[&phi],
            &[&transformed],
            std::slice::from_ref(&p),
            16,
        )
        .unwrap();
        let gap = reference.1[0]
            .re
            .max_abs_diff(&gauge.1[0].re)
            .max(reference.1[0].im.max_abs_diff(&gauge.1[0].im));
        println!(
            "spinor AO {name}: gauge-energy={:.3e} gauge-Fock={gap:.3e}",
            (reference.0 - gauge.0).abs()
        );
        assert!((reference.0 - gauge.0).abs() < 1e-12 && gap < 1e-10);
    }
}
