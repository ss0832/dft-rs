// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electron-phonon coupling on the common driver: the self-consistent
//! vertex checked through its diagonal, the eigenvalue derivatives
//! `d e_n/d x = [C^+ (F^x - e_n S^x) C]_nn`, against an independent four-step
//! fit of central differences of band energies of displaced cells
//! (validation only), on a metallic chain with Fermi-Dirac occupations; and
//! the consistency of lambda, a2F and the critical temperatures.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::dual::{Scalar, Taylor};

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.75, 0.1, 0.05]];

fn mode_path(t: Taylor<5>) -> ([Taylor<5>; 3], [Taylor<5>; 3]) {
    use dft_rs::periodic::electron_phonon::mode_vertex;
    type J = Taylor<5>;
    let f = J::from_f64;
    let one = f(1.0);
    let energies = [
        f(-0.2) / (one - t.scale(0.14)),
        f(0.3) * (one + t.scale(0.17)).powi(2),
    ];
    let average = (energies[0] + energies[1]).scale(0.5);
    let u = f(0.3) + t.scale(0.12);
    let den = one + u * u;
    let mut mode = [f(0.0); 6];
    mode[0] = (one - u * u) / den;
    mode[3] = u.scale(2.0) / den;
    let masses = [
        f(2.0) / (one - t.scale(0.2)),
        f(3.0) * (one + t.scale(0.11)),
    ];
    let omega = f(0.02) / (one - t.scale(0.13));
    let g = [
        (
            f(0.04) / (one - t.scale(0.21)),
            f(-0.02) * (one + t.scale(0.18)).powi(2),
        ),
        (
            f(-0.015) * (one - t.scale(0.16)),
            f(0.01) / (one - t.scale(0.19)),
        ),
    ];
    let mut overlap = [(f(0.0), f(0.0)); 6];
    overlap[0] = (f(0.1) + t.scale(0.02), f(-0.04));
    overlap[3] = (f(-0.03), f(0.07) + t.scale(0.01));
    let mut fock = [(f(0.0), f(0.0)); 6];
    for (k, x) in [0, 3].into_iter().enumerate() {
        fock[x] = (
            average * overlap[x].0 + g[k].0,
            average * overlap[x].1 + g[k].1,
        );
    }
    let (re, im) = mode_vertex(energies, &fock, &overlap, &mode, &masses, omega).unwrap();
    let (mut er, mut ei) = (f(0.0), f(0.0));
    for (k, x) in [0, 3].into_iter().enumerate() {
        let scale = mode[x] / (masses[k] * omega.scale(2.0)).sqrt();
        er = er + scale * g[k].0;
        ei = ei + scale * g[k].1;
    }
    // A common one-electron energy zero must cancel between F^x and S^x.
    let shift = f(1.7);
    for (f, s) in fock.iter_mut().zip(&overlap) {
        f.0 = f.0 + shift * s.0;
        f.1 = f.1 + shift * s.1;
    }
    let shifted = mode_vertex(
        energies.map(|e| e + shift),
        &fock,
        &overlap,
        &mode,
        &masses,
        omega,
    )
    .unwrap();
    for n in 0..5 {
        assert!(
            (re.c[n] - shifted.0.c[n]).abs() < 1e-13 && (im.c[n] - shifted.1.c[n]).abs() < 1e-13
        );
    }
    ([re, im, re * re + im * im], [er, ei, er * er + ei * ei])
}

#[test]
fn pulay_mode_vertex_and_squared_amplitude_have_four_order_independent_fits() {
    let (actual, exact) = mode_path(Taylor::from_coefficients(&[0.0, 1.0]));
    let steps = [0.08, 0.04, 0.02, 0.01];
    for j in 0..3 {
        for n in 0..5 {
            assert!((actual[j].c[n] - exact[j].c[n]).abs() < 1e-12);
        }
    }
    for j in 0..3 {
        for n in 1..=4 {
            let target = n as f64 * actual[j].c[n];
            let estimates: Vec<_> = steps
                .iter()
                .map(|&h| {
                    (mode_path(Taylor::from_coefficients(&[h, 1.0])).1[j].c[n - 1]
                        - mode_path(Taylor::from_coefficients(&[-h, 1.0])).1[j].c[n - 1])
                        / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
            println!("DFPT mode vertex component={j} order={n} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 1e-7);
        }
    }
}

fn chain(atom: usize, axis: usize, h: f64) -> System {
    let mut positions = POSITIONS.to_vec();
    positions[atom][axis] += h;
    System {
        numbers: vec![1, 1],
        positions,
        cell: Some(CELL),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(6), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(14.0)),
        ("grid_level".to_string(), Value::Int(3)),
        ("grid_shell".to_string(), Value::Int(3)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(300)),
        ("electronic_temperature".to_string(), Value::Float(3000.0)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

#[test]
fn vertex_diagonal_matches_band_energy_derivatives() {
    let report = run("electron-phonon", &chain(0, 0, 0.0), &controls(&[])).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let lambda = number(&report, "lambda");
    let modes = flat(&report, "mode_lambda");
    println!(
        "lambda {lambda}, mode lambdas {modes:?}, omega_log {} K, Tc (Allen-Dynes) {:?}, Tc (Eliashberg) {:?}, frequencies {:?}",
        number(&report, "omega_log_k"),
        report.get("tc_allen_dynes_k"),
        report.get("tc_eliashberg_k"),
        report.get("frequencies_cm1")
    );
    assert!((modes.iter().sum::<f64>() - lambda).abs() < 1e-12);
    // The Eliashberg critical temperature is either absent (no solution
    // above the search floor) or a positive finite temperature.
    match report
        .get("tc_eliashberg_k")
        .expect("tc_eliashberg_k is reported")
    {
        Value::Null => {}
        other => {
            let tc = other.as_f64().expect("tc_eliashberg_k is a float or null");
            assert!(tc.is_finite() && tc > 0.0, "tc_eliashberg_k {tc}");
        }
    }
    // derivatives[spin][k][band][x], 6 k points, 2 bands, 6 displacements.
    let derivatives = flat(&report, "eigenvalue_derivatives");
    let index = |k: usize, band: usize, x: usize| ((k * 2) + band) * 6 + x;
    let bands = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let r = run(
            "electronic-structure",
            &chain(atom, axis, h),
            &controls(&[]),
        )
        .unwrap();
        let channels = r.get("channels").unwrap().as_array().unwrap();
        flat(&channels[0], "band_energies")
    };
    // Preserve the original coarse interval explicitly. Thermal occupations
    // make its largest displacements non-asymptotic; two finer steps resolve
    // the quadratic interval without removing any measured error.
    let steps = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025];
    let (k, band) = (1usize, 0usize);
    for (atom, axis) in [(1usize, 0usize), (0, 1)] {
        let x = 3 * atom + axis;
        let analytic = derivatives[index(k, band, x)];
        let estimates: Vec<f64> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (bands(atom, axis, h), bands(atom, axis, -h));
                (plus[2 * k + band] - minus[2 * k + band]) / (2.0 * h)
            })
            .collect();
        let errors: Vec<_> = estimates.iter().map(|v| (v - analytic).abs()).collect();
        for (interval, range) in [("original coarse", 0..4), ("resolved", 2..6)] {
            let fit = convergence::error_power_fit(&steps[range.clone()], &errors[range.clone()]);
            println!("d e/d x (k {k}, band {band}, x {x}) interval={interval}: analytic {analytic:.16e} steps {:?} errors {:?} p={:.9} R2={:.12}",
                &steps[range.clone()],&errors[range],fit.exponent,fit.r_squared);
            if interval == "resolved" {
                assert!(
                    (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999,
                    "x {x}"
                );
            }
        }
        let rich = ((4.0 * estimates[5] - estimates[4]) / 3.0 - analytic).abs();
        println!("metallic LDA x={x} Richardson={rich:.12e}");
        assert!(rich < 1e-7);
    }
}

/// The vertex diagonal with exact exchange on an insulating chain:
/// Hartree-Fock and a global hybrid.
#[test]
fn hybrid_vertex_diagonal_matches_band_energy_derivatives() {
    for xc in ["hf", "pbe0"] {
        let settings = |extra: &[(&str, Value)]| {
            let mut all = vec![
                ("xc", Value::Str(xc.into())),
                (
                    "k_mesh",
                    Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
                ),
                ("overlap_cutoff", Value::Float(12.0)),
                ("exchange_cutoff", Value::Float(8.0)),
                ("grid_level", Value::Int(2)),
                ("grid_shell", Value::Int(1)),
                ("electronic_temperature", Value::Float(0.0)),
            ];
            all.extend(extra.iter().cloned());
            controls(&all)
        };
        let report = run(
            "electron-phonon",
            &chain(0, 0, 0.0),
            &settings(&[("electron_smearing", Value::Float(0.02))]),
        )
        .unwrap();
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        // derivatives[spin][k][band][x], 4 k points, 2 bands, 6 displacements.
        let derivatives = flat(&report, "eigenvalue_derivatives");
        let index = |k: usize, band: usize, x: usize| ((k * 2) + band) * 6 + x;
        let bands = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
            let r = run(
                "electronic-structure",
                &chain(atom, axis, h),
                &settings(&[]),
            )
            .unwrap();
            let channels = r.get("channels").unwrap().as_array().unwrap();
            flat(&channels[0], "band_energies")
        };
        // The second atom is 0.05 Bohr from the midpoint between its two
        // neighbours along the chain, where the energy surface of the
        // insulating chain turns: a displacement of 0.08 Bohr along the
        // chain crosses it (errors 6.33e-3, 1.85e-5, 4.62e-6, 1.15e-6 on
        // 0.08..0.01 for Hartree-Fock, p = 3.927, R2 = 0.8617). The interval
        // below stays on one side.
        let steps = [0.04, 0.02, 0.01, 0.005];
        for (k, band, atom, axis) in [(1usize, 0usize, 1usize, 0usize), (1, 1, 0, 1)] {
            let x = 3 * atom + axis;
            let analytic = derivatives[index(k, band, x)];
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| {
                    let (plus, minus) = (bands(atom, axis, h), bands(atom, axis, -h));
                    ((plus[2 * k + band] - minus[2 * k + band]) / (2.0 * h) - analytic).abs()
                })
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            println!(
                "{xc}: d e/d x (k {k}, band {band}, x {x}): analytic {analytic:.10e} steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
                fit.exponent, fit.r_squared
            );
            assert!(
                (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999,
                "{xc}: k {k} band {band} x {x}"
            );
        }
    }
}

/// The vertex diagonal with exact exchange and Fermi-Dirac occupations: an
/// equidistant chain, whose two bands touch at the zone boundary, with a
/// global hybrid at an electronic temperature that leaves both bands
/// partially occupied.
#[test]
fn hybrid_metallic_vertex_diagonal_matches_band_energy_derivatives() {
    let equidistant = |atom: usize, axis: usize, h: f64| {
        let mut positions = vec![[0.0, 0.0, 0.0], [0.5 * CELL[0][0], 0.1, 0.05]];
        positions[atom][axis] += h;
        System {
            numbers: vec![1, 1],
            positions,
            cell: Some(CELL),
            pbc: [true; 3],
        }
    };
    let settings = || {
        controls(&[
            ("xc", Value::Str("pbe0".into())),
            (
                "k_mesh",
                Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
            ),
            ("overlap_cutoff", Value::Float(12.0)),
            ("exchange_cutoff", Value::Float(8.0)),
            ("grid_level", Value::Int(2)),
            ("grid_shell", Value::Int(1)),
            ("electronic_temperature", Value::Float(20000.0)),
        ])
    };
    let report = run("electron-phonon", &equidistant(0, 0, 0.0), &settings()).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let reference = run("electronic-structure", &equidistant(0, 0, 0.0), &settings()).unwrap();
    let channels = reference.get("channels").unwrap().as_array().unwrap();
    let occupations = flat(&channels[0], "occupations");
    println!(
        "lambda {}, occupations {occupations:?}, band energies {:?}",
        number(&report, "lambda"),
        flat(&channels[0], "band_energies")
    );
    // Both bands carry fractional occupations.
    assert!(occupations.iter().any(|f| *f > 0.02 && *f < 0.98));
    // derivatives[spin][k][band][x], 4 k points, 2 bands, 6 displacements.
    let derivatives = flat(&report, "eigenvalue_derivatives");
    let index = |k: usize, band: usize, x: usize| ((k * 2) + band) * 6 + x;
    let bands = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let r = run(
            "electronic-structure",
            &equidistant(atom, axis, h),
            &settings(),
        )
        .unwrap();
        let channels = r.get("channels").unwrap().as_array().unwrap();
        flat(&channels[0], "band_energies")
    };
    let steps = [0.04, 0.02, 0.01, 0.005];
    for (k, band, atom, axis) in [(1usize, 0usize, 1usize, 1usize), (1, 1, 0, 2), (3, 0, 1, 0)] {
        let x = 3 * atom + axis;
        let analytic = derivatives[index(k, band, x)];
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (bands(atom, axis, h), bands(atom, axis, -h));
                ((plus[2 * k + band] - minus[2 * k + band]) / (2.0 * h) - analytic).abs()
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "metallic pbe0: d e/d x (k {k}, band {band}, x {x}): analytic {analytic:.10e} steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999,
            "k {k} band {band} x {x}"
        );
    }
}
