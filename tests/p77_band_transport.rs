// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic band velocities and effective masses against independent
//! four-step fits of central differences of the band energies in k
//! (validation only; the band energies at displaced k need no SCF), the
//! degenerate p bands of a cubic neon lattice by degenerate perturbation
//! theory, and the Boltzmann transport driver (carrier-concentration
//! round trip, Onsager symmetry, CLI route).

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::periodic::{prepare_periodic, PeriodicMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::math::Vec3;
use dft_rs::periodic::band_transport::BandModel;
use std::process::Command;

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("overlap_cutoff".to_string(), Value::Float(14.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(1)),
        ("conv_tol".to_string(), Value::Float(1e-11)),
        ("conv_tol_dp".to_string(), Value::Float(1e-9)),
        ("max_iter".to_string(), Value::Int(300)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn mesh(k: [i64; 3]) -> Value {
    Value::Array(k.iter().map(|v| Value::Int(*v)).collect())
}

fn chain() -> System {
    System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, 0.0], [1.6, 0.1, 0.05]],
        cell: Some([[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]),
        pbc: [true; 3],
    }
}

/// A tetragonal neon lattice. The product angular grid is invariant under
/// the 90-degree rotation about z (uniform azimuthal points), so p_x and p_y
/// stay exactly degenerate at Gamma; a cubic cell would also need the x/z
/// permutation, which the product grid breaks (measured splitting 9.6e-6
/// Hartree at grid level 2). The longer c axis moves p_z away from the pair.
fn neon() -> System {
    System {
        numbers: vec![10],
        positions: vec![[0.0, 0.0, 0.0]],
        cell: Some([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.5]]),
        pbc: [true; 3],
    }
}

fn model(system: &System, c: &Controls) -> BandModel {
    let method = PeriodicMethod::from_controls(c).unwrap();
    let setup = prepare_periodic(system, &method).unwrap();
    let scf =
        dft_rs::periodic::scf::run_periodic_scf(&setup.cell, &setup.functional, &setup.options)
            .unwrap();
    assert!(scf.converged);
    BandModel::new(&setup.cell, &setup.functional, &scf, &setup.options).unwrap()
}

fn energies(m: &BandModel, k: Vec3) -> Vec<f64> {
    m.bands(0, k).unwrap().energies
}

const STEPS: [f64; 4] = [0.04, 0.02, 0.01, 0.005];

#[test]
fn velocities_and_masses_pass_four_step_fits_of_the_band_energies() {
    let m = model(&chain(), &controls(&[("k_mesh", mesh([4, 1, 1]))]));
    println!("Fock image residual {:e}", m.fock_image_residual);
    assert!(m.fock_image_residual < 1e-10);
    let k = Vec3::new(0.31, 0.07, 0.05);
    let raw = Vec3::new(0.8, 0.5, 0.33);
    let u = raw * (1.0 / raw.norm());
    for band in [0usize, 1] {
        let d = m.derivatives(0, k, band, Some(raw), 1e-6).unwrap();
        let v = d.velocity.unwrap();
        let t = d.inverse_mass.unwrap();
        let uv = (0..3).map(|a| v[a] * u.get(a)).sum::<f64>();
        let utu = (0..3)
            .map(|a| (0..3).map(|b| u.get(a) * t[a][b] * u.get(b)).sum::<f64>())
            .sum::<f64>();
        // The branch route (degenerate formulas on a one-band cluster) agrees.
        assert!(
            (d.branches[0].velocity - uv).abs() < 1e-12
                && (d.branches[0].second_derivative - utu).abs() < 1e-10
        );
        let e0 = energies(&m, k)[band];
        let (mut first, mut second) = (Vec::new(), Vec::new());
        for h in STEPS {
            let ep = energies(&m, k + u * h)[band];
            let em = energies(&m, k + u * (-h))[band];
            first.push(((ep - em) / (2.0 * h) - uv).abs());
            second.push(((ep - 2.0 * e0 + em) / (h * h) - utu).abs());
        }
        let f1 = convergence::error_power_fit(&STEPS, &first);
        let f2 = convergence::error_power_fit(&STEPS, &second);
        println!(
            "band {band}: v.u {uv:.10e} errors {first:?} p={:.6} R2={:.8}; u.M.u {utu:.10e} errors {second:?} p={:.6} R2={:.8}; steps {STEPS:?}",
            f1.exponent, f1.r_squared, f2.exponent, f2.r_squared
        );
        assert!(
            (f1.exponent - 2.0).abs() < 0.1 && f1.r_squared > 0.9999,
            "velocity band {band}"
        );
        assert!(
            (f2.exponent - 2.0).abs() < 0.1 && f2.r_squared > 0.9999,
            "mass band {band}"
        );
    }
    // Off-diagonal tensor elements: mixed second differences along x and y.
    let band = 0usize;
    let t = m
        .derivatives(0, k, band, None, 1e-6)
        .unwrap()
        .inverse_mass
        .unwrap();
    let (ex, ey) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    let errors: Vec<f64> = STEPS
        .iter()
        .map(|&h| {
            let e = |a: f64, b: f64| energies(&m, k + ex * (a * h) + ey * (b * h))[band];
            ((e(1.0, 1.0) - e(1.0, -1.0) - e(-1.0, 1.0) + e(-1.0, -1.0)) / (4.0 * h * h) - t[0][1])
                .abs()
        })
        .collect();
    let fit = convergence::error_power_fit(&STEPS, &errors);
    println!(
        "band 0 M_xy {:.10e} errors {errors:?} p={:.6} R2={:.8}",
        t[0][1], fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.9999);
}

#[test]
fn degenerate_p_bands_follow_degenerate_perturbation_theory() {
    let m = model(&neon(), &controls(&[("k_mesh", mesh([2, 2, 2]))]));
    let gamma = Vec3::zero();
    let e = energies(&m, gamma);
    println!("neon bands at Gamma {e:?}");
    let raw = Vec3::new(0.9, 0.35, 0.2);
    let u = raw * (1.0 / raw.norm());
    // The exactly degenerate p_x/p_y pair among the 2p bands.
    let first = (2..4)
        .find(|&n| (e[n + 1] - e[n]).abs() < 1e-12)
        .expect("a degenerate p pair");
    let pair = first..first + 2;
    let d = m.derivatives(0, gamma, first, Some(raw), 1e-10).unwrap();
    assert_eq!(d.cluster, pair.clone().collect::<Vec<_>>());
    assert!(d.velocity.is_none());
    let others: Vec<f64> = (2..5).filter(|n| !pair.contains(n)).map(|n| e[n]).collect();
    println!(
        "pair {pair:?}, other 2p band {others:?}, separation {:e}",
        (others[0] - e[first]).abs()
    );
    let mut analytic: Vec<f64> = d.branches.iter().map(|b| b.second_derivative).collect();
    analytic.sort_by(|a, b| a.total_cmp(b));
    let velocities: Vec<f64> = d.branches.iter().map(|b| b.velocity).collect();
    println!("branch velocities {velocities:?}, second derivatives {analytic:?}");
    assert!(velocities.iter().all(|v| v.abs() < 1e-10));
    for (i, s) in analytic.iter().enumerate() {
        let mut errors = Vec::new();
        for h in STEPS {
            let (mut p, mut q) = (
                energies(&m, u * h)[pair.clone()].to_vec(),
                energies(&m, u * (-h))[pair.clone()].to_vec(),
            );
            p.sort_by(|a, b| a.total_cmp(b));
            q.sort_by(|a, b| a.total_cmp(b));
            let base = e[first];
            errors.push(((p[i] - 2.0 * base + q[i]) / (h * h) - s).abs());
        }
        let fit = convergence::error_power_fit(&STEPS, &errors);
        println!(
            "branch {i}: d2e/dt2 {s:.10e} errors {errors:?} p={:.6} R2={:.8}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.9999,
            "branch {i}"
        );
    }
}

#[test]
fn transport_round_trips_carrier_concentrations_and_is_symmetric() {
    let c = controls(&[
        ("k_mesh", mesh([4, 1, 1])),
        ("electronic_temperature", Value::Float(3000.0)),
        ("transport_k_mesh", mesh([16, 2, 2])),
        (
            "transport_temperatures_k",
            Value::Array(vec![Value::Float(300.0), Value::Float(1000.0)]),
        ),
        (
            "carrier_concentrations_cm3",
            Value::Array(vec![Value::Float(-1e20), Value::Float(2e20)]),
        ),
    ]);
    let report = run("boltzmann-transport", &chain(), &c).unwrap();
    let rows = report.get("results").unwrap().as_array().unwrap();
    assert_eq!(rows.len(), 4);
    for row in rows {
        let requested = row
            .get("requested_concentration_cm3")
            .and_then(Value::as_f64)
            .unwrap();
        let obtained = row
            .get("carrier_concentration_cm3")
            .and_then(Value::as_f64)
            .unwrap();
        let sigma = row
            .get("conductivity_over_tau")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        let asymmetry = (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .map(|(a, b)| (sigma[3 * a + b] - sigma[3 * b + a]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "T {} K, requested {requested:e} cm^-3, obtained {obtained:e}, sigma_xx/tau {:e} S/(m s), asymmetry {asymmetry:e}, Seebeck {:?}",
            row.get("temperature_k").and_then(Value::as_f64).unwrap(),
            sigma[0],
            row.get("seebeck")
        );
        assert!(((obtained - requested) / requested).abs() < 1e-8);
        assert!(sigma[0] > 0.0 && asymmetry <= 1e-12 * sigma[0]);
    }
}

#[test]
fn effective_mass_cli_route_matches_the_rust_driver() {
    let c = controls(&[
        ("k_mesh", mesh([4, 1, 1])),
        (
            "k_point",
            Value::Array(vec![
                Value::Float(0.1),
                Value::Float(0.0),
                Value::Float(0.0),
            ]),
        ),
        ("band", Value::Int(0)),
    ]);
    let report = run("effective-mass", &chain(), &c).unwrap();
    let system = chain();
    let path = std::env::temp_dir().join(format!("dft_mass_driver_{}.xyz", std::process::id()));
    let a = dft_rs::constants::BOHR_TO_ANGSTROM;
    let mut xyz = String::from("2\nchain\n");
    for p in &system.positions {
        xyz.push_str(&format!(
            "H {:.16} {:.16} {:.16}\n",
            p[0] * a,
            p[1] * a,
            p[2] * a
        ));
    }
    std::fs::write(&path, xyz).unwrap();
    let rows: Vec<String> = system
        .cell
        .unwrap()
        .iter()
        .map(|r| format!("{:.16},{:.16},{:.16}", r[0] * a, r[1] * a, r[2] * a))
        .collect();
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .args([
            "effective-mass",
            path.to_str().unwrap(),
            "--cell",
            &rows.join(";"),
            "--xc",
            "svwn5",
            "--basis",
            "sto-3g",
            "--k-mesh",
            "4,1,1",
            "--overlap-cutoff",
            "14",
            "--grid-level",
            "2",
            "--grid-shell",
            "1",
            "--conv-tol",
            "1e-11",
            "--conv-tol-dp",
            "1e-9",
            "--max-iter",
            "300",
            "--k-point",
            "0.1,0,0",
            "--band",
            "0",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json = Value::parse_json(&String::from_utf8_lossy(&output.stdout)).unwrap();
    let (cli, rust) = (
        json.get("inverse_mass_tensor")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1,
        report
            .get("inverse_mass_tensor")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1,
    );
    let gap = cli
        .iter()
        .zip(&rust)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "CLI against Rust: inverse mass gap {gap:e}, effective masses {:?}",
        report.get("effective_masses")
    );
    assert!(gap < 1e-8);
}
