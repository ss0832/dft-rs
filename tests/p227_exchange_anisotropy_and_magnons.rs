// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance of the molecular magnetic-coupling features on the common
//! driver.
//!
//! `broken-symmetry-exchange`: for stretched H2 the high-spin determinant is
//! the unrestricted triplet of the `energy` feature, the reported coupling
//! is the spin-projected formula of the reported determinants, the coupling
//! is antiferromagnetic and decays exponentially with the distance, and the
//! broken-symmetry determinant tends to an equal mixture of singlet and
//! triplet (`<S^2> -> 1`).
//!
//! `magnetic-anisotropy`: without spin-orbit charges the anisotropy
//! vanishes; for triplet O2 the two axes perpendicular to the bond are
//! equivalent; the force-theorem anisotropy is second order in the
//! spin-orbit strength (four-step power fit over scaled effective charges)
//! and agrees with the self-consistent spinor anisotropy to leading order,
//! their difference being of higher order.
//!
//! `magnon`: for one electron the spin-lowering manifold is complete, so the
//! lowest spin-flip state is the other spin component of the reference and
//! its energy (the Goldstone mode) vanishes, in Hartree-Fock and with the
//! non-collinear kernel of a local functional.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn hydrogen_pair(distance: f64) -> System {
    System::molecule(vec![1, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
}

fn exchange_request() -> Vec<(&'static str, Value)> {
    vec![
        ("basis", Value::Str("6-31g".into())),
        ("xc", Value::Str("hf".into())),
        ("multiplicity", Value::Int(3)),
        ("conv_tol", Value::Float(1e-12)),
        ("site_a", Value::Array(vec![Value::Int(0)])),
        ("site_b", Value::Array(vec![Value::Int(1)])),
    ]
}

#[test]
fn broken_symmetry_coupling_of_stretched_hydrogen() {
    let distances = [4.0, 5.0, 6.0, 7.0];
    let mut couplings = Vec::new();
    for &distance in &distances {
        let report = run(
            "broken-symmetry-exchange",
            &hydrogen_pair(distance),
            &controls(&exchange_request()),
        )
        .unwrap();
        let (e_hs, e_bs) = (
            number(&report, "high_spin_energy"),
            number(&report, "broken_symmetry_energy"),
        );
        let (s_hs, s_bs) = (
            number(&report, "high_spin_s_squared"),
            number(&report, "broken_symmetry_s_squared"),
        );
        let coupling = number(&report, "exchange_coupling");
        let projected = -(e_hs - e_bs) / (s_hs - s_bs);
        let triplet = run(
            "energy",
            &hydrogen_pair(distance),
            &controls(&[
                ("basis", Value::Str("6-31g".into())),
                ("xc", Value::Str("hf".into())),
                ("multiplicity", Value::Int(3)),
                ("unrestricted", Value::Bool(true)),
                ("conv_tol", Value::Float(1e-12)),
            ]),
        )
        .unwrap();
        println!(
            "R = {distance}: E_HS {e_hs:.10} E_BS {e_bs:.10} <S2> {s_hs:.6} {s_bs:.6} \
             J {coupling:.6e} ({:.4} cm-1)",
            number(&report, "exchange_coupling_cm1")
        );
        assert!((e_hs - number(&triplet, "energy")).abs() < 1e-9);
        assert!((s_hs - 2.0).abs() < 1e-8);
        assert!((coupling - projected).abs() <= 1e-12 * projected.abs().max(1e-12));
        assert!(coupling < 0.0, "H2 couples antiferromagnetically");
        couplings.push(coupling);
        if distance >= 6.0 {
            assert!((s_bs - 1.0).abs() < 1e-2, "<S2>_BS = {s_bs}");
        }
    }
    // Exponential decay: ln|J| is linear in the distance.
    let logs: Vec<f64> = couplings.iter().map(|j| j.abs().ln()).collect();
    for pair in logs.windows(2) {
        assert!(pair[1] < pair[0]);
    }
    let n = distances.len() as f64;
    let mean_x = distances.iter().sum::<f64>() / n;
    let mean_y = logs.iter().sum::<f64>() / n;
    let sxx: f64 = distances.iter().map(|x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = distances
        .iter()
        .zip(&logs)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let slope = sxy / sxx;
    let sse: f64 = distances
        .iter()
        .zip(&logs)
        .map(|(x, y)| (y - mean_y - slope * (x - mean_x)).powi(2))
        .sum();
    let sst: f64 = logs.iter().map(|y| (y - mean_y).powi(2)).sum();
    let r_squared = 1.0 - sse / sst;
    println!("ln|J| against R: slope {slope:.6} per Bohr, R2 {r_squared:.8}");
    assert!(slope < 0.0);
    assert!(r_squared > 0.99);
}

fn oxygen() -> System {
    System::molecule(vec![8, 8], vec![[0.0, 0.0, -1.14], [0.0, 0.0, 1.14]])
}

fn anisotropy_request(scale: f64, self_consistent: bool) -> Vec<(&'static str, Value)> {
    vec![
        ("basis", Value::Str("sto-3g".into())),
        ("xc", Value::Str("svwn5".into())),
        ("multiplicity", Value::Int(3)),
        ("unrestricted", Value::Bool(true)),
        ("conv_tol", Value::Float(1e-12)),
        (
            "soc_charges",
            Value::Array(vec![Value::Float(8.0 * scale), Value::Float(8.0 * scale)]),
        ),
        ("spin_orbit_self_consistent", Value::Bool(self_consistent)),
    ]
}

#[test]
fn magnetic_anisotropy_limits_and_order_of_triplet_oxygen() {
    // Default axes x, y, z; the bond is along z.
    let none = run(
        "magnetic-anisotropy",
        &oxygen(),
        &controls(&anisotropy_request(0.0, false)),
    )
    .unwrap();
    let zero = vector(&none, "anisotropy_energies");
    println!("anisotropy without spin-orbit charges {zero:?}");
    assert!(zero.iter().all(|e| e.abs() < 1e-12));

    // Scales kept large enough that the fourth-order difference of the
    // self-consistent and force-theorem values stays above the SCF noise.
    let scales = [1.0, 0.8, 0.6, 0.4];
    let mut force_theorem = Vec::new();
    let mut difference = Vec::new();
    for &scale in &scales {
        let report = run(
            "magnetic-anisotropy",
            &oxygen(),
            &controls(&anisotropy_request(scale, true)),
        )
        .unwrap();
        let bands = vector(&report, "band_energies");
        let consistent = vector(&report, "self_consistent_energies");
        let perpendicular = (bands[0] - bands[1]).abs();
        let anisotropy = bands[2] - 0.5 * (bands[0] + bands[1]);
        let consistent_anisotropy = consistent[2] - 0.5 * (consistent[0] + consistent[1]);
        println!(
            "scale {scale}: force theorem {anisotropy:.6e}, self-consistent \
             {consistent_anisotropy:.6e}, x/y asymmetry {perpendicular:.3e}"
        );
        assert!(perpendicular < 1e-9, "x/y asymmetry {perpendicular:e}");
        force_theorem.push(anisotropy.abs());
        difference.push((anisotropy - consistent_anisotropy).abs());
    }
    let fit = convergence::error_power_fit(&scales, &force_theorem);
    println!(
        "force-theorem anisotropy against the spin-orbit scale: {force_theorem:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.05, "p = {}", fit.exponent);
    assert!(fit.r_squared > 0.9999);
    let higher = convergence::error_power_fit(&scales, &difference);
    println!(
        "self-consistent minus force-theorem anisotropy: {difference:?} p={:.6} R2={:.8}",
        higher.exponent, higher.r_squared
    );
    assert!(higher.exponent > 2.5, "p = {}", higher.exponent);
    assert!(higher.r_squared > 0.999);
}

#[test]
fn one_electron_goldstone_mode_vanishes() {
    let atom = System::molecule(vec![1], vec![[0.0, 0.0, 0.0]]);
    for xc in ["hf", "svwn5"] {
        let report = run(
            "magnon",
            &atom,
            &controls(&[
                ("basis", Value::Str("6-31g".into())),
                ("xc", Value::Str(xc.into())),
                ("multiplicity", Value::Int(2)),
                ("unrestricted", Value::Bool(true)),
                ("conv_tol", Value::Float(1e-12)),
                ("nstates", Value::Int(2)),
            ]),
        )
        .unwrap();
        let lowering = vector(&report, "spin_lowering_energies");
        let goldstone = number(&report, "goldstone_energy");
        println!("{xc}: spin-lowering energies {lowering:?}, Goldstone {goldstone:.3e}");
        assert_eq!(goldstone, lowering[0]);
        assert!(goldstone.abs() < 1e-8, "{xc}: {goldstone:e}");
        assert!(lowering[1] > 0.1);
    }
}
