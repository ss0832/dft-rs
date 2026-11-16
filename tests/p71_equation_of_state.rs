// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The Birch-Murnaghan equation of state on the common driver against two
//! independent analytic routes: the pressure of every scaled cell from the
//! analytic stress, and the bulk modulus at the reference volume from the
//! analytic elastic tensor, `(B11 + 2 B12)/3` of the stress-strain
//! coefficients of a cubic cell; plus the elastic-moduli and Christoffel
//! routes on the same tensor.

use dft_rs::driver::{run, Controls, System, Value};

const EDGE: f64 = 6.0;

fn helium() -> System {
    let h = 0.5 * EDGE;
    System {
        numbers: vec![2],
        positions: vec![[0.0, 0.0, 0.0]],
        cell: Some([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(2), Value::Int(2), Value::Int(2)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(12.0)),
        ("grid_level".to_string(), Value::Int(3)),
        ("grid_shell".to_string(), Value::Int(4)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(200)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

/// Recorded with two Becke shells: the fitted pressures exceeded the analytic
/// ones by a nearly constant 0.0080..0.0148 GPa (largest gap 1.48e-2 GPa
/// against p ~ 10 GPa), the volume-independent offset of the Becke
/// truncation already measured for the stress (periodic.md section 11);
/// the test uses four shells.
#[test]
fn equation_of_state_agrees_with_the_analytic_pressure_and_elastic_tensor() {
    let scales: Vec<Value> = [0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015]
        .iter()
        .map(|s| Value::Float(*s))
        .collect();
    let eos = run(
        "equation-of-state",
        &helium(),
        &controls(&[("scales", Value::Array(scales))]),
    )
    .unwrap();
    let volumes = flat(&eos, "volumes");
    println!("volumes {volumes:?} bohr^3");
    let analytic = flat(&eos, "analytic_pressures_gpa");
    let fitted = flat(&eos, "fitted_pressures_gpa");
    let gap = number(&eos, "pressure_fit_gap_gpa");
    println!(
        "V0 {} bohr^3, B0 {} GPa, B0' {}, residual {:e} Hartree; analytic p {analytic:?}; fitted p {fitted:?}; largest gap {gap:e} GPa",
        number(&eos, "v0"),
        number(&eos, "b0_gpa"),
        number(&eos, "b0_prime"),
        number(&eos, "residual_rms")
    );
    let span = analytic.iter().fold(0.0_f64, |m, p| m.max(p.abs()));
    assert!(
        gap < 1e-3 * span.max(1.0),
        "pressure from the fit against the analytic stress: {gap:e} GPa"
    );

    // B(V_ref) of the fit against (B11 + 2 B12)/3 of the analytic tensor.
    let elastic = run("elastic", &helium(), &controls(&[])).unwrap();
    let birch = flat(&elastic, "relaxed_ion_birch_gpa");
    let sym = |i: usize, j: usize| 0.5 * (birch[6 * i + j] + birch[6 * j + i]);
    let cubic =
        ((sym(0, 0) + sym(1, 1) + sym(2, 2)) + 2.0 * (sym(0, 1) + sym(0, 2) + sym(1, 2))) / 9.0;
    let p0 = analytic[3];
    // B = V d2E/dV2 of the fit at the reference volume.
    let fit_bulk = flat(&eos, "fitted_bulk_moduli_gpa")[3];
    println!("reference pressure {p0} GPa; bulk modulus from the fit {fit_bulk} GPa, from the analytic tensor {cubic} GPa");
    assert!(
        (fit_bulk - cubic).abs() < 2e-2 * cubic.abs(),
        "bulk modulus: fit {fit_bulk} against tensor {cubic}"
    );

    // The moduli and Christoffel routes on the same tensor.
    let moduli = run("elastic-moduli", &helium(), &controls(&[])).unwrap();
    let k_voigt = number(&moduli, "voigt_bulk_gpa");
    assert!((k_voigt - cubic).abs() < 1e-8 * cubic.abs());
    let waves = run(
        "christoffel",
        &helium(),
        &controls(&[(
            "directions",
            Value::Array(vec![Value::Array(vec![
                Value::Float(1.0),
                Value::Float(0.0),
                Value::Float(0.0),
            ])]),
        )]),
    )
    .unwrap();
    let rho = number(&waves, "density_kg_m3");
    let v = flat(&waves, "velocities_m_s");
    let longitudinal = (birch[0] * 1e9 / rho).sqrt();
    println!("density {rho} kg/m^3, [100] velocities {v:?} m/s, sqrt(B11/rho) {longitudinal} m/s");
    assert!((v[2] - longitudinal).abs() < 1e-2 * longitudinal);

    // The material card in rotated axes: the stiffness is the Bond rotation
    // of the analytic tensor, and the CSV file carries the same numbers.
    let path = std::env::temp_dir().join(format!("dft_card_{}.csv", std::process::id()));
    let (c, s) = (0.6, 0.8);
    let rotation = Value::Array(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
            .iter()
            .map(|r| Value::Array(r.iter().map(|x| Value::Float(*x)).collect()))
            .collect(),
    );
    let card = run(
        "material-card",
        &helium(),
        &controls(&[
            ("rotation", rotation),
            ("card_path", Value::Str(path.to_str().unwrap().into())),
        ]),
    )
    .unwrap();
    let stiffness = flat(&card, "stiffness_gpa");
    let csv = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let c11: f64 = csv
        .lines()
        .find(|l| l.starts_with("C,1,1,"))
        .unwrap()
        .split(',')
        .nth(3)
        .unwrap()
        .parse()
        .unwrap();
    assert!((c11 - stiffness[0]).abs() < 1e-9 * stiffness[0].abs());
    // Rotation about z leaves C33 and the bulk modulus unchanged.
    let sym_birch = |i: usize, j: usize| 0.5 * (birch[6 * i + j] + birch[6 * j + i]);
    assert!((stiffness[14] - sym_birch(2, 2)).abs() < 1e-8 * sym_birch(2, 2).abs());
    assert!(
        (number(&card, "hill_bulk_gpa") - number(&moduli, "hill_bulk_gpa")).abs()
            < 1e-8 * cubic.abs()
    );
}
