// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Continuum rock physics on the common driver (no electronic structure):
//! aggregate bounds ordering, Gassmann in its classical form and its
//! limits, Biot consistency, refusal of missing physical inputs, and the
//! CLI route without a structure file.

use dft_rs::driver::{run, Controls, System, Value};
use std::process::Command;

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn pair(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn sandstone(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("mineral_bulk_gpa", Value::Float(37.0)),
        ("mineral_shear_gpa", Value::Float(44.0)),
        ("mineral_density_kg_m3", Value::Float(2650.0)),
        ("porosity", Value::Float(0.2)),
        ("fluid_bulk_gpa", Value::Float(2.25)),
        ("fluid_density_kg_m3", Value::Float(1000.0)),
        ("frame", Value::Str("critical-porosity".into())),
        ("critical_porosity", Value::Float(0.4)),
    ];
    pairs.extend(extra.iter().cloned());
    controls(&pairs)
}

#[test]
fn aggregate_bounds_and_fluid_substitution_follow_their_closed_forms() {
    let phases = Value::Array(
        [[0.6, 37.0, 44.0], [0.3, 76.8, 32.0], [0.1, 21.0, 7.0]]
            .iter()
            .map(|r| Value::Array(r.iter().map(|v| Value::Float(*v)).collect()))
            .collect(),
    );
    let m = run(
        "aggregate-moduli",
        &System::default(),
        &controls(&[("phases", phases)]),
    )
    .unwrap();
    let (v, r) = (pair(&m, "voigt_gpa"), pair(&m, "reuss_gpa"));
    let (lo, hi) = (
        pair(&m, "hashin_shtrikman_lower_gpa"),
        pair(&m, "hashin_shtrikman_upper_gpa"),
    );
    for i in 0..2 {
        assert!(r[i] <= lo[i] + 1e-12 && lo[i] <= hi[i] + 1e-12 && hi[i] <= v[i] + 1e-12);
    }
    let rock = run("rock-physics", &System::default(), &sandstone(&[])).unwrap();
    let (k_s, phi, k_f) = (37.0, 0.2, 2.25);
    let k_dry = number(&rock, "dry_bulk_gpa");
    assert!((k_dry - k_s * (1.0 - phi / 0.4)).abs() < 1e-12);
    let gassmann =
        k_dry + (1.0 - k_dry / k_s).powi(2) / (phi / k_f + (1.0 - phi) / k_s - k_dry / (k_s * k_s));
    assert!((number(&rock, "saturated_bulk_gpa") - gassmann).abs() < 1e-12 * gassmann);
    // Biot: K_u = K_dry + alpha^2 M and B = alpha M / K_u.
    let (alpha, m_biot) = (
        number(&rock, "biot_coefficient"),
        number(&rock, "biot_modulus_gpa"),
    );
    assert!((gassmann - (k_dry + alpha * alpha * m_biot)).abs() < 1e-12 * gassmann);
    assert!((number(&rock, "skempton_b") - alpha * m_biot / gassmann).abs() < 1e-14);
    // No defaults for physical inputs.
    let missing = controls(&[("mineral_bulk_gpa", Value::Float(37.0))]);
    assert!(run("rock-physics", &System::default(), &missing).is_err());
    // CLI route without a structure file.
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .args([
            "rock-physics",
            "--mineral-bulk-gpa",
            "37",
            "--mineral-shear-gpa",
            "44",
            "--mineral-density-kg-m3",
            "2650",
            "--porosity",
            "0.2",
            "--fluid-bulk-gpa",
            "2.25",
            "--fluid-density-kg-m3",
            "1000",
            "--frame",
            "critical-porosity",
            "--critical-porosity",
            "0.4",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json = Value::parse_json(&String::from_utf8_lossy(&output.stdout)).unwrap();
    assert!((number(&json, "saturated_bulk_gpa") - gassmann).abs() < 1e-12 * gassmann);
}

#[test]
fn spheroidal_pore_frames_and_conductivity_bounds_reach_the_driver() {
    let base = |frame: &str, extra: &[(&str, Value)]| {
        let mut pairs = vec![
            ("mineral_bulk_gpa", Value::Float(37.0)),
            ("mineral_shear_gpa", Value::Float(44.0)),
            ("mineral_density_kg_m3", Value::Float(2650.0)),
            ("porosity", Value::Float(0.15)),
            ("fluid_bulk_gpa", Value::Float(2.25)),
            ("fluid_density_kg_m3", Value::Float(1000.0)),
            ("frame", Value::Str(frame.into())),
        ];
        pairs.extend(extra.iter().cloned());
        controls(&pairs)
    };
    let rock = |frame: &str, extra: &[(&str, Value)]| {
        run("rock-physics", &System::default(), &base(frame, extra))
    };
    let sphere = ("pore_aspect_ratio", Value::Float(1.0));
    // Dry spherical pores: Kuster-Toksoz coincides with the upper
    // Hashin-Shtrikman bound, and the other estimates lie below it.
    let hs = rock("hashin-shtrikman", &[]).unwrap();
    let kt = rock("kuster-toksoz", std::slice::from_ref(&sphere)).unwrap();
    let dem = rock("dem", std::slice::from_ref(&sphere)).unwrap();
    let sc = rock("self-consistent", std::slice::from_ref(&sphere)).unwrap();
    for key in ["dry_bulk_gpa", "dry_shear_gpa"] {
        let upper = number(&hs, key);
        println!(
            "{key}: HS upper {upper}, KT {}, DEM {}, SC {}",
            number(&kt, key),
            number(&dem, key),
            number(&sc, key)
        );
        assert!((number(&kt, key) - upper).abs() < 1e-9 * upper);
        assert!(number(&dem, key) < upper && number(&dem, key) > 0.0);
        assert!(number(&sc, key) < upper && number(&sc, key) > 0.0);
    }
    // Flat pores soften the frame more than spheres at the same porosity.
    let flat = rock("dem", &[("pore_aspect_ratio", Value::Float(0.1))]).unwrap();
    assert!(number(&flat, "dry_bulk_gpa") < number(&dem, "dry_bulk_gpa"));
    assert!(number(&flat, "dry_shear_gpa") < number(&dem, "dry_shear_gpa"));
    // The aspect ratio is a required physical input.
    assert!(rock("dem", &[]).is_err());
    // Thermal conductivity: both inputs or neither; ordered means and bounds.
    let solid = ("mineral_conductivity_w_mk", Value::Float(7.7));
    let fluid = ("fluid_conductivity_w_mk", Value::Float(0.6));
    let thermal = rock("dem", &[sphere.clone(), solid.clone(), fluid]).unwrap();
    let t = thermal.get("thermal_conductivity_w_mk").unwrap();
    let (h, lo, g, hi, a) = (
        number(t, "harmonic"),
        number(t, "hashin_shtrikman_lower"),
        number(t, "geometric"),
        number(t, "hashin_shtrikman_upper"),
        number(t, "arithmetic"),
    );
    println!(
        "conductivity: harmonic {h}, HS lower {lo}, geometric {g}, HS upper {hi}, arithmetic {a}"
    );
    assert!(h <= lo && lo <= g && g <= hi && hi <= a);
    assert!((a - (0.85 * 7.7 + 0.15 * 0.6)).abs() < 1e-12);
    assert!(rock("dem", &[sphere, solid]).is_err());
}
