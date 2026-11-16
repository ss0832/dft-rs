// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic calculations on the common driver: energy and analytic forces
//! (independent four-step fit), electronic structure, band structure
//! consistency with the SCF mesh, DOS normalization and the CLI route.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use std::process::Command;

/// A hydrogen chain with two atoms per cell along x (3D cell with vacuum).
fn chain(shift: f64) -> System {
    System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, 0.0], [1.5 + shift, 0.1, 0.0]],
        cell: Some([[3.6, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]]),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(12.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(1)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(200)),
    ];
    pairs.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

/// Controls with converged image, grid and exchange cutoffs: real-space
/// truncations make the energy non-smooth at the 1e-9 level, which is what a
/// finite-difference oracle with small steps resolves.
fn converged(extra: &[(&str, Value)]) -> Controls {
    let mut pairs: Vec<(String, Value)> = Vec::new();
    let replace = ["overlap_cutoff", "grid_level", "grid_shell"];
    if let Value::Object(base) = controls(&[]).to_value() {
        for (k, v) in base {
            if !replace.contains(&k.as_str()) {
                pairs.push((k, v));
            }
        }
    }
    pairs.push(("overlap_cutoff".into(), Value::Float(18.0)));
    pairs.push(("grid_level".into(), Value::Int(4)));
    pairs.push(("grid_shell".into(), Value::Int(2)));
    pairs.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

/// Recorded: with coarse cutoffs (overlap 12, grid level 2, one grid shell)
/// the 0.02..0.0025 interval gave errors 4.5e-5, 2.8e-6, 4.0e-6, 1.1e-5
/// (p = 0.547, R2 = 0.154): cutoff discontinuities dominate. With converged
/// cutoffs the same interval gave 5.4e-5, 1.33e-5, 3.4e-6, 1.39e-6; the last
/// halving ratio (2.45) is limited by an energy noise floor near 2.5e-9, and
/// 0.04..0.005 still gave R2 = 0.99994830 (p = 2.004491) and 0.08..0.01 gave
/// R2 = 0.99998760 (p = 2.014900; halving ratios 4.014, 4.020, 4.098). The
/// interval 0.16..0.02 is fitted, and a Richardson combination of the two
/// smallest steps bounds the analytic error independently.
#[test]
fn periodic_energy_forces_pass_a_four_step_fit() {
    let report = run(
        "periodic-energy",
        &chain(0.0),
        &converged(&[("forces", Value::Bool(true))]),
    )
    .unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    assert!(number(&report, "translation_residual") < 1e-6);
    let gradient = report.get("gradient").unwrap().numeric_array().unwrap().1;
    let analytic = gradient[3]; // atom 1, x
    let steps = [0.16, 0.08, 0.04, 0.02];
    let energy = |h: f64| {
        number(
            &run("periodic-energy", &chain(h), &converged(&[])).unwrap(),
            "energy",
        )
    };
    let errors: Vec<f64> = steps
        .iter()
        .map(|&h| ((energy(h) - energy(-h)) / (2.0 * h) - analytic).abs())
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "periodic dE/dx: analytic {analytic:.12e} steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
    let fd = |h: f64| (energy(h) - energy(-h)) / (2.0 * h);
    let richardson = (4.0 * fd(0.02) - fd(0.04)) / 3.0;
    println!("Richardson derivative {richardson:.12e}, analytic {analytic:.12e}");
    assert!((richardson - analytic).abs() < 2e-6);
}

#[test]
fn bands_reproduce_mesh_eigenvalues_and_dos_counts_the_electrons() {
    let structure = run("electronic-structure", &chain(0.0), &controls(&[])).unwrap();
    assert!(number(&structure, "electron_count_residual").abs() < 1e-8);
    let channel = &structure.get("channels").unwrap().as_array().unwrap()[0];
    let (_, mesh) = channel
        .get("band_energies")
        .unwrap()
        .numeric_array()
        .unwrap();
    let nbands = mesh.len() / 4;
    let edges = channel.get("band_edges").unwrap();
    assert!(number(edges, "gap") > 0.0);
    // Mesh points 0 and 1/4 along a* lie on this path.
    let path = Value::Array(vec![
        Value::Array(vec![
            Value::Float(0.0),
            Value::Float(0.0),
            Value::Float(0.0),
        ]),
        Value::Array(vec![
            Value::Float(0.5),
            Value::Float(0.0),
            Value::Float(0.0),
        ]),
    ]);
    let bands = run(
        "bands",
        &chain(0.0),
        &controls(&[("path", path), ("points", Value::Int(2))]),
    )
    .unwrap();
    let (_, along) = bands.get("channels").unwrap().as_array().unwrap()[0]
        .get("band_energies")
        .unwrap()
        .numeric_array()
        .unwrap();
    let reduced = structure
        .get("k_points_reduced")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    // Path point 1 is k = 1/4 a*; find it on the SCF mesh.
    let index = (0..4)
        .find(|&k| (reduced[3 * k] - 0.25).abs() < 1e-12)
        .expect("mesh point 1/4");
    for b in 0..nbands {
        let difference = (along[nbands + b] - mesh[index * nbands + b]).abs();
        assert!(difference < 1e-8, "band {b}: {difference:e}");
        let difference = (along[b] - mesh[b]).abs();
        assert!(difference < 1e-8, "band {b} at Gamma: {difference:e}");
    }
    // Integrated DOS at mid-gap equals the electron count per cell.
    let dos = run(
        "dos",
        &chain(0.0),
        &controls(&[("sigma", Value::Float(0.002)), ("points", Value::Int(4000))]),
    )
    .unwrap();
    let energies = dos.get("energies").unwrap().numeric_array().unwrap().1;
    let integrated = dos.get("channels").unwrap().as_array().unwrap()[0]
        .get("integrated")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    let mid = 0.5 * (number(edges, "valence_maximum") + number(edges, "conduction_minimum"));
    let i = energies.iter().position(|e| *e >= mid).unwrap();
    assert!(
        (integrated[i] - 2.0).abs() < 1e-6,
        "integrated DOS at mid-gap {}",
        integrated[i]
    );
}

#[test]
fn periodic_cli_route_matches_the_rust_driver() {
    let report = run("periodic-energy", &chain(0.0), &controls(&[])).unwrap();
    let path = std::env::temp_dir().join(format!("dft_periodic_driver_{}.xyz", std::process::id()));
    let a = dft_rs::constants::BOHR_TO_ANGSTROM;
    std::fs::write(
        &path,
        format!("2\nchain\nH 0 0 0\nH {:.16} {:.16} 0\n", 1.5 * a, 0.1 * a),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .args([
            "periodic-energy",
            path.to_str().unwrap(),
            "--cell",
            &format!(
                "{:.16},0,0;0,{:.16},0;0,0,{:.16}",
                3.6 * a,
                7.0 * a,
                7.0 * a
            ),
            "--xc",
            "svwn5",
            "--basis",
            "sto-3g",
            "--k-mesh",
            "4,1,1",
            "--overlap-cutoff",
            "12",
            "--grid-level",
            "2",
            "--grid-shell",
            "1",
            "--conv-tol",
            "1e-12",
            "--conv-tol-dp",
            "1e-10",
            "--max-iter",
            "200",
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
    assert!((number(&json, "energy") - number(&report, "energy")).abs() < 1e-9);
}
