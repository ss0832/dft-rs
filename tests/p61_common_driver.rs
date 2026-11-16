// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance tests for the common calculation driver: the Rust driver, the
//! `dft-rs` CLI (JSON and earlier text output) and direct library calls must
//! agree, the analytic gradient is checked by an independent multi-step
//! error fit, and multipoles obey the exact origin-shift laws.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use std::process::Command;

const WATER: [(u8, [f64; 3]); 3] = [
    (8, [0.0, 0.0, 0.1173]),
    (1, [0.0, 1.4332, -0.9314]),
    (1, [0.0, -1.4332, -0.9314]),
];

fn water() -> System {
    System::molecule(
        WATER.iter().map(|(z, _)| *z).collect(),
        WATER.iter().map(|(_, r)| *r).collect(),
    )
}

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

fn tight() -> Vec<(&'static str, Value)> {
    vec![
        ("basis", Value::Str("sto-3g".into())),
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-12)),
    ]
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

struct TempFile(std::path::PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn water_xyz(tag: &str) -> TempFile {
    let path = std::env::temp_dir().join(format!("dft_driver_{tag}_{}.xyz", std::process::id()));
    let a = dft_rs::constants::BOHR_TO_ANGSTROM;
    let mut text = String::from("3\nwater\n");
    for (z, r) in WATER {
        text.push_str(&format!(
            "{} {:.16} {:.16} {:.16}\n",
            dft_rs::z_to_symbol(z).unwrap(),
            r[0] * a,
            r[1] * a,
            r[2] * a
        ));
    }
    std::fs::write(&path, text).unwrap();
    TempFile(path)
}

fn cli(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .args(args)
        .output()
        .unwrap();
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn driver_energy_matches_the_library_scf_and_both_cli_outputs() {
    let report = run("energy", &water(), &controls(&tight())).unwrap();
    assert_eq!(
        report.get("feature").and_then(Value::as_str),
        Some("energy")
    );
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let mol = dft_rs::Molecule::new(
        WATER
            .iter()
            .map(|(z, r)| dft_rs::Atom {
                z: *z,
                position: dft_rs::Vec3::new(r[0], r[1], r[2]),
            })
            .collect(),
    );
    let mb = dft_rs::basis::MolBasis::build(&mol, &dft_rs::basis::data::load("sto-3g").unwrap())
        .unwrap();
    let opts = dft_rs::ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-13f64.sqrt(),
        ..Default::default()
    };
    let scf = dft_rs::run_hf(&mol, &mb, &opts).unwrap();
    let energy = number(&report, "energy");
    assert!(
        (energy - scf.energy).abs() < 1e-10,
        "{energy} vs {}",
        scf.energy
    );

    let file = water_xyz("energy");
    let path = file.0.to_str().unwrap();
    let common = [
        "--basis",
        "sto-3g",
        "--conv-tol",
        "1e-13",
        "--conv-tol-dp",
        "1e-12",
    ];
    let mut args = vec!["energy", path];
    args.extend(common);
    args.extend(["--format", "json"]);
    let (ok, stdout, stderr) = cli(&args);
    assert!(ok, "{stderr}");
    let json = Value::parse_json(&stdout).unwrap();
    // The XYZ file round-trips positions through Angstrom, so agreement is
    // to SCF precision rather than bit-for-bit.
    assert!((number(&json, "energy") - energy).abs() < 1e-11);

    // The earlier text form stays available and agrees to its printed digits.
    let mut args = vec!["energy", path];
    args.extend(common);
    args.extend(["--log-mode", "none"]);
    let (ok, stdout, stderr) = cli(&args);
    assert!(ok, "{stderr}");
    let line = stdout
        .lines()
        .find(|l| l.starts_with("Total energy:"))
        .expect("legacy total energy line");
    let legacy: f64 = line.split_whitespace().nth(2).unwrap().parse().unwrap();
    assert!((legacy - energy).abs() < 2e-12, "{legacy} vs {energy}");
}

#[test]
fn analytic_gradient_passes_an_independent_four_step_error_fit() {
    let report = run("gradient", &water(), &controls(&tight())).unwrap();
    let gradient = vector(&report, "gradient");
    assert!(number(&report, "translation_residual") < 1e-8);
    // Component: oxygen z (index 2) and hydrogen y (index 4).
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for component in [2usize, 4] {
        let (atom, axis) = (component / 3, component % 3);
        let energy_at = |h: f64| {
            let mut system = water();
            system.positions[atom][axis] += h;
            number(
                &run("energy", &system, &controls(&tight())).unwrap(),
                "energy",
            )
        };
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| ((energy_at(h) - energy_at(-h)) / (2.0 * h) - gradient[component]).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "gradient component {component}: steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.02, "p = {}", fit.exponent);
        assert!(fit.r_squared > 0.99999, "R2 = {}", fit.r_squared);
    }
}

#[test]
fn dipole_and_multipoles_obey_origin_laws_and_match_the_cli() {
    let mut base = tight();
    base.push(("origin", Value::Str("zero".into())));
    let report = run("dipole", &water(), &controls(&base)).unwrap();
    let mu0 = vector(&report, "dipole");
    // Neutral molecule: the dipole does not depend on the origin.
    let mut shifted = tight();
    shifted.push((
        "origin",
        Value::Array(vec![
            Value::Float(0.3),
            Value::Float(-0.7),
            Value::Float(1.1),
        ]),
    ));
    let mu1 = vector(
        &run("dipole", &water(), &controls(&shifted)).unwrap(),
        "dipole",
    );
    for a in 0..3 {
        assert!((mu0[a] - mu1[a]).abs() < 1e-11);
    }
    // Cation: mu(O') = mu(O) - Q (O' - O).
    let mut cation = shifted.clone();
    cation.push(("charge", Value::Int(1)));
    cation.push(("multiplicity", Value::Int(2)));
    let at_shift = vector(
        &run("dipole", &water(), &controls(&cation)).unwrap(),
        "dipole",
    );
    cation[3] = ("origin", Value::Str("zero".into()));
    let at_zero = vector(
        &run("dipole", &water(), &controls(&cation)).unwrap(),
        "dipole",
    );
    for (a, d) in [0.3, -0.7, 1.1].iter().enumerate() {
        assert!((at_shift[a] - (at_zero[a] - d)).abs() < 1e-10, "axis {a}");
    }

    // Multipoles: rank 0 is the total charge, rank 1 the dipole, and raw
    // rank-2 moments follow M'(ab) = M(ab) - d_a mu_b - d_b mu_a + d_a d_b Q.
    let mut multipole = base.clone();
    multipole.push(("max_rank", Value::Int(3)));
    let m0 = run("molecular-multipole", &water(), &controls(&multipole)).unwrap();
    let ranks = m0.get("ranks").unwrap().as_array().unwrap();
    assert!(vector(&ranks[0], "cartesian")[0].abs() < 1e-10);
    let dipole = vector(&ranks[1], "cartesian");
    for a in 0..3 {
        assert!((dipole[a] - mu0[a]).abs() < 1e-12);
    }
    let mut moved = shifted.clone();
    moved.push(("max_rank", Value::Int(2)));
    let m1 = run("molecular-multipole", &water(), &controls(&moved)).unwrap();
    let raw0 = vector(&ranks[2], "cartesian");
    let raw1 = vector(
        &m1.get("ranks").unwrap().as_array().unwrap()[2],
        "cartesian",
    );
    let d = [0.3, -0.7, 1.1];
    // cart_components(2): xx, xy, xz, yy, yz, zz.
    let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    for (c, &(a, b)) in pairs.iter().enumerate() {
        let expected = raw0[c] - d[a] * mu0[b] - d[b] * mu0[a];
        assert!(
            (raw1[c] - expected).abs() < 1e-10,
            "component {c}: {} vs {expected}",
            raw1[c]
        );
    }
    // Spherical rank-2 m=0 equals the traceless zz component.
    let traceless = vector(&ranks[2], "traceless");
    let spherical = vector(&ranks[2], "spherical");
    assert!((spherical[0] - traceless[5]).abs() < 1e-12);

    // The CLI reports the same numbers through the same driver.
    let file = water_xyz("dipole");
    let path = file.0.to_str().unwrap();
    let (ok, stdout, stderr) = cli(&[
        "dipole",
        path,
        "--basis",
        "sto-3g",
        "--conv-tol",
        "1e-13",
        "--conv-tol-dp",
        "1e-12",
        "--origin",
        "zero",
        "--format",
        "json",
    ]);
    assert!(ok, "{stderr}");
    let json = Value::parse_json(&stdout).unwrap();
    let cli_mu = vector(&json, "dipole");
    for a in 0..3 {
        assert!(
            (cli_mu[a] - mu0[a]).abs() < 1e-11,
            "{} vs {}",
            cli_mu[a],
            mu0[a]
        );
    }
}

#[test]
fn unread_controls_and_mismatched_systems_are_refused_by_name() {
    let mut misspelled = tight();
    misspelled.push(("orign", Value::Str("zero".into())));
    let error = run("dipole", &water(), &controls(&misspelled))
        .unwrap_err()
        .to_string();
    assert!(error.contains("orign"), "{error}");
    let file = water_xyz("refuse");
    let (ok, _, stderr) = cli(&["orbitals", file.0.to_str().unwrap(), "--grid-levle", "2"]);
    assert!(!ok && stderr.contains("grid_levle"), "{stderr}");
    let mut periodic = water();
    periodic.cell = Some([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
    periodic.pbc = [true; 3];
    assert!(run("energy", &periodic, &controls(&tight())).is_err());
    let registry = run("feature-registry", &System::default(), &Controls::new()).unwrap();
    let features = registry.get("features").unwrap().as_array().unwrap();
    assert!(features
        .iter()
        .any(|f| f.get("id").and_then(Value::as_str) == Some("molecular-multipole")));
    let (ok, stdout, stderr) = cli(&["feature-registry", "--format", "json"]);
    assert!(ok, "{stderr}");
    assert_eq!(
        Value::parse_json(&stdout).unwrap().get("features"),
        registry.get("features")
    );
}

#[test]
fn exports_write_verified_files_including_fractional_occupations() {
    let dir = std::env::temp_dir();
    let molden = dir.join(format!("dft_driver_{}.molden", std::process::id()));
    let saved = dir.join(format!("dft_driver_orbitals_{}.json", std::process::id()));
    let _cleanup = (TempFile(molden.clone()), TempFile(saved.clone()));
    let mut pairs = tight();
    pairs.push(("path", Value::Str(molden.to_string_lossy().into_owned())));
    let report = run("molden", &water(), &controls(&pairs)).unwrap();
    assert!(number(&report, "occupation_density_deviation") < 1e-12);
    let text = std::fs::read_to_string(&molden).unwrap();
    assert_eq!(text.matches(" Occup= 2.000000").count(), 5);
    assert_eq!(text.matches(" Occup=").count(), 7);

    // A smeared reference: the export carries fractional occupations and
    // the saved file reproduces the printed report.
    let mut hot = tight();
    hot.push(("electronic_temperature", Value::Float(20000.0)));
    hot.push(("path", Value::Str(saved.to_string_lossy().into_owned())));
    let report = run("orbitals-save", &water(), &controls(&hot)).unwrap();
    assert!(number(&report, "occupation_density_deviation") < 1e-12);
    let file = Value::parse_json(&std::fs::read_to_string(&saved).unwrap()).unwrap();
    assert_eq!(
        file.get("feature").and_then(Value::as_str),
        Some("orbitals-save")
    );
    let channel = &file.get("channels").unwrap().as_array().unwrap()[0];
    let (_, occupations) = channel.get("occupations").unwrap().numeric_array().unwrap();
    assert!((occupations.iter().sum::<f64>() - 10.0).abs() < 1e-10);
    assert!(
        occupations.iter().any(|f| *f > 1e-12 && *f < 2.0 - 1e-12),
        "fractional occupations"
    );
    assert_eq!(file.get("energy"), report.get("energy"));
}
