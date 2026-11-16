// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic empirical dispersion on the common driver: the energy, forces and
//! strain derivative of `periodic-energy` and `stress` with `dispersion`
//! equal those without it plus the lattice sum of
//! `periodic::dispersion::periodic_d3` (whose own four-step fits are in its
//! unit tests); the strain-derivative terms add to the total; phonons add
//! the force constants of the lattice sum, the remaining features without
//! dispersion second derivatives refuse the control; and the CLI route
//! agrees with the Rust driver.

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::math::Vec3;
use dft_rs::periodic::dispersion::{
    periodic_d3, periodic_dispersion_hessian, PeriodicDispersion, ThreeBody, DEFAULT_CN_CUTOFF,
    DEFAULT_DISP_CUTOFF,
};
use dft_rs::periodic::Lattice;
use std::process::Command;

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.5, 0.1, 0.05]];

fn chain() -> System {
    System {
        numbers: vec![1, 1],
        positions: POSITIONS.to_vec(),
        cell: Some(CELL),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("pbe".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(12.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(1)),
        ("conv_tol".to_string(), Value::Float(1e-11)),
        ("conv_tol_dp".to_string(), Value::Float(1e-9)),
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

fn direct(
    model: dft_rs::dispersion::Dispersion,
) -> dft_rs::periodic::dispersion::PeriodicDispersionResult {
    let settings =
        PeriodicDispersion::from_model(&model, "pbe", DEFAULT_CN_CUTOFF, DEFAULT_DISP_CUTOFF)
            .unwrap()
            .unwrap();
    let v = |r: [f64; 3]| Vec3::new(r[0], r[1], r[2]);
    let lattice = Lattice::new(v(CELL[0]), v(CELL[1]), v(CELL[2]));
    let positions: Vec<Vec3> = POSITIONS.iter().map(|p| v(*p)).collect();
    periodic_d3(&[1, 1], &positions, &lattice, &settings).unwrap()
}

#[test]
fn stress_with_dispersion_adds_the_lattice_sum() {
    let plain = run("stress", &chain(), &controls(&[])).unwrap();
    for (name, model) in [
        ("d2", dft_rs::dispersion::Dispersion::D2),
        ("d3bj", dft_rs::dispersion::Dispersion::D3Bj),
        ("d3", dft_rs::dispersion::Dispersion::D3Zero),
        ("chg", dft_rs::dispersion::Dispersion::Chg),
    ] {
        let with = run(
            "stress",
            &chain(),
            &controls(&[("dispersion", Value::Str(name.into()))]),
        )
        .unwrap();
        let d3 = direct(model);
        let energy_gap = number(&with, "energy") - number(&plain, "energy") - d3.energy;
        let reported_gap = number(&with, "dispersion_energy") - d3.energy;
        let (fw, fp) = (flat(&with, "forces"), flat(&plain, "forces"));
        let force_gap = (0..6)
            .map(|k| (fw[k] - fp[k] - d3.forces[k / 3].to_array()[k % 3]).abs())
            .fold(0.0_f64, f64::max);
        let (sw, sp) = (
            flat(&with, "strain_derivative"),
            flat(&plain, "strain_derivative"),
        );
        let strain_gap = (0..9)
            .map(|k| (sw[k] - sp[k] - d3.strain_derivative[k / 3][k % 3]).abs())
            .fold(0.0_f64, f64::max);
        let mut sum = [0.0; 9];
        if let Some(Value::Object(terms)) = with.get("strain_derivative_terms") {
            assert!(
                terms.iter().any(|(k, _)| k == "dispersion"),
                "{name}: no dispersion term"
            );
            for (_, term) in terms {
                for (s, x) in sum.iter_mut().zip(term.numeric_array().unwrap().1) {
                    *s += x;
                }
            }
        }
        let term_gap = (0..9)
            .map(|k| (sum[k] - sw[k]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "{name}: E_disp {:.12e}, energy gap {energy_gap:e}, reported gap {reported_gap:e}, force gap {force_gap:e}, \
             strain-derivative gap {strain_gap:e}, term sum gap {term_gap:e}",
            d3.energy
        );
        assert!(d3.energy < 0.0);
        assert!(
            energy_gap.abs() < 1e-10 && reported_gap.abs() < 1e-14,
            "{name}"
        );
        assert!(
            force_gap < 1e-8 && strain_gap < 1e-8 && term_gap < 1e-12,
            "{name}"
        );
    }
}

#[test]
fn features_without_dispersion_derivatives_refuse_it() {
    for feature in ["elastic", "dielectric-tensor"] {
        let err = run(
            feature,
            &chain(),
            &controls(&[("dispersion", Value::Str("d3bj".into()))]),
        );
        println!("{feature}: {:?}", err.as_ref().err());
        assert!(err.is_err(), "{feature} accepted dispersion");
    }
}

#[test]
fn periodic_energy_cli_route_with_dispersion_matches_the_rust_driver() {
    let report = run(
        "periodic-energy",
        &chain(),
        &controls(&[("dispersion", Value::Str("d3bj".into()))]),
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!("dft_d3_driver_{}.xyz", std::process::id()));
    let a = dft_rs::constants::BOHR_TO_ANGSTROM;
    let mut xyz = String::from("2\nchain\n");
    for p in POSITIONS {
        xyz.push_str(&format!(
            "H {:.16} {:.16} {:.16}\n",
            p[0] * a,
            p[1] * a,
            p[2] * a
        ));
    }
    std::fs::write(&path, xyz).unwrap();
    let rows: Vec<String> = CELL
        .iter()
        .map(|r| format!("{:.16},{:.16},{:.16}", r[0] * a, r[1] * a, r[2] * a))
        .collect();
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .args([
            "periodic-energy",
            path.to_str().unwrap(),
            "--cell",
            &rows.join(";"),
            "--xc",
            "pbe",
            "--basis",
            "sto-3g",
            "--k-mesh",
            "2,1,1",
            "--overlap-cutoff",
            "12",
            "--grid-level",
            "2",
            "--grid-shell",
            "1",
            "--conv-tol",
            "1e-11",
            "--conv-tol-dp",
            "1e-9",
            "--max-iter",
            "200",
            "--dispersion",
            "d3bj",
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
    let gap = number(&json, "energy") - number(&report, "energy");
    println!(
        "CLI against Rust driver: energy gap {gap:e}, dispersion {}",
        number(&json, "dispersion_energy")
    );
    assert!(gap.abs() < 1e-9);
    assert!(
        (number(&json, "dispersion_energy") - number(&report, "dispersion_energy")).abs() < 1e-14
    );
}

#[test]
fn a_functional_with_built_in_dispersion_carries_its_lattice_sum() {
    // wB97X-D is defined with the CHG pair term: the periodic energy must
    // include its lattice sum without a request, and refuse a second model.
    let xc = [("xc", Value::Str("wb97xd".into()))];
    let report = run("periodic-energy", &chain(), &controls(&xc)).unwrap();
    let lattice_sum = direct(dft_rs::dispersion::Dispersion::Chg);
    let reported = number(&report, "dispersion_energy");
    println!(
        "wb97xd: energy {:.10}, dispersion {reported:.12e}, lattice sum {:.12e}",
        number(&report, "energy"),
        lattice_sum.energy
    );
    assert!(lattice_sum.energy < 0.0);
    assert!((reported - lattice_sum.energy).abs() < 1e-14);
    let doubled = run(
        "periodic-energy",
        &chain(),
        &controls(&[
            ("xc", Value::Str("wb97xd".into())),
            ("dispersion", Value::Str("d3bj".into())),
        ]),
    );
    assert!(doubled.is_err());
    // Elastic constants of a functional with exact exchange are refused: the
    // strain derivatives of the exchange response are not implemented.
    let refused = run(
        "elastic",
        &chain(),
        &controls(&[("xc", Value::Str("pbe0".into()))]),
    )
    .unwrap_err();
    assert!(refused.to_string().contains("exact exchange"), "{refused}");
}

#[test]
fn phonons_with_dispersion_add_the_force_constants_of_the_lattice_sum() {
    let plain = run("phonons", &chain(), &controls(&[])).unwrap();
    let plain_constants = flat(&plain, "force_constants");
    let v = |r: [f64; 3]| Vec3::new(r[0], r[1], r[2]);
    let lattice = Lattice::new(v(CELL[0]), v(CELL[1]), v(CELL[2]));
    let positions: Vec<Vec3> = POSITIONS.iter().map(|p| v(*p)).collect();
    for (name, model) in [
        ("d2", dft_rs::dispersion::Dispersion::D2),
        ("d3bj", dft_rs::dispersion::Dispersion::D3Bj),
        ("chg", dft_rs::dispersion::Dispersion::Chg),
    ] {
        let with = run(
            "phonons",
            &chain(),
            &controls(&[("dispersion", Value::Str(name.into()))]),
        )
        .unwrap();
        let settings =
            PeriodicDispersion::from_model(&model, "pbe", DEFAULT_CN_CUTOFF, DEFAULT_DISP_CUTOFF)
                .unwrap()
                .unwrap();
        let hessian =
            periodic_dispersion_hessian(&[1, 1], &positions, &lattice, &settings).unwrap();
        let constants = flat(&with, "force_constants");
        let gap = (0..36)
            .map(|k| (constants[k] - plain_constants[k] - hessian[(k / 6, k % 6)]).abs())
            .fold(0.0_f64, f64::max);
        let largest = hessian
            .as_slice()
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        let energy_gap =
            number(&with, "energy") - number(&plain, "energy") - number(&with, "dispersion_energy");
        let (fw, fp) = (
            flat(&with, "frequencies_cm1"),
            flat(&plain, "frequencies_cm1"),
        );
        println!(
            "{name}: largest dispersion force constant {largest:e}, force-constant gap {gap:e}, \
             energy gap {energy_gap:e}, highest frequency {:.4} against {:.4} cm^-1",
            fw[5], fp[5]
        );
        assert!(largest > 1e-6, "{name}");
        assert!(gap < 1e-9, "{name}");
        assert!(energy_gap.abs() < 1e-10, "{name}");
        assert!((fw[5] - fp[5]).abs() > 1e-6, "{name}");
    }
}

#[test]
fn three_body_term_reaches_the_driver() {
    let pair = [("dispersion", Value::Str("d3bj".into()))];
    let triple = [
        ("dispersion", Value::Str("d3bj".into())),
        ("three_body", Value::Bool(true)),
        ("abc_cutoff", Value::Float(16.0)),
    ];
    let without = run("stress", &chain(), &controls(&pair)).unwrap();
    let with = run("stress", &chain(), &controls(&triple)).unwrap();
    let two = PeriodicDispersion::from_model(
        &dft_rs::dispersion::Dispersion::D3Bj,
        "pbe",
        DEFAULT_CN_CUTOFF,
        DEFAULT_DISP_CUTOFF,
    )
    .unwrap()
    .unwrap();
    let v = |r: [f64; 3]| Vec3::new(r[0], r[1], r[2]);
    let lattice = Lattice::new(v(CELL[0]), v(CELL[1]), v(CELL[2]));
    let positions: Vec<Vec3> = POSITIONS.iter().map(|p| v(*p)).collect();
    let direct_two = periodic_d3(&[1, 1], &positions, &lattice, &two).unwrap();
    let three = two
        .with_three_body(ThreeBody {
            cutoff: 16.0,
            ..ThreeBody::default()
        })
        .unwrap();
    let direct_three = periodic_d3(&[1, 1], &positions, &lattice, &three).unwrap();
    let term = direct_three.energy - direct_two.energy;
    let energy_gap = number(&with, "energy") - number(&without, "energy") - term;
    let (fw, fo) = (flat(&with, "forces"), flat(&without, "forces"));
    let force_gap = (0..6)
        .map(|k| {
            let expected = direct_three.forces[k / 3].to_array()[k % 3]
                - direct_two.forces[k / 3].to_array()[k % 3];
            (fw[k] - fo[k] - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    let (sw, so) = (
        flat(&with, "strain_derivative"),
        flat(&without, "strain_derivative"),
    );
    let strain_gap = (0..9)
        .map(|k| {
            let expected = direct_three.strain_derivative[k / 3][k % 3]
                - direct_two.strain_derivative[k / 3][k % 3];
            (sw[k] - so[k] - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    println!(
        "three-body term {term:.6e} Hartree: energy gap {energy_gap:e}, force gap {force_gap:e}, \
         strain-derivative gap {strain_gap:e}"
    );
    assert!(term.abs() > 1e-12);
    assert!(energy_gap.abs() < 1e-10 && force_gap < 1e-8 && strain_gap < 1e-8);
    assert_eq!(with.get("three_body").and_then(Value::as_bool), Some(true));
    // The term needs a D3 model.
    assert!(run(
        "stress",
        &chain(),
        &controls(&[("three_body", Value::Bool(true))])
    )
    .is_err());
    assert!(run(
        "stress",
        &chain(),
        &controls(&[
            ("dispersion", Value::Str("chg".into())),
            ("three_body", Value::Bool(true)),
        ])
    )
    .is_err());
}
