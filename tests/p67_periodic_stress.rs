// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic stress of the self-consistent finite-k energy on the common
//! driver: an independent four-step fit of the SCF energy under homogeneous
//! strain, a Richardson check, the term decomposition and the CLI route; and
//! a variable-cell relaxation driven by the analytic forces and stress.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use std::process::Command;

/// A hydrogen chain with two atoms per cell along x and a skewed cell, so
/// that shear components and all lattice directions carry a strain response.
const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.5, 0.1, 0.05]];

fn strained(eps: [[f64; 3]; 3]) -> System {
    let map = |v: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|a| v[a] + (0..3).map(|b| eps[a][b] * v[b]).sum::<f64>())
    };
    System {
        numbers: vec![1, 1],
        positions: POSITIONS.iter().map(|p| map(*p)).collect(),
        cell: Some(CELL.map(map)),
        pbc: [true; 3],
    }
}

fn unstrained() -> System {
    strained([[0.0; 3]; 3])
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(18.0)),
        ("grid_level".to_string(), Value::Int(4)),
        ("grid_shell".to_string(), Value::Int(4)),
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

fn tensor(v: &Value, key: &str) -> [[f64; 3]; 3] {
    let (_, data) = v.get(key).unwrap().numeric_array().unwrap();
    std::array::from_fn(|a| std::array::from_fn(|b| data[3 * a + b]))
}

/// Recorded with the earlier hard-sphere Becke neighbourhood (kept as the
/// history of the failed intervals): with two Becke shells the [0][0] errors
/// over 0.04..0.005 were
/// 2.27e-3, 5.71e-4, 1.48e-4, 4.08e-5 (p = 1.934, R2 = 0.99978), a
/// strain-independent offset near 5e-6 from the images that cross the
/// Becke neighbourhood. The Richardson difference at 0.01/0.005 was
/// -4.98e-6 with two shells, -3.29e-6 with three and -2.05e-7 with four, so
/// the fit uses four shells. With four shells [0][0] gave p = 1.998296
/// (R2 = 0.99999919, Richardson difference -2.05e-7) and the shear [0][1]
/// gave errors 1.196e-5, 3.137e-6, 8.239e-7, 2.155e-7 with a constant
/// halving ratio of 3.81 (p = 1.931182, R2 = 0.99999960) and a Richardson
/// difference of -1.27e-8, so the exponent bound is 0.1.
///
/// With the smoothly switched Becke partition (four shells) the energy has
/// no jumps and the Richardson differences fall to 5.7e-9 ([0][0]), -1.0e-9
/// ([0][1]) and 4.5e-9 ([2][1]). The shear [2][1] has a small h^2
/// coefficient, so over 0.04..0.005 its h^4 term contaminates the largest
/// steps: errors 1.234e-5, 2.548e-6, 5.856e-7, 1.430e-7 with halving ratios
/// 4.84, 4.35, 4.09 (p = 2.141, R2 = 0.99935), a failed interval. The
/// resolved interval 0.01, 0.007, 0.005, 0.0035 gives errors 5.856e-7,
/// 2.824e-7, 1.430e-7, 6.977e-8 (p = 2.026, R2 = 0.999996), so that column
/// uses it; the thresholds are unchanged.
#[test]
fn scf_stress_passes_a_four_step_strain_fit() {
    let report = run("stress", &unstrained(), &controls(&[])).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let derivative = tensor(&report, "strain_derivative");
    let rotation = number(&report, "rotation_residual");
    println!("dE/de = {derivative:?}, rotation residual {rotation:e}");
    // The terms add to the total.
    let mut sum = [[0.0; 3]; 3];
    if let Some(Value::Object(terms)) = report.get("strain_derivative_terms") {
        for (_, term) in terms {
            let (_, data) = term.numeric_array().unwrap();
            for a in 0..3 {
                for b in 0..3 {
                    sum[a][b] += data[3 * a + b];
                }
            }
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            assert!((sum[a][b] - derivative[a][b]).abs() < 1e-12);
        }
    }
    let energy = |eps: [[f64; 3]; 3]| {
        number(
            &run("periodic-energy", &strained(eps), &controls(&[])).unwrap(),
            "energy",
        )
    };
    let unit = |a: usize, b: usize, h: f64| {
        let mut e = [[0.0; 3]; 3];
        e[a][b] = h;
        e
    };
    let halving = [0.04, 0.02, 0.01, 0.005];
    let resolved = [0.01, 0.007, 0.005, 0.0035];
    // The shear [0][1] also has a finer interval: on 0.04..0.005 its halving
    // ratios 4.34, 4.10, 4.03 (p = 2.053, R2 = 0.99985) carry an h^4 term.
    let finer = [0.02, 0.01, 0.005, 0.0025];
    for (a, b, intervals) in [
        (0usize, 0usize, vec![halving]),
        (0, 1, vec![halving, finer]),
        (2, 1, vec![resolved]),
    ] {
        let mut accepted = None;
        for steps in intervals {
            let analytic = derivative[a][b];
            let fd = |h: f64| (energy(unit(a, b, h)) - energy(unit(a, b, -h))) / (2.0 * h);
            let values: Vec<f64> = steps.iter().map(|&h| fd(h)).collect();
            let errors: Vec<f64> = values.iter().map(|v| (v - analytic).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            // Richardson extrapolation of the two finest steps (error ~ h^2);
            // for a halved step this is (4 f(h/2) - f(h)) / 3.
            let (coarse, fine) = (steps[2] * steps[2], steps[3] * steps[3]);
            let richardson = (coarse * values[3] - fine * values[2]) / (coarse - fine);
            println!(
                "SCF dE/de[{a}][{b}]: analytic {analytic:.12e} steps {steps:?} errors {errors:?} \
             p={:.6} R2={:.8}; Richardson {richardson:.12e} (difference {:e})",
                fit.exponent,
                fit.r_squared,
                richardson - analytic
            );
            if (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.9999 {
                accepted = Some((steps, values, analytic));
                break;
            }
            println!("SCF dE/de[{a}][{b}] on {steps:?}: FAILED");
        }
        let Some((steps, values, analytic)) = accepted else {
            panic!("[{a}][{b}]");
        };
        let (coarse, fine) = (steps[2] * steps[2], steps[3] * steps[3]);
        let richardson = (coarse * values[3] - fine * values[2]) / (coarse - fine);
        assert!(
            (richardson - analytic).abs() < 2e-6,
            "[{a}][{b}] Richardson"
        );
    }
    // The atom-centred angular quadrature is fixed in the laboratory frame,
    // so a rigid rotation changes Exc: the antisymmetric part of dE/de is a
    // property of the quadrature (measured 5.47e-5 Hartree at grid level 4
    // with two Becke shells). The [0][1] and [2][1] fits above validate it
    // component by component; here it only has to stay small against the
    // strain derivative itself.
    let largest = derivative
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(rotation < 1e-3 * largest, "rotation residual {rotation:e}");
}

#[test]
fn stress_cli_route_matches_the_rust_driver() {
    let report = run(
        "stress",
        &unstrained(),
        &coarse(&[("k_mesh", mesh([2, 1, 1]))]),
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!("dft_stress_driver_{}.xyz", std::process::id()));
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
            "stress",
            path.to_str().unwrap(),
            "--cell",
            &rows.join(";"),
            "--xc",
            "svwn5",
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
    let (cli, rust) = (tensor(&json, "stress"), tensor(&report, "stress"));
    for a in 0..3 {
        for b in 0..3 {
            assert!(
                (cli[a][b] - rust[a][b]).abs() < 1e-9,
                "{a}{b}: {} vs {}",
                cli[a][b],
                rust[a][b]
            );
        }
    }
    assert!((number(&json, "pressure_gpa") - number(&report, "pressure_gpa")).abs() < 1e-4);
}

fn coarse(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("overlap_cutoff".to_string(), Value::Float(12.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(1)),
        ("conv_tol".to_string(), Value::Float(1e-11)),
    ];
    pairs.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn mesh(k: [i64; 3]) -> Value {
    Value::Array(k.iter().map(|v| Value::Int(*v)).collect())
}

/// Face-centred cubic helium, one atom per primitive cell: every strain
/// direction has a restoring stress, so the relaxation is a well-posed
/// minimization of the enthalpy at the external pressure.
fn helium(edge: f64) -> System {
    let h = 0.5 * edge;
    System {
        numbers: vec![2],
        positions: vec![[0.0, 0.0, 0.0]],
        cell: Some([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pbc: [true; 3],
    }
}

/// The variable-cell relaxation minimizes the enthalpy with the analytic
/// gradient: the accepted enthalpies decrease monotonically and the final
/// analytic stress equals minus the external pressure. A uniform strain of
/// the relaxed cell then raises the enthalpy on both sides, an independent
/// check that the stationary point is a minimum of `E + pV`.
#[test]
fn cell_relaxation_reaches_the_target_stress_and_an_enthalpy_minimum() {
    let gpa = dft_rs::constants::HARTREE_PER_BOHR3_TO_GPA;
    let pressure_gpa = 5.0;
    let report = run(
        "cell-opt",
        &helium(6.0),
        &coarse(&[
            ("k_mesh", mesh([2, 2, 2])),
            ("pressure_gpa", Value::Float(pressure_gpa)),
            ("stress_tol", Value::Float(2e-6)),
            ("gtol", Value::Float(1e-4)),
            ("max_steps", Value::Int(30)),
        ]),
    )
    .unwrap();
    let trajectory = report.get("trajectory").unwrap().as_array().unwrap();
    let enthalpies: Vec<f64> = trajectory.iter().map(|f| number(f, "enthalpy")).collect();
    let volumes: Vec<f64> = trajectory.iter().map(|f| number(f, "volume")).collect();
    println!("cell-opt: enthalpy {enthalpies:?}, volume {volumes:?}");
    assert_eq!(
        report
            .get("optimization_converged")
            .and_then(Value::as_bool),
        Some(true)
    );
    assert!(enthalpies.windows(2).all(|w| w[1] <= w[0]));
    let stress = tensor(&report, "stress");
    let p = number(&report, "external_pressure");
    assert!((p * gpa - pressure_gpa).abs() < 1e-12);
    for a in 0..3 {
        for b in 0..3 {
            let target = if a == b { -p } else { 0.0 };
            assert!(
                (stress[a][b] - target).abs() < 2e-6,
                "sigma[{a}][{b}] = {}",
                stress[a][b]
            );
        }
    }
    // Enthalpy of the relaxed structure under a uniform strain.
    let (_, rows) = report.get("cell").unwrap().numeric_array().unwrap();
    let enthalpy = |scale: f64| {
        let cell: [[f64; 3]; 3] =
            std::array::from_fn(|r| std::array::from_fn(|c| rows[3 * r + c] * scale));
        let system = System {
            numbers: vec![2],
            positions: vec![[0.0, 0.0, 0.0]],
            cell: Some(cell),
            pbc: [true; 3],
        };
        let r = run(
            "periodic-energy",
            &system,
            &coarse(&[("k_mesh", mesh([2, 2, 2]))]),
        )
        .unwrap();
        number(&r, "energy") + p * number(&r, "volume")
    };
    let relaxed = number(&report, "enthalpy");
    for scale in [0.99, 1.01] {
        let strained = enthalpy(scale);
        println!("uniform scale {scale}: enthalpy {strained:.10} vs relaxed {relaxed:.10}");
        assert!(strained > relaxed);
    }
}

/// The fixed-cell relaxation lowers the energy monotonically and stops where
/// the analytic forces are below `gtol`; the lattice is unchanged.
#[test]
fn fixed_cell_relaxation_reaches_zero_force() {
    let report = run(
        "opt-fixed-cell",
        &unstrained(),
        &coarse(&[
            ("k_mesh", mesh([4, 1, 1])),
            ("gtol", Value::Float(2e-4)),
            ("max_steps", Value::Int(40)),
        ]),
    )
    .unwrap();
    let trajectory = report.get("trajectory").unwrap().as_array().unwrap();
    let objectives: Vec<f64> = trajectory.iter().map(|f| number(f, "objective")).collect();
    println!("opt-fixed-cell: objective {objectives:?}");
    assert_eq!(
        report
            .get("optimization_converged")
            .and_then(Value::as_bool),
        Some(true)
    );
    assert!(objectives.windows(2).all(|w| w[1] <= w[0]));
    let (_, forces) = report.get("forces").unwrap().numeric_array().unwrap();
    assert!(forces.iter().all(|f| f.abs() < 2e-4), "{forces:?}");
    assert!(
        (number(&report, "volume")
            - number(
                &run(
                    "periodic-energy",
                    &unstrained(),
                    &coarse(&[("k_mesh", mesh([4, 1, 1]))])
                )
                .unwrap(),
                "volume"
            ))
        .abs()
            < 1e-12
    );
}
