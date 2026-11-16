// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Gamma-point phonons from analytic coupled-perturbed force constants on the
//! common driver: an independent four-step fit of central differences of
//! the analytic SCF forces (validation only), a Richardson check, the
//! acoustic sum rule, and the CLI route.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use std::process::Command;

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]];

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
            Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(14.0)),
        ("grid_level".to_string(), Value::Int(3)),
        ("grid_shell".to_string(), Value::Int(3)),
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

/// Recorded: over 0.04..0.005 Bohr column (1,0) gave errors 1.294e-3,
/// 3.229e-4, 8.032e-5, 1.994e-5 (p = 2.006675, R2 = 0.99999916) and column
/// (0,1) gave 1.226e-4, 3.045e-5, 7.768e-6, 1.413e-6 (halving ratios 4.03,
/// 3.92, 5.50; p = 2.128805, R2 = 0.99717469): the smallest step reaches the
/// noise of the SCF forces. The Richardson differences were 9.1e-7 and
/// 7.1e-7. The fit uses 0.08..0.01 Bohr.
#[test]
fn gamma_force_constants_pass_a_four_step_fit_of_the_analytic_forces() {
    let report = run("phonons", &chain(0, 0, 0.0), &controls(&[])).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (_, fc) = report
        .get("force_constants")
        .unwrap()
        .numeric_array()
        .unwrap();
    let asymmetry = number(&report, "raw_asymmetry");
    let acoustic = number(&report, "acoustic_residual");
    let (_, frequencies) = report
        .get("frequencies_cm1")
        .unwrap()
        .numeric_array()
        .unwrap();
    println!(
        "raw asymmetry {asymmetry:e}, acoustic residual {acoustic:e}, frequencies {frequencies:?}"
    );
    let forces = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let r = run(
            "periodic-energy",
            &chain(atom, axis, h),
            &controls(&[("forces", Value::Bool(true))]),
        )
        .unwrap();
        r.get("forces").unwrap().numeric_array().unwrap().1
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    for (atom, axis) in [(1usize, 0usize), (0, 1)] {
        let column = 3 * atom + axis;
        let fd: Vec<Vec<f64>> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (forces(atom, axis, h), forces(atom, axis, -h));
                (0..6)
                    .map(|row| -(plus[row] - minus[row]) / (2.0 * h))
                    .collect()
            })
            .collect();
        let errors: Vec<f64> = fd
            .iter()
            .map(|col| {
                (0..6)
                    .map(|row| (col[row] - fc[6 * row + column]).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson: Vec<f64> = (0..6)
            .map(|row| (4.0 * fd[3][row] - fd[2][row]) / 3.0)
            .collect();
        let worst = (0..6)
            .map(|row| (richardson[row] - fc[6 * row + column]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "Gamma force constants column ({atom},{axis}): errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {worst:e}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999,
            "column ({atom},{axis})"
        );
        assert!(worst < 2e-5, "Richardson ({atom},{axis}): {worst:e}");
    }
    assert!(acoustic < 1e-6, "acoustic sum rule residual {acoustic:e}");
    // Three acoustic modes at Gamma.
    let near_zero = frequencies.iter().filter(|f| f.abs() < 5.0).count();
    assert_eq!(near_zero, 3, "{frequencies:?}");
}

#[test]
fn phonon_cli_route_matches_the_rust_driver() {
    let coarse = controls(&[
        ("overlap_cutoff", Value::Float(12.0)),
        ("grid_level", Value::Int(2)),
        ("grid_shell", Value::Int(1)),
        (
            "k_mesh",
            Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
        ),
    ]);
    let report = run("phonons", &chain(0, 0, 0.0), &coarse).unwrap();
    let path = std::env::temp_dir().join(format!("dft_phonons_{}.xyz", std::process::id()));
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
            "phonons",
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
    let (_, cli) = json
        .get("force_constants")
        .unwrap()
        .numeric_array()
        .unwrap();
    let (_, rust) = report
        .get("force_constants")
        .unwrap()
        .numeric_array()
        .unwrap();
    for (a, b) in cli.iter().zip(&rust) {
        assert!((a - b).abs() < 1e-8, "{a} vs {b}");
    }
}

#[test]
fn extended_xyz_lattice_replaces_the_cell_option() {
    let a = dft_rs::constants::BOHR_TO_ANGSTROM;
    let rows: Vec<String> = CELL
        .iter()
        .map(|r| format!("{:.16},{:.16},{:.16}", r[0] * a, r[1] * a, r[2] * a))
        .collect();
    let lattice: Vec<String> = CELL
        .iter()
        .flatten()
        .map(|v| format!("{:.16}", v * a))
        .collect();
    use std::fmt::Write as _;
    let mut atoms = String::new();
    for p in &POSITIONS {
        writeln!(
            &mut atoms,
            "H {:.16} {:.16} {:.16}",
            p[0] * a,
            p[1] * a,
            p[2] * a
        )
        .unwrap();
    }
    let directory = std::env::temp_dir();
    let plain = directory.join(format!("dft_extxyz_plain_{}.xyz", std::process::id()));
    let extended = directory.join(format!("dft_extxyz_lattice_{}.xyz", std::process::id()));
    std::fs::write(&plain, format!("2\nchain\n{atoms}")).unwrap();
    std::fs::write(
        &extended,
        format!(
            "2\nLattice=\"{}\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n{atoms}",
            lattice.join(" ")
        ),
    )
    .unwrap();
    let common = [
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
        "--format",
        "json",
    ];
    let energy = |arguments: &[&str]| -> f64 {
        let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
            .arg("periodic-energy")
            .args(arguments)
            .args(common)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let json = Value::parse_json(&String::from_utf8_lossy(&output.stdout)).unwrap();
        number(&json, "energy")
    };
    let with_option = energy(&[plain.to_str().unwrap(), "--cell", &rows.join(";")]);
    let with_lattice = energy(&[extended.to_str().unwrap()]);
    // An open axis in the comment line is not a three-dimensional crystal.
    std::fs::write(
        &extended,
        format!(
            "2\nLattice=\"{}\" pbc=\"T T F\"\n{atoms}",
            lattice.join(" ")
        ),
    )
    .unwrap();
    let open = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .arg("periodic-energy")
        .arg(extended.to_str().unwrap())
        .args(common)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&plain);
    let _ = std::fs::remove_file(&extended);
    println!("--cell {with_option}, Lattice {with_lattice}");
    assert!((with_option - with_lattice).abs() < 1e-10);
    assert!(!open.status.success());
}

/// A cell whose atoms are all inversion centres: every first-order overlap
/// vanishes by symmetry and the occupied-occupied response density is
/// rounding noise. Before the Hermitian projection of the response density
/// this cell failed with "k-point density 0 is not Hermitian (maximum error
/// 1.88)". Recorded with the projection: column (1,0) over 0.08..0.01 Bohr
/// gave errors 7.892e-4, 1.959e-4, 4.890e-5, 1.222e-5 (p = 2.004223, R2 =
/// 0.99999884) and a Richardson difference of 6.9e-9.
#[test]
fn doubled_helium_cell_with_vanishing_overlap_derivatives_has_phonons() {
    // The coordinates of the failing input: Angstrom values converted to Bohr.
    let a = dft_rs::constants::ANGSTROM_TO_BOHR;
    let edge = 3.175063265418 * a;
    let system = |atom: usize, axis: usize, h: f64| {
        let mut positions = vec![[0.0, 0.0, 0.0], [1.587531632709 * a, 0.0, 0.0]];
        positions[atom][axis] += h;
        System {
            numbers: vec![2, 2],
            positions,
            cell: Some([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
            pbc: [true; 3],
        }
    };
    let settings = |extra: &[(&str, Value)]| {
        let mut all = vec![
            (
                "k_mesh",
                Value::Array(vec![Value::Int(1), Value::Int(1), Value::Int(1)]),
            ),
            // The periodic nuclear attraction is truncated per shell pair
            // like the overlap: at 8 Bohr the He pairs at the cutoff (overlap
            // near 2e-7) enter and leave it as the atom moves, and the force
            // differences stopped converging (errors 7.77e-4, 1.95e-4,
            // 5.09e-5, 1.49e-5 from 0.08 to 0.01 Bohr, p = 1.905,
            // R2 = 0.99927, Richardson difference 2.9e-6).
            ("overlap_cutoff", Value::Float(12.0)),
            ("grid_level", Value::Int(2)),
            ("grid_shell", Value::Int(2)),
        ];
        all.extend(extra.iter().cloned());
        controls(&all)
    };
    let report = run("phonons", &system(0, 0, 0.0), &settings(&[])).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (_, fc) = report
        .get("force_constants")
        .unwrap()
        .numeric_array()
        .unwrap();
    let (_, frequencies) = report
        .get("frequencies_cm1")
        .unwrap()
        .numeric_array()
        .unwrap();
    let acoustic = number(&report, "acoustic_residual");
    println!("doubled helium: acoustic residual {acoustic:e}, frequencies {frequencies:?}");
    let forces = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let r = run(
            "periodic-energy",
            &system(atom, axis, h),
            &settings(&[("forces", Value::Bool(true))]),
        )
        .unwrap();
        r.get("forces").unwrap().numeric_array().unwrap().1
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    let (atom, axis) = (1usize, 0usize);
    let column = 3 * atom + axis;
    let fd: Vec<Vec<f64>> = steps
        .iter()
        .map(|&h| {
            let (plus, minus) = (forces(atom, axis, h), forces(atom, axis, -h));
            (0..6)
                .map(|row| -(plus[row] - minus[row]) / (2.0 * h))
                .collect()
        })
        .collect();
    let errors: Vec<f64> = fd
        .iter()
        .map(|col| {
            (0..6)
                .map(|row| (col[row] - fc[6 * row + column]).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    let worst = (0..6)
        .map(|row| ((4.0 * fd[3][row] - fd[2][row]) / 3.0 - fc[6 * row + column]).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "doubled helium force constants column ({atom},{axis}): steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {worst:e}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999);
    assert!(worst < 2e-5, "Richardson: {worst:e}");
    assert!(acoustic < 1e-6, "acoustic sum rule residual {acoustic:e}");
    let near_zero = frequencies.iter().filter(|f| f.abs() < 5.0).count();
    assert_eq!(near_zero, 3, "{frequencies:?}");
}

/// Exact-exchange force constants of Hartree-Fock, a global hybrid and the
/// range-separated hybrid with its pair dispersion. With a finite exchange
/// cutoff the truncated exchange sum is not symmetric in its two density
/// factors, so the raw matrix is not symmetric; its column `y` is the exact
/// derivative of the analytic gradient, which the four-step fit checks. The
/// symmetrized matrix is the symmetric part of the raw one.
#[test]
fn exact_exchange_force_constants_pass_a_four_step_fit_of_the_analytic_forces() {
    let coarse = |xc: &str, extra: &[(&str, Value)]| {
        let mut all = vec![
            ("xc", Value::Str(xc.into())),
            ("overlap_cutoff", Value::Float(12.0)),
            ("exchange_cutoff", Value::Float(8.0)),
            ("grid_level", Value::Int(2)),
            ("grid_shell", Value::Int(1)),
            (
                "k_mesh",
                Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
            ),
        ];
        all.extend(extra.iter().cloned());
        controls(&all)
    };
    let steps = [0.04, 0.02, 0.01, 0.005];
    for xc in ["hf", "pbe0", "wb97xd"] {
        let report = run("phonons", &chain(0, 0, 0.0), &coarse(xc, &[])).unwrap();
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        let (_, raw) = report
            .get("raw_force_constants")
            .unwrap()
            .numeric_array()
            .unwrap();
        let (_, fc) = report
            .get("force_constants")
            .unwrap()
            .numeric_array()
            .unwrap();
        let asymmetry = number(&report, "raw_asymmetry");
        let acoustic = number(&report, "acoustic_residual");
        let mut gap = 0.0_f64;
        for x in 0..6 {
            for y in 0..6 {
                gap = gap.max((fc[6 * x + y] - 0.5 * (raw[6 * x + y] + raw[6 * y + x])).abs());
            }
        }
        assert!(gap < 1e-14, "{xc}: symmetric part {gap:e}");
        let forces = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
            let r = run(
                "periodic-energy",
                &chain(atom, axis, h),
                &coarse(xc, &[("forces", Value::Bool(true))]),
            )
            .unwrap();
            r.get("forces").unwrap().numeric_array().unwrap().1
        };
        let (atom, axis) = (1usize, 0usize);
        let column = 3 * atom + axis;
        let fd: Vec<Vec<f64>> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (forces(atom, axis, h), forces(atom, axis, -h));
                (0..6)
                    .map(|row| -(plus[row] - minus[row]) / (2.0 * h))
                    .collect()
            })
            .collect();
        let errors: Vec<f64> = fd
            .iter()
            .map(|col| {
                (0..6)
                    .map(|row| (col[row] - raw[6 * row + column]).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let worst = (0..6)
            .map(|row| ((4.0 * fd[3][row] - fd[2][row]) / 3.0 - raw[6 * row + column]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "{xc}: energy {:.10}, raw asymmetry {asymmetry:e}, acoustic residual {acoustic:e}; raw column ({atom},{axis}) steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {worst:e}",
            number(&report, "energy"),
            fit.exponent,
            fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999,
            "{xc}"
        );
        assert!(worst < 2e-5, "{xc} Richardson: {worst:e}");
        assert!(
            acoustic < 1e-6,
            "{xc} acoustic sum rule residual {acoustic:e}"
        );
    }
}

/// Exact-exchange force constants of a Fermi-Dirac reference: an
/// equidistant chain with a global hybrid at an electronic temperature that
/// leaves both bands partially occupied. A raw column is the derivative of
/// the analytic forces of the free energy.
#[test]
fn exact_exchange_force_constants_of_a_fermi_dirac_reference_pass_a_four_step_fit() {
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
    let settings = |extra: &[(&str, Value)]| {
        let mut all = vec![
            ("xc", Value::Str("pbe0".into())),
            ("overlap_cutoff", Value::Float(12.0)),
            ("exchange_cutoff", Value::Float(8.0)),
            ("grid_level", Value::Int(2)),
            ("grid_shell", Value::Int(1)),
            (
                "k_mesh",
                Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
            ),
            ("electronic_temperature", Value::Float(20000.0)),
        ];
        all.extend(extra.iter().cloned());
        controls(&all)
    };
    let report = run("phonons", &equidistant(0, 0, 0.0), &settings(&[])).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (_, raw) = report
        .get("raw_force_constants")
        .unwrap()
        .numeric_array()
        .unwrap();
    let forces = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let r = run(
            "periodic-energy",
            &equidistant(atom, axis, h),
            &settings(&[("forces", Value::Bool(true))]),
        )
        .unwrap();
        r.get("forces").unwrap().numeric_array().unwrap().1
    };
    let steps = [0.04, 0.02, 0.01, 0.005];
    for (atom, axis) in [(1usize, 1usize), (0, 0)] {
        let column = 3 * atom + axis;
        let fd: Vec<Vec<f64>> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (forces(atom, axis, h), forces(atom, axis, -h));
                (0..6)
                    .map(|row| -(plus[row] - minus[row]) / (2.0 * h))
                    .collect()
            })
            .collect();
        let errors: Vec<f64> = fd
            .iter()
            .map(|col| {
                (0..6)
                    .map(|row| (col[row] - raw[6 * row + column]).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let worst = (0..6)
            .map(|row| ((4.0 * fd[3][row] - fd[2][row]) / 3.0 - raw[6 * row + column]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "metallic pbe0: energy {:.10}, raw asymmetry {:e}; raw column ({atom},{axis}) steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {worst:e}",
            number(&report, "energy"),
            number(&report, "raw_asymmetry"),
            fit.exponent,
            fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
        assert!(worst < 1e-5);
    }
}
