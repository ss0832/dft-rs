// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phonon dispersion and lattice thermodynamics from analytic supercell
//! force constants on the common driver: an independent four-step fit of
//! central differences of the analytic supercell forces (validation only),
//! the commensurate-q identity with the supercell Hessian, agreement with
//! the primitive-cell Gamma force constants at the same Brillouin-zone
//! sampling, the thermodynamic identities and the CLI route.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use std::process::Command;

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]];

fn chain() -> System {
    System {
        numbers: vec![1, 1],
        positions: POSITIONS.to_vec(),
        cell: Some(CELL),
        pbc: [true; 3],
    }
}

/// The 2x1x1 supercell in the order of the driver (primitive atom fastest).
fn supercell(atom: usize, axis: usize, h: f64) -> System {
    let mut positions: Vec<[f64; 3]> = Vec::new();
    for n in 0..2 {
        for p in POSITIONS {
            positions.push(std::array::from_fn(|k| p[k] + n as f64 * CELL[0][k]));
        }
    }
    positions[atom][axis] += h;
    let mut cell = CELL;
    for k in 0..3 {
        cell[0][k] *= 2.0;
    }
    System {
        numbers: vec![1; 4],
        positions,
        cell: Some(cell),
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

fn ints(values: &[i64]) -> Value {
    Value::Array(values.iter().map(|v| Value::Int(*v)).collect())
}

fn rows(values: &[[f64; 3]]) -> Value {
    Value::Array(
        values
            .iter()
            .map(|r| Value::Array(r.iter().map(|v| Value::Float(*v)).collect()))
            .collect(),
    )
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

#[test]
fn supercell_force_constants_reproduce_gamma_and_commensurate_modes() {
    let bands = run(
        "phonon-bands",
        &chain(),
        &controls(&[
            ("supercell", ints(&[2, 1, 1])),
            ("path", rows(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])),
            ("points", Value::Int(4)),
        ]),
    )
    .unwrap();
    assert_eq!(bands.get("converged").and_then(Value::as_bool), Some(true));
    let frequencies = flat(&bands, "frequencies_cm1");
    let n_q = frequencies.len() / 6;
    let gamma = &frequencies[..6];
    let zone_boundary = &frequencies[6 * (n_q - 1)..];
    let fc = flat(&bands, "supercell_force_constants");
    println!(
        "acoustic residual {:e}, force-constant asymmetry {:e}, hermiticity {:e}",
        number(&bands, "acoustic_residual"),
        number(&bands, "force_constant_asymmetry"),
        number(&bands, "dynamical_matrix_hermiticity")
    );
    // Commensurate identity: the supercell modes are the union of q = 0 and
    // q = pi/a.
    let mass = dft_rs::constants::atomic_mass_au(1);
    let mut weighted = dft_rs::linalg::Matrix::zeros(12, 12);
    for x in 0..12 {
        for y in 0..12 {
            weighted[(x, y)] = fc[12 * x + y] / mass;
        }
    }
    // Compare eigenvalues: square roots amplify roundoff near zero.
    let (mut supercell_modes, _) = dft_rs::linalg::symmetric_eigen(&weighted).unwrap();
    supercell_modes.sort_by(|a, b| a.total_cmp(b));
    let mut folded: Vec<f64> = gamma
        .iter()
        .chain(zone_boundary)
        .map(|f| f.signum() * (f / dft_rs::constants::HARTREE_TO_CM1).powi(2))
        .collect();
    folded.sort_by(|a, b| a.total_cmp(b));
    let largest = supercell_modes.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = supercell_modes
        .iter()
        .zip(&folded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("supercell eigenvalues {supercell_modes:?}\nfolded {folded:?}\nlargest difference {worst:e}");
    assert!(worst < 1e-10 * largest, "commensurate identity {worst:e}");
    // The primitive Gamma force constants at the same sampling.
    let primitive = run("phonons", &chain(), &controls(&[])).unwrap();
    let reference = flat(&primitive, "frequencies_cm1");
    let gap = reference
        .iter()
        .zip(gamma)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("Gamma primitive {reference:?}\nGamma supercell {gamma:?}\nlargest difference {gap:e} cm^-1");
    assert!(
        gap < 0.5,
        "Gamma from the supercell against the primitive cell: {gap:e} cm^-1"
    );

    // Independent four-step fit of the supercell force constants against
    // central differences of the analytic supercell forces.
    let forces = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let report = run(
            "periodic-energy",
            &supercell(atom, axis, h),
            &controls(&[("forces", Value::Bool(true)), ("k_mesh", ints(&[2, 1, 1]))]),
        )
        .unwrap();
        flat(&report, "forces")
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    {
        let (atom, axis) = (3usize, 0usize);
        let column = 3 * atom + axis;
        let fd: Vec<Vec<f64>> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (forces(atom, axis, h), forces(atom, axis, -h));
                (0..12)
                    .map(|row| -(plus[row] - minus[row]) / (2.0 * h))
                    .collect()
            })
            .collect();
        let errors: Vec<f64> = fd
            .iter()
            .map(|col| {
                (0..12)
                    .map(|row| (col[row] - fc[12 * row + column]).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = (0..12)
            .map(|row| ((4.0 * fd[3][row] - fd[2][row]) / 3.0 - fc[12 * row + column]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "supercell force constants column ({atom},{axis}): steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {richardson:e}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999,
            "({atom},{axis})"
        );
        assert!(richardson < 2e-5);
    }
}

#[test]
fn lattice_thermodynamics_satisfies_its_identities_and_the_cli_route() {
    let coarse = controls(&[
        ("overlap_cutoff", Value::Float(12.0)),
        ("grid_level", Value::Int(2)),
        ("grid_shell", Value::Int(1)),
        ("supercell", ints(&[2, 1, 1])),
        ("q_mesh", ints(&[4, 1, 1])),
        (
            "temperatures",
            Value::Array(vec![
                Value::Float(0.0),
                Value::Float(300.0),
                Value::Float(600.0),
            ]),
        ),
    ]);
    let report = run("phonon-thermo", &chain(), &coarse).unwrap();
    let t = flat(&report, "temperatures");
    let u = flat(&report, "internal_energy");
    let f = flat(&report, "free_energy");
    let s = flat(&report, "entropy");
    let c = flat(&report, "heat_capacity");
    let zpe = number(&report, "zero_point_energy");
    println!(
        "ZPE {zpe:e}, U {u:?}, F {f:?}, S {s:?}, Cv {c:?}, excluded {:?}",
        report.get("excluded_modes")
    );
    assert!((u[0] - zpe).abs() < 1e-15 && (f[0] - zpe).abs() < 1e-15);
    for i in 0..t.len() {
        assert!(
            (f[i] - (u[i] - t[i] * s[i])).abs() < 1e-14,
            "F = U - TS at {} K",
            t[i]
        );
    }
    assert!(f[2] < f[1] && f[1] < f[0] && s[2] > s[1] && c[2] > c[1]);
    // The CLI route.
    let path = std::env::temp_dir().join(format!("dft_phonon_thermo_{}.xyz", std::process::id()));
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
            "phonon-thermo",
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
            "--supercell",
            "2,1,1",
            "--q-mesh",
            "4,1,1",
            "--temperatures",
            "0,300,600",
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
    for (x, y) in flat(&json, "free_energy").iter().zip(&f) {
        assert!((x - y).abs() < 1e-10, "{x} vs {y}");
    }
}

/// Quasi-harmonic states at a pressure and mode Grueneisen parameters on the
/// driver, on compressed fcc helium. Recorded: volumes 53.532, 53.853,
/// 54.516 Bohr^3 and B_T 46.99, 45.30, 42.79 GPa at 0, 300, 600 K and
/// 10.3 GPa; the single stable zone-boundary mode (272.77 cm^-1) has
/// gamma = 2.2319 at the reference volume against the thermodynamic 2.2166
/// and 2.2549 at the volumes of 300 and 600 K. The two transverse
/// zone-boundary modes are imaginary at every volume and excluded.
#[test]
fn quasi_harmonic_states_at_a_pressure_and_mode_gruneisen_parameters() {
    let helium = System {
        numbers: vec![2],
        positions: vec![[0.0, 0.0, 0.0]],
        cell: Some([[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]]),
        pbc: [true; 3],
    };
    let pressure_gpa = 10.3;
    let floats = |values: &[f64]| Value::Array(values.iter().map(|v| Value::Float(*v)).collect());
    let report = run(
        "quasi-harmonic",
        &helium,
        &controls(&[
            ("k_mesh", ints(&[2, 2, 2])),
            ("overlap_cutoff", Value::Float(12.0)),
            ("grid_level", Value::Int(2)),
            ("grid_shell", Value::Int(1)),
            ("conv_tol", Value::Float(1e-11)),
            ("supercell", ints(&[2, 1, 1])),
            ("q_mesh", ints(&[2, 1, 1])),
            ("scales", floats(&[0.98, 1.0, 1.02, 1.04])),
            ("temperatures", floats(&[0.0, 300.0, 600.0])),
            ("pressure_gpa", Value::Float(pressure_gpa)),
        ]),
    )
    .unwrap();
    let gpa = dft_rs::constants::HARTREE_PER_BOHR3_TO_GPA;
    let pressure = pressure_gpa / gpa;
    let volumes = flat(&report, "volumes");
    let statics = flat(&report, "static_energies");
    let vibrational = flat(&report, "vibrational_free_energies");
    let volume = flat(&report, "volume");
    let free = flat(&report, "free_energy");
    let gibbs = flat(&report, "gibbs_energy");
    let bulk = flat(&report, "bulk_modulus_gpa");
    let thermodynamic = flat(&report, "gruneisen");
    let average = flat(&report, "gruneisen_mode_average");
    println!(
        "volumes {volumes:?}\nV(T) {volume:?}, B_T {bulk:?} GPa\nGrueneisen: thermodynamic {thermodynamic:?}, mode average {average:?}\nmode parameters {:?}\nmode frequencies {:?}\nimaginary modes {:?}",
        report.get("mode_gruneisen"),
        report.get("mode_gruneisen_frequencies_cm1"),
        report.get("imaginary_modes_per_volume")
    );
    assert_eq!((volumes.len(), volume.len(), vibrational.len()), (4, 3, 12));
    assert!((number(&report, "pressure_gpa") - pressure_gpa).abs() < 1e-14);
    for t in 0..3 {
        // The state lies on the fitted isotherm of this temperature at the
        // pressure, recomputed here from the reported tables.
        let total: Vec<f64> = (0..4)
            .map(|v| statics[v] + vibrational[3 * v + t])
            .collect();
        let fit = dft_rs::periodic::eos::fit_birch_murnaghan_isotherm(&volumes, &total).unwrap();
        assert!((fit.pressure(volume[t]) - pressure).abs() < 1e-9 * pressure);
        assert!((fit.bulk_modulus(volume[t]) * gpa - bulk[t]).abs() < 1e-8);
        assert!((fit.energy(volume[t]) - free[t]).abs() < 1e-12);
        assert!((gibbs[t] - free[t] - pressure * volume[t]).abs() < 1e-12);
        assert!(volume[t] > volumes[0] && volume[t] < volumes[3]);
    }
    assert!(volume[2] > volume[1] && volume[1] > volume[0]);
    assert!(bulk[2] < bulk[1] && bulk[1] < bulk[0]);
    // Mode Grueneisen parameters: null for the acoustic and imaginary modes.
    let modes: Vec<Option<f64>> = report
        .get("mode_gruneisen")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .flat_map(|q| q.as_array().unwrap().iter().map(Value::as_f64))
        .collect();
    let stable: Vec<f64> = modes.iter().flatten().copied().collect();
    assert_eq!(modes.len(), 6);
    assert!(!stable.is_empty() && stable.iter().all(|g| g.is_finite() && *g > 0.0));
    // The heat-capacity-weighted mode average against the thermodynamic
    // parameter `alpha B_T V / C_v` of the same model.
    assert_eq!(average[0], 0.0);
    for t in 1..3 {
        assert!(
            (average[t] - thermodynamic[t]).abs() < 0.03 * thermodynamic[t],
            "T index {t}: {} against {}",
            average[t],
            thermodynamic[t]
        );
    }
}
