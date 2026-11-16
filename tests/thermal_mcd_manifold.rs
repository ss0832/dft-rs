// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full common-driver gates for the explicitly supplied thermal MCD model.
use dft_rs::driver::{run, Controls, System, Value};

fn request(gap: f64, field: [f64; 3]) -> Value {
    Value::object()
        .with("mcd_model", "thermal-manifold")
        .with(
            "temperature",
            0.08 / dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K,
        )
        .with("bfield", field)
        .with("mcd_thermal_hamiltonian", [[0.0, 0.0], [0.0, gap]])
        .with("mcd_thermal_magnetic_x", [[0.17, 0.23], [0.23, -0.31]])
        .with("mcd_thermal_magnetic_y", [[-0.27, 0.11], [0.11, 0.39]])
        .with("mcd_thermal_magnetic_z", [[0.42, -0.19], [-0.19, -0.13]])
        .with("mcd_thermal_helicity_x", [[0.31, -0.16], [-0.16, 0.12]])
        .with("mcd_thermal_helicity_y", [[-0.23, 0.27], [0.27, 0.41]])
        .with("mcd_thermal_helicity_z", [[0.14, 0.39], [0.39, -0.37]])
}
fn execute(value: &Value) -> Value {
    run(
        "mcd",
        &System::molecule(vec![8, 1], vec![[0.0; 3], [0.0, 0.0, 1.83]]),
        &Controls::from_value(value).unwrap(),
    )
    .unwrap()
}
fn values(r: &Value, key: &str) -> Vec<f64> {
    r.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn actual_driver_all_eighteen_population_and_total_component_fits() {
    for gap in [0.0, 0.017] {
        let base = execute(&request(gap, [0.0; 3]));
        assert_eq!(base.get("geometry_used").unwrap().as_bool(), Some(false));
        let target = values(&base, "mcd_total_thermal_tensor");
        let hs: [f64; 4] = [0.008, 0.004, 0.002, 0.001];
        for axis in 0..3 {
            let mut errors = [[0.0_f64; 4]; 3];
            for (i, h) in hs.iter().enumerate() {
                let mut b = [0.0; 3];
                b[axis] = *h;
                let plus = values(&execute(&request(gap, b)), "mcd_thermal_signal");
                b[axis] = -h;
                let minus = values(&execute(&request(gap, b)), "mcd_thermal_signal");
                for a in 0..3 {
                    errors[a][i] = ((plus[a] - minus[a]) / (2.0 * h) - target[a * 3 + axis]).abs();
                    assert!(errors[a][i] > 0.0);
                }
            }
            for (a, error) in errors.iter().enumerate() {
                let x = hs.map(f64::ln);
                let y = error.map(f64::ln);
                let xm = x.iter().sum::<f64>() / 4.0;
                let ym = y.iter().sum::<f64>() / 4.0;
                let p = x
                    .iter()
                    .zip(y)
                    .map(|(x, y)| (x - xm) * (y - ym))
                    .sum::<f64>()
                    / x.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
                let ss = y.iter().map(|y| (y - ym).powi(2)).sum::<f64>();
                let r2 = 1.0
                    - x.iter()
                        .zip(y)
                        .map(|(x, y)| (y - ym - p * (x - xm)).powi(2))
                        .sum::<f64>()
                        / ss;
                eprintln!("gap={gap} a={a} c={axis} p={p:.10} R2={r2:.12}");
                assert!((p - 2.0).abs() < 0.15 && r2 > 0.999);
            }
        }
    }
}

#[test]
fn actual_driver_rejects_reference_and_invalid_models_without_changing_default() {
    let mut r = request(0.0, [0.0; 3]);
    r.set("reference", "gks");
    assert!(run(
        "mcd",
        &System::default(),
        &Controls::from_value(&r).unwrap()
    )
    .is_err());
    let mut r = request(0.0, [0.0; 3]);
    r.set("mcd_model", "not-a-model");
    assert!(run(
        "mcd",
        &System::default(),
        &Controls::from_value(&r).unwrap()
    )
    .is_err());
    let base = execute(&request(0.0, [0.001, 0.002, 0.003]));
    assert_eq!(base.get("feature").unwrap().as_str(), Some("mcd"));
    let sum: f64 = values(&base, "mcd_thermal_populations").iter().sum();
    assert!((sum - 1.0).abs() < 1e-14);
}

#[test]
fn actual_driver_separate_population_and_mixing_all_eighteen_fits() {
    let zero = request(0.017, [0.0; 3]);
    let base = execute(&zero);
    let c = values(&base, "mcd_population_c_tensor");
    let mixing = values(&base, "mcd_state_mixing_tensor");
    let frozen = |field: [f64; 3]| {
        let mut v = request(0.017, field);
        for axis in ["x", "y", "z"] {
            let key = format!("mcd_thermal_magnetic_{axis}");
            let data = values(&v, &key);
            v.set(&key, [[data[0], 0.0], [0.0, data[3]]]);
        }
        v
    };
    let hs: [f64; 4] = [0.008, 0.004, 0.002, 0.001];
    for axis in 0..3 {
        let mut errors = [[[0.0_f64; 4]; 3]; 2];
        for (i, h) in hs.iter().enumerate() {
            let mut b = [0.0; 3];
            b[axis] = *h;
            let plus = values(&execute(&request(0.017, b)), "mcd_thermal_signal");
            let cp = values(&execute(&frozen(b)), "mcd_thermal_signal");
            b[axis] = -h;
            let minus = values(&execute(&request(0.017, b)), "mcd_thermal_signal");
            let cm = values(&execute(&frozen(b)), "mcd_thermal_signal");
            for a in 0..3 {
                let pop = (cp[a] - cm[a]) / (2.0 * h);
                let total = (plus[a] - minus[a]) / (2.0 * h);
                errors[0][a][i] = (pop - c[a * 3 + axis]).abs();
                errors[1][a][i] = (total - pop - mixing[a * 3 + axis]).abs();
            }
        }
        for (channel, rows) in errors.iter().enumerate() {
            for (a, error) in rows.iter().enumerate() {
                assert!(error.iter().all(|e| *e > 0.0));
                let x = hs.map(f64::ln);
                let y = error.map(f64::ln);
                let xm = x.iter().sum::<f64>() / 4.0;
                let ym = y.iter().sum::<f64>() / 4.0;
                let p = x
                    .iter()
                    .zip(y)
                    .map(|(x, y)| (x - xm) * (y - ym))
                    .sum::<f64>()
                    / x.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
                let r2 = 1.0
                    - x.iter()
                        .zip(y)
                        .map(|(x, y)| (y - ym - p * (x - xm)).powi(2))
                        .sum::<f64>()
                        / y.iter().map(|y| (y - ym).powi(2)).sum::<f64>();
                eprintln!("separate channel={channel} a={a} c={axis} p={p:.10} R2={r2:.12}");
                assert!((p - 2.0).abs() < 0.15 && r2 > 0.999);
            }
        }
    }
}

#[test]
fn actual_driver_finite_field_energy_origin_invariance() {
    let field = [0.017, 0.023, -0.011];
    let base = execute(&request(0.0, field));
    for origin in [-1000.0, -100.0, 1000.0] {
        let mut shifted = request(0.0, field);
        shifted.set("mcd_thermal_hamiltonian", [[origin, 0.0], [0.0, origin]]);
        let r = execute(&shifted);
        for key in [
            "mcd_thermal_signal",
            "mcd_thermal_populations",
            "mcd_thermal_relative_energies",
            "mcd_population_c_tensor",
        ] {
            assert_eq!(values(&r, key), values(&base, key));
        }
        assert_eq!(r.get("mcd_energy_origin").unwrap().as_f64(), Some(origin));
    }
}
