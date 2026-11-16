// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent Pauli-matrix and physical active-space acceptance.
use dft_rs::driver::{run, Controls, System, Value};

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn settings(field: [f64; 3], frozen: bool, phase: f64) -> Value {
    let h = [0.012, 0.017, 0.009];
    let norm = dot(h, h).sqrt();
    let axis = h.map(|v| v / norm);
    let m = [
        [0.17, 0.23, -0.31],
        [-0.27, 0.11, 0.39],
        [0.42, -0.19, -0.13],
    ];
    let q = [
        [0.31, -0.16, 0.12],
        [-0.23, 0.27, 0.41],
        [0.14, 0.39, -0.37],
    ];
    let mut v = Value::object()
        .with("mcd_model", "thermal-complex-manifold")
        .with(
            "temperature",
            0.08 / dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K,
        )
        .with("bfield", field);
    let mut matrix = |key: &str, a: [f64; 3]| {
        // Diagonal unitary U=diag(1,exp(i*phase)) applied to every operator.
        let re = a[0] * phase.cos() + a[1] * phase.sin();
        let im = a[0] * phase.sin() - a[1] * phase.cos();
        v.set(key, [[a[2], re], [re, -a[2]]]);
        v.set(&format!("{key}_imag"), [[0., im], [-im, 0.]]);
    };
    matrix("mcd_thermal_hamiltonian", h);
    for c in 0..3 {
        let a = if frozen {
            axis.map(|x| x * dot(m[c], axis))
        } else {
            m[c]
        };
        matrix(&format!("mcd_thermal_magnetic_{}", ["x", "y", "z"][c]), a);
    }
    for c in 0..3 {
        matrix(
            &format!("mcd_thermal_helicity_{}", ["x", "y", "z"][c]),
            q[c],
        );
    }
    v
}
fn execute(v: &Value) -> Value {
    run("mcd", &System::default(), &Controls::from_value(v).unwrap()).unwrap()
}
fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn complex_pauli_all_twenty_seven_component_fits_and_phase_covariance() {
    let h = [0.012, 0.017, 0.009];
    let r = dot(h, h).sqrt();
    let n = h.map(|v| v / r);
    let kt = 0.08;
    let m = [
        [0.17, 0.23, -0.31],
        [-0.27, 0.11, 0.39],
        [0.42, -0.19, -0.13],
    ];
    let q = [
        [0.31, -0.16, 0.12],
        [-0.23, 0.27, 0.41],
        [0.14, 0.39, -0.37],
    ];
    let base = execute(&settings([0.; 3], false, 0.));
    let rotated = execute(&settings([0.; 3], false, 0.73));
    let keys = [
        "mcd_total_thermal_tensor",
        "mcd_population_c_tensor",
        "mcd_state_mixing_tensor",
    ];
    let mut target = [[[0.; 3]; 3]; 3];
    for a in 0..3 {
        for c in 0..3 {
            let longitudinal = dot(q[a], n) * dot(m[c], n);
            target[1][a][c] = (1. - (r / kt).tanh().powi(2)) * longitudinal / kt;
            target[2][a][c] = (r / kt).tanh() / r * (dot(q[a], m[c]) - longitudinal);
            target[0][a][c] = target[1][a][c] + target[2][a][c];
            for channel in 0..3 {
                assert!(
                    (values(&base, keys[channel])[3 * a + c] - target[channel][a][c]).abs() < 1e-11
                );
                assert!(
                    (values(&rotated, keys[channel])[3 * a + c] - target[channel][a][c]).abs()
                        < 1e-11
                );
            }
        }
    }
    let hs: [f64; 4] = [0.008, 0.004, 0.002, 0.001];
    let mut errors = [[[[0.; 4]; 3]; 3]; 3];
    for c in 0..3 {
        for (i, h) in hs.iter().enumerate() {
            let mut b = [0.; 3];
            b[c] = *h;
            let plus = values(&execute(&settings(b, false, 0.)), "mcd_thermal_signal");
            let cp = values(&execute(&settings(b, true, 0.)), "mcd_thermal_signal");
            b[c] = -h;
            let minus = values(&execute(&settings(b, false, 0.)), "mcd_thermal_signal");
            let cm = values(&execute(&settings(b, true, 0.)), "mcd_thermal_signal");
            for a in 0..3 {
                let total = (plus[a] - minus[a]) / (2. * h);
                let population = (cp[a] - cm[a]) / (2. * h);
                for (channel, value) in [total, population, total - population].iter().enumerate() {
                    errors[channel][a][c][i] = (value - target[channel][a][c]).abs();
                }
            }
        }
    }
    for channel in 0..3 {
        for a in 0..3 {
            for c in 0..3 {
                let x = hs.map(f64::ln);
                let e = errors[channel][a][c];
                assert!(e.iter().all(|x| *x > 0.));
                let y = e.map(f64::ln);
                let xm = x.iter().sum::<f64>() / 4.;
                let ym = y.iter().sum::<f64>() / 4.;
                let p = x
                    .iter()
                    .zip(y)
                    .map(|(x, y)| (x - xm) * (y - ym))
                    .sum::<f64>()
                    / x.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
                let r2 = 1.
                    - x.iter()
                        .zip(y)
                        .map(|(x, y)| (y - ym - p * (x - xm)).powi(2))
                        .sum::<f64>()
                        / y.iter().map(|y| (y - ym).powi(2)).sum::<f64>();
                eprintln!("complex channel={channel} a={a} c={c} p={p:.10} R2={r2:.12}");
                assert!((p - 2.).abs() < 0.15 && r2 > 0.999);
            }
        }
    }
}

#[test]
fn complex_input_is_never_silently_projected_or_symmetrized() {
    let mut v = settings([0.; 3], false, 0.);
    v.set("mcd_thermal_hamiltonian_imag", [[0., 0.1], [0.1, 0.]]);
    assert!(run(
        "mcd",
        &System::default(),
        &Controls::from_value(&v).unwrap()
    )
    .is_err());
    let report = execute(&settings([0.003, -0.002, 0.004], false, 0.));
    assert_eq!(values(&report, "mcd_thermal_populations").len(), 2);
    assert!(
        (values(&report, "mcd_thermal_populations")
            .iter()
            .sum::<f64>()
            - 1.)
            .abs()
            < 1e-14
    );
}

#[test]
fn generated_spin_complete_hydrogen_ci_is_orbital_generator_invariant() {
    let system = System::molecule(vec![1, 1], vec![[0.; 3], [0., 0., 1.4]]);
    for xc in ["hf", "pbe"] {
        let v = Value::object()
            .with("mcd_model", "molecular-casci-soc")
            .with("xc", xc)
            .with("basis", "sto-3g")
            .with("grid_level", 1)
            .with("grid_weight_cutoff", 0.)
            .with("conv_tol", 1e-11)
            .with("conv_tol_dp", 1e-10)
            .with("temperature", 4.2)
            .with("bfield", [0.; 3])
            .with("frequencies", [0.1, 0.2, 0.3])
            .with("soc_scale", 0.);
        let r = run("mcd", &system, &Controls::from_value(&v).unwrap()).unwrap();
        let mut labelled = r.clone();
        assert!(dft_rs::driver::units::complete(&mut labelled).is_empty());
        let energy = values(&r, "state_energies_relative")[0]
            + r.get("energy_origin").unwrap().as_f64().unwrap()
            + 1. / 1.4;
        assert!(
            (energy - (-1.1372759437826976)).abs() < 1e-9,
            "{xc}: {energy}"
        );
        assert!(values(&r, "mcd_cross_section")
            .iter()
            .all(|v| v.abs() < 1e-11));
        assert!(r
            .get("model")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("not TDDFT"));
        let spin_squared = values(&r, "state_spin_squared");
        assert!(spin_squared[0].abs() < 1e-10);
        assert!((spin_squared.iter().sum::<f64>() - 6.).abs() < 1e-10);
        assert!(spin_squared
            .iter()
            .all(|s| s.abs() < 1e-10 || (s - 2.).abs() < 1e-10));
    }
}
