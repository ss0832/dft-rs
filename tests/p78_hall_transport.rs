// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Ordinary Hall response: independent Drude limits, electron/hole signs,
//! four-step Lorentz-direction differences of DFT band velocities, and
//! four-step finite-field fits of an independent Drude resolvent.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::periodic::{prepare_periodic, PeriodicMethod};
use dft_rs::driver::value::ToValue;
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::math::Vec3;
use dft_rs::periodic::band_transport::{
    conductivity_factor, hall_coefficient, hall_coefficient_factor, hall_conductivity_factor,
    hall_integrals, hall_vertex, transport_integrals, BandModel, HallState, TransportState,
};

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("overlap_cutoff".to_string(), Value::Float(14.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(1)),
        ("conv_tol".to_string(), Value::Float(1e-11)),
        ("conv_tol_dp".to_string(), Value::Float(1e-9)),
        ("max_iter".to_string(), Value::Int(300)),
        ("k_mesh".to_string(), vec![4_i64, 1, 1].to_value()),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn chain() -> System {
    System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, 0.0], [1.6, 0.1, 0.05]],
        cell: Some([[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]),
        pbc: [true; 3],
    }
}

fn model() -> BandModel {
    let method = PeriodicMethod::from_controls(&controls(&[])).unwrap();
    let setup = prepare_periodic(&chain(), &method).unwrap();
    let scf =
        dft_rs::periodic::scf::run_periodic_scf(&setup.cell, &setup.functional, &setup.options)
            .unwrap();
    assert!(scf.converged);
    BandModel::new(&setup.cell, &setup.functional, &scf, &setup.options).unwrap()
}

#[test]
fn dft_lorentz_vertices_pass_independent_four_step_velocity_fits() {
    let model = model();
    let k = Vec3::new(0.31, 0.07, 0.05);
    // Small steps resolve the quadratic interval even along the steepest
    // transverse band curvature; every measured error enters the fit.
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for band in [0, 1] {
        let d = model.derivatives(0, k, band, None, 1e-6).unwrap();
        let v = d.velocity.unwrap();
        let vertex = hall_vertex(&v, &d.inverse_mass.unwrap());
        for c in 0..3 {
            // Independent cross product; the direction is frozen at k.
            let field = std::array::from_fn::<_, 3, _>(|a| if a == c { 1.0 } else { 0.0 });
            let cross = Vec3::new(
                v[1] * field[2] - v[2] * field[1],
                v[2] * field[0] - v[0] * field[2],
                v[0] * field[1] - v[1] * field[0],
            );
            let norm = cross.norm();
            assert!(norm > 0.0);
            let direction = cross * (1.0 / norm);
            let mut estimates = Vec::new();
            let mut errors = Vec::new();
            for h in steps {
                let vp = model
                    .derivatives(0, k + direction * h, band, None, 1e-6)
                    .unwrap()
                    .velocity
                    .unwrap();
                let vm = model
                    .derivatives(0, k + direction * (-h), band, None, 1e-6)
                    .unwrap()
                    .velocity
                    .unwrap();
                let estimate: [[f64; 3]; 3] = std::array::from_fn(|a| {
                    std::array::from_fn(|b| v[a] * norm * (vp[b] - vm[b]) / (2.0 * h))
                });
                let error = (0..3)
                    .flat_map(|a| (0..3).map(move |b| (a, b)))
                    .map(|(a, b)| (estimate[a][b] - vertex[a][b][c]).abs())
                    .fold(0.0_f64, f64::max);
                estimates.push(estimate);
                errors.push(error);
            }
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| {
                    ((4.0 * estimates[3][a][b] - estimates[2][a][b]) / 3.0 - vertex[a][b][c]).abs()
                })
                .fold(0.0_f64, f64::max);
            println!(
                "Hall band={band} B-axis={c}, steps={steps:?}, errors={errors:?}, \
                p={:.9}, R2={:.10}, Richardson={richardson:.6e}",
                fit.exponent, fit.r_squared
            );
            assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
            assert!(richardson < 1e-8);
        }
    }
}

#[test]
fn anisotropic_parabolic_electron_and_hole_bands_reproduce_drude_limits() {
    let mass: [f64; 3] = [0.7, 1.1, 1.8];
    let (mu, kt, dq, max_energy) = (0.03_f64, 0.003_f64, 0.013_f64, 0.15_f64);
    let n = ((2.0 * max_energy).sqrt() / dq).ceil() as i32;
    let volume = (2.0 * std::f64::consts::PI / dq).powi(3) / mass.iter().product::<f64>().sqrt();
    let mut states = Vec::new();
    let mut hall_states = Vec::new();
    for i in -n..=n {
        for j in -n..=n {
            for k in -n..=n {
                let q = [i, j, k].map(|v| (v as f64 + 0.5) * dq);
                let energy = q.iter().map(|x| x * x / 2.0).sum::<f64>();
                if energy > max_energy {
                    continue;
                }
                let v = std::array::from_fn(|a| q[a] / mass[a].sqrt());
                states.push(TransportState {
                    energy,
                    velocity2: std::array::from_fn(|a| std::array::from_fn(|b| v[a] * v[b])),
                    count: 1.0,
                });
                hall_states.push(HallState {
                    energy,
                    velocity: v,
                    inverse_mass: std::array::from_fn(|a| {
                        std::array::from_fn(|b| if a == b { 1.0 / mass[a] } else { 0.0 })
                    }),
                });
            }
        }
    }
    let l = transport_integrals(&states, volume, 1, 2.0, mu, kt, 40.0).unwrap();
    let electron = hall_integrals(&hall_states, volume, 1, 2.0, mu, kt, 40.0).unwrap();
    let r = hall_coefficient(&l, &electron).unwrap();
    println!(
        "Drude n={:.9e}, raw Onsager={:.9e}, relative={:?}, R_yxz={:.9e}, n R_yxz={:.9}",
        l.electron_density,
        electron.raw_onsager_residual,
        electron.relative_onsager_residual,
        r[1][0][2],
        l.electron_density * r[1][0][2]
    );
    for a in 0..3 {
        assert!((l.l0[a][a] * mass[a] / l.electron_density - 1.0).abs() < 1e-4);
    }
    assert!((r[1][0][2] * l.electron_density + 1.0).abs() < 1e-4);
    assert!(electron.relative_onsager_residual.unwrap() < 1e-12);
    for s in &mut hall_states {
        s.energy = -s.energy;
        s.velocity = s.velocity.map(|v| -v);
        s.inverse_mass = s.inverse_mass.map(|row| row.map(|v| -v));
    }
    let hole = hall_integrals(&hall_states, volume, 1, 2.0, -mu, kt, 40.0).unwrap();
    let rh = hall_coefficient(&l, &hole).unwrap();
    assert!((rh[1][0][2] * l.electron_density - 1.0).abs() < 1e-4);
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                assert!(
                    (hole.conductivity_derivative_over_tau_squared[a][b][c]
                        + electron.conductivity_derivative_over_tau_squared[a][b][c])
                        .abs()
                        < 1e-13
                );
            }
        }
    }
    // Independent finite-B Drude resolvent of this sampled L0:
    // sigma_xy(B)=-L0_xx B / (m_y + B^2/m_x), tau=1.
    let steps = [0.08, 0.04, 0.02, 0.01];
    let exact = electron.conductivity_derivative_over_tau_squared[0][1][2];
    let estimates: Vec<f64> = steps
        .iter()
        .map(|b| -l.l0[0][0] / (mass[1] + b * b / mass[0]))
        .collect();
    let errors: Vec<f64> = estimates.iter().map(|v| (v - exact).abs()).collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
    println!(
        "Hall Drude finite-B steps={steps:?}, errors={errors:?}, p={:.9}, \
        R2={:.10}, relative Richardson={:.9e}",
        fit.exponent,
        fit.r_squared,
        richardson / exact.abs()
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
    assert!(richardson / exact.abs() < 1e-6);
    // Independent dimensional relation: inversion of the SI conductivity
    // must give the same resistivity derivative as direct unit conversion.
    assert!(
        (hall_conductivity_factor() / conductivity_factor().powi(2) / hall_coefficient_factor()
            - 1.0)
            .abs()
            < 1e-14
    );
}

#[test]
fn hall_driver_preserves_zero_field_transport_and_reports_raw_residuals() {
    let c = controls(&[
        ("transport_k_mesh", vec![8_i64, 2, 2].to_value()),
        ("transport_temperatures_k", vec![2000.0_f64].to_value()),
        ("transport_mu_shifts", vec![0.0_f64].to_value()),
    ]);
    let hall = run("hall-transport", &chain(), &c).unwrap();
    let zero = run("boltzmann-transport", &chain(), &c).unwrap();
    let row = &hall.get("results").unwrap().as_array().unwrap()[0];
    let zrow = &zero.get("results").unwrap().as_array().unwrap()[0];
    assert_eq!(
        row.get("conductivity_over_tau"),
        zrow.get("conductivity_over_tau")
    );
    assert_eq!(row.get("seebeck"), zrow.get("seebeck"));
    let (shape, h) = row
        .get("hall_conductivity_derivative_over_tau_squared")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert_eq!(shape, vec![3, 3, 3]);
    assert!(h.iter().all(|x| x.is_finite()));
    let defect = (0..3)
        .flat_map(|a| (0..3).flat_map(move |b| (0..3).map(move |c| (a, b, c))))
        .map(|(a, b, c)| (h[a * 9 + b * 3 + c] + h[b * 9 + a * 3 + c]).abs())
        .fold(0.0_f64, f64::max);
    let raw = row
        .get("raw_hall_onsager_residual")
        .and_then(Value::as_f64)
        .unwrap();
    println!(
        "DFT Hall driver raw Onsager={raw:.9e}, relative={:?}, Fock defect={:?}",
        row.get("relative_hall_onsager_residual"),
        hall.get("fock_image_residual")
    );
    assert!((defect / hall_conductivity_factor() - raw).abs() < 1e-16);
    assert!(hall
        .get("units")
        .unwrap()
        .get("results.hall_coefficient")
        .is_some());
}

#[test]
fn invalid_inputs_and_singular_conductivity_are_explicit() {
    assert!(hall_integrals(&[], 1.0, 1, 2.0, 0.0, 0.0, 20.0).is_err());
    assert!(hall_integrals(&[], 1.0, 1, 2.0, f64::NAN, 1.0, 20.0).is_err());
    let invalid = HallState {
        energy: 0.0,
        velocity: [f64::NAN; 3],
        inverse_mass: [[0.0; 3]; 3],
    };
    assert!(hall_integrals(&[invalid], 1.0, 1, 2.0, 0.0, 1.0, 20.0).is_err());
    let hall = hall_integrals(&[], 1.0, 1, 2.0, 0.0, 1.0, 20.0).unwrap();
    let l = transport_integrals(&[], 1.0, 1, 2.0, 0.0, 1.0, 20.0).unwrap();
    assert!(hall.relative_onsager_residual.is_none());
    assert!(hall_coefficient(&l, &hall).is_none());
    let c = controls(&[
        ("transport_k_mesh", vec![1_i64, 1, 1].to_value()),
        ("transport_temperatures_k", vec![300.0_f64].to_value()),
        ("transport_mu_shifts", Vec::<f64>::new().to_value()),
    ]);
    assert!(run("hall-transport", &chain(), &c)
        .unwrap_err()
        .to_string()
        .contains("must not be empty"));
}

#[test]
fn unprojected_hall_onsager_relation_converges_on_four_transport_meshes() {
    let system = System {
        numbers: vec![2],
        positions: vec![[0.0; 3]],
        cell: Some([[0.3, 3.1, 2.8], [2.9, 0.2, 3.2], [3.0, 2.7, 0.1]]),
        pbc: [true; 3],
    };
    let mut previous = f64::INFINITY;
    let mut coefficients = Vec::new();
    for mesh in [6_i64, 12, 24, 48] {
        let c = controls(&[
            ("k_mesh", vec![2_i64; 3].to_value()),
            ("overlap_cutoff", Value::Float(12.0)),
            ("guess", Value::Str("core".into())),
            ("transport_k_mesh", vec![mesh; 3].to_value()),
            ("transport_temperatures_k", vec![2000.0_f64].to_value()),
            ("carrier_concentrations_cm3", vec![-1e22_f64].to_value()),
            // Retain the whole thermally weighted band to resolve the
            // integration defect below 1e-10 without a hard 20-kT tail.
            ("delta_window", Value::Float(40.0)),
        ]);
        let report = run("hall-transport", &system, &c).unwrap();
        let row = &report.get("results").unwrap().as_array().unwrap()[0];
        let relative = row
            .get("relative_hall_onsager_residual")
            .and_then(Value::as_f64)
            .unwrap();
        let raw = row
            .get("raw_hall_onsager_residual")
            .and_then(Value::as_f64)
            .unwrap();
        let coefficient = row
            .get("hall_coefficient")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1[11];
        println!(
            "Hall mesh={mesh} window=40 raw={raw:.12e}, relative={relative:.12e}, \
            R_yxz={coefficient:.12e}, diagnostic={:?}",
            row.get("hall_onsager_satisfied")
        );
        assert!(relative < previous);
        previous = relative;
        coefficients.push(coefficient);
        assert!(coefficient > 0.0);
        if mesh == 48 {
            assert!(relative < 1e-10);
            assert_eq!(row.get("hall_onsager_satisfied"), Some(&Value::Bool(true)));
        } else {
            assert_eq!(row.get("hall_onsager_satisfied"), Some(&Value::Bool(false)));
            assert!(report
                .get("warnings")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str().unwrap().contains("Onsager defect")));
        }
    }
    assert!((coefficients[3] / coefficients[2] - 1.0).abs() < 1e-8);
}
