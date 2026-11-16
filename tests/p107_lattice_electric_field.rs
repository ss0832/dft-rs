// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Uniform electric field in a periodic cell on the common driver: the
//! susceptibility and its field derivative against independent four-step
//! fits of the Berry-phase dipole of finite-field self-consistent states,
//! the displacement derivative of the susceptibility against fits of the
//! susceptibility of displaced cells, and the finite-field forces against
//! fits of the finite-field energy (finite differences are validation
//! only). Every component is fitted: the nine susceptibility elements, the
//! 27 field derivatives (mixed fields for b != c), the 54 Raman elements
//! (nine susceptibility elements for each of the six coordinates) and the
//! six forces; a component that fails on the first interval is recomputed
//! on a finer and then a coarser one.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use components::{assert_no_failures, fit_components};

const AXES: [&str; 3] = ["x", "y", "z"];

use dft_rs::driver::{run, Controls, System, Value};

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]];

/// The chain with Cartesian coordinate `coord` (3 atom + axis) moved by
/// `shift`.
fn chain(coord: usize, shift: f64) -> System {
    let mut positions = POSITIONS.to_vec();
    positions[coord / 3][coord % 3] += shift;
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
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions (the partition-weight cutoff otherwise
        // adds noise near 1e-13 to displaced gradients; see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(20.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(2)),
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

fn values(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn field(vector: [f64; 3]) -> Value {
    Value::Array(vector.iter().map(|v| Value::Float(*v)).collect())
}

/// Finite-field state of the chain displaced along `coord`.
fn state(coord: usize, shift: f64, vector: [f64; 3], forces: bool) -> Value {
    run(
        "periodic-efield",
        &chain(coord, shift),
        &controls(&[("field", field(vector)), ("forces", Value::Bool(forces))]),
    )
    .unwrap()
}

#[test]
fn susceptibilities_pass_four_step_fits_of_the_finite_field_dipole_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [4e-3, 3e-3, 2e-3, 1.5e-3],
        [2e-3, 1.5e-3, 1e-3, 7.5e-4],
        [8e-3, 6e-3, 4e-3, 3e-3],
    ];
    let report = run("periodic-beta", &chain(0, 0.0), &controls(&[])).unwrap();
    let volume = number(&report, "volume");
    let chi = values(&report, "susceptibility");
    let second = values(&report, "second_susceptibility");
    assert_eq!((chi.len(), second.len()), (9, 27));
    println!("susceptibility {chi:?}");
    println!("field derivative of the susceptibility {second:?}");
    let dipole = |vector: [f64; 3]| values(&state(0, 0.0, vector, false), "dipole");
    let centre = dipole([0.0; 3]);
    let mut failures = Vec::new();
    for b in 0..3 {
        let along = |h: f64| {
            let mut v = [0.0; 3];
            v[b] = h;
            dipole(v)
        };
        let mut names: Vec<String> = (0..3)
            .map(|a| format!("chi_{}{}", AXES[a], AXES[b]))
            .collect();
        names.extend((0..3).map(|a| format!("d chi_{}{} / dE_{}", AXES[a], AXES[b], AXES[b])));
        fit_components(
            &names,
            &INTERVALS,
            |steps| {
                let mut errors = vec![[0.0; 4]; 6];
                for (s, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (along(h), along(-h));
                    for a in 0..3 {
                        errors[a][s] =
                            ((plus[a] - minus[a]) / (2.0 * h) / volume - chi[3 * a + b]).abs();
                        errors[3 + a][s] =
                            ((plus[a] - 2.0 * centre[a] + minus[a]) / (h * h) / volume
                                - second[9 * a + 3 * b + b])
                                .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    for (b, c) in [(0, 1), (0, 2), (1, 2)] {
        let mixed = |x: f64, y: f64| {
            let mut v = [0.0; 3];
            v[b] = x;
            v[c] = y;
            dipole(v)
        };
        let mut names: Vec<String> = (0..3)
            .map(|a| format!("d chi_{}{} / dE_{}", AXES[a], AXES[b], AXES[c]))
            .collect();
        names.extend((0..3).map(|a| format!("d chi_{}{} / dE_{}", AXES[a], AXES[c], AXES[b])));
        fit_components(
            &names,
            &INTERVALS,
            |steps| {
                let mut errors = vec![[0.0; 4]; 6];
                for (s, &h) in steps.iter().enumerate() {
                    let (pp, pm, mp, mm) = (mixed(h, h), mixed(h, -h), mixed(-h, h), mixed(-h, -h));
                    for a in 0..3 {
                        let value = (pp[a] - pm[a] - mp[a] + mm[a]) / (4.0 * h * h) / volume;
                        errors[a][s] = (value - second[9 * a + 3 * b + c]).abs();
                        errors[3 + a][s] = (value - second[9 * a + 3 * c + b]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    assert_no_failures("periodic susceptibilities", &failures);
}

#[test]
fn raman_tensor_passes_four_step_fits_of_the_susceptibility_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.08, 0.04, 0.02, 0.01],
        [0.04, 0.02, 0.01, 0.005],
        [0.16, 0.08, 0.04, 0.02],
    ];
    let report = run("periodic-raman", &chain(0, 0.0), &controls(&[])).unwrap();
    let raman = values(&report, "raman_tensor");
    assert_eq!(raman.len(), 54);
    let born = values(&report, "born_charges");
    assert_eq!(born.len(), 18);
    // Acoustic sum rule of the Born charges.
    let mut sum_rule = 0.0_f64;
    for axis in 0..3 {
        for a in 0..3 {
            let total: f64 = (0..2).map(|atom| born[3 * (3 * atom + axis) + a]).sum();
            sum_rule = sum_rule.max(total.abs());
        }
    }
    println!("Raman tensor {raman:?}; Born charge sum rule {sum_rule:e}");
    assert!(sum_rule < 1e-5);
    let chi = |coord: usize, shift: f64| {
        values(
            &run("periodic-beta", &chain(coord, shift), &controls(&[])).unwrap(),
            "susceptibility",
        )
    };
    let mut failures = Vec::new();
    for k in 0..6 {
        // d chi_ab / du_k is stored as [(3 a + b) 6 + k].
        let names: Vec<String> = (0..9)
            .map(|ab| {
                format!(
                    "d chi_{}{} / d{}{}",
                    AXES[ab / 3],
                    AXES[ab % 3],
                    AXES[k % 3],
                    k / 3
                )
            })
            .collect();
        fit_components(
            &names,
            &INTERVALS,
            |steps| {
                let mut errors = vec![[0.0; 4]; 9];
                for (s, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (chi(k, h), chi(k, -h));
                    for ab in 0..9 {
                        errors[ab][s] =
                            ((plus[ab] - minus[ab]) / (2.0 * h) - raman[6 * ab + k]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    assert_no_failures("periodic Raman tensor", &failures);
}

#[test]
fn finite_field_forces_pass_four_step_fits_of_the_finite_field_energy_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.04, 0.03, 0.02, 0.015],
        [0.02, 0.015, 0.01, 0.0075],
        [0.08, 0.06, 0.04, 0.03],
    ];
    let vector = [2e-3, 0.0, 0.0];
    let forces = values(&state(0, 0.0, vector, true), "forces");
    assert_eq!(forces.len(), 6);
    println!("finite-field forces {forces:?}");
    let energy =
        |coord: usize, shift: f64| number(&state(coord, shift, vector, false), "field_energy");
    let mut failures = Vec::new();
    for k in 0..6 {
        fit_components(
            &[format!("finite-field force F_{}{}", AXES[k % 3], k / 3)],
            &INTERVALS,
            |steps| {
                let mut errors = [0.0; 4];
                for (s, &h) in steps.iter().enumerate() {
                    errors[s] = (-(energy(k, h) - energy(k, -h)) / (2.0 * h) - forces[k]).abs();
                }
                vec![errors]
            },
            &mut failures,
        );
    }
    assert_no_failures("finite-field forces", &failures);
}
