// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Divide-and-conquer Kohn-Sham of a periodic cell on the common driver:
//! independence of the buffer once it covers the cell, and independent
//! four-step fits of the forces and of the stress against finite
//! differences of the free energy (validation only) with an incomplete
//! buffer, where the response of the partition matters: all twelve force
//! components and all nine strain derivatives (`v_a -> v_a + h v_b` of
//! atoms and lattice vectors); a component that fails on the first interval
//! is recomputed on a finer and then a coarser one. The coarser intervals
//! stay below 0.3 Bohr of atomic motion, the margin between the buffer and
//! the nearest atom of the other fragment.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use components::{assert_no_failures, fit_components};

const AXES: [&str; 3] = ["x", "y", "z"];

use dft_rs::driver::{run, Controls, System, Value};

const CELL: [[f64; 3]; 3] = [[7.2, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.45, 0.1, 0.05],
    [3.6, 0.0, 0.1],
    [5.1, -0.1, 0.0],
];

/// Two hydrogen molecules per cell deformed by `v_a -> v_a + strain v_b`
/// (`deformation = (a, b, strain)`), with Cartesian coordinate `coord`
/// (3 atom + axis) displaced by `shift`.
fn cell(deformation: (usize, usize, f64), coord: usize, shift: f64) -> System {
    let (a, b, strain) = deformation;
    let map = |v: [f64; 3]| -> [f64; 3] {
        let mut w = v;
        w[a] += strain * v[b];
        w
    };
    let lattice = CELL.map(map);
    let mut positions: Vec<[f64; 3]> = POSITIONS.iter().map(|p| map(*p)).collect();
    positions[coord / 3][coord % 3] += shift;
    System {
        numbers: vec![1; 4],
        positions,
        cell: Some(lattice),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions (see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        ("overlap_cutoff".to_string(), Value::Float(20.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(2)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(300)),
        (
            "dc_fragments".to_string(),
            Value::Array(vec![
                Value::Int(0),
                Value::Int(0),
                Value::Int(1),
                Value::Int(1),
            ]),
        ),
        ("dc_smearing".to_string(), Value::Float(5e-3)),
        // 1.8 Bohr reaches no atom of the other fragment (nearest 2.10 and
        // 2.15 Bohr, also under the displacement and strain steps below).
        // The first run used 2.6 Bohr, which covers the whole cell: its
        // "partial" free energy equalled the complete-buffer one to 1e-10.
        ("dc_buffer".to_string(), Value::Float(1.8)),
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

fn energy(system: &System, extra: &[(&str, Value)]) -> f64 {
    number(
        &run("periodic-dc", system, &controls(extra)).unwrap(),
        "energy",
    )
}

#[test]
fn complete_buffers_give_one_free_energy() {
    let reference = cell((0, 0, 0.0), 0, 0.0);
    let first = energy(&reference, &[("dc_buffer", Value::Float(12.0))]);
    let second = energy(&reference, &[("dc_buffer", Value::Float(20.0))]);
    let partial = energy(&reference, &[]);
    println!(
        "divide-and-conquer free energy: complete buffers {first:.10} {second:.10}, partial \
         buffer {partial:.10}"
    );
    assert!((first - second).abs() < 1e-9);
    assert!((first - partial).abs() > 1e-7);
}

#[test]
fn forces_pass_four_step_fits_of_the_free_energy_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.04, 0.03, 0.02, 0.015],
        [0.02, 0.015, 0.01, 0.0075],
        [0.08, 0.06, 0.04, 0.03],
    ];
    let still = (0, 0, 0.0);
    let report = run(
        "periodic-dc",
        &cell(still, 0, 0.0),
        &controls(&[("forces", Value::Bool(true))]),
    )
    .unwrap();
    let forces = report.get("forces").unwrap().numeric_array().unwrap().1;
    assert_eq!(forces.len(), 12);
    println!(
        "divide-and-conquer forces {forces:?}; relaxation residual {:e}",
        number(&report, "relaxation_residual")
    );
    let mut failures = Vec::new();
    for k in 0..12 {
        fit_components(
            &[format!(
                "divide-and-conquer force F_{}{}",
                AXES[k % 3],
                k / 3
            )],
            &INTERVALS,
            |steps| {
                let mut errors = [0.0; 4];
                for (s, &h) in steps.iter().enumerate() {
                    let fd = -(energy(&cell(still, k, h), &[]) - energy(&cell(still, k, -h), &[]))
                        / (2.0 * h);
                    errors[s] = (fd - forces[k]).abs();
                }
                vec![errors]
            },
            &mut failures,
        );
    }
    assert_no_failures("divide-and-conquer forces", &failures);
}

#[test]
fn stress_passes_four_step_fits_of_the_free_energy_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.02, 0.015, 0.01, 0.0075],
        [0.01, 0.0075, 0.005, 0.00375],
        [0.03, 0.0225, 0.015, 0.01125],
    ];
    let report = run("dc-stress", &cell((0, 0, 0.0), 0, 0.0), &controls(&[])).unwrap();
    let derivative = report
        .get("strain_derivative")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    assert_eq!(derivative.len(), 9);
    println!("divide-and-conquer strain derivative {derivative:?}");
    let mut failures = Vec::new();
    for a in 0..3 {
        for b in 0..3 {
            fit_components(
                &[format!(
                    "divide-and-conquer dA/d eps_{}{}",
                    AXES[a], AXES[b]
                )],
                &INTERVALS,
                |steps| {
                    let mut errors = [0.0; 4];
                    for (s, &h) in steps.iter().enumerate() {
                        let fd = (energy(&cell((a, b, h), 0, 0.0), &[])
                            - energy(&cell((a, b, -h), 0, 0.0), &[]))
                            / (2.0 * h);
                        errors[s] = (fd - derivative[3 * a + b]).abs();
                    }
                    vec![errors]
                },
                &mut failures,
            );
        }
    }
    assert_no_failures("divide-and-conquer strain derivative", &failures);
}
