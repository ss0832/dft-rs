// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Raman activities with the dynamic polarizability on the common driver:
//! at a small photon energy they are the static ones, and the derivative of
//! the dynamic polarizability along a normal coordinate against central
//! differences of the dynamic polarizability of displaced molecules by a
//! four-step fit (finite differences are validation only; the displaced
//! polarizabilities come from the independent spinor response without
//! spin-orbit coupling). Every component of the derivative along every
//! normal mode is fitted; the molecule is distorted away from C2v so that
//! no component vanishes by symmetry, and a component that fails on the
//! first interval is recomputed on a finer and then a coarser one.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use components::{assert_no_failures, fit_components};

use dft_rs::driver::{run, Controls, System, Value};

const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;

fn water(shift: &[f64]) -> System {
    let base = [
        [0.0, 0.0, 0.2217],
        [0.0, 1.4309, -0.8867],
        [0.13, -1.4309, -0.8867],
    ];
    System {
        numbers: vec![8, 1, 1],
        positions: (0..3)
            .map(|atom| std::array::from_fn(|axis| base[atom][axis] + shift[3 * atom + axis]))
            .collect(),
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("hf".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions (the partition-weight cutoff otherwise
        // adds noise near 1e-13 to displaced gradients; see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn small_photon_energy_gives_the_static_activities() {
    let report = run(
        "ir-raman",
        &water(&[0.0; 9]),
        &controls(&[("raman_frequency", Value::Float(1e-3))]),
    )
    .unwrap();
    let fixed = values(&report, "raman_activities");
    let dynamic = values(&report, "dynamic_raman_activities");
    let worst = fixed.iter().zip(&dynamic).fold(0.0_f64, |m, (a, b)| {
        m.max((a - b).abs() / a.abs().max(1e-12))
    });
    println!("dynamic against static Raman activities, relative {worst:.3e}");
    assert!(worst < 1e-3);
}

#[test]
fn dynamic_derivatives_pass_four_step_fits_in_every_mode_and_component() {
    check_dynamic_derivatives(&water(&[0.0; 9]), "water");
}

#[test]
fn c1_four_atom_dynamic_derivatives_pass_every_mode_and_component() {
    check_dynamic_derivatives(
        &System::molecule(
            vec![7, 1, 1, 1],
            vec![
                [0.17, -0.11, 0.23],
                [1.43, 0.81, -0.62],
                [-0.63, 1.59, -0.43],
                [-1.12, -1.36, -0.71],
            ],
        ),
        "C1 ammonia",
    );
}

fn check_dynamic_derivatives(reference: &System, label: &str) {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.08, 0.04, 0.02, 0.01],
        [0.04, 0.02, 0.01, 0.005],
        [0.16, 0.08, 0.04, 0.02],
    ];
    const AXES: [&str; 3] = ["x", "y", "z"];
    let omega = 0.1;
    let report = run(
        "ir-raman",
        reference,
        &controls(&[("raman_frequency", Value::Float(omega))]),
    )
    .unwrap();
    let coordinates = 3 * reference.numbers.len();
    // Nine polarizability entries per mode; one displacement per coordinate.
    let modes = values(&report, "normal_modes_cartesian");
    let derivatives = values(&report, "dynamic_polarizability_derivatives");
    let count = modes.len() / coordinates;
    assert_eq!(derivatives.len(), 9 * count);
    let conv = BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM;
    let mut failures = Vec::new();
    for mode in 0..count {
        let pattern = &modes[coordinates * mode..coordinates * (mode + 1)];
        let analytic = &derivatives[9 * mode..9 * mode + 9];
        println!("{label} mode {mode}: analytic dynamic polarizability derivative {analytic:?}");
        let alpha = |amount: f64| {
            let shift: Vec<f64> = pattern.iter().map(|p| amount * p).collect();
            let mut displaced = reference.clone();
            for (point, delta) in displaced.positions.iter_mut().zip(shift.chunks_exact(3)) {
                for axis in 0..3 {
                    point[axis] += delta[axis];
                }
            }
            values(
                &run(
                    "spinor-polarizability",
                    &displaced,
                    &controls(&[
                        ("soc_scale", Value::Float(0.0)),
                        ("frequencies", Value::Array(vec![Value::Float(omega)])),
                    ]),
                )
                .unwrap(),
                "polarizability_real",
            )
        };
        let names: Vec<String> = (0..9)
            .map(|k| format!("{label} mode {mode} d alpha_{}{}", AXES[k / 3], AXES[k % 3]))
            .collect();
        fit_components(
            &names,
            &INTERVALS,
            |steps| {
                let mut errors = vec![[0.0; 4]; 9];
                for (s, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (alpha(h), alpha(-h));
                    for k in 0..9 {
                        errors[k][s] =
                            ((plus[k] - minus[k]) / (2.0 * h) * conv - analytic[k]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    assert_no_failures("dynamic polarizability derivatives", &failures);
}
