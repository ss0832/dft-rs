// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent field in a finite uniform magnetic field on the common
//! driver: the second field derivative of the energy against the London
//! magnetizability by an independent four-step fit (finite differences are
//! validation only), invariance under a translation of the molecule, and
//! the spin Zeeman shift of an open shell. Measured (docs/response.md,
//! "Finite magnetic field"): gradient fit over [0.04, 0.015] Bohr with
//! p = 1.999854, R^2 = 1.00000000; magnetizability fit over fields
//! [0.08, 0.03] with p = 2.005341, R^2 = 0.99999953; translation energy
//! defect 1.421e-14 Hartree; the OH Zeeman shift equals its analytic spin
//! value after the core-seed and terminating-polynomial fixes.
//! The fits now cover every component: all nine magnetizability elements
//! (diagonal from fields along one axis, off-diagonal from the mixed
//! second difference of fields along two axes) and all nine gradient
//! components in the field; the molecule is distorted away from C2v so that
//! no component vanishes by symmetry, and a component that fails on the
//! first interval is recomputed on a finer and then a coarser one.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use components::{assert_no_failures, fit_components};

const AXES: [&str; 3] = ["x", "y", "z"];

use dft_rs::driver::{run, Controls, System, Value};

fn water(shift: [f64; 3]) -> System {
    let positions = [
        [0.0, 0.0, 0.2217],
        [0.0, 1.4309, -0.8867],
        [0.13, -1.4309, -0.8867],
    ];
    System {
        numbers: vec![8, 1, 1],
        positions: positions
            .iter()
            .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
            .collect(),
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("pbe".into())),
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

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn field(vector: [f64; 3]) -> Value {
    Value::Array(vector.iter().map(|v| Value::Float(*v)).collect())
}

fn energy(system: &System, vector: [f64; 3], extra: &[(&str, Value)]) -> f64 {
    let mut all = vec![("bfield", field(vector))];
    all.extend(extra.iter().cloned());
    number(
        &run("bfield", system, &controls(&all)).unwrap(),
        "field_energy",
    )
}

#[test]
fn second_field_derivatives_pass_four_step_fits_of_the_magnetizability_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.08, 0.06, 0.04, 0.03],
        [0.06, 0.045, 0.03, 0.0225],
        [0.12, 0.09, 0.06, 0.045],
    ];
    let system = water([0.0; 3]);
    let report = run(
        "magnetizability",
        &system,
        &controls(&[("gauge", Value::Str("london".into()))]),
    )
    .unwrap();
    let xi = report
        .get("magnetizability")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    assert_eq!(xi.len(), 9);
    println!("London magnetizability {xi:?}");
    let zero = number(&run("energy", &system, &controls(&[])).unwrap(), "energy");
    let at = |a: usize, x: f64, b: usize, y: f64| {
        let mut v = [0.0; 3];
        v[a] += x;
        v[b] += y;
        energy(&system, v, &[])
    };
    let mut failures = Vec::new();
    // E(B) = E(0) - B.xi.B / 2 + O(B^4) for a closed shell.
    for a in 0..3 {
        for b in a..3 {
            let mut names = vec![format!("xi_{}{}", AXES[a], AXES[b])];
            if b != a {
                names.push(format!("xi_{}{}", AXES[b], AXES[a]));
            }
            fit_components(
                &names,
                &INTERVALS,
                |steps| {
                    let mut errors = vec![[0.0; 4]; names.len()];
                    for (s, &h) in steps.iter().enumerate() {
                        let second = if a == b {
                            -2.0 * (at(a, h, a, 0.0) - zero) / (h * h)
                        } else {
                            -(at(a, h, b, h) - at(a, h, b, -h) - at(a, -h, b, h) + at(a, -h, b, -h))
                                / (4.0 * h * h)
                        };
                        errors[0][s] = (second - xi[3 * a + b]).abs();
                        if b != a {
                            errors[1][s] = (second - xi[3 * b + a]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    assert_no_failures("London magnetizability", &failures);
}

#[test]
fn finite_field_energy_does_not_depend_on_the_position_of_the_molecule() {
    let first = energy(&water([0.0; 3]), [0.0, 0.0, 0.05], &[]);
    let second = energy(&water([3.0, -2.0, 1.5]), [0.0, 0.0, 0.05], &[]);
    println!(
        "finite-field energy at two positions: {first:.12} {second:.12}, difference {:e}",
        first - second
    );
    assert!((first - second).abs() < 1e-9);
}

#[test]
fn spin_zeeman_term_shifts_an_open_shell_linearly() {
    // Hydroxyl radical, doublet.
    let system = System {
        numbers: vec![8, 1],
        positions: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.8324]],
        cell: None,
        pbc: [false; 3],
    };
    let doublet = [
        ("multiplicity", Value::Int(2)),
        ("unrestricted", Value::Bool(true)),
    ];
    let strength = 0.01;
    let vector = [0.0, 0.0, strength];
    let mut with = doublet.to_vec();
    with.push(("spin_zeeman", Value::Bool(true)));
    let mut without = doublet.to_vec();
    without.push(("spin_zeeman", Value::Bool(false)));
    let shift = energy(&system, vector, &with) - energy(&system, vector, &without);
    // One unpaired alpha electron: g_e B / 4.
    let expected = 0.25 * 2.002_319_304_362_56 * strength;
    println!("spin Zeeman shift {shift:.10e}, expected {expected:.10e}");
    assert!((shift - expected).abs() < 1e-8);
}

#[test]
fn finite_field_gradient_passes_four_step_fits_of_the_energy_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.04, 0.03, 0.02, 0.015],
        [0.02, 0.015, 0.01, 0.0075],
        [0.08, 0.06, 0.04, 0.03],
    ];
    let strength = [0.0, 0.0, 0.05];
    let report = run(
        "bfield-gradient",
        &water([0.0; 3]),
        &controls(&[("bfield", field(strength))]),
    )
    .unwrap();
    let gradient = report.get("gradient").unwrap().numeric_array().unwrap().1;
    assert_eq!(gradient.len(), 9);
    let translation = number(&report, "translation_residual");
    println!("finite-field gradient {gradient:?}; translation residual {translation:e}");
    assert!(translation < 1e-6);
    let displaced = |coord: usize, h: f64| {
        let mut system = water([0.0; 3]);
        system.positions[coord / 3][coord % 3] += h;
        energy(&system, strength, &[])
    };
    let mut failures = Vec::new();
    for coord in 0..9 {
        fit_components(
            &[format!("finite-field dE/d{}{}", AXES[coord % 3], coord / 3)],
            &INTERVALS,
            |steps| {
                let mut errors = [0.0; 4];
                for (s, &h) in steps.iter().enumerate() {
                    errors[s] = ((displaced(coord, h) - displaced(coord, -h)) / (2.0 * h)
                        - gradient[coord])
                        .abs();
                }
                vec![errors]
            },
            &mut failures,
        );
    }
    assert_no_failures("finite-field gradient", &failures);
}
