// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Path route of the excited-state gradient on the common driver: for a
//! closed-shell Hartree-Fock singlet it is the Z-vector gradient; for a
//! hybrid functional every component of it against central differences of
//! the excited-state energy by four-step fits (finite differences are
//! validation only; the molecule is distorted away from C2v so that no
//! component vanishes by symmetry, and a component that fails on the first
//! interval is recomputed on a finer and then a coarser one); and for an
//! unrestricted open shell it obeys the translation sum rule.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use components::{assert_no_failures, fit_components};

use dft_rs::driver::{run, Controls, System, Value};

/// Water distorted away from C2v, with Cartesian coordinate `coord`
/// (3 atom + axis) moved by `shift`.
fn water(coord: usize, shift: f64) -> System {
    let mut positions = vec![
        [0.0, 0.0, 0.2217],
        [0.0, 1.4309, -0.8867],
        [0.13, -1.4309, -0.8867],
    ];
    positions[coord / 3][coord % 3] += shift;
    System {
        numbers: vec![8, 1, 1],
        positions,
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(xc: &str, extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str(xc.into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions (the partition-weight cutoff otherwise
        // adds noise near 1e-13 to displaced gradients; see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("td_tolerance".to_string(), Value::Float(1e-10)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn path() -> (&'static str, Value) {
    ("td_gradient_route", Value::Str("path".into()))
}

fn gradient(report: &Value) -> Vec<f64> {
    report.get("gradient").unwrap().numeric_array().unwrap().1
}

#[test]
fn path_route_gives_the_z_vector_gradient_of_a_closed_shell_singlet() {
    let direct = gradient(&run("td-state-gradient", &water(0, 0.0), &controls("hf", &[])).unwrap());
    let series = gradient(
        &run(
            "td-state-gradient",
            &water(0, 0.0),
            &controls("hf", &[path()]),
        )
        .unwrap(),
    );
    let difference = direct
        .iter()
        .zip(&series)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    println!("path against Z-vector gradient {difference:.3e}");
    assert!(difference < 1e-6);
}

#[test]
fn hybrid_gradient_passes_four_step_fits_of_the_excited_state_energy_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.04, 0.02, 0.01, 0.005],
        [0.02, 0.01, 0.005, 0.0025],
        [0.08, 0.04, 0.02, 0.01],
    ];
    const AXES: [&str; 3] = ["x", "y", "z"];
    let report = run(
        "td-state-gradient",
        &water(0, 0.0),
        &controls("pbe0", &[path()]),
    )
    .unwrap();
    let analytic = gradient(&report);
    assert_eq!(analytic.len(), 9);
    println!("hybrid excited-state gradient {analytic:?}");
    let energy = |coord: usize, shift: f64| {
        let states = run(
            "excitations",
            &water(coord, shift),
            &controls("pbe0", &[("nstates", Value::Int(1))]),
        )
        .unwrap();
        let ground = states.get("energy").and_then(Value::as_f64).unwrap();
        let excitation = states.get("states").and_then(Value::as_array).unwrap()[0]
            .get("energy")
            .and_then(Value::as_f64)
            .unwrap();
        ground + excitation
    };
    let mut failures = Vec::new();
    for coord in 0..9 {
        fit_components(
            &[format!(
                "excited-state dE/d{}{}",
                AXES[coord % 3],
                coord / 3
            )],
            &INTERVALS,
            |steps| {
                let mut errors = [0.0; 4];
                for (s, &h) in steps.iter().enumerate() {
                    errors[s] = ((energy(coord, h) - energy(coord, -h)) / (2.0 * h)
                        - analytic[coord])
                        .abs();
                }
                vec![errors]
            },
            &mut failures,
        );
    }
    assert_no_failures("hybrid excited-state gradient", &failures);
}

#[test]
fn open_shell_path_gradient_obeys_the_translation_sum_rule() {
    let hydroxyl = System {
        numbers: vec![8, 1],
        positions: vec![[0.1, -0.05, 0.0], [0.9, 0.7, 1.3]],
        cell: None,
        pbc: [false; 3],
    };
    let report = run(
        "td-state-gradient",
        &hydroxyl,
        &controls(
            "pbe",
            &[
                path(),
                ("multiplicity", Value::Int(2)),
                ("unrestricted", Value::Bool(true)),
            ],
        ),
    )
    .unwrap();
    let values = gradient(&report);
    let sum = (0..3).fold(0.0_f64, |m, axis| {
        m.max(
            (0..2)
                .map(|atom| values[3 * atom + axis])
                .sum::<f64>()
                .abs(),
        )
    });
    println!("translation sum of the open-shell path gradient {sum:.3e}");
    assert!(sum < 1e-6);
}
