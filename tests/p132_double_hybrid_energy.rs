// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independently generated parent, exchange, semilocal and separate PT2
//! spin-component oracles. The fixtures are numerical results, not inputs
//! or implementation code from an external project.

use dft_rs::driver::{run, Controls, System, Value};

fn system(name: &str) -> System {
    let (numbers, positions) = match name {
        "hydrogen" => (vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
        "water" => (
            vec![8, 1, 1],
            vec![
                [0.0, 0.0, 0.2217],
                [0.0, 1.4309, -0.8867],
                [0.0, -1.4309, -0.8867],
            ],
        ),
        "lithium_hydride_cation" => (vec![3, 1], vec![[0.0, 0.0, 0.0], [0.2, 0.1, 3.0]]),
        _ => panic!("unknown oracle fixture"),
    };
    System::molecule(numbers, positions)
}
fn number(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn b2plyp_closed_and_open_shell_energies_match_independent_term_oracles() {
    let data = include_str!("oracle_data/double_hybrid_energy.csv");
    let header: Vec<_> = data.lines().next().unwrap().split(',').collect();
    let mut worst = 0.0_f64;
    for line in data.lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let value = |key: &str| {
            fields[header.iter().position(|x| *x == key).unwrap()]
                .parse::<f64>()
                .unwrap()
        };
        let name = fields[0];
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "b2plyp")
                .with("basis", "sto-3g")
                .with("grid_level", 5usize)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-10)
                .with("conv_tol_dp", 1e-12)
                .with("charge", value("charge") as i64)
                .with("multiplicity", value("spin") as i64 + 1)
                .with("unrestricted", value("spin") > 0.0),
        )
        .unwrap();
        let report = run("energy", &system(name), &controls).unwrap();
        let ss = number(&report, "pt2_same_spin_alpha") + number(&report, "pt2_same_spin_beta");
        let components = report.get("semilocal_components").unwrap();
        let Value::Array(parts) = components else {
            panic!("missing semilocal terms")
        };
        for (key, actual) in [
            ("parent", number(&report, "reference_energy")),
            ("energy", number(&report, "energy")),
            ("one", number(&report, "one_electron_energy")),
            ("hartree", number(&report, "hartree_energy")),
            ("exchange", number(&report, "exact_exchange_energy")),
            ("b88", number(&parts[0], "energy")),
            ("lyp", number(&parts[1], "energy")),
            ("same_spin", ss),
            ("opposite_spin", number(&report, "pt2_opposite_spin")),
        ] {
            let defect = (actual - value(key)).abs();
            worst = worst.max(defect);
            println!(
                "{name} {key} computed={actual:.15e} oracle={:.15e} defect={defect:.12e}",
                value(key)
            );
            assert!(defect < 1e-6, "{name} {key}: {defect:e}");
        }
        for key in [
            "pt2_amplitude_residual",
            "pt2_lagrangian_residual",
            "semilocal_component_sum_residual",
            "reference_energy_sum_residual",
        ] {
            let raw = number(&report, key);
            println!("{name} {key}={raw:.12e}");
            assert!(raw < 1e-9);
        }
        assert!(number(&report, "pt2_smallest_denominator") > 0.0);
    }
    println!("independent double-hybrid energy/term maximum defect={worst:.12e}");
}

#[test]
fn custom_zero_correlation_recovers_the_parent_and_one_electron_has_no_pt2() {
    let h2 = system("hydrogen");
    let controls = Controls::from_value(
        &Value::object()
            .with("double_hybrid_model", "custom")
            .with("xc", "pbe0")
            .with("pt2_same_spin", 0.0)
            .with("pt2_opposite_spin", 0.0)
            .with("conv_tol", 1e-12),
    )
    .unwrap();
    let report = run("energy", &h2, &controls).unwrap();
    assert_eq!(
        number(&report, "energy"),
        number(&report, "reference_energy")
    );
    let atom = System::molecule(vec![1], vec![[0.0; 3]]);
    let options = Controls::from_value(
        &Value::object()
            .with("xc", "b2plyp")
            .with("multiplicity", 2i64)
            .with("unrestricted", true),
    )
    .unwrap();
    let report = run("energy", &atom, &options).unwrap();
    assert_eq!(number(&report, "pt2_correlation"), 0.0);
}
