// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Named double hybrids against independent term-by-term oracles
//! (`tools/oracle/gen_double_hybrids.py`): the self-consistent parent, the
//! xDH target on the parent orbitals, the one-electron, Hartree and
//! exact-exchange energies, every weighted semilocal part, the unscaled
//! same-spin and opposite-spin PT2 energies, the D3(BJ) term and the total,
//! for H2, LiH, water, phosphine, ferrocene and the LiH cation doublet.
//! Physical limits: zero PT2 coefficients recover the parent, and the
//! adiabatic-connection recipes keep the exchange fractions summing to one.

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::xc::double_hybrid::DoubleHybridFunctional;

fn oracle() -> Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/double_hybrid_molecules.json"),
    )
    .unwrap();
    Value::parse_json(&text).unwrap()
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("no number `{key}`"))
}

fn system(entry: &Value) -> System {
    let numbers = entry.get("numbers").unwrap().numeric_array().unwrap().1;
    let Value::Array(rows) = entry.get("positions").unwrap() else {
        panic!("positions")
    };
    System::molecule(
        numbers.iter().map(|z| *z as u8).collect(),
        rows.iter()
            .map(|r| {
                let r = r.numeric_array().unwrap().1;
                [r[0], r[1], r[2]]
            })
            .collect(),
    )
}

/// SCF energies agree to the convergence of both SCFs; ferrocene has the
/// largest energy and the loosest absolute agreement.
fn tolerance(molecule: &str) -> f64 {
    if molecule == "ferrocene" {
        1e-5
    } else {
        5e-7
    }
}

#[test]
fn named_double_hybrids_match_independent_term_oracles() {
    let oracle = oracle();
    let molecules = oracle.get("molecules").unwrap();
    let Value::Array(cases) = oracle.get("cases").unwrap() else {
        panic!("cases")
    };
    assert!(!cases.is_empty());
    let mut worst = 0.0_f64;
    for case in cases {
        let model = case.get("model").and_then(Value::as_str).unwrap();
        let name = case.get("molecule").and_then(Value::as_str).unwrap();
        assert!(case.get("converged").and_then(Value::as_bool).unwrap());
        let spin = number(case, "spin");
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", model)
                .with("basis", "sto-3g")
                .with("grid_level", 9usize)
                .with("conv_tol", 1e-12)
                .with("conv_tol_grad", 1e-9)
                .with("charge", number(case, "charge") as i64)
                .with("multiplicity", spin as i64 + 1)
                .with("unrestricted", spin > 0.0),
        )
        .unwrap();
        let report = run("energy", &system(molecules.get(name).unwrap()), &controls).unwrap();
        let dispersion = number(case, "dispersion");
        let Value::Array(parts) = report.get("semilocal_components").unwrap() else {
            panic!("semilocal components")
        };
        let Value::Array(expected_parts) = case.get("parts").unwrap() else {
            panic!("oracle parts")
        };
        assert_eq!(parts.len(), expected_parts.len(), "{model} parts");
        let mut checks = vec![
            ("parent", number(&report, "orbital_reference_energy")),
            ("target", number(&report, "reference_energy") - dispersion),
            ("energy", number(&report, "energy")),
            ("one", number(&report, "one_electron_energy")),
            ("hartree", number(&report, "hartree_energy")),
            ("exchange", number(&report, "exact_exchange_energy")),
            (
                "same_spin",
                number(&report, "pt2_same_spin_alpha") + number(&report, "pt2_same_spin_beta"),
            ),
            ("opposite_spin", number(&report, "pt2_opposite_spin")),
            ("dispersion", number(&report, "model_dispersion_energy")),
        ];
        let part_values: Vec<f64> = parts.iter().map(|p| number(p, "energy")).collect();
        let tol = tolerance(name);
        for (key, actual) in checks.drain(..) {
            let expected = number(case, key);
            let defect = (actual - expected).abs();
            worst = worst.max(defect);
            println!("{model} {name} {key} computed={actual:.12e} oracle={expected:.12e} defect={defect:.3e}");
            assert!(defect < tol, "{model} {name} {key}: {defect:e}");
        }
        for (k, (actual, expected)) in part_values.iter().zip(expected_parts).enumerate() {
            let expected = expected.as_f64().unwrap();
            let defect = (actual - expected).abs();
            worst = worst.max(defect);
            println!("{model} {name} part{k} computed={actual:.12e} oracle={expected:.12e} defect={defect:.3e}");
            assert!(defect < tol, "{model} {name} part {k}: {defect:e}");
        }
        for key in [
            "pt2_amplitude_residual",
            "semilocal_component_sum_residual",
            "reference_energy_sum_residual",
        ] {
            let raw = number(&report, key);
            println!("{model} {name} {key}={raw:.3e}");
            assert!(raw < 1e-9, "{model} {name} {key}: {raw:e}");
        }
    }
    println!("named double hybrids: largest term defect {worst:.3e}");
}

/// Zero PT2 coefficients on the parent of every named recipe recover the
/// parent energy exactly, and the parents of the adiabatic-connection
/// recipes split the exchange between exact and semilocal parts.
#[test]
fn zero_pt2_recovers_the_parent_and_exchange_fractions_sum_to_one() {
    for (name, pt2) in [
        ("pbe0-dh", 0.125),
        ("pbe-qidh", 1.0 / 3.0),
        ("pbe0-2", 0.5),
        ("scan0-dh", 0.125),
        ("r2scan-qidh", 1.0 / 3.0),
    ] {
        let recipe = DoubleHybridFunctional::named(name).unwrap();
        assert_eq!(recipe.correlation.same_spin, pt2);
        assert_eq!(recipe.correlation.opposite_spin, pt2);
        let semilocal_exchange = recipe.parent.parts[0].0;
        let correlation = recipe.parent.parts[1].0;
        assert!((recipe.parent.hybrid.cx + semilocal_exchange - 1.0).abs() < 1e-15);
        assert!((correlation + pt2 - 1.0).abs() < 1e-15);
    }
    let water = System::molecule(
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.2217],
            [0.0, 1.4309, -0.8867],
            [0.0, -1.4309, -0.8867],
        ],
    );
    for name in ["b2gp-plyp", "pbe-qidh", "revdsd-pbep86", "xyg3"] {
        let named = run(
            "energy",
            &water,
            &Controls::from_value(
                &Value::object()
                    .with("xc", name)
                    .with("basis", "sto-3g")
                    .with("conv_tol", 1e-12)
                    .with("dispersion", "none"),
            )
            .unwrap(),
        )
        .unwrap();
        let pt2 = number(&named, "pt2_correlation");
        assert!(pt2 < 0.0, "{name}: PT2 correlation {pt2}");
        let total = number(&named, "energy");
        let reference = number(&named, "reference_energy");
        println!("{name}: energy {total:.12} reference {reference:.12} PT2 {pt2:.3e}");
        assert!((total - reference - pt2).abs() < 1e-12);
    }
}
