// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every functional added from the functional catalogue against frozen
//! PySCF RKS energies of H2, LiH, water, phosphine and ferrocene (STO-3G,
//! grid level 9; `tools/oracle/gen_functional_molecules.py`, file
//! `tests/oracle_data/functional_molecules.json`).

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::Value;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule, Vec3};

fn oracle() -> Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/functional_molecules.json"),
    )
    .unwrap();
    Value::parse_json(&text).unwrap()
}

fn molecule(entry: &Value) -> Molecule {
    let numbers = entry.get("numbers").unwrap().numeric_array().unwrap().1;
    let Value::Array(rows) = entry.get("positions").unwrap() else {
        panic!("positions")
    };
    Molecule::new(
        numbers
            .iter()
            .zip(rows)
            .map(|(z, r)| {
                let r = r.numeric_array().unwrap().1;
                Atom {
                    z: *z as u8,
                    position: Vec3::new(r[0], r[1], r[2]),
                }
            })
            .collect(),
    )
}

/// Tolerance per molecule: the grid integration error of both codes grows
/// with the number of electrons; SCAN carries the oracle's own grid spread.
fn tolerance(name: &str, radial: bool) -> f64 {
    let base = if radial { 2e-5 } else { 5e-7 };
    if name == "ferrocene" {
        base * 20.0
    } else {
        base
    }
}

#[test]
fn catalogue_functionals_match_pyscf_on_five_molecules() {
    let oracle = oracle();
    let Value::Object(molecules) = oracle.get("molecules").unwrap() else {
        panic!("molecules")
    };
    let Value::Array(cases) = oracle.get("cases").unwrap() else {
        panic!("cases")
    };
    let set = data::load("sto-3g").unwrap();
    let mut failures = Vec::new();
    for case in cases {
        let spec = match case.get("spec").unwrap() {
            Value::Str(name) => FunctionalSpec::Combined(name.clone()),
            Value::Array(pair) => FunctionalSpec::Split {
                x: pair[0].as_str().unwrap().into(),
                c: pair[1].as_str().unwrap().into(),
            },
            other => panic!("spec {other:?}"),
        };
        let label = match &spec {
            FunctionalSpec::Combined(n) => n.clone(),
            FunctionalSpec::Split { x, c } => format!("{x}/{c}"),
        };
        let radial = case
            .get("radial")
            .and_then(Value::as_f64)
            .map(|r| r as usize);
        let functional = resolve(&spec).unwrap();
        for (name, entry) in molecules {
            let reference = case
                .get("energies")
                .unwrap()
                .get(name)
                .and_then(Value::as_f64)
                .unwrap();
            let mol = molecule(entry);
            let mb = MolBasis::build(&mol, &set).unwrap();
            let grid = MolecularGrid::build(
                &mol,
                &GridOptions {
                    level: 9,
                    radial_override: radial,
                    // The oracle's explicit (radial, 590) grid: degree 41.
                    angular_override: radial.map(|_| 41),
                    ..GridOptions::default()
                },
            )
            .unwrap();
            let options = ScfOptions {
                e_tol: 1e-11,
                p_tol: 1e-9,
                max_iter: 300,
                eri_mode: EriMode::InCore { limit_mb: 4096 },
                ..ScfOptions::default()
            };
            let result = run_scf(&mol, &mb, &options, &functional, Some(&grid));
            match result {
                Ok(scf) => {
                    let delta = scf.energy - reference;
                    let tol = tolerance(name, radial.is_some());
                    println!(
                        "{label} {name}: E {:.10} PySCF {reference:.10} delta {delta:.3e} (tolerance {tol:.1e}) converged {}",
                        scf.energy, scf.converged
                    );
                    if !scf.converged || delta.abs() > tol {
                        failures.push(format!("{label} {name}: delta {delta:.3e}"));
                    }
                }
                Err(e) => failures.push(format!("{label} {name}: {e}")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "five-molecule failures:\n{}",
        failures.join("\n")
    );
}
