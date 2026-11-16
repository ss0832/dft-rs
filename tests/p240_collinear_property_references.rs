// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Existing population/orbital properties on explicit RKS, UKS and ROKS
//! references. Every AO metric/density entry and every atomic population is
//! checked. No differentiated quantity is introduced by these selectors.
use dft_rs::driver::method::{
    prepare_molecular, run_collinear_property_reference, CollinearPropertyReference,
    MolecularMethod,
};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::linalg::{symmetric_eigen, Matrix};

fn water() -> System {
    System {
        numbers: vec![8, 1, 1],
        positions: vec![
            [0.13, -0.21, 0.19],
            [0.27, 1.21, -0.78],
            [-0.08, -1.69, -0.83],
        ],
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(xc: &str, kind: &str, open: bool, orbitals: bool) -> Controls {
    let mut v = Value::object()
        .with("xc", xc)
        .with("basis", "sto-3g")
        .with("reference", kind)
        .with("charge", i32::from(open))
        .with("multiplicity", if open { 2 } else { 1 })
        .with("grid_level", 2)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_grad", 1e-9)
        .with("max_iter", 300)
        .with("guess", "core");
    if orbitals {
        v.set("coefficients", true);
        v.set("density", true);
    }
    Controls::from_value(&v).unwrap()
}

fn matrix(v: &Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_row_major(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap())
            .map(|v| v.as_f64().unwrap())
            .collect(),
    )
}

fn validate(charges: &Value, orbitals: &Value, open: bool) {
    let s = matrix(orbitals.get("overlap").unwrap());
    let n = s.rows;
    let channels = orbitals.get("channels").unwrap().as_array().unwrap();
    let mut densities = Vec::new();
    let mut worst = 0.0_f64;
    for ch in channels {
        let c = matrix(ch.get("coefficients").unwrap());
        let p = matrix(ch.get("density").unwrap());
        let occ = ch.get("occupations").unwrap().as_array().unwrap();
        let weight = if channels.len() == 1 { 2.0 } else { 1.0 };
        for a in 0..c.cols {
            for b in 0..c.cols {
                let mut metric = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        metric += c[(i, a)] * s[(i, j)] * c[(j, b)];
                    }
                }
                worst = worst.max((metric - if a == b { 1.0 } else { 0.0 }).abs());
            }
        }
        for i in 0..n {
            for j in 0..n {
                let rebuilt: f64 = (0..c.cols)
                    .map(|a| c[(i, a)] * c[(j, a)] * occ[a].as_f64().unwrap() / weight)
                    .sum();
                worst = worst.max((rebuilt - p[(i, j)]).abs());
            }
        }
        densities.push(p);
    }
    if densities.len() == 1 {
        densities.push(densities[0].clone());
    }
    let (values, u) = symmetric_eigen(&s).unwrap();
    assert!(values.iter().all(|v| *v > 0.0));
    let mut half = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            half[(i, j)] = (0..n)
                .map(|a| u[(i, a)] * values[a].sqrt() * u[(j, a)])
                .sum();
        }
    }
    let owner = [0, 0, 0, 0, 0, 1, 2];
    assert_eq!(n, owner.len());
    let mut mulliken = [8.0, 1.0, 1.0];
    let mut lowdin = mulliken;
    let mut spin = [0.0; 3];
    for i in 0..n {
        for j in 0..n {
            mulliken[owner[i]] -= (densities[0][(i, j)] + densities[1][(i, j)]) * s[(j, i)];
            spin[owner[i]] += (densities[0][(i, j)] - densities[1][(i, j)]) * s[(j, i)];
            for k in 0..n {
                lowdin[owner[i]] -=
                    half[(i, j)] * (densities[0][(j, k)] + densities[1][(j, k)]) * half[(k, i)];
            }
        }
    }
    for (key, expected) in [
        ("mulliken", mulliken),
        ("lowdin", lowdin),
        ("mulliken_spin", spin),
    ] {
        for (actual, oracle) in charges
            .get(key)
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .zip(expected)
        {
            worst = worst.max((actual.as_f64().unwrap() - oracle).abs());
        }
    }
    assert!((mulliken.iter().sum::<f64>() - f64::from(u8::from(open))).abs() < 1e-10);
    assert!((lowdin.iter().sum::<f64>() - f64::from(u8::from(open))).abs() < 1e-10);
    assert!((spin.iter().sum::<f64>() - f64::from(u8::from(open))).abs() < 1e-10);
    if orbitals.get("reference_type").unwrap().as_str() == Some("roks") {
        // The beta occupied projector is contained in the alpha one in
        // the common spatial ROKS determinant, despite semicanonical gauges.
        let nested = densities[0].matmul_seq(&s).matmul_seq(&densities[1]);
        worst = worst.max(nested.max_abs_diff(&densities[1]));
    }
    println!(
        "{} all-entry orbital/population defect={worst:.3e}",
        charges.get("functional").unwrap().as_str().unwrap()
    );
    assert!(worst < 1e-9);
}

#[test]
fn roks_orbitals_and_populations_use_the_shared_functional_providers() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0"] {
        let q = run("charges", &water(), &controls(xc, "rohf", true, false)).unwrap();
        let c = run("orbitals", &water(), &controls(xc, "roks", true, true)).unwrap();
        assert_eq!(q.get("reference_type").unwrap().as_str(), Some("roks"));
        assert_eq!(q.get("energy"), c.get("energy"));
        validate(&q, &c, true);
    }
}

#[test]
fn explicit_rks_and_uks_recover_the_closed_shell_population_limit() {
    let mut charges = Vec::new();
    for kind in ["rks", "uks"] {
        let q = run("charges", &water(), &controls("pbe", kind, false, false)).unwrap();
        let c = run("orbitals", &water(), &controls("pbe", kind, false, true)).unwrap();
        validate(&q, &c, false);
        charges.push(q);
    }
    for key in ["mulliken", "lowdin"] {
        for (a, b) in charges[0]
            .get(key)
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .zip(charges[1].get(key).unwrap().as_array().unwrap())
        {
            assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-8);
        }
    }
    assert!(run("charges", &water(), &controls("pbe", "rks", true, false)).is_err());
    assert!(run("charges", &water(), &controls("pbe", "roks", false, false)).is_err());
}

#[test]
fn unregistered_functional_description_does_not_change_reference_dispatch() {
    let controls = controls("pbe", "roks", true, false);
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let mut setup = prepare_molecular(&water().numbers, &water().positions, &method).unwrap();
    setup.functional.description = "unregistered semilocal provider".into();
    let reference =
        run_collinear_property_reference(&setup, CollinearPropertyReference::Roks).unwrap();
    assert!(reference.scf.converged);
    assert_eq!((reference.scf.n_alpha, reference.scf.n_beta), (5, 4));
}
