// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Coupled electronic/quantum-nuclear stability as part of SCF convergence.
//!
//! Oracles: the Davidson eigenvalues against a dense diagonalization of the
//! coupled Hessian assembled column by column from its action; the
//! independent dense curvature of the quantum-nuclear ferrocene saddle
//! (`tests/oracle_data/multicomponent_hf.json`, lambda_min =
//! -0.0435919545037 Hartree in the normalization of one half of the energy
//! Hessian) and its independently optimized local minimum
//! (-1628.4000103624767 Hartree, lowest eigenvalue +0.0002058061276503).

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::multicomponent::correlation::{CorrelationModel, NoCorrelation};
use dft_rs::multicomponent::response::CoupledResponseSolver;
use dft_rs::multicomponent::stability::lowest_curvatures;
use dft_rs::multicomponent::{
    gaussian_particle_basis, solve, MulticomponentOptions, QuantumParticle,
};
use dft_rs::scf::ScfOptions;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

fn oracle(name: &str) -> Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/multicomponent_hf.json"),
    )
    .unwrap();
    let Value::Array(cases) = Value::parse_json(&text).unwrap() else {
        panic!("oracle list")
    };
    cases
        .into_iter()
        .find(|c| c.get("name").and_then(Value::as_str) == Some(name))
        .unwrap()
}

fn rows(v: &Value, key: &str) -> Vec<Vec<f64>> {
    let Value::Array(rows) = v.get(key).unwrap() else {
        panic!("{key}")
    };
    rows.iter().map(|r| r.numeric_array().unwrap().1).collect()
}

/// Davidson eigenvalues of the quantum-proton water reference against the
/// dense coupled Hessian.
#[test]
fn davidson_matches_the_dense_coupled_hessian() {
    let uncorrelated = CorrelationModel::single(Box::new(NoCorrelation));
    let case = oracle("water");
    let numbers: Vec<u8> = case
        .get("numbers")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1
        .iter()
        .map(|z| *z as u8)
        .collect();
    let positions = rows(&case, "positions");
    let quantum_atom = case.get("quantum_atom").and_then(Value::as_i64).unwrap() as usize;
    let exponents = case
        .get("quantum_exponents")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    let molecule = Molecule::new(
        numbers
            .iter()
            .zip(&positions)
            .map(|(z, r)| Atom {
                z: *z,
                position: Vec3::new(r[0], r[1], r[2]),
            })
            .collect(),
    );
    let set = data::load("sto-3g").unwrap();
    let basis = MolBasis::build(&molecule, &set).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let options = ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-11,
        max_iter: 500,
        ..ScfOptions::default()
    };
    let particles = vec![QuantumParticle {
        label: "proton".into(),
        charge: 1.0,
        mass: dft_rs::constants::PROTON_ELECTRON_MASS_RATIO,
        basis: gaussian_particle_basis(molecule.atoms[quantum_atom].position, &exponents, 1)
            .unwrap(),
        replaced_atom: Some(quantum_atom),
        correlation_channel: Some(0),
    }];
    let species = MulticomponentOptions {
        tolerance: 1e-11,
        ..MulticomponentOptions::default()
    };
    let result = solve(
        &molecule,
        &basis,
        &options,
        &functional,
        None,
        &particles,
        &uncorrelated,
        &species,
    )
    .unwrap();
    let solver = CoupledResponseSolver::new(
        &molecule,
        &basis,
        &options,
        &functional,
        None,
        &particles,
        &uncorrelated,
        &result,
    )
    .unwrap();
    let stability = lowest_curvatures(&solver, 3, 1e-9, 200).unwrap();
    let n = solver.dimension();
    let mut dense = Matrix::zeros(n, n);
    for col in 0..n {
        let mut unit = vec![0.0; n];
        unit[col] = 1.0;
        let image = solver.apply_real(&unit).unwrap();
        for row in 0..n {
            dense[(row, col)] = image[row];
        }
    }
    let mut asymmetry = 0.0_f64;
    for r in 0..n {
        for c in 0..r {
            asymmetry = asymmetry.max((dense[(r, c)] - dense[(c, r)]).abs());
            let mean = 0.5 * (dense[(r, c)] + dense[(c, r)]);
            dense[(r, c)] = mean;
            dense[(c, r)] = mean;
        }
    }
    let (values, _) = symmetric_eigen(&dense).unwrap();
    println!(
        "water quantum proton: dimension {n}, Davidson {:?} (residuals {:?}, {} iterations), dense {:?}, raw asymmetry {asymmetry:.3e}",
        stability.eigenvalues,
        stability.residuals,
        stability.iterations,
        &values[..3]
    );
    for k in 0..3 {
        assert!((stability.eigenvalues[k] - values[k]).abs() < 1e-8);
    }
    assert!(stability.eigenvalues[0] > 0.0);
}

/// The ferrocene saddle and its followed minimum against the independent
/// dense curvature and optimization.
#[test]
fn ferrocene_saddle_is_followed_to_the_independent_minimum() {
    let case = oracle("ferrocene");
    let numbers: Vec<u8> = case
        .get("numbers")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1
        .iter()
        .map(|z| *z as u8)
        .collect();
    let positions: Vec<[f64; 3]> = rows(&case, "positions")
        .iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let system = System::molecule(numbers, positions);
    let quantum_atom = case.get("quantum_atom").and_then(Value::as_i64).unwrap();
    let controls = |mode: &str| {
        Controls::from_value(&Value::Object(vec![
            ("xc".into(), Value::Str("hf".into())),
            ("dispersion".into(), Value::Str("none".into())),
            ("interspecies_correlation".into(), Value::Str("none".into())),
            (
                "quantum_atoms".into(),
                Value::Array(vec![Value::Int(quantum_atom)]),
            ),
            (
                "quantum_exponents".into(),
                case.get("quantum_exponents").unwrap().clone(),
            ),
            (
                "quantum_lmax".into(),
                case.get("quantum_lmax").unwrap().clone(),
            ),
            ("species_tolerance".into(), Value::Float(1e-12)),
            ("conv_tol".into(), Value::Float(1e-14)),
            ("conv_tol_grad".into(), Value::Float(1e-12)),
            ("conv_tol_dp".into(), Value::Float(1e-12)),
            ("max_iter".into(), Value::Int(500)),
            ("species_stability".into(), Value::Str(mode.into())),
        ]))
        .unwrap()
    };
    let report = run("multicomponent", &system, &controls("report")).unwrap();
    let stability = report.get("species_stability").unwrap();
    let lowest = stability
        .get("lowest_coupled_eigenvalue")
        .and_then(Value::as_f64)
        .unwrap();
    let energy = report.get("energy").and_then(Value::as_f64).unwrap();
    println!("ferrocene reference: energy {energy:.10}, lowest coupled eigenvalue {lowest:.10}");
    let followed = run("multicomponent", &system, &controls("follow")).unwrap();
    let stability = followed.get("species_stability").unwrap();
    let final_lowest = stability
        .get("lowest_coupled_eigenvalue")
        .and_then(Value::as_f64)
        .unwrap();
    let final_energy = followed.get("energy").and_then(Value::as_f64).unwrap();
    let history = stability
        .get("lowest_eigenvalue_history")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    println!(
        "ferrocene followed: energy {final_energy:.10} (oracle -1628.4000103625), lowering {:.10}, lowest eigenvalue {final_lowest:.10} (oracle 0.0002058061), history {history:?}",
        stability
            .get("followed_energy_lowering")
            .and_then(Value::as_f64)
            .unwrap()
    );
    if lowest < 0.0 {
        assert!(
            (lowest + 0.0435919545037).abs() < 1e-6,
            "saddle curvature {lowest}"
        );
    }
    assert!((final_energy + 1628.4000103624767).abs() < 2e-9);
    assert!((final_lowest - 0.0002058061276503).abs() < 1e-6);
}
