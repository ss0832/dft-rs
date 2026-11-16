// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Berry-phase polarization on the common driver: exact invariance of the
//! total polarization under a rigid translation of all atoms, convergence of
//! the chain component of the Born effective charges from polarization
//! differences (validation only) towards the analytic DFPT charges on growing
//! k meshes, and convergence of the Berry-phase dipole of a boxed molecule with
//! the string mesh and the box edge towards the dipole of the isolated
//! molecule.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]];
/// A hydrogen molecule with a helium atom near one end (Bohr): a polar
/// closed-shell cluster with three basis functions.
const CLUSTER: [(u8, [f64; 3]); 3] = [
    (1, [0.0, 0.0, 0.0]),
    (1, [1.45, 0.1, 0.05]),
    (2, [3.6, -0.1, 0.1]),
];

/// The gapped hydrogen chain, all atoms shifted by `shift` and atom `atom`
/// displaced by `h` along `axis`.
fn chain(shift: [f64; 3], atom: usize, axis: usize, h: f64) -> System {
    let mut positions = POSITIONS.to_vec();
    for p in positions.iter_mut() {
        for c in 0..3 {
            p[c] += shift[c];
        }
    }
    positions[atom][axis] += h;
    System {
        numbers: vec![1, 1],
        positions,
        cell: Some(CELL),
        pbc: [true; 3],
    }
}

fn chain_controls(n: i64) -> Controls {
    let pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(n), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(14.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(1)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(300)),
    ];
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

const POLAR_CELL: [[f64; 3]; 3] = [[6.4, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POLAR_POSITIONS: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05], [3.6, -0.1, 0.1]];

/// A chain of hydrogen molecules and helium atoms along x with atom `atom`
/// displaced by `h` along `axis`. The helium atom is closer to one end of the
/// molecule than to the other, so the chain has no inversion centre and its
/// Born charges do not vanish; two identical atoms per cell would keep an
/// inversion centre for any displacement.
fn polar_chain(atom: usize, axis: usize, h: f64) -> System {
    let mut positions = POLAR_POSITIONS.to_vec();
    positions[atom][axis] += h;
    System {
        numbers: vec![1, 1, 2],
        positions,
        cell: Some(POLAR_CELL),
        pbc: [true; 3],
    }
}

fn polar_controls(n: i64) -> Controls {
    let pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(n), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(12.0)),
        ("grid_level".to_string(), Value::Int(3)),
        ("grid_shell".to_string(), Value::Int(1)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(200)),
    ];
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

/// `x` reduced to `(-1/2, 1/2]`: reduced polarizations are defined modulo one.
fn wrap(x: f64) -> f64 {
    let y = x - x.round();
    if y <= -0.5 {
        y + 1.0
    } else {
        y
    }
}

#[test]
fn total_polarization_is_invariant_under_a_rigid_translation() {
    let c = chain_controls(6);
    // A shift with exactly known reduced coordinates s = (0.1, 0.03, -0.02).
    let s = [0.1, 0.03, -0.02];
    let shift: [f64; 3] = std::array::from_fn(|c| (0..3).map(|i| s[i] * CELL[i][c]).sum::<f64>());
    let p0 = run("berry-polarization", &chain([0.0; 3], 0, 0, 0.0), &c).unwrap();
    let p1 = run("berry-polarization", &chain(shift, 0, 0, 0.0), &c).unwrap();
    let (t0, t1) = (flat(&p0, "total_reduced"), flat(&p1, "total_reduced"));
    let (e0, e1) = (
        flat(&p0, "electronic_reduced"),
        flat(&p1, "electronic_reduced"),
    );
    println!(
        "reference: electronic {e0:?}, ionic {:?}, total {t0:?}, string spread {:?}, min ln|det| {:?}, min gap {:?}",
        flat(&p0, "ionic_reduced"),
        flat(&p0, "string_spread"),
        p0.get("min_link_log_abs_determinant"),
        p0.get("min_gap")
    );
    for i in 0..3 {
        let total_gap = wrap(t1[i] - t0[i]);
        // Two electrons move with the atoms: the electronic part changes by
        // -2 s_i modulo the spin degeneracy 2.
        let expected = e1[i] - e0[i] + 2.0 * s[i];
        let electronic_gap = expected - 2.0 * (expected / 2.0).round();
        println!("axis {i}: total change {total_gap:e}, electronic defect {electronic_gap:e}");
        assert!(total_gap.abs() < 1e-7, "axis {i}");
        assert!(electronic_gap.abs() < 1e-7, "axis {i}");
    }
}

/// Recorded on the meshes 4, 6, 8, 12 with h = 0.005: chain differences
/// 6.43e-3, 1.94e-3, 9.84e-4, 3.39e-4 (p = 2.66). The components along the two
/// directions with a single k point keep constant differences of 3.9e-3 and
/// 9.1e-4: there the Berry phase is the single-point formula and the analytic
/// route a one-point zone sum, and with 7 Bohr between the chains the two
/// samplings are not equivalent. Refining only the Berry strings in those
/// directions (8x16x16) moves the chain component by 4e-3 as well, so the
/// transverse components are not compared; the string convergence is checked
/// on the boxed molecule below.
#[test]
fn born_charges_from_polarization_differences_approach_the_dfpt_charges() {
    let h = 0.005;
    let meshes = [4i64, 6, 8, 12];
    let mut spacings = Vec::new();
    let mut errors = Vec::new();
    for n in meshes {
        let c = polar_controls(n);
        let born = run("born-charges", &polar_chain(0, 0, 0.0), &c).unwrap();
        let z = flat(&born, "born_charges");
        let plus = flat(
            &run("berry-polarization", &polar_chain(1, 0, h), &c).unwrap(),
            "total_reduced",
        );
        let minus = flat(
            &run("berry-polarization", &polar_chain(1, 0, -h), &c).unwrap(),
            "total_reduced",
        );
        // V dP_a = sum_i dp_i a_i[a]; column b = x of atom 1.
        let mut row = [0.0; 3];
        for (a, r) in row.iter_mut().enumerate() {
            *r = (0..3)
                .map(|i| wrap(plus[i] - minus[i]) * POLAR_CELL[i][a])
                .sum::<f64>()
                / (2.0 * h);
        }
        let chain_error = (row[0] - z[9]).abs();
        let transverse = (row[1] - z[12]).abs().max((row[2] - z[15]).abs());
        println!(
            "k mesh {n}x1x1: DFPT Z*(atom 1, a, x) = [{:.8}, {:.8}, {:.8}], from polarization {row:?}, chain difference {chain_error:e}, transverse difference {transverse:e}",
            z[9],
            z[12],
            z[15]
        );
        spacings.push(1.0 / n as f64);
        errors.push(chain_error);
    }
    let fit = convergence::error_power_fit(&spacings, &errors);
    println!(
        "Born charge difference against 1/N: errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!(errors.windows(2).all(|w| w[1] < w[0]), "{errors:?}");
    assert!(fit.exponent > 1.9, "p = {}", fit.exponent);
    assert!(fit.r_squared > 0.99, "R2 = {}", fit.r_squared);
    assert!(errors[3] < 5e-4, "{}", errors[3]);
}

/// `None`: the molecular controls. `Some(m)`: the periodic controls with a
/// string mesh of `m` x `m` x `m` points for the Berry phase.
fn cluster_controls(berry_mesh: Option<i64>) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("grid_level".to_string(), Value::Int(4)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
    ];
    if let Some(m) = berry_mesh {
        pairs.push((
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(1), Value::Int(1), Value::Int(1)]),
        ));
        pairs.push(("overlap_cutoff".to_string(), Value::Float(12.0)));
        pairs.push(("grid_shell".to_string(), Value::Int(1)));
        pairs.push(("max_iter".to_string(), Value::Int(200)));
        pairs.push((
            "berry_mesh".to_string(),
            Value::Array(vec![Value::Int(m), Value::Int(m), Value::Int(m)]),
        ));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

/// A neutral molecule in a cubic box: the Berry-phase dipole converges with
/// the second order in the string mesh, and its extrapolated limit approaches
/// the dipole of the isolated molecule as the polarization of the periodic
/// images decays (third order in 1/L). Recorded with one string point per
/// axis: differences 2.98e-2, 2.24e-2, 1.74e-2, 1.14e-2 for L = 12, 14, 16, 20
/// (p = 1.88): the single-point formula alone converges only as 1/L^2.
#[test]
fn boxed_molecule_reaches_the_molecular_dipole() {
    let molecule = System::molecule(
        CLUSTER.iter().map(|(z, _)| *z).collect(),
        CLUSTER.iter().map(|(_, r)| *r).collect(),
    );
    let dipole = flat(
        &run("dipole", &molecule, &cluster_controls(None)).unwrap(),
        "dipole",
    );
    println!("molecular dipole {dipole:?}");
    let edges = [12.0, 14.0, 16.0, 20.0];
    let mut spacings = Vec::new();
    let mut errors = Vec::new();
    for edge in edges {
        let boxed = System {
            numbers: CLUSTER.iter().map(|(z, _)| *z).collect(),
            positions: CLUSTER.iter().map(|(_, r)| *r).collect(),
            cell: Some([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
            pbc: [true; 3],
        };
        // In a cubic box the dipole per cell is p_i L along axis i.
        let dipole_at = |mesh: i64| -> Vec<f64> {
            let p = flat(
                &run("berry-polarization", &boxed, &cluster_controls(Some(mesh))).unwrap(),
                "total_reduced",
            );
            p.iter().map(|x| x * edge).collect()
        };
        let d2 = dipole_at(2);
        let d4 = dipole_at(4);
        let d8 = dipole_at(8);
        let ratio = (d4[0] - d2[0]) / (d8[0] - d4[0]);
        assert!(ratio > 3.5 && ratio < 4.5, "L = {edge}: {ratio}");
        // Richardson extrapolation for a second-order string error.
        let limit: Vec<f64> = (0..3).map(|a| d8[a] + (d8[a] - d4[a]) / 3.0).collect();
        let worst = (0..3)
            .map(|a| (limit[a] - dipole[a]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "L = {edge}: x dipole on string meshes 2, 4, 8 = [{:.8}, {:.8}, {:.8}], ratio {ratio:.4}, extrapolated limit {limit:?}, largest difference {worst:e}",
            d2[0],
            d4[0],
            d8[0]
        );
        spacings.push(1.0 / edge);
        errors.push(worst);
    }
    let fit = convergence::error_power_fit(&spacings, &errors);
    println!(
        "dipole difference against 1/L: errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!(errors.windows(2).all(|w| w[1] < w[0]), "{errors:?}");
    assert!(
        fit.exponent > 2.5 && fit.exponent < 3.6,
        "p = {}",
        fit.exponent
    );
    assert!(fit.r_squared > 0.99, "R2 = {}", fit.r_squared);
    assert!(errors[3] < 1e-3, "{}", errors[3]);
}
