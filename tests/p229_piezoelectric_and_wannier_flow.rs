// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance of the periodic piezoelectric tensors and of the hybrid
//! Wannier flow on the common driver.
//!
//! `piezoelectric-tensor`: for an HF molecule in cubic cells of growing edge
//! `L` the clamped-ion proper tensor times the volume approaches the affine
//! strain derivative of the molecular dipole measured in the unstrained
//! frame, `sum_A APT_(a,Ab) R_(A,c) - delta_ab mu_c` (symmetrized in `bc`),
//! from the independent molecular atomic polar tensors and dipole; the
//! periodic images polarize each other through the local field, so the
//! error is of order `L^-3` and a four-step fit in `h = 1/L` checks the
//! exponent, as for the Born charges. A molecule in a box has no stable
//! ionic relaxation (its rotations are soft or unstable modes; the first
//! run stopped on a smallest optical force-constant eigenvalue of
//! -0.123 Hartree/Bohr^2), so the test requests the clamped-ion tensor only
//! (`relaxed_ion = false`).
//!
//! `piezo-higher`: along the Lagrangian strain `eta_xx = t` of a polar chain
//! the proper dipole of the strained cell, `D'(t) = F(t)^-1 D(t)` with
//! `F = (1 + 2 eta)^(1/2)` and `D` the Berry-phase dipole of the
//! zero-field state, has `dD'/dt = Omega_0 e_(a,xx)` and
//! `d2D'/dt2 = Omega_0 de_(a,xx)/d eta_xx`; independent four-step fits of
//! central differences check both (finite differences are validation only).
//!
//! `wannier-flow`: a trivial insulator has no winding, and the sum of the
//! hybrid Wannier centres at the transverse wave vector zero is the
//! electronic reduced Berry-phase polarization of the same string, modulo
//! one. The first run found the centres summed with the spin weight equal to
//! plus the electronic reduced polarization (0.338485290294 x 2 against
//! 0.676970580587, defect 0.354 modulo one): the flow reported the
//! eigenphases of the loop product over 2 pi, which are minus the centres.
//! The same run stopped `piezo-higher` on its unit label of the second
//! susceptibility (Hartree and Bohr outside the leading unit expression).
//! With the label fixed, the second strain derivative along z failed over
//! the strains [0.02, 0.015, 0.01, 0.0075]: errors [1.722e-5, 8.144e-6,
//! 3.121e-6, 1.656e-6], p = 2.384, R^2 = 0.99895 (the other five fits
//! passed with p = 1.95 to 2.02). The errors are `c h^2 + d h^4` with
//! c = 0.0273 and d = 39.4 to 5e-9 (an analytic offset fitted with them is
//! -1.3e-8): the interval is not yet asymptotic. The fits now use
//! [0.01, 0.0075, 0.005, 0.0035], where the quartic part falls from 14 % to
//! 2 % and the errors stay above the 1e-8 level of the dipole differences.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn mesh(n: i64) -> Value {
    Value::Array(vec![Value::Int(n), Value::Int(1), Value::Int(1)])
}

/// `x` reduced to `(-1/2, 1/2]`.
fn wrap(x: f64) -> f64 {
    let y = x - x.round();
    if y <= -0.5 {
        y + 1.0
    } else {
        y
    }
}

const HF: [(u8, [f64; 3]); 2] = [(9, [0.0, 0.0, 0.0]), (1, [1.7328, 0.0, 0.0])];

fn molecular() -> Vec<(&'static str, Value)> {
    vec![
        ("xc", Value::Str("svwn5".into())),
        ("basis", Value::Str("sto-3g".into())),
        ("grid_level", Value::Int(4)),
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
    ]
}

#[test]
fn clamped_ion_piezoelectric_tensor_approaches_the_molecular_limit() {
    let molecule = System::molecule(
        HF.iter().map(|(z, _)| *z).collect(),
        HF.iter().map(|(_, r)| *r).collect(),
    );
    let mut request = molecular();
    request.push(("max_rank", Value::Int(1)));
    let gradient = run("multipole-gradient", &molecule, &controls(&request)).unwrap();
    let ranks = gradient.get("ranks").unwrap().as_array().unwrap();
    // apt[a][3 atom + b] = d mu_a / d R_(atom, b), nuclear charges included.
    let apt = flat(&ranks[1], "gradient");
    let dipole = flat(
        &run("dipole", &molecule, &controls(&molecular())).unwrap(),
        "dipole",
    );
    // Affine strain derivative of the dipole in the unstrained frame,
    // [a][3 b + c], symmetric in b and c.
    let mut expected = [[0.0_f64; 9]; 3];
    for (a, row) in expected.iter_mut().enumerate() {
        for b in 0..3 {
            for c in 0..3 {
                let affine = |b: usize, c: usize| -> f64 {
                    let moved: f64 = HF
                        .iter()
                        .enumerate()
                        .map(|(atom, (_, r))| apt[6 * a + 3 * atom + b] * r[c])
                        .sum();
                    moved - if a == b { dipole[c] } else { 0.0 }
                };
                row[3 * b + c] = 0.5 * (affine(b, c) + affine(c, b));
            }
        }
    }
    println!("molecular affine strain derivative of the dipole {expected:?}");
    let edges = [14.0, 16.0, 18.0, 20.0];
    let mut errors = Vec::new();
    for &edge in &edges {
        let system = System {
            numbers: HF.iter().map(|(z, _)| *z).collect(),
            positions: HF.iter().map(|(_, r)| *r).collect(),
            cell: Some([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
            pbc: [true; 3],
        };
        let mut request = molecular();
        request.push(("k_mesh", Value::Array(vec![Value::Int(1); 3])));
        request.push(("overlap_cutoff", Value::Float(24.0)));
        request.push(("grid_shell", Value::Int(1)));
        request.push(("max_iter", Value::Int(200)));
        request.push(("relaxed_ion", Value::Bool(false)));
        let report = run("piezoelectric-tensor", &system, &controls(&request)).unwrap();
        let clamped = flat(&report, "clamped_ion_full");
        let volume = edge * edge * edge;
        let mut error = 0.0_f64;
        for (a, row) in expected.iter().enumerate() {
            for bc in 0..9 {
                let (b, c) = (bc / 3, bc % 3);
                let periodic = 0.5 * (clamped[9 * a + 3 * b + c] + clamped[9 * a + 3 * c + b]);
                error = error.max((periodic * volume - row[bc]).abs());
            }
        }
        println!(
            "L = {edge}: clamped-ion tensor times volume {:?}, error {error:e}, index asymmetry {:e}",
            clamped.iter().map(|v| v * volume).collect::<Vec<_>>(),
            number(&report, "strain_index_asymmetry")
        );
        errors.push(error);
    }
    let h: Vec<f64> = edges.iter().map(|l| 1.0 / l).collect();
    let fit = convergence::error_power_fit(&h, &errors);
    println!(
        "piezoelectric tensor: h = 1/L {h:?} errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 3.0).abs() < 0.15 && fit.r_squared > 0.999);
}

const POLAR_CELL: [[f64; 3]; 3] = [[6.4, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POLAR_POSITIONS: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05], [3.6, -0.1, 0.1]];

/// The polar chain of hydrogen molecules and helium atoms under the
/// Lagrangian strain `eta_xx = t` at fixed reduced coordinates.
fn polar_chain(t: f64) -> System {
    let stretch = (1.0 + 2.0 * t).sqrt();
    let mut cell = POLAR_CELL;
    for row in cell.iter_mut() {
        row[0] *= stretch;
    }
    System {
        numbers: vec![1, 1, 2],
        positions: POLAR_POSITIONS
            .iter()
            .map(|p| [p[0] * stretch, p[1], p[2]])
            .collect(),
        cell: Some(cell),
        pbc: [true; 3],
    }
}

fn chain_controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc", Value::Str("svwn5".into())),
        ("basis", Value::Str("sto-3g".into())),
        ("k_mesh", mesh(2)),
        ("overlap_cutoff", Value::Float(20.0)),
        ("grid_level", Value::Int(2)),
        ("grid_shell", Value::Int(2)),
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
        ("max_iter", Value::Int(200)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((*key, value.clone()));
    }
    controls(&pairs)
}

#[test]
fn piezoelectric_tensor_and_its_strain_derivative_pass_four_step_fits() {
    let report = run("piezo-higher", &polar_chain(0.0), &chain_controls(&[])).unwrap();
    let volume = number(&report, "volume");
    // [a][Voigt] and [a][Voigt][Voigt]; Voigt index 0 is xx.
    let first = flat(&report, "piezoelectric_tensor");
    let second = flat(&report, "second_piezoelectric");
    let proper_dipole = |t: f64| -> [f64; 3] {
        let state = run(
            "periodic-efield",
            &polar_chain(t),
            &chain_controls(&[("field", Value::Array(vec![Value::Float(0.0); 3]))]),
        )
        .unwrap();
        let d = flat(&state, "dipole");
        [d[0] / (1.0 + 2.0 * t).sqrt(), d[1], d[2]]
    };
    let centre = proper_dipole(0.0);
    let steps = [0.01, 0.0075, 0.005, 0.0035];
    let samples: Vec<([f64; 3], [f64; 3])> = steps
        .iter()
        .map(|&h| (proper_dipole(h), proper_dipole(-h)))
        .collect();
    for a in 0..3 {
        let analytic_first = volume * first[6 * a];
        let analytic_second = volume * second[36 * a];
        let first_errors: Vec<f64> = steps
            .iter()
            .zip(&samples)
            .map(|(&h, (plus, minus))| ((plus[a] - minus[a]) / (2.0 * h) - analytic_first).abs())
            .collect();
        let second_errors: Vec<f64> = steps
            .iter()
            .zip(&samples)
            .map(|(&h, (plus, minus))| {
                ((plus[a] - 2.0 * centre[a] + minus[a]) / (h * h) - analytic_second).abs()
            })
            .collect();
        let first_fit = convergence::error_power_fit(&steps, &first_errors);
        let second_fit = convergence::error_power_fit(&steps, &second_errors);
        println!(
            "axis {a}: Omega e_xx {analytic_first:.8e}, steps {steps:?} errors {first_errors:?} \
             p={:.6} R2={:.8}; Omega de_xx/deta_xx {analytic_second:.8e}, errors \
             {second_errors:?} p={:.6} R2={:.8}",
            first_fit.exponent, first_fit.r_squared, second_fit.exponent, second_fit.r_squared
        );
        assert!((first_fit.exponent - 2.0).abs() < 0.2 && first_fit.r_squared > 0.99);
        assert!((second_fit.exponent - 2.0).abs() < 0.3 && second_fit.r_squared > 0.98);
    }
}

#[test]
fn trivial_chain_has_no_winding_and_its_centres_are_the_berry_phase() {
    let divisions = 8;
    let flow = run(
        "wannier-flow",
        &polar_chain(0.0),
        &chain_controls(&[
            (
                "topology_plane_axes",
                Value::Array(vec![Value::Int(0), Value::Int(1)]),
            ),
            ("wannier_loop_divisions", Value::Int(divisions)),
            ("wannier_flow_points", Value::Int(6)),
        ]),
    )
    .unwrap();
    let winding = number(&flow, "winding");
    let weight = number(&flow, "spin_weight");
    let rows = flow.get("results").unwrap().as_array().unwrap();
    let total_at_zero: f64 = rows
        .iter()
        .map(|row| {
            let flow_k = flat(row, "flow");
            let total = flat(row, "total");
            assert!(flow_k[0].abs() < 1e-14, "the flow starts at k = 0");
            total[0]
        })
        .sum();
    let berry = run(
        "berry-polarization",
        &polar_chain(0.0),
        &chain_controls(&[(
            "berry_mesh",
            Value::Array(vec![Value::Int(divisions), Value::Int(1), Value::Int(1)]),
        )]),
    )
    .unwrap();
    let electronic = flat(&berry, "electronic_reduced")[0];
    let defect = wrap(electronic + weight * total_at_zero);
    println!(
        "winding {winding:.3e}; centres at k = 0 summed with weight {weight}: {total_at_zero:.12}, \
         electronic reduced Berry phase {electronic:.12}, defect modulo one {defect:.3e}"
    );
    assert!(winding.abs() < 1e-8);
    assert!(defect.abs() < 1e-8);
}
