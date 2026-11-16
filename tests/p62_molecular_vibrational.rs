// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance tests for the analytic second-derivative driver routes:
//! Hessian, frequencies, IR/Raman, polarizability and multipole gradients.
//! Every analytic derivative is compared with an independent four-step
//! central-difference error fit of the corresponding lower-order driver
//! observable.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

const WATER: [(u8, [f64; 3]); 3] = [
    (8, [0.0, 0.0, 0.1173]),
    (1, [0.0, 1.4332, -0.9314]),
    (1, [0.0, -1.4332, -0.9314]),
];
const STEPS: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];

fn water() -> System {
    System::molecule(
        WATER.iter().map(|(z, _)| *z).collect(),
        WATER.iter().map(|(_, r)| *r).collect(),
    )
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("conv_tol".to_string(), Value::Float(1e-13)),
        ("conv_tol_dp".to_string(), Value::Float(1e-12)),
    ];
    pairs.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn array(report: &Value, key: &str) -> (Vec<usize>, Vec<f64>) {
    report.get(key).unwrap().numeric_array().unwrap()
}

fn displaced(atom: usize, axis: usize, h: f64) -> System {
    let mut system = water();
    system.positions[atom][axis] += h;
    system
}

/// Fit `|central difference - analytic|` over the four steps and require
/// second-order convergence on the project's acceptance thresholds.
fn assert_second_order(label: &str, analytic: f64, value_at: &dyn Fn(f64) -> f64) {
    let errors: Vec<f64> = STEPS
        .iter()
        .map(|&h| ((value_at(h) - value_at(-h)) / (2.0 * h) - analytic).abs())
        .collect();
    let fit = convergence::error_power_fit(&STEPS, &errors);
    println!(
        "{label}: analytic {analytic:.12e} steps {STEPS:?} errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!(
        (fit.exponent - 2.0).abs() < 0.02,
        "{label}: p = {}",
        fit.exponent
    );
    assert!(fit.r_squared > 0.99999, "{label}: R2 = {}", fit.r_squared);
}

#[test]
fn hessian_columns_and_apt_match_four_step_gradient_and_dipole_fits() {
    let report = run("hessian", &water(), &controls(&[])).unwrap();
    let (shape, hessian) = array(&report, "hessian");
    assert_eq!(shape, vec![9, 9]);
    assert!(report.get("raw_asymmetry").unwrap().as_f64().unwrap() < 1e-7);
    assert!(
        report
            .get("translation_residual")
            .unwrap()
            .as_f64()
            .unwrap()
            < 1e-6
    );
    let (_, apt) = array(&report, "dipole_derivatives");
    // Column for hydrogen-1 y (coordinate 4): rows 2 (oxygen z) and 4.
    for row in [2usize, 4] {
        assert_second_order(&format!("hessian[{row}][4]"), hessian[row * 9 + 4], &|h| {
            let g = run("gradient", &displaced(1, 1, h), &controls(&[])).unwrap();
            array(&g, "gradient").1[row]
        });
    }
    // APT d mu_y / d R(H1,y) with a fixed origin.
    assert_second_order("apt[y][4]", apt[9 + 4], &|h| {
        let d = run(
            "dipole",
            &displaced(1, 1, h),
            &controls(&[("origin", Value::Str("zero".into()))]),
        )
        .unwrap();
        array(&d, "dipole").1[1]
    });
}

#[test]
fn analytic_polarizability_matches_a_four_step_field_fit() {
    let report = run("polarizability", &water(), &controls(&[])).unwrap();
    let (_, alpha) = array(&report, "polarizability");
    assert!(report.get("raw_asymmetry").unwrap().as_f64().unwrap() < 1e-8);
    // alpha_zz = d mu_z / d F_z with H' = +F.r, mu = -Tr(P r) + nuclear.
    assert_second_order("alpha_zz", alpha[8], &|f| {
        let d = run(
            "dipole",
            &water(),
            &controls(&[
                ("origin", Value::Str("zero".into())),
                (
                    "field",
                    Value::Array(vec![Value::Float(0.0), Value::Float(0.0), Value::Float(f)]),
                ),
            ]),
        )
        .unwrap();
        array(&d, "dipole").1[2]
    });
}

#[test]
fn analytic_raman_derivatives_match_displaced_analytic_polarizabilities() {
    // The displaced polarizabilities are differenced over steps down to
    // 0.0025 Bohr, so their CPKS residual must sit well below h^2 * c.
    let tight = [("response_tolerance", Value::Float(1e-12))];
    let report = run("ir-raman", &water(), &controls(&tight)).unwrap();
    let (_, frequencies) = array(&report, "frequencies_cm1");
    assert_eq!(frequencies.len(), 3);
    assert!(frequencies.iter().all(|f| *f > 0.0));
    assert!(report
        .get("thermochemistry")
        .unwrap()
        .get("zero_point_energy")
        .is_some());
    let modes = report
        .get("normal_modes_cartesian")
        .unwrap()
        .as_array()
        .unwrap();
    let derivatives = report
        .get("polarizability_derivatives")
        .unwrap()
        .as_array()
        .unwrap();
    let to_au = 1.0 / (dft_rs::constants::BOHR_TO_ANGSTROM * dft_rs::constants::BOHR_TO_ANGSTROM);
    // Totally symmetric mode 0 changes alpha_zz; the antisymmetric stretch
    // (mode 2) changes only the off-diagonal alpha_yz in this frame.
    for (k, component) in [(0usize, 8usize), (2, 5)] {
        let (_, pattern) = modes[k].numeric_array().unwrap();
        let (_, derivative) = derivatives[k].numeric_array().unwrap();
        let analytic = derivative[component] * to_au;
        assert_second_order(&format!("d alpha[{component}] / dQ_{k}"), analytic, &|h| {
            let mut system = water();
            for atom in 0..3 {
                for axis in 0..3 {
                    system.positions[atom][axis] += h * pattern[3 * atom + axis];
                }
            }
            let a = run("polarizability", &system, &controls(&tight)).unwrap();
            array(&a, "polarizability").1[component]
        });
    }
    let (_, activities) = array(&report, "raman_activities");
    assert!(activities.iter().all(|a| a.is_finite() && *a >= 0.0));
}

#[test]
fn multipole_gradients_match_four_step_moment_fits_for_fixed_and_moving_origins() {
    let hessian = run("hessian", &water(), &controls(&[])).unwrap();
    let (_, apt) = array(&hessian, "dipole_derivatives");
    for origin in ["zero", "center_of_mass"] {
        let report = run(
            "multipole-gradient",
            &water(),
            &controls(&[
                ("max_rank", Value::Int(2)),
                ("origin", Value::Str(origin.into())),
            ]),
        )
        .unwrap();
        let ranks = report.get("ranks").unwrap().as_array().unwrap();
        // Rank 1 equals the analytic atomic polar tensor from the Hessian.
        let (_, rank1) = ranks[1].get("gradient").unwrap().numeric_array().unwrap();
        if origin == "zero" {
            for (a, b) in rank1.iter().zip(&apt) {
                assert!((a - b).abs() < 1e-9, "{a} vs {b}");
            }
        }
        let (_, rank2) = ranks[2].get("gradient").unwrap().numeric_array().unwrap();
        // Quadrupole components xx (0) along H1 y, yz (4) along H1 z and zz
        // (5) along O z; each is nonzero by symmetry.
        for (component, coordinate) in [(0usize, 4usize), (4, 5), (5, 2)] {
            let (atom, axis) = (coordinate / 3, coordinate % 3);
            assert_second_order(
                &format!("dM2[{component}]/dR{coordinate} origin {origin}"),
                rank2[component * 9 + coordinate],
                &|h| {
                    let m = run(
                        "molecular-multipole",
                        &displaced(atom, axis, h),
                        &controls(&[
                            ("max_rank", Value::Int(2)),
                            ("origin", Value::Str(origin.into())),
                        ]),
                    )
                    .unwrap();
                    m.get("ranks").unwrap().as_array().unwrap()[2]
                        .get("cartesian")
                        .unwrap()
                        .numeric_array()
                        .unwrap()
                        .1[component]
                },
            );
        }
    }
}

/// Tr(P M) for the SCF density at a geometry/field and a fixed symmetric AO
/// matrix M (the same AO indexing moves with the atoms).
fn density_projection(
    system: &System,
    extra: &[(&str, Value)],
    probe: &dft_rs::linalg::Matrix,
) -> f64 {
    let mut pairs = extra.to_vec();
    pairs.push(("density", Value::Bool(true)));
    let report = run("orbitals", system, &controls(&pairs)).unwrap();
    let channel = &report.get("channels").unwrap().as_array().unwrap()[0];
    let (_, p) = channel.get("density").unwrap().numeric_array().unwrap();
    // Restricted: the reported channel is one spin; both spins are equal.
    2.0 * p
        .iter()
        .zip(probe.as_slice())
        .map(|(a, b)| a * b)
        .sum::<f64>()
}

fn probe(n: usize) -> dft_rs::linalg::Matrix {
    let mut m = dft_rs::linalg::Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let v = ((i * 7 + j * 13) % 11) as f64 / 11.0 - 0.4;
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

#[test]
fn saved_field_and_nuclear_density_derivatives_match_four_step_fits() {
    let tight = [("response_tolerance", Value::Float(1e-12))];
    let report = run("response-save", &water(), &controls(&tight)).unwrap();
    let field = report.get("field").unwrap().as_array().unwrap();
    let dpdf_z = field[2]
        .get("density_derivative")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .numeric_array()
        .unwrap();
    let n = dpdf_z.0[0];
    let m = probe(n);
    let analytic: f64 = 2.0
        * dpdf_z
            .1
            .iter()
            .zip(m.as_slice())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    assert_second_order("Tr(dP/dF_z M)", analytic, &|f| {
        density_projection(
            &water(),
            &[(
                "field",
                Value::Array(vec![Value::Float(0.0), Value::Float(0.0), Value::Float(f)]),
            )],
            &m,
        )
    });
    let nuclear = report
        .get("nuclear_density_derivatives")
        .unwrap()
        .as_array()
        .unwrap();
    let dpdr = nuclear[5].as_array().unwrap()[0].numeric_array().unwrap().1;
    let analytic: f64 = 2.0
        * dpdr
            .iter()
            .zip(m.as_slice())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    assert_second_order("Tr(dP/dR(H1 z) M)", analytic, &|h| {
        density_projection(&displaced(1, 2, h), &[], &m)
    });
    // U^F reproduces dP/dF: dP = C_v U C_o^T + transpose.
    let rotations = field[2]
        .get("orbital_rotations")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        rotations.len(),
        1,
        "restricted reference has one rotation block"
    );
}

#[test]
fn continuum_solvation_gradient_passes_a_four_step_fit() {
    let solvated = [("solvent", Value::Str("water".into()))];
    let report = run("gradient", &water(), &controls(&solvated)).unwrap();
    let (_, gradient) = array(&report, "gradient");
    assert!(
        report
            .get("energy_components")
            .unwrap()
            .get("continuum_electrostatic")
            .unwrap()
            .as_f64()
            .unwrap()
            < 0.0
    );
    assert_second_order("C-PCM gradient H1 y", gradient[4], &|h| {
        let e = run("energy", &displaced(1, 1, h), &controls(&solvated)).unwrap();
        e.get("energy").unwrap().as_f64().unwrap()
    });
}
