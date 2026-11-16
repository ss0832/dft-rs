// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Born effective charges and the electronic dielectric tensor on the common
//! driver against the independent molecular analytic implementation: an HF
//! molecule in cubic cells of growing edge L. At zero macroscopic field the
//! periodic images polarize each other through the Lorentz local field, so
//! the Born charges approach the molecular atomic polar tensor and
//! `(eps_inf - 1) V / (4 pi)` the molecular polarizability with an error
//! of order `alpha / V`, i.e. `L^-3`; a four-step fit in `h = 1/L` checks the
//! exponent. A polar chain checks charge neutrality with the k derivative
//! active, and the CLI route.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use std::process::Command;

const HF: [(u8, [f64; 3]); 2] = [(9, [0.0, 0.0, 0.0]), (1, [1.7328, 0.0, 0.0])];

fn molecule() -> System {
    System::molecule(
        HF.iter().map(|(z, _)| *z).collect(),
        HF.iter().map(|(_, r)| *r).collect(),
    )
}

fn boxed(edge: f64) -> System {
    System {
        numbers: HF.iter().map(|(z, _)| *z).collect(),
        positions: HF.iter().map(|(_, r)| *r).collect(),
        cell: Some([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("grid_level".to_string(), Value::Int(4)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn periodic(extra: &[(&str, Value)]) -> Controls {
    let mut all = vec![
        (
            "k_mesh",
            Value::Array(vec![Value::Int(1), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff", Value::Float(12.0)),
        ("grid_shell", Value::Int(1)),
        ("max_iter", Value::Int(200)),
    ];
    all.extend(extra.iter().cloned());
    controls(&all)
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn born_charges_and_dielectric_tensor_approach_the_molecular_limit() {
    let gradient = run(
        "multipole-gradient",
        &molecule(),
        &controls(&[("max_rank", Value::Int(1))]),
    )
    .unwrap();
    let ranks = gradient.get("ranks").unwrap().as_array().unwrap();
    // gradient[a][3 atom + b] = d mu_a / d R_(atom, b).
    let apt = flat(&ranks[1], "gradient");
    let alpha = flat(
        &run("polarizability", &molecule(), &controls(&[])).unwrap(),
        "polarizability",
    );
    println!("molecular APT {apt:?}\nmolecular polarizability {alpha:?}");
    // The original 10..16 interval mixes exponential AO-image overlap with
    // the asymptotic L^-3 local field. Its raw failed fit is retained in
    // docs/response.md. Use four independently resolved asymptotic sizes,
    // without removing a point from either fit or changing its gates.
    let edges = [14.0, 16.0, 18.0, 20.0];
    let mut z_errors = Vec::new();
    let mut alpha_errors = Vec::new();
    for &edge in &edges {
        let report = run(
            "born-charges",
            &boxed(edge),
            &periodic(&[("overlap_cutoff", Value::Float(24.0))]),
        )
        .unwrap();
        let z = flat(&report, "born_charges");
        let eps = flat(&report, "epsilon_infinity");
        let volume = edge * edge * edge;
        let mut z_error = 0.0_f64;
        for atom in 0..2 {
            for a in 0..3 {
                for b in 0..3 {
                    z_error =
                        z_error.max((z[9 * atom + 3 * a + b] - apt[6 * a + 3 * atom + b]).abs());
                }
            }
        }
        let mut a_error = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let periodic_alpha = (eps[3 * a + b] - if a == b { 1.0 } else { 0.0 }) * volume
                    / (4.0 * std::f64::consts::PI);
                a_error = a_error.max((periodic_alpha - alpha[3 * a + b]).abs());
            }
        }
        println!(
            "L = {edge}: Born {z:?}\n  eps_inf {eps:?}\n  |Z* - APT| {z_error:e}, |alpha_periodic - alpha| {a_error:e}, sum rule {:?}",
            flat(&report, "born_sum_rule")
        );
        z_errors.push(z_error);
        alpha_errors.push(a_error);
    }
    let h: Vec<f64> = edges.iter().map(|l| 1.0 / l).collect();
    for (label, errors) in [
        ("Born charges", &z_errors),
        ("polarizability", &alpha_errors),
    ] {
        let fit = convergence::error_power_fit(&h, errors);
        println!(
            "{label}: h = 1/L {h:?} errors {errors:?} p={:.6} R2={:.8}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 3.0).abs() < 0.15 && fit.r_squared > 0.999,
            "{label}"
        );
    }
}

/// The same molecular limit with exact exchange: Hartree-Fock and a global
/// hybrid. The k derivative of the non-local exchange operator and the
/// exchange kernel of the coupled-perturbed equations are both needed for
/// the periodic values to approach the molecular ones.
#[test]
fn hybrid_born_charges_and_dielectric_tensor_approach_the_molecular_limit() {
    for xc in ["hf", "pbe0"] {
        let functional = [("xc", Value::Str(xc.into()))];
        let mut with_rank = functional.to_vec();
        with_rank.push(("max_rank", Value::Int(1)));
        let gradient = run("multipole-gradient", &molecule(), &controls(&with_rank)).unwrap();
        let ranks = gradient.get("ranks").unwrap().as_array().unwrap();
        let apt = flat(&ranks[1], "gradient");
        let alpha = flat(
            &run("polarizability", &molecule(), &controls(&functional)).unwrap(),
            "polarizability",
        );
        println!("{xc}: molecular APT {apt:?}\n{xc}: molecular polarizability {alpha:?}");
        let edges = [14.0, 16.0, 18.0, 20.0];
        let mut z_errors = Vec::new();
        let mut alpha_errors = Vec::new();
        for &edge in &edges {
            let report = run(
                "born-charges",
                &boxed(edge),
                &periodic(&[
                    ("xc", Value::Str(xc.into())),
                    ("overlap_cutoff", Value::Float(24.0)),
                ]),
            )
            .unwrap();
            let z = flat(&report, "born_charges");
            let eps = flat(&report, "epsilon_infinity");
            let volume = edge * edge * edge;
            let mut z_error = 0.0_f64;
            for atom in 0..2 {
                for a in 0..3 {
                    for b in 0..3 {
                        z_error = z_error
                            .max((z[9 * atom + 3 * a + b] - apt[6 * a + 3 * atom + b]).abs());
                    }
                }
            }
            let mut a_error = 0.0_f64;
            for a in 0..3 {
                for b in 0..3 {
                    let periodic_alpha = (eps[3 * a + b] - if a == b { 1.0 } else { 0.0 }) * volume
                        / (4.0 * std::f64::consts::PI);
                    a_error = a_error.max((periodic_alpha - alpha[3 * a + b]).abs());
                }
            }
            println!(
                "{xc}: L = {edge}: |Z* - APT| {z_error:e}, |alpha_periodic - alpha| {a_error:e}, sum rule {:?}, Fock image residual {:?}",
                flat(&report, "born_sum_rule"),
                report.get("fock_image_residual")
            );
            z_errors.push(z_error);
            alpha_errors.push(a_error);
        }
        let h: Vec<f64> = edges.iter().map(|l| 1.0 / l).collect();
        for (label, errors) in [
            ("Born charges", &z_errors),
            ("polarizability", &alpha_errors),
        ] {
            let fit = convergence::error_power_fit(&h, errors);
            println!(
                "{xc}: {label}: h = 1/L {h:?} errors {errors:?} p={:.6} R2={:.8}",
                fit.exponent, fit.r_squared
            );
            assert!(
                (fit.exponent - 3.0).abs() < 0.15 && fit.r_squared > 0.999,
                "{xc}: {label}"
            );
        }
    }
}

#[test]
fn molecular_limit_interval_reports_exponential_overlap_separately() {
    use dft_rs::periodic::integrals::{gamma_sum, image_blocks, Operator};
    use dft_rs::periodic::Lattice;
    let system = molecule();
    let mol =
        dft_rs::driver::method::molecule_from(&system.numbers, &system.positions, 0, 1).unwrap();
    let set = dft_rs::driver::method::build_basis("sto-3g", None).unwrap();
    let basis = dft_rs::basis::MolBasis::build(&mol, &set).unwrap();
    let isolated = dft_rs::integrals::overlap_matrix(&basis);
    let mut previous = f64::INFINITY;
    for edge in [10.0, 12.0, 14.0, 16.0, 18.0, 20.0] {
        let physical = gamma_sum(&image_blocks(
            &basis,
            &Lattice::cubic(edge),
            24.0,
            Operator::Overlap,
        ));
        let finite = gamma_sum(&image_blocks(
            &basis,
            &Lattice::cubic(edge),
            12.0,
            Operator::Overlap,
        ));
        let overlap = physical.max_abs_diff(&isolated);
        let truncation = physical.max_abs_diff(&finite);
        println!("molecular box L={edge}: AO-image-overlap={overlap:.12e} cutoff12-to24={truncation:.12e}");
        assert!(overlap < previous);
        previous = overlap;
    }
}

#[test]
fn polar_chain_satisfies_charge_neutrality_and_the_cli_route() {
    // LiH chain along x: one ionic pair per cell, the k derivative active.
    let chain = System {
        numbers: vec![3, 1],
        positions: vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell: Some([[6.2, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]),
        pbc: [true; 3],
    };
    let settings = periodic(&[
        (
            "k_mesh",
            Value::Array(vec![Value::Int(6), Value::Int(1), Value::Int(1)]),
        ),
        ("grid_level", Value::Int(3)),
    ]);
    let report = run("born-charges", &chain, &settings).unwrap();
    let z = flat(&report, "born_charges");
    let rule = flat(&report, "born_sum_rule");
    let imaginary = report
        .get("imaginary_residual")
        .and_then(Value::as_f64)
        .unwrap();
    println!("LiH chain Born {z:?}\n sum rule {rule:?}\n imaginary residual {imaginary:e}\n eps_inf {:?}", flat(&report, "epsilon_infinity"));
    let largest = z.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(largest > 0.1, "an ionic chain has nonzero Born charges");
    assert!(
        rule.iter().all(|v| v.abs() < 1e-5 * largest),
        "charge neutrality {rule:?}"
    );
    assert!(imaginary < 1e-8, "imaginary residual {imaginary:e}");
    let polar = run("crystal-ir", &chain, &settings).unwrap();
    let eps0 = flat(&polar, "epsilon_static");
    let eps_inf = flat(&polar, "epsilon_infinity");
    println!(
        "eps_0 {eps0:?}; TO {:?}; LO {:?}",
        polar.get("frequencies_cm1"),
        polar.get("lo_frequencies_cm1")
    );
    assert!(
        eps0[0] >= eps_inf[0],
        "the ionic part is non-negative along a stable polar axis"
    );
    // CLI route.
    let path = std::env::temp_dir().join(format!("dft_born_{}.xyz", std::process::id()));
    let a = dft_rs::constants::BOHR_TO_ANGSTROM;
    std::fs::write(
        &path,
        format!("2\nchain\nLi 0 0 0\nH {:.16} 0 0\n", 3.0 * a),
    )
    .unwrap();
    let cell = format!(
        "{:.16},0,0;0,{:.16},0;0,0,{:.16}",
        6.2 * a,
        8.0 * a,
        8.0 * a
    );
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .args([
            "born-charges",
            path.to_str().unwrap(),
            "--cell",
            &cell,
            "--xc",
            "svwn5",
            "--basis",
            "sto-3g",
            "--k-mesh",
            "6,1,1",
            "--overlap-cutoff",
            "12",
            "--grid-level",
            "3",
            "--grid-shell",
            "1",
            "--conv-tol",
            "1e-12",
            "--conv-tol-dp",
            "1e-10",
            "--max-iter",
            "200",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json = Value::parse_json(&String::from_utf8_lossy(&output.stdout)).unwrap();
    for (x, y) in flat(&json, "born_charges").iter().zip(&z) {
        assert!((x - y).abs() < 1e-8, "{x} vs {y}");
    }
}

#[test]
fn omega_b97x_d_polar_modes_share_the_dispersion_corrected_gamma_hessian() {
    // A generically oriented NaH bond in a triclinic molecular-crystal cell.
    // This compact ECP basis keeps the test on the shared 3D response route.
    let crystal = System {
        numbers: vec![11, 1],
        positions: vec![[0.13, -0.21, 0.07], [2.885, -1.879, 2.282]],
        cell: Some([[11.4, 0.71, 0.33], [0.45, 12.8, 0.69], [0.24, -0.55, 13.2]]),
        pbc: [true; 3],
    };
    let settings = Value::object()
        .with("xc", "wb97x-d")
        .with("reference", "rks")
        .with(
            "basis",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/periodic_scalar_ecp.g94"
            ),
        )
        .with("overlap_cutoff", 5.5)
        .with("exchange_cutoff", 5.5)
        .with("xc_ao_cutoff", 13.0)
        .with("disp_cutoff", 14.0)
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("response_tolerance", 1e-11);
    let calculate = |feature, settings: &Value| {
        let report = run(feature, &crystal, &Controls::from_value(settings).unwrap()).unwrap();
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        assert_eq!(
            report.get("dispersion_model").and_then(Value::as_str),
            Some("Chg")
        );
        report
    };
    let born = calculate("born-charges", &settings);
    let phonons = calculate("phonons", &settings);
    let directions = [[0.53, -0.41, 0.72], [0.11, 0.63, 0.44]];
    let polar = calculate("crystal-ir", &settings.with("directions", directions));
    let number = |report: &Value, key| report.get(key).and_then(Value::as_f64).unwrap();
    let frequencies = flat(&polar, "frequencies_cm1");
    let lo = flat(&polar, "lo_frequencies_cm1");
    let epsilon = flat(&polar, "epsilon_infinity");
    let charges = flat(&polar, "born_charges");
    let sum = flat(&polar, "born_sum_rule");
    for (a, b) in frequencies.iter().zip(flat(&phonons, "frequencies_cm1")) {
        // The near-zero acoustic frequencies differ by a square root of
        // round-off in the mass-weighted matrix.
        let bound = if a.abs() < 1e-2 { 1e-3 } else { 1e-6 };
        assert!((a - b).abs() < bound);
    }
    for key in ["epsilon_infinity", "born_charges"] {
        assert!(flat(&polar, key)
            .iter()
            .zip(flat(&born, key))
            .all(|(a, b)| (a - b).abs() < 1e-9));
    }
    for report in [&born, &phonons] {
        for key in ["energy", "dispersion_energy"] {
            assert!((number(report, key) - number(&polar, key)).abs() < 1e-11);
        }
    }
    assert!(number(&polar, "dispersion_energy") < 0.0);
    assert!(sum.iter().all(|x| x.abs() < 1e-8));
    assert!(number(&polar, "acoustic_residual") < 1e-8);
    assert!(number(&polar, "fock_image_residual") < 1e-8);
    // A rank-one LO correction raises the trace by its closed-form
    // mass-normalized squared longitudinal charge, for each direction.
    let volume = number(&polar, "volume");
    let to_wavenumber = dft_rs::constants::HARTREE_TO_CM1;
    let signed_square = |w: &f64| w.signum() * (w / to_wavenumber).powi(2);
    let trace: f64 = frequencies.iter().map(signed_square).sum();
    for (direction, modes) in directions.iter().zip(lo.chunks_exact(frequencies.len())) {
        let screening = (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .map(|(a, b)| direction[a] * epsilon[3 * a + b] * direction[b])
            .sum::<f64>();
        let strength: f64 = crystal
            .numbers
            .iter()
            .enumerate()
            .map(|(atom, &z)| {
                (0..3)
                    .map(|b| {
                        (0..3)
                            .map(|a| direction[a] * charges[9 * atom + 3 * a + b])
                            .sum::<f64>()
                            .powi(2)
                    })
                    .sum::<f64>()
                    / dft_rs::constants::atomic_mass_au(z)
            })
            .sum();
        let expected = 4.0 * std::f64::consts::PI * strength / (volume * screening);
        assert!(expected > 0.0);
        assert!((modes.iter().map(signed_square).sum::<f64>() - trace - expected).abs() < 1e-11);
    }
    println!("NaH omegaB97X-D E={:.15e} E_disp={:.15e} epsilon={epsilon:?} Born={charges:?} TO={frequencies:?} LO={lo:?} solves={:?}",
        number(&polar, "energy"), number(&polar, "dispersion_energy"), polar.get("displacement_solves"));
}
