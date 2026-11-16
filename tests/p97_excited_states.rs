// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance tests for molecular excited states on the common driver: the
//! driver reproduces the library solver, the complete RPA spectrum
//! reproduces the analytic CPKS polarizability through its sum over states,
//! TDA energies bound the RPA ones from above, broadened spectra integrate
//! to their line strengths and change sign under reflection, and the
//! analytic excited-state gradient passes an independent four-step error
//! fit.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use std::process::Command;

/// Water distorted away from C2v (Bohr): no gradient component vanishes by
/// symmetry and no low-lying root is degenerate.
const WATER: [(u8, [f64; 3]); 3] = [
    (8, [0.0, 0.0, 0.2300]),
    (1, [0.0, 1.4309, -0.8867]),
    (1, [0.1200, -1.4809, -0.9067]),
];

fn water() -> System {
    System::molecule(
        WATER.iter().map(|(z, _)| *z).collect(),
        WATER.iter().map(|(_, r)| *r).collect(),
    )
}

/// Twisted hydrogen peroxide (Bohr), a chiral molecule with a dihedral angle
/// of 110 degrees; `mirror` reflects it through the xy plane.
fn peroxide(mirror: bool) -> System {
    let s = if mirror { -1.0 } else { 1.0 };
    System::molecule(
        vec![8, 8, 1, 1],
        vec![
            [-1.37, 0.0, 0.0],
            [1.37, 0.0, 0.0],
            [-1.6883, 1.03539, 1.4787 * s],
            [1.6883, 1.03539, -1.4787 * s],
        ],
    )
}

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

fn tight(xc: &str) -> Vec<(&'static str, Value)> {
    vec![
        ("basis", Value::Str("sto-3g".into())),
        ("xc", Value::Str(xc.into())),
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-12)),
    ]
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn states(report: &Value) -> &[Value] {
    report.get("states").unwrap().as_array().unwrap()
}

fn excitation_request(xc: &str, nstates: i64, method: &str) -> Vec<(&'static str, Value)> {
    let mut request = tight(xc);
    request.push(("nstates", Value::Int(nstates)));
    request.push(("td_method", Value::Str(method.into())));
    request
}

#[test]
fn driver_excitations_reproduce_the_library_solver() {
    use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
    use dft_rs::response::tddft::{TdaSolver, TdaSpin};
    let system = water();
    let method = MolecularMethod::from_controls(&controls(&tight("svwn5"))).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    for (name, tamm_dancoff) in [("tda", true), ("rpa", false)] {
        let solver = TdaSolver::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            TdaSpin::Singlet,
        )
        .unwrap();
        let expected = if tamm_dancoff {
            solver.solve(4).unwrap()
        } else {
            solver.solve_rpa(4).unwrap()
        };
        let report = run(
            "excitations",
            &system,
            &controls(&excitation_request("svwn5", 4, name)),
        )
        .unwrap();
        assert_eq!(report.get("td_method").and_then(Value::as_str), Some(name));
        assert_eq!(
            report.get("td_spin").and_then(Value::as_str),
            Some("singlet")
        );
        let found = states(&report);
        assert_eq!(found.len(), 4);
        for (state, library) in found.iter().zip(&expected) {
            assert!((number(state, "energy") - library.energy).abs() < 1e-10);
            assert!(
                (number(state, "oscillator_strength") - library.oscillator_strength).abs() < 1e-10
            );
            // Rotatory strengths are reported for the Tamm-Dancoff states only.
            assert_eq!(state.get("rotatory_strength").is_some(), tamm_dancoff);
        }
    }
}

#[test]
fn complete_rpa_spectrum_reproduces_the_cpks_polarizability() {
    // hf: Hartree-Fock exchange; svwn5: LDA; pbe: GGA; tpss: meta-GGA;
    // b3lyp and pbe0: global hybrids; camb3lyp: range-separated hybrid.
    // wb97xd is a range-separated hybrid with an empirical dispersion term and m06 a hybrid meta-GGA.
    for xc in [
        "hf", "svwn5", "pbe", "tpss", "b3lyp", "pbe0", "camb3lyp", "wb97xd", "m06",
    ] {
        let rpa = run(
            "excitations",
            &water(),
            &controls(&excitation_request(xc, 1000, "rpa")),
        )
        .unwrap();
        let tda = run(
            "excitations",
            &water(),
            &controls(&excitation_request(xc, 1000, "tda")),
        )
        .unwrap();
        // Five doubly occupied and two empty orbitals: ten singlet excitations.
        assert_eq!(states(&rpa).len(), 10);
        assert_eq!(states(&tda).len(), 10);
        // alpha_ab = sum_n 2 d_a d_b / omega_n over the complete spectrum.
        let mut alpha = [[0.0_f64; 3]; 3];
        for state in states(&rpa) {
            let omega = number(state, "energy");
            let d = vector(state, "transition_dipole");
            for a in 0..3 {
                for b in 0..3 {
                    alpha[a][b] += 2.0 * d[a] * d[b] / omega;
                }
            }
        }
        let mut request = tight(xc);
        request.push(("response_tolerance", Value::Float(1e-11)));
        let cpks = vector(
            &run("polarizability", &water(), &controls(&request)).unwrap(),
            "polarizability",
        );
        let mut deviation = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                deviation = deviation.max((alpha[a][b] - cpks[3 * a + b]).abs());
            }
        }
        println!(
            "{xc}: sum-over-states polarizability deviates from CPKS by {deviation:.3e} \
             (alpha_xx {:.8}, alpha_yy {:.8}, alpha_zz {:.8})",
            alpha[0][0], alpha[1][1], alpha[2][2]
        );
        assert!(deviation < 1e-7, "{xc}: {deviation:e}");
        // The Tamm-Dancoff energies bound the RPA ones from above, root by root.
        for (t, r) in states(&tda).iter().zip(states(&rpa)) {
            assert!(number(t, "energy") >= number(r, "energy") - 1e-10);
        }
    }
}

#[test]
fn broadened_spectra_integrate_to_line_strengths_and_mirror_images_are_opposite() {
    for xc in ["hf", "pbe", "b3lyp", "wb97xd", "m06"] {
        let mut request = tight(xc);
        request.push(("nstates", Value::Int(6)));
        request.push(("broadening_ev", Value::Float(0.3)));
        request.push(("points", Value::Int(4001)));
        let left = run("ecd-spectrum", &peroxide(false), &controls(&request)).unwrap();
        let right = run("ecd-spectrum", &peroxide(true), &controls(&request)).unwrap();
        // Sum rules over the default window, which ends five standard deviations
        // beyond the outer bands.
        let f_sum = number(&left, "oscillator_strength_sum");
        let f_integral = number(&left, "absorption_integral_oscillator_strength");
        let r_sum = vector(&left, "rotatory_strength_sum_cgs");
        let r_integral = vector(&left, "ecd_integral_rotatory_strength_cgs");
        let scale: f64 = states(&left)
            .iter()
            .map(|s| number(s, "rotatory_strength_velocity_cgs").abs())
            .sum();
        println!(
            "{xc} peroxide: sum f {f_sum:.8} integral {f_integral:.8}; sum R {r_sum:?} integral \
             {r_integral:?} (1e-40 erg cm^3), sum |R_velocity| {scale:.4}"
        );
        assert!(scale > 0.1, "the test molecule must be optically active");
        assert!((f_integral / f_sum - 1.0).abs() < 1e-5);
        for gauge in 0..2 {
            assert!((r_integral[gauge] - r_sum[gauge]).abs() < 1e-5 * scale);
        }
        // Reflection keeps the absorption and reverses the circular dichroism.
        let absorption = (
            vector(&left, "molar_absorptivity"),
            vector(&right, "molar_absorptivity"),
        );
        let peak = absorption.0.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(peak > 0.0);
        for (a, b) in absorption.0.iter().zip(&absorption.1) {
            assert!((a - b).abs() < 1e-6 * peak);
        }
        for key in ["ecd_length", "ecd_velocity"] {
            let (a, b) = (vector(&left, key), vector(&right, key));
            let peak = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(peak > 0.0, "{key} vanishes");
            for (x, y) in a.iter().zip(&b) {
                assert!((x + y).abs() < 1e-6 * peak, "{key}: {x} {y}");
            }
        }
    }
}

/// Phase-independent similarity of two amplitude sets, `sum |x| |y|`: an
/// orbital that changes sign between two geometries does not change it.
fn amplitude_similarity(a: &Value, b: &Value) -> f64 {
    let x = a.get("amplitudes").unwrap().numeric_array().unwrap().1;
    let y = b.get("amplitudes").unwrap().numeric_array().unwrap().1;
    x.iter().zip(&y).map(|(p, q)| (p * q).abs()).sum()
}

fn amplitude_request(xc: &str) -> Vec<(&'static str, Value)> {
    let mut request = excitation_request(xc, 4, "tda");
    request.push(("amplitudes", Value::Bool(true)));
    request
}

/// Total energy of the excited state that matches `reference` at a
/// displaced geometry.
fn followed_energy(system: &System, xc: &str, reference: &Value) -> f64 {
    let report = run("excitations", system, &controls(&amplitude_request(xc))).unwrap();
    let best = states(&report)
        .iter()
        .max_by(|u, v| {
            amplitude_similarity(reference, u)
                .partial_cmp(&amplitude_similarity(reference, v))
                .unwrap()
        })
        .unwrap();
    let similarity = amplitude_similarity(reference, best);
    assert!(
        similarity > 0.9,
        "root following lost the state: {similarity}"
    );
    number(&report, "energy") + number(best, "energy")
}

#[test]
fn excited_state_gradient_passes_an_independent_four_step_error_fit() {
    // The analytic excited-state gradient exists for Hartree-Fock and pure LDA
    // references only; GGA, meta-GGA and hybrid references are refused (checked
    // in the refusal test below).
    for xc in ["hf", "svwn5"] {
        let mut request = tight(xc);
        request.push(("root", Value::Int(1)));
        request.push(("nstates", Value::Int(4)));
        let report = run("td-state-gradient", &water(), &controls(&request)).unwrap();
        let gradient = vector(&report, "gradient");
        let residual = number(&report, "translation_residual");
        let unperturbed = run("excitations", &water(), &controls(&amplitude_request(xc))).unwrap();
        let state = states(&unperturbed)[0].clone();
        assert!(
            (number(&report, "excited_state_energy")
                - number(&unperturbed, "energy")
                - number(&state, "energy"))
            .abs()
                < 1e-10
        );
        println!("{xc}: translation residual {residual:.3e}");
        assert!(residual < 1e-6, "{xc}: {residual:e}");
        // Components: oxygen z (index 2) and first hydrogen y (index 4).
        let steps = [0.02, 0.01, 0.005, 0.0025];
        for component in [2usize, 4] {
            let (atom, axis) = (component / 3, component % 3);
            let energy_at = |h: f64| {
                let mut system = water();
                system.positions[atom][axis] += h;
                followed_energy(&system, xc, &state)
            };
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| ((energy_at(h) - energy_at(-h)) / (2.0 * h) - gradient[component]).abs())
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            println!(
                "{xc} excited-state gradient component {component}: analytic {:.10} steps \
                 {steps:?} errors {errors:?} p={:.6} R2={:.8}",
                gradient[component], fit.exponent, fit.r_squared
            );
            assert!((fit.exponent - 2.0).abs() < 0.02, "p = {}", fit.exponent);
            assert!(fit.r_squared > 0.99999, "R2 = {}", fit.r_squared);
        }
    }
}

#[test]
fn unsupported_excited_state_requests_are_refused_by_name() {
    let failure = |feature: &str, xc: &str, extra: &[(&'static str, Value)]| -> String {
        let mut request = tight(xc);
        request.extend(extra.iter().cloned());
        run(feature, &water(), &controls(&request))
            .unwrap_err()
            .to_string()
    };
    let gga = failure("td-state-gradient", "pbe", &[]);
    assert!(gga.contains("GGA"), "{gga}");
    let meta_gga = failure("td-state-gradient", "tpss", &[]);
    assert!(meta_gga.contains("GGA"), "{meta_gga}");
    let hybrid = failure("td-state-gradient", "b3lyp", &[]);
    assert!(!hybrid.is_empty(), "the hybrid refusal has no text");
    println!("b3lyp excited-state gradient refusal: {hybrid}");
    // B3LYP is refused through its gradient-corrected part, which is checked before the exchange admixture.
    assert!(hybrid.contains("GGA"), "{hybrid}");
    let field = failure(
        "td-state-gradient",
        "hf",
        &[(
            "field",
            Value::Array(vec![
                Value::Float(0.0),
                Value::Float(0.0),
                Value::Float(0.001),
            ]),
        )],
    );
    assert!(field.contains("external electric field"), "{field}");
    let rpa = failure(
        "ecd-spectrum",
        "hf",
        &[
            ("broadening_ev", Value::Float(0.3)),
            ("td_method", Value::Str("rpa".into())),
        ],
    );
    assert!(rpa.contains("RPA"), "{rpa}");
    let missing = failure("ecd-spectrum", "hf", &[]);
    assert!(missing.contains("broadening_ev"), "{missing}");
    let smeared = failure(
        "excitations",
        "hf",
        &[("electronic_temperature", Value::Float(3000.0))],
    );
    assert!(smeared.contains("fractionally occupied"), "{smeared}");
    let misspelled = failure("excitations", "hf", &[("nstate", Value::Int(3))]);
    assert!(misspelled.contains("nstate"), "{misspelled}");
    let root = failure("td-state-gradient", "hf", &[("root", Value::Int(0))]);
    assert!(root.contains("root"), "{root}");
}

#[test]
fn cli_excitations_print_the_driver_report() {
    let path = std::env::temp_dir().join(format!("dft_excited_{}.xyz", std::process::id()));
    let a = dft_rs::constants::BOHR_TO_ANGSTROM;
    let mut text = String::from("3\nwater\n");
    for (z, r) in WATER {
        text.push_str(&format!(
            "{} {:.16} {:.16} {:.16}\n",
            dft_rs::z_to_symbol(z).unwrap(),
            r[0] * a,
            r[1] * a,
            r[2] * a
        ));
    }
    std::fs::write(&path, text).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .args([
            "excitations",
            path.to_str().unwrap(),
            "--basis",
            "sto-3g",
            "--xc",
            "hf",
            "--conv-tol",
            "1e-13",
            "--conv-tol-dp",
            "1e-12",
            "--nstates",
            "3",
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
    let printed = Value::parse_json(&String::from_utf8_lossy(&output.stdout)).unwrap();
    let report = run(
        "excitations",
        &water(),
        &controls(&excitation_request("hf", 3, "tda")),
    )
    .unwrap();
    assert_eq!(states(&printed).len(), 3);
    for (a, b) in states(&printed).iter().zip(states(&report)) {
        assert!((number(a, "energy") - number(b, "energy")).abs() < 1e-9);
        assert!((number(a, "oscillator_strength") - number(b, "oscillator_strength")).abs() < 1e-8);
    }
}

/// Bent magnesium hydride (Bohr) with a small-core effective potential on
/// magnesium: no symmetry, so no excited state is degenerate.
fn magnesium_hydride() -> System {
    System::molecule(
        vec![12, 1, 1],
        vec![[0.0, 0.0, 0.0], [3.1, 0.6, 0.2], [-2.9, 0.9, -0.3]],
    )
}

#[test]
fn excited_state_gradient_with_an_effective_core_potential_passes_a_four_step_fit() {
    for xc in ["hf", "svwn5"] {
        let request = |extra: &[(&'static str, Value)]| {
            let mut pairs = vec![
                ("basis", Value::Str("lanl2dz".into())),
                ("xc", Value::Str(xc.into())),
                ("conv_tol", Value::Float(1e-13)),
                ("conv_tol_dp", Value::Float(1e-12)),
                ("nstates", Value::Int(4)),
            ];
            pairs.extend(extra.iter().cloned());
            controls(&pairs)
        };
        let report = run(
            "td-state-gradient",
            &magnesium_hydride(),
            &request(&[("root", Value::Int(1))]),
        )
        .unwrap();
        let gradient = vector(&report, "gradient");
        let residual = number(&report, "translation_residual");
        println!("{xc} with an effective core potential: translation residual {residual:.3e}");
        assert!(residual < 1e-6, "{xc}: {residual:e}");
        let with_amplitudes = [("amplitudes", Value::Bool(true))];
        let unperturbed = run(
            "excitations",
            &magnesium_hydride(),
            &request(&with_amplitudes),
        )
        .unwrap();
        let state = states(&unperturbed)[0].clone();
        // Components: magnesium x (index 0) and first hydrogen y (index 4).
        let steps = [0.02, 0.01, 0.005, 0.0025];
        for component in [0usize, 4] {
            let (atom, axis) = (component / 3, component % 3);
            let energy_at = |h: f64| {
                let mut system = magnesium_hydride();
                system.positions[atom][axis] += h;
                let displaced = run("excitations", &system, &request(&with_amplitudes)).unwrap();
                let best = states(&displaced)
                    .iter()
                    .max_by(|u, v| {
                        amplitude_similarity(&state, u)
                            .partial_cmp(&amplitude_similarity(&state, v))
                            .unwrap()
                    })
                    .unwrap();
                let similarity = amplitude_similarity(&state, best);
                assert!(
                    similarity > 0.9,
                    "root following lost the state: {similarity}"
                );
                number(&displaced, "energy") + number(best, "energy")
            };
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| ((energy_at(h) - energy_at(-h)) / (2.0 * h) - gradient[component]).abs())
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            println!(
                "{xc} effective-core-potential gradient component {component}: analytic {:.10} \
                 steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
                gradient[component], fit.exponent, fit.r_squared
            );
            assert!((fit.exponent - 2.0).abs() < 0.02, "p = {}", fit.exponent);
            assert!(fit.r_squared > 0.99999, "R2 = {}", fit.r_squared);
        }
    }
}
