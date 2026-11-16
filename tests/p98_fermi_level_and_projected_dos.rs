// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance tests for the projected density of states and the Fermi-level
//! properties on the common driver: projections sum to the total and
//! integrate to the Mulliken populations of the density matrix; the Pauli
//! susceptibility and the electronic heat capacity pass independent
//! four-step error fits against central differences of the magnetization
//! and of the band energy; and the unit conversions match tabulated
//! constants.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};

/// Hydrogen fluoride chain along x in a cell with vacuum (Bohr).
fn hydrogen_fluoride() -> System {
    System {
        numbers: vec![9, 1],
        positions: vec![[0.0, 0.0, 0.0], [1.75, 0.15, 0.1]],
        cell: Some([[4.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]),
        pbc: [true; 3],
    }
}

/// Hydrogen chain with nearly equal bonds: partially occupied bands at an
/// electronic temperature of 3000 K.
fn metal() -> System {
    System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, 0.0], [1.75, 0.1, 0.05]],
        cell: Some([[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]),
        pbc: [true; 3],
    }
}

fn controls(mesh: i64, extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(mesh), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(14.0)),
        ("grid_level".to_string(), Value::Int(3)),
        ("grid_shell".to_string(), Value::Int(3)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(300)),
    ];
    // An extra control replaces the default entry with the same key.
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn first_channel(report: &Value) -> &Value {
    &report.get("channels").unwrap().as_array().unwrap()[0]
}

fn trapezoid(grid: &[f64], values: &[f64], upper: f64) -> f64 {
    let mut sum = 0.0;
    for i in 1..grid.len() {
        if grid[i] > upper {
            break;
        }
        sum += 0.5 * (grid[i] - grid[i - 1]) * (values[i] + values[i - 1]);
    }
    sum
}

#[test]
fn projected_dos_sums_to_the_total_and_integrates_to_mulliken_populations() {
    let structure = run(
        "electronic-structure",
        &hydrogen_fluoride(),
        &controls(4, &[]),
    )
    .unwrap();
    let edges = first_channel(&structure).get("band_edges").unwrap();
    let gap = number(edges, "gap");
    let mid = 0.5 * (number(edges, "valence_maximum") + number(edges, "conduction_minimum"));
    let charges = flat(&structure, "mulliken");
    println!("HF chain: gap {gap:.6} Hartree, Mulliken charges {charges:?}");
    assert!(gap > 0.05, "the test cell must be insulating: gap {gap}");
    let sigma = 0.004;
    let request = [
        ("sigma", Value::Float(sigma)),
        ("points", Value::Int(6000)),
        ("projected", Value::Bool(true)),
    ];
    let report = run("dos", &hydrogen_fluoride(), &controls(4, &request)).unwrap();
    let channel = first_channel(&report);
    let grid = flat(&report, "energies");
    let total = flat(channel, "dos");
    let atoms = flat(channel, "projected_atoms");
    let shells = flat(channel, "projected_shells");
    let groups = report.get("projection_groups").unwrap().as_array().unwrap();
    let points = grid.len();
    let peak = total.iter().fold(0.0_f64, |m, v| m.max(*v));
    // Fluorine carries s and p functions, hydrogen s only.
    let labels: Vec<(i64, &str)> = groups
        .iter()
        .map(|g| {
            (
                g.get("atom").and_then(Value::as_i64).unwrap(),
                g.get("label").and_then(Value::as_str).unwrap(),
            )
        })
        .collect();
    assert_eq!(labels, vec![(0, "s"), (0, "p"), (1, "s")]);
    assert_eq!(atoms.len(), 2 * points);
    assert_eq!(shells.len(), 3 * points);
    // Shells sum to their atom, atoms sum to the total.
    let mut worst = 0.0_f64;
    for i in 0..points {
        let fluorine = shells[i] + shells[points + i];
        worst = worst.max((fluorine - atoms[i]).abs());
        worst = worst.max((shells[2 * points + i] - atoms[points + i]).abs());
        worst = worst.max((atoms[i] + atoms[points + i] - total[i]).abs());
    }
    let residual = number(channel, "projection_sum_residual");
    println!("projection sum residual {residual:.3e}, worst partition defect {worst:.3e}, peak {peak:.4}");
    assert!(worst < 1e-9 * peak);
    assert!(residual < 1e-9 * peak);
    // The occupied part of each atom-projected density of states is the
    // Mulliken population of the density matrix.
    for atom in 0..2 {
        let population = trapezoid(&grid, &atoms[atom * points..(atom + 1) * points], mid);
        let expected = [9.0, 1.0][atom] - charges[atom];
        println!(
            "atom {atom}: integrated projection {population:.8}, Mulliken population {expected:.8}"
        );
        assert!((population - expected).abs() < 1e-5);
    }
    // The reconstruction route on the same mesh gives the same curves.
    let mut again = request.to_vec();
    again.push((
        "dos_mesh",
        Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
    ));
    let rebuilt = run("dos", &hydrogen_fluoride(), &controls(4, &again)).unwrap();
    let rebuilt_atoms = flat(first_channel(&rebuilt), "projected_atoms");
    let mut difference = 0.0_f64;
    for (a, b) in atoms.iter().zip(&rebuilt_atoms) {
        difference = difference.max((a - b).abs());
    }
    println!("SCF mesh against reconstructed mesh: {difference:.3e}");
    assert!(difference < 1e-4 * peak);
    // Without the projection the total is unchanged.
    let plain = run(
        "dos",
        &hydrogen_fluoride(),
        &controls(
            4,
            &[("sigma", Value::Float(sigma)), ("points", Value::Int(6000))],
        ),
    )
    .unwrap();
    assert!(first_channel(&plain).get("projected_atoms").is_none());
    let plain_total = flat(first_channel(&plain), "dos");
    for (a, b) in total.iter().zip(&plain_total) {
        assert!((a - b).abs() < 1e-4 * peak);
    }
}

#[test]
fn projected_dos_holds_for_gradient_corrected_meta_and_hybrid_functionals() {
    let system = System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, 0.0], [1.5, 0.1, 0.0]],
        cell: Some([[3.6, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]]),
        pbc: [true; 3],
    };
    for xc in ["pbe", "tpss", "pbe0", "m06"] {
        let functional = [("xc", Value::Str(xc.into()))];
        let structure = run("electronic-structure", &system, &controls(4, &functional)).unwrap();
        let edges = first_channel(&structure).get("band_edges").unwrap();
        let gap = number(edges, "gap");
        let mid = 0.5 * (number(edges, "valence_maximum") + number(edges, "conduction_minimum"));
        let charges = flat(&structure, "mulliken");
        let request = [
            ("xc", Value::Str(xc.into())),
            ("sigma", Value::Float(0.004)),
            ("points", Value::Int(4000)),
            ("projected", Value::Bool(true)),
        ];
        let report = run("dos", &system, &controls(4, &request)).unwrap();
        let channel = first_channel(&report);
        let grid = flat(&report, "energies");
        let total = flat(channel, "dos");
        let atoms = flat(channel, "projected_atoms");
        let points = grid.len();
        let peak = total.iter().fold(0.0_f64, |m, v| m.max(*v));
        let residual = number(channel, "projection_sum_residual");
        assert!(residual < 1e-9 * peak, "{xc}: residual {residual:e}");
        // The occupied part of each atom-projected density of states is the
        // Mulliken population of the density matrix (hydrogen carries one
        // valence electron).
        let populations: Vec<f64> = (0..2)
            .map(|atom| trapezoid(&grid, &atoms[atom * points..(atom + 1) * points], mid))
            .collect();
        println!(
            "{xc}: gap {gap:.6} Hartree, projection sum residual {residual:.3e}, integrated \
             populations {:.8} and {:.8}, Mulliken populations {:.8} and {:.8}",
            populations[0],
            populations[1],
            1.0 - charges[0],
            1.0 - charges[1]
        );
        for atom in 0..2 {
            let expected = 1.0 - charges[atom];
            assert!(
                (populations[atom] - expected).abs() < 1e-5,
                "{xc} atom {atom}: {} against {expected}",
                populations[atom]
            );
        }
    }
}

fn occupation(x: f64) -> f64 {
    1.0 / (1.0 + x.exp())
}

/// Chemical potential of `levels` (each with the weight `weight`) holding
/// `electrons` electrons at the temperature `kt`, by bisection.
fn chemical_potential(levels: &[f64], weight: f64, electrons: f64, kt: f64) -> f64 {
    let mut lo = levels.iter().copied().fold(f64::INFINITY, f64::min) - 1.0;
    let mut hi = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let count = weight
            * levels
                .iter()
                .map(|e| occupation((e - mid) / kt))
                .sum::<f64>();
        if count < electrons {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Shared Fermi-level procedure on the metal with the extra controls: the
/// chemical potential, the Pauli susceptibility and the electronic heat
/// capacity are checked against independent evaluations and four-step error
/// fits. Returns the pauli-spin-susceptibility report.
fn fermi_level_fits(extra: &[(&str, Value)]) -> Value {
    let structure = run("electronic-structure", &metal(), &controls(6, extra)).unwrap();
    let bands = flat(first_channel(&structure), "band_energies");
    let scf_mu = number(first_channel(&structure), "chemical_potential");
    let report = run("pauli-spin-susceptibility", &metal(), &controls(6, extra)).unwrap();
    let row = &report.get("rows").unwrap().as_array().unwrap()[0];
    assert_eq!(number(row, "temperature_k"), 3000.0);
    let kt = 3000.0 * dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let weight = 2.0 / 6.0;
    let electrons = 2.0;
    // Independent evaluation from the band energies of the SCF mesh.
    let mu = chemical_potential(&bands, weight, electrons, kt);
    let l0 = weight
        * bands
            .iter()
            .map(|e| {
                let f = occupation((e - mu) / kt);
                f * (1.0 - f) / kt
            })
            .sum::<f64>();
    let dos = number(row, "dos_at_fermi_level");
    let heat_capacity = number(row, "heat_capacity");
    println!(
        "metal: mu {mu:.10} (driver {:.10}, SCF {scf_mu:.10}), L0 {l0:.8} (driver {dos:.8}), \
         C {heat_capacity:.8} k_B, Wilson ratio {:.6}",
        number(row, "chemical_potential"),
        number(row, "wilson_ratio")
    );
    assert!((number(row, "chemical_potential") - mu).abs() < 1e-10);
    assert!((scf_mu - mu).abs() < 1e-6);
    assert!((number(row, "electrons") - electrons).abs() < 1e-12);
    assert!(
        l0 > 1e-3,
        "the test cell must have states at the Fermi level: {l0}"
    );
    assert!((dos / l0 - 1.0).abs() < 1e-9);

    // Magnetization in Bohr magnetons under a rigid Zeeman shift b of the
    // two spin channels at the same electron count: dm/db = L0.
    let magnetization = |b: f64| -> f64 {
        let mut levels: Vec<f64> = bands.iter().map(|e| e - b).collect();
        levels.extend(bands.iter().map(|e| e + b));
        let half = 0.5 * weight;
        let shifted = chemical_potential(&levels, half, electrons, kt);
        half * bands
            .iter()
            .map(|e| occupation((e - b - shifted) / kt) - occupation((e + b - shifted) / kt))
            .sum::<f64>()
    };
    let steps: Vec<f64> = [0.1, 0.05, 0.025, 0.0125].iter().map(|s| s * kt).collect();
    let errors: Vec<f64> = steps
        .iter()
        .map(|&b| ((magnetization(b) - magnetization(-b)) / (2.0 * b) - dos).abs())
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "Pauli susceptibility: steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.02, "p = {}", fit.exponent);
    assert!(fit.r_squared > 0.99999, "R2 = {}", fit.r_squared);

    // Heat capacity in k_B: central differences of the band energy in kT at
    // the same electron count.
    let band_energy = |t: f64| -> f64 {
        let shifted = chemical_potential(&bands, weight, electrons, t);
        weight
            * bands
                .iter()
                .map(|e| e * occupation((e - shifted) / t))
                .sum::<f64>()
    };
    assert!((number(row, "band_energy") - band_energy(kt)).abs() < 1e-10);
    // Recorded: on the interval [0.1, 0.05, 0.025, 0.0125] kT the errors were
    // 9.62e-5, 2.19e-5, 5.35e-6, 1.33e-6 (p = 2.057, R2 = 0.99980): with six
    // k points the fourth-order term of the difference is still visible at
    // the coarse steps. The interval below resolves the second order.
    let steps: Vec<f64> = [0.025, 0.0125, 0.00625, 0.003125]
        .iter()
        .map(|s| s * kt)
        .collect();
    let errors: Vec<f64> = steps
        .iter()
        .map(|&h| ((band_energy(kt + h) - band_energy(kt - h)) / (2.0 * h) - heat_capacity).abs())
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "electronic heat capacity: steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.02, "p = {}", fit.exponent);
    assert!(fit.r_squared > 0.99999, "R2 = {}", fit.r_squared);
    report
}

#[test]
fn fermi_level_properties_pass_independent_four_step_fits() {
    let smeared = [("electronic_temperature", Value::Float(3000.0))];
    let report = fermi_level_fits(&smeared);
    let row = &report.get("rows").unwrap().as_array().unwrap()[0];

    // Unit conversions against tabulated constants (CODATA 2018):
    // chi_molar = mu_B^2 N_A g and gamma = (pi^2/3) k_B^2 N_A g.
    let bohr_magneton_erg_per_gauss: f64 = 9.2740100783e-21;
    let avogadro: f64 = 6.02214076e23;
    let ev_in_erg: f64 = 1.602176634e-12;
    let boltzmann_j_per_k: f64 = 1.380649e-23;
    let per_ev = number(row, "dos_at_fermi_level_per_ev");
    let molar = bohr_magneton_erg_per_gauss.powi(2) * avogadro * per_ev / ev_in_erg;
    let gamma = std::f64::consts::PI.powi(2) / 3.0 * boltzmann_j_per_k.powi(2) * avogadro * per_ev
        / (ev_in_erg * 1e-7)
        * 1000.0;
    println!(
        "chi_molar {:.6e} cm^3/mol (tabulated route {molar:.6e}), gamma from DOS {:.6} \
         mJ/(mol K^2) (tabulated route {gamma:.6})",
        number(row, "pauli_susceptibility_molar_cgs"),
        number(row, "sommerfeld_from_dos_mj_mol_k2")
    );
    assert!((number(row, "pauli_susceptibility_molar_cgs") / molar - 1.0).abs() < 1e-6);
    assert!((number(row, "sommerfeld_from_dos_mj_mol_k2") / gamma - 1.0).abs() < 1e-6);
    let volume = number(&report, "volume");
    let cgs = number(row, "pauli_susceptibility_volume_cgs");
    assert!(
        (number(row, "pauli_susceptibility_volume_si") / (4.0 * std::f64::consts::PI * cgs) - 1.0)
            .abs()
            < 1e-12
    );
    let bohr_cm = dft_rs::constants::BOHR_TO_METER * 100.0;
    assert!((cgs * volume * bohr_cm.powi(3) * avogadro / molar - 1.0).abs() < 1e-6);
}

#[test]
fn fermi_level_fits_hold_for_a_gradient_corrected_functional() {
    let _report = fermi_level_fits(&[
        ("electronic_temperature", Value::Float(3000.0)),
        ("xc", Value::Str("pbe".into())),
    ]);
}

#[test]
fn fermi_level_requests_follow_their_controls() {
    let smeared = [("electronic_temperature", Value::Float(3000.0))];
    let base = run(
        "pauli-spin-susceptibility",
        &metal(),
        &controls(6, &smeared),
    )
    .unwrap();
    let reference = number(
        &base.get("rows").unwrap().as_array().unwrap()[0],
        "dos_at_fermi_level",
    );
    // The reconstruction route on the SCF mesh reproduces the SCF bands.
    let mut request = smeared.to_vec();
    request.push((
        "fermi_k_mesh",
        Value::Array(vec![Value::Int(6), Value::Int(1), Value::Int(1)]),
    ));
    request.push((
        "temperatures_k",
        Value::Array(vec![Value::Float(1000.0), Value::Float(3000.0)]),
    ));
    let rebuilt = run(
        "pauli-spin-susceptibility",
        &metal(),
        &controls(6, &request),
    )
    .unwrap();
    let rows = rebuilt.get("rows").unwrap().as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(number(&rows[0], "temperature_k"), 1000.0);
    let same = number(&rows[1], "dos_at_fermi_level");
    println!("SCF mesh {reference:.8}, reconstructed mesh {same:.8}");
    assert!((same / reference - 1.0).abs() < 1e-4);
    // A zero-temperature reference needs explicit temperatures.
    let error = run(
        "pauli-spin-susceptibility",
        &hydrogen_fluoride(),
        &controls(4, &[]),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("temperatures_k"), "{error}");
    // An insulator has no Fermi-level response at low temperature.
    let insulator = run(
        "pauli-spin-susceptibility",
        &hydrogen_fluoride(),
        &controls(4, &[("temperatures_k", Value::Float(100.0))]),
    )
    .unwrap();
    let row = &insulator.get("rows").unwrap().as_array().unwrap()[0];
    println!(
        "insulator at 100 K: L0 {:.3e}, C {:.3e}",
        number(row, "dos_at_fermi_level"),
        number(row, "heat_capacity")
    );
    assert!(number(row, "dos_at_fermi_level") < 1e-12);
    assert!(number(row, "heat_capacity") < 1e-12);
}
