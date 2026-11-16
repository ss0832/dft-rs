// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Unequal rigid spin bands at fixed total number, with independent field
//! and temperature stencils and the original equal-channel limit.
#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::periodic::fermi_thermodynamics::{
    collinear_fermi_thermodynamics, fermi_thermodynamics,
};
use dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K;

// Independent occupation/root solve; it does not call production moment,
// susceptibility, occupation or response routines.
fn state(bands: [&[f64]; 2], weight: f64, number: f64, kt: f64, field: f64, g: f64) -> [f64; 3] {
    let levels: [Vec<_>; 2] = std::array::from_fn(|s| {
        bands[s]
            .iter()
            .map(|e| {
                e + if s == 0 {
                    g * field / 4.0
                } else {
                    -g * field / 4.0
                }
            })
            .collect()
    });
    let occupation = |e: f64, mu: f64| 1.0 / (1.0 + ((e - mu) / kt).exp());
    let mut lo = levels
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min)
        - 50.0 * kt;
    let mut hi = levels
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        + 50.0 * kt;
    for _ in 0..200 {
        let mu = 0.5 * (lo + hi);
        let count: f64 = weight
            * levels
                .iter()
                .flatten()
                .map(|&e| occupation(e, mu))
                .sum::<f64>();
        if count < number {
            lo = mu;
        } else {
            hi = mu;
        }
    }
    let mu = 0.5 * (lo + hi);
    let numbers: [f64; 2] =
        std::array::from_fn(|s| weight * levels[s].iter().map(|&e| occupation(e, mu)).sum::<f64>());
    assert!((numbers.iter().sum::<f64>() - number).abs() < 1e-12);
    let energy = weight
        * levels
            .iter()
            .flatten()
            .map(|&e| e * occupation(e, mu))
            .sum::<f64>();
    [mu, -g * (numbers[0] - numbers[1]) / 4.0, energy]
}

fn fit(label: &str, exact: f64, steps: &[f64], values: &[f64]) {
    let errors: Vec<_> = values.iter().map(|value| (value - exact).abs()).collect();
    let fit = convergence::error_power_fit(steps, &errors);
    let last = values.len() - 1;
    let richardson = ((4.0 * values[last] - values[last - 1]) / 3.0 - exact).abs();
    println!(
        "{label}: steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e}",
        fit.exponent, fit.r_squared
    );
    assert!(
        (fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999,
        "{label}"
    );
    assert!(richardson < 1e-7, "{label}: {richardson:e}");
}

fn accept(
    bands: [&[f64]; 2],
    weight: f64,
    number: f64,
    kt: f64,
    g: f64,
    field_steps: &[f64],
    thermal_steps: &[f64],
    label: &str,
) {
    let result = collinear_fermi_thermodynamics(bands, weight, number, kt, g).unwrap();
    let values = state(bands, weight, number, kt, 0.0, g);
    assert!((result.thermal.chemical_potential - values[0]).abs() < 1e-12);
    assert!((result.magnetization - values[1]).abs() < 1e-12);
    assert!((result.thermal.band_energy - values[2]).abs() < 1e-12);
    assert!(result.charge_constraint_residual < 1e-12);
    let magnetic: Vec<_> = field_steps
        .iter()
        .map(|&h| {
            let a = state(bands, weight, number, kt, h, g);
            let b = state(bands, weight, number, kt, -h, g);
            [(a[1] - b[1]) / (2.0 * h), (a[0] - b[0]) / (2.0 * h)]
        })
        .collect();
    fit(
        &format!("{label} susceptibility"),
        result.susceptibility,
        field_steps,
        &magnetic.iter().map(|v| v[0]).collect::<Vec<_>>(),
    );
    fit(
        &format!("{label} common mu/B"),
        result.chemical_potential_response.unwrap(),
        field_steps,
        &magnetic.iter().map(|v| v[1]).collect::<Vec<_>>(),
    );
    let thermal: Vec<_> = thermal_steps
        .iter()
        .map(|&h| {
            let a = state(bands, weight, number, kt + h, 0.0, g);
            let b = state(bands, weight, number, kt - h, 0.0, g);
            [(a[2] - b[2]) / (2.0 * h), (a[0] - b[0]) / (2.0 * h)]
        })
        .collect();
    fit(
        &format!("{label} heat capacity"),
        result.thermal.heat_capacity,
        thermal_steps,
        &thermal.iter().map(|v| v[0]).collect::<Vec<_>>(),
    );
    fit(
        &format!("{label} common mu/kT"),
        result.thermal.dmu_dkt,
        thermal_steps,
        &thermal.iter().map(|v| v[1]).collect::<Vec<_>>(),
    );
}

#[test]
fn unequal_rigid_band_field_and_temperature_derivatives_have_four_step_fits() {
    accept(
        [&[-0.5, 0.1, 0.7], &[-0.3, 0.35, 0.9]],
        0.5,
        1.3,
        0.2,
        2.0,
        &[0.008, 0.004, 0.002, 0.001],
        &[0.008, 0.004, 0.002, 0.001],
        "asymmetric levels",
    );
}

#[test]
fn restricted_inert_and_g_scaling_limits_preserve_physical_denominators() {
    let band = [-0.5, 0.1, 0.7];
    let restricted = collinear_fermi_thermodynamics([&band, &band], 0.5, 1.3, 0.2, 2.0).unwrap();
    let original = fermi_thermodynamics(&band, 1.0, 1.3, 0.2).unwrap();
    assert!((restricted.susceptibility - 0.25 * original.dos).abs() < 1e-14);
    assert!((restricted.thermal.heat_capacity - original.heat_capacity).abs() < 1e-14);
    assert_eq!(restricted.chemical_potential_response, Some(0.0));
    let other = [-0.3, 0.35, 0.9];
    let a = collinear_fermi_thermodynamics([&band, &other], 0.5, 1.3, 0.2, 1.5).unwrap();
    let b = collinear_fermi_thermodynamics([&band, &other], 0.5, 1.3, 0.2, 3.0).unwrap();
    assert!((b.susceptibility - 4.0 * a.susceptibility).abs() < 1e-14);
    assert!(
        (b.chemical_potential_response.unwrap() - 2.0 * a.chemical_potential_response.unwrap())
            .abs()
            < 1e-14
    );
    let inert =
        collinear_fermi_thermodynamics([&[-1.0, 1.0], &[-1.0, 1.0]], 1.0, 2.0, 1e-6, 2.0).unwrap();
    assert_eq!(inert.susceptibility, 0.0);
    assert_eq!(inert.chemical_potential_response, None);
    assert_eq!(inert.charge_constraint_residual, 0.0);
    let empty = collinear_fermi_thermodynamics([&[], &band], 1.0, 1.3, 0.2, 2.0).unwrap();
    assert_eq!(empty.susceptibility, 0.0);
    assert_eq!(empty.chemical_potential_response, Some(-0.5));
    assert_eq!(empty.delta_spin_numbers, [0.0; 2]);
    for (weight, kt, g) in [
        (f64::INFINITY, 0.2, 2.0),
        (1.0, f64::INFINITY, 2.0),
        (1.0, 0.2, f64::NAN),
    ] {
        assert!(collinear_fermi_thermodynamics([&band, &other], weight, 1.3, kt, g).is_err());
    }
}

#[test]
fn polarized_physical_parent_bands_use_the_shared_rigid_response_driver() {
    let system = System {
        numbers: vec![1, 1, 1],
        positions: vec![[0.0, 0.0, -0.9], [0.0, 0.0, 0.9], [1.7, 0.3, 0.2]],
        cell: Some([[9.5, 0.0, 0.0], [0.0, 9.5, 0.0], [0.0, 0.0, 9.5]]),
        pbc: [true; 3],
    };
    for xc in ["svwn5", "pbe", "tpss", "pbe0"] {
        let base = Value::object()
            .with("xc", xc)
            .with("basis", "sto-3g")
            .with("multiplicity", 2i64)
            .with("unrestricted", true)
            .with("grid_level", 1usize)
            .with("grid_shell", 1usize)
            .with("overlap_cutoff", 7.0)
            .with("k_mesh", [1usize; 3])
            .with("electronic_temperature", 150000.0)
            .with("conv_tol", 1e-12)
            .with("guess", "core")
            .with("max_iter", 250usize);
        let structure = run(
            "electronic-structure",
            &system,
            &Controls::from_value(&base).unwrap(),
        )
        .unwrap();
        let channels = structure.get("channels").unwrap().as_array().unwrap();
        assert_eq!(channels.len(), 2);
        let bands: [Vec<_>; 2] = std::array::from_fn(|s| {
            channels[s]
                .get("band_energies")
                .unwrap()
                .numeric_array()
                .unwrap()
                .1
        });
        let kt = 150000.0 * BOLTZMANN_HARTREE_PER_K;
        accept(
            [&bands[0], &bands[1]],
            1.0,
            3.0,
            kt,
            2.0,
            &[0.08, 0.04, 0.02, 0.01],
            &[0.04, 0.02, 0.01, 0.005],
            xc,
        );
        let mut settings = base.clone();
        settings.set("temperatures_k", [150000.0]);
        settings.set("spin_g_factor", 2.0);
        let report = run(
            "pauli-spin-susceptibility",
            &system,
            &Controls::from_value(&settings).unwrap(),
        )
        .unwrap();
        let row = &report.get("rows").unwrap().as_array().unwrap()[0];
        let expected =
            collinear_fermi_thermodynamics([&bands[0], &bands[1]], 1.0, 3.0, kt, 2.0).unwrap();
        let number = |key| row.get(key).unwrap().as_f64().unwrap();
        assert!((number("pauli_susceptibility_per_cell") - expected.susceptibility).abs() < 1e-11);
        assert!(
            (number("chemical_potential_response") - expected.chemical_potential_response.unwrap())
                .abs()
                < 1e-11
        );
        assert!(number("raw_charge_constraint_residual") < 1e-12);
        assert!(number("thermal_number_residual") < 1e-12);
        println!(
            "{xc} common-mu rigid driver: chi={:.12e} mu'={:.12e} charge={:.12e} number={:.12e}",
            number("pauli_susceptibility_per_cell"),
            number("chemical_potential_response"),
            number("raw_charge_constraint_residual"),
            number("thermal_number_residual")
        );
    }
}
