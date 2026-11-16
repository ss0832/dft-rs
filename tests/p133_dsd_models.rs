// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Original 2013 DSD-D3(BJ) recipes: independent electronic/dispersion
//! energy terms and four-step analytic dispersion derivative fits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::dual::DualN;
use dft_rs::xc::double_hybrid::DoubleHybridFunctional;

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn dsd_models_match_independent_parent_spin_and_dispersion_terms() {
    let data = include_str!("oracle_data/dsd_energy.csv");
    let header: Vec<_> = data.lines().next().unwrap().split(',').collect();
    for line in data.lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let expected = |key: &str| {
            fields[header.iter().position(|x| *x == key).unwrap()]
                .parse::<f64>()
                .unwrap()
        };
        let model = fields[0];
        let name = fields[1];
        let system = if name == "hydrogen" {
            System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
        } else {
            System::molecule(
                vec![8, 1, 1],
                vec![
                    [0.0, 0.0, 0.2217],
                    [0.0, 1.4309, -0.8867],
                    [0.0, -1.4309, -0.8867],
                ],
            )
        };
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", model)
                .with("basis", "sto-3g")
                .with("grid_level", 5usize)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-10)
                .with("conv_tol_dp", 1e-12),
        )
        .unwrap();
        let report = run("energy", &system, &controls).unwrap();
        let Value::Array(parts) = report.get("semilocal_components").unwrap() else {
            panic!("missing semilocal terms")
        };
        for (key, actual) in [
            ("parent", number(&report, "orbital_reference_energy")),
            ("energy", number(&report, "energy")),
            ("dispersion", number(&report, "model_dispersion_energy")),
            ("one", number(&report, "one_electron_energy")),
            ("hartree", number(&report, "hartree_energy")),
            ("exchange", number(&report, "exact_exchange_energy")),
            ("semilocal_x", number(&parts[0], "energy")),
            ("semilocal_c", number(&parts[1], "energy")),
            (
                "same_spin",
                number(&report, "pt2_same_spin_alpha") + number(&report, "pt2_same_spin_beta"),
            ),
            ("opposite_spin", number(&report, "pt2_opposite_spin")),
        ] {
            let defect = (actual - expected(key)).abs();
            println!(
                "{model} {name} {key} actual={actual:.15e} oracle={:.15e} defect={defect:.12e}",
                expected(key)
            );
            assert!(defect < 1e-6, "{model} {name} {key}");
            if key == "dispersion" {
                assert!(defect < 1e-11);
            }
        }
        assert!(number(&report, "pt2_amplitude_residual") < 1e-12);
        assert!(number(&report, "reference_energy_sum_residual") < 1e-9);
    }
}

#[test]
fn each_dsd_dispersion_gradient_passes_a_four_step_independent_error_fit() {
    let data = include_str!("oracle_data/dsd_dispersion.csv");
    for model in ["dsd-pbep86", "dsd-blyp"] {
        let recipe = DoubleHybridFunctional::named(model).unwrap();
        let params = recipe.dispersion.unwrap();
        let positions = [
            [DualN::<f64, 1>::constant(0.0); 3],
            [
                DualN::constant(0.0),
                DualN::constant(0.0),
                DualN::variable(6.0, 0),
            ],
        ];
        let derivative =
            dft_rs::dispersion::d3_energy_g(&[1, 1], &positions, None, Some(&params)).im[0];
        let mut steps = Vec::new();
        let mut errors = Vec::new();
        let mut oracle_defect = 0.0_f64;
        for line in data
            .lines()
            .skip(1)
            .filter(|line| line.starts_with(&format!("{model},")))
        {
            let fields: Vec<_> = line.split(',').collect();
            let values: Vec<f64> = fields[1..].iter().map(|x| x.parse().unwrap()).collect();
            let h = values[0];
            steps.push(h);
            errors.push(((values[1] - values[2]) / (2.0 * h) - derivative).abs());
            oracle_defect = oracle_defect.max((values[3] - derivative).abs());
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        println!("{model} dispersion derivative={derivative:.12e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} independent_gradient={oracle_defect:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.15 && fit.r_squared > 0.999);
        assert!(oracle_defect < 1e-12);
    }
}
