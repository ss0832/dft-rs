// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Independent SCF-gradient, density, weighted-density and partition-function oracles.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::linalg::Matrix;
use dft_rs::properties::thermochemistry::{
    thermochemistry, thermochemistry_from_mermin, ThermochemistryOptions, BOLTZMANN_HARTREE_K,
};

fn system(bond: f64) -> System {
    System::molecule(vec![1, 1], vec![[0.0; 3], [0.0, 0.0, bond]])
}
fn method(xc: &str, temperature: f64) -> MolecularMethod {
    let values = Value::object()
        .with("xc", xc)
        .with("basis", "6-31g")
        .with("grid_level", 4usize)
        .with("dispersion", "none")
        .with("max_iter", 500usize)
        .with("conv_tol", 1e-14)
        .with("conv_tol_dp", 1e-12)
        .with("electronic_temperature", temperature);
    MolecularMethod::from_controls(&Controls::from_value(&values).unwrap()).unwrap()
}
fn weighted(reference: &dft_rs::scf::ScfResult) -> Matrix {
    let nao = reference.overlap.rows;
    let mut out = Matrix::zeros(nao, nao);
    for spin in 0..2 {
        let c = &reference.mo_coeff[spin];
        let mut diagonal = Matrix::zeros(c.cols, c.cols);
        for i in 0..c.cols {
            diagonal[(i, i)] = reference.mo_energy[spin][i] * reference.occupations[spin][i];
        }
        let term = c.matmul(&diagonal).matmul(&c.transpose());
        for (x, y) in out.as_mut_slice().iter_mut().zip(term.as_slice()) {
            *x += y;
        }
    }
    out
}
fn fit(label: &str, steps: &[f64], errors: &[f64]) -> bool {
    let result = convergence::error_power_fit(steps, errors);
    let accepted = (result.exponent - 2.0).abs() < 0.025 && result.r_squared > 0.9999;
    println!(
        "Mermin {label} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} C={:.9e}{}",
        result.exponent,
        result.r_squared,
        result.prefactor,
        if accepted { "" } else { " FAILED" }
    );
    accepted
}
#[test]
fn moving_metric_hessian_and_densities_have_four_step_fits_for_functional_classes() {
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0"] {
        let model = method(xc, 8000.0);
        let geometry = system(2.0);
        let setup = prepare_molecular(&geometry.numbers, &geometry.positions, &model).unwrap();
        let reference = run_molecular_reference(&setup).unwrap();
        let analytic = dft_rs::hessian::analytic::analytic_hessian_full(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            &reference.scf,
            setup.grid.as_ref(),
        )
        .unwrap();
        let mut unresolved = [true; 3];
        // Retain the original interval, then test the asymptotic regime
        // for entries with nonquadratic errors at the original steps.
        for steps in [[0.08, 0.04, 0.02, 0.01], [0.008, 0.004, 0.002, 0.001]] {
            let mut errors = [Vec::new(), Vec::new(), Vec::new()];
            for h in steps {
                let mut samples = Vec::new();
                for sign in [-1.0, 1.0] {
                    let geometry = system(2.0 + sign * h);
                    let setup =
                        prepare_molecular(&geometry.numbers, &geometry.positions, &model).unwrap();
                    let r = run_molecular_reference(&setup).unwrap();
                    let gradient = dft_rs::gradient::assemble_gradient(
                        &setup.molecule,
                        &setup.basis,
                        &setup.functional,
                        setup.grid.as_ref(),
                        &r.scf,
                        &setup.options,
                    )
                    .unwrap();
                    samples.push((gradient[1].z, r.scf.density.clone(), weighted(&r.scf)));
                }
                errors[0].push(
                    ((samples[1].0 - samples[0].0) / (2.0 * h) - analytic.hessian[(5, 5)]).abs(),
                );
                let mut pdiff = 0.0_f64;
                let mut wdiff = 0.0_f64;
                for spin in 0..2 {
                    for i in 0..setup.basis.nao_sph {
                        for j in 0..setup.basis.nao_sph {
                            pdiff = pdiff.max(
                                ((samples[1].1[spin][(i, j)] - samples[0].1[spin][(i, j)])
                                    / (2.0 * h)
                                    - analytic.density_derivatives[5][spin][(i, j)])
                                    .abs(),
                            );
                        }
                    }
                }
                for (i, (&a, &b)) in samples[1]
                    .2
                    .as_slice()
                    .iter()
                    .zip(samples[0].2.as_slice())
                    .enumerate()
                {
                    wdiff = wdiff.max(
                        ((a - b) / (2.0 * h)
                            - analytic.energy_weighted_density_derivatives[5].as_slice()[i])
                            .abs(),
                    );
                }
                errors[1].push(pdiff);
                errors[2].push(wdiff);
            }
            for (index, (label, error)) in ["Hessian", "density", "weighted density"]
                .iter()
                .zip(errors)
                .enumerate()
            {
                if unresolved[index] {
                    unresolved[index] = !fit(&format!("{xc} {label}"), &steps, &error);
                }
            }
            if !unresolved.iter().any(|&failed| failed) {
                break;
            }
        }
        assert_eq!(
            unresolved, [false; 3],
            "{xc} Hessian/density/weighted density"
        );
        let residual = analytic
            .ensemble_response_residuals
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let number = analytic
            .ensemble_number_defects
            .iter()
            .flatten()
            .copied()
            .fold(0.0_f64, f64::max);
        println!(
            "Mermin {xc} raw asymmetry={:.12e} equation={residual:.12e} number={number:.12e}",
            analytic.raw_asymmetry
        );
        assert!(residual < 1e-10 && number < 1e-10 && analytic.raw_asymmetry < 1e-8);
    }
}

#[test]
fn canonical_integer_limit_needs_no_artificial_fermi_surface_weight() {
    let geometry = system(1.4);
    let mut hessians = Vec::new();
    for temperature in [0.0, 1.0] {
        let setup = prepare_molecular(
            &geometry.numbers,
            &geometry.positions,
            &method("hf", temperature),
        )
        .unwrap();
        let r = run_molecular_reference(&setup).unwrap();
        hessians.push(
            dft_rs::hessian::analytic::analytic_hessian(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                &r.scf,
                setup.grid.as_ref(),
            )
            .unwrap(),
        );
    }
    let difference = hessians[0].max_abs_diff(&hessians[1]);
    println!("Mermin integer-limit Hessian difference={difference:.12e}");
    assert!(difference < 1e-10);
}

#[test]
fn oscillator_entropy_and_electronic_mermin_gibbs_have_independent_partition_oracles() {
    let mol = dft_rs::Molecule::from_xyz_str(
        "3\npartition oracle\nO 0 0 0\nH 0 0.7586 0.5043\nH 0 -0.7586 0.5043\n",
    )
    .unwrap();
    let frequencies = [100.0, 500.0, 1200.0];
    let at = |temperature: f64| {
        let kb = dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
        let x = 0.001 / (kb * temperature);
        let logz = (2.0 * x.cosh()).ln();
        let a = -kb * temperature * logz;
        let entropy = logz - x * x.tanh();
        thermochemistry_from_mermin(
            &mol,
            &frequencies,
            a,
            temperature,
            entropy,
            ThermochemistryOptions {
                temperature_kelvin: temperature,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let center = at(500.0);
    let steps = [40.0, 20.0, 10.0, 5.0];
    let errors: Vec<_> = steps
        .iter()
        .map(|&h| {
            ((at(500.0 + h).gibbs_free_energy - at(500.0 - h).gibbs_free_energy) / (2.0 * h)
                + center.entropy)
                .abs()
        })
        .collect();
    assert!(fit("dG/dT + total entropy", &steps, &errors));
    let opts = ThermochemistryOptions {
        temperature_kelvin: 500.0,
        ..Default::default()
    };
    let rrho = thermochemistry(&mol, &frequencies, 0.0, opts).unwrap();
    let bose_entropy: f64 = frequencies
        .iter()
        .map(|frequency| {
            let x = frequency / dft_rs::constants::HARTREE_TO_CM1 / (BOLTZMANN_HARTREE_K * 500.0);
            let occupation = 1.0 / x.exp_m1();
            BOLTZMANN_HARTREE_K
                * ((1.0 + occupation) * (1.0 + occupation).ln() - occupation * occupation.ln())
        })
        .sum();
    let error = (rrho.entropy_vibrational - bose_entropy).abs();
    println!("Bose partition entropy term error={error:.12e}");
    assert!(error < 1e-18);
    assert!((center.gibbs_free_energy - center.enthalpy + 500.0 * center.entropy).abs() < 1e-16);
    let bad = thermochemistry_from_mermin(&mol, &frequencies, 0.0, 600.0, 0.1, opts);
    assert!(
        bad.is_err(),
        "different bath temperatures must not be presented as equilibrium Gibbs"
    );
}
