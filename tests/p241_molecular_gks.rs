// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full complex GKS provider: all Hermitian density directions, independent
//! four-step variational fits, collinear limits and a complex spin source.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::MolecularGrid;
use dft_rs::math::Vec3;
use dft_rs::periodic::spinor_scf::SpinorEnergyBackend;
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};

fn molecule() -> System {
    System {
        numbers: vec![1; 4],
        positions: vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
        cell: None,
        pbc: [false; 3],
    }
}
fn controls(xc: &str) -> Controls {
    Controls::from_value(
        &Value::object()
            .with("xc", xc)
            .with("basis", "sto-3g")
            .with("grid_level", 2)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-12)
            .with("conv_tol_grad", 1e-9)
            .with("conv_tol_dp", 1e-11)
            .with("max_iter", 300)
            .with("guess", "core"),
    )
    .unwrap()
}

#[test]
fn every_hermitian_density_component_has_a_variational_four_step_fit() {
    let system = molecule();
    let method = MolecularMethod::from_controls(&controls("hf")).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    let grid = MolecularGrid {
        points: vec![
            Vec3::new(0.21, 0.32, 0.43),
            Vec3::new(0.91, -0.25, 0.62),
            Vec3::new(-0.31, 0.74, 1.26),
            Vec3::new(1.11, 0.83, -0.23),
            Vec3::new(0.57, -0.18, 1.49),
        ],
        weights: vec![0.43, 0.67, 0.39, 0.51, 0.62],
        raw_weights: vec![0.43, 0.67, 0.39, 0.51, 0.62],
        atom_of: vec![0, 1, 2, 1, 3],
    };
    let functional =
        dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("tpss".into())).unwrap();
    let n = 2 * setup.basis.nao_sph;
    let backend = MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &functional,
        Some(&grid),
        &seed.scf,
        &setup.options,
        BlochMatrix::zeros(n, n),
        [0.3, 0.4, 0.5],
        MolecularSpinorFunctionalModel::SpinCovariant,
        1e-12,
    )
    .unwrap();
    let mut p = BlochMatrix::zeros(n, n);
    for i in 0..n {
        p.re[(i, i)] = 0.31 + 0.017 * i as f64;
        for j in 0..i {
            let re = 0.013 * (0.31 + 0.17 * (i + 2 * j) as f64).sin();
            let im = 0.011 * (0.29 + 0.13 * (2 * i + j) as f64).cos();
            p.re[(i, j)] = re;
            p.re[(j, i)] = re;
            p.im[(i, j)] = im;
            p.im[(j, i)] = -im;
        }
    }
    let f = backend.evaluate(&[p.clone()]).unwrap().fock.remove(0);
    let mut directions = Vec::new();
    let mut names = Vec::new();
    for i in 0..n {
        let mut d = BlochMatrix::zeros(n, n);
        d.re[(i, i)] = 1.0;
        directions.push(d);
        names.push(format!("GKS density diagonal {i}"));
        for j in 0..i {
            for imaginary in [false, true] {
                let mut d = BlochMatrix::zeros(n, n);
                if imaginary {
                    d.im[(i, j)] = 1.0;
                    d.im[(j, i)] = -1.0;
                } else {
                    d.re[(i, j)] = 1.0;
                    d.re[(j, i)] = 1.0;
                }
                directions.push(d);
                names.push(format!("GKS density ({i},{j}) imaginary={imaginary}"));
            }
        }
    }
    assert_eq!(directions.len(), n * n);
    let errors = |steps: &[f64; 4], stencil: stencils::Stencil| {
        directions
            .iter()
            .map(|d| {
                let analytic = f.trace_product(d);
                let energy = |h: f64| {
                    let mut q = p.clone();
                    q.add_scaled(d, h);
                    backend.evaluate(&[q]).unwrap().energy
                };
                steps.map(|h| {
                    let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                        (energy(2.0 * h), energy(-2.0 * h))
                    } else {
                        (0.0, 0.0)
                    };
                    (stencil.first(energy(h), energy(-h), p2, m2, h) - analytic).abs()
                })
            })
            .collect::<Vec<_>>()
    };
    let mut failures = Vec::new();
    let initial_steps = [0.04, 0.03, 0.02, 0.015];
    let initial_errors = errors(&initial_steps, stencils::Stencil::Second);
    for (name, e) in names.iter().zip(&initial_errors) {
        if e.iter().all(|v| *v > 0.0 && v.is_finite()) {
            let fit = convergence::error_power_fit(&initial_steps, e);
            println!("{name} log fit RMSE={:.3e}", fit.log_rmse);
        }
    }
    components::fit_components(
        &names,
        &[initial_steps],
        |_| initial_errors.clone(),
        &mut failures,
    );
    if !failures.is_empty() {
        println!("Initial failed components retained: {failures:?}");
        failures.clear();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
            ],
            &errors,
            &mut failures,
        );
    }
    components::assert_no_failures("complete complex GKS energy/Fock derivative", &failures);
}

#[test]
fn gks_population_and_orbitals_recover_collinear_limits_for_functional_classes() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0"] {
        let mut value = controls(xc).to_value();
        value.set("reference", "rks");
        let rks = run(
            "charges",
            &molecule(),
            &Controls::from_value(&value).unwrap(),
        )
        .unwrap();
        value.set("reference", "gks");
        value.set("spin_axis", vec![0.3, 0.4, 0.5]);
        let gks = run(
            "charges",
            &molecule(),
            &Controls::from_value(&value).unwrap(),
        )
        .unwrap();
        let mut worst = 0.0_f64;
        for key in ["mulliken", "lowdin"] {
            for (a, b) in rks
                .get(key)
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .zip(gks.get(key).unwrap().as_array().unwrap())
            {
                worst = worst.max((a.as_f64().unwrap() - b.as_f64().unwrap()).abs());
            }
        }
        println!("{xc} GKS/RKS population defect={worst:.3e}");
        assert!(worst < 1e-8);
        assert!(
            (rks.get("energy").unwrap().as_f64().unwrap()
                - gks.get("energy").unwrap().as_f64().unwrap())
            .abs()
                < 1e-9
        );
    }
}

#[test]
fn a_complex_spin_operator_retains_every_spin_block() {
    let mut c = controls("hf").to_value();
    c.set("reference", "ghf");
    c.set("coefficients", true);
    c.set("density", true);
    let mut re = vec![vec![0.0; 8]; 8];
    let mut im = re.clone();
    for i in 0..4 {
        re[i][i] = -0.013 * (i + 1) as f64;
        re[4 + i][4 + i] = -re[i][i];
        re[i][4 + i] = 0.007 * (i + 1) as f64;
        re[4 + i][i] = re[i][4 + i];
        im[i][4 + i] = 0.009 * (4 - i) as f64;
        im[4 + i][i] = -im[i][4 + i];
    }
    c.set("gks_operator_real", re);
    c.set("gks_operator_imaginary", im);
    let report = run("orbitals", &molecule(), &Controls::from_value(&c).unwrap()).unwrap();
    let channel = &report.get("channels").unwrap().as_array().unwrap()[0];
    let (_, density) = channel
        .get("density_imaginary")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert!(density.iter().any(|v| v.abs() > 1e-3));
    assert!(
        report
            .get("particle_number_error")
            .unwrap()
            .as_f64()
            .unwrap()
            < 1e-10
    );
    assert!(report.get("orbital_residual").unwrap().as_f64().unwrap() < 1e-9);
    assert!(report.get("metric_defect").unwrap().as_f64().unwrap() < 1e-10);
}
