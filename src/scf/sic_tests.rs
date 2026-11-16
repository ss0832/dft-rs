// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use super::{evaluate, JkEngine, Matrix, MolecularDensityModel};
use crate::acceptance::{components, stencils};
use crate::driver::method::{prepare_molecular, MolecularMethod};
use crate::driver::{run, Controls, System, Value};
use crate::scf::run_scf;

#[test]
fn pz_sic_orbital_gradient_passes_every_rotation() {
    let setup = prepare_molecular(
        &[1; 4],
        &[
            [0.13, -0.27, 0.19],
            [1.77, 0.36, -0.08],
            [2.44, 2.35, 1.2],
            [3.97, 2.88, 1.67],
        ],
        &MolecularMethod {
            xc: Some("pbe".into()),
            unrestricted: true,
            grid_level: 0,
            grid_weight_cutoff: 0.0,
            ..MolecularMethod::default()
        },
    )
    .unwrap();
    let reference = run_scf(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )
    .unwrap();
    let jk = JkEngine::from_options(&setup.molecule, &setup.basis, &setup.options).unwrap();
    let model = MolecularDensityModel {
        mol: &setup.molecule,
        mb: &setup.basis,
        opts: &setup.options,
        functional: &setup.functional,
        grid: setup.grid.as_ref(),
        h_core: &reference.h_core,
        extension: None,
        jk: &jk,
        jk_lr: None,
        pcm_solver: None,
        nlc_grid: None,
        restricted: false,
    };
    let counts = [reference.n_alpha, reference.n_beta];
    let central = evaluate(&model, &reference.mo_coeff, counts, reference.e_nuc).unwrap();
    let probes: Vec<_> = counts
        .iter()
        .enumerate()
        .flat_map(|(s, &count)| {
            let size = reference.mo_coeff[s].cols;
            (0..count).flat_map(move |i| (i + 1..size).map(move |j| (s, i, j)))
        })
        .collect();
    let labels: Vec<_> = probes
        .iter()
        .map(|(s, i, j)| format!("PZ-SIC orbital gradient ({s},{i},{j})"))
        .collect();
    let grid = setup.grid.as_ref().unwrap();
    let zero = Matrix::zeros(model.mb.nao_sph, model.mb.nao_sph);
    // Independent energy oracle: ordinary parent energy and explicit
    // one-orbital density Coulomb and spin-polarized XC evaluations.
    let energy = |orbitals: &[Matrix; 2]| {
        let density = std::array::from_fn(|s| orbitals[s].leading_columns_gram(counts[s], 1.0));
        let mut energy = model.evaluate(&density).unwrap().electronic_energy + reference.e_nuc;
        for (c, count) in orbitals.iter().zip(counts) {
            for i in 0..count {
                let mut density = Matrix::zeros(c.rows, c.rows);
                for a in 0..c.rows {
                    for b in 0..c.rows {
                        density[(a, b)] = c[(a, i)] * c[(b, i)];
                    }
                }
                let (j, _) = jk
                    .build(model.mb, &density, [&density, &zero], false)
                    .unwrap();
                let xc =
                    crate::ks::xc_contribution(model.functional, model.mb, grid, [&density, &zero])
                        .unwrap();
                energy -= xc.exc + 0.5 * j.frobenius_dot(&density);
            }
        }
        energy
    };
    assert!((central.energy - energy(&reference.mo_coeff)).abs() < 2e-13);
    let mut failed = Vec::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            probes
                .iter()
                .map(|&(s, i, j)| {
                    let physical = central.gradient[s][(j, i)];
                    let rotate = |angle: f64| {
                        let mut orbitals = reference.mo_coeff.clone();
                        let (sin, cos) = angle.sin_cos();
                        for a in 0..orbitals[s].rows {
                            let b = reference.mo_coeff[s][(a, i)];
                            let c = reference.mo_coeff[s][(a, j)];
                            orbitals[s][(a, i)] = cos * b + sin * c;
                            orbitals[s][(a, j)] = cos * c - sin * b;
                        }
                        energy(&orbitals)
                    };
                    steps.map(|h| {
                        let (plus2, minus2) = match stencil {
                            stencils::Stencil::Second => (0.0, 0.0),
                            stencils::Stencil::Fourth => (rotate(2.0 * h), rotate(-2.0 * h)),
                        };
                        (stencil.first(rotate(h), rotate(-h), plus2, minus2, h) - physical).abs()
                    })
                })
                .collect()
        },
        &mut failed,
    );
    components::assert_no_failures("PZ-SIC orbital gradient", &failed);
}

#[test]
fn pz_sic_one_electron_exact_limit_for_required_functionals() {
    let hydrogen_ion = System::molecule(vec![1, 1], vec![[0.13, -0.27, 0.19], [1.77, 0.36, -0.08]]);
    let controls = Value::object()
        .with("charge", 1)
        .with("multiplicity", 2usize)
        .with("basis", "sto-3g")
        .with("grid_level", 0usize)
        .with("vv10_grid_level", 0usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol_grad", 1e-9)
        .with("reference", "uks");
    let report = |feature: &str, xc: &str| {
        // Remove the geometry-only empirical tail for the exact one-electron
        // check, while retaining the density-dependent VV10 contribution.
        let dispersion = if xc == "wb97x-v" { "auto" } else { "none" };
        run(
            feature,
            &hydrogen_ion,
            &Controls::from_value(
                &controls
                    .clone()
                    .with("xc", xc)
                    .with("dispersion", dispersion),
            )
            .unwrap(),
        )
        .unwrap()
    };
    let value = |report: &Value, key: &str| report.get(key).unwrap().as_f64().unwrap();
    let expected = value(&report("energy", "hf"), "energy");
    for xc in ["wb97x-d", "wb97x-v"] {
        let result = report("pz-sic", xc);
        let actual = value(&result, "energy");
        let correction = value(&result, "self_interaction_correction");
        let residual = value(&result, "scf_orbital_residual");
        let orbital = &result
            .get("orbital_self_energies")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .as_array()
            .unwrap()[0];
        let nonlocal = value(orbital, "nonlocal_correlation");
        println!("PZ-SIC H2+ {xc}: E={actual:.14e}, correction={correction:.12e}, residual={residual:.6e}");
        assert!((actual - expected).abs() < 2e-11);
        assert!(residual <= 1e-9);
        assert_eq!(nonlocal.abs() > 1e-6, xc == "wb97x-v");
    }
}
