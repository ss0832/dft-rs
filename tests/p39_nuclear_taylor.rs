// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent derivative gates use displaced SCFs only as validation.
//! The level-2 Richardson interval failed its unchanged fourth-order ratio
//! gate. Resolving the validation quadrature at levels 3 and 4 passes the
//! original four steps [0.02,0.01,0.005,0.0025] and every original fit/ratio
//! gate. Level 3 gives D3/D4 p=2.000325081801/1.998279256917, minimum
//! R2=0.999999800960. Other tests keep their original grids. This verifies
//! derivative consistency at the stated grids, not continuum grid convergence;
//! all failed intervals and the separate level-4 check are in response.md.

use dft_rs::basis::{data, MolBasis};
use dft_rs::gradient::assemble_gradient;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::hessian::analytic_hessian_full;
use dft_rs::math::Vec3;
use dft_rs::response::nuclear::{mixed_nuclear_response, nuclear_taylor};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::{run_scf, EriMode, JkEngine, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn molecule(r: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::new(0.1, -0.2, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.1, -0.2, r),
        },
    ])
    .with_charge(1)
}
fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}
fn grid_options() -> GridOptions {
    GridOptions {
        level: 2,
        prune: false,
        weight_cutoff: 0.0,
        ..GridOptions::default()
    }
}
fn directional_hessian(r: f64, name: &str) -> f64 {
    directional_hessian_on_grid(r, name, &grid_options())
}

fn directional_hessian_on_grid(r: f64, name: &str, grid_options: &GridOptions) -> f64 {
    let mol = molecule(r);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let grid =
        (!functional.parts.is_empty()).then(|| MolecularGrid::build(&mol, grid_options).unwrap());
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).unwrap();
    assert!(scf.converged);
    analytic_hessian_full(&mol, &mb, &opts, &functional, &scf, grid.as_ref())
        .unwrap()
        .hessian[(5, 5)]
}

#[test]
fn cubic_and_quartic_hf_match_independent_hessian_richardson() {
    let r = 1.5;
    let mol = molecule(r);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let direction = [[0.0; 3], [0.0, 0.0, 1.0]];
    let report = nuclear_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &direction,
        4,
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    eprintln!(
        "HF nuclear coefficients {:?}; stationarity {:?}; energy/Fock errors {}, {}",
        report.value.energy_coefficients,
        report.value.stationarity_defects,
        report.value.reference_energy_error,
        report.value.reference_fock_error
    );
    assert!(report.converged, "{:?}", report.warnings);
    let v = &report.value;
    assert!((v.energy_coefficients[0] - scf.energy).abs() < 1e-11);
    assert!((v.energy_components[0].nuclear_repulsion - scf.e_nuc).abs() < 1e-12);
    assert!(
        (v.energy_components[0].one_electron - scf.p_total().frobenius_dot(&scf.h_core)).abs()
            < 1e-11
    );
    let gradient = assemble_gradient(&mol, &mb, &functional, None, &scf, &opts).unwrap();
    assert!((v.energy_coefficients[1] - gradient[1].z).abs() < 1e-9);
    let h0 = directional_hessian(r, "hf");
    assert!((2.0 * v.energy_coefficients[2] - h0).abs() < 1e-9);
    let estimate = |h: f64| {
        let plus = directional_hessian(r + h, "hf");
        let minus = directional_hessian(r - h, "hf");
        [
            (plus - minus) / (2.0 * h),
            (plus + minus - 2.0 * h0) / (h * h),
        ]
    };
    let coarse = estimate(0.02);
    let fine = estimate(0.01);
    let oracle = std::array::from_fn::<_, 2, _>(|i| (4.0 * fine[i] - coarse[i]) / 3.0);
    for (i, factor) in [6.0, 24.0].into_iter().enumerate() {
        let analytic = factor * v.energy_coefficients[3 + i];
        eprintln!(
            "order {} analytic {analytic:.14e}, oracle {:.14e}",
            3 + i,
            oracle[i]
        );
        assert!((analytic - oracle[i]).abs() < [2e-7, 3e-6][i]);
    }
    assert!(v.density_coefficients[1][0]
        .as_slice()
        .iter()
        .any(|x| x.abs() > 1e-3));
    for defect in v.particle_number_coefficients.iter().flatten() {
        assert!(defect.abs() < 1e-10);
    }
}

#[test]
fn quartic_translation_and_mixed_force_constants_preserve_sum_rules() {
    let mol = molecule(1.5);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let config = ResponseSolverConfig::default();
    let stretch = vec![[0.0; 3], [0.0, 0.0, 1.0]];
    let opposite = vec![[0.0, 0.0, 1.0], [0.0; 3]];
    let uniform = vec![[0.3, -0.7, 1.2]; 2];
    let translation = nuclear_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &uniform,
        4,
        &config,
    )
    .unwrap();
    for e in &translation.value.energy_coefficients[1..] {
        assert!(e.abs() < 1e-11);
    }
    let direct = nuclear_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &stretch,
        4,
        &config,
    )
    .unwrap();
    for n in [3, 4] {
        let mut directions = vec![stretch.clone(); n];
        directions[n - 1] = opposite.clone();
        let mixed = mixed_nuclear_response(
            &mol,
            &mb,
            &functional,
            None,
            &scf,
            &opts,
            &directions,
            &config,
        )
        .unwrap();
        assert!(mixed.converged, "{:?}", mixed.warnings);
        let factor = if n == 3 { 6.0 } else { 24.0 };
        assert!(
            (mixed.value.energy_derivative + factor * direct.value.energy_coefficients[n]).abs()
                < 1e-9
        );
        directions[n - 1] = uniform.clone();
        let acoustic = mixed_nuclear_response(
            &mol,
            &mb,
            &functional,
            None,
            &scf,
            &opts,
            &directions,
            &config,
        )
        .unwrap();
        assert!(acoustic.value.energy_derivative.abs() < 1e-9);
    }
}

#[test]
fn moving_grid_semilocal_and_hybrid_families_match_existing_analytic_derivatives() {
    let mol = molecule(1.5);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(&mol, &grid_options()).unwrap();
    let opts = options();
    let config = ResponseSolverConfig::default();
    let direction = [[0.0; 3], [0.0, 0.0, 1.0]];
    for name in ["svwn5", "pbe", "r2scan", "b3lyp", "cam-b3lyp"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let scf = run_scf(&mol, &mb, &opts, &functional, Some(&grid)).unwrap();
        let report = nuclear_taylor(
            &mol,
            &mb,
            &functional,
            Some(&grid),
            &scf,
            &opts,
            &direction,
            4,
            &config,
        )
        .unwrap();
        eprintln!(
            "{name}: {:?}; stationarity {:?}; reference errors {}, {}",
            report.value.energy_coefficients,
            report.value.stationarity_defects,
            report.value.reference_energy_error,
            report.value.reference_fock_error
        );
        assert!(report.converged, "{name}: {:?}", report.warnings);
        let v = &report.value;
        assert!((v.energy_components[0].semilocal_xc - scf.e_xc).abs() < 1e-10);
        let engine = JkEngine::new(
            &mb,
            opts.eri_mode,
            dft_rs::integrals::CoulombKernel::Coulomb,
        )
        .unwrap();
        let (j, k) = engine
            .build(
                &mb,
                &scf.p_total(),
                [&scf.density[0], &scf.density[1]],
                true,
            )
            .unwrap();
        let hartree = 0.5 * scf.p_total().frobenius_dot(&j);
        let exchange = -0.5
            * functional.hybrid.cx
            * (scf.density[0].frobenius_dot(&k[0]) + scf.density[1].frobenius_dot(&k[1]));
        assert!(
            (v.energy_components[0].hartree - hartree).abs() < 1e-10,
            "{name}"
        );
        assert!(
            (v.energy_components[0].exchange - exchange).abs() < 1e-10,
            "{name}"
        );
        if functional.hybrid.clr != 0.0 {
            let engine = JkEngine::new(
                &mb,
                opts.eri_mode,
                dft_rs::integrals::CoulombKernel::Erf {
                    omega: functional.hybrid.omega,
                },
            )
            .unwrap();
            let k = engine
                .build_k_only(&mb, [&scf.density[0], &scf.density[1]])
                .unwrap();
            let exchange = -0.5
                * functional.hybrid.clr
                * (scf.density[0].frobenius_dot(&k[0]) + scf.density[1].frobenius_dot(&k[1]));
            assert!(
                (v.energy_components[0].long_range_exchange - exchange).abs() < 1e-10,
                "{name}"
            );
        }
        let gradient = assemble_gradient(&mol, &mb, &functional, Some(&grid), &scf, &opts).unwrap();
        assert!(
            (v.energy_coefficients[1] - gradient[1].z).abs() < 1e-8,
            "{name}"
        );
        let hessian =
            analytic_hessian_full(&mol, &mb, &opts, &functional, &scf, Some(&grid)).unwrap();
        assert!(
            (2.0 * v.energy_coefficients[2] - hessian.hessian[(5, 5)]).abs() < 1e-8,
            "{name}"
        );
        let estimate = |h: f64| {
            let plus = directional_hessian(1.5 + h, name);
            let minus = directional_hessian(1.5 - h, name);
            [
                (plus - minus) / (2.0 * h),
                (plus + minus - 2.0 * hessian.hessian[(5, 5)]) / (h * h),
            ]
        };
        let coarse = estimate(0.01);
        let fine = estimate(0.005);
        for (i, factor) in [6.0, 24.0].into_iter().enumerate() {
            let oracle = (4.0 * fine[i] - coarse[i]) / 3.0;
            let analytic = factor * v.energy_coefficients[3 + i];
            eprintln!(
                "{name} order {} analytic {analytic:.14e}, Richardson {oracle:.14e}",
                3 + i
            );
            assert!(
                (analytic - oracle).abs() < [2e-6, 2e-5][i],
                "{name} order {} error {}",
                3 + i,
                analytic - oracle
            );
        }
        for n in 0..4 {
            assert!(v.metric_defects[n] < 1e-10);
            assert!(v.projector_defects[n] < 1e-10);
        }
    }
}

#[test]
fn noncommuting_metric_transport_and_unrestricted_quartic_response() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.5, 0.1, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.2, 1.7, 0.3),
        },
    ])
    .with_charge(1);
    let direction = [[0.0; 3], [0.4, -0.2, 0.3], [-0.1, 0.3, 0.2]];
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let report = nuclear_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &direction,
        4,
        &Default::default(),
    )
    .unwrap();
    assert!(
        report.converged,
        "{:?}; stationarity {:?}",
        report.warnings, report.value.stationarity_defects
    );
    let s = &report.value.overlap_coefficients;
    let comm = s[1].matmul_seq(&s[2]).max_abs_diff(&s[2].matmul_seq(&s[1]));
    assert!(
        comm > 1e-4,
        "fixture must require noncommuting matrix Taylor products"
    );
    let hessian_at = |t: f64| {
        let mut displaced = mol.clone();
        for (atom, d) in displaced.atoms.iter_mut().zip(direction) {
            atom.position += Vec3::new(d[0] * t, d[1] * t, d[2] * t);
        }
        let mb = MolBasis::build(&displaced, &data::load("sto-3g").unwrap()).unwrap();
        let scf = run_scf(&displaced, &mb, &opts, &functional, None).unwrap();
        let h = analytic_hessian_full(&displaced, &mb, &opts, &functional, &scf, None)
            .unwrap()
            .hessian;
        let d: Vec<_> = direction.iter().flatten().copied().collect();
        let mut value = 0.0;
        for i in 0..d.len() {
            for j in 0..d.len() {
                value += d[i] * h[(i, j)] * d[j];
            }
        }
        value
    };
    let h0 = hessian_at(0.0);
    let coarse = [hessian_at(0.02), hessian_at(-0.02)];
    let fine = [hessian_at(0.01), hessian_at(-0.01)];
    let third = (4.0 * (fine[0] - fine[1]) / 0.02 - (coarse[0] - coarse[1]) / 0.04) / 3.0;
    let fourth = (4.0 * (fine[0] + fine[1] - 2.0 * h0) / 0.0001
        - (coarse[0] + coarse[1] - 2.0 * h0) / 0.0004)
        / 3.0;
    assert!((6.0 * report.value.energy_coefficients[3] - third).abs() < 2e-7);
    assert!((24.0 * report.value.energy_coefficients[4] - fourth).abs() < 3e-6);
    let mol = molecule(1.5).with_charge(0).with_multiplicity(2);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    assert!(scf.unrestricted);
    let report = nuclear_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &[[0.0; 3], [0.0, 0.0, 1.0]],
        4,
        &Default::default(),
    )
    .unwrap();
    assert!(report.converged, "{:?}", report.warnings);
    let h = analytic_hessian_full(&mol, &mb, &opts, &functional, &scf, None)
        .unwrap()
        .hessian;
    assert!((2.0 * report.value.energy_coefficients[2] - h[(5, 5)]).abs() < 1e-9);
    for n in report.value.particle_number_coefficients.iter().flatten() {
        assert!(n.abs() < 1e-10);
    }
}

#[test]
fn central_hessian_differences_have_second_order_error() {
    #[path = "support/convergence.rs"]
    mod convergence;
    let mol = molecule(1.5);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    // The level-2 moving quadrature does not resolve this Richardson
    // interval: finer finite-difference steps enter cancellation first.
    // Keep all fit/ratio thresholds and resolve the quadrature instead.
    // The retained failed level-2 measurements are recorded in response.md.
    let resolved_grid = GridOptions {
        level: 3,
        ..grid_options()
    };
    let grid = MolecularGrid::build(&mol, &resolved_grid).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &functional, Some(&grid)).unwrap();
    let report = nuclear_taylor(
        &mol,
        &mb,
        &functional,
        Some(&grid),
        &scf,
        &opts,
        &[[0.0; 3], [0.0, 0.0, 1.0]],
        4,
        &Default::default(),
    )
    .unwrap();
    let analytic = [
        6.0 * report.value.energy_coefficients[3],
        24.0 * report.value.energy_coefficients[4],
    ];
    let h0 = directional_hessian_on_grid(1.5, "svwn5", &resolved_grid);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let mut estimates = Vec::new();
    for h in steps {
        let plus = directional_hessian_on_grid(1.5 + h, "svwn5", &resolved_grid);
        let minus = directional_hessian_on_grid(1.5 - h, "svwn5", &resolved_grid);
        let estimate = [
            (plus - minus) / (2.0 * h),
            (plus + minus - 2.0 * h0) / (h * h),
        ];
        eprintln!(
            "h={h:.6}: raw D3={:.14e} error={:.14e}; raw D4={:.14e} error={:.14e}",
            estimate[0],
            (estimate[0] - analytic[0]).abs(),
            estimate[1],
            (estimate[1] - analytic[1]).abs()
        );
        estimates.push(estimate);
    }
    for axis in 0..2 {
        let errors: Vec<_> = estimates
            .iter()
            .map(|v| (v[axis] - analytic[axis]).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        eprintln!(
            "D{} four-step error fit: p={:.12}, C={:.12e}, R_squared={:.12}, log_rmse={:.12e}",
            axis + 3,
            fit.exponent,
            fit.prefactor,
            fit.r_squared,
            fit.log_rmse
        );
        assert!((fit.exponent - 2.0).abs() < 0.02);
        assert!(fit.r_squared > 0.99999);
        for k in 0..steps.len() - 1 {
            let coarse = (estimates[k][axis] - analytic[axis]).abs();
            let fine = (estimates[k + 1][axis] - analytic[axis]).abs();
            let ratio = coarse / fine;
            eprintln!(
                "D{} raw error ratio h/h2 = {ratio:.10}; effective order {:.8}",
                axis + 3,
                ratio.log2()
            );
            assert!(
                (ratio - 4.0).abs() < 0.15,
                "central difference must retain second-order convergence before roundoff"
            );
        }
        let richardson0 = (4.0 * estimates[1][axis] - estimates[0][axis]) / 3.0;
        let richardson1 = (4.0 * estimates[2][axis] - estimates[1][axis]) / 3.0;
        let ratio = (richardson0 - analytic[axis]).abs() / (richardson1 - analytic[axis]).abs();
        eprintln!(
            "D{} Richardson error ratio h/h2 = {ratio:.10}; effective order {:.8}",
            axis + 3,
            ratio.log2()
        );
        assert!(
            (ratio - 16.0).abs() < 1.0,
            "Richardson must cancel the leading h^2 error"
        );
    }
}
