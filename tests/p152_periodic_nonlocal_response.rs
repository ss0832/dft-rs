// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! A variational periodic nonlocal reference and independently reconverged
//! moving-cell energy/gradient differences. Finite differences are oracles.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_cartesian_directions.rs"]
mod symmetric_cartesian_directions;
use dft_rs::basis::data;
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::nonlocal::PeriodicVv10Options;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::nuclear::lattice_path_energy;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

#[test]
fn omega_b97x_v_finite_k_forces_and_stress_fit_all_components() {
    use dft_rs::periodic::{forces::analytic_forces, stress::analytic_stress};
    let positions = [[0.13, -0.21, 0.07], [2.21, -1.28, 1.61]];
    let rows = [[9.4, 0.71, 0.33], [0.45, 10.8, 0.69], [0.24, -0.55, 11.2]];
    let basis = dft_rs::driver::method::build_basis(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/periodic_scalar_ecp.g94"
        ),
        None,
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined("wb97x-v".into())).unwrap();
    let options = PeriodicScfOptions {
        k_mesh: [3, 1, 1],
        exchange_cutoff: Some(5.5),
        xc_ao_cutoff: Some(9.0),
        vv10: Some(PeriodicVv10Options {
            parameters: dft_rs::nlc::Vv10Params::WB97XV,
            cutoff: 9.0,
        }),
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        max_iter: 400,
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(10),
            angular_override: Some(7),
            prune: false,
            weight_cutoff: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let vector = |[x, y, z]: [f64; 3]| Vec3::new(x, y, z);
    let build = |component: usize, h: f64| {
        let (mut p, mut a) = (positions, rows);
        if component < 6 {
            p[component / 3][component % 3] += h;
        } else {
            let (i, j) = ((component - 6) / 3, (component - 6) % 3);
            for r in p.iter_mut().chain(&mut a) {
                r[i] += h * r[j];
            }
        }
        let molecule = Molecule::new(
            [11, 1]
                .into_iter()
                .zip(p)
                .map(|(z, r)| Atom {
                    z,
                    position: vector(r),
                })
                .collect(),
        );
        Cell::new(
            molecule,
            Lattice::new(vector(a[0]), vector(a[1]), vector(a[2])),
            &basis,
            5.5,
        )
        .unwrap()
    };
    let solve = |cell: &Cell| {
        let scf = run_periodic_scf(cell, &functional, &options).unwrap();
        assert!(scf.converged && scf.e_nlc.abs() > 1e-8);
        scf
    };
    let cell = build(0, 0.0);
    let scf = solve(&cell);
    let forces = analytic_forces(&cell, &functional, &scf, &options)
        .unwrap()
        .total;
    let stress = analytic_stress(&cell, &functional, &scf, &options).unwrap();
    let values: Vec<_> = forces
        .iter()
        .flat_map(|f| (-*f).to_array())
        .chain(stress.strain_derivative.iter().flatten().copied())
        .collect();
    assert!(forces.iter().fold(Vec3::zero(), |s, v| s + *v).norm() < 1e-7);
    println!("NaH finite-k omegaB97X-V energy={:.15e} VV10={:.15e} orbital={:?} forces={forces:?} stress={:?} raw rotation={:.15e}",
        scf.energy, scf.e_nlc, scf.orbital_residual, stress.stress, stress.rotation_residual);
    let names: Vec<_> = (0..15)
        .map(|i| format!("NaH finite-k omegaB97X-V first derivative {i}"))
        .collect();
    let mut failures = Vec::new();
    let mut cache = std::collections::BTreeMap::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.018, 0.012, 0.008, 0.005], stencils::Stencil::Second),
            ([0.009, 0.006, 0.004, 0.0025], stencils::Stencil::Second),
            ([0.025, 0.018, 0.013, 0.009], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            values
                .iter()
                .enumerate()
                .map(|(i, target)| {
                    let mut energy = |h: f64| {
                        *cache
                            .entry((i, h.to_bits()))
                            .or_insert_with(|| solve(&build(i, h)).energy)
                    };
                    steps.map(|h| {
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (energy(2.0 * h), energy(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(energy(h), energy(-h), p2, m2, h) - target).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("finite-k VV10 forces and stress", &failures);
}

#[test]
fn periodic_nonlocal_moving_density_response_has_independent_four_step_fits() {
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let options = PeriodicScfOptions {
        vv10: Some(PeriodicVv10Options {
            cutoff: 11.3,
            ..Default::default()
        }),
        e_tol: 1e-13,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        max_iter: 250,
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(10),
            angular_override: Some(7),
            prune: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let direction = [[0.1, -0.05, -0.3], [-0.1, 0.05, 0.3]];
    let at = |t: f64, solved: usize| {
        let positions = [[-0.1, -0.05, -0.7], [0.1, 0.05, 0.7]];
        let molecule = Molecule::new(
            (0..2)
                .map(|i| Atom {
                    z: 1,
                    position: Vec3::new(
                        positions[i][0] + t * direction[i][0],
                        positions[i][1] + t * direction[i][1],
                        positions[i][2] + t * direction[i][2],
                    ),
                })
                .collect(),
        );
        let cell = Cell::new(
            molecule,
            Lattice::cubic(9.0),
            &data::load("sto-3g").unwrap(),
            7.0,
        )
        .unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged);
        let result = lattice_path_energy(
            &cell,
            &functional,
            &reference,
            &options,
            1,
            &direction,
            solved,
            &ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged);
        assert!(result.residuals.iter().all(|r| r.abs() < 1e-10));
        result.energy
    };
    let exact = at(0.0, 1);
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut estimates = [Vec::new(), Vec::new()];
    for h in steps {
        let plus = at(h, 0);
        let minus = at(-h, 0);
        estimates[0].push((plus[0] - minus[0]) / (2.0 * h));
        estimates[1].push((plus[1] - minus[1]) / (2.0 * h));
    }
    for n in 1..=2 {
        let value = exact[n] * if n == 2 { 2.0 } else { 1.0 };
        let errors: Vec<_> = estimates[n - 1].iter().map(|v| (v - value).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[n - 1][3] - estimates[n - 1][2]) / 3.0 - value).abs();
        println!("periodic nonlocal moving order={n} value={value:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
        assert!(rich < 1e-7);
    }
}

#[test]
fn omega_b97x_v_gamma_hessian_fits_all_independent_directions() {
    use dft_rs::driver::periodic::{prepare_periodic, PeriodicMethod};
    use dft_rs::driver::{run, Controls, System, Value};
    use stencils::Stencil;

    let system = System {
        numbers: vec![11, 1],
        positions: vec![[0.13, -0.21, 0.07], [2.21, -1.28, 1.61]],
        cell: Some([[9.4, 0.71, 0.33], [0.45, 10.8, 0.69], [0.24, -0.55, 11.2]]),
        pbc: [true; 3],
    };
    let input = Value::object()
        .with("xc", "wb97x-v")
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
        .with("xc_ao_cutoff", 9.0)
        .with("vv10_cutoff", 9.0)
        .with("grid_level", 0)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-12)
        .with("max_iter", 400);
    let report = run(
        "phonons",
        &system,
        &Controls::from_value(&input.clone().with("response_tolerance", 1e-12)).unwrap(),
    )
    .unwrap();
    let scalar = |key| report.get(key).and_then(Value::as_f64).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (shape, hessian) = report
        .get("force_constants")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert_eq!(shape, vec![6, 6]);
    assert!(scalar("acoustic_residual") < 1e-7);
    println!(
        "NaH omegaB97X-V E={:.17e} H={hessian:?} frequencies={:?} residuals={:?} stationary energy defect={:.17e}",
        scalar("energy"),
        report.get("frequencies_cm1"),
        report.get("response_residuals"),
        scalar("series_reference_energy_difference")
    );
    let settings = Controls::from_value(&input).unwrap();
    let method = PeriodicMethod::from_controls_with_dispersion(&settings).unwrap();
    let solver = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let directions = symmetric_cartesian_directions::directions(6, 2);
    let names: Vec<_> = (0..directions.len())
        .map(|i| format!("NaH omegaB97X-V Hessian direction {i}"))
        .collect();
    let exact: Vec<_> = directions
        .iter()
        .map(|u| {
            (0..6)
                .flat_map(|a| (0..6).map(move |b| (a, b)))
                .map(|(a, b)| u[a] * hessian[6 * a + b] * u[b])
                .sum::<f64>()
        })
        .collect();
    let slope = |index: usize, h: f64| {
        let mut displaced = system.clone();
        let direction: Vec<[f64; 3]> = directions[index]
            .chunks_exact(3)
            .map(|u| [u[0], u[1], u[2]])
            .collect();
        for (r, u) in displaced.positions.iter_mut().zip(&direction) {
            for a in 0..3 {
                r[a] += h * u[a];
            }
        }
        let setup = prepare_periodic(&displaced, &method).unwrap();
        let scf = run_periodic_scf(&setup.cell, &setup.functional, &setup.options).unwrap();
        assert!(scf.converged && scf.e_nlc.abs() > 1e-8);
        // Rebuild and reoptimize the displaced geometry. The first
        // derivative uses stationarity alone with no orbital response solve.
        let path = lattice_path_energy(
            &setup.cell,
            &setup.functional,
            &scf,
            &setup.options,
            1,
            &direction,
            0,
            &solver,
        )
        .unwrap();
        assert!(path.converged);
        println!(
            "NaH omegaB97X-V oracle [{index}] h={h:.17e} slope={:.17e} E_nlc={:.17e} SCF={:.17e}",
            path.energy[1],
            scf.e_nlc,
            scf.orbital_residual.unwrap()
        );
        path.energy[1]
    };
    let mut samples = std::collections::BTreeMap::new();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
        ],
        |steps, stencil| {
            exact
                .iter()
                .enumerate()
                .map(|(index, target)| {
                    let mut at = |h: f64| {
                        *samples
                            .entry((index, h.to_bits()))
                            .or_insert_with(|| slope(index, h))
                    };
                    steps.map(|h| {
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(at(h), at(-h), plus2, minus2, h) - target).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("omegaB97X-V full symmetric Gamma Hessian", &failures);
    let mut worst = 0.0_f64;
    for (i, target) in exact.iter().enumerate() {
        let derivative =
            |h: f64| (samples[&(i, h.to_bits())] - samples[&(i, (-h).to_bits())]) / (2.0 * h);
        worst = worst.max(((4.0 * derivative(0.005) - derivative(0.01)) / 3.0 - target).abs());
    }
    println!("NaH omegaB97X-V maximum Richardson defect={worst:.17e}");
    assert!(worst < 2e-7);
}
