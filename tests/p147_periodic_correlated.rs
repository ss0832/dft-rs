// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic correlated electric and moving-basis response acceptance.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::data;
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::nuclear::{LatticeCorrelatedSolver, LatticePath};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::double_hybrid::{DoubleHybridFunctional, Pt2Coefficients};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

fn options() -> PeriodicScfOptions {
    PeriodicScfOptions {
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
    }
}
fn cell(numbers: &[u8], positions: &[[f64; 3]]) -> Cell {
    Cell::new(
        Molecule::new(
            numbers
                .iter()
                .zip(positions)
                .map(|(&z, p)| Atom {
                    z,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        ),
        Lattice::cubic(9.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap()
}
fn fit(label: &str, exact: f64, estimates: &[f64], steps: &[f64; 4], rich_bound: f64) {
    let errors: Vec<_> = estimates.iter().map(|v| (v - exact).abs()).collect();
    let fit = convergence::error_power_fit(steps, &errors);
    let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
    println!("{label} value={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
    assert!(
        (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999,
        "{label}"
    );
    assert!(rich < rich_bound, "{label}");
}

#[test]
fn finite_bias_periodic_pt2_energy_dipole_alpha_beta_gamma_have_four_step_fits() {
    let cell = cell(&[1, 1], &[[-0.1, -0.05, -0.7], [0.1, 0.05, 0.7]]);
    let options = options();
    let config = ResponseSolverConfig {
        tolerance: 1e-13,
        ..Default::default()
    };
    let u = [0.2, -0.3, 0.7];
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for model in ["b2plyp", "pbe", "tpss", "pbe0"] {
        if std::env::var("DFT_RS_TEST_MODEL")
            .ok()
            .is_some_and(|m| m != model)
        {
            continue;
        }
        let recipe = DoubleHybridFunctional::named(model);
        let (functional, weights) = if let Some(recipe) = recipe {
            (recipe.parent, recipe.correlation)
        } else {
            (
                resolve(&FunctionalSpec::Combined(model.into())).unwrap(),
                Pt2Coefficients {
                    same_spin: 0.2,
                    opposite_spin: 0.3,
                },
            )
        };
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        let bias = u.map(|v| 0.02 * v);
        let solver = LatticeCorrelatedSolver::new(
            &cell,
            &functional,
            None,
            &reference,
            &options,
            1,
            bias,
            weights,
            512,
        )
        .unwrap();
        let line = LatticePath {
            field: vec![u],
            ..Default::default()
        };
        let exact = solver.solve(&line, 4, &config).unwrap();
        for (key, values) in [
            ("response", &exact.response_residuals),
            ("pair", &exact.amplitude_residuals),
            ("projector", &exact.projector_residuals),
            ("stationarity", &exact.stationarity_residuals),
            ("metric", &exact.metric_residuals),
        ] {
            let max = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            println!("periodic {model} {key} residual={max:.12e}");
            assert!(max < 1e-9, "{key}");
        }
        assert!(exact
            .number_residuals
            .iter()
            .flatten()
            .all(|v| v.abs() < 1e-9));
        assert!(exact.orbital_frame_residual < 1e-9);
        let centre = solver.solve(&line, 1, &config).unwrap();
        let mut values: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        let mut os: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for h in steps {
            let mut displaced = Vec::new();
            for factor in [1.0, -1.0, 2.0, -2.0] {
                let field = std::array::from_fn(|a| bias[a] + factor * h * u[a]);
                let solver = LatticeCorrelatedSolver::new(
                    &cell,
                    &functional,
                    None,
                    &reference,
                    &options,
                    1,
                    field,
                    weights,
                    512,
                )
                .unwrap();
                displaced.push(solver.solve(&line, 1, &config).unwrap());
            }
            for (out, choose, base) in [
                (&mut values, 0, centre.energy[1]),
                (&mut os, 1, centre.opposite_spin[1]),
            ] {
                let e = |j: usize, n: usize| {
                    if choose == 0 {
                        displaced[j].energy[n]
                    } else {
                        displaced[j].opposite_spin[n]
                    }
                };
                out[0].push((e(0, 0) - e(1, 0)) / (2.0 * h));
                out[1].push((e(0, 1) - e(1, 1)) / (2.0 * h));
                out[2].push((e(0, 1) - 2.0 * base + e(1, 1)) / (h * h));
                out[3]
                    .push((e(2, 1) - 2.0 * e(0, 1) + 2.0 * e(1, 1) - e(3, 1)) / (2.0 * h.powi(3)));
            }
        }
        for (channel, series, estimates) in [
            ("total", &exact.energy, &values),
            ("OS", &exact.opposite_spin, &os),
        ] {
            for n in 1..=4 {
                let factorial = (1..=n).product::<usize>() as f64;
                fit(
                    &format!("periodic {model} {channel} energy order {n}"),
                    factorial * series[n],
                    &estimates[n - 1],
                    &steps,
                    1e-5,
                );
            }
        }
        let zero = LatticeCorrelatedSolver::new(
            &cell,
            &functional,
            None,
            &reference,
            &options,
            1,
            bias,
            Pt2Coefficients {
                same_spin: 0.0,
                opposite_spin: 0.0,
            },
            512,
        )
        .unwrap()
        .solve(&line, 2, &config)
        .unwrap();
        assert!(zero.correlation.iter().all(|e| *e == 0.0));
        assert!(zero.energy.iter().zip(&zero.parent).all(|(a, b)| a == b));
    }
}

#[test]
fn periodic_correlated_nuclear_response_includes_metric_and_integral_derivatives() {
    let positions = [[-0.1, -0.05, -0.7], [0.1, 0.05, 0.7]];
    let direction = [[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]];
    let cell0 = cell(&[1, 1], &positions);
    let functional = DoubleHybridFunctional::b2plyp();
    let options = options();
    let config = ResponseSolverConfig {
        tolerance: 1e-13,
        ..Default::default()
    };
    let reference = run_periodic_scf(&cell0, &functional.parent, &options).unwrap();
    let exact = LatticeCorrelatedSolver::new(
        &cell0,
        &functional.parent,
        None,
        &reference,
        &options,
        1,
        [0.0; 3],
        functional.correlation,
        512,
    )
    .unwrap()
    .solve(&LatticePath::displacement(&direction), 2, &config)
    .unwrap();
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut estimates: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    let mut os: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for h in steps {
        let mut values = Vec::new();
        for sign in [1.0, -1.0] {
            let p: Vec<_> = positions
                .iter()
                .zip(direction)
                .map(|(p, d)| std::array::from_fn(|a| p[a] + sign * h * d[a]))
                .collect();
            let moving = cell(&[1, 1], &p);
            let reference = run_periodic_scf(&moving, &functional.parent, &options).unwrap();
            values.push(
                LatticeCorrelatedSolver::new(
                    &moving,
                    &functional.parent,
                    None,
                    &reference,
                    &options,
                    1,
                    [0.0; 3],
                    functional.correlation,
                    512,
                )
                .unwrap()
                .solve(&LatticePath::default(), 0, &config)
                .unwrap(),
            );
        }
        for (out, a, b, c) in [
            (
                &mut estimates,
                values[0].energy[0],
                values[1].energy[0],
                exact.energy[0],
            ),
            (
                &mut os,
                values[0].opposite_spin[0],
                values[1].opposite_spin[0],
                exact.opposite_spin[0],
            ),
        ] {
            out[0].push((a - b) / (2.0 * h));
            out[1].push((a - 2.0 * c + b) / (h * h));
        }
    }
    for (label, series, estimates) in [
        ("total", &exact.energy, &estimates),
        ("OS", &exact.opposite_spin, &os),
    ] {
        for n in 1..=2 {
            fit(
                &format!("periodic B2PLYP nuclear {label} order {n}"),
                (1..=n).product::<usize>() as f64 * series[n],
                &estimates[n - 1],
                &steps,
                1e-6,
            );
        }
    }
    assert!(exact.metric_residuals.iter().all(|v| *v < 1e-10));
    assert!(exact.projector_residuals.iter().all(|v| *v < 1e-10));
    assert!(exact.stationarity_residuals.iter().all(|v| *v < 1e-9));
    assert!(exact
        .number_residuals
        .iter()
        .flatten()
        .all(|v| v.abs() < 1e-10));
}

#[test]
fn water_periodic_ss_os_have_independent_hartree_energy_oracles_and_response_fits() {
    use dft_rs::linalg::Matrix;
    let cell = cell(
        &[8, 1, 1],
        &[[0., 0., 0.], [1.43, 0., 1.10], [-1.43, 0., 1.10]],
    );
    let options = options();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let config = ResponseSolverConfig {
        tolerance: 1e-13,
        ..Default::default()
    };
    let weights = Pt2Coefficients {
        same_spin: 0.2,
        opposite_spin: 0.3,
    };
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    let solver = LatticeCorrelatedSolver::new(
        &cell,
        &functional,
        None,
        &reference,
        &options,
        1,
        [0.; 3],
        weights,
        512,
    )
    .unwrap();
    let line = LatticePath {
        field: vec![[0., 0., 1.]],
        ..Default::default()
    };
    let exact = solver.solve(&line, 2, &config).unwrap();
    // Independent Coulomb oracle: contract a periodic Hartree operator for
    // each AO transition density. This does not read or transform the
    // response engine's stored four-index Ewald tensor.
    let (orbitals, orbital_energies, occupied) = solver.reference_orbitals();
    let c = &orbitals[0];
    let energies = &orbital_energies[0];
    let no = occupied[0];
    let nv = c.cols - no;
    let nao = c.rows;
    let mut g = vec![0.0; no * nv * no * nv];
    let ix = |i: usize, a: usize, j: usize, b: usize| ((i * nv + a) * no + j) * nv + b;
    for j in 0..no {
        for b in 0..nv {
            let mut density = Matrix::zeros(nao, nao);
            for mu in 0..nao {
                for nu in 0..nao {
                    density[(mu, nu)] =
                        0.5 * (c[(mu, j)] * c[(nu, no + b)] + c[(mu, no + b)] * c[(nu, j)]);
                }
            }
            // Hartree shell pairs have their own screening reach. An
            // overlap-only density-image cutoff drops transition-density
            // images and is not an independent Coulomb oracle.
            let images = cell
                .density_images_from_k(
                    &[Vec3::zero()],
                    &[(density, Matrix::zeros(nao, nao))],
                    2.0 * cell.hartree_real_space_cutoff().max(cell.overlap_cutoff()),
                )
                .unwrap();
            let (hartree, _) = cell.hartree(&images, Vec3::zero()).unwrap();
            let mo = c.transpose().matmul_seq(&hartree).matmul_seq(c);
            for i in 0..no {
                for a in 0..nv {
                    g[ix(i, a, j, b)] = mo[(i, no + a)];
                }
            }
        }
    }
    let mut ss = 0.0;
    let mut os = 0.0;
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    let delta = energies[i] + energies[j] - energies[no + a] - energies[no + b];
                    let direct = g[ix(i, a, j, b)];
                    let anti = direct - g[ix(i, b, j, a)];
                    ss += 0.25 * anti * anti / delta;
                    os += direct * direct / delta;
                }
            }
        }
    }
    println!("water periodic independent Hartree PT2 oracle SS={ss:.15e} OS={os:.15e} SS_error={:.12e} OS_error={:.12e}",
        (ss-exact.same_spin_alpha[0]).abs(),(os-exact.opposite_spin[0]).abs());
    println!(
        "water periodic series-reference energy difference={:.12e}",
        exact.reference.energy_difference
    );
    assert!((ss - exact.same_spin_alpha[0]).abs() < 1e-8);
    assert!((ss - exact.same_spin_beta[0]).abs() < 1e-8);
    assert!((os - exact.opposite_spin[0]).abs() < 1e-8);
    assert!(ss.abs() > 1e-6 && os.abs() > 1e-4);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let mut estimates: [[Vec<f64>; 2]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for h in steps {
        let mut values = Vec::new();
        for sign in [1., -1.] {
            let field = [0., 0., sign * h];
            let solver = LatticeCorrelatedSolver::new(
                &cell,
                &functional,
                None,
                &reference,
                &options,
                1,
                field,
                weights,
                512,
            )
            .unwrap();
            values.push(solver.solve(&LatticePath::default(), 0, &config).unwrap());
        }
        for (index, plus, minus, centre) in [
            (
                0,
                values[0].same_spin_alpha[0],
                values[1].same_spin_alpha[0],
                exact.same_spin_alpha[0],
            ),
            (
                1,
                values[0].opposite_spin[0],
                values[1].opposite_spin[0],
                exact.opposite_spin[0],
            ),
            (
                2,
                values[0].same_spin_beta[0],
                values[1].same_spin_beta[0],
                exact.same_spin_beta[0],
            ),
        ] {
            estimates[index][0].push((plus - minus) / (2. * h));
            estimates[index][1].push((plus - 2. * centre + minus) / (h * h));
        }
    }
    for (index, label, series) in [
        (0, "SS-alpha", &exact.same_spin_alpha),
        (1, "OS", &exact.opposite_spin),
        (2, "SS-beta", &exact.same_spin_beta),
    ] {
        for n in 1..=2 {
            fit(
                &format!("water periodic {label} order {n}"),
                (1..=n).product::<usize>() as f64 * series[n],
                &estimates[index][n - 1],
                &steps,
                1e-6,
            );
        }
    }
}
