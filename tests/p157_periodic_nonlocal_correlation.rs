// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! A distinct nonlocal double-hybrid target on fully responding periodic
//! orbitals. Energy oracles do not use the production pair functional.
#[path = "support/convergence.rs"]
mod convergence;
mod nonlocal_oracle {
    use super::PeriodicVv10Options;
    include!("support/nonlocal_oracle.rs");
}
use dft_rs::basis::{data, n_sph};
use dft_rs::grid::{eval::eval_aos_full, GridOptions};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::periodic::grid::PeriodicGrid;
use dft_rs::periodic::nonlocal::PeriodicVv10Options;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::nuclear::{LatticeCorrelatedSolver, LatticePath};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::double_hybrid::DoubleHybridFunctional;
use dft_rs::{Atom, Molecule};

fn independent_nonlocal(
    cell: &Cell,
    options: &PeriodicScfOptions,
    c: &Matrix,
    occupied: usize,
    settings: PeriodicVv10Options,
) -> f64 {
    let grid = PeriodicGrid::build(
        cell.molecule(),
        cell.lattice(),
        &options.grid,
        options.grid_shell,
    )
    .unwrap();
    let nao = c.rows;
    let points = grid.points.len();
    let mut phi = Matrix::zeros(points, nao);
    let mut gradient: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(points, nao));
    // Direct integer-cube Bloch sum with the declared physical AO cutoff.
    // No response grid's AO arrays or density contractions are read.
    for x in -4..=4 {
        for y in -4..=4 {
            for z in -4..=4 {
                let translation = cell.lattice().translation([x, y, z]);
                let shifted: Vec<_> = grid.points.iter().map(|p| *p - translation).collect();
                let batch = eval_aos_full(cell.basis(), &shifted, true, false);
                for shell in &cell.basis().shells {
                    for i in 0..points {
                        if (shell.center - shifted[i]).norm()
                            > options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff())
                        {
                            continue;
                        }
                        for mu in shell.sph_off..shell.sph_off + n_sph(shell.l) {
                            phi[(i, mu)] += batch.phi[(i, mu)];
                            for axis in 0..3 {
                                gradient[axis][(i, mu)] +=
                                    batch.grad.as_ref().unwrap()[axis][(i, mu)];
                            }
                        }
                    }
                }
            }
        }
    }
    let mo = phi.matmul_seq(c);
    let derivatives = gradient.map(|v| v.matmul_seq(c));
    let rho: Vec<_> = (0..points)
        .map(|i| (0..occupied).map(|n| 2.0 * mo[(i, n)].powi(2)).sum())
        .collect();
    let sigma: Vec<_> = (0..points)
        .map(|i| {
            (0..3)
                .map(|a| {
                    (0..occupied)
                        .map(|n| 4.0 * mo[(i, n)] * derivatives[a][(i, n)])
                        .sum::<f64>()
                        .powi(2)
                })
                .sum()
        })
        .collect();
    nonlocal_oracle::energy(
        cell.lattice().a.map(|a| a.to_array()),
        [true; 3],
        &grid.points.iter().map(|p| p.to_array()).collect::<Vec<_>>(),
        &rho,
        &sigma,
        &grid.weights,
        settings,
    )
}

fn independent_pair(cell: &Cell, c: &Matrix, energies: &[f64]) -> f64 {
    assert_eq!(c.cols, 2);
    let mut density = Matrix::zeros(c.rows, c.rows);
    for mu in 0..c.rows {
        for nu in 0..c.rows {
            density[(mu, nu)] = 0.5 * (c[(mu, 0)] * c[(nu, 1)] + c[(mu, 1)] * c[(nu, 0)]);
        }
    }
    let images = cell
        .density_images_from_k(
            &[Vec3::zero()],
            &[(density, Matrix::zeros(c.rows, c.rows))],
            2.0 * cell.hartree_real_space_cutoff().max(cell.overlap_cutoff()),
        )
        .unwrap();
    let (j, _) = cell.hartree(&images, Vec3::zero()).unwrap();
    let g = c.transpose().matmul_seq(&j).matmul_seq(c)[(0, 1)];
    g * g / (2.0 * (energies[0] - energies[1]))
}

#[test]
fn distinct_nonlocal_target_and_periodic_pair_response_have_four_step_fits() {
    let recipe = DoubleHybridFunctional::wb97m2();
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(-0.1, -0.05, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.1, 0.05, 0.7),
            },
        ]),
        Lattice::cubic(9.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        vv10: Some(PeriodicVv10Options {
            parameters: recipe.reference_vv10.unwrap(),
            cutoff: 11.3,
        }),
        e_tol: 1e-13,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        max_iter: 250,
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(8),
            angular_override: Some(5),
            prune: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let (coefficient, parameters) = recipe.target_vv10.unwrap();
    let target_settings = PeriodicVv10Options {
        parameters,
        cutoff: 11.3,
    };
    let initial = run_periodic_scf(&cell, &recipe.parent, &options).unwrap();
    assert!(initial.converged);
    let make = |field| {
        LatticeCorrelatedSolver::new(
            &cell,
            &recipe.parent,
            recipe.target.as_ref(),
            &initial,
            &options,
            1,
            field,
            recipe.correlation,
            512,
        )
        .unwrap()
        .with_target_nonlocal(coefficient, target_settings)
        .unwrap()
    };
    let u = [0.2, -0.3, 0.7];
    let bias = u.map(|v| 0.02 * v);
    let config = ResponseSolverConfig {
        tolerance: 1e-13,
        ..Default::default()
    };
    let line = LatticePath {
        field: vec![u],
        ..Default::default()
    };
    let solver = make(bias);
    let exact = solver.solve(&line, 2, &config).unwrap();
    let (c, energies, occupied) = solver.reference_orbitals();
    assert_eq!(occupied, [1, 1]);
    let parent_oracle = independent_nonlocal(&cell, &options, &c[0], 1, options.vv10.unwrap());
    let target_oracle =
        coefficient * independent_nonlocal(&cell, &options, &c[0], 1, target_settings);
    let pair_oracle = independent_pair(&cell, &c[0], &energies[0]);
    println!("periodic distinct nonlocal energy oracles parent={parent_oracle:.15e} target={target_oracle:.15e} OS={pair_oracle:.15e} errors=[{:.12e},{:.12e},{:.12e}]",
        (parent_oracle-exact.parent_nonlocal[0]).abs(),(target_oracle-exact.target_nonlocal[0]).abs(),(pair_oracle-exact.opposite_spin[0]).abs());
    assert!((parent_oracle - exact.parent_nonlocal[0]).abs() < 1e-12);
    assert!((target_oracle - exact.target_nonlocal[0]).abs() < 1e-12);
    assert!((pair_oracle - exact.opposite_spin[0]).abs() < 1e-8);
    assert!((exact.target_nonlocal[0] - coefficient * exact.parent_nonlocal[0]).abs() > 1e-7);
    assert!(exact
        .same_spin_alpha
        .iter()
        .chain(&exact.same_spin_beta)
        .all(|v| v.abs() < 1e-15));
    for residual in [
        &exact.response_residuals,
        &exact.amplitude_residuals,
        &exact.projector_residuals,
        &exact.metric_residuals,
        &exact.stationarity_residuals,
    ] {
        assert!(residual.iter().all(|v| v.abs() < 1e-9));
    }
    println!("periodic distinct raw residuals orbital={:?} pair={:?} projector={:?} metric={:?} stationarity={:?}",
        exact.response_residuals, exact.amplitude_residuals, exact.projector_residuals,
        exact.metric_residuals, exact.stationarity_residuals);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let mut estimates: [[Vec<f64>; 2]; 4] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for h in steps {
        let mut results = Vec::new();
        for sign in [-1.0, 1.0] {
            let displaced = make(std::array::from_fn(|a| bias[a] + sign * h * u[a]));
            results.push(displaced.solve(&line, 1, &config).unwrap());
        }
        for (channel, values) in [
            (&results[0].energy, &results[1].energy),
            (&results[0].parent_nonlocal, &results[1].parent_nonlocal),
            (&results[0].target_nonlocal, &results[1].target_nonlocal),
            (&results[0].opposite_spin, &results[1].opposite_spin),
        ]
        .into_iter()
        .enumerate()
        {
            for n in 0..2 {
                estimates[channel][n].push((values.1[n] - values.0[n]) / (2.0 * h));
            }
        }
    }
    for (channel, (label, series)) in [
        ("total", &exact.energy),
        ("parent NLC", &exact.parent_nonlocal),
        ("target NLC", &exact.target_nonlocal),
        ("OS", &exact.opposite_spin),
    ]
    .into_iter()
    .enumerate()
    {
        for n in 1..=2 {
            let value = series[n] * if n == 2 { 2.0 } else { 1.0 };
            let errors: Vec<_> = estimates[channel][n - 1]
                .iter()
                .map(|e| (e - value).abs())
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[channel][n - 1][3] - estimates[channel][n - 1][2]) / 3.0
                - value)
                .abs();
            println!("periodic distinct {label} order={n} value={value:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
            assert!(rich < 1e-6);
        }
    }
    let zero = solver
        .with_target_nonlocal(0.0, target_settings)
        .unwrap()
        .solve(&line, 2, &config)
        .unwrap();
    for n in 0..=2 {
        assert_eq!(zero.target_nonlocal[n], 0.0);
        assert!((exact.energy[n] - zero.energy[n] - exact.target_nonlocal[n]).abs() < 1e-11);
    }
}
