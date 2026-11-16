// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{eval::eval_aos_full, GridOptions, MolecularGrid};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::nlc::{rvv10_potential, vv10_potential, Vv10Params};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use stencils::Stencil;

const NUMBERS: [u8; 4] = [7, 1, 1, 1];
const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.78, 0.34, -0.08],
    [-0.49, 1.65, 0.41],
    [0.31, -0.78, 1.91],
];

// Independently collocate a symmetric density and evaluate Eq. (5) as two
// full loops, using q and the damping product directly.
fn oracle_terms(basis: &MolBasis, grid: &MolecularGrid, density: &Matrix) -> [f64; 2] {
    let pi = std::f64::consts::PI;
    let b: f64 = 6.3;
    let c = 0.0093;
    let beta = (3.0 / b.powi(2)).powf(0.75) / 32.0;
    let ao = eval_aos_full(basis, &grid.points, true, false);
    let grad = ao.grad.as_ref().unwrap();
    let endpoints: Vec<_> = (0..grid.len())
        .map(|i| {
            let mut rho: f64 = 0.0;
            let mut drho = [0.0; 3];
            for a in 0..basis.nao_sph {
                for b in 0..basis.nao_sph {
                    let left = ao.phi[(i, a)];
                    let right = ao.phi[(i, b)];
                    rho += density[(a, b)] * left * right;
                    for x in 0..3 {
                        drho[x] +=
                            density[(a, b)] * (grad[x][(i, a)] * right + left * grad[x][(i, b)]);
                    }
                }
            }
            (rho, drho)
        })
        .enumerate()
        .filter_map(|(i, (rho, drho))| {
            if rho < 1e-8 {
                return None;
            }
            let kappa = b * (3.0 * pi / 2.0) * (rho / (9.0 * pi)).powf(1.0 / 6.0);
            let relative_gradient_sq = drho.iter().map(|d| (d / rho).powi(2)).sum::<f64>();
            let omega = (c * relative_gradient_sq.powi(2) + (4.0 * pi / 3.0) * rho).sqrt();
            Some((i, grid.weights[i] * rho, kappa, omega / kappa))
        })
        .collect();
    let local = beta
        * endpoints
            .iter()
            .map(|(_, weighted, _, _)| weighted)
            .sum::<f64>();
    let mut pair = 0.0;
    for &(i, n, k, q) in &endpoints {
        for &(j, np, kp, qp) in &endpoints {
            let r2 = (grid.points[i] - grid.points[j]).norm2();
            pair -= 0.75 * n * np
                / ((k * kp).powf(1.5) * (1.0 + q * r2) * (1.0 + qp * r2) * (2.0 + (q + qp) * r2));
        }
    }
    [local, pair]
}

#[test]
fn revised_kernel_ammonia_energy_and_every_ao_potential_component() {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let mol = Molecule::new(
        NUMBERS
            .into_iter()
            .zip(POSITIONS)
            .map(|(z, [x, y, zz])| Atom {
                z,
                position: Vec3::new(x, y, zz),
            })
            .collect(),
    );
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let reference_grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let derivative_grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            radial_override: Some(10),
            angular_override: Some(3),
            prune: false,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let pbe = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let options = ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &basis, &options, &pbe, Some(&reference_grid)).unwrap();
    assert!(scf.converged);
    let params = Vv10Params::REVISED;
    let mut density = scf.p_total();
    let (energy, _) = rvv10_potential(&basis, &reference_grid, &density, params).unwrap();
    let [local, pair] = oracle_terms(&basis, &reference_grid, &density);
    let original = vv10_potential(&basis, &reference_grid, &density, params)
        .unwrap()
        .0;
    assert!((energy - (local + pair)).abs() < 1e-11);
    assert!((energy - original).abs() > 1e-6);
    assert!(local > 0.0 && pair < 0.0);
    println!(
        "NH3 RKS/PBE/STO-3G: {} SCF iterations, {} quadrature points, reference={:.12e} Eh, rVV10={energy:.12e} Eh, local={local:.12e} Eh, pair={pair:.12e} Eh, original at same b and C={original:.12e} Eh, frozen total={:.12e} Eh",
        scf.iterations,
        reference_grid.len(),
        scf.energy,
        scf.energy + energy
    );

    // A small diagonal shift keeps both signs of every AO variation in the
    // interior of the positive-definite cone, including occupied-virtual mixes.
    for a in 0..basis.nao_sph {
        density[(a, a)] += 0.07;
    }
    let (energy, potential) = rvv10_potential(&basis, &derivative_grid, &density, params).unwrap();
    assert!(
        (energy
            - oracle_terms(&basis, &derivative_grid, &density)
                .into_iter()
                .sum::<f64>())
        .abs()
            < 1e-12
    );
    let axes: Vec<_> = (0..basis.nao_sph)
        .flat_map(|a| (0..=a).map(move |b| (a, b)))
        .collect();
    let names: Vec<_> = axes
        .iter()
        .map(|(a, b)| format!("rVV10 AO potential [{a},{b}] "))
        .collect();
    let stages = [
        ([0.016, 0.009, 0.005, 0.003], Stencil::Second),
        ([0.003, 0.002, 0.0013, 0.0008], Stencil::Second),
        ([0.04, 0.025, 0.016, 0.009], Stencil::Second),
        ([0.04, 0.025, 0.016, 0.009], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &stages,
        |steps, stencil| {
            axes.iter()
                .map(|&(a, b)| {
                    let oracle = |step: f64| {
                        let mut trial = density.clone();
                        if a == b {
                            trial[(a, b)] += step;
                        } else {
                            trial[(a, b)] += 0.5 * step;
                            trial[(b, a)] += 0.5 * step;
                        }
                        oracle_terms(&basis, &derivative_grid, &trial)
                            .into_iter()
                            .sum::<f64>()
                    };
                    steps.map(|h| {
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (oracle(2.0 * h), oracle(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(oracle(h), oracle(-h), plus2, minus2, h) - potential[(a, b)])
                            .abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("rVV10 AO potential", &failures);

    let zero = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    let (empty_energy, empty_potential) =
        rvv10_potential(&basis, &derivative_grid, &zero, params).unwrap();
    assert_eq!(empty_energy, 0.0);
    assert_eq!(empty_potential, zero);
}
