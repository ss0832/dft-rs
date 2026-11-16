// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::data;
use dft_rs::dfpt::coulomb::{plane_wave_pair, FiniteQHartree};
use dft_rs::dfpt::orbitals::adjoint;
use dft_rs::dfpt::xc::FiniteQXc;
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::bloch::BlochMatrix;
use dft_rs::periodic::xc::{
    xc_contribution, xc_contribution_at_grid, AoDerivativeOrder, PeriodicXcGrid,
};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn cell() -> Cell {
    Cell::new(
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(-0.4, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.3, 0.1, 0.7),
            },
        ]),
        Lattice::cubic(9.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap()
}
fn mesh_density(nk: usize) -> [Vec<BlochMatrix>; 2] {
    std::array::from_fn(|s| {
        (0..nk)
            .map(|_| {
                let mut d = BlochMatrix::zeros(2, 2);
                d.re[(0, 0)] = 0.6 + 0.1 * s as f64;
                d.re[(1, 1)] = 0.5;
                d.re[(0, 1)] = 0.1;
                d.re[(1, 0)] = 0.1;
                d
            })
            .collect()
    })
}
fn delta(nk: usize) -> [Vec<BlochMatrix>; 2] {
    std::array::from_fn(|s| {
        (0..nk)
            .map(|ik| {
                let mut d = BlochMatrix::zeros(2, 2);
                d.re[(0, 0)] = 0.02;
                d.re[(1, 1)] = -0.03;
                d.re[(0, 1)] = 0.01;
                d.re[(1, 0)] = 0.01;
                d.im[(0, 1)] = 0.017 * (1.0 + ik as f64 + s as f64);
                d.im[(1, 0)] = -d.im[(0, 1)];
                d
            })
            .collect()
    })
}
fn difference(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}

#[test]
fn finite_q_plane_waves_recover_overlap_and_crossing_identity() {
    let cell = cell();
    let k = Vec3::new(0.11, 0.07, -0.03);
    let q = Vec3::new(0.043, -0.02, 0.01);
    let overlap = plane_wave_pair(&cell, k, Vec3::zero(), Vec3::zero(), 7.0).unwrap();
    assert!(difference(&overlap, &BlochMatrix::from_pair(cell.overlap(k))) < 1e-12);
    let wave = q + cell.lattice().reciprocal()[0];
    let plus = plane_wave_pair(&cell, k, q, wave, 7.0).unwrap();
    let minus = plane_wave_pair(&cell, k + q, q * (-1.0), wave * (-1.0), 7.0).unwrap();
    assert!(difference(&plus, &adjoint(&minus).unwrap()) < 2e-12);
    assert!(plus.hermiticity_error() > 1e-4);
    assert!(plane_wave_pair(&cell, k, q, Vec3::new(0.2, 0.0, 0.0), 7.0).is_err());
}

#[test]
fn analytic_periodic_xc_kernels_match_independent_density_derivatives() {
    let cell = cell();
    let points = cell.k_points([2, 1, 1]);
    let ground = mesh_density(points.len());
    let change = delta(points.len());
    let grid = PeriodicXcGrid::build(
        &cell,
        &points,
        &GridOptions {
            radial_override: Some(10),
            angular_override: Some(7),
            prune: false,
            weight_cutoff: 0.0,
            ..Default::default()
        },
        1,
        7.0,
        AoDerivativeOrder::Gradient,
    )
    .unwrap();
    for name in ["svwn5", "pbe", "r2scan"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let kernel = FiniteQXc::new(&functional, &grid, &grid, &ground).unwrap();
        let analytic = kernel.apply(&change).unwrap();
        for step in [1e-3, 5e-4] {
            let mut plus = ground.clone();
            let mut minus = ground.clone();
            for s in 0..2 {
                for k in 0..points.len() {
                    plus[s][k].add_scaled(&change[s][k], step);
                    minus[s][k].add_scaled(&change[s][k], -step);
                }
            }
            let vp = xc_contribution(&functional, &grid, &plus).unwrap();
            let vm = xc_contribution(&functional, &grid, &minus).unwrap();
            for s in 0..2 {
                for k in 0..points.len() {
                    let mut numerical = vp.vxc[s][k].clone();
                    numerical.add_scaled(&vm.vxc[s][k], -1.0);
                    numerical.scale(0.5 / step);
                    assert!(
                        difference(&analytic[s][k], &numerical) < 2e-7,
                        "{name}, step {step}: {}",
                        difference(&analytic[s][k], &numerical)
                    );
                }
            }
        }
        let q = Vec3::new(0.047, -0.018, 0.023);
        let shifted = grid
            .at_k_points(
                &cell,
                &points.iter().map(|&k| k + q).collect::<Vec<_>>(),
                7.0,
            )
            .unwrap();
        let forward = FiniteQXc::new(&functional, &grid, &shifted, &ground).unwrap();
        let reverse =
            FiniteQXc::with_reference(&functional, &grid, &ground, &shifted, &grid).unwrap();
        let mut arbitrary = change.clone();
        arbitrary[0][0].re[(0, 1)] += 0.07;
        arbitrary[1][1].im[(1, 1)] = 0.04;
        let conjugated: [Vec<_>; 2] =
            std::array::from_fn(|s| arbitrary[s].iter().map(|d| adjoint(d).unwrap()).collect());
        let a = forward.apply(&arbitrary).unwrap();
        let b = reverse.apply(&conjugated).unwrap();
        for s in 0..2 {
            for k in 0..points.len() {
                assert!(difference(&a[s][k], &adjoint(&b[s][k]).unwrap()) < 1e-12);
            }
        }
        let ground_shifted =
            xc_contribution_at_grid(&functional, &grid, &ground, &shifted).unwrap();
        let original = xc_contribution(&functional, &grid, &ground).unwrap();
        assert_eq!(ground_shifted.exc, original.exc);
        assert_eq!(ground_shifted.n_elec, original.n_elec);
        assert!(difference(&ground_shifted.vxc[0][0], &original.vxc[0][0]) > 1e-7);
    }
}

#[test]
fn finite_q_hartree_is_complex_linear_and_preserves_the_energy_pairing() {
    let cell = cell();
    let points = vec![Vec3::new(0.1, 0.02, 0.0)];
    let q = Vec3::new(0.043, 0.013, 0.0);
    let kernel = FiniteQHartree::new(&cell, &points, q, 0.9, 7.0).unwrap();
    let mut change = delta(1);
    change[0][0].re[(1, 0)] += 0.08;
    change[1][0].im[(1, 1)] = 0.03;
    let potential = kernel.apply(&change).unwrap();
    let rho = kernel.density_fourier(&change).unwrap();
    let mut pairing = 0.0;
    for s in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                pairing += change[s][0].re[(i, j)] * potential[s][0].re[(i, j)]
                    + change[s][0].im[(i, j)] * potential[s][0].im[(i, j)];
            }
        }
    }
    let expected: f64 = rho
        .iter()
        .zip(&kernel.waves)
        .map(|(r, g)| {
            4.0 * std::f64::consts::PI * cell.lattice().volume() * r.abs().powi(2) / g.norm2()
        })
        .sum();
    assert!((pairing - expected).abs() < 1e-12);
    assert!(pairing > 0.0);
    let mut imaginary = change.clone();
    for d in imaginary.iter_mut().flatten() {
        for i in 0..2 {
            for j in 0..2 {
                let r = d.re[(i, j)];
                d.re[(i, j)] = -d.im[(i, j)];
                d.im[(i, j)] = r;
            }
        }
    }
    let shifted = kernel.apply(&imaginary).unwrap();
    for s in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                assert!((shifted[s][0].re[(i, j)] + potential[s][0].im[(i, j)]).abs() < 1e-12);
                assert!((shifted[s][0].im[(i, j)] - potential[s][0].re[(i, j)]).abs() < 1e-12);
            }
        }
    }
}
