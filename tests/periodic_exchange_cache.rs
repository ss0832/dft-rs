// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Cache capacity, changed sources and immutable operator identity.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;

use dft_rs::basis::{BasisSet, ElementBasis, Shell};
use dft_rs::dfpt::exchange::FiniteQExchange;
use dft_rs::integrals::CoulombKernel;
use dft_rs::periodic::{BlochMatrix, Cell, Lattice};
use dft_rs::{Atom, Molecule, Vec3};

fn fixture(displacement: f64) -> Cell {
    let mut positions = [
        [0.1, -0.2, 0.3],
        [1.35, 0.1, 0.2],
        [-0.3, 3.2, 0.9],
        [0.2, 3.4, 2.2],
    ];
    positions[2][2] += displacement;
    c1_geometry::assert_c1(&[1; 4], &positions);
    let mut basis = BasisSet::new("C1 immutable exchange cache regression");
    basis.insert(ElementBasis {
        z: 1,
        shells: vec![
            Shell {
                l: 0,
                exps: vec![0.91],
                coefs: vec![1.],
            },
            Shell {
                l: 1,
                exps: vec![0.73],
                coefs: vec![1.],
            },
        ],
    });
    Cell::new(
        Molecule::new(
            positions
                .into_iter()
                .map(|p| Atom {
                    z: 1,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        ),
        Lattice::new(
            Vec3::new(8.7, 0.31, 0.13),
            Vec3::new(0.2, 9.2, 0.43),
            Vec3::new(0.11, 0.27, 9.7),
        ),
        &basis,
        4.5,
    )
    .unwrap()
}

fn sources(cell: &Cell) -> Vec<BlochMatrix> {
    (0..2)
        .map(|k| {
            let n = cell.nao();
            let mut d = BlochMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    d.re[(i, j)] = (0.3 * (1 + i + 3 * j + 5 * k) as f64).sin();
                    d.im[(i, j)] = (0.17 * (2 + 2 * i + j + 7 * k) as f64).cos();
                }
            }
            d
        })
        .collect()
}

fn defect(a: &[BlochMatrix], b: &[BlochMatrix]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| [a.re.max_abs_diff(&b.re), a.im.max_abs_diff(&b.im)])
        .fold(0., f64::max)
}

#[test]
fn disabled_partial_and_full_caches_preserve_every_complex_action_and_changed_source() {
    let cell = fixture(0.);
    let q = Vec3::new(0.043, 0.017, -0.021);
    let points = [Vec3::zero(), cell.lattice().reciprocal()[0] * 0.5];
    let density = sources(&cell);
    for kernel in [
        CoulombKernel::Coulomb,
        CoulombKernel::Erf { omega: 0.3 },
        CoulombKernel::Erfc { omega: 0.3 },
    ] {
        let direct = FiniteQExchange::with_cache_budget(&cell, &points, q, 4.5, kernel, 0).unwrap();
        let expected = direct.apply(&density).unwrap();
        for budget in [0, 256, 4096, 32 * 1024 * 1024] {
            let cached =
                FiniteQExchange::with_cache_budget(&cell, &points, q, 4.5, kernel, budget).unwrap();
            assert_eq!(cached.cached_integral_storage_bound(), 0);
            let mut invalid = density.clone();
            invalid[0].re[(0, 0)] = f64::NAN;
            assert!(cached.apply(&invalid).is_err());
            assert_eq!(cached.cached_integral_storage_bound(), 0);
            let cold = cached.apply(&density).unwrap();
            let warm = cached.apply(&density).unwrap();
            assert!(defect(&expected, &cold) < 2e-12);
            assert!(defect(&expected, &warm) < 2e-12);
            assert!(cached.cached_integral_storage_bound() <= budget);
            let mut changed = density.clone();
            changed[0].re[(0, 1)] += 0.37;
            changed[1].im[(1, 0)] -= 0.23;
            let expected_changed = direct.apply(&changed).unwrap();
            let actual_changed = cached.apply(&changed).unwrap();
            assert!(defect(&expected, &expected_changed) > 1e-6);
            assert!(defect(&expected_changed, &actual_changed) < 2e-12);
        }
    }
}

#[test]
fn concurrent_cold_actions_share_one_bounded_initialization() {
    let cell = fixture(0.);
    let points = [Vec3::zero(), cell.lattice().reciprocal()[0] * 0.5];
    let cached = FiniteQExchange::new(
        &cell,
        &points,
        Vec3::new(0.043, 0.017, -0.021),
        4.5,
        CoulombKernel::Coulomb,
    )
    .unwrap();
    let density = sources(&cell);
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(|| cached.apply(&density).unwrap());
        let b = scope.spawn(|| cached.apply(&density).unwrap());
        (a.join().unwrap(), b.join().unwrap())
    });
    assert!(defect(&a, &b) < 2e-12);
    assert!(cached.cached_integral_storage_bound() > 0);
    assert!(cached.cached_integral_storage_bound() <= 32 * 1024 * 1024);
}

#[test]
fn a_new_geometry_and_kernel_start_distinct_empty_operator_caches() {
    let first = fixture(0.);
    let second = fixture(0.07);
    let q = Vec3::new(0.043, 0.017, -0.021);
    let points = [Vec3::zero(), first.lattice().reciprocal()[0] * 0.5];
    let density = sources(&first);
    let a = FiniteQExchange::new(&first, &points, q, 4.5, CoulombKernel::Coulomb).unwrap();
    let first_result = a.apply(&density).unwrap();
    for (cell, kernel) in [
        (&second, CoulombKernel::Coulomb),
        (&first, CoulombKernel::Erf { omega: 0.3 }),
    ] {
        let b = FiniteQExchange::new(cell, &points, q, 4.5, kernel).unwrap();
        assert_eq!(b.cached_integral_storage_bound(), 0);
        let fresh = b.apply(&density).unwrap();
        let direct = FiniteQExchange::with_cache_budget(cell, &points, q, 4.5, kernel, 0).unwrap();
        assert!(defect(&fresh, &direct.apply(&density).unwrap()) < 2e-12);
        assert!(defect(&first_result, &fresh) > 1e-6);
    }
}

#[test]
fn partial_periodic_image_caches_preserve_transfer_phases() {
    let mut basis = BasisSet::new("periodic image cache fixture");
    for (z, exponent) in [(1, 0.91), (2, 1.23)] {
        basis.insert(ElementBasis {
            z,
            shells: vec![Shell {
                l: 0,
                exps: vec![exponent],
                coefs: vec![1.],
            }],
        });
    }
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.1, -0.2, 0.3),
            },
            Atom {
                z: 2,
                position: Vec3::new(1.3, 0.3, -0.1),
            },
        ])
        .with_multiplicity(2),
        Lattice::new(
            Vec3::new(3.2, 0.2, 0.1),
            Vec3::new(0.3, 3.8, -0.1),
            Vec3::new(0.1, 0.2, 4.1),
        ),
        &basis,
        2.5,
    )
    .unwrap();
    let points = [Vec3::zero(), cell.lattice().reciprocal()[0] * 0.5];
    let density = sources(&cell);
    let mut previous: Option<Vec<BlochMatrix>> = None;
    for q in [Vec3::zero(), Vec3::new(0.13, -0.07, 0.03)] {
        let direct =
            FiniteQExchange::with_cache_budget(&cell, &points, q, 2.5, CoulombKernel::Coulomb, 0)
                .unwrap();
        let expected = direct.apply(&density).unwrap();
        for budget in [512, 4096, 65536] {
            let cached = FiniteQExchange::with_cache_budget(
                &cell,
                &points,
                q,
                2.5,
                CoulombKernel::Coulomb,
                budget,
            )
            .unwrap();
            assert_eq!(cached.cached_integral_storage_bound(), 0);
            assert!(defect(&expected, &cached.apply(&density).unwrap()) < 2e-12);
            assert!(defect(&expected, &cached.apply(&density).unwrap()) < 2e-12);
            assert!(cached.cached_integral_storage_bound() <= budget);
        }
        if let Some(before) = previous {
            assert!(defect(&before, &expected) > 1e-6);
        }
        previous = Some(expected);
    }
}
