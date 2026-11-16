// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent dimensional XC density/potential oracles and variational fits.

use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::basis_boundary::dimensional_bloch_aos;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::integrals::{bloch, dimensional_image_blocks, Operator};
use dft_rs::periodic::xc_boundary::{DimensionalXcBackend, PeriodicXcOptions};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, Family, FunctionalSpec, HybridParams, ResolvedFunctional};
use std::f64::consts::PI;

fn fixture(
    d: usize,
) -> (
    Molecule,
    MolBasis,
    PeriodicGeometry,
    PeriodicXcOptions,
    [Vec<BlochMatrix>; 2],
) {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.2, -0.1, 0.3),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.6, 0.5, -0.4),
        },
    ])
    .with_charge(1);
    let mut definition = BasisSet::new("contracted-XC-reference");
    definition.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7, 0.3],
            coefs: vec![0.6, 0.4],
        }],
    });
    let mb = MolBasis::build(&mol, &definition).unwrap();
    let geometry = PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(5.2, 0.5, 0.3),
            Vec3::new(0.8, 5.6, 0.4),
            Vec3::new(0.3, 0.2, 6.0),
        ),
        PeriodicBoundary::new(std::array::from_fn(|i| i < d)),
    )
    .unwrap();
    let options = PeriodicXcOptions {
        k_mesh: [3, 1, 1],
        basis_cutoff: 8.0,
        grid_cutoff: 10.0,
        grid: GridOptions {
            radial_override: Some(24),
            angular_override: Some(13),
            prune: false,
            ..Default::default()
        },
        grid_tolerance: 2e-3,
        ..Default::default()
    };
    let k_points = geometry.k_points(options.k_mesh).unwrap();
    let images =
        dimensional_image_blocks(&mb, &geometry, options.basis_cutoff, Operator::Overlap).unwrap();
    let overlaps: Vec<_> = k_points
        .iter()
        .map(|&k| BlochMatrix::from_pair(bloch(&images, k)))
        .collect();
    let mut p: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
        (0..3)
            .map(|k| {
                let mut p = BlochMatrix::zeros(2, 2);
                let scale = if spin == 0 { 0.65 } else { 0.35 };
                p.re[(0, 0)] = 0.6 * scale;
                p.re[(1, 1)] = 0.5 * scale;
                p.re[(0, 1)] = 0.2 * scale;
                p.re[(1, 0)] = 0.2 * scale;
                p.im[(0, 1)] = 0.07 * scale * (k as f64 + 0.3).cos();
                p.im[(1, 0)] = -p.im[(0, 1)];
                p
            })
            .collect()
    });
    let ne = p
        .iter()
        .map(|s| {
            s.iter()
                .zip(&overlaps)
                .map(|(p, s)| p.trace_product(s))
                .sum::<f64>()
                / 3.0
        })
        .sum::<f64>();
    for matrix in p.iter_mut().flatten() {
        matrix.scale(1.0 / ne);
    }
    (mol, mb, geometry, options, p)
}
fn functional(name: &str) -> ResolvedFunctional {
    let spec = if name == "slater" {
        FunctionalSpec::Split {
            x: "slater".into(),
            c: "none".into(),
        }
    } else {
        FunctionalSpec::Combined(name.into())
    };
    resolve(&spec).unwrap()
}
fn difference(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}
fn fit(label: &str, errors: [f64; 4]) {
    let h = [0.04_f64, 0.02, 0.01, 0.005];
    assert!(errors.iter().all(|e| e.is_finite() && *e > 0.0));
    let x = h.map(f64::ln);
    let y = errors.map(f64::ln);
    let xm = x.iter().sum::<f64>() / 4.0;
    let ym = y.iter().sum::<f64>() / 4.0;
    let p = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - xm) * (b - ym))
        .sum::<f64>()
        / x.iter().map(|a| (a - xm).powi(2)).sum::<f64>();
    let r2 = 1.0
        - x.iter()
            .zip(y)
            .map(|(a, b)| (b - ym - p * (a - xm)).powi(2))
            .sum::<f64>()
            / y.iter().map(|b| (b - ym).powi(2)).sum::<f64>();
    println!("{label} errors={errors:?}; density steps=0.04..0.005 p={p:.12} R2={r2:.12}");
    assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
    for pair in errors.windows(2) {
        assert!((pair[0] / pair[1] / 4.0 - 1.0).abs() < 0.03);
    }
}

#[test]
fn slater_matches_independent_closed_spin_density_energy_and_complex_potentials() {
    for d in 1..=3 {
        let (mol, mb, geometry, options, density) = fixture(d);
        let functional = functional("slater");
        let backend =
            DimensionalXcBackend::new(&mol, &mb, &geometry, &functional, &options).unwrap();
        let report = backend.evaluate(&functional, &density).unwrap();
        let grid = backend.grid();
        let aos: Vec<_> = report
            .k_points
            .iter()
            .map(|&k| {
                dimensional_bloch_aos(&mb, &geometry, &grid.points, k, options.basis_cutoff)
                    .unwrap()
            })
            .collect();
        let factor = -0.75 * (6.0 / PI).cbrt();
        let mut energy = 0.0;
        let mut v: [Vec<BlochMatrix>; 2] =
            std::array::from_fn(|_| (0..3).map(|_| BlochMatrix::zeros(2, 2)).collect());
        for spin in 0..2 {
            for point in 0..grid.len() {
                let mut rho = 0.0;
                for k in 0..3 {
                    for a in 0..2 {
                        for b in 0..2 {
                            let ar = aos[k].re[(point, a)];
                            let ai = aos[k].im[(point, a)];
                            let br = aos[k].re[(point, b)];
                            let bi = aos[k].im[(point, b)];
                            rho += (density[spin][k].re[(a, b)] * (ar * br + ai * bi)
                                - density[spin][k].im[(a, b)] * (ai * br - ar * bi))
                                / 3.0;
                        }
                    }
                }
                assert!(rho >= 0.0);
                energy += grid.weights[point] * factor * rho.powf(4.0 / 3.0);
                let local = grid.weights[point] * (4.0 / 3.0) * factor * rho.cbrt();
                for k in 0..3 {
                    for a in 0..2 {
                        for b in 0..2 {
                            let ar = aos[k].re[(point, a)];
                            let ai = aos[k].im[(point, a)];
                            let br = aos[k].re[(point, b)];
                            let bi = aos[k].im[(point, b)];
                            v[spin][k].re[(a, b)] += local * (ar * br + ai * bi);
                            v[spin][k].im[(a, b)] += local * (ar * bi - ai * br);
                        }
                    }
                }
            }
        }
        let defect = report
            .potential
            .iter()
            .flatten()
            .zip(v.iter().flatten())
            .map(|(a, b)| difference(a, b))
            .fold(0.0, f64::max);
        println!(
            "{d}D Slater closed energy defect={} complex potential defect={defect}",
            (report.energy - energy).abs()
        );
        assert!((report.energy - energy).abs() < 1e-11 && defect < 1e-11);
        assert!(report.hermiticity_defect < 1e-11 && report.component_energy_defect < 1e-11);
    }
}

#[test]
fn all_functional_classes_have_four_step_variational_density_fits() {
    for d in 1..=3 {
        let (mol, mb, geometry, options, density) = fixture(d);
        let backend =
            DimensionalXcBackend::new(&mol, &mb, &geometry, &functional("r2scan"), &options)
                .unwrap();
        let mut delta = density.clone();
        for (index, matrix) in delta.iter_mut().flatten().enumerate() {
            matrix.scale(0.2);
            matrix.re[(0, 1)] += 0.017 * (index as f64).sin();
            matrix.re[(1, 0)] = matrix.re[(0, 1)];
        }
        for name in ["slater", "pbe", "r2scan", "pbe0", "cam-b3lyp", "wb97mv"] {
            let f = functional(name);
            let exact = backend.evaluate(&f, &density).unwrap();
            let analytic = delta
                .iter()
                .flatten()
                .zip(exact.potential.iter().flatten())
                .map(|(p, v)| p.trace_product(v) / 3.0)
                .sum::<f64>();
            let mut errors = [0.0; 4];
            let mut estimates = [0.0; 4];
            for (i, h) in [0.04, 0.02, 0.01, 0.005].into_iter().enumerate() {
                let mut plus = density.clone();
                let mut minus = density.clone();
                for ((a, b), dp) in plus
                    .iter_mut()
                    .flatten()
                    .zip(minus.iter_mut().flatten())
                    .zip(delta.iter().flatten())
                {
                    a.add_scaled(dp, h);
                    b.add_scaled(dp, -h);
                }
                estimates[i] = (backend.evaluate(&f, &plus).unwrap().energy
                    - backend.evaluate(&f, &minus).unwrap().energy)
                    / (2.0 * h);
                errors[i] = (estimates[i] - analytic).abs();
            }
            fit(&format!("{d}D {name} XC variational"), errors);
            assert!(((4.0 * estimates[3] - estimates[2]) / 3.0 - analytic).abs() < 1e-8);
            assert!(exact.component_energy_defect < 1e-11 && exact.hermiticity_defect < 1e-11);
            if name == "wb97mv" {
                assert_eq!(exact.required_nonlocal, Some("VV10"));
            }
        }
    }
}

#[test]
fn unregistered_provider_parts_and_all_masks_preserve_semilocal_assembly() {
    let first = functional("slater");
    let second = functional("r2scan");
    let parts = vec![
        (0.7, first.parts[0].1.clone()),
        (0.3, second.parts[0].1.clone()),
    ];
    let custom = ResolvedFunctional::custom(
        "weighted provider",
        parts,
        HybridParams::PURE,
        Family::MetaGga,
    );
    let (mol, mb, original, mut options, density) = fixture(3);
    for mask in 1..8 {
        let axes = std::array::from_fn(|i| mask & (1 << i) != 0);
        options.k_mesh = [1; 3];
        options.k_mesh[(0..3).find(|&i| axes[i]).unwrap()] = 3;
        let mut sparse = original.lattice().clone();
        for i in 0..3 {
            if !axes[i] {
                sparse.a[i] = Vec3::zero();
            }
        }
        let geometry =
            PeriodicGeometry::new(original.lattice().clone(), PeriodicBoundary::new(axes)).unwrap();
        let removed = PeriodicGeometry::new(sparse, PeriodicBoundary::new(axes)).unwrap();
        let a = DimensionalXcBackend::new(&mol, &mb, &geometry, &custom, &options).unwrap();
        let b = DimensionalXcBackend::new(&mol, &mb, &removed, &custom, &options).unwrap();
        let ra = a.evaluate(&custom, &density).unwrap();
        let rb = b.evaluate(&custom, &density).unwrap();
        assert!((ra.energy - rb.energy).abs() < 1e-12);
        assert!(ra
            .potential
            .iter()
            .flatten()
            .zip(rb.potential.iter().flatten())
            .all(|(a, b)| difference(a, b) < 1e-12));
        let mut v: [Vec<BlochMatrix>; 2] =
            std::array::from_fn(|_| (0..3).map(|_| BlochMatrix::zeros(2, 2)).collect());
        for (i, (weight, part)) in custom.parts.iter().enumerate() {
            let component = ResolvedFunctional::custom(
                "component",
                vec![(1.0, part.clone())],
                HybridParams::PURE,
                part.family(),
            );
            let single = a.evaluate(&component, &density).unwrap();
            assert!((ra.component_energies[i] - weight * single.energy).abs() < 1e-12);
            for (target, source) in v
                .iter_mut()
                .flatten()
                .zip(single.potential.iter().flatten())
            {
                target.add_scaled(source, *weight);
            }
        }
        assert!(ra
            .potential
            .iter()
            .flatten()
            .zip(v.iter().flatten())
            .all(|(a, b)| difference(a, b) < 1e-12));
    }
}

#[test]
fn invalid_spin_density_and_incompatible_cached_family_are_rejected() {
    let (mol, mb, geometry, options, mut density) = fixture(1);
    let f = functional("slater");
    let backend = DimensionalXcBackend::new(&mol, &mb, &geometry, &f, &options).unwrap();
    assert!(backend.evaluate(&functional("r2scan"), &density).is_err());
    density[0][0].im[(0, 1)] = f64::NAN;
    assert!(backend.evaluate(&f, &density).is_err());
    let mut invalid = mb.clone();
    invalid.shells[0].sph_off = usize::MAX;
    assert!(DimensionalXcBackend::new(&mol, &invalid, &geometry, &f, &options).is_err());
}
