// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Dimensional SCF conservation, component oracles and variational fits.
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::electrostatics::ChargeCompensation;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::scf_boundary::{
    run_dimensional_scf, DimensionalScfBackend, DimensionalScfOptions,
};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use std::f64::consts::PI;

fn basis(mol: &Molecule) -> MolBasis {
    let mut set = BasisSet::new("single-primitive-oracle");
    for z in [1, 2] {
        set.insert(ElementBasis {
            z,
            shells: vec![Shell {
                l: 0,
                exps: vec![0.7],
                coefs: vec![1.0],
            }],
        });
    }
    MolBasis::build(mol, &set).unwrap()
}
fn options(boundary: PeriodicBoundary) -> DimensionalScfOptions {
    let mut o = DimensionalScfOptions::for_boundary(boundary);
    o.basis_cutoff = 4.3;
    o.grid_cutoff = 10.0;
    o.scf.k_mesh = std::array::from_fn(|i| if boundary.axes[i] { 3 } else { 1 });
    o.scf.grid.radial_override = Some(32);
    o.scf.grid.angular_override = Some(13);
    o.scf.grid.prune = false;
    o.scf.max_iter = 150;
    o.scf.conv_tol_grad = 1e-9;
    o.electrostatics.alpha = Some(0.55);
    o.electrostatics.real_cutoff = Some(16.0);
    o.electrostatics.reciprocal_cutoff = Some(8.0);
    o
}
#[test]
fn all_masks_and_signed_formal_charge_use_actual_self_consistent_electrons() {
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    for mask in 1..8 {
        let boundary = PeriodicBoundary::new(std::array::from_fn(|i| mask & (1 << i) != 0));
        let geometry = PeriodicGeometry::new(Lattice::cubic(5.5), boundary).unwrap();
        for charge in [-1, 0, 1] {
            let mol = Molecule::new(vec![Atom {
                z: 1,
                position: Vec3::new(0.2, -0.1, 0.3),
            }])
            .with_charge(charge)
            .with_multiplicity(if charge == 0 { 2 } else { 1 });
            let mb = basis(&mol);
            let mut o = options(boundary);
            o.electrostatics.compensation = ChargeCompensation::GaussianCountercharge {
                center: Vec3::new(-0.5, 0.7, 0.9),
                width: 1.1,
            };
            let report = run_dimensional_scf(&mol, &mb, &geometry, &functional, &o).unwrap();
            assert!(
                report.converged,
                "mask={mask} charge={charge} warnings={:?}",
                report.warnings
            );
            assert!(report.reference_diagnostics.particle_number_error < 1e-12);
            assert!(report.reference_diagnostics.orbital_residual < 1e-12);
            assert!(report.occupation_consistency < 1e-12);
            assert!(report.component_energy_defect < 1e-12);
            assert_eq!(report.compensating_charge, -(charge as f64));
            assert_eq!(
                report.reference.n_alpha + report.reference.n_beta,
                (1 - charge) as usize
            );
            println!(
                "dimensional SCF mask={mask} charge={charge} E={:.15e} residual={:.3e}",
                report.reference.energy, report.reference_diagnostics.orbital_residual
            );
        }
    }
}
#[test]
fn isolated_neutral_gaussian_hf_matches_closed_energy_components() {
    let mol = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol);
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    // Spherical neutral charge distributions have no electrostatic image
    // interaction outside their Gaussian tails. Nuclear gauge shares cancel.
    let geometry = PeriodicGeometry::new(Lattice::cubic(40.0), PeriodicBoundary::FULL_3D).unwrap();
    let mut o = options(PeriodicBoundary::FULL_3D);
    o.scf.k_mesh = [1; 3];
    o.electrostatics.alpha = Some(0.12);
    o.electrostatics.real_cutoff = Some(100.0);
    o.electrostatics.reciprocal_cutoff = Some(4.0);
    let r = run_dimensional_scf(&mol, &mb, &geometry, &functional, &o).unwrap();
    let kinetic = 3.0 * 0.7;
    let attraction = -8.0 * (2.0 * 0.7 / PI).sqrt();
    let repulsion = 4.0 * (0.7 / PI).sqrt();
    let exchange = -2.0 * (0.7 / PI).sqrt();
    assert!((r.kinetic_energy - kinetic).abs() < 1e-12);
    assert!((r.full_exchange_energy - exchange).abs() < 1e-12);
    let coulomb = r.reference.e_nuc + r.electron_nuclear_energy + r.hartree_energy;
    assert!(
        (coulomb - attraction - repulsion).abs() < 1e-10,
        "Coulomb closed defect {}",
        coulomb - attraction - repulsion
    );
    assert!((r.reference.energy - kinetic - attraction - repulsion - exchange).abs() < 1e-10);
}
#[test]
fn complete_dimensional_energy_has_variational_fock_on_four_step_orbital_paths() {
    for d in 1..=3 {
        let boundary = PeriodicBoundary::new(std::array::from_fn(|i| i < d));
        let geometry = PeriodicGeometry::new(Lattice::cubic(5.5), boundary).unwrap();
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.4, 0.3, 0.2),
            },
        ]);
        let mb = basis(&mol);
        let mut o = options(boundary);
        o.scf.k_mesh = [1; 3];
        o.scf.exchange_cutoff = Some(6.0);
        for name in ["hf", "svwn5", "pbe", "pbe0", "cam-b3lyp", "r2scan"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let backend =
                DimensionalScfBackend::new(&mol, &mb, &geometry, &functional, &o).unwrap();
            let s = &backend.overlap()[0].re;
            let c0 = [1.0 / s[(0, 0)].sqrt(), 0.0];
            let x = s[(0, 1)] / s[(0, 0)];
            let scale = (s[(1, 1)] - x * s[(0, 1)]).sqrt();
            let c1 = [-x / scale, 1.0 / scale];
            let theta = 0.37_f64;
            let density = |t: f64| {
                let c = std::array::from_fn::<_, 2, _>(|i| {
                    c0[i] * (theta + t).cos() + c1[i] * (theta + t).sin()
                });
                let mut p = BlochMatrix::zeros(2, 2);
                for i in 0..2 {
                    for j in 0..2 {
                        p.re[(i, j)] = c[i] * c[j];
                    }
                }
                [vec![p.clone()], vec![p]]
            };
            let c = std::array::from_fn::<_, 2, _>(|i| c0[i] * theta.cos() + c1[i] * theta.sin());
            let dc = std::array::from_fn::<_, 2, _>(|i| -c0[i] * theta.sin() + c1[i] * theta.cos());
            let mut dp = BlochMatrix::zeros(2, 2);
            for i in 0..2 {
                for j in 0..2 {
                    dp.re[(i, j)] = dc[i] * c[j] + c[i] * dc[j];
                }
            }
            let base = backend.evaluate(&density(0.0)).unwrap();
            let analytic = dp.trace_product(&base.fock[0][0]) + dp.trace_product(&base.fock[1][0]);
            let hs = [0.004_f64, 0.002, 0.001, 0.0005];
            let mut values = [0.0; 4];
            for (index, h) in hs.iter().enumerate() {
                values[index] = (backend.evaluate(&density(*h)).unwrap().electronic_energy
                    - backend.evaluate(&density(-*h)).unwrap().electronic_energy)
                    / (2.0 * h);
            }
            let errors = values.map(|v| (v - analytic).abs());
            assert!(errors.iter().all(|e| e.is_finite() && *e > 0.0));
            let x = hs.map(f64::ln);
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
            println!("dimensional full Fock d={d} {name} h=0.004..0.0005 errors={errors:?} p={p:.12} R2={r2:.12}");
            assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
            for e in errors.windows(2) {
                assert!((e[0] / e[1] / 4.0 - 1.0).abs() < 0.03);
            }
            assert!(((4.0 * values[3] - values[2]) / 3.0 - analytic).abs() < 1e-8);
            let solved = run_dimensional_scf(&mol, &mb, &geometry, &functional, &o).unwrap();
            assert!(solved.reference_diagnostics.orbital_residual < 1e-9);
            assert!(solved.occupation_consistency < 1e-8);
        }
    }
}
