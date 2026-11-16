// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Masked ECP operators and hybrid reference fields against independent
//! molecular ECP and closed one-electron energy/component oracles.
use dft_rs::basis::{data, BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::integrals::{bloch, dimensional_ecp_image_blocks, ecp_image_blocks};
use dft_rs::periodic::scf_boundary::{
    run_dimensional_scf, DimensionalScfBackend, DimensionalScfOptions,
};
use dft_rs::periodic::{Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn system() -> (Molecule, MolBasis) {
    let mol = Molecule::new(vec![Atom {
        z: 11,
        position: Vec3::zero(),
    }])
    .with_multiplicity(2);
    let physical = MolBasis::build(&mol, &data::load("lanl2dz").unwrap()).unwrap();
    let mut set = BasisSet::new("compact-valence-oracle");
    set.insert(ElementBasis {
        z: 11,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
    let mut mb = MolBasis::build(&mol, &set).unwrap();
    mb.ecps = physical.ecps;
    mb.n_core_electrons = physical.n_core_electrons;
    (mol, mb)
}

#[test]
fn ecp_masks_match_molecular_and_three_dimensional_operators() {
    let (_, mb) = system();
    let molecular = dft_rs::integrals::ecp_matrix(&mb).unwrap();
    for mask in 1..8 {
        let boundary = PeriodicBoundary::new(std::array::from_fn(|i| mask & (1 << i) != 0));
        let geometry = PeriodicGeometry::new(Lattice::cubic(40.0), boundary).unwrap();
        let blocks = dimensional_ecp_image_blocks(&mb, &geometry, 4.3, 4.3).unwrap();
        let (re, im) = bloch(&blocks, Vec3::new(0.21, 0.13, 0.08));
        assert!(re.max_abs_diff(&molecular) < 1e-13);
        assert!(im.as_slice().iter().all(|x| x.abs() < 1e-13));
        assert!(blocks
            .iter()
            .all(|b| (0..3).all(|a| boundary.axes[a] || b.translation.get(a) == 0.0)));
    }
    let lattice = Lattice::new(
        Vec3::new(3.7, 0.0, 0.0),
        Vec3::new(0.3, 4.2, 0.0),
        Vec3::new(0.1, 0.2, 4.8),
    );
    let geometry = PeriodicGeometry::new(lattice.clone(), PeriodicBoundary::FULL_3D).unwrap();
    let dimensional = dimensional_ecp_image_blocks(&mb, &geometry, 4.3, 4.3).unwrap();
    let original = ecp_image_blocks(&mb, &lattice, 4.3, 4.3).unwrap();
    for k in [Vec3::zero(), Vec3::new(0.21, 0.13, 0.08)] {
        let (a, b) = bloch(&dimensional, k);
        let (c, d) = bloch(&original, k);
        assert!(a.max_abs_diff(&c) < 1e-13 && b.max_abs_diff(&d) < 1e-13);
    }
}

#[test]
fn dimensional_ecp_hf_has_separate_energy_oracles_and_consistent_shifted_fields() {
    let (mol, mb) = system();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let ecp = dft_rs::integrals::ecp_matrix(&mb).unwrap()[(0, 0)];
    let kinetic = 1.5 * 0.7;
    let attraction = -2.0 * (1.4 / std::f64::consts::PI).sqrt();
    let expected = kinetic + attraction + ecp;
    for axes in [
        [true, false, false],
        [true, true, false],
        [true, true, true],
    ] {
        let boundary = PeriodicBoundary::new(axes);
        let geometry = PeriodicGeometry::new(Lattice::cubic(40.0), boundary).unwrap();
        let mut o = DimensionalScfOptions::for_boundary(boundary);
        o.basis_cutoff = 4.3;
        o.grid_cutoff = 10.0;
        o.scf.k_mesh = [1; 3];
        o.scf.max_iter = 30;
        o.scf.conv_tol_grad = 1e-9;
        o.electrostatics.alpha = Some(0.55);
        o.electrostatics.real_cutoff = Some(16.0);
        o.electrostatics.reciprocal_cutoff = Some(8.0);
        let report = run_dimensional_scf(&mol, &mb, &geometry, &functional, &o).unwrap();
        println!(
            "dimensional ECP axes={axes:?} E={:.12e} expected={expected:.12e} \
            kinetic={:.12e} ecp={:.12e} components={:.3e} orbital={:.3e}",
            report.reference.energy,
            report.kinetic_energy,
            report.ecp_energy,
            report.component_energy_defect,
            report.reference_diagnostics.orbital_residual
        );
        assert!(report.converged);
        assert_eq!(report.reference.n_alpha + report.reference.n_beta, 1);
        assert!((report.reference.energy - expected).abs() < 2e-8);
        assert!((report.kinetic_energy - kinetic).abs() < 1e-13);
        assert!((report.ecp_energy - ecp).abs() < 1e-13);
        assert!(report.component_energy_defect < 1e-13);
        let backend = DimensionalScfBackend::new(&mol, &mb, &geometry, &functional, &o).unwrap();
        let (_, fock) = backend
            .reference_fock_at(&report.reference.density, backend.k_points())
            .unwrap();
        for spin in 0..2 {
            assert!(
                fock[spin][0]
                    .re
                    .max_abs_diff(&report.reference.fock[spin][0].re)
                    < 1e-10
            );
        }
        let k: [f64; 3] = std::array::from_fn(|i| {
            if axes[i] {
                [0.021, 0.013, 0.008][i]
            } else {
                0.0
            }
        });
        let (_, shifted) = backend
            .reference_fock_at(&report.reference.density, &[Vec3::new(k[0], k[1], k[2])])
            .unwrap();
        for spin in 0..2 {
            assert!(shifted[spin][0].re.max_abs_diff(&fock[spin][0].re) < 1e-10);
        }
    }
}
