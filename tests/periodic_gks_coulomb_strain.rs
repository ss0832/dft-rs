// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent complete affine energy checks at fixed complex Bloch density.
use dft_rs::basis::{g94, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::periodic::coulomb_boundary::{DimensionalCoulombBackend, PeriodicCoulombOptions};
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/stencils.rs"]
pub mod stencils;

fn frame(component: usize, amount: f64, mesh: [usize; 3]) -> DimensionalCoulombBackend {
    let map = |r: Vec3| {
        let mut x = r.to_array();
        if component < 9 {
            x[component / 3] += amount * x[component % 3];
        }
        Vec3::new(x[0], x[1], x[2])
    };
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: map(Vec3::new(0.13, -0.21, 0.07)),
        },
        Atom {
            z: 2,
            position: map(Vec3::new(1.61, 0.45, -0.31)),
        },
    ])
    .with_multiplicity(2);
    let basis = g94::parse(
        "affine-complex-density",
        "H 0\nS 1 1.0\n0.8 1.0\n****\nHe 0\nS 1 1.0\n1.1 1.0\n****\n",
    )
    .unwrap();
    let basis = MolBasis::build(&mol, &basis).unwrap();
    let geometry = PeriodicGeometry::new(
        Lattice::new(
            map(Vec3::new(4.7, 0.31, 0.17)),
            map(Vec3::new(0.21, 5.3, 0.19)),
            map(Vec3::new(0.13, 0.27, 5.9)),
        ),
        PeriodicBoundary::FULL_3D,
    )
    .unwrap();
    let mut options = PeriodicCoulombOptions {
        k_mesh: mesh,
        basis_cutoff: 4.4,
        ..Default::default()
    };
    options.electrostatics.alpha = Some(0.5);
    options.electrostatics.real_cutoff = Some(12.0);
    options.electrostatics.reciprocal_cutoff = Some(7.0);
    options.electrostatics.quadrature_order = 16;
    options.electrostatics.tolerance = 1e-11;
    DimensionalCoulombBackend::new(&mol, &basis, &geometry, &options).unwrap()
}
fn density(backend: &DimensionalCoulombBackend) -> Vec<BlochMatrix> {
    let mut ps = Vec::new();
    for (i, _) in backend.k_points().iter().enumerate() {
        let mut p = BlochMatrix::zeros(2, 2);
        p.re[(0, 0)] = 1.2 + 0.03 * i as f64;
        p.re[(1, 1)] = 1.3 - 0.03 * i as f64;
        p.re[(0, 1)] = 0.17;
        p.re[(1, 0)] = 0.17;
        p.im[(0, 1)] = 0.09 + 0.02 * i as f64;
        p.im[(1, 0)] = -p.im[(0, 1)];
        ps.push(p);
    }
    let number = backend.evaluate(Some(&ps)).unwrap().electron_count.unwrap();
    for p in &mut ps {
        p.scale(3.0 / number);
    }
    ps
}
#[test]
fn all_bloch_coulomb_strain_components_have_four_step_energy_fits() {
    dft_rs::resources::ResourceLimits::new(Some(1),Some(512)).unwrap().install(|| {
        for mesh in [[1,1,1],[3,1,1]] {
            let base=frame(9,0.0,mesh);let ps=density(&base);
            let reference=base.evaluate(Some(&ps)).unwrap();assert!(reference.converged,"{mesh:?} Coulomb reference unconverged");
            let analytic=base.affine_strain(&ps).unwrap();assert!(analytic.converged,"Coulomb strain provider unconverged");
            assert!(analytic.reference_energy_defect<=1e-11);
            let tensors=[analytic.source,analytic.electron_nuclear,analytic.hartree,analytic.total];
            let names:Vec<_>=(0..36).map(|i|format!("Bloch Coulomb {mesh:?} term={} component={}",i/9,i%9)).collect();
            let mut failures=Vec::new();
            stencils::fit_components_staged(&names,&[([0.04,0.02,0.01,0.005],stencils::Stencil::Second)],|steps,stencil| {
                let mut errors=vec![[0.0;4];36];
                for component in 0..9 {for (i,&h) in steps.iter().enumerate() {
                    let energy=|amount| {
                        let report=frame(component,amount,mesh).evaluate(Some(&ps)).unwrap();
                        [report.source_energy,report.electron_nuclear_components.unwrap().iter().sum(),report.hartree_energy_components.unwrap().iter().sum(),report.total_coulomb_energy.unwrap()]
                    };
                    let plus=energy(h);let minus=energy(-h);
                    for term in 0..4 {
                        let target=tensors[term][component];let independent=stencil.first(plus[term],minus[term],0.0,0.0,h);let error=(target-independent).abs();errors[9*term+component][i]=error;
                        println!("BLOCH_COULOMB_SIGNED mesh={mesh:?} row={} h={h:.17e} analytic={target:.17e} reference={independent:.17e} error={error:.17e}",9*term+component);
                    }
                }}errors
            },&mut failures);
            components::assert_no_failures("all 36 Coulomb affine components per mesh",&failures);
        }
    }).unwrap();
}

#[test]
fn exact_ao_pair_cutoff_strain_is_rejected() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.25, 0.0, 0.0),
        },
    ]);
    let set = g94::parse("cutoff-domain", "H 0\nS 1 1.0\n0.8 1.0\n****\n").unwrap();
    let basis = MolBasis::build(&mol, &set).unwrap();
    let geometry = PeriodicGeometry::new(Lattice::cubic(7.0), PeriodicBoundary::FULL_3D).unwrap();
    let options = PeriodicCoulombOptions {
        basis_cutoff: 0.25,
        ..Default::default()
    };
    let backend = DimensionalCoulombBackend::new(&mol, &basis, &geometry, &options).unwrap();
    let mut p = BlochMatrix::zeros(2, 2);
    p.re[(0, 0)] = 1.0;
    p.re[(1, 1)] = 1.0;
    let error = backend
        .affine_strain(&[p])
        .err()
        .expect("exact hard cutoff has no affine derivative");
    assert!(error.to_string().contains("cutoff"));
}
