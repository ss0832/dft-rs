// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every homogeneous-strain derivative on a genuinely heteronuclear C1 cell.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{BasisSet, ElementBasis, Shell};
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::stress::analytic_stress;
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule, Vec3};

fn fixture(axis: Option<(usize, usize, f64)>, finite_k: bool) -> (Cell, PeriodicScfOptions) {
    let deform = |v: Vec3| {
        let mut row = [v.x, v.y, v.z];
        if let Some((a, b, h)) = axis {
            row[a] += h * row[b];
        }
        Vec3::new(row[0], row[1], row[2])
    };
    let mut basis = BasisSet::new("C1 compact heteronuclear strain validation");
    for (z, exponent) in [(1, 1.2), (2, 1.6)] {
        basis.insert(ElementBasis {
            z,
            shells: vec![Shell {
                l: 0,
                exps: vec![exponent],
                coefs: vec![1.0],
            }],
        });
    }
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: deform(Vec3::new(0.11, -0.17, 0.23)),
        },
        Atom {
            z: 2,
            position: deform(Vec3::new(1.24, 0.19, 0.57)),
        },
    ])
    .with_charge(1);
    let scale = if finite_k { 0.65 } else { 1.0 };
    let cell = Cell::new(
        molecule,
        Lattice::new(
            deform(Vec3::new(7.3, 0.21, 0.17) * scale),
            deform(Vec3::new(0.35, 7.9, 0.41) * scale),
            deform(Vec3::new(0.13, 0.29, 8.6) * scale),
        ),
        &basis,
        if finite_k { 6.0 } else { 3.0 },
    )
    .unwrap();
    let options = PeriodicScfOptions {
        max_iter: 500,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        xc_ao_cutoff: Some(8.0),
        grid_shell: 1,
        grid: dft_rs::grid::GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..Default::default()
        },
        k_mesh: if finite_k { [2, 1, 1] } else { [1, 1, 1] },
        ..Default::default()
    };
    (cell, options)
}

fn check_all_components(finite_k: bool, functionals: &[&str]) {
    for &name in functionals {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let (cell, options) = fixture(None, finite_k);
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged);
        if finite_k {
            let contrast = reference.overlap[0]
                .re
                .as_slice()
                .iter()
                .zip(reference.overlap[1].re.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                contrast > 1e-6,
                "The finite-k fixture must contain measurable intercell overlap"
            );
        }
        let stress = analytic_stress(&cell, &functional, &reference, &options).unwrap();
        println!(
            "{name} C1 reference energy={:.16e} derivative={:?}",
            reference.energy, stress.strain_derivative
        );
        let names: Vec<String> = (0..3)
            .flat_map(|a| (0..3).map(move |b| format!("{name} strain[{a},{b}]")))
            .collect();
        let energy = |a, b, h| {
            let (cell, options) = fixture(Some((a, b, h)), finite_k);
            let result = run_periodic_scf(&cell, &functional, &options).unwrap();
            assert!(result.converged);
            result.energy
        };
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 9];
                for a in 0..3 {
                    for b in 0..3 {
                        for (k, &h) in steps.iter().enumerate() {
                            let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                                (energy(a, b, 2.0 * h), energy(a, b, -2.0 * h))
                            } else {
                                (0.0, 0.0)
                            };
                            let finite =
                                stencil.first(energy(a, b, h), energy(a, b, -h), plus2, minus2, h);
                            errors[3 * a + b][k] = (finite - stress.strain_derivative[a][b]).abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("every C1 XC stress component", &failures);
    }
}

#[test]
fn all_c1_xc_stress_components_keep_independent_four_step_fits() {
    check_all_components(false, &["svwn5", "pbe", "tpss"]);
}

#[test]
fn every_finite_k_c1_stress_component_has_an_independent_fit() {
    check_all_components(true, &["pbe"]);
}
