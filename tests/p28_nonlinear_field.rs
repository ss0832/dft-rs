// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::properties::dipole_moment;
use dft_rs::response::{electric_field_taylor, polarizability::polarizability};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[test]
fn analytic_nonlinear_field_response_matches_independent_finite_fields() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    for name in ["hf", "svwn5", "pbe", "b3lyp", "cam-b3lyp", "r2scan"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let grid = if functional.parts.is_empty() {
            None
        } else {
            Some(
                MolecularGrid::build(
                    &mol,
                    &GridOptions {
                        level: 1,
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
        };
        let opts = ScfOptions {
            max_iter: 250,
            e_tol: 1e-13,
            p_tol: 1e-12,
            conv_tol_grad: 1e-10,
            ..Default::default()
        };
        let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).unwrap();
        let report = electric_field_taylor(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            [0.0, 0.0, 1.0],
            3,
            &Default::default(),
        )
        .unwrap();
        let c = &report.value.dipole_coefficients;
        let alpha = polarizability(&mol, &mb, &functional, grid.as_ref(), &scf, &opts).unwrap();
        assert!((c[1][2] - alpha[2][2]).abs() < 1e-8, "{name}: alpha");
        assert!(c[2][2].abs() < 1e-8, "{name}: inversion symmetry");
        assert!(
            report.value.projector_defects.iter().all(|x| *x < 1e-9),
            "{name}: idempotency"
        );
        assert!(report.residuals.iter().all(|x| *x <= 1e-10));
        assert!(
            report.value.stationarity_defects.iter().all(|x| *x < 1e-9),
            "{name}: stationarity"
        );
        assert!(
            report
                .value
                .particle_number_derivatives
                .iter()
                .flatten()
                .all(|x| x.abs() < 1e-9),
            "{name}: particle conservation"
        );
        assert!(report.reference.orbital_residual < 1e-9);
        assert!(report.reference.projector_defect < 1e-9);
        let mu_at = |field: f64| {
            let mut field_opts = opts.clone();
            field_opts.external_field = Some(Vec3::new(0.0, 0.0, field));
            let shifted = run_scf(&mol, &mb, &field_opts, &functional, grid.as_ref()).unwrap();
            assert!(shifted.converged);
            dipole_moment(&mol, &mb, &shifted).z
        };
        // Eliminate the linear term using two field magnitudes, then
        // Richardson-extrapolate the cubic coefficient at a second scale.
        let cubic = |h: f64| {
            ((mu_at(2.0 * h) - mu_at(-2.0 * h)) - 2.0 * (mu_at(h) - mu_at(-h))) / (12.0 * h * h * h)
        };
        let fd = (4.0 * cubic(0.005) - cubic(0.01)) / 3.0;
        assert!(
            (fd - c[3][2]).abs() < 2e-4 * (1.0 + c[3][2].abs()),
            "{name}: cubic analytic {} vs extrapolated {fd}",
            c[3][2]
        );
    }
}

#[test]
fn biased_reference_has_nonzero_quadratic_response_and_higher_order_projectors() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = ScfOptions {
        max_iter: 250,
        e_tol: 1e-13,
        p_tol: 1e-12,
        external_field: Some(Vec3::new(0.0, 0.0, 0.03)),
        ..Default::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let report = electric_field_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        [0.0, 0.0, 1.0],
        6,
        &Default::default(),
    )
    .unwrap();
    assert!(report.value.dipole_coefficients[2][2].abs() > 1e-3);
    let mixed = dft_rs::response::mixed_electric_field_response(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &[[0.0, 0.0, 1.0], [0.2, -0.1, 0.5]],
        &Default::default(),
    )
    .unwrap();
    assert!(
        (mixed.value.dipole_derivative[2] - report.value.dipole_coefficients[2][2]).abs() < 1e-9
    );
    assert!(report.value.projector_defects.iter().all(|x| *x < 1e-8));
    let h: f64 = 0.001;
    let predicted: f64 = report
        .value
        .dipole_coefficients
        .iter()
        .enumerate()
        .map(|(n, c)| c[2] * h.powi(n as i32))
        .sum();
    let mut shifted = opts.clone();
    shifted.external_field = Some(Vec3::new(0.0, 0.0, 0.031));
    let shifted = run_scf(&mol, &mb, &shifted, &functional, None).unwrap();
    assert!((predicted - dipole_moment(&mol, &mb, &shifted).z).abs() < 1e-9);
}
