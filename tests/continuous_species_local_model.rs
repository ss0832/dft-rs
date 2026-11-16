// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Full local-potential/steric/frame derivative acceptance on a C1 molecule.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::continuous_species::frame::AttachedFrame;
use dft_rs::driver::continuous_species::local::{LocalCenter, LocalGaussianSpec};
use dft_rs::driver::continuous_species::{
    ChargeChart, ChargeParameter, ContinuousSpeciesModel, SpeciesFunctional,
};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::{Frequency, ResponseSolverConfig};
use dft_rs::scf::{EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};

fn molecule() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.13, -0.21, 0.17),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.34, 0.82, -0.23),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.74, 1.13, 0.91),
        },
    ])
}

fn make(mol: Molecule, f: SpeciesFunctional, open: bool) -> ContinuousSpeciesModel {
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let chart = ChargeChart {
        reference_charges: vec![8., 1., 1.],
        conserve_total_charge: false,
        parameters: vec![
            ChargeParameter {
                name: "local".into(),
                unit: "dimensionless".into(),
                scale: 1.,
                bounds: [-0.5, 0.5],
                charge_direction: vec![0.; 3],
            },
            ChargeParameter {
                name: "steric".into(),
                unit: "dimensionless".into(),
                scale: 1.,
                bounds: [0., 0.5],
                charge_direction: vec![0.; 3],
            },
            ChargeParameter {
                name: "mixed_charge".into(),
                unit: "dimensionless".into(),
                scale: 1.,
                bounds: [-0.5, 0.5],
                charge_direction: vec![0.17, -0.31, 0.14],
            },
        ],
    };
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let origin = Vec3::new(0.11, 0.07, -0.09);
    let model = if open {
        ContinuousSpeciesModel::new_with_spin(mol, basis, grid, [5, 4], chart, f, options, origin)
    } else {
        ContinuousSpeciesModel::new(mol, basis, grid, 10, chart, f, options, origin)
    }
    .unwrap();
    model
        .with_local_potentials(vec![LocalGaussianSpec {
            center: LocalCenter::Attached(AttachedFrame {
                anchors: [0, 1, 2],
                offset: Vec3::new(0.27, -0.19, 0.33),
                min_anchor_distance: 1e-8,
                min_sine_angle: 1e-6,
            }),
            exponent: 0.63,
            reference_amplitude: 0.12,
            amplitude_direction: vec![1., 0., 0.03],
            reference_steric_strength: 0.18,
            steric_direction: vec![0., 1., 0.02],
            environment_exponents: vec![0.47, 0.81, 0.59],
            environment_hardness: vec![0.31, 0.73, 0.23],
            excluded_atoms: vec![true, true, false],
        }])
        .unwrap()
}

fn shifted(v: Vec3, axis: usize, delta: f64) -> Vec3 {
    let mut a = v.to_array();
    a[axis] += delta;
    Vec3::new(a[0], a[1], a[2])
}

fn acceptance(f: SpeciesFunctional, open: bool) {
    let q = [0.23, 0.07, -0.19];
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let m = make(molecule(), f, open);
    let state = m.evaluate(&q, field, None).unwrap();
    let config = ResponseSolverConfig::default();
    let response = m.sensitivity(&state, &config).unwrap();
    println!(
        "local model {} open={open} SCF={} CPKS={:?}",
        f.name(),
        state.diagnostics.orbital_residual,
        response.response_residuals
    );
    assert!(response.hessian_symmetry_defect < 1e-8);
    assert!(response.mixed_derivative_defect < 1e-8);
    // g is an exactly affine classical term. Its electronic responses and
    // Hessian row are structural zeros, checked without fitting roundoff.
    for i in 0..3 {
        assert!(response.energy_hessian[1][i].abs() < 1e-12);
        assert!(response.dipole_jacobian[i][1].abs() < 1e-12);
    }
    for h in [0.04, 0.02, 0.01, 0.005] {
        let mut p = q;
        p[1] += h;
        let mut n = q;
        n[1] -= h;
        let ps = m.evaluate(&p, field, None).unwrap();
        let ns = m.evaluate(&n, field, None).unwrap();
        assert!(ps.total_density().max_abs_diff(&ns.total_density()) < 1e-10);
        assert!(
            ((ps.energy() - ns.energy()) / (2. * h) - response.energy_gradient[1]).abs() < 1e-9
        );
    }
    let mut failures = Vec::new();
    for a in [0, 2] {
        let names: Vec<_> = std::iter::once(format!("{} open={open} local E_q[{a}]", f.name()))
            .chain((0..3).map(|i| format!("{} open={open} local mu_q[{i},{a}]", f.name())))
            .collect();
        components::fit_components(
            &names,
            &[[0.08, 0.04, 0.02, 0.01], [0.16, 0.08, 0.04, 0.02]],
            |steps| {
                let mut errors = vec![[0.; 4]; 4];
                for (k, &h) in steps.iter().enumerate() {
                    let mut p = q;
                    p[a] += h;
                    let mut n = q;
                    n[a] -= h;
                    let p = m.evaluate(&p, field, None).unwrap();
                    let n = m.evaluate(&n, field, None).unwrap();
                    errors[0][k] =
                        ((p.energy() - n.energy()) / (2. * h) - response.energy_gradient[a]).abs();
                    for i in 0..3 {
                        errors[1 + i][k] = ((p.dipole.get(i) - n.dipole.get(i)) / (2. * h)
                            - response.dipole_jacobian[i][a])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    let names: Vec<_> = [(0, 0), (0, 2), (2, 2)]
        .into_iter()
        .map(|(a, b)| format!("{} open={open} local E_qq[{a},{b}]", f.name()))
        .collect();
    components::fit_components(
        &names,
        &[[0.08, 0.04, 0.02, 0.01], [0.16, 0.08, 0.04, 0.02]],
        |steps| {
            let mut errors = vec![[0.; 4]; 3];
            for (k, &h) in steps.iter().enumerate() {
                for (c, (a, b)) in [(0, 0), (0, 2), (2, 2)].into_iter().enumerate() {
                    let energy = |da: f64, db: f64| {
                        let mut point = q;
                        point[a] += da;
                        point[b] += db;
                        m.evaluate(&point, field, None).unwrap().energy()
                    };
                    let finite = if a == b {
                        (energy(h, 0.) - 2. * state.energy() + energy(-h, 0.)) / (h * h)
                    } else {
                        (energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h))
                            / (4. * h * h)
                    };
                    errors[c][k] = (finite - response.energy_hessian[a][b]).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    for axis in 0..3 {
        let names: Vec<_> = (0..3)
            .map(|i| format!("{} open={open} local alpha[{i},{axis}]", f.name()))
            .collect();
        components::fit_components(
            &names,
            &[[0.008, 0.004, 0.002, 0.001], [0.016, 0.008, 0.004, 0.002]],
            |steps| {
                let mut errors = vec![[0.; 4]; 3];
                for (k, &h) in steps.iter().enumerate() {
                    let p = m.evaluate(&q, shifted(field, axis, h), None).unwrap();
                    let n = m.evaluate(&q, shifted(field, axis, -h), None).unwrap();
                    for i in 0..3 {
                        errors[i][k] = ((p.dipole.get(i) - n.dipole.get(i)) / (2. * h)
                            - response.polarizability[i][axis])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    let gradient = m.nuclear_gradient(&state).unwrap();
    let names: Vec<_> = (0..3)
        .flat_map(|a| {
            (0..3).map(move |i| format!("{} open={open} local nuclear[{a},{i}]", f.name()))
        })
        .collect();
    components::fit_components(
        &names,
        &[[0.02, 0.01, 0.005, 0.0025], [0.04, 0.02, 0.01, 0.005]],
        |steps| {
            let mut errors = vec![[0.; 4]; 9];
            for a in 0..3 {
                for axis in 0..3 {
                    for (k, &h) in steps.iter().enumerate() {
                        let energy = |delta| {
                            let mut mol = molecule();
                            mol.atoms[a].position = shifted(mol.atoms[a].position, axis, delta);
                            make(mol, f, open)
                                .evaluate(&q, field, None)
                                .unwrap()
                                .energy()
                        };
                        errors[3 * a + axis][k] =
                            ((energy(h) - energy(-h)) / (2. * h) - gradient[a].get(axis)).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all local model active derivative components", &failures);
    let dynamic = m
        .dynamic_polarizability(&state, Frequency::Static, &config)
        .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((dynamic.value.alpha[i][j].re - response.polarizability[i][j]).abs() < 1e-7);
            assert!(dynamic.value.alpha[i][j].im.abs() < 1e-12);
        }
    }
    let probes = [Vec3::new(2.31, -1.47, 0.86), Vec3::new(-1.83, 0.67, 2.14)];
    let (rho, esp) = m.density_and_potential_at_points(&state, &probes).unwrap();
    assert!(rho.iter().all(|x| x.is_finite() && *x > 0.));
    assert!(esp.iter().all(|x| x.is_finite()));
    assert!(m
        .density_and_potential_at_points(&state, &[molecule().atoms[0].position])
        .is_err());
}

#[test]
fn local_lda_rks() {
    acceptance(SpeciesFunctional::Lda, false);
}
#[test]
fn local_pbe_rks() {
    acceptance(SpeciesFunctional::Pbe, false);
}
#[test]
fn local_lda_uks() {
    acceptance(SpeciesFunctional::Lda, true);
}
#[test]
fn local_pbe_uks() {
    acceptance(SpeciesFunctional::Pbe, true);
}
