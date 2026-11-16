// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Independent finite-field SCF fits of mixed electromagnetic response.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{data, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::response::magnetic_mixed::mixed_electromagnetic_energy;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::magnetic::{magnetic_field_gradient, magnetic_field_scf, MagneticFieldOptions};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

fn options() -> ScfOptions {
    ScfOptions {
        max_iter: 300,
        e_tol: 1e-14,
        p_tol: 1e-13,
        ..Default::default()
    }
}
fn field(b: [f64; 3]) -> MagneticFieldOptions {
    MagneticFieldOptions {
        field: b,
        spin_zeeman: false,
        series_order: Some(16),
        series_tolerance: 1e-14,
    }
}
fn diatomic(z: u8, r: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, r),
        },
    ])
}

#[test]
fn mixed_third_and_fourth_electromagnetic_energy_has_independent_four_step_fits() {
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let zero = [0.0; 3];
    let e = [0.0, 0.0, 0.2];
    let b = [0.6, 0.0, 0.0];
    let steps = [0.24_f64, 0.18, 0.12, 0.06];
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    for (z, r, order) in [(1, 1.4, 4), (3, 3.0, 3)] {
        let molecule = diatomic(z, r);
        let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
        let opts = options();
        let reference = run_scf(&molecule, &basis, &opts, &functional, None).unwrap();
        let (es, bs) = if order == 4 {
            (vec![e, e, zero, zero], vec![zero, zero, b, b])
        } else {
            (vec![e, zero, zero], vec![zero, b, b])
        };
        let actual = mixed_electromagnetic_energy(
            &molecule,
            &basis,
            &functional,
            None,
            &reference,
            &opts,
            Vec3::zero(),
            true,
            &es,
            &bs,
            &config,
        )
        .unwrap();
        assert!(actual.derivative.abs() > 1e-8);
        let mut estimates = Vec::new();
        for h in steps {
            let mut energies = [[0.0; 3]; 3];
            for (i, a) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
                for (j, c) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
                    let opts = ScfOptions {
                        external_field: Some(Vec3::new(e[0] * a * h, e[1] * a * h, e[2] * a * h)),
                        ..options()
                    };
                    let state = magnetic_field_scf(
                        &molecule,
                        &basis,
                        &functional,
                        None,
                        &reference,
                        &opts,
                        &field(b.map(|x| x * c * h)),
                    )
                    .unwrap();
                    assert!(state.converged);
                    energies[i][j] = state.energy;
                }
            }
            let ew = if order == 4 {
                [1.0, -2.0, 1.0]
            } else {
                [-0.5, 0.0, 0.5]
            };
            let bw = [1.0, -2.0, 1.0];
            let mut estimate = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    estimate += ew[i] * bw[j] * energies[i][j];
                }
            }
            estimates.push(estimate / h.powi(order));
        }
        let errors: Vec<_> = estimates
            .iter()
            .map(|x| (x - actual.derivative).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - actual.derivative).abs();
        println!("joint electromagnetic z={z} order={order} target={:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e} residual={:.12e} metric={:.12e} stationarity={:.12e}",actual.derivative,fit.exponent,fit.r_squared,actual.response_residual,actual.projector_defect,actual.stationarity_defect);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999 && rich < 1e-6);
    }
}

#[test]
fn joint_field_gradient_and_charged_translation_have_four_step_fits() {
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let electric = Vec3::new(0.01, -0.015, 0.02);
    let magnetic = field([0.1, 0.05, 0.08]);
    let ion = diatomic(1, 1.4).with_charge(1).with_multiplicity(2);
    let water = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.43, 1.1),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.43, 1.1),
        },
    ]);
    for (case, molecule) in [ion, water].into_iter().enumerate() {
        let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
        let opts = ScfOptions {
            external_field: Some(electric),
            ..options()
        };
        let seed = run_scf(&molecule, &basis, &opts, &functional, None).unwrap();
        let zero = magnetic_field_scf(
            &molecule,
            &basis,
            &functional,
            None,
            &seed,
            &opts,
            &field([0.0; 3]),
        )
        .unwrap();
        assert!((zero.energy - seed.energy).abs() < 1e-10);
        let state = magnetic_field_scf(
            &molecule,
            &basis,
            &functional,
            None,
            &seed,
            &opts,
            &magnetic,
        )
        .unwrap();
        let gradient =
            magnetic_field_gradient(&molecule, &basis, &functional, None, &magnetic, &state)
                .unwrap();
        for axis in 0..3 {
            let sum: f64 = gradient.iter().map(|g| g[axis]).sum();
            assert!((sum + molecule.charge as f64 * electric.get(axis)).abs() < 1e-10);
        }
        let steps = [0.08, 0.04, 0.02, 0.01];
        let target = gradient[1][2];
        let energy = |h: f64| {
            let mut moved = molecule.clone();
            moved.atoms[1].position.z += h;
            let basis = MolBasis::build(&moved, &data::load("sto-3g").unwrap()).unwrap();
            let reference = run_scf(&moved, &basis, &opts, &functional, None).unwrap();
            magnetic_field_scf(
                &moved,
                &basis,
                &functional,
                None,
                &reference,
                &opts,
                &magnetic,
            )
            .unwrap()
            .energy
        };
        let estimates: Vec<_> = steps
            .iter()
            .map(|&h| (energy(h) - energy(-h)) / (2.0 * h))
            .collect();
        let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
        println!("joint field gradient case={case} target={target:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999 && rich < 1e-7);
        let translation = Vec3::new(0.3, -0.2, 0.4);
        let mut moved = molecule.clone();
        for a in &mut moved.atoms {
            a.position += translation;
        }
        let basis = MolBasis::build(&moved, &data::load("sto-3g").unwrap()).unwrap();
        let seed = run_scf(&moved, &basis, &opts, &functional, None).unwrap();
        let translated =
            magnetic_field_scf(&moved, &basis, &functional, None, &seed, &opts, &magnetic).unwrap();
        assert!(
            (translated.energy - state.energy + molecule.charge as f64 * electric.dot(translation))
                .abs()
                < 1e-10
        );
    }
}
