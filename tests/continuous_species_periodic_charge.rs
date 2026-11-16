// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Quadratic charge identities have zero truncation error: no fabricated p2 fit.
pub use dft_rs::{driver, linalg, math, periodic, DftError, Result};
#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "../src/driver/continuous_species/periodic_charge.rs"]
pub mod periodic_charge;
use driver::continuous_species::{ChargeChart, ChargeParameter};
use math::Vec3;
use periodic::{ewald::EwaldParams, Lattice};
use periodic_charge::NeutralEwaldChart;

fn setup(translation: Vec3, permutation: &[usize]) -> NeutralEwaldChart {
    let lattice = Lattice::new(
        Vec3::new(7.3, 0.21, -0.13),
        Vec3::new(0.17, 8.1, 0.31),
        Vec3::new(-0.23, 0.19, 9.2),
    );
    let positions = [
        Vec3::new(0.17, -0.24, 0.38),
        Vec3::new(1.41, 0.13, -0.08),
        Vec3::new(-0.31, 2.87, 0.79),
        Vec3::new(0.29, 3.39, 2.13),
    ];
    let reference = [1.2, 0.8, 1.1, 0.9];
    let directions = [[0.7, -0.2, -0.1, -0.4], [0.11, 0.27, -0.19, -0.19]];
    let parameters = directions
        .iter()
        .enumerate()
        .map(|(i, b)| ChargeParameter {
            name: format!("q{i}"),
            unit: "dimensionless".into(),
            scale: 1.0,
            bounds: [-0.5, 0.5],
            charge_direction: permutation.iter().map(|&a| b[a]).collect(),
        })
        .collect();
    let params = EwaldParams::for_accuracy(&lattice, 4, 1e-13);
    NeutralEwaldChart::new(
        lattice,
        permutation
            .iter()
            .map(|&a| positions[a] + translation)
            .collect(),
        vec![1; 4],
        4,
        ChargeChart {
            reference_charges: permutation.iter().map(|&a| reference[a]).collect(),
            parameters,
            conserve_total_charge: true,
        },
        params,
    )
    .unwrap()
}

#[test]
fn all_gradient_and_hessian_entries_obey_four_step_quadratic_identities() {
    let model = setup(Vec3::zero(), &[0, 1, 2, 3]);
    let q = [0.29, -0.17];
    let base = model.evaluate(&q).unwrap();
    let steps = [0.08, 0.04, 0.02, 0.01];
    let mut failures = Vec::new();
    assert_eq!(model.scope(), "classical-neutral-fixed-cell-ewald-only");
    assert_eq!(model.atomic_numbers(), &[1, 1, 1, 1]);
    assert_eq!(model.fixed_electrons(), 4);
    assert!(base
        .physical_charges
        .iter()
        .any(|z| (*z - 1.0).abs() > 0.01));
    assert!(base.physical_total_charge.abs() < 1e-14);
    assert!((base.energy - model.direct_energy(&q).unwrap()).abs() < 2e-13);
    for a in 0..2 {
        let mut gradient_errors = [0.0; 4];
        let mut hessian_errors = [[0.0; 4]; 2];
        for (k, &h) in steps.iter().enumerate() {
            let mut plus = q;
            plus[a] += h;
            let mut minus = q;
            minus[a] -= h;
            let ep = model.evaluate(&plus).unwrap();
            let em = model.evaluate(&minus).unwrap();
            // Direct lattice-sum oracle; quadratic central difference is exact
            // in real arithmetic. Log every measured roundoff error unchanged.
            let fd = (model.direct_energy(&plus).unwrap() - model.direct_energy(&minus).unwrap())
                / (2.0 * h);
            gradient_errors[k] = (fd - base.parameter_gradient[a]).abs();
            if gradient_errors[k] > 1e-11 {
                failures.push(format!("Eq[{a}],h={h}: {}", gradient_errors[k]));
            }
            for b in 0..2 {
                let fd = (ep.parameter_gradient[b] - em.parameter_gradient[b]) / (2.0 * h);
                hessian_errors[b][k] = (fd - base.parameter_hessian[(b, a)]).abs();
                if hessian_errors[b][k] > 1e-11 {
                    failures.push(format!("Eqq[{b},{a}],h={h}: {}", hessian_errors[b][k]));
                }
                assert_eq!(ep.parameter_hessian[(b, a)], base.parameter_hessian[(b, a)]);
                assert_eq!(em.parameter_hessian[(b, a)], base.parameter_hessian[(b, a)]);
            }
            let taylor = base.energy
                + h * base.parameter_gradient[a]
                + 0.5 * h * h * base.parameter_hessian[(a, a)];
            assert!((ep.energy - taylor).abs() < 2e-14);
        }
        println!("structural quadratic Eq[{a}] steps={steps:?} raw_errors={gradient_errors:?}; no truncation-order fit applicable");
        for (b, errors) in hessian_errors.iter().enumerate() {
            println!("structural constant Eqq[{b},{a}] steps={steps:?} raw_errors={errors:?}; no truncation-order fit applicable");
        }
    }
    components::assert_no_failures(
        "all quadratic structural components at four steps",
        &failures,
    );
    let shifted = setup(Vec3::new(0.47, -0.31, 0.29), &[0, 1, 2, 3])
        .evaluate(&q)
        .unwrap();
    let permuted = setup(Vec3::zero(), &[2, 0, 3, 1]).evaluate(&q).unwrap();
    for test in [shifted, permuted] {
        assert!((test.energy - base.energy).abs() < 2e-13);
        for a in 0..2 {
            assert!((test.parameter_gradient[a] - base.parameter_gradient[a]).abs() < 2e-13);
            for b in 0..2 {
                assert!(
                    (test.parameter_hessian[(a, b)] - base.parameter_hessian[(a, b)]).abs() < 2e-13
                );
            }
        }
    }
    println!("Ewald classical E={:.15} direct_defect={:.9e} matrix_asymmetry={:.9e} gradient={:?} Hessian={:?}",base.energy,(base.energy-model.direct_energy(&q).unwrap()).abs(),base.raw_matrix_asymmetry,base.parameter_gradient,base.parameter_hessian.as_slice());
}

#[test]
fn refuses_nonneutral_invalid_or_colliding_charts() {
    let valid = setup(Vec3::zero(), &[0, 1, 2, 3]);
    assert!(valid.evaluate(&[0.8, 0.0]).is_err());
    let lattice = Lattice::cubic(7.0);
    let params = EwaldParams::for_accuracy(&lattice, 2, 1e-12);
    let chart = ChargeChart {
        reference_charges: vec![1.0, 1.0],
        parameters: vec![ChargeParameter {
            name: "q".into(),
            unit: "dimensionless".into(),
            scale: 1.0,
            bounds: [-0.2, 0.2],
            charge_direction: vec![1.0, -1.0],
        }],
        conserve_total_charge: true,
    };
    let positions = vec![Vec3::zero(), Vec3::new(0.7, 0.1, 0.2)];
    assert!(NeutralEwaldChart::new(
        lattice.clone(),
        positions.clone(),
        vec![1, 1],
        3,
        chart.clone(),
        params
    )
    .is_err());
    assert!(NeutralEwaldChart::new(
        lattice.clone(),
        vec![Vec3::zero(), lattice.a[0]],
        vec![1, 1],
        2,
        chart.clone(),
        params
    )
    .is_err());
    let mut bad = chart.clone();
    bad.parameters[0].charge_direction[1] = 0.0;
    assert!(NeutralEwaldChart::new(
        lattice.clone(),
        positions.clone(),
        vec![1, 1],
        2,
        bad,
        params
    )
    .is_err());
    let mut bad = chart.clone();
    bad.parameters[0].bounds = [-2.0, 2.0];
    assert!(NeutralEwaldChart::new(
        lattice.clone(),
        positions.clone(),
        vec![1, 1],
        2,
        bad,
        params
    )
    .is_err());
    assert!(NeutralEwaldChart::new(
        lattice,
        positions,
        vec![0, 1],
        2,
        chart,
        EwaldParams {
            alpha: f64::NAN,
            ..params
        }
    )
    .is_err());
}
