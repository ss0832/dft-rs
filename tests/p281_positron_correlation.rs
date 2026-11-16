// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent energy/potential/kernel oracles of the dilute-positron model.
//! Four-step fits cover both kernel rows in two independent density directions.
//! Intrinsically linear partner-density entries have additional exact checks;
//! this local acceptance does not establish molecular derivative tensors.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::multicomponent::annihilation::Enhancement;
use dft_rs::multicomponent::correlation::{InterspeciesCorrelation, PositronRelaxation};

fn independent_potentials(n: f64, p: f64) -> [f64; 2] {
    let r = (3.0 / (4.0 * std::f64::consts::PI * n)).cbrt();
    let a = -0.260361 / r - 0.261762 + 0.00375534 * r - 0.262005 * 0.0270912 * r * r;
    let da = 0.260361 / r.powi(2) + 0.00375534 - 2.0 * 0.262005 * 0.0270912 * r;
    let b = 1.0 + 0.113718 * r + 0.0270912 * r * r;
    let db = 0.113718 + 2.0 * 0.0270912 * r;
    [p * (da * b - a * db) / b.powi(2) * -r / (3.0 * n), a / b]
}

#[test]
fn all_local_terms_match_independent_high_precision_oracles_including_dilute_tails() {
    let model = PositronRelaxation::drummond_2011();
    for row in include_str!("oracle_data/positron_relaxation.csv")
        .lines()
        .filter(|r| !r.starts_with('#'))
        .skip(1)
    {
        let r: Vec<f64> = row.split(',').map(|v| v.parse().unwrap()).collect();
        let point = model.evaluate(r[0], r[1]).unwrap();
        let ee = model.kernel_action(r[0], r[1], [1.0, 0.0]).unwrap();
        let ep = model.kernel_action(r[0], r[1], [0.0, 1.0]).unwrap();
        for (label, value, expected) in [
            ("energy", point.energy, r[2]),
            ("electron potential", point.electron_potential, r[3]),
            ("partner potential", point.partner_potential, r[4]),
            ("electron/electron kernel", ee[0], r[5]),
            ("electron/partner kernel", ep[0], r[6]),
            ("partner/electron kernel", ee[1], r[6]),
        ] {
            let relative = ((value - expected) / expected).abs();
            println!("positron density={:.4e} {label} value={value:.16e} oracle={expected:.16e} relative={relative:.12e}", r[0]);
            assert!(value.is_finite() && relative < 2e-12);
        }
        assert_eq!(ep[1], 0.0);
        let tangent = [0.17 * r[0], -0.21 * r[1]];
        let action = model.kernel_action(r[0], r[1], tangent).unwrap();
        let expected = [r[5] * tangent[0] + r[6] * tangent[1], r[6] * tangent[0]];
        for (v, e) in action.iter().zip(expected) {
            assert!(v.is_finite() && ((v - e) / e).abs() < 3e-12);
        }
    }
}

#[test]
fn every_local_kernel_row_has_independent_four_step_fits() {
    let model = PositronRelaxation::drummond_2011();
    let mut failures = Vec::new();
    for (n, p) in [(0.02, 0.7), (0.2, 0.03), (1.3, 2.1), (5.0, 0.5)] {
        // Independent non-axis directions recover the entire symmetric
        // two-density kernel, including its exactly zero partner diagonal.
        for direction in [[0.17 * n, -0.21 * p], [-0.31 * n, 0.19 * p]] {
            let analytic = model.kernel_action(n, p, direction).unwrap();
            let names = (0..2)
                .map(|row| format!("positron kernel n={n} p={p} direction={direction:?} row={row}"))
                .collect::<Vec<_>>();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    (0..2)
                        .map(|row| {
                            std::array::from_fn(|i| {
                                let h = steps[i];
                                let at = |t: f64| {
                                    independent_potentials(
                                        n + t * direction[0],
                                        p + t * direction[1],
                                    )[row]
                                };
                                let estimate =
                                    stencil.first(at(h), at(-h), at(2.0 * h), at(-2.0 * h), h);
                                (estimate - analytic[row]).abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("all local positron kernel rows", &failures);
}

#[test]
fn enhancement_values_and_physical_density_limits_match_independent_forms() {
    let model = PositronRelaxation::drummond_2011();
    for r in [0.1_f64, 0.5, 1.0, 2.0, 5.0, 8.0] {
        let x = r.cbrt();
        let expected = 1.0 + 1.23 * r - 3.38208 * r * r.sqrt() + 8.6957 * r * r
            - 7.37037 * x.powi(7)
            + 1.75648 * x.powi(8)
            + 0.173694 * r.powi(3);
        let actual = Enhancement::Drummond2011.factor(r);
        println!(
            "positron enhancement rs={r} value={actual:.16e} defect={:.12e}",
            (actual - expected).abs()
        );
        assert!((actual - expected).abs() < 1e-12 * expected.abs());
        assert_eq!(Enhancement::IndependentParticle.factor(r), 1.0);
    }
    let dilute = model.evaluate(1e-300, 0.7).unwrap();
    assert!((dilute.partner_potential + 0.262005).abs() < 1e-15);
    let absent = model.evaluate(0.2, 0.0).unwrap();
    assert_eq!(absent.energy, 0.0);
    assert_eq!(absent.electron_potential, 0.0);
    assert!(model.evaluate(-1.0, 0.2).is_err());
    assert!(model.kernel_action(0.2, 0.7, [f64::NAN, 0.0]).is_err());
}

#[test]
fn annihilation_matches_closed_gaussian_integrals_and_zero_gradient_correction_limit() {
    use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
    use dft_rs::grid::{GridOptions, MolecularGrid};
    use dft_rs::linalg::Matrix;
    use dft_rs::math::Vec3;
    use dft_rs::multicomponent::{annihilation::annihilation, gaussian_particle_basis};

    let method = MolecularMethod {
        xc: Some("hf".into()),
        dispersion: "none".into(),
        ..Default::default()
    };
    let setup = prepare_molecular(&[1, 1], &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], &method).unwrap();
    let (a, b) = (1.2_f64, 0.7_f64);
    let center = Vec3::new(0.13, -0.24, 0.31);
    let partner = Vec3::new(-0.2, 0.15, -0.13);
    let electron_basis = gaussian_particle_basis(center, &[a], 0).unwrap();
    let partner_basis = gaussian_particle_basis(partner, &[b], 0).unwrap();
    let density = Matrix::from_row_major(1, 1, vec![1.0]);
    let spin = [density.clone(), Matrix::zeros(1, 1)];
    let grid = MolecularGrid::build(
        &setup.molecule,
        &GridOptions {
            radial_override: Some(120),
            angular_override: Some(65),
            prune: false,
            weight_cutoff: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let pi = std::f64::consts::PI;
    // Integral of n_+(r)*n_-(r)^power, from Gaussian completion of the square.
    let moment = |power: f64| {
        let exponent = 2.0 * (power * a + b);
        (2.0 * a / pi).powf(1.5 * power)
            * (2.0 * b / pi).powf(1.5)
            * (pi / exponent).powf(1.5)
            * (-4.0 * power * a * b / exponent * (center - partner).norm2()).exp()
    };
    let sphere = 3.0 / (4.0 * pi);
    for (model, powers) in [
        (Enhancement::IndependentParticle, vec![]),
        (
            Enhancement::Drummond2011,
            vec![
                (1.0, 1.23),
                (1.5, -3.38208),
                (2.0, 8.6957),
                (7.0 / 3.0, -7.37037),
                (8.0 / 3.0, 1.75648),
                (3.0, 0.173694),
            ],
        ),
        (
            Enhancement::Barbiellini { p: -0.0742 },
            vec![(1.0, 1.23), (2.0, -0.0742), (3.0, 1.0 / 3.0)],
        ),
        (
            Enhancement::Barbiellini { p: -0.1375 },
            vec![(1.0, 1.23), (2.0, -0.1375), (3.0, 1.0 / 3.0)],
        ),
        (
            Enhancement::Barbiellini { p: -0.22 },
            vec![(1.0, 1.23), (2.0, -0.22), (3.0, 1.0 / 3.0)],
        ),
    ] {
        let expected = moment(1.0)
            + powers
                .iter()
                .map(|(power, coefficient)| {
                    coefficient * sphere.powf(power / 3.0) * moment(1.0 - power / 3.0)
                })
                .sum::<f64>();
        let actual = annihilation(
            &electron_basis,
            &spin,
            &partner_basis,
            &density,
            &grid,
            model,
            None,
        )
        .unwrap();
        let zero = annihilation(
            &electron_basis,
            &spin,
            &partner_basis,
            &density,
            &grid,
            model,
            Some(0.0),
        )
        .unwrap();
        println!(
            "Gaussian annihilation {model:?} contact defect={:.12e} enhanced defect={:.12e}",
            (actual.contact_density - moment(1.0)).abs(),
            (actual.enhanced_contact_density - expected).abs()
        );
        assert!((actual.contact_density - moment(1.0)).abs() < 2e-10);
        assert!((actual.enhanced_contact_density - expected).abs() < 2e-10);
        assert_eq!(
            actual.enhanced_contact_density,
            zero.enhanced_contact_density
        );
        let rate = pi * dft_rs::constants::FINE_STRUCTURE.powi(3) * expected;
        assert!(
            (actual.rate - rate).abs() < 2e-10 * pi * dft_rs::constants::FINE_STRUCTURE.powi(3)
        );
    }
    // One independent far-tail quadrature point reproduces the former
    // 0*infinity in the alpha=0 correction. No physical density is floored.
    let tail = MolecularGrid {
        points: vec![Vec3::new(16.0, 0.0, 0.0)],
        weights: vec![1.0],
        raw_weights: vec![1.0],
        atom_of: vec![0],
    };
    for alpha in [None, Some(0.0), Some(0.22)] {
        let rate = annihilation(
            &electron_basis,
            &spin,
            &partner_basis,
            &density,
            &tail,
            Enhancement::Drummond2011,
            alpha,
        )
        .unwrap();
        assert!(rate.enhanced_contact_density.is_finite() && rate.rate.is_finite());
    }
}
