// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::properties::soc_zero_field_splitting as candidate;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::{run_scf, Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[test]
fn actual_moving_soc_nuclear_response() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| {
            let coordinates = [
                [0.13, -0.21, 0.17],
                [2.05, 0.25, 0.38],
                [-0.65, 1.76, -0.44],
                [-1.19, -1.48, 0.85],
            ];
            let mol = Molecule::new(
                coordinates
                    .iter()
                    .map(|p| Atom {
                        z: 1,
                        position: Vec3::new(p[0], p[1], p[2]),
                    })
                    .collect(),
            )
            .with_charge(2)
            .with_multiplicity(3);
            let mut definition = BasisSet::new("triplet-response-fixture");
            definition.insert(ElementBasis {
                z: 1,
                shells: vec![Shell {
                    l: 0,
                    exps: vec![0.7],
                    coefs: vec![1.0],
                }],
            });
            let basis = MolBasis::build(&mol, &definition).unwrap();
            let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
            let options = ScfOptions {
                reference: Reference::Uhf,
                e_tol: 1e-14,
                p_tol: 1e-13,
                conv_tol_grad: 1e-12,
                max_iter: 500,
                ..ScfOptions::default()
            };
            let config = ResponseSolverConfig {
                tolerance: 1e-12,
                max_iterations: 1000,
                ..ResponseSolverConfig::default()
            };
            let scf = run_scf(&mol, &basis, &options, &functional, None).unwrap();
            assert!(scf.converged);
            let direction = [
                [0.13, -0.21, 0.31],
                [-0.17, 0.27, 0.05],
                [0.08, -0.11, 0.19],
                [-0.04, 0.16, -0.23],
            ];
            let report = candidate::soc_nuclear_path(
                &mol,
                &basis,
                &functional,
                &scf,
                &options,
                &direction,
                2,
                &config,
            )
            .unwrap();
            assert!(report.response.converged);
            assert_eq!(report.orientation_independent_energy_coefficients.len(), 3);
            for degree in 0..=2 {
                let chi = &report.susceptibility_coefficients[degree];
                let transverse = (0..3).map(|a| chi[(3 * a, 3 * a)]).sum::<f64>();
                let scale = (dft_rs::constants::FINE_STRUCTURE.powi(2) / 4.0).powi(2);
                for a in 0..3 {
                    let energy = 0.5
                        * scale
                        * (transverse + chi[(3 * a + 2, 3 * a + 2)] - chi[(3 * a, 3 * a)]);
                    let mapped = report.orientation_independent_energy_coefficients[degree]
                        + 0.5 * report.coefficients[degree][a][a];
                    assert!((energy - mapped).abs() < 2e-26);
                }
            }
            println!(
                "SOC_NUCLEAR_DIAGNOSTICS metric={:?} density={:?} KKT={:?}",
                report.metric_error, report.density_error, report.kkt_residuals
            );
            for (degree, values) in report.coefficients.iter().enumerate() {
                println!("SOC_NUCLEAR_D degree={degree} values={values:?}");
            }
            for (degree, values) in report.susceptibility_coefficients.iter().enumerate() {
                println!(
                    "SOC_NUCLEAR_CHI degree={degree} values={:?}",
                    values.as_slice()
                );
            }
            assert!(report.metric_error < 2e-10 && report.density_error < 2e-10);
            assert!(report.kkt_residuals.iter().all(|x| *x < 2e-10));
            let shift = [[0.31, -0.17, 0.23]; 4];
            let translated = candidate::soc_nuclear_path(
                &mol,
                &basis,
                &functional,
                &scf,
                &options,
                &shift,
                2,
                &config,
            )
            .unwrap();
            let residual = translated
                .coefficients
                .iter()
                .skip(1)
                .flatten()
                .flatten()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);
            println!("SOC_NUCLEAR_TRANSLATION {residual:?}");
            assert!(residual < 1e-20);
            for order in 0..2 {
                let lower = candidate::soc_nuclear_path(
                    &mol,
                    &basis,
                    &functional,
                    &scf,
                    &options,
                    &direction,
                    order,
                    &config,
                )
                .unwrap();
                assert_eq!(lower.coefficients.len(), order + 1);
                assert!(lower.metric_error < 2e-10 && lower.density_error < 2e-10);
                for degree in 0..=order {
                    for a in 0..3 {
                        for b in 0..3 {
                            assert!(
                                (lower.coefficients[degree][a][b]
                                    - report.coefficients[degree][a][b])
                                    .abs()
                                    < 2e-22
                            );
                        }
                    }
                }
            }
            assert!(matches!(
                candidate::soc_nuclear_path(
                    &mol,
                    &basis,
                    &functional,
                    &scf,
                    &options,
                    &direction,
                    3,
                    &config
                ),
                Err(dft_rs::DftError::Unsupported { .. })
            ));
            let mut modified = options.clone();
            modified.extra_hcore = Some(dft_rs::linalg::Matrix::zeros(4, 4));
            assert!(matches!(
                candidate::soc_nuclear_path(
                    &mol,
                    &basis,
                    &functional,
                    &scf,
                    &modified,
                    &direction,
                    2,
                    &config
                ),
                Err(dft_rs::DftError::Unsupported { .. })
            ));
        })
        .unwrap();
}

#[test]
fn multiple_occupied_spin_spaces_preserve_quantum_spin_normalization() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| {
            let positions = [
                [0.13, -0.21, 0.17],
                [2.05, 0.25, 0.38],
                [-0.65, 1.76, -0.44],
                [-1.19, -1.48, 0.85],
            ];
            let mut definition = BasisSet::new("occupied-spin-response-fixture");
            definition.insert(ElementBasis {
                z: 1,
                shells: vec![Shell {
                    l: 0,
                    exps: vec![0.7],
                    coefs: vec![1.0],
                }],
            });
            let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
            let options = ScfOptions {
                reference: Reference::Uhf,
                e_tol: 1e-14,
                p_tol: 1e-13,
                conv_tol_grad: 1e-12,
                max_iter: 500,
                ..ScfOptions::default()
            };
            let config = ResponseSolverConfig {
                tolerance: 1e-12,
                max_iterations: 1000,
                ..ResponseSolverConfig::default()
            };
            let direction = [
                [0.13, -0.21, 0.31],
                [-0.17, 0.27, 0.05],
                [0.08, -0.11, 0.19],
                [-0.04, 0.16, -0.23],
            ];
            for (charge, multiplicity) in [(2, 3), (1, 4), (0, 3), (3, 2)] {
                let mol = Molecule::new(
                    positions
                        .iter()
                        .map(|p| Atom {
                            z: 1,
                            position: Vec3::new(p[0], p[1], p[2]),
                        })
                        .collect(),
                )
                .with_charge(charge)
                .with_multiplicity(multiplicity);
                let basis = MolBasis::build(&mol, &definition).unwrap();
                let scf = run_scf(&mol, &basis, &options, &functional, None).unwrap();
                assert!(scf.converged);
                let result = candidate::soc_nuclear_path(
                    &mol,
                    &basis,
                    &functional,
                    &scf,
                    &options,
                    &direction,
                    2,
                    &config,
                );
                if multiplicity == 2 {
                    assert!(matches!(result, Err(dft_rs::DftError::Unsupported { .. })));
                    continue;
                }
                let report = result.unwrap();
                let spin = 0.5 * (multiplicity - 1) as f64;
                assert_eq!(report.nominal_spin, spin);
                assert_eq!(
                    report.reference_s_squared.to_bits(),
                    scf.s_squared.to_bits()
                );
                assert!(
                    report.response.converged
                        && report.metric_error < 2e-10
                        && report.density_error < 2e-10
                );
                let scale = (dft_rs::constants::FINE_STRUCTURE.powi(2) / 4.0).powi(2);
                for degree in 0..=2 {
                    let tensor = &report.coefficients[degree];
                    let trace = tensor[0][0] + tensor[1][1] + tensor[2][2];
                    // Highest-weight spin coherent states have <S_parallel^2>=S^2
                    // and <S_transverse^2>=S/2. Compare the resulting quantum
                    // energy differences to the independently returned electronic
                    // SOC susceptibility, not to a second copy of the D prefactor.
                    for a in 0..3 {
                        for b in 0..3 {
                            let qa = 0.5 * spin * trace + (spin * spin - 0.5 * spin) * tensor[a][a];
                            let qb = 0.5 * spin * trace + (spin * spin - 0.5 * spin) * tensor[b][b];
                            let chi = &report.susceptibility_coefficients[degree];
                            let ra = chi[(3 * a + 2, 3 * a + 2)] - chi[(3 * a, 3 * a)];
                            let rb = chi[(3 * b + 2, 3 * b + 2)] - chi[(3 * b, 3 * b)];
                            assert!((qa - qb - 0.5 * scale * (ra - rb)).abs() < 2e-24);
                        }
                    }
                }
            }
        })
        .unwrap();
}

#[test]
fn actual_nh_paired_spin_cartesian_response_preserves_translation() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| {
            let factor = 2.0 / 21.0_f64.sqrt();
            let mol = Molecule::new(vec![
                Atom {
                    z: 7,
                    position: Vec3::new(0.13, -0.21, 0.17),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.13 + factor, -0.21 + 2.0 * factor, 0.17 + 4.0 * factor),
                },
            ])
            .with_multiplicity(3);
            let basis =
                MolBasis::build(&mol, &dft_rs::basis::data::load("sto-3g").unwrap()).unwrap();
            let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
            let options = ScfOptions {
                reference: Reference::Uhf,
                e_tol: 1e-14,
                p_tol: 1e-13,
                conv_tol_grad: 1e-12,
                max_iter: 500,
                ..ScfOptions::default()
            };
            let config = ResponseSolverConfig {
                tolerance: 1e-12,
                max_iterations: 1000,
                ..ResponseSolverConfig::default()
            };
            let scf = run_scf(&mol, &basis, &options, &functional, None).unwrap();
            assert!(scf.converged);
            assert_eq!((scf.n_alpha, scf.n_beta), (5, 3));
            let report = candidate::soc_nuclear_derivatives(
                &mol,
                &basis,
                &functional,
                &scf,
                &options,
                2,
                &config,
            )
            .unwrap();
            let gradient = report.gradient.as_ref().unwrap();
            let hessian = report.hessian.as_ref().unwrap();
            assert_eq!(report.paths.len(), 21);
            assert_eq!(gradient.len(), 6);
            assert_eq!(hessian.len(), 6);
            assert!(report.paths.iter().all(|p| p.nominal_spin == 1.0
                && p.reference_s_squared == scf.s_squared
                && p.response.converged));
            for axis in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        assert!((gradient[axis][a][b] + gradient[3 + axis][a][b]).abs() < 1e-20);
                        for k in 0..6 {
                            assert!(
                                (hessian[axis][k][a][b] + hessian[3 + axis][k][a][b]).abs() < 1e-20
                            );
                            assert_eq!(hessian[axis][k][a][b], hessian[k][axis][a][b]);
                        }
                    }
                }
            }
            println!(
                "NH_PAIRED_CARTESIAN value={:?} gradient={:?} hessian={:?}",
                report.value, gradient, hessian
            );
        })
        .unwrap();
}
