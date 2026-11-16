// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::complex::Complex;
use dft_rs::response::dynamic::dynamic_amplitudes;
use dft_rs::response::tddft::{TdaSolver, TdaSpin};
use dft_rs::response::{dynamic_polarizability, Frequency, ResponseSolverConfig};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn h2() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ])
}

#[test]
fn dynamic_zero_frequency_matches_independent_static_cpks() {
    let mol = h2();
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
                        level: 4,
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
        };
        let opts = ScfOptions {
            max_iter: 250,
            e_tol: 1e-12,
            p_tol: 1e-10,
            eri_mode: EriMode::InCore { limit_mb: 128 },
            ..Default::default()
        };
        let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).unwrap();
        let static_alpha = dft_rs::response::polarizability::polarizability(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
        )
        .unwrap();
        let report = dynamic_polarizability(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            Frequency::Static,
            &Default::default(),
        )
        .unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (static_alpha[a][b] - report.value.alpha[a][b].re).abs() < 1e-8,
                    "{name} ({a},{b})"
                );
                assert!(report.value.alpha[a][b].im.abs() < 1e-12);
            }
        }
    }
}

#[test]
fn retarded_response_is_causal_and_has_the_correct_imaginary_axis_limit() {
    let mol = h2();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        ..Default::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let calculate = |frequency| {
        dynamic_polarizability(
            &mol,
            &mb,
            &functional,
            None,
            &scf,
            &opts,
            frequency,
            &ResponseSolverConfig::default(),
        )
        .unwrap()
    };
    let plus = calculate(Frequency::Retarded {
        omega: 0.2,
        damping: 0.03,
    });
    let minus = calculate(Frequency::Retarded {
        omega: -0.2,
        damping: 0.03,
    });
    let static_result = calculate(Frequency::Static);
    let imaginary = calculate(Frequency::Imaginary { frequency: 0.2 });
    for a in 0..3 {
        for b in 0..3 {
            assert!((plus.value.alpha[a][b] - minus.value.alpha[a][b].conj()).abs() < 1e-10);
            assert!(imaginary.value.alpha[a][b].im.abs() < 1e-12);
        }
    }
    assert!(plus.value.alpha[2][2].im > 0.0);
    assert!(imaginary.value.alpha[2][2].re > 0.0);
    assert!(imaginary.value.alpha[2][2].re < static_result.value.alpha[2][2].re);
    assert!(Frequency::Retarded {
        omega: 1.0,
        damping: -0.1
    }
    .energy()
    .is_err());
}

#[test]
fn report_and_amplitude_views_match_independent_two_by_two_resolvents() {
    for charge in [0, 1] {
        let mol = h2()
            .with_charge(charge)
            .with_multiplicity(if charge == 0 { 1 } else { 2 });
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
        let opts = ScfOptions {
            e_tol: 1e-12,
            p_tol: 1e-10,
            ..Default::default()
        };
        let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
        let spin = if scf.unrestricted {
            TdaSpin::SpinResolved
        } else {
            TdaSpin::Singlet
        };
        let solver = TdaSolver::new(&mol, &mb, &functional, None, &scf, &opts, spin).unwrap();
        assert_eq!(solver.dim(), 1);
        let a_plus_b = solver.apb_vec(&[1.0]).unwrap()[0];
        let a_minus_b = solver.amb_vec(&[1.0]).unwrap()[0];
        let c = &scf.mo_coeff[0];
        let dipoles = dft_rs::integrals::dipole_matrices(&mb, Vec3::zero());
        let r: [f64; 3] =
            std::array::from_fn(|axis| c.transpose_matmul(&dipoles[axis]).matmul(c)[(1, 0)]);
        let mut worst = 0.0_f64;
        for frequency in [
            Frequency::Static,
            Frequency::Imaginary { frequency: 0.2 },
            Frequency::Retarded {
                omega: 0.2,
                damping: 0.03,
            },
            Frequency::Retarded {
                omega: -0.2,
                damping: 0.03,
            },
        ] {
            let z = frequency.energy().unwrap();
            let determinant = Complex::from(a_plus_b * a_minus_b) - z * z;
            let report = dynamic_polarizability(
                &mol,
                &mb,
                &functional,
                None,
                &scf,
                &opts,
                frequency,
                &Default::default(),
            )
            .unwrap();
            let amplitudes = dynamic_amplitudes(
                &mol,
                &mb,
                &functional,
                None,
                &scf,
                &opts,
                frequency,
                &Default::default(),
            )
            .unwrap();
            for axis in 0..3 {
                let sum = Complex::from(-2.0 * r[axis] * a_minus_b) / determinant;
                let difference = z * (-2.0 * r[axis]) / determinant;
                worst = worst
                    .max((amplitudes.sum[axis][0] - sum).abs())
                    .max((amplitudes.difference[axis][0] - difference).abs());
                for observable in 0..3 {
                    let alpha = sum * (-r[observable] * if scf.unrestricted { 1.0 } else { 2.0 });
                    worst = worst
                        .max((report.value.alpha[observable][axis] - alpha).abs())
                        .max((amplitudes.alpha[observable][axis] - alpha).abs());
                }
                assert!(report.residuals[axis] < 1e-10 && amplitudes.residuals[axis] < 1e-10);
            }
        }
        println!(
            "shared dynamic solve charge={charge}: independent dense resolvent defect={worst:.3e}"
        );
        assert!(worst < 1e-9);
    }
}

#[test]
fn direct_integer_dynamic_routes_reject_a_converged_thermal_hf_reference() {
    fn expect_fractional_refusal<T>(result: dft_rs::Result<T>) {
        match result {
            Err(dft_rs::DftError::Unsupported { feature, missing }) => {
                assert!(feature.contains("integer-occupation orbital response"));
                assert!(missing.contains("integer projector"));
            }
            Err(error) => panic!("wrong fractional-reference error: {error}"),
            Ok(_) => panic!("integer response accepted a thermal fractional projector"),
        }
    }
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 3.0),
        },
    ]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let frequency = Frequency::Retarded {
        omega: 0.03,
        damping: 0.01,
    };
    let config = ResponseSolverConfig::default();
    for temperature in [0.0, 40000.0] {
        let opts = ScfOptions {
            max_iter: 300,
            e_tol: 1e-12,
            p_tol: 1e-11,
            conv_tol_grad: 1e-10,
            electronic_temperature: temperature
                * dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K,
            eri_mode: EriMode::InCore { limit_mb: 64 },
            ..Default::default()
        };
        // Genuine supported Fermi-Dirac SCF: no population or matrix edits.
        let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
        assert!(scf.converged);
        let tda = TdaSolver::new(&mol, &mb, &functional, None, &scf, &opts, TdaSpin::Singlet);
        let amplitudes = dynamic_amplitudes(
            &mol,
            &mb,
            &functional,
            None,
            &scf,
            &opts,
            frequency,
            &config,
        );
        let report = dynamic_polarizability(
            &mol,
            &mb,
            &functional,
            None,
            &scf,
            &opts,
            frequency,
            &config,
        );
        if temperature > 0.0 {
            assert!(!scf.has_integer_occupations());
            for population in &scf.occupations {
                assert!(population.iter().all(|&f| f > 0.0 && f < 1.0));
                assert!((population.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            }
            expect_fractional_refusal(tda);
            expect_fractional_refusal(amplitudes);
            expect_fractional_refusal(report);
        } else {
            assert!(scf.has_integer_occupations());
            assert_eq!(tda.unwrap().dim(), 1);
            let amplitudes = amplitudes.unwrap();
            let report = report.unwrap();
            for a in 0..3 {
                assert!(amplitudes.residuals[a] <= config.tolerance);
                assert!(report.residuals[a] <= config.tolerance);
                for b in 0..3 {
                    assert!((amplitudes.alpha[a][b] - report.value.alpha[a][b]).abs() < 1e-12);
                }
            }
        }
    }
}
