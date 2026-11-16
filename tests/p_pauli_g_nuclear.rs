// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::properties::g_nuclear::PauliGNuclearContext;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::{run_scf, Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn fixture(displacement: &[f64]) -> (Molecule, MolBasis) {
    let positions = [
        [0.13, -0.21, 0.17],
        [2.05, 0.25, 0.38],
        [-0.65, 1.76, -0.44],
        [-1.19, -1.48, 0.85],
    ];
    let molecule = Molecule::new(
        positions
            .iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: 1,
                position: Vec3::new(
                    p[0] + displacement[3 * i],
                    p[1] + displacement[3 * i + 1],
                    p[2] + displacement[3 * i + 2],
                ),
            })
            .collect(),
    )
    .with_charge(3)
    .with_multiplicity(2);
    let mut definition = BasisSet::new("one-electron-four-centre-fixture");
    definition.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
    let basis = MolBasis::build(&molecule, &definition).unwrap();
    (molecule, basis)
}

fn options() -> ScfOptions {
    ScfOptions {
        reference: Reference::Uhf,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        ..ScfOptions::default()
    }
}

fn config() -> ResponseSolverConfig {
    ResponseSolverConfig {
        tolerance: 1e-12,
        max_iterations: 1000,
        ..ResponseSolverConfig::default()
    }
}

#[test]
fn complete_cartesian_g() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| {
            let (mol, basis) = fixture(&[0.0; 12]);
            let opts = options();
            let cfg = config();
            let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
            let scf = run_scf(&mol, &basis, &opts, &functional, None).unwrap();
            assert!(scf.converged);
            let context = PauliGNuclearContext::new(
                &mol,
                &basis,
                &functional,
                None,
                &scf,
                &opts,
                &cfg,
                Vec3::new(0.4, -0.3, 0.7),
                &[1.0; 4],
            )
            .unwrap()
            .with_orbital_time_reversal();
            assert!(matches!(
                context.cartesian_derivatives(3),
                Err(dft_rs::DftError::Unsupported { .. })
            ));
            let zero = context.cartesian_derivatives(0).unwrap();
            assert_eq!(zero.order, 0);
            assert!(zero.gradient.is_none() && zero.hessian.is_none());
            assert_eq!(zero.paths.len(), 1);
            let one = context.cartesian_derivatives(1).unwrap();
            assert!(one.hessian.is_none());
            assert_eq!(one.paths.len(), 12);
            let two = context.cartesian_derivatives(2).unwrap();
            assert_eq!(two.paths.len(), 78);
            let g: Vec<_> = two
                .gradient
                .as_ref()
                .unwrap()
                .iter()
                .map(|v| v.components())
                .collect();
            let h: Vec<Vec<_>> = two
                .hessian
                .as_ref()
                .unwrap()
                .iter()
                .map(|r| r.iter().map(|v| v.components()).collect())
                .collect();
            for (a, b) in one
                .gradient
                .unwrap()
                .iter()
                .zip(two.gradient.as_ref().unwrap())
            {
                for (x, y) in a.components().iter().zip(b.components()) {
                    assert!((x - y).abs() < 1e-15);
                }
            }
            for (a, b) in zero.value.components().iter().zip(two.value.components()) {
                assert!((a - b).abs() < 1e-15);
            }

            for i in 0..12 {
                for j in 0..12 {
                    assert_eq!(h[i][j], h[j][i]);
                }
            }
            for axis in 0..3 {
                for entry in 19..28 {
                    let sum: f64 = (0..4).map(|atom| g[3 * atom + axis][entry]).sum();
                    assert!(sum.abs() < 1e-13);
                    for j in 0..12 {
                        let sum: f64 = (0..4).map(|atom| h[3 * atom + axis][j][entry]).sum();
                        assert!(sum.abs() < 1e-13);
                    }
                }
            }
            assert!(two
                .paths
                .iter()
                .all(|p| p.real_response.converged && p.mixed_reports.len() == 3));
        })
        .unwrap();
}

#[test]
fn independent_precision_and_full_signed_curve_complement_rate_test() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| {
            let (mol, basis) = fixture(&[0.0; 12]);
            let opts = options();
            let cfg = config();
            let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
            let scf = run_scf(&mol, &basis, &opts, &functional, None).unwrap();
            assert!(scf.converged);
            let context = PauliGNuclearContext::new(
                &mol,
                &basis,
                &functional,
                None,
                &scf,
                &opts,
                &cfg,
                Vec3::new(0.4, -0.3, 0.7),
                &[1.0; 4],
            )
            .unwrap()
            .with_orbital_time_reversal();
            let direction_bits = [
                13819891624278298871,
                13824923642368853785,
                13826311909087494985,
                13798384866866070496,
                4600697522295609352,
                4582192719547487031,
                4596028765686908401,
                4586441737124806021,
                4597771164052910095,
                13822443838791250796,
                4600249379071200418,
                4593747035949598484,
            ];
            let direction: Vec<_> = direction_bits
                .chunks_exact(3)
                .map(|v| {
                    [
                        f64::from_bits(v[0]),
                        f64::from_bits(v[1]),
                        f64::from_bits(v[2]),
                    ]
                })
                .collect();
            let path = context.nuclear_path(&direction, 2).unwrap();
            assert!(path.real_response.converged && path.mixed_reports.len() == 3);
            let actual = 2.0 * path.tensor[2][1][1];

            // Independent 100-digit Gaussian/eigenpair derivative and complete
            // scalar central differences at all four original steps. The rate
            // fit remains failed: this separate check tests formula precision.
            let reference = -1.4886554737533363e-6_f64;
            // gamma_m uses m=2*n_AO^3*(nuclear_order+1)^2. The fixed atomic-unit
            // overlap/eigenpair/magnetic border condition maximum is 11.627754.
            // This is an engineering precision envelope, not a certified bound.
            let m = 1152.0;
            let gamma = m * f64::EPSILON / (1.0 - m * f64::EPSILON);
            let kappa = 11.627753639537876;
            let natural_scale = 7.2973525693e-3_f64.powi(2) * (2.0 * 0.7);
            let allowance = gamma * kappa * (reference.abs() + natural_scale);
            assert!((actual - reference).abs() <= allowance);

            let steps = [0.04, 0.02, 0.01, 0.005];
            let scalar_differences = [
                -1.4886541660376773e-6,
                -1.488654973882171e-6,
                -1.4886553379695634e-6,
                -1.488655439131826e-6,
            ];
            let predicted_errors = [
                1.3077156590347592e-12,
                4.998711652816559e-13,
                1.357837729743668e-13,
                3.4621510355648695e-14,
            ];
            let ulp = |v: f64| f64::from_bits(v.abs().to_bits() + 1) - v.abs();
            for ((h, sample), prediction) in steps
                .into_iter()
                .zip(scalar_differences)
                .zip(predicted_errors)
            {
                let budget = allowance
                    + 2.0 * (ulp(sample) + ulp(actual))
                    + 4.0 * 5.672117367325826e-55 / (h * h);
                assert!(((sample - actual) - prediction).abs() <= budget);
            }
        })
        .unwrap();
}
