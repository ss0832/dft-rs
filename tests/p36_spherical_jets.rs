// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::dual::{DualN, Scalar};
use dft_rs::integrals::analytic::GaussianKernel;
use dft_rs::integrals::analytic_basis::{self as jets, AnalyticNucleus, AnalyticShell};
use dft_rs::integrals::deriv_mat::{
    kinetic_derivative_matrices, nuclear_derivative_matrices, overlap_derivative_matrices,
};
use dft_rs::integrals::{
    eri_quartet, kinetic_matrix, nuclear_matrix, overlap_matrix, CoulombKernel,
};
use dft_rs::math::Vec3;
use dft_rs::response::series::{evaluate_series_many, SeriesScalar};
use dft_rs::system::{Atom, Molecule};

fn fixture() -> (Molecule, MolBasis) {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(-0.3, 0.2, 0.1),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.8, -0.4, 1.2),
        },
    ]);
    let mut basis = BasisSet::new("analytic-test");
    basis.insert(ElementBasis {
        z: 1,
        shells: vec![
            Shell {
                l: 1,
                exps: vec![0.7, 0.3],
                coefs: vec![0.4, 0.6],
            },
            Shell {
                l: 2,
                exps: vec![0.6, 0.25],
                coefs: vec![0.5, 0.3],
            },
        ],
    });
    let mb = MolBasis::build(&mol, &basis).unwrap();
    (mol, mb)
}

#[test]
fn spherical_dipole_jets_match_native_operators_and_translation_identity() {
    type D = DualN<f64, 1>;
    let (mol, mb) = fixture();
    let origin = Vec3::new(0.2, -0.3, 0.4);
    let native = dft_rs::integrals::dipole_matrices(&mb, origin);
    let derivative = dft_rs::integrals::deriv_mat::dipole_derivative_matrices(&mb, mol.len());
    let shifted: Vec<_> = mb
        .shells
        .iter()
        .map(|s| {
            let mut center = [
                D::from_f64(s.center.x),
                D::from_f64(s.center.y),
                D::from_f64(s.center.z),
            ];
            if s.atom == 0 {
                center[0].im[0] = 1.0;
            }
            AnalyticShell::from_placed(s, center).unwrap()
        })
        .collect();
    for (ia, a) in shifted.iter().enumerate() {
        for (ib, b) in shifted.iter().enumerate() {
            for axis in 0..3 {
                let block = jets::dipole(
                    a,
                    b,
                    axis,
                    [
                        D::from_f64(origin.x),
                        D::from_f64(origin.y),
                        D::from_f64(origin.z),
                    ],
                )
                .unwrap();
                let ni = 2 * mb.shells[ia].l as usize + 1;
                let nj = 2 * mb.shells[ib].l as usize + 1;
                for i in 0..ni {
                    for j in 0..nj {
                        let ij = (mb.shells[ia].sph_off + i, mb.shells[ib].sph_off + j);
                        let v = block.values[i * nj + j];
                        assert!((v.re - native[axis][ij]).abs() < 3e-12);
                        // Native derivatives use origin zero. Correct their moving
                        // metric explicitly rather than changing that convention.
                        let overlap = jets::overlap(a, b).unwrap().values[i * nj + j];
                        let expected = derivative[axis][0][0][ij]
                            - [origin.x, origin.y, origin.z][axis] * overlap.im[0];
                        assert!((v.im[0] - expected).abs() < 4e-11);
                    }
                }
            }
        }
    }
    let translation = [0.4, -0.2, 0.3];
    let rigid: Vec<_> = mb
        .shells
        .iter()
        .map(|s| {
            let center = std::array::from_fn(|i| {
                let mut x = D::from_f64([s.center.x, s.center.y, s.center.z][i]);
                x.im[0] = translation[i];
                x
            });
            AnalyticShell::from_placed(s, center).unwrap()
        })
        .collect();
    for a in &rigid {
        for b in &rigid {
            let overlap = jets::overlap(a, b).unwrap();
            for axis in 0..3 {
                let dipole = jets::dipole(a, b, axis, [D::from_f64(0.0); 3]).unwrap();
                for (mu, s) in dipole.values.iter().zip(&overlap.values) {
                    assert!(s.im[0].abs() < 1e-12);
                    assert!((mu.im[0] - translation[axis] * s.re).abs() < 3e-12);
                }
            }
        }
    }
}

#[test]
fn spherical_integral_values_and_nuclear_derivatives_match_existing_engines() {
    type D = DualN<f64, 1>;
    let (mol, mb) = fixture();
    let shells: Vec<_> = mb
        .shells
        .iter()
        .map(|s| {
            let mut center = [
                D::from_f64(s.center.x),
                D::from_f64(s.center.y),
                D::from_f64(s.center.z),
            ];
            if s.atom == 0 {
                center[0].im[0] = 1.0;
            }
            AnalyticShell::from_placed(s, center).unwrap()
        })
        .collect();
    let nuclei: Vec<_> = mol
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut center = [
                D::from_f64(a.position.x),
                D::from_f64(a.position.y),
                D::from_f64(a.position.z),
            ];
            if i == 0 {
                center[0].im[0] = 1.0;
            }
            AnalyticNucleus {
                center,
                charge: D::from_f64(a.z as f64),
            }
        })
        .collect();
    let a = &shells[1];
    let b = &shells[2];
    let blocks = [
        jets::overlap(a, b).unwrap(),
        jets::kinetic(a, b).unwrap(),
        jets::attraction(a, b, &nuclei, GaussianKernel::Coulomb, 1).unwrap(),
    ];
    let values = [
        overlap_matrix(&mb),
        kinetic_matrix(&mb),
        nuclear_matrix(&mb, &mol, None),
    ];
    let derivatives = [
        overlap_derivative_matrices(&mb, 2),
        kinetic_derivative_matrices(&mb, 2),
        nuclear_derivative_matrices(&mb, &mol, &[1.0, 1.0]),
    ];
    for operator in 0..3 {
        for i in 0..5 {
            for j in 0..3 {
                let ij = (mb.shells[1].sph_off + i, mb.shells[2].sph_off + j);
                let v = blocks[operator].values[i * 3 + j];
                assert!((v.re - values[operator][ij]).abs() < 2e-12);
                assert!((v.im[0] - derivatives[operator][0][0][ij]).abs() < 2e-11);
            }
        }
    }
    let real: Vec<_> = mb
        .shells
        .iter()
        .map(|s| AnalyticShell::from_placed(s, [s.center.x, s.center.y, s.center.z]).unwrap())
        .collect();
    for (kernel, native) in [
        (GaussianKernel::Coulomb, CoulombKernel::Coulomb),
        (
            GaussianKernel::LongRange { omega: 0.4 },
            CoulombKernel::Erf { omega: 0.4 },
        ),
        (
            GaussianKernel::ShortRange { omega: 0.4 },
            CoulombKernel::Erfc { omega: 0.4 },
        ),
    ] {
        let v = jets::repulsion(&real[1], &real[2], &real[0], &real[2], kernel, 0).unwrap();
        let reference = eri_quartet(
            &mb.shells[1],
            &mb.shells[2],
            &mb.shells[0],
            &mb.shells[2],
            native,
        );
        assert_eq!(v.shape, [5, 3, 3, 3]);
        for (x, y) in v.values.iter().zip(&reference.data) {
            assert!((x - y).abs() < 2e-12);
        }
    }
}

#[test]
fn normalization_differentiates_exponents_and_contractions_at_high_order() {
    let order = 8;
    let mut exponent = vec![0.0; order + 1];
    exponent[0] = 0.6;
    exponent[1] = 0.07;
    let mut coefficient = vec![0.0; order + 1];
    coefficient[0] = 0.5;
    coefficient[1] = -0.11;
    for l in [0, 1, 2, 3] {
        let metric = evaluate_series_many(&[exponent.clone(), coefficient.clone()], |v| {
            let zero = SeriesScalar::from_f64(0.0);
            let shell = AnalyticShell::normalized(
                l,
                [zero; 3],
                vec![v[0], SeriesScalar::from_f64(0.3)],
                vec![v[1], SeriesScalar::from_f64(0.6)],
            )
            .unwrap();
            jets::overlap(&shell, &shell).unwrap().values
        })
        .unwrap();
        let n = 2 * l as usize + 1;
        for i in 0..n {
            for j in 0..n {
                assert!((metric[i * n + j][0] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
                for derivative in &metric[i * n + j][1..] {
                    assert!(derivative.abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn moving_basis_and_nuclei_preserve_translation_at_all_orders() {
    let (mol, mb) = fixture();
    let mut translation = vec![0.0; 9];
    translation[1] = 1.0;
    let output = evaluate_series_many(&[translation], |t| {
        let shells: Vec<_> = mb
            .shells
            .iter()
            .map(|s| {
                AnalyticShell::from_placed(
                    s,
                    [
                        t[0] + SeriesScalar::from_f64(s.center.x),
                        SeriesScalar::from_f64(s.center.y),
                        SeriesScalar::from_f64(s.center.z),
                    ],
                )
                .unwrap()
            })
            .collect();
        let nuclei: Vec<_> = mol
            .atoms
            .iter()
            .map(|a| AnalyticNucleus {
                center: [
                    t[0] + SeriesScalar::from_f64(a.position.x),
                    SeriesScalar::from_f64(a.position.y),
                    SeriesScalar::from_f64(a.position.z),
                ],
                charge: SeriesScalar::from_f64(a.z as f64),
            })
            .collect();
        let mut result = jets::overlap(&shells[1], &shells[2]).unwrap().values;
        result.extend(jets::kinetic(&shells[1], &shells[2]).unwrap().values);
        result.extend(
            jets::attraction(&shells[1], &shells[2], &nuclei, GaussianKernel::Coulomb, 8)
                .unwrap()
                .values,
        );
        // One s-shell pair keeps the high-order four-centre gate inexpensive.
        let s = |center| {
            AnalyticShell::normalized(
                0,
                center,
                vec![SeriesScalar::from_f64(0.4)],
                vec![SeriesScalar::from_f64(1.0)],
            )
            .unwrap()
        };
        let a = s(shells[0].center);
        let b = s(shells[2].center);
        result.extend(
            jets::repulsion(&a, &a, &b, &b, GaussianKernel::Coulomb, 8)
                .unwrap()
                .values,
        );
        result
    })
    .unwrap();
    for coefficients in output {
        for c in &coefficients[1..] {
            assert!(c.abs() < 2e-12);
        }
    }
    assert!(AnalyticShell::normalized(7, [0.0; 3], vec![0.4], vec![1.0]).is_err());
    assert!(AnalyticShell::normalized(0, [0.0; 3], vec![0.4], vec![0.0]).is_err());
}
