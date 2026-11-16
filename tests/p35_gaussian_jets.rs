// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::dual::{DualN, Scalar};
use dft_rs::integrals::analytic::{
    attraction, boys_jet, boys_values, kinetic, overlap, repulsion, GaussianKernel, Primitive,
};
use dft_rs::integrals::eri::eri_prim_cart;
use dft_rs::integrals::CoulombKernel;
use dft_rs::math::Vec3;
use dft_rs::response::series::{evaluate_series, SeriesScalar};
use std::f64::consts::PI;

fn primitive<S: Scalar>(exponent: f64, center: [S; 3], angular: [usize; 3]) -> Primitive<S> {
    Primitive {
        exponent: S::from_f64(exponent),
        center,
        angular,
    }
}

#[test]
fn arbitrary_order_gaussian_center_jets_match_closed_forms() {
    let order = 12;
    for operator in 0..4 {
        let mut seed = vec![0.0; order + 1];
        seed[1] = 1.0;
        let coefficients = evaluate_series(&[seed], |v| {
            let zero = SeriesScalar::from_f64(0.0);
            let moved = [v[0], zero, zero];
            let home = [zero; 3];
            let a = primitive(0.6, moved, [0; 3]);
            let b = primitive(0.4, if operator < 2 { home } else { moved }, [0; 3]);
            match operator {
                0 => overlap(&a, &b).unwrap(),
                1 => kinetic(&a, &b).unwrap(),
                2 => attraction(
                    &a,
                    &b,
                    home,
                    SeriesScalar::from_f64(1.3),
                    GaussianKernel::Coulomb,
                    order,
                )
                .unwrap(),
                _ => repulsion(
                    &a,
                    &b,
                    &primitive(0.7, home, [0; 3]),
                    &primitive(0.5, home, [0; 3]),
                    GaussianKernel::Coulomb,
                    order,
                )
                .unwrap(),
            }
        })
        .unwrap();
        let mut factorial = 1.0;
        for k in 0..=order / 2 {
            if k > 0 {
                factorial *= k as f64;
            }
            let s0 = PI.powf(1.5);
            let mu = 0.24_f64;
            let expected = match operator {
                0 => s0 * (-mu).powi(k as i32) / factorial,
                1 => {
                    s0 * (3.0 * mu * (-mu).powi(k as i32) / factorial
                        - if k > 0 {
                            2.0 * mu * mu * (-mu).powi(k as i32 - 1) / (factorial / k as f64)
                        } else {
                            0.0
                        })
                }
                2 => -1.3 * 2.0 * PI * (-1.0_f64).powi(k as i32) / (factorial * (2 * k + 1) as f64),
                _ => {
                    2.0 * PI.powf(2.5) / (1.2 * 2.2_f64.sqrt()) * (-(1.2 / 2.2_f64)).powi(k as i32)
                        / (factorial * (2 * k + 1) as f64)
                }
            };
            assert!(
                (coefficients[2 * k] - expected).abs() < 1e-11,
                "operator {operator}, order {}: {} versus {expected}",
                2 * k,
                coefficients[2 * k]
            );
            if 2 * k < order {
                assert!(coefficients[2 * k + 1].abs() < 1e-14);
            }
        }
    }
}

#[test]
fn cartesian_generic_repulsion_matches_the_independent_native_engine() {
    let positions = [
        Vec3::new(0.2, -0.1, 0.4),
        Vec3::new(-0.4, 0.3, -0.2),
        Vec3::new(0.1, 0.6, -0.3),
        Vec3::new(-0.2, -0.1, 0.5),
    ];
    let angulars = [[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 2]];
    let exponents = [0.6, 0.4, 0.7, 0.5];
    let primitives: [Primitive<f64>; 4] = std::array::from_fn(|i| Primitive {
        exponent: exponents[i],
        center: [positions[i].x, positions[i].y, positions[i].z],
        angular: angulars[i],
    });
    for (kernel, native) in [
        (GaussianKernel::Coulomb, CoulombKernel::Coulomb),
        (
            GaussianKernel::LongRange { omega: 0.31 },
            CoulombKernel::Erf { omega: 0.31 },
        ),
        (
            GaussianKernel::ShortRange { omega: 0.31 },
            CoulombKernel::Erfc { omega: 0.31 },
        ),
    ] {
        let value = repulsion(
            &primitives[0],
            &primitives[1],
            &primitives[2],
            &primitives[3],
            kernel,
            0,
        )
        .unwrap();
        let l = |i: usize| {
            (
                angulars[i][0] as i32,
                angulars[i][1] as i32,
                angulars[i][2] as i32,
            )
        };
        let expected = eri_prim_cart(
            exponents[0],
            l(0),
            positions[0],
            exponents[1],
            l(1),
            positions[1],
            exponents[2],
            l(2),
            positions[2],
            exponents[3],
            l(3),
            positions[3],
            native,
        );
        assert!((value - expected).abs() < 1e-13);
    }
}

#[test]
fn boys_high_orders_and_zero_base_keep_exact_derivatives() {
    let order = 16;
    let mut seed = vec![0.0; order + 1];
    seed[1] = 1.0;
    let coefficients = evaluate_series(&[seed], |t| boys_jet(3, t[0], order).unwrap()).unwrap();
    let mut factorial = 1.0;
    for n in 0..=order {
        if n > 0 {
            factorial *= n as f64;
        }
        let expected = (-1.0_f64).powi(n as i32) / (factorial * (2 * (3 + n) + 1) as f64);
        assert!((coefficients[n] - expected).abs() < 1e-15);
    }
    for x in [0.0, 0.1, 36.0, 120.0, 900.0] {
        let f = boys_values(160, x).unwrap();
        assert!(f.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(f.windows(2).all(|p| p[0] >= p[1]));
        for n in 0..160 {
            let residual = (2 * n + 1) as f64 * f[n] - 2.0 * x * f[n + 1] - (-x).exp();
            assert!(residual.abs() < 2e-12 * f[n].max(1e-300));
        }
    }
    let t = DualN::<f64, 1>::variable(0.4, 0);
    let value = boys_jet(2, t, 1).unwrap();
    assert!((value.im[0] + boys_values(3, 0.4).unwrap()[3]).abs() < 1e-14);
}
