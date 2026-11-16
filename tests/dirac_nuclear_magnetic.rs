// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact finite-Gaussian magnetic-dipole vertices on an atomic s orbital.

use dft_rs::dual::{Scalar, Taylor};
use dft_rs::integrals::analytic_basis::AnalyticShell;
use dft_rs::integrals::dirac::{
    matrix_coefficients, nuclear_magnetic_dipole_vertices, GaussianMagnetization,
};
use dft_rs::math::Vec3;
use dft_rs::response::complex::Complex;
use std::f64::consts::PI;

fn pauli(v: Vec3) -> [Complex; 4] {
    [
        Complex::new(v.z, 0.0),
        Complex::new(v.x, -v.y),
        Complex::new(v.x, v.y),
        Complex::new(-v.z, 0.0),
    ]
}

#[test]
fn gaussian_magnetization_contact_and_uniform_field_terms_match_direct_integration() {
    type Jet = Taylor<3>;
    let center = [0.45, -0.72, 0.29].map(Jet::from_f64);
    let alpha = 0.849_f64;
    let radius = 0.354_f64;
    let c = 11.7_f64;
    let [field, direction] = [Vec3::new(0.024, 0.017, -0.031), Vec3::new(0.1, -0.26, 0.31)];
    let shell = AnalyticShell::normalized(
        0,
        center,
        vec![Jet::from_f64(alpha)],
        vec![Jet::from_f64(1.)],
    )
    .unwrap();
    let nucleus = GaussianMagnetization {
        center,
        rms_radius: Jet::from_f64(radius),
    };
    let magnetic =
        std::array::from_fn(|i| Jet::from_coefficients(&[field.get(i), direction.get(i)]));
    let vertices =
        nuclear_magnetic_dipole_vertices(&[shell], nucleus, Jet::from_f64(c), magnetic, 2, 2)
            .unwrap();
    let omega = 1.5_f64.sqrt() / radius;
    let p = 2. * alpha;
    // The independent radial moment is <r_x^2 erf'(omega,r)/r>.
    // Laplace integration of the Gaussian magnetization gives
    // (2 sqrt(p)/(3 sqrt(pi))) * (omega/sqrt(p+omega^2))^3.
    let radial = 2. * p.sqrt() / (3. * PI.sqrt()) * (omega / (p + omega * omega).sqrt()).powi(3);
    let unit = [
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 1., 0.),
        Vec3::new(0., 0., 1.),
    ];
    let mut max_error = 0.0_f64;
    for (axis, [real, imaginary]) in vertices.into_iter().enumerate() {
        let coefficients = matrix_coefficients(&real, Some(&imaginary), 2).unwrap();
        for (order, matrix) in coefficients.iter().enumerate() {
            let value = [field, direction, Vec3::zero()][order];
            let mu = unit[axis];
            let cross = pauli(mu.cross(value));
            let contact = pauli(mu);
            let factor = radial / (2. * c * c);
            let expected = std::array::from_fn::<_, 4, _>(|k| {
                let scalar = if k == 0 || k == 3 { mu.dot(value) } else { 0. };
                let p = if order == 0 {
                    contact[k] * (4. * alpha)
                } else {
                    Complex::ZERO
                };
                (Complex::from(scalar) + p + Complex::new(0., 0.5) * cross[k]) * factor
            });
            for i in 0..4 {
                for j in 0..4 {
                    let target = match (i < 2, j < 2) {
                        (true, false) => expected[2 * i + j - 2],
                        (false, true) => expected[2 * j + i - 2].conj(),
                        _ => Complex::ZERO,
                    };
                    let actual = Complex::new(matrix.re[(i, j)], matrix.im[(i, j)]);
                    max_error = max_error.max((actual - target).abs());
                }
            }
            assert!(matrix.hermiticity_error() < 1e-13);
        }
    }
    println!("finite Gaussian magnetic dipole contact {:.16e}; maximum matrix and field coefficient discrepancy {max_error:.6e}",2.*alpha*radial/c.powi(2));
    assert!(max_error < 1e-13);
}
