// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Closed-form London s-Gaussian H/S and independently rediagonalized
//! electronic-sector energy on an asymmetric ammonia geometry.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::dual::{Scalar, Taylor};
use dft_rs::grid::gauss_legendre;
use dft_rs::integrals::analytic_basis::AnalyticShell;
use dft_rs::integrals::dirac::{
    london_magnetic_balance, matrix_coefficients, restricted_kinetic_balance, GaussianNucleus,
};
use dft_rs::math::Vec3;
use dft_rs::periodic::BlochMatrix;
use dft_rs::properties::no_pair::no_pair_projector_taylor;
use dft_rs::properties::no_pair::stationary::NoPairSpace;
use dft_rs::response::complex::Complex;
use std::f64::consts::PI;

const N: usize = 4;
const SPEED: f64 = 13.7;
const NUMBERS: [u8; N] = [7, 1, 1, 1];
const COORDINATES: [[f64; 3]; N] = [
    [0.12, -0.21, 0.15],
    [1.74, 0.18, 0.78],
    [-0.76, 1.27, -0.57],
    [-0.61, -1.52, -0.73],
];
const ALPHA: [f64; N] = [0.935, 0.682, 0.817, 0.751];
const RADII: [f64; N] = [4.84e-5, 1.59e-5, 1.59e-5, 1.59e-5];

type Jet = Taylor<3>;

fn vector([x, y, z]: [f64; 3]) -> Vec3 {
    Vec3::new(x, y, z)
}

fn inputs<S: Scalar>(translation: Vec3) -> (Vec<AnalyticShell<S>>, Vec<GaussianNucleus<S>>) {
    let at = |i| {
        (vector(COORDINATES[i]) + translation)
            .to_array()
            .map(S::from_f64)
    };
    let shells = (0..N)
        .map(|i| {
            AnalyticShell::normalized(
                0,
                at(i),
                vec![S::from_f64(ALPHA[i])],
                vec![S::from_f64(1.0)],
            )
            .unwrap()
        })
        .collect();
    let nuclei = (0..N)
        .map(|i| GaussianNucleus {
            center: at(i),
            charge: S::from_f64(f64::from(NUMBERS[i])),
            rms_radius: S::from_f64(RADII[i]),
        })
        .collect();
    (shells, nuclei)
}

fn production(field: Vec3, direction: Vec3, translation: Vec3) -> [Vec<BlochMatrix>; 2] {
    let (shells, nuclei) = inputs::<Jet>(translation);
    let magnetic = std::array::from_fn(|i| {
        Jet::from_coefficients(&[field.to_array()[i], direction.to_array()[i]])
    });
    let phase_order = if field == Vec3::zero() { 2 } else { 6 };
    let integrals = london_magnetic_balance(
        &shells,
        &nuclei,
        Jet::from_f64(SPEED),
        magnetic,
        2,
        phase_order,
    )
    .unwrap();
    [
        matrix_coefficients(
            &integrals.hamiltonian_real,
            Some(&integrals.hamiltonian_imaginary),
            2,
        ),
        matrix_coefficients(&integrals.overlap, Some(&integrals.overlap_imaginary), 2),
    ]
    .map(Result::unwrap)
}

fn pauli(product: [[Complex; 3]; 3]) -> [Complex; 4] {
    let diag = product[0][0] + product[1][1] + product[2][2];
    let [x, y, z] = [(1, 2), (2, 0), (0, 1)].map(|(a, b)| product[a][b] - product[b][a]);
    let i = Complex::new(0.0, 1.0);
    [diag + i * z, i * x + y, i * x - y, diag - i * z]
}

fn oracle(field: Vec3) -> [BlochMatrix; 2] {
    let (nodes, weights) = gauss_legendre(32);
    let [mut h, mut s] = std::array::from_fn(|_| BlochMatrix::zeros(4 * N, 4 * N));
    for a in 0..N {
        let ra = vector(COORDINATES[a]);
        for b in 0..N {
            let rb = vector(COORDINATES[b]);
            let k = field.cross(ra - rb) * 0.5;
            let p = ALPHA[a] + ALPHA[b];
            let p0 = (ra * ALPHA[a] + rb * ALPHA[b]) / p;
            let gaussian = (4.0 * ALPHA[a] * ALPHA[b]).powf(0.75) / p.powf(1.5)
                * (-ALPHA[a] * ALPHA[b] / p * (ra - rb).dot(ra - rb)).exp();
            // The two factors of pi are affine in r. Integrating their
            // product only needs the first Fourier-shifted Gaussian moment
            // and the exact Gaussian covariance.
            let coefficients = |exponent: f64, sign: f64, axis: usize| {
                let mut unit = [0.0; 3];
                unit[axis] = 1.0;
                let row = vector(unit).cross(field) * 0.5;
                let row = row.to_array();
                std::array::from_fn::<_, 3, _>(|j| {
                    Complex::new(
                        row[j],
                        if j == axis {
                            sign * 2.0 * exponent
                        } else {
                            0.0
                        },
                    )
                })
            };
            let left = std::array::from_fn::<_, 3, _>(|i| coefficients(ALPHA[a], -1.0, i));
            let right = std::array::from_fn::<_, 3, _>(|j| coefficients(ALPHA[b], 1.0, j));
            let integrand = |center: Vec3, invp: f64| {
                let angle = k.dot(center);
                let factor =
                    Complex::new(angle.cos(), angle.sin()) * (-0.25 * invp * k.dot(k)).exp();
                let mean = |r| {
                    let displacement = (center - r).to_array();
                    std::array::from_fn::<_, 3, _>(|i| {
                        Complex::new(displacement[i], 0.5 * invp * k.to_array()[i])
                    })
                };
                let u = mean(ra);
                let v = mean(rb);
                let quadratic = std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        let mut sum = Complex::ZERO;
                        for l in 0..3 {
                            for m in 0..3 {
                                let covariance = if l == m { 0.5 * invp } else { 0.0 };
                                sum += left[i][l]
                                    * right[j][m]
                                    * (u[l] * v[m] + Complex::from(covariance));
                            }
                        }
                        factor * sum
                    })
                });
                (factor, pauli(quadratic))
            };
            let (overlap, kinetic) = integrand(p0, p.recip());
            let mut vll = Complex::ZERO;
            let mut vss = [Complex::ZERO; 4];
            // Integrate the finite nuclear potential as a Laplace mixture.
            // v = u/sqrt(p+u^2) cancels the 3D Gaussian factor in the
            // Jacobian, also when the nuclear radius is much smaller
            // than the spatial AO width.
            for n in 0..N {
                let c = vector(COORDINATES[n]);
                let vmax = (1.0 + (2.0 / 3.0) * p * RADII[n].powi(2)).sqrt().recip();
                let scale = -f64::from(NUMBERS[n]) * vmax * (p / PI).sqrt();
                for (&x, &w) in nodes.iter().zip(&weights) {
                    let v = 0.5 * vmax * (1.0 + x);
                    let d = p0 - c;
                    let decay = (-p * v * v * d.dot(d)).exp();
                    let (value, product) = integrand(p0 - d * v.powi(2), (1.0 - v * v) / p);
                    vll += value * (scale * w * decay);
                    for i in 0..4 {
                        vss[i] += product[i] * (scale * w * decay / (4.0 * SPEED.powi(2)));
                    }
                }
            }
            let set = |m: &mut BlochMatrix, row, column, value: Complex| {
                m.re[(row, column)] = value.re * gaussian;
                m.im[(row, column)] = value.im * gaussian;
            };
            for spin in 0..2 {
                set(&mut s, spin * N + a, spin * N + b, overlap);
                set(&mut h, spin * N + a, spin * N + b, vll);
            }
            for index in 0..4 {
                let row = (index / 2) * N + a;
                let column = (index % 2) * N + b;
                let t = kinetic[index] * 0.5;
                set(&mut h, row, 2 * N + column, t);
                set(&mut h, 2 * N + row, column, t);
                set(&mut h, 2 * N + row, 2 * N + column, vss[index] - t);
                set(
                    &mut s,
                    2 * N + row,
                    2 * N + column,
                    t / (2.0 * SPEED.powi(2)),
                );
            }
        }
    }
    [h, s]
}

fn oracle_energy(field: Vec3) -> f64 {
    let [h, s] = oracle(field);
    let space = NoPairSpace::new(&h, &s, -SPEED * SPEED, 1e-12).unwrap();
    let hc = space.project_operator(&h).unwrap();
    (0..hc.rows()).map(|i| hc.re[(i, i)]).sum()
}

fn response_energy([h, s]: &[Vec<BlochMatrix>; 2]) -> [f64; 3] {
    let response = no_pair_projector_taylor(h, s, 2, -SPEED * SPEED, 1e-12).unwrap();
    for defect in response
        .relative_commutator_defects
        .iter()
        .chain(&response.projector_defects)
    {
        assert!(*defect < 2e-10, "{response:?}");
    }
    std::array::from_fn(|n| {
        (0..=n)
            .map(|i| response.projector[i].trace_product(&h[n - i]))
            .sum()
    })
}

fn matrix_error(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}

#[test]
fn london_metric_and_core_agree_with_an_independent_finite_field_oracle() {
    c1_geometry::assert_c1(&NUMBERS, &COORDINATES);
    let field = Vec3::new(0.013, -0.019, 0.011);
    let direction = Vec3::new(-0.31, 0.46, 0.67);
    let [h, s] = production(field, direction, Vec3::zero());
    let [href, sref] = oracle(field);
    let defects = [matrix_error(&h[0], &href), matrix_error(&s[0], &sref)];
    for m in h.iter().chain(&s) {
        assert!(m.hermiticity_error() < 2e-12);
    }
    println!("C1 ammonia finite-field H/S absolute discrepancies {defects:?}");
    assert!(defects[0] < 1e-11 && defects[1] < 2e-12);
    let base = response_energy(&[h, s]);
    let shifted = response_energy(&production(field, direction, Vec3::new(1.31, -2.11, 0.93)));
    let errors = std::array::from_fn::<_, 3, _>(|i| (base[i] - shifted[i]).abs());
    println!("C1 ammonia electronic trace {base:?}, translation errors {errors:?}");
    assert!(errors.iter().all(|x| *x < 2e-10));
    let (shells, nuclei) = inputs::<Jet>(Vec3::zero());
    let rkb = restricted_kinetic_balance(&shells, &nuclei, Jet::from_f64(SPEED), 0).unwrap();
    let href =
        matrix_coefficients(&rkb.hamiltonian_real, Some(&rkb.hamiltonian_imaginary), 0).unwrap();
    let sref = matrix_coefficients(&rkb.overlap, Some(&rkb.overlap_imaginary), 0).unwrap();
    let [h0, s0] = production(Vec3::zero(), direction, Vec3::zero());
    assert!(matrix_error(&h0[0], &href[0]) < 1e-13);
    assert!(matrix_error(&s0[0], &sref[0]) < 1e-14);
}

#[test]
fn moving_london_sector_first_and_second_field_tensors_pass_independent_differences() {
    use stencils::Stencil;
    let field = Vec3::new(0.013, -0.019, 0.011);
    let axes = [
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 1., 0.),
        Vec3::new(0., 0., 1.),
    ];
    let axes_values = axes.map(|a| response_energy(&production(field, a, Vec3::zero())));
    let mut exact = axes_values.map(|e| e[1]).to_vec();
    let components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    exact.extend(components.map(|(i, j)| {
        if i == j {
            return 2.0 * axes_values[i][2];
        }
        response_energy(&production(field, axes[i] + axes[j], Vec3::zero()))[2]
            - axes_values[i][2]
            - axes_values[j][2]
    }));
    let names: Vec<_> = (0..3)
        .map(|i| format!("core trace first {i}"))
        .chain(components.map(|(i, j)| format!("core trace second {i},{j}")))
        .collect();
    let base = oracle_energy(field);
    let estimate = |index: usize, h: f64, stencil: Stencil| {
        if index < 3 {
            let along = axes[index] * h;
            let plus = oracle_energy(field + along);
            let minus = oracle_energy(field - along);
            let (p2, m2) = if stencil == Stencil::Fourth {
                (
                    oracle_energy(field + along * 2.0),
                    oracle_energy(field - along * 2.0),
                )
            } else {
                (0.0, 0.0)
            };
            stencil.first(plus, minus, p2, m2, h)
        } else {
            let (i, j) = components[index - 3];
            let central = |h: f64| {
                let a = axes[i] * h;
                let b = axes[j] * h;
                if i == j {
                    ((oracle_energy(field + a) - base) + (oracle_energy(field - a) - base))
                        / h.powi(2)
                } else {
                    ((oracle_energy(field + a + b) - oracle_energy(field + a - b))
                        - (oracle_energy(field - a + b) - oracle_energy(field - a - b)))
                        / (4.0 * h.powi(2))
                }
            };
            let value = central(h);
            if stencil == Stencil::Fourth {
                (4.0 * value - central(2.0 * h)) / 3.0
            } else {
                value
            }
        }
    };
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.09, 0.065, 0.046, 0.032], Stencil::Second),
            ([0.043, 0.031, 0.023, 0.016], Stencil::Second),
            ([0.11, 0.084, 0.061, 0.046], Stencil::Fourth),
        ],
        |steps, stencil| {
            exact
                .iter()
                .enumerate()
                .map(|(i, value)| steps.map(|h| (estimate(i, h, stencil) - value).abs()))
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("moving London electronic sector", &failures);
    let mut max_absolute: f64 = 0.0;
    for (i, value) in exact.iter().enumerate() {
        let error = (estimate(i, 0.004, Stencil::Fourth) - value).abs();
        max_absolute = max_absolute.max(error);
        assert!(error < 2e-7, "{}: {error:.8e}", names[i]);
    }
    println!("C1 ammonia electronic trace {base:.16e}, derivative vector {exact:?}, largest fourth-order absolute check {max_absolute:.7e}");
}
