// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent complex-Gaussian moments, explicit translation sums and
//! four-step fits of magnetic nuclear operators in seven crystal metrics.
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/crystals.rs"]
mod crystals;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_cell::{MagneticCell, MagneticPairOperator};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::{Atom, Molecule};
use std::f64::consts::PI;

const POSITIONS: [[f64; 3]; 2] = [[0.1, 0.2, -0.1], [0.6, 0.7, 0.3]];
const DIRECTION: [[f64; 3]; 2] = [[0.4, -0.3, 0.2], [-0.3, 0.5, -0.25]];
const CUTOFF: f64 = 8.0;
fn phase(x: f64) -> Complex {
    Complex::new(x.cos(), x.sin())
}
fn values(matrix: &BlochMatrix) -> Vec<Complex> {
    matrix
        .re
        .as_slice()
        .iter()
        .zip(matrix.im.as_slice())
        .map(|(r, i)| Complex::new(*r, *i))
        .collect()
}
fn maximum_difference(a: &[Complex], b: &[Complex]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0, f64::max)
}

/// Closed complex Gaussian mean and covariance, with no Hermite recurrence,
/// angular shifts, generic Scalar arithmetic or production image enumerator.
fn oracle(
    cell: &MagneticCell,
    basis: &MolBasis,
    wave: Vec3,
    kind: MagneticPairOperator,
    t: f64,
) -> Vec<Complex> {
    let mut output = vec![Complex::ZERO; 4];
    for (a, sa) in basis.shells.iter().enumerate() {
        for (b, sb) in basis.shells.iter().enumerate() {
            let alpha = sa.exps[0];
            let beta = sb.exps[0];
            let p = alpha + beta;
            for x in -2..=2 {
                for y in -2..=2 {
                    for zz in -2..=2 {
                        let index = [x, y, zz];
                        if (0..3).any(|k| !cell.geometry.boundary().axes[k] && index[k] != 0) {
                            continue;
                        }
                        let translation = cell.geometry.lattice().translation(index);
                        if (sb.center + translation - sa.center).norm() > CUTOFF {
                            continue;
                        }
                        let left = sa.center
                            + Vec3::new(DIRECTION[a][0], DIRECTION[a][1], DIRECTION[a][2]) * t;
                        let right = sb.center
                            + translation
                            + Vec3::new(DIRECTION[b][0], DIRECTION[b][1], DIRECTION[b][2]) * t;
                        let original = right - translation;
                        let q = wave - cell.field.cross(right - left) * 0.5;
                        let centre = (left * alpha + right * beta) / p;
                        let gaussian = phase(q.dot(centre))
                            * ((PI / p).powf(1.5)
                                * (-alpha * beta / p * (left - right).norm2()
                                    - q.norm2() / (4.0 * p))
                                    .exp()
                                * sa.coefs[0]
                                * sb.coefs[0]);
                        let da: [Complex; 3] = std::array::from_fn(|k| {
                            Complex::new(centre.get(k) - left.get(k), q.get(k) / (2.0 * p))
                        });
                        let db: [Complex; 3] = std::array::from_fn(|k| {
                            Complex::new(centre.get(k) - right.get(k), q.get(k) / (2.0 * p))
                        });
                        let bra: [Complex; 3] = std::array::from_fn(|k| {
                            let (j, l) = ((k + 1) % 3, (k + 2) % 3);
                            da[k] * Complex::new(0.0, -2.0 * alpha)
                                + (da[l] * cell.field.get(j) - da[j] * cell.field.get(l)) * 0.5
                        });
                        let ket: [Complex; 3] = std::array::from_fn(|k| {
                            let (j, l) = ((k + 1) % 3, (k + 2) % 3);
                            db[k] * Complex::new(0.0, 2.0 * beta)
                                + (db[l] * cell.field.get(j) - db[j] * cell.field.get(l)) * 0.5
                        });
                        let factor = match kind {
                            MagneticPairOperator::Fourier => Complex::ONE,
                            MagneticPairOperator::Momentum(k) => ket[k],
                            MagneticPairOperator::Kinetic => {
                                let mean = (0..3)
                                    .map(|k| bra[k] * ket[k])
                                    .fold(Complex::ZERO, |v, w| v + w);
                                (mean
                                    + Complex::from(
                                        (12.0 * alpha * beta + 0.5 * cell.field.norm2())
                                            / (2.0 * p),
                                    ))
                                    * 0.5
                            }
                        };
                        let m = cell.flux_quanta;
                        let parity = (m[0] * x * y + m[1] * y * zz - m[2] * x * zz).rem_euclid(2);
                        let coefficient = phase(
                            cell.bloch_wave.dot(translation)
                                + 0.5 * cell.field.dot(original.cross(translation)),
                        ) * if parity == 0 { 1.0 } else { -1.0 };
                        output[2 * a + b] += coefficient * gaussian * factor;
                    }
                }
            }
        }
    }
    output
}

fn accept(dimension: usize) {
    let rows = crystals::rows();
    let molecule = Molecule::new(
        POSITIONS
            .iter()
            .map(|r| Atom {
                z: 1,
                position: Vec3::new(r[0], r[1], r[2]),
            })
            .collect(),
    );
    let mut set = BasisSet::new("magnetic-nuclear-moment-oracle");
    set.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
    let basis = MolBasis::build(&molecule, &set).unwrap();
    let steps = [0.08, 0.06, 0.04, 0.02];
    let mut fits = Vec::new();
    for (crystal, row) in rows.iter().enumerate() {
        let a = row.map(|r| Vec3::new(r[0], r[1], r[2]));
        let lattice = Lattice::new(a[0], a[1], a[2]);
        let geometry = PeriodicGeometry::new(
            lattice,
            PeriodicBoundary::new(std::array::from_fn(|k| k < dimension)),
        )
        .unwrap();
        let field = match dimension {
            1 => Vec3::new(0.12, 0.08, 0.16),
            2 => a[0].cross(a[1]) * (2.0 * PI / a[0].cross(a[1]).norm2()),
            _ => a[2] * (2.0 * PI / geometry.lattice().volume()),
        };
        let cell = MagneticCell::new(geometry, field, Vec3::new(0.2, -0.1, 0.3)).unwrap();
        let wave = cell.geometry.reciprocal()[0];
        for (operator, kind) in [
            MagneticPairOperator::Fourier,
            MagneticPairOperator::Kinetic,
            MagneticPairOperator::Momentum(0),
            MagneticPairOperator::Momentum(1),
            MagneticPairOperator::Momentum(2),
        ]
        .into_iter()
        .enumerate()
        {
            let g = if matches!(kind, MagneticPairOperator::Kinetic) {
                Vec3::zero()
            } else {
                wave
            };
            let series = cell
                .nuclear_operator_taylor(&basis, CUTOFF, g, kind, &DIRECTION, 4)
                .unwrap();
            let zero = oracle(&cell, &basis, g, kind, 0.0);
            let baseline = maximum_difference(&values(&series[0]), &zero);
            assert!(baseline<2e-13,"dimension={dimension} crystal={crystal} operator={operator} baseline={baseline:.16e}");
            let mut approximations = Vec::new();
            for h in steps {
                let (p, m, pp, mm) = (
                    oracle(&cell, &basis, g, kind, h),
                    oracle(&cell, &basis, g, kind, -h),
                    oracle(&cell, &basis, g, kind, 2.0 * h),
                    oracle(&cell, &basis, g, kind, -2.0 * h),
                );
                let estimates: [Vec<Complex>; 4] = std::array::from_fn(|n| {
                    (0..4)
                        .map(|i| match n {
                            0 => (p[i] - m[i]) / (2.0 * h),
                            1 => (p[i] + m[i] - zero[i] * 2.0) / (h * h),
                            2 => (pp[i] - p[i] * 2.0 + m[i] * 2.0 - mm[i]) / (2.0 * h * h * h),
                            _ => {
                                (pp[i] - p[i] * 4.0 + zero[i] * 6.0 - m[i] * 4.0 + mm[i])
                                    / (h * h * h * h)
                            }
                        })
                        .collect()
                });
                approximations.push(estimates);
            }
            for n in 1..=4 {
                let target: Vec<_> = values(&series[n])
                    .into_iter()
                    .map(|z| z * (1..=n).product::<usize>() as f64)
                    .collect();
                let errors: Vec<_> = approximations
                    .iter()
                    .map(|v| maximum_difference(&v[n - 1], &target))
                    .collect();
                let fit = convergence::error_power_fit(&steps, &errors);
                let rich: Vec<_> = approximations[3][n - 1]
                    .iter()
                    .zip(&approximations[2][n - 1])
                    .map(|(a, b)| (*a * 4.0 - *b) / 3.0)
                    .collect();
                let richardson = maximum_difference(&rich, &target);
                println!("magnetic nuclear operator dimension={dimension} crystal={crystal} operator={operator} order={n} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.16e} baseline={baseline:.16e}",fit.exponent,fit.r_squared);
                fits.push((dimension, crystal, operator, n, fit, richardson));
            }
        }
    }
    for (dimension, crystal, operator, n, fit, richardson) in fits {
        assert!((fit.exponent-2.0).abs()<0.05 && fit.r_squared>0.9999 && richardson<2e-6,
            "dimension={dimension} crystal={crystal} operator={operator} order={n} p={} R2={} Richardson={richardson}",fit.exponent,fit.r_squared);
    }
}

#[test]
fn wire_magnetic_nuclear_operators_against_closed_gaussian_moments() {
    accept(1);
}
#[test]
fn slab_magnetic_nuclear_operators_against_closed_gaussian_moments() {
    accept(2);
}
#[test]
fn bulk_magnetic_nuclear_operators_against_closed_gaussian_moments() {
    accept(3);
}
