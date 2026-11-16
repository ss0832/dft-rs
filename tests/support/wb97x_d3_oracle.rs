// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Direct evaluation of the published ωB97X-D3 channels.

use dft_rs::dual::Scalar;
use std::f64::consts::PI;

pub const PARAMETERS: [f64; 19] = [
    0.004, 0.2, 0.006, 0.25, 0.804272, 0.698900, 0.508940, -3.744903, 10.060790, 1.0, -4.868902,
    21.295726, -36.020866, 19.177018, 1.0, 2.433266, -15.446008, 17.644390, -8.879494,
];

fn gas_channel<S: Scalar>(rs: S, spin: usize) -> S {
    let coefficients = [
        [0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294],
        [0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517],
        [0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671],
    ];
    let [a, a1, b1, b2, b3, b4] = coefficients[spin];
    let root = rs.sqrt();
    let f = root.scale(b1) + (rs.scale(b2) + root.scale(b3) * rs) + rs.powi(2).scale(b4);
    -(S::from_f64(1.0) + rs.scale(a1)) * f.scale(2.0 * a).recip().ln_1p().scale(2.0 * a)
}

fn radius<S: Scalar>(rho: S) -> S {
    rho.recip().scale(0.75 / PI).powf(1.0 / 3.0)
}

fn gas<S: Scalar>(a: S, b: S) -> S {
    let rs = radius(a + b);
    let z = (a - b) / (a + b);
    let one = S::from_f64(1.0);
    let eo = gas_channel(rs, 0);
    let ef = gas_channel(rs, 1);
    let spin_interpolation =
        (one + z).powf(4.0 / 3.0) + (one - z).powf(4.0 / 3.0) - S::from_f64(2.0);
    eo + spin_interpolation
        * ((ef - eo) * z.powi(4) / S::from_f64(2.0_f64.powf(4.0 / 3.0) - 2.0)
            - gas_channel(rs, 2) * (one - z.powi(4)).scale(9.0 / 8.0))
}

fn short_range_fraction<S: Scalar>(u: S) -> S {
    // In reciprocal space the fraction is (8/3) times the integral over
    // x=q/(2*k_F) in [0,1] of (1-3*x/2+x^3/2)*(1-exp(-u^2*x^2)).
    // The cubic is the intersection volume of two Fermi spheres.
    let u2 = u * u;
    if u2.value() <= 0.25 {
        let mut sum = S::from_f64(0.0);
        let mut power = u2;
        for n in 1..=16 {
            let m = f64::from(2 * n);
            sum = sum + power.scale(8.0 / ((m + 1.0) * (m + 2.0) * (m + 4.0)));
            power = -(power * u2).scale(1.0 / f64::from(n + 1));
        }
        return sum;
    }
    let one = S::from_f64(1.0);
    let endpoint = (-u2).exp();
    let i0 = u.erf().scale(0.5 * PI.sqrt()) / u;
    let i1 = (one - endpoint) / u2.scale(2.0);
    let i3 = (one - (one + u2) * endpoint) / u2.powi(2).scale(2.0);
    one - (i0 - i1.scale(1.5) + i3.scale(0.5)).scale(8.0 / 3.0)
}

fn polynomial<S: Scalar>(coefficient: &[S], gamma: S, gradient: S) -> S {
    let one = S::from_f64(1.0);
    let numerator = gamma * gradient;
    let u = numerator / (one + numerator);
    coefficient
        .iter()
        .rev()
        .fold(S::from_f64(0.0), |y, &c| y * u + c)
}

/// Independent short-range X, same-spin C and opposite-spin C energy densities.
pub fn terms<S: Scalar>(point: [S; 5], parameters: &[S; 19]) -> [S; 3] {
    let [a, b, aa, _, bb] = point;
    let zero = S::from_f64(0.0);
    let mut ex = zero;
    let mut ss = zero;
    let mut gradients = [zero; 2];
    let mut ec_uniform = zero;
    for (i, (rho, sigma)) in [(a, aa), (b, bb)].into_iter().enumerate() {
        if rho.value() < 1e-13 {
            continue;
        }
        let rho43 = rho.powf(4.0 / 3.0);
        let gradient = sigma / rho43.powi(2);
        gradients[i] = gradient;
        let kf = rho.scale(6.0 * PI * PI).cbrt();
        let attenuation = if parameters[3].value() == 0.0 {
            S::from_f64(1.0)
        } else {
            short_range_fraction(kf / parameters[3])
        };
        ex = ex
            - (kf * rho).scale(3.0 / (4.0 * PI))
                * attenuation
                * polynomial(&parameters[4..9], parameters[0], gradient);
        let ep = rho * gas_channel(radius(rho), 1);
        ss = ss + ep * polynomial(&parameters[9..14], parameters[1], gradient);
        ec_uniform = ec_uniform + ep;
    }
    let os = if a.value().min(b.value()) >= 1e-13 {
        ((a + b) * gas(a, b) - ec_uniform)
            * polynomial(
                &parameters[14..19],
                parameters[2],
                (gradients[0] + gradients[1]).scale(0.5),
            )
    } else {
        zero
    };
    [ex, ss, os]
}

const TABLE_RADII: &str = include_str!("../../src/dispersion/data/d3_radii.csv");
const TABLE_PAIR_RADII: &str = include_str!("../../src/dispersion/data/d3_r0ab.csv");
const TABLE_REFERENCES: &str = include_str!("../../src/dispersion/data/d3_c6_reference.csv");

fn table_rows<const N: usize>(text: &str) -> impl Iterator<Item = [f64; N]> + '_ {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .skip(1)
        .map(|line| {
            let mut columns = line.split(',').map(|v| v.trim().parse().unwrap());
            std::array::from_fn(|_| columns.next().unwrap())
        })
}

#[derive(Clone, Copy)]
struct ImagePair {
    i: usize,
    j: usize,
    translation: [f64; 3],
    coordination: bool,
    dispersion: bool,
}

struct ReferencePair {
    covalent_sum: f64,
    cutoff_radius: f64,
    c8_over_c6: f64,
    references: Vec<[f64; 3]>,
}

/// Published pair parameters, independently interpolated over explicit images.
pub struct DispersionOracle<const N: usize> {
    reference: [[ReferencePair; N]; N],
    images: Vec<ImagePair>,
}

impl<const N: usize> DispersionOracle<N> {
    pub fn new(
        numbers: [u8; N],
        positions: &[[f64; 3]; N],
        lattice: [[f64; 3]; 3],
        pbc: [bool; 3],
        coordination_cutoff: f64,
        pair_cutoff: f64,
        margin: f64,
    ) -> Self {
        let radii: [[f64; 3]; N] = numbers.map(|z| {
            table_rows(TABLE_RADII)
                .find(|row| row[0] == f64::from(z))
                .unwrap()
        });
        let pair_radii: Vec<_> = table_rows::<1>(TABLE_PAIR_RADII)
            .map(|row| row[0])
            .collect();
        let records: Vec<[f64; 5]> = table_rows(TABLE_REFERENCES).collect();
        let conversion = dft_rs::constants::ANGSTROM_TO_BOHR;
        let reference = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let [zi, moment_i, covalent_i] = radii[i];
                let [zj, moment_j, covalent_j] = radii[j];
                let k = usize::from(numbers[i].max(numbers[j]));
                let l = usize::from(numbers[i].min(numbers[j]));
                let mut references = Vec::new();
                for &[a, b, c6, cn_a, cn_b] in &records {
                    if (a, b) == (zi, zj) {
                        references.push([c6, cn_a, cn_b]);
                    }
                    if (b, a) == (zi, zj) && (a != b || cn_a != cn_b) {
                        references.push([c6, cn_b, cn_a]);
                    }
                }
                assert!(!references.is_empty());
                ReferencePair {
                    cutoff_radius: pair_radii[k * (k - 1) / 2 + l - 1] * conversion,
                    c8_over_c6: 1.5 * (moment_i * moment_j * (zi * zj).sqrt()).sqrt(),
                    covalent_sum: (covalent_i + covalent_j) * (4.0 / 3.0) * conversion,
                    references,
                }
            })
        });
        let mut images = Vec::new();
        for i in 0..N {
            for j in 0..N {
                for x in -3_i32..=3 {
                    for y in -3_i32..=3 {
                        for z in -3_i32..=3 {
                            let image = [x, y, z];
                            if (i == j && image == [0; 3])
                                || image
                                    .into_iter()
                                    .zip(pbc)
                                    .any(|(n, periodic)| n != 0 && !periodic)
                            {
                                continue;
                            }
                            let translation = std::array::from_fn(|axis| {
                                (0..3).map(|k| f64::from(image[k]) * lattice[k][axis]).sum()
                            });
                            let mut pair = ImagePair {
                                i,
                                j,
                                translation,
                                coordination: false,
                                dispersion: false,
                            };
                            let distance = Self::distance_squared(positions, pair).sqrt();
                            assert!((distance - coordination_cutoff).abs() > margin);
                            assert!((distance - pair_cutoff).abs() > margin);
                            pair.coordination = distance < coordination_cutoff;
                            pair.dispersion = distance < pair_cutoff;
                            if pair.coordination || pair.dispersion {
                                assert!(image.into_iter().all(|v| v.abs() < 3));
                                images.push(pair);
                            }
                        }
                    }
                }
            }
        }
        Self { reference, images }
    }

    fn distance_squared<S: Scalar>(positions: &[[S; 3]; N], pair: ImagePair) -> S {
        (0..3).fold(S::from_f64(0.0), |value, axis| {
            let displacement = positions[pair.j][axis] - positions[pair.i][axis]
                + S::from_f64(pair.translation[axis]);
            value + displacement.powi(2)
        })
    }

    pub fn terms<S: Scalar>(&self, positions: &[[S; 3]; N]) -> [S; 2] {
        let zero = S::from_f64(0.0);
        let one = S::from_f64(1.0);
        let mut cn = [zero; N];
        for &pair in self.images.iter().filter(|pair| pair.coordination) {
            let distance = Self::distance_squared(positions, pair).sqrt();
            let covalent = self.reference[pair.i][pair.j].covalent_sum;
            let exponent = (one - distance.recip().scale(covalent)).scale(16.0);
            cn[pair.i] = cn[pair.i] + (one + exponent.exp()).recip();
        }
        let mut terms = [zero; 2];
        for &pair in self.images.iter().filter(|pair| pair.dispersion) {
            let r2 = Self::distance_squared(positions, pair);
            let r = r2.sqrt();
            let data = &self.reference[pair.i][pair.j];
            let (mut weights, mut sum) = (zero, zero);
            for &[coefficient, ci, cj] in &data.references {
                let squared =
                    (cn[pair.i] - S::from_f64(ci)).powi(2) + (cn[pair.j] - S::from_f64(cj)).powi(2);
                let weight = squared.scale(-4.0).exp();
                weights = weights + weight;
                sum = sum + weight.scale(coefficient);
            }
            let c6 = sum / weights;
            for (slot, damping_radius, exponent, coefficient) in [
                (0, 1.281, 14, c6),
                (1, 1.094, 16, c6.scale(data.c8_over_c6)),
            ] {
                let damping = one
                    + r.recip()
                        .scale(damping_radius * data.cutoff_radius)
                        .powi(exponent)
                        .scale(6.0);
                terms[slot] =
                    terms[slot] - coefficient / (r2.powi(3 + slot as i32) * damping).scale(2.0);
            }
        }
        terms
    }
}
