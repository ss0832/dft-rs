// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fourier Gaussian products without raising angular momentum by the phase
//! order. A real formal variable u=i t shifts each Gaussian product centre;
//! multiplying coefficient n by i^n restores the physical field series.
use super::analytic::{self, GaussianKernel, Primitive};
use crate::dual::{Scalar, Taylor};
use crate::Result;

fn product_shift<const W: usize>(
    a: &Primitive<f64>,
    b: &Primitive<f64>,
    k: [f64; 3],
) -> ([Taylor<W>; 3], Taylor<W>) {
    let p = a.exponent + b.exponent;
    let centre: [f64; 3] =
        std::array::from_fn(|x| (a.exponent * a.center[x] + b.exponent * b.center[x]) / p);
    let linear = (0..3).map(|x| k[x] * centre[x]).sum::<f64>();
    let quadratic = k.iter().map(|x| x * x).sum::<f64>() / (4.0 * p);
    (
        k.map(|x| Taylor::from_coefficients(&[0.0, x / (2.0 * p)])),
        Taylor::from_coefficients(&[0.0, linear, quadratic]),
    )
}

fn shifted<const W: usize>(
    p: &Primitive<f64>,
    delta: [Taylor<W>; 3],
) -> Vec<(Taylor<W>, Primitive<Taylor<W>>)> {
    let primitive = Primitive {
        angular: p.angular,
        exponent: Taylor::from_f64(p.exponent),
        center: std::array::from_fn(|a| Taylor::from_f64(p.center[a]) + delta[a]),
    };
    let mut terms = vec![(Taylor::from_f64(1.0), primitive)];
    for (axis, &d) in delta.iter().enumerate() {
        let l = p.angular[axis];
        let mut next = Vec::new();
        for (weight, primitive) in terms {
            let mut binomial = 1.0;
            for removed in 0..=l {
                let mut value = primitive;
                value.angular[axis] = l - removed;
                let coefficient = weight * d.powi(removed as i32).scale(binomial);
                // Only an identically zero coefficient may be omitted.
                if coefficient.c.iter().any(|&x| x != 0.0) {
                    next.push((coefficient, value));
                }
                binomial *= (l - removed) as f64 / (removed + 1) as f64;
            }
        }
        terms = next;
    }
    terms
}

fn coefficients<const W: usize>(
    primitives: [&Primitive<f64>; 4],
    kernel: GaussianKernel<f64>,
    bra: [f64; 3],
    ket: [f64; 3],
    first: usize,
    last: usize,
) -> Result<Vec<(f64, f64)>> {
    let [a, b, c, d] = primitives;
    let mut factor = Taylor::<W>::from_f64(0.0);
    let shifts: [[Taylor<W>; 3]; 2] = [(a, b, bra), (c, d, ket)].map(|(a, b, k)| {
        let (shift, value) = product_shift(a, b, k);
        factor = factor + value;
        shift
    });
    let terms = [
        shifted(a, shifts[0]),
        shifted(b, shifts[0]),
        shifted(c, shifts[1]),
        shifted(d, shifts[1]),
    ];
    let kernel = match kernel {
        GaussianKernel::Coulomb => GaussianKernel::Coulomb,
        GaussianKernel::LongRange { omega } => GaussianKernel::LongRange {
            omega: Taylor::from_f64(omega),
        },
        GaussianKernel::ShortRange { omega } => GaussianKernel::ShortRange {
            omega: Taylor::from_f64(omega),
        },
    };
    let mut sum = Taylor::<W>::from_f64(0.0);
    for (wa, a) in &terms[0] {
        for (wb, b) in &terms[1] {
            for (wc, c) in &terms[2] {
                for (wd, d) in &terms[3] {
                    sum = sum
                        + *wa * *wb * *wc * *wd * analytic::repulsion(a, b, c, d, kernel, last)?;
                }
            }
        }
    }
    let result = sum * factor.exp();
    Ok((first..=last)
        .map(|n| match n % 4 {
            0 => (result.c[n], 0.0),
            1 => (0.0, result.c[n]),
            2 => (-result.c[n], 0.0),
            _ => (0.0, -result.c[n]),
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn pair<const W: usize>(
    a: &Primitive<f64>,
    b: &Primitive<f64>,
    k: [f64; 3],
    nuclei: &[([f64; 3], f64)],
    electric: [f64; 3],
    magnetic: [f64; 3],
    gauge: [f64; 3],
    last: usize,
) -> Result<Vec<Vec<(f64, f64)>>> {
    let (delta, factor) = product_shift(a, b, k);
    let left = shifted(a, delta);
    let ket = super::Terms::single(*b);
    let operators = [
        ket.clone(),
        ket.kinetic(),
        ket.angular(magnetic, gauge),
        ket.dotted(electric),
        ket.diamagnetic(magnetic, gauge),
        ket.shifted(0, 0.0),
        ket.shifted(1, 0.0),
        ket.shifted(2, 0.0),
    ];
    let mut values = [Taylor::<W>::from_f64(0.0); 8];
    for (operator, terms) in operators.iter().enumerate() {
        for (coefficient, b) in &terms.items {
            let right = shifted(b, delta);
            for (wa, a) in &left {
                for (wb, b) in &right {
                    let mut value = analytic::overlap(a, b)?;
                    if operator == 0 {
                        for (centre, charge) in nuclei {
                            values[1] = values[1]
                                + (*wa * *wb).scale(*coefficient)
                                    * analytic::attraction(
                                        a,
                                        b,
                                        centre.map(Taylor::from_f64),
                                        Taylor::from_f64(*charge),
                                        GaussianKernel::Coulomb,
                                        last,
                                    )?;
                        }
                    }
                    value = value * (*wa * *wb).scale(*coefficient);
                    values[operator] = values[operator] + value;
                }
            }
        }
    }
    let factor = factor.exp();
    values = values.map(|v| v * factor);
    let complex = |v: Taylor<W>, n: usize| match n % 4 {
        0 => (v.c[n], 0.0),
        1 => (0.0, v.c[n]),
        2 => (-v.c[n], 0.0),
        _ => (0.0, -v.c[n]),
    };
    let mut output = vec![vec![(0.0, 0.0); last + 1]; 5];
    for n in 0..=last {
        output[0][n] = complex(values[0], n);
        output[1][n] = complex(values[1], n);
        for axis in 0..3 {
            output[2 + axis][n] = complex(values[5 + axis], n);
        }
        if n > 0 {
            let (re, im) = complex(values[2], n - 1);
            let electric = complex(values[3], n - 1);
            output[1][n].0 += 0.5 * im + electric.0;
            output[1][n].1 -= 0.5 * re - electric.1;
        }
        if n > 1 {
            let (re, im) = complex(values[4], n - 2);
            output[1][n].0 += re;
            output[1][n].1 += im;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pair_coefficients(
    a: &Primitive<f64>,
    b: &Primitive<f64>,
    k: [f64; 3],
    nuclei: &[([f64; 3], f64)],
    electric: [f64; 3],
    magnetic: [f64; 3],
    gauge: [f64; 3],
    last: usize,
) -> Result<Option<Vec<Vec<(f64, f64)>>>> {
    let values = match last {
        0..=2 => pair::<3>(a, b, k, nuclei, electric, magnetic, gauge, last)?,
        3..=4 => pair::<5>(a, b, k, nuclei, electric, magnetic, gauge, last)?,
        5..=8 => pair::<9>(a, b, k, nuclei, electric, magnetic, gauge, last)?,
        9..=12 => pair::<13>(a, b, k, nuclei, electric, magnetic, gauge, last)?,
        13..=16 => pair::<17>(a, b, k, nuclei, electric, magnetic, gauge, last)?,
        17..=24 => pair::<25>(a, b, k, nuclei, electric, magnetic, gauge, last)?,
        25..=32 => pair::<33>(a, b, k, nuclei, electric, magnetic, gauge, last)?,
        _ => return Ok(None),
    };
    Ok(Some(values))
}

pub(super) fn repulsion_coefficients(
    primitives: [&Primitive<f64>; 4],
    kernel: GaussianKernel<f64>,
    bra: [f64; 3],
    ket: [f64; 3],
    first: usize,
    last: usize,
) -> Result<Vec<(f64, f64)>> {
    match last {
        0..=2 => coefficients::<3>(primitives, kernel, bra, ket, first, last),
        3..=4 => coefficients::<5>(primitives, kernel, bra, ket, first, last),
        5..=8 => coefficients::<9>(primitives, kernel, bra, ket, first, last),
        9..=12 => coefficients::<13>(primitives, kernel, bra, ket, first, last),
        13..=16 => coefficients::<17>(primitives, kernel, bra, ket, first, last),
        17..=24 => coefficients::<25>(primitives, kernel, bra, ket, first, last),
        25..=32 => coefficients::<33>(primitives, kernel, bra, ket, first, last),
        // Preserve the arbitrary-order API rather than changing its cap.
        _ => super::primitive_repulsion_coefficients(
            primitives,
            kernel,
            bra,
            ket,
            first,
            last,
            0,
            |x| x == 0.0,
        ),
    }
}

#[cfg(test)]
#[path = "london_phase_tests.rs"]
mod tests;
