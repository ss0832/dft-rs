// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Second-order forward jet in nine variables with a symmetric Hessian:
//! value, gradient and the upper triangle of the Hessian (55 numbers), the
//! truncated Taylor algebra of `DualN<DualN<f64, 9>, 9>` without its
//! redundant mixed copies. Univariate functions lift as
//! `f(x) = (f, f' g, f' H + f'' g g^T)`.

use super::Scalar;
use std::ops::{Add, Div, Mul, Neg, Sub};

const N: usize = 9;
const PAIRS: usize = N * (N + 1) / 2;

/// Index of `(i, j)` with `i <= j` in the packed upper triangle.
#[inline]
pub const fn packed(i: usize, j: usize) -> usize {
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    a * N - a * (a + 1) / 2 + b
}

#[derive(Clone, Copy, Debug)]
pub struct Jet2 {
    pub value: f64,
    pub gradient: [f64; N],
    pub hessian: [f64; PAIRS],
}

impl Default for Jet2 {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl Jet2 {
    pub const fn constant(value: f64) -> Self {
        Self {
            value,
            gradient: [0.0; N],
            hessian: [0.0; PAIRS],
        }
    }

    /// The variable `k` at `value`.
    pub fn variable(value: f64, k: usize) -> Self {
        let mut out = Self::constant(value);
        out.gradient[k] = 1.0;
        out
    }

    /// `d2/dx_i dx_j`.
    pub fn second(&self, i: usize, j: usize) -> f64 {
        self.hessian[packed(i, j)]
    }

    #[inline]
    fn lift(self, f: f64, d1: f64, d2: f64) -> Self {
        let mut out = Self::constant(f);
        for i in 0..N {
            out.gradient[i] = d1 * self.gradient[i];
        }
        let mut p = 0;
        for i in 0..N {
            let gi = self.gradient[i];
            for j in i..N {
                out.hessian[p] = d1 * self.hessian[p] + d2 * gi * self.gradient[j];
                p += 1;
            }
        }
        out
    }
}

impl Add for Jet2 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self.value += rhs.value;
        for (a, b) in self.gradient.iter_mut().zip(rhs.gradient) {
            *a += b;
        }
        for (a, b) in self.hessian.iter_mut().zip(rhs.hessian) {
            *a += b;
        }
        self
    }
}

impl Sub for Jet2 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self.value -= rhs.value;
        for (a, b) in self.gradient.iter_mut().zip(rhs.gradient) {
            *a -= b;
        }
        for (a, b) in self.hessian.iter_mut().zip(rhs.hessian) {
            *a -= b;
        }
        self
    }
}

impl Mul for Jet2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::constant(self.value * rhs.value);
        for i in 0..N {
            out.gradient[i] = self.gradient[i] * rhs.value + self.value * rhs.gradient[i];
        }
        let mut p = 0;
        for i in 0..N {
            let (ai, bi) = (self.gradient[i], rhs.gradient[i]);
            for j in i..N {
                out.hessian[p] = self.hessian[p] * rhs.value
                    + self.value * rhs.hessian[p]
                    + ai * rhs.gradient[j]
                    + bi * self.gradient[j];
                p += 1;
            }
        }
        out
    }
}

impl Div for Jet2 {
    type Output = Self;
    // Division is multiplication by the reciprocal jet.
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

impl Neg for Jet2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl Scalar for Jet2 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }
    #[inline]
    fn scale(mut self, x: f64) -> Self {
        self.value *= x;
        for v in self.gradient.iter_mut().chain(self.hessian.iter_mut()) {
            *v *= x;
        }
        self
    }
    #[inline]
    fn value(self) -> f64 {
        self.value
    }
    fn recip(self) -> Self {
        let r = 1.0 / self.value;
        self.lift(r, -r * r, 2.0 * r * r * r)
    }
    fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        self.lift(s, 0.5 / s, -0.25 / (s * self.value))
    }
    fn cbrt(self) -> Self {
        let c = libm::cbrt(self.value);
        let x = self.value;
        self.lift(c, c / (3.0 * x), -2.0 * c / (9.0 * x * x))
    }
    fn ln(self) -> Self {
        let x = self.value;
        self.lift(x.ln(), 1.0 / x, -1.0 / (x * x))
    }
    fn exp(self) -> Self {
        let e = self.value.exp();
        self.lift(e, e, e)
    }
    fn atan(self) -> Self {
        let x = self.value;
        let d = 1.0 / (1.0 + x * x);
        self.lift(x.atan(), d, -2.0 * x * d * d)
    }
    fn asinh(self) -> Self {
        let x = self.value;
        let r = 1.0 / (x * x + 1.0).sqrt();
        self.lift(x.asinh(), r, -x * r * r * r)
    }
    fn erf(self) -> Self {
        let x = self.value;
        let d = 2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp();
        self.lift(libm::erf(x), d, -2.0 * x * d)
    }
    fn ln_1p(self) -> Self {
        let x = self.value;
        let d = 1.0 / (1.0 + x);
        self.lift(libm::log1p(x), d, -d * d)
    }
    fn erfcx(self) -> Self {
        let x = self.value;
        let f = x.erfcx();
        let d1 = 2.0 * x * f - 2.0 / std::f64::consts::PI.sqrt();
        self.lift(f, d1, 2.0 * f + 2.0 * x * d1)
    }
    fn e1_scaled(self) -> Self {
        let x = self.value;
        let g = x.e1_scaled();
        let d1 = g - 1.0 / x;
        self.lift(g, d1, d1 + 1.0 / (x * x))
    }
    fn powf(self, p: f64) -> Self {
        let x = self.value;
        self.lift(
            x.powf(p),
            p * x.powf(p - 1.0),
            p * (p - 1.0) * x.powf(p - 2.0),
        )
    }
    fn powi(self, n: i32) -> Self {
        let x = self.value;
        let nf = n as f64;
        let d1 = if n == 0 { 0.0 } else { nf * x.powi(n - 1) };
        let d2 = if n == 0 || n == 1 {
            0.0
        } else {
            nf * (nf - 1.0) * x.powi(n - 2)
        };
        self.lift(x.powi(n), d1, d2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::DualN;

    type Nested = DualN<DualN<f64, 9>, 9>;

    fn nested_variable(value: f64, k: usize) -> Nested {
        let mut x = Nested::constant(DualN::<f64, 9>::variable(value, k));
        x.im[k] = DualN::<f64, 9>::constant(1.0);
        x
    }

    /// Every operation against the nested dual numbers it replaces.
    #[test]
    fn jet2_matches_nested_second_order_duals() {
        let values = [0.7, 1.3, 0.4, 2.1, 0.9, 1.7, 0.55, 1.1, 0.8];
        let j: Vec<Jet2> = values
            .iter()
            .enumerate()
            .map(|(k, v)| Jet2::variable(*v, k))
            .collect();
        let n: Vec<Nested> = values
            .iter()
            .enumerate()
            .map(|(k, v)| nested_variable(*v, k))
            .collect();
        let fj = |x: &[Jet2]| -> Jet2 {
            let a = x[0] * x[1] + x[2].exp() * x[3].sqrt() - x[4] / x[5];
            let b = (x[6] * x[7]).ln() + x[8].cbrt() * x[0].powf(1.7) + x[1].powi(3) - x[2].atan();
            let c =
                (x[3] * x[4]).asinh() + x[5].erf() + x[6].ln_1p() + x[7].erfcx() + x[8].e1_scaled();
            a * b + c.recip() + (-a).scale(0.3)
        };
        let fn_ = |x: &[Nested]| -> Nested {
            let a = x[0] * x[1] + x[2].exp() * x[3].sqrt() - x[4] / x[5];
            let b = (x[6] * x[7]).ln() + x[8].cbrt() * x[0].powf(1.7) + x[1].powi(3) - x[2].atan();
            let c =
                (x[3] * x[4]).asinh() + x[5].erf() + x[6].ln_1p() + x[7].erfcx() + x[8].e1_scaled();
            a * b + c.recip() + (-a).scale(0.3)
        };
        let (r, s) = (fj(&j), fn_(&n));
        assert!((r.value - s.re.re).abs() < 1e-13 * s.re.re.abs().max(1.0));
        for i in 0..9 {
            assert!(
                (r.gradient[i] - s.im[i].re).abs() < 1e-12 * s.im[i].re.abs().max(1.0),
                "gradient {i}"
            );
            for k in 0..9 {
                assert!(
                    (r.second(i, k) - s.im[i].im[k]).abs() < 1e-11 * s.im[i].im[k].abs().max(1.0),
                    "hessian {i} {k}"
                );
            }
        }
    }
}
