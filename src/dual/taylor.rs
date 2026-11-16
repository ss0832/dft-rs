// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-width truncated Taylor series in one variable.
//!
//! `Taylor<W>` holds the coefficients `c_0 .. c_{W-1}` of `sum_n c_n t^n`
//! and implements [`Scalar`] by the series recurrences, so that a
//! `Scalar`-generic body evaluated on it returns its Taylor coefficients
//! along a path through order `W - 1`. It is a plain copyable value that
//! can cross threads, for bodies that are evaluated in parallel.

use super::Scalar;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Taylor<const W: usize> {
    pub c: [f64; W],
}

impl<const W: usize> Taylor<W> {
    /// The series with the leading coefficients `values`; missing orders are
    /// zero and orders beyond the width are dropped.
    pub fn from_coefficients(values: &[f64]) -> Self {
        let mut c = [0.0; W];
        for (target, value) in c.iter_mut().zip(values) {
            *target = *value;
        }
        Self { c }
    }

    /// Sine and cosine of an arbitrary curved Taylor path, through W-1.
    pub fn sin_cos(self) -> (Self, Self) {
        let mut sine = Self { c: [0.0; W] };
        let mut cosine = sine;
        if W > 0 {
            (sine.c[0], cosine.c[0]) = self.c[0].sin_cos();
        }
        // s'=x' c, c'=-x' s; each right-hand side uses lower orders.
        for n in 1..W {
            for k in 1..=n {
                let derivative = k as f64 * self.c[k] / n as f64;
                sine.c[n] += derivative * cosine.c[n - k];
                cosine.c[n] -= derivative * sine.c[n - k];
            }
        }
        (sine, cosine)
    }

    /// `d/dt`, exact through order `W - 2`.
    fn derivative(self) -> Self {
        let mut c = [0.0; W];
        for n in 0..W.saturating_sub(1) {
            c[n] = (n + 1) as f64 * self.c[n + 1];
        }
        Self { c }
    }

    /// The antiderivative with the constant term `value`.
    fn integral(self, value: f64) -> Self {
        let mut c = [0.0; W];
        if W > 0 {
            c[0] = value;
        }
        for n in 1..W {
            c[n] = self.c[n - 1] / n as f64;
        }
        Self { c }
    }

    /// `f(x)` from `f(x_0) = value` and `f'(x) = slope`.
    fn chain(self, value: f64, slope: Self) -> Self {
        (self.derivative() * slope).integral(value)
    }
}

impl<const W: usize> Add for Taylor<W> {
    type Output = Self;
    #[inline]
    fn add(self, b: Self) -> Self {
        Self {
            c: std::array::from_fn(|n| self.c[n] + b.c[n]),
        }
    }
}

impl<const W: usize> Sub for Taylor<W> {
    type Output = Self;
    #[inline]
    fn sub(self, b: Self) -> Self {
        Self {
            c: std::array::from_fn(|n| self.c[n] - b.c[n]),
        }
    }
}

impl<const W: usize> Mul for Taylor<W> {
    type Output = Self;
    #[inline]
    fn mul(self, b: Self) -> Self {
        Self {
            c: std::array::from_fn(|n| (0..=n).map(|k| self.c[k] * b.c[n - k]).sum()),
        }
    }
}

impl<const W: usize> Div for Taylor<W> {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, b: Self) -> Self {
        self * b.recip()
    }
}

impl<const W: usize> Neg for Taylor<W> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            c: std::array::from_fn(|n| -self.c[n]),
        }
    }
}

impl<const W: usize> Scalar for Taylor<W> {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::from_coefficients(&[v])
    }
    #[inline]
    fn scale(self, x: f64) -> Self {
        Self {
            c: std::array::from_fn(|n| self.c[n] * x),
        }
    }
    #[inline]
    fn value(self) -> f64 {
        self.c[0]
    }
    fn recip(self) -> Self {
        let mut y = [0.0; W];
        y[0] = 1.0 / self.c[0];
        for n in 1..W {
            y[n] = -(1..=n).map(|k| self.c[k] * y[n - k]).sum::<f64>() / self.c[0];
        }
        Self { c: y }
    }
    fn sqrt(self) -> Self {
        self.powf(0.5)
    }
    fn cbrt(self) -> Self {
        if self.value() < 0.0 {
            -(-self).powf(1.0 / 3.0)
        } else {
            self.powf(1.0 / 3.0)
        }
    }
    fn ln(self) -> Self {
        self.chain(self.c[0].ln(), self.recip())
    }
    fn exp(self) -> Self {
        let mut y = [0.0; W];
        y[0] = self.c[0].exp();
        for n in 1..W {
            y[n] = (1..=n)
                .map(|k| k as f64 * self.c[k] * y[n - k])
                .sum::<f64>()
                / n as f64;
        }
        Self { c: y }
    }
    fn atan(self) -> Self {
        self.chain(
            self.c[0].atan(),
            (Self::from_f64(1.0) + self * self).recip(),
        )
    }
    fn asinh(self) -> Self {
        self.chain(
            self.c[0].asinh(),
            (Self::from_f64(1.0) + self * self).sqrt().recip(),
        )
    }
    fn erf(self) -> Self {
        self.chain(
            libm::erf(self.c[0]),
            (-self * self)
                .exp()
                .scale(2.0 / std::f64::consts::PI.sqrt()),
        )
    }
    fn ln_1p(self) -> Self {
        self.chain(libm::log1p(self.c[0]), (Self::from_f64(1.0) + self).recip())
    }
    fn erfcx(self) -> Self {
        // y' = (2 x y - 2 / sqrt(pi)) x'.
        let dx = self.derivative();
        let c = (self * dx).scale(2.0);
        let mut y = [0.0; W];
        y[0] = <f64 as Scalar>::erfcx(self.c[0]);
        for n in 1..W {
            y[n] = ((0..n).map(|k| c.c[k] * y[n - 1 - k]).sum::<f64>()
                - 2.0 / std::f64::consts::PI.sqrt() * dx.c[n - 1])
                / n as f64;
        }
        Self { c: y }
    }
    fn e1_scaled(self) -> Self {
        // y' = (y - 1 / x) x'.
        let dx = self.derivative();
        let forcing = dx * self.recip();
        let mut y = [0.0; W];
        y[0] = <f64 as Scalar>::e1_scaled(self.c[0]);
        for n in 1..W {
            y[n] =
                ((0..n).map(|k| dx.c[k] * y[n - 1 - k]).sum::<f64>() - forcing.c[n - 1]) / n as f64;
        }
        Self { c: y }
    }
    fn powf(self, p: f64) -> Self {
        if p == 0.0 {
            return Self::from_f64(1.0);
        }
        if p == 1.0 {
            return self;
        }
        let mut y = [0.0; W];
        y[0] = self.c[0].powf(p);
        for n in 1..W {
            y[n] = (1..=n)
                .map(|k| ((p + 1.0) * k as f64 - n as f64) * self.c[k] * y[n - k])
                .sum::<f64>()
                / (n as f64 * self.c[0]);
        }
        Self { c: y }
    }
    fn powi(self, n: i32) -> Self {
        let mut power = n.unsigned_abs();
        let mut base = self;
        let mut y = Self::from_f64(1.0);
        while power > 0 {
            if power & 1 == 1 {
                y = y * base;
            }
            power >>= 1;
            if power > 0 {
                base = base * base;
            }
        }
        if n < 0 {
            y.recip()
        } else {
            y
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::DualN;

    /// A body that uses every operation of the trait.
    fn body<S: Scalar>(x: S) -> S {
        let one = S::from_f64(1.0);
        let a = (x * x + one).sqrt() / (x + S::from_f64(2.0));
        let b = (x.scale(0.7)).exp() - x.ln() + x.cbrt() + (-x).cbrt();
        let c = x.atan() * x.asinh() + x.erf() - x.ln_1p();
        let d = x.erfcx() + x.e1_scaled() + x.powf(1.7) + x.powi(-3) + x.recip();
        a * b + c - d
    }

    #[test]
    fn coefficients_are_the_scaled_derivatives_of_nested_duals() {
        type D1 = DualN<f64, 1>;
        type D2 = DualN<D1, 1>;
        type D3 = DualN<D2, 1>;
        let (x0, direction) = (0.83, 0.6);
        // The line x(t) = x0 + direction t in both arithmetics.
        let series = body(Taylor::<4>::from_coefficients(&[x0, direction]));
        let seed = |v: f64, d: f64| -> D3 {
            let inner = |value: f64, slope: f64| D1 {
                re: value,
                im: [slope],
            };
            let middle = |value: D1, slope: D1| D2 {
                re: value,
                im: [slope],
            };
            D3 {
                re: middle(inner(v, d), inner(d, 0.0)),
                im: [middle(inner(d, 0.0), inner(0.0, 0.0))],
            }
        };
        let nested = body(seed(x0, direction));
        let derivatives = [
            nested.re.re.re,
            nested.im[0].re.re,
            nested.im[0].im[0].re,
            nested.im[0].im[0].im[0],
        ];
        let factorial = [1.0, 1.0, 2.0, 6.0];
        for n in 0..4 {
            let expected = derivatives[n] / factorial[n];
            let error = (series.c[n] - expected).abs();
            println!(
                "order {n}: series {:.12e}, nested duals {expected:.12e}, difference {error:e}",
                series.c[n]
            );
            assert!(error < 1e-11 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn products_and_quotients_compose_a_curved_path() {
        // (1 + t)^2 / (1 + t) = 1 + t, and exp(ln(x)) = x on a curved path.
        let x = Taylor::<6>::from_coefficients(&[1.0, 1.0]);
        let ratio = x * x / x;
        let curved = Taylor::<6>::from_coefficients(&[1.3, -0.4, 0.25, 0.1, -0.05, 0.02]);
        let back = curved.ln().exp();
        let cube = curved.cbrt().powi(3);
        for n in 0..6 {
            assert!((ratio.c[n] - x.c[n]).abs() < 1e-14);
            assert!((back.c[n] - curved.c[n]).abs() < 1e-13);
            assert!((cube.c[n] - curved.c[n]).abs() < 1e-13);
        }
    }
}
