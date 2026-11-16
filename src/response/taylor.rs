// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Taylor-normalized forward arithmetic. N coefficients represent orders
//! 0..N-1; nesting represents independent mixed perturbations. No numeric
//! displacement, finite field or differentiation through an SCF iteration
//! is involved. This type implements the existing functional Scalar trait.

use crate::dual::Scalar;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Taylor<S: Scalar, const N: usize> {
    pub coefficients: [S; N],
}

impl<S: Scalar, const N: usize> Taylor<S, N> {
    pub fn constant(value: S) -> Self {
        assert!(N > 0, "Taylor arithmetic requires a constant coefficient");
        let mut coefficients = [S::from_f64(0.0); N];
        coefficients[0] = value;
        Self { coefficients }
    }
    pub fn variable(value: S, direction: S) -> Self {
        assert!(N > 1, "a Taylor variable needs at least two coefficients");
        let mut result = Self::constant(value);
        result.coefficients[1] = direction;
        result
    }
    pub fn derivative(self, order: usize) -> S {
        assert!(order < N);
        let factorial = (1..=order).fold(1.0, |a, b| a * b as f64);
        self.coefficients[order] * S::from_f64(factorial)
    }
    fn derivative_series(self) -> Self {
        let mut result = Self::constant(S::from_f64(0.0));
        for n in 0..N - 1 {
            result.coefficients[n] = self.coefficients[n + 1] * S::from_f64((n + 1) as f64);
        }
        result
    }
    fn integrate(mut derivative: Self, constant: S) -> Self {
        for n in (1..N).rev() {
            derivative.coefficients[n] = derivative.coefficients[n - 1] / S::from_f64(n as f64);
        }
        derivative.coefficients[0] = constant;
        derivative
    }
}

impl<S: Scalar, const N: usize> Add for Taylor<S, N> {
    type Output = Self;
    fn add(mut self, b: Self) -> Self {
        for n in 0..N {
            self.coefficients[n] = self.coefficients[n] + b.coefficients[n];
        }
        self
    }
}
impl<S: Scalar, const N: usize> Sub for Taylor<S, N> {
    type Output = Self;
    fn sub(mut self, b: Self) -> Self {
        for n in 0..N {
            self.coefficients[n] = self.coefficients[n] - b.coefficients[n];
        }
        self
    }
}
impl<S: Scalar, const N: usize> Neg for Taylor<S, N> {
    type Output = Self;
    fn neg(mut self) -> Self {
        for x in &mut self.coefficients {
            *x = -*x;
        }
        self
    }
}
impl<S: Scalar, const N: usize> Mul for Taylor<S, N> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        let mut result = Self::constant(S::from_f64(0.0));
        for n in 0..N {
            for k in 0..=n {
                result.coefficients[n] =
                    result.coefficients[n] + self.coefficients[k] * b.coefficients[n - k];
            }
        }
        result
    }
}
impl<S: Scalar, const N: usize> Div for Taylor<S, N> {
    type Output = Self;
    // Division is multiplication by the reciprocal series.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, b: Self) -> Self {
        self * b.recip()
    }
}

impl<S: Scalar, const N: usize> Scalar for Taylor<S, N> {
    fn from_f64(value: f64) -> Self {
        Self::constant(S::from_f64(value))
    }
    fn value(self) -> f64 {
        self.coefficients[0].value()
    }
    fn recip(self) -> Self {
        let mut result = Self::constant(self.coefficients[0].recip());
        for n in 1..N {
            let mut sum = S::from_f64(0.0);
            for k in 1..=n {
                sum = sum + self.coefficients[k] * result.coefficients[n - k];
            }
            result.coefficients[n] = -sum / self.coefficients[0];
        }
        result
    }
    fn exp(self) -> Self {
        let mut result = Self::constant(self.coefficients[0].exp());
        for n in 1..N {
            let mut sum = S::from_f64(0.0);
            for k in 1..=n {
                sum =
                    sum + S::from_f64(k as f64) * self.coefficients[k] * result.coefficients[n - k];
            }
            result.coefficients[n] = sum / S::from_f64(n as f64);
        }
        result
    }
    fn ln(self) -> Self {
        Self::integrate(self.derivative_series() / self, self.coefficients[0].ln())
    }
    fn ln_1p(self) -> Self {
        Self::integrate(
            self.derivative_series() / (Self::from_f64(1.0) + self),
            self.coefficients[0].ln_1p(),
        )
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
    fn atan(self) -> Self {
        Self::integrate(
            self.derivative_series() / (Self::from_f64(1.0) + self * self),
            self.coefficients[0].atan(),
        )
    }
    fn asinh(self) -> Self {
        Self::integrate(
            self.derivative_series() / (Self::from_f64(1.0) + self * self).sqrt(),
            self.coefficients[0].asinh(),
        )
    }
    fn erf(self) -> Self {
        let derivative = self.derivative_series()
            * (-self * self).exp()
            * Self::from_f64(2.0 / std::f64::consts::PI.sqrt());
        Self::integrate(derivative, self.coefficients[0].erf())
    }
    fn erfcx(self) -> Self {
        // dy/dt = (2xy - 2/sqrt(pi)) dx/dt.
        let mut y = Self::constant(self.coefficients[0].erfcx());
        let dx = self.derivative_series();
        for n in 1..N {
            let dy = (Self::from_f64(2.0) * self * y
                - Self::from_f64(2.0 / std::f64::consts::PI.sqrt()))
                * dx;
            y.coefficients[n] = dy.coefficients[n - 1] / S::from_f64(n as f64);
        }
        y
    }
    fn e1_scaled(self) -> Self {
        // dy/dt = (y - 1/x) dx/dt.
        let mut y = Self::constant(self.coefficients[0].e1_scaled());
        let dx = self.derivative_series();
        let reciprocal = self.recip();
        for n in 1..N {
            let dy = (y - reciprocal) * dx;
            y.coefficients[n] = dy.coefficients[n - 1] / S::from_f64(n as f64);
        }
        y
    }
    fn powf(self, p: f64) -> Self {
        if p == 0.0 {
            return Self::from_f64(1.0);
        }
        if p == 1.0 {
            return self;
        }
        let mut y = Self::constant(self.coefficients[0].powf(p));
        for n in 1..N {
            let mut sum = S::from_f64(0.0);
            for k in 1..=n {
                sum = sum
                    + S::from_f64((p + 1.0) * k as f64 - n as f64)
                        * self.coefficients[k]
                        * y.coefficients[n - k];
            }
            y.coefficients[n] = sum / (S::from_f64(n as f64) * self.coefficients[0]);
        }
        y
    }
    fn powi(self, n: i32) -> Self {
        let mut power = n.unsigned_abs();
        let mut base = self;
        let mut result = Self::from_f64(1.0);
        while power > 0 {
            if power & 1 == 1 {
                result = result * base;
            }
            power >>= 1;
            if power > 0 {
                base = base * base;
            }
        }
        if n < 0 {
            result.recip()
        } else {
            result
        }
    }
}
