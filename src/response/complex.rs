// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex response arithmetic without an additional dependency.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0);
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl From<f64> for Complex {
    fn from(x: f64) -> Self {
        Self::new(x, 0.0)
    }
}
impl Add for Complex {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
}
impl Sub for Complex {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
}
impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
impl Mul for Complex {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
}
impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self::new(self.re * b, self.im * b)
    }
}
impl Div<f64> for Complex {
    type Output = Self;
    fn div(self, b: f64) -> Self {
        Self::new(self.re / b, self.im / b)
    }
}
impl Div for Complex {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        // Scale before squaring: neither |b|^2 nor its reciprocal needs
        // to be representable for the quotient to be representable.
        let scale = b.re.abs().max(b.im.abs());
        let br = b.re / scale;
        let bi = b.im / scale;
        let denominator = br * br + bi * bi;
        let ar = self.re / scale;
        let ai = self.im / scale;
        Self::new(
            (ar * br + ai * bi) / denominator,
            (ai * br - ar * bi) / denominator,
        )
    }
}
impl AddAssign for Complex {
    fn add_assign(&mut self, b: Self) {
        *self = *self + b;
    }
}
impl SubAssign for Complex {
    fn sub_assign(&mut self, b: Self) {
        *self = *self - b;
    }
}

pub(crate) fn dot(a: &[Complex], b: &[Complex]) -> Complex {
    fn add(sum: &mut f64, correction: &mut f64, value: f64) {
        let next = *sum + value;
        *correction += if sum.abs() >= value.abs() {
            (*sum - next) + value
        } else {
            (value - next) + *sum
        };
        *sum = next;
    }
    fn product(sum: &mut f64, correction: &mut f64, a: f64, b: f64) {
        let value = a * b;
        add(sum, correction, value);
        add(sum, correction, a.mul_add(b, -value));
    }
    let (mut real, mut imaginary, mut cr, mut ci) = (0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        product(&mut real, &mut cr, x.re, y.re);
        product(&mut real, &mut cr, x.im, y.im);
        product(&mut imaginary, &mut ci, x.re, y.im);
        product(&mut imaginary, &mut ci, -x.im, y.re);
    }
    Complex::new(real + cr, imaginary + ci)
}

pub(crate) fn norm(a: &[Complex]) -> f64 {
    a.iter().fold(0.0, |s, x| s.hypot(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermitian_dot_preserves_cancelling_sums_and_product_roundoff() {
        let a = [
            Complex::new(1e16, -1e16),
            Complex::new(1.0, 2.0),
            Complex::new(-1e16, 1e16),
        ];
        assert_eq!(dot(&a, &[Complex::ONE; 3]), Complex::new(1.0, -2.0));
        let large = (1_u64 << 27) as f64;
        let a = [Complex::from(large + 1.0), Complex::from(-large)];
        let b = [Complex::from(large - 1.0), Complex::from(large)];
        assert_eq!(dot(&a, &b), Complex::from(-1.0));
        let b = [Complex::new(0.2, -0.3), Complex::new(-0.7, 0.1)];
        assert!((dot(&a, &b) - dot(&b, &a).conj()).abs() < 1e-15);
    }
}
