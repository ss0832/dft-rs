// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex pair coefficients with rounding remainders kept in the physical
//! recurrence and its independent equation check.

use super::{AmplitudeNumber, FockCoefficient};
use crate::integrals::london::ComplexMatrix;
use crate::linalg::AccurateSum;
use crate::response::complex::Complex;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

#[derive(Clone, Copy, Debug)]
pub(super) struct RetainedComplex {
    high: Complex,
    low: Complex,
}

impl RetainedComplex {
    fn from_sums(real: AccurateSum, imaginary: AccurateSum) -> Self {
        let (real, real_low) = real.parts();
        let (imaginary, imaginary_low) = imaginary.parts();
        Self {
            high: Complex::new(real, imaginary),
            low: Complex::new(real_low, imaginary_low),
        }
    }
}

impl From<Complex> for RetainedComplex {
    fn from(high: Complex) -> Self {
        Self {
            high,
            low: Complex::ZERO,
        }
    }
}

impl From<f64> for RetainedComplex {
    fn from(value: f64) -> Self {
        Complex::from(value).into()
    }
}

impl From<RetainedComplex> for Complex {
    fn from(value: RetainedComplex) -> Self {
        value.high + value.low
    }
}

impl Add for RetainedComplex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut real = AccurateSum::default();
        let mut imaginary = AccurateSum::default();
        for part in [self.high, self.low, other.high, other.low] {
            real.add_product(part.re, 1.0);
            imaginary.add_product(part.im, 1.0);
        }
        Self::from_sums(real, imaginary)
    }
}

impl Sub for RetainedComplex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + Self {
            high: -other.high,
            low: -other.low,
        }
    }
}

impl AddAssign for RetainedComplex {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl SubAssign for RetainedComplex {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Mul for RetainedComplex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut real = AccurateSum::default();
        let mut imaginary = AccurateSum::default();
        for a in [self.high, self.low] {
            for b in [other.high, other.low] {
                real.add_product(a.re, b.re);
                real.add_product(-a.im, b.im);
                imaginary.add_product(a.re, b.im);
                imaginary.add_product(a.im, b.re);
            }
        }
        Self::from_sums(real, imaginary)
    }
}

impl Div<f64> for RetainedComplex {
    type Output = Self;

    fn div(self, denominator: f64) -> Self {
        let quotient = |high, low| {
            let estimate: f64 = high / denominator;
            let mut remainder = AccurateSum::default();
            remainder.add_product(-estimate, denominator);
            remainder.add_product(high, 1.0);
            remainder.add_product(low, 1.0);
            let mut sum = AccurateSum::default();
            sum.add_product(estimate, 1.0);
            sum.add_product(remainder.value() / denominator, 1.0);
            sum
        };
        Self::from_sums(
            quotient(self.high.re, self.low.re),
            quotient(self.high.im, self.low.im),
        )
    }
}

impl AmplitudeNumber for RetainedComplex {
    fn conjugate(self) -> Self {
        Self {
            high: self.high.conj(),
            low: self.low.conj(),
        }
    }

    fn magnitude(self) -> f64 {
        Complex::from(self).abs()
    }

    fn finite(self) -> bool {
        self.high.is_finite() && self.low.is_finite()
    }
}

impl FockCoefficient<RetainedComplex> for ComplexMatrix {
    fn rows(&self) -> usize {
        self.re.rows
    }

    fn cols(&self) -> usize {
        self.re.cols
    }

    fn element(&self, row: usize, col: usize) -> RetainedComplex {
        FockCoefficient::<Complex>::element(self, row, col).into()
    }

    fn finite(&self) -> bool {
        FockCoefficient::<Complex>::finite(self)
    }
}
