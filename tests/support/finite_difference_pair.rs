// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Stable arithmetic for two finite, distinct function evaluations.
//! `mean` and `half_difference` represent (f(+h)+f(-h))/2 and
//! (f(+h)-f(-h))/2. This carries finite differences, not infinitesimal AD.
//! Elementary divided differences avoid subtracting a large unchanged local
//! potential when testing tiny nonlocal cross-grid couplings.
use dft_rs::dual::Scalar;
use dft_rs::skala::Number;
use std::ops::{Add, Div, Mul, Neg, Sub};
#[derive(Clone, Copy, Debug)]
pub struct FinitePair {
    pub mean: f64,
    pub half_difference: f64,
}
impl FinitePair {
    pub fn constant(x: f64) -> Self {
        Self {
            mean: x,
            half_difference: 0.0,
        }
    }
    pub fn shifted(x: f64, delta: f64) -> Self {
        Self {
            mean: x,
            half_difference: delta,
        }
    }
    pub fn plus(self) -> f64 {
        self.mean + self.half_difference
    }
    pub fn minus(self) -> f64 {
        self.mean - self.half_difference
    }
}
impl Add for FinitePair {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            mean: self.mean + b.mean,
            half_difference: self.half_difference + b.half_difference,
        }
    }
}
impl Sub for FinitePair {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            mean: self.mean - b.mean,
            half_difference: self.half_difference - b.half_difference,
        }
    }
}
impl Neg for FinitePair {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            mean: -self.mean,
            half_difference: -self.half_difference,
        }
    }
}
fn multiply_pair(a: FinitePair, b: FinitePair) -> FinitePair {
    FinitePair {
        mean: a.mean * b.mean + a.half_difference * b.half_difference,
        half_difference: a.mean * b.half_difference + a.half_difference * b.mean,
    }
}
impl Mul for FinitePair {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        multiply_pair(self, b)
    }
}
impl Div for FinitePair {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        multiply_pair(self, b.recip())
    }
}
impl Number for FinitePair {
    fn is_zero(self) -> bool {
        self.mean == 0.0 && self.half_difference == 0.0
    }
}
impl Scalar for FinitePair {
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }
    fn value(self) -> f64 {
        self.mean
    }
    fn recip(self) -> Self {
        let ratio = self.half_difference / self.mean;
        let factor = self.mean.recip() / (1.0 - ratio * ratio);
        Self {
            mean: factor,
            half_difference: -ratio * factor,
        }
    }
    fn sqrt(self) -> Self {
        let a = self.plus().sqrt();
        let b = self.minus().sqrt();
        Self {
            mean: 0.5 * (a + b),
            half_difference: self.half_difference / (a + b),
        }
    }
    fn cbrt(self) -> Self {
        let a = self.plus().cbrt();
        let b = self.minus().cbrt();
        Self {
            mean: 0.5 * (a + b),
            half_difference: self.half_difference / (a * a + a * b + b * b),
        }
    }
    fn ln(self) -> Self {
        assert!(self.plus() > 0.0 && self.minus() > 0.0);
        let r = self.half_difference / self.mean;
        Self {
            mean: self.mean.ln() + 0.5 * (-r * r).ln_1p(),
            half_difference: 0.5 * (r.ln_1p() - (-r).ln_1p()),
        }
    }
    fn ln_1p(self) -> Self {
        let r = self.half_difference / (1.0 + self.mean);
        Self {
            mean: self.mean.ln_1p() + 0.5 * (-r * r).ln_1p(),
            half_difference: 0.5 * (r.ln_1p() - (-r).ln_1p()),
        }
    }
    fn exp(self) -> Self {
        let e = self.mean.exp();
        Self {
            mean: e * self.half_difference.cosh(),
            half_difference: e * self.half_difference.sinh(),
        }
    }
    fn powf(self, p: f64) -> Self {
        self.ln().scale(p).exp()
    }
    fn powi(self, n: i32) -> Self {
        if n < 0 {
            return self.recip().powi(-n);
        }
        let mut a = self;
        let mut result = Self::constant(1.0);
        let mut power = n as u32;
        while power != 0 {
            if power & 1 != 0 {
                result = result * a
            }
            power >>= 1;
            if power != 0 {
                a = a * a
            }
        }
        result
    }
    fn scale(self, x: f64) -> Self {
        Self {
            mean: self.mean * x,
            half_difference: self.half_difference * x,
        }
    }
    fn atan(self) -> Self {
        panic!("atan is outside this finite-pair network oracle")
    }
    fn asinh(self) -> Self {
        panic!("asinh is outside this finite-pair network oracle")
    }
    fn erf(self) -> Self {
        panic!("erf is outside this finite-pair network oracle")
    }
    fn erfcx(self) -> Self {
        panic!("erfcx is outside this finite-pair network oracle")
    }
    fn e1_scaled(self) -> Self {
        panic!("e1_scaled is outside this finite-pair network oracle")
    }
}
