// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Forward-mode automatic differentiation: a `Scalar` trait abstracting the
//! arithmetic that functional/integral kernels need, and a fixed-width dual
//! number `DualN<S, N>` that nests to arbitrary derivative order
//! (`DualN<DualN<f64, N>, N>` = second derivatives, three levels = third, …).
//!
//! Seeding a variable's `im[k] = 1` extracts ∂/∂x_k; because parameters are
//! passed through the same `Scalar`-generic code path as densities, the same
//! machinery differentiates with respect to functional parameters or basis
//! parameters (project requirement).

use std::ops::{Add, Div, Mul, Neg, Sub};

pub mod jet2;
pub mod taylor;
pub use jet2::Jet2;
pub use taylor::Taylor;

/// Arithmetic closure required by the numeric kernels.
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn from_f64(v: f64) -> Self;
    /// Undifferentiated value (for branching, screening, diagnostics).
    fn value(self) -> f64;
    fn recip(self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn atan(self) -> Self;
    /// asinh(x) = ln(x + √(x²+1)).
    fn asinh(self) -> Self;
    /// Error function (d/dx erf = 2/√π·e^{−x²}).
    fn erf(self) -> Self;
    /// ln(1+x) (d/dx = 1/(1+x)); cancellation-free for small |x|.
    fn ln_1p(self) -> Self;
    /// Scaled complementary error function e^{x²}·erfc(x) for x ≥ 0
    /// (d/dx erfcx = 2x·erfcx − 2/√π); avoids the e^{x²} overflow and
    /// the 1−erf cancellation of the naive form.
    fn erfcx(self) -> Self;
    /// Scaled exponential integral e^{x}·E₁(x) for x > 0
    /// (d/dx = e^x·E₁(x) − 1/x).
    fn e1_scaled(self) -> Self;
    /// Real power with a constant exponent.
    fn powf(self, p: f64) -> Self;
    fn powi(self, n: i32) -> Self;
    /// Product with a plain constant, without forming a constant jet.
    #[inline]
    fn scale(self, x: f64) -> Self {
        self * Self::from_f64(x)
    }
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn scale(self, x: f64) -> Self {
        self * x
    }
    #[inline]
    fn value(self) -> f64 {
        self
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn cbrt(self) -> Self {
        libm::cbrt(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn atan(self) -> Self {
        f64::atan(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        f64::asinh(self)
    }
    #[inline]
    fn erf(self) -> Self {
        libm::erf(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        libm::log1p(self)
    }
    fn erfcx(self) -> Self {
        let x = self;
        if x < 5.0 {
            // erfc still has full relative precision here and e^{x²} ≤ e^{25}.
            f64::exp(x * x) * libm::erfc(x)
        } else {
            // Laplace continued fraction, evaluated bottom-up:
            // erfcx(x) = π^{-1/2}/(x + (1/2)/(x + 1/(x + (3/2)/(x + …)))).
            let mut c = 0.0;
            for k in (1..=60).rev() {
                c = (k as f64 * 0.5) / (x + c);
            }
            1.0 / (std::f64::consts::PI.sqrt() * (x + c))
        }
    }
    fn e1_scaled(self) -> Self {
        let x = self;
        if x <= 1.5 {
            // E₁(x) = −γ − ln x + Σ_{k≥1} (−1)^{k+1} x^k/(k·k!), then ·e^x.
            const EULER_GAMMA: f64 = 0.5772156649015329;
            let mut sum = -EULER_GAMMA - f64::ln(x);
            let mut term = 1.0;
            for k in 1..=40 {
                term *= -x / k as f64;
                sum -= term / k as f64;
            }
            sum * f64::exp(x)
        } else {
            // e^x·E₁(x) = 1/(x + 1/(1 + 1/(x + 2/(1 + 2/(x + 3/(1 + …)))))),
            // evaluated bottom-up.
            let mut t = 0.0;
            for k in (1..=100).rev() {
                let kf = k as f64;
                t = kf / (1.0 + kf / (x + t));
            }
            1.0 / (x + t)
        }
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        f64::powf(self, p)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
}

/// Fixed-width forward dual: value `re` plus `N` tangent components.
#[derive(Clone, Copy, Debug)]
pub struct DualN<S: Scalar, const N: usize> {
    pub re: S,
    pub im: [S; N],
}

impl<S: Scalar, const N: usize> DualN<S, N> {
    #[inline]
    pub fn constant(v: S) -> Self {
        Self {
            re: v,
            im: [S::from_f64(0.0); N],
        }
    }

    /// A variable seeded in tangent slot `k`.
    #[inline]
    pub fn variable(v: S, k: usize) -> Self {
        let mut im = [S::from_f64(0.0); N];
        im[k] = S::from_f64(1.0);
        Self { re: v, im }
    }

    /// Chain rule for a univariate function: given f(re) and f'(re).
    #[inline]
    fn lift(self, f: S, df: S) -> Self {
        let mut im = self.im;
        for c in im.iter_mut() {
            *c = *c * df;
        }
        Self { re: f, im }
    }
}

impl<S: Scalar, const N: usize> Add for DualN<S, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut im = self.im;
        for (a, b) in im.iter_mut().zip(rhs.im) {
            *a = *a + b;
        }
        Self {
            re: self.re + rhs.re,
            im,
        }
    }
}
impl<S: Scalar, const N: usize> Sub for DualN<S, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut im = self.im;
        for (a, b) in im.iter_mut().zip(rhs.im) {
            *a = *a - b;
        }
        Self {
            re: self.re - rhs.re,
            im,
        }
    }
}
impl<S: Scalar, const N: usize> Mul for DualN<S, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut im = [S::from_f64(0.0); N];
        for k in 0..N {
            im[k] = self.im[k] * rhs.re + self.re * rhs.im[k];
        }
        Self {
            re: self.re * rhs.re,
            im,
        }
    }
}
impl<S: Scalar, const N: usize> Div for DualN<S, N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = rhs.re.recip();
        let re = self.re * inv;
        let mut im = [S::from_f64(0.0); N];
        for k in 0..N {
            im[k] = (self.im[k] - re * rhs.im[k]) * inv;
        }
        Self { re, im }
    }
}
impl<S: Scalar, const N: usize> Neg for DualN<S, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut im = self.im;
        for a in im.iter_mut() {
            *a = -*a;
        }
        Self { re: -self.re, im }
    }
}

impl<S: Scalar, const N: usize> Scalar for DualN<S, N> {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(S::from_f64(v))
    }
    #[inline]
    fn scale(self, x: f64) -> Self {
        let mut im = self.im;
        for c in im.iter_mut() {
            *c = c.scale(x);
        }
        Self {
            re: self.re.scale(x),
            im,
        }
    }
    #[inline]
    fn value(self) -> f64 {
        self.re.value()
    }
    #[inline]
    fn recip(self) -> Self {
        let inv = self.re.recip();
        self.lift(inv, -inv * inv)
    }
    #[inline]
    fn sqrt(self) -> Self {
        let r = self.re.sqrt();
        self.lift(r, r.recip() * S::from_f64(0.5))
    }
    #[inline]
    fn cbrt(self) -> Self {
        let r = self.re.cbrt();
        // d/dx x^{1/3} = (1/3) x^{-2/3} = 1/(3 r²).
        self.lift(r, (S::from_f64(3.0) * r * r).recip())
    }
    #[inline]
    fn ln(self) -> Self {
        self.lift(self.re.ln(), self.re.recip())
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.lift(e, e)
    }
    #[inline]
    fn atan(self) -> Self {
        let one = S::from_f64(1.0);
        self.lift(self.re.atan(), (one + self.re * self.re).recip())
    }
    #[inline]
    fn asinh(self) -> Self {
        let one = S::from_f64(1.0);
        self.lift(self.re.asinh(), (self.re * self.re + one).sqrt().recip())
    }
    #[inline]
    fn erf(self) -> Self {
        let two_over_sqrt_pi = S::from_f64(2.0 / std::f64::consts::PI.sqrt());
        self.lift(
            self.re.erf(),
            two_over_sqrt_pi * (-(self.re * self.re)).exp(),
        )
    }
    #[inline]
    fn ln_1p(self) -> Self {
        let one = S::from_f64(1.0);
        self.lift(self.re.ln_1p(), (one + self.re).recip())
    }
    #[inline]
    fn erfcx(self) -> Self {
        let f = self.re.erfcx();
        let two_over_sqrt_pi = S::from_f64(2.0 / std::f64::consts::PI.sqrt());
        self.lift(f, S::from_f64(2.0) * self.re * f - two_over_sqrt_pi)
    }
    #[inline]
    fn e1_scaled(self) -> Self {
        let f = self.re.e1_scaled();
        self.lift(f, f - self.re.recip())
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        if p.fract() == 0.0 && p >= i32::MIN as f64 && p <= i32::MAX as f64 {
            return self.powi(p as i32);
        }
        let f = self.re.powf(p);
        self.lift(f, S::from_f64(p) * self.re.powf(p - 1.0))
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        // Polynomial multiplication remains analytic at zero at every nesting
        // depth. A lifted n*x^(n-1) would form 0/0 for a constant power and
        // propagate NaNs into otherwise regular mixed derivatives.
        let mut power = n.unsigned_abs();
        let mut result = Self::from_f64(1.0);
        let mut base = if n < 0 { self.recip() } else { self };
        while power != 0 {
            if power & 1 != 0 {
                result = result * base;
            }
            power >>= 1;
            if power != 0 {
                base = base * base;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type D2 = DualN<f64, 2>;
    type DD2 = DualN<DualN<f64, 2>, 2>;

    fn test_fn<S: Scalar>(x: S, y: S) -> S {
        // f = x²·ln(y) + exp(x)·√y + atan(x/y) + y^{1.7} + x^{1/3}
        x * x * y.ln() + x.exp() * y.sqrt() + (x / y).atan() + y.powf(1.7) + x.cbrt()
    }

    /// First derivatives vs central differences.
    #[test]
    fn first_derivatives_match_fd() {
        let (x0, y0) = (0.8, 1.3);
        let f = test_fn(D2::variable(x0, 0), D2::variable(y0, 1));
        let h = 1e-6;
        let dfdx = (test_fn(x0 + h, y0) - test_fn(x0 - h, y0)) / (2.0 * h);
        let dfdy = (test_fn(x0, y0 + h) - test_fn(x0, y0 - h)) / (2.0 * h);
        assert!((f.im[0] - dfdx).abs() < 1e-8, "{} vs {dfdx}", f.im[0]);
        assert!((f.im[1] - dfdy).abs() < 1e-8, "{} vs {dfdy}", f.im[1]);
    }

    /// Nested duals: full Hessian vs finite differences of the dual gradient
    /// (which is itself exact) — tight tolerance.
    #[test]
    fn second_derivatives_match_fd_of_gradient() {
        let (x0, y0) = (0.8, 1.3);
        // Second-order seeding: inner duals carry d/dx in slot 0, d/dy in
        // slot 1; the outer tangents mirror the same seeding.
        let x = DD2 {
            re: DualN::variable(x0, 0),
            im: [DualN::constant(1.0), DualN::constant(0.0)],
        };
        let y = DD2 {
            re: DualN::variable(y0, 1),
            im: [DualN::constant(0.0), DualN::constant(1.0)],
        };
        let f = test_fn(x, y);
        // f.im[a].im[b] = ∂²f/∂a∂b.
        let h = 1e-6;
        let grad = |x0: f64, y0: f64| {
            let g = test_fn(D2::variable(x0, 0), D2::variable(y0, 1));
            [g.im[0], g.im[1]]
        };
        let gxp = grad(x0 + h, y0);
        let gxm = grad(x0 - h, y0);
        let gyp = grad(x0, y0 + h);
        let gym = grad(x0, y0 - h);
        let hess_fd = [
            [(gxp[0] - gxm[0]) / (2.0 * h), (gyp[0] - gym[0]) / (2.0 * h)],
            [(gxp[1] - gxm[1]) / (2.0 * h), (gyp[1] - gym[1]) / (2.0 * h)],
        ];
        for a in 0..2 {
            for b in 0..2 {
                let got = f.im[a].im[b];
                assert!(
                    (got - hess_fd[a][b]).abs() < 1e-7,
                    "H[{a}][{b}] = {got} vs {}",
                    hess_fd[a][b]
                );
            }
        }
        // Symmetry of mixed partials.
        assert!((f.im[0].im[1] - f.im[1].im[0]).abs() < 1e-12);
    }
}
