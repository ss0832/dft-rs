// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Runtime-order Taylor arithmetic for Scalar-generic functional bodies.
//! Scalars are copyable handles to a scoped, thread-local coefficient arena.
//! The arena is removed on normal return and on panic; generation checks
//! prevent stale or cross-thread handles from referring to another series.
//!
//! The arena stores every coefficient array in one contiguous buffer, and
//! each operation writes its result directly into a new slot at the end of
//! that buffer, so the arithmetic of the inner loops allocates nothing
//! beyond the amortized growth of the buffer.

use crate::dual::Scalar;
use crate::{DftError, Result};
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering};

struct Arena {
    generation: u64,
    width: usize,
    /// Slot `s` holds the coefficients `values[s * width..(s + 1) * width]`.
    values: Vec<f64>,
}

impl Arena {
    /// Offset of the coefficients of `scalar`.
    fn offset(&self, scalar: SeriesScalar) -> usize {
        assert_eq!(
            scalar.generation, self.generation,
            "stale or cross-thread SeriesScalar"
        );
        scalar.slot * self.width
    }

    /// Append a slot filled by `fill` from the earlier slots.
    fn push(&mut self, fill: impl FnOnce(&[f64], &mut [f64])) -> SeriesScalar {
        let start = self.values.len();
        self.values.resize(start + self.width, 0.0);
        let (earlier, slot) = self.values.split_at_mut(start);
        fill(earlier, slot);
        SeriesScalar {
            generation: self.generation,
            slot: start / self.width,
        }
    }
}

thread_local! { static ACTIVE: RefCell<Option<Arena>> = const { RefCell::new(None) }; }
static GENERATION: AtomicU64 = AtomicU64::new(1);

/// Initial capacity of the arena, in coefficient arrays.
const INITIAL_SLOTS: usize = 256;

fn with_arena<R>(action: impl FnOnce(&mut Arena) -> R) -> R {
    ACTIVE.with(|cell| {
        let mut state = cell.borrow_mut();
        let arena = state
            .as_mut()
            .expect("SeriesScalar requires evaluate_series");
        action(arena)
    })
}

#[derive(Clone, Copy, Debug)]
pub struct SeriesScalar {
    generation: u64,
    slot: usize,
}

impl SeriesScalar {
    fn insert(coefficients: &[f64]) -> Self {
        with_arena(|arena| {
            assert_eq!(coefficients.len(), arena.width);
            arena.push(|_, slot| slot.copy_from_slice(coefficients))
        })
    }
    fn coefficients(self) -> Vec<f64> {
        ACTIVE.with(|cell| {
            let state = cell.borrow();
            let arena = state
                .as_ref()
                .expect("SeriesScalar has left its evaluation scope");
            let start = arena.offset(self);
            arena.values[start..start + arena.width].to_vec()
        })
    }
    fn binary(self, b: Self, f: impl FnOnce(&[f64], &[f64], &mut [f64])) -> Self {
        with_arena(|arena| {
            let (x, y, width) = (arena.offset(self), arena.offset(b), arena.width);
            arena.push(|earlier, slot| f(&earlier[x..x + width], &earlier[y..y + width], slot))
        })
    }
    fn unary(self, f: impl FnOnce(&[f64], &mut [f64])) -> Self {
        with_arena(|arena| {
            let (x, width) = (arena.offset(self), arena.width);
            arena.push(|earlier, slot| f(&earlier[x..x + width], slot))
        })
    }
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        ACTIVE.with(|cell| *cell.borrow_mut() = None);
    }
}

/// Evaluate a single Scalar-generic expression at any supplied Taylor order.
/// Inputs and returned values use factorial-normalized coefficients. Calls
/// on independent threads are independent; recursive scopes are rejected.
pub fn evaluate_series(
    inputs: &[Vec<f64>],
    expression: impl FnOnce(&[SeriesScalar]) -> SeriesScalar,
) -> Result<Vec<f64>> {
    Ok(evaluate_series_many(inputs, |v| vec![expression(v)])?.remove(0))
}

/// Evaluate several outputs in one scope, including tangents nested over
/// the Taylor scalar. All outputs share the input normalization and order.
pub fn evaluate_series_many(
    inputs: &[Vec<f64>],
    expression: impl FnOnce(&[SeriesScalar]) -> Vec<SeriesScalar>,
) -> Result<Vec<Vec<f64>>> {
    let width = inputs.first().map_or(0, Vec::len);
    if width == 0
        || inputs
            .iter()
            .any(|v| v.len() != width || v.iter().any(|c| !c.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "Taylor inputs require equal nonempty finite coefficient arrays".into(),
        ));
    }
    let occupied = ACTIVE.with(|cell| cell.borrow().is_some());
    if occupied {
        return Err(DftError::InvalidInput(
            "recursive runtime Taylor evaluation is not allowed".into(),
        ));
    }
    ACTIVE.with(|cell| {
        *cell.borrow_mut() = Some(Arena {
            generation: GENERATION.fetch_add(1, Ordering::Relaxed),
            width,
            values: Vec::with_capacity(width * INITIAL_SLOTS.max(inputs.len())),
        })
    });
    let reset = Reset;
    let variables: Vec<_> = inputs.iter().map(|v| SeriesScalar::insert(v)).collect();
    let result: Vec<_> = expression(&variables)
        .into_iter()
        .map(SeriesScalar::coefficients)
        .collect();
    drop(reset);
    if result.iter().flatten().any(|c| !c.is_finite()) {
        return Err(DftError::InvalidInput(
            "functional Taylor expansion is non-finite at the supplied point".into(),
        ));
    }
    Ok(result)
}

/// Evaluate fallible analytic kernels without leaking the coefficient arena.
pub fn evaluate_series_many_result(
    inputs: &[Vec<f64>],
    expression: impl FnOnce(&[SeriesScalar]) -> Result<Vec<SeriesScalar>>,
) -> Result<Vec<Vec<f64>>> {
    let mut failure = None;
    let result = evaluate_series_many(inputs, |v| match expression(v) {
        Ok(values) => values,
        Err(error) => {
            failure = Some(error);
            vec![SeriesScalar::from_f64(0.0)]
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    result
}

fn product_into(a: &[f64], b: &[f64], out: &mut [f64]) {
    for (n, value) in out.iter_mut().enumerate() {
        *value = (0..=n).map(|k| a[k] * b[n - k]).sum();
    }
}
fn product(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    product_into(a, b, &mut out);
    out
}
fn reciprocal_into(x: &[f64], y: &mut [f64]) {
    y[0] = 1.0 / x[0];
    for n in 1..x.len() {
        y[n] = -(1..=n).map(|k| x[k] * y[n - k]).sum::<f64>() / x[0];
    }
}
pub(crate) fn reciprocal(x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    reciprocal_into(x, &mut y);
    y
}
fn derivative(x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for n in 0..x.len() - 1 {
        y[n] = (n + 1) as f64 * x[n + 1];
    }
    y
}
fn integral(x: &[f64], value: f64) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    y[0] = value;
    for n in 1..x.len() {
        y[n] = x[n - 1] / n as f64;
    }
    y
}

impl Add for SeriesScalar {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        self.binary(b, |a, b, out| {
            for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
                *o = a + b;
            }
        })
    }
}
impl Sub for SeriesScalar {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self.binary(b, |a, b, out| {
            for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
                *o = a - b;
            }
        })
    }
}
impl Mul for SeriesScalar {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        self.binary(b, product_into)
    }
}
impl Div for SeriesScalar {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        self.binary(b, |a, b, out| product_into(a, &reciprocal(b), out))
    }
}
impl Neg for SeriesScalar {
    type Output = Self;
    fn neg(self) -> Self {
        self.unary(|x, out| {
            for (o, v) in out.iter_mut().zip(x) {
                *o = -v;
            }
        })
    }
}

impl Scalar for SeriesScalar {
    fn from_f64(value: f64) -> Self {
        with_arena(|arena| arena.push(|_, slot| slot[0] = value))
    }
    fn value(self) -> f64 {
        ACTIVE.with(|cell| {
            let state = cell.borrow();
            let arena = state
                .as_ref()
                .expect("SeriesScalar has left its evaluation scope");
            arena.values[arena.offset(self)]
        })
    }
    fn recip(self) -> Self {
        self.unary(reciprocal_into)
    }
    fn exp(self) -> Self {
        self.unary(|x, y| {
            y[0] = x[0].exp();
            for n in 1..x.len() {
                y[n] = (1..=n).map(|k| k as f64 * x[k] * y[n - k]).sum::<f64>() / n as f64;
            }
        })
    }
    fn ln(self) -> Self {
        self.unary(|x, out| {
            out.copy_from_slice(&integral(
                &product(&derivative(x), &reciprocal(x)),
                x[0].ln(),
            ))
        })
    }
    fn ln_1p(self) -> Self {
        self.unary(|x, out| {
            let mut denominator = x.to_vec();
            denominator[0] += 1.0;
            out.copy_from_slice(&integral(
                &product(&derivative(x), &reciprocal(&denominator)),
                x[0].ln_1p(),
            ))
        })
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
        let derivative =
            Self::insert(&derivative(&self.coefficients())) / (Self::from_f64(1.0) + self * self);
        Self::insert(&integral(&derivative.coefficients(), self.value().atan()))
    }
    fn asinh(self) -> Self {
        let derivative = Self::insert(&derivative(&self.coefficients()))
            / (Self::from_f64(1.0) + self * self).sqrt();
        Self::insert(&integral(&derivative.coefficients(), self.value().asinh()))
    }
    fn erf(self) -> Self {
        let derivative = Self::insert(&derivative(&self.coefficients()))
            * (-self * self).exp()
            * Self::from_f64(2.0 / std::f64::consts::PI.sqrt());
        Self::insert(&integral(
            &derivative.coefficients(),
            libm::erf(self.value()),
        ))
    }
    fn erfcx(self) -> Self {
        self.unary(|x, y| {
            let dx = derivative(x);
            let c: Vec<_> = product(x, &dx).iter().map(|v| 2.0 * v).collect();
            y[0] = <f64 as Scalar>::erfcx(x[0]);
            for n in 1..x.len() {
                y[n] = ((0..n).map(|k| c[k] * y[n - 1 - k]).sum::<f64>()
                    - 2.0 / std::f64::consts::PI.sqrt() * dx[n - 1])
                    / n as f64;
            }
        })
    }
    fn e1_scaled(self) -> Self {
        self.unary(|x, y| {
            let dx = derivative(x);
            let forcing = product(&dx, &reciprocal(x));
            y[0] = <f64 as Scalar>::e1_scaled(x[0]);
            for n in 1..x.len() {
                y[n] =
                    ((0..n).map(|k| dx[k] * y[n - 1 - k]).sum::<f64>() - forcing[n - 1]) / n as f64;
            }
        })
    }
    fn powf(self, p: f64) -> Self {
        if p == 0.0 {
            return Self::from_f64(1.0);
        }
        if p == 1.0 {
            return self;
        }
        // Integral powers are polynomials at a zero constant coefficient.
        // The general recurrence divides by that coefficient and would turn
        // a smooth spin-balanced term such as zeta^6 into 0/0.
        if p.fract() == 0.0 && p >= i32::MIN as f64 && p <= i32::MAX as f64 {
            return self.powi(p as i32);
        }
        self.unary(|x, y| {
            y[0] = x[0].powf(p);
            for n in 1..x.len() {
                y[n] = (1..=n)
                    .map(|k| ((p + 1.0) * k as f64 - n as f64) * x[k] * y[n - k])
                    .sum::<f64>()
                    / (n as f64 * x[0]);
            }
        })
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
