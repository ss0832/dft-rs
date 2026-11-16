// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Scalar-generic Cartesian Gaussian primitives for analytic jets of
//! centres, exponents, nuclear charges, and range-separation parameters.
//! Contraction, spherical transformation, moving grids, and response
//! equations must propagate the resulting jets in their physical driver.

use crate::dual::Scalar;
use crate::{DftError, Result};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Primitive<S: Scalar> {
    pub angular: [usize; 3],
    pub exponent: S,
    pub center: [S; 3],
}

#[derive(Clone, Copy, Debug)]
pub enum GaussianKernel<S: Scalar> {
    Coulomb,
    LongRange { omega: S },
    ShortRange { omega: S },
}

fn constant<S: Scalar>(x: f64) -> S {
    S::from_f64(x)
}
fn validate<S: Scalar>(p: &Primitive<S>) -> Result<()> {
    if !p.exponent.value().is_finite()
        || p.exponent.value() <= 0.0
        || p.center.iter().any(|x| !x.value().is_finite())
    {
        return Err(DftError::InvalidInput(
            "Gaussian primitive requires finite centres and positive finite exponent".into(),
        ));
    }
    Ok(())
}
fn norm2<S: Scalar>(v: [S; 3]) -> S {
    v.into_iter().fold(constant::<S>(0.0), |s, x| s + x * x)
}

/// Stable base Boys values at arbitrary order. A logarithmic positive
/// highest-order series and downward recurrence avoid both cancellation
/// and underflow of the seed. Upward recurrence is used only well below
/// its instability region.
pub fn boys_values(nmax: usize, x: f64) -> Result<Vec<f64>> {
    if !x.is_finite() || x < 0.0 || nmax > usize::MAX / 2 - 2 {
        return Err(DftError::InvalidInput(
            "invalid Boys argument or order overflow".into(),
        ));
    }
    if x == 0.0 {
        return Ok((0..=nmax).map(|n| 1.0 / (2 * n + 1) as f64).collect());
    }
    if x > 50.0 && x > 2.0 * (nmax as f64 + 1.0) {
        let mut out = vec![0.0; nmax + 1];
        out[0] = 0.5 * (PI / x).sqrt();
        let exponential = (-x).exp();
        for n in 1..=nmax {
            out[n] = ((2 * n - 1) as f64 * out[n - 1] - exponential) / (2.0 * x);
        }
        return Ok(out);
    }
    let log_add = |a: f64, b: f64| {
        let m = a.max(b);
        m + ((a.min(b) - m).exp()).ln_1p()
    };
    let mut log_term = -((2 * nmax + 1) as f64).ln();
    let mut log_sum = log_term;
    let mut k = 1usize;
    loop {
        let denominator = 2.0 * nmax as f64 + 2.0 * k as f64 + 1.0;
        let ratio = 2.0 * x / denominator;
        log_term += (2.0_f64.ln() + x.ln()) - denominator.ln();
        log_sum = log_add(log_sum, log_term);
        if ratio < 1.0 && (log_term - log_sum).exp() < 1e-17 {
            break;
        }
        k = k
            .checked_add(1)
            .ok_or_else(|| DftError::InvalidInput("Boys series index overflow".into()))?;
    }
    let highest_log = log_sum - x;
    if highest_log >= f64::MIN_POSITIVE.ln() {
        let mut out = vec![0.0; nmax + 1];
        out[nmax] = highest_log.exp();
        let exponential = (-x).exp();
        for n in (0..nmax).rev() {
            out[n] = (2.0 * x * out[n + 1] + exponential) / (2 * n + 1) as f64;
        }
        return Ok(out);
    }
    let mut logs = vec![0.0; nmax + 1];
    logs[nmax] = log_sum - x;
    for n in (0..nmax).rev() {
        logs[n] = log_add(2.0_f64.ln() + x.ln() + logs[n + 1], -x) - ((2 * n + 1) as f64).ln();
    }
    Ok(logs.into_iter().map(f64::exp).collect())
}

/// Exact composition through `order` in a nilpotent AD algebra, using
/// d^r F_n(t)/dt^r=(-1)^r F_(n+r)(t). The requested order must include all
/// nested dual levels as well as the Taylor order. For ordinary f64 only
/// the base value contributes. No finite parameter increment is taken.
pub fn boys_jet<S: Scalar>(n: usize, t: S, order: usize) -> Result<S> {
    let highest = n
        .checked_add(order)
        .ok_or_else(|| DftError::InvalidInput("Boys derivative order overflow".into()))?;
    let values = boys_values(highest, t.value())?;
    let offset = t - constant::<S>(t.value());
    let mut term = constant::<S>(1.0);
    let mut sum = constant::<S>(values[n]);
    for r in 1..=order {
        term = term * (-offset) / constant::<S>(r as f64);
        sum = sum + term * constant::<S>(values[n + r]);
    }
    Ok(sum)
}

struct EAxis<S: Scalar> {
    imax: usize,
    jmax: usize,
    tdim: usize,
    data: Vec<S>,
}
impl<S: Scalar> EAxis<S> {
    fn new(imax: usize, jmax: usize, delta: S, a: S, b: S) -> Self {
        let p = a + b;
        let mu = a * b / p;
        let pa = -b * delta / p;
        let pb = a * delta / p;
        let tdim = imax + jmax + 1;
        let mut data = vec![constant::<S>(0.0); (imax + 1) * (jmax + 1) * tdim];
        let index = |i: usize, j: usize, t: usize| (i * (jmax + 1) + j) * tdim + t;
        data[index(0, 0, 0)] = (-mu * delta * delta).exp();
        for i in 0..imax {
            for t in 0..=i + 1 {
                let mut value = pa * data[index(i, 0, t)];
                if t > 0 {
                    value = value + data[index(i, 0, t - 1)] / (constant::<S>(2.0) * p);
                }
                if t < i {
                    value = value + constant::<S>((t + 1) as f64) * data[index(i, 0, t + 1)];
                }
                data[index(i + 1, 0, t)] = value;
            }
        }
        for i in 0..=imax {
            for j in 0..jmax {
                for t in 0..=i + j + 1 {
                    let mut value = pb * data[index(i, j, t)];
                    if t > 0 {
                        value = value + data[index(i, j, t - 1)] / (constant::<S>(2.0) * p);
                    }
                    if t < i + j {
                        value = value + constant::<S>((t + 1) as f64) * data[index(i, j, t + 1)];
                    }
                    data[index(i, j + 1, t)] = value;
                }
            }
        }
        Self {
            imax,
            jmax,
            tdim,
            data,
        }
    }
    fn get(&self, i: usize, j: usize, t: usize) -> S {
        if i > self.imax || j > self.jmax || t > i + j {
            constant::<S>(0.0)
        } else {
            self.data[(i * (self.jmax + 1) + j) * self.tdim + t]
        }
    }
}

fn axes<S: Scalar>(a: &Primitive<S>, b: &Primitive<S>, extra: usize) -> [EAxis<S>; 3] {
    std::array::from_fn(|axis| {
        EAxis::new(
            a.angular[axis],
            b.angular[axis] + extra,
            a.center[axis] - b.center[axis],
            a.exponent,
            b.exponent,
        )
    })
}

pub fn overlap<S: Scalar>(a: &Primitive<S>, b: &Primitive<S>) -> Result<S> {
    validate(a)?;
    validate(b)?;
    let e = axes(a, b, 0);
    let factor = (constant::<S>(PI) / (a.exponent + b.exponent)).sqrt();
    Ok((0..3).fold(constant::<S>(1.0), |v, axis| {
        v * factor * e[axis].get(a.angular[axis], b.angular[axis], 0)
    }))
}
pub fn kinetic<S: Scalar>(a: &Primitive<S>, b: &Primitive<S>) -> Result<S> {
    validate(a)?;
    validate(b)?;
    let e = axes(a, b, 2);
    let factor = (constant::<S>(PI) / (a.exponent + b.exponent)).sqrt();
    let s: [S; 3] =
        std::array::from_fn(|axis| factor * e[axis].get(a.angular[axis], b.angular[axis], 0));
    let t: [S; 3] = std::array::from_fn(|axis| {
        let la = a.angular[axis];
        let lb = b.angular[axis];
        let mut value = b.exponent * constant::<S>((2 * lb + 1) as f64) * s[axis]
            - constant::<S>(2.0) * b.exponent * b.exponent * factor * e[axis].get(la, lb + 2, 0);
        if lb >= 2 {
            value = value
                - constant::<S>(0.5 * (lb * (lb - 1)) as f64) * factor * e[axis].get(la, lb - 2, 0);
        }
        value
    });
    Ok(t[0] * s[1] * s[2] + s[0] * t[1] * s[2] + s[0] * s[1] * t[2])
}

fn adjusted_boys<S: Scalar>(
    nmax: usize,
    rho: S,
    t: S,
    kernel: GaussianKernel<S>,
    order: usize,
) -> Result<Vec<S>> {
    let full = || {
        (0..=nmax)
            .map(|n| boys_jet(n, t, order))
            .collect::<Result<Vec<_>>>()
    };
    let (omega, short) = match kernel {
        GaussianKernel::Coulomb => return full(),
        GaussianKernel::LongRange { omega } => (omega, false),
        GaussianKernel::ShortRange { omega } => (omega, true),
    };
    if !omega.value().is_finite() || omega.value() < 0.0 {
        return Err(DftError::InvalidInput(
            "range separation must be finite and nonnegative".into(),
        ));
    }
    let theta = omega / (rho + omega * omega).sqrt();
    let ratio = theta * theta;
    let mut scale = theta;
    let mut long = Vec::with_capacity(nmax + 1);
    for n in 0..=nmax {
        long.push(scale * boys_jet(n, ratio * t, order)?);
        scale = scale * ratio;
    }
    if short {
        let full = full()?;
        Ok(full.into_iter().zip(long).map(|(f, l)| f - l).collect())
    } else {
        Ok(long)
    }
}

struct CoulombTensor<S: Scalar> {
    dim: usize,
    data: Vec<S>,
}
impl<S: Scalar> CoulombTensor<S> {
    fn new(l: usize, p: S, pc: [S; 3], f: &[S]) -> Self {
        let dim = l + 1;
        let index = |t: usize, u: usize, v: usize| (t * dim + u) * dim + v;
        let mut current = vec![constant::<S>(0.0); dim * dim * dim];
        let mut next = current.clone();
        current[0] = (-constant::<S>(2.0) * p).powi(l as i32) * f[l];
        for n in (0..l).rev() {
            next.fill(constant::<S>(0.0));
            next[0] = (-constant::<S>(2.0) * p).powi(n as i32) * f[n];
            for t in 0..=l - n {
                for u in 0..=l - n - t {
                    for v in 0..=l - n - t - u {
                        if t + u + v == 0 {
                            continue;
                        }
                        let (axis, powers) = if t > 0 {
                            (0, [t, u, v])
                        } else if u > 0 {
                            (1, [t, u, v])
                        } else {
                            (2, [t, u, v])
                        };
                        let mut lower = powers;
                        lower[axis] -= 1;
                        let mut value = pc[axis] * current[index(lower[0], lower[1], lower[2])];
                        if powers[axis] > 1 {
                            lower[axis] -= 1;
                            value = value
                                + constant::<S>((powers[axis] - 1) as f64)
                                    * current[index(lower[0], lower[1], lower[2])];
                        }
                        next[index(t, u, v)] = value;
                    }
                }
            }
            std::mem::swap(&mut current, &mut next);
        }
        Self { dim, data: current }
    }
    fn get(&self, powers: [usize; 3]) -> S {
        self.data[(powers[0] * self.dim + powers[1]) * self.dim + powers[2]]
    }
}

/// Hermite (McMurchie-Davidson) data of one primitive quartet shared by
/// every member of a family of quartets that differ only in the Cartesian
/// powers of their primitives: the Hermite expansion coefficients of the
/// bra and ket pairs and the Coulomb tensor are built once for the largest
/// powers, so a family of `n` raised quartets costs `n` contractions
/// instead of `n` complete integral evaluations. The values are those of
/// [`repulsion`] for each member.
pub(crate) struct HermiteQuartet<S: Scalar> {
    bra: [EAxis<S>; 3],
    ket: [EAxis<S>; 3],
    coulomb: CoulombTensor<S>,
    prefactor: S,
    capacity: usize,
}

/// Hermite coefficients of a linear combination of bra (or ket) pairs:
/// `(t, u, v) -> sum_k c_k E^x_t E^y_u E^z_v`, with the ket sign
/// `(-1)^(t+u+v)` already applied for a ket.
pub(crate) struct HermiteVector<S: Scalar> {
    entries: Vec<([usize; 3], S)>,
}

impl<S: Scalar> HermiteQuartet<S> {
    /// `raise[k]` bounds the extra Cartesian power, per axis, of primitive
    /// `k` above its own angular momentum, and `total` bounds the sum of
    /// all extra powers of a member quartet.
    pub(crate) fn new(
        primitives: [&Primitive<S>; 4],
        kernel: GaussianKernel<S>,
        order: usize,
        raise: [usize; 4],
        total: usize,
    ) -> Result<Self> {
        Self::new_radial(primitives, raise, total, |capacity, rho, t| {
            adjusted_boys(capacity, rho, t, kernel, order)
        })
    }

    /// Hermite tensor for the radial distance |r1-r2| with two extra
    /// separation derivative slots. Its ssss auxiliary at T=rho|P-Q|^2 is
    /// [(T+1/2)F_0(T)+exp(-T)/2]/rho in Coulomb prefactor units.
    pub(crate) fn new_distance(
        primitives: [&Primitive<S>; 4],
        order: usize,
        raise: [usize; 4],
        total: usize,
    ) -> Result<Self> {
        Self::new_radial(primitives, raise, total + 2, |capacity, rho, t| {
            let f = adjusted_boys(capacity, rho, t, GaussianKernel::Coulomb, order)?;
            let half = constant::<S>(0.5);
            let mut distance = Vec::with_capacity(capacity + 1);
            distance.push(((t + half) * f[0] + half * (-t).exp()) / rho);
            // The Hermite recurrence expects (-1)^n times the nth T
            // derivative. The Boys recurrence eliminates cancellation
            // between the exponential and the T F_n terms.
            distance.extend(f.windows(2).map(|pair| half * (pair[1] - pair[0]) / rho));
            Ok(distance)
        })
    }

    fn new_radial(
        primitives: [&Primitive<S>; 4],
        raise: [usize; 4],
        total: usize,
        auxiliary: impl FnOnce(usize, S, S) -> Result<Vec<S>>,
    ) -> Result<Self> {
        let [a, b, c, d] = primitives;
        for primitive in primitives {
            validate(primitive)?;
        }
        let p = a.exponent + b.exponent;
        let q = c.exponent + d.exponent;
        let rho = p * q / (p + q);
        let pq = std::array::from_fn(|axis| {
            (a.exponent * a.center[axis] + b.exponent * b.center[axis]) / p
                - (c.exponent * c.center[axis] + d.exponent * d.center[axis]) / q
        });
        let capacity = [a, b, c, d]
            .iter()
            .map(|x| x.angular.iter().sum::<usize>())
            .sum::<usize>()
            + total;
        let boys = auxiliary(capacity, rho, rho * norm2(pq))?;
        let pair = |x: &Primitive<S>, y: &Primitive<S>, rx: usize, ry: usize| -> [EAxis<S>; 3] {
            std::array::from_fn(|axis| {
                EAxis::new(
                    x.angular[axis] + rx,
                    y.angular[axis] + ry,
                    x.center[axis] - y.center[axis],
                    x.exponent,
                    y.exponent,
                )
            })
        };
        Ok(Self {
            bra: pair(a, b, raise[0], raise[1]),
            ket: pair(c, d, raise[2], raise[3]),
            coulomb: CoulombTensor::new(capacity, rho, pq, &boys),
            prefactor: constant::<S>(2.0 * PI.powf(2.5)) / (p * q * (p + q).sqrt()),
            capacity,
        })
    }

    fn vector(
        axes: &[EAxis<S>; 3],
        members: impl IntoIterator<Item = (S, [usize; 3], [usize; 3])>,
        ket: bool,
    ) -> HermiteVector<S> {
        // Dense accumulation over the Hermite indices the axes can reach,
        // then the touched entries only.
        let dims: [usize; 3] = std::array::from_fn(|axis| axes[axis].imax + axes[axis].jmax + 1);
        let index = |t: usize, u: usize, v: usize| (t * dims[1] + u) * dims[2] + v;
        let mut dense = vec![constant::<S>(0.0); dims[0] * dims[1] * dims[2]];
        let mut touched = vec![false; dense.len()];
        for (weight, left, right) in members {
            let top: [usize; 3] = std::array::from_fn(|axis| left[axis] + right[axis]);
            for t in 0..=top[0] {
                let ex = weight * axes[0].get(left[0], right[0], t);
                for u in 0..=top[1] {
                    let exy = ex * axes[1].get(left[1], right[1], u);
                    for v in 0..=top[2] {
                        let mut value = exy * axes[2].get(left[2], right[2], v);
                        if ket && (t + u + v) % 2 == 1 {
                            value = -value;
                        }
                        let k = index(t, u, v);
                        dense[k] = dense[k] + value;
                        touched[k] = true;
                    }
                }
            }
        }
        let mut entries = Vec::new();
        for t in 0..dims[0] {
            for u in 0..dims[1] {
                for v in 0..dims[2] {
                    let k = index(t, u, v);
                    if touched[k] {
                        entries.push(([t, u, v], dense[k]));
                    }
                }
            }
        }
        HermiteVector { entries }
    }

    /// Hermite coefficients of `sum_k w_k (a_k b_k|`, given the Cartesian
    /// powers of the first and second primitives of every member.
    pub(crate) fn bra(
        &self,
        members: impl IntoIterator<Item = (S, [usize; 3], [usize; 3])>,
    ) -> HermiteVector<S> {
        Self::vector(&self.bra, members, false)
    }

    /// Hermite coefficients of `|sum_k w_k c_k d_k)`.
    pub(crate) fn ket(
        &self,
        members: impl IntoIterator<Item = (S, [usize; 3], [usize; 3])>,
    ) -> HermiteVector<S> {
        Self::vector(&self.ket, members, true)
    }

    /// `(bra | ket)` of two Hermite vectors of this quartet.
    pub(crate) fn contract(&self, bra: &HermiteVector<S>, ket: &HermiteVector<S>) -> S {
        let mut sum = constant::<S>(0.0);
        for (left, x) in &bra.entries {
            for (right, y) in &ket.entries {
                let key: [usize; 3] = std::array::from_fn(|axis| left[axis] + right[axis]);
                debug_assert!(key.iter().sum::<usize>() <= self.capacity);
                sum = sum + *x * *y * self.coulomb.get(key);
            }
        }
        self.prefactor * sum
    }

    /// Hessian of the distance integral with respect to separation,
    /// ordered xx, xy, xz, yy, yz, zz. The pair expansions are held fixed.
    pub(crate) fn distance_hessian(
        &self,
        bra: &HermiteVector<S>,
        ket: &HermiteVector<S>,
    ) -> [S; 6] {
        let mut sum = [constant::<S>(0.0); 6];
        for (left, x) in &bra.entries {
            for (right, y) in &ket.entries {
                let key: [usize; 3] = std::array::from_fn(|axis| left[axis] + right[axis]);
                debug_assert!(key.iter().sum::<usize>() + 2 <= self.capacity);
                for (entry, (i, j)) in
                    sum.iter_mut()
                        .zip([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])
                {
                    let mut shifted = key;
                    shifted[i] += 1;
                    shifted[j] += 1;
                    *entry = *entry + *x * *y * self.coulomb.get(shifted);
                }
            }
        }
        sum.map(|entry| self.prefactor * entry)
    }
}

pub fn attraction<S: Scalar>(
    a: &Primitive<S>,
    b: &Primitive<S>,
    nuclear_center: [S; 3],
    charge: S,
    kernel: GaussianKernel<S>,
    order: usize,
) -> Result<S> {
    validate(a)?;
    validate(b)?;
    if nuclear_center.iter().any(|x| !x.value().is_finite()) || !charge.value().is_finite() {
        return Err(DftError::InvalidInput(
            "non-finite nuclear primitive input".into(),
        ));
    }
    let p = a.exponent + b.exponent;
    let pc = std::array::from_fn(|axis| {
        (a.exponent * a.center[axis] + b.exponent * b.center[axis]) / p - nuclear_center[axis]
    });
    let angular: [usize; 3] = std::array::from_fn(|axis| a.angular[axis] + b.angular[axis]);
    let l = angular.iter().sum();
    let boys = adjusted_boys(l, p, p * norm2(pc), kernel, order)?;
    let r = CoulombTensor::new(l, p, pc, &boys);
    let e = axes(a, b, 0);
    let mut sum = constant::<S>(0.0);
    for t in 0..=angular[0] {
        for u in 0..=angular[1] {
            for v in 0..=angular[2] {
                sum = sum
                    + e[0].get(a.angular[0], b.angular[0], t)
                        * e[1].get(a.angular[1], b.angular[1], u)
                        * e[2].get(a.angular[2], b.angular[2], v)
                        * r.get([t, u, v]);
            }
        }
    }
    Ok(-constant::<S>(2.0 * PI) * charge * sum / p)
}

pub fn repulsion<S: Scalar>(
    a: &Primitive<S>,
    b: &Primitive<S>,
    c: &Primitive<S>,
    d: &Primitive<S>,
    kernel: GaussianKernel<S>,
    order: usize,
) -> Result<S> {
    for primitive in [a, b, c, d] {
        validate(primitive)?;
    }
    let p = a.exponent + b.exponent;
    let q = c.exponent + d.exponent;
    let rho = p * q / (p + q);
    let pq = std::array::from_fn(|axis| {
        (a.exponent * a.center[axis] + b.exponent * b.center[axis]) / p
            - (c.exponent * c.center[axis] + d.exponent * d.center[axis]) / q
    });
    let left: [usize; 3] = std::array::from_fn(|axis| a.angular[axis] + b.angular[axis]);
    let right: [usize; 3] = std::array::from_fn(|axis| c.angular[axis] + d.angular[axis]);
    let l = left.iter().chain(&right).sum();
    let boys = adjusted_boys(l, rho, rho * norm2(pq), kernel, order)?;
    let r = CoulombTensor::new(l, rho, pq, &boys);
    let e = axes(a, b, 0);
    let f = axes(c, d, 0);
    let mut sum = constant::<S>(0.0);
    for t in 0..=left[0] {
        for u in 0..=left[1] {
            for v in 0..=left[2] {
                let bra = e[0].get(a.angular[0], b.angular[0], t)
                    * e[1].get(a.angular[1], b.angular[1], u)
                    * e[2].get(a.angular[2], b.angular[2], v);
                for x in 0..=right[0] {
                    for y in 0..=right[1] {
                        for z in 0..=right[2] {
                            let ket = f[0].get(c.angular[0], d.angular[0], x)
                                * f[1].get(c.angular[1], d.angular[1], y)
                                * f[2].get(c.angular[2], d.angular[2], z);
                            let sign = if (x + y + z) % 2 == 0 { 1.0 } else { -1.0 };
                            sum = sum
                                + bra * ket * constant::<S>(sign) * r.get([t + x, u + y, v + z]);
                        }
                    }
                }
            }
        }
    }
    Ok(constant::<S>(2.0 * PI.powf(2.5)) * sum / (p * q * (p + q).sqrt()))
}
