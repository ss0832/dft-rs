// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Contracted spherical Gaussian integrals over the existing Scalar algebra.
//! Radial normalization, centres, exponents, contraction coefficients, nuclear
//! charges, and range separation retain every analytic coefficient. Angular
//! transforms use the same constant CCA convention as the production basis.

use super::analytic::{self, GaussianKernel, Primitive};
use crate::basis::{aos, cart_components, double_factorial, MolShell};
use crate::dual::Scalar;
use crate::{DftError, Result};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct AnalyticShell<S: Scalar> {
    pub l: u8,
    pub center: [S; 3],
    pub exponents: Vec<S>,
    /// Coefficients including primitive and contracted radial normalization.
    pub coefficients: Vec<S>,
}

impl<S: Scalar> AnalyticShell<S> {
    /// Lift an already normalized production shell, retaining supplied centre
    /// derivatives. For exponent/coefficient derivatives use `normalized`.
    pub fn from_placed(shell: &MolShell, center: [S; 3]) -> Result<Self> {
        let result = Self {
            l: shell.l,
            center,
            exponents: shell.exps.iter().map(|&x| S::from_f64(x)).collect(),
            coefficients: shell.coefs.iter().map(|&x| S::from_f64(x)).collect(),
        };
        result.validate()?;
        Ok(result)
    }

    /// Normalize raw Gaussian94-style coefficients analytically, including
    /// exponent and coefficient derivatives of both normalization factors.
    pub fn normalized(
        l: u8,
        center: [S; 3],
        exponents: Vec<S>,
        coefficients: Vec<S>,
    ) -> Result<Self> {
        let mut result = Self {
            l,
            center,
            exponents,
            coefficients,
        };
        result.validate()?;
        let c = S::from_f64;
        let df = double_factorial(2 * l as i32 - 1);
        for (a, coefficient) in result.exponents.iter().zip(&mut result.coefficients) {
            let norm = ((c(2.0 / PI) * *a).powf(1.5) * (c(4.0) * *a).powi(l as i32) / c(df)).sqrt();
            *coefficient = *coefficient * norm;
        }
        let mut norm2 = c(0.0);
        for (a, ca) in result.exponents.iter().zip(&result.coefficients) {
            for (b, cb) in result.exponents.iter().zip(&result.coefficients) {
                let p = *a + *b;
                norm2 =
                    norm2 + *ca * *cb * c(df) / (c(2.0) * p).powi(l as i32) * (c(PI) / p).powf(1.5);
            }
        }
        if !norm2.value().is_finite() || norm2.value() <= 0.0 {
            return Err(DftError::InvalidInput(
                "nonpositive analytic contraction norm".into(),
            ));
        }
        let norm = norm2.sqrt();
        for coefficient in &mut result.coefficients {
            *coefficient = *coefficient / norm;
        }
        Ok(result)
    }

    pub fn validate(&self) -> Result<()> {
        if self.l > aos::MAX_L
            || self.exponents.is_empty()
            || self.exponents.len() != self.coefficients.len()
            || self.center.iter().any(|x| !x.value().is_finite())
            || self
                .exponents
                .iter()
                .any(|x| !x.value().is_finite() || x.value() <= 0.0)
            || self.coefficients.iter().any(|x| !x.value().is_finite())
        {
            return Err(DftError::InvalidInput(
                "invalid analytic shell or angular transform".into(),
            ));
        }
        Ok(())
    }
    fn primitive(&self, component: (i32, i32, i32), index: usize) -> Primitive<S> {
        Primitive {
            angular: [
                component.0 as usize,
                component.1 as usize,
                component.2 as usize,
            ],
            exponent: self.exponents[index],
            center: self.center,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnalyticNucleus<S: Scalar> {
    pub center: [S; 3],
    pub charge: S,
}

/// Row-major spherical integral tensor. The last shell index varies fastest;
/// component ordering is m=0,+1,-1,... for each shell.
#[derive(Clone, Debug)]
pub struct IntegralBlock<S: Scalar, const N: usize> {
    pub shape: [usize; N],
    pub values: Vec<S>,
}

fn zeros<S: Scalar>(size: usize) -> Result<Vec<S>> {
    let mut values = Vec::new();
    values.try_reserve_exact(size).map_err(|_| {
        DftError::InvalidInput("analytic integral tensor exceeds available allocation".into())
    })?;
    values.resize(size, S::from_f64(0.0));
    Ok(values)
}

fn size<const N: usize>(shape: &[usize; N]) -> Result<usize> {
    shape.iter().try_fold(1usize, |s, &n| {
        s.checked_mul(n)
            .ok_or_else(|| DftError::InvalidInput("analytic integral shape overflow".into()))
    })
}

fn spherical<S: Scalar, const N: usize>(
    mut block: IntegralBlock<S, N>,
    shells: [&AnalyticShell<S>; N],
) -> Result<IntegralBlock<S, N>> {
    for (axis, shell) in shells.into_iter().enumerate() {
        let transform = aos::cart_to_sph(shell.l);
        let old = block.shape[axis];
        let stride = size(&block.shape)? / block.shape[..=axis].iter().product::<usize>();
        let outer = block.values.len() / (old * stride);
        let mut shape = block.shape;
        shape[axis] = transform.cols;
        let mut values = zeros(size(&shape)?)?;
        for prefix in 0..outer {
            for source in 0..old {
                for target in 0..transform.cols {
                    let weight = transform[(source, target)];
                    if weight == 0.0 {
                        continue;
                    }
                    for tail in 0..stride {
                        let src = (prefix * old + source) * stride + tail;
                        let dst = (prefix * transform.cols + target) * stride + tail;
                        values[dst] = values[dst] + block.values[src] * S::from_f64(weight);
                    }
                }
            }
        }
        block = IntegralBlock { shape, values };
    }
    Ok(block)
}

/// Contract several primitive operator channels over one normalized shell tuple.
/// The Cartesian/spherical convention and scalar normalization are shared with
/// all existing real integrals; complex operators use separate real/imaginary
/// channels without copying any physical symmetry in the primitive operator.
pub(crate) fn contracted_components<S: Scalar, const CHANNELS: usize, const RANK: usize>(
    shells: [&AnalyticShell<S>; RANK],
    operator: impl Fn([Primitive<S>; RANK]) -> Result<[S; CHANNELS]>,
) -> Result<[IntegralBlock<S, RANK>; CHANNELS]> {
    if RANK == 0 {
        return Err(DftError::InvalidInput(
            "an analytic shell tuple must be nonempty".into(),
        ));
    }
    for shell in shells {
        shell.validate()?;
    }
    let components = shells.map(|shell| cart_components(shell.l));
    let shape = components.each_ref().map(Vec::len);
    let count = size(&shape)?;
    let mut channels: Vec<_> = (0..CHANNELS)
        .map(|_| zeros::<S>(count))
        .collect::<Result<_>>()?;
    let primitive_count = shells.iter().try_fold(1_usize, |product, shell| {
        product
            .checked_mul(shell.exponents.len())
            .ok_or_else(|| DftError::InvalidInput("analytic primitive tuple overflow".into()))
    })?;
    for index in 0..count {
        let mut remainder = index;
        let mut angular = [(0, 0, 0); RANK];
        for axis in (0..RANK).rev() {
            angular[axis] = components[axis][remainder % shape[axis]];
            remainder /= shape[axis];
        }
        for primitive in 0..primitive_count {
            let mut remainder = primitive;
            let mut indices = [0; RANK];
            for axis in (0..RANK).rev() {
                indices[axis] = remainder % shells[axis].exponents.len();
                remainder /= shells[axis].exponents.len();
            }
            let mut weight = shells[0].coefficients[indices[0]];
            for axis in 1..RANK {
                weight = weight * shells[axis].coefficients[indices[axis]];
            }
            let primitives = std::array::from_fn(|a| shells[a].primitive(angular[a], indices[a]));
            let values = operator(primitives)?;
            for (channel, value) in channels.iter_mut().zip(values) {
                channel[index] = channel[index] + weight * value;
            }
        }
    }
    let result = channels
        .into_iter()
        .map(|values| spherical(IntegralBlock { shape, values }, shells))
        .collect::<Result<Vec<_>>>()?;
    result.try_into().map_err(|_| {
        DftError::InvalidInput("analytic channel count changed during contraction".into())
    })
}

fn pair<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    operator: impl Fn(&Primitive<S>, &Primitive<S>) -> Result<S>,
) -> Result<IntegralBlock<S, 2>> {
    let [value] = contracted_components([a, b], |[pa, pb]| Ok([operator(&pa, &pb)?]))?;
    Ok(value)
}

pub fn overlap<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
) -> Result<IntegralBlock<S, 2>> {
    pair(a, b, analytic::overlap)
}
pub fn kinetic<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
) -> Result<IntegralBlock<S, 2>> {
    pair(a, b, analytic::kinetic)
}

/// Position operator (r_axis - origin_axis) in the production spherical
/// convention. The angular recurrence preserves centre and origin jets.
pub fn dipole<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    axis: usize,
    origin: [S; 3],
) -> Result<IntegralBlock<S, 2>> {
    if axis >= 3 || origin.iter().any(|x| !x.value().is_finite()) {
        return Err(DftError::InvalidInput(
            "invalid analytic dipole axis or origin".into(),
        ));
    }
    pair(a, b, |pa, pb| {
        let mut raised = *pb;
        raised.angular[axis] = raised.angular[axis]
            .checked_add(1)
            .ok_or_else(|| DftError::InvalidInput("dipole angular index overflow".into()))?;
        Ok(analytic::overlap(pa, &raised)?
            + (pb.center[axis] - origin[axis]) * analytic::overlap(pa, pb)?)
    })
}
/// Overlap of `a` with `b` times `prod_k (r_k - origin_k)^power_k`.
fn moment_primitive<S: Scalar>(
    pa: &Primitive<S>,
    pb: &Primitive<S>,
    power: [usize; 3],
    origin: [S; 3],
) -> Result<S> {
    polynomial_primitive(pa, pb, power, origin, analytic::overlap)
}

fn polynomial_primitive<S: Scalar>(
    pa: &Primitive<S>,
    pb: &Primitive<S>,
    power: [usize; 3],
    origin: [S; 3],
    operator: impl Fn(&Primitive<S>, &Primitive<S>) -> Result<S>,
) -> Result<S> {
    // (r - O)^n = sum_j C(n, j) (B - O)^(n - j) (r - B)^j about the centre B.
    let mut value = S::from_f64(0.0);
    for jx in 0..=power[0] {
        for jy in 0..=power[1] {
            for jz in 0..=power[2] {
                let shift = [jx, jy, jz];
                let mut raised = *pb;
                let mut weight = S::from_f64(1.0);
                for axis in 0..3 {
                    raised.angular[axis] += shift[axis];
                    let mut choose = 1.0;
                    for step in 0..shift[axis] {
                        choose *= (power[axis] - step) as f64 / (step + 1) as f64;
                    }
                    weight = weight
                        * S::from_f64(choose)
                        * (pb.center[axis] - origin[axis]).powi((power[axis] - shift[axis]) as i32);
                }
                value = value + weight * operator(pa, &raised)?;
            }
        }
    }
    Ok(value)
}

/// `moment_primitive` with the derivative along `axis` acting on `b`.
fn moment_gradient_primitive<S: Scalar>(
    pa: &Primitive<S>,
    pb: &Primitive<S>,
    power: [usize; 3],
    origin: [S; 3],
    axis: usize,
) -> Result<S> {
    let mut raised = *pb;
    raised.angular[axis] += 1;
    let mut value =
        -(S::from_f64(2.0) * pb.exponent) * moment_primitive(pa, &raised, power, origin)?;
    if pb.angular[axis] > 0 {
        let mut lowered = *pb;
        lowered.angular[axis] -= 1;
        value = value
            + S::from_f64(pb.angular[axis] as f64) * moment_primitive(pa, &lowered, power, origin)?;
    }
    Ok(value)
}

/// Cartesian polynomial moment times a ket derivative.
pub(crate) fn moment_gradient<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    power: [usize; 3],
    origin: [S; 3],
    axis: usize,
) -> Result<IntegralBlock<S, 2>> {
    pair(a, b, |pa, pb| {
        moment_gradient_primitive(pa, pb, power, origin, axis)
    })
}

/// Cartesian moment `prod_k (r_k - origin_k)^power_k` in the production
/// spherical convention.
pub fn moment<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    power: [usize; 3],
    origin: [S; 3],
) -> Result<IntegralBlock<S, 2>> {
    if origin.iter().any(|x| !x.value().is_finite()) {
        return Err(DftError::InvalidInput(
            "invalid analytic moment origin".into(),
        ));
    }
    pair(a, b, |pa, pb| moment_primitive(pa, pb, power, origin))
}

/// Gradient `d/dr_axis` acting on `b`. The block changes sign when the two
/// shells are exchanged.
pub fn gradient<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    axis: usize,
) -> Result<IntegralBlock<S, 2>> {
    if axis >= 3 {
        return Err(DftError::InvalidInput(
            "invalid analytic gradient axis".into(),
        ));
    }
    pair(a, b, |pa, pb| {
        moment_gradient_primitive(pa, pb, [0; 3], pb.center, axis)
    })
}

/// Orbital angular momentum `((r - origin) x grad)_axis` acting on `b`,
/// without the factor `-i`. The block changes sign when the two shells are
/// exchanged.
pub fn angular_momentum<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    axis: usize,
    origin: [S; 3],
) -> Result<IntegralBlock<S, 2>> {
    if axis >= 3 || origin.iter().any(|x| !x.value().is_finite()) {
        return Err(DftError::InvalidInput(
            "invalid analytic angular momentum axis or origin".into(),
        ));
    }
    let (j, k) = ((axis + 1) % 3, (axis + 2) % 3);
    let unit = |direction: usize| {
        let mut power = [0_usize; 3];
        power[direction] = 1;
        power
    };
    pair(a, b, |pa, pb| {
        Ok(moment_gradient_primitive(pa, pb, unit(j), origin, k)?
            - moment_gradient_primitive(pa, pb, unit(k), origin, j)?)
    })
}

/// Velocity form of the second moment about `origin`:
/// `(r_i - O_i) d_j + (r_j - O_j) d_i + delta_ij` acting on `b`. The block
/// changes sign when the two shells are exchanged.
pub fn velocity_quadrupole<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    i: usize,
    j: usize,
    origin: [S; 3],
) -> Result<IntegralBlock<S, 2>> {
    if i >= 3 || j >= 3 || origin.iter().any(|x| !x.value().is_finite()) {
        return Err(DftError::InvalidInput(
            "invalid analytic velocity quadrupole axes or origin".into(),
        ));
    }
    let unit = |direction: usize| {
        let mut power = [0_usize; 3];
        power[direction] = 1;
        power
    };
    pair(a, b, |pa, pb| {
        let mut value = moment_gradient_primitive(pa, pb, unit(i), origin, j)?
            + moment_gradient_primitive(pa, pb, unit(j), origin, i)?;
        if i == j {
            value = value + analytic::overlap(pa, pb)?;
        }
        Ok(value)
    })
}

pub fn attraction<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    nuclei: &[AnalyticNucleus<S>],
    kernel: GaussianKernel<S>,
    order: usize,
) -> Result<IntegralBlock<S, 2>> {
    attraction_moment(a, b, nuclei, kernel, order, [0; 3], [S::from_f64(0.0); 3])
}

/// Nuclear attraction times a Cartesian polynomial about a fixed origin.
/// The same primitive angular recurrence as position moments retains all
/// center, finite-nucleus, kernel and origin derivatives.
pub fn attraction_moment<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    nuclei: &[AnalyticNucleus<S>],
    kernel: GaussianKernel<S>,
    order: usize,
    power: [usize; 3],
    origin: [S; 3],
) -> Result<IntegralBlock<S, 2>> {
    if origin.iter().any(|x| !x.value().is_finite()) {
        return Err(DftError::InvalidInput(
            "invalid attraction moment origin".into(),
        ));
    }
    pair(a, b, |pa, pb| {
        polynomial_primitive(pa, pb, power, origin, |pa, pb| {
            let mut value = S::from_f64(0.0);
            for nucleus in nuclei {
                value = value
                    + analytic::attraction(pa, pb, nucleus.center, nucleus.charge, kernel, order)?;
            }
            Ok(value)
        })
    })
}

pub fn repulsion<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    c: &AnalyticShell<S>,
    d: &AnalyticShell<S>,
    kernel: GaussianKernel<S>,
    order: usize,
) -> Result<IntegralBlock<S, 4>> {
    let (shape, values) = contract_repulsion(
        [a, b, c, d],
        S::from_f64(0.0),
        |primitives, raise| {
            analytic::HermiteQuartet::new(primitives, kernel, order, raise, raise.iter().sum())
        },
        |slot, weight, quartet, bra, ket| *slot = *slot + weight * quartet.contract(bra, ket),
    )?;
    Ok(IntegralBlock { shape, values })
}

/// The six symmetric Cartesian separation derivatives of the distance
/// operator contracted in the same spherical basis as scalar repulsion.
pub(crate) fn repulsion_distance_hessian<S: Scalar>(
    shells: [&AnalyticShell<S>; 4],
    order: usize,
) -> Result<([usize; 4], Vec<[S; 6]>)> {
    contract_repulsion(
        shells,
        [S::from_f64(0.0); 6],
        |primitives, raise| {
            analytic::HermiteQuartet::new_distance(primitives, order, raise, raise.iter().sum())
        },
        |slot, weight, quartet, bra, ket| {
            for (entry, integral) in slot.iter_mut().zip(quartet.distance_hessian(bra, ket)) {
                *entry = *entry + weight * integral;
            }
        },
    )
}

fn contract_repulsion<S: Scalar, T: Copy>(
    shells: [&AnalyticShell<S>; 4],
    zero: T,
    build: impl Fn([&Primitive<S>; 4], [usize; 4]) -> Result<analytic::HermiteQuartet<S>>,
    accumulate: impl Fn(
        &mut T,
        S,
        &analytic::HermiteQuartet<S>,
        &analytic::HermiteVector<S>,
        &analytic::HermiteVector<S>,
    ),
) -> Result<([usize; 4], Vec<T>)> {
    // Build primitive Hermite/Coulomb data once, and transform the linear
    // bra/ket expansions before their bilinear contraction. No Cartesian
    // fourth-rank intermediate or per-component Coulomb rebuild is needed.
    for shell in shells {
        shell.validate()?;
    }
    let angular = shells.map(|shell| cart_components(shell.l));
    let shape = shells.map(|shell| crate::basis::n_sph(shell.l));
    let count = size(&shape)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| {
        DftError::InvalidInput("analytic integral tensor exceeds available allocation".into())
    })?;
    values.resize(count, zero);
    let members = |left: usize, right: usize| {
        let first = aos::cart_to_sph(shells[left].l);
        let second = aos::cart_to_sph(shells[right].l);
        let first_count = (0..shape[left])
            .map(|m| (0..first.rows).filter(|&i| first[(i, m)] != 0.0).count())
            .collect::<Vec<_>>();
        let second_count = (0..shape[right])
            .map(|n| (0..second.rows).filter(|&j| second[(j, n)] != 0.0).count())
            .collect::<Vec<_>>();
        let mut result = Vec::with_capacity(shape[left] * shape[right]);
        for m in 0..shape[left] {
            for n in 0..shape[right] {
                let mut pair = Vec::with_capacity(first_count[m] * second_count[n]);
                for (i, &(x, y, z)) in angular[left].iter().enumerate() {
                    for (j, &(u, v, w)) in angular[right].iter().enumerate() {
                        let coefficient = first[(i, m)] * second[(j, n)];
                        if coefficient != 0.0 {
                            pair.push((
                                S::from_f64(coefficient),
                                [x as usize, y as usize, z as usize],
                                [u as usize, v as usize, w as usize],
                            ));
                        }
                    }
                }
                result.push(pair);
            }
        }
        result
    };
    let left = members(0, 1);
    let right = members(2, 3);
    let primitive_count = shells.iter().try_fold(1_usize, |n, s| {
        n.checked_mul(s.exponents.len())
            .ok_or_else(|| DftError::InvalidInput("analytic primitive tuple overflow".into()))
    })?;
    for primitive in 0..primitive_count {
        let mut remainder = primitive;
        let mut indices = [0; 4];
        for axis in (0..4).rev() {
            indices[axis] = remainder % shells[axis].exponents.len();
            remainder /= shells[axis].exponents.len();
        }
        let primitives =
            std::array::from_fn::<_, 4, _>(|axis| shells[axis].primitive((0, 0, 0), indices[axis]));
        let raise = shells.map(|s| s.l as usize);
        let quartet = build(primitives.each_ref(), raise)?;
        let bra = left
            .iter()
            .map(|m| quartet.bra(m.iter().copied()))
            .collect::<Vec<_>>();
        let ket = right
            .iter()
            .map(|m| quartet.ket(m.iter().copied()))
            .collect::<Vec<_>>();
        let mut weight = shells[0].coefficients[indices[0]];
        for axis in 1..4 {
            weight = weight * shells[axis].coefficients[indices[axis]];
        }
        for (i, a) in bra.iter().enumerate() {
            for (j, b) in ket.iter().enumerate() {
                let index = i * ket.len() + j;
                accumulate(&mut values[index], weight, &quartet, a, b);
            }
        }
    }
    Ok((shape, values))
}

#[cfg(test)]
mod repulsion_tests {
    use super::*;
    use crate::dual::Taylor;

    fn direct<S: Scalar>(
        shells: &[AnalyticShell<S>; 4],
        kernel: GaussianKernel<S>,
        order: usize,
    ) -> Result<IntegralBlock<S, 4>> {
        let [value] = contracted_components(shells.each_ref(), |[a, b, c, d]| {
            Ok([analytic::repulsion(&a, &b, &c, &d, kernel, order)?])
        })?;
        Ok(value)
    }

    fn check(angular: [u8; 4], radial_derivatives: bool) {
        let positions = [
            [0.13, -0.27, 0.08],
            [1.39, 0.21, -0.34],
            [-0.41, 1.52, 0.31],
            [0.28, -0.33, 1.71],
        ];
        let shells: [AnalyticShell<Taylor<5>>; 4] = std::array::from_fn(|atom| {
            let centre = std::array::from_fn(|axis| {
                Taylor::from_coefficients(&[
                    positions[atom][axis],
                    0.12 * (3 * atom + axis + 1) as f64,
                    0.01 * ((atom + axis + 1) as f64).sin(),
                    0.005 * ((2 * atom + axis + 1) as f64).cos(),
                    0.002 * ((atom + 2 * axis + 1) as f64).sin(),
                ])
            });
            let mut exponents = vec![Taylor::from_f64(0.51 + 0.11 * atom as f64)];
            let mut coefficients = vec![Taylor::from_f64(1.0)];
            if radial_derivatives {
                exponents[0].c[1..].copy_from_slice(&[0.013, -0.007, 0.002, 0.001]);
                coefficients[0].c[1..].copy_from_slice(&[-0.031, 0.005, 0.003, -0.001]);
                exponents.push(Taylor::from_coefficients(&[
                    0.21, -0.011, 0.004, 0.002, 0.001,
                ]));
                coefficients.push(Taylor::from_coefficients(&[
                    0.37, 0.023, -0.005, 0.002, -0.001,
                ]));
            }
            AnalyticShell::normalized(angular[atom], centre, exponents, coefficients).unwrap()
        });
        for kernel in [
            GaussianKernel::Coulomb,
            GaussianKernel::LongRange {
                omega: Taylor::from_coefficients(&[0.43, 0.013, -0.004, 0.002, 0.001]),
            },
            GaussianKernel::ShortRange {
                omega: Taylor::from_coefficients(&[0.43, 0.013, -0.004, 0.002, 0.001]),
            },
        ] {
            let start = std::time::Instant::now();
            let expected = direct(&shells, kernel, 4).unwrap();
            let direct_seconds = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            let actual =
                repulsion(&shells[0], &shells[1], &shells[2], &shells[3], kernel, 4).unwrap();
            let shared_seconds = start.elapsed().as_secs_f64();
            assert_eq!(actual.shape, expected.shape);
            let mut maximum = 0.0_f64;
            for (index, (a, b)) in actual.values.iter().zip(&expected.values).enumerate() {
                for n in 0..5 {
                    let defect = (a.c[n] - b.c[n]).abs();
                    maximum = maximum.max(defect);
                    assert!(
                        defect < 2e-13 * (1.0 + b.c[n].abs()),
                        "angular={angular:?} entry={index} order={n} defect={defect:.12e}"
                    );
                }
            }
            println!(
                "shared Hermite angular={angular:?} radial_derivatives={radial_derivatives} coefficients={} max_defect={maximum:.12e} direct_seconds={direct_seconds:.6} shared_seconds={shared_seconds:.6}",
                actual.values.len() * 5
            );
        }
    }

    #[test]
    fn all_mixed_angular_coefficients_match_direct_cartesian_quartets() {
        for angular in [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 1, 0, 0],
            [3, 1, 0, 0],
            [4, 1, 0, 0],
            [5, 1, 0, 0],
            [6, 1, 0, 0],
            [4, 3, 0, 0],
            [5, 4, 0, 0],
            [6, 5, 0, 0],
            [3, 3, 3, 0],
            [4, 4, 4, 0],
            [6, 6, 0, 0],
        ] {
            check(angular, false);
        }
    }

    #[test]
    fn radial_normalization_and_range_coefficients_remain_differentiable() {
        for angular in [[1, 1, 0, 0], [4, 1, 0, 0], [6, 1, 0, 0]] {
            check(angular, true);
        }
    }
}
