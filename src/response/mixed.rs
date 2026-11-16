// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact polarization of factorial-normalized directional Taylor
//! coefficients. This is an analytic mixed derivative identity; no finite
//! displacement or step size is involved. Repeated directions share a
//! binomial sum instead of an exponential list of equivalent subsets.

use crate::{DftError, Result};
use std::collections::BTreeMap;

/// Exact homogeneous coefficient reuse across nuclear and electric providers.
/// Keys retain every normalized floating-point bit; nearby paths are never merged.
#[derive(Default)]
pub(crate) struct DirectionalCoefficients {
    values: BTreeMap<(bool, usize, Vec<u64>), Vec<f64>>,
    pub evaluations: usize,
}

impl DirectionalCoefficients {
    pub fn get(
        &mut self,
        direction: &[f64],
        order: usize,
        channels: usize,
        evaluate: impl FnOnce(&[f64], usize) -> Result<Vec<f64>>,
    ) -> Result<Vec<f64>> {
        self.get_with_precision(direction, order, channels, false, evaluate)
    }

    /// Adjacent high/low channels retain exact binary path scaling and
    /// product rounding before the mixed derivative is accumulated.
    pub fn get_retained(
        &mut self,
        direction: &[f64],
        order: usize,
        channels: usize,
        evaluate: impl FnOnce(&[f64], usize) -> Result<Vec<f64>>,
    ) -> Result<Vec<f64>> {
        if channels % 2 != 0 {
            return Err(DftError::InvalidInput(
                "retained coefficient channels require high/low pairs".into(),
            ));
        }
        self.get_with_precision(direction, order, channels, true, evaluate)
    }

    fn get_with_precision(
        &mut self,
        direction: &[f64],
        order: usize,
        channels: usize,
        retained: bool,
        evaluate: impl FnOnce(&[f64], usize) -> Result<Vec<f64>>,
    ) -> Result<Vec<f64>> {
        if order == 0 || channels == 0 || direction.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "directional coefficient caches require finite directions, positive order and channels".into(),
            ));
        }
        let magnitude = direction.iter().map(|v| v.abs()).fold(0.0, f64::max);
        // Opposite paths are the same homogeneous coefficient with parameter
        // t reversed. Retain its signed scale, including every odd order.
        let mut scale =
            magnitude.copysign(direction.iter().copied().find(|v| *v != 0.0).unwrap_or(1.0));
        if scale == 0.0 {
            return Ok(vec![0.0; channels]);
        }
        if retained {
            let bits = magnitude.to_bits();
            let exponent = bits & 0x7ff0_0000_0000_0000;
            let power = if exponent != 0 {
                f64::from_bits(exponent)
            } else {
                f64::from_bits(1_u64 << (63 - bits.leading_zeros()))
            };
            scale = power.copysign(scale);
            // An extreme exponent spread must not erase a path component.
            // In that case keep the original finite vector without scaling.
            let factor = scale.powi(order as i32);
            if !factor.is_normal() || direction.iter().any(|&x| (x / scale) * scale != x) {
                scale = 1.0;
            }
        }
        let normalized: Vec<_> = direction
            .iter()
            .map(|&x| if x == 0.0 { 0.0 } else { x / scale })
            .collect();
        let key = (
            retained,
            order,
            normalized.iter().map(|v| v.to_bits()).collect(),
        );
        if !self.values.contains_key(&key) {
            let value = evaluate(&normalized, order)?;
            if value.len() != channels || value.iter().any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput(
                    "incompatible high-order energy coefficient channels".into(),
                ));
            }
            self.values.insert(key.clone(), value);
            self.evaluations += 1;
        }
        if self.values[&key].len() != channels {
            return Err(DftError::InvalidInput(
                "cached energy coefficient channel count changed".into(),
            ));
        }
        let factor = scale.powi(order as i32);
        if retained {
            let mut values = Vec::with_capacity(channels);
            for pair in self.values[&key].chunks_exact(2) {
                let mut sum = crate::linalg::AccurateSum::default();
                sum.add_product(pair[0], factor);
                sum.add_product(pair[1], factor);
                let (high, low) = sum.parts();
                values.extend([high, low]);
            }
            Ok(values)
        } else {
            Ok(self.values[&key].iter().map(|v| v * factor).collect())
        }
    }
}

pub(crate) struct PolarizedTensor {
    pub values: Vec<Vec<f64>>,
    pub roundoff: Vec<Vec<f64>>,
    pub permutation_defects: Vec<f64>,
    pub permutation_comparisons: usize,
    pub independent_components: usize,
    pub directional_evaluations: usize,
}

pub(crate) fn tensor_indices(mut flat: usize, dimension: usize, order: usize) -> Vec<usize> {
    let mut tuple = vec![0; order];
    for axis in tuple.iter_mut().rev() {
        *axis = flat % dimension;
        flat /= dimension;
    }
    tuple
}

/// Complete row-major symmetric tensors of independent energy channels.
/// Ordered reverse comparisons are optional and never inferred from reused
/// permutations. The callback supplies factorial-normalized energy coefficients.
pub(crate) fn polarized_tensor(
    patterns: &[Vec<f64>],
    order: usize,
    channels: usize,
    memory_mb: usize,
    compare_reverse: bool,
    mut coefficient: impl FnMut(&[f64], usize) -> Result<Vec<f64>>,
) -> Result<PolarizedTensor> {
    let dimension = patterns.len();
    let width = patterns.first().map_or(0, Vec::len);
    if dimension == 0
        || width == 0
        || channels == 0
        || !(3..=4).contains(&order)
        || patterns
            .iter()
            .any(|p| p.len() != width || p.iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput("force constants require order three/four, channels and nonempty finite equal-sized patterns".into()));
    }
    let size = dimension
        .checked_pow(order as u32)
        .ok_or_else(|| DftError::InvalidInput("force-constant tensor size overflow".into()))?;
    let bytes = size
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(2 * std::mem::size_of::<f64>()))
        .ok_or_else(|| DftError::InvalidInput("force-constant storage overflow".into()))?;
    let limit = memory_mb
        .checked_mul(1024 * 1024)
        .ok_or_else(|| DftError::InvalidInput("force-constant memory limit overflow".into()))?;
    if bytes > limit {
        return Err(DftError::InvalidInput("force-constant tensor exceeds incore_mb; request a smaller coordinate block or pattern set".into()));
    }
    let mut out = PolarizedTensor {
        values: vec![vec![0.0; size]; channels],
        roundoff: vec![vec![0.0; size]; channels],
        permutation_defects: vec![0.0; channels],
        permutation_comparisons: 0,
        independent_components: 0,
        directional_evaluations: 0,
    };
    for flat in 0..size {
        let tuple = tensor_indices(flat, dimension, order);
        let mut canonical = tuple.clone();
        canonical.sort_unstable();
        if tuple != canonical {
            let source = canonical.iter().fold(0, |i, &a| i * dimension + a);
            for c in 0..channels {
                out.values[c][flat] = out.values[c][source];
                out.roundoff[c][flat] = out.roundoff[c][source];
            }
            continue;
        }
        let directions: Vec<_> = tuple.iter().map(|&a| patterns[a].clone()).collect();
        let mut value = polarized_mixed(width, &directions, &mut coefficient)?;
        if value.value.len() != channels {
            return Err(DftError::InvalidInput(
                "incompatible force-constant energy channels".into(),
            ));
        }
        out.directional_evaluations += value.directional_evaluations;
        if compare_reverse && tuple.iter().rev().ne(tuple.iter()) {
            let reverse: Vec<_> = directions.into_iter().rev().collect();
            let other = polarized_mixed(width, &reverse, &mut coefficient)?;
            if other.value.len() != channels {
                return Err(DftError::InvalidInput(
                    "incompatible reverse force-constant channels".into(),
                ));
            }
            out.directional_evaluations += other.directional_evaluations;
            out.permutation_comparisons += 1;
            for c in 0..channels {
                out.permutation_defects[c] =
                    out.permutation_defects[c].max((value.value[c] - other.value[c]).abs());
                value.accumulation_roundoff[c] =
                    value.accumulation_roundoff[c].max(other.accumulation_roundoff[c]);
            }
        }
        out.independent_components += 1;
        for c in 0..channels {
            out.values[c][flat] = value.value[c];
            out.roundoff[c][flat] = value.accumulation_roundoff[c];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tensor_tests {
    use super::*;

    #[test]
    fn all_retained_mixed_components_fit_an_independent_exponential() {
        fit_cached_mixed_exponential(true);
    }

    #[test]
    fn all_cached_mixed_components_fit_an_independent_exponential() {
        fit_cached_mixed_exponential(false);
    }

    fn fit_cached_mixed_exponential(retained: bool) {
        use crate::acceptance::{components, stencils};
        let rates = [0.23_f64, -0.41, 0.37];
        let mut cache = DirectionalCoefficients::default();
        let mut exact = Vec::new();
        let mut tuples = Vec::new();
        let mut names = Vec::new();
        for order in 1..=4 {
            for flat in 0..3_usize.pow(order as u32) {
                let tuple = tensor_indices(flat, 3, order);
                let legs: Vec<_> = tuple
                    .iter()
                    .map(|&axis| (0..3).map(|i| f64::from(i == axis)).collect())
                    .collect();
                let value = polarized_mixed(3, &legs, |d, n| {
                    let coefficient = |d: &[f64], n: usize| {
                        let slope: f64 = rates.iter().zip(d).map(|(a, d)| a * d).sum();
                        let high = slope.powi(n as i32) / (1..=n).product::<usize>() as f64;
                        Ok(vec![high, high * 2.0_f64.powi(-54)])
                    };
                    if retained {
                        cache.get_retained(d, n, 2, coefficient)
                    } else {
                        cache.get(d, n, 2, coefficient)
                    }
                })
                .unwrap();
                let mut sum = crate::linalg::AccurateSum::default();
                for x in value.value {
                    sum.add_product(x, 1.0);
                }
                let actual = sum.value();
                let closed: f64 = tuple.iter().map(|&a| rates[a]).product();
                assert!((actual - closed).abs() < 1e-12);
                names.push(format!(
                    "cached mixed retained={retained} order={order} component={flat}"
                ));
                exact.push(actual);
                tuples.push(tuple);
            }
        }
        let stages = [
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                tuples
                    .iter()
                    .zip(&exact)
                    .map(|(tuple, &analytic)| {
                        let prefix: f64 =
                            tuple[..tuple.len() - 1].iter().map(|&a| rates[a]).product();
                        let rate = rates[*tuple.last().unwrap()];
                        let oracle = |h: f64| {
                            let high = prefix * (rate * h).exp();
                            high + high * 2.0_f64.powi(-54)
                        };
                        std::array::from_fn(|i| {
                            let h = steps[i];
                            (stencil.first(
                                oracle(h),
                                oracle(-h),
                                oracle(2.0 * h),
                                oracle(-2.0 * h),
                                h,
                            ) - analytic)
                                .abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
        assert_eq!(exact.len(), 120);
        components::assert_no_failures("all120 retained mixed derivative components", &failures);
    }

    #[test]
    fn retained_cache_preserves_binary_paths_and_subulp_coefficients() {
        for order in 1..=4 {
            let mut cache = DirectionalCoefficients::default();
            let direction = [3.0, 1.0, -0.19];
            let normalized = direction.map(|v| v / 2.0);
            let value = cache
                .get_retained(&direction, order, 2, |d, n| {
                    assert_eq!(d, normalized);
                    assert_eq!(n, order);
                    for i in 0..3 {
                        assert_eq!(d[i] * 2.0, direction[i]);
                    }
                    Ok(vec![1.0, 2.0_f64.powi(-54)])
                })
                .unwrap();
            let factor = 2.0_f64.powi(order as i32);
            assert_eq!(value, vec![factor, factor * 2.0_f64.powi(-54)]);
            let opposite = direction.map(|v| -v);
            let reverse = cache
                .get_retained(&opposite, order, 2, |_, _| {
                    panic!("the exactly reversed path reuses its homogeneous coefficient")
                })
                .unwrap();
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(reverse, value.iter().map(|v| sign * v).collect::<Vec<_>>());
            assert_eq!(cache.evaluations, 1);
            assert!(cache
                .get_retained(&direction, order, 1, |_, _| Ok(vec![1.0]))
                .is_err());
        }
        // Normalization must not erase a finite component whose exponent
        // lies beyond the range of a normalized binary64 vector.
        let direction = [f64::MAX, f64::MIN_POSITIVE];
        let mut cache = DirectionalCoefficients::default();
        assert_eq!(
            cache
                .get_retained(&direction, 4, 2, |d, _| {
                    assert_eq!(d, direction);
                    Ok(vec![0.25, 2.0_f64.powi(-56)])
                })
                .unwrap(),
            vec![0.25, 2.0_f64.powi(-56)]
        );
    }

    #[test]
    fn signed_homogeneous_cache_preserves_every_channel_and_nearby_path() {
        for order in 1..=4 {
            let mut cache = DirectionalCoefficients::default();
            let direction = [-0.5, 1.0, 0.25];
            let coefficient = |d: &[f64], n: usize| -> Result<Vec<f64>> {
                Ok(vec![
                    (0.31 * d[0] - 0.27 * d[1] + 0.19 * d[2]).powi(n as i32),
                    (-0.23 * d[0] + 0.47 * d[1] + 0.41 * d[2]).powi(n as i32),
                ])
            };
            let expected = coefficient(&direction, order).unwrap();
            for scale in [1.0_f64, -1.0, 2.0, -2.0] {
                let path = direction.map(|x| x * scale);
                let actual = cache.get(&path, order, 2, coefficient).unwrap();
                for channel in 0..2 {
                    assert!(
                        (actual[channel] - expected[channel] * scale.powi(order as i32)).abs()
                            < 1e-14
                    );
                }
            }
            assert_eq!(cache.evaluations, 1);
            let mut nearby = direction;
            nearby[2] = f64::from_bits(nearby[2].to_bits() + 1);
            cache.get(&nearby, order, 2, coefficient).unwrap();
            assert_eq!(cache.evaluations, 2, "nearby paths must remain independent");
            assert_eq!(
                cache
                    .get(&[0.0; 3], order, 2, |_, _| panic!(
                        "zero homogeneous coefficient"
                    ))
                    .unwrap(),
                vec![0.0; 2]
            );
        }
    }

    #[test]
    fn polarization_recovers_represented_small_terms_between_large_ones() {
        let directions: Vec<_> = (0..4)
            .map(|i| (0..4).map(|j| f64::from(i == j)).collect())
            .collect();
        let mut index = 0;
        let result = polarized_mixed(4, &directions, |_, _| {
            // Opposite corners have identical fourth-order coefficients.
            // Their exact represented signed sum is two in channel zero.
            let value = match index {
                0 | 2 | 13 | 15 => 1e16,
                1 | 14 => -1.0,
                _ => 0.0,
            };
            index += 1;
            Ok(vec![value, value * 2.0_f64.powi(-20)])
        })
        .unwrap();
        assert_eq!(result.value, vec![2.0, 2.0_f64.powi(-19)]);
        assert_eq!(index, 16);
        assert!(result
            .accumulation_roundoff
            .iter()
            .all(|e| *e > 0.0 && *e < 1e-12));
    }

    #[test]
    fn unrepresented_high_order_weights_retain_the_first_order_bound() {
        let directions = vec![vec![1.0]; 128];
        let result = polarized_mixed(1, &directions, |d, n| {
            Ok(vec![(d[0] / 64.0).powi(n as i32)])
        })
        .unwrap();
        assert_eq!(result.directional_evaluations, 128);
        assert!(result.value[0].is_finite());
        assert!(result.accumulation_roundoff[0] > f64::EPSILON.powi(2));
    }

    #[test]
    fn every_multichannel_tensor_element_matches_independent_exponential_derivatives() {
        let patterns = vec![
            vec![0.7, -0.3, 0.2, 0.9],
            vec![0.2, 0.6, -0.7, 0.4],
            vec![-0.4, 0.8, 0.5, 0.3],
        ];
        let axes = [[0.31, -0.23, 0.17, 0.41], [-0.37, 0.29, 0.47, 0.13]];
        for order in [3usize, 4] {
            let factorial: f64 = (1..=order).map(|i| i as f64).product();
            let result = polarized_tensor(&patterns, order, 2, 1, true, |d, _| {
                Ok(axes
                    .iter()
                    .map(|a| {
                        a.iter()
                            .zip(d)
                            .map(|(a, d)| a * d)
                            .sum::<f64>()
                            .powi(order as i32)
                            / factorial
                    })
                    .collect())
            })
            .unwrap();
            assert_eq!(
                result.independent_components,
                if order == 3 { 10 } else { 15 }
            );
            assert_eq!(
                result.permutation_comparisons,
                result.independent_components - 3
            );
            for (c, a) in axes.iter().enumerate() {
                for flat in 0..3usize.pow(order as u32) {
                    let exact: f64 = tensor_indices(flat, 3, order)
                        .iter()
                        .map(|&p| patterns[p].iter().zip(a).map(|(p, a)| p * a).sum::<f64>())
                        .product();
                    assert!((result.values[c][flat] - exact).abs() < 1e-12);
                }
                assert!(result.permutation_defects[c] < 1e-12);
            }
        }
        assert!(polarized_tensor(&patterns, 4, 2, 0, false, |_, _| panic!(
            "memory guard must precede evaluation"
        ))
        .is_err());
        assert!(polarized_tensor(&patterns, 4, 2, 1, false, |_, _| Ok(vec![1.0])).is_err());
    }
}

#[derive(Clone, Debug)]
pub struct MixedDerivative {
    pub value: Vec<f64>,
    /// Conservative floating-point accumulation estimate; errors in the
    /// directional backend are separate and must also be reported.
    pub accumulation_roundoff: Vec<f64>,
    pub directional_evaluations: usize,
}

/// Assemble a scalar Hessian or a higher derivative contracted with leading
/// directions. Intrinsic permutation reuse is explicit; this does not measure
/// independent ordered asymmetry. Both molecular and lattice drivers use it.
pub(crate) fn polarized_matrix(
    dimension: usize,
    leading: &[Vec<f64>],
    mut coefficient: impl FnMut(&[f64], usize) -> Result<Vec<f64>>,
) -> Result<(Vec<Vec<f64>>, f64)> {
    let (mut matrices, roundoff) = polarized_matrices(dimension, leading, &mut coefficient)?;
    if matrices.len() != 1 {
        return Err(DftError::InvalidInput(
            "polarized energy matrix needs one scalar coefficient per direction".into(),
        ));
    }
    Ok((matrices.remove(0), roundoff[0]))
}

/// Exact polarization of several independent energy channels. Only matrix
/// interchange is reused in production; acceptance fits every output entry.
pub(crate) fn polarized_matrices(
    dimension: usize,
    leading: &[Vec<f64>],
    mut coefficient: impl FnMut(&[f64], usize) -> Result<Vec<f64>>,
) -> Result<(Vec<Vec<Vec<f64>>>, Vec<f64>)> {
    if dimension == 0 || dimension.checked_mul(dimension).is_none() {
        return Err(DftError::InvalidInput(
            "invalid polarized matrix dimension".into(),
        ));
    }
    let mut matrices: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut roundoff: Vec<f64> = Vec::new();
    for i in 0..dimension {
        for j in 0..=i {
            let mut directions = leading.to_vec();
            for axis in [i, j] {
                let mut unit = vec![0.0; dimension];
                unit[axis] = 1.0;
                directions.push(unit);
            }
            let value = polarized_mixed(dimension, &directions, &mut coefficient)?;
            if matrices.is_empty() && !value.value.is_empty() {
                matrices = vec![vec![vec![0.0; dimension]; dimension]; value.value.len()];
                roundoff = vec![0.0; value.value.len()];
            }
            if value.value.is_empty() || value.value.len() != matrices.len() {
                return Err(DftError::InvalidInput(
                    "polarized matrix energy channels must be nonempty and consistent".into(),
                ));
            }
            for c in 0..matrices.len() {
                matrices[c][i][j] = value.value[c];
                matrices[c][j][i] = value.value[c];
                roundoff[c] = roundoff[c].max(value.accumulation_roundoff[c]);
            }
        }
    }
    Ok((matrices, roundoff))
}

pub fn polarized_mixed(
    dimension: usize,
    directions: &[Vec<f64>],
    mut coefficient: impl FnMut(&[f64], usize) -> Result<Vec<f64>>,
) -> Result<MixedDerivative> {
    if directions
        .iter()
        .any(|d| d.len() != dimension || d.iter().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "invalid mixed derivative directions".into(),
        ));
    }
    if directions.is_empty() {
        let value = coefficient(&vec![0.0; dimension], 0)?;
        if value.iter().any(|x| !x.is_finite()) {
            return Err(DftError::InvalidInput(
                "non-finite mixed derivative output".into(),
            ));
        }
        return Ok(MixedDerivative {
            accumulation_roundoff: vec![0.0; value.len()],
            value,
            directional_evaluations: 1,
        });
    }
    let mut groups: Vec<(Vec<f64>, usize)> = Vec::new();
    for direction in directions {
        if let Some((_, count)) = groups.iter_mut().find(|(d, _)| d == direction) {
            *count += 1;
        } else {
            groups.push((direction.clone(), 1));
        }
    }
    let evaluations = groups
        .iter()
        .try_fold(1_usize, |n, (_, m)| n.checked_mul(m + 1))
        .ok_or_else(|| {
            DftError::InvalidInput("mixed derivative polarization size overflow".into())
        })?;
    let mut counters = vec![0_usize; groups.len()];
    let mut sum: Vec<crate::linalg::AccurateSum> = Vec::new();
    let mut magnitude = Vec::new();
    let mut calls = 0_usize;
    let mut exact_weights = true;
    for _ in 0..evaluations {
        let mut direction = vec![0.0; dimension];
        let mut weight = 1.0;
        for ((v, m), &k) in groups.iter().zip(&counters) {
            let mut choose = 1.0;
            let mut integer_choose = Some(1u128);
            for j in 0..k.min(m - k) {
                choose *= (m - j) as f64 / (j + 1) as f64;
                integer_choose = integer_choose
                    .and_then(|v| v.checked_mul((m - j) as u128))
                    .map(|v| v / (j + 1) as u128);
            }
            let represented =
                integer_choose.is_some_and(|v| v <= (1u128 << 53) && choose == v as f64);
            let factor = if (m - k) % 2 == 0 { choose } else { -choose };
            let next_weight = weight * factor;
            exact_weights &= represented && weight.mul_add(factor, -next_weight) == 0.0;
            weight = next_weight;
            for (d, &v) in direction.iter_mut().zip(v) {
                // Centering is an exact homogeneous polarization identity,
                // not a finite-difference displacement. It reduces cancellation
                // between large translated directional coefficients.
                *d += (k as f64 - 0.5 * *m as f64) * v;
            }
        }
        if !weight.is_finite() || direction.iter().any(|x| !x.is_finite()) {
            return Err(DftError::InvalidInput(
                "mixed derivative polarization overflows".into(),
            ));
        }
        if direction.iter().any(|v| *v != 0.0) {
            let value = coefficient(&direction, directions.len())?;
            calls += 1;
            if calls == 1 {
                sum = vec![crate::linalg::AccurateSum::default(); value.len()];
                magnitude = vec![0.0; value.len()];
            }
            if value.len() != sum.len() || value.iter().any(|x| !x.is_finite()) {
                return Err(DftError::InvalidInput(
                    "mixed derivative backend returned inconsistent outputs".into(),
                ));
            }
            for i in 0..value.len() {
                // Keep both multiplication and summation remainders. A
                // Kahan sum that first rounds weight*coefficient loses the
                // product remainder before cancellation can recover it.
                sum[i].add_product(weight, value[i]);
                let product = weight * value[i];
                magnitude[i] += product.abs();
            }
        }
        for g in 0..groups.len() {
            counters[g] += 1;
            if counters[g] <= groups[g].1 {
                break;
            }
            counters[g] = 0;
        }
    }
    if sum.iter().any(|x| !x.value().is_finite()) || magnitude.iter().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput(
            "mixed derivative summation overflows".into(),
        ));
    }
    let rounding = 4.0 * (calls as f64 + 1.0) * f64::EPSILON;
    let denominator = 1.0 - rounding;
    let gamma = rounding / denominator;
    Ok(MixedDerivative {
        value: sum.iter().map(|s| s.value()).collect(),
        // The retained low part measures the final rounding. Accumulating
        // the product/sum remainders adds a second-order error bounded by
        // the absolute term sum. This bounds this accumulation only; input
        // coefficient/backend errors remain separate, as for MixedDerivative.
        // Actual correction losses are tracked with upward rounding. The
        // scale-based padding is not a substitute for that a posteriori
        // bound. Unrepresented binomial weights keep a first-order bound.
        accumulation_roundoff: sum
            .iter()
            .zip(&magnitude)
            .map(|(sum, m)| {
                if denominator <= 0.0 {
                    f64::INFINITY
                } else if exact_weights {
                    sum.roundoff_bound() + rounding * f64::EPSILON * (m / denominator)
                } else {
                    // Very high orders may not represent their binomial
                    // weights exactly. Preserve the earlier first-order bound
                    // until a retained-weight provider is supplied as well.
                    sum.roundoff_bound() + gamma * (m / denominator)
                }
            })
            .collect(),
        directional_evaluations: calls,
    })
}
