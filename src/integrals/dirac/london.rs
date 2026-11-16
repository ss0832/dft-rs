// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Products of London phases with the Gaussian moments used by magnetic balance.

use super::{
    constant, potential_moment, CenterJet, GaussianNucleus, MagneticMoments, PairMoments, POWERS,
};
use crate::dual::Scalar;
use crate::integrals::analytic_basis::{self, AnalyticShell};
use crate::Result;
use std::collections::BTreeMap;

pub(super) fn half_cross<S: Scalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [
        (a[1] * b[2] - a[2] * b[1]).scale(0.5),
        (a[2] * b[0] - a[0] * b[2]).scale(0.5),
        (a[0] * b[1] - a[1] * b[0]).scale(0.5),
    ]
}

/// Rotate the truncated local phase by the exact phase at the expansion
/// point. In particular the truncation never acts on a rigid translation.
fn phase_at<S: Scalar>(argument: S, order: usize) -> [S; 2] {
    let delta = argument - S::from_f64(argument.value());
    let (sin, cos) = argument.value().sin_cos();
    let mut term = [S::from_f64(cos), S::from_f64(sin)];
    let mut value = term;
    for n in 1..=order {
        let scale = delta.scale(1.0 / n as f64);
        term = [-scale * term[1], scale * term[0]];
        value[0] = value[0] + term[0];
        value[1] = value[1] + term[1];
    }
    value
}

pub(super) fn multiply_phase<S: Scalar>(value: [S; 2], degree: usize) -> [S; 2] {
    let [re, im] = value;
    match degree % 4 {
        0 => [re, im],
        1 => [-im, re],
        2 => [-re, -im],
        _ => [im, -re],
    }
}

/// Polynomial coefficients of exp(i*k*(r-origin)), including the exact
/// exp(i*k*origin) factor. Powers and the plane-wave phase are shared by
/// the one- and two-electron providers.
pub(super) fn phase_coefficients<S: Scalar>(
    k: [S; 3],
    origin: [S; 3],
    order: usize,
    phase_order: usize,
) -> Vec<([usize; 3], [S; 2])> {
    let phase = phase_at(
        (0..3).fold(S::from_f64(0.0), |v, i| v + k[i] * origin[i]),
        order,
    );
    let powers: [Vec<S>; 3] = k.map(|component| {
        let mut powers = vec![S::from_f64(1.0); phase_order + 1];
        for n in 1..=phase_order {
            powers[n] = (powers[n - 1] * component).scale(1.0 / n as f64);
        }
        powers
    });
    let mut terms = Vec::new();
    for x in 0..=phase_order {
        for y in 0..=phase_order - x {
            for z in 0..=phase_order - x - y {
                let weight = powers[0][x] * powers[1][y] * powers[2][z];
                terms.push((
                    [x, y, z],
                    multiply_phase(phase, x + y + z).map(|v| v * weight),
                ));
            }
        }
    }
    terms
}

/// The small basis uses A centred on its own spatial AO after the derivative
/// of the London phase has cancelled the global vector-potential offset.
/// Here k, the midpoint and the local gauge offsets retain outer derivatives.
/// The inner endpoint jets differentiate only the unphased Gaussian factors.
#[allow(clippy::too_many_arguments)]
pub(super) fn moments<S: Scalar>(
    a: &AnalyticShell<CenterJet<S>>,
    b: &AnalyticShell<CenterJet<S>>,
    centers: [[S; 3]; 2],
    nuclei: &[GaussianNucleus<S>],
    field: [S; 3],
    order: usize,
    phase_order: usize,
) -> Result<MagneticMoments<S>> {
    let origin = std::array::from_fn(|i| (centers[0][i] + centers[1][i]).scale(0.5));
    let k = half_cross(
        field,
        std::array::from_fn(|i| centers[0][i] - centers[1][i]),
    );
    let offsets = centers.map(|r| half_cross(field, std::array::from_fn(|i| origin[i] - r[i])));
    let zero = S::from_f64(0.0);
    let count = (2 * a.l as usize + 1) * (2 * b.l as usize + 1);
    let mut overlap = std::array::from_fn(|_| vec![[constant(zero); 2]; count]);
    let mut potential = overlap.clone();
    let mut integrals = BTreeMap::new();
    for ([x, y, z], coefficient) in phase_coefficients(k, origin, order, phase_order) {
        let coefficient = coefficient.map(constant);
        for (i, &[px, py, pz]) in POWERS.iter().enumerate() {
            let key = [px + x, py + y, pz + z];
            let (s, v) = match integrals.entry(key) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let s = analytic_basis::moment(a, b, key, origin.map(constant))?.values;
                    let v = potential_moment(a, b, nuclei, order, key, origin)?;
                    entry.insert((s, v))
                }
            };
            for j in 0..count {
                for c in 0..2 {
                    overlap[i][j][c] = overlap[i][j][c] + s[j] * coefficient[c];
                    potential[i][j][c] = potential[i][j][c] + v[j] * coefficient[c];
                }
            }
        }
    }
    Ok(MagneticMoments {
        overlap,
        potential,
        offsets,
    })
}

/// Three dipole operators in the same moving magnetic basis. The phase is
/// expanded at the pair midpoint. Multiplication by r-origin adds one
/// Gaussian power and a constant shift for each Cartesian component.
pub(super) fn dipole_moments(
    a: &AnalyticShell<CenterJet<f64>>,
    b: &AnalyticShell<CenterJet<f64>>,
    centers: [[f64; 3]; 2],
    position_origin: [f64; 3],
    field: [f64; 3],
    phase_order: usize,
) -> Result<([PairMoments<f64>; 3], [[f64; 3]; 2])> {
    let center = std::array::from_fn(|i| 0.5 * (centers[0][i] + centers[1][i]));
    let k = half_cross(
        field,
        std::array::from_fn(|i| centers[0][i] - centers[1][i]),
    );
    let offsets = centers.map(|r| half_cross(field, std::array::from_fn(|i| center[i] - r[i])));
    let count = (2 * a.l as usize + 1) * (2 * b.l as usize + 1);
    let mut dipoles =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![[constant(0.0); 2]; count]));
    let mut integrals: BTreeMap<[usize; 3], Vec<CenterJet<f64>>> = BTreeMap::new();
    for (phase, coefficient) in phase_coefficients(k, center, 0, phase_order) {
        if coefficient == [0.0; 2] {
            continue;
        }
        let coefficient = coefficient.map(constant);
        for (i, base) in POWERS.iter().enumerate() {
            let key = std::array::from_fn(|axis| base[axis] + phase[axis]);
            let mut lookup = |power| -> Result<Vec<CenterJet<f64>>> {
                match integrals.entry(power) {
                    std::collections::btree_map::Entry::Occupied(entry) => Ok(entry.get().clone()),
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        let data = analytic_basis::moment(a, b, power, center.map(constant))?;
                        Ok(entry.insert(data.values).clone())
                    }
                }
            };
            let overlap = lookup(key)?;
            for axis in 0..3 {
                let mut raised = key;
                raised[axis] += 1;
                let position = lookup(raised)?;
                let shift = center[axis] - position_origin[axis];
                for (j, (pos, s)) in position.into_iter().zip(&overlap).enumerate() {
                    let value = pos + s.scale(shift);
                    for (c, coef) in coefficient.iter().copied().enumerate() {
                        dipoles[axis][i][j][c] = dipoles[axis][i][j][c] + value * coef;
                    }
                }
            }
        }
    }
    Ok((dipoles, offsets))
}
