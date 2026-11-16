// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Four-component finite Gaussian nuclear magnetic-dipole vertices.

use super::london::{half_cross, phase_coefficients};
use super::{block, constant, endpoint, pauli_tensor, shell_offsets, CenterJet};
use crate::dual::{DualN, Scalar};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{self, AnalyticNucleus, AnalyticShell, IntegralBlock};
use crate::{DftError, Result};
use std::collections::BTreeMap;

/// Gaussian nuclear magnetization centre and RMS radius in Bohr.
/// The magnetic radius is independent of the nuclear charge radius.
#[derive(Clone, Copy, Debug)]
pub struct GaussianMagnetization<S: Scalar> {
    pub center: [S; 3],
    pub rms_radius: S,
}

fn gradient_moment<S: Scalar>(
    a: &AnalyticShell<CenterJet<S>>,
    b: &AnalyticShell<CenterJet<S>>,
    nucleus: GaussianMagnetization<S>,
    origin: [S; 3],
    power: [usize; 3],
    order: usize,
) -> Result<Vec<[CenterJet<S>; 3]>> {
    let shell = |x: &AnalyticShell<CenterJet<S>>| AnalyticShell {
        l: x.l,
        center: x.center.map(DualN::constant),
        exponents: x.exponents.iter().copied().map(DualN::constant).collect(),
        coefficients: x
            .coefficients
            .iter()
            .copied()
            .map(DualN::constant)
            .collect(),
    };
    let a = shell(a);
    let b = shell(b);
    let source = AnalyticNucleus {
        center: std::array::from_fn(|i| DualN::variable(constant(nucleus.center[i]), i)),
        charge: DualN::constant(constant(S::from_f64(1.0))),
    };
    let result = analytic_basis::attraction_moment(
        &a,
        &b,
        &[source],
        GaussianKernel::LongRange {
            omega: DualN::constant(constant(S::from_f64(1.5_f64.sqrt()) / nucleus.rms_radius)),
        },
        order + 3,
        power,
        origin.map(|x| DualN::constant(constant(x))),
    )?;
    // Attraction uses the minus sign of an electron in the scalar field.
    // The magnetic potential needs +grad_R[erf(omega|r-R|)/|r-R|].
    Ok(result
        .values
        .into_iter()
        .map(|jet| jet.im.map(|x| -x))
        .collect())
}

type PhasedGradient<S> = [CenterJet<S>; 2];
type CartesianGradient<S> = [PhasedGradient<S>; 3];
type PairGradients<S> = [Vec<CartesianGradient<S>>; 4];

fn phased_gradient_moments<S: Scalar + PartialEq>(
    a: &AnalyticShell<CenterJet<S>>,
    b: &AnalyticShell<CenterJet<S>>,
    centers: [[S; 3]; 2],
    nucleus: GaussianMagnetization<S>,
    field: [S; 3],
    order: usize,
    phase_order: usize,
) -> Result<PairGradients<S>> {
    let origin = std::array::from_fn(|i| (centers[0][i] + centers[1][i]).scale(0.5));
    let wave = half_cross(
        field,
        std::array::from_fn(|i| centers[0][i] - centers[1][i]),
    );
    let zero = S::from_f64(0.0);
    let count = (2 * a.l as usize + 1) * (2 * b.l as usize + 1);
    let mut result = std::array::from_fn(|_| vec![[[constant(zero); 2]; 3]; count]);
    let mut cached = BTreeMap::new();
    for (initial, coefficient) in phase_coefficients(wave, origin, order, phase_order) {
        if coefficient == [zero; 2] {
            continue;
        }
        let coefficients = coefficient.map(constant);
        for (k, target) in result.iter_mut().enumerate() {
            let mut power = initial;
            if k > 0 {
                power[k - 1] += 1;
            }
            let source = match cached.entry(power) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(gradient_moment(a, b, nucleus, origin, power, order)?)
                }
            };
            for (out, values) in target.iter_mut().zip(source) {
                for (row, value) in out.iter_mut().zip(values) {
                    for (component, weight) in row.iter_mut().zip(coefficients) {
                        *component = *component + *value * weight;
                    }
                }
            }
        }
    }
    Ok(result)
}

/// Covariant vertices dH/d(mu_i), where the moment unit is
/// Hartree per atomic magnetic field unit. A Gaussian magnetization gives
/// A_mu=(mu cross grad_R(erf(omega|r-R|)/|r-R|))/c^2. The nuclear moment
/// acts in the Dirac operator. The small basis is balanced with respect to
/// the external uniform field. All three Cartesian vertices contain both
/// large-small and small-large blocks on the same London-RMB path.
pub fn nuclear_magnetic_dipole_vertices<S: Scalar + PartialEq>(
    shells: &[AnalyticShell<S>],
    nucleus: GaussianMagnetization<S>,
    speed_of_light: S,
    external_field: [S; 3],
    order: usize,
    phase_order: usize,
) -> Result<[[IntegralBlock<S, 2>; 2]; 3]> {
    if !speed_of_light.value().is_finite()
        || speed_of_light.value() <= 0.0
        || !nucleus.rms_radius.value().is_finite()
        || nucleus.rms_radius.value() <= 0.0
        || nucleus
            .center
            .iter()
            .chain(&external_field)
            .any(|x| !x.value().is_finite())
        || order > phase_order
        || phase_order > 8
    {
        return Err(DftError::InvalidInput(
            "a nuclear magnetic moment needs finite centres and field, positive magnetic radius and c, and a phase order no smaller than the derivative order through eight".into(),
        ));
    }
    let offsets = shell_offsets(shells)?;
    let n = *offsets.last().unwrap();
    let size = n.checked_mul(4).filter(|&x| x > 0).ok_or_else(|| {
        DftError::InvalidInput("invalid four-component nuclear magnetic dimension".into())
    })?;
    let mut matrices: Vec<_> = (0..3)
        .map(|_| Ok([block(size, size)?, block(size, size)?]))
        .collect::<Result<_>>()?;
    let zero = S::from_f64(0.0);
    let [bx, by, bz] = external_field.map(|x| x.scale(0.5));
    let vector = [[zero, -bz, by], [bz, zero, -bx], [-by, bx, zero]];
    let factor = S::from_f64(0.5) / speed_of_light.powi(2);
    let left: Vec<_> = shells.iter().map(|s| endpoint(s, true)).collect();
    let right: Vec<_> = shells.iter().map(|s| endpoint(s, false)).collect();
    for (a, bra) in left.iter().enumerate() {
        for (b, ket) in right.iter().enumerate() {
            let centers = [shells[a].center, shells[b].center];
            let moments = phased_gradient_moments(
                bra,
                ket,
                centers,
                nucleus,
                external_field,
                order,
                phase_order,
            )?;
            let displacement = |s: usize, t: usize| {
                std::array::from_fn(|i| (centers[s][i] - centers[t][i]).scale(0.5))
            };
            let offset_a = half_cross(external_field, displacement(1, 0));
            let offset_b = half_cross(external_field, displacement(0, 1));
            let (rows, cols) = (offsets[a + 1] - offsets[a], offsets[b + 1] - offsets[b]);
            for (axis, result) in matrices.iter_mut().enumerate() {
                for ia in 0..rows {
                    for ib in 0..cols {
                        let index = ia * cols + ib;
                        let mut ls = [[zero; 3]; 3];
                        let mut ls_im = ls;
                        let mut sl = ls;
                        let mut sl_im = ls;
                        let component = |i: usize, k: usize| {
                            if i == axis {
                                return [constant(zero); 2];
                            }
                            let sign = if (axis + 1) % 3 == i { -1.0 } else { 1.0 };
                            moments[k][index][3 - i - axis].map(|x| x.scale(sign))
                        };
                        for i in 0..3 {
                            for j in 0..3 {
                                let [re, im] = component(i, 0);
                                ls[i][j] = -im.im[j].re + offset_b[j] * re.re.re;
                                ls_im[i][j] = re.im[j].re + offset_b[j] * im.re.re;
                                let [re, im] = component(j, 0);
                                sl[i][j] = im.re.im[i] + offset_a[i] * re.re.re;
                                sl_im[i][j] = -re.re.im[i] + offset_a[i] * im.re.re;
                                for k in 0..3 {
                                    let [re, im] = component(i, k + 1);
                                    ls[i][j] = ls[i][j] + vector[j][k] * re.re.re;
                                    ls_im[i][j] = ls_im[i][j] + vector[j][k] * im.re.re;
                                    let [re, im] = component(j, k + 1);
                                    sl[i][j] = sl[i][j] + vector[i][k] * re.re.re;
                                    sl_im[i][j] = sl_im[i][j] + vector[i][k] * im.re.re;
                                }
                            }
                        }
                        for (pair, ((lsr, lsi), (slr, sli))) in pauli_tensor(ls, ls_im)
                            .into_iter()
                            .zip(pauli_tensor(sl, sl_im))
                            .enumerate()
                        {
                            let i = (pair / 2) * n + offsets[a] + ia;
                            let j = (pair % 2) * n + offsets[b] + ib;
                            let upper = i * size + 2 * n + j;
                            let lower = (i + 2 * n) * size + j;
                            (result[0].values[upper], result[1].values[upper]) =
                                (lsr * factor, lsi * factor);
                            (result[0].values[lower], result[1].values[lower]) =
                                (slr * factor, sli * factor);
                        }
                    }
                }
            }
        }
    }
    matrices
        .try_into()
        .map_err(|_| DftError::LinearAlgebra("three nuclear magnetic operators required".into()))
}
