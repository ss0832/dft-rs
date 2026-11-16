// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving-centre continuum magnetic-translation operators. The same scalar
//! contraction supplies the SCF matrices and nuclear Taylor coefficients.

use super::*;
use crate::basis::n_sph;
use crate::dual::{Scalar, Taylor};
use crate::integrals::analytic_basis::{contracted_components, AnalyticShell};
use crate::periodic::planewave::plane_wave_primitive_g;

/// One-electron operator in an ordered magnetic-translation basis.
#[derive(Clone, Copy, Debug)]
pub enum MagneticPairOperator {
    /// exp(i G.r), including G=0 overlap.
    Fourier,
    /// Bilinear kinetic-energy density: sum_axis (pi_axis bra)* exp(iG.r)
    /// (pi_axis ket)/2, with pi=p+A. At G=0 this is (p+A)^2/2.
    Kinetic,
    /// exp(i G.r) (p+A)_axis, with axis 0..2.
    Momentum(usize),
}

fn weighted<S: Scalar>(target: &mut (S, S), value: (S, S), coefficient: Complex) {
    target.0 = target.0 + value.0.scale(coefficient.re) - value.1.scale(coefficient.im);
    target.1 = target.1 + value.0.scale(coefficient.im) + value.1.scale(coefficient.re);
}

pub(super) fn scalar_pair_operator<S: Scalar>(
    cell: &MagneticCell,
    basis: &MolBasis,
    cutoff: f64,
    wave: Vec3,
    kind: Option<usize>,
    centres: &[[S; 3]],
    cos_sin: impl Fn(S) -> (S, S) + Copy,
) -> Result<[Vec<S>; 2]> {
    if basis.nao_sph == 0
        || centres.len() != basis.shells.len()
        || !cutoff.is_finite()
        || cutoff <= 0.0
        || !wave.norm().is_finite()
        || kind.is_some_and(|k| k > 3)
        || centres.iter().flatten().any(|x| !x.value().is_finite())
    {
        return Err(DftError::InvalidInput(
            "magnetic operators require AOs, finite centres/wave vector and positive cutoff".into(),
        ));
    }
    let n = basis.nao_sph;
    let zero = S::from_f64(0.0);
    let mut out = [vec![zero; n * n], vec![zero; n * n]];
    let shells = basis
        .shells
        .iter()
        .zip(centres)
        .map(|(shell, centre)| AnalyticShell::from_placed(shell, *centre))
        .collect::<Result<Vec<_>>>()?;
    for (a, left) in basis.shells.iter().enumerate() {
        for (b, right) in basis.shells.iter().enumerate() {
            // Fix the image partition at the reference geometry; convergence
            // of this truncation is an independent integration requirement.
            for index in cell
                .geometry
                .nearby_translation_indices(right.center - left.center, cutoff)?
            {
                let translation = cell.geometry.lattice().translation(index);
                let mut ket = shells[b].clone();
                for axis in 0..3 {
                    ket.center[axis] = ket.center[axis] + S::from_f64(translation.get(axis));
                }
                let delta: [S; 3] =
                    std::array::from_fn(|axis| ket.center[axis] - shells[a].center[axis]);
                let q: [S; 3] = std::array::from_fn(|axis| {
                    let (x, y) = ((axis + 1) % 3, (axis + 2) % 3);
                    S::from_f64(wave.get(axis))
                        - (delta[y].scale(cell.field.get(x)) - delta[x].scale(cell.field.get(y)))
                            .scale(0.5)
                });
                let block = contracted_components::<S, 2, 2>([&shells[a], &ket], |[pa, pb]| {
                    let integral = |la, lb| {
                        plane_wave_primitive_g(
                            la,
                            lb,
                            pa.exponent.value(),
                            pb.exponent.value(),
                            pa.center,
                            pb.center,
                            q,
                            cos_sin,
                        )
                    };
                    let mut value = (zero, zero);
                    match kind {
                        None => value = integral(pa.angular, pb.angular),
                        Some(3) => {
                            for axis in 0..3 {
                                for (la, ca) in momentum_terms(
                                    pa.angular,
                                    pa.exponent.value(),
                                    cell.field,
                                    axis,
                                ) {
                                    for (lb, cb) in momentum_terms(
                                        pb.angular,
                                        pb.exponent.value(),
                                        cell.field,
                                        axis,
                                    ) {
                                        weighted(
                                            &mut value,
                                            integral(la, lb),
                                            ca.conj() * cb * 0.5,
                                        );
                                    }
                                }
                            }
                        }
                        Some(axis) => {
                            for (lb, c) in
                                momentum_terms(pb.angular, pb.exponent.value(), cell.field, axis)
                            {
                                weighted(&mut value, integral(pa.angular, lb), c);
                            }
                        }
                    }
                    Ok([value.0, value.1])
                })?;
                let phase = (centres[b][1].scale(translation.z)
                    - centres[b][2].scale(translation.y))
                .scale(0.5 * cell.field.x)
                    + (centres[b][2].scale(translation.x) - centres[b][0].scale(translation.z))
                        .scale(0.5 * cell.field.y)
                    + (centres[b][0].scale(translation.y) - centres[b][1].scale(translation.x))
                        .scale(0.5 * cell.field.z);
                let (c, s) = cos_sin(phase);
                let constant = cell.image_phase(Vec3::zero(), index);
                let coefficient = (
                    c.scale(constant.re) - s.scale(constant.im),
                    s.scale(constant.re) + c.scale(constant.im),
                );
                for i in 0..n_sph(left.l) {
                    for j in 0..n_sph(right.l) {
                        let k = i * n_sph(right.l) + j;
                        let r = block[0].values[k];
                        let im = block[1].values[k];
                        let target = (left.sph_off + i) * n + right.sph_off + j;
                        out[0][target] = out[0][target] + coefficient.0 * r - coefficient.1 * im;
                        out[1][target] = out[1][target] + coefficient.0 * im + coefficient.1 * r;
                    }
                }
            }
        }
    }
    Ok(out)
}

impl MagneticCell {
    /// Taylor coefficients of a fixed-cell continuum magnetic operator as
    /// nuclei and their Gaussian centres move together. Field, reciprocal
    /// wave, magnetic Bloch sector and translation cocycle stay fixed.
    /// This integral component does not itself solve the interacting nuclear
    /// response, Berry electric bias or finite-q DFPT equations.
    pub fn nuclear_operator_taylor(
        &self,
        basis: &MolBasis,
        cutoff: f64,
        wave: Vec3,
        operator: MagneticPairOperator,
        direction: &[[f64; 3]],
        order: usize,
    ) -> Result<Vec<BlochMatrix>> {
        if order > 4
            || direction.is_empty()
            || direction.iter().flatten().any(|x| !x.is_finite())
            || basis.shells.iter().any(|s| s.atom >= direction.len())
            || matches!(operator,MagneticPairOperator::Momentum(axis) if axis>=3)
        {
            return Err(DftError::InvalidInput("magnetic nuclear operators need finite atom directions, axis 0..2 and order zero through four".into()));
        }
        self.validate_wave(wave)?;
        let centres: Vec<_> = basis
            .shells
            .iter()
            .map(|s| {
                std::array::from_fn(|a| {
                    Taylor::<5>::from_coefficients(&[s.center.get(a), direction[s.atom][a]])
                })
            })
            .collect();
        let kind = match operator {
            MagneticPairOperator::Fourier => None,
            MagneticPairOperator::Kinetic => Some(3),
            MagneticPairOperator::Momentum(a) => Some(a),
        };
        let values = scalar_pair_operator(self, basis, cutoff, wave, kind, &centres, |x| {
            let (s, c) = x.sin_cos();
            (c, s)
        })?;
        Ok((0..=order)
            .map(|k| BlochMatrix {
                re: crate::linalg::Matrix::from_row_major(
                    basis.nao_sph,
                    basis.nao_sph,
                    values[0].iter().map(|v| v.c[k]).collect(),
                ),
                im: crate::linalg::Matrix::from_row_major(
                    basis.nao_sph,
                    basis.nao_sph,
                    values[1].iter().map(|v| v.c[k]).collect(),
                ),
            })
            .collect())
    }
}
