// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spin-resolved semicanonical full-Coulomb PT2 and its stationary amplitude
//! Lagrangian. Scalar-generic orbitals, integrals and energies carry analytic
//! mixed amplitude/orbital/parameter paths through the same contractions.

use crate::dual::Scalar;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct OrbitalSpace<S> {
    pub energies: Vec<S>,
    /// Row-major AO by MO coefficients.
    pub coefficients: Vec<S>,
    pub occupied: usize,
}

#[derive(Clone, Debug)]
pub struct Pt2Problem<S> {
    pub energies: [Vec<S>; 2],
    pub occupied: [usize; 2],
    /// Blocks alpha-alpha, alpha-beta, beta-beta, each [i,a,j,b].
    pub ovov: [Vec<S>; 3],
}

#[derive(Clone, Debug)]
pub struct Pt2Solution<S> {
    pub same_spin_alpha: S,
    pub same_spin_beta: S,
    pub opposite_spin: S,
    pub correlation: S,
    /// Antisymmetrized same-spin amplitudes and direct opposite-spin ones.
    pub amplitudes: [Vec<S>; 3],
    /// max |Delta t - g|, before projection or denominator modification.
    pub amplitude_residual: f64,
    pub smallest_excitation_denominator: Option<f64>,
}

fn product(shape: [usize; 4]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| {
            DftError::InvalidInput("PT2 tensor shape overflows addressable memory".into())
        })
}
fn index(shape: [usize; 4], x: [usize; 4]) -> usize {
    ((x[0] * shape[1] + x[1]) * shape[2] + x[2]) * shape[3] + x[3]
}

/// Four successive one-index transformations, O(N^5), with the same scalar
/// operations for energies and AD paths. No finite differences are used.
pub fn from_ao<S: Scalar>(
    nao: usize,
    integrals: &[S],
    orbitals: &[OrbitalSpace<S>; 2],
) -> Result<Pt2Problem<S>> {
    if nao == 0
        || integrals.len() != product([nao; 4])?
        || integrals.iter().any(|x| !x.value().is_finite())
        || orbitals.iter().any(|o| {
            o.occupied > o.energies.len()
                || o.coefficients.len() != nao.saturating_mul(o.energies.len())
                || o.energies
                    .iter()
                    .chain(&o.coefficients)
                    .any(|v| !v.value().is_finite())
        })
    {
        return Err(DftError::InvalidInput(
            "PT2 needs finite AO integrals and dimensioned orbital spaces".into(),
        ));
    }
    let mut blocks = [Vec::new(), Vec::new(), Vec::new()];
    for (channel, (left, right)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        let spins = [left, left, right, right];
        let spaces = [true, false, true, false];
        let mut shape = [nao; 4];
        let mut tensor = integrals.to_vec();
        for axis in 0..4 {
            let o = &orbitals[spins[axis]];
            let (offset, count) = if spaces[axis] {
                (0, o.occupied)
            } else {
                (o.occupied, o.energies.len() - o.occupied)
            };
            let mut next_shape = shape;
            next_shape[axis] = count;
            let mut transformed = vec![S::from_f64(0.0); product(next_shape)?];
            for i in 0..next_shape[0] {
                for a in 0..next_shape[1] {
                    for j in 0..next_shape[2] {
                        for b in 0..next_shape[3] {
                            let x = [i, a, j, b];
                            let mut source = x;
                            let mut sum = S::from_f64(0.0);
                            for mu in 0..nao {
                                source[axis] = mu;
                                sum = sum
                                    + tensor[index(shape, source)]
                                        * o.coefficients[mu * o.energies.len() + offset + x[axis]];
                            }
                            transformed[index(next_shape, x)] = sum;
                        }
                    }
                }
            }
            shape = next_shape;
            tensor = transformed;
        }
        blocks[channel] = tensor;
    }
    Ok(Pt2Problem {
        energies: std::array::from_fn(|s| orbitals[s].energies.clone()),
        occupied: std::array::from_fn(|s| orbitals[s].occupied),
        ovov: blocks,
    })
}

impl<S: Scalar> Pt2Problem<S> {
    pub(super) fn visit(&self, mut term: impl FnMut(usize, usize, S, S, f64)) -> Result<()> {
        if self
            .energies
            .iter()
            .zip(self.occupied)
            .any(|(e, n)| n > e.len() || e.iter().any(|v| !v.value().is_finite()))
        {
            return Err(DftError::InvalidInput(
                "invalid PT2 semicanonical orbital energies".into(),
            ));
        }
        for (channel, (left, right)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let (nl, nr) = (self.occupied[left], self.occupied[right]);
            let (vl, vr) = (
                self.energies[left].len() - nl,
                self.energies[right].len() - nr,
            );
            let shape = [nl, vl, nr, vr];
            if self.ovov[channel].len() != product(shape)?
                || self.ovov[channel].iter().any(|v| !v.value().is_finite())
            {
                return Err(DftError::InvalidInput(
                    "invalid PT2 occupied-virtual integral block".into(),
                ));
            }
            for i in 0..nl {
                for a in 0..vl {
                    for j in 0..nr {
                        for b in 0..vr {
                            let p = index(shape, [i, a, j, b]);
                            let coupling = if left == right {
                                self.ovov[channel][p]
                                    - self.ovov[channel][index(shape, [i, b, j, a])]
                            } else {
                                self.ovov[channel][p]
                            };
                            let denominator = self.energies[left][i] + self.energies[right][j]
                                - self.energies[left][nl + a]
                                - self.energies[right][nr + b];
                            if !denominator.value().is_finite() || denominator.value() >= 0.0 {
                                return Err(DftError::InvalidInput("PT2 needs strictly negative physical pair denominators; no regularization or floor is applied".into()));
                            }
                            term(
                                channel,
                                p,
                                coupling,
                                denominator,
                                if left == right { 0.25 } else { 1.0 },
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub fn solve(&self, coefficients: Pt2Coefficients) -> Result<Pt2Solution<S>> {
        coefficients.validate()?;
        let mut amplitudes = self
            .ovov
            .each_ref()
            .map(|block| vec![S::from_f64(0.0); block.len()]);
        let mut energies = [S::from_f64(0.0); 3];
        let mut residual = 0.0_f64;
        let mut smallest: Option<f64> = None;
        let mut nonfinite = false;
        self.visit(|channel, p, g, delta, weight| {
            let t = g / delta;
            nonfinite |= !g.value().is_finite() || !t.value().is_finite();
            amplitudes[channel][p] = t;
            energies[channel] = energies[channel] + (g * t).scale(weight);
            residual = residual.max((delta * t - g).value().abs());
            smallest = Some(smallest.map_or(-delta.value(), |v| v.min(-delta.value())));
        })?;
        if nonfinite || energies.iter().any(|e| !e.value().is_finite()) {
            return Err(DftError::InvalidInput(
                "unregularized PT2 amplitudes or energy overflow the scalar value range".into(),
            ));
        }
        Ok(Pt2Solution {
            same_spin_alpha: energies[0],
            same_spin_beta: energies[2],
            opposite_spin: energies[1],
            correlation: (energies[0] + energies[2]).scale(coefficients.same_spin)
                + energies[1].scale(coefficients.opposite_spin),
            amplitudes,
            amplitude_residual: residual,
            smallest_excitation_denominator: smallest,
        })
    }
    /// Stationary amplitude functional sum w(2 g t - Delta t^2).
    /// AD differentiates amplitudes, orbital coefficients, denominators,
    /// integrals and their mixed paths together. Orbital paths must come
    /// from the reference response; this contraction does not solve CPKS.
    pub fn lagrangian(&self, amplitudes: &[Vec<S>; 3], coefficients: Pt2Coefficients) -> Result<S> {
        coefficients.validate()?;
        if amplitudes
            .iter()
            .zip(&self.ovov)
            .any(|(a, g)| a.len() != g.len() || a.iter().any(|v| !v.value().is_finite()))
        {
            return Err(DftError::InvalidInput(
                "invalid PT2 Lagrangian amplitudes".into(),
            ));
        }
        let mut value = S::from_f64(0.0);
        self.visit(|channel, p, g, delta, weight| {
            let t = amplitudes[channel][p];
            value = value
                + (g * t.scale(2.0) - delta * t * t).scale(
                    weight
                        * if channel == 1 {
                            coefficients.opposite_spin
                        } else {
                            coefficients.same_spin
                        },
                );
        })?;
        Ok(value)
    }
}
