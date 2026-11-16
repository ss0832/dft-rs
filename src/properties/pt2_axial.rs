// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Normalized first-order pair-wavefunction overlaps for VCD. Anti-Hermitian
//! orbital connections reduce the imaginary orbital/orbital overlap to a
//! one-body commutator; no triple-substitution states are stored. The shared
//! PT2 tensor action gives O(N^5) contractions with O(N^4) coefficient storage.
use crate::integrals::london::ComplexMatrix;
use crate::properties::pt2_response::complex_pair_action;
use crate::response::complex::Complex;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};

#[cfg(test)]
#[path = "pt2_axial_tests.rs"]
mod tests;

/// Complete anti-Hermitian MO connections, including the nuclear ket-basis
/// half derivative. Amplitudes follow the shared `[i,a,j,b]` convention.
pub struct PairStateTangent<'a> {
    pub connection: &'a [ComplexMatrix; 2],
    pub amplitudes: &'a [Vec<Complex>; 3],
}

/// The `Phi + T2 Phi` pair state in a complete collinear orbital frame.
/// Orbital labels have arbitrary capacity rather than a fixed-word mask.
pub struct PairAxialState {
    dimensions: [usize; 2],
    occupied: [usize; 2],
    amplitudes: [Vec<Complex>; 3],
}

/// Imaginary derivative-state overlap and its unscaled channel/norm data.
/// Channel bilinears are intermediate normalized, not additive normalized
/// corrections. Signed weights need a positive resulting norm.
pub struct PairAxialOverlap {
    pub value: f64,
    pub parent: f64,
    pub channel_bilinears: [f64; 3],
    pub channel_norms: [f64; 3],
    pub norm: f64,
    pub norm_derivative: f64,
    pub magnetic_connection: f64,
    pub connection_defect: f64,
    pub cancellation_defect: f64,
}

fn element(k: &ComplexMatrix, i: usize, j: usize) -> Complex {
    Complex::new(k.re[(i, j)], k.im[(i, j)])
}
fn inner(a: &[Complex], b: &[Complex], channel: usize) -> Complex {
    a.iter()
        .zip(b)
        .fold(Complex::ZERO, |s, (a, b)| s + a.conj() * *b)
        * if channel == 1 { 1.0 } else { 0.25 }
}

impl PairAxialState {
    pub fn new(
        dimensions: [usize; 2],
        occupied: [usize; 2],
        amplitudes: &[Vec<f64>; 3],
    ) -> Result<Self> {
        Self::with_memory_limit(
            dimensions,
            occupied,
            amplitudes,
            crate::scf::DEFAULT_INCORE_MB,
        )
    }
    /// Check the coefficient/connection workspace before allocating it.
    pub fn with_memory_limit(
        dimensions: [usize; 2],
        occupied: [usize; 2],
        amplitudes: &[Vec<f64>; 3],
        memory_mb: usize,
    ) -> Result<Self> {
        if occupied.iter().zip(dimensions).any(|(o, n)| *o > n)
            || dimensions[0].checked_add(dimensions[1]).is_none()
        {
            return Err(DftError::InvalidInput(
                "pair-state orbital dimensions or occupations are invalid".into(),
            ));
        }
        let mut count = 0usize;
        for (s, t) in [(0, 0), (0, 1), (1, 1)] {
            let length = [
                occupied[s],
                dimensions[s] - occupied[s],
                occupied[t],
                dimensions[t] - occupied[t],
            ]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .ok_or_else(|| {
                DftError::InvalidInput(
                    "pair-state tensor shape overflows addressable memory".into(),
                )
            })?;
            count = count.checked_add(length).ok_or_else(|| {
                DftError::InvalidInput(
                    "pair-state tensor workspace overflows addressable memory".into(),
                )
            })?;
        }
        let matrices = dimensions
            .iter()
            .try_fold(0usize, |sum, n| {
                n.checked_mul(*n).and_then(|v| sum.checked_add(v))
            })
            .and_then(|n| n.checked_mul(8));
        let bytes = count
            .checked_mul(10)
            .and_then(|n| matrices.and_then(|m| n.checked_add(m)))
            .and_then(|n| n.checked_mul(std::mem::size_of::<Complex>()))
            .ok_or_else(|| {
                DftError::InvalidInput(
                    "pair-state derivative workspace overflows addressable memory".into(),
                )
            })?;
        let required_mb = bytes.div_ceil(1024 * 1024);
        if required_mb > memory_mb {
            return Err(DftError::ResourceLimit {
                operation: "pair-state derivative overlaps",
                required_mb,
                limit_mb: memory_mb,
            });
        }
        let mut result = Self {
            dimensions,
            occupied,
            amplitudes: std::array::from_fn(|_| Vec::new()),
        };
        let complex = amplitudes
            .each_ref()
            .map(|v| v.iter().copied().map(Complex::from).collect());
        result.amplitudes = result.canonical(&complex)?;
        Ok(result)
    }
    fn shape(&self, c: usize) -> [usize; 4] {
        let (s, t) = [(0, 0), (0, 1), (1, 1)][c];
        [
            self.occupied[s],
            self.dimensions[s] - self.occupied[s],
            self.occupied[t],
            self.dimensions[t] - self.occupied[t],
        ]
    }
    fn canonical(&self, amplitudes: &[Vec<Complex>; 3]) -> Result<[Vec<Complex>; 3]> {
        let mut result = amplitudes.clone();
        for c in 0..3 {
            let [no, nv, nr, vr] = self.shape(c);
            if amplitudes[c].len() != no * nv * nr * vr
                || amplitudes[c].iter().any(|z| !z.is_finite())
            {
                return Err(DftError::InvalidInput("pair-state amplitudes need finite [i,a,j,b] blocks matching the orbital spaces".into()));
            }
            if c != 1 {
                let at = |i, a, j, b| amplitudes[c][((i * nv + a) * nr + j) * vr + b];
                // This is the exact antisymmetric part of the second-
                // quantized T2 operator, not an energy/denominator change.
                for i in 0..no {
                    for a in 0..nv {
                        for j in 0..nr {
                            for b in 0..vr {
                                result[c][((i * nv + a) * nr + j) * vr + b] =
                                    (at(i, a, j, b) - at(j, a, i, b) - at(i, b, j, a)
                                        + at(j, b, i, a))
                                        * 0.25;
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }
    fn trace(&self, k: &[ComplexMatrix; 2]) -> Complex {
        (0..2)
            .flat_map(|s| (0..self.occupied[s]).map(move |i| element(&k[s], i, i)))
            .fold(Complex::ZERO, |a, b| a + b)
    }
    fn action(&self, k: &[ComplexMatrix; 2], amplitudes: &[Complex], c: usize) -> Vec<Complex> {
        if amplitudes.iter().all(|z| *z == Complex::ZERO) {
            return vec![Complex::ZERO; amplitudes.len()];
        }
        let (s, t) = [(0, 0), (0, 1), (1, 1)][c];
        let mut result = complex_pair_action(self.shape(c), [&k[s], &k[t]], amplitudes);
        let trace = self.trace(k);
        for (v, t) in result.iter_mut().zip(amplitudes) {
            *v = trace * *t - *v;
        }
        result
    }
    fn singles(&self, k: &[ComplexMatrix; 2], c: usize) -> [Vec<Complex>; 2] {
        let mut result = std::array::from_fn(|s| {
            vec![Complex::ZERO; self.occupied[s] * (self.dimensions[s] - self.occupied[s])]
        });
        let (s, t) = [(0, 0), (0, 1), (1, 1)][c];
        let [no, nv, nr, vr] = self.shape(c);
        for i in 0..no {
            for a in 0..nv {
                for j in 0..nr {
                    for b in 0..vr {
                        let v = self.amplitudes[c][((i * nv + a) * nr + j) * vr + b];
                        result[s][a * no + i] += element(&k[t], j, nr + b) * v;
                        if s != t {
                            result[t][b * nr + j] += element(&k[s], i, no + a) * v;
                        }
                    }
                }
            }
        }
        result
    }

    /// Real nuclear / imaginary magnetic zero-field tangents. The exact
    /// identity `Im <G_R T|G_B T> = -Im <T|[G_R,G_B]|T>/2` eliminates
    /// triple states. The commutator is one-body and cannot connect distinct
    /// pair-spin sectors or a reference determinant to a double. Linear
    /// cancellation is also evaluated explicitly on the single projections.
    pub fn overlap(
        &self,
        nuclear: &PairStateTangent<'_>,
        magnetic: &PairStateTangent<'_>,
        weights: Pt2Coefficients,
        tolerance: f64,
    ) -> Result<PairAxialOverlap> {
        weights.validate()?;
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(DftError::InvalidInput(
                "pair-state acceptance tolerance must be positive and finite".into(),
            ));
        }
        let mut defect = 0.0_f64;
        for (tangent, is_magnetic) in [(nuclear, false), (magnetic, true)] {
            for (s, matrix) in tangent.connection.iter().enumerate() {
                let n = self.dimensions[s];
                if matrix.re.rows != n
                    || matrix.re.cols != n
                    || matrix.im.rows != n
                    || matrix.im.cols != n
                {
                    return Err(DftError::InvalidInput(
                        "pair-state orbital connection shape does not match its spin space".into(),
                    ));
                }
                for i in 0..n {
                    for j in 0..n {
                        let z = element(matrix, i, j);
                        if !z.is_finite() {
                            return Err(DftError::InvalidInput(
                                "pair-state connections must be finite".into(),
                            ));
                        }
                        defect = defect
                            .max((matrix.re[(i, j)] + matrix.re[(j, i)]).abs())
                            .max((matrix.im[(i, j)] - matrix.im[(j, i)]).abs())
                            .max(if is_magnetic { z.re.abs() } else { z.im.abs() });
                    }
                }
            }
            for z in tangent.amplitudes.iter().flatten() {
                defect = defect.max(if is_magnetic { z.re.abs() } else { z.im.abs() });
            }
        }
        let ar = self.canonical(nuclear.amplitudes)?;
        let ab = self.canonical(magnetic.amplitudes)?;
        if defect > tolerance {
            return Err(DftError::ResponseNotConverged {
                iterations: 0,
                residual: defect,
            });
        }
        let commutator: [ComplexMatrix; 2] = std::array::from_fn(|s| {
            if self.dimensions[s] == 0 {
                return ComplexMatrix::zeros(0, 0);
            }
            let mut value = nuclear.connection[s].matmul_seq(&magnetic.connection[s]);
            let reverse = magnetic.connection[s].matmul_seq(&nuclear.connection[s]);
            for (a, b) in value
                .re
                .as_mut_slice()
                .iter_mut()
                .zip(reverse.re.as_slice())
            {
                *a -= b;
            }
            for (a, b) in value
                .im
                .as_mut_slice()
                .iter_mut()
                .zip(reverse.im.as_slice())
            {
                *a -= b;
            }
            value
        });
        let parent = -0.5 * self.trace(&commutator).im;
        let mut channel_bilinears = [0.0; 3];
        let channel_norms =
            std::array::from_fn(|c| inner(&self.amplitudes[c], &self.amplitudes[c], c).re);
        let mut norm_changes = [0.0; 3];
        let mut connections = [0.0; 3];
        let mut cancellation_defect = 0.0_f64;
        for c in 0..3 {
            let t = &self.amplitudes[c];
            let bt = self.action(magnetic.connection, t, c);
            channel_bilinears[c] = -0.5 * inner(t, &self.action(&commutator, t, c), c).im
                - inner(t, &self.action(nuclear.connection, &ab[c], c), c).im
                + inner(&ar[c], &bt, c).im
                + inner(&ar[c], &ab[c], c).im;
            norm_changes[c] = 2.0 * inner(t, &ar[c], c).re;
            connections[c] = (inner(t, &bt, c) + inner(t, &ab[c], c)).im;
            let sr = self.singles(nuclear.connection, c);
            let sb = self.singles(magnetic.connection, c);
            let mut linear = Complex::ZERO;
            for s in 0..2 {
                let no = self.occupied[s];
                for i in 0..no {
                    for a in no..self.dimensions[s] {
                        linear += element(&nuclear.connection[s], a, i).conj()
                            * sb[s][(a - no) * no + i]
                            + sr[s][(a - no) * no + i].conj()
                                * element(&magnetic.connection[s], a, i);
                    }
                }
                for i in 0..self.dimensions[s] {
                    for j in 0..self.dimensions[s] {
                        cancellation_defect = cancellation_defect
                            .max((commutator[s].re[(i, j)] + commutator[s].re[(j, i)]).abs())
                            .max((commutator[s].im[(i, j)] - commutator[s].im[(j, i)]).abs());
                    }
                }
            }
            cancellation_defect = cancellation_defect.max(linear.im.abs());
        }
        if cancellation_defect > tolerance {
            return Err(DftError::ResponseNotConverged {
                iterations: 0,
                residual: cancellation_defect,
            });
        }
        let w = [weights.same_spin, weights.opposite_spin, weights.same_spin];
        let norm = 1.0 + (0..3).map(|c| w[c] * channel_norms[c]).sum::<f64>();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(DftError::InvalidInput(
                "the signed spin-weighted pair-state norm must be positive".into(),
            ));
        }
        let norm_derivative = (0..3).map(|c| w[c] * norm_changes[c]).sum::<f64>();
        let magnetic_connection =
            self.trace(magnetic.connection).im + (0..3).map(|c| w[c] * connections[c]).sum::<f64>();
        let value = (parent + (0..3).map(|c| w[c] * channel_bilinears[c]).sum::<f64>()) / norm
            - norm_derivative * magnetic_connection / (2.0 * norm * norm);
        Ok(PairAxialOverlap {
            value,
            parent,
            channel_bilinears,
            channel_norms,
            norm,
            norm_derivative,
            magnetic_connection,
            connection_defect: defect,
            cancellation_defect,
        })
    }
}
