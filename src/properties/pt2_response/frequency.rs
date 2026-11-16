// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Floquet PT2 amplitude action in a prescribed moving orbital frame.

use super::{index, pair_action};
use crate::integrals::london::ComplexMatrix;
use crate::properties::pt2::Pt2Problem;
use crate::response::complex::Complex;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};

#[cfg(test)]
mod tests;

const SPINS: [(usize, usize); 3] = [(0, 0), (0, 1), (1, 1)];

/// One coefficient of the moving-frame pair action. The two coupling
/// tensors are raw Coulomb ovov blocks in alpha-alpha, alpha-beta and
/// beta-beta order. The left coefficient is the analytic dual of the
/// right operator at the opposite harmonic. At a complex frequency it
/// must not be constructed by conjugating the same-frequency coefficient.
pub struct Pt2FrequencyVertex {
    pub coupling_left: [Vec<Complex>; 3],
    pub coupling_right: [Vec<Complex>; 3],
    /// Covariant MO generator F - U^{-1} i dU/dt. Only the occupied and
    /// virtual diagonal blocks enter the pair action.
    pub orbital_generator: [ComplexMatrix; 2],
}

/// Negative mixed quasi-energy derivatives and the unmodified pair equations.
pub struct Pt2FrequencyResponse {
    /// Unscaled same-spin alpha, opposite spin and same-spin beta values.
    pub channels: [Complex; 3],
    pub correlation: Complex,
    /// Per harmonic (+z, -z), then per spin channel.
    pub right_absolute_residuals: [[f64; 3]; 2],
    pub left_absolute_residuals: [[f64; 3]; 2],
    pub right_relative_residuals: [[f64; 3]; 2],
    pub left_relative_residuals: [[f64; 3]; 2],
    /// Minimum |Delta +/- z| over Pauli-allowed channels with a source.
    pub minimum_driven_denominator: Option<f64>,
}

struct PairBlock {
    shape: [usize; 4],
    denominators: Vec<f64>,
    stationary: Vec<Complex>,
}

/// Reusable semicanonical base for electric or other one-body orbital paths.
pub struct Pt2FrequencyKernel {
    blocks: [PairBlock; 3],
    dimensions: [usize; 2],
    coefficients: Pt2Coefficients,
}

impl Pt2FrequencyKernel {
    pub fn new(problem: &Pt2Problem<f64>, coefficients: Pt2Coefficients) -> Result<Self> {
        coefficients.validate()?;
        let dimensions = problem.energies.each_ref().map(Vec::len);
        let mut blocks = std::array::from_fn(|c| {
            let (s, t) = SPINS[c];
            PairBlock {
                shape: [
                    problem.occupied[s],
                    dimensions[s].saturating_sub(problem.occupied[s]),
                    problem.occupied[t],
                    dimensions[t].saturating_sub(problem.occupied[t]),
                ],
                denominators: vec![0.0; problem.ovov[c].len()],
                stationary: vec![Complex::ZERO; problem.ovov[c].len()],
            }
        });
        problem.visit(|channel, p, coupling, denominator, _| {
            blocks[channel].denominators[p] = denominator;
            if blocks[channel].allowed(p, channel != 1) {
                blocks[channel].stationary[p] = Complex::from(coupling / denominator);
            }
        })?;
        if blocks
            .iter()
            .any(|b| b.stationary.iter().any(|a| !a.is_finite()))
        {
            return Err(DftError::InvalidInput(
                "nonfinite stationary PT2 frequency-response amplitudes".into(),
            ));
        }
        Ok(Self {
            blocks,
            dimensions,
            coefficients,
        })
    }

    /// Apply two labelled harmonics +z and -z. The mixed vertex is their
    /// actual mixed derivative at zero total frequency.
    ///
    /// For t_R and t_L the action is
    /// t_L^T g_R + g_L^T t_R - t_L^T (A + i*d/dt) t_R, with the static
    /// occupied-minus-virtual denominator A. At +z the right resolvent
    /// is (A+z)^{-1}. Integrating the time derivative by parts gives
    /// (A-z)^{-T} on the left. Stationarity eliminates the unknown mixed
    /// amplitudes from the zero-frequency quasi-energy coefficient.
    pub fn quadratic(
        &self,
        z: Complex,
        harmonic: [&Pt2FrequencyVertex; 2],
        mixed: &Pt2FrequencyVertex,
    ) -> Result<Pt2FrequencyResponse> {
        if !z.is_finite() {
            return Err(DftError::InvalidInput(
                "PT2 frequency must be finite".into(),
            ));
        }
        for vertex in [harmonic[0], harmonic[1], mixed] {
            self.validate_vertex(vertex)?;
        }
        let left_generators = harmonic.map(|vertex| {
            vertex.orbital_generator.each_ref().map(|f| ComplexMatrix {
                re: f.re.transpose(),
                im: f.im.transpose(),
            })
        });
        let mut report = Pt2FrequencyResponse {
            channels: [Complex::ZERO; 3],
            correlation: Complex::ZERO,
            right_absolute_residuals: [[0.0; 3]; 2],
            left_absolute_residuals: [[0.0; 3]; 2],
            right_relative_residuals: [[0.0; 3]; 2],
            left_relative_residuals: [[0.0; 3]; 2],
            minimum_driven_denominator: None,
        };
        for (c, (s, t)) in SPINS.into_iter().enumerate() {
            let block = &self.blocks[c];
            let q_right = harmonic.map(|vertex| {
                block.source(
                    &vertex.coupling_right[c],
                    [&vertex.orbital_generator[s], &vertex.orbital_generator[t]],
                    s == t,
                )
            });
            let q_left = std::array::from_fn::<_, 2, _>(|h| {
                let f = &left_generators[h];
                block.source(&harmonic[h].coupling_left[c], [&f[s], &f[t]], s == t)
            });
            let mixed_right = block.source(
                &mixed.coupling_right[c],
                [&mixed.orbital_generator[s], &mixed.orbital_generator[t]],
                s == t,
            );
            let mixed_left = block.coupling(&mixed.coupling_left[c], s == t);
            let mut coefficient = bilinear(&mixed_left, &block.stationary)
                + bilinear(&block.stationary, &mixed_right);
            for (h, shift) in [z, -z].into_iter().enumerate() {
                let right = block.solve(shift, &q_right[h])?;
                let left = block.solve(-shift, &q_left[h])?;
                coefficient += bilinear(&q_left[1 - h], &right.amplitudes);
                report.right_absolute_residuals[h][c] = right.absolute;
                report.left_absolute_residuals[h][c] = left.absolute;
                report.right_relative_residuals[h][c] = right.relative;
                report.left_relative_residuals[h][c] = left.relative;
                for minimum in [right.minimum, left.minimum].into_iter().flatten() {
                    report.minimum_driven_denominator = Some(
                        report
                            .minimum_driven_denominator
                            .map_or(minimum, |d| d.min(minimum)),
                    );
                }
            }
            // Redundant same-spin amplitudes represent each unique pair four times.
            report.channels[c] = -coefficient * if s == t { 0.25 } else { 1.0 };
        }
        report.correlation = (report.channels[0] + report.channels[2])
            * self.coefficients.same_spin
            + report.channels[1] * self.coefficients.opposite_spin;
        if !report.correlation.is_finite() || report.channels.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "nonfinite PT2 mixed frequency quasi-energy".into(),
            ));
        }
        Ok(report)
    }

    fn validate_vertex(&self, vertex: &Pt2FrequencyVertex) -> Result<()> {
        let couplings_valid = vertex
            .coupling_left
            .iter()
            .zip(&vertex.coupling_right)
            .zip(&self.blocks)
            .all(|((left, right), block)| {
                left.len() == block.stationary.len()
                    && right.len() == left.len()
                    && left.iter().chain(right).all(|v| v.is_finite())
            });
        let generator_valid = vertex
            .orbital_generator
            .iter()
            .zip(self.dimensions)
            .all(|(f, n)| {
                f.re.rows == n
                    && f.re.cols == n
                    && f.im.rows == n
                    && f.im.cols == n
                    && f.re
                        .as_slice()
                        .iter()
                        .chain(f.im.as_slice())
                        .all(|v| v.is_finite())
            });
        if !couplings_valid || !generator_valid {
            return Err(DftError::InvalidInput(
                "invalid PT2 frequency-response pair or orbital vertex".into(),
            ));
        }
        Ok(())
    }
}

fn bilinear(left: &[Complex], right: &[Complex]) -> Complex {
    // Analytic continuation uses a transpose pairing, not a Hermitian norm.
    left.iter()
        .zip(right)
        .fold(Complex::ZERO, |sum, (l, r)| sum + *l * *r)
}

struct ShiftedSolution {
    amplitudes: Vec<Complex>,
    absolute: f64,
    relative: f64,
    minimum: Option<f64>,
}

impl PairBlock {
    fn allowed(&self, p: usize, same_spin: bool) -> bool {
        if !same_spin {
            return true;
        }
        let [_, nv, nr, vr] = self.shape;
        let b = p % vr;
        let j = (p / vr) % nr;
        let a = (p / (vr * nr)) % nv;
        let i = p / (vr * nr * nv);
        i != j && a != b
    }

    fn coupling(&self, raw: &[Complex], same_spin: bool) -> Vec<Complex> {
        let mut out = raw.to_vec();
        if same_spin {
            let [no, nv, _, _] = self.shape;
            for i in 0..no {
                for a in 0..nv {
                    for j in 0..no {
                        for b in 0..nv {
                            let p = index(self.shape, i, a, j, b);
                            out[p] = if i == j || a == b {
                                Complex::ZERO
                            } else {
                                raw[p] - raw[index(self.shape, i, b, j, a)]
                            };
                        }
                    }
                }
            }
        }
        out
    }

    fn source(
        &self,
        raw: &[Complex],
        generator: [&ComplexMatrix; 2],
        same_spin: bool,
    ) -> Vec<Complex> {
        let mut result = self.coupling(raw, same_spin);
        for (p, (q, image)) in result
            .iter_mut()
            .zip(pair_action(self.shape, generator, &self.stationary))
            .enumerate()
        {
            // Exact Pauli projection also removes floating-point cancellation
            // residue in the generator action on the redundant tensor.
            *q = if self.allowed(p, same_spin) {
                *q - image
            } else {
                Complex::ZERO
            };
        }
        result
    }

    fn solve(&self, shift: Complex, source: &[Complex]) -> Result<ShiftedSolution> {
        let mut amplitudes = Vec::with_capacity(source.len());
        let (mut absolute, mut residual_norm, mut source_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut minimum = None::<f64>;
        for (&a, &q) in self.denominators.iter().zip(source) {
            let denominator = Complex::from(a) + shift;
            let amplitude = if q == Complex::ZERO {
                Complex::ZERO
            } else {
                if denominator == Complex::ZERO {
                    return Err(DftError::Unsupported {
                        feature: "undamped PT2 frequency at a driven pair pole".into(),
                        missing: "a damped frequency or enlarged degenerate response space".into(),
                    });
                }
                let d = denominator.abs();
                minimum = Some(minimum.map_or(d, |value| value.min(d)));
                q / denominator
            };
            let residual = (denominator * amplitude - q).abs();
            if !amplitude.is_finite() || !residual.is_finite() {
                return Err(DftError::InvalidInput(
                    "nonfinite PT2 shifted amplitude equation".into(),
                ));
            }
            absolute = absolute.max(residual);
            residual_norm = residual_norm.hypot(residual);
            source_norm = source_norm.hypot(q.abs());
            amplitudes.push(amplitude);
        }
        let relative = if source_norm > 0.0 {
            residual_norm / source_norm
        } else {
            0.0
        };
        Ok(ShiftedSolution {
            amplitudes,
            absolute,
            relative,
            minimum,
        })
    }
}
