// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Second-order polarization-propagator ingredients on a collinear parent.
//! The residual interaction is the normal-ordered full Coulomb operator;
//! the supplied DFT parent retains its own zeroth-order response kernel.
//! No functional names, finite differences or within-subspace gap divisions
//! occur in these contractions. Signed spin coefficients enter bilinears.

use super::pt2::Pt2Problem;
use super::pt2_density::{transform, unrelaxed_mo_density};
use super::pt2_response::real_pair_action;
use crate::integrals::eri::EriTensor;
use crate::linalg::Matrix;
use crate::scf::ScfResult;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};
use rayon::prelude::*;

mod path;
pub use path::{collinear_transition_path, CollinearTransitionPath};

#[cfg(test)]
#[path = "pt2_transition_tests.rs"]
mod tests;

/// Reusable full-Coulomb PT2 transition algebra in a semicanonical MO frame.
/// Storage O(N^4), orbital Hessian actions O(N^5). All spins are explicit;
/// spin adaptation is performed by the parent response manifold.
pub struct Pt2Transition {
    energies: [Vec<f64>; 2],
    occupied: [usize; 2],
    integrals: [Vec<f64>; 3],
    amplitudes: [Vec<f64>; 3],
    coefficients: Pt2Coefficients,
    density: [Matrix; 2],
    property_density: [Matrix; 2],
}

impl Pt2Transition {
    pub(crate) fn workspace_bytes(nao: usize) -> Result<usize> {
        nao.checked_pow(4)
            .and_then(|v| v.checked_mul(8 * 24))
            .ok_or_else(|| DftError::InvalidInput("PT2 transition workspace overflow".into()))
    }
    /// Build from the same real reference, AO integrals and amplitudes used
    /// by the stationary energy. The workspace budget is checked first.
    pub fn from_reference(
        scf: &ScfResult,
        eri: &EriTensor,
        problem: &Pt2Problem<f64>,
        amplitudes: &[Vec<f64>; 3],
        coefficients: Pt2Coefficients,
        memory_mb: usize,
    ) -> Result<Self> {
        let n = eri.nao;
        let bytes = Self::workspace_bytes(n)?;
        let required_mb = bytes.div_ceil(1024 * 1024);
        if required_mb > memory_mb {
            return Err(DftError::ResourceLimit {
                operation: "PT2 transition tensors",
                required_mb,
                limit_mb: memory_mb,
            });
        }
        if [scf.n_alpha, scf.n_beta] != problem.occupied
            || scf.mo_energy != problem.energies
            || scf
                .mo_coeff
                .iter()
                .enumerate()
                .any(|(s, c)| c.rows != n || c.cols != problem.energies[s].len())
        {
            return Err(DftError::InvalidInput(
                "PT2 transition orbitals and energy spaces disagree".into(),
            ));
        }
        let integrals = [(0, 0), (0, 1), (1, 1)]
            .into_par_iter()
            .map(|(s, t)| {
                transform(
                    eri,
                    [
                        &scf.mo_coeff[s],
                        &scf.mo_coeff[s],
                        &scf.mo_coeff[t],
                        &scf.mo_coeff[t],
                    ],
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Self::from_mo(problem, integrals, amplitudes.clone(), coefficients)
    }

    fn from_mo(
        problem: &Pt2Problem<f64>,
        integrals: [Vec<f64>; 3],
        amplitudes: [Vec<f64>; 3],
        coefficients: Pt2Coefficients,
    ) -> Result<Self> {
        let density = unrelaxed_mo_density(problem, &amplitudes, coefficients)?;
        let ns = [problem.energies[0].len(), problem.energies[1].len()];
        if integrals
            .iter()
            .zip([(0, 0), (0, 1), (1, 1)])
            .any(|(g, (s, t))| {
                g.len() != ns[s] * ns[s] * ns[t] * ns[t] || g.iter().any(|v| !v.is_finite())
            })
        {
            return Err(DftError::InvalidInput(
                "PT2 transition integral dimensions disagree".into(),
            ));
        }
        let mut out = Self {
            energies: problem.energies.clone(),
            occupied: problem.occupied,
            integrals,
            amplitudes,
            coefficients,
            density: density.clone(),
            property_density: density,
        };
        let source = out.coupling(&out.amplitudes);
        for s in 0..2 {
            let o = out.occupied[s];
            for a in o..ns[s] {
                for i in 0..o {
                    let gap = out.energies[s][a] - out.energies[s][i];
                    if !(gap > 0.0) {
                        return Err(DftError::InvalidInput(
                            "PT2 transition ground singles need positive occupied-virtual gaps"
                                .into(),
                        ));
                    }
                    let value = -source[s][(a - o, i)] / gap;
                    out.property_density[s][(a, i)] = value;
                    out.property_density[s][(i, a)] = value;
                }
            }
        }
        Ok(out)
    }

    fn dimensions(&self) -> [usize; 2] {
        [self.energies[0].len(), self.energies[1].len()]
    }
    fn weight(&self, s: usize, t: usize) -> f64 {
        if s == t {
            self.coefficients.same_spin
        } else {
            self.coefficients.opposite_spin
        }
    }
    fn g(
        &self,
        blocks: &[Vec<f64>; 3],
        s: usize,
        t: usize,
        p: usize,
        q: usize,
        r: usize,
        u: usize,
    ) -> f64 {
        let n = self.dimensions();
        if s == 1 && t == 0 {
            blocks[1][((r * n[0] + u) * n[1] + p) * n[1] + q]
        } else {
            blocks[if s == t { 2 * s } else { 1 }][((p * n[s] + q) * n[t] + r) * n[t] + u]
        }
    }
    fn t(
        &self,
        amp: &[Vec<f64>; 3],
        s: usize,
        t: usize,
        i: usize,
        a: usize,
        j: usize,
        b: usize,
    ) -> f64 {
        let v = [
            self.energies[0].len() - self.occupied[0],
            self.energies[1].len() - self.occupied[1],
        ];
        if s == 1 && t == 0 {
            amp[1][((j * v[0] + b) * self.occupied[1] + i) * v[1] + a]
        } else {
            amp[if s == t { 2 * s } else { 1 }][((i * v[s] + a) * self.occupied[t] + j) * v[t] + b]
        }
    }
    fn contraction(&self, blocks: &[Vec<f64>; 3], amp: &[Vec<f64>; 3]) -> [Matrix; 2] {
        let n = self.dimensions();
        std::array::from_fn(|s| {
            let mut l = Matrix::zeros(n[s], n[s]);
            for t in 0..2 {
                for i in 0..self.occupied[s] {
                    for a in self.occupied[s]..n[s] {
                        for j in 0..self.occupied[t] {
                            for b in self.occupied[t]..n[t] {
                                let value = self.weight(s, t)
                                    * self.t(
                                        amp,
                                        s,
                                        t,
                                        i,
                                        a - self.occupied[s],
                                        j,
                                        b - self.occupied[t],
                                    );
                                for m in 0..n[s] {
                                    l[(m, i)] += self.g(blocks, s, t, a, m, b, j) * value;
                                    l[(m, a)] += self.g(blocks, s, t, i, m, j, b) * value;
                                }
                            }
                        }
                    }
                }
            }
            l
        })
    }
    /// Singles/doubles coupling, contracted with full same-spin
    /// antisymmetric and opposite-spin direct amplitudes [i,a,j,b].
    pub fn coupling(&self, amplitudes: &[Vec<f64>; 3]) -> [Matrix; 2] {
        let l = self.contraction(&self.integrals, amplitudes);
        std::array::from_fn(|s| {
            let o = self.occupied[s];
            let n = self.energies[s].len();
            let mut out = Matrix::zeros(n - o, o);
            for a in o..n {
                for i in 0..o {
                    out[(a - o, i)] = l[s][(a, i)] - l[s][(i, a)];
                }
            }
            out
        })
    }
    fn rotation(&self, x: &[Matrix; 2], odd: bool) -> [Matrix; 2] {
        std::array::from_fn(|s| {
            let n = self.energies[s].len();
            let o = self.occupied[s];
            let mut k = Matrix::zeros(n, n);
            for a in o..n {
                for i in 0..o {
                    k[(a, i)] = x[s][(a - o, i)];
                    k[(i, a)] = if odd {
                        x[s][(a - o, i)]
                    } else {
                        -x[s][(a - o, i)]
                    };
                }
            }
            k
        })
    }
    fn integral_rotation(&self, k: &[Matrix; 2], odd: bool) -> [Vec<f64>; 3] {
        let n = self.dimensions();
        let bra = if odd { -1.0 } else { 1.0 };
        let blocks: Vec<_> = [(0, 0), (0, 1), (1, 1)]
            .into_par_iter()
            .map(|(s, t)| {
                let mut out = vec![0.0; n[s] * n[s] * n[t] * n[t]];
                for p in 0..n[s] {
                    for q in 0..n[s] {
                        for r in 0..n[t] {
                            for u in 0..n[t] {
                                let mut value = 0.0;
                                for m in 0..n[s] {
                                    value += bra
                                        * k[s][(m, p)]
                                        * self.g(&self.integrals, s, t, m, q, r, u)
                                        + k[s][(m, q)] * self.g(&self.integrals, s, t, p, m, r, u);
                                }
                                for m in 0..n[t] {
                                    value += bra
                                        * k[t][(m, r)]
                                        * self.g(&self.integrals, s, t, p, q, m, u)
                                        + k[t][(m, u)] * self.g(&self.integrals, s, t, p, q, r, m);
                                }
                                out[((p * n[s] + q) * n[t] + r) * n[t] + u] = value;
                            }
                        }
                    }
                }
                out
            })
            .collect();
        blocks.try_into().unwrap()
    }
    fn parity_kernel(&self, x: &[Matrix; 2], odd: bool) -> [Matrix; 2] {
        let k = self.rotation(x, odd);
        let dg = self.integral_rotation(&k, odd);
        let l = self.contraction(&dg, &self.amplitudes);
        std::array::from_fn(|s| {
            let n = self.energies[s].len();
            let o = self.occupied[s];
            let mut df = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    df[(i, j)] = (self.energies[s][i] - self.energies[s][j]) * k[s][(i, j)];
                }
            }
            let comm = commutator(&df, &self.density[s]);
            let mut out = Matrix::zeros(n - o, o);
            for a in o..n {
                for i in 0..o {
                    out[(a - o, i)] = comm[(a, i)]
                        + if odd {
                            -l[s][(a, i)] - l[s][(i, a)]
                        } else {
                            l[s][(a, i)] - l[s][(i, a)]
                        };
                }
            }
            out
        })
    }
    /// Second-order singles A and B actions, from the time-even/odd
    /// Hylleraas orbital kernels, with fixed first-order amplitudes.
    pub fn kernels(&self, x: &[Matrix; 2]) -> ([Matrix; 2], [Matrix; 2]) {
        let even = self.parity_kernel(x, false);
        let odd = self.parity_kernel(x, true);
        let a = std::array::from_fn(|s| combine(&even[s], &odd[s], 0.5, 0.5));
        let b = std::array::from_fn(|s| combine(&even[s], &odd[s], 0.5, -0.5));
        (a, b)
    }
    /// Correlation part of the singles commutator metric.
    pub fn metric(&self, x: &[Matrix; 2]) -> [Matrix; 2] {
        std::array::from_fn(|s| {
            let o = self.occupied[s];
            let n = self.energies[s].len();
            let mut out = Matrix::zeros(n - o, o);
            for a in o..n {
                for i in 0..o {
                    for j in 0..o {
                        out[(a - o, i)] += x[s][(a - o, j)] * self.density[s][(j, i)];
                    }
                    for b in o..n {
                        out[(a - o, i)] -= self.density[s][(a, b)] * x[s][(b - o, i)];
                    }
                }
            }
            out
        })
    }
    /// Transpose singles/doubles coupling in the unscaled pair basis.
    pub fn coupling_transpose(&self, x: &[Matrix; 2]) -> [Vec<f64>; 3] {
        let dg = self.integral_rotation(&self.rotation(x, false), false);
        self.amplitudes_from_integrals(&dg)
    }
    fn amplitudes_from_integrals(&self, g: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        std::array::from_fn(|c| {
            let (s, t) = [(0, 0), (0, 1), (1, 1)][c];
            let o = self.occupied;
            let n = self.dimensions();
            let v = [n[0] - o[0], n[1] - o[1]];
            let mut out = vec![0.0; o[s] * v[s] * o[t] * v[t]];
            for i in 0..o[s] {
                for a in o[s]..n[s] {
                    for j in 0..o[t] {
                        for b in o[t]..n[t] {
                            if s == t && (i == j || a == b) {
                                continue;
                            }
                            out[((i * v[s] + a - o[s]) * o[t] + j) * v[t] + b - o[t]] = self
                                .g(g, s, t, i, a, j, b)
                                - if s == t {
                                    self.g(g, s, t, i, b, j, a)
                                } else {
                                    0.0
                                };
                        }
                    }
                }
            }
            out
        })
    }
    /// Doubles resolvent without denominator modification. Returns the
    /// unscaled amplitudes, their signed metric norm and smallest |gap|.
    pub fn doubles(&self, omega: f64, x: &[Matrix; 2]) -> Result<([Vec<f64>; 3], f64, f64)> {
        let mut out = self.coupling_transpose(x);
        let mut norm = 0.0;
        let mut minimum = f64::INFINITY;
        for (c, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            if self.weight(s, t) == 0.0 {
                out[c].fill(0.0);
                continue;
            }
            let o = self.occupied;
            let n = self.dimensions();
            let v = [n[0] - o[0], n[1] - o[1]];
            for i in 0..o[s] {
                for a in o[s]..n[s] {
                    for j in 0..o[t] {
                        for b in o[t]..n[t] {
                            let k = ((i * v[s] + a - o[s]) * o[t] + j) * v[t] + b - o[t];
                            // Pauli-forbidden substitutions are absent
                            // states, not poles. This is an exact exclusion,
                            // without a numerical denominator floor.
                            if s == t && (i == j || a == b) {
                                out[c][k] = 0.0;
                                continue;
                            }
                            let denominator = omega - self.energies[s][a] + self.energies[s][i]
                                - self.energies[t][b]
                                + self.energies[t][j];
                            minimum = minimum.min(denominator.abs());
                            if denominator == 0.0 {
                                if out[c][k] != 0.0 {
                                    return Err(DftError::InvalidInput("PT2 transition root is resonant with a coupled double excitation; an enlarged degenerate propagator space is required".into()));
                                }
                            } else {
                                out[c][k] /= denominator;
                            }
                            if !out[c][k].is_finite() {
                                return Err(DftError::InvalidInput(
                                    "nonfinite PT2 transition resolvent".into(),
                                ));
                            }
                            norm += self.weight(s, t)
                                * if s == t { 0.25 } else { 1.0 }
                                * out[c][k]
                                * out[c][k];
                        }
                    }
                }
            }
        }
        Ok((out, norm, minimum))
    }
    /// Second-order singles property correction, including MP2 ground
    /// singles, plus the first-order doubles observable gradient. `operator`
    /// may be real Hermitian or the real part of an anti-Hermitian electric,
    /// momentum or angular-momentum operator. The same algebra treats all.
    pub fn property(&self, operator: &[Matrix; 2]) -> ([Matrix; 2], [Vec<f64>; 3]) {
        let singles = std::array::from_fn(|s| {
            let comm = commutator(&operator[s], &self.property_density[s]);
            let o = self.occupied[s];
            let n = self.energies[s].len();
            let mut out = Matrix::zeros(n - o, o);
            for a in o..n {
                for i in 0..o {
                    out[(a - o, i)] = comm[(a, i)];
                }
            }
            out
        });
        let doubles = std::array::from_fn(|c| {
            let (s, t) = [(0, 0), (0, 1), (1, 1)][c];
            let n = self.dimensions();
            let o = self.occupied;
            real_pair_action(
                [o[s], n[s] - o[s], o[t], n[t] - o[t]],
                [&operator[s], &operator[t]],
                &self.amplitudes[c],
            )
            .into_iter()
            .map(|v| -v)
            .collect()
        });
        (singles, doubles)
    }
    /// Signed spin-weighted inner product in the redundant pair tensors.
    pub fn pair_dot(&self, a: &[Vec<f64>; 3], b: &[Vec<f64>; 3]) -> f64 {
        a.iter()
            .zip(b)
            .enumerate()
            .map(|(c, (a, b))| {
                let weight = if c == 1 {
                    self.coefficients.opposite_spin
                } else {
                    0.25 * self.coefficients.same_spin
                };
                weight * a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>()
            })
            .sum()
    }
}

fn commutator(a: &Matrix, b: &Matrix) -> Matrix {
    combine(&a.matmul(b), &b.matmul(a), 1.0, -1.0)
}
fn combine(a: &Matrix, b: &Matrix, sa: f64, sb: f64) -> Matrix {
    let mut out = a.clone();
    for (x, y) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x = sa * *x + sb * y;
    }
    out
}
