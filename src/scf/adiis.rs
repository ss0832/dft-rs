// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! ADIIS: augmented direct inversion in the iterative subspace.
//!
//! Hu & Yang, J. Chem. Phys. **132**, 054109 (2010). CDIIS is a residual
//! extrapolation: it drives `FPS − SPF` to zero, which is the right thing
//! NEAR a solution and an unreliable one far from it, where the
//! extrapolated Fock can be worse than any of its inputs. ADIIS instead
//! minimizes a variational estimate of the energy over the convex hull of
//! the stored densities, so every coefficient is non-negative and the
//! result is a genuine interpolation rather than an extrapolation.
//!
//! With `D_n`, `F_n` the newest pair, the objective is
//!
//! ```text
//! f(c) = 2 sum_i c_i <D_i - D_n | F_n>
//!      +   sum_ij c_i c_j <D_i - D_n | F_j - F_n>
//! ```
//!
//! minimized subject to `c_i >= 0` and `sum_i c_i = 1`. The constraint is
//! handled by the substitution `c_i = t_i^2 / sum_j t_j^2`, which enforces
//! both conditions exactly and turns the problem into an unconstrained one
//! in `t` — no active-set bookkeeping, and no way to leave the simplex by
//! accident.
//!
//! The usual driving pattern, which is what this module is for: ADIIS while
//! the error is large, CDIIS once it is small, and a blend in between.

use crate::linalg::Matrix;

/// Stored (density, Fock) pairs and the ADIIS objective built from them.
pub struct Adiis {
    max_vecs: usize,
    /// One entry per stored iterate: the per-spin densities.
    densities: Vec<Vec<Matrix>>,
    /// One entry per stored iterate: the per-spin Fock matrices.
    focks: Vec<Vec<Matrix>>,
}

impl Adiis {
    pub fn new(max_vecs: usize) -> Self {
        Self {
            max_vecs: max_vecs.max(2),
            densities: Vec::new(),
            focks: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.densities.clear();
        self.focks.clear();
    }

    pub fn len(&self) -> usize {
        self.focks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.focks.is_empty()
    }

    /// Record one iterate. Densities and Focks are per spin and must be in
    /// the same order every call.
    pub fn push(&mut self, densities: &[Matrix], focks: &[Matrix]) {
        self.densities.push(densities.to_vec());
        self.focks.push(focks.to_vec());
        if self.focks.len() > self.max_vecs {
            self.densities.remove(0);
            self.focks.remove(0);
        }
    }

    /// Trace inner product summed over spin channels.
    fn dot(a: &[Matrix], b: &[Matrix]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x.frobenius_dot(y)).sum()
    }

    /// Minimize the ADIIS objective and return the interpolated Fock, or
    /// `None` when there is nothing to interpolate.
    pub fn extrapolate(&self) -> Option<Vec<Matrix>> {
        let n = self.focks.len();
        if n < 2 {
            return None;
        }
        let last = n - 1;
        // g_i = <D_i - D_n | F_n>,  h_ij = <D_i - D_n | F_j - F_n>
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        let dn = &self.densities[last];
        let fnn = &self.focks[last];
        let diff_d: Vec<Vec<Matrix>> = (0..n)
            .map(|i| {
                self.densities[i]
                    .iter()
                    .zip(dn)
                    .map(|(a, b)| {
                        let mut m = a.clone();
                        for (x, y) in m.as_mut_slice().iter_mut().zip(b.as_slice()) {
                            *x -= *y;
                        }
                        m
                    })
                    .collect()
            })
            .collect();
        let diff_f: Vec<Vec<Matrix>> = (0..n)
            .map(|j| {
                self.focks[j]
                    .iter()
                    .zip(fnn)
                    .map(|(a, b)| {
                        let mut m = a.clone();
                        for (x, y) in m.as_mut_slice().iter_mut().zip(b.as_slice()) {
                            *x -= *y;
                        }
                        m
                    })
                    .collect()
            })
            .collect();
        for i in 0..n {
            g[i] = Self::dot(&diff_d[i], fnn);
            for j in 0..n {
                h[i * n + j] = Self::dot(&diff_d[i], &diff_f[j]);
            }
        }
        // Symmetrize: the objective only ever sees (h_ij + h_ji)/2.
        for i in 0..n {
            for j in (i + 1)..n {
                let m = 0.5 * (h[i * n + j] + h[j * n + i]);
                h[i * n + j] = m;
                h[j * n + i] = m;
            }
        }

        let c = minimize_simplex(&g, &h, n)?;
        // F = sum_i c_i F_i, per spin.
        let nspin = self.focks[0].len();
        let mut out: Vec<Matrix> = (0..nspin)
            .map(|sp| Matrix::zeros(self.focks[0][sp].rows, self.focks[0][sp].cols))
            .collect();
        for (i, ci) in c.iter().enumerate() {
            if *ci == 0.0 {
                continue;
            }
            for sp in 0..nspin {
                for (o, f) in out[sp]
                    .as_mut_slice()
                    .iter_mut()
                    .zip(self.focks[i][sp].as_slice())
                {
                    *o += ci * f;
                }
            }
        }
        Some(out)
    }
}

/// Minimize `f(c) = 2 g.c + c^T h c` on the simplex, through the
/// substitution `c_i = t_i^2 / sum_j t_j^2`.
///
/// The substitution is what keeps this simple: it enforces `c_i >= 0` and
/// `sum c_i = 1` identically, so gradient descent can never step outside
/// the feasible set and there is no active set to track. Backtracking on
/// the objective makes the descent unconditionally safe.
pub(crate) fn minimize_simplex(g: &[f64], h: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 || g.len() != n || h.len() != n.checked_mul(n)? {
        return None;
    }
    // Start from the newest iterate, which is the CDIIS-free fallback.
    let mut t = vec![0.35; n];
    t[n - 1] = 1.0;

    let coeffs = |t: &[f64]| -> Vec<f64> {
        let s: f64 = t.iter().map(|x| x * x).sum();
        if s <= 0.0 {
            let mut c = vec![0.0; n];
            c[n - 1] = 1.0;
            return c;
        }
        t.iter().map(|x| x * x / s).collect()
    };
    let value = |c: &[f64]| -> f64 {
        let mut v = 0.0;
        for i in 0..n {
            v += 2.0 * g[i] * c[i];
            for j in 0..n {
                v += c[i] * c[j] * h[i * n + j];
            }
        }
        v
    };

    let mut f_cur = value(&coeffs(&t));
    for _ in 0..200 {
        let c = coeffs(&t);
        let s: f64 = t.iter().map(|x| x * x).sum();
        if s <= 0.0 {
            break;
        }
        // df/dc_i, then chain through c_i = t_i^2 / s.
        let dfdc: Vec<f64> = (0..n)
            .map(|i| 2.0 * g[i] + 2.0 * (0..n).map(|j| h[i * n + j] * c[j]).sum::<f64>())
            .collect();
        let dot: f64 = dfdc.iter().zip(&c).map(|(a, b)| a * b).sum();
        let grad: Vec<f64> = (0..n).map(|i| 2.0 * t[i] / s * (dfdc[i] - dot)).collect();
        let gnorm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !gnorm.is_finite() {
            return None;
        }
        if gnorm < 1e-12 {
            break;
        }
        // Backtracking line search: never accept a worse objective.
        let mut step = 1.0 / gnorm;
        let mut moved = false;
        for _ in 0..40 {
            let trial: Vec<f64> = t.iter().zip(&grad).map(|(x, gr)| x - step * gr).collect();
            let f_try = value(&coeffs(&trial));
            if f_try.is_finite() && f_try < f_cur {
                t = trial;
                f_cur = f_try;
                moved = true;
                break;
            }
            step *= 0.5;
        }
        if !moved {
            break;
        }
    }
    let mut best = coeffs(&t);
    let mut best_value = value(&best);
    if best.iter().any(|x| !x.is_finite()) || !best_value.is_finite() {
        return None;
    }
    // The local t-space descent is not guaranteed to find the global
    // simplex minimum for an indefinite quadratic model. Compare it with
    // every pure-history vertex; the newest standard ADIIS vertex is zero.
    for i in 0..n {
        let mut vertex = vec![0.0; n];
        vertex[i] = 1.0;
        let vertex_value = value(&vertex);
        if vertex_value.is_finite() && vertex_value < best_value {
            best = vertex;
            best_value = vertex_value;
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coefficients stay on the simplex whatever the objective looks
    /// like — that is the whole point of the parameterization, and it is
    /// what CDIIS cannot promise.
    #[test]
    fn coefficients_stay_on_the_simplex() {
        for seed in 0..8 {
            let n = 4;
            let g: Vec<f64> = (0..n)
                .map(|i| ((i * 7 + seed * 3) % 11) as f64 / 11.0 - 0.5)
                .collect();
            let mut h = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let v = (((i * 5 + j * 3 + seed) % 7) as f64 / 7.0 - 0.4) * 0.5;
                    h[i * n + j] = v;
                    h[j * n + i] = v;
                }
            }
            let c = minimize_simplex(&g, &h, n).expect("minimizer");
            let sum: f64 = c.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "sum = {sum}");
            assert!(
                c.iter().all(|&x| x >= -1e-15),
                "negative coefficient: {c:?}"
            );
        }
    }

    #[test]
    fn local_candidate_never_loses_to_a_pure_endpoint() {
        let g = [0.8, 0.0];
        let h = [-0.1, 0.0, 0.0, 0.0];
        let c = minimize_simplex(&g, &h, 2).unwrap();
        let objective = |c: &[f64]| {
            2.0 * (g[0] * c[0] + g[1] * c[1])
                + c[0] * c[0] * h[0]
                + c[0] * c[1] * h[1]
                + c[1] * c[0] * h[2]
                + c[1] * c[1] * h[3]
        };
        let got = objective(&c);
        assert!(got <= objective(&[1.0, 0.0]) + 1e-14);
        assert!(got <= objective(&[0.0, 1.0]) + 1e-14);
    }

    /// With one stored iterate there is nothing to interpolate, and the
    /// caller must get `None` rather than a degenerate answer.
    #[test]
    fn single_iterate_declines() {
        let mut a = Adiis::new(8);
        let d = vec![Matrix::zeros(2, 2)];
        let f = vec![Matrix::zeros(2, 2)];
        a.push(&d, &f);
        assert!(a.extrapolate().is_none());
    }

    /// The objective is minimized, not merely evaluated: the returned
    /// coefficients must beat both pure endpoints on a case where the
    /// interior is genuinely better.
    #[test]
    fn interior_beats_the_endpoints() {
        let n = 2;
        // f(c) = 2 g.c + c^T h c with a minimum strictly inside.
        let g = vec![-1.0, 0.0];
        let h = vec![2.0, 0.0, 0.0, 2.0];
        let c = minimize_simplex(&g, &h, n).expect("minimizer");
        let val = |c: &[f64]| -> f64 {
            let mut v = 0.0;
            for i in 0..n {
                v += 2.0 * g[i] * c[i];
                for j in 0..n {
                    v += c[i] * c[j] * h[i * n + j];
                }
            }
            v
        };
        assert!(val(&c) <= val(&[1.0, 0.0]) + 1e-12);
        assert!(val(&c) <= val(&[0.0, 1.0]) + 1e-12);
    }
}
