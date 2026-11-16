// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete-active-space self-consistent field, state-specific or state
//! averaged.
//!
//! The orbitals are ordered core (doubly occupied), active, virtual. With
//! the core density `P_c = 2 C_c C_c^T`, the core Fock matrix
//! `F^I = h + J[P_c] - K[P_c]/2` and the active density
//! `P_a = C_a gamma C_a^T` with `F^A = J[P_a] - K[P_a]/2`, the energy of a
//! state is
//!
//! ```text
//! E = V_nn + Tr[P_c (h + F^I)]/2 + sum_tu F^I_tu gamma_tu
//!       + 1/2 sum_tuvw (tu|vw) Gamma_tuvw.
//! ```
//!
//! The generalized Fock matrix has the rows `F_iq = 2 (F^I_qi + F^A_qi)` for
//! a core orbital `i`, `F_tq = sum_u gamma_tu F^I_qu + sum_uvw Gamma_tuvw
//! (qu|vw)` for an active orbital `t` and zero for a virtual one. For the
//! rotation `C' = C exp(K)` with `K_pq = kappa_pq = -K_qp` (`p > q`) the
//! orbital gradient is `dE/dkappa_pq = 2 (F_qp - F_pq)`. The state-averaged
//! energy uses the weighted density matrices of the selected states. The
//! orbitals are optimized by a preconditioned limited-memory BFGS step on
//! the core-active, core-virtual and active-virtual rotations with an energy
//! backtracking line search; every energy evaluation solves the CI problem
//! again.

use super::ci::{ActiveHamiltonian, CiSpace, CiStates};
use crate::integrals::EriTensor;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::{DftError, Result};
use rayon::prelude::*;

/// Active space and state selection.
#[derive(Clone, Debug)]
pub struct CasSpec {
    /// Doubly occupied core orbitals.
    pub n_core: usize,
    /// Active orbitals.
    pub n_active: usize,
    /// Active alpha and beta electrons (`n_alpha - n_beta = 2 S`).
    pub n_alpha: usize,
    pub n_beta: usize,
    /// State-average weights of the lowest states of spin `S`, summing to one.
    pub weights: Vec<f64>,
    pub ci_tolerance: f64,
    pub ci_max_iterations: usize,
}

impl CasSpec {
    pub fn spin(&self) -> f64 {
        0.5 * (self.n_alpha as f64 - self.n_beta as f64)
    }
    pub fn roots(&self) -> usize {
        self.weights.len()
    }
}

/// Atomic-orbital operators of the molecule.
pub struct CasOperators<'a> {
    pub h_core: &'a Matrix,
    pub eri: &'a EriTensor,
    pub e_nuc: f64,
}

/// `J[P]_mn = sum (mn|ls) P_ls` and `K[P]_mn = sum (ml|ns) P_ls`.
pub fn coulomb_exchange(eri: &EriTensor, p: &Matrix) -> (Matrix, Matrix) {
    let n = eri.nao;
    let data = eri.as_slice();
    let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|m| {
            let mut j = vec![0.0; n];
            let mut k = vec![0.0; n];
            for nu in 0..n {
                let base = (m * n + nu) * n * n;
                j[nu] = data[base..base + n * n]
                    .iter()
                    .zip(p.as_slice())
                    .map(|(a, b)| a * b)
                    .sum();
            }
            for l in 0..n {
                for nu in 0..n {
                    let base = ((m * n + l) * n + nu) * n;
                    let row = &p.as_slice()[l * n..(l + 1) * n];
                    k[nu] += data[base..base + n]
                        .iter()
                        .zip(row)
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                }
            }
            (j, k)
        })
        .collect();
    let mut j = Matrix::zeros(n, n);
    let mut k = Matrix::zeros(n, n);
    for (m, (jr, kr)) in rows.into_iter().enumerate() {
        for nu in 0..n {
            j[(m, nu)] = jr[nu];
            k[(m, nu)] = kr[nu];
        }
    }
    (j, k)
}

/// `(qu|vw)` for every orbital `q` and active `u, v, w`, at
/// `((q n + u) n + v) n + w` with `n` active orbitals.
fn mixed_integrals(eri: &EriTensor, c: &Matrix, active: &Matrix) -> Vec<f64> {
    let nao = eri.nao;
    let n = active.cols;
    let data = eri.as_slice();
    // Y[(mu nu), (v w)] = sum_ls (mu nu|l s) C_lv C_sw.
    let y: Vec<Vec<f64>> = (0..nao * nao)
        .into_par_iter()
        .map(|row| {
            let block = Matrix::from_row_major(
                nao,
                nao,
                data[row * nao * nao..(row + 1) * nao * nao].to_vec(),
            );
            active
                .transpose_matmul_seq(&block.matmul_seq(active))
                .as_slice()
                .to_vec()
        })
        .collect();
    // Z[mu][(u v w)] = sum_nu C_nu,u Y[(mu nu), (v w)].
    let mut z = Matrix::zeros(nao, n * n * n);
    for mu in 0..nao {
        for nu in 0..nao {
            let yr = &y[mu * nao + nu];
            for u in 0..n {
                let factor = active[(nu, u)];
                if factor == 0.0 {
                    continue;
                }
                let target = &mut z.as_mut_slice()[mu * n * n * n + u * n * n..][..n * n];
                for (t, v) in target.iter_mut().zip(yr) {
                    *t += factor * v;
                }
            }
        }
    }
    c.transpose_matmul(&z).as_slice().to_vec()
}

/// `exp(K)` of a real antisymmetric matrix.
pub fn antisymmetric_exponential(k: &Matrix) -> Result<Matrix> {
    if k.rows != k.cols || k.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "an orbital-rotation generator must be finite and square".into(),
        ));
    }
    let scale = k.as_slice().iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    for i in 0..k.rows {
        for j in 0..=i {
            if (k[(i, j)] + k[(j, i)]).abs() > 64.0 * f64::EPSILON * scale {
                return Err(DftError::InvalidInput(
                    "an orbital-rotation generator must be antisymmetric".into(),
                ));
            }
        }
    }
    Ok(crate::scf::soscf::orthogonal_exponential(k))
}

/// Energies, density matrices and orbital derivatives at one set of orbitals.
#[derive(Clone, Debug)]
pub struct CasPoint {
    pub orbitals: Matrix,
    pub hamiltonian: ActiveHamiltonian,
    pub states: CiStates,
    /// Total energy of every selected state, Hartree.
    pub energies: Vec<f64>,
    pub average_energy: f64,
    /// State-averaged active one- and two-particle density matrices.
    pub gamma: Matrix,
    pub gamma2: Vec<f64>,
    /// `dE_avg/dkappa_pq` at `(p, q)`, `p > q`, zero for redundant pairs.
    pub gradient: Matrix,
    /// Positive diagonal model of the orbital Hessian on the same pairs.
    pub hessian_diagonal: Matrix,
    /// Core Fock matrix `F^I` in the atomic-orbital basis.
    pub core_fock: Matrix,
}

impl CasPoint {
    pub fn gradient_norm(&self) -> f64 {
        self.gradient
            .as_slice()
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
    }
}

/// Result of an orbital optimization.
#[derive(Clone, Debug)]
pub struct CasResult {
    pub point: CasPoint,
    pub iterations: usize,
    pub converged: bool,
    pub energy_history: Vec<f64>,
}

/// One active-space problem: operators, partition and CI space.
pub struct CasProblem<'a> {
    pub operators: CasOperators<'a>,
    pub spec: CasSpec,
    pub space: CiSpace,
}

impl<'a> CasProblem<'a> {
    pub fn new(operators: CasOperators<'a>, spec: CasSpec) -> Result<Self> {
        let total: f64 = spec.weights.iter().sum();
        if spec.weights.is_empty()
            || spec.weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || (total - 1.0).abs() > 1e-12
        {
            return Err(DftError::InvalidInput(
                "state-average weights are non-negative and sum to one".into(),
            ));
        }
        if spec.n_alpha < spec.n_beta {
            return Err(DftError::InvalidInput(
                "the active space is set up with at least as many alpha as beta electrons".into(),
            ));
        }
        let space = CiSpace::new(spec.n_active, spec.n_alpha, spec.n_beta)?;
        Ok(Self {
            operators,
            spec,
            space,
        })
    }

    fn core_and_active(&self, c: &Matrix) -> Result<(Matrix, Matrix)> {
        let (nc, na) = (self.spec.n_core, self.spec.n_active);
        if c.cols < nc + na || c.rows != self.operators.eri.nao {
            return Err(DftError::InvalidInput(
                "the orbitals do not hold the core and active spaces".into(),
            ));
        }
        let take = |start: usize, count: usize| -> Matrix {
            let mut m = Matrix::zeros(c.rows, count);
            for i in 0..c.rows {
                for j in 0..count {
                    m[(i, j)] = c[(i, start + j)];
                }
            }
            m
        };
        Ok((take(0, nc), take(nc, na)))
    }

    /// Active Hamiltonian of the orbitals `c`, the core Fock matrix (AO) and
    /// the mixed integrals `(qu|vw)`.
    pub fn active_hamiltonian(&self, c: &Matrix) -> Result<(ActiveHamiltonian, Matrix, Vec<f64>)> {
        let (core, active) = self.core_and_active(c)?;
        let h = self.operators.h_core;
        let p_core = core.matmul_transpose_seq(&core);
        let mut p_core2 = p_core.clone();
        p_core2.as_mut_slice().iter_mut().for_each(|v| *v *= 2.0);
        let (j, k) = coulomb_exchange(self.operators.eri, &p_core2);
        let mut fock = h.clone();
        for ((f, jv), kv) in fock
            .as_mut_slice()
            .iter_mut()
            .zip(j.as_slice())
            .zip(k.as_slice())
        {
            *f += jv - 0.5 * kv;
        }
        let mut sum = h.clone();
        for (s, f) in sum.as_mut_slice().iter_mut().zip(fock.as_slice()) {
            *s += f;
        }
        let constant = self.operators.e_nuc + 0.5 * p_core2.frobenius_dot(&sum);
        let mixed = mixed_integrals(self.operators.eri, c, &active);
        let n = self.spec.n_active;
        let nc = self.spec.n_core;
        let mut g = vec![0.0; n * n * n * n];
        for t in 0..n {
            g[t * n * n * n..(t + 1) * n * n * n]
                .copy_from_slice(&mixed[(nc + t) * n * n * n..(nc + t + 1) * n * n * n]);
        }
        Ok((
            ActiveHamiltonian {
                n,
                constant,
                h: active.transpose_matmul_seq(&fock.matmul_seq(&active)),
                g,
            },
            fock,
            mixed,
        ))
    }

    /// The selected lowest states of spin `S` of `ham`.
    pub fn states(&self, ham: &ActiveHamiltonian, guess: Option<&[Vec<f64>]>) -> Result<CiStates> {
        let spin = self.spec.spin();
        let target = spin * (spin + 1.0);
        let wanted = self.spec.roots();
        let dim = self.space.dimension();
        let mut count = (wanted + 2).min(dim).max(wanted);
        loop {
            let all = self.space.spin_adapted(
                ham,
                self.space.lowest(
                    ham,
                    count,
                    self.spec.ci_tolerance,
                    self.spec.ci_max_iterations,
                    guess,
                )?,
                1e-6,
            )?;
            let keep: Vec<usize> = (0..count)
                .filter(|&r| (all.spin_squared[r] - target).abs() < 1e-6)
                .take(wanted)
                .collect();
            if keep.len() == wanted {
                return Ok(CiStates {
                    energies: keep.iter().map(|&r| all.energies[r]).collect(),
                    vectors: keep.iter().map(|&r| all.vectors[r].clone()).collect(),
                    residuals: keep.iter().map(|&r| all.residuals[r]).collect(),
                    spin_squared: keep.iter().map(|&r| all.spin_squared[r]).collect(),
                });
            }
            if count == dim {
                return Err(DftError::InvalidInput(format!(
                    "the active space holds fewer than {wanted} states with S = {spin}"
                )));
            }
            count = (2 * count).min(dim);
        }
    }

    /// Energies, density matrices and the orbital gradient at `c`.
    pub fn evaluate(&self, c: &Matrix, guess: Option<&[Vec<f64>]>) -> Result<CasPoint> {
        let (ham, core_fock, mixed) = self.active_hamiltonian(c)?;
        let states = self.states(&ham, guess)?;
        let n = self.spec.n_active;
        let nc = self.spec.n_core;
        let nmo = c.cols;
        let mut gamma = Matrix::zeros(n, n);
        let mut gamma2 = vec![0.0; n * n * n * n];
        let mut energies = Vec::with_capacity(states.vectors.len());
        for (v, &w) in states.vectors.iter().zip(&self.spec.weights) {
            let (g1, g2) = self.space.rdm12(v, v)?;
            let mut e = ham.constant + g1.frobenius_dot(&ham.h);
            e += 0.5 * g2.iter().zip(&ham.g).map(|(a, b)| a * b).sum::<f64>();
            energies.push(e);
            for (t, s) in gamma.as_mut_slice().iter_mut().zip(g1.as_slice()) {
                *t += w * s;
            }
            for (t, s) in gamma2.iter_mut().zip(&g2) {
                *t += w * s;
            }
        }
        let average_energy = energies
            .iter()
            .zip(&self.spec.weights)
            .map(|(e, w)| e * w)
            .sum();
        let (_, active) = self.core_and_active(c)?;
        let p_active = active.matmul_seq(&gamma).matmul_transpose_seq(&active);
        let (ja, ka) = coulomb_exchange(self.operators.eri, &p_active);
        let mut fa = ja;
        for (f, k) in fa.as_mut_slice().iter_mut().zip(ka.as_slice()) {
            *f -= 0.5 * k;
        }
        let fi_mo = c.transpose_matmul(&core_fock.matmul(c));
        let fa_mo = c.transpose_matmul(&fa.matmul(c));
        let mut fgen = Matrix::zeros(nmo, nmo);
        for i in 0..nc {
            for q in 0..nmo {
                fgen[(i, q)] = 2.0 * (fi_mo[(q, i)] + fa_mo[(q, i)]);
            }
        }
        for t in 0..n {
            for q in 0..nmo {
                let mut value = 0.0;
                for u in 0..n {
                    value += gamma[(t, u)] * fi_mo[(q, nc + u)];
                }
                let row = &mixed[q * n * n * n..(q + 1) * n * n * n];
                for u in 0..n {
                    for vw in 0..n * n {
                        value += gamma2[(t * n + u) * n * n + vw] * row[u * n * n + vw];
                    }
                }
                fgen[(nc + t, q)] = value;
            }
        }
        let class = |p: usize| -> usize {
            if p < nc {
                0
            } else if p < nc + n {
                1
            } else {
                2
            }
        };
        let mut gradient = Matrix::zeros(nmo, nmo);
        let mut hessian = Matrix::zeros(nmo, nmo);
        for p in 0..nmo {
            for q in 0..p {
                if class(p) == class(q) {
                    continue;
                }
                gradient[(p, q)] = 2.0 * (fgen[(q, p)] - fgen[(p, q)]);
                let fiv = |r: usize| fi_mo[(r, r)] + fa_mo[(r, r)];
                let model = match (class(q), class(p)) {
                    (0, 2) => 4.0 * (fiv(p) - fiv(q)),
                    (1, 2) => {
                        let t = q - nc;
                        2.0 * gamma[(t, t)] * fiv(p) - 2.0 * fgen[(q, q)]
                    }
                    _ => {
                        let t = p - nc;
                        4.0 * (fiv(p) - fiv(q)) + 2.0 * gamma[(t, t)] * fiv(q) - 2.0 * fgen[(p, p)]
                    }
                };
                hessian[(p, q)] = model.max(0.05);
            }
        }
        Ok(CasPoint {
            orbitals: c.clone(),
            hamiltonian: ham,
            states,
            energies,
            average_energy,
            gamma,
            gamma2,
            gradient,
            hessian_diagonal: hessian,
            core_fock,
        })
    }

    /// Orbitals `c exp(K)` for the lower-triangle rotation parameters `kappa`.
    pub fn rotate(c: &Matrix, kappa: &Matrix) -> Result<Matrix> {
        let n = kappa.rows;
        let mut k = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..p {
                k[(p, q)] = kappa[(p, q)];
                k[(q, p)] = -kappa[(p, q)];
            }
        }
        Ok(c.matmul(&antisymmetric_exponential(&k)?))
    }

    /// Orbital optimization from `c` until the largest gradient element is
    /// below `gradient_tolerance` and the energy changes by less than
    /// `energy_tolerance`.
    pub fn optimize(
        &self,
        c: &Matrix,
        max_iterations: usize,
        gradient_tolerance: f64,
        energy_tolerance: f64,
    ) -> Result<CasResult> {
        let mut point = self.evaluate(c, None)?;
        let mut history = vec![point.average_energy];
        let mut steps: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        let mut last_change = f64::INFINITY;
        let pairs: Vec<(usize, usize)> = (0..c.cols)
            .flat_map(|p| (0..p).map(move |q| (p, q)))
            .filter(|&(p, q)| point.hessian_diagonal[(p, q)] > 0.0)
            .collect();
        let gather = |m: &Matrix| -> Vec<f64> { pairs.iter().map(|&(p, q)| m[(p, q)]).collect() };
        for iteration in 1..=max_iterations {
            crate::logger::log(&format!(
                "[casscf] iter {iteration:>3} E={:.12} dE={last_change:.3e} gradient={:.3e}",
                point.average_energy,
                point.gradient_norm()
            ));
            if point.gradient_norm() < gradient_tolerance && last_change.abs() < energy_tolerance {
                return Ok(CasResult {
                    point,
                    iterations: iteration - 1,
                    converged: true,
                    energy_history: history,
                });
            }
            let g = gather(&point.gradient);
            let h0 = gather(&point.hessian_diagonal);
            // Two-loop recursion with the diagonal model as initial inverse.
            let mut q = g.clone();
            let mut alphas = Vec::with_capacity(steps.len());
            for (s, y) in steps.iter().rev() {
                let rho = 1.0 / dot(y, s);
                let alpha = rho * dot(s, &q);
                for (qi, yi) in q.iter_mut().zip(y) {
                    *qi -= alpha * yi;
                }
                alphas.push(alpha);
            }
            for (qi, hi) in q.iter_mut().zip(&h0) {
                *qi /= hi;
            }
            for ((s, y), alpha) in steps.iter().zip(alphas.iter().rev()) {
                let rho = 1.0 / dot(y, s);
                let beta = rho * dot(y, &q);
                for (qi, si) in q.iter_mut().zip(s) {
                    *qi += (alpha - beta) * si;
                }
            }
            let mut step: Vec<f64> = q.iter().map(|v| -v).collect();
            if dot(&step, &g) >= 0.0 {
                // Not a descent direction: fall back to the diagonal model.
                steps.clear();
                step = g.iter().zip(&h0).map(|(gi, hi)| -gi / hi).collect();
            }
            let largest = step.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if largest > 0.2 {
                step.iter_mut().for_each(|v| *v *= 0.2 / largest);
            }
            let guess = point.states.vectors.clone();
            let mut scale = 1.0;
            let mut accepted = None;
            for _ in 0..12 {
                let mut kappa = Matrix::zeros(c.cols, c.cols);
                for (&(p, qq), s) in pairs.iter().zip(&step) {
                    kappa[(p, qq)] = scale * s;
                }
                let trial = self.evaluate(&Self::rotate(&point.orbitals, &kappa)?, Some(&guess))?;
                if trial.average_energy <= point.average_energy + 1e-12 {
                    accepted = Some((trial, scale));
                    break;
                }
                scale *= 0.5;
            }
            let Some((trial, scale)) = accepted else {
                return Err(DftError::InvalidInput(format!(
                    "the CASSCF line search found no lower energy at iteration {iteration} \
                     (largest orbital gradient {:.3e})",
                    point.gradient_norm()
                )));
            };
            let s: Vec<f64> = step.iter().map(|v| v * scale).collect();
            let y: Vec<f64> = gather(&trial.gradient)
                .iter()
                .zip(&g)
                .map(|(a, b)| a - b)
                .collect();
            if dot(&s, &y) > 1e-12 {
                steps.push((s, y));
                if steps.len() > 8 {
                    steps.remove(0);
                }
            }
            last_change = trial.average_energy - point.average_energy;
            point = trial;
            history.push(point.average_energy);
        }
        Ok(CasResult {
            point,
            iterations: max_iterations,
            converged: false,
            energy_history: history,
        })
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Natural occupation numbers (descending) and natural orbitals of an
/// active one-particle density matrix, in the active orbital basis.
pub fn natural_orbitals(gamma: &Matrix) -> Result<(Vec<f64>, Matrix)> {
    let (values, vectors) = symmetric_eigen(gamma)?;
    let n = values.len();
    let mut occupations = Vec::with_capacity(n);
    let mut sorted = Matrix::zeros(n, n);
    for (col, r) in (0..n).rev().enumerate() {
        occupations.push(values[r]);
        for i in 0..n {
            sorted[(i, col)] = vectors[(i, r)];
        }
    }
    Ok((occupations, sorted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antisymmetric_exponential_is_orthogonal_and_matches_the_series() {
        let mut k = Matrix::zeros(3, 3);
        let values = [(1, 0, 0.3), (2, 0, -0.2), (2, 1, 0.45)];
        for (p, q, v) in values {
            k[(p, q)] = v;
            k[(q, p)] = -v;
        }
        let u = antisymmetric_exponential(&k).unwrap();
        let identity = u.transpose_matmul_seq(&u);
        let mut series = Matrix::identity(3);
        let mut term = Matrix::identity(3);
        for order in 1..30 {
            term = term.matmul_seq(&k);
            term.as_mut_slice()
                .iter_mut()
                .for_each(|v| *v /= order as f64);
            for (s, t) in series.as_mut_slice().iter_mut().zip(term.as_slice()) {
                *s += t;
            }
        }
        assert!(identity.max_abs_diff(&Matrix::identity(3)) < 1e-14);
        assert!(u.max_abs_diff(&series) < 1e-14);
    }
}
