// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Determinant configuration interaction in an active space.
//!
//! An occupation string of `n` active orbitals is a bit mask, and a CI vector
//! is the matrix `C[alpha string][beta string]` stored row by row. The
//! active Hamiltonian
//!
//! ```text
//! H = c + sum_kl h_kl E_kl + 1/2 sum_klmn (kl|mn) (E_kl E_mn - delta_lm E_kn)
//! ```
//!
//! acts through the one-particle replacement lists of the two spin strings:
//! with `D_mn = E_mn C` and `G_kl = sum_mn (kl|mn) D_mn`,
//! `H C = c C + sum_kl htilde_kl D_kl + 1/2 sum_kl E_kl G_kl`,
//! `htilde_kl = h_kl - 1/2 sum_m (km|ml)`. The same replacement vectors
//! give the spin-free reduced density matrices
//! `gamma_kl = <bra|E_kl|ket>` and
//! `Gamma_klmn = <bra|E_kl E_mn|ket> - delta_lm gamma_kn`, with
//! `<bra|E_kl E_mn|ket> = (E_lk bra) . (E_mn ket)` for real vectors.

use crate::linalg::{symmetric_eigen, Matrix};
use crate::{DftError, Result};
use rayon::prelude::*;

/// Active-space Hamiltonian: constant, one-electron matrix and two-electron
/// integrals `(kl|mn)` at `((k n + l) n + m) n + o`, all real.
#[derive(Clone, Debug)]
pub struct ActiveHamiltonian {
    pub n: usize,
    pub constant: f64,
    pub h: Matrix,
    pub g: Vec<f64>,
}

impl ActiveHamiltonian {
    fn g(&self, k: usize, l: usize, m: usize, o: usize) -> f64 {
        let n = self.n;
        self.g[((k * n + l) * n + m) * n + o]
    }
}

#[derive(Clone, Copy, Debug)]
struct Replacement {
    source: usize,
    target: usize,
    sign: f64,
}

/// All strings of `n_elec` electrons in `n_orb` orbitals, ascending, with
/// the replacements `a+_k a_l` grouped by the orbital pair `k n + l`.
#[derive(Clone, Debug)]
struct StringSpace {
    strings: Vec<u64>,
    by_pair: Vec<Vec<Replacement>>,
}

/// Ascending bit masks with `k` of the lowest `n` bits set.
fn combinations(n: usize, k: usize) -> Vec<u64> {
    if k > n {
        return Vec::new();
    }
    if k == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    let mut mask: u64 = (1u64 << k) - 1;
    let limit = 1u64 << n;
    while mask < limit {
        out.push(mask);
        let lowest = mask & mask.wrapping_neg();
        let ripple = mask + lowest;
        mask = (((ripple ^ mask) >> 2) / lowest) | ripple;
    }
    out
}

impl StringSpace {
    fn new(n_orb: usize, n_elec: usize) -> Self {
        let strings = combinations(n_orb, n_elec);
        let mut by_pair = vec![Vec::new(); n_orb * n_orb];
        for (source, &string) in strings.iter().enumerate() {
            for l in (0..n_orb).filter(|&l| (string >> l) & 1 == 1) {
                let removed = string & !(1u64 << l);
                let below_l = (string & ((1u64 << l) - 1)).count_ones();
                for k in (0..n_orb).filter(|&k| k == l || (removed >> k) & 1 == 0) {
                    let target_string = removed | (1u64 << k);
                    let below_k = (removed & ((1u64 << k) - 1)).count_ones();
                    let target = strings
                        .binary_search(&target_string)
                        .expect("a one-particle replacement stays in the string space");
                    let sign = if (below_l + below_k) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    by_pair[k * n_orb + l].push(Replacement {
                        source,
                        target,
                        sign,
                    });
                }
            }
        }
        Self { strings, by_pair }
    }

    fn len(&self) -> usize {
        self.strings.len()
    }
}

/// Determinant space of a fixed number of alpha and beta electrons.
#[derive(Clone, Debug)]
pub struct CiSpace {
    pub n_orb: usize,
    pub n_alpha: usize,
    pub n_beta: usize,
    alpha: StringSpace,
    beta: StringSpace,
}

/// Eigenstates of an active Hamiltonian.
#[derive(Clone, Debug)]
pub struct CiStates {
    pub energies: Vec<f64>,
    pub vectors: Vec<Vec<f64>>,
    /// Norm of `H C - E C` of every returned state.
    pub residuals: Vec<f64>,
    /// `<S^2>` of every returned state.
    pub spin_squared: Vec<f64>,
}

/// Number of determinants below which the Hamiltonian is diagonalized
/// densely instead of iteratively.
const DENSE_LIMIT: usize = 400;

impl CiSpace {
    pub fn new(n_orb: usize, n_alpha: usize, n_beta: usize) -> Result<Self> {
        if n_orb > 62 || n_alpha > n_orb || n_beta > n_orb {
            return Err(DftError::InvalidInput(format!(
                "an active space holds at most 62 orbitals and at most one electron of each spin \
                 per orbital (got {n_orb} orbitals, {n_alpha} alpha and {n_beta} beta electrons)"
            )));
        }
        Ok(Self {
            n_orb,
            n_alpha,
            n_beta,
            alpha: StringSpace::new(n_orb, n_alpha),
            beta: StringSpace::new(n_orb, n_beta),
        })
    }

    pub fn dimension(&self) -> usize {
        self.alpha.len() * self.beta.len()
    }

    /// Alpha and beta occupation masks for the determinant at a CI vector index.
    pub(crate) fn occupations(&self, index: usize) -> [u64; 2] {
        [
            self.alpha.strings[index / self.beta.len()],
            self.beta.strings[index % self.beta.len()],
        ]
    }

    /// Bytes of the replacement vectors `E_kl C` of one CI vector.
    pub fn replacement_bytes(&self) -> usize {
        self.n_orb * self.n_orb * self.dimension() * 8
    }

    /// `D[kl] = E_kl C` as the rows of an `n^2 x dim` matrix.
    fn replacements(&self, c: &[f64]) -> Matrix {
        let (n, nb, dim) = (self.n_orb, self.beta.len(), self.dimension());
        let rows: Vec<Vec<f64>> = (0..n * n)
            .into_par_iter()
            .map(|pair| {
                let mut row = vec![0.0; dim];
                for r in &self.alpha.by_pair[pair] {
                    let (src, dst) = (r.source * nb, r.target * nb);
                    for ib in 0..nb {
                        row[dst + ib] += r.sign * c[src + ib];
                    }
                }
                for r in &self.beta.by_pair[pair] {
                    for ia in 0..self.alpha.len() {
                        row[ia * nb + r.target] += r.sign * c[ia * nb + r.source];
                    }
                }
                row
            })
            .collect();
        Matrix::from_row_major(n * n, dim, rows.concat())
    }

    /// `sum_kl E_kl X[kl]` for the rows `X[kl]` of an `n^2 x dim` matrix.
    fn apply_replacements(&self, x: &Matrix) -> Vec<f64> {
        let (n, nb, dim) = (self.n_orb, self.beta.len(), self.dimension());
        (0..n * n)
            .into_par_iter()
            .fold(
                || vec![0.0; dim],
                |mut out, pair| {
                    let row = &x.as_slice()[pair * dim..(pair + 1) * dim];
                    for r in &self.alpha.by_pair[pair] {
                        let (src, dst) = (r.source * nb, r.target * nb);
                        for ib in 0..nb {
                            out[dst + ib] += r.sign * row[src + ib];
                        }
                    }
                    for r in &self.beta.by_pair[pair] {
                        for ia in 0..self.alpha.len() {
                            out[ia * nb + r.target] += r.sign * row[ia * nb + r.source];
                        }
                    }
                    out
                },
            )
            .reduce(
                || vec![0.0; dim],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            )
    }

    fn check(&self, ham: &ActiveHamiltonian) -> Result<()> {
        let n = self.n_orb;
        if ham.n != n || ham.h.rows != n || ham.h.cols != n || ham.g.len() != n * n * n * n {
            return Err(DftError::InvalidInput(
                "the active Hamiltonian does not match the orbital count of the CI space".into(),
            ));
        }
        Ok(())
    }

    /// `H C` for the active Hamiltonian.
    pub fn sigma(&self, ham: &ActiveHamiltonian, c: &[f64]) -> Result<Vec<f64>> {
        self.check(ham)?;
        let n = self.n_orb;
        let dim = self.dimension();
        if c.len() != dim {
            return Err(DftError::InvalidInput(
                "a CI vector does not match the determinant space".into(),
            ));
        }
        let d = self.replacements(c);
        let mut tilde = ham.h.clone();
        for k in 0..n {
            for l in 0..n {
                tilde[(k, l)] -= 0.5 * (0..n).map(|m| ham.g(k, m, m, l)).sum::<f64>();
            }
        }
        let integrals = Matrix::from_row_major(n * n, n * n, ham.g.clone());
        let g = integrals.matmul(&d);
        let mut sigma = self.apply_replacements(&g);
        let one: Vec<f64> = (0..dim)
            .into_par_iter()
            .map(|i| {
                (0..n * n)
                    .map(|pair| tilde.as_slice()[pair] * d.as_slice()[pair * dim + i])
                    .sum::<f64>()
            })
            .collect();
        for ((s, o), ci) in sigma.iter_mut().zip(one).zip(c) {
            *s = 0.5 * *s + o + ham.constant * ci;
        }
        Ok(sigma)
    }

    /// Diagonal of the Hamiltonian in the determinant basis.
    pub fn diagonal(&self, ham: &ActiveHamiltonian) -> Result<Vec<f64>> {
        self.check(ham)?;
        let n = self.n_orb;
        let occupied = |s: u64| -> Vec<usize> { (0..n).filter(|&i| (s >> i) & 1 == 1).collect() };
        let same = |occ: &[usize]| -> f64 {
            let mut e = 0.0;
            for &i in occ {
                e += ham.h[(i, i)];
                for &j in occ {
                    e += 0.5 * (ham.g(i, i, j, j) - ham.g(i, j, j, i));
                }
            }
            e
        };
        let alpha: Vec<(Vec<usize>, f64)> = self
            .alpha
            .strings
            .iter()
            .map(|&s| {
                let o = occupied(s);
                let e = same(&o);
                (o, e)
            })
            .collect();
        let beta: Vec<(Vec<usize>, f64)> = self
            .beta
            .strings
            .iter()
            .map(|&s| {
                let o = occupied(s);
                let e = same(&o);
                (o, e)
            })
            .collect();
        let mut out = Vec::with_capacity(self.dimension());
        for (oa, ea) in &alpha {
            for (ob, eb) in &beta {
                let mut cross = 0.0;
                for &i in oa {
                    for &j in ob {
                        cross += ham.g(i, i, j, j);
                    }
                }
                out.push(ham.constant + ea + eb + cross);
            }
        }
        Ok(out)
    }

    /// Spin-free one- and two-particle (transition) density matrices
    /// `<bra|E_kl|ket>` and `<bra|E_kl E_mn|ket> - delta_lm <bra|E_kn|ket>`.
    pub fn rdm12(&self, bra: &[f64], ket: &[f64]) -> Result<(Matrix, Vec<f64>)> {
        let dim = self.dimension();
        if bra.len() != dim || ket.len() != dim {
            return Err(DftError::InvalidInput(
                "a CI vector does not match the determinant space".into(),
            ));
        }
        let n = self.n_orb;
        let db = self.replacements(bra);
        let dk = if std::ptr::eq(bra, ket) {
            db.clone()
        } else {
            self.replacements(ket)
        };
        let mut gamma = Matrix::zeros(n, n);
        for k in 0..n {
            for l in 0..n {
                gamma[(k, l)] = bra
                    .iter()
                    .zip(&dk.as_slice()[(k * n + l) * dim..(k * n + l + 1) * dim])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        let products = db.matmul_transpose_seq(&dk);
        let mut two = vec![0.0; n * n * n * n];
        for k in 0..n {
            for l in 0..n {
                for m in 0..n {
                    for o in 0..n {
                        let mut value = products[(l * n + k, m * n + o)];
                        if l == m {
                            value -= gamma[(k, o)];
                        }
                        two[((k * n + l) * n + m) * n + o] = value;
                    }
                }
            }
        }
        Ok((gamma, two))
    }

    /// Spin-resolved one-particle (transition) density matrices
    /// `<bra|a+_(k s) a_(l s)|ket>`, alpha then beta.
    pub fn rdm1_spin(&self, bra: &[f64], ket: &[f64]) -> Result<[Matrix; 2]> {
        let dim = self.dimension();
        if bra.len() != dim || ket.len() != dim {
            return Err(DftError::InvalidInput(
                "a CI vector does not match the determinant space".into(),
            ));
        }
        let (n, na, nb) = (self.n_orb, self.alpha.len(), self.beta.len());
        let mut out = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
        for k in 0..n {
            for l in 0..n {
                let pair = k * n + l;
                let mut alpha = 0.0;
                for r in &self.alpha.by_pair[pair] {
                    for ib in 0..nb {
                        alpha += r.sign * bra[r.target * nb + ib] * ket[r.source * nb + ib];
                    }
                }
                let mut beta = 0.0;
                for r in &self.beta.by_pair[pair] {
                    for ia in 0..na {
                        beta += r.sign * bra[ia * nb + r.target] * ket[ia * nb + r.source];
                    }
                }
                out[0][(k, l)] = alpha;
                out[1][(k, l)] = beta;
            }
        }
        Ok(out)
    }

    /// `<S^2> = -N (N - 4) / 4 - 1/2 sum_pq Gamma_pqqp` of a normalized state.
    pub fn spin_squared(&self, c: &[f64]) -> Result<f64> {
        let (_, two) = self.rdm12(c, c)?;
        let n = self.n_orb;
        let electrons = (self.n_alpha + self.n_beta) as f64;
        let mut exchange = 0.0;
        for p in 0..n {
            for q in 0..n {
                exchange += two[((p * n + q) * n + q) * n + p];
            }
        }
        Ok(-electrons * (electrons - 4.0) / 4.0 - 0.5 * exchange)
    }

    /// Spin eigenstates of groups of states whose energies agree within
    /// `tolerance`: a degenerate group of eigenvectors may mix spin
    /// multiplets of the same `M_S`, so `S^2` is diagonalized in the group
    /// and the Hamiltonian again in every block of equal `<S^2>`.
    pub fn spin_adapted(
        &self,
        ham: &ActiveHamiltonian,
        states: CiStates,
        tolerance: f64,
    ) -> Result<CiStates> {
        let count = states.energies.len();
        let electrons = (self.n_alpha + self.n_beta) as f64;
        let constant = -electrons * (electrons - 4.0) / 4.0;
        let n = self.n_orb;
        let mut vectors = states.vectors;
        let mut start = 0;
        while start < count {
            let mut end = start + 1;
            while end < count && (states.energies[end] - states.energies[start]).abs() < tolerance {
                end += 1;
            }
            let k = end - start;
            if k > 1 {
                let group: Vec<Vec<f64>> = vectors[start..end].to_vec();
                let mut spin = Matrix::zeros(k, k);
                for i in 0..k {
                    for j in 0..k {
                        let (_, two) = self.rdm12(&group[i], &group[j])?;
                        let mut exchange = 0.0;
                        for p in 0..n {
                            for q in 0..n {
                                exchange += two[((p * n + q) * n + q) * n + p];
                            }
                        }
                        spin[(i, j)] = if i == j { constant } else { 0.0 } - 0.5 * exchange;
                    }
                }
                for i in 0..k {
                    for j in 0..i {
                        let average = 0.5 * (spin[(i, j)] + spin[(j, i)]);
                        spin[(i, j)] = average;
                        spin[(j, i)] = average;
                    }
                }
                let (values, rotation) = symmetric_eigen(&spin)?;
                let combine =
                    |coefficients: &dyn Fn(usize) -> f64, from: &[Vec<f64>]| -> Vec<f64> {
                        let mut v = vec![0.0; from[0].len()];
                        for (i, source) in from.iter().enumerate() {
                            let c = coefficients(i);
                            for (x, y) in v.iter_mut().zip(source) {
                                *x += c * y;
                            }
                        }
                        v
                    };
                let spin_vectors: Vec<Vec<f64>> = (0..k)
                    .map(|col| combine(&|i| rotation[(i, col)], &group))
                    .collect();
                let mut adapted: Vec<(f64, Vec<f64>)> = Vec::with_capacity(k);
                let mut block = 0;
                while block < k {
                    let mut stop = block + 1;
                    while stop < k && (values[stop] - values[block]).abs() < 1e-6 {
                        stop += 1;
                    }
                    let members = &spin_vectors[block..stop];
                    let m = members.len();
                    let images: Vec<Vec<f64>> = members
                        .iter()
                        .map(|v| self.sigma(ham, v))
                        .collect::<Result<_>>()?;
                    let mut small = Matrix::zeros(m, m);
                    for i in 0..m {
                        for j in 0..m {
                            small[(i, j)] = 0.5
                                * (members[i]
                                    .iter()
                                    .zip(&images[j])
                                    .map(|(a, b)| a * b)
                                    .sum::<f64>()
                                    + members[j]
                                        .iter()
                                        .zip(&images[i])
                                        .map(|(a, b)| a * b)
                                        .sum::<f64>());
                        }
                    }
                    let (energies, y) = symmetric_eigen(&small)?;
                    for (col, e) in energies.into_iter().enumerate() {
                        adapted.push((e, combine(&|i| y[(i, col)], members)));
                    }
                    block = stop;
                }
                adapted.sort_by(|a, b| a.0.total_cmp(&b.0));
                for (offset, (_, v)) in adapted.into_iter().enumerate() {
                    vectors[start + offset] = v;
                }
            }
            start = end;
        }
        let mut energies = Vec::with_capacity(count);
        let mut residuals = Vec::with_capacity(count);
        let mut spin_squared = Vec::with_capacity(count);
        for v in &vectors {
            let hv = self.sigma(ham, v)?;
            let e: f64 = v.iter().zip(&hv).map(|(a, b)| a * b).sum();
            residuals.push(
                hv.iter()
                    .zip(v)
                    .map(|(a, b)| (a - e * b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            );
            energies.push(e);
            spin_squared.push(self.spin_squared(v)?);
        }
        Ok(CiStates {
            energies,
            vectors,
            residuals,
            spin_squared,
        })
    }

    /// The `count` lowest eigenstates (dense below [`DENSE_LIMIT`]
    /// determinants, Davidson iterations above), started from `guess` when
    /// given.
    pub fn lowest(
        &self,
        ham: &ActiveHamiltonian,
        count: usize,
        tolerance: f64,
        max_iterations: usize,
        guess: Option<&[Vec<f64>]>,
    ) -> Result<CiStates> {
        let dim = self.dimension();
        if count == 0 || count > dim {
            return Err(DftError::InvalidInput(format!(
                "{count} CI states requested from a space of {dim} determinants"
            )));
        }
        let (energies, vectors) = if dim <= DENSE_LIMIT {
            let mut matrix = Matrix::zeros(dim, dim);
            for j in 0..dim {
                let mut unit = vec![0.0; dim];
                unit[j] = 1.0;
                let column = self.sigma(ham, &unit)?;
                for (i, value) in column.into_iter().enumerate() {
                    matrix[(i, j)] = value;
                }
            }
            for i in 0..dim {
                for j in 0..i {
                    let average = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
                    matrix[(i, j)] = average;
                    matrix[(j, i)] = average;
                }
            }
            let (values, vectors) = symmetric_eigen(&matrix)?;
            let states = (0..count)
                .map(|r| (0..dim).map(|i| vectors[(i, r)]).collect())
                .collect();
            (values[..count].to_vec(), states)
        } else {
            self.davidson(ham, count, tolerance, max_iterations, guess)?
        };
        let mut residuals = Vec::with_capacity(count);
        let mut spin_squared = Vec::with_capacity(count);
        for (e, v) in energies.iter().zip(&vectors) {
            let hv = self.sigma(ham, v)?;
            residuals.push(
                hv.iter()
                    .zip(v)
                    .map(|(a, b)| (a - e * b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            );
            spin_squared.push(self.spin_squared(v)?);
        }
        Ok(CiStates {
            energies,
            vectors,
            residuals,
            spin_squared,
        })
    }

    fn davidson(
        &self,
        ham: &ActiveHamiltonian,
        count: usize,
        tolerance: f64,
        max_iterations: usize,
        guess: Option<&[Vec<f64>]>,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let dim = self.dimension();
        let diagonal = self.diagonal(ham)?;
        let max_subspace = (8 * count + 24).min(dim);
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut images: Vec<Vec<f64>> = Vec::new();
        let add = |basis: &mut Vec<Vec<f64>>, mut v: Vec<f64>| -> bool {
            for _ in 0..2 {
                for b in basis.iter() {
                    let overlap: f64 = b.iter().zip(&v).map(|(x, y)| x * y).sum();
                    for (x, y) in v.iter_mut().zip(b) {
                        *x -= overlap * y;
                    }
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                return false;
            }
            v.iter_mut().for_each(|x| *x /= norm);
            basis.push(v);
            true
        };
        if let Some(vectors) = guess {
            for v in vectors.iter().filter(|v| v.len() == dim) {
                add(&mut basis, v.clone());
            }
        }
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&a, &b| diagonal[a].total_cmp(&diagonal[b]));
        for &i in &order {
            if basis.len() >= (2 * count).max(count + 2).min(dim) {
                break;
            }
            let mut unit = vec![0.0; dim];
            unit[i] = 1.0;
            add(&mut basis, unit);
        }
        for _ in 0..max_iterations {
            while images.len() < basis.len() {
                images.push(self.sigma(ham, &basis[images.len()])?);
            }
            let k = basis.len();
            let mut small = Matrix::zeros(k, k);
            for i in 0..k {
                for j in 0..=i {
                    let value: f64 = basis[i].iter().zip(&images[j]).map(|(a, b)| a * b).sum();
                    small[(i, j)] = value;
                    small[(j, i)] = value;
                }
            }
            let (theta, y) = symmetric_eigen(&small)?;
            let ritz = |r: usize, from: &[Vec<f64>]| -> Vec<f64> {
                let mut v = vec![0.0; dim];
                for (j, b) in from.iter().enumerate() {
                    for (x, z) in v.iter_mut().zip(b) {
                        *x += y[(j, r)] * z;
                    }
                }
                v
            };
            let mut corrections = Vec::new();
            let mut converged = true;
            for r in 0..count {
                let x = ritz(r, &basis);
                let ax = ritz(r, &images);
                let residual: Vec<f64> = ax.iter().zip(&x).map(|(a, b)| a - theta[r] * b).collect();
                let norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > tolerance {
                    converged = false;
                    corrections.push(
                        residual
                            .iter()
                            .zip(&diagonal)
                            .map(|(r_i, d)| {
                                let denominator = theta[r] - d;
                                if denominator.abs() < 1e-8 {
                                    r_i / 1e-8_f64.copysign(denominator)
                                } else {
                                    r_i / denominator
                                }
                            })
                            .collect::<Vec<f64>>(),
                    );
                }
            }
            if converged {
                let energies = theta[..count].to_vec();
                let vectors = (0..count).map(|r| ritz(r, &basis)).collect();
                return Ok((energies, vectors));
            }
            if basis.len() + corrections.len() > max_subspace {
                let keep = (2 * count).min(k);
                let collapsed: Vec<Vec<f64>> = (0..keep).map(|r| ritz(r, &basis)).collect();
                basis.clear();
                images.clear();
                for v in collapsed {
                    add(&mut basis, v);
                }
            }
            let before = basis.len();
            for v in corrections {
                add(&mut basis, v);
            }
            if basis.len() == before {
                return Err(DftError::InvalidInput(
                    "the Davidson iteration found no new direction before converging".into(),
                ));
            }
        }
        Err(DftError::InvalidInput(format!(
            "the CI Davidson iteration did not converge in {max_iterations} iterations"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_signs_follow_the_fermion_order() {
        // |{1,2}> = a+_1 a+_2 |0>: a_2 passes the occupied orbital 1 (sign
        // -1), then a+_0 passes nothing, giving -|{0,1}>.
        let space = StringSpace::new(3, 2);
        let pair = 2;
        let r = space.by_pair[pair]
            .iter()
            .find(|r| space.strings[r.source] == 0b110)
            .unwrap();
        assert_eq!(space.strings[r.target], 0b011);
        assert_eq!(r.sign, -1.0);
    }
}
