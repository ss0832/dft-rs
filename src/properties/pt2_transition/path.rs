// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Collinear propagator tensors on complete noncanonical orbital paths.
use crate::dual::Taylor;
use crate::linalg::Matrix;
use crate::periodic::BlochMatrix;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug)]
struct J(Taylor<3>);
impl From<f64> for J {
    fn from(x: f64) -> Self {
        Self(Taylor::from_coefficients(&[x]))
    }
}
impl Add for J {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self(self.0 + b.0)
    }
}
impl Sub for J {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self(self.0 - b.0)
    }
}
impl Mul for J {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self(self.0 * b.0)
    }
}
impl Div for J {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        Self(self.0 / b.0)
    }
}
impl Neg for J {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl Mul<f64> for J {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        self * Self::from(b)
    }
}
impl Mul<J> for f64 {
    type Output = J;
    fn mul(self, b: J) -> J {
        J::from(self) * b
    }
}
impl AddAssign for J {
    fn add_assign(&mut self, b: Self) {
        *self = *self + b
    }
}
impl SubAssign for J {
    fn sub_assign(&mut self, b: Self) {
        *self = *self - b
    }
}
#[derive(Clone)]
struct Jm {
    rows: usize,
    cols: usize,
    data: Vec<J>,
}
impl Jm {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![J::from(0.0); rows * cols],
        }
    }
    fn as_mut_slice(&mut self) -> &mut [J] {
        &mut self.data
    }
    fn as_slice(&self) -> &[J] {
        &self.data
    }
    fn matmul(&self, b: &Self) -> Self {
        let mut out = Self::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for j in 0..b.cols {
                for k in 0..self.cols {
                    out[(i, j)] += self[(i, k)] * b[(k, j)];
                }
            }
        }
        out
    }
}
impl Index<(usize, usize)> for Jm {
    type Output = J;
    fn index(&self, (i, j): (usize, usize)) -> &J {
        &self.data[i * self.cols + j]
    }
}
impl IndexMut<(usize, usize)> for Jm {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut J {
        &mut self.data[i * self.cols + j]
    }
}
fn combine(a: &Jm, b: &Jm, sa: f64, sb: f64) -> Jm {
    let mut out = a.clone();
    for (x, y) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x = *x * sa + *y * sb;
    }
    out
}
fn commutator(a: &Jm, b: &Jm) -> Jm {
    combine(&a.matmul(b), &b.matmul(a), 1.0, -1.0)
}
struct Core {
    n: [usize; 2],
    occupied: [usize; 2],
    integrals: [Vec<J>; 3],
    amplitudes: [Vec<J>; 3],
    coefficients: Pt2Coefficients,
    density: [Jm; 2],
    fock: [Jm; 2],
}
impl Core {
    fn dimensions(&self) -> [usize; 2] {
        [self.n[0], self.n[1]]
    }
    fn weight(&self, s: usize, t: usize) -> J {
        if s == t {
            J::from(self.coefficients.same_spin)
        } else {
            J::from(self.coefficients.opposite_spin)
        }
    }
    fn g(
        &self,
        blocks: &[Vec<J>; 3],
        s: usize,
        t: usize,
        p: usize,
        q: usize,
        r: usize,
        u: usize,
    ) -> J {
        let n = self.dimensions();
        if s == 1 && t == 0 {
            blocks[1][((r * n[0] + u) * n[1] + p) * n[1] + q]
        } else {
            blocks[if s == t { 2 * s } else { 1 }][((p * n[s] + q) * n[t] + r) * n[t] + u]
        }
    }
    fn t(
        &self,
        amp: &[Vec<J>; 3],
        s: usize,
        t: usize,
        i: usize,
        a: usize,
        j: usize,
        b: usize,
    ) -> J {
        let v = [self.n[0] - self.occupied[0], self.n[1] - self.occupied[1]];
        if s == 1 && t == 0 {
            amp[1][((j * v[0] + b) * self.occupied[1] + i) * v[1] + a]
        } else {
            amp[if s == t { 2 * s } else { 1 }][((i * v[s] + a) * self.occupied[t] + j) * v[t] + b]
        }
    }
    fn contraction(&self, blocks: &[Vec<J>; 3], amp: &[Vec<J>; 3]) -> [Jm; 2] {
        let n = self.dimensions();
        std::array::from_fn(|s| {
            let mut l = Jm::zeros(n[s], n[s]);
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
    pub fn coupling(&self, amplitudes: &[Vec<J>; 3]) -> [Jm; 2] {
        let l = self.contraction(&self.integrals, amplitudes);
        std::array::from_fn(|s| {
            let o = self.occupied[s];
            let n = self.n[s];
            let mut out = Jm::zeros(n - o, o);
            for a in o..n {
                for i in 0..o {
                    out[(a - o, i)] = l[s][(a, i)] - l[s][(i, a)];
                }
            }
            out
        })
    }
    fn rotation(&self, x: &[Jm; 2], odd: bool) -> [Jm; 2] {
        std::array::from_fn(|s| {
            let n = self.n[s];
            let o = self.occupied[s];
            let mut k = Jm::zeros(n, n);
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
    fn integral_rotation(&self, k: &[Jm; 2], odd: bool) -> [Vec<J>; 3] {
        let n = self.dimensions();
        let bra = if odd { -1.0 } else { 1.0 };
        let blocks: Vec<_> = [(0, 0), (0, 1), (1, 1)]
            .into_iter()
            .map(|(s, t)| {
                let mut out = vec![J::from(0.0); n[s] * n[s] * n[t] * n[t]];
                for p in 0..n[s] {
                    for q in 0..n[s] {
                        for r in 0..n[t] {
                            for u in 0..n[t] {
                                let mut value = J::from(0.0);
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
    fn parity_kernel(&self, x: &[Jm; 2], odd: bool) -> [Jm; 2] {
        let k = self.rotation(x, odd);
        let dg = self.integral_rotation(&k, odd);
        let l = self.contraction(&dg, &self.amplitudes);
        std::array::from_fn(|s| {
            let n = self.n[s];
            let o = self.occupied[s];
            let df = commutator(&self.fock[s], &k[s]);
            let comm = commutator(&df, &self.density[s]);
            let mut out = Jm::zeros(n - o, o);
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
    pub fn kernels(&self, x: &[Jm; 2]) -> ([Jm; 2], [Jm; 2]) {
        let even = self.parity_kernel(x, false);
        let odd = self.parity_kernel(x, true);
        let a = std::array::from_fn(|s| combine(&even[s], &odd[s], 0.5, 0.5));
        let b = std::array::from_fn(|s| combine(&even[s], &odd[s], 0.5, -0.5));
        (a, b)
    }
    /// Correlation part of the singles commutator metric.
    pub fn metric(&self, x: &[Jm; 2]) -> [Jm; 2] {
        std::array::from_fn(|s| {
            let o = self.occupied[s];
            let n = self.n[s];
            let mut out = Jm::zeros(n - o, o);
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
    pub fn coupling_transpose(&self, x: &[Jm; 2]) -> [Vec<J>; 3] {
        let dg = self.integral_rotation(&self.rotation(x, false), false);
        self.amplitudes_from_integrals(&dg)
    }
    fn amplitudes_from_integrals(&self, g: &[Vec<J>; 3]) -> [Vec<J>; 3] {
        std::array::from_fn(|c| {
            let (s, t) = [(0, 0), (0, 1), (1, 1)][c];
            let o = self.occupied;
            let n = self.dimensions();
            let v = [n[0] - o[0], n[1] - o[1]];
            let mut out = vec![J::from(0.0); o[s] * v[s] * o[t] * v[t]];
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
                                    J::from(0.0)
                                };
                        }
                    }
                }
            }
            out
        })
    }
}

/// Moving propagator tensors in the complete spin-conserving singles space.
/// Same-spin doubles use unique antisymmetric pairs; opposite-spin doubles
/// are direct products. All arrays contain factorial-normalized coefficients.
pub struct CollinearTransitionPath {
    pub singles: Vec<BlochMatrix>,
    pub metric: Vec<BlochMatrix>,
    pub doubles: Vec<BlochMatrix>,
    pub coupling: Vec<BlochMatrix>,
    pub coupling_duality_defect: f64,
}

fn pair_index(
    o: [usize; 2],
    v: [usize; 2],
    s: usize,
    t: usize,
    i: usize,
    a: usize,
    j: usize,
    b: usize,
) -> usize {
    ((i * v[s] + a) * o[t] + j) * v[t] + b
}

impl Core {
    fn build_density(&mut self) {
        let o = self.occupied;
        let v = [self.n[0] - o[0], self.n[1] - o[1]];
        for s in 0..2 {
            for t in 0..2 {
                let factor = self.weight(s, t) * if s == t { 0.5 } else { 1.0 };
                for a in 0..v[s] {
                    for c in 0..v[s] {
                        let mut value = J::from(0.0);
                        for i in 0..o[s] {
                            for j in 0..o[t] {
                                for b in 0..v[t] {
                                    value += factor
                                        * self.t(&self.amplitudes, s, t, i, a, j, b)
                                        * self.t(&self.amplitudes, s, t, i, c, j, b);
                                }
                            }
                        }
                        self.density[s][(o[s] + a, o[s] + c)] += value;
                    }
                }
                for k in 0..o[s] {
                    for i in 0..o[s] {
                        let mut value = J::from(0.0);
                        for j in 0..o[t] {
                            for a in 0..v[s] {
                                for b in 0..v[t] {
                                    value -= factor
                                        * self.t(&self.amplitudes, s, t, i, a, j, b)
                                        * self.t(&self.amplitudes, s, t, k, a, j, b);
                                }
                            }
                        }
                        self.density[s][(k, i)] += value;
                    }
                }
            }
        }
    }
    fn double_action(&self, s: usize, t: usize, unit: &[f64]) -> Vec<J> {
        let o = self.occupied;
        let v = [self.n[0] - o[0], self.n[1] - o[1]];
        let index = |i, a, j, b| pair_index(o, v, s, t, i, a, j, b);
        let mut out = vec![J::from(0.0); unit.len()];
        for i in 0..o[s] {
            for a in 0..v[s] {
                for j in 0..o[t] {
                    for b in 0..v[t] {
                        let p = index(i, a, j, b);
                        for k in 0..o[s] {
                            out[p] -= self.fock[s][(k, i)] * unit[index(k, a, j, b)];
                        }
                        for k in 0..o[t] {
                            out[p] -= self.fock[t][(k, j)] * unit[index(i, a, k, b)];
                        }
                        for c in 0..v[s] {
                            out[p] += self.fock[s][(o[s] + a, o[s] + c)] * unit[index(i, c, j, b)];
                        }
                        for c in 0..v[t] {
                            out[p] += self.fock[t][(o[t] + b, o[t] + c)] * unit[index(i, a, j, c)];
                        }
                    }
                }
            }
        }
        out
    }
}

/// Construct the same normal-ordered full-Coulomb kernels as `Pt2Transition`,
/// including all noncanonical Fock blocks and ground-amplitude responses.
/// Coefficients must be nonnegative for this positive-metric pair basis.
pub fn collinear_transition_path(
    occupied: [usize; 2],
    fock: &[Vec<Matrix>; 2],
    integrals: &[[Vec<f64>; 3]],
    amplitudes: &[[Vec<f64>; 3]],
    coefficients: Pt2Coefficients,
    memory_mb: usize,
    tolerance: f64,
) -> Result<CollinearTransitionPath> {
    coefficients.validate()?;
    let width = integrals.len();
    if !(2..=3).contains(&width)
        || amplitudes.len() != width
        || fock.iter().any(|f| f.len() != width)
        || !tolerance.is_finite()
        || tolerance <= 0.0
    {
        return Err(DftError::InvalidInput(
            "transition paths need matching finite orders zero through one or two".into(),
        ));
    }
    if coefficients.same_spin < 0.0 || coefficients.opposite_spin < 0.0 {
        return Err(DftError::Unsupported {
            feature: "signed PT2 double-space metric along a nuclear excitation path".into(),
            missing: "an indefinite metric propagator adapter".into(),
        });
    }
    let n = [fock[0][0].rows, fock[1][0].rows];
    if occupied.iter().zip(n).any(|(&o, n)| o > n)
        || fock.iter().enumerate().any(|(s, f)| {
            f.iter().any(|m| {
                m.rows != n[s] || m.cols != n[s] || m.as_slice().iter().any(|x| !x.is_finite())
            })
        })
    {
        return Err(DftError::InvalidInput(
            "invalid transition orbital/Fock spaces".into(),
        ));
    }
    let v = [n[0] - occupied[0], n[1] - occupied[1]];
    let channels = [(0, 0), (0, 1), (1, 1)];
    // Check the actual unique-pair dimensions before allocating their list.
    let dimensions = (|| {
        let ns = occupied[0]
            .checked_mul(v[0])?
            .checked_add(occupied[1].checked_mul(v[1])?)?;
        let choose_two = |n: usize| n.checked_mul(n.saturating_sub(1)).map(|x| x / 2);
        let mut nd = 0usize;
        if coefficients.same_spin != 0.0 {
            for s in 0..2 {
                nd = nd.checked_add(choose_two(occupied[s])?.checked_mul(choose_two(v[s])?)?)?;
            }
        }
        if coefficients.opposite_spin != 0.0 {
            nd = nd.checked_add(
                occupied[0]
                    .checked_mul(v[0])?
                    .checked_mul(occupied[1])?
                    .checked_mul(v[1])?,
            )?;
        }
        let tensors = n.iter().try_fold(0usize, |sum, n| {
            n.checked_pow(4)?.checked_mul(256)?.checked_add(sum)
        })?;
        let dense = ns.checked_add(nd)?.checked_pow(2)?.checked_mul(512)?;
        Some((ns, nd, tensors.checked_add(dense)?))
    })()
    .ok_or_else(|| DftError::InvalidInput("transition path workspace overflow".into()))?;
    let early_required_mb = dimensions.2.div_ceil(1024 * 1024);
    if early_required_mb > memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "moving collinear propagator tensors",
            required_mb: early_required_mb,
            limit_mb: memory_mb,
        });
    }
    let mut pairs = Vec::new();
    for (c, (s, t)) in channels.into_iter().enumerate() {
        let w = if s == t {
            coefficients.same_spin
        } else {
            coefficients.opposite_spin
        };
        if w == 0.0 {
            continue;
        }
        for i in 0..occupied[s] {
            for a in 0..v[s] {
                for j in 0..occupied[t] {
                    for b in 0..v[t] {
                        if s != t || (i < j && a < b) {
                            pairs.push((c, s, t, i, a, j, b, w));
                        }
                    }
                }
            }
        }
    }
    let singles: Vec<_> = (0..2)
        .flat_map(|s| (0..v[s]).flat_map(move |a| (0..occupied[s]).map(move |i| (s, a, i))))
        .collect();
    let ns = singles.len();
    let nd = pairs.len();
    let bytes = n
        .iter()
        .try_fold(0usize, |sum, n| {
            n.checked_pow(4)
                .and_then(|b| b.checked_mul(256))
                .and_then(|b| sum.checked_add(b))
        })
        .and_then(|b| {
            ns.checked_add(nd)
                .and_then(|d| d.checked_mul(d))
                .and_then(|d| d.checked_mul(512))
                .and_then(|d| b.checked_add(d))
        })
        .ok_or_else(|| DftError::InvalidInput("transition path workspace overflow".into()))?;
    let required_mb = bytes.div_ceil(1024 * 1024);
    if required_mb > memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "moving collinear propagator tensors",
            required_mb,
            limit_mb: memory_mb,
        });
    }
    for (k, g) in integrals.iter().enumerate() {
        for (c, (s, t)) in channels.into_iter().enumerate() {
            let expected = n[s] * n[s] * n[t] * n[t];
            let expected_t = occupied[s] * v[s] * occupied[t] * v[t];
            if g[c].len() != expected
                || amplitudes[k][c].len() != expected_t
                || g[c].iter().chain(&amplitudes[k][c]).any(|x| !x.is_finite())
            {
                return Err(DftError::InvalidInput(
                    "invalid full MO integral or ground-amplitude coefficient dimensions".into(),
                ));
            }
        }
    }
    let jet = |values: Vec<f64>| J(Taylor::from_coefficients(&values));
    let mut core = Core {
        n,
        occupied,
        coefficients,
        fock: std::array::from_fn(|s| {
            let mut m = Jm::zeros(n[s], n[s]);
            for i in 0..n[s] {
                for j in 0..n[s] {
                    m[(i, j)] = jet((0..width).map(|k| fock[s][k][(i, j)]).collect());
                }
            }
            m
        }),
        integrals: std::array::from_fn(|c| {
            (0..integrals[0][c].len())
                .map(|i| jet((0..width).map(|k| integrals[k][c][i]).collect()))
                .collect()
        }),
        amplitudes: std::array::from_fn(|c| {
            (0..amplitudes[0][c].len())
                .map(|i| jet((0..width).map(|k| amplitudes[k][c][i]).collect()))
                .collect()
        }),
        density: std::array::from_fn(|s| Jm::zeros(n[s], n[s])),
    };
    core.build_density();
    let mut out = CollinearTransitionPath {
        singles: vec![BlochMatrix::zeros(ns, ns); width],
        metric: vec![BlochMatrix::zeros(ns, ns); width],
        doubles: vec![BlochMatrix::zeros(nd, nd); width],
        coupling: vec![BlochMatrix::zeros(ns, nd); width],
        coupling_duality_defect: 0.0,
    };
    for (col, &(s, a, i)) in singles.iter().enumerate() {
        let mut x = std::array::from_fn(|s| Jm::zeros(v[s], occupied[s]));
        x[s][(a, i)] = J::from(1.0);
        let (a, _) = core.kernels(&x);
        let m = core.metric(&x);
        let coupling = core.coupling_transpose(&x);
        for (row, &(s, b, j)) in singles.iter().enumerate() {
            for k in 0..width {
                out.singles[k].re[(row, col)] = a[s][(b, j)].0.c[k];
                out.metric[k].re[(row, col)] = m[s][(b, j)].0.c[k];
            }
        }
        for (d, &(c, s, t, i, a, j, b, w)) in pairs.iter().enumerate() {
            let p = pair_index(occupied, v, s, t, i, a, j, b);
            for k in 0..width {
                out.coupling[k].re[(col, d)] = w.sqrt() * coupling[c][p].0.c[k];
            }
        }
    }
    for (col, &(c, s, t, i, a, j, b, w)) in pairs.iter().enumerate() {
        let mut unit = vec![0.0; amplitudes[0][c].len()];
        let ix = |i, a, j, b| pair_index(occupied, v, s, t, i, a, j, b);
        unit[ix(i, a, j, b)] = 1.0;
        if s == t {
            unit[ix(j, b, i, a)] = 1.0;
            unit[ix(i, b, j, a)] = -1.0;
            unit[ix(j, a, i, b)] = -1.0;
        }
        let d = core.double_action(s, t, &unit);
        for (row, &(cc, ss, tt, ii, aa, jj, bb, _)) in pairs.iter().enumerate() {
            if cc == c {
                for k in 0..width {
                    out.doubles[k].re[(row, col)] =
                        d[pair_index(occupied, v, ss, tt, ii, aa, jj, bb)].0.c[k];
                }
            }
        }
        let mut redundant: [Vec<J>; 3] =
            std::array::from_fn(|c| vec![J::from(0.0); amplitudes[0][c].len()]);
        for (target, value) in redundant[c].iter_mut().zip(&unit) {
            *target = J::from(*value / w.sqrt());
        }
        let right = core.coupling(&redundant);
        for (row, &(s, a, i)) in singles.iter().enumerate() {
            for k in 0..width {
                out.coupling_duality_defect = out
                    .coupling_duality_defect
                    .max((right[s][(a, i)].0.c[k] - out.coupling[k].re[(row, col)]).abs());
            }
        }
    }
    if !out.coupling_duality_defect.is_finite() || out.coupling_duality_defect > tolerance {
        return Err(DftError::ResponseNotConverged {
            iterations: width - 1,
            residual: out.coupling_duality_defect,
        });
    }
    Ok(out)
}
