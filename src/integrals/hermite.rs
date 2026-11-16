// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Hermite-Gaussian machinery of the McMurchie–Davidson scheme:
//! E-coefficient tables and the flat-array Hermite–Coulomb R-tensor.
//!
//! Conventions: primitive pair with exponents `a` on center A and `b` on B,
//! `p = a + b`, `μ = ab/p`, `q = A − B` (per axis), product center
//! `P = (aA + bB)/p`, `X_PA = −(b/p)q`, `X_PB = (a/p)q`.

use crate::math::Vec3;

/// Table of Hermite expansion coefficients E^{ij}_t for one axis, all
/// `i ≤ imax`, `j ≤ jmax`, `t ≤ i + j`. Built once per primitive pair and
/// axis by the standard two-index recursion; `E(0,0,0)` carries the
/// Gaussian-product prefactor `exp(−μ q²)`.
#[derive(Clone, Debug)]
pub struct ETable {
    imax: usize,
    jmax: usize,
    tdim: usize,
    data: Vec<f64>,
}

impl ETable {
    pub fn build(imax: usize, jmax: usize, q: f64, a: f64, b: f64) -> Self {
        let p = a + b;
        let mu = a * b / p;
        let x_pa = -(b / p) * q;
        let x_pb = (a / p) * q;
        let tdim = imax + jmax + 1;
        let mut t_of = vec![0.0; (imax + 1) * (jmax + 1) * tdim];
        let idx = |i: usize, j: usize, t: usize| (i * (jmax + 1) + j) * tdim + t;
        t_of[idx(0, 0, 0)] = (-mu * q * q).exp();
        // Raise i at j = 0.
        for i in 0..imax {
            for t in 0..=(i + 1) {
                let mut v = x_pa * t_of[idx(i, 0, t)];
                if t > 0 {
                    v += t_of[idx(i, 0, t - 1)] / (2.0 * p);
                }
                if t < i {
                    v += (t + 1) as f64 * t_of[idx(i, 0, t + 1)];
                }
                t_of[idx(i + 1, 0, t)] = v;
            }
        }
        // Raise j at every i.
        for i in 0..=imax {
            for j in 0..jmax {
                for t in 0..=(i + j + 1) {
                    let mut v = x_pb * t_of[idx(i, j, t)];
                    if t > 0 {
                        v += t_of[idx(i, j, t - 1)] / (2.0 * p);
                    }
                    if t < i + j {
                        v += (t + 1) as f64 * t_of[idx(i, j, t + 1)];
                    }
                    t_of[idx(i, j + 1, t)] = v;
                }
            }
        }
        Self {
            imax,
            jmax,
            tdim,
            data: t_of,
        }
    }

    /// E^{ij}_t (0 outside the valid range).
    #[inline]
    pub fn get(&self, i: usize, j: usize, t: usize) -> f64 {
        if i > self.imax || j > self.jmax || t > i + j {
            return 0.0;
        }
        self.data[(i * (self.jmax + 1) + j) * self.tdim + t]
    }
}

/// Flat dense Hermite–Coulomb tensor R⁰_{tuv}, valid for `t+u+v ≤ l`.
///
/// Built iteratively (no recursion, no hashing): starting from the base
/// values `R^n_{000} = (−2p)^n f[n]` for the kernel-adjusted Boys values
/// `f`, the auxiliary index n is peeled down while the (t,u,v) range grows.
#[derive(Clone, Debug)]
pub struct RTensor {
    l: usize,
    dim: usize,
    data: Vec<f64>,
}

impl RTensor {
    /// `l`: maximum total Hermite order; `p`: composite exponent entering
    /// the base factor `(−2p)^n`; `pc`: P − C (or P − Q); `f`: Boys values
    /// `f[0..=l]` already adjusted for the operator kernel.
    pub fn build(l: usize, p: f64, pc: Vec3, f: &[f64]) -> Self {
        let dim = l + 1;
        let cube = dim * dim * dim;
        let idx = |t: usize, u: usize, v: usize| (t * dim + u) * dim + v;
        let mut cur = vec![0.0; cube];
        let mut next = vec![0.0; cube];
        // n = l level: only (0,0,0).
        cur[idx(0, 0, 0)] = (-2.0 * p).powi(l as i32) * f[l];
        for n in (0..l).rev() {
            let reach = l - n;
            // Overwrite the complete valid simplex below. Each read of cur
            // has total order at most reach - 1, written by the preceding
            // level; clearing the rest of the cube cannot affect the result.
            next[idx(0, 0, 0)] = (-2.0 * p).powi(n as i32) * f[n];
            for t in 0..=reach {
                for u in 0..=(reach - t) {
                    for v in 0..=(reach - t - u) {
                        if t + u + v == 0 {
                            continue;
                        }
                        // Reduce along the highest nonzero index.
                        let val = if t > 0 {
                            let mut x = pc.x * cur[idx(t - 1, u, v)];
                            if t > 1 {
                                x += (t - 1) as f64 * cur[idx(t - 2, u, v)];
                            }
                            x
                        } else if u > 0 {
                            let mut x = pc.y * cur[idx(t, u - 1, v)];
                            if u > 1 {
                                x += (u - 1) as f64 * cur[idx(t, u - 2, v)];
                            }
                            x
                        } else {
                            let mut x = pc.z * cur[idx(t, u, v - 1)];
                            if v > 1 {
                                x += (v - 1) as f64 * cur[idx(t, u, v - 2)];
                            }
                            x
                        };
                        next[idx(t, u, v)] = val;
                    }
                }
            }
            std::mem::swap(&mut cur, &mut next);
        }
        Self { l, dim, data: cur }
    }

    /// R⁰_{tuv} (caller must respect `t+u+v ≤ l`).
    #[inline]
    pub fn get(&self, t: usize, u: usize, v: usize) -> f64 {
        debug_assert!(t + u + v <= self.l);
        self.data[(t * self.dim + u) * self.dim + v]
    }

    #[inline]
    pub fn max_order(&self) -> usize {
        self.l
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::boys::boys;

    /// Slow reference E by the textbook per-call recursion.
    fn e_ref(i: i32, j: i32, t: i32, q: f64, a: f64, b: f64) -> f64 {
        let p = a + b;
        let mu = a * b / p;
        if t < 0 || t > i + j || i < 0 || j < 0 {
            return 0.0;
        }
        if i == 0 && j == 0 && t == 0 {
            return (-mu * q * q).exp();
        }
        if i > 0 {
            e_ref(i - 1, j, t - 1, q, a, b) / (2.0 * p) - (b / p) * q * e_ref(i - 1, j, t, q, a, b)
                + (t + 1) as f64 * e_ref(i - 1, j, t + 1, q, a, b)
        } else {
            e_ref(i, j - 1, t - 1, q, a, b) / (2.0 * p)
                + (a / p) * q * e_ref(i, j - 1, t, q, a, b)
                + (t + 1) as f64 * e_ref(i, j - 1, t + 1, q, a, b)
        }
    }

    /// Slow reference R by memo-free recursion.
    fn r_ref(t: i32, u: i32, v: i32, n: i32, p: f64, pc: Vec3, f: &[f64]) -> f64 {
        if t < 0 || u < 0 || v < 0 {
            return 0.0;
        }
        if t == 0 && u == 0 && v == 0 {
            return (-2.0 * p).powi(n) * f[n as usize];
        }
        if t > 0 {
            (t - 1) as f64 * r_ref(t - 2, u, v, n + 1, p, pc, f)
                + pc.x * r_ref(t - 1, u, v, n + 1, p, pc, f)
        } else if u > 0 {
            (u - 1) as f64 * r_ref(t, u - 2, v, n + 1, p, pc, f)
                + pc.y * r_ref(t, u - 1, v, n + 1, p, pc, f)
        } else {
            (v - 1) as f64 * r_ref(t, u, v - 2, n + 1, p, pc, f)
                + pc.z * r_ref(t, u, v - 1, n + 1, p, pc, f)
        }
    }

    #[test]
    fn e_table_matches_reference_recursion() {
        let (a, b, q) = (1.3, 0.6, 0.9);
        let table = ETable::build(4, 3, q, a, b);
        for i in 0..=4i32 {
            for j in 0..=3i32 {
                for t in 0..=(i + j) {
                    let want = e_ref(i, j, t, q, a, b);
                    let got = table.get(i as usize, j as usize, t as usize);
                    assert!(
                        (got - want).abs() < 1e-13 * want.abs().max(1.0),
                        "E({i},{j},{t}): {got} vs {want}"
                    );
                }
            }
        }
    }

    /// Σ_i E^{i0}_0 must reproduce the Gaussian product integral identity:
    /// E^{00}_0 = exp(−μq²) exactly.
    #[test]
    fn e_base_case_prefactor() {
        let (a, b, q) = (0.7, 1.9, -1.3);
        let table = ETable::build(0, 0, q, a, b);
        let mu = a * b / (a + b);
        assert!((table.get(0, 0, 0) - (-mu * q * q).exp()).abs() < 1e-16);
    }

    #[test]
    fn r_tensor_matches_reference_recursion() {
        let l = 6;
        let p = 1.7;
        let pc = Vec3::new(0.3, -0.8, 0.5);
        let f = boys(l, p * pc.norm2());
        let rt = RTensor::build(l, p, pc, &f);
        for t in 0..=l {
            for u in 0..=(l - t) {
                for v in 0..=(l - t - u) {
                    let want = r_ref(t as i32, u as i32, v as i32, 0, p, pc, &f);
                    let got = rt.get(t, u, v);
                    assert!(
                        (got - want).abs() < 1e-12 * want.abs().max(1.0),
                        "R({t},{u},{v}): {got} vs {want}"
                    );
                }
            }
        }
    }

    /// Memoized recursive oracle for every valid entry, including the origin,
    /// co-centered pairs and orders above ordinary spdf quartets. Preserve the
    /// conventional axis priority: changing it at high order compares distinct
    /// cancellation-sensitive floating-point recurrences, rather than validating
    /// the iterative storage sweep against the recursive construction.
    #[test]
    fn r_tensor_all_entries_across_orders_and_centers() {
        use std::collections::HashMap;

        fn reference(
            orders: [usize; 4],
            p: f64,
            pc: Vec3,
            f: &[f64],
            memo: &mut HashMap<[usize; 4], f64>,
        ) -> f64 {
            if let Some(&value) = memo.get(&orders) {
                return value;
            }
            let [t, u, v, n] = orders;
            let value = if t + u + v == 0 {
                (-2.0 * p).powi(n as i32) * f[n]
            } else {
                let axis = if t > 0 {
                    0
                } else if u > 0 {
                    1
                } else {
                    2
                };
                let mut lower = orders;
                lower[axis] -= 1;
                lower[3] += 1;
                let first = pc.get(axis) * reference(lower, p, pc, f, memo);
                if orders[axis] > 1 {
                    lower[axis] -= 1;
                    first + (orders[axis] - 1) as f64 * reference(lower, p, pc, f, memo)
                } else {
                    first
                }
            };
            memo.insert(orders, value);
            value
        }

        for l in [0, 1, 2, 4, 6, 8, 12, 16] {
            for (p, pc) in [
                (0.013, Vec3::zero()),
                (0.17, Vec3::new(0.31, -0.83, 0.57)),
                (1.7, Vec3::new(-0.21, 0.73, -0.46)),
                (3.4, Vec3::new(1.13, -0.37, 0.91)),
            ] {
                let f = boys(l, p * pc.norm2());
                let rt = RTensor::build(l, p, pc, &f);
                let mut memo = HashMap::new();
                for t in 0..=l {
                    for u in 0..=l - t {
                        for v in 0..=l - t - u {
                            let want = reference([t, u, v, 0], p, pc, &f, &mut memo);
                            let got = rt.get(t, u, v);
                            assert!(
                                (got - want).abs() < 1e-12 * want.abs().max(1.0),
                                "l={l}, p={p}, pc={pc:?}, R({t},{u},{v}): {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }
}
