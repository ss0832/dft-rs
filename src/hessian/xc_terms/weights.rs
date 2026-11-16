// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Cached owner-riding Becke derivatives shared by molecular Hessians.
use crate::dual::{DualN, Scalar};
use crate::grid::becke_weight_gradient;
use crate::math::Vec3;

/// Scalar-generic Becke weight (k = 3, size-adjusted) - mirror of
/// `grid::becke_weight`, evaluated on dual numbers for derivatives.
fn becke_weight_s<S: Scalar>(centers: &[[S; 3]], aij: &[f64], ia: usize, p: [S; 3]) -> S {
    let n_at = centers.len();
    if n_at == 1 {
        return S::from_f64(1.0);
    }
    let dist: Vec<S> = centers
        .iter()
        .map(|c| {
            let dx = p[0] - c[0];
            let dy = p[1] - c[1];
            let dz = p[2] - c[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();
    let mut num = S::from_f64(0.0);
    let mut den = S::from_f64(0.0);
    for b in 0..n_at {
        let mut cell = S::from_f64(1.0);
        for c in 0..n_at {
            if c == b {
                continue;
            }
            let dx = centers[b][0] - centers[c][0];
            let dy = centers[b][1] - centers[c][1];
            let dz = centers[b][2] - centers[c][2];
            let rbc = (dx * dx + dy * dy + dz * dz).sqrt();
            cell = cell * partition_factor(dist[b], dist[c], rbc, aij[b * n_at + c]);
        }
        if b == ia {
            num = cell;
        }
        den = den + cell;
    }
    if den.value() > 0.0 {
        num / den
    } else {
        S::from_f64(0.0)
    }
}

type D2 = DualN<DualN<f64, 3>, 3>;

/// Coordinate seeded for the nested second derivative: `outer`/`inner`
/// select which axis slot (if any) this coordinate occupies.
fn seed(val: f64, outer: Option<usize>, inner: Option<usize>) -> D2 {
    let re = match inner {
        Some(k) => DualN::<f64, 3>::variable(val, k),
        None => DualN::<f64, 3>::constant(val),
    };
    let mut d = D2 {
        re,
        im: [DualN::<f64, 3>::constant(0.0); 3],
    };
    if let Some(k) = outer {
        d.im[k] = DualN::<f64, 3>::constant(1.0);
    }
    d
}

/// Full 3x3 block of total second derivatives `D_A D_B w` for one grid
/// point (owner-riding convention: seeding atom G also seeds the point
/// when G is the owner).
pub fn becke_weight_second(
    centers: &[Vec3],
    aij: &[f64],
    owner: usize,
    a: usize,
    b: usize,
    p: Vec3,
) -> [[f64; 3]; 3] {
    let n_at = centers.len();
    let cs: Vec<[D2; 3]> = (0..n_at)
        .map(|g| {
            let c = centers[g];
            [
                seed(c.x, (g == a).then_some(0), (g == b).then_some(0)),
                seed(c.y, (g == a).then_some(1), (g == b).then_some(1)),
                seed(c.z, (g == a).then_some(2), (g == b).then_some(2)),
            ]
        })
        .collect();
    let pp = [
        seed(p.x, (owner == a).then_some(0), (owner == b).then_some(0)),
        seed(p.y, (owner == a).then_some(1), (owner == b).then_some(1)),
        seed(p.z, (owner == a).then_some(2), (owner == b).then_some(2)),
    ];
    let w = becke_weight_s(&cs, aij, owner, pp);
    let mut out = [[0.0; 3]; 3];
    for (x, row) in out.iter_mut().enumerate() {
        for (y, v) in row.iter_mut().enumerate() {
            *v = w.im[x].im[y];
        }
    }
    out
}

/// A Becke pair factor depends on only its two centers at a fixed point.
/// Cache all six-coordinate derivatives once per ordered pair and reuse
/// them in every nuclear Hessian block. Products never divide by a cell
/// factor, so an exactly zero switching factor remains well defined.
#[derive(Clone, Copy)]
struct PairFactor {
    value: f64,
    gradient: [f64; 6],
    hessian: [[f64; 6]; 6],
}

impl PairFactor {
    fn evaluate(left: Vec3, right: Vec3, point: Vec3, adjustment: f64) -> Self {
        type PairDual = DualN<DualN<f64, 6>, 6>;
        let variable = |value, axis| {
            let mut out = PairDual::constant(DualN::variable(value, axis));
            out.im[axis] = DualN::constant(1.0);
            out
        };
        let l: [PairDual; 3] = std::array::from_fn(|k| variable(left.get(k), k));
        let r: [PairDual; 3] = std::array::from_fn(|k| variable(right.get(k), k + 3));
        let p: [PairDual; 3] = std::array::from_fn(|k| PairDual::from_f64(point.get(k)));
        let norm = |v: [PairDual; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let dl = norm(std::array::from_fn(|k| p[k] - l[k]));
        let dr = norm(std::array::from_fn(|k| p[k] - r[k]));
        let distance = norm(std::array::from_fn(|k| l[k] - r[k]));
        let factor = partition_factor(dl, dr, distance, adjustment);
        Self {
            value: factor.re.re,
            gradient: factor.re.im,
            hessian: std::array::from_fn(|i| factor.im[i].im),
        }
    }

    fn block(self, left: usize, right: usize, outer: usize, inner: usize) -> D2 {
        let local = |atom| {
            if atom == left {
                Some(0)
            } else if atom == right {
                Some(3)
            } else {
                None
            }
        };
        let mut out = D2::from_f64(self.value);
        if let Some(j) = local(inner) {
            for y in 0..3 {
                out.re.im[y] = self.gradient[j + y];
            }
        }
        if let Some(i) = local(outer) {
            for x in 0..3 {
                out.im[x].re = self.gradient[i + x];
                if let Some(j) = local(inner) {
                    for y in 0..3 {
                        out.im[x].im[y] = self.hessian[i + x][j + y];
                    }
                }
            }
        }
        out
    }
}

/// Shared size-adjusted, three-times-smoothed Becke switching polynomial.
fn partition_factor<S: Scalar>(left: S, right: S, distance: S, adjustment: f64) -> S {
    let mu = (left - right) / distance;
    let one = S::from_f64(1.0);
    let mut f = mu + S::from_f64(adjustment) * (one - mu * mu);
    for _ in 0..3 {
        f = S::from_f64(1.5) * f - S::from_f64(0.5) * f * f * f;
    }
    S::from_f64(0.5) * (one - f)
}

/// Reconstruct a non-owner block from cached pair derivatives. Only factors
/// incident on a seeded atom need dual multiplication; other factors are
/// folded into a plain scalar. The owner's blocks follow by translation.
fn becke_weight_second_sparse(
    factors: &[PairFactor],
    natoms: usize,
    owner: usize,
    a: usize,
    b: usize,
) -> [[f64; 3]; 3] {
    let mut num = D2::from_f64(0.0);
    let mut den = D2::from_f64(0.0);
    for left in 0..natoms {
        let mut constant = 1.0;
        let mut dual = D2::from_f64(1.0);
        for right in 0..natoms {
            if right == left {
                continue;
            }
            let factor = factors[left * natoms + right];
            if left != a && left != b && right != a && right != b {
                constant *= factor.value;
            } else {
                dual = dual * factor.block(left, right, a, b);
            }
        }
        let cell = D2::from_f64(constant) * dual;
        if left == owner {
            num = cell;
        }
        den = den + cell;
    }
    let w = if den.value() > 0.0 {
        num / den
    } else {
        D2::from_f64(0.0)
    };
    std::array::from_fn(|x| w.im[x].im)
}

/// Total (owner-riding) partition-weight derivatives of one grid point:
/// `(D_A w, D_A D_B w)` over the `3N` Hessian coordinates, row-major.
pub(crate) trait WeightResponse: Sync {
    fn derivatives(&self, point: Vec3, owner: usize) -> (Vec<f64>, Vec<f64>);
}

/// Molecular Becke weights: the fixed-point gradient with the owner-riding
/// point, and the sparse nested-dual second derivatives of every pair not
/// involving the owner, whose row and column follow from translational
/// invariance.
pub(crate) struct MolecularWeights<'a> {
    pub(crate) centers: &'a [Vec3],
    pub(crate) aij: &'a [f64],
}

impl WeightResponse for MolecularWeights<'_> {
    fn derivatives(&self, point: Vec3, owner: usize) -> (Vec<f64>, Vec<f64>) {
        let natoms = self.centers.len();
        let dim = 3 * natoms;
        let (_, dwb) = becke_weight_gradient(self.centers, self.aij, natoms, owner, point);
        let mut sum = Vec3::zero();
        for v in &dwb {
            sum += *v;
        }
        let mut first = vec![0.0; dim];
        for (at, v) in dwb.iter().enumerate() {
            for ax in 0..3 {
                first[3 * at + ax] = v.get(ax) - if at == owner { sum.get(ax) } else { 0.0 };
            }
        }
        let zero = PairFactor {
            value: 0.0,
            gradient: [0.0; 6],
            hessian: [[0.0; 6]; 6],
        };
        let mut factors = vec![zero; natoms * natoms];
        for left in 0..natoms {
            for right in 0..natoms {
                if left != right {
                    factors[left * natoms + right] = PairFactor::evaluate(
                        self.centers[left],
                        self.centers[right],
                        point,
                        self.aij[left * natoms + right],
                    );
                }
            }
        }
        let mut ddw = vec![vec![[[0.0; 3]; 3]; natoms]; natoms];
        for aa in 0..natoms {
            if aa == owner {
                continue;
            }
            for bb in aa..natoms {
                if bb == owner {
                    continue;
                }
                let blk = becke_weight_second_sparse(&factors, natoms, owner, aa, bb);
                ddw[aa][bb] = blk;
                if aa != bb {
                    for (x, r) in blk.iter().enumerate() {
                        for (y, v) in r.iter().enumerate() {
                            ddw[bb][aa][y][x] = *v;
                        }
                    }
                }
            }
        }
        // Owner row/column, then the owner-owner block.
        for bb in 0..natoms {
            if bb == owner {
                continue;
            }
            let mut acc = [[0.0; 3]; 3];
            for aa in 0..natoms {
                if aa == owner {
                    continue;
                }
                for x in 0..3 {
                    for y in 0..3 {
                        acc[x][y] -= ddw[aa][bb][x][y];
                    }
                }
            }
            ddw[owner][bb] = acc;
            for x in 0..3 {
                for y in 0..3 {
                    ddw[bb][owner][y][x] = acc[x][y];
                }
            }
        }
        let mut acc = [[0.0; 3]; 3];
        for bb in 0..natoms {
            if bb == owner {
                continue;
            }
            for x in 0..3 {
                for y in 0..3 {
                    acc[x][y] -= ddw[owner][bb][x][y];
                }
            }
        }
        ddw[owner][owner] = acc;
        let mut second = vec![0.0; dim * dim];
        for aa in 0..natoms {
            for bb in 0..natoms {
                for x in 0..3 {
                    for y in 0..3 {
                        second[(3 * aa + x) * dim + 3 * bb + y] = ddw[aa][bb][x][y];
                    }
                }
            }
        }
        (first, second)
    }
}

#[cfg(test)]
#[path = "weight_tests.rs"]
mod tests;
