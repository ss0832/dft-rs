// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Natural resonance theory: the density as a convex combination of the
//! idealized densities of Lewis structures.
//!
//! Candidate structures keep the sigma skeleton of the natural Lewis
//! structure (every bonded pair keeps at least one bond) and distribute the
//! remaining valence electron pairs (single electrons for a spin density)
//! over lone pairs and further bond units between pairs bonded in the
//! reference or with a Wiberg index of at least 0.1, with at most four
//! valence units per atom (one for H and He) and bond orders up to three.
//! The idealized density of a structure is `q sum_i |i><i|` over the Lewis
//! orbitals of that prescribed structure (leading eigenvectors of the
//! depleted density blocks, orthogonalized as hybrids; `q` the maximum
//! occupancy). The structures with the smallest single-structure deviation
//! `||D - Gamma_alpha||` (the reference always included) receive the weights
//! that minimize `||D - sum_alpha w_alpha Gamma_alpha||_F` with `w >= 0` and
//! `sum w = 1`, solved by an active-set method on the simplex. NRT bond
//! orders are `sum_alpha w_alpha b_AB^alpha`, valencies their row sums and
//! formal charges `Z_val - q n_A - (q/2) sum_B b_AB` for the spin-summed
//! density.

use super::lewis::prescribed_lewis;
use super::{NaoKind, NaturalAtomicOrbitals, NboChannel, NboKind};
use crate::error::{DftError, Result};
use crate::linalg::{symmetric_eigen, Matrix};

/// One Lewis structure of the resonance hybrid.
#[derive(Clone, Debug)]
pub struct ResonanceStructure {
    /// Lone pairs (single electrons for a spin density) per atom.
    pub lone_pairs: Vec<usize>,
    /// `(a, b, order)` with `a < b`.
    pub bonds: Vec<(usize, usize, usize)>,
    pub weight: f64,
    /// `||D - Gamma||_F / ||D||_F` of this structure alone.
    pub deviation: f64,
    /// Formal charges (spin-summed density only).
    pub formal_charges: Option<Vec<f64>>,
}

/// Resonance weights and NRT bond orders of one density.
#[derive(Clone, Debug)]
pub struct Resonance {
    /// Structures with a weight above 1e-6, descending weight.
    pub structures: Vec<ResonanceStructure>,
    /// `sum_alpha w_alpha b_AB^alpha` in bond units of the density.
    pub bond_orders: Matrix,
    pub valencies: Vec<f64>,
    /// `||D - sum w Gamma||_F / ||D||_F`.
    pub fractional_deviation: f64,
    /// The same of the reference structure alone.
    pub reference_deviation: f64,
    pub enumerated: usize,
    pub evaluated: usize,
    /// Structures whose Lewis orbitals could not be formed (dependent
    /// hybrids), left out.
    pub skipped: usize,
    /// True when the enumeration stopped at its limit.
    pub truncated: bool,
    /// Largest violation of the optimality conditions of the weights.
    pub optimality_residual: f64,
    /// Linearly dependent combinations of the idealized densities (Gram
    /// eigenvalues below 1e-10 of the largest): weights differing along
    /// them fit equally well, and the active set returns the minimum-norm
    /// weights of its face.
    pub degenerate_directions: usize,
}

#[derive(Clone, PartialEq, Eq)]
struct Assignment {
    lone: Vec<usize>,
    /// Orders of the allowed pairs.
    orders: Vec<usize>,
}

/// Largest number of enumerated structures.
const ENUMERATION_LIMIT: usize = 20000;
/// Largest number of structures whose idealized density is built.
const EVALUATION_LIMIT: usize = 400;

struct Enumerator<'a> {
    pairs: &'a [(usize, usize)],
    minimum: &'a [usize],
    capacity: &'a [usize],
    out: Vec<Assignment>,
    truncated: bool,
}

impl Enumerator<'_> {
    fn pairs(
        &mut self,
        index: usize,
        used: &mut [usize],
        orders: &mut Vec<usize>,
        remaining: usize,
    ) {
        if self.truncated {
            return;
        }
        if index == self.pairs.len() {
            let free: usize = self
                .capacity
                .iter()
                .zip(used.iter())
                .map(|(c, u)| c - u)
                .sum();
            if remaining <= free {
                let mut lone = vec![0; self.capacity.len()];
                self.lone(0, used, &mut lone, orders, remaining);
            }
            return;
        }
        let (a, b) = self.pairs[index];
        for extra in 0..=(3 - self.minimum[index]) {
            if extra > remaining {
                break;
            }
            let order = self.minimum[index] + extra;
            if extra > 0
                && (used[a] + extra > self.capacity[a] || used[b] + extra > self.capacity[b])
            {
                break;
            }
            used[a] += extra;
            used[b] += extra;
            orders.push(order);
            self.pairs(index + 1, used, orders, remaining - extra);
            orders.pop();
            used[a] -= extra;
            used[b] -= extra;
        }
    }

    fn lone(
        &mut self,
        atom: usize,
        used: &[usize],
        lone: &mut [usize],
        orders: &[usize],
        remaining: usize,
    ) {
        if self.truncated {
            return;
        }
        if atom == self.capacity.len() {
            if remaining == 0 {
                if self.out.len() >= ENUMERATION_LIMIT {
                    self.truncated = true;
                    return;
                }
                self.out.push(Assignment {
                    lone: lone.to_vec(),
                    orders: orders.to_vec(),
                });
            }
            return;
        }
        let room = self.capacity[atom] - used[atom];
        let later: usize = (atom + 1..self.capacity.len())
            .map(|b| self.capacity[b] - used[b])
            .sum();
        for count in 0..=room.min(remaining) {
            if remaining - count > later {
                continue;
            }
            lone[atom] = count;
            self.lone(atom + 1, used, lone, orders, remaining - count);
        }
        lone[atom] = 0;
    }
}

fn frobenius(a: &Matrix) -> f64 {
    a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Solves `[G_FF 1; 1^T 0] [w; nu] = [c_F; 1]` by a pseudo-inverse.
fn kkt_solve(g: &Matrix, c: &[f64], free: &[usize]) -> Result<(Vec<f64>, f64)> {
    let k = free.len();
    let mut a = Matrix::zeros(k + 1, k + 1);
    let mut rhs = vec![0.0; k + 1];
    for (i, &p) in free.iter().enumerate() {
        for (j, &q) in free.iter().enumerate() {
            a[(i, j)] = g[(p, q)];
        }
        a[(i, k)] = 1.0;
        a[(k, i)] = 1.0;
        rhs[i] = c[p];
    }
    rhs[k] = 1.0;
    let (values, vectors) = symmetric_eigen(&a)?;
    let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut x = vec![0.0; k + 1];
    for (m, &value) in values.iter().enumerate() {
        if value.abs() <= 1e-12 * largest {
            continue;
        }
        let projection: f64 = (0..=k).map(|i| vectors[(i, m)] * rhs[i]).sum();
        for (i, xi) in x.iter_mut().enumerate() {
            *xi += vectors[(i, m)] * projection / value;
        }
    }
    let nu = x[k];
    x.truncate(k);
    Ok((x, nu))
}

/// `argmin 1/2 w^T G w - c^T w` on the simplex by an active-set method,
/// started at the vertex `start`; returns the weights and the largest
/// violation of the optimality conditions.
pub(super) fn simplex_least_squares(
    g: &Matrix,
    c: &[f64],
    start: usize,
) -> Result<(Vec<f64>, f64)> {
    let m = c.len();
    let mut w = vec![0.0; m];
    w[start] = 1.0;
    let mut free = vec![start];
    for _ in 0..(20 * m + 100) {
        let (target, nu) = kkt_solve(g, c, &free)?;
        if target.iter().all(|&x| x >= -1e-14) {
            w = vec![0.0; m];
            for (&p, &x) in free.iter().zip(&target) {
                w[p] = x.max(0.0);
            }
            let gradient: Vec<f64> = (0..m)
                .map(|i| (0..m).map(|j| g[(i, j)] * w[j]).sum::<f64>() - c[i] + nu)
                .collect();
            let worst = (0..m)
                .filter(|i| !free.contains(i))
                .min_by(|&i, &j| gradient[i].total_cmp(&gradient[j]));
            match worst {
                Some(j) if gradient[j] < -1e-12 => free.push(j),
                _ => {
                    let sum: f64 = w.iter().sum();
                    let residual = (0..m)
                        .filter(|i| !free.contains(i))
                        .map(|i| (-gradient[i]).max(0.0))
                        .fold((sum - 1.0).abs(), f64::max);
                    return Ok((w, residual));
                }
            }
        } else {
            // Step towards the target until the first weight vanishes.
            let mut step = 1.0_f64;
            for (&p, &x) in free.iter().zip(&target) {
                if x < 0.0 {
                    step = step.min(w[p] / (w[p] - x));
                }
            }
            for (&p, &x) in free.iter().zip(&target) {
                w[p] += step * (x - w[p]);
            }
            free.retain(|&p| w[p] > 1e-15);
            for (i, wi) in w.iter_mut().enumerate() {
                if !free.contains(&i) {
                    *wi = 0.0;
                }
            }
            if free.is_empty() {
                return Err(DftError::LinearAlgebra(
                    "the resonance weights left the simplex".into(),
                ));
            }
        }
    }
    Err(DftError::LinearAlgebra(
        "the resonance weights did not converge in the active-set iteration limit".into(),
    ))
}

/// Natural resonance analysis of one density (NAO basis) of maximum
/// occupancy `q` with its natural Lewis structure `reference`.
pub fn natural_resonance(
    nao: &NaturalAtomicOrbitals,
    numbers: &[u8],
    density: &Matrix,
    fock: Option<&Matrix>,
    maximum_occupancy: f64,
    reference: &NboChannel,
    max_structures: usize,
) -> Result<Resonance> {
    let natoms = numbers.len();
    if reference
        .nbos
        .iter()
        .any(|b| matches!(b.kind, NboKind::ThreeCenter))
    {
        return Err(DftError::Unsupported {
            feature: "natural resonance theory of a structure with three-centre bonds".into(),
            missing: "resonance structures of three-centre bonds".into(),
        });
    }
    let atom_orbitals = nao.atom_orbitals(natoms);
    let core: Vec<Vec<usize>> = atom_orbitals
        .iter()
        .map(|list| {
            list.iter()
                .copied()
                .filter(|&i| nao.labels[i].kind == NaoKind::Core)
                .collect()
        })
        .collect();
    let mut reference_lone = vec![0usize; natoms];
    let mut reference_bonds: Vec<(usize, usize)> = Vec::new();
    for nbo in reference.nbos.iter().filter(|b| b.kind.is_lewis()) {
        match nbo.kind {
            NboKind::LonePair => reference_lone[nbo.atoms[0]] += 1,
            NboKind::Bond => reference_bonds.push((nbo.atoms[0], nbo.atoms[1])),
            _ => {}
        }
    }
    let cores: usize = core.iter().map(Vec::len).sum();
    let units = reference.expected_lewis_count.saturating_sub(cores);
    // Allowed pairs and their minimal orders.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for a in 0..natoms {
        for b in a + 1..natoms {
            if reference_bonds.contains(&(a, b)) || nao.wiberg_bond_indices[(a, b)] >= 0.1 {
                pairs.push((a, b));
            }
        }
    }
    let minimum: Vec<usize> = pairs
        .iter()
        .map(|p| usize::from(reference_bonds.contains(p)))
        .collect();
    let capacity: Vec<usize> = numbers
        .iter()
        .map(|&z| if z <= 2 { 1 } else { 4 })
        .collect();
    let mut used = vec![0usize; natoms];
    for (&(a, b), &order) in pairs.iter().zip(&minimum) {
        used[a] += order;
        used[b] += order;
    }
    if used.iter().zip(&capacity).any(|(u, c)| u > c) {
        return Err(DftError::InvalidInput(
            "the sigma skeleton of the natural Lewis structure exceeds the valence capacity".into(),
        ));
    }
    let skeleton: usize = minimum.iter().sum();
    let reference_orders: Vec<usize> = pairs
        .iter()
        .map(|p| reference_bonds.iter().filter(|q| *q == p).count())
        .collect();
    let mut enumerator = Enumerator {
        pairs: &pairs,
        minimum: &minimum,
        capacity: &capacity,
        out: Vec::new(),
        truncated: false,
    };
    let remaining = units.checked_sub(skeleton).ok_or_else(|| {
        DftError::InvalidInput(
            "the natural Lewis structure has more bonds than electron pairs".into(),
        )
    })?;
    let mut orders = Vec::new();
    enumerator.pairs(0, &mut used.clone(), &mut orders, remaining);
    let truncated = enumerator.truncated;
    let mut candidates = enumerator.out;
    let reference_assignment = Assignment {
        lone: reference_lone.clone(),
        orders: reference_orders.clone(),
    };
    if !candidates.contains(&reference_assignment) {
        candidates.push(reference_assignment.clone());
    }
    let enumerated = candidates.len();
    // Pre-ranking by agreement with the Wiberg indices when there are many.
    if candidates.len() > EVALUATION_LIMIT {
        let wiberg_score = |s: &Assignment| -> f64 {
            pairs
                .iter()
                .zip(&s.orders)
                .map(|(&(a, b), &o)| {
                    let scale = 0.5 * maximum_occupancy;
                    (o as f64 * scale - nao.wiberg_bond_indices[(a, b)]).powi(2)
                })
                .sum()
        };
        candidates.sort_by(|x, y| wiberg_score(x).total_cmp(&wiberg_score(y)));
        candidates.truncate(EVALUATION_LIMIT);
        if !candidates.contains(&reference_assignment) {
            candidates.push(reference_assignment.clone());
        }
    }
    let norm = frobenius(density);
    let ideal = |s: &Assignment| -> Result<Matrix> {
        let bonds: Vec<(usize, usize, usize)> = pairs
            .iter()
            .zip(&s.orders)
            .filter(|(_, &o)| o > 0)
            .map(|(&(a, b), &o)| (a, b, o))
            .collect();
        let vectors = prescribed_lewis(density, fock, &atom_orbitals, &core, &s.lone, &bonds)?;
        let n = density.rows;
        let mut gamma = Matrix::zeros(n, n);
        for v in &vectors {
            for (i, vi) in v.iter().enumerate() {
                if *vi == 0.0 {
                    continue;
                }
                for (j, vj) in v.iter().enumerate() {
                    gamma[(i, j)] += maximum_occupancy * vi * vj;
                }
            }
        }
        Ok(gamma)
    };
    let mut evaluated: Vec<(Assignment, Matrix, f64)> = Vec::with_capacity(candidates.len());
    let mut skipped = 0;
    for s in candidates {
        // The reference is the natural Lewis structure itself: its idealized
        // density is built from the NBOs of the analysis.
        let built = if s == reference_assignment {
            let n = density.rows;
            let mut gamma = Matrix::zeros(n, n);
            for (j, nbo) in reference.nbos.iter().enumerate() {
                if !nbo.kind.is_lewis() {
                    continue;
                }
                for p in 0..n {
                    for q in 0..n {
                        gamma[(p, q)] += maximum_occupancy
                            * reference.coefficients[(p, j)]
                            * reference.coefficients[(q, j)];
                    }
                }
            }
            Ok(gamma)
        } else {
            ideal(&s)
        };
        let gamma = match built {
            Ok(gamma) => gamma,
            Err(DftError::LinearAlgebra(_)) if s != reference_assignment => {
                skipped += 1;
                continue;
            }
            Err(error) => return Err(error),
        };
        let deviation = frobenius(&{
            let mut d = density.clone();
            for (x, y) in d.as_mut_slice().iter_mut().zip(gamma.as_slice()) {
                *x -= y;
            }
            d
        }) / norm;
        evaluated.push((s, gamma, deviation));
    }
    let evaluated_count = evaluated.len();
    evaluated.sort_by(|x, y| x.2.total_cmp(&y.2));
    let reference_position = evaluated
        .iter()
        .position(|(s, _, _)| *s == reference_assignment)
        .expect("the reference structure is evaluated");
    let reference_deviation = evaluated[reference_position].2;
    let mut chosen: Vec<usize> = (0..evaluated.len().min(max_structures.max(1))).collect();
    if !chosen.contains(&reference_position) {
        chosen.push(reference_position);
    }
    let m = chosen.len();
    let mut g = Matrix::zeros(m, m);
    let mut c = vec![0.0; m];
    for (i, &p) in chosen.iter().enumerate() {
        c[i] = evaluated[p].1.frobenius_dot(density);
        for (j, &q) in chosen.iter().enumerate() {
            g[(i, j)] = evaluated[p].1.frobenius_dot(&evaluated[q].1);
        }
    }
    let start = chosen
        .iter()
        .position(|&p| p == reference_position)
        .expect("the reference is chosen");
    let (weights, optimality_residual) = simplex_least_squares(&g, &c, start)?;
    let (gram, _) = symmetric_eigen(&g)?;
    let top = gram.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let degenerate_directions = gram.iter().filter(|v| v.abs() < 1e-10 * top).count();
    let mut hybrid = density.clone();
    let mut bond_orders = Matrix::zeros(natoms, natoms);
    for (&w, &p) in weights.iter().zip(&chosen) {
        for (x, y) in hybrid
            .as_mut_slice()
            .iter_mut()
            .zip(evaluated[p].1.as_slice())
        {
            *x -= w * y;
        }
        for (&(a, b), &o) in pairs.iter().zip(&evaluated[p].0.orders) {
            bond_orders[(a, b)] += w * o as f64;
            bond_orders[(b, a)] += w * o as f64;
        }
    }
    let fractional_deviation = frobenius(&hybrid) / norm;
    let valencies: Vec<f64> = (0..natoms)
        .map(|a| (0..natoms).map(|b| bond_orders[(a, b)]).sum())
        .collect();
    let summed = maximum_occupancy > 1.5;
    let valence_charge: Vec<f64> = (0..natoms)
        .map(|a| {
            nao.atomic_charges[a] + nao.atomic_populations[a]
                - maximum_occupancy * core[a].len() as f64
        })
        .collect();
    let mut structures: Vec<ResonanceStructure> = weights
        .iter()
        .zip(&chosen)
        .filter(|(&w, _)| w > 1e-6)
        .map(|(&w, &p)| {
            let (s, _, deviation) = &evaluated[p];
            let bonds: Vec<(usize, usize, usize)> = pairs
                .iter()
                .zip(&s.orders)
                .filter(|(_, &o)| o > 0)
                .map(|(&(a, b), &o)| (a, b, o))
                .collect();
            let formal_charges = summed.then(|| {
                (0..natoms)
                    .map(|a| {
                        let shared: usize = bonds
                            .iter()
                            .filter(|(x, y, _)| *x == a || *y == a)
                            .map(|(_, _, o)| o)
                            .sum();
                        valence_charge[a]
                            - maximum_occupancy * s.lone[a] as f64
                            - 0.5 * maximum_occupancy * shared as f64
                    })
                    .collect()
            });
            ResonanceStructure {
                lone_pairs: s.lone.clone(),
                bonds,
                weight: w,
                deviation: *deviation,
                formal_charges,
            }
        })
        .collect();
    structures.sort_by(|x, y| y.weight.total_cmp(&x.weight));
    Ok(Resonance {
        structures,
        bond_orders,
        valencies,
        fractional_deviation,
        reference_deviation,
        enumerated,
        evaluated: evaluated_count,
        skipped,
        truncated,
        optimality_residual,
        degenerate_directions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The active-set weights agree with a dense scan of the simplex.
    #[test]
    fn simplex_weights_minimize_the_quadratic() {
        let g = Matrix::from_row_major(3, 3, vec![2.0, 0.6, 0.3, 0.6, 1.5, 0.4, 0.3, 0.4, 1.2]);
        let c = vec![1.1, 0.9, 0.2];
        let (w, residual) = simplex_least_squares(&g, &c, 0).unwrap();
        let value = |w: &[f64]| -> f64 {
            let mut v = 0.0;
            for (i, (wi, ci)) in w.iter().zip(&c).enumerate() {
                for (j, wj) in w.iter().enumerate() {
                    v += 0.5 * wi * g[(i, j)] * wj;
                }
                v -= ci * wi;
            }
            v
        };
        let best = value(&w);
        let steps = 400;
        let mut scanned = f64::INFINITY;
        for i in 0..=steps {
            for j in 0..=(steps - i) {
                let x = [
                    i as f64 / steps as f64,
                    j as f64 / steps as f64,
                    (steps - i - j) as f64 / steps as f64,
                ];
                scanned = scanned.min(value(&x));
            }
        }
        assert!(residual < 1e-12);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(w.iter().all(|&x| x >= 0.0));
        assert!(best <= scanned + 1e-12);
    }
}
