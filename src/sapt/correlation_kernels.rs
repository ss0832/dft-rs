// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Product-space contractions for the mixed-order SAPT recurrence.

use super::third_order::{Determinant, Operators, Transition};
use crate::linalg::Matrix;
use crate::{resources, Result};

pub(super) fn bare_actions<const N: usize>(
    operators: &Operators,
    monomers: [&[Determinant]; 2],
    states: &[Matrix; N],
) -> [Matrix; N] {
    // Each bare intermonomer vertex changes at most one spin orbital
    // on each fragment. Read the input once and scatter only those
    // transitions allowed by the one-particle selection rule.
    let transitions = monomers.map(|dets| {
        dets.iter()
            .map(|right| {
                dets.iter()
                    .enumerate()
                    .filter_map(|(i, left)| Transition::between(left, right).map(|t| (i, t)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    let [a, b] = &transitions;
    let mut result = std::array::from_fn(|_| Matrix::zeros(a.len(), b.len()));
    for (i, ta) in a.iter().enumerate() {
        for (j, tb) in b.iter().enumerate() {
            let amplitudes = states.each_ref().map(|x| x[(i, j)]);
            if amplitudes == [0.0; N] {
                continue;
            }
            for (k, x) in ta {
                for (l, y) in tb {
                    let element = operators.bare(x, y);
                    for (m, c) in result.iter_mut().zip(amplitudes) {
                        m[(*k, *l)] += element * c;
                    }
                }
            }
        }
    }
    result
}

#[derive(Clone, Copy)]
struct Amplitudes {
    beta: usize,
    ground: [f64; 4],
    response: [[f64; 3]; 3],
    ground_order: u8,
    response_order: u8,
}

impl Amplitudes {
    fn minimum(&self) -> u8 {
        self.ground_order.min(1 + self.response_order)
    }
}

pub(super) fn vector_bytes() -> usize {
    std::mem::size_of::<Amplitudes>() + 14 * std::mem::size_of::<f64>()
}

pub(super) fn spin_bytes(strings: usize, electrons: usize) -> usize {
    triangular(strings)
        .saturating_mul(electrons.saturating_pow(2).saturating_add(3))
        .saturating_mul(std::mem::size_of::<f64>())
}

fn triangular(n: usize) -> usize {
    n.saturating_mul(n.saturating_add(1)) / 2
}

#[derive(Clone, Copy)]
struct SpinVertex<'a> {
    first: &'a [f64],
    overlap: f64,
    interaction: f64,
    reversed: bool,
    electrons: usize,
}

impl SpinVertex<'_> {
    fn cofactor(self, i: usize, j: usize) -> f64 {
        let (a, b) = if self.reversed { (j, i) } else { (i, j) };
        self.first[a * self.electrons + b]
    }
}

pub(super) struct Expectations {
    /// Columns are <V A> and <A>, respectively.
    pub(super) first: [[f64; 2]; 4],
    pub(super) second: [[[f64; 2]; 3]; 3],
}

fn weight<const N: usize>(
    out: &mut [[f64; 2]; N],
    bra: &[f64; 4],
    ket: &[f64; N],
    element: [f64; 2],
) {
    for (order, [va, ov]) in out.iter_mut().enumerate() {
        let c = (0..=order).map(|i| bra[i] * ket[order - i]).sum::<f64>();
        *va += c * element[0];
        *ov += c * element[1];
    }
}

pub(super) fn exchange_expectations(
    ops: &Operators,
    monomers: [&[Determinant]; 2],
    string_counts: [usize; 2],
    orbital_count: usize,
    ground: &[Matrix; 4],
    responses: &[[Matrix; 3]; 3],
) -> Result<Expectations> {
    // Restricted alpha and beta occupation spaces are identical. Store
    // one table of spin-overlap determinants, cofactors and same-spin
    // vertices. List each spin determinant by its two monomer strings.
    let [sa, sb] = string_counts;
    let ns = sa * sb;
    let split = monomers[0][0].orbitals[0].len();
    let spins = (0..ns)
        .map(|i| {
            let ma = &monomers[0][(i / sb) * sa];
            let mb = &monomers[1][(i % sb) * sb];
            ma.orbitals[0]
                .iter()
                .chain(&mb.orbitals[0])
                .copied()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let electrons = spins[0].len();
    let cofactor_size = electrons * electrons;
    let columns = cofactor_size + 3;
    let mut spin_table = Matrix::zeros(triangular(ns), columns);
    for (p, bra) in spins.iter().enumerate() {
        for (q, ket) in spins.iter().enumerate().take(p + 1) {
            let (overlap, v, cofactor) = ops.spin_terms(bra, ket, split, p != q);
            let first = (triangular(p) + q) * columns;
            let row = &mut spin_table.as_mut_slice()[first..first + columns];
            row[..cofactor_size].copy_from_slice(cofactor.as_slice());
            row[cofactor_size] = overlap;
            row[cofactor_size + 1..].copy_from_slice(&v);
        }
    }

    let _amplitudes_memory = resources::reserve(
        ns * ns * std::mem::size_of::<Amplitudes>(),
        "SAPT product amplitudes grouped by spin",
    )?;
    let amplitudes = (0..ns)
        .map(|alpha| {
            let mut row = Vec::with_capacity(ns);
            for beta in 0..ns {
                let pos = [(alpha / sb) * sa + beta / sb, (alpha % sb) * sb + beta % sb];
                let ground = ground.each_ref().map(|m| m[(pos[0], pos[1])]);
                let response = responses
                    .each_ref()
                    .map(|v| v.each_ref().map(|m| m[(pos[0], pos[1])]));
                let ground_order = ground.iter().position(|&x| x != 0.0).unwrap_or(4) as u8;
                let response_order = response
                    .iter()
                    .filter_map(|v| v.iter().position(|&x| x != 0.0))
                    .min()
                    .unwrap_or(3) as u8;
                if ground_order < 4 || response_order < 3 {
                    row.push(Amplitudes {
                        beta,
                        ground,
                        response,
                        ground_order,
                        response_order,
                    });
                }
            }
            row.sort_unstable_by_key(|item| (item.minimum(), item.beta));
            row
        })
        .collect::<Vec<_>>();
    let row_of = |a: usize, b: usize| {
        let reversed = a < b;
        let row = (triangular(a.max(b)) + a.min(b)) * columns;
        let data = &spin_table.as_slice()[row..row + columns];
        SpinVertex {
            first: &data[..cofactor_size],
            overlap: data[cofactor_size],
            interaction: data[cofactor_size + 1 + usize::from(reversed)],
            reversed,
            electrons,
        }
    };
    let orbital_pairs = orbital_count * orbital_count;
    let eri = Matrix::from_row_major(
        orbital_pairs,
        orbital_pairs,
        (0..orbital_pairs)
            .flat_map(|p| {
                (0..orbital_pairs).map(move |q| {
                    ops.eri(
                        p / orbital_count,
                        p % orbital_count,
                        q / orbital_count,
                        q % orbital_count,
                    )
                })
            })
            .collect(),
    );
    let mut result = Expectations {
        first: [[0.0; 2]; 4],
        second: [[[0.0; 2]; 3]; 3],
    };
    let moments = amplitudes
        .iter()
        .map(|r| {
            r.iter().fold([4, 3], |[g, w], x| {
                [g.min(x.ground_order), w.min(x.response_order)]
            })
        })
        .collect::<Vec<_>>();

    // Build Coulomb fields from several alpha spin vertices in one
    // dense integral transform. Each beta spin vertex then contributes
    // a single contraction against the field on the opposite monomer.
    const BLOCK: usize = 32;
    let mut density = Matrix::zeros(orbital_pairs, 2 * BLOCK);
    for (p, left) in amplitudes.iter().enumerate() {
        let bra_order = moments[p][0];
        if bra_order > 3 {
            continue;
        }
        for first in (0..ns).step_by(BLOCK) {
            let mut active = [false; BLOCK];
            let count = BLOCK.min(ns - first);
            density.as_mut_slice().fill(0.0);
            for (col, a) in active.iter_mut().enumerate().take(count) {
                let q = first + col;
                let [ket_ground, ket_response] = moments[q];
                *a = bra_order + ket_ground < 4 || bra_order + ket_response < 3;
                if *a {
                    let spin = row_of(p, q);
                    for (i, &k) in spins[p].iter().enumerate() {
                        let fragment = usize::from(i >= split);
                        for (j, &l) in spins[q].iter().enumerate() {
                            density[(k * orbital_count + l, 2 * col + fragment)] +=
                                spin.cofactor(i, j);
                        }
                    }
                }
            }
            if !active.contains(&true) {
                continue;
            }
            let fields = eri.matmul(&density);
            for (col, _) in active.iter().enumerate().filter(|(_, a)| **a) {
                let q = first + col;
                let alpha = row_of(p, q);
                for g in left {
                    let order = g.ground_order;
                    if order > 3 {
                        continue;
                    }
                    let bra = &spins[g.beta];
                    for k in &amplitudes[q] {
                        if order + k.minimum() >= 4 {
                            break;
                        }
                        let first_order = order + k.ground_order < 4;
                        let response_order = order + k.response_order < 3;
                        let beta = row_of(g.beta, k.beta);
                        let ov = alpha.overlap * beta.overlap;
                        let mut va = ov * ops.nuclear()
                            + alpha.interaction * beta.overlap
                            + beta.interaction * alpha.overlap;
                        let ket = &spins[k.beta];
                        for (i, &l) in bra.iter().enumerate() {
                            let partner = usize::from(i < split);
                            for (j, &r) in ket.iter().enumerate() {
                                va += beta.cofactor(i, j)
                                    * fields[(l * orbital_count + r, 2 * col + partner)];
                            }
                        }
                        if first_order {
                            weight(&mut result.first, &g.ground, &k.ground, [va, ov]);
                        }
                        if response_order {
                            for (out, v) in result.second.iter_mut().zip(&k.response) {
                                weight(out, &g.ground, v, [va, ov]);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(result)
}
