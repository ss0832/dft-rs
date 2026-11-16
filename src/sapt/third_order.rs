// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Third intermolecular order at zeroth intramonomer correlation.
//!
//! The monomer Fock sums define the zeroth-order product-determinant
//! spectrum. A single interaction can change one orbital of each
//! monomer, so the second-order ket needs at most two replacements in
//! each monomer. The one-sided interaction uses explicit overlap
//! cofactors and also works for singular overlaps of excited determinants.

use super::antisymmetry::Interaction;
use crate::linalg::{solve_linear_matrix, Matrix};
use crate::properties::pt2_density::transform;
use crate::scf::ScfResult;
use crate::{DftError, Result};

/// Uncoupled third-order polarization and exchange components.
/// All energies are in Hartree.
#[derive(Clone, Copy, Debug)]
pub struct SaptThirdOrder {
    pub induction: f64,
    pub dispersion: f64,
    pub induction_dispersion: f64,
    /// Sum of the three exchange components.
    pub exchange: f64,
    pub exchange_induction: f64,
    pub exchange_dispersion: f64,
    pub exchange_induction_dispersion: f64,
}

impl SaptThirdOrder {
    /// Full third-order contribution to the interaction energy.
    pub fn total(self) -> f64 {
        self.induction + self.dispersion + self.induction_dispersion + self.exchange
    }
}

pub(super) struct Terms {
    pub(super) first: f64,
    pub(super) electrostatics: f64,
    pub(super) second_bare: [f64; 3],
    pub(super) second_antisymmetrized: [f64; 3],
    pub(super) third: SaptThirdOrder,
}

pub(super) struct Determinant {
    pub(super) orbitals: [Vec<usize>; 2],
    pub(super) rank: usize,
    pub(super) gap: f64,
}

pub(super) struct Transition {
    orbitals: Vec<(usize, usize)>,
    unchanged: bool,
    phase: f64,
}

impl Transition {
    pub(super) fn between(bra: &Determinant, ket: &Determinant) -> Option<Self> {
        let mut difference = None;
        for (left, right) in bra.orbitals.iter().zip(&ket.orbitals) {
            for (i, &p) in left.iter().enumerate() {
                if right.binary_search(&p).is_err() {
                    let (j, q) = right
                        .iter()
                        .enumerate()
                        .find(|&(_, q)| left.binary_search(q).is_err())?;
                    if difference
                        .replace((p, *q, if (i + j) % 2 == 0 { 1.0 } else { -1.0 }))
                        .is_some()
                    {
                        return None;
                    }
                }
            }
        }
        Some(if let Some((p, q, phase)) = difference {
            Self {
                orbitals: vec![(p, q)],
                unchanged: false,
                phase,
            }
        } else {
            Self {
                orbitals: bra.orbitals.iter().flatten().map(|&p| (p, p)).collect(),
                unchanged: true,
                phase: 1.0,
            }
        })
    }
}

pub(super) struct Operators {
    potential: [Matrix; 2],
    eri: Vec<f64>,
    overlap: Matrix,
    nuclear: f64,
}

impl Operators {
    pub(super) fn eri(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let n = self.overlap.rows;
        self.eri[((p * n + q) * n + r) * n + s]
    }

    // Centered one-body electrostatic field in the reference product.
    fn field(
        &self,
        transition: &Transition,
        reference: &Determinant,
        partner: &Determinant,
        source: usize,
    ) -> f64 {
        let element = |p, q| {
            self.potential[source][(p, q)]
                + partner
                    .orbitals
                    .iter()
                    .flatten()
                    .map(|&r| self.eri(p, q, r, r))
                    .sum::<f64>()
        };
        let mut value = transition
            .orbitals
            .iter()
            .map(|&(p, q)| element(p, q))
            .sum::<f64>();
        if transition.unchanged {
            value -= reference
                .orbitals
                .iter()
                .flatten()
                .map(|&p| element(p, p))
                .sum::<f64>();
        }
        transition.phase * value
    }

    pub(super) fn bare(&self, a: &Transition, b: &Transition) -> f64 {
        let mut result = 0.0;
        if b.unchanged {
            result += a
                .orbitals
                .iter()
                .map(|&ij| self.potential[1][ij])
                .sum::<f64>();
        }
        if a.unchanged {
            result += b
                .orbitals
                .iter()
                .map(|&ij| self.potential[0][ij])
                .sum::<f64>();
        }
        if a.unchanged && b.unchanged {
            result += self.nuclear;
        }
        for &(p, q) in &a.orbitals {
            for &(r, s) in &b.orbitals {
                result += self.eri(p, q, r, s);
            }
        }
        a.phase * b.phase * result
    }

    pub(super) fn nuclear(&self) -> f64 {
        self.nuclear
    }

    /// Overlap and the same-spin interaction vertex in the selected
    /// orientation. A reversed bra/ket pair reuses the transposed
    /// overlap cofactors, with the intermonomer operator on the new bra.
    fn spin_vertex(
        &self,
        left: &[usize],
        right: &[usize],
        split: usize,
        cofactor: &Cofactors<'_>,
        reversed: bool,
    ) -> f64 {
        let mut potential = 0.0;
        for (i, &p) in left.iter().enumerate() {
            let partner = usize::from(i < split);
            for (j, &q) in right.iter().enumerate() {
                let c = if reversed {
                    cofactor.first[(j, i)]
                } else {
                    cofactor.first[(i, j)]
                };
                potential += self.potential[partner][(p, q)] * c;
            }
            if i < split {
                for (k, &r) in left.iter().enumerate().skip(split) {
                    for (j, &q) in right.iter().enumerate() {
                        for (l, &s) in right.iter().enumerate().skip(j + 1) {
                            let c = if reversed {
                                cofactor.second(j, l, i, k)
                            } else {
                                cofactor.second(i, k, j, l)
                            };
                            potential += (self.eri(p, q, r, s) - self.eri(p, s, r, q)) * c;
                        }
                    }
                }
            }
        }
        potential
    }

    pub(super) fn spin_terms(
        &self,
        left: &[usize],
        right: &[usize],
        split: usize,
        both_orientations: bool,
    ) -> (f64, [f64; 2], Matrix) {
        let n = left.len();
        let overlap = Matrix::from_row_major(
            n,
            n,
            left.iter()
                .flat_map(|&p| right.iter().map(move |&q| self.overlap[(p, q)]))
                .collect(),
        );
        let cofactor = Cofactors::new(&overlap);
        let forward = self.spin_vertex(left, right, split, &cofactor, false);
        let reverse = if both_orientations {
            self.spin_vertex(right, left, split, &cofactor, true)
        } else {
            forward
        };
        (cofactor.determinant, [forward, reverse], cofactor.first)
    }

    pub(super) fn antisymmetrized(
        &self,
        bra: [&Determinant; 2],
        ket: [&Determinant; 2],
    ) -> (f64, f64) {
        let left: [Vec<usize>; 2] = std::array::from_fn(|spin| {
            bra.iter()
                .flat_map(|d| d.orbitals[spin].iter().copied())
                .collect()
        });
        let right: [Vec<usize>; 2] = std::array::from_fn(|spin| {
            ket.iter()
                .flat_map(|d| d.orbitals[spin].iter().copied())
                .collect()
        });
        let split = bra[0].orbitals.each_ref().map(Vec::len);
        let [(da, va, ca), (db, vb, cb)] =
            [0, 1].map(|spin| self.spin_terms(&left[spin], &right[spin], split[spin], false));
        let norm = da * db;
        let mut value = norm * self.nuclear + va[0] * db + vb[0] * da;
        let first = [ca, cb];
        for spin in 0..2 {
            let other = 1 - spin;
            for (i, &p) in left[spin].iter().enumerate().take(split[spin]) {
                for (j, &q) in right[spin].iter().enumerate() {
                    for k in split[other]..left[other].len() {
                        for (l, &s) in right[other].iter().enumerate() {
                            value += self.eri(p, q, left[other][k], s)
                                * first[spin][(i, j)]
                                * first[other][(k, l)];
                        }
                    }
                }
            }
        }
        (value, norm)
    }
}

struct Cofactors<'a> {
    overlap: &'a Matrix,
    determinant: f64,
    inverse: Option<Matrix>,
    first: Matrix,
}

impl<'a> Cofactors<'a> {
    fn new(overlap: &'a Matrix) -> Self {
        let determinant = minor(overlap, &[], &[]);
        // Jacobi's identity reuses one inverse for every cofactor when the
        // overlap is well conditioned. Pivoted direct minors cover exactly
        // and nearly singular excited-state overlaps.
        let inverse = if determinant == 0.0 || overlap.rows == 0 {
            None
        } else {
            let inf_norm = |m: &Matrix| {
                m.as_slice()
                    .chunks_exact(m.cols)
                    .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
                    .fold(0.0_f64, f64::max)
            };
            solve_linear_matrix(overlap, &Matrix::identity(overlap.rows))
                .ok()
                .filter(|inv| inf_norm(overlap) * inf_norm(inv) * f64::EPSILON < 1e-9)
        };
        let mut first = Matrix::zeros(overlap.rows, overlap.cols);
        for i in 0..overlap.rows {
            for j in 0..overlap.cols {
                first[(i, j)] = inverse.as_ref().map_or_else(
                    || minor(overlap, &[i], &[j]),
                    |inverse| determinant * inverse[(j, i)],
                );
            }
        }
        Self {
            overlap,
            determinant,
            inverse,
            first,
        }
    }

    fn second(&self, i: usize, k: usize, j: usize, l: usize) -> f64 {
        self.inverse.as_ref().map_or_else(
            || minor(self.overlap, &[i, k], &[j, l]),
            |m| self.determinant * (m[(j, i)] * m[(l, k)] - m[(j, k)] * m[(l, i)]),
        )
    }
}

/// Signed minor, with pivoting and no inverse of an excited-state overlap.
fn minor(m: &Matrix, rows: &[usize], cols: &[usize]) -> f64 {
    let n = m.rows - rows.len();
    let sign_sum = rows.iter().chain(cols).sum::<usize>();
    let mut determinant = if sign_sum % 2 == 0 { 1.0 } else { -1.0 };
    let mut a = Vec::with_capacity(n * n);
    for i in 0..m.rows {
        if rows.contains(&i) {
            continue;
        }
        for j in 0..m.cols {
            if !cols.contains(&j) {
                a.push(m[(i, j)]);
            }
        }
    }
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap_or(k);
        let p = a[pivot * n + k];
        if p == 0.0 {
            return 0.0;
        }
        if pivot != k {
            determinant = -determinant;
            for j in k..n {
                a.swap(pivot * n + j, k * n + j);
            }
        }
        determinant *= p;
        for i in k + 1..n {
            let alpha = a[i * n + k] / p;
            for j in k + 1..n {
                a[i * n + j] -= alpha * a[k * n + j];
            }
        }
    }
    determinant
}

#[derive(Clone, Copy)]
struct Replacement {
    spin: usize,
    occupied: usize,
    virtual_orbital: usize,
    gap: f64,
}

fn determinants(reference: [Vec<usize>; 2], replacements: &[Replacement]) -> Vec<Determinant> {
    let excited = |a: Replacement, b: Option<Replacement>| {
        let mut orbitals = reference.clone();
        orbitals[a.spin][a.occupied] = a.virtual_orbital;
        if let Some(b) = b {
            orbitals[b.spin][b.occupied] = b.virtual_orbital;
        }
        for part in &mut orbitals {
            part.sort_unstable();
        }
        Determinant {
            orbitals,
            rank: 1 + usize::from(b.is_some()),
            gap: a.gap + b.map_or(0.0, |b| b.gap),
        }
    };
    let mut states = vec![Determinant {
        orbitals: reference.clone(),
        rank: 0,
        gap: 0.0,
    }];
    for &a in replacements {
        states.push(excited(a, None));
    }
    for (i, &a) in replacements.iter().enumerate() {
        for &b in &replacements[i + 1..] {
            if a.spin != b.spin
                || (a.occupied < b.occupied && a.virtual_orbital < b.virtual_orbital)
            {
                states.push(excited(a, Some(b)));
            }
        }
    }
    states
}

pub(super) type Spectrum = [Vec<Determinant>; 2];

pub(super) fn prepare(
    interaction: &Interaction<'_>,
    monomers: [&ScfResult; 2],
    limit_mb: usize,
) -> Result<(Operators, Spectrum)> {
    let nao = interaction.overlap.rows;
    let orbital_count: usize = monomers
        .iter()
        .map(|m| {
            m.mo_coeff[0].cols
                + if m.unrestricted {
                    m.mo_coeff[1].cols
                } else {
                    0
                }
        })
        .sum();
    let memory_limit = limit_mb.saturating_mul(1024 * 1024);
    let eri_entries = nao
        .saturating_pow(4)
        .saturating_add(orbital_count.saturating_mul(nao.saturating_pow(3)))
        .saturating_add(
            orbital_count
                .saturating_pow(2)
                .saturating_mul(nao.saturating_pow(2)),
        )
        .saturating_add(orbital_count.saturating_pow(4));
    if eri_entries.saturating_mul(8) > memory_limit {
        return Err(DftError::InvalidInput(format!(
            "SAPT(30) orbital-integral workspace exceeds the {limit_mb} MiB integral limit"
        )));
    }
    let local_states = monomers.map(|m| {
        let [no_a, no_b] = [m.n_alpha, m.n_beta];
        let va = m.mo_coeff[0].cols - no_a;
        let vb = m.mo_coeff[usize::from(m.unrestricted)].cols - no_b;
        let choose_two = |n: usize| n.saturating_mul(n.saturating_sub(1)) / 2;
        1 + no_a * va
            + no_b * vb
            + no_a * va * no_b * vb
            + choose_two(no_a) * choose_two(va)
            + choose_two(no_b) * choose_two(vb)
    });
    let amplitudes = local_states[0]
        .saturating_mul(local_states[1])
        .saturating_mul(32);
    let local_orbitals = local_states
        .into_iter()
        .zip(monomers)
        .map(|(count, m)| count.saturating_mul(64 + (m.n_alpha + m.n_beta) * 8))
        .fold(0, usize::saturating_add);
    let live_integrals = nao
        .saturating_pow(4)
        .saturating_add(orbital_count.saturating_pow(4))
        .saturating_mul(8);
    if amplitudes
        .saturating_add(local_orbitals)
        .saturating_add(live_integrals)
        > memory_limit
    {
        return Err(DftError::InvalidInput(format!(
            "SAPT(30) determinant workspace exceeds the {limit_mb} MiB integral limit"
        )));
    }
    let mut c = Matrix::zeros(nao, orbital_count);
    let mut ao_offset = 0;
    let mut mo_offset = 0;
    let mut spectrum = [Vec::new(), Vec::new()];
    for (fragment, monomer) in monomers.into_iter().enumerate() {
        let mut reference = [Vec::new(), Vec::new()];
        let mut replacements = Vec::new();
        let alpha_start = mo_offset;
        for (spin, count) in [monomer.n_alpha, monomer.n_beta].into_iter().enumerate() {
            let channel = if monomer.unrestricted { spin } else { 0 };
            let coeff = &monomer.mo_coeff[channel];
            let energy = &monomer.mo_energy[channel];
            let current = if spin == 0 || monomer.unrestricted {
                let current = mo_offset;
                for i in 0..coeff.rows {
                    for j in 0..coeff.cols {
                        c[(ao_offset + i, mo_offset + j)] = coeff[(i, j)];
                    }
                }
                mo_offset += coeff.cols;
                current
            } else {
                alpha_start
            };
            reference[spin] = (current..current + count).collect();
            for i in 0..count {
                for a in count..coeff.cols {
                    let gap = energy[a] - energy[i];
                    if !gap.is_finite() || gap <= 0.0 {
                        return Err(DftError::InvalidInput(
                            "SAPT(30) requires positive Aufbau monomer orbital gaps".into(),
                        ));
                    }
                    replacements.push(Replacement {
                        spin,
                        occupied: i,
                        virtual_orbital: current + a,
                        gap,
                    });
                }
            }
        }
        ao_offset += monomer.mo_coeff[0].rows;
        spectrum[fragment] = determinants(reference, &replacements);
    }
    let one_body = |op: &Matrix| c.transpose_matmul(&op.matmul(&c));
    let operators = Operators {
        potential: interaction.external.map(one_body),
        overlap: one_body(interaction.overlap),
        eri: transform(interaction.eri, [&c; 4]),
        nuclear: interaction.nuclear,
    };
    Ok((operators, spectrum))
}

fn vertices(states: &[Determinant], ket: usize) -> Vec<(usize, Transition)> {
    states
        .iter()
        .enumerate()
        .filter_map(|(i, d)| Transition::between(d, &states[ket]).map(|t| (i, t)))
        .collect()
}

pub(super) fn terms(
    interaction: &Interaction<'_>,
    monomers: [&ScfResult; 2],
    limit_mb: usize,
) -> Result<Terms> {
    let (operators, spectrum) = prepare(interaction, monomers, limit_mb)?;
    prepared_terms(&operators, &spectrum)
}

pub(super) fn prepared_terms(operators: &Operators, [a, b]: &Spectrum) -> Result<Terms> {
    let reference = [&a[0], &b[0]];
    let gap = |i: usize, j: usize| a[i].gap + b[j].gap;
    let flat = |i, j| i * b.len() + j;
    let transitions_a = vertices(a, 0);
    let transitions_b = vertices(b, 0);
    let electrostatics = operators.bare(&transitions_a[0].1, &transitions_b[0].1);
    let (numerator, norm) = operators.antisymmetrized(reference, reference);
    if !norm.is_finite() || norm <= 1e-12 {
        return Err(DftError::InvalidInput(
            "SAPT(30) requires a nonsingular reference determinant overlap".into(),
        ));
    }
    let first = numerator / norm;
    let mut linear = vec![0.0; a.len() * b.len()];
    let mut quadratic: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; linear.len()]);
    let mut second_bare = [0.0; 3];
    let mut second_antisymmetrized = [0.0; 3];
    let mut first_overlap = [0.0; 2];
    let mut active = Vec::new();
    for (i, ta) in &transitions_a {
        for (j, tb) in &transitions_b {
            let (i, j) = (*i, *j);
            if i == 0 && j == 0 {
                continue;
            }
            let category = if i == 0 {
                1
            } else if j == 0 {
                0
            } else {
                2
            };
            let interaction_element = operators.bare(ta, tb);
            let amplitude = -interaction_element / gap(i, j);
            if amplitude == 0.0 {
                continue;
            }
            let index = flat(i, j);
            linear[index] = amplitude;
            active.push((i, j, category));
            second_bare[category] += interaction_element * amplitude;
            let (num, overlap) = operators.antisymmetrized(reference, [&a[i], &b[j]]);
            second_antisymmetrized[category] += (num - first * overlap) * amplitude / norm;
            first_overlap[usize::from(category == 2)] += overlap * amplitude / norm;
        }
    }
    let mut polarization = [0.0; 3];
    for (i, j, incoming) in active {
        let amplitude = linear[flat(i, j)];
        let neighbors_a = vertices(a, i);
        let neighbors_b = vertices(b, j);
        for (k, ta) in &neighbors_a {
            for (l, tb) in &neighbors_b {
                let (k, l) = (*k, *l);
                if k == 0 && l == 0 {
                    continue;
                }
                let mut value = operators.bare(ta, tb);
                if i == k && j == l {
                    value -= electrostatics;
                }
                let out = flat(k, l);
                let one_sided = k == 0 || l == 0;
                let one_excitation = a[k].rank + b[l].rank == 1;
                let mut field = 0.0;
                if tb.unchanged {
                    field += operators.field(ta, &a[0], &b[0], 1);
                }
                if ta.unchanged {
                    field += operators.field(tb, &b[0], &a[0], 0);
                }
                let fluctuation = value - field;
                // The field between dispersion amplitudes is mixed
                // induction-dispersion. A fluctuation between single
                // excitations on opposite monomers is mutual induction.
                let field_group = if incoming == 2 && !one_sided { 2 } else { 0 };
                let fluctuation_group = if incoming == 2 {
                    if one_excitation {
                        2
                    } else {
                        1
                    }
                } else if one_sided {
                    0
                } else {
                    2
                };
                for (group, element) in [(field_group, field), (fluctuation_group, fluctuation)] {
                    quadratic[group][out] -= element * amplitude / gap(k, l);
                    polarization[group] += linear[out] * element * amplitude;
                }
            }
        }
    }
    // Normalization through third order with both pure and mixed products.
    let s_ind = second_antisymmetrized[0] + second_antisymmetrized[1];
    let s_disp = second_antisymmetrized[2];
    let mut third_antisymmetrized = [
        -s_ind * first_overlap[0],
        -s_disp * first_overlap[1],
        -s_ind * first_overlap[1] - s_disp * first_overlap[0],
    ];
    for (i, da) in a.iter().enumerate() {
        for (j, db) in b.iter().enumerate() {
            let amplitudes = quadratic.each_ref().map(|q| q[flat(i, j)]);
            if amplitudes != [0.0; 3] {
                debug_assert!(da.rank <= 2 && db.rank <= 2);
                let (numerator, overlap) = operators.antisymmetrized(reference, [da, db]);
                let vertex = (numerator - first * overlap) / norm;
                for (term, amplitude) in third_antisymmetrized.iter_mut().zip(amplitudes) {
                    *term += vertex * amplitude;
                }
            }
        }
    }
    let [exchange_induction, exchange_dispersion, exchange_induction_dispersion] =
        std::array::from_fn(|i| third_antisymmetrized[i] - polarization[i]);
    Ok(Terms {
        first,
        electrostatics,
        second_bare,
        second_antisymmetrized,
        third: SaptThirdOrder {
            induction: polarization[0],
            dispersion: polarization[1],
            induction_dispersion: polarization[2],
            exchange: exchange_induction + exchange_dispersion + exchange_induction_dispersion,
            exchange_induction,
            exchange_dispersion,
            exchange_induction_dispersion,
        },
    })
}
