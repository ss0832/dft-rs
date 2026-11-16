// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Natural hybrid orbitals, natural bond orbitals of the natural Lewis
//! structure, natural localized molecular orbitals and second-order
//! donor-acceptor energies, for one density in the NAO basis.
//!
//! Lewis search at an occupancy threshold `t` (a fraction of the maximum
//! occupancy, two electrons for the spin-summed density and one for a spin
//! density): the core NAOs are core orbitals (CR); every accepted orbital
//! is projected out of the density, `D <- (1 - U U^T) D (1 - U U^T)`; the
//! eigenvectors of the atomic blocks above `t` are lone pairs (LP), then
//! those of the two-atom blocks of the density depleted of the one-centre
//! orbitals are bonds (BD; all pairs see the same matrix, so equivalent
//! bonds get equivalent hybrids), and optionally those of three-atom blocks
//! of the density depleted of all of them are three-centre bonds (3C). Every orbital is split into its normalized
//! atomic parts, the hybrids; the hybrids of one atom are symmetrically
//! orthogonalized and completed by the natural orbitals of the atomic
//! density in their orthogonal complement (Rydberg hybrids, RY). A bond is
//! the bonding eigenvector of the density in its two hybrids, its antibond
//! (BD*) the orthogonal combination; a three-centre bond has two
//! antibonds (3C*). The NBOs form an orthonormal basis of the NAO space.
//! Without a fixed threshold the search runs from 0.95 down to 0.75 in
//! steps of 0.05 and keeps the structure with the smallest non-Lewis
//! occupancy among those with at most the expected number of Lewis
//! orbitals (ties: the higher threshold). Degenerate eigenvectors (empty
//! Rydberg orbitals, equivalent lone pairs) are fixed by diagonalizing the
//! Fock matrix within their degenerate space.
//!
//! Natural localized molecular orbitals: Jacobi rotations between Lewis and
//! non-Lewis NBOs annihilate the Lewis/non-Lewis block of the density; the
//! rotated Lewis NBOs are the NLMOs. Second-order donor-acceptor energies:
//! `E2 = q_i F_ij^2 / (F_jj - F_ii)` for a Lewis donor `i` with occupancy
//! `q_i` and a non-Lewis acceptor `j`, with the Fock matrix in the NBO
//! basis (pairs with a non-positive gap are counted and not evaluated).

use super::{inverse_square_root, NaoKind, NaturalAtomicOrbitals};
use crate::error::{DftError, Result};
use crate::linalg::{symmetric_eigen, Matrix};

/// Controls of the Lewis search and of the reported donor-acceptor pairs.
#[derive(Clone, Debug)]
pub struct NboOptions {
    /// Fixed occupancy threshold as a fraction of the maximum occupancy;
    /// `None` scans 0.95, 0.90, 0.85, 0.80 and 0.75.
    pub lewis_threshold: Option<f64>,
    /// Search three-centre bonds after the two-centre bonds.
    pub three_center: bool,
    /// Smallest reported second-order stabilization (Hartree).
    pub perturbation_threshold: f64,
}

impl Default for NboOptions {
    fn default() -> Self {
        NboOptions {
            lewis_threshold: None,
            three_center: false,
            perturbation_threshold: 0.5 * crate::constants::KCAL_MOL_TO_HARTREE,
        }
    }
}

/// Type of a natural bond orbital.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NboKind {
    Core,
    LonePair,
    Bond,
    ThreeCenter,
    Antibond,
    ThreeCenterAntibond,
    Rydberg,
}

impl NboKind {
    /// `CR`, `LP`, `BD`, `3C`, `BD*`, `3C*`, `RY`.
    pub fn name(self) -> &'static str {
        match self {
            NboKind::Core => "CR",
            NboKind::LonePair => "LP",
            NboKind::Bond => "BD",
            NboKind::ThreeCenter => "3C",
            NboKind::Antibond => "BD*",
            NboKind::ThreeCenterAntibond => "3C*",
            NboKind::Rydberg => "RY",
        }
    }

    /// Part of the natural Lewis structure.
    pub fn is_lewis(self) -> bool {
        matches!(
            self,
            NboKind::Core | NboKind::LonePair | NboKind::Bond | NboKind::ThreeCenter
        )
    }
}

/// Natural hybrid of one atom in one NBO.
#[derive(Clone, Debug)]
pub struct Hybrid {
    pub atom: usize,
    /// Polarization coefficient of the hybrid in the NBO.
    pub coefficient: f64,
    /// Percent s, p, d, f, ... character of the normalized hybrid.
    pub composition: Vec<f64>,
}

/// One natural bond orbital.
#[derive(Clone, Debug)]
pub struct Nbo {
    pub kind: NboKind,
    pub atoms: Vec<usize>,
    /// 1, 2, ... for orbitals of the same type on the same atoms.
    pub serial: usize,
    /// `BD(1) O1-H2`.
    pub label: String,
    pub occupancy: f64,
    /// Diagonal Fock element (Hartree).
    pub energy: Option<f64>,
    pub hybrids: Vec<Hybrid>,
}

/// Second-order stabilization of one donor-acceptor pair.
#[derive(Clone, Copy, Debug)]
pub struct DonorAcceptor {
    pub donor: usize,
    pub acceptor: usize,
    /// `q_i F_ij^2 / (F_jj - F_ii)` (Hartree).
    pub stabilization: f64,
    pub fock_element: f64,
    pub energy_gap: f64,
}

/// One natural localized molecular orbital.
#[derive(Clone, Debug)]
pub struct Nlmo {
    /// Index of the parent Lewis NBO.
    pub parent: usize,
    pub occupancy: f64,
    /// Percent of the parent NBO.
    pub parent_percentage: f64,
    /// Percent on every atom.
    pub atomic_percentages: Vec<f64>,
}

/// One threshold of the Lewis search.
#[derive(Clone, Copy, Debug)]
pub struct LewisTrial {
    /// Occupancy threshold (electrons).
    pub threshold: f64,
    pub lewis_count: usize,
    pub non_lewis_occupancy: f64,
    /// The hybrids of an atom were linearly dependent at this threshold;
    /// the trial is left out of the selection.
    pub dependent: bool,
}

/// Natural bond orbital analysis of one density.
#[derive(Clone, Debug)]
pub struct NboChannel {
    /// `total`, `alpha` or `beta`.
    pub spin: String,
    pub maximum_occupancy: f64,
    pub electrons: f64,
    pub expected_lewis_count: usize,
    /// Selected threshold (electrons).
    pub lewis_threshold: f64,
    pub trials: Vec<LewisTrial>,
    pub lewis_count: usize,
    pub lewis_occupancy: f64,
    pub non_lewis_occupancy: f64,
    /// Lewis NBOs first, then antibonds and Rydberg orbitals.
    pub nbos: Vec<Nbo>,
    /// NAO coefficients of the NBOs (columns).
    pub coefficients: Matrix,
    pub orthonormality_residual: f64,
    /// Pairs above the reporting threshold, descending.
    pub perturbation: Vec<DonorAcceptor>,
    /// Sum over all donor-acceptor pairs with a positive gap.
    pub perturbation_total: Option<f64>,
    pub nonpositive_gaps: usize,
    pub nlmos: Vec<Nlmo>,
    /// NAO coefficients of the NLMOs (columns, in Lewis NBO order).
    pub nlmo_coefficients: Matrix,
    /// Largest Lewis/non-Lewis density element after the rotations.
    pub nlmo_residual: f64,
    pub nlmo_sweeps: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Seed {
    Core,
    Lone,
    Bond,
    Three,
}

struct Candidate {
    seed: Seed,
    atoms: Vec<usize>,
    vector: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn quadratic(d: &Matrix, a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .enumerate()
        .filter(|(_, x)| **x != 0.0)
        .map(|(i, x)| {
            x * b
                .iter()
                .enumerate()
                .map(|(j, y)| d[(i, j)] * y)
                .sum::<f64>()
        })
        .sum()
}

fn block(d: &Matrix, index: &[usize]) -> Matrix {
    let mut out = Matrix::zeros(index.len(), index.len());
    for (i, &p) in index.iter().enumerate() {
        for (j, &q) in index.iter().enumerate() {
            out[(i, j)] = d[(p, q)];
        }
    }
    out
}

fn embed(n: usize, index: &[usize], local: &Matrix, column: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for (i, &p) in index.iter().enumerate() {
        v[p] = local[(i, column)];
    }
    v
}

/// Orthonormal accepted orbitals and the depletion of a density.
#[derive(Default)]
struct Accepted {
    vectors: Vec<Vec<f64>>,
}

impl Accepted {
    /// Adds the part of `v` orthogonal to the accepted orbitals and
    /// returns it normalized, or `None` when `v` depends on them.
    fn add(&mut self, v: &[f64]) -> Option<Vec<f64>> {
        let mut w = v.to_vec();
        for _ in 0..2 {
            for u in &self.vectors {
                let projection = dot(u, &w);
                for (x, y) in w.iter_mut().zip(u) {
                    *x -= projection * y;
                }
            }
        }
        let norm = dot(&w, &w).sqrt();
        if norm > 1e-8 {
            w.iter_mut().for_each(|x| *x /= norm);
            self.vectors.push(w.clone());
            Some(w)
        } else {
            None
        }
    }

    fn deplete(&self, d: &Matrix) -> Matrix {
        let n = d.rows;
        let mut q = Matrix::identity(n);
        for u in &self.vectors {
            for (i, ui) in u.iter().enumerate() {
                if *ui == 0.0 {
                    continue;
                }
                for (j, uj) in u.iter().enumerate() {
                    q[(i, j)] -= ui * uj;
                }
            }
        }
        q.matmul(&d.matmul(&q))
    }
}

/// Eigenvalues (ascending) and eigenvectors of the density block `d`. A
/// group of degenerate occupancies (within 1e-10 electrons) is rotated to
/// diagonalize the Fock block `f`, the lowest Fock energy last (the
/// consumers read the vectors from the end), so that the orbitals of a
/// degenerate group, empty Rydberg orbitals in particular, do not depend on
/// the orientation of the molecule.
fn natural_with_ties(d: &Matrix, f: Option<&Matrix>) -> Result<(Vec<f64>, Matrix)> {
    let (values, mut vectors) = symmetric_eigen(d)?;
    let Some(f) = f else {
        return Ok((values, vectors));
    };
    let n = values.len();
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && values[end] - values[start] <= 1e-10 {
            end += 1;
        }
        if end - start > 1 {
            let group: Vec<usize> = (start..end).collect();
            let mut v = Matrix::zeros(n, group.len());
            for (c, &k) in group.iter().enumerate() {
                for i in 0..n {
                    v[(i, c)] = vectors[(i, k)];
                }
            }
            let (_, r) = symmetric_eigen(&v.transpose_matmul(&f.matmul(&v)))?;
            let rotated = v.matmul(&r);
            let size = group.len();
            for (c, &k) in group.iter().enumerate() {
                for i in 0..n {
                    vectors[(i, k)] = rotated[(i, size - 1 - c)];
                }
            }
        }
        start = end;
    }
    Ok((values, vectors))
}

struct Search<'a> {
    density: &'a Matrix,
    fock: Option<&'a Matrix>,
    atom_orbitals: &'a [Vec<usize>],
    threshold: f64,
    accepted: Accepted,
    depleted: Matrix,
    /// Deplete the density after every accepted block (core and lone
    /// pairs, three-centre bonds); two-centre bonds of different pairs are
    /// all searched in the density depleted of the one-centre orbitals, so
    /// that equivalent bonds receive equivalent hybrids.
    deplete: bool,
    /// Largest number of Lewis hybrids per atom (its core and valence
    /// NAOs) and the hybrids placed so far.
    capacity: Vec<usize>,
    hybrids: Vec<usize>,
    candidates: Vec<Candidate>,
}

impl Search<'_> {
    /// Accepts the leading eigenvectors of the depleted block of `atoms`:
    /// those above the threshold, or exactly `count` of them.
    fn block(&mut self, atoms: &[usize], seed: Seed, count: Option<usize>) -> Result<()> {
        let index: Vec<usize> = atoms
            .iter()
            .flat_map(|&a| self.atom_orbitals[a].iter().copied())
            .collect();
        if index.is_empty() {
            return Ok(());
        }
        let fock = self.fock.map(|f| block(f, &index));
        let (values, vectors) = natural_with_ties(&block(&self.depleted, &index), fock.as_ref())?;
        let mut found = false;
        let mut taken = 0;
        for k in (0..index.len()).rev() {
            let enough = match count {
                Some(count) => taken >= count,
                None => values[k] <= self.threshold,
            };
            if enough {
                break;
            }
            if count.is_none() && atoms.iter().any(|&a| self.hybrids[a] >= self.capacity[a]) {
                // A further hybrid would exceed the natural minimal basis of
                // an atom (two bonds sharing one hydrogen hybrid).
                break;
            }
            let v = embed(self.density.rows, &index, &vectors, k);
            // A prescribed count may reach eigenvectors of the depleted null
            // space, which can contain accepted orbitals: those are skipped
            // and the rest orthogonalized to them.
            let vector = match (count, self.accepted.add(&v)) {
                (Some(_), Some(w)) => w,
                (Some(_), None) => continue,
                (None, _) => v,
            };
            self.candidates.push(Candidate {
                seed,
                atoms: atoms.to_vec(),
                vector,
            });
            for &a in atoms {
                self.hybrids[a] += 1;
            }
            found = true;
            taken += 1;
        }
        if found && self.deplete {
            self.depleted = self.accepted.deplete(self.density);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn search(
    density: &Matrix,
    fock: Option<&Matrix>,
    atom_orbitals: &[Vec<usize>],
    core: &[Vec<usize>],
    capacity: &[usize],
    threshold: f64,
    three_center: bool,
) -> Result<Vec<Candidate>> {
    let n = density.rows;
    let natoms = atom_orbitals.len();
    let mut accepted = Accepted::default();
    let mut candidates = Vec::new();
    for (atom, list) in core.iter().enumerate() {
        for &i in list {
            let mut v = vec![0.0; n];
            v[i] = 1.0;
            accepted.add(&v);
            candidates.push(Candidate {
                seed: Seed::Core,
                atoms: vec![atom],
                vector: v,
            });
        }
    }
    let depleted = accepted.deplete(density);
    let mut state = Search {
        density,
        fock,
        atom_orbitals,
        threshold,
        accepted,
        depleted,
        deplete: true,
        capacity: capacity.to_vec(),
        hybrids: core.iter().map(Vec::len).collect(),
        candidates,
    };
    for atom in 0..natoms {
        state.block(&[atom], Seed::Lone, None)?;
    }
    let coupling = |a: usize, b: usize| -> f64 {
        let mut sum = 0.0;
        for &i in &atom_orbitals[a] {
            for &j in &atom_orbitals[b] {
                sum += density[(i, j)].powi(2);
            }
        }
        sum
    };
    let mut pairs: Vec<(f64, usize, usize)> = (0..natoms)
        .flat_map(|a| (a + 1..natoms).map(move |b| (a, b)))
        .map(|(a, b)| (coupling(a, b), a, b))
        .collect();
    pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
    state.deplete = false;
    for &(_, a, b) in &pairs {
        state.block(&[a, b], Seed::Bond, None)?;
    }
    state.depleted = state.accepted.deplete(density);
    state.deplete = true;
    if three_center {
        let mut triples: Vec<(f64, usize, usize, usize)> = Vec::new();
        for a in 0..natoms {
            for b in a + 1..natoms {
                for c in b + 1..natoms {
                    triples.push((coupling(a, b) + coupling(a, c) + coupling(b, c), a, b, c));
                }
            }
        }
        triples.sort_by(|x, y| y.0.total_cmp(&x.0));
        for &(_, a, b, c) in &triples {
            state.block(&[a, b, c], Seed::Three, None)?;
        }
    }
    Ok(state.candidates)
}

/// Squared density coupling of the NAO blocks of two atoms.
fn coupling(density: &Matrix, atom_orbitals: &[Vec<usize>], a: usize, b: usize) -> f64 {
    let mut sum = 0.0;
    for &i in &atom_orbitals[a] {
        for &j in &atom_orbitals[b] {
            sum += density[(i, j)].powi(2);
        }
    }
    sum
}

/// NAO vectors of the Lewis orbitals of a prescribed structure: the core
/// NAOs, `lone[a]` lone pairs on atom `a` and `order` bonds of every listed
/// pair `(a, b, order)`, each the leading eigenvectors of its block of the
/// depleted density whatever their occupancy (pairs in descending
/// coupling), completed and orthogonalized as hybrids.
pub(super) fn prescribed_lewis(
    density: &Matrix,
    fock: Option<&Matrix>,
    atom_orbitals: &[Vec<usize>],
    core: &[Vec<usize>],
    lone: &[usize],
    bonds: &[(usize, usize, usize)],
) -> Result<Vec<Vec<f64>>> {
    let n = density.rows;
    let mut accepted = Accepted::default();
    let mut candidates = Vec::new();
    for (atom, list) in core.iter().enumerate() {
        for &i in list {
            let mut v = vec![0.0; n];
            v[i] = 1.0;
            accepted.add(&v);
            candidates.push(Candidate {
                seed: Seed::Core,
                atoms: vec![atom],
                vector: v,
            });
        }
    }
    let capacity = vec![usize::MAX; atom_orbitals.len()];
    let depleted = accepted.deplete(density);
    let mut state = Search {
        density,
        fock,
        atom_orbitals,
        threshold: f64::INFINITY,
        accepted,
        depleted,
        deplete: true,
        capacity: capacity.to_vec(),
        hybrids: core.iter().map(Vec::len).collect(),
        candidates,
    };
    // Bonds first, all from the density depleted of the cores, then the
    // lone pairs from the density depleted of everything placed: a forced
    // lone pair takes what the bonds leave (the pi orbital of a carbanion)
    // instead of the leading sigma direction of its atom.
    let mut ordered: Vec<(f64, usize, usize, usize)> = bonds
        .iter()
        .map(|&(a, b, order)| (coupling(density, atom_orbitals, a, b), a, b, order))
        .collect();
    ordered.sort_by(|x, y| y.0.total_cmp(&x.0));
    state.deplete = false;
    for &(_, a, b, order) in &ordered {
        if order > 0 {
            state.block(&[a, b], Seed::Bond, Some(order))?;
        }
    }
    state.depleted = state.accepted.deplete(density);
    state.deplete = true;
    for (atom, &count) in lone.iter().enumerate() {
        if count > 0 {
            state.block(&[atom], Seed::Lone, Some(count))?;
        }
    }
    let built = build(&state.candidates, density, fock, atom_orbitals)?;
    Ok((0..n)
        .filter(|&j| built.kinds[j].is_lewis())
        .map(|j| (0..n).map(|i| built.coefficients[(i, j)]).collect())
        .collect())
}

/// NBOs of one Lewis structure: kinds, atoms, NAO vectors and the hybrids
/// (atom, normalized hybrid, coefficient).
struct Built {
    kinds: Vec<NboKind>,
    atoms: Vec<Vec<usize>>,
    hybrids: Vec<Vec<(usize, Vec<f64>, f64)>>,
    coefficients: Matrix,
}

fn build(
    candidates: &[Candidate],
    density: &Matrix,
    fock: Option<&Matrix>,
    atom_orbitals: &[Vec<usize>],
) -> Result<Built> {
    let n = density.rows;
    let natoms = atom_orbitals.len();
    let mut per_atom: Vec<Vec<(usize, Vec<f64>)>> = vec![Vec::new(); natoms];
    for (ci, candidate) in candidates.iter().enumerate() {
        for &atom in &candidate.atoms {
            let local: Vec<f64> = atom_orbitals[atom]
                .iter()
                .map(|&i| candidate.vector[i])
                .collect();
            let norm = dot(&local, &local).sqrt();
            if norm < 1e-10 {
                return Err(DftError::InvalidInput(format!(
                    "a natural bond orbital candidate has no component on atom {atom}"
                )));
            }
            per_atom[atom].push((ci, local.iter().map(|x| x / norm).collect()));
        }
    }
    let mut candidate_hybrids: Vec<Vec<(usize, Vec<f64>)>> = vec![Vec::new(); candidates.len()];
    let mut rydberg: Vec<(usize, Vec<f64>)> = Vec::new();
    for atom in 0..natoms {
        let index = &atom_orbitals[atom];
        let size = index.len();
        let count = per_atom[atom].len();
        if count > size {
            return Err(DftError::InvalidInput(format!(
                "the Lewis structure places {count} hybrids on atom {atom}, which has {size} NAOs"
            )));
        }
        let mut orthonormal = Matrix::zeros(size, count);
        if count > 0 {
            let mut h = Matrix::zeros(size, count);
            for (j, (_, local)) in per_atom[atom].iter().enumerate() {
                for (i, value) in local.iter().enumerate() {
                    h[(i, j)] = *value;
                }
            }
            let gram = h.transpose_matmul(&h);
            orthonormal = h.matmul(&inverse_square_root(&gram, "the hybrid metric of an atom")?);
            for (j, (ci, _)) in per_atom[atom].iter().enumerate() {
                candidate_hybrids[*ci].push((atom, embed(n, index, &orthonormal, j)));
            }
        }
        if count < size {
            let mut projector = Matrix::identity(size);
            let outer = orthonormal.matmul_transpose_seq(&orthonormal);
            for (p, o) in projector.as_mut_slice().iter_mut().zip(outer.as_slice()) {
                *p -= o;
            }
            let (values, vectors) = symmetric_eigen(&projector)?;
            let kept: Vec<usize> = (0..size).filter(|&k| values[k] > 0.5).collect();
            if kept.len() != size - count {
                return Err(DftError::LinearAlgebra(format!(
                    "the complement of the hybrids on atom {atom} has dimension {} instead of {}",
                    kept.len(),
                    size - count
                )));
            }
            let mut complement = Matrix::zeros(size, kept.len());
            for (c, &k) in kept.iter().enumerate() {
                for i in 0..size {
                    complement[(i, c)] = vectors[(i, k)];
                }
            }
            let local_density =
                complement.transpose_matmul(&block(density, index).matmul(&complement));
            let local_fock =
                fock.map(|f| complement.transpose_matmul(&block(f, index).matmul(&complement)));
            let (_, natural) = natural_with_ties(&local_density, local_fock.as_ref())?;
            let ordered = complement.matmul(&natural);
            for k in (0..kept.len()).rev() {
                rydberg.push((atom, embed(n, index, &ordered, k)));
            }
        }
    }
    let mut lewis: Vec<(NboKind, Vec<usize>, Vec<(usize, Vec<f64>, f64)>)> = Vec::new();
    let mut non_lewis: Vec<(NboKind, Vec<usize>, Vec<(usize, Vec<f64>, f64)>)> = Vec::new();
    for (candidate, hybrids) in candidates.iter().zip(candidate_hybrids) {
        match candidate.seed {
            Seed::Core | Seed::Lone => {
                let kind = if candidate.seed == Seed::Core {
                    NboKind::Core
                } else {
                    NboKind::LonePair
                };
                let (atom, h) = hybrids.into_iter().next().unwrap();
                lewis.push((kind, vec![atom], vec![(atom, h, 1.0)]));
            }
            Seed::Bond | Seed::Three => {
                let mut hybrids = hybrids;
                // Signs so that the couplings to the first hybrid are not
                // negative.
                let (first, rest) = hybrids.split_at_mut(1);
                for (_, h) in rest.iter_mut() {
                    if quadratic(density, &first[0].1, h) < 0.0 {
                        h.iter_mut().for_each(|x| *x = -*x);
                    }
                }
                let size = hybrids.len();
                let mut m = Matrix::zeros(size, size);
                for (i, (_, hi)) in hybrids.iter().enumerate() {
                    for (j, (_, hj)) in hybrids.iter().enumerate() {
                        m[(i, j)] = quadratic(density, hi, hj);
                    }
                }
                let (_, vectors) = symmetric_eigen(&m)?;
                let atoms: Vec<usize> = hybrids.iter().map(|(a, _)| *a).collect();
                for rank in 0..size {
                    let k = size - 1 - rank;
                    let sign = if vectors[(0, k)] < 0.0 { -1.0 } else { 1.0 };
                    let parts: Vec<(usize, Vec<f64>, f64)> = hybrids
                        .iter()
                        .enumerate()
                        .map(|(i, (a, h))| (*a, h.clone(), sign * vectors[(i, k)]))
                        .collect();
                    let entry = (
                        match (candidate.seed, rank) {
                            (Seed::Bond, 0) => NboKind::Bond,
                            (Seed::Bond, _) => NboKind::Antibond,
                            (_, 0) => NboKind::ThreeCenter,
                            _ => NboKind::ThreeCenterAntibond,
                        },
                        atoms.clone(),
                        parts,
                    );
                    if rank == 0 {
                        lewis.push(entry);
                    } else {
                        non_lewis.push(entry);
                    }
                }
            }
        }
    }
    for (atom, h) in rydberg {
        non_lewis.push((NboKind::Rydberg, vec![atom], vec![(atom, h, 1.0)]));
    }
    let all: Vec<_> = lewis.into_iter().chain(non_lewis).collect();
    if all.len() != n {
        return Err(DftError::LinearAlgebra(format!(
            "the natural bond orbitals number {} for {n} NAOs",
            all.len()
        )));
    }
    let mut coefficients = Matrix::zeros(n, n);
    let mut kinds = Vec::with_capacity(n);
    let mut atoms = Vec::with_capacity(n);
    let mut hybrids = Vec::with_capacity(n);
    for (column, (kind, on, parts)) in all.into_iter().enumerate() {
        for (_, h, c) in &parts {
            for (i, value) in h.iter().enumerate() {
                coefficients[(i, column)] += c * value;
            }
        }
        kinds.push(kind);
        atoms.push(on);
        hybrids.push(parts);
    }
    Ok(Built {
        kinds,
        atoms,
        hybrids,
        coefficients,
    })
}

/// Natural bond orbitals, NLMOs and donor-acceptor energies of `density`
/// (NAO basis) with at most `maximum_occupancy` electrons per orbital;
/// `fock` is the Fock matrix in the NAO basis.
pub fn natural_bond_orbitals(
    nao: &NaturalAtomicOrbitals,
    numbers: &[u8],
    spin: &str,
    density: &Matrix,
    fock: Option<&Matrix>,
    maximum_occupancy: f64,
    options: &NboOptions,
) -> Result<NboChannel> {
    let n = density.rows;
    let natoms = numbers.len();
    if density.cols != n
        || nao.labels.len() != n
        || fock.is_some_and(|f| f.rows != n || f.cols != n)
    {
        return Err(DftError::InvalidInput(
            "NBO density, Fock and NAO dimensions disagree".into(),
        ));
    }
    let fractions = match options.lewis_threshold {
        Some(f) if f > 0.0 && f <= 1.0 => vec![f],
        Some(f) => {
            return Err(DftError::InvalidInput(format!(
                "nbo_lewis_threshold must lie in (0, 1], not {f}"
            )))
        }
        None => vec![0.95, 0.90, 0.85, 0.80, 0.75],
    };
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
    let capacity: Vec<usize> = atom_orbitals
        .iter()
        .map(|list| {
            list.iter()
                .filter(|&&i| nao.labels[i].kind != NaoKind::Rydberg)
                .count()
        })
        .collect();
    let electrons: f64 = (0..n).map(|i| density[(i, i)]).sum();
    let expected = ((electrons / maximum_occupancy) - 1e-6).ceil().max(0.0) as usize;
    let mut trials = Vec::new();
    let mut best: Option<(bool, f64, f64, Built)> = None;
    for fraction in fractions {
        let threshold = fraction * maximum_occupancy;
        let candidates = search(
            density,
            fock,
            &atom_orbitals,
            &core,
            &capacity,
            threshold,
            options.three_center,
        )?;
        let built = match build(&candidates, density, fock, &atom_orbitals) {
            Ok(built) => built,
            Err(DftError::LinearAlgebra(_)) => {
                trials.push(LewisTrial {
                    threshold,
                    lewis_count: candidates.len(),
                    non_lewis_occupancy: electrons,
                    dependent: true,
                });
                continue;
            }
            Err(error) => return Err(error),
        };
        let lewis_count = built.kinds.iter().filter(|k| k.is_lewis()).count();
        let lewis_occupancy: f64 = (0..n)
            .filter(|&j| built.kinds[j].is_lewis())
            .map(|j| {
                let column: Vec<f64> = (0..n).map(|i| built.coefficients[(i, j)]).collect();
                quadratic(density, &column, &column)
            })
            .sum();
        let non_lewis = electrons - lewis_occupancy;
        trials.push(LewisTrial {
            threshold,
            lewis_count,
            non_lewis_occupancy: non_lewis,
            dependent: false,
        });
        let valid = lewis_count <= expected;
        let better = match &best {
            None => true,
            Some((best_valid, best_non_lewis, _, _)) => {
                (valid && !best_valid)
                    || (valid == *best_valid && non_lewis < best_non_lewis - 1e-10)
            }
        };
        if better {
            best = Some((valid, non_lewis, threshold, built));
        }
    }
    let (_, _, lewis_threshold, built) = best.ok_or_else(|| {
        DftError::LinearAlgebra(
            "the hybrids of every Lewis threshold were linearly dependent".into(),
        )
    })?;
    let b = built.coefficients;
    let metric = b.transpose_matmul(&b);
    let orthonormality_residual = metric.max_abs_diff(&Matrix::identity(n));
    let d_nbo = b.transpose_matmul(&density.matmul(&b));
    let f_nbo = fock.map(|f| b.transpose_matmul(&f.matmul(&b)));
    let lewis: Vec<usize> = (0..n).filter(|&j| built.kinds[j].is_lewis()).collect();
    let non_lewis: Vec<usize> = (0..n).filter(|&j| !built.kinds[j].is_lewis()).collect();
    // NBO records.
    let atom_of: Vec<usize> = nao.labels.iter().map(|l| l.atom).collect();
    let max_l = nao.labels.iter().map(|l| l.l).max().unwrap_or(0);
    let mut seen: Vec<(NboKind, Vec<usize>, usize)> = Vec::new();
    let mut nbos = Vec::with_capacity(n);
    for (j, ((&kind, on), parts)) in built
        .kinds
        .iter()
        .zip(&built.atoms)
        .zip(&built.hybrids)
        .enumerate()
    {
        let atoms = on.clone();
        let serial = match seen.iter_mut().find(|(k, a, _)| *k == kind && *a == atoms) {
            Some(entry) => {
                entry.2 += 1;
                entry.2
            }
            None => {
                seen.push((kind, atoms.clone(), 1));
                1
            }
        };
        let names: Vec<String> = atoms
            .iter()
            .map(|&a| {
                format!(
                    "{}{}",
                    crate::system::z_to_symbol(numbers[a]).unwrap_or("X"),
                    a + 1
                )
            })
            .collect();
        let hybrids = parts
            .iter()
            .map(|(atom, h, c)| {
                let mut composition = vec![0.0; max_l + 1];
                for (i, value) in h.iter().enumerate() {
                    if atom_of[i] == *atom {
                        composition[nao.labels[i].l] += 100.0 * value * value;
                    }
                }
                Hybrid {
                    atom: *atom,
                    coefficient: *c,
                    composition,
                }
            })
            .collect();
        nbos.push(Nbo {
            kind,
            atoms,
            serial,
            label: format!("{}({serial}) {}", kind.name(), names.join("-")),
            occupancy: d_nbo[(j, j)],
            energy: f_nbo.as_ref().map(|f| f[(j, j)]),
            hybrids,
        });
    }
    let lewis_occupancy: f64 = lewis.iter().map(|&j| d_nbo[(j, j)]).sum();
    // Second-order donor-acceptor energies.
    let mut perturbation = Vec::new();
    let mut nonpositive_gaps = 0;
    let perturbation_total = f_nbo.as_ref().map(|f| {
        let mut total = 0.0;
        for &i in &lewis {
            for &j in &non_lewis {
                let gap = f[(j, j)] - f[(i, i)];
                if gap <= 0.0 {
                    nonpositive_gaps += 1;
                    continue;
                }
                let stabilization = d_nbo[(i, i)] * f[(i, j)].powi(2) / gap;
                total += stabilization;
                if stabilization >= options.perturbation_threshold {
                    perturbation.push(DonorAcceptor {
                        donor: i,
                        acceptor: j,
                        stabilization,
                        fock_element: f[(i, j)],
                        energy_gap: gap,
                    });
                }
            }
        }
        total
    });
    perturbation.sort_by(|x, y| y.stabilization.total_cmp(&x.stabilization));
    // NLMOs by Jacobi rotations of the Lewis/non-Lewis block.
    let mut d = d_nbo.clone();
    let mut u = Matrix::identity(n);
    let off_block = |d: &Matrix| -> f64 {
        lewis
            .iter()
            .flat_map(|&i| non_lewis.iter().map(move |&j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(d[(i, j)].abs()))
    };
    let mut sweeps = 0;
    let mut residual = off_block(&d);
    while residual > 1e-12 && sweeps < 500 {
        for &i in &lewis {
            for &j in &non_lewis {
                let a = d[(i, j)];
                if a.abs() < 1e-15 {
                    continue;
                }
                let theta = 0.5 * (2.0 * a).atan2(d[(i, i)] - d[(j, j)]);
                let (s, c) = theta.sin_cos();
                for k in 0..n {
                    let (x, y) = (d[(k, i)], d[(k, j)]);
                    d[(k, i)] = c * x + s * y;
                    d[(k, j)] = -s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (d[(i, k)], d[(j, k)]);
                    d[(i, k)] = c * x + s * y;
                    d[(j, k)] = -s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (u[(k, i)], u[(k, j)]);
                    u[(k, i)] = c * x + s * y;
                    u[(k, j)] = -s * x + c * y;
                }
            }
        }
        sweeps += 1;
        residual = off_block(&d);
    }
    if residual > 1e-8 {
        return Err(DftError::LinearAlgebra(format!(
            "the Lewis and non-Lewis spaces did not separate after {sweeps} Jacobi sweeps \
             (largest coupling {residual:e})"
        )));
    }
    // The rotations give the Lewis subspace; the NLMOs are the symmetric
    // orthogonalization of the projections of their parents onto it, which
    // does not depend on the order of the rotations (equivalent bonds have
    // equivalent NLMOs).
    let rotated = b.matmul(&u);
    let count = lewis.len();
    let mut subspace = Matrix::zeros(n, count);
    let mut parents = Matrix::zeros(n, count);
    for (c, &i) in lewis.iter().enumerate() {
        for k in 0..n {
            subspace[(k, c)] = rotated[(k, i)];
            parents[(k, c)] = b[(k, i)];
        }
    }
    let projection = subspace.transpose_matmul(&parents);
    let orthogonal = projection.matmul(&inverse_square_root(
        &projection.transpose_matmul(&projection),
        "the parent NBOs projected on the Lewis subspace",
    )?);
    let nlmo_coefficients = subspace.matmul(&orthogonal);
    let occupancies = nlmo_coefficients.transpose_matmul(&density.matmul(&nlmo_coefficients));
    let mut nlmos = Vec::with_capacity(count);
    for (c, &i) in lewis.iter().enumerate() {
        let mut atomic = vec![0.0; natoms];
        let mut parent = 0.0;
        for (k, &atom) in atom_of.iter().enumerate() {
            atomic[atom] += 100.0 * nlmo_coefficients[(k, c)].powi(2);
            parent += parents[(k, c)] * nlmo_coefficients[(k, c)];
        }
        nlmos.push(Nlmo {
            parent: i,
            occupancy: occupancies[(c, c)],
            parent_percentage: 100.0 * parent * parent,
            atomic_percentages: atomic,
        });
    }
    Ok(NboChannel {
        spin: spin.to_string(),
        maximum_occupancy,
        electrons,
        expected_lewis_count: expected,
        lewis_threshold,
        trials,
        lewis_count: lewis.len(),
        lewis_occupancy,
        non_lewis_occupancy: electrons - lewis_occupancy,
        nbos,
        coefficients: b,
        orthonormality_residual,
        perturbation,
        perturbation_total,
        nonpositive_gaps,
        nlmos,
        nlmo_coefficients,
        nlmo_residual: residual,
        nlmo_sweeps: sweeps,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{natural_atomic_orbitals, ShellLayout};
    use super::*;

    /// A homonuclear two-electron bond in a minimal basis: one bond, one
    /// antibond, no donor-acceptor energy, the NLMO is the bond.
    #[test]
    fn two_center_bond_and_its_antibond() {
        let shells = vec![
            ShellLayout {
                atom: 0,
                l: 0,
                offset: 0,
            },
            ShellLayout {
                atom: 1,
                l: 0,
                offset: 1,
            },
        ];
        let s = 0.4;
        let overlap = Matrix::from_row_major(2, 2, vec![1.0, s, s, 1.0]);
        let c = 1.0 / (2.0 * (1.0 + s)).sqrt();
        let p = Matrix::from_row_major(2, 2, vec![c * c; 4]);
        let nao =
            natural_atomic_orbitals(&overlap, [&p, &p], &shells, &[1, 1], &[0, 0], &[1.0, 1.0])
                .unwrap();
        let fock = Matrix::from_row_major(2, 2, vec![-0.5, -0.3, -0.3, -0.5]);
        let channel = natural_bond_orbitals(
            &nao,
            &[1, 1],
            "total",
            &nao.density_nao,
            Some(&fock),
            2.0,
            &NboOptions::default(),
        )
        .unwrap();
        assert_eq!(channel.lewis_count, 1);
        assert_eq!(channel.nbos[0].kind, NboKind::Bond);
        assert_eq!(channel.nbos[0].label, "BD(1) H1-H2");
        assert_eq!(channel.nbos[1].kind, NboKind::Antibond);
        assert!((channel.nbos[0].occupancy - 2.0).abs() < 1e-12);
        assert!(channel.nbos[1].occupancy.abs() < 1e-12);
        for h in &channel.nbos[0].hybrids {
            assert!((h.coefficient - 0.5_f64.sqrt()).abs() < 1e-12);
            assert!((h.composition[0] - 100.0).abs() < 1e-12);
        }
        assert!(channel.orthonormality_residual < 1e-14);
        assert!((channel.nlmos[0].parent_percentage - 100.0).abs() < 1e-10);
    }
}
