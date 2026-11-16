// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Natural bond orbital analysis of one-particle density matrices.
//!
//! Natural atomic orbitals (NAOs). For every atom and angular momentum `l`
//! the blocks of `Q = S P S` and `S` are averaged over the `2l + 1`
//! components of each shell and the generalized eigenproblem
//! `Q_l v = w S_l v` gives the pre-NAOs and their occupancies, assigned to
//! the shell slots in descending order. The first slots of each atom and
//! `l` form the natural minimal basis: the core and valence subshells of
//! the element (`ns np` for main-group elements, `ns (n-1)d` for transition
//! metals, `ns (n-1)d (n-2)f` for lanthanides and actinides, with the core
//! of the preceding noble gas, the filled `(n-1)d` and `(n-2)f` shells of
//! the later rows and without the subshells an effective core potential
//! removes); the rest is the Rydberg set. The core set is symmetrically
//! orthogonalized, the valence set is Schmidt-orthogonalized to it and
//! orthogonalized with the occupancy weights, `O = W (W M W)^(-1/2)`, and
//! the Rydberg set is Schmidt-orthogonalized to both and symmetrically
//! orthogonalized. The natural character is then restored by diagonalizing
//! the component-averaged atom and `l` blocks of the density in the
//! orthonormal basis. The NAOs are orthonormal, `C^T S C = I`, and the
//! analysis is invariant under rotations of the molecule.
//!
//! Natural population analysis sums the NAO occupancies by atom (charges
//! against the effective nuclear charges, natural spin populations from the
//! spin densities), by subshell (natural electron configurations) and gives
//! Wiberg bond indices `W_AB = 2 sum_(mu in A, nu in B) (D^a_mu,nu^2 +
//! D^b_mu,nu^2)` in the NAO basis. The natural hybrid and bond orbitals, the
//! natural localized molecular orbitals and the second-order donor-acceptor
//! energies are in [`lewis`].

pub mod lewis;
pub mod corresponding;
pub mod periodic;
pub mod resonance;
pub mod spectral;
pub mod spinor;

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::scf::ScfResult;
use crate::system::Molecule;

pub use lewis::{
    natural_bond_orbitals, DonorAcceptor, Hybrid, LewisTrial, Nbo, NboChannel, NboKind, NboOptions,
    Nlmo,
};

/// Role of a natural atomic orbital.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NaoKind {
    Core,
    Valence,
    Rydberg,
}

impl NaoKind {
    /// `core`, `valence` or `rydberg`.
    pub fn name(self) -> &'static str {
        match self {
            NaoKind::Core => "core",
            NaoKind::Valence => "valence",
            NaoKind::Rydberg => "rydberg",
        }
    }

    fn tag(self) -> &'static str {
        match self {
            NaoKind::Core => "Cor",
            NaoKind::Valence => "Val",
            NaoKind::Rydberg => "Ryd",
        }
    }
}

/// One shell of the spherical AO basis.
#[derive(Clone, Copy, Debug)]
pub struct ShellLayout {
    pub atom: usize,
    pub l: usize,
    /// Index of its first spherical AO; the `2l + 1` components follow.
    pub offset: usize,
}

/// Atom, angular momentum, component, principal quantum number and role of
/// one NAO.
#[derive(Clone, Copy, Debug)]
pub struct NaoLabel {
    pub atom: usize,
    pub l: usize,
    /// Component index `0..2l` in the order of the spherical AOs.
    pub component: usize,
    pub n: usize,
    pub kind: NaoKind,
}

impl NaoLabel {
    /// `Cor(1s)`, `Val(2p)`, `Ryd(3d)`.
    pub fn text(&self) -> String {
        format!("{}({}{})", self.kind.tag(), self.n, angular_letter(self.l))
    }
}

/// Summed occupancy of one subshell of one atom.
#[derive(Clone, Copy, Debug)]
pub struct SubshellPopulation {
    pub n: usize,
    pub l: usize,
    pub kind: NaoKind,
    pub occupancy: f64,
}

/// Natural atomic orbitals and natural population analysis.
#[derive(Clone, Debug)]
pub struct NaturalAtomicOrbitals {
    /// AO coefficients of the NAOs, `C^T S C = I`.
    pub nao_coefficients: Matrix,
    pub labels: Vec<NaoLabel>,
    /// Spin densities in the NAO basis, `C^T S P_s S C`.
    pub spin_density_nao: [Matrix; 2],
    /// Spin-summed density in the NAO basis.
    pub density_nao: Matrix,
    /// NAO occupancies (diagonal of the spin-summed density).
    pub occupations: Vec<f64>,
    /// NAO spin densities (alpha minus beta diagonal).
    pub spin_densities: Vec<f64>,
    pub orbital_atoms: Vec<usize>,
    pub atomic_populations: Vec<f64>,
    pub atomic_charges: Vec<f64>,
    pub spin_populations: Vec<f64>,
    pub core_populations: Vec<f64>,
    pub valence_populations: Vec<f64>,
    pub rydberg_populations: Vec<f64>,
    /// Subshell occupancies per atom: core, valence, Rydberg, ascending n.
    pub configurations: Vec<Vec<SubshellPopulation>>,
    /// `[core]2s(1.71)2p(5.11)3p(0.01)`: valence subshells and Rydberg
    /// subshells with at least 0.005 electrons.
    pub configuration_text: Vec<String>,
    /// Wiberg bond indices in the NAO basis, zero diagonal.
    pub wiberg_bond_indices: Matrix,
    /// Row sums of the Wiberg matrix.
    pub wiberg_atom_totals: Vec<f64>,
    /// Largest `|C^T S C - I|`.
    pub orthonormality_residual: f64,
}

impl NaturalAtomicOrbitals {
    /// NAO indices of every atom, ascending.
    pub fn atom_orbitals(&self, natoms: usize) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); natoms];
        for (index, label) in self.labels.iter().enumerate() {
            out[label.atom].push(index);
        }
        out
    }
}

/// Shells of a molecular basis.
pub fn shell_layout(basis: &MolBasis) -> Vec<ShellLayout> {
    basis
        .shells
        .iter()
        .map(|shell| ShellLayout {
            atom: shell.atom,
            l: shell.l as usize,
            offset: shell.sph_off,
        })
        .collect()
}

/// Letter of an angular momentum.
pub fn angular_letter(l: usize) -> char {
    ['s', 'p', 'd', 'f', 'g', 'h', 'i']
        .get(l)
        .copied()
        .unwrap_or('?')
}

/// Core and valence subshells `(n, l)` of the natural minimal basis of the
/// element `z` after an effective core potential removed `ecp_core`
/// electrons from the lowest closed subshells (ordered by `n`, then `l`).
pub fn minimal_basis(z: u8, ecp_core: u32) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>)> {
    const ENDS: [u8; 7] = [2, 10, 18, 36, 54, 86, 118];
    const NOBLE: [&[(usize, usize)]; 6] = [
        &[(1, 0)],
        &[(2, 0), (2, 1)],
        &[(3, 0), (3, 1)],
        &[(4, 0), (3, 2), (4, 1)],
        &[(5, 0), (4, 2), (5, 1)],
        &[(6, 0), (4, 3), (5, 2), (6, 1)],
    ];
    if z == 0 || z > 118 {
        return Err(DftError::InvalidInput(format!(
            "no natural minimal basis for atomic number {z}"
        )));
    }
    let period = ENDS.iter().position(|&end| z <= end).unwrap() + 1;
    let start = if period == 1 { 1 } else { ENDS[period - 2] + 1 };
    let position = (z - start) as usize;
    let mut core: Vec<(usize, usize)> = NOBLE[..period - 1].concat();
    let n = period;
    let valence = match period {
        1 => vec![(1, 0)],
        2 | 3 => vec![(n, 0), (n, 1)],
        4 | 5 => match position {
            0 | 1 => vec![(n, 0), (n, 1)],
            2..=11 => vec![(n, 0), (n - 1, 2)],
            _ => {
                core.push((n - 1, 2));
                vec![(n, 0), (n, 1)]
            }
        },
        _ => match position {
            0 | 1 => vec![(n, 0), (n, 1)],
            2..=16 => vec![(n, 0), (n - 1, 2), (n - 2, 3)],
            17..=25 => {
                core.push((n - 2, 3));
                vec![(n, 0), (n - 1, 2)]
            }
            _ => {
                core.push((n - 2, 3));
                core.push((n - 1, 2));
                vec![(n, 0), (n, 1)]
            }
        },
    };
    core.sort_unstable();
    let mut removed = 0u32;
    while removed < ecp_core {
        if core.is_empty() {
            break;
        }
        let (_, l) = core.remove(0);
        removed += 2 * (2 * l as u32 + 1);
    }
    if removed != ecp_core {
        return Err(DftError::InvalidInput(format!(
            "the effective core potential of element {z} removes {ecp_core} electrons, \
             which is not a sequence of closed core subshells"
        )));
    }
    Ok((core, valence))
}

fn inverse_square_root(m: &Matrix, what: &str) -> Result<Matrix> {
    let (values, vectors) = symmetric_eigen(m)?;
    let largest = values.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    if values
        .iter()
        .any(|&v| v.is_nan() || v <= 1e-14 * largest || v.is_infinite())
    {
        return Err(DftError::LinearAlgebra(format!(
            "{what} is singular or not positive definite (eigenvalues {values:?})"
        )));
    }
    let mut scaled = vectors.clone();
    for (k, value) in values.iter().enumerate() {
        let factor = 1.0 / value.sqrt();
        for i in 0..scaled.rows {
            scaled[(i, k)] *= factor;
        }
    }
    Ok(scaled.matmul_transpose_seq(&vectors))
}

fn columns(m: &Matrix, index: &[usize]) -> Matrix {
    let mut out = Matrix::zeros(m.rows, index.len());
    for (c, &j) in index.iter().enumerate() {
        for i in 0..m.rows {
            out[(i, c)] = m[(i, j)];
        }
    }
    out
}

fn set_columns(target: &mut Matrix, index: &[usize], source: &Matrix) {
    for (c, &j) in index.iter().enumerate() {
        for i in 0..target.rows {
            target[(i, j)] = source[(i, c)];
        }
    }
}

/// Symmetric orthogonalization of the columns of `c` in the metric `s`.
fn symmetric_orthogonalization(c: &Matrix, s: &Matrix, what: &str) -> Result<Matrix> {
    if c.cols == 0 {
        return Ok(c.clone());
    }
    let m = c.transpose_matmul(&s.matmul(c));
    Ok(c.matmul(&inverse_square_root(&m, what)?))
}

/// `c - b b^T s c` for `b` orthonormal in `s`.
fn schmidt(c: &Matrix, b: &Matrix, s: &Matrix) -> Matrix {
    if b.cols == 0 || c.cols == 0 {
        return c.clone();
    }
    let projection = b.transpose_matmul(&s.matmul(c));
    let mut out = c.clone();
    let removed = b.matmul(&projection);
    for (value, r) in out.as_mut_slice().iter_mut().zip(removed.as_slice()) {
        *value -= r;
    }
    out
}

/// Shells of one atom and angular momentum, ascending offsets.
struct Group {
    atom: usize,
    l: usize,
    offsets: Vec<usize>,
}

impl Group {
    fn slots(&self, ranks: std::ops::Range<usize>) -> Vec<usize> {
        let degeneracy = 2 * self.l + 1;
        ranks
            .flat_map(|rank| (0..degeneracy).map(move |k| (rank, k)))
            .map(|(rank, k)| self.offsets[rank] + k)
            .collect()
    }
}

fn groups(shells: &[ShellLayout]) -> Vec<Group> {
    let mut sorted: Vec<&ShellLayout> = shells.iter().collect();
    sorted.sort_by_key(|shell| (shell.atom, shell.l, shell.offset));
    let mut out: Vec<Group> = Vec::new();
    for shell in sorted {
        match out.last_mut() {
            Some(group) if group.atom == shell.atom && group.l == shell.l => {
                group.offsets.push(shell.offset)
            }
            _ => out.push(Group {
                atom: shell.atom,
                l: shell.l,
                offsets: vec![shell.offset],
            }),
        }
    }
    out
}

/// Component average of the shell-pair blocks of `m` in one group.
fn averaged(m: &Matrix, group: &Group) -> Matrix {
    let size = group.offsets.len();
    let degeneracy = 2 * group.l + 1;
    let mut out = Matrix::zeros(size, size);
    for (i, &oi) in group.offsets.iter().enumerate() {
        for (j, &oj) in group.offsets.iter().enumerate() {
            let sum: f64 = (0..degeneracy).map(|k| m[(oi + k, oj + k)]).sum();
            out[(i, j)] = sum / degeneracy as f64;
        }
    }
    out
}

/// Natural orbitals of the component-averaged block of `q` in one group,
/// in the metric `s` (identity when `None`), written into `rotation` and
/// `weights` at the slots of the group in descending occupancy.
fn natural_group(
    q: &Matrix,
    s: Option<&Matrix>,
    group: &Group,
    rotation: &mut Matrix,
    weights: &mut [f64],
) -> Result<()> {
    let size = group.offsets.len();
    let block = averaged(q, group);
    let (values, vectors) = match s {
        Some(s) => {
            let x = inverse_square_root(&averaged(s, group), "an atomic overlap block")?;
            let (values, u) = symmetric_eigen(&x.matmul(&block.matmul(&x)))?;
            (values, x.matmul(&u))
        }
        None => symmetric_eigen(&block)?,
    };
    let degeneracy = 2 * group.l + 1;
    for (rank, &slot) in group.offsets.iter().enumerate() {
        let source = size - 1 - rank;
        for k in 0..degeneracy {
            let column = slot + k;
            weights[column] = values[source];
            for (r, &offset) in group.offsets.iter().enumerate() {
                rotation[(offset + k, column)] = vectors[(r, source)];
            }
        }
    }
    Ok(())
}

/// Natural atomic orbitals and natural population analysis of the spin
/// densities `density` (AO basis) in the AO metric `overlap`. `numbers`
/// and `ecp_core` give the element and the electrons removed by an
/// effective core potential of every atom, `effective_charges` the charges
/// the populations are measured against.
pub fn natural_atomic_orbitals(
    overlap: &Matrix,
    density: [&Matrix; 2],
    shells: &[ShellLayout],
    numbers: &[u8],
    ecp_core: &[u32],
    effective_charges: &[f64],
) -> Result<NaturalAtomicOrbitals> {
    let n = overlap.rows;
    let natoms = numbers.len();
    let covered: usize = shells.iter().map(|shell| 2 * shell.l + 1).sum();
    if overlap.cols != n
        || density.iter().any(|m| m.rows != n || m.cols != n)
        || covered != n
        || ecp_core.len() != natoms
        || effective_charges.len() != natoms
        || shells.iter().any(|shell| shell.atom >= natoms)
    {
        return Err(DftError::InvalidInput(
            "natural atomic orbitals: overlap, density, shell, atom and charge dimensions disagree"
                .into(),
        ));
    }
    let mut total = density[0].clone();
    for (value, beta) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
        *value += beta;
    }
    let q = overlap.matmul(&total.matmul(overlap));
    let groups = groups(shells);
    // Pre-NAOs.
    let mut pre = Matrix::zeros(n, n);
    let mut weights = vec![0.0; n];
    for group in &groups {
        natural_group(&q, Some(overlap), group, &mut pre, &mut weights)?;
    }
    // Natural minimal basis per group.
    let mut minimal = Vec::with_capacity(natoms);
    for (&z, &core) in numbers.iter().zip(ecp_core) {
        minimal.push(minimal_basis(z, core)?);
    }
    let counts = |group: &Group| -> (usize, usize) {
        let (core, valence) = &minimal[group.atom];
        let size = group.offsets.len();
        let nc = core.iter().filter(|(_, l)| *l == group.l).count().min(size);
        let nv = valence
            .iter()
            .filter(|(_, l)| *l == group.l)
            .count()
            .min(size - nc);
        (nc, nv)
    };
    let (mut core_slots, mut valence_slots, mut rydberg_slots) =
        (Vec::new(), Vec::new(), Vec::new());
    for group in &groups {
        let (nc, nv) = counts(group);
        core_slots.extend(group.slots(0..nc));
        valence_slots.extend(group.slots(nc..nc + nv));
        rydberg_slots.extend(group.slots(nc + nv..group.offsets.len()));
    }
    // Core: symmetric orthogonalization.
    let core = symmetric_orthogonalization(
        &columns(&pre, &core_slots),
        overlap,
        "the core pre-NAO metric",
    )?;
    // Valence: Schmidt to the core, occupancy-weighted symmetric
    // orthogonalization. Occupancies below 1e-6 are raised to 1e-6 so that
    // an empty valence orbital (an isolated atom) keeps a regular metric.
    let valence = {
        let c = schmidt(&columns(&pre, &valence_slots), &core, overlap);
        if c.cols == 0 {
            c
        } else {
            let m = c.transpose_matmul(&overlap.matmul(&c));
            let w: Vec<f64> = valence_slots
                .iter()
                .map(|&j| weights[j].max(1e-6))
                .collect();
            let mut wmw = m.clone();
            for (i, wi) in w.iter().enumerate() {
                for (j, wj) in w.iter().enumerate() {
                    wmw[(i, j)] = wi * m[(i, j)] * wj;
                }
            }
            let mut o = inverse_square_root(&wmw, "the weighted valence pre-NAO metric")?;
            for (i, wi) in w.iter().enumerate() {
                for j in 0..o.cols {
                    o[(i, j)] *= wi;
                }
            }
            c.matmul(&o)
        }
    };
    let mut minimal_set = Matrix::zeros(n, core.cols + valence.cols);
    for i in 0..n {
        for j in 0..core.cols {
            minimal_set[(i, j)] = core[(i, j)];
        }
        for j in 0..valence.cols {
            minimal_set[(i, core.cols + j)] = valence[(i, j)];
        }
    }
    let rydberg = symmetric_orthogonalization(
        &schmidt(&columns(&pre, &rydberg_slots), &minimal_set, overlap),
        overlap,
        "the Rydberg pre-NAO metric",
    )?;
    let mut orthonormal = Matrix::zeros(n, n);
    set_columns(&mut orthonormal, &core_slots, &core);
    set_columns(&mut orthonormal, &valence_slots, &valence);
    set_columns(&mut orthonormal, &rydberg_slots, &rydberg);
    // Restoration of the natural character.
    let restored_density = orthonormal.transpose_matmul(&q.matmul(&orthonormal));
    let mut rotation = Matrix::zeros(n, n);
    let mut restored = vec![0.0; n];
    for group in &groups {
        natural_group(&restored_density, None, group, &mut rotation, &mut restored)?;
    }
    let coefficients = orthonormal.matmul(&rotation);
    let metric = coefficients.transpose_matmul(&overlap.matmul(&coefficients));
    let orthonormality_residual = metric.max_abs_diff(&Matrix::identity(n));
    // Labels.
    let mut labels = vec![
        NaoLabel {
            atom: 0,
            l: 0,
            component: 0,
            n: 0,
            kind: NaoKind::Rydberg,
        };
        n
    ];
    for group in &groups {
        let (core_list, valence_list) = &minimal[group.atom];
        let of_l = |list: &[(usize, usize)]| -> Vec<usize> {
            list.iter()
                .filter(|(_, l)| *l == group.l)
                .map(|(n, _)| *n)
                .collect()
        };
        let (core_n, valence_n) = (of_l(core_list), of_l(valence_list));
        let (nc, nv) = counts(group);
        let highest = valence_list.iter().map(|(n, _)| *n).max().unwrap_or(1);
        let first_rydberg = core_n
            .iter()
            .chain(&valence_n)
            .max()
            .map_or((group.l + 1).max(highest), |n| n + 1);
        for (rank, &slot) in group.offsets.iter().enumerate() {
            let (kind, principal) = if rank < nc {
                (NaoKind::Core, core_n[rank])
            } else if rank < nc + nv {
                (NaoKind::Valence, valence_n[rank - nc])
            } else {
                (NaoKind::Rydberg, first_rydberg + rank - nc - nv)
            };
            for k in 0..2 * group.l + 1 {
                labels[slot + k] = NaoLabel {
                    atom: group.atom,
                    l: group.l,
                    component: k,
                    n: principal,
                    kind,
                };
            }
        }
    }
    let transform = |p: &Matrix| {
        coefficients.transpose_matmul(&overlap.matmul(&p.matmul(&overlap.matmul(&coefficients))))
    };
    let spin_density_nao = [transform(density[0]), transform(density[1])];
    let mut density_nao = spin_density_nao[0].clone();
    for (value, beta) in density_nao
        .as_mut_slice()
        .iter_mut()
        .zip(spin_density_nao[1].as_slice())
    {
        *value += beta;
    }
    let occupations: Vec<f64> = (0..n).map(|i| density_nao[(i, i)]).collect();
    if occupations
        .iter()
        .any(|&o| !(-1e-6..=2.0 + 1e-6).contains(&o))
    {
        return Err(DftError::InvalidInput(format!(
            "NAO occupancies fall outside [0, 2]: {occupations:?}"
        )));
    }
    let spin_densities: Vec<f64> = (0..n)
        .map(|i| spin_density_nao[0][(i, i)] - spin_density_nao[1][(i, i)])
        .collect();
    let orbital_atoms: Vec<usize> = labels.iter().map(|label| label.atom).collect();
    let mut atomic_populations = vec![0.0; natoms];
    let mut spin_populations = vec![0.0; natoms];
    let mut by_kind = [vec![0.0; natoms], vec![0.0; natoms], vec![0.0; natoms]];
    for (i, label) in labels.iter().enumerate() {
        atomic_populations[label.atom] += occupations[i];
        spin_populations[label.atom] += spin_densities[i];
        let slot = match label.kind {
            NaoKind::Core => 0,
            NaoKind::Valence => 1,
            NaoKind::Rydberg => 2,
        };
        by_kind[slot][label.atom] += occupations[i];
    }
    let atomic_charges: Vec<f64> = effective_charges
        .iter()
        .zip(&atomic_populations)
        .map(|(z, p)| z - p)
        .collect();
    // Natural electron configurations.
    let mut configurations: Vec<Vec<SubshellPopulation>> = vec![Vec::new(); natoms];
    for (i, label) in labels.iter().enumerate() {
        let list = &mut configurations[label.atom];
        match list
            .iter_mut()
            .find(|s| s.n == label.n && s.l == label.l && s.kind == label.kind)
        {
            Some(entry) => entry.occupancy += occupations[i],
            None => list.push(SubshellPopulation {
                n: label.n,
                l: label.l,
                kind: label.kind,
                occupancy: occupations[i],
            }),
        }
    }
    let order = |kind: NaoKind| match kind {
        NaoKind::Core => 0,
        NaoKind::Valence => 1,
        NaoKind::Rydberg => 2,
    };
    for list in &mut configurations {
        list.sort_by_key(|s| (order(s.kind), s.n, s.l));
    }
    let configuration_text: Vec<String> = configurations
        .iter()
        .map(|list| {
            let mut text = String::new();
            if list.iter().any(|s| s.kind == NaoKind::Core) {
                text.push_str("[core]");
            }
            for s in list {
                let shown = match s.kind {
                    NaoKind::Core => false,
                    NaoKind::Valence => true,
                    NaoKind::Rydberg => s.occupancy >= 0.005,
                };
                if shown {
                    text.push_str(&format!(
                        "{}{}({:.2})",
                        s.n,
                        angular_letter(s.l),
                        s.occupancy
                    ));
                }
            }
            text
        })
        .collect();
    // Wiberg bond indices.
    let mut wiberg = Matrix::zeros(natoms, natoms);
    for (mu, first) in labels.iter().enumerate() {
        for (nu, second) in labels.iter().enumerate() {
            let (a, b) = (first.atom, second.atom);
            if a != b {
                wiberg[(a, b)] += 2.0
                    * (spin_density_nao[0][(mu, nu)].powi(2)
                        + spin_density_nao[1][(mu, nu)].powi(2));
            }
        }
    }
    let wiberg_atom_totals: Vec<f64> = (0..natoms)
        .map(|a| (0..natoms).map(|b| wiberg[(a, b)]).sum())
        .collect();
    let [core_populations, valence_populations, rydberg_populations] = by_kind;
    Ok(NaturalAtomicOrbitals {
        nao_coefficients: coefficients,
        labels,
        spin_density_nao,
        density_nao,
        occupations,
        spin_densities,
        orbital_atoms,
        atomic_populations,
        atomic_charges,
        spin_populations,
        core_populations,
        valence_populations,
        rydberg_populations,
        configurations,
        configuration_text,
        wiberg_bond_indices: wiberg,
        wiberg_atom_totals,
        orthonormality_residual,
    })
}

/// Electrons removed by effective core potentials, per atom.
pub fn ecp_core_electrons(basis: &MolBasis, natoms: usize) -> Vec<u32> {
    let mut out = vec![0; natoms];
    for ecp in &basis.ecps {
        out[ecp.atom] = ecp.potential.n_core;
    }
    out
}

/// Spin treatment of the bond analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NboSpin {
    /// One analysis of the spin-summed density for closed shells, one per
    /// spin otherwise.
    Auto,
    /// The spin-summed density with at most two electrons per orbital.
    Total,
    /// One Lewis structure per spin, at most one electron per orbital.
    Separate,
}

impl NboSpin {
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "auto" => Ok(NboSpin::Auto),
            "total" => Ok(NboSpin::Total),
            "separate" => Ok(NboSpin::Separate),
            other => Err(DftError::InvalidInput(format!(
                "nbo_spin must be auto, total or separate, not `{other}`"
            ))),
        }
    }
}

/// NAOs, natural population analysis and the natural bond orbitals of every
/// analysed density.
#[derive(Clone, Debug)]
pub struct NaturalBondAnalysis {
    pub nao: NaturalAtomicOrbitals,
    /// `total` or `alpha` and `beta`.
    pub channels: Vec<NboChannel>,
    /// True when the spin densities differ (open shell or broken symmetry).
    pub open_shell: bool,
}

/// Complete natural bond orbital analysis of the spin densities and Fock
/// matrices of a molecule. `fock` (AO basis, per spin) enables the
/// orbital energies and the second-order donor-acceptor energies.
#[allow(clippy::too_many_arguments)]
pub fn natural_bond_from_density(
    overlap: &Matrix,
    density: [&Matrix; 2],
    fock: Option<[&Matrix; 2]>,
    shells: &[ShellLayout],
    numbers: &[u8],
    ecp_core: &[u32],
    effective_charges: &[f64],
    spin: NboSpin,
    options: &NboOptions,
) -> Result<NaturalBondAnalysis> {
    let nao = natural_atomic_orbitals(
        overlap,
        density,
        shells,
        numbers,
        ecp_core,
        effective_charges,
    )?;
    let open_shell = nao.spin_density_nao[0].max_abs_diff(&nao.spin_density_nao[1]) > 1e-8;
    let separate = match spin {
        NboSpin::Auto => open_shell,
        NboSpin::Total => false,
        NboSpin::Separate => true,
    };
    let c = &nao.nao_coefficients;
    let to_nao = |f: &Matrix| c.transpose_matmul(&f.matmul(c));
    let mut channels = Vec::new();
    if separate {
        for (s, name) in ["alpha", "beta"].into_iter().enumerate() {
            let f = fock.map(|f| to_nao(f[s]));
            channels.push(natural_bond_orbitals(
                &nao,
                numbers,
                name,
                &nao.spin_density_nao[s],
                f.as_ref(),
                1.0,
                options,
            )?);
        }
    } else {
        // Closed shells have equal Fock matrices; otherwise their average.
        let f = fock.map(|f| {
            let mut average = to_nao(f[0]);
            let beta = to_nao(f[1]);
            for (value, b) in average.as_mut_slice().iter_mut().zip(beta.as_slice()) {
                *value = 0.5 * (*value + b);
            }
            average
        });
        channels.push(natural_bond_orbitals(
            &nao,
            numbers,
            "total",
            &nao.density_nao,
            f.as_ref(),
            2.0,
            options,
        )?);
    }
    Ok(NaturalBondAnalysis {
        nao,
        channels,
        open_shell,
    })
}

/// Natural bond orbital analysis of a converged SCF reference.
pub fn natural_bond_analysis(
    molecule: &Molecule,
    basis: &MolBasis,
    scf: &ScfResult,
    spin: NboSpin,
    options: &NboOptions,
) -> Result<NaturalBondAnalysis> {
    let numbers: Vec<u8> = molecule.atoms.iter().map(|atom| atom.z).collect();
    natural_bond_from_density(
        &scf.overlap,
        [&scf.density[0], &scf.density[1]],
        Some([&scf.fock[0], &scf.fock[1]]),
        &shell_layout(basis),
        &numbers,
        &ecp_core_electrons(basis, numbers.len()),
        &basis.effective_charges(molecule),
        spin,
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_basis_follows_the_rows_of_the_periodic_table() {
        let (core, valence) = minimal_basis(1, 0).unwrap();
        assert!(core.is_empty());
        assert_eq!(valence, vec![(1, 0)]);
        let (core, valence) = minimal_basis(8, 0).unwrap();
        assert_eq!(core, vec![(1, 0)]);
        assert_eq!(valence, vec![(2, 0), (2, 1)]);
        let (core, valence) = minimal_basis(26, 0).unwrap();
        assert_eq!(core.len(), 5);
        assert_eq!(valence, vec![(4, 0), (3, 2)]);
        let (core, _) = minimal_basis(35, 0).unwrap();
        assert!(core.contains(&(3, 2)));
        let electrons =
            |list: &[(usize, usize)]| -> usize { list.iter().map(|(_, l)| 2 * (2 * l + 1)).sum() };
        assert_eq!(electrons(&minimal_basis(36, 0).unwrap().0), 28);
        assert_eq!(electrons(&minimal_basis(79, 0).unwrap().0), 68);
        // Gold with a 60-electron effective core: 5s 5p remain in the core.
        let (core, valence) = minimal_basis(79, 60).unwrap();
        assert_eq!(core, vec![(5, 0), (5, 1)]);
        assert_eq!(valence, vec![(6, 0), (5, 2)]);
        assert!(minimal_basis(79, 61).is_err());
    }

    /// Two shells on each of two atoms: an s shell on atom 0, s and p
    /// shells on atom 1 (element 3, minimal basis 1s 2s 2p).
    fn fixture() -> (Matrix, [Matrix; 2], Vec<ShellLayout>) {
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
            ShellLayout {
                atom: 1,
                l: 0,
                offset: 2,
            },
            ShellLayout {
                atom: 1,
                l: 1,
                offset: 3,
            },
        ];
        let n = 6;
        let mut overlap = Matrix::identity(n);
        let couplings = [(0, 1, 0.2), (0, 2, 0.35), (1, 2, 0.25), (0, 5, 0.3)];
        for &(i, j, v) in &couplings {
            overlap[(i, j)] = v;
            overlap[(j, i)] = v;
        }
        // Spin densities from occupied orbitals orthonormal in the metric.
        let x = inverse_square_root(&overlap, "test").unwrap();
        let occupied = |rows: &[[f64; 6]]| -> Matrix {
            let mut c = Matrix::zeros(n, rows.len());
            for (j, row) in rows.iter().enumerate() {
                for (i, value) in row.iter().enumerate() {
                    c[(i, j)] = *value;
                }
            }
            let c = x.matmul(&c);
            let c = symmetric_orthogonalization(&c, &overlap, "test").unwrap();
            c.matmul_transpose_seq(&c)
        };
        let alpha = occupied(&[
            [0.1, 0.9, 0.1, 0.0, 0.0, 0.1],
            [0.6, 0.1, 0.5, 0.0, 0.0, 0.4],
        ]);
        let beta = occupied(&[[0.1, 0.9, 0.1, 0.0, 0.0, 0.1]]);
        (overlap, [alpha, beta], shells)
    }

    #[test]
    fn naos_are_orthonormal_and_conserve_electrons_and_spin() {
        let (overlap, density, shells) = fixture();
        let nao = natural_atomic_orbitals(
            &overlap,
            [&density[0], &density[1]],
            &shells,
            &[1, 3],
            &[0, 0],
            &[1.0, 3.0],
        )
        .unwrap();
        assert!(nao.orthonormality_residual < 1e-12);
        assert!((nao.occupations.iter().sum::<f64>() - 3.0).abs() < 1e-12);
        assert!((nao.atomic_charges.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((nao.spin_populations.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(nao.labels[1].kind, NaoKind::Core);
        assert_eq!(nao.labels[2].kind, NaoKind::Valence);
        assert_eq!(nao.labels[3].kind, NaoKind::Valence);
        assert_eq!(nao.labels[3].text(), "Val(2p)");
        for a in 0..2 {
            assert_eq!(nao.wiberg_bond_indices[(a, a)], 0.0);
        }
    }

    #[test]
    fn broken_symmetry_hydrogen_pair_has_no_bond_index() {
        // At infinite separation the alpha electron sits on one atom and the
        // beta electron on the other: no interatomic density block.
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
        let overlap = Matrix::identity(2);
        let alpha = Matrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 0.0]);
        let beta = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.0, 1.0]);
        let nao = natural_atomic_orbitals(
            &overlap,
            [&alpha, &beta],
            &shells,
            &[1, 1],
            &[0, 0],
            &[1.0, 1.0],
        )
        .unwrap();
        assert!(nao.wiberg_bond_indices[(0, 1)].abs() < 1e-14);
        assert!((nao.spin_populations[0] - 1.0).abs() < 1e-14);
        assert!((nao.spin_populations[1] + 1.0).abs() < 1e-14);
    }

    #[test]
    fn homonuclear_bond_has_unit_bond_index() {
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
        let s = 0.3;
        let overlap = Matrix::from_row_major(2, 2, vec![1.0, s, s, 1.0]);
        let c = 1.0 / (2.0 * (1.0 + s)).sqrt();
        let p = Matrix::from_row_major(2, 2, vec![c * c; 4]);
        let nao =
            natural_atomic_orbitals(&overlap, [&p, &p], &shells, &[1, 1], &[0, 0], &[1.0, 1.0])
                .unwrap();
        assert!((nao.wiberg_bond_indices[(0, 1)] - 1.0).abs() < 1e-12);
        assert!(nao.atomic_charges.iter().all(|q| q.abs() < 1e-12));
    }
}
