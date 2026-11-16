// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Atom-pair partition of the electrostatic and induction energies on the
//! natural localized molecular orbitals of the monomers.
//!
//! The occupied space of every monomer is spanned by its NLMOs; NLMO `k`
//! belongs to atom `a` with the fraction `w_ka` of its norm on the natural
//! atomic orbitals of `a`, and every nucleus belongs to its atom. The atom
//! densities `rho_a = sum_k w_ka l_k l_k^T` (both spins) sum to the monomer
//! density, so
//! `E_elst(a, b) = Tr[rho_a (J[rho_b] + V_b)] + Tr[rho_b V_a] + Z_a Z_b / R_ab`
//! sums to the electrostatic energy. The induction of A in the potential of
//! B, `1/2 Tr[dP_A omega^B]`, is split over the atoms `b` of B through
//! `omega^B = sum_b (V_b + J[rho_b])`, and over the atoms `a` of A through
//! the response of every NLMO, `dP_k = u_k l_k^T + l_k u_k^T` with the
//! canonical first-order amplitudes rotated to the NLMOs
//! (`T = C_o^T S L`); both sums are exact.

use crate::basis::MolBasis;
use crate::linalg::Matrix;
use crate::properties::natural_bond::{natural_bond_analysis, NboOptions, NboSpin};
use crate::response::ResponseSolver;
use crate::scf::ScfResult;
use crate::system::Molecule;
use crate::{DftError, Result};

/// NLMOs (monomer AO basis) of both spins and their atomic weights.
pub(crate) struct LocalOrbitals {
    pub orbitals: [Matrix; 2],
    /// `[spin][orbital][atom]`, fractions summing to one per orbital.
    pub weights: [Vec<Vec<f64>>; 2],
}

/// NLMOs of a monomer reference.
pub(crate) fn local_orbitals(
    mol: &Molecule,
    mb: &MolBasis,
    scf: &ScfResult,
) -> Result<LocalOrbitals> {
    let analysis = natural_bond_analysis(mol, mb, scf, NboSpin::Auto, &NboOptions::default())?;
    let c = &analysis.nao.nao_coefficients;
    let of = |channel: &crate::properties::natural_bond::NboChannel| {
        let orbitals = c.matmul(&channel.nlmo_coefficients);
        let weights: Vec<Vec<f64>> = channel
            .nlmos
            .iter()
            .map(|m| m.atomic_percentages.iter().map(|p| p / 100.0).collect())
            .collect();
        (orbitals, weights)
    };
    let ((a, wa), (b, wb)) = if analysis.channels.len() == 2 {
        (of(&analysis.channels[0]), of(&analysis.channels[1]))
    } else {
        let x = of(&analysis.channels[0]);
        (x.clone(), x)
    };
    for (s, (l, occupied)) in [(&a, scf.n_alpha), (&b, scf.n_beta)]
        .into_iter()
        .enumerate()
    {
        if l.cols != occupied {
            return Err(DftError::InvalidInput(format!(
                "the natural Lewis structure of spin {s} has {} orbitals for {occupied} \
                 occupied ones; the SAPT partition needs NLMOs spanning the occupied space",
                l.cols
            )));
        }
    }
    Ok(LocalOrbitals {
        orbitals: [a, b],
        weights: [wa, wb],
    })
}

/// Atom densities `rho_a` (both spins) of a monomer, embedded in the dimer
/// AO space at `offset`.
pub(crate) fn atom_densities(
    local: &LocalOrbitals,
    natoms: usize,
    nao_dimer: usize,
    offset: usize,
) -> Vec<Matrix> {
    let mut out = vec![Matrix::zeros(nao_dimer, nao_dimer); natoms];
    for (orbitals, weights) in local.orbitals.iter().zip(&local.weights) {
        let n = orbitals.rows;
        for (k, w) in weights.iter().enumerate() {
            for (a, &fraction) in w.iter().enumerate() {
                if fraction == 0.0 {
                    continue;
                }
                for p in 0..n {
                    let lp = orbitals[(p, k)];
                    if lp == 0.0 {
                        continue;
                    }
                    for q in 0..n {
                        out[a][(p + offset, q + offset)] += fraction * lp * orbitals[(q, k)];
                    }
                }
            }
        }
    }
    out
}

/// Induction of a monomer (its solver, reference and NLMOs) in the partner
/// potentials `potentials[b]` (monomer AO block): `[atom a][atom b]`.
pub(crate) fn induction_partition(
    solver: &ResponseSolver,
    scf: &ScfResult,
    overlap: &Matrix,
    local: &LocalOrbitals,
    natoms: usize,
    potentials: &[Matrix],
) -> Result<Matrix> {
    let total = {
        let mut m = potentials[0].clone();
        for p in &potentials[1..] {
            for (x, y) in m.as_mut_slice().iter_mut().zip(p.as_slice()) {
                *x += y;
            }
        }
        m
    };
    let (n_occ, n_vir) = solver.ov_dims();
    let spins = if scf.unrestricted { 2 } else { 1 };
    let length: usize = (0..spins).map(|s| n_occ[s] * n_vir[s]).sum();
    let mut source = vec![0.0; length];
    solver.project_ov([&total, &total], &mut source);
    let u = solver.solve_ov(&source)?;
    let nao = overlap.rows;
    let mut out = Matrix::zeros(natoms, potentials.len());
    let mut offset = 0;
    for s in 0..2 {
        let block = if spins == 2 { s } else { 0 };
        let c = &scf.mo_coeff[block];
        let (no, nv) = (n_occ[block], n_vir[block]);
        let l = &local.orbitals[s];
        let mut co = Matrix::zeros(nao, no);
        let mut cv = Matrix::zeros(nao, nv);
        for m in 0..nao {
            for i in 0..no {
                co[(m, i)] = c[(m, i)];
            }
            for a in 0..nv {
                cv[(m, a)] = c[(m, no + a)];
            }
        }
        let t = co.transpose_matmul(&overlap.matmul(l));
        let mut amplitudes = Matrix::zeros(nv, no);
        for a in 0..nv {
            for i in 0..no {
                amplitudes[(a, i)] = u[offset + a * no + i];
            }
        }
        // First-order orbitals of the NLMOs, AO x orbital.
        let first = cv.matmul(&amplitudes.matmul(&t));
        for (k, weights) in local.weights[s].iter().enumerate() {
            let mut d = Matrix::zeros(nao, nao);
            for p in 0..nao {
                for q in 0..nao {
                    d[(p, q)] = first[(p, k)] * l[(q, k)] + l[(p, k)] * first[(q, k)];
                }
            }
            for (b, potential) in potentials.iter().enumerate() {
                let value = 0.5 * d.frobenius_dot(potential);
                for (a, &fraction) in weights.iter().enumerate() {
                    out[(a, b)] += fraction * value;
                }
            }
        }
        if spins == 2 {
            offset += no * nv;
        }
    }
    Ok(out)
}
