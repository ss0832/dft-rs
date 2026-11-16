// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spin-flip Tamm-Dancoff excitations of an unrestricted reference: the
//! transverse spin excitations (magnon-type modes) of a molecule or cluster.
//!
//! A spin-lowering excitation moves an alpha electron of an occupied
//! orbital `i` to an empty beta orbital `a`; a spin-raising one moves a beta
//! electron to an empty alpha orbital. With a collinear functional only the
//! exact exchange couples the flips:
//!
//! ```text
//! A(ai),(bj) = delta_ij F'_ab - delta_ab F_ij - c_x (ab|ij)
//! ```
//!
//! with `F` the Fock matrix of the spin that loses the electron and `F'` of
//! the spin that gains it. For exact exchange alone the uniform rotation of
//! the total spin is an eigenvector of the spin-lowering block of a
//! high-spin reference with energy zero (the Goldstone mode); a semilocal
//! functional with the collinear kernel does not reproduce that.

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

/// Spin-flip excitations of one direction.
#[derive(Clone, Debug)]
pub struct SpinFlipStates {
    /// Excitation energies, ascending (Hartree).
    pub energies: Vec<f64>,
    /// Amplitudes of every state, rows = empty orbitals of the gaining
    /// spin, columns = occupied orbitals of the losing spin.
    pub amplitudes: Vec<Matrix>,
    /// Largest `|A_ij - A_ji|` of the matrix as built.
    pub asymmetry: f64,
}

/// Spin-flip Tamm-Dancoff states of the unrestricted reference `scf`:
/// `lowering = true` flips alpha to beta, `false` beta to alpha.
#[allow(clippy::too_many_arguments)]
pub fn spin_flip_tda(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    opts: &ScfOptions,
    grid: Option<&crate::grid::MolecularGrid>,
    scf: &ScfResult,
    lowering: bool,
    nroots: usize,
    noncollinear: bool,
) -> Result<SpinFlipStates> {
    if !scf.converged {
        return Err(DftError::InvalidInput(
            "spin-flip response needs a converged reference".into(),
        ));
    }
    let (from, to) = if lowering { (0, 1) } else { (1, 0) };
    let occupied_count = if from == 0 { scf.n_alpha } else { scf.n_beta };
    let gaining_occupied = if to == 0 { scf.n_alpha } else { scf.n_beta };
    let c_from = &scf.mo_coeff[from];
    let c_to = &scf.mo_coeff[to];
    let no = occupied_count;
    let nv = c_to.cols - gaining_occupied;
    let dim = no * nv;
    if dim == 0 {
        return Ok(SpinFlipStates {
            energies: Vec::new(),
            amplitudes: Vec::new(),
            asymmetry: 0.0,
        });
    }
    let nao = c_from.rows;
    let operator = crate::response::spin_flip_kernel::SpinFlipOperator::new(
        mol,
        mb,
        functional,
        opts,
        grid,
        [&scf.density[0], &scf.density[1]],
        noncollinear,
    )?;
    let mut occupied = Matrix::zeros(nao, no);
    let mut empty = Matrix::zeros(nao, nv);
    for row in 0..nao {
        for i in 0..no {
            occupied[(row, i)] = c_from[(row, i)];
        }
        for a in 0..nv {
            empty[(row, a)] = c_to[(row, gaining_occupied + a)];
        }
    }
    let f_from = occupied.transpose_matmul(&scf.fock[from].matmul(&occupied));
    let f_to = empty.transpose_matmul(&scf.fock[to].matmul(&empty));
    let index = |a: usize, i: usize| a * no + i;
    let mut matrix = Matrix::zeros(dim, dim);
    for b in 0..nv {
        for j in 0..no {
            // Transition density of the unit amplitude (b, j).
            let mut t = Matrix::zeros(nao, nao);
            for mu in 0..nao {
                for nu in 0..nao {
                    t[(mu, nu)] = empty[(mu, b)] * occupied[(nu, j)];
                }
            }
            let response = operator.apply(&t)?;
            let coupled = empty.transpose_matmul(&response.matmul(&occupied));
            let col = index(b, j);
            for a in 0..nv {
                for i in 0..no {
                    let mut value = coupled[(a, i)];
                    if i == j {
                        value += f_to[(a, b)];
                    }
                    if a == b {
                        value -= f_from[(j, i)];
                    }
                    matrix[(index(a, i), col)] = value;
                }
            }
        }
    }
    let mut asymmetry = 0.0_f64;
    for p in 0..dim {
        for q in (p + 1)..dim {
            asymmetry = asymmetry.max((matrix[(p, q)] - matrix[(q, p)]).abs());
            let average = 0.5 * (matrix[(p, q)] + matrix[(q, p)]);
            matrix[(p, q)] = average;
            matrix[(q, p)] = average;
        }
    }
    let (values, vectors) = symmetric_eigen(&matrix)?;
    let count = nroots.min(dim);
    let amplitudes = (0..count)
        .map(|root| {
            let mut x = Matrix::zeros(nv, no);
            for a in 0..nv {
                for i in 0..no {
                    x[(a, i)] = vectors[(index(a, i), root)];
                }
            }
            x
        })
        .collect();
    Ok(SpinFlipStates {
        energies: values[..count].to_vec(),
        amplitudes,
        asymmetry,
    })
}
