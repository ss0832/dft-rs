// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Energies of mixed-reference spin-flip states along a nuclear path.
//!
//! The restricted open-shell triplet is followed along the path in the
//! transported frame ([`super::open_shell`]), with its rotation equations
//! solved at every order: the excitation energy is not stationary in the
//! orbitals of the reference. The moving projectors on the closed, open and
//! virtual orbitals give orthonormal orbitals of each class,
//! `U_x(t) = P_x(t) E_x (E_x^T P_x(t) E_x)^(-1/2)`; the states do not
//! depend on the rotations inside a class. The spin-flip matrices are
//! linear in the Kohn-Sham blocks and in the projected kernel responses, so
//! every Taylor coefficient of the channel matrix follows from the series
//! of those, and the eigenvalue of a non-degenerate root from perturbation
//! theory of the symmetric matrix series.
//!
//! No displaced calculation enters.

use super::open_shell::{open_shell_path, OpenShellPath};
use super::{add, matrix_product, metric_transport, MolecularDirection};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::response::krylov::ResponseSolverConfig;
use crate::response::mrsf::{assemble, Channel, MrsfCoupling, SpinFlipBlocks};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// One state of a spin channel, counted from the lowest.
#[derive(Clone, Copy, Debug)]
pub struct MixedReferenceState {
    pub singlet: bool,
    pub root: usize,
    pub coupling: MrsfCoupling,
    pub noncollinear: bool,
}

/// Energy coefficients of one mixed-reference spin-flip state along a path.
#[derive(Clone, Debug)]
pub struct MixedReferencePath {
    /// Energy coefficients of the reference triplet, Hartree.
    pub ground: Vec<f64>,
    /// Coefficients of the energy of the state relative to the reference;
    /// entry 0 is its vertical value.
    pub excitation: Vec<f64>,
    /// Distance to the nearest other root of the channel, Hartree.
    pub gap: f64,
    pub residual: f64,
    pub flag_defect: f64,
    pub stationarity_defect: f64,
    pub converged: bool,
}

fn transposed(series: &[Matrix]) -> Vec<Matrix> {
    series.iter().map(Matrix::transpose).collect()
}

/// Columns `start..start + count` of every coefficient.
fn columns(series: &[Matrix], start: usize, count: usize) -> Vec<Matrix> {
    series
        .iter()
        .map(|m| {
            let mut out = Matrix::zeros(m.rows, count);
            for row in 0..m.rows {
                for col in 0..count {
                    out[(row, col)] = m[(row, start + col)];
                }
            }
            out
        })
        .collect()
}

/// Columns of `left` followed by those of `right`, per coefficient.
fn joined(left: &[Matrix], right: &[Matrix]) -> Vec<Matrix> {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let mut out = Matrix::zeros(a.rows, a.cols + b.cols);
            for row in 0..a.rows {
                for col in 0..a.cols {
                    out[(row, col)] = a[(row, col)];
                }
                for col in 0..b.cols {
                    out[(row, a.cols + col)] = b[(row, col)];
                }
            }
            out
        })
        .collect()
}

/// Orthonormal orbitals of the class with the projector series `projector`
/// that start from the reference orbitals `start..start + count`, in the
/// transported frame.
fn class_orbitals(projector: &[Matrix], start: usize, count: usize, order: usize) -> Vec<Matrix> {
    let mut selection = Matrix::zeros(projector[0].rows, count);
    for k in 0..count {
        selection[(start + k, k)] = 1.0;
    }
    let inner = metric_transport(&selection, projector, order);
    matrix_product(projector, &inner, order)
}

/// Taylor coefficients of the eigenvalue `root` of the symmetric matrix
/// series `h`, and its distance to the nearest other eigenvalue at the
/// reference point.
fn eigenvalue_series(h: &[Matrix], root: usize) -> Result<(Vec<f64>, f64)> {
    let (values, vectors) = symmetric_eigen(&h[0])?;
    let size = values.len();
    if root >= size {
        return Err(DftError::InvalidInput(format!(
            "root {root} exceeds the {size} states of the channel"
        )));
    }
    let gap = values
        .iter()
        .enumerate()
        .filter(|(m, _)| *m != root)
        .map(|(_, e)| (e - values[root]).abs())
        .fold(f64::INFINITY, f64::min);
    if gap < 1e-6 {
        return Err(DftError::Unsupported {
            feature: "nuclear derivatives of a mixed-reference spin-flip root within 1e-6 \
                      hartree of another root of its channel"
                .into(),
            missing: "degenerate perturbation theory of the response eigenvalue".into(),
        });
    }
    let apply = |m: &Matrix, v: &[f64]| -> Vec<f64> {
        (0..size)
            .map(|i| (0..size).map(|j| m[(i, j)] * v[j]).sum())
            .collect()
    };
    let column = |m: usize| -> Vec<f64> { (0..size).map(|i| vectors[(i, m)]).collect() };
    let start = column(root);
    let mut omega = vec![values[root]];
    let mut v: Vec<Vec<f64>> = vec![start.clone()];
    for n in 1..h.len() {
        // sum_{k=1}^{n} H_k v_{n-k}.
        let mut image = vec![0.0; size];
        for k in 1..=n {
            for (t, x) in image.iter_mut().zip(apply(&h[k], &v[n - k])) {
                *t += x;
            }
        }
        let shift: f64 = start.iter().zip(&image).map(|(a, b)| a * b).sum();
        omega.push(shift);
        // (H_0 - w_0) v_n = -sum_{k=1}^{n} (H_k - w_k) v_{n-k}, v_n
        // orthogonal to the reference vector.
        let mut rhs: Vec<f64> = image.iter().map(|x| -x).collect();
        for k in 1..=n {
            for (t, x) in rhs.iter_mut().zip(&v[n - k]) {
                *t += omega[k] * x;
            }
        }
        let mut next = vec![0.0; size];
        for (m, value) in values.iter().enumerate() {
            if m == root {
                continue;
            }
            let other = column(m);
            let weight =
                other.iter().zip(&rhs).map(|(a, b)| a * b).sum::<f64>() / (value - values[root]);
            for (t, x) in next.iter_mut().zip(&other) {
                *t += weight * x;
            }
        }
        v.push(next);
    }
    Ok((omega, gap))
}

/// Excitation energy coefficients of `state` on the reference path.
fn excitation_series(
    functional: &ResolvedFunctional,
    path: &OpenShellPath,
    state: MixedReferenceState,
    order: usize,
) -> Result<(Vec<f64>, f64)> {
    let [nc, open] = path.counts;
    let nmo = path.closed[0].rows;
    if open != 2 || nmo < nc + 2 {
        return Err(DftError::InvalidInput(
            "mixed-reference spin flip needs a triplet reference with two open orbitals".into(),
        ));
    }
    let nv = nmo - nc - 2;
    let (n_o, n_v) = (nc + 2, nv + 2);
    let virtual_projector: Vec<Matrix> = (0..=order)
        .map(|n| {
            let mut q = if n == 0 {
                Matrix::identity(nmo)
            } else {
                Matrix::zeros(nmo, nmo)
            };
            add(&mut q, &path.closed[n], -1.0);
            add(&mut q, &path.open[n], -1.0);
            q
        })
        .collect();
    // Closed, open and virtual orbitals in the frame and in the AO basis.
    let frame = [
        class_orbitals(&path.closed, 0, nc, order),
        class_orbitals(&path.open, nc, 2, order),
        class_orbitals(&virtual_projector, nc + 2, nv, order),
    ];
    let ao: [Vec<Matrix>; 3] = std::array::from_fn(|x| matrix_product(&path.b, &frame[x], order));
    let frame_o = joined(&frame[0], &frame[1]);
    let frame_v = joined(&frame[1], &frame[2]);
    let c_o = joined(&ao[0], &ao[1]);
    let c_v = joined(&ao[1], &ao[2]);
    let sandwich = |left: &[Matrix], middle: &[Matrix], right: &[Matrix]| -> Vec<Matrix> {
        matrix_product(
            &matrix_product(&transposed(left), middle, order),
            right,
            order,
        )
    };
    let f_alpha = sandwich(&frame_o, &path.f[0], &frame_o);
    let f_beta = sandwich(&frame_v, &path.f[1], &frame_v);
    // Kernel response series of the transition density |left><right|.
    let respond = |left: &[Matrix], right: &[Matrix]| -> Result<Vec<Matrix>> {
        let t = matrix_product(left, &transposed(right), order);
        path.geometry
            .evaluate_spin_flip(functional, &path.p, &t, order, state.noncollinear)
    };
    let mut column_series = Vec::with_capacity(n_v * n_o);
    for s in 0..n_v {
        for r in 0..n_o {
            let response = respond(&columns(&c_v, s, 1), &columns(&c_o, r, 1))?;
            column_series.push(sandwich(&c_v, &response, &c_o));
        }
    }
    let mut closed_series: Vec<Vec<Vec<Matrix>>> = Vec::with_capacity(2);
    let mut virtual_series: Vec<Vec<Vec<Matrix>>> = Vec::with_capacity(2);
    for q in 0..2 {
        let mut closed_row = Vec::with_capacity(2);
        let mut virtual_row = Vec::with_capacity(2);
        for s in 0..2 {
            let response = respond(&columns(&ao[1], q, 1), &columns(&ao[1], s, 1))?;
            closed_row.push(sandwich(&ao[0], &response, &ao[0]));
            virtual_row.push(sandwich(&ao[2], &response, &ao[2]));
        }
        closed_series.push(closed_row);
        virtual_series.push(virtual_row);
    }
    let mut mixed_series: Vec<Vec<Vec<Matrix>>> = Vec::with_capacity(nv);
    for bv in 0..nv {
        let mut row = Vec::with_capacity(2);
        for r in 0..2 {
            let response = respond(&columns(&ao[1], r, 1), &columns(&ao[2], bv, 1))?;
            row.push(sandwich(&ao[0], &response, &ao[1]));
        }
        mixed_series.push(row);
    }
    let coefficient = |table: &[Vec<Vec<Matrix>>], n: usize| -> Vec<Vec<Matrix>> {
        table
            .iter()
            .map(|row| row.iter().map(|series| series[n].clone()).collect())
            .collect()
    };
    let channel = Channel::new(nc, nv, state.singlet, state.coupling);
    let h: Vec<Matrix> = (0..=order)
        .map(|n| {
            let blocks = SpinFlipBlocks {
                f_alpha: f_alpha[n].clone(),
                f_beta: f_beta[n].clone(),
                columns: column_series
                    .iter()
                    .map(|series| series[n].clone())
                    .collect(),
                closed: coefficient(&closed_series, n),
                virtuals: coefficient(&virtual_series, n),
                mixed: coefficient(&mixed_series, n),
            };
            let (a, u) = assemble(nc, nv, &blocks);
            channel.matrix(&a, &u)
        })
        .collect();
    eigenvalue_series(&h, state.root)
}

/// Energy of the reference triplet and of `state` relative to it along
/// `R(t) = R(0) + t direction` through `order`. `orbitals` are the common
/// orbitals of the restricted open-shell triplet `scf`.
#[allow(clippy::too_many_arguments)]
pub fn mixed_reference_path(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    state: MixedReferenceState,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<MixedReferencePath> {
    let (taylor, path) = open_shell_path(
        mol,
        mb,
        functional,
        grid,
        scf,
        orbitals,
        opts,
        &MolecularDirection::nuclear(direction.to_vec()),
        order,
        order,
        config,
    )?;
    let (excitation, gap) = excitation_series(functional, &path, state, order)?;
    let largest = |values: &[f64]| values.iter().copied().fold(0.0_f64, f64::max);
    Ok(MixedReferencePath {
        ground: taylor.energy.clone(),
        excitation,
        gap,
        residual: largest(&taylor.residuals),
        flag_defect: largest(&taylor.flag_defects),
        stationarity_defect: largest(&taylor.stationarity_defects),
        converged: taylor.converged,
    })
}

/// Vertical energy, Cartesian gradient and, on request, Hessian of the
/// total energy of one mixed-reference spin-flip state.
#[derive(Clone, Debug)]
pub struct MixedReferenceSurface {
    /// Energy of the state relative to the reference triplet, Hartree.
    pub omega: f64,
    /// Gradient of the total energy of the state, Hartree/Bohr, index
    /// `3 * atom + axis`.
    pub gradient: Vec<f64>,
    /// Hessian of the total energy of the state, Hartree/Bohr^2.
    pub hessian: Option<Matrix>,
    pub gap: f64,
    pub residual: f64,
    pub flag_defect: f64,
    pub stationarity_defect: f64,
    pub converged: bool,
    /// Number of nuclear paths evaluated.
    pub paths: usize,
}

/// First derivatives of `E_reference + omega` from the first-order path
/// coefficients of every Cartesian direction; with `with_hessian` the
/// second-order coefficients `c(u)` of those and of every pair sum give
/// `H_ii = 2 c(e_i)` and `H_ij = c(e_i + e_j) - c(e_i) - c(e_j)`.
#[allow(clippy::too_many_arguments)]
pub fn mixed_reference_surface(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    state: MixedReferenceState,
    with_hessian: bool,
    config: &ResponseSolverConfig,
) -> Result<MixedReferenceSurface> {
    let n = 3 * mol.len();
    let order = if with_hessian { 2 } else { 1 };
    let unit = |index: usize| -> Vec<[f64; 3]> {
        let mut direction = vec![[0.0; 3]; mol.len()];
        direction[index / 3][index % 3] = 1.0;
        direction
    };
    let mut paths = 0_usize;
    let mut converged = true;
    let mut gap = f64::INFINITY;
    let mut residual = 0.0_f64;
    let mut flag_defect = 0.0_f64;
    let mut stationarity_defect = 0.0_f64;
    let mut run = |direction: Vec<[f64; 3]>| -> Result<MixedReferencePath> {
        let path = mixed_reference_path(
            mol, mb, functional, grid, scf, orbitals, opts, state, &direction, order, config,
        )?;
        paths += 1;
        converged &= path.converged;
        gap = gap.min(path.gap);
        residual = residual.max(path.residual);
        flag_defect = flag_defect.max(path.flag_defect);
        stationarity_defect = stationarity_defect.max(path.stationarity_defect);
        Ok(path)
    };
    let mut omega = 0.0;
    let mut gradient = vec![0.0; n];
    let mut second = vec![0.0; n];
    let mut hessian = Matrix::zeros(n, n);
    for i in 0..n {
        let path = run(unit(i))?;
        omega = path.excitation[0];
        gradient[i] = path.ground[1] + path.excitation[1];
        if with_hessian {
            second[i] = path.ground[2] + path.excitation[2];
            hessian[(i, i)] = 2.0 * second[i];
        }
    }
    if with_hessian {
        for i in 0..n {
            for j in 0..i {
                let mut direction = unit(i);
                direction[j / 3][j % 3] += 1.0;
                let path = run(direction)?;
                let value = path.ground[2] + path.excitation[2] - second[i] - second[j];
                hessian[(i, j)] = value;
                hessian[(j, i)] = value;
            }
        }
    }
    Ok(MixedReferenceSurface {
        omega,
        gradient,
        hessian: with_hessian.then_some(hessian),
        gap,
        residual,
        flag_defect,
        stationarity_defect,
        converged,
        paths,
    })
}
