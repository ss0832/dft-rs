// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Mixed-reference spin-flip Tamm-Dancoff states on a restricted open-shell
//! triplet.
//!
//! The reference has closed orbitals `C`, two open orbitals `O = {x, y}` and
//! virtual orbitals `V`. A spin flip takes an alpha electron of `o' = C + O`
//! to a beta orbital of `v' = O + V`. With `K(a, b; c, d) = <a| K_SF[|c><d|] |b>`
//! the response of the spin-flip Kohn-Sham operator (the exact exchange
//! `-c_x (ac|bd)` and, when requested, the non-collinear kernel of
//! [`super::spin_flip_kernel`]):
//!
//! ```text
//! A+(pq),(rs) = delta_pr F_beta,qs - delta_qs F_alpha,rp + K(q, p; s, r)
//! closed->open / closed->open:   U(iq),(js) = -K(i, j; q, s) + delta_qs sum_a K(i, j; a, a)
//! open->virtual / open->virtual: U(pa),(rb) = -K(a, b; p, r) + delta_pr sum_c K(a, b; c, c)
//! closed->open / open->virtual:  U(iq),(rb) = -K(i, q; r, b) + K(i, r; q, b)
//! ```
//!
//! `A+` is the spin-flip matrix of the `M_S = +1` reference and `U` couples
//! its configurations to those of the `M_S = -1` reference. The singlet
//! channel diagonalizes `A+ - U`, the triplet channel `A+ + U`; of the four
//! open-to-open configurations the singlet keeps `(xx - yy)/sqrt(2)`, `xy`
//! and `yx` and the triplet keeps `(xx + yy)/sqrt(2)`. Energies are measured
//! from the reference triplet.
//!
//! Both matrices are linear in the Kohn-Sham blocks and in the projected
//! kernel responses ([`SpinFlipBlocks`]), so the same assembly gives every
//! Taylor coefficient of the channel matrix along a nuclear path.

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::integrals::dipole_matrices;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

/// Coupling of the open-to-open configurations, which both references
/// share, to the paired configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MrsfCoupling {
    /// The projection of `A+` (the published method).
    Published,
    /// `sqrt(2)` times the projection, as in the configuration interaction
    /// of the union of the two spin-flip spaces.
    UnionSpace,
}

/// One state of a channel.
#[derive(Clone, Debug)]
pub struct MrsfState {
    /// Energy relative to the reference triplet (Hartree).
    pub omega: f64,
    /// `<S^2>`: twice the closed-to-virtual weight for a singlet-channel
    /// state, 2 for the triplet channel.
    pub s_squared: f64,
    /// Amplitudes, rows `v'` (open, then virtual), columns `o'` (closed,
    /// then open).
    pub amplitudes: Matrix,
    /// Transition dipole from the lowest singlet (singlets above it only).
    pub transition_dipole: Option<[f64; 3]>,
    /// Length-gauge oscillator strength from the lowest singlet.
    pub oscillator_strength: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct MrsfStates {
    pub singlets: Vec<MrsfState>,
    pub triplets: Vec<MrsfState>,
    /// Largest `|A+_ij - A+_ji|` of the spin-flip matrix as built.
    pub asymmetry: f64,
    pub closed: usize,
    pub virtuals: usize,
}

/// Kohn-Sham blocks and projected kernel responses that the spin-flip
/// matrices are linear in.
pub(crate) struct SpinFlipBlocks {
    /// Alpha Kohn-Sham matrix in `o'`.
    pub f_alpha: Matrix,
    /// Beta Kohn-Sham matrix in `v'`.
    pub f_beta: Matrix,
    /// `columns[s * n_o + r]`: the `v' x o'` block of the response to
    /// `|v'_s><o'_r|`.
    pub columns: Vec<Matrix>,
    /// `closed[q][s]`: the closed-closed block of the response to
    /// `|open_q><open_s|`.
    pub closed: Vec<Vec<Matrix>>,
    /// `virtuals[q][s]`: its virtual-virtual block.
    pub virtuals: Vec<Vec<Matrix>>,
    /// `mixed[b][r]`: the closed-open block of the response to
    /// `|open_r><virtual_b|`.
    pub mixed: Vec<Vec<Matrix>>,
}

/// `A+` as built (not symmetrized) and the pairing block `U` for `nc`
/// closed and `nv` virtual orbitals.
pub(crate) fn assemble(nc: usize, nv: usize, blocks: &SpinFlipBlocks) -> (Matrix, Matrix) {
    let (n_o, n_v) = (nc + 2, nv + 2);
    let dim = n_o * n_v;
    let index = |q: usize, p: usize| q * n_o + p;
    let mut a = Matrix::zeros(dim, dim);
    for s in 0..n_v {
        for r in 0..n_o {
            let response = &blocks.columns[s * n_o + r];
            let col = index(s, r);
            for q in 0..n_v {
                for p in 0..n_o {
                    let mut value = response[(q, p)];
                    if p == r {
                        value += blocks.f_beta[(q, s)];
                    }
                    if q == s {
                        value -= blocks.f_alpha[(r, p)];
                    }
                    a[(index(q, p), col)] = value;
                }
            }
        }
    }
    let mut u = Matrix::zeros(dim, dim);
    for q in 0..2 {
        for s in 0..2 {
            for i in 0..nc {
                for j in 0..nc {
                    let mut value = -blocks.closed[q][s][(i, j)];
                    if q == s {
                        value += blocks.closed[0][0][(i, j)] + blocks.closed[1][1][(i, j)];
                    }
                    u[(index(q, i), index(s, j))] = value;
                }
            }
            // Open orbital p = q, r = s as the hole of an open-to-virtual pair.
            for av in 0..nv {
                for bv in 0..nv {
                    let mut value = -blocks.virtuals[q][s][(av, bv)];
                    if q == s {
                        value += blocks.virtuals[0][0][(av, bv)] + blocks.virtuals[1][1][(av, bv)];
                    }
                    u[(index(2 + av, nc + q), index(2 + bv, nc + s))] = value;
                }
            }
        }
    }
    // Closed-to-open with open-to-virtual: L[r][b](i, q) = K(i, q; r, b).
    for (bv, mixed) in blocks.mixed.iter().enumerate() {
        for q in 0..2 {
            for r in 0..2 {
                for i in 0..nc {
                    let value = -mixed[r][(i, q)] + mixed[q][(i, r)];
                    u[(index(q, i), index(2 + bv, nc + r))] = value;
                    u[(index(2 + bv, nc + r), index(q, i))] = value;
                }
            }
        }
    }
    (a, u)
}

/// Basis of one spin channel: unit vectors of the paired configurations
/// and the combinations of the open-to-open ones.
pub(crate) struct Channel {
    /// Basis vectors as lists of (index, weight); the flag marks the
    /// open-to-open ones.
    basis: Vec<(Vec<(usize, f64)>, bool)>,
    sign: f64,
    scale: f64,
    dim: usize,
}

impl Channel {
    pub(crate) fn new(nc: usize, nv: usize, singlet: bool, coupling: MrsfCoupling) -> Self {
        let (n_o, n_v) = (nc + 2, nv + 2);
        let dim = n_o * n_v;
        let index = |q: usize, p: usize| q * n_o + p;
        let open_open = [
            index(0, nc),
            index(1, nc + 1),
            index(1, nc),
            index(0, nc + 1),
        ];
        let half = std::f64::consts::FRAC_1_SQRT_2;
        let mut basis: Vec<(Vec<(usize, f64)>, bool)> = (0..dim)
            .filter(|i| !open_open.contains(i))
            .map(|i| (vec![(i, 1.0)], false))
            .collect();
        if singlet {
            basis.push((vec![(open_open[0], half), (open_open[1], -half)], true));
            basis.push((vec![(open_open[2], 1.0)], true));
            basis.push((vec![(open_open[3], 1.0)], true));
        } else {
            basis.push((vec![(open_open[0], half), (open_open[1], half)], true));
        }
        Self {
            basis,
            sign: if singlet { -1.0 } else { 1.0 },
            scale: match coupling {
                MrsfCoupling::Published => 1.0,
                MrsfCoupling::UnionSpace => std::f64::consts::SQRT_2,
            },
            dim,
        }
    }

    /// Number of states of the channel.
    pub(crate) fn size(&self) -> usize {
        self.basis.len()
    }

    /// The symmetrized channel matrix of `A+` and `U`.
    pub(crate) fn matrix(&self, a: &Matrix, u: &Matrix) -> Matrix {
        let n = self.basis.len();
        let mut h = Matrix::zeros(n, n);
        for (row, (left, left_open)) in self.basis.iter().enumerate() {
            for (col, (right, right_open)) in self.basis.iter().enumerate() {
                let mut value = 0.0;
                for (i, wi) in left {
                    for (j, wj) in right {
                        value += wi * wj * (a[(*i, *j)] + self.sign * u[(*i, *j)]);
                    }
                }
                if left_open != right_open {
                    value *= self.scale;
                }
                h[(row, col)] = value;
            }
        }
        for row in 0..n {
            for col in (row + 1)..n {
                let average = 0.5 * (h[(row, col)] + h[(col, row)]);
                h[(row, col)] = average;
                h[(col, row)] = average;
            }
        }
        h
    }

    /// A channel vector in the space of all spin-flip configurations.
    pub(crate) fn embed(&self, vector: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; self.dim];
        for ((terms, _), weight) in self.basis.iter().zip(vector) {
            for (i, w) in terms {
                x[*i] += w * weight;
            }
        }
        x
    }
}

fn column(c: &Matrix, index: usize) -> Vec<f64> {
    (0..c.rows).map(|row| c[(row, index)]).collect()
}

fn outer(a: &[f64], b: &[f64]) -> Matrix {
    let mut out = Matrix::zeros(a.len(), b.len());
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[(i, j)] = x * y;
        }
    }
    out
}

fn columns(c: &Matrix, start: usize, count: usize) -> Matrix {
    let mut out = Matrix::zeros(c.rows, count);
    for row in 0..c.rows {
        for col in 0..count {
            out[(row, col)] = c[(row, start + col)];
        }
    }
    out
}

/// Mixed-reference spin-flip states of the restricted open-shell triplet
/// `reference` with the common orbitals `orbitals` (closed, open, virtual).
#[allow(clippy::too_many_arguments)]
pub fn mrsf_tda(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    opts: &ScfOptions,
    grid: Option<&crate::grid::MolecularGrid>,
    reference: &ScfResult,
    orbitals: &Matrix,
    nroots: usize,
    coupling: MrsfCoupling,
    noncollinear: bool,
) -> Result<MrsfStates> {
    let (n_alpha, n_beta) = (reference.n_alpha, reference.n_beta);
    if n_alpha != n_beta + 2 {
        return Err(DftError::InvalidInput(
            "mixed-reference spin flip needs a triplet reference with two open orbitals".into(),
        ));
    }
    if !reference.converged {
        return Err(DftError::InvalidInput(
            "mixed-reference spin flip needs a converged reference".into(),
        ));
    }
    let nmo = orbitals.cols;
    let nc = n_beta;
    let nv = nmo - n_alpha;
    let (n_o, n_v) = (nc + 2, nv + 2);
    let dim = n_o * n_v;
    if nv == 0 && nc == 0 {
        return Err(DftError::InvalidInput(
            "the spin-flip space holds only the open orbitals".into(),
        ));
    }
    // Response of the spin-flip operator to a transition density (row beta,
    // column alpha).
    let operator = crate::response::spin_flip_kernel::SpinFlipOperator::new(
        mol,
        mb,
        functional,
        opts,
        grid,
        [&reference.density[0], &reference.density[1]],
        noncollinear,
    )?;
    let kernel = |t: &Matrix| -> Result<Matrix> { operator.apply(t) };
    let c_o = columns(orbitals, 0, n_o);
    let c_v = columns(orbitals, nc, n_v);
    let c_closed = columns(orbitals, 0, nc);
    let c_open = columns(orbitals, nc, 2);
    let c_virtual = columns(orbitals, n_alpha, nv);
    let f_alpha = c_o.transpose_matmul(&reference.fock[0].matmul(&c_o));
    let f_beta = c_v.transpose_matmul(&reference.fock[1].matmul(&c_v));
    let index = |q: usize, p: usize| q * n_o + p;

    // Projected kernel responses of every spin-flip configuration, of the
    // four open-open densities and of the open-virtual ones.
    let mut column_blocks = Vec::with_capacity(dim);
    for s in 0..n_v {
        for r in 0..n_o {
            let t = outer(&column(&c_v, s), &column(&c_o, r));
            column_blocks.push(c_v.transpose_matmul(&kernel(&t)?.matmul(&c_o)));
        }
    }
    let mut closed_block: Vec<Vec<Matrix>> = Vec::with_capacity(2);
    let mut virtual_block: Vec<Vec<Matrix>> = Vec::with_capacity(2);
    for q in 0..2 {
        let mut closed_row = Vec::with_capacity(2);
        let mut virtual_row = Vec::with_capacity(2);
        for s in 0..2 {
            let response = kernel(&outer(&column(&c_open, q), &column(&c_open, s)))?;
            closed_row.push(c_closed.transpose_matmul(&response.matmul(&c_closed)));
            virtual_row.push(c_virtual.transpose_matmul(&response.matmul(&c_virtual)));
        }
        closed_block.push(closed_row);
        virtual_block.push(virtual_row);
    }
    let mut mixed_block: Vec<Vec<Matrix>> = Vec::with_capacity(nv);
    for bv in 0..nv {
        let mut row = Vec::with_capacity(2);
        for r in 0..2 {
            let response = kernel(&outer(&column(&c_open, r), &column(&c_virtual, bv)))?;
            row.push(c_closed.transpose_matmul(&response.matmul(&c_open)));
        }
        mixed_block.push(row);
    }
    let blocks = SpinFlipBlocks {
        f_alpha,
        f_beta,
        columns: column_blocks,
        closed: closed_block,
        virtuals: virtual_block,
        mixed: mixed_block,
    };
    let (a, u) = assemble(nc, nv, &blocks);
    let mut asymmetry = 0.0_f64;
    for i in 0..dim {
        for j in (i + 1)..dim {
            asymmetry = asymmetry.max((a[(i, j)] - a[(j, i)]).abs());
        }
    }

    let solve = |singlet: bool| -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let channel = Channel::new(nc, nv, singlet, coupling);
        let n = channel.size();
        let (values, vectors) = symmetric_eigen(&channel.matrix(&a, &u))?;
        let count = nroots.min(n);
        let embedded = (0..count)
            .map(|root| {
                let vector: Vec<f64> = (0..n).map(|k| vectors[(k, root)]).collect();
                channel.embed(&vector)
            })
            .collect();
        Ok((values[..count].to_vec(), embedded))
    };
    let to_matrix = |x: &[f64]| -> Matrix {
        let mut m = Matrix::zeros(n_v, n_o);
        for q in 0..n_v {
            for p in 0..n_o {
                m[(q, p)] = x[index(q, p)];
            }
        }
        m
    };
    let closed_to_virtual = |x: &[f64]| -> f64 {
        let mut weight = 0.0;
        for q in 2..n_v {
            for p in 0..nc {
                weight += x[index(q, p)] * x[index(q, p)];
            }
        }
        weight
    };
    let (singlet_energies, singlet_vectors) = solve(true)?;
    let (triplet_energies, triplet_vectors) = solve(false)?;

    // Transition dipoles from the lowest singlet: the one-body part of the
    // spin-flip bilinear form.
    let dipoles = dipole_matrices(mb, Vec3::zero());
    let dipole_o: Vec<Matrix> = dipoles
        .iter()
        .map(|d| c_o.transpose_matmul(&d.matmul(&c_o)))
        .collect();
    let dipole_v: Vec<Matrix> = dipoles
        .iter()
        .map(|d| c_v.transpose_matmul(&d.matmul(&c_v)))
        .collect();
    let mut singlets = Vec::with_capacity(singlet_energies.len());
    for (k, (omega, vector)) in singlet_energies.iter().zip(&singlet_vectors).enumerate() {
        let amplitudes = to_matrix(vector);
        let (transition_dipole, oscillator_strength) = if k == 0 {
            (None, None)
        } else {
            let ground = to_matrix(&singlet_vectors[0]);
            // M_beta = sym(X_K X_0^T), M_alpha = -sym(X_0^T X_K).
            let particle = amplitudes.matmul_transpose_seq(&ground);
            let hole = ground.transpose_matmul(&amplitudes);
            let mut moment = [0.0_f64; 3];
            for (axis, value) in moment.iter_mut().enumerate() {
                let mut sum = 0.0;
                for q in 0..n_v {
                    for s in 0..n_v {
                        sum += 0.5 * (particle[(q, s)] + particle[(s, q)]) * dipole_v[axis][(s, q)];
                    }
                }
                for p in 0..n_o {
                    for r in 0..n_o {
                        sum -= 0.5 * (hole[(p, r)] + hole[(r, p)]) * dipole_o[axis][(r, p)];
                    }
                }
                *value = sum;
            }
            let strength = 2.0 / 3.0
                * (omega - singlet_energies[0])
                * moment.iter().map(|m| m * m).sum::<f64>();
            (Some(moment), Some(strength))
        };
        singlets.push(MrsfState {
            omega: *omega,
            s_squared: 2.0 * closed_to_virtual(vector),
            amplitudes,
            transition_dipole,
            oscillator_strength,
        });
    }
    let triplets = triplet_energies
        .iter()
        .zip(&triplet_vectors)
        .map(|(omega, vector)| MrsfState {
            omega: *omega,
            s_squared: 2.0,
            amplitudes: to_matrix(vector),
            transition_dipole: None,
            oscillator_strength: None,
        })
        .collect();
    Ok(MrsfStates {
        singlets,
        triplets,
        asymmetry,
        closed: nc,
        virtuals: nv,
    })
}
