// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! XC contributions to the analytic Hessian (Stage B: LDA + GGA).
//!
//! Total-derivative convention: grid points ride their parent atom, so
//! for a scalar field q the derivative operator is
//! `D_a q = (explicit d/dR_A_x) + delta_{A,owner} (d/dr)_x`.
//! With c^A_mu = delta_{A,owner} - delta_{A,atom(mu)} the AO derivative is
//! `D_a phi_mu = c^A_mu (d phi_mu/dr_x)`, which collapses every orbital
//! term into per-atom-restricted sums plus full sums.
//!
//! Pieces delivered per SCF density (fixed P) include the `skeleton`
//! contribution, a 3N x 3N matrix sum_i D_a D_b (w_i e_i). Its product rule
//! is D_aD_b(we) = (DDw)e + (Dw_a)(De_b) + (Dw_b)(De_a) + w DDe, with DDe
//! assembled over {rho_a, rho_b, s_aa, s_ab, s_bb}. GGA gradient-variable
//! second derivatives use third AO derivatives on the grid.
//!
//! The `vxc_deriv[spin][atom][axis]` contribution contains the derivative XC
//! potential matrices entering the skeleton Fock derivative and CPKS RHS.
//!
//! Becke weight second derivatives come from nested forward duals over a
//! Scalar-generic weight function (the point seeded together with its
//! owner) - exact, no finite differences.

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::eval::{eval_aos_deriv3, third_index, AoBatch};
use crate::grid::MolecularGrid;
use crate::integrals::deriv_mat::AtomDerivs;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use rayon::prelude::*;

mod weights;
pub use weights::becke_weight_second;
pub(crate) use weights::{MolecularWeights, WeightResponse};

/// One batch of grid points: positions, full and raw weights, owners.
pub(crate) struct BatchPoints<'a> {
    pub(crate) points: &'a [Vec3],
    pub(crate) weights: &'a [f64],
    pub(crate) raw_weights: &'a [f64],
    pub(crate) owners: &'a [usize],
    /// Optional periodic shell membership, indexed by local shell and point.
    pub(crate) active_shells: Option<&'a [Vec<bool>]>,
}

impl BatchPoints<'_> {
    pub(crate) fn ao_derivatives(&self, basis: &MolBasis, third: bool) -> AoBatch {
        let mut batch = eval_aos_deriv3(basis, self.points, true, true, third);
        if let Some(active) = self.active_shells {
            // A translated shell can be selected for only part of a batch.
            // Preserve the SCF's pointwise AO selection in every derivative.
            for (shell, active_points) in basis.shells.iter().zip(active) {
                let range = shell.sph_off..shell.sph_off + crate::basis::n_sph(shell.l);
                for (point, &included) in active_points.iter().enumerate() {
                    if !included {
                        for ao in range.clone() {
                            batch.phi[(point, ao)] = 0.0;
                            for matrix in batch
                                .grad
                                .iter_mut()
                                .flatten()
                                .chain(batch.hess.iter_mut().flatten())
                                .chain(batch.third.iter_mut().flatten())
                            {
                                matrix[(point, ao)] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        batch
    }
}

/// XC Hessian pieces at fixed density.
pub struct XcHessianTerms {
    /// 3N x 3N skeleton matrix sum_i D_a D_b (w_i e_i).
    pub skeleton: Matrix,
    /// D_a V_xc^spin matrices, indexed [spin][atom][axis].
    pub vxc_deriv: [Vec<AtomDerivs>; 2],
}

use crate::grid::eval::{second_index as h6, second_index as ss};

pub fn xc_hessian_terms(
    functional: &ResolvedFunctional,
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
) -> Result<XcHessianTerms> {
    let gga = match functional.family {
        Family::Lda => false,
        Family::Gga => true,
        Family::MetaGga => {
            return Err(DftError::InvalidInput(
                "xc_hessian_terms covers LDA and GGA functionals only; a meta-GGA must \
                 go through xc_hessian_terms_mgga, which carries the tau channel. \
                 analytic_hessian_full dispatches on the family, so reaching this \
                 means a caller bypassed that dispatch"
                    .to_string(),
            ))
        }
    };
    xc_hessian_terms_with(mol, mb, grid, p, gga, &|vars, _start, _npts| {
        let mut out = XcOut::default();
        functional.eval(vars, 2, &mut out)?;
        Ok(out)
    })
}

/// Kernel-injected core of `xc_hessian_terms`: the per-batch order-2
/// kernel data (e, vrho, vsigma, second derivatives) comes from `kernel`
/// instead of a functional evaluation. This lets the VV10 Hessian reuse
/// the entire AO/weight second-derivative machinery with its EFFECTIVE
/// local kernel (per-point pair-sum accumulators); the nonlocal Gram
/// corrections are then added separately (`hessian/nlc_terms.rs`).
pub(crate) fn xc_hessian_terms_with(
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
    gga: bool,
    kernel: &(dyn Fn(&GridVars, usize, usize) -> Result<XcOut> + Sync),
) -> Result<XcHessianTerms> {
    let natoms = mol.len();
    let nao = mb.nao_sph;
    let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
    let aij = MolecularGrid::becke_aij(mol);
    let weights = MolecularWeights {
        centers: &centers,
        aij: &aij,
    };
    let dim = 3 * natoms;
    let zeros3 = || -> AtomDerivs {
        [
            Matrix::zeros(nao, nao),
            Matrix::zeros(nao, nao),
            Matrix::zeros(nao, nao),
        ]
    };
    let fresh = || -> (Matrix, [Vec<AtomDerivs>; 2]) {
        (
            Matrix::zeros(dim, dim),
            [
                (0..natoms).map(|_| zeros3()).collect(),
                (0..natoms).map(|_| zeros3()).collect(),
            ],
        )
    };

    // Grid batches are independent, so this is a map-reduce over them. It
    // is worth saying explicitly: this stage was the single largest cost in
    // the analytic Hessian AND the only major one still running on one
    // thread, while the integral stages around it were already parallel.
    const BATCH: usize = 128;
    let n = grid.len();
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(BATCH)
        .map(|s| (s, (s + BATCH).min(n)))
        .collect();
    let (mut skeleton, mut vxc_deriv) = fresh();
    // Keep only a bounded wave of batch matrices alive, then sum in the
    // original grid order. Peak matrix storage scales with worker count,
    // not the total number of quadrature points; summation order is unchanged.
    let wave_size = 2 * rayon::current_num_threads().max(1);
    for wave in ranges.chunks(wave_size) {
        let partials: Vec<Result<(Matrix, [Vec<AtomDerivs>; 2])>> = wave
            .par_iter()
            .map(|&(start, end)| {
                let (mut skeleton, mut vxc_deriv) = fresh();
                let points = BatchPoints {
                    points: &grid.points[start..end],
                    weights: &grid.weights[start..end],
                    raw_weights: &grid.raw_weights[start..end],
                    owners: &grid.atom_of[start..end],
                    active_shells: None,
                };
                batch_terms(
                    start,
                    &points,
                    mb,
                    p,
                    natoms,
                    gga,
                    kernel,
                    &weights,
                    &mut skeleton,
                    &mut vxc_deriv,
                )?;
                Ok((skeleton, vxc_deriv))
            })
            .collect();
        for part in partials {
            let (sk, vd) = part?;
            for (t, v) in skeleton.as_mut_slice().iter_mut().zip(sk.as_slice()) {
                *t += *v;
            }
            for spin in 0..2 {
                for at in 0..natoms {
                    for ax in 0..3 {
                        for (t, v) in vxc_deriv[spin][at][ax]
                            .as_mut_slice()
                            .iter_mut()
                            .zip(vd[spin][at][ax].as_slice())
                        {
                            *t += *v;
                        }
                    }
                }
            }
        }
    }
    Ok(XcHessianTerms {
        skeleton,
        vxc_deriv,
    })
}

/// One grid batch's contribution, accumulated into caller-owned buffers.
///
/// `basis` is evaluated on the batch points; its shell atom labels are the
/// Hessian atoms (a periodic caller passes lattice images labelled by their
/// parent atom). `start` is the global index of the first point, passed to
/// the kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_terms(
    start: usize,
    points: &BatchPoints,
    basis: &MolBasis,
    p: [&Matrix; 2],
    natoms: usize,
    gga: bool,
    kernel: &(dyn Fn(&GridVars, usize, usize) -> Result<XcOut> + Sync),
    weights: &dyn WeightResponse,
    skeleton: &mut Matrix,
    vxc_deriv: &mut [Vec<AtomDerivs>; 2],
) -> Result<()> {
    let nao = basis.nao_sph;
    let ao_atom = basis.sph_ao_atoms();
    // AO index lists per atom, and their complements: the AO-derivative
    // coefficient c^A_mu is nonzero only on one of the two.
    let on_atom: Vec<Vec<usize>> = (0..natoms)
        .map(|a| (0..nao).filter(|&mu| ao_atom[mu] == a).collect())
        .collect();
    let not_on: Vec<Vec<usize>> = (0..natoms)
        .map(|a| (0..nao).filter(|&mu| ao_atom[mu] != a).collect())
        .collect();
    {
        let pts = points.points;
        let wts = points.weights;
        let raws = points.raw_weights;
        let owners = points.owners;
        let npts = pts.len();
        let batch = points.ao_derivatives(basis, gga);
        let g = batch.grad.as_ref().expect("gradients");
        let hess = batch.hess.as_ref().expect("hessians");
        let third = batch.third.as_ref();
        // x[s] = phi P^s rows; rho; yfull[s][u] = g_u P rows.
        let mut x = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        let mut rho = [vec![0.0; npts], vec![0.0; npts]];
        let mut yfull: Vec<[Matrix; 3]> = Vec::with_capacity(2);
        let mut yres: Vec<Vec<[Matrix; 3]>> = Vec::with_capacity(2);
        for spin in 0..2 {
            x[spin] = batch.phi.matmul_seq(p[spin]);
            for i in 0..npts {
                let mut acc = 0.0;
                for mu in 0..nao {
                    acc += x[spin][(i, mu)] * batch.phi[(i, mu)];
                }
                rho[spin][i] = acc.max(0.0);
            }
            yfull.push([
                g[0].matmul_seq(p[spin]),
                g[1].matmul_seq(p[spin]),
                g[2].matmul_seq(p[spin]),
            ]);
            // yres[at][u] = g_u * (P with all rows but atom `at` zeroed).
            // Cloning and multiplying a full nao x nao matrix per atom is
            // O(natoms * nao^2 * npts) of which all but the atom's own
            // rows are multiplications by zero. Accumulating only the
            // owned rows gives the same result at O(nao * npts) total.
            let mut per_atom: Vec<[Matrix; 3]> = (0..natoms)
                .map(|_| {
                    [
                        Matrix::zeros(npts, nao),
                        Matrix::zeros(npts, nao),
                        Matrix::zeros(npts, nao),
                    ]
                })
                .collect();
            for mu in 0..nao {
                let at = ao_atom[mu];
                for u in 0..3 {
                    let dst = &mut per_atom[at][u];
                    for i in 0..npts {
                        let gv = g[u][(i, mu)];
                        if gv == 0.0 {
                            continue;
                        }
                        for nu in 0..nao {
                            let pv = p[spin][(mu, nu)];
                            if pv != 0.0 {
                                dst[(i, nu)] += gv * pv;
                            }
                        }
                    }
                }
            }
            yres.push(per_atom);
        }
        // sigma on the batch (gr = grad rho components).
        let mut gr = [
            [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]],
            [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]],
        ];
        for spin in 0..2 {
            for u in 0..3 {
                for i in 0..npts {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        acc += x[spin][(i, mu)] * g[u][(i, mu)];
                    }
                    gr[spin][u][i] = 2.0 * acc;
                }
            }
        }
        let sigma_store;
        let vars = if gga {
            let mut s3 = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
            for i in 0..npts {
                let ga = [gr[0][0][i], gr[0][1][i], gr[0][2][i]];
                let gb = [gr[1][0][i], gr[1][1][i], gr[1][2][i]];
                s3[0][i] = ga.iter().map(|v| v * v).sum();
                s3[1][i] = ga.iter().zip(&gb).map(|(a, b)| a * b).sum();
                s3[2][i] = gb.iter().map(|v| v * v).sum();
            }
            sigma_store = s3;
            GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: Some([&sigma_store[0], &sigma_store[1], &sigma_store[2]]),
                tau: None,
            }
        } else {
            GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: None,
                tau: None,
            }
        };
        let out = kernel(&vars, start, npts)?;

        // Per-point scratch, allocated ONCE per batch and cleared per
        // point. Allocating these inside the point loop dominated the whole
        // analytic Hessian: `hu_at` alone is natoms^2 x 72 doubles, so at
        // twelve atoms it was ~83 kB of fresh zeroed heap for every single
        // grid point. The GGA-only buffers stay empty for LDA, where they
        // are never read.
        let gga_n = if gga { natoms } else { 0 };
        // Per-point coefficients of the deferred rank-one term, indexed
        // [(spin*natoms + atom)*3 + axis][point]; LDA path only.
        let lda_n = if gga { 0 } else { 2 * natoms * 3 };
        let mut rank1 = vec![vec![0.0; npts]; lda_n];
        let mut q = vec![[[0.0; 3]; 2]; natoms];
        let mut hx_at = vec![[[0.0; 6]; 2]; natoms];
        let mut gy_at = vec![[[[0.0; 3]; 3]; 2]; natoms];
        let mut hy_at = vec![[[[0.0; 3]; 6]; 2]; natoms];
        let mut tx_at = vec![[[0.0; 10]; 2]; natoms];
        let mut ddw_all = vec![vec![[[0.0; 3]; 3]; natoms]; natoms];
        let mut gu_full = vec![[[[0.0; 3]; 3]; 2]; natoms];
        let mut gu_at = vec![vec![[[[0.0; 3]; 3]; 2]; natoms]; natoms];
        let mut hu_full = vec![[[[0.0; 6]; 3]; 2]; gga_n];
        let mut hu_at = vec![vec![[[[0.0; 6]; 3]; 2]; gga_n]; gga_n];
        for i in 0..npts {
            let w = wts[i];
            let raw = raws[i];
            let owner = owners[i];
            let e_i = out.e[i];
            let vr = [out.vrho[0][i], out.vrho[1][i]];
            let vs = if gga {
                [out.vsigma[0][i], out.vsigma[1][i], out.vsigma[2][i]]
            } else {
                [0.0; 3]
            };
            // 5x5 symmetric kernel over (rho_a, rho_b, s_aa, s_ab, s_bb).
            let mut ker = [[0.0; 5]; 5];
            ker[0][0] = out.v2rho2[0][i];
            ker[0][1] = out.v2rho2[1][i];
            ker[1][1] = out.v2rho2[2][i];
            if gga {
                for pch in 0..3 {
                    ker[0][2 + pch] = out.v2rhosigma[pch][i];
                    ker[1][2 + pch] = out.v2rhosigma[3 + pch][i];
                    for qch in pch..3 {
                        ker[2 + pch][2 + qch] = out.v2sigma2[ss(pch, qch)][i];
                    }
                }
            }
            for r in 0..5 {
                for c in 0..r {
                    ker[r][c] = ker[c][r];
                }
            }
            // ---- per-point restricted sums ----
            // q[G][s][x] = sum_{mu in G} g_x x_mu; full grad rho = gr.
            // hx: h-contraction with x -> hx_at[G][s][h6], hx_full[s][h6].
            // gy: g_x with yfull_u -> gy_at[G][s][u][x], gy_full[s][u][x].
            // hy: h_{xy} with yfull_u -> hy_at[G][s][h6][u], hy_full.
            // t3x: third derivs with x -> tx_at[G][s][t10], tx_full[s][t10].
            q.iter_mut().for_each(|v| *v = [[0.0; 3]; 2]);
            hx_at.iter_mut().for_each(|v| *v = [[0.0; 6]; 2]);
            gy_at.iter_mut().for_each(|v| *v = [[[0.0; 3]; 3]; 2]);
            hy_at.iter_mut().for_each(|v| *v = [[[0.0; 3]; 6]; 2]);
            tx_at.iter_mut().for_each(|v| *v = [[0.0; 10]; 2]);
            gu_at
                .iter_mut()
                .for_each(|r| r.iter_mut().for_each(|v| *v = [[[0.0; 3]; 3]; 2]));
            let mut hx_full = [[0.0; 6]; 2];
            let mut gy_full = [[[0.0; 3]; 3]; 2];
            let mut hy_full = [[[0.0; 3]; 6]; 2];
            let mut tx_full = [[0.0; 10]; 2];
            for spin in 0..2 {
                for mu in 0..nao {
                    let xm = x[spin][(i, mu)];
                    let at = ao_atom[mu];
                    if xm != 0.0 {
                        for ax in 0..3 {
                            let gv = g[ax][(i, mu)] * xm;
                            q[at][spin][ax] += gv;
                        }
                        for hh in 0..6 {
                            let hv = hess[hh][(i, mu)] * xm;
                            hx_at[at][spin][hh] += hv;
                            hx_full[spin][hh] += hv;
                        }
                        if let Some(t3) = third {
                            for tt in 0..10 {
                                let tv = t3[tt][(i, mu)] * xm;
                                tx_at[at][spin][tt] += tv;
                                tx_full[spin][tt] += tv;
                            }
                        }
                    }
                    if gga {
                        for u in 0..3 {
                            let yv = yfull[spin][u][(i, mu)];
                            if yv == 0.0 {
                                continue;
                            }
                            for ax in 0..3 {
                                let gv = g[ax][(i, mu)] * yv;
                                gy_at[at][spin][u][ax] += gv;
                                gy_full[spin][u][ax] += gv;
                            }
                            for hh in 0..6 {
                                let hv = hess[hh][(i, mu)] * yv;
                                hy_at[at][spin][hh][u] += hv;
                                hy_full[spin][hh][u] += hv;
                            }
                        }
                    }
                }
            }
            // hu: h_{h6}(nu) against uvec^{A,x}(nu) =
            //     dAo yfull[s][x](nu) - yres[s][A][x](nu):
            // hu_full[s][A][x][h6], hu_at[s][A][x][B][h6].
            if gga {
                hu_full.iter_mut().for_each(|v| *v = [[[0.0; 6]; 3]; 2]);
                hu_at
                    .iter_mut()
                    .for_each(|r| r.iter_mut().for_each(|v| *v = [[[0.0; 6]; 3]; 2]));
                for spin in 0..2 {
                    for a_at in 0..natoms {
                        let d_ao = a_at == owner;
                        for ax in 0..3 {
                            for nu in 0..nao {
                                let u_v = if d_ao {
                                    yfull[spin][ax][(i, nu)] - yres[spin][a_at][ax][(i, nu)]
                                } else {
                                    -yres[spin][a_at][ax][(i, nu)]
                                };
                                if u_v == 0.0 {
                                    continue;
                                }
                                let b_at = ao_atom[nu];
                                for hh in 0..6 {
                                    let hv = hess[hh][(i, nu)] * u_v;
                                    hu_full[a_at][spin][ax][hh] += hv;
                                    hu_at[a_at][b_at][spin][ax][hh] += hv;
                                }
                            }
                        }
                    }
                }
            }

            // The skeleton''s DDrho needs sum_nu c^B_nu g_y(nu) u^{A,x}(nu),
            // which depends on (A,x) only through u and on (B,y) only
            // through g. Rebuilding it inside the (A,x,B,y) quadruple loop
            // cost one nao sweep per element of the 3N x 3N block at every
            // grid point; hoisting it here makes the loop a table lookup.
            for spin in 0..2 {
                for a_at in 0..natoms {
                    let d_ao = a_at == owner;
                    for ax in 0..3 {
                        let mut full = [0.0; 3];
                        for nu in 0..nao {
                            let u_v = if d_ao {
                                yfull[spin][ax][(i, nu)] - yres[spin][a_at][ax][(i, nu)]
                            } else {
                                -yres[spin][a_at][ax][(i, nu)]
                            };
                            if u_v == 0.0 {
                                continue;
                            }
                            let b_at = ao_atom[nu];
                            for by in 0..3 {
                                let t = g[by][(i, nu)] * u_v;
                                full[by] += t;
                                gu_at[a_at][b_at][spin][ax][by] += t;
                            }
                        }
                        gu_full[a_at][spin][ax] = full;
                    }
                }
            }
            // D_a rho_s.
            let drho = |spin: usize, at: usize, ax: usize| -> f64 {
                let own = if at == owner { gr[spin][ax][i] } else { 0.0 };
                own - 2.0 * q[at][spin][ax]
            };
            // D_a (grad rho_s)_u.
            let dgr = |spin: usize, at: usize, ax: usize, u: usize| -> f64 {
                let mut v = -2.0 * (gy_at[at][spin][u][ax] + hx_at[at][spin][h6(ax, u)]);
                if at == owner {
                    v += 2.0 * (gy_full[spin][u][ax] + hx_full[spin][h6(ax, u)]);
                }
                v
            };
            // D_a sigma_p for p = aa, ab, bb.
            let dsig = |pch: usize, at: usize, ax: usize| -> f64 {
                let (s, t) = [(0, 0), (0, 1), (1, 1)][pch];
                let mut v = 0.0;
                for u in 0..3 {
                    v += dgr(s, at, ax, u) * gr[t][u][i] + gr[s][u][i] * dgr(t, at, ax, u);
                }
                v
            };
            // 5-vector of variable first derivatives.
            let dvars = |at: usize, ax: usize| -> [f64; 5] {
                let mut v = [0.0; 5];
                v[0] = drho(0, at, ax);
                v[1] = drho(1, at, ax);
                if gga {
                    for pch in 0..3 {
                        v[2 + pch] = dsig(pch, at, ax);
                    }
                }
                v
            };
            // ---- weight first derivatives (total, owner-riding) ----
            // A point that sits entirely inside its own Becke cell has
            // w = 1 EXACTLY in f64: every other cell function contains a
            // factor 0.5(1 - f) with f = 1 to machine precision, and the
            // three Newton iterations make the smoothing polynomial flat
            // there, so both derivative orders vanish identically. Most
            // grid points are of this kind, and skipping them is exact --
            // not a tolerance. It matters: the second-derivative call is
            // O(natoms^2) in nested duals and was being made once per
            // ATOM PAIR per point, i.e. O(natoms^4) overall.
            let becke_w = if raw != 0.0 { w / raw } else { 1.0 };
            let w_flat = (becke_w - 1.0).abs() < 1e-14;
            let weight_derivatives = (!w_flat).then(|| weights.derivatives(pts[i], owner));
            let dw = |at: usize, ax: usize| -> f64 {
                weight_derivatives
                    .as_ref()
                    .map_or(0.0, |(first, _)| raw * first[3 * at + ax])
            };
            // ---- Vxc derivative matrices ----
            for spin in 0..2 {
                let (own_ch, cross_ch) = if spin == 0 { (0usize, 1usize) } else { (2, 1) };
                let other = 1 - spin;
                let rowbase = if spin == 0 { 0 } else { 3 };
                for at in 0..natoms {
                    for ax in 0..3 {
                        let dw_v = dw(at, ax);
                        let dv = dvars(at, ax);
                        // D_a vrho_s and D_a vsigma_p.
                        let mut dvr = ker[spin][0] * dv[0] + ker[spin][1] * dv[1];
                        if gga {
                            for pch in 0..3 {
                                dvr += out.v2rhosigma[rowbase + pch][i] * dv[2 + pch];
                            }
                        }
                        let mut dvs = [0.0; 3];
                        if gga {
                            for pch in 0..3 {
                                dvs[pch] = out.v2rhosigma[pch][i] * dv[0]
                                    + out.v2rhosigma[3 + pch][i] * dv[1];
                                for qch in 0..3 {
                                    dvs[pch] += out.v2sigma2[ss(pch, qch)][i] * dv[2 + qch];
                                }
                            }
                        }
                        // b vector and its derivative.
                        let mut bvec = [0.0; 3];
                        let mut dbvec = [0.0; 3];
                        if gga {
                            for u in 0..3 {
                                bvec[u] = 2.0 * vs[own_ch] * gr[spin][u][i]
                                    + vs[cross_ch] * gr[other][u][i];
                                dbvec[u] = 2.0 * dvs[own_ch] * gr[spin][u][i]
                                    + 2.0 * vs[own_ch] * dgr(spin, at, ax, u)
                                    + dvs[cross_ch] * gr[other][u][i]
                                    + vs[cross_ch] * dgr(other, at, ax, u);
                            }
                        }
                        let m = &mut vxc_deriv[spin][at][ax];
                        if !gga {
                            // LDA fast path. Two structural facts turn the
                            // nao^2 sweep that dominated the whole analytic
                            // Hessian into a pair of sparse ones:
                            //   * (Dw) vrho phi phi + w (Dvrho) phi phi is a
                            //     SCALAR times the same rank-one phi phi^T
                            //     for every (spin, atom, axis), so it is
                            //     deferred to one weighted Gram matmul per
                            //     batch (after this loop);
                            //   * the rest carries c^A_mu = delta_{A,owner}
                            //     - delta_{A,atom(mu)}, which vanishes unless
                            //     mu sits on `at` (or, when `at` owns the
                            //     point, unless it does not).
                            rank1[(spin * natoms + at) * 3 + ax][i] = dw_v * vr[spin] + w * dvr;
                            let live: &[usize] = if at == owner {
                                &not_on[at]
                            } else {
                                &on_atom[at]
                            };
                            let wv = w * vr[spin];
                            for &mu in live {
                                let c_mu = (at == owner) as i32 - (ao_atom[mu] == at) as i32;
                                let cg = wv * (c_mu as f64) * g[ax][(i, mu)];
                                for nu in 0..nao {
                                    // The nu-derivative term is the exact
                                    // TRANSPOSE of the mu-derivative one
                                    // (swap the labels in
                                    // w vrho c_nu phi_mu g_nu), so one value
                                    // serves both entries.
                                    let t = cg * batch.phi[(i, nu)];
                                    m[(mu, nu)] += t;
                                    m[(nu, mu)] += t;
                                }
                            }
                            continue;
                        }
                        for mu in 0..nao {
                            let pm = batch.phi[(i, mu)];
                            let gm = g[ax][(i, mu)];
                            let c_mu = (at == owner) as i32 - (ao_atom[mu] == at) as i32;
                            for nu in 0..nao {
                                let pn = batch.phi[(i, nu)];
                                let gn = g[ax][(i, nu)];
                                let c_nu = (at == owner) as i32 - (ao_atom[nu] == at) as i32;
                                // (Dw)(vrho phi phi) + w (Dvrho) phi phi
                                let mut v = (dw_v * vr[spin] + w * dvr) * pm * pn;
                                // w vrho D(phi phi)
                                if c_mu != 0 {
                                    v += w * vr[spin] * (c_mu as f64) * gm * pn;
                                }
                                if c_nu != 0 {
                                    v += w * vr[spin] * (c_nu as f64) * pm * gn;
                                }
                                if gga {
                                    for u in 0..3 {
                                        let gpm = g[u][(i, mu)];
                                        let gpn = g[u][(i, nu)];
                                        // (Dw) b + w Db on (grad phi phi + ...)
                                        v +=
                                            (dw_v * bvec[u] + w * dbvec[u]) * (gpm * pn + pm * gpn);
                                        // w b . D(grad phi phi + phi grad phi)
                                        let mut dterm = 0.0;
                                        if c_mu != 0 {
                                            dterm += (c_mu as f64)
                                                * (hess[h6(ax, u)][(i, mu)] * pn + gm * gpn);
                                        }
                                        if c_nu != 0 {
                                            dterm += (c_nu as f64)
                                                * (gpm * gn + pm * hess[h6(ax, u)][(i, nu)]);
                                        }
                                        if dterm != 0.0 {
                                            v += w * bvec[u] * dterm;
                                        }
                                    }
                                }
                                if v != 0.0 {
                                    m[(mu, nu)] += v;
                                }
                            }
                        }
                    }
                }
            }
            // ---- skeleton second derivative ----
            let grad1 = [vr[0], vr[1], vs[0], vs[1], vs[2]];
            let de = |dv: &[f64; 5]| -> f64 {
                let mut v = 0.0;
                for k in 0..5 {
                    v += grad1[k] * dv[k];
                }
                v
            };
            // DD(grad rho_s)_u for the pair (A,x),(B,y).
            let ddgr =
                |spin: usize, a_at: usize, ax: usize, b_at: usize, by: usize, u: usize| -> f64 {
                    let d_ao = a_at == owner;
                    let d_bo = b_at == owner;
                    // term1: c^A c^B same-mu with h_{xy} . yfull_u.
                    let mut t1 = 0.0;
                    {
                        let hyf = hy_full[spin][h6(ax, by)][u];
                        let hya = hy_at[a_at][spin][h6(ax, by)][u];
                        let hyb = hy_at[b_at][spin][h6(ax, by)][u];
                        if d_ao && d_bo {
                            t1 += hyf;
                        }
                        if d_ao {
                            t1 -= hyb;
                        }
                        if d_bo {
                            t1 -= hya;
                        }
                        if a_at == b_at {
                            t1 += hya;
                        }
                    }
                    // term4: c^A c^B same-nu with t3_{xyu} . x.
                    let mut t4 = 0.0;
                    {
                        let tt = third_index(ax, by, u);
                        let txf = tx_full[spin][tt];
                        let txa = tx_at[a_at][spin][tt];
                        let txb = tx_at[b_at][spin][tt];
                        if d_ao && d_bo {
                            t4 += txf;
                        }
                        if d_ao {
                            t4 -= txb;
                        }
                        if d_bo {
                            t4 -= txa;
                        }
                        if a_at == b_at {
                            t4 += txa;
                        }
                    }
                    // term2: uvec^{A,x} against c^B_nu h_{yu}.
                    let t2 = {
                        let full = hu_full[a_at][spin][ax][h6(by, u)];
                        let res = hu_at[a_at][b_at][spin][ax][h6(by, u)];
                        if d_bo {
                            full - res
                        } else {
                            -res
                        }
                    };
                    // term3: uvec^{B,y} against c^A_nu h_{xu}.
                    let t3v = {
                        let full = hu_full[b_at][spin][by][h6(ax, u)];
                        let res = hu_at[b_at][a_at][spin][by][h6(ax, u)];
                        if d_ao {
                            full - res
                        } else {
                            -res
                        }
                    };
                    2.0 * (t1 + t2 + t3v + t4)
                };
            // All D_A D_B w blocks for this point.
            for row in ddw_all.iter_mut() {
                for v in row.iter_mut() {
                    *v = [[0.0; 3]; 3];
                }
            }
            if let Some((_, second)) = weight_derivatives.as_ref() {
                let dim = 3 * natoms;
                for (aa, row) in ddw_all.iter_mut().enumerate() {
                    for (bb, blk) in row.iter_mut().enumerate() {
                        for x in 0..3 {
                            for y in 0..3 {
                                blk[x][y] = second[(3 * aa + x) * dim + 3 * bb + y];
                            }
                        }
                    }
                }
            }
            for a_at in 0..natoms {
                for b_at in a_at..natoms {
                    let ddw = ddw_all[a_at][b_at];
                    for ax in 0..3 {
                        let by0 = if a_at == b_at { ax } else { 0 };
                        for by in by0..3 {
                            let dva = dvars(a_at, ax);
                            let dvb = dvars(b_at, by);
                            let mut v = raw * ddw[ax][by] * e_i;
                            v += dw(a_at, ax) * de(&dvb);
                            v += dw(b_at, by) * de(&dva);
                            // kernel-quadratic part.
                            let mut dde = 0.0;
                            for r in 0..5 {
                                for c in 0..5 {
                                    dde += ker[r][c] * dva[r] * dvb[c];
                                }
                            }
                            // vrho . DDrho.
                            for s in 0..2 {
                                let mut t1 = 0.0;
                                {
                                    let hf = hx_full[s][h6(ax, by)];
                                    let ha = hx_at[a_at][s][h6(ax, by)];
                                    let hb = hx_at[b_at][s][h6(ax, by)];
                                    let d_ao = a_at == owner;
                                    let d_bo = b_at == owner;
                                    if d_ao && d_bo {
                                        t1 += hf;
                                    }
                                    if d_ao {
                                        t1 -= hb;
                                    }
                                    if d_bo {
                                        t1 -= ha;
                                    }
                                    if a_at == b_at {
                                        t1 += ha;
                                    }
                                }
                                // cross: sum_nu c^B_nu g_y(nu) uvec^{A,x}(nu),
                                // precomputed once per (spin, A, x) above.
                                let cross_full = gu_full[a_at][s][ax][by];
                                let cross_b = gu_at[a_at][b_at][s][ax][by];
                                let d_bo = b_at == owner;
                                let cross = if d_bo { cross_full - cross_b } else { -cross_b };
                                let ddrho = 2.0 * t1 + 2.0 * cross;
                                dde += vr[s] * ddrho;
                            }
                            // vsigma . DDsigma.
                            if gga {
                                for pch in 0..3 {
                                    let (s, t) = [(0, 0), (0, 1), (1, 1)][pch];
                                    let mut dds = 0.0;
                                    for u in 0..3 {
                                        dds += ddgr(s, a_at, ax, b_at, by, u) * gr[t][u][i]
                                            + dgr(s, a_at, ax, u) * dgr(t, b_at, by, u)
                                            + dgr(s, b_at, by, u) * dgr(t, a_at, ax, u)
                                            + gr[s][u][i] * ddgr(t, a_at, ax, b_at, by, u);
                                    }
                                    dde += vs[pch] * dds;
                                }
                            }
                            v += w * dde;
                            let (r, c) = (3 * a_at + ax, 3 * b_at + by);
                            skeleton[(r, c)] += v;
                            if r != c {
                                skeleton[(c, r)] += v;
                            }
                        }
                    }
                }
            }
        }
        // Deferred rank-one accumulation for the LDA path: one weighted
        // Gram matrix per (spin, atom, axis) instead of an nao^2 sweep at
        // every grid point.
        for (k, coef) in rank1.iter().enumerate() {
            let (rest, ax) = (k / 3, k % 3);
            let (spin, at) = (rest / natoms, rest % natoms);
            let mut scaled = batch.phi.clone();
            for (i, &ci) in coef.iter().enumerate() {
                for mu in 0..nao {
                    scaled[(i, mu)] *= ci;
                }
            }
            let gram = scaled.transpose_matmul(&batch.phi);
            let m = &mut vxc_deriv[spin][at][ax];
            for (mv, gv) in m.as_mut_slice().iter_mut().zip(gram.as_slice()) {
                *mv += *gv;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::grid::{becke_weight_gradient, GridOptions};
    use crate::system::Atom;
    use crate::xc::{resolve, FunctionalSpec};

    fn mol() -> Molecule {
        Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.05, 0.2217),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.03, 1.4309, -0.8867),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, -1.4309, -0.8867),
            },
        ])
    }

    fn bump(m: &Molecule, atom: usize, ax: usize, h: f64) -> Molecule {
        let mut mm = m.clone();
        match ax {
            0 => mm.atoms[atom].position.x += h,
            1 => mm.atoms[atom].position.y += h,
            _ => mm.atoms[atom].position.z += h,
        }
        mm
    }

    /// Nested-dual weight second derivatives vs FD of the analytic first
    /// derivative (owner-riding convention on both sides).
    #[test]
    fn weight_second_matches_fd_of_first() {
        let m = mol();
        let centers: Vec<Vec3> = m.atoms.iter().map(|a| a.position).collect();
        let aij = MolecularGrid::becke_aij(&m);
        let owner = 0usize;
        let offset = Vec3::new(0.9, -0.4, 0.7); // point rides atom 0
        let h = 1e-6;
        for a in 0..3 {
            for b in 0..3 {
                let ddw = becke_weight_second(&centers, &aij, owner, a, b, centers[owner] + offset);
                for ax in 0..3 {
                    for by in 0..3 {
                        let dfd = |mm: &Molecule| -> f64 {
                            let cs: Vec<Vec3> = mm.atoms.iter().map(|at| at.position).collect();
                            let (_, dwb) =
                                becke_weight_gradient(&cs, &aij, 3, owner, cs[owner] + offset);
                            let mut s = Vec3::zero();
                            for v in &dwb {
                                s += *v;
                            }
                            let mut v = dwb[a].get(ax);
                            if a == owner {
                                v -= s.get(ax);
                            }
                            v
                        };
                        let fd = (dfd(&bump(&m, b, by, h)) - dfd(&bump(&m, b, by, -h))) / (2.0 * h);
                        assert!(
                            (ddw[ax][by] - fd).abs() < 1e-7,
                            "ddw a={a} b={b} ({ax},{by}): {} vs FD {fd}",
                            ddw[ax][by]
                        );
                    }
                }
            }
        }
    }

    /// Fixed-P reference: E_xc and V_xc at an arbitrary geometry (grid
    /// rebuilt; points ride atoms by construction).
    fn exc_and_vxc(
        mm: &Molecule,
        func: &ResolvedFunctional,
        p: [&Matrix; 2],
        nao: usize,
        gga: bool,
    ) -> (f64, [Matrix; 2]) {
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(mm, &set).unwrap();
        let grid = MolecularGrid::build(mm, &GridOptions::default()).unwrap();
        let mut e_tot = 0.0;
        let mut v = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
        const BATCH: usize = 256;
        let n = grid.len();
        let mut start = 0;
        while start < n {
            let end = (start + BATCH).min(n);
            let pts = &grid.points[start..end];
            let wts = &grid.weights[start..end];
            let npts = pts.len();
            let batch = eval_aos_deriv3(&mb, pts, gga, false, false);
            let g = batch.grad.as_ref();
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut x = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
            let mut gr: [[Vec<f64>; 3]; 2] = Default::default();
            for spin in 0..2 {
                x[spin] = batch.phi.matmul_seq(p[spin]);
                for i in 0..npts {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        acc += x[spin][(i, mu)] * batch.phi[(i, mu)];
                    }
                    rho[spin][i] = acc.max(0.0);
                }
                if gga {
                    let g = g.expect("grad");
                    for u in 0..3 {
                        let mut vv = vec![0.0; npts];
                        for i in 0..npts {
                            let mut acc = 0.0;
                            for mu in 0..nao {
                                acc += x[spin][(i, mu)] * g[u][(i, mu)];
                            }
                            vv[i] = 2.0 * acc;
                        }
                        gr[spin][u] = vv;
                    }
                }
            }
            let sigma_store;
            let vars = if gga {
                let mut s3 = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
                for i in 0..npts {
                    let ga = [gr[0][0][i], gr[0][1][i], gr[0][2][i]];
                    let gb = [gr[1][0][i], gr[1][1][i], gr[1][2][i]];
                    s3[0][i] = ga.iter().map(|v| v * v).sum();
                    s3[1][i] = ga.iter().zip(&gb).map(|(a, b)| a * b).sum();
                    s3[2][i] = gb.iter().map(|v| v * v).sum();
                }
                sigma_store = s3;
                GridVars {
                    rho: [&rho[0], &rho[1]],
                    sigma: Some([&sigma_store[0], &sigma_store[1], &sigma_store[2]]),
                    tau: None,
                }
            } else {
                GridVars {
                    rho: [&rho[0], &rho[1]],
                    sigma: None,
                    tau: None,
                }
            };
            let mut out = XcOut::default();
            func.eval(&vars, 1, &mut out).unwrap();
            for i in 0..npts {
                e_tot += wts[i] * out.e[i];
                for spin in 0..2 {
                    let f = wts[i] * out.vrho[spin][i];
                    let mut bvec = [0.0; 3];
                    if gga {
                        let (own, cross) = if spin == 0 { (0, 1) } else { (2, 1) };
                        let other = 1 - spin;
                        for u in 0..3 {
                            bvec[u] = wts[i]
                                * (2.0 * out.vsigma[own][i] * gr[spin][u][i]
                                    + out.vsigma[cross][i] * gr[other][u][i]);
                        }
                    }
                    for mu in 0..nao {
                        let pm = batch.phi[(i, mu)];
                        for nu in 0..nao {
                            let pn = batch.phi[(i, nu)];
                            let mut val = f * pm * pn;
                            if gga {
                                let g = g.expect("grad");
                                for u in 0..3 {
                                    val += bvec[u] * (g[u][(i, mu)] * pn + pm * g[u][(i, nu)]);
                                }
                            }
                            v[spin][(mu, nu)] += val;
                        }
                    }
                }
            }
            start = end;
        }
        (e_tot, v)
    }

    fn fixed_p(nao: usize) -> Matrix {
        let mut p = Matrix::zeros(nao, nao);
        let mut state = 0xabc0_1234_u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for i in 0..nao {
            for j in i..nao {
                let v = 0.2 * rng();
                p[(i, j)] = v;
                p[(j, i)] = v;
            }
        }
        for i in 0..nao {
            p[(i, i)] += 0.7;
        }
        p
    }

    fn run_case(name: &str, gga: bool) {
        let m = mol();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&m, &set).unwrap();
        let nao = mb.nao_sph;
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let p = fixed_p(nao);
        let grid = MolecularGrid::build(&m, &GridOptions::default()).unwrap();
        let terms = xc_hessian_terms(&func, &m, &mb, &grid, [&p, &p]).unwrap();
        let h = 1e-4;
        // V_xc^(a) vs FD.
        for at in 0..3 {
            for ax in 0..3 {
                let (_, vp) = exc_and_vxc(&bump(&m, at, ax, h), &func, [&p, &p], nao, gga);
                let (_, vm) = exc_and_vxc(&bump(&m, at, ax, -h), &func, [&p, &p], nao, gga);
                for mu in 0..nao {
                    for nu in 0..nao {
                        let fd = (vp[0][(mu, nu)] - vm[0][(mu, nu)]) / (2.0 * h);
                        let an = terms.vxc_deriv[0][at][ax][(mu, nu)];
                        assert!(
                            (an - fd).abs() < 2e-5,
                            "{name} Vxc^(a) at={at} ax={ax} ({mu},{nu}): {an} vs FD {fd}"
                        );
                    }
                }
            }
        }
        // Skeleton vs double FD of E_xc (representative entries).
        for &(a, ax, b, by) in &[
            (0usize, 0usize, 0usize, 0usize),
            (0, 1, 1, 1),
            (0, 2, 1, 0),
            (1, 2, 2, 2),
            (2, 0, 2, 0),
        ] {
            let e = |mm: &Molecule| exc_and_vxc(mm, &func, [&p, &p], nao, gga).0;
            let pp = e(&bump(&bump(&m, a, ax, h), b, by, h));
            let pm = e(&bump(&bump(&m, a, ax, h), b, by, -h));
            let mp = e(&bump(&bump(&m, a, ax, -h), b, by, h));
            let mmv = e(&bump(&bump(&m, a, ax, -h), b, by, -h));
            let fd = (pp - pm - mp + mmv) / (4.0 * h * h);
            let an = terms.skeleton[(3 * a + ax, 3 * b + by)];
            assert!(
                (an - fd).abs() < 2e-4 * fd.abs().max(1.0),
                "{name} XC skeleton ({a},{ax})x({b},{by}): {an} vs FD {fd}"
            );
        }
    }

    /// LDA terms vs finite differences at fixed density.
    #[test]
    fn xc_terms_match_finite_differences_lda() {
        run_case("svwn5", false);
    }

    /// GGA terms (third-AO-derivative machinery) vs finite differences.
    #[test]
    fn xc_terms_match_finite_differences_gga() {
        run_case("blyp", true);
    }
}
