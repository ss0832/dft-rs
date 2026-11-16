// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! VV10 nonlocal-correlation contributions to the analytic Hessian.
//!
//! The fixed-density VV10 energy is a closed form in the grid variables
//! E = F({w_i}, {ρ_i, σ_i}, {R²_ij}) = β Σ_i w_iρ_i + ½ Σ_ij w_iw_j e_ij,
//! e_ij = ρ_iρ_j Φ(ρ_i,σ_i,ρ_j,σ_j,R²_ij). Its second nuclear derivative
//! splits into
//!  (a) a LOCAL part with the effective per-point kernel
//!      (ε^eff_i = ∂F/∂w_i, vρ_i, vσ_i and the same-slot curvatures
//!      Σ_l w_l ∂₁²e(i,l)) — evaluated by the shared GGA machinery
//!      (`xc_terms::xc_hessian_terms_with`) through a spin-mapped kernel
//!      injection, covering DDw/DDρ/DDσ, the Dw·Dε crossings, AO
//!      second/third derivatives and Becke-weight seconds; and
//!  (b) NONLOCAL Gram corrections: cross-point channel curvatures
//!      Σ_ij ∂ch_i∂ch'_j, assembled through per-point M-vectors
//!      (O(m²·3N)), plus the sparse kernel-motion R² rows/columns and
//!      ½ e_{R²} DDR² (points ride their owner atoms).
//! Every pair partial comes from ONE nested 5-variable dual evaluation of
//! the closed-form pair energy (`nlc::vv10_pair_e5`) — no hand-derived
//! kernel curvature formulas.
//!
//! Also delivered: D_a V_nlc potential-matrix corrections (the nonlocal
//! part of D_a vρ_i / D_a vσ_i contracted like the potential assembly),
//! completing F^{(a),skel} for the CPKS right-hand sides.

use crate::basis::MolBasis;
use crate::dual::DualN;
use crate::error::Result;
use crate::grid::eval::{eval_aos_full, second_index as h6};
use crate::grid::{becke_weight_gradient, MolecularGrid};
use crate::integrals::deriv_mat::AtomDerivs;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::nlc::{molecular_vv10_amplitude, vv10_pair_e5, Vv10Params, MOLECULAR_VV10_RHO_CUT};
use crate::system::Molecule;
use crate::xc::{Family, XcOut};
use rayon::prelude::*;

use super::xc_terms::xc_hessian_terms_with;

/// VV10 pieces of the analytic Hessian at fixed density.
pub struct NlcHessianTerms {
    /// 3N × 3N skeleton D_aD_b E_VV10 (weights + densities + kernel R²).
    pub skeleton: Matrix,
    /// D_a V_nlc matrices [atom][axis] (total-density channel — the same
    /// matrix enters both spin Fock derivatives).
    pub v_deriv: Vec<AtomDerivs>,
}

pub fn nlc_hessian_terms(
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
    params: Vv10Params,
) -> Result<NlcHessianTerms> {
    let n = grid.len();
    let nao = mb.nao_sph;
    let natoms = mol.len();
    let dim = 3 * natoms;
    let ao_atom = mb.sph_ao_atoms();
    let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
    let aij = MolecularGrid::becke_aij(mol);
    let p_tot = {
        let mut t = p[0].clone();
        for mu in 0..nao {
            for nu in 0..nao {
                t[(mu, nu)] += p[1][(mu, nu)];
            }
        }
        t
    };

    // ---- pass 1: rho, sigma, grad rho (total channel) ----
    let mut rho = vec![0.0; n];
    let mut sig = vec![0.0; n];
    let mut grho = vec![[0.0; 3]; n];
    const BATCH: usize = 256;
    let chunks: Vec<(usize, Vec<f64>, Vec<f64>, Vec<[f64; 3]>)> = (0..n)
        .step_by(BATCH)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&start| {
            let end = (start + BATCH).min(n);
            let pts = &grid.points[start..end];
            let npts = pts.len();
            let batch = eval_aos_full(mb, pts, true, false);
            let g = batch.grad.as_ref().expect("grads");
            let phip = batch.phi.matmul_seq(&p_tot);
            let mut r = vec![0.0; npts];
            let mut s = vec![0.0; npts];
            let mut gr = vec![[0.0; 3]; npts];
            for i in 0..npts {
                let mut acc = 0.0;
                for mu in 0..nao {
                    acc += phip[(i, mu)] * batch.phi[(i, mu)];
                }
                r[i] = acc.max(0.0);
                for ax in 0..3 {
                    let mut a = 0.0;
                    for mu in 0..nao {
                        a += phip[(i, mu)] * g[ax][(i, mu)];
                    }
                    gr[i][ax] = 2.0 * a;
                }
                s[i] = gr[i][0] * gr[i][0] + gr[i][1] * gr[i][1] + gr[i][2] * gr[i][2];
            }
            (start, r, s, gr)
        })
        .collect();
    for (start, r, s, gr) in chunks {
        let len = r.len();
        rho[start..start + len].copy_from_slice(&r);
        sig[start..start + len].copy_from_slice(&s);
        grho[start..start + len].copy_from_slice(&gr);
    }
    let idx: Vec<usize> = (0..n)
        .filter(|&i| rho[i] > MOLECULAR_VV10_RHO_CUT)
        .collect();
    let m = idx.len();
    let beta = (1.0 / 32.0) * (3.0 / (params.b * params.b)).powf(0.75);

    // ---- pass 2: fixed-P first derivatives of (w, rho, sigma) on kept
    // points, dense over the 3N displacement slots ----
    let mut dw = vec![0.0; m * dim];
    let mut dr = vec![0.0; m * dim];
    let mut ds = vec![0.0; m * dim];
    {
        let slices: Vec<(usize, Vec<f64>, Vec<f64>, Vec<f64>)> = (0..m)
            .step_by(BATCH)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&kstart| {
                let kend = (kstart + BATCH).min(m);
                let kn = kend - kstart;
                let pts: Vec<Vec3> = (kstart..kend).map(|a| grid.points[idx[a]]).collect();
                let batch = eval_aos_full(mb, &pts, true, true);
                let g = batch.grad.as_ref().expect("grads");
                let hess = batch.hess.as_ref().expect("hessians");
                let x = batch.phi.matmul_seq(&p_tot);
                let y = [
                    g[0].matmul_seq(&p_tot),
                    g[1].matmul_seq(&p_tot),
                    g[2].matmul_seq(&p_tot),
                ];
                let mut dwl = vec![0.0; kn * dim];
                let mut drl = vec![0.0; kn * dim];
                let mut dsl = vec![0.0; kn * dim];
                for k in 0..kn {
                    let gi = idx[kstart + k];
                    let owner = grid.atom_of[gi];
                    let raw = grid.raw_weights[gi];
                    // per-atom restricted sums.
                    let mut q = vec![[0.0; 3]; natoms];
                    let mut gy_at = vec![[[0.0; 3]; 3]; natoms];
                    let mut hx_at = vec![[0.0; 6]; natoms];
                    let mut gy_full = [[0.0; 3]; 3];
                    let mut hx_full = [0.0; 6];
                    for mu in 0..nao {
                        let xm = x[(k, mu)];
                        let at = ao_atom[mu];
                        if xm != 0.0 {
                            for ax in 0..3 {
                                q[at][ax] += g[ax][(k, mu)] * xm;
                            }
                            for hh in 0..6 {
                                let hv = hess[hh][(k, mu)] * xm;
                                hx_at[at][hh] += hv;
                                hx_full[hh] += hv;
                            }
                        }
                        for u in 0..3 {
                            let yv = y[u][(k, mu)];
                            if yv == 0.0 {
                                continue;
                            }
                            for ax in 0..3 {
                                let gv = g[ax][(k, mu)] * yv;
                                gy_at[at][u][ax] += gv;
                                gy_full[u][ax] += gv;
                            }
                        }
                    }
                    let (_, dwb) = becke_weight_gradient(&centers, &aij, natoms, owner, pts[k]);
                    let mut dwb_sum = Vec3::zero();
                    for v in &dwb {
                        dwb_sum += *v;
                    }
                    for at in 0..natoms {
                        for ax in 0..3 {
                            let xslot = 3 * at + ax;
                            let mut wv = dwb[at].get(ax);
                            if at == owner {
                                wv -= dwb_sum.get(ax);
                            }
                            dwl[k * dim + xslot] = raw * wv;
                            let own = if at == owner { grho[gi][ax] } else { 0.0 };
                            drl[k * dim + xslot] = own - 2.0 * q[at][ax];
                            let mut sv = 0.0;
                            for u in 0..3 {
                                let mut dgr_u = -2.0 * (gy_at[at][u][ax] + hx_at[at][h6(ax, u)]);
                                if at == owner {
                                    dgr_u += 2.0 * (gy_full[u][ax] + hx_full[h6(ax, u)]);
                                }
                                sv += 2.0 * dgr_u * grho[gi][u];
                            }
                            dsl[k * dim + xslot] = sv;
                        }
                    }
                }
                (kstart, dwl, drl, dsl)
            })
            .collect();
        for (kstart, dwl, drl, dsl) in slices {
            let len = dwl.len();
            dw[kstart * dim..kstart * dim + len].copy_from_slice(&dwl);
            dr[kstart * dim..kstart * dim + len].copy_from_slice(&drl);
            ds[kstart * dim..kstart * dim + len].copy_from_slice(&dsl);
        }
    }

    // Sparsity of the fixed-P first derivatives: a grid point only feels
    // the atoms whose basis functions reach it (plus its owner), so most
    // of the 3N slots are exactly zero. Listing the live ones turns the
    // pair loop below from O(m^2 * 3N) into O(m^2 * nnz), i.e. a cost per
    // pair that stops growing with the molecule.
    let live: Vec<Vec<usize>> = (0..m)
        .map(|a| {
            (0..dim)
                .filter(|&xs| {
                    dw[a * dim + xs] != 0.0 || dr[a * dim + xs] != 0.0 || ds[a * dim + xs] != 0.0
                })
                .collect()
        })
        .collect();

    // ---- pass 3 (O(m²)): effective kernel + nonlocal M-vectors + sparse
    // R² Hessian pieces ----
    type D5 = DualN<DualN<f64, 5>, 5>;
    let seed5 = |vals: [f64; 5]| -> [D5; 5] {
        let mut out = [D5 {
            re: DualN::<f64, 5>::constant(0.0),
            im: [DualN::<f64, 5>::constant(0.0); 5],
        }; 5];
        for (k, o) in out.iter_mut().enumerate() {
            o.re = DualN::<f64, 5>::variable(vals[k], k);
            o.im[k] = DualN::<f64, 5>::constant(1.0);
        }
        out
    };
    struct PairOut {
        eeff: f64,
        vrho: f64,
        vsig: f64,
        crr: f64,
        crs: f64,
        css: f64,
        dew: Vec<f64>,
        dvr: Vec<f64>,
        dvs: Vec<f64>,
        hpart: Matrix,
    }
    let pair_rows: Vec<PairOut> = (0..m)
        .into_par_iter()
        .map(|a| {
            let i = idx[a];
            let pi = grid.points[i];
            let oi = grid.atom_of[i];
            let wi = grid.weights[i];
            // Per-thread scratch for the axis-independent R² row.
            let mut row_buf = vec![0.0; dim];
            let mut row_live: Vec<usize> = Vec::with_capacity(dim);
            let mut row_touched = vec![false; dim];
            let (amplitude, derivative) =
                molecular_vv10_amplitude(DualN::<f64, 1>::variable(rho[i], 0));
            let mut o = PairOut {
                eeff: beta * amplitude.re,
                vrho: beta * derivative.re,
                vsig: 0.0,
                crr: beta * derivative.im[0],
                crs: 0.0,
                css: 0.0,
                dew: vec![0.0; dim],
                dvr: vec![0.0; dim],
                dvs: vec![0.0; dim],
                hpart: Matrix::zeros(dim, dim),
            };
            for b_ in 0..m {
                let j = idx[b_];
                let wj = grid.weights[j];
                let dvec = pi - grid.points[j];
                let r2 = dvec.norm2();
                let vars = seed5([rho[i], sig[i], rho[j], sig[j], r2]);
                let e = vv10_pair_e5(vars[0], vars[1], vars[2], vars[3], vars[4], params);
                let e0 = e.re.re;
                let e1 = [e.re.im[0], e.re.im[1], e.re.im[2], e.re.im[3], e.re.im[4]];
                let e2 = |r: usize, c: usize| e.im[r].im[c];
                o.eeff += wj * e0;
                o.vrho += wj * e1[0];
                o.vsig += wj * e1[1];
                o.crr += wj * e2(0, 0);
                o.crs += wj * e2(0, 1);
                o.css += wj * e2(1, 1);
                let oj = grid.atom_of[j];
                // j-side contractions over the LIVE slots only.
                let (c_ew_r, c_ew_s) = (wj * e1[2], wj * e1[3]);
                let (c_vr_r, c_vr_s) = (wj * e2(0, 2), wj * e2(0, 3));
                let (c_vs_r, c_vs_s) = (wj * e2(1, 2), wj * e2(1, 3));
                for &xs in &live[b_] {
                    let dwj = dw[b_ * dim + xs];
                    let drj = dr[b_ * dim + xs];
                    let dsj = ds[b_ * dim + xs];
                    o.dew[xs] += e0 * dwj + c_ew_r * drj + c_ew_s * dsj;
                    o.dvr[xs] += e1[0] * dwj + c_vr_r * drj + c_vr_s * dsj;
                    o.dvs[xs] += e1[1] * dwj + c_vs_r * drj + c_vs_s * dsj;
                }
                // Kernel-motion R² columns (sparse in the displacement) and
                // the R² row / DDR² into the Hessian.
                if oi != oj {
                    // Build the axis-independent R² row once, over the
                    // union of the two points' live slots plus the two
                    // owner atoms (everything else is exactly zero).
                    row_live.clear();
                    for &ys in live[a].iter().chain(live[b_].iter()) {
                        if !row_touched[ys] {
                            row_touched[ys] = true;
                            row_live.push(ys);
                        }
                    }
                    for at in [oi, oj] {
                        for k in 0..3 {
                            let ys = 3 * at + k;
                            if !row_touched[ys] {
                                row_touched[ys] = true;
                                row_live.push(ys);
                            }
                        }
                    }
                    for &ys in &row_live {
                        let mut row = wi
                            * wj
                            * (e2(4, 0) * dr[a * dim + ys]
                                + e2(4, 1) * ds[a * dim + ys]
                                + e2(4, 2) * dr[b_ * dim + ys]
                                + e2(4, 3) * ds[b_ * dim + ys])
                            + wj * e1[4] * dw[a * dim + ys]
                            + wi * e1[4] * dw[b_ * dim + ys];
                        let (yat, yax) = (ys / 3, ys % 3);
                        let dyr2 = if yat == oi {
                            2.0 * dvec.get(yax)
                        } else if yat == oj {
                            -2.0 * dvec.get(yax)
                        } else {
                            0.0
                        };
                        if dyr2 != 0.0 {
                            row += wi * wj * e2(4, 4) * dyr2;
                        }
                        row_buf[ys] = row;
                    }
                    for &ys in &row_live {
                        row_touched[ys] = false;
                    }
                    for ax in 0..3 {
                        let dr2 = 2.0 * dvec.get(ax);
                        // R² columns of the M-vectors.
                        let xi = 3 * oi + ax;
                        let xj = 3 * oj + ax;
                        o.dew[xi] += wj * e1[4] * dr2;
                        o.dew[xj] -= wj * e1[4] * dr2;
                        o.dvr[xi] += wj * e2(0, 4) * dr2;
                        o.dvr[xj] -= wj * e2(0, 4) * dr2;
                        o.dvs[xi] += wj * e2(1, 4) * dr2;
                        o.dvs[xj] -= wj * e2(1, 4) * dr2;
                        // R² row: identical for all three axes (only the
                        // dr2 prefactor differs), so it is built once per
                        // pair above and merely scaled here.
                        for &ys in &row_live {
                            let v = 0.5 * dr2 * row_buf[ys];
                            o.hpart[(xi, ys)] += v;
                            o.hpart[(xj, ys)] -= v;
                        }
                        // ½ e_{R²} DDR²: DD = 2δ_{xy}(±1)(±1).
                        let tt = 0.5 * wi * wj * e1[4] * 2.0;
                        o.hpart[(xi, xi)] += tt;
                        o.hpart[(xj, xj)] += tt;
                        o.hpart[(xi, xj)] -= tt;
                        o.hpart[(xj, xi)] -= tt;
                    }
                }
            }
            o
        })
        .collect();

    // ---- local part through the shared GGA Hessian machinery ----
    let mut eeff = vec![0.0; n];
    let mut vrho = vec![0.0; n];
    let mut vsig = vec![0.0; n];
    let mut crr = vec![0.0; n];
    let mut crs = vec![0.0; n];
    let mut css = vec![0.0; n];
    for a in 0..m {
        let i = idx[a];
        eeff[i] = pair_rows[a].eeff;
        vrho[i] = pair_rows[a].vrho;
        vsig[i] = pair_rows[a].vsig;
        crr[i] = pair_rows[a].crr;
        crs[i] = pair_rows[a].crs;
        css[i] = pair_rows[a].css;
    }
    let kernel = |_vars: &crate::xc::GridVars, start: usize, npts: usize| -> Result<XcOut> {
        let mut out = XcOut::default();
        out.prepare(npts, 2, Family::Gga);
        for k in 0..npts {
            let gi = start + k;
            out.e[k] = eeff[gi];
            out.vrho[0][k] = vrho[gi];
            out.vrho[1][k] = vrho[gi];
            // total-σ chain: vσaa = vσbb = v, vσab = 2v.
            out.vsigma[0][k] = vsig[gi];
            out.vsigma[1][k] = 2.0 * vsig[gi];
            out.vsigma[2][k] = vsig[gi];
            for ch in 0..3 {
                out.v2rho2[ch][k] = crr[gi];
            }
            let rs_scale = [1.0, 2.0, 1.0];
            for pch in 0..3 {
                out.v2rhosigma[pch][k] = crs[gi] * rs_scale[pch];
                out.v2rhosigma[3 + pch][k] = crs[gi] * rs_scale[pch];
            }
            let ss_scale = [1.0, 2.0, 1.0, 4.0, 2.0, 1.0];
            for sch in 0..6 {
                out.v2sigma2[sch][k] = css[gi] * ss_scale[sch];
            }
        }
        Ok(out)
    };
    let local = xc_hessian_terms_with(mol, mb, grid, p, true, &kernel)?;

    // ---- assemble: local + Gram + sparse R² parts ----
    let mut skeleton = local.skeleton;
    for a in 0..m {
        let po = &pair_rows[a];
        for xs in 0..dim {
            let (dwx, drx, dsx) = (dw[a * dim + xs], dr[a * dim + xs], ds[a * dim + xs]);
            if dwx == 0.0 && drx == 0.0 && dsx == 0.0 {
                continue;
            }
            let wi = grid.weights[idx[a]];
            for ys in 0..dim {
                skeleton[(xs, ys)] += dwx * po.dew[ys] + wi * (drx * po.dvr[ys] + dsx * po.dvs[ys]);
            }
        }
        for xs in 0..dim {
            for ys in 0..dim {
                skeleton[(xs, ys)] += po.hpart[(xs, ys)];
            }
        }
    }

    // ---- V^{(a)} nonlocal corrections ----
    let mut v_deriv: Vec<AtomDerivs> = (0..natoms)
        .map(|_| {
            [
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
            ]
        })
        .collect();
    let mut kstart = 0;
    while kstart < m {
        let kend = (kstart + BATCH).min(m);
        let pts: Vec<Vec3> = (kstart..kend).map(|a| grid.points[idx[a]]).collect();
        let batch = eval_aos_full(mb, &pts, true, false);
        let g = batch.grad.as_ref().expect("grads");
        for k in 0..kend - kstart {
            let a = kstart + k;
            let gi = idx[a];
            let w = grid.weights[gi];
            let po = &pair_rows[a];
            for at in 0..natoms {
                for ax in 0..3 {
                    let xs = 3 * at + ax;
                    let (dvr, dvs) = (po.dvr[xs], po.dvs[xs]);
                    if dvr == 0.0 && dvs == 0.0 {
                        continue;
                    }
                    let f = w * dvr;
                    let bvec = [
                        2.0 * w * dvs * grho[gi][0],
                        2.0 * w * dvs * grho[gi][1],
                        2.0 * w * dvs * grho[gi][2],
                    ];
                    let mmat = &mut v_deriv[at][ax];
                    for mu in 0..nao {
                        let pm = batch.phi[(k, mu)];
                        for nu in 0..nao {
                            let pn = batch.phi[(k, nu)];
                            let mut val = f * pm * pn;
                            for u in 0..3 {
                                val += bvec[u] * (g[u][(k, mu)] * pn + pm * g[u][(k, nu)]);
                            }
                            mmat[(mu, nu)] += val;
                        }
                    }
                }
            }
        }
        kstart = kend;
    }
    // Merge the local vxc_deriv (identical for both spins by construction —
    // take the α channel).
    for at in 0..natoms {
        for ax in 0..3 {
            for mu in 0..nao {
                for nu in 0..nao {
                    v_deriv[at][ax][(mu, nu)] += local.vxc_deriv[0][at][ax][(mu, nu)];
                }
            }
        }
    }
    Ok(NlcHessianTerms { skeleton, v_deriv })
}
