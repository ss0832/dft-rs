// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Meta-GGA XC contributions to the analytic Hessian - the tau-channel
//! extension of `hessian/xc_terms.rs`, kept in its own module per the
//! project layout policy (meta machinery grows here, not as branches in
//! the GGA paths).
//!
//! Additions over the GGA version:
//! - 7-variable kernel quadratic (rho_a rho_b s_aa s_ab s_bb tau_a tau_b)
//!   using v2rhotau / v2sigmatau / v2tau2 from the order-2 MggaDriver;
//! - D_a tau_s = delta_Ao HY1_full - HY1_at[A],
//!   HY1[x] = sum_u sum_mu h_{xu,mu} (g_u P)_mu;
//! - D_aD_b tau_s: same-index third-AO-derivative term (t3 . gP rows) and
//!   the cross term over hP = h P row matrices;
//! - V_xc^(a) gains the tau channel
//!   D_a [ w 1/2 vtau_s sum_u g_u,mu g_u,nu ].

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::eval::{second_index as h6, second_index as ss, third_index};
use crate::grid::MolecularGrid;
use crate::hessian::xc_terms::{BatchPoints, MolecularWeights, WeightResponse, XcHessianTerms};
use crate::integrals::deriv_mat::AtomDerivs;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};

pub fn xc_hessian_terms_mgga(
    functional: &ResolvedFunctional,
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
) -> Result<XcHessianTerms> {
    if functional.family != Family::MetaGga {
        return Err(DftError::InvalidInput(
            "xc_hessian_terms_mgga expects a meta-GGA functional".to_string(),
        ));
    }
    let natoms = mol.len();
    let nao = mb.nao_sph;
    let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
    let aij = MolecularGrid::becke_aij(mol);
    let weights = MolecularWeights {
        centers: &centers,
        aij: &aij,
    };
    let dim = 3 * natoms;
    let mut skeleton = Matrix::zeros(dim, dim);
    let zeros3 = || -> AtomDerivs {
        [
            Matrix::zeros(nao, nao),
            Matrix::zeros(nao, nao),
            Matrix::zeros(nao, nao),
        ]
    };
    let mut vxc_deriv: [Vec<AtomDerivs>; 2] = [
        (0..natoms).map(|_| zeros3()).collect(),
        (0..natoms).map(|_| zeros3()).collect(),
    ];

    const BATCH: usize = 96;
    let n = grid.len();
    let mut start = 0;
    while start < n {
        let end = (start + BATCH).min(n);
        mgga_batch_terms(
            &BatchPoints {
                points: &grid.points[start..end],
                weights: &grid.weights[start..end],
                raw_weights: &grid.raw_weights[start..end],
                owners: &grid.atom_of[start..end],
                active_shells: None,
            },
            mb,
            p,
            natoms,
            functional,
            &weights,
            &mut skeleton,
            &mut vxc_deriv,
        )?;
        start = end;
    }
    Ok(XcHessianTerms {
        skeleton,
        vxc_deriv,
    })
}

/// Accumulate one meta-GGA grid batch. For a periodic local basis, every
/// image carries the atom label of its home shell and the supplied partition
/// derivatives include the motion of the corresponding lattice images.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mgga_batch_terms(
    points: &BatchPoints,
    mb: &MolBasis,
    p: [&Matrix; 2],
    natoms: usize,
    functional: &ResolvedFunctional,
    weights: &dyn WeightResponse,
    skeleton: &mut Matrix,
    vxc_deriv: &mut [Vec<AtomDerivs>; 2],
) -> Result<()> {
    let pts = points.points;
    let wts = points.weights;
    let raws = points.raw_weights;
    let owners = points.owners;
    let nao = mb.nao_sph;
    let dim = 3 * natoms;
    let ao_atom = mb.sph_ao_atoms();
    let npts = pts.len();
    let batch = points.ao_derivatives(mb, true);
    let g = batch.grad.as_ref().expect("gradients");
    let hess = batch.hess.as_ref().expect("hessians");
    let third = batch.third.as_ref().expect("third derivatives");
    // Row matrices: x = phi P, yfull_u = g_u P, hp_h6 = h_h6 P, and the
    // per-atom masked versions of yfull/hp for the c^A decompositions.
    let mut x = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
    let mut rho = [vec![0.0; npts], vec![0.0; npts]];
    let mut tau = [vec![0.0; npts], vec![0.0; npts]];
    let mut yfull: Vec<[Matrix; 3]> = Vec::with_capacity(2);
    let mut yres: Vec<Vec<[Matrix; 3]>> = Vec::with_capacity(2);
    let mut hp: Vec<Vec<Matrix>> = Vec::with_capacity(2); // [s][h6]
    let mut hp_res: Vec<Vec<Vec<Matrix>>> = Vec::with_capacity(2); // [s][at][h6]
    for spin in 0..2 {
        x[spin] = batch.phi.matmul_seq(p[spin]);
        yfull.push([
            g[0].matmul_seq(p[spin]),
            g[1].matmul_seq(p[spin]),
            g[2].matmul_seq(p[spin]),
        ]);
        let mut hps = Vec::with_capacity(6);
        for hh in 0..6 {
            hps.push(hess[hh].matmul_seq(p[spin]));
        }
        hp.push(hps);
        let mut per_atom = Vec::with_capacity(natoms);
        let mut per_atom_h = Vec::with_capacity(natoms);
        for at in 0..natoms {
            let mut masked = p[spin].clone();
            for mu in 0..nao {
                if ao_atom[mu] != at {
                    for nu in 0..nao {
                        masked[(mu, nu)] = 0.0;
                    }
                }
            }
            per_atom.push([
                g[0].matmul_seq(&masked),
                g[1].matmul_seq(&masked),
                g[2].matmul_seq(&masked),
            ]);
            let mut hs = Vec::with_capacity(6);
            for hh in 0..6 {
                hs.push(hess[hh].matmul_seq(&masked));
            }
            per_atom_h.push(hs);
        }
        yres.push(per_atom);
        hp_res.push(per_atom_h);
        for i in 0..npts {
            let mut acc = 0.0;
            for mu in 0..nao {
                acc += x[spin][(i, mu)] * batch.phi[(i, mu)];
            }
            rho[spin][i] = acc.max(0.0);
            let mut t_acc = 0.0;
            for u in 0..3 {
                for mu in 0..nao {
                    t_acc += yfull[spin][u][(i, mu)] * g[u][(i, mu)];
                }
            }
            tau[spin][i] = (0.5 * t_acc).max(0.0);
        }
    }
    // grad rho and sigma.
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
    let mut sig = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
    for i in 0..npts {
        let ga = [gr[0][0][i], gr[0][1][i], gr[0][2][i]];
        let gb = [gr[1][0][i], gr[1][1][i], gr[1][2][i]];
        sig[0][i] = ga.iter().map(|v| v * v).sum();
        sig[1][i] = ga.iter().zip(&gb).map(|(a, b)| a * b).sum();
        sig[2][i] = gb.iter().map(|v| v * v).sum();
    }
    let vars = GridVars {
        rho: [&rho[0], &rho[1]],
        sigma: Some([&sig[0], &sig[1], &sig[2]]),
        tau: Some([&tau[0], &tau[1]]),
    };
    let mut out = XcOut::default();
    functional.eval(&vars, 2, &mut out)?;

    for i in 0..npts {
        let w = wts[i];
        let raw = raws[i];
        let owner = owners[i];
        let e_i = out.e[i];
        let vr = [out.vrho[0][i], out.vrho[1][i]];
        let vs = [out.vsigma[0][i], out.vsigma[1][i], out.vsigma[2][i]];
        let vt = [out.vtau[0][i], out.vtau[1][i]];
        // 7x7 symmetric kernel: (ra rb saa sab sbb ta tb).
        let mut ker = [[0.0; 7]; 7];
        ker[0][0] = out.v2rho2[0][i];
        ker[0][1] = out.v2rho2[1][i];
        ker[1][1] = out.v2rho2[2][i];
        for pch in 0..3 {
            ker[0][2 + pch] = out.v2rhosigma[pch][i];
            ker[1][2 + pch] = out.v2rhosigma[3 + pch][i];
            for qch in pch..3 {
                ker[2 + pch][2 + qch] = out.v2sigma2[ss(pch, qch)][i];
            }
            ker[2 + pch][5] = out.v2sigmatau[2 * pch][i];
            ker[2 + pch][6] = out.v2sigmatau[2 * pch + 1][i];
        }
        ker[0][5] = out.v2rhotau[0][i];
        ker[0][6] = out.v2rhotau[1][i];
        ker[1][5] = out.v2rhotau[2][i];
        ker[1][6] = out.v2rhotau[3][i];
        ker[5][5] = out.v2tau2[0][i];
        ker[5][6] = out.v2tau2[1][i];
        ker[6][6] = out.v2tau2[2][i];
        for r in 0..7 {
            for c in 0..r {
                ker[r][c] = ker[c][r];
            }
        }
        // ---- per-point restricted accumulators ----
        let mut q = vec![[[0.0; 3]; 2]; natoms];
        let mut hx_at = vec![[[0.0; 6]; 2]; natoms];
        let mut hx_full = [[0.0; 6]; 2];
        let mut gy_at = vec![[[[0.0; 3]; 3]; 2]; natoms];
        let mut gy_full = [[[0.0; 3]; 3]; 2];
        let mut hy_at = vec![[[[0.0; 3]; 6]; 2]; natoms];
        let mut hy_full = [[[0.0; 3]; 6]; 2];
        let mut tx_at = vec![[[0.0; 10]; 2]; natoms];
        let mut tx_full = [[0.0; 10]; 2];
        // tau extras: HY1[x] = sum_u h_{xu} yfull_u; T3Y[(x,y)] =
        // sum_u t3_{xyu} yfull_u.
        let mut hy1_at = vec![[[0.0; 3]; 2]; natoms];
        let mut hy1_full = [[0.0; 3]; 2];
        let mut t3y_at = vec![[[0.0; 6]; 2]; natoms];
        let mut t3y_full = [[0.0; 6]; 2];
        for spin in 0..2 {
            for mu in 0..nao {
                let xm = x[spin][(i, mu)];
                let at = ao_atom[mu];
                if xm != 0.0 {
                    for ax in 0..3 {
                        q[at][spin][ax] += g[ax][(i, mu)] * xm;
                    }
                    for hh in 0..6 {
                        let hv = hess[hh][(i, mu)] * xm;
                        hx_at[at][spin][hh] += hv;
                        hx_full[spin][hh] += hv;
                    }
                    for tt in 0..10 {
                        let tv = third[tt][(i, mu)] * xm;
                        tx_at[at][spin][tt] += tv;
                        tx_full[spin][tt] += tv;
                    }
                }
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
                    for ax in 0..3 {
                        let hv = hess[h6(ax, u)][(i, mu)] * yv;
                        hy1_at[at][spin][ax] += hv;
                        hy1_full[spin][ax] += hv;
                    }
                    for ax in 0..3 {
                        for by in ax..3 {
                            let tv = third[third_index(ax, by, u)][(i, mu)] * yv;
                            t3y_at[at][spin][h6(ax, by)] += tv;
                            t3y_full[spin][h6(ax, by)] += tv;
                        }
                    }
                }
            }
        }
        // Cross accumulators for DDrho (g-side) and DDtau (h-side).
        let mut hu_full = vec![[[[0.0; 6]; 3]; 2]; natoms];
        let mut hu_at = vec![vec![[[[0.0; 6]; 3]; 2]; natoms]; natoms];
        // hh-scans: for DDtau term2: per (s, A, x): sum over u of
        // h_{yu}(nu) . U2^{A,xu}(nu) with U2 from hp rows.
        let mut hh_full = vec![[[[0.0; 3]; 3]; 2]; natoms]; // [A][s][x][y]
        let mut hh_at = vec![vec![[[[0.0; 3]; 3]; 2]; natoms]; natoms];
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
                        if u_v != 0.0 {
                            let b_at = ao_atom[nu];
                            for hh in 0..6 {
                                let hv = hess[hh][(i, nu)] * u_v;
                                hu_full[a_at][spin][ax][hh] += hv;
                                hu_at[a_at][b_at][spin][ax][hh] += hv;
                            }
                        }
                        // DDtau cross: U2(nu) summed over u.
                        for u in 0..3 {
                            let u2 = if d_ao {
                                hp[spin][h6(ax, u)][(i, nu)]
                                    - hp_res[spin][a_at][h6(ax, u)][(i, nu)]
                            } else {
                                -hp_res[spin][a_at][h6(ax, u)][(i, nu)]
                            };
                            if u2 == 0.0 {
                                continue;
                            }
                            let b_at = ao_atom[nu];
                            for by in 0..3 {
                                let hv = hess[h6(by, u)][(i, nu)] * u2;
                                hh_full[a_at][spin][ax][by] += hv;
                                hh_at[a_at][b_at][spin][ax][by] += hv;
                            }
                        }
                    }
                }
            }
        }

        let drho = |spin: usize, at: usize, ax: usize| -> f64 {
            let own = if at == owner { gr[spin][ax][i] } else { 0.0 };
            own - 2.0 * q[at][spin][ax]
        };
        let dgr_f = |spin: usize, at: usize, ax: usize, u: usize| -> f64 {
            let mut v = -2.0 * (gy_at[at][spin][u][ax] + hx_at[at][spin][h6(ax, u)]);
            if at == owner {
                v += 2.0 * (gy_full[spin][u][ax] + hx_full[spin][h6(ax, u)]);
            }
            v
        };
        let dsig = |pch: usize, at: usize, ax: usize| -> f64 {
            let (s, t) = [(0, 0), (0, 1), (1, 1)][pch];
            let mut v = 0.0;
            for u in 0..3 {
                v += dgr_f(s, at, ax, u) * gr[t][u][i] + gr[s][u][i] * dgr_f(t, at, ax, u);
            }
            v
        };
        let dtau = |spin: usize, at: usize, ax: usize| -> f64 {
            let own = if at == owner { hy1_full[spin][ax] } else { 0.0 };
            own - hy1_at[at][spin][ax]
        };
        let dvars = |at: usize, ax: usize| -> [f64; 7] {
            let mut v = [0.0; 7];
            v[0] = drho(0, at, ax);
            v[1] = drho(1, at, ax);
            for pch in 0..3 {
                v[2 + pch] = dsig(pch, at, ax);
            }
            v[5] = dtau(0, at, ax);
            v[6] = dtau(1, at, ax);
            v
        };
        // Reuse the same cached owner-riding partition derivatives as
        // LDA/GGA and nonlocal Hessians; the tau response remains here.
        let (weight_first, weight_second) = weights.derivatives(pts[i], owner);
        let dw = |at: usize, ax: usize| -> f64 { raw * weight_first[3 * at + ax] };
        // ---- Vxc derivative matrices (rho + sigma + tau channels) ----
        for spin in 0..2 {
            let (own_ch, cross_ch) = if spin == 0 { (0usize, 1usize) } else { (2, 1) };
            let other = 1 - spin;
            for at in 0..natoms {
                for ax in 0..3 {
                    let dw_v = dw(at, ax);
                    let dv = dvars(at, ax);
                    // First-order kernel rows for this spin.
                    let row_r = if spin == 0 { 0 } else { 1 };
                    let mut dvr = 0.0;
                    for k in 0..7 {
                        dvr += ker[row_r][k] * dv[k];
                    }
                    let mut dvs = [0.0; 3];
                    for pch in 0..3 {
                        for k in 0..7 {
                            dvs[pch] += ker[2 + pch][k] * dv[k];
                        }
                    }
                    let row_t = if spin == 0 { 5 } else { 6 };
                    let mut dvt = 0.0;
                    for k in 0..7 {
                        dvt += ker[row_t][k] * dv[k];
                    }
                    let mut bvec = [0.0; 3];
                    let mut dbvec = [0.0; 3];
                    for u in 0..3 {
                        bvec[u] =
                            2.0 * vs[own_ch] * gr[spin][u][i] + vs[cross_ch] * gr[other][u][i];
                        dbvec[u] = 2.0 * dvs[own_ch] * gr[spin][u][i]
                            + 2.0 * vs[own_ch] * dgr_f(spin, at, ax, u)
                            + dvs[cross_ch] * gr[other][u][i]
                            + vs[cross_ch] * dgr_f(other, at, ax, u);
                    }
                    let m = &mut vxc_deriv[spin][at][ax];
                    for mu in 0..nao {
                        let pm = batch.phi[(i, mu)];
                        let gm = g[ax][(i, mu)];
                        let c_mu = (at == owner) as i32 - (ao_atom[mu] == at) as i32;
                        for nu in 0..nao {
                            let pn = batch.phi[(i, nu)];
                            let gn = g[ax][(i, nu)];
                            let c_nu = (at == owner) as i32 - (ao_atom[nu] == at) as i32;
                            let mut v = (dw_v * vr[spin] + w * dvr) * pm * pn;
                            if c_mu != 0 {
                                v += w * vr[spin] * (c_mu as f64) * gm * pn;
                            }
                            if c_nu != 0 {
                                v += w * vr[spin] * (c_nu as f64) * pm * gn;
                            }
                            for u in 0..3 {
                                let gpm = g[u][(i, mu)];
                                let gpn = g[u][(i, nu)];
                                v += (dw_v * bvec[u] + w * dbvec[u]) * (gpm * pn + pm * gpn);
                                let mut dterm = 0.0;
                                if c_mu != 0 {
                                    dterm +=
                                        (c_mu as f64) * (hess[h6(ax, u)][(i, mu)] * pn + gm * gpn);
                                }
                                if c_nu != 0 {
                                    dterm +=
                                        (c_nu as f64) * (gpm * gn + pm * hess[h6(ax, u)][(i, nu)]);
                                }
                                if dterm != 0.0 {
                                    v += w * bvec[u] * dterm;
                                }
                                // tau channel: D_a[w 1/2 vt g_u g_u].
                                let mut tterm = (dw_v * vt[spin] + w * dvt) * 0.5 * gpm * gpn;
                                if c_mu != 0 {
                                    tterm += w
                                        * vt[spin]
                                        * 0.5
                                        * (c_mu as f64)
                                        * hess[h6(ax, u)][(i, mu)]
                                        * gpn;
                                }
                                if c_nu != 0 {
                                    tterm += w
                                        * vt[spin]
                                        * 0.5
                                        * (c_nu as f64)
                                        * gpm
                                        * hess[h6(ax, u)][(i, nu)];
                                }
                                v += tterm;
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
        let grad1 = [vr[0], vr[1], vs[0], vs[1], vs[2], vt[0], vt[1]];
        let de = |dv: &[f64; 7]| -> f64 {
            let mut v = 0.0;
            for k in 0..7 {
                v += grad1[k] * dv[k];
            }
            v
        };
        let ddgr = |spin: usize, a_at: usize, ax: usize, b_at: usize, by: usize, u: usize| -> f64 {
            let d_ao = a_at == owner;
            let d_bo = b_at == owner;
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
            let t2 = {
                let full = hu_full[a_at][spin][ax][h6(by, u)];
                let res = hu_at[a_at][b_at][spin][ax][h6(by, u)];
                if d_bo {
                    full - res
                } else {
                    -res
                }
            };
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
        for a_at in 0..natoms {
            for b_at in a_at..natoms {
                for ax in 0..3 {
                    let by0 = if a_at == b_at { ax } else { 0 };
                    for by in by0..3 {
                        let dva = dvars(a_at, ax);
                        let dvb = dvars(b_at, by);
                        let mut v =
                            raw * weight_second[(3 * a_at + ax) * dim + 3 * b_at + by] * e_i;
                        v += dw(a_at, ax) * de(&dvb);
                        v += dw(b_at, by) * de(&dva);
                        let mut dde = 0.0;
                        for r in 0..7 {
                            for c in 0..7 {
                                dde += ker[r][c] * dva[r] * dvb[c];
                            }
                        }
                        // vrho . DDrho.
                        for s in 0..2 {
                            let d_ao = a_at == owner;
                            let d_bo = b_at == owner;
                            let mut t1 = 0.0;
                            {
                                let hf = hx_full[s][h6(ax, by)];
                                let ha = hx_at[a_at][s][h6(ax, by)];
                                let hb = hx_at[b_at][s][h6(ax, by)];
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
                            let mut cross_full = 0.0;
                            let mut cross_b = 0.0;
                            for nu in 0..nao {
                                let u_v = if d_ao {
                                    yfull[s][ax][(i, nu)] - yres[s][a_at][ax][(i, nu)]
                                } else {
                                    -yres[s][a_at][ax][(i, nu)]
                                };
                                if u_v == 0.0 {
                                    continue;
                                }
                                let gy = g[by][(i, nu)];
                                cross_full += gy * u_v;
                                if ao_atom[nu] == b_at {
                                    cross_b += gy * u_v;
                                }
                            }
                            let cross = if d_bo { cross_full - cross_b } else { -cross_b };
                            dde += vr[s] * (2.0 * t1 + 2.0 * cross);
                        }
                        // vsigma . DDsigma.
                        for pch in 0..3 {
                            let (s, t) = [(0, 0), (0, 1), (1, 1)][pch];
                            let mut dds = 0.0;
                            for u in 0..3 {
                                dds += ddgr(s, a_at, ax, b_at, by, u) * gr[t][u][i]
                                    + dgr_f(s, a_at, ax, u) * dgr_f(t, b_at, by, u)
                                    + dgr_f(s, b_at, by, u) * dgr_f(t, a_at, ax, u)
                                    + gr[s][u][i] * ddgr(t, a_at, ax, b_at, by, u);
                            }
                            dde += vs[pch] * dds;
                        }
                        // vtau . DDtau: same-index t3 term + hP cross.
                        for s in 0..2 {
                            let d_ao = a_at == owner;
                            let d_bo = b_at == owner;
                            let mut t1 = 0.0;
                            {
                                let tf = t3y_full[s][h6(ax, by)];
                                let ta = t3y_at[a_at][s][h6(ax, by)];
                                let tb = t3y_at[b_at][s][h6(ax, by)];
                                if d_ao && d_bo {
                                    t1 += tf;
                                }
                                if d_ao {
                                    t1 -= tb;
                                }
                                if d_bo {
                                    t1 -= ta;
                                }
                                if a_at == b_at {
                                    t1 += ta;
                                }
                            }
                            let t2 = {
                                let full = hh_full[a_at][s][ax][by];
                                let res = hh_at[a_at][b_at][s][ax][by];
                                if d_bo {
                                    full - res
                                } else {
                                    -res
                                }
                            };
                            dde += vt[s] * (t1 + t2);
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::grid::GridOptions;
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

    /// Fixed-P reference: E_xc and V_xc (rho + sigma + tau channels) at an
    /// arbitrary geometry with the grid rebuilt.
    fn exc_and_vxc(
        mm: &Molecule,
        func: &ResolvedFunctional,
        p: [&Matrix; 2],
        nao: usize,
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
            let batch = crate::grid::eval::eval_aos_deriv3(&mb, pts, true, false, false);
            let g = batch.grad.as_ref().unwrap();
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut tau = [vec![0.0; npts], vec![0.0; npts]];
            let mut x = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
            let mut gr: [[Vec<f64>; 3]; 2] = Default::default();
            let mut y: Vec<[Matrix; 3]> = Vec::new();
            for spin in 0..2 {
                x[spin] = batch.phi.matmul_seq(p[spin]);
                let yg = [
                    g[0].matmul_seq(p[spin]),
                    g[1].matmul_seq(p[spin]),
                    g[2].matmul_seq(p[spin]),
                ];
                for i in 0..npts {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        acc += x[spin][(i, mu)] * batch.phi[(i, mu)];
                    }
                    rho[spin][i] = acc.max(0.0);
                    let mut t = 0.0;
                    for u in 0..3 {
                        for mu in 0..nao {
                            t += yg[u][(i, mu)] * g[u][(i, mu)];
                        }
                    }
                    tau[spin][i] = (0.5 * t).max(0.0);
                }
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
                y.push(yg);
            }
            let mut s3 = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
            for i in 0..npts {
                let ga = [gr[0][0][i], gr[0][1][i], gr[0][2][i]];
                let gb = [gr[1][0][i], gr[1][1][i], gr[1][2][i]];
                s3[0][i] = ga.iter().map(|v| v * v).sum();
                s3[1][i] = ga.iter().zip(&gb).map(|(a, b)| a * b).sum();
                s3[2][i] = gb.iter().map(|v| v * v).sum();
            }
            let vars = GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: Some([&s3[0], &s3[1], &s3[2]]),
                tau: Some([&tau[0], &tau[1]]),
            };
            let mut out = XcOut::default();
            func.eval(&vars, 1, &mut out).unwrap();
            for i in 0..npts {
                e_tot += wts[i] * out.e[i];
                for spin in 0..2 {
                    let f = wts[i] * out.vrho[spin][i];
                    let (own, cross) = if spin == 0 { (0, 1) } else { (2, 1) };
                    let other = 1 - spin;
                    let mut bvec = [0.0; 3];
                    for u in 0..3 {
                        bvec[u] = wts[i]
                            * (2.0 * out.vsigma[own][i] * gr[spin][u][i]
                                + out.vsigma[cross][i] * gr[other][u][i]);
                    }
                    let ft = 0.5 * wts[i] * out.vtau[spin][i];
                    for mu in 0..nao {
                        let pm = batch.phi[(i, mu)];
                        for nu in 0..nao {
                            let pn = batch.phi[(i, nu)];
                            let mut val = f * pm * pn;
                            for u in 0..3 {
                                val += bvec[u] * (g[u][(i, mu)] * pn + pm * g[u][(i, nu)]);
                                val += ft * g[u][(i, mu)] * g[u][(i, nu)];
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

    /// Meta XC Hessian terms (M06-L) vs finite differences at fixed P.
    #[test]
    fn mgga_terms_match_finite_differences() {
        let m = mol();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&m, &set).unwrap();
        let nao = mb.nao_sph;
        let func = resolve(&FunctionalSpec::Combined("m06l".into())).unwrap();
        let mut p = Matrix::zeros(nao, nao);
        let mut state = 0x5150_cafe_u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for i in 0..nao {
            for j in i..nao {
                let v = 0.15 * rng();
                p[(i, j)] = v;
                p[(j, i)] = v;
            }
        }
        for i in 0..nao {
            p[(i, i)] += 0.7;
        }
        let grid = MolecularGrid::build(&m, &GridOptions::default()).unwrap();
        let terms = xc_hessian_terms_mgga(&func, &m, &mb, &grid, [&p, &p]).unwrap();
        let h = 1e-4;
        for at in 0..3 {
            for ax in 0..3 {
                let (_, vp) = exc_and_vxc(&bump(&m, at, ax, h), &func, [&p, &p], nao);
                let (_, vm) = exc_and_vxc(&bump(&m, at, ax, -h), &func, [&p, &p], nao);
                for mu in 0..nao {
                    for nu in 0..nao {
                        let fd = (vp[0][(mu, nu)] - vm[0][(mu, nu)]) / (2.0 * h);
                        let an = terms.vxc_deriv[0][at][ax][(mu, nu)];
                        assert!(
                            (an - fd).abs() < 5e-5,
                            "meta Vxc^(a) at={at} ax={ax} ({mu},{nu}): {an} vs FD {fd}"
                        );
                    }
                }
            }
        }
        for &(a, ax, b, by) in &[
            (0usize, 0usize, 0usize, 0usize),
            (0, 1, 1, 1),
            (0, 2, 1, 0),
            (1, 2, 2, 2),
            (2, 0, 2, 0),
        ] {
            let e = |mm: &Molecule| exc_and_vxc(mm, &func, [&p, &p], nao).0;
            let pp = e(&bump(&bump(&m, a, ax, h), b, by, h));
            let pm = e(&bump(&bump(&m, a, ax, h), b, by, -h));
            let mp = e(&bump(&bump(&m, a, ax, -h), b, by, h));
            let mmv = e(&bump(&bump(&m, a, ax, -h), b, by, -h));
            let fd = (pp - pm - mp + mmv) / (4.0 * h * h);
            let an = terms.skeleton[(3 * a + ax, 3 * b + by)];
            assert!(
                (an - fd).abs() < 5e-4 * fd.abs().max(1.0),
                "meta XC skeleton ({a},{ax})x({b},{by}): {an} vs FD {fd}"
            );
        }
    }
}
