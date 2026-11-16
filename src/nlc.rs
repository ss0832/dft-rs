// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! VV10 nonlocal correlation (Vydrov & Van Voorhis, JCP 133, 244103
//! (2010)):
//!
//! `E_nlc = ∫ ρ(r) [ β + ½ ∫ ρ(r′) Φ(r,r′) dr′ ] dr`,
//! `Φ = −3/2 · 1/(g g′ (g+g′))`, `g = ω₀(r)R² + κ(r)`,
//! `ω₀ = √(ω_g² + ω_p²/3)`, `ω_p² = 4πρ`, `ω_g² = C |∇ρ|⁴/ρ⁴`,
//! `κ = b·(3π/2)·[ρ/(9π)]^{1/6}`, `β = (1/32)·(3/b²)^{3/4}`.
//!
//! Evaluated post-SCF on the converged density over a (typically coarse)
//! dedicated grid — the self-consistency correction is at the µEh scale
//! and is documented in the gates. Parameters: VV10 (b = 5.9, C = 0.0093);
//! ωB97X-V uses (b = 6.0, C = 0.01).

use crate::basis::MolBasis;
use crate::dual::{DualN, Scalar};
use crate::error::Result;
use crate::grid::eval::eval_aos_full;
use crate::grid::{becke_weight_gradient, MolecularGrid};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use rayon::prelude::*;

pub mod vdw_df;
pub mod vv09;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vv10Params {
    pub b: f64,
    pub c: f64,
}

impl Vv10Params {
    pub const ORIGINAL: Self = Self { b: 5.9, c: 0.0093 };
    pub const WB97XV: Self = Self { b: 6.0, c: 0.01 };
    /// Sabatini, Gorni and de Gironcoli's parameters for the revised kernel.
    pub const REVISED: Self = Self { b: 6.3, c: 0.0093 };

    /// VV10 parameters carried by a named `-V` functional, if any.
    ///
    /// Name normalization mirrors the XC resolver. Keeping this decision in
    /// the library prevents the CLI and language bindings from accidentally
    /// evaluating only the semilocal part of a functional whose definition
    /// includes nonlocal correlation.
    pub fn for_functional(name: &str) -> Option<Self> {
        crate::xc::built_in_nonlocal(name).0
    }
}

pub(crate) const RHO_CUT: f64 = 1e-8;

// Molecular VV10 evaluates the literal low-density kernel until the tail
// is below the quadrature accuracy. The distinct periodic and revised
// kernels keep their own vacuum convention above.
pub(crate) const MOLECULAR_VV10_RHO_CUT: f64 = 1e-24;

/// Pair amplitude and its density derivative. The beta(6,6) integral
/// joins the numerical vacuum to the literal tail with five continuous
/// derivatives. The true density still determines omega and kappa.
pub(crate) fn molecular_vv10_amplitude<S: Scalar>(rho: S) -> (S, S) {
    let one = S::from_f64(1.0);
    let zero = S::from_f64(0.0);
    let cutoff = MOLECULAR_VV10_RHO_CUT;
    if rho.value() <= cutoff {
        return (zero, zero);
    }
    if rho.value() >= 2.0 * cutoff {
        return (rho, one);
    }
    let x = (rho - S::from_f64(cutoff)) / S::from_f64(cutoff);
    let smooth = |t: S| {
        t.powi(6)
            * [462.0, -1980.0, 3465.0, -3080.0, 1386.0, -252.0]
                .iter()
                .rev()
                .fold(zero, |acc, &coefficient| acc * t + S::from_f64(coefficient))
    };
    let switch = if x.value() < 0.5 {
        smooth(x)
    } else {
        one - smooth(one - x)
    };
    let slope = S::from_f64(2772.0 / cutoff) * (x * (one - x)).powi(5);
    (rho * switch, switch + rho * slope)
}

/// Total VV10 energy for the density `p_tot` on `grid`.
pub fn vv10_energy(
    mb: &MolBasis,
    grid: &MolecularGrid,
    p_tot: &Matrix,
    params: Vv10Params,
) -> Result<f64> {
    let n = grid.len();
    let nao = mb.nao_sph;
    // ρ and |∇ρ|² on every point (batched).
    let mut rho = vec![0.0; n];
    let mut sig = vec![0.0; n];
    const BATCH: usize = 512;
    let chunks: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..n)
        .step_by(BATCH)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&start| {
            let end = (start + BATCH).min(n);
            let pts = &grid.points[start..end];
            let npts = pts.len();
            let batch = eval_aos_full(mb, pts, true, false);
            let g = batch.grad.as_ref().expect("grads");
            let phip = batch.phi.matmul_seq(p_tot);
            let mut r = vec![0.0; npts];
            let mut s = vec![0.0; npts];
            for i in 0..npts {
                let mut acc = 0.0;
                for mu in 0..nao {
                    acc += phip[(i, mu)] * batch.phi[(i, mu)];
                }
                r[i] = acc.max(0.0);
                let mut gv = [0.0; 3];
                for ax in 0..3 {
                    let mut a = 0.0;
                    for mu in 0..nao {
                        a += phip[(i, mu)] * g[ax][(i, mu)];
                    }
                    gv[ax] = 2.0 * a;
                }
                s[i] = gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2];
            }
            (start, r, s)
        })
        .collect();
    for (start, r, s) in chunks {
        rho[start..start + r.len()].copy_from_slice(&r);
        sig[start..start + s.len()].copy_from_slice(&s);
    }

    // Per-point ω₀ and κ; skip vacuum points.
    let mut idx = Vec::new();
    let mut w0 = Vec::new();
    let mut kappa = Vec::new();
    for i in 0..n {
        if rho[i] <= MOLECULAR_VV10_RHO_CUT {
            continue;
        }
        let (omega, damping) = vv10_local_fields(rho[i], sig[i], params);
        w0.push(omega);
        kappa.push(damping);
        idx.push(i);
    }
    let amplitude: Vec<_> = idx
        .iter()
        .map(|&i| molecular_vv10_amplitude(rho[i]).0)
        .collect();
    let m = idx.len();
    let beta = (1.0 / 32.0) * (3.0 / (params.b * params.b)).powf(0.75);

    let inner: f64 = (0..m)
        .into_par_iter()
        .map(|a| {
            let i = idx[a];
            let pi = grid.points[i];
            let mut acc = 0.0;
            for b_ in 0..m {
                let j = idx[b_];
                let r2 = (pi - grid.points[j]).norm2();
                let g = w0[a] * r2 + kappa[a];
                let gp = w0[b_] * r2 + kappa[b_];
                let phi = -1.5 / (g * gp * (g + gp));
                acc += grid.weights[j] * amplitude[b_] * phi;
            }
            grid.weights[i] * amplitude[a] * (beta + 0.5 * acc)
        })
        .sum();
    Ok(inner)
}

/// VV10 energy AND its Kohn-Sham potential matrix
/// `V_mn = sum_i w_i [vrho_i phi_m phi_n + b_i . (grad phi_m phi_n +
/// phi_m grad phi_n)]` with the effective local derivatives
/// vrho = P1 + rho (dw0/drho P2R + dkap/drho P2K), vsig = rho dw0/dsig P2R
///
/// Self-consistent VV10 adds this to the Fock matrix each iteration, making
/// the composite energy variational and enabling exact analytic derivatives.
pub fn vv10_potential(
    mb: &MolBasis,
    grid: &MolecularGrid,
    p_tot: &Matrix,
    params: Vv10Params,
) -> Result<(f64, Matrix)> {
    nonlocal_potential::<false>(mb, grid, p_tot, params)
}

/// rVV10 nonlocal correlation energy and its derivative with respect to the
/// total AO density matrix. The density and molecular grid are supplied by
/// the caller. Uses the direct real-space kernel of Sabatini et al.,
/// Phys. Rev. B 87, 041108(R) (2013), Eq. (5).
pub fn rvv10_potential(
    mb: &MolBasis,
    grid: &MolecularGrid,
    p_tot: &Matrix,
    params: Vv10Params,
) -> Result<(f64, Matrix)> {
    if p_tot.rows != mb.nao_sph || p_tot.cols != mb.nao_sph {
        return Err(crate::DftError::InvalidInput(
            "rVV10 total density dimensions must match the AO basis".into(),
        ));
    }
    nonlocal_potential::<true>(mb, grid, p_tot, params)
}

fn nonlocal_potential<const REVISED: bool>(
    mb: &MolBasis,
    grid: &MolecularGrid,
    p_tot: &Matrix,
    params: Vv10Params,
) -> Result<(f64, Matrix)> {
    let n = grid.len();
    let nao = mb.nao_sph;
    // Pass 1: rho, sigma, grad rho.
    let mut rho = vec![0.0; n];
    let mut sig = vec![0.0; n];
    let mut grho = vec![[0.0; 3]; n];
    const BATCH: usize = 512;
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
            let phip = batch.phi.matmul_seq(p_tot);
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
        rho[start..start + r.len()].copy_from_slice(&r);
        sig[start..start + s.len()].copy_from_slice(&s);
        grho[start..start + gr.len()].copy_from_slice(&gr);
    }
    let fields = nonlocal_density_fields::<_, REVISED>(grid, &rho, &sig, params)?;
    let energy = fields.energy;
    let vrho = fields.d_rho;
    let vsig = fields.d_sigma;
    // Matrix assembly.
    let mut v = Matrix::zeros(nao, nao);
    let mut start2 = 0;
    while start2 < n {
        let end = (start2 + BATCH).min(n);
        let pts = &grid.points[start2..end];
        let npts = pts.len();
        let batch = eval_aos_full(mb, pts, true, false);
        let g = batch.grad.as_ref().expect("grads");
        for i in 0..npts {
            let gi = start2 + i;
            let (vr, vs) = (vrho[gi], vsig[gi]);
            if vr == 0.0 && vs == 0.0 {
                continue;
            }
            let w = grid.weights[gi];
            let bvec = [
                2.0 * w * vs * grho[gi][0],
                2.0 * w * vs * grho[gi][1],
                2.0 * w * vs * grho[gi][2],
            ];
            let f = w * vr;
            for mu in 0..nao {
                let pm = batch.phi[(i, mu)];
                for nu in 0..nao {
                    let pn = batch.phi[(i, nu)];
                    let mut val = f * pm * pn;
                    for u in 0..3 {
                        val += bvec[u] * (g[u][(i, mu)] * pn + pm * g[u][(i, nu)]);
                    }
                    v[(mu, nu)] += val;
                }
            }
        }
        start2 = end;
    }
    Ok((energy, v))
}

/// Analytic nuclear gradient of the VV10 energy at fixed density
/// (post-SCF convention, matching `vv10_energy`): orbital terms through
/// D_a rho / D_a sigma, Becke weight response, grid-point motion, and the
/// explicit kernel R_ij motion. Grid points ride their parent atoms.
pub fn vv10_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p_tot: &Matrix,
    params: Vv10Params,
) -> Result<Vec<Vec3>> {
    let n = grid.len();
    let nao = mb.nao_sph;
    let natoms = mol.len();
    // Pass 1: rho, sigma, grad-rho everywhere.
    let mut rho = vec![0.0; n];
    let mut sig = vec![0.0; n];
    let mut grho = vec![[0.0; 3]; n];
    const BATCH: usize = 512;
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
            let phip = batch.phi.matmul_seq(p_tot);
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
        rho[start..start + r.len()].copy_from_slice(&r);
        sig[start..start + s.len()].copy_from_slice(&s);
        grho[start..start + gr.len()].copy_from_slice(&gr);
    }
    // Kernel ingredients on kept points.
    let mut idx = Vec::new();
    let mut w0 = Vec::new();
    let mut kap = Vec::new();
    for i in 0..n {
        if rho[i] <= MOLECULAR_VV10_RHO_CUT {
            continue;
        }
        let (omega, damping) = vv10_local_fields(rho[i], sig[i], params);
        w0.push(omega);
        kap.push(damping);
        idx.push(i);
    }
    let amplitudes: Vec<_> = idx
        .iter()
        .map(|&i| molecular_vv10_amplitude(rho[i]))
        .collect();
    let m = idx.len();
    let beta = (1.0 / 32.0) * (3.0 / (params.b * params.b)).powf(0.75);

    // Pass 2 (O(m^2)): P1 = beta + sum wrho Phi; the dPhi/dg accumulators
    // with and without R^2; and the explicit R-motion pair force.
    struct KernelOut {
        p1: f64,
        p2r: f64,
        p2k: f64,
        rforce: Vec3,
    }
    let kout: Vec<KernelOut> = (0..m)
        .into_par_iter()
        .map(|a| {
            let i = idx[a];
            let pi = grid.points[i];
            let (mut p1, mut p2r, mut p2k) = (beta, 0.0, 0.0);
            let mut rf = Vec3::zero();
            for b_ in 0..m {
                let j = idx[b_];
                let dvec = pi - grid.points[j];
                let r2 = dvec.norm2();
                let g = w0[a] * r2 + kap[a];
                let gp = w0[b_] * r2 + kap[b_];
                let phi = -1.5 / (g * gp * (g + gp));
                let wr = grid.weights[j] * amplitudes[b_].0;
                p1 += wr * phi;
                // dPhi/dg = -Phi (1/g + 1/(g+g')).
                let dphi_dg = -phi * (1.0 / g + 1.0 / (g + gp));
                let dphi_dgp = -phi * (1.0 / gp + 1.0 / (g + gp));
                p2r += wr * dphi_dg * r2;
                p2k += wr * dphi_dg;
                // dT/dR^2 pair force (both g's move with R^2).
                rf += dvec * (wr * (dphi_dg * w0[a] + dphi_dgp * w0[b_]));
            }
            KernelOut {
                p1,
                p2r,
                p2k,
                rforce: rf,
            }
        })
        .collect();
    // Effective local potentials: dE/drho_i = w_i vrho_i etc.
    let mut vrho = vec![0.0; n];
    let mut vsig = vec![0.0; n];
    let mut e_w = vec![0.0; n]; // dE/dw_i = rho_i P1_i
    let mut acc = vec![Vec3::zero(); natoms];
    for a in 0..m {
        let i = idx[a];
        let (amplitude, derivative) = amplitudes[a];
        let [dw0_drho, dw0_dsig, dkap_drho] =
            vv10_local_partials(rho[i], sig[i], w0[a], kap[a], params.c);
        vrho[i] = derivative * kout[a].p1
            + amplitude * (dw0_drho * kout[a].p2r + dkap_drho * kout[a].p2k);
        vsig[i] = amplitude * dw0_dsig * kout[a].p2r;
        e_w[i] = amplitude * kout[a].p1;
        // Explicit R motion: the pair 1/2 cancels against the delta_oa /
        // delta_ob relabeling, leaving a net factor 2 on the owner side.
        acc[grid.atom_of[i]] += kout[a].rforce * (2.0 * grid.weights[i] * amplitude);
    }

    // Pass 3: orbital + weight + point-motion contraction (total-density
    // channel; mirrors xc_gradient_full).
    let ao_atom = mb.sph_ao_atoms();
    let centers: Vec<Vec3> = mol.atoms.iter().map(|at| at.position).collect();
    let aij = MolecularGrid::becke_aij(mol);
    let ranges: Vec<(usize, usize)> = (0..n).step_by(256).map(|s| (s, (s + 256).min(n))).collect();
    let partials: Vec<Vec<Vec3>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let pts = &grid.points[start..end];
            let wts = &grid.weights[start..end];
            let raws = &grid.raw_weights[start..end];
            let owners = &grid.atom_of[start..end];
            let npts = pts.len();
            let batch = eval_aos_full(mb, pts, true, true);
            let g = batch.grad.as_ref().expect("grads");
            let hess = batch.hess.as_ref().expect("hessians");
            let x = batch.phi.matmul_seq(p_tot);
            let y = [
                g[0].matmul_seq(p_tot),
                g[1].matmul_seq(p_tot),
                g[2].matmul_seq(p_tot),
            ];
            let hidx = |u: usize, v: usize| -> usize {
                let (a, b) = if u <= v { (u, v) } else { (v, u) };
                match (a, b) {
                    (0, 0) => 0,
                    (0, 1) => 1,
                    (0, 2) => 2,
                    (1, 1) => 3,
                    (1, 2) => 4,
                    _ => 5,
                }
            };
            let mut part = vec![Vec3::zero(); natoms];
            for ip in 0..npts {
                let i = start + ip;
                let (vr, vs) = (vrho[i], vsig[i]);
                if vr == 0.0 && vs == 0.0 && e_w[i] == 0.0 {
                    continue;
                }
                let w = wts[ip];
                let b_vec = [
                    2.0 * vs * grho[i][0],
                    2.0 * vs * grho[i][1],
                    2.0 * vs * grho[i][2],
                ];
                // Orbital term.
                for mu in 0..nao {
                    let xm = x[(ip, mu)];
                    let atom = ao_atom[mu];
                    let mut gvec = [0.0; 3];
                    for ax in 0..3 {
                        let mut t = vr * g[ax][(ip, mu)] * xm;
                        for u in 0..3 {
                            t += b_vec[u]
                                * (hess[hidx(ax, u)][(ip, mu)] * xm
                                    + g[ax][(ip, mu)] * y[u][(ip, mu)]);
                        }
                        gvec[ax] = t;
                    }
                    part[atom] +=
                        Vec3::new(-2.0 * w * gvec[0], -2.0 * w * gvec[1], -2.0 * w * gvec[2]);
                }
                // Weight response (dE/dw_i = e_w).
                let (_, dwb) = becke_weight_gradient(&centers, &aij, natoms, owners[ip], pts[ip]);
                let mut dwb_sum = Vec3::zero();
                for gat in 0..natoms {
                    part[gat] += dwb[gat] * (raws[ip] * e_w[i]);
                    dwb_sum += dwb[gat];
                }
                part[owners[ip]] -= dwb_sum * (raws[ip] * e_w[i]);
                // Point motion: w (vr grad rho + vs grad sigma).
                let mut hr = [[0.0; 3]; 3];
                for u in 0..3 {
                    for v in u..3 {
                        let mut t1 = 0.0;
                        let mut t2 = 0.0;
                        for mu in 0..nao {
                            t1 += y[u][(ip, mu)] * g[v][(ip, mu)];
                            t2 += x[(ip, mu)] * hess[hidx(u, v)][(ip, mu)];
                        }
                        let val = 2.0 * (t1 + t2);
                        hr[u][v] = val;
                        hr[v][u] = val;
                    }
                }
                let mut ge = Vec3::zero();
                for u in 0..3 {
                    let mut dsig_u = 0.0;
                    for v in 0..3 {
                        dsig_u += 2.0 * hr[u][v] * grho[i][v];
                    }
                    let val = vr * grho[i][u] + vs * dsig_u;
                    match u {
                        0 => ge.x = val,
                        1 => ge.y = val,
                        _ => ge.z = val,
                    }
                }
                part[owners[ip]] += ge * w;
            }
            part
        })
        .collect();
    for p in partials {
        for (t, v) in acc.iter_mut().zip(p) {
            *t += v;
        }
    }
    Ok(acc)
}

/// The VV10 pair energy density e(i,j) = ρ_i ρ_j Φ_ij as a Scalar-generic
/// closed form (R² also generic for the Hessian's kernel-motion terms) —
/// nested duals extract every kernel partial exactly.
pub(crate) fn vv10_pair_e5<S: Scalar>(ri: S, si: S, rj: S, sj: S, r2: S, params: Vv10Params) -> S {
    let endpoint = |rho, sigma| {
        let (omega, kappa) = vv10_local_fields(rho, sigma, params);
        Vv10Endpoint {
            rho: molecular_vv10_amplitude(rho).0,
            omega,
            kappa,
        }
    };
    vv10_prepared_pair(endpoint(ri, si), endpoint(rj, sj), r2)
}

/// Point-local ingredients of the same VV10 pair. Periodic lattice images
/// share their endpoint density, so roots and fractional powers need only
/// one evaluation per point rather than one per pair image.
#[derive(Clone, Copy)]
pub(crate) struct Vv10Endpoint<S> {
    rho: S,
    omega: S,
    kappa: S,
}

impl<S: Scalar> Vv10Endpoint<S> {
    pub(crate) fn new(rho: S, sigma: S, params: Vv10Params) -> Self {
        let four_pi_3 = 4.0 * std::f64::consts::PI / 3.0;
        let kap_c =
            params.b * (1.5 * std::f64::consts::PI) / (9.0 * std::f64::consts::PI).powf(1.0 / 6.0);
        let r2i = (rho * rho).recip();
        Self {
            rho,
            omega: (S::from_f64(params.c) * sigma * sigma * r2i * r2i
                + S::from_f64(four_pi_3) * rho)
                .sqrt(),
            kappa: S::from_f64(kap_c) * rho.powf(1.0 / 6.0),
        }
    }
    pub(crate) fn map<T: Scalar>(self, f: impl Fn(S) -> T) -> Vv10Endpoint<T> {
        Vv10Endpoint {
            rho: f(self.rho),
            omega: f(self.omega),
            kappa: f(self.kappa),
        }
    }
}

pub(crate) fn vv10_prepared_pair<S: Scalar>(i: Vv10Endpoint<S>, j: Vv10Endpoint<S>, r2: S) -> S {
    let g = i.omega * r2 + i.kappa;
    let gp = j.omega * r2 + j.kappa;
    i.rho * j.rho * S::from_f64(-1.5) / (g * gp * (g + gp))
}

/// f_nlc response: the second functional derivative of the VV10 energy
/// contracted with a (symmetrized) total-density response ΔP:
///   (f_nlc·ΔP)_μν = d/dt V_nlc[P + tΔP]_μν |_{t=0}.
/// Differentiate the shared variational potential fields with both pair
/// endpoints displaced. The assembly also differentiates the local
/// density-gradient factor multiplying the sigma potential.
/// Used by the CPKS/TDDFT response path of the -V functionals.
pub fn vv10_fxc_contraction(
    mb: &MolBasis,
    grid: &MolecularGrid,
    p_tot: &Matrix,
    dp_tot: &Matrix,
    params: Vv10Params,
) -> Result<Matrix> {
    let n = grid.len();
    let nao = mb.nao_sph;
    // Pass 1: rho/sigma/grad-rho of P and the response drho/dgrad-rho of ΔP.
    let mut rho = vec![0.0; n];
    let mut sig = vec![0.0; n];
    let mut grho = vec![[0.0; 3]; n];
    let mut drho = vec![0.0; n];
    let mut dgrho = vec![[0.0; 3]; n];
    const BATCH: usize = 512;
    #[allow(clippy::type_complexity)]
    let chunks: Vec<(
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<[f64; 3]>,
        Vec<f64>,
        Vec<[f64; 3]>,
    )> = (0..n)
        .step_by(BATCH)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&start| {
            let end = (start + BATCH).min(n);
            let pts = &grid.points[start..end];
            let npts = pts.len();
            let batch = eval_aos_full(mb, pts, true, false);
            let g = batch.grad.as_ref().expect("grads");
            let phip = batch.phi.matmul_seq(p_tot);
            let phidp = batch.phi.matmul_seq(dp_tot);
            let mut r = vec![0.0; npts];
            let mut s = vec![0.0; npts];
            let mut gr = vec![[0.0; 3]; npts];
            let mut dr = vec![0.0; npts];
            let mut dgr = vec![[0.0; 3]; npts];
            for i in 0..npts {
                let (mut acc, mut dacc) = (0.0, 0.0);
                for mu in 0..nao {
                    acc += phip[(i, mu)] * batch.phi[(i, mu)];
                    dacc += phidp[(i, mu)] * batch.phi[(i, mu)];
                }
                r[i] = acc.max(0.0);
                dr[i] = dacc;
                for ax in 0..3 {
                    let (mut a, mut da) = (0.0, 0.0);
                    for mu in 0..nao {
                        a += phip[(i, mu)] * g[ax][(i, mu)];
                        da += phidp[(i, mu)] * g[ax][(i, mu)];
                    }
                    gr[i][ax] = 2.0 * a;
                    dgr[i][ax] = 2.0 * da;
                }
                s[i] = gr[i][0] * gr[i][0] + gr[i][1] * gr[i][1] + gr[i][2] * gr[i][2];
            }
            (start, r, s, gr, dr, dgr)
        })
        .collect();
    for (start, r, s, gr, dr, dgr) in chunks {
        let len = r.len();
        rho[start..start + len].copy_from_slice(&r);
        sig[start..start + len].copy_from_slice(&s);
        grho[start..start + len].copy_from_slice(&gr);
        drho[start..start + len].copy_from_slice(&dr);
        dgrho[start..start + len].copy_from_slice(&dgr);
    }
    // The shared endpoint kernel evaluates the frequency and damping only
    // once per point. A single directional dual carries both density changes
    // through the same pair sum, including its original vacuum selection.
    let density: Vec<_> = rho
        .into_iter()
        .zip(drho)
        .map(|(re, derivative)| DualN {
            re,
            im: [derivative],
        })
        .collect();
    let sigma: Vec<_> = sig
        .into_iter()
        .zip(grho.iter().zip(&dgrho))
        .map(|(re, (gradient, response))| {
            let contraction: f64 = gradient.iter().zip(response).map(|(g, dg)| g * dg).sum();
            DualN {
                re,
                im: [2.0 * contraction],
            }
        })
        .collect();
    let fields = vv10_density_fields(grid, &density, &sigma, params)?;

    // Assembly: ΔV = Σ w [Δvρ φφ + (2Δvσ ∇ρ + 2vσ ∇Δρ)·(∇φ φ + φ ∇φ)].
    let mut v = Matrix::zeros(nao, nao);
    let mut start2 = 0;
    while start2 < n {
        let end = (start2 + BATCH).min(n);
        let pts = &grid.points[start2..end];
        let npts = pts.len();
        let batch = eval_aos_full(mb, pts, true, false);
        let g = batch.grad.as_ref().expect("grads");
        for i in 0..npts {
            let gi = start2 + i;
            let potential = fields.d_sigma[gi];
            let (dvr, dvs, vs) = (fields.d_rho[gi].im[0], potential.im[0], potential.re);
            if dvr == 0.0 && dvs == 0.0 && vs == 0.0 {
                continue;
            }
            let w = grid.weights[gi];
            let bvec = [
                w * (2.0 * dvs * grho[gi][0] + 2.0 * vs * dgrho[gi][0]),
                w * (2.0 * dvs * grho[gi][1] + 2.0 * vs * dgrho[gi][1]),
                w * (2.0 * dvs * grho[gi][2] + 2.0 * vs * dgrho[gi][2]),
            ];
            let f = w * dvr;
            for mu in 0..nao {
                let pm = batch.phi[(i, mu)];
                for nu in 0..nao {
                    let pn = batch.phi[(i, nu)];
                    let mut val = f * pm * pn;
                    for u in 0..3 {
                        val += bvec[u] * (g[u][(i, mu)] * pn + pm * g[u][(i, nu)]);
                    }
                    v[(mu, nu)] += val;
                }
            }
        }
        start2 = end;
    }
    Ok(v)
}

/// Shared unweighted endpoint partials of the molecular pair functional.
pub(crate) struct Vv10DensityFields<S> {
    pub energy: S,
    pub d_rho: Vec<S>,
    pub d_sigma: Vec<S>,
}

pub(crate) fn vv10_local_fields<S: Scalar>(rho: S, sigma: S, parameters: Vv10Params) -> (S, S) {
    let pi = std::f64::consts::PI;
    let omega = local_frequency(rho, sigma, parameters.c);
    let kappa =
        S::from_f64(parameters.b * 1.5 * pi / (9.0 * pi).powf(1.0 / 6.0)) * rho.powf(1.0 / 6.0);
    (omega, kappa)
}

fn local_frequency<S: Scalar>(rho: S, sigma: S, gap: f64) -> S {
    let reduced = sigma / rho / rho;
    (S::from_f64(gap) * reduced * reduced + S::from_f64(4.0 * std::f64::consts::PI / 3.0) * rho)
        .sqrt()
}

fn vv10_local_partials<S: Scalar>(rho: S, sigma: S, omega: S, kappa: S, gap: f64) -> [S; 3] {
    let reduced = sigma / rho / rho;
    let gap = S::from_f64(gap);
    [
        (S::from_f64(4.0 * std::f64::consts::PI / 3.0)
            - S::from_f64(4.0) * gap * reduced * reduced / rho)
            / (S::from_f64(2.0) * omega),
        gap * reduced / rho / rho / omega,
        kappa / (S::from_f64(6.0) * rho),
    ]
}

/// One endpoint row; the symmetric partner derivative is included in its
/// potential, whereas energy retains the exact pair factor one half.
pub(crate) fn vv10_density_row<S: Scalar>(
    grid: &MolecularGrid,
    indices: &[usize],
    a: usize,
    rho: &[S],
    omega: &[S],
    kappa: &[S],
    sigma: S,
    parameters: Vv10Params,
) -> [S; 3] {
    let amplitudes: Vec<_> = rho.iter().copied().map(molecular_vv10_amplitude).collect();
    nonlocal_geometry_row(
        indices,
        a,
        rho,
        &amplitudes,
        (omega[a], kappa[a]),
        sigma,
        parameters,
        |i, j| S::from_f64((grid.points[i] - grid.points[j]).norm2()),
        |i| S::from_f64(grid.weights[i]),
        |b, r2, wr| vv10_pair_partials(omega[a], kappa[a], omega[b], kappa[b], r2, wr),
    )
}

// Return the weighted kernel and its derivatives with respect to the left
// endpoint's frequency and damping. Both kernels share the density chain rule.
fn vv10_pair_partials<S: Scalar>(w: S, k: S, wp: S, kp: S, r2: S, wr: S) -> [S; 3] {
    let g = w * r2 + k;
    let gp = wp * r2 + kp;
    let phi = S::from_f64(-1.5) / (g * gp * (g + gp));
    let derivative = -phi * (g.recip() + (g + gp).recip());
    [wr * phi, wr * derivative * r2, wr * derivative]
}

#[derive(Clone, Copy)]
struct Rvv10Endpoint<S> {
    inv_kappa: S,
    q: S,
    amplitude: S,
}

impl<S: Scalar> Rvv10Endpoint<S> {
    fn new(omega: S, kappa: S) -> Self {
        let inv_kappa = kappa.recip();
        Self {
            inv_kappa,
            q: omega * inv_kappa,
            amplitude: inv_kappa * inv_kappa.sqrt(),
        }
    }

    fn pair_partials(self, other: Self, r2: S, wr: S) -> [S; 3] {
        // The revised denominator attaches a unit intercept to each qR².
        // The removed kappa factors form the product of endpoint amplitudes.
        let one = S::from_f64(1.0);
        let g = one + self.q * r2;
        let gp = one + other.q * r2;
        let phi = S::from_f64(-1.5) * self.amplitude * other.amplitude * wr / (g * gp * (g + gp));
        let d_q = -phi * (g.recip() + (g + gp).recip()) * r2;
        [
            phi,
            d_q * self.inv_kappa,
            (-S::from_f64(1.5) * phi - d_q * self.q) * self.inv_kappa,
        ]
    }
}

#[allow(clippy::too_many_arguments)]
fn nonlocal_geometry_row<S: Scalar>(
    indices: &[usize],
    a: usize,
    rho: &[S],
    amplitudes: &[(S, S)],
    local: (S, S),
    sigma: S,
    parameters: Vv10Params,
    distance: impl Fn(usize, usize) -> S,
    weight: impl Fn(usize) -> S,
    pair_partials: impl Fn(usize, S, S) -> [S; 3],
) -> [S; 3] {
    let beta = (3.0 / parameters.b.powi(2)).powf(0.75) / 32.0;
    let r = rho[a];
    let (amplitude, derivative) = amplitudes[a];
    let (w, k) = local;
    let mut p1 = S::from_f64(beta);
    let mut p2r = S::from_f64(0.0);
    let mut p2k = p2r;
    for b in 0..indices.len() {
        let r2 = distance(indices[a], indices[b]);
        let wr = weight(indices[b]) * amplitudes[b].0;
        let [phi, d_omega, d_kappa] = pair_partials(b, r2, wr);
        p1 = p1 + phi;
        p2r = p2r + d_omega;
        p2k = p2k + d_kappa;
    }
    let [drw, dsw, drk] = vv10_local_partials(r, sigma, w, k, parameters.c);
    [
        amplitude * (S::from_f64(beta) + S::from_f64(0.5) * (p1 - S::from_f64(beta))),
        derivative * p1 + amplitude * (drw * p2r + drk * p2k),
        amplitude * dsw * p2r,
    ]
}

pub(crate) fn vv10_density_fields<S: Scalar + Send + Sync>(
    grid: &MolecularGrid,
    rho: &[S],
    sigma: &[S],
    parameters: Vv10Params,
) -> Result<Vv10DensityFields<S>> {
    nonlocal_density_fields::<_, false>(grid, rho, sigma, parameters)
}

fn nonlocal_density_fields<S: Scalar + Send + Sync, const REVISED: bool>(
    grid: &MolecularGrid,
    rho: &[S],
    sigma: &[S],
    parameters: Vv10Params,
) -> Result<Vv10DensityFields<S>> {
    let n = grid.len();
    if rho.len() != n
        || sigma.len() != n
        || grid.weights.len() != n
        || !parameters.b.is_finite()
        || parameters.b <= 0.0
        || !parameters.c.is_finite()
        || parameters.c < 0.0
        || rho.iter().chain(sigma).any(|v| !v.value().is_finite())
        || sigma.iter().any(|v| v.value() < 0.0)
        || grid.weights.iter().any(|w| !w.is_finite())
        || grid.points.iter().any(|p| !p.norm().is_finite())
    {
        return Err(crate::DftError::InvalidInput(
            "invalid molecular VV10 endpoint fields, grid or parameters".into(),
        ));
    }
    nonlocal_geometry_fields::<_, REVISED>(
        rho,
        sigma,
        parameters,
        |i, j| S::from_f64((grid.points[i] - grid.points[j]).norm2()),
        |i| S::from_f64(grid.weights[i]),
    )
}

/// Moving molecular coordinates and weights with the SCF vacuum convention.
pub(crate) fn vv10_moving_density_fields<S: Scalar + Send + Sync>(
    positions: &[[S; 3]],
    weights: &[S],
    rho: &[S],
    sigma: &[S],
    parameters: Vv10Params,
) -> Result<Vv10DensityFields<S>> {
    if positions.len() != rho.len()
        || weights.len() != rho.len()
        || positions
            .iter()
            .flatten()
            .chain(weights)
            .any(|v| !v.value().is_finite())
    {
        return Err(crate::DftError::InvalidInput(
            "invalid moving molecular VV10 geometry or weights".into(),
        ));
    }
    nonlocal_geometry_fields::<_, false>(
        rho,
        sigma,
        parameters,
        |i, j| {
            (0..3)
                .map(|axis| {
                    let d = positions[i][axis] - positions[j][axis];
                    d * d
                })
                .fold(S::from_f64(0.0), |a, b| a + b)
        },
        |i| weights[i],
    )
}

fn nonlocal_geometry_fields<S: Scalar + Send + Sync, const REVISED: bool>(
    rho: &[S],
    sigma: &[S],
    parameters: Vv10Params,
    distance: impl Fn(usize, usize) -> S + Sync,
    weight: impl Fn(usize) -> S + Sync,
) -> Result<Vv10DensityFields<S>> {
    if rho.len() != sigma.len()
        || !parameters.b.is_finite()
        || parameters.b <= 0.0
        || !parameters.c.is_finite()
        || parameters.c < 0.0
        || rho.iter().chain(sigma).any(|v| !v.value().is_finite())
        || sigma.iter().any(|v| v.value() < 0.0)
    {
        return Err(crate::DftError::InvalidInput(
            "invalid molecular VV10 endpoint fields or parameters".into(),
        ));
    }
    let indices: Vec<_> = (0..rho.len())
        .filter(|&i| {
            if REVISED {
                rho[i].value() >= RHO_CUT
            } else {
                rho[i].value() > MOLECULAR_VV10_RHO_CUT
            }
        })
        .collect();
    let values: Vec<_> = indices.iter().map(|&i| rho[i]).collect();
    let amplitudes: Vec<_> = values
        .iter()
        .copied()
        .map(|r| {
            if REVISED {
                (r, S::from_f64(1.0))
            } else {
                molecular_vv10_amplitude(r)
            }
        })
        .collect();
    let local: Vec<_> = indices
        .iter()
        .map(|&i| vv10_local_fields(rho[i], sigma[i], parameters))
        .collect();
    let revised: Vec<_> = if REVISED {
        local
            .iter()
            .map(|&(w, k)| Rvv10Endpoint::new(w, k))
            .collect()
    } else {
        Vec::new()
    };
    let rows: Vec<_> = (0..indices.len())
        .into_par_iter()
        .map(|a| {
            nonlocal_geometry_row(
                &indices,
                a,
                &values,
                &amplitudes,
                local[a],
                sigma[indices[a]],
                parameters,
                &distance,
                &weight,
                |b, r2, wr| {
                    if REVISED {
                        revised[a].pair_partials(revised[b], r2, wr)
                    } else {
                        vv10_pair_partials(local[a].0, local[a].1, local[b].0, local[b].1, r2, wr)
                    }
                },
            )
        })
        .collect();
    let mut result = Vv10DensityFields {
        energy: S::from_f64(0.0),
        d_rho: vec![S::from_f64(0.0); rho.len()],
        d_sigma: vec![S::from_f64(0.0); rho.len()],
    };
    for (&i, row) in indices.iter().zip(rows) {
        result.energy = result.energy + weight(i) * row[0];
        result.d_rho[i] = row[1];
        result.d_sigma[i] = row[2];
    }
    Ok(result)
}
