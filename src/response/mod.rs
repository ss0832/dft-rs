// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Generic coupled-perturbed KS/HF (linear-response) solver.
//!
//! Unknowns are the occupied→virtual rotation amplitudes `U^σ_ai`; the
//! orbital-Hessian × vector product
//! `(ε_a−ε_i)U + [J(ΔP) − cx K(ΔP^σ) − clr K_lr(ΔP^σ) + f_xc·ΔP]_ai`
//! is applied matrix-free (direct AO builds + grid kernel contraction) and
//! the system is solved by Jacobi-preconditioned conjugate gradients. The
//! same product serves the analytic Hessian (nuclear perturbations), the
//! analytic polarizability (electric field), and later TDDFT (Davidson).

pub mod anharmonic;
pub(crate) mod berry_field;
pub mod complex;
pub mod constraints;
pub mod correlated_field;
pub mod density_kernel;
pub mod diagnostics;
pub mod dynamic;
pub mod dynamic_tensor;
pub mod electric_process;
pub mod electric_tensor;
pub mod ensemble;
pub mod field_path;
pub mod force_constants;
pub mod krylov;
pub mod london_ecd;
pub mod magnetic_mixed;
pub mod magnetic_taylor;
pub mod medium_optics;
pub mod mgga_kernel;
pub mod mixed;
pub mod mrsf;
pub mod nambu;
pub mod nonlinear;
pub mod nonlocal;
pub mod nuclear;
pub mod orbital;
pub mod thermal_dynamic;
pub use orbital::{
    frontier_gap_gradient, orbital_energy_gradients, FrontierGap, FrontierGapGradient,
    OrbitalCluster, OrbitalEnergyGradients, OrbitalGradientOptions,
};
pub mod series;
pub mod spin_flip;
pub mod spin_flip_kernel;
pub mod spin_spin;
pub mod spinor;
pub mod spinor_tda;
pub mod stability;
pub mod superconducting;
pub mod thermal_matrix;
pub use stability::{
    molecular_stability, rotated_density, stability_from_state, stabilize, DescentStep,
    MolecularStability, StabilityBlock, StabilityChannel, StabilityExcitation, StabilityMode,
    StabilityOptions, StabilizeOptions, StabilizeReport,
};
pub mod taylor;
pub use dynamic::{dynamic_polarizability, DynamicPolarizability, Frequency, ResponseReport};
pub use dynamic_tensor::{
    dynamic_beta, dynamic_electric_tensor, dynamic_gamma, DynamicElectricTensor,
};
pub use electric_process::{dynamic_electric_process, ElectricProcess};
pub use electric_tensor::{
    static_beta, static_beta_variational, static_electric_tensor, static_gamma,
    StaticElectricTensor,
};
pub use krylov::ResponseSolverConfig;
pub use nonlinear::{
    electric_field_taylor, mixed_electric_field_response, ElectricFieldTaylor,
    MixedElectricFieldResponse,
};
pub mod correlated_tda;
mod orbital_frame;
pub mod polarizability;
pub mod tddft;
pub mod tddft_grad;

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::eval::eval_aos_full;
use crate::grid::MolecularGrid;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::scf::{JkEngine, ScfOptions, ScfResult};
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use rayon::prelude::*;
use std::sync::OnceLock;

// Shared batch size keeps the derivative-cache slots aligned with contractions.
const SEMILOCAL_RESPONSE_BATCH: usize = 256;

fn semilocal_cache_bytes(points: usize, gga: bool) -> Option<usize> {
    let batches = points.div_ceil(SEMILOCAL_RESPONSE_BATCH);
    points
        .checked_mul(if gga { 18 * 8 } else { 3 * 8 })
        .and_then(|data| {
            batches
                .checked_mul(std::mem::size_of::<OnceLock<XcOut>>())
                .and_then(|headers| data.checked_add(headers))
        })
}

fn semilocal_cache_slots(points: usize, gga: bool, budget: usize) -> usize {
    if semilocal_cache_bytes(points, gga).is_some_and(|bytes| bytes <= budget) {
        points.div_ceil(SEMILOCAL_RESPONSE_BATCH)
    } else {
        0
    }
}

fn semilocal_cache(
    points: usize,
    gga: bool,
    budget: usize,
) -> Result<(Vec<OnceLock<XcOut>>, crate::resources::MemoryReservation)> {
    let slots = semilocal_cache_slots(points, gga, budget);
    if slots != 0 {
        let bytes = semilocal_cache_bytes(points, gga).expect("cache size checked");
        if let Some(lease) =
            crate::resources::reserve_cache(bytes, "semilocal XC derivative cache")?
        {
            return Ok(((0..slots).map(|_| OnceLock::new()).collect(), lease));
        }
    }
    Ok((Vec::new(), Default::default()))
}

/// Shared semilocal density Hessian. Both ordinary orbital response and
/// frozen off-shell density models supply their immutable spin densities.
fn semilocal_density_contraction(
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
    cache: &[OnceLock<XcOut>],
    dp: &[Matrix; 2],
) -> Result<[Matrix; 2]> {
    let gga = functional.family != Family::Lda;
    let nao = mb.nao_sph;
    const BATCH: usize = SEMILOCAL_RESPONSE_BATCH;
    let n = grid.len();
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(BATCH)
        .map(|s| (s, (s + BATCH).min(n)))
        .collect();
    let partials: Result<Vec<[Matrix; 2]>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let pts = &grid.points[start..end];
            let wts = &grid.weights[start..end];
            let npts = pts.len();
            let batch = eval_aos_full(mb, pts, gga, false);
            let g = batch.grad.as_ref();
            // Ground-state and perturbed densities/gradients.
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut drho = [vec![0.0; npts], vec![0.0; npts]];
            let mut u_g: [[Vec<f64>; 3]; 2] = Default::default();
            let mut du_g: [[Vec<f64>; 3]; 2] = Default::default();
            for spin in 0..2 {
                let x0 = batch.phi.matmul_seq(p[spin]);
                let x1 = batch.phi.matmul_seq(&dp[spin]);
                for i in 0..npts {
                    let mut r0 = 0.0;
                    let mut r1 = 0.0;
                    for mu in 0..nao {
                        r0 += x0[(i, mu)] * batch.phi[(i, mu)];
                        r1 += x1[(i, mu)] * batch.phi[(i, mu)];
                    }
                    rho[spin][i] = r0.max(0.0);
                    drho[spin][i] = r1;
                }
                if gga {
                    let g = g.expect("grads");
                    for ax in 0..3 {
                        let mut v0 = vec![0.0; npts];
                        let mut v1 = vec![0.0; npts];
                        for i in 0..npts {
                            let mut a0 = 0.0;
                            let mut a1 = 0.0;
                            for mu in 0..nao {
                                a0 += x0[(i, mu)] * g[ax][(i, mu)];
                                a1 += x1[(i, mu)] * g[ax][(i, mu)];
                            }
                            v0[i] = 2.0 * a0;
                            v1[i] = 2.0 * a1;
                        }
                        u_g[spin][ax] = v0;
                        du_g[spin][ax] = v1;
                    }
                }
            }
            let sigma_store;
            let vars = if gga {
                let mut s3 = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
                for i in 0..npts {
                    let ua = [u_g[0][0][i], u_g[0][1][i], u_g[0][2][i]];
                    let ub = [u_g[1][0][i], u_g[1][1][i], u_g[1][2][i]];
                    s3[0][i] = ua.iter().map(|v| v * v).sum();
                    s3[1][i] = ua.iter().zip(&ub).map(|(a, b)| a * b).sum();
                    s3[2][i] = ub.iter().map(|v| v * v).sum();
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
            let slot = cache.get(start / BATCH);
            let cached = slot.and_then(OnceLock::get);
            let mut out = cached.cloned().unwrap_or_default();
            if cached.is_none() {
                #[cfg(test)]
                crate::xc::branch_freeze::set_batch(Some(start));
                let evaluated = functional.eval(&vars, 2, &mut out);
                #[cfg(test)]
                crate::xc::branch_freeze::set_batch(None);
                evaluated?;
                let cache_shape_ok = out.v2rho2.iter().all(|v| v.len() == npts)
                    && (!gga
                        || out
                            .vsigma
                            .iter()
                            .chain(&out.v2rhosigma)
                            .chain(&out.v2sigma2)
                            .all(|v| v.len() == npts));
                if let Some(slot) = slot.filter(|_| cache_shape_ok) {
                    // Retain only active arrays used below. Copies have
                    // exact logical lengths, matching the cache budget.
                    let frozen = XcOut {
                        vsigma: std::array::from_fn(|p| {
                            if gga {
                                out.vsigma[p].to_vec()
                            } else {
                                Vec::new()
                            }
                        }),
                        v2rho2: std::array::from_fn(|p| out.v2rho2[p].to_vec()),
                        v2rhosigma: std::array::from_fn(|p| {
                            if gga {
                                out.v2rhosigma[p].to_vec()
                            } else {
                                Vec::new()
                            }
                        }),
                        v2sigma2: std::array::from_fn(|p| {
                            if gga {
                                out.v2sigma2[p].to_vec()
                            } else {
                                Vec::new()
                            }
                        }),
                        ..XcOut::default()
                    };
                    // Concurrent first callers may compute the same batch;
                    // only one immutable, identical result is retained.
                    let _ = slot.set(frozen);
                }
            }
            // Per-point first-order potential changes.
            let mut dv = [Matrix::zeros(npts, nao), Matrix::zeros(npts, nao)];
            let mut dv_grad: [Option<Matrix>; 2] = [None, None];
            if gga {
                dv_grad = [
                    Some(Matrix::zeros(npts, nao)),
                    Some(Matrix::zeros(npts, nao)),
                ];
            }
            // Δσ contractions.
            let mut dsig = vec![[0.0; 3]; npts];
            if gga {
                for i in 0..npts {
                    let ua = [u_g[0][0][i], u_g[0][1][i], u_g[0][2][i]];
                    let ub = [u_g[1][0][i], u_g[1][1][i], u_g[1][2][i]];
                    let dua = [du_g[0][0][i], du_g[0][1][i], du_g[0][2][i]];
                    let dub = [du_g[1][0][i], du_g[1][1][i], du_g[1][2][i]];
                    dsig[i] = [
                        2.0 * dot3(&ua, &dua),
                        dot3(&ua, &dub) + dot3(&ub, &dua),
                        2.0 * dot3(&ub, &dub),
                    ];
                }
            }
            for spin in 0..2 {
                // v2rho2 blocks: (aa, ab, bb); rows for this spin.
                let frr: [usize; 2] = if spin == 0 { [0, 1] } else { [1, 2] };
                // v2rhosigma rows: (ρσ, σ_p) p = aa,ab,bb.
                let frs0 = if spin == 0 { 0 } else { 3 };
                for i in 0..npts {
                    let w = wts[i];
                    let mut dvr =
                        out.v2rho2[frr[0]][i] * drho[0][i] + out.v2rho2[frr[1]][i] * drho[1][i];
                    if gga {
                        for pth in 0..3 {
                            dvr += out.v2rhosigma[frs0 + pth][i] * dsig[i][pth];
                        }
                    }
                    let f = w * dvr;
                    for mu in 0..nao {
                        dv[spin][(i, mu)] += f * batch.phi[(i, mu)];
                    }
                    if gga {
                        let g = g.expect("grads");
                        // Δvσ for the three pair channels.
                        // block index helper for v2sigma2 (p ≤ q).
                        let ss = |p: usize, q: usize| -> usize {
                            let (a, b) = if p <= q { (p, q) } else { (q, p) };
                            match (a, b) {
                                (0, 0) => 0,
                                (0, 1) => 1,
                                (0, 2) => 2,
                                (1, 1) => 3,
                                (1, 2) => 4,
                                _ => 5,
                            }
                        };
                        let mut dvs = [0.0; 3];
                        for pch in 0..3 {
                            // ρ-σ cross: rows (ρa,p) idx p; (ρb,p) idx 3+p.
                            dvs[pch] = out.v2rhosigma[pch][i] * drho[0][i]
                                + out.v2rhosigma[3 + pch][i] * drho[1][i];
                            for qch in 0..3 {
                                dvs[pch] += out.v2sigma2[ss(pch, qch)][i] * dsig[i][qch];
                            }
                        }
                        let (own, cross) = if spin == 0 { (0, 1) } else { (2, 1) };
                        let other = 1 - spin;
                        // b-vector change: 2Δvσ_own u_own + 2vσ_own Δu_own
                        //               + Δvσ_ab u_other + vσ_ab Δu_other.
                        let mut bvec = [0.0; 3];
                        for ax in 0..3 {
                            bvec[ax] = w
                                * (2.0 * dvs[own] * u_g[spin][ax][i]
                                    + 2.0 * out.vsigma[own][i] * du_g[spin][ax][i]
                                    + dvs[cross] * u_g[other][ax][i]
                                    + out.vsigma[cross][i] * du_g[other][ax][i]);
                        }
                        let dvg = dv_grad[spin].as_mut().expect("gga buffer");
                        for mu in 0..nao {
                            dvg[(i, mu)] += bvec[0] * g[0][(i, mu)]
                                + bvec[1] * g[1][(i, mu)]
                                + bvec[2] * g[2][(i, mu)];
                        }
                    }
                }
            }
            let mut vxc = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
            for spin in 0..2 {
                let mut v = batch.phi.transpose_matmul_seq(&dv[spin]);
                // dv built as (w·Δvρ)φ per point row: Φᵀ diag Φ needs the
                // transpose orientation — v = Φᵀ · dv gives Σ_i φμ dv_iν ✓.
                if let Some(dvg) = &dv_grad[spin] {
                    let cross = dvg.transpose_matmul_seq(&batch.phi);
                    for r in 0..nao {
                        for c in 0..nao {
                            v[(r, c)] += cross[(r, c)] + cross[(c, r)];
                        }
                    }
                }
                vxc[spin] = v;
            }
            Ok(vxc)
        })
        .collect();
    let mut total = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    for part in partials? {
        for spin in 0..2 {
            for (t, s) in total[spin]
                .as_mut_slice()
                .iter_mut()
                .zip(part[spin].as_slice())
            {
                *t += s;
            }
        }
    }
    // Symmetrize.
    for m in total.iter_mut() {
        for i in 0..nao {
            for j in 0..i {
                let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                m[(i, j)] = avg;
                m[(j, i)] = avg;
            }
        }
    }
    Ok(total)
}

/// Occupied/virtual dimensions per spin.
struct OvDims {
    n_occ: [usize; 2],
    n_vir: [usize; 2],
}

struct ReferencePolish {
    rotations: [Matrix; 2],
    initial_residual: f64,
    final_residual: f64,
    iterations: usize,
}

pub struct ResponseSolver<'a> {
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    /// Lazy, bounded derivatives of this solver's immutable reference only.
    /// A missing slot retains the identical streamed provider evaluation.
    semilocal_fxc: Vec<OnceLock<XcOut>>,
    _semilocal_allocation: crate::resources::MemoryReservation,
    jk: JkEngine,
    jk_lr: Option<JkEngine>,
    dims: OvDims,
    frame: orbital_frame::OrbitalFrame<'a>,
    /// Bounded local derivative cache of this immutable meta-GGA reference.
    /// Missing cache retains the identical on-demand kernel evaluation.
    mgga_kernel: Option<mgga_kernel::MetaGgaKernel<'a>>,
    /// Self-consistent VV10 state (params, dedicated coarse grid, P_tot):
    /// when present, `fxc_contraction` includes the exact nonlocal f_nlc
    /// kernel, so CPKS / polarizability / TDDFT responses of the -V
    /// functionals carry the full VV10 coupling.
    nlc: Option<(crate::nlc::Vv10Params, MolecularGrid, Matrix)>,
    /// Continuum reaction field: when present, `fxc_contraction` adds the
    /// polarization kernel, so CPKS / polarizability / TDDFT responses see
    /// the solvent relaxing with the density instead of frozen.
    pcm: Option<crate::pcm::PcmSolver>,
    /// Learned functional and the molecule it is evaluated for: when
    /// present, `fxc_contraction` adds the second derivatives of its energy
    /// contracted with the density response.
    learned: Option<(
        std::sync::Arc<crate::skala::SkalaModel>,
        crate::system::Molecule,
    )>,
    pub max_iter: usize,
    pub tol: f64,
}

impl<'a> ResponseSolver<'a> {
    pub fn new(
        mol: &crate::system::Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        scf: &'a ScfResult,
        opts: &ScfOptions,
    ) -> Result<Self> {
        diagnostics::reference_diagnostics(scf)?;
        Self::construct(mol, mb, functional, grid, scf, opts, None)
    }

    /// The same kernel engines for a reference of any occupations, used by
    /// thermal and prescribed-population responses that carry their own
    /// spectral weights. Only [`Self::fock_response`] and
    /// [`Self::fxc_contraction`] are meaningful for such a reference; the
    /// occupied-virtual CPKS methods assume aufbau occupations.
    pub(crate) fn kernel_only(
        mol: &crate::system::Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        scf: &'a ScfResult,
        opts: &ScfOptions,
    ) -> Result<Self> {
        Self::construct(mol, mb, functional, grid, scf, opts, None)
    }

    /// Reuse immutable integral caches during orbital optimization. The
    /// density-dependent XC kernel and direct-screening scale are rebuilt for
    /// the current trial determinant, while the ERI/RI tensors are shared.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kernel_with_engines(
        mol: &crate::system::Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        scf: &'a ScfResult,
        opts: &ScfOptions,
        jk: &JkEngine,
        jk_lr: Option<&JkEngine>,
    ) -> Result<Self> {
        Self::construct(mol, mb, functional, grid, scf, opts, Some((jk, jk_lr)))
    }

    #[allow(clippy::too_many_arguments)]
    fn construct(
        mol: &crate::system::Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        scf: &'a ScfResult,
        opts: &ScfOptions,
        engines: Option<(&JkEngine, Option<&JkEngine>)>,
    ) -> Result<Self> {
        if mb.nao_sph != scf.overlap.rows {
            return Err(DftError::InvalidInput(
                "response basis and reference AO dimensions disagree".into(),
            ));
        }
        if !functional.parts.is_empty() && grid.is_none() {
            return Err(DftError::InvalidInput(
                "response solver needs a grid for functionals with XC parts".into(),
            ));
        }
        let mut effective_options = opts.clone();
        effective_options.eri_mode = opts.effective_eri_mode(mb.nao_sph);
        let opts = &effective_options;
        let reference_total = scf.p_total();
        let reference_spin = [&scf.density[0], &scf.density[1]];
        let jk = match engines {
            Some((engine, _)) => engine.clone(),
            None => JkEngine::from_options(mol, mb, opts)?,
        }
        .with_reference_screening(&reference_total, reference_spin);
        let jk_lr = if functional.hybrid.clr != 0.0 {
            let engine = match engines {
                Some((_, Some(engine))) => engine.clone(),
                Some((_, None)) => {
                    return Err(DftError::InvalidInput(
                        "orbital kernel requires the reference long-range integral engine".into(),
                    ))
                }
                None => JkEngine::new(
                    mb,
                    opts.eri_mode,
                    CoulombKernel::Erf {
                        omega: functional.hybrid.omega,
                    },
                )?,
            };
            Some(engine.with_reference_screening(&reference_total, reference_spin))
        } else {
            None
        };
        let nmo = [scf.mo_coeff[0].cols, scf.mo_coeff[1].cols];
        let dims = OvDims {
            n_occ: [scf.n_alpha, scf.n_beta],
            n_vir: [nmo[0] - scf.n_alpha, nmo[1] - scf.n_beta],
        };
        let nlc = match opts.vv10 {
            Some(params) => {
                let nlc_grid = opts.nonlocal_grid(mol)?;
                Some((params, nlc_grid, scf.p_total()))
            }
            None => None,
        };
        let learned = match &opts.skala {
            Some(model) => {
                if grid.is_none() {
                    return Err(DftError::InvalidInput(
                        "response solver needs a grid for the learned functional".into(),
                    ));
                }
                Some((model.clone(), mol.clone()))
            }
            None => None,
        };
        let pcm = match &opts.pcm {
            Some(p) => Some(crate::pcm::PcmSolver::build(
                mol,
                mb,
                &mb.effective_charges(mol),
                p,
            )?),
            None => None,
        };
        let mgga_kernel = if functional.family == Family::MetaGga && !functional.parts.is_empty() {
            mgga_kernel::MetaGgaKernel::try_new(
                mb,
                grid.expect("grid checked at construction"),
                functional,
                reference_spin,
                opts.incore_mb.saturating_mul(1024 * 1024) / 8,
            )?
        } else {
            None
        };
        let frame = orbital_frame::OrbitalFrame::new(
            &scf.mo_coeff,
            &scf.fock,
            [scf.n_alpha, scf.n_beta],
            scf.unrestricted,
        );
        let (semilocal_fxc, semilocal_allocation) =
            if !functional.parts.is_empty() && functional.family != Family::MetaGga {
                let budget = (opts.incore_mb.saturating_mul(1024 * 1024) / 8).min(64 * 1024 * 1024);
                semilocal_cache(
                    grid.expect("grid checked at construction").len(),
                    functional.family != Family::Lda,
                    budget,
                )?
            } else {
                (Vec::new(), Default::default())
            };
        Ok(Self {
            mb,
            functional,
            grid,
            scf,
            jk,
            jk_lr,
            dims,
            frame,
            mgga_kernel,
            semilocal_fxc,
            _semilocal_allocation: semilocal_allocation,
            nlc,
            pcm,
            learned,
            max_iter: 100,
            tol: 1e-8,
        })
    }

    fn spins(&self) -> usize {
        if self.scf.unrestricted {
            2
        } else {
            1
        }
    }

    fn u_len(&self) -> usize {
        self.frame.dimension()
    }

    /// ΔP^σ = C_v U^σ C_oᵀ + C_o U^σᵀ C_vᵀ (symmetric real response).
    pub(crate) fn delta_density(&self, u: &[f64]) -> [Matrix; 2] {
        self.frame.density(u)
    }

    /// Project an AO matrix per spin into the OV block: (C_vᵀ M C_o)_ai.
    pub(crate) fn project_ov(&self, m: [&Matrix; 2], out: &mut [f64]) {
        self.frame.project(m, out)
    }

    /// Orbital-Hessian product `A·u`.
    /// Solve the CPKS equations for MANY right-hand sides at once.
    ///
    /// The per-vector path rebuilds J/K (and the XC kernel) on every PCG
    /// iteration of every vector, so the analytic Hessian pays 3N times the
    /// Fock work. Running one PCG over all vectors keeps each iteration's
    /// J/K to a single pass over the shared machinery while every vector
    /// keeps its own residual and step lengths — the iterates are
    /// identical to solving them separately, only the scheduling changes.
    pub fn solve_ov_block(&self, rhs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let nvec = rhs.len();
        if nvec == 0 {
            return Ok(Vec::new());
        }
        let len = self.u_len();
        let mut diag = vec![0.0; len];
        let mut off = 0;
        for spin in 0..self.spins() {
            let (no, nv) = (self.dims.n_occ[spin], self.dims.n_vir[spin]);
            let eps = &self.scf.mo_energy[spin];
            for a in 0..nv {
                for i in 0..no {
                    diag[off + a * no + i] = (eps[no + a] - eps[i]).max(1e-3);
                }
            }
            off += no * nv;
        }
        let mut u = vec![vec![0.0; len]; nvec];
        let mut r: Vec<Vec<f64>> = rhs
            .iter()
            .map(|b| b.iter().map(|&v| -v).collect())
            .collect();
        let mut z: Vec<Vec<f64>> = r
            .iter()
            .map(|rv| rv.iter().zip(&diag).map(|(x, d)| x / d).collect())
            .collect();
        let mut p = z.clone();
        let mut rz: Vec<f64> = r
            .iter()
            .zip(&z)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y).sum())
            .collect();
        let rmax = |v: &[f64]| v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let mut active: Vec<bool> = r.iter().map(|rv| rmax(rv) >= self.tol).collect();
        // Stagnation tracking. Conjugate gradients loses conjugacy to
        // rounding on ill-conditioned systems — exactly the CPKS regime
        // when the gap is small — and then burns iterations without
        // progress. Restarting the direction from the preconditioned
        // residual recovers a descent direction and costs one iteration.
        let mut best: Vec<f64> = r.iter().map(|rv| rmax(rv)).collect();
        let mut since_best = vec![0usize; nvec];
        let mut restarts = vec![0usize; nvec];
        const STALL_WINDOW: usize = 8;
        const MAX_RESTARTS: usize = 6;
        for _ in 0..self.max_iter {
            if !active.iter().any(|&a| a) {
                break;
            }
            // One Hessian-vector product per ACTIVE perturbation, and they
            // are independent -- so they run together. Sequentially this
            // re-walked the XC grid 3N times per CG iteration, which was
            // the largest remaining cost in the analytic Hessian.
            let idx: Vec<usize> = (0..nvec).filter(|&k| active[k]).collect();
            let aps: Vec<Result<Vec<f64>>> =
                idx.par_iter().map(|&k| self.hess_vec(&p[k])).collect();
            let mut ap_of: Vec<Option<Vec<f64>>> = (0..nvec).map(|_| None).collect();
            for (slot, res) in idx.iter().zip(aps) {
                ap_of[*slot] = Some(res?);
            }
            for k in 0..nvec {
                if !active[k] {
                    continue;
                }
                let ap = ap_of[k].take().expect("active perturbation has a product");
                let pap: f64 = p[k].iter().zip(&ap).map(|(a, b)| a * b).sum();
                // Non-positive curvature means the orbital Hessian is not
                // positive definite along this direction, i.e. the SCF
                // solution is a saddle rather than a minimum. Restarting
                // may get past a rounding artefact; if it does not, say so
                // — the response of an unstable reference is meaningless
                // and reporting it as a number is worse than refusing.
                if !pap.is_finite() || pap <= 0.0 {
                    if restarts[k] < MAX_RESTARTS {
                        restarts[k] += 1;
                        since_best[k] = 0;
                        for i in 0..len {
                            z[k][i] = r[k][i] / diag[i];
                            p[k][i] = z[k][i];
                        }
                        rz[k] = r[k].iter().zip(&z[k]).map(|(a, b)| a * b).sum();
                        continue;
                    }
                    return Err(DftError::InvalidInput(format!(
                        "CPKS: the orbital Hessian has non-positive curvature \
                         (p·Ap = {pap:.3e}) for perturbation {k} even after \
                         {MAX_RESTARTS} restarts. The SCF reference is a saddle \
                         point, not a minimum, so its linear response is not \
                         defined; re-converge the SCF (a broken-symmetry start \
                         via guess_mix often finds the stable solution)."
                    )));
                }
                let alpha = rz[k] / pap;
                for i in 0..len {
                    u[k][i] += alpha * p[k][i];
                    r[k][i] -= alpha * ap[i];
                }
                let cur = rmax(&r[k]);
                if cur < self.tol {
                    active[k] = false;
                    continue;
                }
                for i in 0..len {
                    z[k][i] = r[k][i] / diag[i];
                }
                // Restart when the residual has not beaten its best for a
                // while: keeping a stale direction is what stalls CG here.
                if cur < best[k] * 0.9 {
                    best[k] = cur;
                    since_best[k] = 0;
                } else {
                    since_best[k] += 1;
                }
                let restart = since_best[k] >= STALL_WINDOW && restarts[k] < MAX_RESTARTS;
                if restart {
                    restarts[k] += 1;
                    since_best[k] = 0;
                    best[k] = cur;
                }
                let rz_new: f64 = r[k].iter().zip(&z[k]).map(|(a, b)| a * b).sum();
                let beta = if restart { 0.0 } else { rz_new / rz[k] };
                rz[k] = rz_new;
                for i in 0..len {
                    p[k][i] = z[k][i] + beta * p[k][i];
                }
            }
        }
        // Silence here would be the worst outcome: an unconverged U looks
        // exactly like a converged one to every consumer downstream (the
        // analytic Hessian, the polarizability, SAPT induction), and the
        // error it carries is invisible in the result.
        if let Some(k) = (0..nvec).find(|&k| active[k]) {
            let worst = (0..nvec)
                .filter(|&j| active[j])
                .fold(0.0_f64, |m, j| m.max(rmax(&r[j])));
            let n_stuck = active.iter().filter(|&&a| a).count();
            return Err(DftError::ResponseNotConverged {
                iterations: self.max_iter,
                residual: worst,
            })
            .map_err(|e: DftError| {
                DftError::InvalidInput(format!(
                    "{e} ({n_stuck} of {nvec} perturbations unconverged; \
                     worst is #{k}). A small HOMO-LUMO gap is the usual cause."
                ))
            });
        }
        Ok(u)
    }

    /// Build the self-consistent Fock response G[dP] using the same cached
    /// full-range/LR J/K engines and density kernels as the orbital Hessian.
    /// Analytic derivative post-processing reuses this path so it cannot
    /// allocate duplicate O(n^4) ERI tensors or drift from the CPKS operator.
    pub(crate) fn fock_response(&self, dp: &[Matrix; 2]) -> Result<[Matrix; 2]> {
        let mut df = crate::scf::fock::coulomb_exchange_response(
            self.mb,
            &self.jk,
            self.jk_lr.as_ref(),
            self.functional.hybrid.cx,
            self.functional.hybrid.clr,
            dp,
        )?;
        // The reaction field responds even when there is no XC part at all
        // (HF in solvent), so the kernel branch keys on both.
        let fxc = if self.functional.parts.is_empty()
            && self.pcm.is_none()
            && self.nlc.is_none()
            && self.learned.is_none()
        {
            None
        } else {
            Some(self.fxc_contraction(dp)?)
        };
        if let Some(fxc) = &fxc {
            for spin in 0..2 {
                for (f, x) in df[spin].as_mut_slice().iter_mut().zip(fxc[spin].as_slice()) {
                    *f += x;
                }
            }
        }
        Ok(df)
    }

    /// Imaginary antisymmetric density at a real collinear reference. Local
    /// density/gradient/kinetic descriptors, Hartree, scalar nonlocal and
    /// electrostatic solvent terms have zero first variation; exchange keeps
    /// both AO endpoints. This is the time-odd partner of `fock_response`.
    pub(crate) fn imaginary_fock_response(&self, dp: &[Matrix; 2]) -> Result<[Matrix; 2]> {
        crate::scf::fock::coulomb_exchange_response(
            self.mb,
            &self.jk,
            self.jk_lr.as_ref(),
            self.functional.hybrid.cx,
            self.functional.hybrid.clr,
            dp,
        )
    }

    fn hess_vec(&self, u: &[f64]) -> Result<Vec<f64>> {
        self.hess_vec_in_frame(&self.frame, u)
    }

    fn hess_vec_in_frame(
        &self,
        frame: &orbital_frame::OrbitalFrame<'_>,
        u: &[f64],
    ) -> Result<Vec<f64>> {
        let dp = frame.density(u);
        let df = self.fock_response(&dp)?;
        let mut out = vec![0.0; frame.dimension()];
        frame.project([&df[0], &df[1]], &mut out);
        frame.add_fock_action(u, &mut out);
        Ok(out)
    }

    /// XC kernel contraction: ΔVxc^σ from Δρ (and Δ∇ρ for GGA), plus the
    /// exact nonlocal VV10 f_nlc when SC-VV10 is active.
    pub(crate) fn fxc_contraction(&self, dp: &[Matrix; 2]) -> Result<[Matrix; 2]> {
        let nao = self.mb.nao_sph;
        let mut out = if self.functional.parts.is_empty() {
            [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)]
        } else {
            self.fxc_contraction_semilocal(dp)?
        };
        if let Some((params, nlc_grid, p_tot)) = &self.nlc {
            let mut dp_tot = dp[0].clone();
            for (t, b) in dp_tot.as_mut_slice().iter_mut().zip(dp[1].as_slice()) {
                *t += *b;
            }
            let f = crate::nlc::vv10_fxc_contraction(self.mb, nlc_grid, p_tot, &dp_tot, *params)?;
            for o in out.iter_mut() {
                for (ov, fv) in o.as_mut_slice().iter_mut().zip(f.as_slice()) {
                    *ov += *fv;
                }
            }
        }
        if let Some((model, molecule)) = &self.learned {
            let grid = self.grid.expect("grid checked at construction");
            let f = crate::skala::skala_fxc_contraction(
                model,
                molecule,
                self.mb,
                grid,
                [&self.scf.density[0], &self.scf.density[1]],
                [&dp[0], &dp[1]],
            )?;
            for (o, fv) in out.iter_mut().zip(&f) {
                for (ov, v) in o.as_mut_slice().iter_mut().zip(fv.as_slice()) {
                    *ov += *v;
                }
            }
        }
        if let Some(solver) = &self.pcm {
            let mut dp_tot = dp[0].clone();
            for (t, b) in dp_tot.as_mut_slice().iter_mut().zip(dp[1].as_slice()) {
                *t += *b;
            }
            let f = solver.fock_response(&dp_tot)?;
            for o in out.iter_mut() {
                for (ov, fv) in o.as_mut_slice().iter_mut().zip(f.as_slice()) {
                    *ov += *fv;
                }
            }
        }
        Ok(out)
    }

    fn fxc_contraction_semilocal(&self, dp: &[Matrix; 2]) -> Result<[Matrix; 2]> {
        let grid = self.grid.expect("grid checked at construction");
        if self.functional.family == Family::MetaGga {
            // The tau-channel kernel lives in its own module.
            if let Some(kernel) = &self.mgga_kernel {
                return kernel.apply(dp);
            }
            return mgga_kernel::mgga_fxc_contraction(
                self.mb,
                grid,
                self.functional,
                [&self.scf.density[0], &self.scf.density[1]],
                dp,
            );
        }
        semilocal_density_contraction(
            self.mb,
            self.functional,
            grid,
            [&self.scf.density[0], &self.scf.density[1]],
            &self.semilocal_fxc,
            dp,
        )
    }

    /// Solve `A·u = −rhs_ov` for each right-hand side (AO perturbation
    /// matrices), returning the symmetric response densities ΔP per spin.
    pub fn solve(&self, rhs_ao: &[Matrix]) -> Result<Vec<[Matrix; 2]>> {
        let mut results = Vec::with_capacity(rhs_ao.len());
        for rhs_m in rhs_ao {
            let b = self.ao_perturbation(rhs_m);
            let u = self.solve_ov(&b)?;
            results.push(self.delta_density(&u));
        }
        Ok(results)
    }

    fn ao_perturbation(&self, rhs: &Matrix) -> Vec<f64> {
        let mut source = vec![0.0; self.u_len()];
        self.project_ov([rhs, rhs], &mut source);
        source
    }

    /// Maximum absolute defect of the original CPKS equation `A u + b`.
    /// Recompute the physical operator: a recursively updated CG residual
    /// alone cannot certify the returned orbital response.
    pub(crate) fn ov_equation_residual(&self, u: &[f64], b: &[f64]) -> Result<f64> {
        if u.len() != self.u_len() || b.len() != self.u_len() {
            return Err(DftError::InvalidInput(
                "CPKS residual requires one source and solution entry per orbital rotation".into(),
            ));
        }
        Ok(self
            .hess_vec(u)?
            .iter()
            .zip(b)
            .fold(0.0_f64, |worst, (au, b)| {
                let defect = (au + b).abs();
                if defect.is_finite() {
                    worst.max(defect)
                } else {
                    f64::INFINITY
                }
            }))
    }

    /// Return each induced density and the maximum fresh OV defect of
    /// A U + b in the original equation. The iterative CG residual is
    /// updated recursively and can differ from this check.
    pub(crate) fn solve_ao_block_report(
        &self,
        rhs_ao: &[Matrix],
    ) -> Result<Vec<([Matrix; 2], f64)>> {
        let sources: Vec<_> = rhs_ao.iter().map(|rhs| self.ao_perturbation(rhs)).collect();
        self.solve_ov_block(&sources)?
            .iter()
            .zip(sources)
            .map(|(u, b)| {
                let residual = self.ov_equation_residual(u, &b)?;
                Ok((self.delta_density(u), residual))
            })
            .collect()
    }

    /// Solve `A·u = −b` for a raw OV right-hand-side vector, returning the
    /// rotation amplitudes U_ai (nuclear-perturbation CPKS needs U itself,
    /// not just ΔP).
    pub fn solve_ov(&self, b_in: &[f64]) -> Result<Vec<f64>> {
        // One right-hand side is the one-element case of the block solver.
        // Keeping a second copy of the PCG here was how the two paths came
        // to differ: the block one used to return an unconverged answer
        // silently while this one raised. There is now one implementation,
        // so a fix to the iteration cannot reach only half the callers.
        Ok(self
            .solve_ov_block(std::slice::from_ref(&b_in.to_vec()))?
            .swap_remove(0))
    }

    /// Common response driver for A U = -b, with a fresh relative residual
    /// and iteration diagnostics. The operator uses the physical Fock blocks;
    /// canonical gaps and their floor enter only its Jacobi preconditioner.
    pub fn solve_ov_report(
        &self,
        b: &[f64],
        config: &ResponseSolverConfig,
    ) -> Result<krylov::LinearSolution> {
        self.solve_ov_report_with_structure(b, config, krylov::OperatorStructure::PositiveDefinite)
    }

    /// Select the solver from a stated property of the physical operator.
    /// Use Hermitian when positivity has not been established.
    pub fn solve_ov_report_with_structure(
        &self,
        b: &[f64],
        config: &ResponseSolverConfig,
        structure: krylov::OperatorStructure,
    ) -> Result<krylov::LinearSolution> {
        self.solve_in_frame(&self.frame, b, config, structure)
    }

    fn solve_in_frame(
        &self,
        frame: &orbital_frame::OrbitalFrame<'_>,
        b: &[f64],
        config: &ResponseSolverConfig,
        structure: krylov::OperatorStructure,
    ) -> Result<krylov::LinearSolution> {
        struct OrbitalOperator<'a, 'b, 'c>(
            &'a ResponseSolver<'b>,
            &'a orbital_frame::OrbitalFrame<'c>,
            krylov::OperatorStructure,
        );
        impl krylov::LinearOperator for OrbitalOperator<'_, '_, '_> {
            fn dimension(&self) -> usize {
                self.1.dimension()
            }
            fn structure(&self) -> krylov::OperatorStructure {
                self.2
            }
            fn apply(&self, x: &[complex::Complex]) -> Result<Vec<complex::Complex>> {
                let re: Vec<_> = x.iter().map(|x| x.re).collect();
                let im: Vec<_> = x.iter().map(|x| x.im).collect();
                let r = self.0.hess_vec_in_frame(self.1, &re)?;
                let i = if im.iter().any(|x| *x != 0.0) {
                    self.0.hess_vec_in_frame(self.1, &im)?
                } else {
                    vec![0.0; im.len()]
                };
                Ok(r.into_iter()
                    .zip(i)
                    .map(|(re, im)| complex::Complex::new(re, im))
                    .collect())
            }
        }
        let diagonal: Vec<_> = (0..self.spins())
            .flat_map(|s| {
                let no = self.dims.n_occ[s];
                (0..self.dims.n_vir[s]).flat_map(move |a| {
                    (0..no).map(move |i| {
                        (self.scf.mo_energy[s][no + a] - self.scf.mo_energy[s][i]).abs()
                    })
                })
            })
            .map(|v| v.max(config.preconditioner_floor))
            .collect();
        let rhs: Vec<_> = b.iter().map(|&b| complex::Complex::from(-b)).collect();
        krylov::solve(
            &OrbitalOperator(self, frame, structure),
            &rhs,
            Some(&diagonal),
            config,
        )
    }

    /// Polish only the representation of a converged integer determinant.
    /// The initial density kernel is reused as a near-reference Newton
    /// operator; every accepted trial is judged with the supplied actual
    /// energy/Fock evaluator. Original reference diagnostics remain external.
    fn polish_reference(
        &self,
        config: &ResponseSolverConfig,
        mut evaluate: impl FnMut(&[Matrix; 2]) -> Result<(f64, [Matrix; 2])>,
    ) -> Result<ReferencePolish> {
        let initial = &self.scf.mo_coeff;
        let occupied = [self.scf.n_alpha, self.scf.n_beta];
        let mut rotations = std::array::from_fn(|s| Matrix::identity(initial[s].cols));
        let mut coefficients = self.scf.mo_coeff.clone();
        let mut fock = self.scf.fock.clone();
        let mut gradient = vec![0.0; self.u_len()];
        self.frame.project([&fock[0], &fock[1]], &mut gradient);
        let largest = |v: &[f64]| v.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let initial_residual = largest(&gradient);
        let mut final_residual = initial_residual;
        let mut iterations = 0;
        for iteration in 0..4 {
            if final_residual == 0.0 {
                break;
            }
            let frame = orbital_frame::OrbitalFrame::new(
                &coefficients,
                &fock,
                occupied,
                self.scf.unrestricted,
            );
            let source_norm = gradient.iter().fold(0.0_f64, |a, &b| a.hypot(b));
            let solution = self.solve_in_frame(
                &frame,
                &gradient,
                &config.for_absolute_residual(source_norm)?,
                krylov::OperatorStructure::Hermitian,
            )?;
            let amplitudes: Vec<_> = solution.amplitudes.iter().map(|z| z.re).collect();
            let mut accepted = None;
            for scale in [1.0, 0.5, 0.25] {
                let step: Vec<_> = amplitudes.iter().map(|v| v * scale).collect();
                let rotation = frame.rotations(&step);
                let candidate_rotation: [Matrix; 2] =
                    std::array::from_fn(|s| rotations[s].matmul_accurate(&rotation[s], false));
                let candidate: [Matrix; 2] = std::array::from_fn(|s| {
                    initial[s].matmul_accurate(&candidate_rotation[s], false)
                });
                let density = orbital_frame::OrbitalFrame::new(
                    &candidate,
                    &fock,
                    occupied,
                    self.scf.unrestricted,
                )
                .occupied_density();
                let (energy, candidate_fock) = evaluate(&density)?;
                let trial = orbital_frame::OrbitalFrame::new(
                    &candidate,
                    &candidate_fock,
                    occupied,
                    self.scf.unrestricted,
                );
                let mut trial_gradient = vec![0.0; self.u_len()];
                trial.project(
                    [&candidate_fock[0], &candidate_fock[1]],
                    &mut trial_gradient,
                );
                let residual = largest(&trial_gradient);
                if residual < final_residual {
                    accepted = Some((
                        candidate_rotation,
                        candidate,
                        candidate_fock,
                        trial_gradient,
                        residual,
                        energy,
                    ));
                    break;
                }
            }
            let Some((r, c, f, g, residual, energy)) = accepted else {
                break;
            };
            rotations = r;
            coefficients = c;
            fock = f;
            gradient = g;
            final_residual = residual;
            iterations = iteration + 1;
            crate::logger::log(&format!("[response] reference refinement {iterations}: E={energy:.16e} orbital residual={residual:.6e}"));
        }
        Ok(ReferencePolish {
            rotations,
            initial_residual,
            final_residual,
            iterations,
        })
    }

    /// Occupied/virtual counts per spin (Hessian assembly needs them).
    pub fn ov_dims(&self) -> ([usize; 2], [usize; 2]) {
        (self.dims.n_occ, self.dims.n_vir)
    }
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod semilocal_cache_regression {
    use super::*;
    use crate::basis::data;
    use crate::math::Vec3;
    use crate::scf::{run_scf, EriMode};
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    #[test]
    fn allocation_bound_includes_headers_and_handles_overflow() {
        assert_eq!(semilocal_cache_slots(1024, true, 0), 0);
        assert_eq!(semilocal_cache_slots(usize::MAX, true, usize::MAX), 0);
        for gga in [false, true] {
            let bytes =
                257 * if gga { 144 } else { 24 } + 2 * std::mem::size_of::<OnceLock<XcOut>>();
            assert_eq!(semilocal_cache_slots(257, gga, bytes), 2);
            assert_eq!(semilocal_cache_slots(257, gga, bytes - 1), 0);
        }
    }

    #[test]
    fn shared_budget_holds_cached_payload_and_falls_back_without_room() {
        use crate::resources::{current_usage, remaining_bytes, reserve, ResourceLimits, MIB};
        ResourceLimits::new(Some(1), Some(1))
            .unwrap()
            .run(|| {
                let bytes = semilocal_cache_bytes(257, true).unwrap();
                let (slots, lease) = semilocal_cache(257, true, MIB)?;
                assert_eq!(slots.len(), 2);
                assert_eq!(current_usage().unwrap().reserved_bytes, bytes);
                let required = reserve(MIB - bytes - 1024, "cache fallback fixture")?;
                let (fallback, unused) = semilocal_cache(257, true, MIB)?;
                assert!(fallback.is_empty());
                assert_eq!(unused.bytes(), 0);
                assert_eq!(remaining_bytes(), Some(1024));
                drop(required);
                drop(slots);
                drop(lease);
                assert_eq!(remaining_bytes(), Some(MIB));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn every_symmetric_ao_source_and_spin_matches_streamed_kernel() {
        for name in ["hf", "svwn5", "pbe", "pbe0"] {
            for open in [false, true] {
                for shifted in [false, true] {
                    let mut mol = Molecule::new(vec![
                        Atom {
                            z: 1,
                            position: Vec3::new(0.13, -0.21, 0.19),
                        },
                        Atom {
                            z: 1,
                            position: Vec3::new(1.57, 0.16, -0.08),
                        },
                        Atom {
                            z: 1,
                            position: Vec3::new(-0.19, 1.93, 0.37),
                        },
                        Atom {
                            z: 1,
                            position: Vec3::new(0.42, 0.29, 2.28),
                        },
                    ]);
                    if shifted {
                        mol.atoms[1].position.x += 0.027;
                    }
                    if open {
                        mol.charge = 1;
                        mol.multiplicity = 2;
                    }
                    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
                    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
                    // A fixed discrete grid spans two batches including a
                    // partial last batch. This tests reuse, not quadrature accuracy.
                    let count = if shifted { 259 } else { 258 };
                    let points: Vec<_> = (0..count)
                        .map(|i| {
                            let q = (i % 5) as f64;
                            Vec3::new(0.3 + 0.17 * q, 0.2 - 0.11 * q, 0.1 + 0.23 * q)
                        })
                        .collect();
                    let grid = MolecularGrid {
                        points,
                        weights: vec![4.0 / count as f64; count],
                        raw_weights: vec![4.0 / count as f64; count],
                        atom_of: vec![0; count],
                    };
                    let grid = (!functional.parts.is_empty()).then_some(&grid);
                    let opts = ScfOptions {
                        max_iter: 400,
                        e_tol: 1e-13,
                        p_tol: 1e-12,
                        conv_tol_grad: 1e-11,
                        incore_mb: 64,
                        eri_mode: EriMode::InCore { limit_mb: 128 },
                        ..Default::default()
                    };
                    let scf = run_scf(&mol, &basis, &opts, &functional, grid).unwrap();
                    assert!(scf.converged, "{name} open={open} shifted={shifted}");
                    let cached =
                        ResponseSolver::new(&mol, &basis, &functional, grid, &scf, &opts).unwrap();
                    let mut streamed =
                        ResponseSolver::new(&mol, &basis, &functional, grid, &scf, &opts).unwrap();
                    streamed.semilocal_fxc.clear();
                    for source_spin in 0..2 {
                        for i in 0..basis.nao_sph {
                            for j in 0..=i {
                                let mut density = [
                                    Matrix::zeros(basis.nao_sph, basis.nao_sph),
                                    Matrix::zeros(basis.nao_sph, basis.nao_sph),
                                ];
                                density[source_spin][(i, j)] = 0.37;
                                density[source_spin][(j, i)] = 0.37;
                                let reference = streamed.fxc_contraction(&density).unwrap();
                                // First and subsequent batch calls must have
                                // identical full AO output for both spin channels.
                                for _ in 0..2 {
                                    let actual = cached.fxc_contraction(&density).unwrap();
                                    for spin in 0..2 {
                                        for (a, b) in actual[spin]
                                            .as_slice()
                                            .iter()
                                            .zip(reference[spin].as_slice())
                                        {
                                            assert_eq!(a, b, "{name} open={open} shifted={shifted} source={source_spin},{i},{j}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if name != "hf" {
                        assert_eq!(cached.semilocal_fxc.len(), 2);
                        assert!(cached.semilocal_fxc.iter().all(|slot| slot.get().is_some()));
                        let mut no_budget = opts.clone();
                        no_budget.incore_mb = 0;
                        let fallback =
                            ResponseSolver::new(&mol, &basis, &functional, grid, &scf, &no_budget)
                                .unwrap();
                        assert!(fallback.semilocal_fxc.is_empty());
                    }
                }
            }
        }
    }
}
