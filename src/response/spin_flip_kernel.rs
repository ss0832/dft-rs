// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Response of the Kohn-Sham operator to a spin-flip transition density.
//!
//! A transverse (spin-flip) density change does not alter the densities of
//! a collinear functional to first order, so its only coupling there is the
//! exact exchange, `-c_x K[T]`. Rotating the local spin axis of a spin-
//! polarized density by a transverse change `d` changes every spin
//! difference `s_q = q_alpha - q_beta` of a variable `q` (density, density
//! gradient, kinetic energy density) at second order,
//! `s_q -> s_q - s_q d^2 / (2 m^2) + d d_q / m`, with the spin density `m`
//! and the transverse change `d_q` of that variable. With the spin
//! differences `Dv`, `DG` and `Dv_tau` of `de/drho`, `de/dgrad rho` and
//! `de/dtau`, this gives the non-collinear kernel
//!
//! ```text
//! K_nc[T]_mn = sum_p w_p { a rho_T phi_m phi_n
//!                        + b . [rho_T grad(phi_m phi_n) + phi_m phi_n grad rho_T]
//!                        + c [rho_T grad phi_m . grad phi_n / 2 + phi_m phi_n tau_T] },
//! a = Dv / m - (DG . grad m + Dv_tau (tau_alpha - tau_beta)) / m^2,
//! b = DG / m,   c = Dv_tau / m,
//! ```
//!
//! with the transition density `rho_T`, its gradient and the transition
//! kinetic energy density `tau_T` on the grid. For a local functional only
//! `a = (v_alpha - v_beta) / m` remains. `b` holds a term proportional to
//! `grad m / m`, which does not stay finite where the spin density vanishes:
//! at points with `|m| <= 1e-6 (rho_alpha + rho_beta)` the kernel is its
//! local limit `a = (f_aa - 2 f_ab + f_bb) / 2` from the second derivatives
//! of the energy density, with `b = c = 0`.

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::eval::eval_aos_full;
use crate::grid::MolecularGrid;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::scf::fock::JkEngine;
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};

/// Grid points handled together.
const BATCH: usize = 256;

/// The non-collinear kernel of a reference density on its grid.
pub struct NoncollinearKernel<'a> {
    mb: &'a MolBasis,
    grid: &'a MolecularGrid,
    /// Whether the gradient and kinetic terms are present.
    gradients: bool,
    /// Quadrature weight times `a` at every point.
    weighted_local: Vec<f64>,
    /// Quadrature weight times `b`.
    weighted_gradient: Vec<[f64; 3]>,
    /// Quadrature weight times `c`.
    weighted_kinetic: Vec<f64>,
}

impl<'a> NoncollinearKernel<'a> {
    pub fn new(
        functional: &ResolvedFunctional,
        mb: &'a MolBasis,
        grid: &'a MolecularGrid,
        density: [&Matrix; 2],
    ) -> Result<Self> {
        let gga = functional.family != Family::Lda;
        let meta = functional.family == Family::MetaGga;
        let nao = mb.nao_sph;
        let npoints = grid.points.len();
        let mut weighted_local = Vec::with_capacity(npoints);
        let mut weighted_gradient = Vec::with_capacity(npoints);
        let mut weighted_kinetic = Vec::with_capacity(npoints);
        let mut start = 0;
        while start < npoints {
            let end = (start + BATCH).min(npoints);
            let n = end - start;
            let batch = eval_aos_full(mb, &grid.points[start..end], gga, false);
            let mut rho = [vec![0.0_f64; n], vec![0.0_f64; n]];
            let mut slope = [vec![[0.0_f64; 3]; n], vec![[0.0_f64; 3]; n]];
            let mut tau = [vec![0.0_f64; n], vec![0.0_f64; n]];
            for spin in 0..2 {
                let x = batch.phi.matmul(density[spin]);
                for i in 0..n {
                    let mut value = 0.0;
                    for mu in 0..nao {
                        value += x[(i, mu)] * batch.phi[(i, mu)];
                    }
                    rho[spin][i] = value.max(0.0);
                }
                if let Some(g) = &batch.grad {
                    for axis in 0..3 {
                        let y = g[axis].matmul(density[spin]);
                        for i in 0..n {
                            let mut first = 0.0;
                            let mut kinetic = 0.0;
                            for mu in 0..nao {
                                first += x[(i, mu)] * g[axis][(i, mu)];
                                kinetic += y[(i, mu)] * g[axis][(i, mu)];
                            }
                            slope[spin][i][axis] = 2.0 * first;
                            tau[spin][i] += 0.5 * kinetic;
                        }
                    }
                }
            }
            let sigma: [Vec<f64>; 3] = std::array::from_fn(|pair| {
                let (a, b) = [(0, 0), (0, 1), (1, 1)][pair];
                (0..n)
                    .map(|i| (0..3).map(|k| slope[a][i][k] * slope[b][i][k]).sum())
                    .collect()
            });
            let vars = GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: gga.then_some([&sigma[0], &sigma[1], &sigma[2]]),
                tau: meta.then_some([&tau[0], &tau[1]]),
            };
            let mut out = XcOut::default();
            functional.eval(&vars, 2, &mut out)?;
            for i in 0..n {
                let weight = grid.weights[start + i];
                let total = rho[0][i] + rho[1][i];
                let m = rho[0][i] - rho[1][i];
                if total <= 1e-12 {
                    weighted_local.push(0.0);
                    weighted_gradient.push([0.0; 3]);
                    weighted_kinetic.push(0.0);
                } else if m.abs() <= 1e-6 * total {
                    let limit =
                        0.5 * (out.v2rho2[0][i] - 2.0 * out.v2rho2[1][i] + out.v2rho2[2][i]);
                    weighted_local.push(weight * limit);
                    weighted_gradient.push([0.0; 3]);
                    weighted_kinetic.push(0.0);
                } else {
                    let dv = out.vrho[0][i] - out.vrho[1][i];
                    // DG = G_alpha - G_beta with
                    // G_s = 2 v_ss grad rho_s + v_ab grad rho_other.
                    let mut dg = [0.0_f64; 3];
                    let mut projected = 0.0;
                    if gga {
                        let own_a = 2.0 * out.vsigma[0][i] - out.vsigma[1][i];
                        let own_b = 2.0 * out.vsigma[2][i] - out.vsigma[1][i];
                        for k in 0..3 {
                            dg[k] = own_a * slope[0][i][k] - own_b * slope[1][i][k];
                            projected += dg[k] * (slope[0][i][k] - slope[1][i][k]);
                        }
                    }
                    let (dv_tau, d_tau) = if meta {
                        (out.vtau[0][i] - out.vtau[1][i], tau[0][i] - tau[1][i])
                    } else {
                        (0.0, 0.0)
                    };
                    weighted_local.push(weight * (dv / m - (projected + dv_tau * d_tau) / (m * m)));
                    weighted_gradient.push([
                        weight * dg[0] / m,
                        weight * dg[1] / m,
                        weight * dg[2] / m,
                    ]);
                    weighted_kinetic.push(weight * dv_tau / m);
                }
            }
            start = end;
        }
        Ok(Self {
            mb,
            grid,
            gradients: gga,
            weighted_local,
            weighted_gradient,
            weighted_kinetic,
        })
    }

    /// `K_nc[T]` for the transition density matrix `t`.
    pub fn apply(&self, t: &Matrix) -> Matrix {
        let nao = self.mb.nao_sph;
        let npoints = self.grid.points.len();
        let mut out = Matrix::zeros(nao, nao);
        let mut start = 0;
        while start < npoints {
            let end = (start + BATCH).min(npoints);
            let n = end - start;
            let batch = eval_aos_full(
                self.mb,
                &self.grid.points[start..end],
                self.gradients,
                false,
            );
            let x = batch.phi.matmul(t);
            // `left` pairs with the orbital values, `side[k]` with their
            // gradients along `k`.
            let mut left = Matrix::zeros(n, nao);
            let mut side: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, nao));
            let y: Option<[Matrix; 3]> = batch
                .grad
                .as_ref()
                .map(|g| std::array::from_fn(|k| g[k].matmul(t)));
            for i in 0..n {
                let point = start + i;
                let mut transition = 0.0;
                for mu in 0..nao {
                    transition += x[(i, mu)] * batch.phi[(i, mu)];
                }
                let mut factor = self.weighted_local[point] * transition;
                if let (Some(g), Some(y)) = (&batch.grad, &y) {
                    let b = self.weighted_gradient[point];
                    let c = self.weighted_kinetic[point];
                    let mut kinetic = 0.0;
                    for k in 0..3 {
                        let mut slope = 0.0;
                        for mu in 0..nao {
                            slope +=
                                y[k][(i, mu)] * batch.phi[(i, mu)] + x[(i, mu)] * g[k][(i, mu)];
                            kinetic += y[k][(i, mu)] * g[k][(i, mu)];
                        }
                        factor += b[k] * slope;
                    }
                    factor += c * 0.5 * kinetic;
                    for mu in 0..nao {
                        let mut along = 0.0;
                        for k in 0..3 {
                            along += b[k] * g[k][(i, mu)];
                            side[k][(i, mu)] =
                                transition * (b[k] * batch.phi[(i, mu)] + 0.5 * c * g[k][(i, mu)]);
                        }
                        left[(i, mu)] = factor * batch.phi[(i, mu)] + transition * along;
                    }
                } else {
                    for mu in 0..nao {
                        left[(i, mu)] = factor * batch.phi[(i, mu)];
                    }
                }
            }
            let mut part = batch.phi.transpose_matmul(&left);
            if let Some(g) = &batch.grad {
                for k in 0..3 {
                    let extra = g[k].transpose_matmul(&side[k]);
                    for (target, value) in part.as_mut_slice().iter_mut().zip(extra.as_slice()) {
                        *target += *value;
                    }
                }
            }
            for (target, value) in out.as_mut_slice().iter_mut().zip(part.as_slice()) {
                *target += *value;
            }
            start = end;
        }
        out
    }
}

/// Exact exchange and, when requested, the non-collinear kernel acting on a
/// spin-flip transition density.
pub struct SpinFlipOperator<'a> {
    mb: &'a MolBasis,
    jk: JkEngine,
    jk_lr: Option<JkEngine>,
    cx: f64,
    clr: f64,
    noncollinear: Option<NoncollinearKernel<'a>>,
}

impl<'a> SpinFlipOperator<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mol: &Molecule,
        mb: &'a MolBasis,
        functional: &ResolvedFunctional,
        opts: &ScfOptions,
        grid: Option<&'a MolecularGrid>,
        density: [&Matrix; 2],
        noncollinear: bool,
    ) -> Result<Self> {
        let cx = functional.hybrid.cx;
        let clr = functional.hybrid.clr;
        let jk = JkEngine::from_options(mol, mb, opts)?;
        let jk_lr = if clr != 0.0 {
            Some(JkEngine::new(
                mb,
                opts.eri_mode,
                CoulombKernel::Erf {
                    omega: functional.hybrid.omega,
                },
            )?)
        } else {
            None
        };
        let noncollinear = if noncollinear && !functional.parts.is_empty() {
            let grid = grid.ok_or_else(|| {
                DftError::InvalidInput(
                    "the non-collinear spin-flip kernel needs the molecular grid".into(),
                )
            })?;
            Some(NoncollinearKernel::new(functional, mb, grid, density)?)
        } else {
            None
        };
        Ok(Self {
            mb,
            jk,
            jk_lr,
            cx,
            clr,
            noncollinear,
        })
    }

    /// True when some term couples the spin-flip configurations.
    pub fn couples(&self) -> bool {
        self.cx != 0.0 || self.clr != 0.0 || self.noncollinear.is_some()
    }

    /// Response to the transition density `t` (row: the spin that gains the
    /// electron, column: the spin that loses it).
    pub fn apply(&self, t: &Matrix) -> Result<Matrix> {
        let mut out = Matrix::zeros(t.rows, t.cols);
        if self.cx != 0.0 {
            let k = self.jk.build_k_only(self.mb, [t, t])?;
            for (target, value) in out.as_mut_slice().iter_mut().zip(k[0].as_slice()) {
                *target -= self.cx * value;
            }
        }
        if let Some(engine) = &self.jk_lr {
            let k = engine.build_k_only(self.mb, [t, t])?;
            for (target, value) in out.as_mut_slice().iter_mut().zip(k[0].as_slice()) {
                *target -= self.clr * value;
            }
        }
        if let Some(kernel) = &self.noncollinear {
            let local = kernel.apply(t);
            for (target, value) in out.as_mut_slice().iter_mut().zip(local.as_slice()) {
                *target += *value;
            }
        }
        Ok(out)
    }
}
