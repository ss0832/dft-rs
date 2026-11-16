// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Meta-GGA response kernel: the fxc contraction with the tau channel
//! (Delta v_rho / Delta v_sigma / Delta v_tau from Delta rho / sigma /
//! tau), kept in its own module so future meta-GGA machinery (analytic
//! Hessian tau terms, meta TDDFT extensions) grows here instead of
//! branching through the GGA paths.
//!
//! Kernel blocks (libxc-style packing, from `MggaDriver` order 2):
//! v2rhotau (ra ta, ra tb, rb ta, rb tb), v2sigmatau (sigma_p x tau_s,
//! p-major), v2tau2 (aa, ab, bb). The tau-channel matrix contribution is
//! `1/2 sum_u g_u^T diag(w dvtau) g_u` - the response image of the
//! ground-state KS tau term.

use crate::basis::MolBasis;
use crate::error::Result;
use crate::grid::eval::eval_aos_full;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::xc::{GridVars, ResolvedFunctional, XcOut};

const BATCH: usize = 256;

/// Local derivatives tied to one immutable basis, grid and spin density.
/// AO contractions retain the same arithmetic as the uncached application.
pub(crate) struct MetaGgaKernel<'a> {
    basis: &'a MolBasis,
    grid: &'a MolecularGrid,
    functional: &'a ResolvedFunctional,
    density: [&'a Matrix; 2],
    outputs: Vec<XcOut>,
    _allocation: crate::resources::MemoryReservation,
}

impl<'a> MetaGgaKernel<'a> {
    pub fn try_new(
        basis: &'a MolBasis,
        grid: &'a MolecularGrid,
        functional: &'a ResolvedFunctional,
        density: [&'a Matrix; 2],
        budget_bytes: usize,
    ) -> Result<Option<Self>> {
        let Some(batches) = grid.len().checked_add(BATCH - 1).map(|n| n / BATCH) else {
            return Ok(None);
        };
        // An order-two meta-GGA output owns 36 arrays. Budget a full last
        // batch plus the headers, without relying on allocator rounding.
        let required = batches
            .checked_mul(BATCH * 36 * std::mem::size_of::<f64>() + std::mem::size_of::<XcOut>());
        if budget_bytes == 0 || required.is_none_or(|bytes| bytes > budget_bytes) {
            return Ok(None);
        }
        let Some(allocation) = crate::resources::reserve_cache(
            required.expect("cache size checked"),
            "meta-GGA XC derivative cache",
        )?
        else {
            return Ok(None);
        };
        let n = basis.nao_sph;
        let zero = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
        let mut outputs = Vec::with_capacity(batches);
        contraction(
            basis,
            grid,
            functional,
            density,
            &zero,
            None,
            Some(&mut outputs),
        )?;
        Ok(Some(Self {
            basis,
            grid,
            functional,
            density,
            outputs,
            _allocation: allocation,
        }))
    }

    pub fn apply(&self, direction: &[Matrix; 2]) -> Result<[Matrix; 2]> {
        contraction(
            self.basis,
            self.grid,
            self.functional,
            self.density,
            direction,
            Some(&self.outputs),
            None,
        )
    }
}

/// Meta-GGA XC kernel contraction: `DeltaVxc^s` matrices from per-spin
/// symmetric density changes `dp` at the ground-state density `p`.
pub(crate) fn mgga_fxc_contraction(
    mb: &MolBasis,
    grid: &MolecularGrid,
    functional: &ResolvedFunctional,
    p: [&Matrix; 2],
    dp: &[Matrix; 2],
) -> Result<[Matrix; 2]> {
    contraction(mb, grid, functional, p, dp, None, None)
}

fn contraction(
    mb: &MolBasis,
    grid: &MolecularGrid,
    functional: &ResolvedFunctional,
    p: [&Matrix; 2],
    dp: &[Matrix; 2],
    cached: Option<&[XcOut]>,
    mut retain: Option<&mut Vec<XcOut>>,
) -> Result<[Matrix; 2]> {
    let nao = mb.nao_sph;
    let n = grid.len();
    let mut vxc = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    let mut start = 0;
    while start < n {
        let end = (start + BATCH).min(n);
        let pts = &grid.points[start..end];
        let wts = &grid.weights[start..end];
        let npts = pts.len();
        let batch = eval_aos_full(mb, pts, true, false);
        let g = batch.grad.as_ref().expect("gradients");
        // Ground-state and perturbed variables.
        let mut rho = [vec![0.0; npts], vec![0.0; npts]];
        let mut tau = [vec![0.0; npts], vec![0.0; npts]];
        let mut drho = [vec![0.0; npts], vec![0.0; npts]];
        let mut dtau = [vec![0.0; npts], vec![0.0; npts]];
        let mut gr: [[Vec<f64>; 3]; 2] = Default::default();
        let mut dgr: [[Vec<f64>; 3]; 2] = Default::default();
        let mut x_gs = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        for spin in 0..2 {
            x_gs[spin] = batch.phi.matmul_seq(p[spin]);
            let x_dp = batch.phi.matmul_seq(&dp[spin]);
            let yg = [
                g[0].matmul_seq(p[spin]),
                g[1].matmul_seq(p[spin]),
                g[2].matmul_seq(p[spin]),
            ];
            let yd = [
                g[0].matmul_seq(&dp[spin]),
                g[1].matmul_seq(&dp[spin]),
                g[2].matmul_seq(&dp[spin]),
            ];
            for i in 0..npts {
                let mut r_acc = 0.0;
                let mut dr_acc = 0.0;
                for mu in 0..nao {
                    r_acc += x_gs[spin][(i, mu)] * batch.phi[(i, mu)];
                    dr_acc += x_dp[(i, mu)] * batch.phi[(i, mu)];
                }
                rho[spin][i] = r_acc.max(0.0);
                drho[spin][i] = dr_acc;
                let mut t_acc = 0.0;
                let mut dt_acc = 0.0;
                for u in 0..3 {
                    for mu in 0..nao {
                        t_acc += yg[u][(i, mu)] * g[u][(i, mu)];
                        dt_acc += yd[u][(i, mu)] * g[u][(i, mu)];
                    }
                }
                tau[spin][i] = (0.5 * t_acc).max(0.0);
                dtau[spin][i] = 0.5 * dt_acc;
            }
            for u in 0..3 {
                let mut gv = vec![0.0; npts];
                let mut dgv = vec![0.0; npts];
                for i in 0..npts {
                    let mut a = 0.0;
                    let mut da = 0.0;
                    for mu in 0..nao {
                        a += x_gs[spin][(i, mu)] * g[u][(i, mu)];
                        da += x_dp[(i, mu)] * g[u][(i, mu)];
                    }
                    gv[i] = 2.0 * a;
                    dgv[i] = 2.0 * da;
                }
                gr[spin][u] = gv;
                dgr[spin][u] = dgv;
            }
        }
        let mut sig = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
        let mut dsig = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
        for i in 0..npts {
            let ga = [gr[0][0][i], gr[0][1][i], gr[0][2][i]];
            let gb = [gr[1][0][i], gr[1][1][i], gr[1][2][i]];
            let da = [dgr[0][0][i], dgr[0][1][i], dgr[0][2][i]];
            let db = [dgr[1][0][i], dgr[1][1][i], dgr[1][2][i]];
            sig[0][i] = ga.iter().map(|v| v * v).sum();
            sig[1][i] = ga.iter().zip(&gb).map(|(x, y)| x * y).sum();
            sig[2][i] = gb.iter().map(|v| v * v).sum();
            dsig[0][i] = 2.0 * ga.iter().zip(&da).map(|(x, y)| x * y).sum::<f64>();
            dsig[1][i] = ga.iter().zip(&db).map(|(x, y)| x * y).sum::<f64>()
                + gb.iter().zip(&da).map(|(x, y)| x * y).sum::<f64>();
            dsig[2][i] = 2.0 * gb.iter().zip(&db).map(|(x, y)| x * y).sum::<f64>();
        }
        let vars = GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: Some([&sig[0], &sig[1], &sig[2]]),
            tau: Some([&tau[0], &tau[1]]),
        };
        let mut evaluated_output = XcOut::default();
        let out = if let Some(outputs) = cached {
            &outputs[start / BATCH]
        } else {
            #[cfg(test)]
            crate::xc::branch_freeze::set_batch(Some(start));
            let evaluated = functional.eval(&vars, 2, &mut evaluated_output);
            #[cfg(test)]
            crate::xc::branch_freeze::set_batch(None);
            evaluated?;
            if let Some(outputs) = &mut retain {
                outputs.push(evaluated_output);
                start = end;
                continue;
            }
            &evaluated_output
        };
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
        for spin in 0..2 {
            let rowbase = if spin == 0 { 0 } else { 3 };
            let frr: [usize; 2] = if spin == 0 { [0, 1] } else { [1, 2] };
            let (own, cross) = if spin == 0 { (0usize, 1usize) } else { (2, 1) };
            let other = 1 - spin;
            // Per-point row buffers: value channel, gradient channel, tau.
            let mut dv = Matrix::zeros(npts, nao);
            let mut dvg = Matrix::zeros(npts, nao);
            let mut dvt = vec![0.0; npts];
            for i in 0..npts {
                let w = wts[i];
                // Delta v_rho.
                let mut dvr =
                    out.v2rho2[frr[0]][i] * drho[0][i] + out.v2rho2[frr[1]][i] * drho[1][i];
                for pch in 0..3 {
                    dvr += out.v2rhosigma[rowbase + pch][i] * dsig[pch][i];
                }
                // v2rhotau rows: (ra ta, ra tb, rb ta, rb tb).
                let rt = if spin == 0 { [0, 1] } else { [2, 3] };
                dvr += out.v2rhotau[rt[0]][i] * dtau[0][i] + out.v2rhotau[rt[1]][i] * dtau[1][i];
                let f = w * dvr;
                for mu in 0..nao {
                    dv[(i, mu)] += f * batch.phi[(i, mu)];
                }
                // Delta v_sigma for the three channels.
                let mut dvs = [0.0; 3];
                for pch in 0..3 {
                    dvs[pch] = out.v2rhosigma[pch][i] * drho[0][i]
                        + out.v2rhosigma[3 + pch][i] * drho[1][i];
                    for qch in 0..3 {
                        dvs[pch] += out.v2sigma2[ss(pch, qch)][i] * dsig[qch][i];
                    }
                    dvs[pch] += out.v2sigmatau[2 * pch][i] * dtau[0][i]
                        + out.v2sigmatau[2 * pch + 1][i] * dtau[1][i];
                }
                let mut bvec = [0.0; 3];
                for ax in 0..3 {
                    bvec[ax] = w
                        * (2.0 * dvs[own] * gr[spin][ax][i]
                            + 2.0 * out.vsigma[own][i] * dgr[spin][ax][i]
                            + dvs[cross] * gr[other][ax][i]
                            + out.vsigma[cross][i] * dgr[other][ax][i]);
                }
                for mu in 0..nao {
                    dvg[(i, mu)] +=
                        bvec[0] * g[0][(i, mu)] + bvec[1] * g[1][(i, mu)] + bvec[2] * g[2][(i, mu)];
                }
                // Delta v_tau.
                let tt = if spin == 0 { [0, 1] } else { [1, 2] };
                let mut dvtau =
                    out.v2tau2[tt[0]][i] * dtau[0][i] + out.v2tau2[tt[1]][i] * dtau[1][i];
                let rt2 = if spin == 0 { [0, 2] } else { [1, 3] };
                dvtau +=
                    out.v2rhotau[rt2[0]][i] * drho[0][i] + out.v2rhotau[rt2[1]][i] * drho[1][i];
                for pch in 0..3 {
                    dvtau += out.v2sigmatau[2 * pch + spin][i] * dsig[pch][i];
                }
                dvt[i] = w * dvtau;
            }
            // Assemble: phi^T dv + (dvg^T phi + phi^T dvg) + tau term.
            let mut v = batch.phi.transpose_matmul_seq(&dv);
            let cross_m = dvg.transpose_matmul_seq(&batch.phi);
            for r in 0..nao {
                for c in 0..nao {
                    v[(r, c)] += cross_m[(r, c)] + cross_m[(c, r)];
                }
            }
            // Tau channel: 1/2 sum_u g_u^T diag(dvt) g_u.
            for u in 0..3 {
                let mut scaled = g[u].clone();
                for i in 0..npts {
                    let f = 0.5 * dvt[i];
                    for mu in 0..nao {
                        scaled[(i, mu)] *= f;
                    }
                }
                let t = g[u].transpose_matmul_seq(&scaled);
                for (vv, tv) in v.as_mut_slice().iter_mut().zip(t.as_slice()) {
                    *vv += *tv;
                }
            }
            for (acc, vv) in vxc[spin].as_mut_slice().iter_mut().zip(v.as_slice()) {
                *acc += *vv;
            }
        }
        start = end;
    }
    Ok(vxc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use crate::basis::data;
    use crate::grid::GridOptions;
    use crate::math::Vec3;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};
    use std::time::Instant;

    fn fixture() -> (MolBasis, MolecularGrid, [Matrix; 2]) {
        let molecule = Molecule::new(
            [
                [-0.81, 0.12, -0.27],
                [0.93, -0.38, 0.14],
                [0.24, 1.47, 0.63],
                [-0.36, -0.74, 1.51],
            ]
            .map(|r| Atom {
                z: 1,
                position: Vec3::new(r[0], r[1], r[2]),
            })
            .to_vec(),
        );
        let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
        let grid = MolecularGrid::build(
            &molecule,
            &GridOptions {
                level: 0,
                weight_cutoff: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        let density = std::array::from_fn(|spin| {
            let mut p = Matrix::zeros(4, 4);
            for i in 0..4 {
                for j in i..4 {
                    let value = if i == j {
                        0.24 + 0.031 * (i + 1) as f64 + 0.047 * spin as f64
                    } else {
                        0.014 * ((i + 2 * j + spin + 1) as f64).sin()
                    };
                    p[(i, j)] = value;
                    p[(j, i)] = value;
                }
            }
            p
        });
        (basis, grid, density)
    }

    #[test]
    fn shared_budget_tracks_cache_and_preserves_streamed_fallback() {
        use crate::resources::{current_usage, reserve, ResourceLimits, MIB};
        let (basis, mut grid, density) = fixture();
        grid.points.truncate(17);
        grid.weights.truncate(17);
        grid.raw_weights.truncate(17);
        grid.atom_of.truncate(17);
        let functional = resolve(&FunctionalSpec::Combined("tpss".into())).unwrap();
        let refs = [&density[0], &density[1]];
        let expected = mgga_fxc_contraction(&basis, &grid, &functional, refs, &density).unwrap();
        ResourceLimits::new(Some(1), Some(1))
            .unwrap()
            .run(|| {
                let cache = MetaGgaKernel::try_new(&basis, &grid, &functional, refs, MIB)?.unwrap();
                let reserved = current_usage().unwrap().reserved_bytes;
                assert!(reserved > 0 && reserved < MIB / 2);
                assert_eq!(cache.apply(&density)?, expected);
                drop(cache);
                assert_eq!(current_usage().unwrap().reserved_bytes, 0);
                let pressure = reserve(MIB - 65536, "meta-GGA fallback fixture")?;
                assert!(MetaGgaKernel::try_new(&basis, &grid, &functional, refs, MIB)?.is_none());
                assert_eq!(
                    mgga_fxc_contraction(&basis, &grid, &functional, refs, &density)?,
                    expected
                );
                drop(pressure);
                assert_eq!(current_usage().unwrap().reserved_bytes, 0);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn bounded_cache_preserves_every_kernel_entry_and_four_step_density_fit() {
        let (basis, grid, density) = fixture();
        let stages = [
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            (
                [0.005, 0.0025, 0.00125, 0.000625],
                stencils::Stencil::Second,
            ),
            (
                [0.0025, 0.00125, 0.000625, 0.0003125],
                stencils::Stencil::Second,
            ),
            ([0.16, 0.08, 0.04, 0.02], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        for name in ["tpss", "revtpss", "m06l", "m06-2x"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let refs = [&density[0], &density[1]];
            assert!(MetaGgaKernel::try_new(&basis, &grid, &functional, refs, 0)
                .unwrap()
                .is_none());
            assert!(MetaGgaKernel::try_new(&basis, &grid, &functional, refs, 1)
                .unwrap()
                .is_none());
            let cache = MetaGgaKernel::try_new(&basis, &grid, &functional, refs, usize::MAX)
                .unwrap()
                .unwrap();
            let mut cached_seconds = 0.0;
            let mut uncached_seconds = 0.0;
            for spin in 0..2 {
                for i in 0..4 {
                    for j in i..4 {
                        let mut direction = [Matrix::zeros(4, 4), Matrix::zeros(4, 4)];
                        direction[spin][(i, j)] = 1.0;
                        direction[spin][(j, i)] = 1.0;
                        let started = Instant::now();
                        let image = cache.apply(&direction).unwrap();
                        cached_seconds += started.elapsed().as_secs_f64();
                        let started = Instant::now();
                        let ordinary =
                            mgga_fxc_contraction(&basis, &grid, &functional, refs, &direction)
                                .unwrap();
                        uncached_seconds += started.elapsed().as_secs_f64();
                        let exact: Vec<_> =
                            image.iter().flat_map(|m| m.as_slice()).copied().collect();
                        for (a, b) in exact.iter().zip(ordinary.iter().flat_map(|m| m.as_slice())) {
                            assert_eq!(a.to_bits(), b.to_bits(), "{name} cache changed the kernel");
                        }
                        let names: Vec<_> = (0..32)
                            .map(|k| {
                                format!(
                                    "{name} density[{spin}][{i}][{j}] kernel[{}][{}][{}]",
                                    k / 16,
                                    (k % 16) / 4,
                                    k % 4
                                )
                            })
                            .collect();
                        stencils::fit_components_staged(
                            &names,
                            &stages,
                            |steps, stencil| {
                                let mut errors = vec![[0.0; 4]; 32];
                                for (step, &h) in steps.iter().enumerate() {
                                    let at = |t: f64| {
                                        let mut shifted = density.clone();
                                        shifted[spin][(i, j)] += t;
                                        if i != j {
                                            shifted[spin][(j, i)] += t;
                                        }
                                        crate::ks::xc_contribution(
                                            &functional,
                                            &basis,
                                            &grid,
                                            [&shifted[0], &shifted[1]],
                                        )
                                        .unwrap()
                                        .vxc
                                        .iter()
                                        .flat_map(|m| m.as_slice())
                                        .copied()
                                        .collect::<Vec<_>>()
                                    };
                                    let plus = at(h);
                                    let minus = at(-h);
                                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                                        (at(2.0 * h), at(-2.0 * h))
                                    } else {
                                        (Vec::new(), Vec::new())
                                    };
                                    for k in 0..32 {
                                        let fd = stencil.first(
                                            plus[k],
                                            minus[k],
                                            plus2.get(k).copied().unwrap_or(0.0),
                                            minus2.get(k).copied().unwrap_or(0.0),
                                            h,
                                        );
                                        errors[k][step] = (fd - exact[k]).abs();
                                    }
                                }
                                errors
                            },
                            &mut failures,
                        );
                    }
                }
            }
            println!("{name} all 640 AO kernel entries: cached={cached_seconds:.6}s uncached={uncached_seconds:.6}s grid_points={}", grid.len());
        }
        components::assert_no_failures("meta-GGA cached kernels", &failures);
    }
}
