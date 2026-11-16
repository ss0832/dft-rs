// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Kohn–Sham XC quadrature assembly: densities (and gradients) on the
//! grid, Exc, and the Vxc matrices. LDA and GGA branches; the meta-GGA
//! variables (τ, ∇²ρ) extend the same structure later.

use crate::basis::MolBasis;
use crate::error::Result;
use crate::grid::eval::eval_aos;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use rayon::prelude::*;

/// XC contribution of one density: energy, spin potentials, and the
/// numerically integrated electron count (grid-quality diagnostic).
pub struct XcContribution {
    pub exc: f64,
    pub vxc: [Matrix; 2],
    pub n_elec: f64,
}

/// Number of grid points per parallel work batch.
const BATCH: usize = 256;

// Keep completed partial matrices bounded independently of grid length.
// A single unavoidable batch may exceed this target; the existing managed
// allocation guard remains authoritative. This is not a whole-RSS bound.
fn ordered_partial_group_len(nao: usize) -> Result<usize> {
    let partial_bytes = crate::resources::checked_bytes(
        &[2, nao, nao, std::mem::size_of::<f64>()],
        "ordered XC partial matrices",
    )?;
    let target = crate::resources::remaining_bytes()
        .unwrap_or(64 * crate::resources::MIB)
        .min(64 * crate::resources::MIB);
    let preferred = rayon::current_num_threads().saturating_mul(8).clamp(1, 32);
    Ok(preferred.min((target / partial_bytes.max(1)).max(1)))
}

/// Evaluate Exc and Vxc for the spin densities `p`.
pub fn xc_contribution(
    functional: &ResolvedFunctional,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
) -> Result<XcContribution> {
    let gga = functional.family != Family::Lda;
    let meta = functional.family == Family::MetaGga;
    let nao = mb.nao_sph;
    let n = grid.len();
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(BATCH)
        .map(|s| (s, (s + BATCH).min(n)))
        .collect();

    let mut totals: Option<(f64, f64, [Matrix; 2])> = None;
    let group_len = ordered_partial_group_len(nao)?;
    for group in ranges.chunks(group_len) {
        let partials: Result<Vec<(f64, f64, Matrix, Matrix)>> = group
            .par_iter()
            .map(|&(start, end)| {
                let pts = &grid.points[start..end];
                let wts = &grid.weights[start..end];
                let npts = pts.len();
                let batch = eval_aos(mb, pts, gga);
                // ρ_σ(r_i) = Σ_μ (Φ P)_iμ Φ_iμ; ∇ρ_σ = 2 Σ_μ (Φ P)_iμ ∇Φ_iμ.
                let mut rho = [vec![0.0; npts], vec![0.0; npts]];
                let mut grad: [[Vec<f64>; 3]; 2] = Default::default();
                if gga {
                    for spin in grad.iter_mut() {
                        for ax in spin.iter_mut() {
                            *ax = vec![0.0; npts];
                        }
                    }
                }
                let mut phip = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
                let mut tau = [vec![0.0; 0], vec![0.0; 0]];
                if meta {
                    tau = [vec![0.0; npts], vec![0.0; npts]];
                }
                for spin in 0..2 {
                    phip[spin] = batch.phi.matmul_seq(p[spin]);
                    for i in 0..npts {
                        let mut acc = 0.0;
                        for mu in 0..nao {
                            acc += phip[spin][(i, mu)] * batch.phi[(i, mu)];
                        }
                        rho[spin][i] = acc.max(0.0);
                    }
                    if gga {
                        let g = batch.grad.as_ref().expect("gradients requested");
                        for ax in 0..3 {
                            let gp = if meta {
                                Some(g[ax].matmul_seq(p[spin]))
                            } else {
                                None
                            };
                            for i in 0..npts {
                                let mut acc = 0.0;
                                for mu in 0..nao {
                                    acc += phip[spin][(i, mu)] * g[ax][(i, mu)];
                                }
                                grad[spin][ax][i] = 2.0 * acc;
                                if let Some(gp) = &gp {
                                    // τσ = ½ Σ_ax Σ_μν Pσ ∂φμ ∂φν.
                                    let mut t = 0.0;
                                    for mu in 0..nao {
                                        t += gp[(i, mu)] * g[ax][(i, mu)];
                                    }
                                    tau[spin][i] += 0.5 * t;
                                }
                            }
                        }
                    }
                }
                // σ contractions.
                let sigma_store;
                let vars = if gga {
                    let mut saa = vec![0.0; npts];
                    let mut sab = vec![0.0; npts];
                    let mut sbb = vec![0.0; npts];
                    for i in 0..npts {
                        let ga = [grad[0][0][i], grad[0][1][i], grad[0][2][i]];
                        let gb = [grad[1][0][i], grad[1][1][i], grad[1][2][i]];
                        saa[i] = ga[0] * ga[0] + ga[1] * ga[1] + ga[2] * ga[2];
                        sab[i] = ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2];
                        sbb[i] = gb[0] * gb[0] + gb[1] * gb[1] + gb[2] * gb[2];
                    }
                    sigma_store = [saa, sab, sbb];
                    GridVars {
                        rho: [&rho[0], &rho[1]],
                        sigma: Some([&sigma_store[0], &sigma_store[1], &sigma_store[2]]),
                        tau: if meta { Some([&tau[0], &tau[1]]) } else { None },
                    }
                } else {
                    GridVars {
                        rho: [&rho[0], &rho[1]],
                        sigma: None,
                        tau: None,
                    }
                };
                let mut out = XcOut::default();
                #[cfg(test)]
                crate::xc::branch_freeze::set_batch(Some(start));
                let evaluated = functional.eval(&vars, 1, &mut out);
                #[cfg(test)]
                crate::xc::branch_freeze::set_batch(None);
                evaluated?;
                let mut exc = 0.0;
                let mut n_elec = 0.0;
                for i in 0..npts {
                    exc += wts[i] * out.e[i];
                    n_elec += wts[i] * (rho[0][i] + rho[1][i]);
                }
                // V^σ = Φᵀ diag(w·vρσ) Φ
                //     + [Aᵀ Φ + Φᵀ A],  A_iμ = w_i b_i·∇Φ_iμ,
                //   with b_i = 2 vσ_σσ ∇ρσ + vσ_αβ ∇ρσ̄  (GGA term).
                let mut vxc = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
                for spin in 0..2 {
                    let mut scaled = batch.phi.clone();
                    for i in 0..npts {
                        let f = wts[i] * out.vrho[spin][i];
                        for mu in 0..nao {
                            scaled[(i, mu)] *= f;
                        }
                    }
                    let mut v = batch.phi.transpose_matmul_seq(&scaled);
                    if gga {
                        let g = batch.grad.as_ref().expect("gradients requested");
                        let other = 1 - spin;
                        let (vs_own, vs_cross) = if spin == 0 {
                            (&out.vsigma[0], &out.vsigma[1])
                        } else {
                            (&out.vsigma[2], &out.vsigma[1])
                        };
                        let mut a = Matrix::zeros(npts, nao);
                        for i in 0..npts {
                            let b = [
                                wts[i]
                                    * (2.0 * vs_own[i] * grad[spin][0][i]
                                        + vs_cross[i] * grad[other][0][i]),
                                wts[i]
                                    * (2.0 * vs_own[i] * grad[spin][1][i]
                                        + vs_cross[i] * grad[other][1][i]),
                                wts[i]
                                    * (2.0 * vs_own[i] * grad[spin][2][i]
                                        + vs_cross[i] * grad[other][2][i]),
                            ];
                            for mu in 0..nao {
                                a[(i, mu)] = b[0] * g[0][(i, mu)]
                                    + b[1] * g[1][(i, mu)]
                                    + b[2] * g[2][(i, mu)];
                            }
                        }
                        let cross = a.transpose_matmul_seq(&batch.phi);
                        for r in 0..nao {
                            for c in 0..nao {
                                v[(r, c)] += cross[(r, c)] + cross[(c, r)];
                            }
                        }
                        // Meta-GGA τ term: ½ Σ_ax ∂φμ diag(w·vτσ) ∂φν.
                        if meta {
                            for ax in 0..3 {
                                let mut scaled_g = g[ax].clone();
                                for i in 0..npts {
                                    let f = 0.5 * wts[i] * out.vtau[spin][i];
                                    for mu in 0..nao {
                                        scaled_g[(i, mu)] *= f;
                                    }
                                }
                                let vt = g[ax].transpose_matmul_seq(&scaled_g);
                                for (dst, src) in v.as_mut_slice().iter_mut().zip(vt.as_slice()) {
                                    *dst += *src;
                                }
                            }
                        }
                    }
                    vxc[spin] = v;
                }
                let [va, vb] = vxc;
                Ok((exc, n_elec, va, vb))
            })
            .collect();
        let partials = partials?;
        let (exc, n_elec, vxc) = totals
            .get_or_insert_with(|| (0.0, 0.0, [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)]));
        for (e, ne, va, vb) in partials {
            *exc += e;
            *n_elec += ne;
            for (dst, src) in vxc[0].as_mut_slice().iter_mut().zip(va.as_slice()) {
                *dst += *src;
            }
            for (dst, src) in vxc[1].as_mut_slice().iter_mut().zip(vb.as_slice()) {
                *dst += *src;
            }
        }
    }
    let (exc, n_elec, mut vxc) =
        totals.unwrap_or_else(|| (0.0, 0.0, [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)]));
    // Symmetrize against the tiny asymmetry of floating accumulation.
    for m in vxc.iter_mut() {
        for i in 0..nao {
            for j in 0..i {
                let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                m[(i, j)] = avg;
                m[(j, i)] = avg;
            }
        }
    }
    Ok(XcContribution { exc, vxc, n_elec })
}

/// Gradient-regulated asymptotic correction (GRAC) of the exchange-
/// correlation potential of `functional`: the Fock contribution that turns
/// its potential into `(1 - f) (v_bulk - shift) + f v_LB94`, with
/// `f = 1 / (1 + exp(-0.5 (x - 40)))` of the reduced gradient
/// `x = |grad rho| / rho^(4/3)` of the spin-summed density and the LB94
/// potential `v_LDA,s - 0.05 rho_s^(1/3) x_s^2 / (1 + 0.15 x_s asinh x_s)`
/// (`x_s` of the spin density, `v_LDA` the SVWN5 potential), whose tail is
/// `-1/r`. The bulk potential is weighted point by point in its matrix
/// integrand, gradient terms included. `shift` is the ionization potential
/// plus the highest occupied orbital energy of the bulk functional. The
/// correction changes the potential only; the energy stays that of
/// `functional`. Meta-GGA potentials are not corrected.
pub fn grac_correction(
    functional: &ResolvedFunctional,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
    shift: f64,
) -> Result<[Matrix; 2]> {
    if functional.family == Family::MetaGga || functional.parts.is_empty() || !shift.is_finite() {
        return Err(crate::DftError::InvalidInput(
            "the asymptotic correction needs a local or gradient-corrected density functional \
             and a finite shift"
                .into(),
        ));
    }
    let lda = crate::xc::resolve(&crate::xc::FunctionalSpec::Combined("svwn5".into()))?;
    let gga = functional.family != Family::Lda;
    let nao = mb.nao_sph;
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
            let batch = eval_aos(mb, pts, true);
            let g = batch.grad.as_ref().expect("gradients requested");
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut grad: [[Vec<f64>; 3]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; npts]));
            for spin in 0..2 {
                let phip = batch.phi.matmul_seq(p[spin]);
                for i in 0..npts {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        acc += phip[(i, mu)] * batch.phi[(i, mu)];
                    }
                    rho[spin][i] = acc.max(0.0);
                    for ax in 0..3 {
                        let mut acc = 0.0;
                        for mu in 0..nao {
                            acc += phip[(i, mu)] * g[ax][(i, mu)];
                        }
                        grad[spin][ax][i] = 2.0 * acc;
                    }
                }
            }
            let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let at = |s: usize, i: usize| [grad[s][0][i], grad[s][1][i], grad[s][2][i]];
            let sigma: [Vec<f64>; 3] = [
                (0..npts).map(|i| dot(at(0, i), at(0, i))).collect(),
                (0..npts).map(|i| dot(at(0, i), at(1, i))).collect(),
                (0..npts).map(|i| dot(at(1, i), at(1, i))).collect(),
            ];
            let vars = GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: if gga {
                    Some([&sigma[0], &sigma[1], &sigma[2]])
                } else {
                    None
                },
                tau: None,
            };
            let mut bulk = XcOut::default();
            functional.eval(&vars, 1, &mut bulk)?;
            let local = GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: None,
                tau: None,
            };
            let mut lb = XcOut::default();
            lda.eval(&local, 1, &mut lb)?;
            // Switching function of the spin-summed reduced gradient.
            let switch: Vec<f64> = (0..npts)
                .map(|i| {
                    let total = rho[0][i] + rho[1][i];
                    let gradient = [
                        grad[0][0][i] + grad[1][0][i],
                        grad[0][1][i] + grad[1][1][i],
                        grad[0][2][i] + grad[1][2][i],
                    ];
                    if total < 1e-14 {
                        return 1.0;
                    }
                    let x = dot(gradient, gradient).sqrt() / total.powf(4.0 / 3.0);
                    1.0 / (1.0 + (-0.5 * (x - 40.0)).exp())
                })
                .collect();
            let mut out = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
            for (spin, target) in out.iter_mut().enumerate() {
                let mut scaled = batch.phi.clone();
                for i in 0..npts {
                    let f = switch[i];
                    let r = rho[spin][i];
                    let lb94 = if r > 1e-14 {
                        let gs = at(spin, i);
                        let xs = dot(gs, gs).sqrt() / r.powf(4.0 / 3.0);
                        lb.vrho[spin][i]
                            - 0.05 * r.cbrt() * xs * xs / (1.0 + 0.15 * xs * xs.asinh())
                    } else {
                        0.0
                    };
                    let c = wts[i] * (-f * bulk.vrho[spin][i] - (1.0 - f) * shift + f * lb94);
                    for mu in 0..nao {
                        scaled[(i, mu)] *= c;
                    }
                }
                let mut v = batch.phi.transpose_matmul_seq(&scaled);
                if gga {
                    let other = 1 - spin;
                    let (own, cross) = if spin == 0 {
                        (&bulk.vsigma[0], &bulk.vsigma[1])
                    } else {
                        (&bulk.vsigma[2], &bulk.vsigma[1])
                    };
                    let mut a = Matrix::zeros(npts, nao);
                    for i in 0..npts {
                        let weight = -switch[i] * wts[i];
                        let b: [f64; 3] = std::array::from_fn(|ax| {
                            weight
                                * (2.0 * own[i] * grad[spin][ax][i] + cross[i] * grad[other][ax][i])
                        });
                        for mu in 0..nao {
                            a[(i, mu)] =
                                b[0] * g[0][(i, mu)] + b[1] * g[1][(i, mu)] + b[2] * g[2][(i, mu)];
                        }
                    }
                    let term = a.transpose_matmul_seq(&batch.phi);
                    for r in 0..nao {
                        for c in 0..nao {
                            v[(r, c)] += term[(r, c)] + term[(c, r)];
                        }
                    }
                }
                *target = v;
            }
            Ok(out)
        })
        .collect();
    let mut total = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    for part in partials? {
        for (t, m) in total.iter_mut().zip(&part) {
            for (x, y) in t.as_mut_slice().iter_mut().zip(m.as_slice()) {
                *x += y;
            }
        }
    }
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
