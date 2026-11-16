// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Coulomb (J) and exchange (K) matrix builders.
//!
//! Spin-resolved throughout: J is built from the total density, K per spin
//! channel. Two paths share one contraction definition:
//! - **in-core**: dense contraction against a stored ERI tensor (small
//!   systems; the screening-free reference),
//! - **direct**: screened shell-quartet recomputation with Schwarz bounds
//!   `|(ab|cd)| ≤ Q_ab·Q_cd` and density weighting.
//!
//! Chemists' notation: J_μν = Σ_λσ P_λσ (μν|λσ),
//! K^s_μν = Σ_λσ P^s_λσ (μλ|νσ).

use super::ScfOptions;
use crate::basis::MolBasis;
use crate::error::Result;
use crate::integrals::{eri_incore, eri_shell_quartet, schwarz_matrix, CoulombKernel, EriTensor};
use crate::linalg::Matrix;
use rayon::prelude::*;
use std::collections::HashSet;

/// How the two-electron contribution is evaluated.
#[derive(Clone, Copy, Debug)]
pub enum EriMode {
    /// Dense stored tensor (guarded by a memory limit in MiB).
    InCore { limit_mb: usize },
    /// Screened recomputation each iteration.
    Direct { screen_threshold: f64 },
}

impl Default for EriMode {
    fn default() -> Self {
        EriMode::Direct {
            screen_threshold: 1e-12,
        }
    }
}

/// Shared linear Coulomb/exchange density kernel for response and orbital
/// optimization, using the existing full- and long-range integral caches.
pub(crate) fn coulomb_exchange_response(
    basis: &MolBasis,
    engine: &JkEngine,
    long_range: Option<&JkEngine>,
    exchange: f64,
    range_exchange: f64,
    density: &[Matrix; 2],
) -> Result<[Matrix; 2]> {
    let mut total = density[0].clone();
    for (a, b) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
        *a += b;
    }
    let (j, k) = engine.build(basis, &total, [&density[0], &density[1]], exchange != 0.0)?;
    let lr = long_range
        .map(|engine| engine.build_k_only(basis, [&density[0], &density[1]]))
        .transpose()?;
    Ok(std::array::from_fn(|s| {
        let mut f = j.clone();
        for i in 0..f.as_slice().len() {
            f.as_mut_slice()[i] -= exchange * k[s].as_slice()[i];
            if let Some(lr) = &lr {
                f.as_mut_slice()[i] -= range_exchange * lr[s].as_slice()[i];
            }
        }
        f
    }))
}

/// J/K engine holding whatever the chosen path needs.
/// Clones share immutable integral/RI storage; each density response retains
/// its own frozen screening scale without duplicating the integral tensors.
#[derive(Clone)]
pub struct JkEngine {
    mode: EriMode,
    kernel: CoulombKernel,
    incore: Option<std::sync::Arc<EriTensor>>,
    schwarz: Option<std::sync::Arc<Matrix>>,
    /// Density fitting for J (K stays exact) when enabled.
    ri: Option<std::sync::Arc<super::ri::RiJ>>,
    /// For a response operator, freeze the direct integral mask at the
    /// reference density so changing a trial vector's magnitude cannot
    /// change the operator being solved.
    screening_density: Option<f64>,
}

impl JkEngine {
    pub fn new(mb: &MolBasis, mode: EriMode, kernel: CoulombKernel) -> Result<Self> {
        match mode {
            EriMode::InCore { limit_mb } => Ok(Self {
                mode,
                kernel,
                incore: Some(std::sync::Arc::new(eri_incore(mb, kernel, limit_mb)?)),
                schwarz: None,
                ri: None,
                screening_density: None,
            }),
            EriMode::Direct { .. } => Ok(Self {
                mode,
                kernel,
                incore: None,
                schwarz: Some(std::sync::Arc::new(schwarz_matrix(mb, kernel))),
                ri: None,
                screening_density: None,
            }),
        }
    }

    pub fn from_options(
        mol: &crate::system::Molecule,
        mb: &MolBasis,
        opts: &ScfOptions,
    ) -> Result<Self> {
        let mut engine = Self::new(mb, opts.eri_mode, opts.kernel)?;
        if let Some(aux_name) = &opts.density_fit {
            let limit = match opts.eri_mode {
                EriMode::InCore { limit_mb } => limit_mb,
                _ => 2048,
            };
            engine.ri = Some(std::sync::Arc::new(super::ri::RiJ::build(
                mol,
                mb,
                aux_name,
                opts.kernel,
                limit,
            )?));
        }
        Ok(engine)
    }

    /// Whether the Coulomb build is density-fitted.
    pub fn density_fitted(&self) -> bool {
        self.ri.is_some()
    }

    pub(crate) fn with_reference_screening(mut self, total: &Matrix, spin: [&Matrix; 2]) -> Self {
        let scale = total
            .as_slice()
            .iter()
            .chain(spin[0].as_slice())
            .chain(spin[1].as_slice())
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        self.screening_density = Some(if scale == 0.0 { 1.0 } else { scale });
        self
    }

    /// Build `J(P_tot)` and (when `want_k`) `[K(P_a), K(P_b)]`; with
    /// `want_k = false` (pure functionals) the K matrices are returned
    /// zeroed and their work is skipped. With density fitting enabled the
    /// J comes from the RI path and only K (if wanted) hits the exact path.
    pub fn build(
        &self,
        mb: &MolBasis,
        p_tot: &Matrix,
        p_spin: [&Matrix; 2],
        want_k: bool,
    ) -> Result<(Matrix, [Matrix; 2])> {
        let want_j_exact = self.ri.is_none();
        let (mut j, k) = if want_j_exact || want_k {
            match self.mode {
                EriMode::InCore { .. } => jk_incore_impl(
                    self.incore.as_ref().expect("in-core tensor present"),
                    p_tot,
                    p_spin,
                    want_j_exact,
                    want_k,
                ),
                EriMode::Direct { screen_threshold } => jk_direct(
                    mb,
                    self.kernel,
                    self.schwarz.as_ref().expect("schwarz matrix present"),
                    screen_threshold,
                    p_tot,
                    p_spin,
                    want_j_exact,
                    want_k,
                    self.screening_density,
                ),
            }
        } else {
            let n = mb.nao_sph;
            (
                Matrix::zeros(n, n),
                [Matrix::zeros(n, n), Matrix::zeros(n, n)],
            )
        };
        if let Some(ri) = &self.ri {
            j = ri.j(p_tot)?;
        }
        Ok((j, k))
    }

    /// Exchange-only build (long-range K for range-separated hybrids).
    pub fn build_k_only(&self, mb: &MolBasis, p_spin: [&Matrix; 2]) -> Result<[Matrix; 2]> {
        let n = mb.nao_sph;
        let zero = Matrix::zeros(n, n);
        let (_, k) = match self.mode {
            EriMode::InCore { .. } => jk_incore_impl(
                self.incore.as_ref().expect("in-core tensor present"),
                &zero,
                p_spin,
                false,
                true,
            ),
            EriMode::Direct { screen_threshold } => jk_direct(
                mb,
                self.kernel,
                self.schwarz.as_ref().expect("schwarz matrix present"),
                screen_threshold,
                &zero,
                p_spin,
                false,
                true,
                self.screening_density,
            ),
        };
        Ok(k)
    }
}

/// Dense in-core contraction (reference path).
pub fn jk_incore(
    eri: &EriTensor,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    want_k: bool,
) -> (Matrix, [Matrix; 2]) {
    jk_incore_impl(eri, p_tot, p_spin, true, want_k)
}

fn jk_incore_impl(
    eri: &EriTensor,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    want_j: bool,
    want_k: bool,
) -> (Matrix, [Matrix; 2]) {
    let n = eri.nao;
    let mut j = Matrix::zeros(n, n);
    let mut k = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
    if n == 0 {
        return (j, k);
    }
    let [ka, kb] = &mut k;
    // Independent output rows share the existing rayon pool. Each entry keeps
    // its original contraction order, including compensated product errors.
    j.as_mut_slice()
        .par_chunks_mut(n)
        .zip(ka.as_mut_slice().par_chunks_mut(n))
        .zip(kb.as_mut_slice().par_chunks_mut(n))
        .enumerate()
        .for_each(|(mu, ((jr, ar), br))| {
            for nu in 0..n {
                let mut jv = crate::linalg::AccurateSum::default();
                let mut kv = [crate::linalg::AccurateSum::default(); 2];
                for lam in 0..n {
                    for sig in 0..n {
                        if want_j {
                            jv.add_product(p_tot[(lam, sig)], eri.get(mu, nu, lam, sig));
                        }
                        if want_k {
                            let exch = eri.get(mu, lam, nu, sig);
                            kv[0].add_product(p_spin[0][(lam, sig)], exch);
                            kv[1].add_product(p_spin[1][(lam, sig)], exch);
                        }
                    }
                }
                jr[nu] = jv.value();
                ar[nu] = kv[0].value();
                br[nu] = kv[1].value();
            }
        });
    (j, k)
}

/// Screened direct build: shell quartets above the Schwarz×density bound are
/// recomputed and digested into J and K with explicit permutation handling.
#[allow(clippy::too_many_arguments)]
fn jk_direct(
    mb: &MolBasis,
    kernel: CoulombKernel,
    q: &Matrix,
    threshold: f64,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    want_j: bool,
    want_k: bool,
    screening_density: Option<f64>,
) -> (Matrix, [Matrix; 2]) {
    let ns = mb.shells.len();
    let n = mb.nao_sph;
    // Density-based screening weight per shell pair: the largest |P| element
    // touching the pair (either J or K pathways).
    let pmax_shell = {
        let mut m = Matrix::zeros(ns, ns);
        for a in 0..ns {
            for b in 0..ns {
                let (oa, na) = (mb.shells[a].sph_off, crate::basis::n_sph(mb.shells[a].l));
                let (ob, nb) = (mb.shells[b].sph_off, crate::basis::n_sph(mb.shells[b].l));
                let mut mx: f64 = 0.0;
                for i in 0..na {
                    for jj in 0..nb {
                        mx = mx.max(p_tot[(oa + i, ob + jj)].abs());
                        mx = mx.max(p_spin[0][(oa + i, ob + jj)].abs());
                        mx = mx.max(p_spin[1][(oa + i, ob + jj)].abs());
                    }
                }
                m[(a, b)] = mx;
            }
        }
        m
    };
    let global_pmax = screening_density.unwrap_or_else(|| {
        pmax_shell
            .as_slice()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v))
            .max(1e-30)
    });

    // Unique canonical quartets, screened.
    let mut quartets = Vec::new();
    for a in 0..ns {
        for b in a..ns {
            let qab = q[(a, b)];
            if qab * q_max_row(q, ns) * global_pmax < threshold {
                continue;
            }
            for c in a..ns {
                let d0 = if c == a { b } else { c };
                for d in d0..ns {
                    let bound = qab * q[(c, d)];
                    if bound * global_pmax < threshold {
                        continue;
                    }
                    quartets.push((a, b, c, d));
                }
            }
        }
    }

    // Parallel over quartets with per-thread accumulators.
    let partials: Vec<(Matrix, Matrix, Matrix)> = quartets
        .par_chunks(64.max(quartets.len() / (8 * rayon::current_num_threads()).max(1)))
        .map(|chunk| {
            let mut j = Matrix::zeros(n, n);
            let mut ka = Matrix::zeros(n, n);
            let mut kb = Matrix::zeros(n, n);
            for &(a, b, c, d) in chunk {
                let quartet = eri_shell_quartet(mb, a, b, c, d, kernel);
                // The 8 slot permutations σ (image position → canonical
                // slot). Images with identical shell tuples are equivalent by
                // integral permutational symmetry: keep the first σ per tuple.
                const SIGMAS: [[usize; 4]; 8] = [
                    [0, 1, 2, 3],
                    [1, 0, 2, 3],
                    [0, 1, 3, 2],
                    [1, 0, 3, 2],
                    [2, 3, 0, 1],
                    [3, 2, 0, 1],
                    [2, 3, 1, 0],
                    [3, 2, 1, 0],
                ];
                let shells = [a, b, c, d];
                let mut seen = HashSet::new();
                for sigma in SIGMAS {
                    let tuple = (
                        shells[sigma[0]],
                        shells[sigma[1]],
                        shells[sigma[2]],
                        shells[sigma[3]],
                    );
                    if !seen.insert(tuple) {
                        continue;
                    }
                    scatter_quartet(
                        mb, &quartet, shells, sigma, p_tot, p_spin, want_j, want_k, &mut j,
                        &mut ka, &mut kb,
                    );
                }
            }
            (j, ka, kb)
        })
        .collect();

    let mut j = Matrix::zeros(n, n);
    let mut k = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
    for (pj, pka, pkb) in partials {
        for (dst, src) in j.as_mut_slice().iter_mut().zip(pj.as_slice()) {
            *dst += *src;
        }
        for (dst, src) in k[0].as_mut_slice().iter_mut().zip(pka.as_slice()) {
            *dst += *src;
        }
        for (dst, src) in k[1].as_mut_slice().iter_mut().zip(pkb.as_slice()) {
            *dst += *src;
        }
    }
    (j, k)
}

fn q_max_row(q: &Matrix, ns: usize) -> f64 {
    let mut mx: f64 = 0.0;
    for i in 0..ns {
        for j in 0..ns {
            mx = mx.max(q[(i, j)]);
        }
    }
    mx
}

/// Add one permuted image of the canonical quartet block to J and K.
///
/// `sigma` maps image position → canonical slot; the image's shells are
/// `shells[sigma[pos]]` and its element `(i0,i1,i2,i3)` equals the canonical
/// block element with `canon_idx[sigma[pos]] = i_pos`.
#[allow(clippy::too_many_arguments)]
fn scatter_quartet(
    mb: &MolBasis,
    quartet: &crate::integrals::Quartet,
    shells: [usize; 4],
    sigma: [usize; 4],
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
    want_j: bool,
    want_k: bool,
    j: &mut Matrix,
    ka: &mut Matrix,
    kb: &mut Matrix,
) {
    let (pa, pb, pc, pd) = (
        shells[sigma[0]],
        shells[sigma[1]],
        shells[sigma[2]],
        shells[sigma[3]],
    );
    let off = |s: usize| mb.shells[s].sph_off;
    let nfun = |s: usize| crate::basis::n_sph(mb.shells[s].l);
    let (na, nb, nc, nd) = (nfun(pa), nfun(pb), nfun(pc), nfun(pd));
    let (oa, ob, oc, od) = (off(pa), off(pb), off(pc), off(pd));
    let value = |ip: usize, jp: usize, kp: usize, lp: usize| -> f64 {
        let mut canon_idx = [0usize; 4];
        canon_idx[sigma[0]] = ip;
        canon_idx[sigma[1]] = jp;
        canon_idx[sigma[2]] = kp;
        canon_idx[sigma[3]] = lp;
        quartet.get(canon_idx[0], canon_idx[1], canon_idx[2], canon_idx[3])
    };
    if want_j {
        for i in 0..na {
            for jj in 0..nb {
                let mut jv = 0.0;
                for k in 0..nc {
                    for l in 0..nd {
                        jv += p_tot[(oc + k, od + l)] * value(i, jj, k, l);
                    }
                }
                j[(oa + i, ob + jj)] += jv;
            }
        }
    }
    if !want_k {
        return;
    }
    // K^s_{μλ}: μ ∈ pa, ν ∈ pb, λ ∈ pc, σ ∈ pd — contribution
    // K[μ, λ... using (μν|λσ): K^s_{μλ} += P^s_{νσ} (μν|λσ).
    for i in 0..na {
        for k in 0..nc {
            let mut kva = 0.0;
            let mut kvb = 0.0;
            for jj in 0..nb {
                for l in 0..nd {
                    let v = value(i, jj, k, l);
                    kva += p_spin[0][(ob + jj, od + l)] * v;
                    kvb += p_spin[1][(ob + jj, od + l)] * v;
                }
            }
            ka[(oa + i, oc + k)] += kva;
            kb[(oa + i, oc + k)] += kvb;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::math::Vec3;
    use crate::system::{Atom, Molecule};

    /// The screened direct build must reproduce the in-core contraction on
    /// a mixed s/p/d system with an asymmetric density.
    #[test]
    fn direct_matches_incore() {
        let set = data::load("def2-svp").unwrap();
        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.2217),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 1.4309, -0.8867),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, -1.4309, -0.8867),
            },
        ]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let n = mb.nao_sph;
        // Deterministic pseudo-random, genuinely non-symmetric transition
        // densities. Ordinary SCF densities are symmetric, so mirroring the
        // upper triangle here would leave the TDA/CIS contraction untested.
        let mut p_a = Matrix::zeros(n, n);
        let mut p_b = Matrix::zeros(n, n);
        let mut state = 0x12345678u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for i in 0..n {
            for j in 0..n {
                p_a[(i, j)] = 0.1 * rng();
                p_b[(i, j)] = 0.1 * rng();
            }
        }
        assert!((p_a[(0, 1)] - p_a[(1, 0)]).abs() > 1e-6);
        assert!((p_b[(0, 1)] - p_b[(1, 0)]).abs() > 1e-6);
        let mut p_tot = p_a.clone();
        for (t, b) in p_tot.as_mut_slice().iter_mut().zip(p_b.as_slice()) {
            *t += *b;
        }
        let incore = JkEngine::new(
            &mb,
            EriMode::InCore { limit_mb: 512 },
            CoulombKernel::Coulomb,
        )
        .unwrap();
        let direct = JkEngine::new(
            &mb,
            EriMode::Direct {
                screen_threshold: 1e-14,
            },
            CoulombKernel::Coulomb,
        )
        .unwrap();
        let (j1, k1) = incore.build(&mb, &p_tot, [&p_a, &p_b], true).unwrap();
        let (j2, k2) = direct.build(&mb, &p_tot, [&p_a, &p_b], true).unwrap();
        assert!(
            j1.max_abs_diff(&j2) < 1e-10,
            "J mismatch {}",
            j1.max_abs_diff(&j2)
        );
        assert!(k1[0].max_abs_diff(&k2[0]) < 1e-10, "Ka mismatch");
        assert!(k1[1].max_abs_diff(&k2[1]) < 1e-10, "Kb mismatch");
    }

    #[test]
    fn shared_cache_policy_respects_budget_explicit_limits_and_density_fitting() {
        let mut opts = ScfOptions {
            incore_mb: 283,
            ..ScfOptions::default()
        };
        assert!(matches!(
            opts.effective_eri_mode(78),
            EriMode::InCore { limit_mb: 291 }
        ));
        opts.incore_mb = 282;
        assert!(matches!(
            opts.effective_eri_mode(78),
            EriMode::Direct { .. }
        ));
        opts.incore_mb = usize::MAX;
        assert!(matches!(
            opts.effective_eri_mode(usize::MAX),
            EriMode::Direct { .. }
        ));
        opts.density_fit = Some("auxiliary".into());
        assert!(matches!(opts.effective_eri_mode(4), EriMode::Direct { .. }));
        opts.eri_mode = EriMode::InCore { limit_mb: 7 };
        opts.incore_mb = 0;
        assert!(matches!(
            opts.effective_eri_mode(78),
            EriMode::InCore { limit_mb: 7 }
        ));
    }
}
