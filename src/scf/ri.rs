// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Density fitting (resolution of the identity) for the Coulomb matrix:
//! `J_μν ≈ Σ_PQ (μν|P) [V⁻¹]_PQ (Q|λσ) P_λσ` with `V = (P|Q)`.
//!
//! An optional acceleration (off by default): enabling it replaces only the
//! J build; exchange K stays on the exact path. The RI-J energy error with
//! the def2-universal-JFIT auxiliary basis is at the usual tens of µEh per
//! atom level; the toggle lets any result be re-checked against the exact
//! path.

use crate::basis::{data, MolBasis};
use crate::error::{DftError, Result};
use crate::integrals::{eri_2c, eri_3c, CoulombKernel};
use crate::linalg::{solve_linear, Matrix};
use crate::system::Molecule;
use rayon::prelude::*;

pub struct RiJ {
    naux: usize,
    nao: usize,
    /// Two-center metric V = (P|Q).
    v: Matrix,
    /// Three-center integrals, `naux × ntri` with packed μ ≤ ν pairs.
    three: Vec<f64>,
}

#[inline]
fn tri(mu: usize, nu: usize) -> usize {
    // μ ≤ ν packed index.
    nu * (nu + 1) / 2 + mu
}

impl RiJ {
    /// Assemble the RI machinery for `aux_name` on this molecule.
    pub fn build(
        mol: &Molecule,
        mb: &MolBasis,
        aux_name: &str,
        kernel: CoulombKernel,
        limit_mb: usize,
    ) -> Result<Self> {
        let aux_set = data::load_or_read(aux_name)?;
        let aux = MolBasis::build(mol, &aux_set)?;
        let naux = aux.nao_sph;
        let nao = mb.nao_sph;
        let ntri = nao * (nao + 1) / 2;
        let required_mb = naux * ntri * std::mem::size_of::<f64>() / (1024 * 1024);
        if required_mb > limit_mb {
            return Err(DftError::ResourceLimit {
                operation: "RI-J three-center integral store",
                required_mb,
                limit_mb,
            });
        }

        // V = (P|Q).
        let n_aux_shell = aux.shells.len();
        let mut v = Matrix::zeros(naux, naux);
        let blocks: Vec<(usize, usize, crate::integrals::Quartet)> = (0..n_aux_shell)
            .flat_map(|p| (p..n_aux_shell).map(move |q| (p, q)))
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&(p, q)| (p, q, eri_2c(&aux, p, q, kernel)))
            .collect();
        for (p, q, block) in blocks {
            let (op, oq) = (aux.shells[p].sph_off, aux.shells[q].sph_off);
            for i in 0..block.na {
                for k in 0..block.nc {
                    v[(op + i, oq + k)] = block.get(i, 0, k, 0);
                    v[(oq + k, op + i)] = block.get(i, 0, k, 0);
                }
            }
        }

        // (μν|P), packed lower triangle in μν.
        let ns = mb.shells.len();
        let pairs: Vec<(usize, usize)> =
            (0..ns).flat_map(|a| (a..ns).map(move |b| (a, b))).collect();
        let mut three = vec![0.0; naux * ntri];
        let chunks: Vec<Vec<(usize, usize, usize, crate::integrals::Quartet)>> = (0..n_aux_shell)
            .into_par_iter()
            .map(|p| {
                pairs
                    .iter()
                    .map(|&(a, b)| (a, b, p, eri_3c(mb, &aux, a, b, p, kernel)))
                    .collect()
            })
            .collect();
        for chunk in chunks {
            for (a, b, p, block) in chunk {
                let (oa, ob, op) = (
                    mb.shells[a].sph_off,
                    mb.shells[b].sph_off,
                    aux.shells[p].sph_off,
                );
                for i in 0..block.na {
                    for j in 0..block.nb {
                        let (mu, nu) = (oa + i, ob + j);
                        if mu > nu {
                            continue;
                        }
                        for k in 0..block.nc {
                            three[(op + k) * ntri + tri(mu, nu)] = block.get(i, j, k, 0);
                        }
                    }
                }
            }
        }
        Ok(Self {
            naux,
            nao,
            v,
            three,
        })
    }

    /// Fitted Coulomb matrix for the total density.
    pub fn j(&self, p_tot: &Matrix) -> Result<Matrix> {
        let (nao, naux) = (self.nao, self.naux);
        let ntri = nao * (nao + 1) / 2;
        // b_P = Σ_{μν} (μν|P) P_μν.
        let mut b = vec![0.0; naux];
        for p in 0..naux {
            let row = &self.three[p * ntri..(p + 1) * ntri];
            let mut acc = 0.0;
            for nu in 0..nao {
                for mu in 0..=nu {
                    let scale = if mu == nu { 1.0 } else { 2.0 };
                    acc += scale * p_tot[(mu, nu)] * row[tri(mu, nu)];
                }
            }
            b[p] = acc;
        }
        let c = solve_linear(&self.v, &b)?;
        let mut j = Matrix::zeros(nao, nao);
        for p in 0..naux {
            let cp = c[p];
            if cp == 0.0 {
                continue;
            }
            let row = &self.three[p * ntri..(p + 1) * ntri];
            for nu in 0..nao {
                for mu in 0..=nu {
                    j[(mu, nu)] += cp * row[tri(mu, nu)];
                }
            }
        }
        for nu in 0..nao {
            for mu in 0..nu {
                j[(nu, mu)] = j[(mu, nu)];
            }
        }
        Ok(j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::scf::{EriMode, JkEngine};
    use crate::system::Atom;

    /// `Tr[P J]`, written with the mathematical index order rather than
    /// relying on symmetry of either matrix.
    fn trace_pj(p: &Matrix, j: &Matrix) -> f64 {
        assert_eq!((p.rows, p.cols), (j.rows, j.cols));
        let mut value = 0.0;
        for mu in 0..p.rows {
            for nu in 0..p.cols {
                value += p[(mu, nu)] * j[(nu, mu)];
            }
        }
        value
    }

    /// A literal leading principal subspace of an existing auxiliary basis.
    /// Its functions, metric, and three-center rows are byte-for-byte a subset
    /// of `full`, so the small and large fitting spaces are rigorously nested.
    fn leading_auxiliary_subspace(full: &RiJ, naux: usize) -> RiJ {
        assert!(naux > 0 && naux < full.naux);
        let mut v = Matrix::zeros(naux, naux);
        for p in 0..naux {
            for q in 0..naux {
                v[(p, q)] = full.v[(p, q)];
            }
        }
        let ntri = full.nao * (full.nao + 1) / 2;
        RiJ {
            naux,
            nao: full.nao,
            v,
            three: full.three[..naux * ntri].to_vec(),
        }
    }

    /// RI-J against the exact J at the CONVERGED HF density (the density
    /// class the fitting basis is designed for): the Coulomb-energy fitting
    /// error must be at the documented µEh–sub-mEh scale, and the fitted J
    /// must be symmetric.
    #[test]
    fn ri_j_close_to_exact_j() {
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
        let opts = crate::scf::ScfOptions {
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..Default::default()
        };
        let scf = crate::scf::run_hf(&mol, &mb, &opts).unwrap();
        let p = scf.p_total();
        let ri = RiJ::build(
            &mol,
            &mb,
            "def2-universal-jfit",
            CoulombKernel::Coulomb,
            1024,
        )
        .unwrap();
        let j_ri = ri.j(&p).unwrap();
        let exact = JkEngine::new(
            &mb,
            EriMode::InCore { limit_mb: 512 },
            CoulombKernel::Coulomb,
        )
        .unwrap();
        let (j_exact, _) = exact.build(&mb, &p, [&p, &p], false).unwrap();
        // Coulomb-energy fitting error ½|Tr[P(J_RI − J_exact)]|.
        let mut de = 0.0;
        let n = mb.nao_sph;
        for i in 0..n {
            for jj in 0..n {
                de += p[(i, jj)] * (j_ri[(i, jj)] - j_exact[(i, jj)]);
            }
        }
        de = 0.5 * de.abs();
        assert!(de < 5e-4, "RI-J Coulomb-energy error too large: {de}");
        assert!(
            de > 1e-12,
            "suspiciously exact — fitting path not exercised?"
        );
        for i in 0..n {
            for jj in 0..i {
                assert!((j_ri[(i, jj)] - j_ri[(jj, i)]).abs() < 1e-12);
            }
        }
    }

    /// RI-J is a self-adjoint Coulomb-metric projection. Consequently it is
    /// linear, reciprocal for arbitrary symmetric P and Q, and variational
    /// for a positive-semidefinite density:
    ///
    /// `Tr[P J_RI(Q)] = Tr[Q J_RI(P)]`,
    /// `0 <= 1/2 Tr[D J_RI(D)] <= 1/2 Tr[D J_exact(D)]`.
    ///
    /// The metric quadratic form independently pins triangular packing and
    /// off-diagonal factors. A literal nested auxiliary subspace additionally
    /// pins the variational monotonicity under basis enlargement.
    #[test]
    fn ri_j_obeys_the_metric_projection_identities() {
        let set = data::load("def2-svp").unwrap();
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.1, 0.0, 0.7),
            },
        ]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let ri = RiJ::build(
            &mol,
            &mb,
            "def2-universal-jfit",
            CoulombKernel::Coulomb,
            512,
        )
        .unwrap();
        let n = mb.nao_sph;
        let mut p = Matrix::zeros(n, n);
        let mut q = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                // Signed, dense, unrelated matrices: reciprocity must not be
                // tested only on physical/positive densities or diagonals.
                let pv = (((3 * i + 5 * j + 1) % 9) as f64 - 4.0) / (7 + i + j) as f64;
                let qv = (((7 * i + 2 * j + 3) % 11) as f64 - 5.0) / (9 + 2 * i + j) as f64;
                p[(i, j)] = pv;
                p[(j, i)] = pv;
                q[(i, j)] = qv;
                q[(j, i)] = qv;
            }
        }
        let (a, b) = (1.7, -0.4);
        let mut apbq = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                apbq[(i, j)] = a * p[(i, j)] + b * q[(i, j)];
            }
        }
        let jp = ri.j(&p).unwrap();
        let jq = ri.j(&q).unwrap();
        let japbq = ri.j(&apbq).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert!((japbq[(i, j)] - a * jp[(i, j)] - b * jq[(i, j)]).abs() < 2e-11);
            }
        }

        let p_jq = trace_pj(&p, &jq);
        let q_jp = trace_pj(&q, &jp);
        let reciprocity_scale = 1.0 + p_jq.abs().max(q_jp.abs());
        assert!(
            (p_jq - q_jp).abs() < 2e-11 * reciprocity_scale,
            "RI-J is not self-adjoint: Tr[P J(Q)]={p_jq:.16e}, \
             Tr[Q J(P)]={q_jp:.16e}"
        );

        // D = u u^T + 0.4 v v^T is PSD by construction, while its signed
        // off-diagonal entries exercise the packed-density factor of two.
        let mut density = Matrix::zeros(n, n);
        let norm = (n as f64).sqrt();
        for i in 0..n {
            let ui = (i + 1) as f64 / ((n + 1) as f64 * norm);
            let vi = if i % 2 == 0 { 1.0 } else { -1.0 } / norm;
            for j in 0..n {
                let uj = (j + 1) as f64 / ((n + 1) as f64 * norm);
                let vj = if j % 2 == 0 { 1.0 } else { -1.0 } / norm;
                density[(i, j)] = ui * uj + 0.4 * vi * vj;
            }
        }
        let j_density = ri.j(&density).unwrap();

        let ntri = n * (n + 1) / 2;
        let mut rhs = vec![0.0; ri.naux];
        for aux in 0..ri.naux {
            for nu in 0..n {
                for mu in 0..=nu {
                    let scale = if mu == nu { 1.0 } else { 2.0 };
                    rhs[aux] += scale * density[(mu, nu)] * ri.three[aux * ntri + tri(mu, nu)];
                }
            }
        }
        let coeff = solve_linear(&ri.v, &rhs).unwrap();
        let metric_energy = 0.5 * rhs.iter().zip(&coeff).map(|(x, y)| x * y).sum::<f64>();
        let ri_energy = 0.5 * trace_pj(&density, &j_density);
        assert!(
            metric_energy >= -1e-12,
            "negative metric energy {metric_energy}"
        );
        assert!((metric_energy - ri_energy).abs() < 2e-11);

        // Independent four-center ERI oracle for the Coulomb-metric upper
        // bound. No fitted quantity enters J_exact.
        let exact = JkEngine::new(
            &mb,
            EriMode::InCore { limit_mb: 512 },
            CoulombKernel::Coulomb,
        )
        .unwrap();
        let (j_exact, _) = exact
            .build(&mb, &density, [&density, &density], false)
            .unwrap();
        let exact_energy = 0.5 * trace_pj(&density, &j_exact);
        let energy_scale = 1.0 + exact_energy.abs().max(ri_energy.abs());
        let roundoff = 2e-11 * energy_scale;
        assert!(
            ri_energy >= -1e-12,
            "RI Coulomb energy must be nonnegative: {ri_energy:.16e}"
        );
        assert!(
            ri_energy <= exact_energy + roundoff,
            "Coulomb-metric projection exceeded exact J: \
             E_RI={ri_energy:.16e}, E_exact={exact_energy:.16e}"
        );

        // The first atom's complete auxiliary block is a literal leading
        // subspace of the two-atom fitting basis; no named-basis nesting is
        // assumed. Projection norm can only increase on the larger space.
        assert_eq!(ri.naux % mol.len(), 0);
        let ri_small = leading_auxiliary_subspace(&ri, ri.naux / mol.len());
        let j_small = ri_small.j(&density).unwrap();
        let small_energy = 0.5 * trace_pj(&density, &j_small);
        assert!(
            small_energy >= -1e-12,
            "nested-subspace Coulomb energy is negative: {small_energy:.16e}"
        );
        assert!(
            small_energy <= ri_energy + roundoff,
            "enlarging a rigorously nested auxiliary space lowered the projection norm: \
             E_small={small_energy:.16e}, E_large={ri_energy:.16e}"
        );
    }
}
