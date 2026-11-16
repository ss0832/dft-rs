// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Variational Fermi ensemble in one shared real spatial orbital frame.

use crate::linalg::Matrix;
use crate::response::complex::Complex;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::ResponseSolver;
use crate::{DftError, Result};

use super::super::fields::{MolecularDensityFields, MolecularDensityModel};
use super::super::soscf::orthogonal_exponential;
use super::{occupied_density, ScfResult};

struct State {
    c: Matrix,
    levels: [Vec<f64>; 2],
    f: [Vec<f64>; 2],
    mu: [Option<f64>; 2],
    entropy: f64,
    density: [Matrix; 2],
    fields: MolecularDensityFields,
    fock: [Matrix; 2],
    a: f64,
}

impl State {
    fn new(
        model: &MolecularDensityModel<'_>,
        c: Matrix,
        levels: [Vec<f64>; 2],
        counts: [usize; 2],
    ) -> Result<Self> {
        let mut f = [Vec::new(), Vec::new()];
        let mut mu = [None; 2];
        let mut entropy = 0.0;
        let kt = model.opts.electronic_temperature;
        for spin in 0..2 {
            let count = counts[spin];
            if count == 0 || count == c.cols {
                f[spin] = vec![f64::from(count != 0); c.cols];
                continue;
            }
            let (occupations, level, contribution) =
                super::super::fermi_dirac_occupations(&levels[spin], count as f64, kt)?;
            f[spin] = occupations;
            mu[spin] = Some(level);
            entropy += contribution;
        }
        let density = std::array::from_fn(|s| occupied_density(&c, &f[s]));
        let fields = model.evaluate(&density)?;
        let fock = fields.fock.each_ref().map(|f| project(f, &c));
        let a = fields.electronic_energy - kt * entropy;
        Ok(Self {
            c,
            levels,
            f,
            mu,
            entropy,
            density,
            fields,
            fock,
            a,
        })
    }

    fn pairs(&self) -> Vec<(usize, usize)> {
        (0..self.c.cols)
            .flat_map(|p| (p + 1..self.c.cols).map(move |q| (p, q)))
            .filter(|&(p, q)| self.f.iter().any(|f| f[p] != f[q]))
            .collect()
    }

    fn residual(&self, pairs: &[(usize, usize)]) -> Vec<f64> {
        pairs
            .iter()
            .map(|&(p, q)| {
                (0..2)
                    .map(|s| (self.f[s][p] - self.f[s][q]) * self.fock[s][(p, q)])
                    .sum()
            })
            .chain((0..2).flat_map(|s| {
                (0..self.c.cols).map(move |i| self.levels[s][i] - self.fock[s][(i, i)])
            }))
            .collect()
    }

    fn trial(
        &self,
        x: &[f64],
        pairs: &[(usize, usize)],
        fraction: f64,
        model: &MolecularDensityModel<'_>,
        counts: [usize; 2],
    ) -> Result<Self> {
        let k = rotation(self.c.cols, pairs, x, fraction);
        let c = self.c.matmul_seq(&orthogonal_exponential(&k));
        let levels = std::array::from_fn(|s| {
            (0..c.cols)
                .map(|i| self.levels[s][i] + fraction * x[pairs.len() + s * c.cols + i])
                .collect()
        });
        Self::new(model, c, levels, counts)
    }

    fn populate(&self, result: &mut ScfResult) {
        result.mo_coeff = [self.c.clone(), self.c.clone()];
        result.mo_energy = self.fock.each_ref().map(diagonal);
        result.density = self.density.clone();
        result.fock = self.fields.fock.clone();
        result.e_elec = self.fields.electronic_energy;
        result.e_xc = self.fields.xc_energy;
        result.e_nlc = self.fields.nonlocal_energy;
        result.e_pcm = self.fields.solvent_energy;
        result.grid_electron_count = self.fields.grid_electron_count;
        result.occupations = self.f.clone();
        result.electronic_entropy = self.entropy;
        result.chemical_potentials = self.mu;
        result.energy = self.a + result.e_nuc;
        // Independent thermal spin occupations, including number fluctuations.
        let sz = 0.5 * (result.n_alpha as f64 - result.n_beta as f64);
        let variance: f64 = self.f.iter().flatten().map(|f| f * (1.0 - f)).sum();
        let cross: f64 = self.f[0].iter().zip(&self.f[1]).map(|(a, b)| a * b).sum();
        result.s_squared =
            sz * sz + 0.5 * (result.n_alpha + result.n_beta) as f64 - cross + 0.25 * variance;
    }
}

fn project(f: &Matrix, c: &Matrix) -> Matrix {
    c.matmul_accurate(&f.matmul_accurate(c, false), true)
}

fn diagonal(f: &Matrix) -> Vec<f64> {
    (0..f.rows).map(|i| f[(i, i)]).collect()
}

fn rotation(n: usize, pairs: &[(usize, usize)], x: &[f64], fraction: f64) -> Matrix {
    let mut k = Matrix::zeros(n, n);
    for (&(p, q), &value) in pairs.iter().zip(x) {
        k[(q, p)] = fraction * value;
        k[(p, q)] = -k[(q, p)];
    }
    k
}

pub(crate) struct Jacobian<'a> {
    pub c: &'a Matrix,
    pub occupations: &'a [Vec<f64>; 2],
    pub fock: &'a [Matrix; 2],
    pub pairs: &'a [(usize, usize)],
    pub kernel: &'a ResponseSolver<'a>,
    pub kt: f64,
}

impl Jacobian<'_> {
    fn image(&self, x: &[f64]) -> Result<Vec<f64>> {
        let n = self.c.cols;
        let k = rotation(n, self.pairs, x, 1.0);
        // dmu removes the change in particle count separately in each spin.
        let df: [Vec<f64>; 2] = std::array::from_fn(|spin| {
            let levels = &x[self.pairs.len() + spin * n..][..n];
            let weight: Vec<_> = self.occupations[spin]
                .iter()
                .map(|f| f * (1.0 - f))
                .collect();
            let total: f64 = weight.iter().sum();
            if total == 0.0 {
                return vec![0.0; n];
            }
            let mu = weight.iter().zip(levels).map(|(w, e)| w * e).sum::<f64>() / total;
            weight
                .iter()
                .zip(levels)
                .map(|(w, e)| -w * ((e - mu) / self.kt))
                .collect()
        });
        // dP = C ([K,f] + diag(df)) C^T in the fixed AO frame.
        let density = std::array::from_fn(|spin| {
            let mut p = k.clone();
            for i in 0..n {
                for j in 0..n {
                    p[(i, j)] *= self.occupations[spin][j] - self.occupations[spin][i];
                }
                p[(i, i)] = df[spin][i];
            }
            self.c.matmul_seq(&p).matmul_transpose_seq(self.c)
        });
        let induced = self.kernel.fock_response(&density)?;
        let d_fock: [Matrix; 2] = std::array::from_fn(|spin| {
            let f = &self.fock[spin];
            let mut change = project(&induced[spin], self.c);
            super::add_scaled(&mut change, &f.matmul_seq(&k), 1.0);
            super::add_scaled(&mut change, &k.matmul_seq(f), -1.0);
            change
        });
        Ok(self
            .pairs
            .iter()
            .map(|&(p, q)| {
                (0..2)
                    .map(|spin| {
                        (df[spin][p] - df[spin][q]) * self.fock[spin][(p, q)]
                            + (self.occupations[spin][p] - self.occupations[spin][q])
                                * d_fock[spin][(p, q)]
                    })
                    .sum()
            })
            .chain(
                x[self.pairs.len()..]
                    .iter()
                    .enumerate()
                    .map(|(i, &level)| level - d_fock[i / n][(i % n, i % n)]),
            )
            .collect())
    }
}

impl LinearOperator for Jacobian<'_> {
    fn dimension(&self) -> usize {
        self.pairs.len() + 2 * self.c.cols
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| self.image(v))
    }
}

pub(super) fn solve(
    model: &MolecularDensityModel<'_>,
    mut result: ScfResult,
) -> Result<(ScfResult, Matrix)> {
    let c = result.mo_coeff[0].clone();
    let levels = result.fock.each_ref().map(|f| diagonal(&project(f, &c)));
    let counts = [result.n_alpha, result.n_beta];
    let opts = model.opts;
    let mut state = State::new(model, c, levels, counts)?;
    let mut last_change = 0.0_f64;
    for iteration in 1..=opts.max_iter {
        let pairs = state.pairs();
        let residual = state.residual(&pairs);
        let maximum = |entries: &[f64]| entries.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let error = maximum(&residual);
        crate::logger::log(&format!(
            "[roks-scf] iter {iteration:>3} A={:.12} dA={last_change:.3e} residual={error:.3e}",
            state.a + result.e_nuc
        ));
        if error <= opts.conv_tol_grad {
            if last_change <= opts.e_tol {
                state.populate(&mut result);
                result.converged = true;
                result.iterations = iteration;
                result.common_orbital_residual = Some(maximum(&residual[..pairs.len()]));
                return Ok((result, state.c));
            }
            // Check the energy again at the stationary state. The step that
            // reached stationarity may still have had a large energy change.
            let next = State::new(model, state.c.clone(), state.levels.clone(), counts)?;
            last_change = (state.a - next.a).abs();
            state = next;
            continue;
        }
        let diagonal: Vec<f64> = pairs
            .iter()
            .map(|&(p, q)| {
                (0..2)
                    .map(|s| {
                        (state.f[s][p] - state.f[s][q])
                            * (state.fock[s][(q, q)] - state.fock[s][(p, p)])
                    })
                    .sum::<f64>()
                    .abs()
                    .max(1e-4)
            })
            .chain(std::iter::repeat_n(1.0, 2 * state.c.cols))
            .collect();
        state.populate(&mut result);
        let newton = match ResponseSolver::kernel_with_engines(
            model.mol,
            model.mb,
            model.functional,
            model.grid,
            &result,
            opts,
            model.jk,
            model.jk_lr,
        ) {
            Ok(kernel) => {
                let jac = Jacobian {
                    c: &state.c,
                    occupations: &state.f,
                    fock: &state.fock,
                    pairs: &pairs,
                    kernel: &kernel,
                    kt: opts.electronic_temperature,
                };
                let rhs: Vec<_> = residual.iter().map(|v| Complex::new(-v, 0.0)).collect();
                let config = ResponseSolverConfig {
                    tolerance: 1e-4,
                    ..Default::default()
                };
                match krylov::solve(&jac, &rhs, Some(&diagonal), &config) {
                    Ok(step) => Some(
                        step.amplitudes
                            .into_iter()
                            .map(|x| x.re)
                            .collect::<Vec<_>>(),
                    ),
                    Err(DftError::ResponseNotConverged { .. }) => None,
                    Err(error) => return Err(error),
                }
            }
            Err(DftError::Unsupported { .. } | DftError::DerivOrderUnsupported { .. }) => None,
            Err(error) => return Err(error),
        };
        let descent = residual
            .iter()
            .zip(&diagonal)
            .map(|(r, d)| -r / d)
            .collect();
        let arithmetic = 16.0 * f64::EPSILON * state.a.abs().max(1.0);
        let mut accepted = None;
        for step in newton.into_iter().chain(std::iter::once(descent)) {
            let length = step[..pairs.len()]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            let mut fraction = if length > opts.soscf_trust {
                opts.soscf_trust / length
            } else {
                1.0
            };
            for _ in 0..24 {
                let trial = state.trial(&step, &pairs, fraction, model, counts)?;
                let trial_error = maximum(&trial.residual(&pairs));
                if trial.a < state.a - arithmetic
                    || (trial.a <= state.a + arithmetic && trial_error < error)
                {
                    accepted = Some(trial);
                    break;
                }
                fraction *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }
        let Some(next) = accepted else {
            return Err(DftError::ScfNotConverged {
                iterations: iteration,
                error,
            });
        };
        last_change = (state.a - next.a).abs();
        state = next;
    }
    let error = state
        .residual(&state.pairs())
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max);
    Err(DftError::ScfNotConverged {
        iterations: opts.max_iter,
        error,
    })
}
