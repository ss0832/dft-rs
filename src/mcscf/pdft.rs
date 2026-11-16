// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Multiconfiguration pair-density functional theory.
//!
//! The energy of a multiconfigurational state keeps its one-electron energy
//! and replaces its two-electron energy by the classical Coulomb energy of
//! its density and an on-top functional of the density and the on-top pair
//! density:
//!
//! ```text
//! E = V_nn + Tr[h P] + 1/2 Tr[P J[P]] + E_ot[rho, Pi],
//! Pi(r) = rho_c^2/4 + rho_c rho_a/2 + 1/2 sum_tuvw Gamma_tuvw phi_t phi_u phi_v phi_w,
//! ```
//!
//! with the core density `rho_c`, the active density `rho_a` and the
//! spin-free active two-particle density matrix `Gamma`. An on-top
//! functional is a registered local or gradient-corrected exchange-
//! correlation functional evaluated at translated spin densities: with
//! `R = 4 Pi / rho^2`, the translated polarization is
//! `zeta = (1 - R)^(1/2)` for `R < 1` and zero otherwise, and
//! `rho_alpha,beta = rho (1 +- zeta)/2`,
//! `grad rho_alpha,beta = (1 +- zeta)/2 grad rho`. The fully translated form
//! joins `(1 - R)^(1/2)` (below `R0 = 0.9`) and zero (above `R1 = 1.15`) by
//! `A x^5 + B x^4 + C x^3`, `x = R - R1`, continuous with its first two
//! derivatives, and adds `+- rho/2 grad zeta` to the spin-density gradients.
//! A hybrid fraction `lambda` mixes `lambda E_CAS + (1 - lambda) E_PDFT`.
//!
//! Several states: state-specific energies of the state-averaged reference;
//! linearized PDFT, whose Hamiltonian expands the classical Coulomb and
//! on-top energies to first order about the state-averaged density and is
//! diagonalized in the model space of the reference states; and the
//! compressed and extended
//! multistate forms, which rotate the reference states into intermediate
//! states (maximal sum of classical Coulomb self-energies of the active
//! densities, or the eigenstates of the state-averaged Fock operator) and
//! diagonalize the matrix with PDFT energies on the diagonal and the
//! active-space Hamiltonian couplings off the diagonal.

use super::casscf::{coulomb_exchange, CasPoint, CasProblem};
use super::ci::ActiveHamiltonian;
use crate::basis::MolBasis;
use crate::dual::{DualN, Scalar};
use crate::grid::eval::eval_aos;
use crate::grid::MolecularGrid;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::xc::{resolve, Family, FunctionalSpec, GridVars, ResolvedFunctional, XcOut};
use crate::{DftError, Result};
use rayon::prelude::*;

/// Translation of a spin-density functional to the density and the on-top
/// pair density.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Translation {
    Translated,
    FullyTranslated,
}

const R0: f64 = 0.9;
const R1: f64 = 1.15;
const FT_A: f64 = -475.60656009;
const FT_B: f64 = -379.47331922;
const FT_C: f64 = -85.38149682;
/// Densities below this value give no on-top energy.
const DENSITY_FLOOR: f64 = 1e-14;

/// Hybrid on-top functionals with conventional names: name, translation,
/// semilocal parent and hybrid fraction.
const HYBRID_NAMES: [(&str, Translation, &str, f64); 2] = [
    ("tpbe0", Translation::Translated, "pbe", 0.25),
    ("ftpbe0", Translation::FullyTranslated, "pbe", 0.25),
];

/// A translated on-top functional with an optional hybrid fraction.
pub struct OnTopFunctional {
    pub name: String,
    pub translation: Translation,
    pub base: ResolvedFunctional,
    pub lambda: f64,
}

impl OnTopFunctional {
    /// `t<name>` or `ft<name>` for a registered local or gradient-corrected
    /// functional `<name>`, or one of the conventional hybrid names;
    /// `lambda` overrides the hybrid fraction.
    pub fn parse(name: &str, lambda: Option<f64>) -> Result<Self> {
        let lower = name.trim().to_ascii_lowercase();
        let (translation, parent, fraction) =
            if let Some((_, t, parent, f)) = HYBRID_NAMES.iter().find(|(n, ..)| *n == lower) {
                (*t, parent.to_string(), *f)
            } else if let Some(rest) = lower.strip_prefix("ft") {
                (Translation::FullyTranslated, rest.to_string(), 0.0)
            } else if let Some(rest) = lower.strip_prefix('t') {
                (Translation::Translated, rest.to_string(), 0.0)
            } else {
                return Err(DftError::InvalidInput(format!(
                    "on-top functional `{name}`: write t<functional> (translated) or \
                     ft<functional> (fully translated) for a registered local or \
                     gradient-corrected functional, or tpbe0 / ftpbe0"
                )));
            };
        let base = resolve(&FunctionalSpec::Combined(parent.clone()))?;
        if base.family == Family::MetaGga {
            return Err(DftError::Unsupported {
                feature: format!("on-top translation of the meta-GGA `{parent}`"),
                missing: "a translation of the kinetic-energy density".into(),
            });
        }
        if base.hybrid.cx != 0.0
            || base.hybrid.clr != 0.0
            || base.required_nonlocal_kind().is_some()
            || base.parts.is_empty()
        {
            return Err(DftError::InvalidInput(format!(
                "the parent `{parent}` of an on-top functional must be a pure local or \
                 gradient-corrected functional; give hybrids through pdft_lambda"
            )));
        }
        let lambda = lambda.unwrap_or(fraction);
        if !(0.0..1.0).contains(&lambda) {
            return Err(DftError::InvalidInput(
                "the hybrid fraction pdft_lambda lies in [0, 1)".into(),
            ));
        }
        Ok(Self {
            name: lower,
            translation,
            base,
            lambda,
        })
    }
}

/// Translated `[rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb]` of
/// `(rho, Pi, grad rho, grad Pi)`.
pub fn translate<S: Scalar>(
    translation: Translation,
    rho: S,
    pi: S,
    grad_rho: [S; 3],
    grad_pi: [S; 3],
) -> [S; 5] {
    let raw = S::from_f64(4.0) * pi / (rho * rho);
    let ratio = if raw.value() < 0.0 {
        S::from_f64(0.0)
    } else {
        raw
    };
    let one = S::from_f64(1.0);
    let r = ratio.value();
    let (zeta, slope) = match translation {
        Translation::Translated => {
            if r < 1.0 {
                ((one - ratio).sqrt(), S::from_f64(0.0))
            } else {
                (S::from_f64(0.0), S::from_f64(0.0))
            }
        }
        Translation::FullyTranslated => {
            if r < R0 {
                let root = (one - ratio).sqrt();
                (root, -S::from_f64(0.5) / root)
            } else if r <= R1 {
                let x = ratio - S::from_f64(R1);
                let x2 = x * x;
                (
                    x2 * x * (S::from_f64(FT_A) * x2 + S::from_f64(FT_B) * x + S::from_f64(FT_C)),
                    x2 * (S::from_f64(5.0 * FT_A) * x2
                        + S::from_f64(4.0 * FT_B) * x
                        + S::from_f64(3.0 * FT_C)),
                )
            } else {
                (S::from_f64(0.0), S::from_f64(0.0))
            }
        }
    };
    let half = S::from_f64(0.5);
    let up = half * (one + zeta);
    let down = half * (one - zeta);
    let mut grad_up = [S::from_f64(0.0); 3];
    let mut grad_down = [S::from_f64(0.0); 3];
    for x in 0..3 {
        grad_up[x] = up * grad_rho[x];
        grad_down[x] = down * grad_rho[x];
        if translation == Translation::FullyTranslated {
            // grad zeta = zeta'(R) grad R, grad R = 4 grad Pi/rho^2 - 2 R grad rho/rho.
            let grad_r = (S::from_f64(4.0) * grad_pi[x]
                - S::from_f64(2.0) * ratio * rho * grad_rho[x])
                / (rho * rho);
            let shift = half * rho * slope * grad_r;
            grad_up[x] = grad_up[x] + shift;
            grad_down[x] = grad_down[x] - shift;
        }
    }
    let dot = |a: [S; 3], b: [S; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    [
        rho * up,
        rho * down,
        dot(grad_up, grad_up),
        dot(grad_up, grad_down),
        dot(grad_down, grad_down),
    ]
}

/// Density matrices of one state for the on-top evaluation.
pub struct OnTopDensity<'a> {
    /// Core density `2 C_c C_c^T`, atomic-orbital basis.
    pub core: &'a Matrix,
    /// Active orbitals, atomic orbitals by active orbitals.
    pub active: &'a Matrix,
    pub gamma: &'a Matrix,
    pub gamma2: &'a [f64],
}

/// Derivatives of the on-top energy with respect to the active density
/// matrices at fixed core: `dE/dgamma_tu` and `dE/dGamma_tuvw`.
#[derive(Clone, Debug)]
pub struct OnTopPotential {
    pub one: Matrix,
    pub two: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct OnTopEnergy {
    pub energy: f64,
    /// Integrated density (electron count of the grid).
    pub electrons: f64,
    /// Integrated on-top pair density.
    pub on_top_pairs: f64,
    pub potential: Option<OnTopPotential>,
}

/// Grid points per batch.
const BATCH: usize = 256;

/// Symmetrize a two-particle density matrix over the index permutations of
/// real orbitals: `(tu vw)`, `(vw tu)`, `(ut wv)`, `(wv ut)`.
fn symmetrized(n: usize, gamma2: &[f64]) -> Matrix {
    let index = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
    let mut out = Matrix::zeros(n * n, n * n);
    for t in 0..n {
        for u in 0..n {
            for v in 0..n {
                for w in 0..n {
                    out[(t * n + u, v * n + w)] = 0.25
                        * (gamma2[index(t, u, v, w)]
                            + gamma2[index(v, w, t, u)]
                            + gamma2[index(u, t, w, v)]
                            + gamma2[index(w, v, u, t)]);
                }
            }
        }
    }
    out
}

/// On-top energy of one state, with its active-density potential when
/// `potential` is set.
pub fn on_top_energy(
    functional: &OnTopFunctional,
    mb: &MolBasis,
    grid: &MolecularGrid,
    density: &OnTopDensity,
    potential: bool,
) -> Result<OnTopEnergy> {
    let n = density.active.cols;
    let nao = mb.nao_sph;
    let gga = functional.base.family != Family::Lda;
    let pair = symmetrized(n, density.gamma2);
    let ranges: Vec<(usize, usize)> = (0..grid.len())
        .step_by(BATCH)
        .map(|s| (s, (s + BATCH).min(grid.len())))
        .collect();
    type Partial = (f64, f64, f64, Option<(Matrix, Matrix)>);
    let partials: Result<Vec<Partial>> = ranges
        .par_iter()
        .map(|&(start, end)| -> Result<Partial> {
            let points = &grid.points[start..end];
            let weights = &grid.weights[start..end];
            let npts = points.len();
            let batch = eval_aos(mb, points, gga);
            let phi_core = batch.phi.matmul_seq(density.core);
            let act = batch.phi.matmul_seq(density.active);
            let act_gamma = act.matmul_seq(density.gamma);
            let grads: Option<[Matrix; 3]> = batch
                .grad
                .as_ref()
                .map(|g| std::array::from_fn(|x| g[x].matmul_seq(density.active)));
            let mut q = Matrix::zeros(npts, n * n);
            for i in 0..npts {
                for t in 0..n {
                    for u in 0..n {
                        q[(i, t * n + u)] = act[(i, t)] * act[(i, u)];
                    }
                }
            }
            let pair_q = q.matmul_seq(&pair);
            let mut rho = vec![0.0; npts];
            let mut pi = vec![0.0; npts];
            let mut rho_core = vec![0.0; npts];
            let mut grad_rho = vec![[0.0; 3]; npts];
            let mut grad_pi = vec![[0.0; 3]; npts];
            let mut grad_core = vec![[0.0; 3]; npts];
            for i in 0..npts {
                let mut rc = 0.0;
                for mu in 0..nao {
                    rc += phi_core[(i, mu)] * batch.phi[(i, mu)];
                }
                let mut ra = 0.0;
                for t in 0..n {
                    ra += act_gamma[(i, t)] * act[(i, t)];
                }
                let mut paa = 0.0;
                for tu in 0..n * n {
                    paa += q[(i, tu)] * pair_q[(i, tu)];
                }
                paa *= 0.5;
                rho_core[i] = rc;
                rho[i] = rc + ra;
                pi[i] = 0.25 * rc * rc + 0.5 * rc * ra + paa;
                if let (Some(g), Some(ga)) = (batch.grad.as_ref(), grads.as_ref()) {
                    for x in 0..3 {
                        let mut gc = 0.0;
                        for mu in 0..nao {
                            gc += 2.0 * phi_core[(i, mu)] * g[x][(i, mu)];
                        }
                        let mut gra = 0.0;
                        let mut gpa = 0.0;
                        for t in 0..n {
                            gra += 2.0 * act_gamma[(i, t)] * ga[x][(i, t)];
                            for u in 0..n {
                                gpa += 2.0 * ga[x][(i, t)] * act[(i, u)] * pair_q[(i, t * n + u)];
                            }
                        }
                        grad_core[i][x] = gc;
                        grad_rho[i][x] = gc + gra;
                        grad_pi[i][x] = 0.5 * rc * gc + 0.5 * (gc * ra + rc * gra) + gpa;
                    }
                }
            }
            // Translated spin variables.
            let mut vars = vec![[0.0; 5]; npts];
            let mut jacobians = vec![[[0.0; 8]; 5]; npts];
            for i in 0..npts {
                if rho[i] < DENSITY_FLOOR {
                    continue;
                }
                if potential {
                    type D = DualN<f64, 8>;
                    let out = translate(
                        functional.translation,
                        D::variable(rho[i], 0),
                        D::variable(pi[i], 1),
                        std::array::from_fn(|x| D::variable(grad_rho[i][x], 2 + x)),
                        std::array::from_fn(|x| D::variable(grad_pi[i][x], 5 + x)),
                    );
                    for (k, value) in out.iter().enumerate() {
                        vars[i][k] = value.re;
                        jacobians[i][k] = value.im;
                    }
                } else {
                    vars[i] = translate(
                        functional.translation,
                        rho[i],
                        pi[i],
                        grad_rho[i],
                        grad_pi[i],
                    );
                }
            }
            let column = |k: usize| -> Vec<f64> { vars.iter().map(|v| v[k].max(0.0)).collect() };
            let (ra, rb) = (column(0), column(1));
            let sigma: Vec<Vec<f64>> = (2..5)
                .map(|k| vars.iter().map(|v| v[k]).collect())
                .collect();
            let grid_vars = GridVars {
                rho: [&ra, &rb],
                sigma: if gga {
                    Some([&sigma[0], &sigma[1], &sigma[2]])
                } else {
                    None
                },
                tau: None,
            };
            let mut out = XcOut::default();
            functional
                .base
                .eval(&grid_vars, usize::from(potential), &mut out)?;
            let mut energy = 0.0;
            let mut electrons = 0.0;
            let mut pairs = 0.0;
            for i in 0..npts {
                electrons += weights[i] * rho[i];
                pairs += weights[i] * pi[i];
                if rho[i] >= DENSITY_FLOOR {
                    energy += weights[i] * out.e[i];
                }
            }
            if !potential {
                return Ok((energy, electrons, pairs, None));
            }
            // dE/d(rho, Pi, grad rho, grad Pi) at every point.
            let mut a = vec![0.0; npts];
            let mut b = vec![[0.0; 3]; npts];
            let mut c = vec![0.0; npts];
            let mut d = vec![[0.0; 3]; npts];
            for i in 0..npts {
                if rho[i] < DENSITY_FLOOR {
                    continue;
                }
                let mut de = [out.vrho[0][i], out.vrho[1][i], 0.0, 0.0, 0.0];
                if gga {
                    de[2] = out.vsigma[0][i];
                    de[3] = out.vsigma[1][i];
                    de[4] = out.vsigma[2][i];
                }
                let mut total = [0.0; 8];
                for (k, dk) in de.iter().enumerate() {
                    for (t, j) in total.iter_mut().zip(&jacobians[i][k]) {
                        *t += dk * j;
                    }
                }
                let w = weights[i];
                let v_pi = total[1];
                let v_grad_pi = [total[5], total[6], total[7]];
                let core_coupling: f64 = (0..3).map(|x| v_grad_pi[x] * grad_core[i][x]).sum();
                a[i] = w * (total[0] + 0.5 * v_pi * rho_core[i] + 0.5 * core_coupling);
                for x in 0..3 {
                    b[i][x] = w * (total[2 + x] + 0.5 * v_grad_pi[x] * rho_core[i]);
                    d[i][x] = w * v_grad_pi[x];
                }
                c[i] = w * v_pi;
            }
            // One-particle part: act^T diag(a) act + B^T act + act^T B.
            let mut scaled = act.clone();
            for i in 0..npts {
                for t in 0..n {
                    scaled[(i, t)] *= a[i];
                }
            }
            let mut one = act.transpose_matmul_seq(&scaled);
            if let Some(ga) = grads.as_ref() {
                let mut bmat = Matrix::zeros(npts, n);
                for i in 0..npts {
                    for t in 0..n {
                        bmat[(i, t)] = (0..3).map(|x| b[i][x] * ga[x][(i, t)]).sum();
                    }
                }
                let cross = bmat.transpose_matmul_seq(&act);
                for t in 0..n {
                    for u in 0..n {
                        one[(t, u)] += cross[(t, u)] + cross[(u, t)];
                    }
                }
            }
            // Two-particle part: 1/2 [Q^T diag(c) Q + M^T Q + Q^T M].
            let mut cq = q.clone();
            for i in 0..npts {
                for tu in 0..n * n {
                    cq[(i, tu)] *= c[i];
                }
            }
            let mut two = q.transpose_matmul_seq(&cq);
            if let Some(ga) = grads.as_ref() {
                let mut m = Matrix::zeros(npts, n * n);
                for i in 0..npts {
                    for t in 0..n {
                        for u in 0..n {
                            m[(i, t * n + u)] = (0..3)
                                .map(|x| {
                                    d[i][x]
                                        * (ga[x][(i, t)] * act[(i, u)]
                                            + act[(i, t)] * ga[x][(i, u)])
                                })
                                .sum();
                        }
                    }
                }
                let cross = m.transpose_matmul_seq(&q);
                for r in 0..n * n {
                    for s in 0..n * n {
                        two[(r, s)] += cross[(r, s)] + cross[(s, r)];
                    }
                }
            }
            two.as_mut_slice().iter_mut().for_each(|v| *v *= 0.5);
            Ok((energy, electrons, pairs, Some((one, two))))
        })
        .collect();
    let mut energy = 0.0;
    let mut electrons = 0.0;
    let mut on_top_pairs = 0.0;
    let mut one = Matrix::zeros(n, n);
    let mut two = Matrix::zeros(n * n, n * n);
    for (e, el, p, pot) in partials? {
        energy += e;
        electrons += el;
        on_top_pairs += p;
        if let Some((o, t)) = pot {
            for (x, y) in one.as_mut_slice().iter_mut().zip(o.as_slice()) {
                *x += y;
            }
            for (x, y) in two.as_mut_slice().iter_mut().zip(t.as_slice()) {
                *x += y;
            }
        }
    }
    Ok(OnTopEnergy {
        energy,
        electrons,
        on_top_pairs,
        potential: potential.then(|| OnTopPotential {
            one,
            two: two.as_slice().to_vec(),
        }),
    })
}

/// Energy terms of one state.
#[derive(Clone, Debug)]
pub struct PdftEnergy {
    pub nuclear_repulsion: f64,
    pub one_electron: f64,
    pub coulomb: f64,
    pub on_top: f64,
    /// `V_nn + one_electron + coulomb + on_top`.
    pub pdft: f64,
    /// Active-space (CASSCF or CASCI) energy of the same state.
    pub reference: f64,
    /// `lambda reference + (1 - lambda) pdft`.
    pub total: f64,
    pub electrons: f64,
    pub on_top_pairs: f64,
}

/// Core density, active orbitals and the total density of `gamma`.
fn densities(problem: &CasProblem, orbitals: &Matrix, gamma: &Matrix) -> (Matrix, Matrix, Matrix) {
    let (nc, na) = (problem.spec.n_core, problem.spec.n_active);
    let take = |start: usize, count: usize| -> Matrix {
        let mut m = Matrix::zeros(orbitals.rows, count);
        for i in 0..orbitals.rows {
            for j in 0..count {
                m[(i, j)] = orbitals[(i, start + j)];
            }
        }
        m
    };
    let core = take(0, nc);
    let active = take(nc, na);
    let mut p_core = core.matmul_transpose_seq(&core);
    p_core.as_mut_slice().iter_mut().for_each(|v| *v *= 2.0);
    let mut total = active.matmul_seq(gamma).matmul_transpose_seq(&active);
    for (t, c) in total.as_mut_slice().iter_mut().zip(p_core.as_slice()) {
        *t += c;
    }
    (p_core, active, total)
}

/// PDFT energy of the state with active density matrices `gamma`, `gamma2`
/// and reference energy `reference` at the orbitals `orbitals`.
#[allow(clippy::too_many_arguments)]
pub fn state_energy(
    problem: &CasProblem,
    functional: &OnTopFunctional,
    mb: &MolBasis,
    grid: &MolecularGrid,
    orbitals: &Matrix,
    gamma: &Matrix,
    gamma2: &[f64],
    reference: f64,
) -> Result<PdftEnergy> {
    let (p_core, active, total) = densities(problem, orbitals, gamma);
    let (j, _) = coulomb_exchange(problem.operators.eri, &total);
    let one_electron = total.frobenius_dot(problem.operators.h_core);
    let coulomb = 0.5 * total.frobenius_dot(&j);
    let ot = on_top_energy(
        functional,
        mb,
        grid,
        &OnTopDensity {
            core: &p_core,
            active: &active,
            gamma,
            gamma2,
        },
        false,
    )?;
    let pdft = problem.operators.e_nuc + one_electron + coulomb + ot.energy;
    Ok(PdftEnergy {
        nuclear_repulsion: problem.operators.e_nuc,
        one_electron,
        coulomb,
        on_top: ot.energy,
        pdft,
        reference,
        total: functional.lambda * reference + (1.0 - functional.lambda) * pdft,
        electrons: ot.electrons,
        on_top_pairs: ot.on_top_pairs,
    })
}

/// Multistate PDFT methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiState {
    /// State-specific energies of the reference states.
    StateSpecific,
    /// Linearized PDFT Hamiltonian diagonalized in the active space.
    Linearized,
    /// Intermediate states of maximal classical Coulomb self-energy.
    Compressed,
    /// Intermediate states that diagonalize the state-averaged Fock operator.
    Extended,
}

impl MultiState {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name.trim().to_ascii_lowercase().as_str() {
            "state-specific" => Self::StateSpecific,
            "l-pdft" => Self::Linearized,
            "cms-pdft" => Self::Compressed,
            "xms-pdft" => Self::Extended,
            other => {
                return Err(DftError::InvalidInput(format!(
                    "unknown pdft_method `{other}` (state-specific, l-pdft, cms-pdft, xms-pdft)"
                )))
            }
        })
    }
}

/// Result of a multistate PDFT calculation.
#[derive(Clone, Debug)]
pub struct MultiStateResult {
    pub method: MultiState,
    /// Final energies, ascending, Hartree.
    pub energies: Vec<f64>,
    /// Energy terms of the reference or intermediate states.
    pub state_terms: Vec<PdftEnergy>,
    /// Rotation from the reference states to the intermediate states
    /// (columns), identity for the state-specific and linearized forms.
    pub rotation: Matrix,
    /// Effective Hamiltonian in the intermediate basis (diagonal for the
    /// state-specific form; for the linearized form, in the reference basis).
    pub hamiltonian: Matrix,
    /// Eigenvectors of the effective Hamiltonian (columns).
    pub eigenvectors: Matrix,
    /// Linearized form: PDFT energy at the expansion point and the energy
    /// of the expansion density under the linearized Hamiltonian.
    pub expansion_energy: Option<(f64, f64)>,
    /// Compressed form: maximized sum of Coulomb self-energies and sweeps.
    pub objective: Option<(f64, usize)>,
}

/// Coulomb bilinear form in the active orbital basis.
fn coulomb_inner(g: &[f64], left: &Matrix, right: &Matrix) -> f64 {
    let n = left.rows;
    let mut e = 0.0;
    for t in 0..n {
        for u in 0..n {
            for v in 0..n {
                for w in 0..n {
                    e += left[(t, u)] * right[(v, w)] * g[((t * n + u) * n + v) * n + w];
                }
            }
        }
    }
    e
}

fn compressed_pair_angle(g: &[f64], difference: &Matrix, transition: &Matrix) -> f64 {
    // The two rotated densities are A +/- [B cos(2 theta) + C sin(2 theta)].
    // Their sum of self energies therefore varies only with 4 theta.
    let bb = coulomb_inner(g, difference, difference);
    let cc = coulomb_inner(g, transition, transition);
    let sin = 2.0 * coulomb_inner(g, difference, transition);
    let cos = bb - cc;
    // A flat pair has no preferred rotation in the current subspace.
    if sin.hypot(cos) <= 16.0 * f64::EPSILON * (bb.abs() + cc.abs()) {
        return 0.0;
    }
    0.25 * sin.atan2(cos)
}

fn rotated_vectors(vectors: &[Vec<f64>], rotation: &Matrix) -> Vec<Vec<f64>> {
    let k = vectors.len();
    (0..k)
        .map(|col| {
            let mut v = vec![0.0; vectors[0].len()];
            for (row, source) in vectors.iter().enumerate() {
                for (x, y) in v.iter_mut().zip(source) {
                    *x += rotation[(row, col)] * y;
                }
            }
            v
        })
        .collect()
}

/// Multistate PDFT on the converged (state-averaged) reference `point`.
pub fn multistate(
    problem: &CasProblem,
    point: &CasPoint,
    functional: &OnTopFunctional,
    mb: &MolBasis,
    grid: &MolecularGrid,
    method: MultiState,
) -> Result<MultiStateResult> {
    let k = point.states.vectors.len();
    let space = &problem.space;
    let lambda = functional.lambda;
    let terms_of = |vectors: &[Vec<f64>], references: &[f64]| -> Result<Vec<PdftEnergy>> {
        vectors
            .iter()
            .zip(references)
            .map(|(v, &reference)| {
                let (g1, g2) = space.rdm12(v, v)?;
                state_energy(
                    problem,
                    functional,
                    mb,
                    grid,
                    &point.orbitals,
                    &g1,
                    &g2,
                    reference,
                )
            })
            .collect()
    };
    match method {
        MultiState::StateSpecific => {
            let terms = terms_of(&point.states.vectors, &point.energies)?;
            let mut hamiltonian = Matrix::zeros(k, k);
            for (i, t) in terms.iter().enumerate() {
                hamiltonian[(i, i)] = t.total;
            }
            let (energies, eigenvectors) = symmetric_eigen(&hamiltonian)?;
            Ok(MultiStateResult {
                method,
                energies,
                state_terms: terms,
                rotation: Matrix::identity(k),
                hamiltonian,
                eigenvectors,
                expansion_energy: None,
                objective: None,
            })
        }
        MultiState::Linearized => {
            let terms = terms_of(&point.states.vectors, &point.energies)?;
            let (p_core, active, p0) = densities(problem, &point.orbitals, &point.gamma);
            let (j0, _) = coulomb_exchange(problem.operators.eri, &p0);
            let ot = on_top_energy(
                functional,
                mb,
                grid,
                &OnTopDensity {
                    core: &p_core,
                    active: &active,
                    gamma: &point.gamma,
                    gamma2: &point.gamma2,
                },
                true,
            )?;
            let potential = ot.potential.expect("potential requested");
            let h_core = problem.operators.h_core;
            let h_act = active.transpose_matmul_seq(&h_core.matmul_seq(&active));
            let j_act = active.transpose_matmul_seq(&j0.matmul_seq(&active));
            let n = problem.spec.n_active;
            let mut h1 = Matrix::zeros(n, n);
            for t in 0..n {
                for u in 0..n {
                    h1[(t, u)] = h_act[(t, u)] + j_act[(t, u)] + potential.one[(t, u)];
                }
            }
            let linear_two: f64 = potential
                .two
                .iter()
                .zip(&point.gamma2)
                .map(|(a, b)| a * b)
                .sum();
            let expansion = problem.operators.e_nuc
                + p0.frobenius_dot(h_core)
                + 0.5 * p0.frobenius_dot(&j0)
                + ot.energy;
            // E_L(gamma, Gamma) = constant + sum h1 gamma + sum V2 Gamma equals
            // the PDFT energy at the expansion point (gamma0, Gamma0).
            let constant = expansion - h1.frobenius_dot(&point.gamma) - linear_two;
            let linearized = ActiveHamiltonian {
                n,
                constant,
                h: h1,
                g: potential.two.iter().map(|v| 2.0 * v).collect(),
            };
            let reference = &point.hamiltonian;
            let mixed = ActiveHamiltonian {
                n,
                constant: lambda * reference.constant + (1.0 - lambda) * linearized.constant,
                h: {
                    let mut h = linearized.h.clone();
                    for (x, y) in h.as_mut_slice().iter_mut().zip(reference.h.as_slice()) {
                        *x = lambda * y + (1.0 - lambda) * *x;
                    }
                    h
                },
                g: linearized
                    .g
                    .iter()
                    .zip(&reference.g)
                    .map(|(l, r)| lambda * r + (1.0 - lambda) * l)
                    .collect(),
            };
            let at_expansion = mixed.constant
                + mixed.h.frobenius_dot(&point.gamma)
                + 0.5
                    * mixed
                        .g
                        .iter()
                        .zip(&point.gamma2)
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
            let average_reference = point.average_energy;
            // The linearized Hamiltonian in the model space of the reference
            // states; with one state it is the PDFT energy of that state.
            let mut hamiltonian = Matrix::zeros(k, k);
            for i in 0..k {
                for jx in 0..k {
                    let hv = space.sigma(&mixed, &point.states.vectors[jx])?;
                    hamiltonian[(i, jx)] = point.states.vectors[i]
                        .iter()
                        .zip(&hv)
                        .map(|(a, b)| a * b)
                        .sum();
                }
            }
            for i in 0..k {
                for jx in 0..i {
                    let average = 0.5 * (hamiltonian[(i, jx)] + hamiltonian[(jx, i)]);
                    hamiltonian[(i, jx)] = average;
                    hamiltonian[(jx, i)] = average;
                }
            }
            let (energies, eigenvectors) = symmetric_eigen(&hamiltonian)?;
            Ok(MultiStateResult {
                method,
                energies,
                state_terms: terms,
                rotation: Matrix::identity(k),
                hamiltonian,
                eigenvectors,
                expansion_energy: Some((
                    lambda * average_reference + (1.0 - lambda) * expansion,
                    at_expansion,
                )),
                objective: None,
            })
        }
        MultiState::Compressed | MultiState::Extended => {
            let n = problem.spec.n_active;
            let mut transition = vec![vec![Matrix::zeros(n, n); k]; k];
            for i in 0..k {
                for jx in 0..k {
                    transition[i][jx] = space
                        .rdm12(&point.states.vectors[i], &point.states.vectors[jx])?
                        .0;
                }
            }
            let gamma_of = |rotation: &Matrix, bra: usize, ket: usize| -> Matrix {
                let mut g = Matrix::zeros(n, n);
                for i in 0..k {
                    for jx in 0..k {
                        let f = rotation[(i, bra)] * rotation[(jx, ket)];
                        if f != 0.0 {
                            for (x, y) in g
                                .as_mut_slice()
                                .iter_mut()
                                .zip(transition[i][jx].as_slice())
                            {
                                *x += f * y;
                            }
                        }
                    }
                }
                g
            };
            let (rotation, objective) = if method == MultiState::Compressed {
                let g = &point.hamiltonian.g;
                let mut rotation = Matrix::identity(k);
                let total = |rotation: &Matrix| -> f64 {
                    (0..k)
                        .map(|c| {
                            let density = gamma_of(rotation, c, c);
                            0.5 * coulomb_inner(g, &density, &density)
                        })
                        .sum()
                };
                let mut sweeps = 0;
                loop {
                    sweeps += 1;
                    let before = total(&rotation);
                    for a in 0..k {
                        for b in (a + 1)..k {
                            let mut difference = gamma_of(&rotation, a, a);
                            let other = gamma_of(&rotation, b, b);
                            for (x, y) in difference.as_mut_slice().iter_mut().zip(other.as_slice()) {
                                *x = 0.5 * (*x - y);
                            }
                            let mut cross = gamma_of(&rotation, a, b);
                            for i in 0..n {
                                for j in 0..i {
                                    let value = 0.5 * (cross[(i, j)] + cross[(j, i)]);
                                    cross[(i, j)] = value;
                                    cross[(j, i)] = value;
                                }
                            }
                            let theta = compressed_pair_angle(g, &difference, &cross);
                            if theta.abs() > 16.0 * f64::EPSILON {
                                let (s, c) = theta.sin_cos();
                                for row in 0..k {
                                    let (x, y) = (rotation[(row, a)], rotation[(row, b)]);
                                    rotation[(row, a)] = c * x + s * y;
                                    rotation[(row, b)] = -s * x + c * y;
                                }
                            }
                        }
                    }
                    let after = total(&rotation);
                    if (after - before).abs() < 1e-12 || sweeps >= 200 || k < 2 {
                        break (rotation, Some((after, sweeps)));
                    }
                }
            } else {
                // State-averaged Fock operator in the reference states.
                let (_, active, _) = densities(problem, &point.orbitals, &point.gamma);
                let p_active = active
                    .matmul_seq(&point.gamma)
                    .matmul_transpose_seq(&active);
                let (ja, ka) = coulomb_exchange(problem.operators.eri, &p_active);
                let mut fock = point.core_fock.clone();
                for ((f, j), kv) in fock
                    .as_mut_slice()
                    .iter_mut()
                    .zip(ja.as_slice())
                    .zip(ka.as_slice())
                {
                    *f += j - 0.5 * kv;
                }
                let f_act = active.transpose_matmul_seq(&fock.matmul_seq(&active));
                let mut f_states = Matrix::zeros(k, k);
                for i in 0..k {
                    for jx in 0..k {
                        f_states[(i, jx)] = 0.5
                            * (transition[i][jx].frobenius_dot(&f_act)
                                + transition[jx][i].frobenius_dot(&f_act));
                    }
                }
                (symmetric_eigen(&f_states)?.1, None)
            };
            let intermediate = rotated_vectors(&point.states.vectors, &rotation);
            let reference_energies: Vec<f64> = (0..k)
                .map(|c| {
                    (0..k)
                        .map(|i| rotation[(i, c)] * rotation[(i, c)] * point.energies[i])
                        .sum()
                })
                .collect();
            let terms = terms_of(&intermediate, &reference_energies)?;
            let mut hamiltonian = Matrix::zeros(k, k);
            for a in 0..k {
                for b in 0..k {
                    hamiltonian[(a, b)] = if a == b {
                        terms[a].total
                    } else {
                        (0..k)
                            .map(|i| rotation[(i, a)] * rotation[(i, b)] * point.energies[i])
                            .sum()
                    };
                }
            }
            let (energies, eigenvectors) = symmetric_eigen(&hamiltonian)?;
            Ok(MultiStateResult {
                method,
                energies,
                state_terms: terms,
                rotation,
                hamiltonian,
                eigenvectors,
                expansion_energy: None,
                objective,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fully_translated_polarization_is_smooth_at_both_joints() {
        let zeta = |r: f64| -> f64 {
            let rho = 1.0;
            translate(
                Translation::FullyTranslated,
                rho,
                0.25 * r,
                [0.0; 3],
                [0.0; 3],
            )[0] * 2.0
                - 1.0
        };
        for (joint, inside) in [(R0, R0 + 1e-9), (R1, R1 + 1e-9)] {
            let below = zeta(joint - 1e-9);
            let above = zeta(inside);
            assert!((below - above).abs() < 1e-6, "{joint}: {below} {above}");
        }
        assert!((zeta(0.5) - 0.5_f64.sqrt()).abs() < 1e-15);
        assert_eq!(zeta(1.3), 0.0);
    }

    #[test]
    fn translated_variables_of_a_determinant_are_its_spin_densities() {
        // Pi = rho_a rho_b for a single determinant.
        let (ra, rb) = (0.7, 0.2);
        let out = translate(
            Translation::Translated,
            ra + rb,
            ra * rb,
            [0.0; 3],
            [0.0; 3],
        );
        assert!((out[0] - ra).abs() < 1e-15 && (out[1] - rb).abs() < 1e-15);
    }
}
