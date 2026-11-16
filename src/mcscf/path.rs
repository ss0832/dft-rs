// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear Taylor series of the complete-active-space and MC-PDFT energies.
//!
//! Along the path `R(t) = R(0) + t d` the moving Gaussian integrals, the
//! moving Becke weights and the atomic-orbital values are Taylor series. The
//! reference orbitals are carried by the transported frame
//! `B(t) = C_0 (I + X(t))^(-1/2)`, `X = C_0^T S(t) C_0 - I`, and rotated by
//! `exp(K(t))` with an antisymmetric `K(t) = sum_n K_n t^n` on the
//! core-active, core-virtual and active-virtual pairs; the CI vector of
//! every state is `c(t) = sum_n c_n t^n`. At every order the unknown
//! coefficients follow from the stationarity conditions of the reference,
//! the (state-averaged) orbital gradient and the CI eigenvalue residual
//! `(H(t) - E(t)) c(t)` of every state, with the normalization fixing the
//! component of `c_n` along `c_0`:
//!
//! ```text
//! J [K_n, c_n] = -r_n,
//! ```
//!
//! `r_n` the order-`n` residual with zero order-`n` unknowns, and `J` the
//! Jacobian of the residuals at the reference, applied as the first-order
//! coefficient of the same residual series at fixed geometry. With the
//! solved series the active-space energy and the MC-PDFT energy of every
//! state are Taylor series: the classical Coulomb energy of the moving
//! density and the on-top functional of the moving density and on-top pair
//! density integrated with the moving weights. No energy is assumed
//! stationary in the orbitals, so the PDFT coefficients carry the complete
//! wavefunction response. Gradients are the first-order coefficients along
//! the Cartesian directions and Hessians follow from second-order
//! coefficients of the directions and their pair sums.

use super::casscf::{CasPoint, CasProblem};
use super::ci::ActiveHamiltonian;
use super::pdft::{translate, OnTopFunctional};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::response::nuclear::geometry::GeometryTaylor;
use crate::response::nuclear::metric_transport;
use crate::response::series::evaluate_series_many_result;
use crate::system::Molecule;
use crate::xc::{ResolvedFunctional, XcTaylorInput};
use crate::{DftError, Result};
use rayon::prelude::*;

type Series = Vec<Matrix>;

/// Cauchy product of two matrix series through `order`.
fn product(a: &[Matrix], b: &[Matrix], order: usize) -> Series {
    (0..=order)
        .map(|n| {
            let mut out = Matrix::zeros(a[0].rows, b[0].cols);
            for k in 0..=n {
                if k < a.len() && n - k < b.len() {
                    let term = a[k].matmul_seq(&b[n - k]);
                    for (o, t) in out.as_mut_slice().iter_mut().zip(term.as_slice()) {
                        *o += t;
                    }
                }
            }
            out
        })
        .collect()
}

fn transposed(a: &[Matrix]) -> Series {
    a.iter().map(Matrix::transpose).collect()
}

fn add_into(target: &mut Matrix, source: &Matrix, scale: f64) {
    for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *t += scale * s;
    }
}

fn columns(m: &Matrix, start: usize, count: usize) -> Matrix {
    let mut out = Matrix::zeros(m.rows, count);
    for i in 0..m.rows {
        for j in 0..count {
            out[(i, j)] = m[(i, start + j)];
        }
    }
    out
}

/// `J[P]` and `K[P]` of one Coulomb tensor.
fn jk(eri: &[f64], n: usize, p: &Matrix) -> (Matrix, Matrix) {
    let mut j = Matrix::zeros(n, n);
    let mut k = Matrix::zeros(n, n);
    for m in 0..n {
        for nu in 0..n {
            let base = (m * n + nu) * n * n;
            j[(m, nu)] = eri[base..base + n * n]
                .iter()
                .zip(p.as_slice())
                .map(|(a, b)| a * b)
                .sum();
        }
        for l in 0..n {
            let row = &p.as_slice()[l * n..(l + 1) * n];
            for nu in 0..n {
                let base = ((m * n + l) * n + nu) * n;
                k[(m, nu)] += eri[base..base + n]
                    .iter()
                    .zip(row)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            }
        }
    }
    (j, k)
}

/// Series of `J[P(t)]` and `K[P(t)]` with the moving tensor.
fn jk_series(eri: &[Vec<f64>], n: usize, p: &[Matrix], order: usize) -> (Series, Series) {
    let mut js = vec![Matrix::zeros(n, n); order + 1];
    let mut ks = js.clone();
    for (a, tensor) in eri.iter().enumerate().take(order + 1) {
        for (b, density) in p.iter().enumerate().take(order + 1 - a) {
            if density.as_slice().iter().all(|v| *v == 0.0) {
                continue;
            }
            let (j, k) = jk(tensor, n, density);
            add_into(&mut js[a + b], &j, 1.0);
            add_into(&mut ks[a + b], &k, 1.0);
        }
    }
    (js, ks)
}

/// `(q u|v w)` with the four orbital matrices `q, u, v, w`, at
/// `((q n + u) n + v) n + w`.
fn transform4(eri: &[f64], nao: usize, q: &Matrix, u: &Matrix, v: &Matrix, w: &Matrix) -> Vec<f64> {
    let (nq, n) = (q.cols, u.cols);
    let y: Vec<Vec<f64>> = (0..nao * nao)
        .into_par_iter()
        .map(|row| {
            let block = Matrix::from_row_major(
                nao,
                nao,
                eri[row * nao * nao..(row + 1) * nao * nao].to_vec(),
            );
            v.transpose_matmul_seq(&block.matmul_seq(w))
                .as_slice()
                .to_vec()
        })
        .collect();
    let mut z = Matrix::zeros(nao, n * n * n);
    for mu in 0..nao {
        for nu in 0..nao {
            let yr = &y[mu * nao + nu];
            for a in 0..n {
                let factor = u[(nu, a)];
                if factor == 0.0 {
                    continue;
                }
                let target = &mut z.as_mut_slice()[mu * n * n * n + a * n * n..][..n * n];
                for (t, value) in target.iter_mut().zip(yr) {
                    *t += factor * value;
                }
            }
        }
    }
    let out = q.transpose_matmul(&z);
    debug_assert_eq!(out.rows, nq);
    out.as_slice().to_vec()
}

/// Compositions of `n` into `parts` non-negative integers.
fn compositions(n: usize, parts: usize) -> Vec<Vec<usize>> {
    if parts == 1 {
        return vec![vec![n]];
    }
    let mut out = Vec::new();
    for first in 0..=n {
        for mut rest in compositions(n - first, parts - 1) {
            rest.insert(0, first);
            out.push(rest);
        }
    }
    out
}

/// Series of `(q u|v w)` for every orbital `q` of `c` and active `u, v, w`.
fn mixed_series(
    eri: &[Vec<f64>],
    nao: usize,
    c: &[Matrix],
    active: &[Matrix],
    order: usize,
) -> Vec<Vec<f64>> {
    (0..=order)
        .map(|n| {
            let size = c[0].cols * active[0].cols.pow(3);
            let mut out = vec![0.0; size];
            for parts in compositions(n, 5) {
                let (k, a, b, d, e) = (parts[0], parts[1], parts[2], parts[3], parts[4]);
                if k >= eri.len()
                    || a >= c.len()
                    || b >= active.len()
                    || d >= active.len()
                    || e >= active.len()
                {
                    continue;
                }
                let zero = |m: &Matrix| m.as_slice().iter().all(|v| *v == 0.0);
                if zero(&c[a]) || zero(&active[b]) || zero(&active[d]) || zero(&active[e]) {
                    continue;
                }
                let term = transform4(&eri[k], nao, &c[a], &active[b], &active[d], &active[e]);
                for (o, t) in out.iter_mut().zip(term) {
                    *o += t;
                }
            }
            out
        })
        .collect()
}

/// `exp(K)` through `order` for a series `K` without constant term.
fn exponential(k: &[Matrix], order: usize) -> Series {
    let n = k[0].rows;
    let mut result = vec![Matrix::zeros(n, n); order + 1];
    result[0] = Matrix::identity(n);
    let mut power = result.clone();
    for m in 1..=order {
        power = product(&power, k, order);
        for (r, p) in result.iter_mut().zip(&power) {
            add_into(r, p, 1.0 / factorial(m));
        }
    }
    result
}

fn factorial(m: usize) -> f64 {
    (1..=m).map(|v| v as f64).product()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Moving-geometry operators of one path.
struct Operators {
    order: usize,
    nao: usize,
    frame: Series,
    core: Series,
    eri: Vec<Vec<f64>>,
    nuclear: Vec<f64>,
}

/// Residuals and density matrices of the series at one set of coefficients.
struct Evaluation {
    /// Orbital gradient on the rotation pairs.
    gradient: Vec<Vec<f64>>,
    /// CI residuals `(H - E) c` of every state.
    residuals: Vec<Vec<Vec<f64>>>,
    /// Active-space energies of every state.
    energies: Vec<Vec<f64>>,
    gamma: Vec<Series>,
    gamma2: Vec<Vec<Vec<f64>>>,
    orbitals: Series,
}

/// Coefficients of the stationarity conditions along a nuclear path.
struct PathSolver<'a> {
    problem: &'a CasProblem<'a>,
    reference: &'a CasPoint,
    pairs: Vec<(usize, usize)>,
}

impl PathSolver<'_> {
    fn nmo(&self) -> usize {
        self.reference.orbitals.cols
    }

    /// Residual series through `order` for the rotation series `kappa` (on
    /// the pairs) and the CI series of every state.
    fn evaluate(
        &self,
        ops: &Operators,
        kappa: &[Vec<f64>],
        ci: &[Vec<Vec<f64>>],
        order: usize,
    ) -> Result<Evaluation> {
        let spec = &self.problem.spec;
        let (nc, na, nmo, nao) = (spec.n_core, spec.n_active, self.nmo(), ops.nao);
        let mut k = vec![Matrix::zeros(nmo, nmo); order + 1];
        for (n, values) in kappa.iter().enumerate().take(order + 1) {
            for (&(p, q), v) in self.pairs.iter().zip(values) {
                k[n][(p, q)] = *v;
                k[n][(q, p)] = -*v;
            }
        }
        let rotation = exponential(&k, order);
        let c = product(&ops.frame, &rotation, order);
        let core: Series = c.iter().map(|m| columns(m, 0, nc)).collect();
        let active: Series = c.iter().map(|m| columns(m, nc, na)).collect();
        let mut p_core = product(&core, &transposed(&core), order);
        p_core
            .iter_mut()
            .for_each(|m| m.as_mut_slice().iter_mut().for_each(|v| *v *= 2.0));
        let (j, kx) = jk_series(&ops.eri, nao, &p_core, order);
        let mut fock: Series = (0..=order)
            .map(|n| {
                ops.core
                    .get(n)
                    .cloned()
                    .unwrap_or_else(|| Matrix::zeros(nao, nao))
            })
            .collect();
        for n in 0..=order {
            add_into(&mut fock[n], &j[n], 1.0);
            add_into(&mut fock[n], &kx[n], -0.5);
        }
        let mut constant = vec![0.0; order + 1];
        for n in 0..=order {
            constant[n] = ops.nuclear.get(n).copied().unwrap_or(0.0);
            for a in 0..=n {
                let mut sum = fock[n - a].clone();
                if n - a < ops.core.len() {
                    add_into(&mut sum, &ops.core[n - a], 1.0);
                }
                constant[n] += 0.5 * p_core[a].frobenius_dot(&sum);
            }
        }
        let h_act = product(&product(&transposed(&active), &fock, order), &active, order);
        let mixed = mixed_series(&ops.eri, nao, &c, &active, order);
        let hams: Vec<ActiveHamiltonian> = (0..=order)
            .map(|n| ActiveHamiltonian {
                n: na,
                constant: constant[n],
                h: h_act[n].clone(),
                g: mixed[n][nc * na * na * na..(nc + na) * na * na * na].to_vec(),
            })
            .collect();
        let space = &self.problem.space;
        let mut residuals = Vec::new();
        let mut energies = Vec::new();
        let mut gamma_states = Vec::new();
        let mut gamma2_states = Vec::new();
        for series in ci {
            let dim = series[0].len();
            let mut sigma = vec![vec![0.0; dim]; order + 1];
            for (kk, ham) in hams.iter().enumerate() {
                for jj in 0..=(order - kk) {
                    if series[jj].iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    let s = space.sigma(ham, &series[jj])?;
                    for (x, y) in sigma[kk + jj].iter_mut().zip(s) {
                        *x += y;
                    }
                }
            }
            let mut numerator = vec![0.0; order + 1];
            let mut norm = vec![0.0; order + 1];
            for n in 0..=order {
                for a in 0..=n {
                    numerator[n] += dot(&series[a], &sigma[n - a]);
                    norm[n] += dot(&series[a], &series[n - a]);
                }
            }
            let mut energy = vec![0.0; order + 1];
            for n in 0..=order {
                let mut value = numerator[n];
                for a in 1..=n {
                    value -= norm[a] * energy[n - a];
                }
                energy[n] = value / norm[0];
            }
            let mut residual = sigma.clone();
            for n in 0..=order {
                for a in 0..=n {
                    for (r, x) in residual[n].iter_mut().zip(&series[n - a]) {
                        *r -= energy[a] * x;
                    }
                }
            }
            let mut g1 = vec![Matrix::zeros(na, na); order + 1];
            let mut g2 = vec![vec![0.0; na.pow(4)]; order + 1];
            for a in 0..=order {
                for b in 0..=(order - a) {
                    if series[a].iter().all(|v| *v == 0.0) || series[b].iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    let (one, two) = space.rdm12(&series[a], &series[b])?;
                    add_into(&mut g1[a + b], &one, 1.0);
                    for (x, y) in g2[a + b].iter_mut().zip(two) {
                        *x += y;
                    }
                }
            }
            residuals.push(residual);
            energies.push(energy);
            gamma_states.push(g1);
            gamma2_states.push(g2);
        }
        // State-averaged density matrices and the orbital gradient.
        let mut gamma = vec![Matrix::zeros(na, na); order + 1];
        let mut gamma2 = vec![vec![0.0; na.pow(4)]; order + 1];
        for ((g1, g2), w) in gamma_states.iter().zip(&gamma2_states).zip(&spec.weights) {
            for n in 0..=order {
                add_into(&mut gamma[n], &g1[n], *w);
                for (x, y) in gamma2[n].iter_mut().zip(&g2[n]) {
                    *x += w * y;
                }
            }
        }
        let p_active = product(
            &product(&active, &gamma, order),
            &transposed(&active),
            order,
        );
        let (ja, ka) = jk_series(&ops.eri, nao, &p_active, order);
        let mut fa = ja;
        for n in 0..=order {
            add_into(&mut fa[n], &ka[n], -0.5);
        }
        let ct = transposed(&c);
        let fi_mo = product(&product(&ct, &fock, order), &c, order);
        let fa_mo = product(&product(&ct, &fa, order), &c, order);
        let mut gradient = vec![vec![0.0; self.pairs.len()]; order + 1];
        for n in 0..=order {
            let mut fgen = Matrix::zeros(nmo, nmo);
            for i in 0..nc {
                for q in 0..nmo {
                    fgen[(i, q)] = 2.0 * (fi_mo[n][(q, i)] + fa_mo[n][(q, i)]);
                }
            }
            for t in 0..na {
                for q in 0..nmo {
                    let mut value = 0.0;
                    for a in 0..=n {
                        for u in 0..na {
                            value += gamma[a][(t, u)] * fi_mo[n - a][(q, nc + u)];
                        }
                        let row = &mixed[n - a][q * na * na * na..(q + 1) * na * na * na];
                        let block = &gamma2[a][t * na * na * na..(t + 1) * na * na * na];
                        value += dot(block, row);
                    }
                    fgen[(nc + t, q)] = value;
                }
            }
            for (slot, &(p, q)) in self.pairs.iter().enumerate() {
                gradient[n][slot] = 2.0 * (fgen[(q, p)] - fgen[(p, q)]);
            }
        }
        Ok(Evaluation {
            gradient,
            residuals,
            energies,
            gamma: gamma_states,
            gamma2: gamma2_states,
            orbitals: c,
        })
    }

    /// Pack the order-`order` residuals: orbital gradient, then every CI
    /// residual projected on the complement of its reference vector.
    fn pack(&self, e: &Evaluation, order: usize) -> Vec<f64> {
        let mut out = e.gradient[order].clone();
        for (state, residual) in e.residuals.iter().enumerate() {
            let c0 = &self.reference.states.vectors[state];
            let r = &residual[order];
            let along = dot(c0, r);
            out.extend(r.iter().zip(c0).map(|(x, y)| x - along * y));
        }
        out
    }

    /// Jacobian of the residuals at the reference applied to `v`.
    fn jacobian(&self, fixed: &Operators, v: &[f64]) -> Result<Vec<f64>> {
        let np = self.pairs.len();
        let kappa = vec![vec![0.0; np], v[..np].to_vec()];
        let mut offset = np;
        let ci: Vec<Vec<Vec<f64>>> = self
            .reference
            .states
            .vectors
            .iter()
            .map(|c0| {
                let dim = c0.len();
                let mut d = v[offset..offset + dim].to_vec();
                offset += dim;
                let along = dot(c0, &d);
                d.iter_mut().zip(c0).for_each(|(x, y)| *x -= along * y);
                vec![c0.clone(), d]
            })
            .collect();
        let e = self.evaluate(fixed, &kappa, &ci, 1)?;
        Ok(self.pack(&e, 1))
    }
}

/// The shared matrix-free GMRES, including an original-operator residual
/// check and a failure when the requested accuracy is not reached.
fn gmres(
    apply: &dyn Fn(&[f64]) -> Result<Vec<f64>>,
    b: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Vec<f64>, f64)> {
    use crate::response::complex::Complex;
    use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};

    struct StationarityOperator<'a> {
        apply: &'a dyn Fn(&[f64]) -> Result<Vec<f64>>,
        dimension: usize,
    }
    impl LinearOperator for StationarityOperator<'_> {
        fn dimension(&self) -> usize {
            self.dimension
        }
        fn structure(&self) -> OperatorStructure {
            OperatorStructure::General
        }
        fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
            krylov::apply_real_linear(x, self.apply)
        }
    }
    let config = ResponseSolverConfig {
        tolerance,
        max_iterations,
        ..ResponseSolverConfig::default()
    };
    let rhs: Vec<_> = b.iter().map(|&v| Complex::from(v)).collect();
    let solution = krylov::solve(
        &StationarityOperator {
            apply,
            dimension: b.len(),
        },
        &rhs,
        None,
        &config,
    )?;
    Ok((
        solution.amplitudes.iter().map(|v| v.re).collect(),
        solution.relative_residual,
    ))
}

/// Energy coefficients of one path for every state.
#[derive(Clone, Debug)]
pub struct PdftPath {
    /// `E(t) = sum_n E_n t^n`, Hartree, per state.
    pub reference: Vec<Vec<f64>>,
    pub pdft: Vec<Vec<f64>>,
    /// `lambda reference + (1 - lambda) pdft`.
    pub total: Vec<Vec<f64>>,
    /// Largest stationarity residual left at every order.
    pub stationarity: Vec<f64>,
    /// Relative residual of the linear solve at every order >= 1.
    pub solves: Vec<f64>,
}

/// Taylor coefficients through `order` of the active-space and MC-PDFT
/// energies of every state of the converged reference `point` along the
/// nuclear direction `direction` (one Cartesian vector per atom).
#[allow(clippy::too_many_arguments)]
pub fn pdft_path(
    problem: &CasProblem,
    point: &CasPoint,
    functional: Option<&OnTopFunctional>,
    mol: &Molecule,
    mb: &MolBasis,
    grid: Option<&MolecularGrid>,
    direction: &[[f64; 3]],
    order: usize,
    memory_mb: usize,
) -> Result<PdftPath> {
    if direction.len() != mol.len() || direction.iter().flatten().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "a nuclear direction holds one finite Cartesian vector per atom".into(),
        ));
    }
    let energies = &point.energies;
    for i in 0..energies.len() {
        for j in 0..i {
            if (energies[i] - energies[j]).abs() < 1e-6 {
                return Err(DftError::Unsupported {
                    feature: "nuclear derivatives of degenerate active-space states".into(),
                    missing: "the response of a degenerate set of reference states".into(),
                });
            }
        }
    }
    let hartree_fock = ResolvedFunctional::hartree_fock();
    let (base, grid) = match (functional, grid) {
        (Some(f), Some(g)) => (&f.base, Some(g)),
        (None, _) => (&hartree_fock, None),
        (Some(_), None) => {
            return Err(DftError::InvalidInput(
                "the on-top energy series needs the molecular grid".into(),
            ))
        }
    };
    let geometry = GeometryTaylor::new(
        mol, mb, grid, base, None, direction, [0.0; 3], [0.0; 3], order, memory_mb,
    )?;
    let nao = mb.nao_sph;
    let ops = Operators {
        order,
        nao,
        frame: metric_transport(&point.orbitals, &geometry.overlap, order),
        core: geometry.core_coefficients()[..=order].to_vec(),
        eri: geometry.coulomb_coefficients()[..=order].to_vec(),
        nuclear: geometry.nuclear_coefficients()[..=order].to_vec(),
    };
    let fixed = Operators {
        order: 0,
        nao,
        frame: vec![ops.frame[0].clone()],
        core: vec![ops.core[0].clone()],
        eri: vec![ops.eri[0].clone()],
        nuclear: vec![ops.nuclear[0]],
    };
    let nmo = point.orbitals.cols;
    let pairs: Vec<(usize, usize)> = (0..nmo)
        .flat_map(|p| (0..p).map(move |q| (p, q)))
        .filter(|&(p, q)| point.hessian_diagonal[(p, q)] > 0.0)
        .collect();
    let solver = PathSolver {
        problem,
        reference: point,
        pairs,
    };
    let np = solver.pairs.len();
    let mut kappa = vec![vec![0.0; np]];
    let mut ci: Vec<Vec<Vec<f64>>> = point
        .states
        .vectors
        .iter()
        .map(|c| vec![c.clone()])
        .collect();
    let mut stationarity = Vec::with_capacity(ops.order + 1);
    let mut solves = Vec::with_capacity(ops.order);
    let zero = solver.evaluate(&ops, &kappa, &ci, 0)?;
    stationarity.push(
        solver
            .pack(&zero, 0)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs())),
    );
    for n in 1..=order {
        kappa.push(vec![0.0; np]);
        for series in ci.iter_mut() {
            let dim = series[0].len();
            let mut normalization = 0.0;
            for a in 1..n {
                normalization += dot(&series[a], &series[n - a]);
            }
            let c0 = series[0].clone();
            series.push(
                c0.iter()
                    .map(|v| -0.5 * normalization * v)
                    .collect::<Vec<f64>>(),
            );
            debug_assert_eq!(series[n].len(), dim);
        }
        let current = solver.evaluate(&ops, &kappa, &ci, n)?;
        let rhs: Vec<f64> = solver.pack(&current, n).iter().map(|v| -v).collect();
        let (delta, relative) = gmres(&|v| solver.jacobian(&fixed, v), &rhs, 1e-12, 2000)?;
        solves.push(relative);
        for (slot, value) in kappa[n].iter_mut().enumerate() {
            *value += delta[slot];
        }
        let mut offset = np;
        for (state, series) in ci.iter_mut().enumerate() {
            let c0 = &point.states.vectors[state];
            let dim = c0.len();
            let mut d = delta[offset..offset + dim].to_vec();
            offset += dim;
            let along = dot(c0, &d);
            d.iter_mut().zip(c0).for_each(|(x, y)| *x -= along * y);
            for (x, y) in series[n].iter_mut().zip(d) {
                *x += y;
            }
        }
        let check = solver.evaluate(&ops, &kappa, &ci, n)?;
        stationarity.push(
            solver
                .pack(&check, n)
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs())),
        );
    }
    let last = solver.evaluate(&ops, &kappa, &ci, order)?;
    let (pdft, lambda): (Vec<Vec<f64>>, f64) = match functional {
        Some(f) => (
            (0..ci.len())
                .map(|state| pdft_energy_series(problem, &ops, &geometry, f, &last, state, order))
                .collect::<Result<_>>()?,
            f.lambda,
        ),
        None => (last.energies.clone(), 0.0),
    };
    let total = last
        .energies
        .iter()
        .zip(&pdft)
        .map(|(r, p)| {
            r.iter()
                .zip(p)
                .map(|(a, b)| lambda * a + (1.0 - lambda) * b)
                .collect()
        })
        .collect();
    Ok(PdftPath {
        reference: last.energies.clone(),
        pdft,
        total,
        stationarity,
        solves,
    })
}

/// MC-PDFT energy series of one state.
fn pdft_energy_series(
    problem: &CasProblem,
    ops: &Operators,
    geometry: &GeometryTaylor,
    functional: &OnTopFunctional,
    e: &Evaluation,
    state: usize,
    order: usize,
) -> Result<Vec<f64>> {
    let spec = &problem.spec;
    let (nc, na, nao) = (spec.n_core, spec.n_active, ops.nao);
    let core: Series = e.orbitals.iter().map(|m| columns(m, 0, nc)).collect();
    let active: Series = e.orbitals.iter().map(|m| columns(m, nc, na)).collect();
    let gamma = &e.gamma[state];
    let gamma2 = &e.gamma2[state];
    let mut p_core = product(&core, &transposed(&core), order);
    p_core
        .iter_mut()
        .for_each(|m| m.as_mut_slice().iter_mut().for_each(|v| *v *= 2.0));
    let p_active = product(&product(&active, gamma, order), &transposed(&active), order);
    let total: Series = p_core
        .iter()
        .zip(&p_active)
        .map(|(a, b)| {
            let mut m = a.clone();
            add_into(&mut m, b, 1.0);
            m
        })
        .collect();
    let (j, _) = jk_series(&ops.eri, nao, &total, order);
    let mut energy = vec![0.0; order + 1];
    for n in 0..=order {
        energy[n] += ops.nuclear[n];
        for a in 0..=n {
            energy[n] += total[a].frobenius_dot(&ops.core[n - a]);
            energy[n] += 0.5 * total[a].frobenius_dot(&j[n - a]);
        }
    }
    // Symmetrized active two-particle density matrix series.
    let index = |a: usize, b: usize, c: usize, d: usize| ((a * na + b) * na + c) * na + d;
    let pair: Vec<Matrix> = gamma2
        .iter()
        .map(|g| {
            let mut out = Matrix::zeros(na * na, na * na);
            for t in 0..na {
                for u in 0..na {
                    for v in 0..na {
                        for w in 0..na {
                            out[(t * na + u, v * na + w)] = 0.25
                                * (g[index(t, u, v, w)]
                                    + g[index(v, w, t, u)]
                                    + g[index(u, t, w, v)]
                                    + g[index(w, v, u, t)]);
                        }
                    }
                }
            }
            out
        })
        .collect();
    let width = order + 1;
    let conv = |a: &[f64], b: &[f64]| -> Vec<f64> {
        (0..width)
            .map(|n| (0..=n).map(|k| a[k] * b[n - k]).sum())
            .collect()
    };
    let gga = functional.base.family != crate::xc::Family::Lda;
    let on_top: Result<Vec<f64>> = geometry
        .grid_points()
        .par_iter()
        .try_fold(
            || vec![0.0; width],
            |mut acc, point| -> Result<Vec<f64>> {
                // Orbital values and gradients as series: [orbital][component][order].
                let orbital = |m: &Series, col: usize| -> [Vec<f64>; 4] {
                    std::array::from_fn(|x| {
                        let mut out = vec![0.0; width];
                        for (mu, ao) in point.aos.iter().enumerate() {
                            let coefficient: Vec<f64> =
                                (0..width).map(|n| m[n][(mu, col)]).collect();
                            let term = conv(&ao[x][..width], &coefficient);
                            for (o, t) in out.iter_mut().zip(term) {
                                *o += t;
                            }
                        }
                        out
                    })
                };
                let cores: Vec<[Vec<f64>; 4]> = (0..nc).map(|i| orbital(&core, i)).collect();
                let acts: Vec<[Vec<f64>; 4]> = (0..na).map(|t| orbital(&active, t)).collect();
                let mut rho_c = vec![0.0; width];
                let mut grad_c = [vec![0.0; width], vec![0.0; width], vec![0.0; width]];
                for phi in &cores {
                    let square = conv(&phi[0], &phi[0]);
                    for (r, s) in rho_c.iter_mut().zip(&square) {
                        *r += 2.0 * s;
                    }
                    for x in 0..3 {
                        let g = conv(&phi[0], &phi[1 + x]);
                        for (r, s) in grad_c[x].iter_mut().zip(&g) {
                            *r += 4.0 * s;
                        }
                    }
                }
                // Pair products Q_tu and their gradients as series.
                let mut q = vec![vec![0.0; width]; na * na];
                let mut grad_q =
                    vec![[vec![0.0; width], vec![0.0; width], vec![0.0; width]]; na * na];
                for t in 0..na {
                    for u in 0..na {
                        q[t * na + u] = conv(&acts[t][0], &acts[u][0]);
                        for x in 0..3 {
                            let a = conv(&acts[t][1 + x], &acts[u][0]);
                            let b = conv(&acts[t][0], &acts[u][1 + x]);
                            grad_q[t * na + u][x] = a.iter().zip(&b).map(|(p, r)| p + r).collect();
                        }
                    }
                }
                let mut rho_a = vec![0.0; width];
                let mut grad_a = [vec![0.0; width], vec![0.0; width], vec![0.0; width]];
                let mut pi_aa = vec![0.0; width];
                let mut grad_pi_aa = [vec![0.0; width], vec![0.0; width], vec![0.0; width]];
                for n in 0..width {
                    for a in 0..=n {
                        for tu in 0..na * na {
                            let (t, u) = (tu / na, tu % na);
                            rho_a[n] += gamma[a][(t, u)] * q[tu][n - a];
                            for x in 0..3 {
                                grad_a[x][n] += gamma[a][(t, u)] * grad_q[tu][x][n - a];
                            }
                        }
                        for b in 0..=(n - a) {
                            for tu in 0..na * na {
                                for vw in 0..na * na {
                                    let g = pair[a][(tu, vw)];
                                    if g == 0.0 {
                                        continue;
                                    }
                                    pi_aa[n] += 0.5 * g * q[tu][b] * q[vw][n - a - b];
                                    for x in 0..3 {
                                        grad_pi_aa[x][n] += 0.5
                                            * g
                                            * (grad_q[tu][x][b] * q[vw][n - a - b]
                                                + q[tu][b] * grad_q[vw][x][n - a - b]);
                                    }
                                }
                            }
                        }
                    }
                }
                let rho: Vec<f64> = rho_c.iter().zip(&rho_a).map(|(a, b)| a + b).collect();
                if rho[0] < 1e-14 {
                    return Ok(acc);
                }
                let cc = conv(&rho_c, &rho_c);
                let ca = conv(&rho_c, &rho_a);
                let pi: Vec<f64> = (0..width)
                    .map(|n| 0.25 * cc[n] + 0.5 * ca[n] + pi_aa[n])
                    .collect();
                let grad_rho: [Vec<f64>; 3] = std::array::from_fn(|x| {
                    grad_c[x]
                        .iter()
                        .zip(&grad_a[x])
                        .map(|(a, b)| a + b)
                        .collect()
                });
                let grad_pi: [Vec<f64>; 3] = std::array::from_fn(|x| {
                    let a = conv(&rho_c, &grad_c[x]);
                    let b = conv(&grad_c[x], &rho_a);
                    let c = conv(&rho_c, &grad_a[x]);
                    (0..width)
                        .map(|n| 0.5 * a[n] + 0.5 * (b[n] + c[n]) + grad_pi_aa[x][n])
                        .collect()
                });
                let mut inputs = vec![rho.clone(), pi];
                inputs.extend(grad_rho.iter().cloned());
                inputs.extend(grad_pi.iter().cloned());
                let translated = evaluate_series_many_result(&inputs, |v| {
                    Ok(translate(
                        functional.translation,
                        v[0],
                        v[1],
                        [v[2], v[3], v[4]],
                        [v[5], v[6], v[7]],
                    )
                    .to_vec())
                })?;
                let zero = vec![0.0; width];
                let variables: [Vec<f64>; 7] = [
                    translated[0].clone(),
                    translated[1].clone(),
                    if gga {
                        translated[2].clone()
                    } else {
                        zero.clone()
                    },
                    if gga {
                        translated[3].clone()
                    } else {
                        zero.clone()
                    },
                    if gga {
                        translated[4].clone()
                    } else {
                        zero.clone()
                    },
                    zero.clone(),
                    zero,
                ];
                let density = functional.base.semilocal_taylor_energy(&XcTaylorInput {
                    variables,
                    parameters: None,
                })?;
                let weighted = conv(&point.weight[..width], &density);
                for (a, w) in acc.iter_mut().zip(weighted) {
                    *a += w;
                }
                Ok(acc)
            },
        )
        .try_reduce(
            || vec![0.0; width],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                Ok(a)
            },
        );
    for (e, o) in energy.iter_mut().zip(on_top?) {
        *e += o;
    }
    Ok(energy)
}

/// Nuclear gradients and (optionally) Hessians of every state.
#[derive(Clone, Debug)]
pub struct PdftDerivatives {
    /// `dE/dx` per state, Hartree/Bohr, `3 atom + axis`.
    pub reference_gradient: Vec<Vec<f64>>,
    pub gradient: Vec<Vec<f64>>,
    /// Hessians per state, Hartree/Bohr^2.
    pub reference_hessian: Option<Vec<Matrix>>,
    pub hessian: Option<Vec<Matrix>>,
    pub paths: usize,
    pub stationarity: f64,
    pub solve_residual: f64,
}

/// Gradients (`order = 1`) or gradients and Hessians (`order = 2`) of the
/// total (hybrid) PDFT and of the active-space energies of every state;
/// without an on-top functional only the active-space energies, whose
/// values then also fill the PDFT entries.
#[allow(clippy::too_many_arguments)]
pub fn pdft_derivatives(
    problem: &CasProblem,
    point: &CasPoint,
    functional: Option<&OnTopFunctional>,
    mol: &Molecule,
    mb: &MolBasis,
    grid: Option<&MolecularGrid>,
    order: usize,
    memory_mb: usize,
) -> Result<PdftDerivatives> {
    if !(1..=2).contains(&order) {
        return Err(DftError::InvalidInput(
            "MC-PDFT nuclear derivatives are of order 1 or 2".into(),
        ));
    }
    let n = 3 * mol.len();
    let states = point.energies.len();
    let unit = |i: usize| -> Vec<[f64; 3]> {
        let mut d = vec![[0.0; 3]; mol.len()];
        d[i / 3][i % 3] = 1.0;
        d
    };
    let mut paths = 0;
    let mut stationarity = 0.0_f64;
    let mut solve_residual = 0.0_f64;
    let mut run = |direction: Vec<[f64; 3]>| -> Result<PdftPath> {
        let path = pdft_path(
            problem, point, functional, mol, mb, grid, &direction, order, memory_mb,
        )?;
        paths += 1;
        stationarity = path
            .stationarity
            .iter()
            .copied()
            .fold(stationarity, f64::max);
        solve_residual = path.solves.iter().copied().fold(solve_residual, f64::max);
        Ok(path)
    };
    let mut gradient = vec![vec![0.0; n]; states];
    let mut reference_gradient = vec![vec![0.0; n]; states];
    let mut second = vec![vec![0.0; n]; states];
    let mut reference_second = vec![vec![0.0; n]; states];
    for i in 0..n {
        let path = run(unit(i))?;
        for s in 0..states {
            gradient[s][i] = path.total[s][1];
            reference_gradient[s][i] = path.reference[s][1];
            if order == 2 {
                second[s][i] = path.total[s][2];
                reference_second[s][i] = path.reference[s][2];
            }
        }
    }
    let (hessian, reference_hessian) = if order == 2 {
        let mut h = vec![Matrix::zeros(n, n); states];
        let mut hr = h.clone();
        for s in 0..states {
            for i in 0..n {
                h[s][(i, i)] = 2.0 * second[s][i];
                hr[s][(i, i)] = 2.0 * reference_second[s][i];
            }
        }
        for i in 0..n {
            for j in 0..i {
                let mut d = unit(i);
                d[j / 3][j % 3] += 1.0;
                let path = run(d)?;
                for s in 0..states {
                    let value = path.total[s][2] - second[s][i] - second[s][j];
                    h[s][(i, j)] = value;
                    h[s][(j, i)] = value;
                    let value =
                        path.reference[s][2] - reference_second[s][i] - reference_second[s][j];
                    hr[s][(i, j)] = value;
                    hr[s][(j, i)] = value;
                }
            }
        }
        (Some(h), Some(hr))
    } else {
        (None, None)
    };
    Ok(PdftDerivatives {
        reference_gradient,
        gradient,
        reference_hessian,
        hessian,
        paths,
        stationarity,
        solve_residual,
    })
}
