// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Orbital-manifold globalization using only a variational energy and Fock.
//! Cayley steps preserve the occupied subspace and their stable density
//! increments retain quadratic diagonal work near a stationary reference.

use super::{put, z, BlochMatrix};
use crate::dfpt::orbitals::{adjoint, matmul};
use crate::linalg::{solve_linear_matrix, AccurateSum, Matrix};
use crate::response::complex::Complex;
use crate::Result;

fn inner(a: &[BlochMatrix], b: &[BlochMatrix]) -> f64 {
    let mut sum = AccurateSum::default();
    for (a, b) in a.iter().zip(b) {
        for (x, y) in a.re.as_slice().iter().zip(b.re.as_slice()) {
            sum.add_product(*x, *y);
        }
        for (x, y) in a.im.as_slice().iter().zip(b.im.as_slice()) {
            sum.add_product(*x, *y);
        }
    }
    sum.value()
}

fn commutator(a: &BlochMatrix, b: &BlochMatrix) -> Result<BlochMatrix> {
    let mut value = matmul(a, b)?;
    value.add_scaled(&matmul(b, a)?, -1.0);
    Ok(value)
}

fn antihermitian(value: &mut BlochMatrix) {
    for i in 0..value.rows() {
        for j in i..value.cols() {
            let v = (z(value, i, j) - z(value, j, i).conj()) * 0.5;
            put(value, i, j, v);
            put(value, j, i, -v.conj());
        }
    }
}

fn tangent(projector: &BlochMatrix, vector: &BlochMatrix) -> Result<BlochMatrix> {
    let left = matmul(projector, vector)?;
    let mut value = left.clone();
    value.add_scaled(&matmul(vector, projector)?, 1.0);
    value.add_scaled(&matmul(&left, projector)?, -2.0);
    antihermitian(&mut value);
    Ok(value)
}

fn solve(a: &BlochMatrix, b: &BlochMatrix) -> Result<BlochMatrix> {
    let n = a.rows();
    let mut rhs = Matrix::zeros(2 * n, b.cols());
    for i in 0..n {
        for j in 0..b.cols() {
            rhs[(i, j)] = b.re[(i, j)];
            rhs[(n + i, j)] = b.im[(i, j)];
        }
    }
    let solution = solve_linear_matrix(&a.real_embedding(), &rhs)?;
    let mut result = BlochMatrix::zeros(n, b.cols());
    for i in 0..n {
        for j in 0..b.cols() {
            put(
                &mut result,
                i,
                j,
                Complex::new(solution[(i, j)], solution[(n + i, j)]),
            );
        }
    }
    Ok(result)
}

/// U P U^H-P = alpha A^-1 [D,P] (A^H)^-1, A=I-alpha D/2.
/// This exact rational identity avoids subtracting two nearly equal P's.
fn cayley_change(p: &BlochMatrix, d: &BlochMatrix, alpha: f64) -> Result<BlochMatrix> {
    let mut a = d.clone();
    a.scale(-0.5 * alpha);
    for i in 0..a.rows() {
        a.re[(i, i)] += 1.0;
    }
    let left = solve(&a, &commutator(d, p)?)?;
    let mut change = adjoint(&solve(&a, &adjoint(&left)?)?)?;
    change.scale(alpha);
    // Both endpoints are physical Hermitian densities; only linear-solve
    // roundoff is paired. No independent finite-q response is projected.
    for i in 0..change.rows() {
        for j in i..change.cols() {
            let v = (z(&change, i, j) + z(&change, j, i).conj()) * 0.5;
            put(&mut change, i, j, v);
            put(&mut change, j, i, v.conj());
        }
    }
    Ok(change)
}

pub(crate) struct OrbitalMinimizer {
    frame: Vec<BlochMatrix>,
    projector: Vec<BlochMatrix>,
    previous_gradient: Vec<BlochMatrix>,
    previous_step: Vec<BlochMatrix>,
    history: Vec<Secant>,
}

struct Secant {
    step: Vec<BlochMatrix>,
    curvature: Vec<BlochMatrix>,
}

pub(crate) struct OrbitalStep {
    pub density: Vec<BlochMatrix>,
    pub work: f64,
    pub fraction: f64,
    pub history_size: usize,
}

impl OrbitalMinimizer {
    /// Full complex metric-orthonormal frames define fixed coordinates.
    /// The caller restarts at the supplied canonical integer projectors.
    pub fn new(frame: Vec<BlochMatrix>, occupations: &[Vec<f64>]) -> Self {
        Self {
            frame,
            projector: occupations
                .iter()
                .map(|f| {
                    let mut p = BlochMatrix::zeros(f.len(), f.len());
                    for (i, value) in f.iter().enumerate() {
                        p.re[(i, i)] = *value;
                    }
                    p
                })
                .collect(),
            previous_gradient: Vec::new(),
            previous_step: Vec::new(),
            history: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        density: &[BlochMatrix],
        fock: &[BlochMatrix],
        energy: f64,
        block_weight: f64,
        quadratic: bool,
        mut evaluate: impl FnMut(&[BlochMatrix]) -> Result<(f64, Vec<BlochMatrix>)>,
    ) -> Result<Option<OrbitalStep>> {
        let mut gradient = Vec::new();
        for ((b, p), f) in self.frame.iter().zip(&self.projector).zip(fock) {
            let f = matmul(&adjoint(b)?, &matmul(f, b)?)?;
            let mut g = commutator(p, &f)?;
            antihermitian(&mut g);
            gradient.push(g);
        }
        let transport = |v: &[BlochMatrix]| -> Result<Vec<BlochMatrix>> {
            self.projector
                .iter()
                .zip(v)
                .map(|(p, x)| tangent(p, x))
                .collect()
        };
        // All secants live in the current occupied-subspace tangent space.
        // The stored gradient is the *negative* energy gradient, hence
        // y = transported(old gradient) - new gradient in the BFGS update.
        for secant in &mut self.history {
            secant.step = transport(&secant.step)?;
            secant.curvature = transport(&secant.curvature)?;
        }
        self.history.retain(|s| inner(&s.step, &s.curvature) > 0.0);
        if !self.previous_gradient.is_empty() {
            let step = transport(&self.previous_step)?;
            let mut curvature = transport(&self.previous_gradient)?;
            for (y, g) in curvature.iter_mut().zip(&gradient) {
                y.add_scaled(g, -1.0);
            }
            let sy = inner(&step, &curvature);
            let scale = inner(&step, &step).sqrt() * inner(&curvature, &curvature).sqrt();
            if sy.is_finite() && sy > 1e-12 * scale {
                if self.history.len() == 12 {
                    self.history.remove(0);
                }
                self.history.push(Secant { step, curvature });
            }
        }
        let mut direction = gradient.clone();
        let mut coefficients = Vec::with_capacity(self.history.len());
        for secant in self.history.iter().rev() {
            let alpha = inner(&secant.step, &direction) / inner(&secant.step, &secant.curvature);
            coefficients.push(alpha);
            for (q, y) in direction.iter_mut().zip(&secant.curvature) {
                q.add_scaled(y, -alpha);
            }
        }
        if let Some(last) = self.history.last() {
            let scale =
                inner(&last.step, &last.curvature) / inner(&last.curvature, &last.curvature);
            for q in &mut direction {
                q.scale(scale);
            }
        }
        for (secant, alpha) in self.history.iter().zip(coefficients.into_iter().rev()) {
            let beta =
                inner(&secant.curvature, &direction) / inner(&secant.step, &secant.curvature);
            for (q, s) in direction.iter_mut().zip(&secant.step) {
                q.add_scaled(s, alpha - beta);
            }
        }
        for restart in 0..2 {
            if restart > 0 || inner(&gradient, &direction) <= 0.0 {
                direction = gradient.clone();
                self.history.clear();
            }
            let slope = -block_weight * inner(&gradient, &direction);
            let norm = direction
                .iter()
                .map(|d| inner(std::slice::from_ref(d), std::slice::from_ref(d)).sqrt())
                .fold(0.0_f64, f64::max);
            if !slope.is_finite() || slope >= 0.0 || !norm.is_finite() || norm == 0.0 {
                return Ok(None);
            }
            // A rotation trust radius bounds the retraction, not a physical
            // denominator or the Fock. Backtracking may reduce this radius.
            let maximum = if self.history.is_empty() {
                0.25 / norm
            } else {
                (0.25 / norm).min(1.0)
            };
            let mut fraction = 1.0;
            for _ in 0..32 {
                let alpha = maximum * fraction;
                let mut next_projector = self.projector.clone();
                let mut change = Vec::new();
                for (((p, next), d), b) in self
                    .projector
                    .iter()
                    .zip(&mut next_projector)
                    .zip(&direction)
                    .zip(&self.frame)
                {
                    let delta = cayley_change(p, d, alpha)?;
                    next.add_scaled(&delta, 1.0);
                    change.push(matmul(&matmul(b, &delta)?, &adjoint(b)?)?);
                }
                let mut candidate = density.to_vec();
                for (p, delta) in candidate.iter_mut().zip(&change) {
                    p.add_scaled(delta, 1.0);
                }
                let (next_energy, next_fock) = evaluate(&candidate)?;
                let work = if quadratic {
                    block_weight * quadratic_density_work(&change, fock, &next_fock)
                } else {
                    next_energy - energy
                };
                if work.is_finite() && work <= 1e-4 * alpha * slope {
                    self.projector = next_projector;
                    self.previous_gradient = gradient;
                    self.previous_step = direction;
                    for step in &mut self.previous_step {
                        step.scale(alpha);
                    }
                    return Ok(Some(OrbitalStep {
                        density: candidate,
                        work,
                        fraction,
                        history_size: self.history.len(),
                    }));
                }
                fraction *= 0.5;
            }
        }
        Ok(None)
    }
}

/// Compensated endpoint work before statistical/block weights. It is exact
/// for a quadratic density energy, including all real and imaginary entries.
pub(crate) fn quadratic_density_work(
    change: &[BlochMatrix],
    old: &[BlochMatrix],
    new: &[BlochMatrix],
) -> f64 {
    let mut work = AccurateSum::default();
    for ((d, a), b) in change.iter().zip(old).zip(new) {
        for i in 0..d.rows() {
            for j in 0..d.cols() {
                work.add_product(d.re[(i, j)], 0.5 * (a.re[(j, i)] + b.re[(j, i)]));
                work.add_product(-d.im[(i, j)], 0.5 * (a.im[(j, i)] + b.im[(j, i)]));
            }
        }
    }
    work.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cayley_density_entry_matches_a_closed_complex_two_level_rotation() {
        let mut p = BlochMatrix::zeros(2, 2);
        p.re[(0, 0)] = 1.0;
        let a = Complex::new(-0.31, 0.23);
        let r2 = a.re * a.re + a.im * a.im;
        let mut d = BlochMatrix::zeros(2, 2);
        put(&mut d, 1, 0, a);
        put(&mut d, 0, 1, -a.conj());
        for h in [0.2, 0.1, 0.05, 0.025, 1e-8, 1e-12] {
            let change = cayley_change(&p, &d, h).unwrap();
            let s = h * h * r2 / 4.0;
            let diagonal = h * h * r2 / (1.0 + s).powi(2);
            let off = a * (h * (1.0 - s) / (1.0 + s).powi(2));
            let exact = [
                [Complex::new(-diagonal, 0.0), off.conj()],
                [off, Complex::new(diagonal, 0.0)],
            ];
            for (i, row) in exact.iter().enumerate() {
                for (j, value) in row.iter().enumerate() {
                    let error = (z(&change, i, j) - *value).abs();
                    println!("Cayley density h={h:.1e} entry=({i},{j}) defect={error:.12e}");
                    assert!(error <= 32.0 * f64::EPSILON * value.abs());
                }
            }
            let mut next = p.clone();
            next.add_scaled(&change, 1.0);
            let mut defect = matmul(&next, &next).unwrap();
            defect.add_scaled(&next, -1.0);
            assert!(defect
                .re
                .as_slice()
                .iter()
                .chain(defect.im.as_slice())
                .all(|v| v.abs() < 1e-14));
            assert!((next.re[(0, 0)] + next.re[(1, 1)] - 1.0).abs() < 1e-14);
        }
    }
}

/// Full complex occupied/virtual frame for the shared real AH eigensolver.
/// Each complex rotation contributes independent real and imaginary entries.
pub(crate) struct AugmentedHessianMinimizer {
    frame: Vec<BlochMatrix>,
    counts: Vec<usize>,
    trust: crate::scf::trah::TrustRegion,
}
impl AugmentedHessianMinimizer {
    pub fn new(frame: Vec<BlochMatrix>, occupations: &[Vec<f64>]) -> Result<Self> {
        if occupations.iter().flatten().any(|f| *f != 0.0 && *f != 1.0) {
            return Err(crate::DftError::InvalidInput(
                "spinor TRAH requires integer occupations".into(),
            ));
        }
        Ok(Self {
            frame,
            counts: occupations
                .iter()
                .map(|f| f.iter().filter(|v| **v == 1.0).count())
                .collect(),
            trust: crate::scf::trah::TrustRegion::new(0.2)?,
        })
    }
    fn generators(&self, x: &[f64]) -> Vec<BlochMatrix> {
        let mut off = 0;
        self.frame
            .iter()
            .zip(&self.counts)
            .map(|(c, &no)| {
                let mut k = BlochMatrix::zeros(c.cols(), c.cols());
                for a in no..c.cols() {
                    for i in 0..no {
                        let value = Complex::new(x[off], x[off + 1]);
                        off += 2;
                        put(&mut k, a, i, value);
                        put(&mut k, i, a, -value.conj());
                    }
                }
                k
            })
            .collect()
    }
    fn project(&self, fock: &[BlochMatrix], weight: f64) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        for ((c, &no), f) in self.frame.iter().zip(&self.counts).zip(fock) {
            let f = matmul(&adjoint(c)?, &matmul(f, c)?)?;
            for a in no..c.cols() {
                for i in 0..no {
                    let v = z(&f, a, i);
                    out.extend([2.0 * weight * v.re, 2.0 * weight * v.im]);
                }
            }
        }
        Ok(out)
    }
    fn curvature(
        &self,
        x: &[f64],
        projectors: &[BlochMatrix],
        mo_fock: &[BlochMatrix],
        kernel: &dyn super::SpinorDensityKernel,
        weight: f64,
    ) -> Result<Vec<f64>> {
        let generators = self.generators(x);
        let dp = generators
            .iter()
            .zip(projectors)
            .zip(&self.frame)
            .map(|((k, p), c)| matmul(&matmul(c, &commutator(k, p)?)?, &adjoint(c)?))
            .collect::<Result<Vec<_>>>()?;
        let df = kernel.apply(&dp)?;
        let mut projected = self.project(&df, weight)?;
        let mut off = 0;
        for ((f, k), &no) in mo_fock.iter().zip(&generators).zip(&self.counts) {
            let comm = commutator(f, k)?;
            for a in no..f.rows() {
                for i in 0..no {
                    projected[off] += 2.0 * weight * comm.re[(a, i)];
                    projected[off + 1] += 2.0 * weight * comm.im[(a, i)];
                    off += 2;
                }
            }
        }
        Ok(projected)
    }
    pub fn step(
        &mut self,
        backend: &impl super::SpinorEnergyBackend,
        density: &[BlochMatrix],
        fock: &[BlochMatrix],
        energy: f64,
    ) -> Result<Option<OrbitalStep>> {
        let weight = 1.0 / density.len() as f64;
        let gradient = self.project(fock, weight)?;
        if gradient.is_empty() {
            return Ok(None);
        }
        let mut diagonal = Vec::new();
        let mut mo_fock = Vec::new();
        let mut projectors = Vec::new();
        for ((c, &no), f) in self.frame.iter().zip(&self.counts).zip(fock) {
            let f = matmul(&adjoint(c)?, &matmul(f, c)?)?;
            for a in no..c.cols() {
                for i in 0..no {
                    let gap = 2.0 * weight * (f.re[(a, a)] - f.re[(i, i)]);
                    diagonal.extend([gap, gap]);
                }
            }
            let mut p = BlochMatrix::zeros(c.cols(), c.cols());
            for i in 0..no {
                p.re[(i, i)] = 1.0;
            }
            projectors.push(p);
            mo_fock.push(f);
        }
        let kernel = backend.orbital_kernel(density)?;
        for rejection in 0..12 {
            let hessian =
                |x: &[f64]| self.curvature(x, &projectors, &mo_fock, kernel.as_ref(), weight);

            let step = crate::scf::trah::direction(
                &diagonal,
                &gradient,
                hessian,
                self.trust.radius,
                40,
                1e-3,
            )?;
            if step.predicted_work >= 0.0 {
                return Ok(None);
            }
            let generators = self.generators(&step.rotation);
            let mut frames = Vec::new();
            let mut changes = Vec::new();
            let mut next = Vec::new();
            for ((k, p), c) in generators.iter().zip(&projectors).zip(&self.frame) {
                let change = cayley_change(p, k, 1.0)?;
                let mut a = k.clone();
                a.scale(-0.5);
                let mut b = k.clone();
                b.scale(0.5);
                for i in 0..a.rows() {
                    a.re[(i, i)] += 1.0;
                    b.re[(i, i)] += 1.0;
                }
                frames.push(matmul(c, &solve(&a, &b)?)?);
                let mut q = p.clone();
                q.add_scaled(&change, 1.0);
                next.push(matmul(&matmul(c, &q)?, &adjoint(c)?)?);
                changes.push(change);
            }
            let fields = backend.evaluate(&next)?;
            let work = if backend.quadratic_density_energy() {
                let new_mo = self
                    .frame
                    .iter()
                    .zip(&fields.fock)
                    .map(|(c, f)| matmul(&adjoint(c)?, &matmul(f, c)?))
                    .collect::<Result<Vec<_>>>()?;
                weight * quadratic_density_work(&changes, &mo_fock, &new_mo)
            } else {
                let Some(value) = crate::scf::trah::energy_work(energy, fields.energy, |t| {
                    let intermediate = self
                        .frame
                        .iter()
                        .zip(&projectors)
                        .zip(&changes)
                        .map(|((c, p), d)| {
                            let mut q = p.clone();
                            q.add_scaled(d, t);
                            matmul(&matmul(c, &q)?, &adjoint(c)?)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let f = backend.evaluate(&intermediate)?.fock;
                    let mo = self
                        .frame
                        .iter()
                        .zip(&f)
                        .map(|(c, f)| matmul(&adjoint(c)?, &matmul(f, c)?))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(weight * quadratic_density_work(&changes, &mo, &mo))
                })?
                else {
                    return Ok(None);
                };
                value
            };
            let length = step.rotation.iter().map(|x| x * x).sum::<f64>().sqrt();
            let radius = self.trust.radius;
            let (accepted, ratio) = self.trust.assess(step.predicted_work, work, length);
            crate::logger::log(&format!(
                "[spinor-scf] TRAH accepted={accepted} rejection={rejection} radius={radius:.12e} step={length:.12e} predicted={:.12e} actual={work:.12e} ratio={ratio:.12e} shift={:.12e} inner_residual={:.12e} products={}",
                step.predicted_work,step.shift,step.residual,step.products));
            if !accepted && length < 0.5 * self.trust.radius {
                return Ok(None);
            }
            if accepted {
                self.frame = frames;
                return Ok(Some(OrbitalStep {
                    density: next,
                    work,
                    fraction: 1.0,
                    history_size: 0,
                }));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
#[path = "trah_tests.rs"]
mod trah_tests;
