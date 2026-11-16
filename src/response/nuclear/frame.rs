// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Linear response and excitation energies along a nuclear path.
//!
//! The nuclear Taylor driver transports the reference orbitals so that
//! `B(t)^T S(t) B(t) = 1`. In that frame the reference density `D(t)` is a
//! projector that commutes with the Kohn-Sham matrix `F(t)`, and a
//! first-order density `d(t)` oscillating at the frequency `w` under the
//! operator `V(t)` obeys
//!
//! ```text
//! w d = [F, d] + [V + G[d], D],        d = D d + d D,
//! ```
//!
//! where `G[d]` is the change of the Kohn-Sham matrix caused by `d`. Both
//! relations hold order by order in `t`. The projector condition fixes the
//! occupied-occupied and virtual-virtual blocks of every coefficient from the
//! lower ones; its occupied-virtual blocks solve the paired response
//! equations of the reference geometry with a source collected from the
//! lower orders. With `V = 0` the same recursion is an eigenvalue problem
//! and gives the Taylor coefficients of an excitation energy. The
//! Tamm-Dancoff form keeps `d = (1 - D) d D` and projects the equation on
//! the same block.
//!
//! No displaced calculation enters: the derivatives of the integrals, of the
//! quadrature and of the reference solution are the coefficients of the
//! nuclear Taylor driver.

use super::geometry::{operator_series, GeometryTaylor};
use super::{add, matrix_product, molecular_taylor_parts, MolecularDirection};
use crate::basis::MolBasis;
use crate::dual::Scalar;
use crate::grid::MolecularGrid;
use crate::integrals::analytic_basis as jets;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::series::SeriesScalar;
use crate::response::tddft::{TdaSolver, TdaSpin, TdaState};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// The reference solution along one nuclear path.
pub(super) struct TransportedPath {
    pub retained: Option<super::precision::IntegerPath>,
    pub geometry: GeometryTaylor,
    /// Transport coefficients `B(t)` per spin (AO by orbital).
    pub b: [Vec<Matrix>; 2],
    /// Density coefficients in the transported frame.
    pub d: [Vec<Matrix>; 2],
    /// Kohn-Sham matrix coefficients in the transported frame.
    pub f: [Vec<Matrix>; 2],
    /// Density coefficients in the AO basis.
    pub p: [Vec<Matrix>; 2],
    pub n_occ: [usize; 2],
    pub unrestricted: bool,
    /// Energy coefficients of the reference state.
    pub energy: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Order-`n` coefficient of `Tr[O(t)^T X(t)]`.
fn contract(operator: &[Matrix], density: &[Matrix], n: usize) -> f64 {
    (0..=n)
        .map(|k| operator[k].frobenius_dot(&density[n - k]))
        .sum()
}

/// The response operator of the reference geometry on the occupied-virtual
/// amplitudes: the paired form on `(X + Y, X - Y)` or the Tamm-Dancoff
/// matrix, minus the frequency.
struct FrameOperator<'a, 'b> {
    solver: &'b TdaSolver<'a>,
    paired: bool,
    frequency: f64,
    /// Unit vector whose direction is shifted by one: the null vector of an
    /// eigenvalue problem.
    deflation: Option<&'b [f64]>,
}

impl FrameOperator<'_, '_> {
    fn image(&self, x: &[f64]) -> Result<Vec<f64>> {
        let n = self.solver.dim();
        let mut out = if self.paired {
            let sum = self.solver.apb_vec(&x[..n])?;
            let difference = self.solver.amb_vec(&x[n..])?;
            let mut out = vec![0.0; 2 * n];
            for k in 0..n {
                out[k] = sum[k] - self.frequency * x[n + k];
                out[n + k] = difference[k] - self.frequency * x[k];
            }
            out
        } else {
            let image = self.solver.a_vec(x)?;
            image
                .iter()
                .zip(x)
                .map(|(a, x)| a - self.frequency * x)
                .collect()
        };
        if let Some(unit) = self.deflation {
            let overlap = dot(unit, x);
            for (value, u) in out.iter_mut().zip(unit) {
                *value += overlap * u;
            }
        }
        Ok(out)
    }
}

impl LinearOperator for FrameOperator<'_, '_> {
    fn dimension(&self) -> usize {
        if self.paired {
            2 * self.solver.dim()
        } else {
            self.solver.dim()
        }
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::Hermitian
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| self.image(v))
    }
}

/// The response equations of one path in its transported frame.
struct Frame<'a, 'b> {
    path: &'b TransportedPath,
    functional: &'a ResolvedFunctional,
    solver: &'b TdaSolver<'a>,
    /// Transposed transport coefficients.
    transposed: [Vec<Matrix>; 2],
    /// Coefficients of the complement `1 - D(t)`.
    complement: [Vec<Matrix>; 2],
    paired: bool,
    /// Sign of the beta block of a spin-adapted restricted response.
    beta_sign: f64,
}

impl<'a, 'b> Frame<'a, 'b> {
    fn new(
        path: &'b TransportedPath,
        functional: &'a ResolvedFunctional,
        solver: &'b TdaSolver<'a>,
        paired: bool,
        beta_sign: f64,
    ) -> Self {
        let transposed: [Vec<Matrix>; 2] =
            std::array::from_fn(|s| path.b[s].iter().map(Matrix::transpose).collect());
        let complement: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
            path.d[s]
                .iter()
                .enumerate()
                .map(|(k, d)| {
                    let mut q = if k == 0 {
                        Matrix::identity(d.rows)
                    } else {
                        Matrix::zeros(d.rows, d.cols)
                    };
                    add(&mut q, d, -1.0);
                    q
                })
                .collect()
        });
        Self {
            path,
            functional,
            solver,
            transposed,
            complement,
            paired,
            beta_sign,
        }
    }

    fn spins(&self) -> usize {
        if self.path.unrestricted {
            2
        } else {
            1
        }
    }

    /// AO coefficients of a frame matrix series.
    fn to_ao(&self, s: usize, x: &[Matrix], n: usize) -> Vec<Matrix> {
        matrix_product(
            &matrix_product(&self.path.b[s], x, n),
            &self.transposed[s],
            n,
        )
    }

    /// Frame coefficients of an AO operator series.
    fn to_frame(&self, s: usize, x: &[Matrix], n: usize) -> Vec<Matrix> {
        matrix_product(
            &matrix_product(&self.transposed[s], x, n),
            &self.path.b[s],
            n,
        )
    }

    /// AO coefficients of the first-order density summed over the spins.
    fn total(&self, d1: &[Vec<Matrix>; 2], order: usize) -> Vec<Matrix> {
        let mut out = self.to_ao(0, &d1[0], order);
        let other = self.to_ao(1, &d1[1], order);
        for (a, b) in out.iter_mut().zip(&other) {
            add(a, b, 1.0);
        }
        out
    }

    /// Set the blocks of the order-`n` coefficient that the projector
    /// condition fixes from the lower orders. The coefficient must be zero on
    /// entry.
    fn constrain(&self, d1: &mut [Vec<Matrix>; 2], n: usize) {
        for s in 0..2 {
            let d = &self.path.d[s];
            let no = self.path.n_occ[s];
            let nmo = d[0].rows;
            let fixed = if self.paired {
                let mut c = Matrix::zeros(nmo, nmo);
                for k in 1..=n {
                    add(&mut c, &d[k].matmul_seq(&d1[s][n - k]), 1.0);
                    add(&mut c, &d1[s][n - k].matmul_seq(&d[k]), 1.0);
                }
                c
            } else {
                matrix_product(&matrix_product(&self.complement[s], &d1[s], n), d, n)[n].clone()
            };
            for i in 0..nmo {
                for j in 0..nmo {
                    let (io, jo) = (i < no, j < no);
                    d1[s][n][(i, j)] = if self.paired {
                        if io && jo {
                            -fixed[(i, j)]
                        } else if !io && !jo {
                            fixed[(i, j)]
                        } else {
                            0.0
                        }
                    } else if !io && jo {
                        0.0
                    } else {
                        fixed[(i, j)]
                    };
                }
            }
        }
    }

    /// Order-`n` coefficient of the response equation for the trial
    /// first-order density `d1`, with the frame operator series `v` and the
    /// frequency series `omega`.
    fn residual(
        &self,
        d1: &[Vec<Matrix>; 2],
        v: Option<&[Vec<Matrix>; 2]>,
        omega: &[f64],
        n: usize,
    ) -> Result<[Matrix; 2]> {
        let q: [Vec<Matrix>; 2] = std::array::from_fn(|s| self.to_ao(s, &d1[s], n));
        let g = self
            .path
            .geometry
            .evaluate_linear(self.functional, &self.path.p, &q, n)?;
        let mut out = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        for s in 0..2 {
            let mut w = self.to_frame(s, &g[s], n);
            if let Some(v) = v {
                for k in 0..=n {
                    add(&mut w[k], &v[s][k], 1.0);
                }
            }
            let f = &self.path.f[s];
            let d = &self.path.d[s];
            let nmo = d[0].rows;
            let mut r = Matrix::zeros(nmo, nmo);
            if self.paired {
                for k in 0..=n {
                    add(&mut r, &f[k].matmul_seq(&d1[s][n - k]), 1.0);
                    add(&mut r, &d1[s][n - k].matmul_seq(&f[k]), -1.0);
                    add(&mut r, &w[k].matmul_seq(&d[n - k]), 1.0);
                    add(&mut r, &d[n - k].matmul_seq(&w[k]), -1.0);
                }
            } else {
                let mut inner = matrix_product(f, &d1[s], n);
                let reversed = matrix_product(&d1[s], f, n);
                for k in 0..=n {
                    add(&mut inner[k], &reversed[k], -1.0);
                    add(&mut inner[k], &w[k], 1.0);
                }
                r = matrix_product(&matrix_product(&self.complement[s], &inner, n), d, n)[n]
                    .clone();
            }
            for k in 0..=n {
                add(&mut r, &d1[s][n - k], -omega[k]);
            }
            out[s] = r;
        }
        Ok(out)
    }

    /// Occupied-virtual source of the amplitude equations from a residual:
    /// its antisymmetric and symmetric parts for the paired form.
    fn source(&self, r: &[Matrix; 2]) -> Vec<f64> {
        let dim = self.solver.dim();
        let mut out = vec![0.0; if self.paired { 2 * dim } else { dim }];
        let mut off = 0;
        for s in 0..self.spins() {
            let no = self.path.n_occ[s];
            let nv = r[s].rows - no;
            for a in 0..nv {
                for i in 0..no {
                    let k = off + a * no + i;
                    let lower = r[s][(no + a, i)];
                    let upper = r[s][(i, no + a)];
                    if self.paired {
                        out[k] = 0.5 * (lower - upper);
                        out[dim + k] = 0.5 * (lower + upper);
                    } else {
                        out[k] = lower;
                    }
                }
            }
            off += no * nv;
        }
        out
    }

    /// Store the occupied-virtual amplitudes `u` in the order-`n`
    /// coefficient.
    fn set_amplitudes(&self, d1: &mut [Vec<Matrix>; 2], n: usize, u: &[f64]) {
        let dim = self.solver.dim();
        let mut off = 0;
        for s in 0..self.spins() {
            let no = self.path.n_occ[s];
            let nv = d1[s][n].rows - no;
            for a in 0..nv {
                for i in 0..no {
                    let k = off + a * no + i;
                    if self.paired {
                        d1[s][n][(no + a, i)] = u[k] + u[dim + k];
                        d1[s][n][(i, no + a)] = u[k] - u[dim + k];
                    } else {
                        d1[s][n][(no + a, i)] = u[k];
                    }
                }
            }
            off += no * nv;
        }
        if self.spins() == 1 {
            let mut other = d1[0][n].clone();
            for value in other.as_mut_slice() {
                *value *= self.beta_sign;
            }
            d1[1][n] = other;
        }
    }

    fn solve(
        &self,
        rhs: &[f64],
        frequency: f64,
        deflation: Option<&[f64]>,
        config: &ResponseSolverConfig,
    ) -> Result<(Vec<f64>, f64)> {
        if rhs.iter().all(|v| *v == 0.0) {
            return Ok((vec![0.0; rhs.len()], 0.0));
        }
        let operator = FrameOperator {
            solver: self.solver,
            paired: self.paired,
            frequency,
            deflation,
        };
        let rhs: Vec<Complex> = rhs.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let solution = krylov::solve(&operator, &rhs, None, config)?;
        Ok((
            solution.amplitudes.iter().map(|z| z.re).collect(),
            solution.relative_residual,
        ))
    }

    fn empty(&self) -> [Matrix; 2] {
        std::array::from_fn(|s| {
            let nmo = self.path.d[s][0].rows;
            Matrix::zeros(nmo, nmo)
        })
    }

    /// First-order density coefficients of the response to the AO operator
    /// series `v` at the frequency `omega`, with the solver residuals.
    fn driven(
        &self,
        v: &[Matrix],
        omega: f64,
        order: usize,
        config: &ResponseSolverConfig,
    ) -> Result<([Vec<Matrix>; 2], Vec<f64>)> {
        let frame: [Vec<Matrix>; 2] = std::array::from_fn(|s| self.to_frame(s, v, order));
        let mut frequencies = vec![0.0; order + 1];
        frequencies[0] = omega;
        let mut d1: [Vec<Matrix>; 2] = [Vec::new(), Vec::new()];
        let mut residuals = Vec::with_capacity(order + 1);
        for n in 0..=order {
            for (s, zero) in self.empty().into_iter().enumerate() {
                d1[s].push(zero);
            }
            self.constrain(&mut d1, n);
            let r = self.residual(&d1, Some(&frame), &frequencies, n)?;
            let rhs: Vec<f64> = self.source(&r).iter().map(|v| -v).collect();
            let (u, residual) = self.solve(&rhs, omega, None, config)?;
            self.set_amplitudes(&mut d1, n, &u);
            residuals.push(residual);
        }
        Ok((d1, residuals))
    }

    /// Taylor coefficients of the excitation energy of the state with the
    /// reference amplitudes `start` and energy `omega`, the solver residuals
    /// and the relative residual of the reference eigenvalue equation.
    fn excited(
        &self,
        start: &[f64],
        omega: f64,
        order: usize,
        config: &ResponseSolverConfig,
    ) -> Result<(Vec<f64>, Vec<f64>, f64)> {
        let dim = self.solver.dim();
        // The metric partner of the null vector: (X - Y, X + Y) for the
        // paired form, the vector itself for the Tamm-Dancoff form.
        let partner: Vec<f64> = if self.paired {
            start[dim..].iter().chain(&start[..dim]).copied().collect()
        } else {
            start.to_vec()
        };
        let norm = dot(start, start).sqrt();
        let metric = dot(start, &partner);
        if !(norm > 0.0) || metric.abs() <= 1e-12 * norm * norm {
            return Err(DftError::InvalidInput(
                "the excited state has no normalizable response vector".into(),
            ));
        }
        let unit: Vec<f64> = start.iter().map(|v| v / norm).collect();
        let mut frequencies = vec![omega];
        let mut d1: [Vec<Matrix>; 2] = [Vec::new(), Vec::new()];
        for (s, zero) in self.empty().into_iter().enumerate() {
            d1[s].push(zero);
        }
        self.set_amplitudes(&mut d1, 0, start);
        let reference = {
            let r = self.residual(&d1, None, &frequencies, 0)?;
            let source = self.source(&r);
            dot(&source, &source).sqrt() / norm
        };
        let mut residuals = Vec::with_capacity(order);
        for n in 1..=order {
            for (s, zero) in self.empty().into_iter().enumerate() {
                d1[s].push(zero);
            }
            self.constrain(&mut d1, n);
            frequencies.push(0.0);
            let r = self.residual(&d1, None, &frequencies, n)?;
            let source = self.source(&r);
            let shift = dot(start, &source) / metric;
            frequencies[n] = shift;
            let rhs: Vec<f64> = source
                .iter()
                .zip(&partner)
                .map(|(r, w)| shift * w - r)
                .collect();
            let (u, residual) = self.solve(&rhs, omega, Some(&unit), config)?;
            self.set_amplitudes(&mut d1, n, &u);
            residuals.push(residual);
        }
        Ok((frequencies, residuals, reference))
    }
}

/// Taylor coefficients along a nuclear direction of the electric dipole
/// polarizability, of the optical rotation tensor and of the electric
/// dipole - electric quadrupole polarizability at one frequency.
#[derive(Clone, Debug)]
pub struct OpticalTensorPath {
    /// Photon energy, Hartree.
    pub frequency: f64,
    /// Origin of the magnetic dipole and of the quadrupole, Bohr.
    pub origin: [f64; 3],
    /// Whether the tensors are in the velocity gauge.
    pub velocity: bool,
    /// `alpha[n][a][b]`: observable axis `a`, field axis `b`.
    pub polarizability: Vec<[[f64; 3]; 3]>,
    /// `G'[n][a][b]`: field axis `a`, magnetic-dipole axis `b`.
    pub rotation: Vec<[[f64; 3]; 3]>,
    /// `A[n][a][b][c]`: field axis `a`, traceless quadrupole component `bc`.
    pub quadrupole: Vec<[[[f64; 3]; 3]; 3]>,
    pub residuals: Vec<f64>,
    pub converged: bool,
    pub warnings: Vec<String>,
}

/// Optical tensors of the paired response along the nuclear path
/// `R(t) = R(0) + t direction`, about the origin `O`, which stays fixed in
/// space.
///
/// Length gauge: with `H' = +E.r` and the first-order density `d` of a field
/// along `a`, `alpha[b][a] = -Tr[r_b d]`, `G'[a][b] = -Tr[L_b^T d] / 2` with
/// `L = (r - O) x grad`, and `A[a][bc] = -Tr[(3 r_b r_c - r^2 delta_bc) d] / 2`.
///
/// Velocity gauge: `d` is the difference of the first-order densities of the
/// momentum `p_a = -i grad_a` at the frequency and at zero frequency, and
/// with the antisymmetric operators `grad_b`, `L_b` and
/// `T_bc = r_b grad_c + r_c grad_b + delta_bc`,
/// `alpha[b][a] = Tr[grad_b^T d] / w^2`, `G'[a][b] = -Tr[L_b^T d] / (2 w)`,
/// `M[a][bc] = Tr[T_bc^T d] / w^2` and `A = 3 M / 2 - delta_bc Tr M / 2`.
/// The origin enters `L` and `T` only, through `-d x grad` and
/// `-d_b grad_c - d_c grad_b`, so the tensors change with the origin exactly
/// as in a complete basis and their rotational invariants do not depend on it.
#[allow(clippy::too_many_arguments)]
pub fn optical_tensor_path<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    direction: &[[f64; 3]],
    order: usize,
    omega: f64,
    origin: Vec3,
    velocity: bool,
    config: &ResponseSolverConfig,
) -> Result<OpticalTensorPath> {
    if !omega.is_finite() || omega < 0.0 || (velocity && omega == 0.0) {
        return Err(DftError::InvalidInput(
            "the optical tensors need a finite non-negative photon energy, and a positive one \
             in the velocity gauge"
                .into(),
        ));
    }
    let (report, path) = molecular_taylor_parts(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &MolecularDirection::nuclear(direction.to_vec()),
        order,
        config,
    )?;
    let spin = if scf.unrestricted {
        TdaSpin::SpinResolved
    } else {
        TdaSpin::Singlet
    };
    let solver = TdaSolver::new(mol, mb, functional, grid, scf, opts, spin)?;
    let frame = Frame::new(&path, functional, &solver, true, 1.0);
    let center = origin.to_array();
    let mut angular = Vec::with_capacity(3);
    for axis in 0..3 {
        angular.push(operator_series(mol, mb, direction, order, true, |x, y| {
            Ok(jets::angular_momentum(x, y, axis, center.map(SeriesScalar::from_f64))?.values)
        })?);
    }
    // Second moments in the length gauge, their velocity form otherwise.
    let mut moments: Vec<Vec<Matrix>> = Vec::with_capacity(9);
    for b in 0..3 {
        for c in 0..3 {
            if c < b {
                let copy = moments[3 * c + b].clone();
                moments.push(copy);
                continue;
            }
            let series = if velocity {
                operator_series(mol, mb, direction, order, true, |x, y| {
                    Ok(
                        jets::velocity_quadrupole(x, y, b, c, center.map(SeriesScalar::from_f64))?
                            .values,
                    )
                })?
            } else {
                let mut power = [0_usize; 3];
                power[b] += 1;
                power[c] += 1;
                operator_series(mol, mb, direction, order, false, |x, y| {
                    Ok(jets::moment(x, y, power, center.map(SeriesScalar::from_f64))?.values)
                })?
            };
            moments.push(series);
        }
    }
    let mut gradient = Vec::new();
    if velocity {
        for axis in 0..3 {
            gradient.push(operator_series(mol, mb, direction, order, true, |x, y| {
                Ok(jets::gradient(x, y, axis)?.values)
            })?);
        }
    }
    let mut polarizability = vec![[[0.0; 3]; 3]; order + 1];
    let mut rotation = vec![[[0.0; 3]; 3]; order + 1];
    let mut quadrupole = vec![[[[0.0; 3]; 3]; 3]; order + 1];
    let mut residuals = Vec::new();
    for a in 0..3 {
        // First-order density of a field along `a`: the response to the
        // position operator, or the difference of the responses to the
        // momentum `i (-grad_a)` at the frequency and at zero frequency.
        let total = if velocity {
            let source: Vec<Matrix> = gradient[a]
                .iter()
                .map(|m| {
                    let mut v = m.clone();
                    for x in v.as_mut_slice() {
                        *x = -*x;
                    }
                    v
                })
                .collect();
            let (at_frequency, solved) = frame.driven(&source, omega, order, config)?;
            residuals.extend(solved);
            let (at_zero, solved) = frame.driven(&source, 0.0, order, config)?;
            residuals.extend(solved);
            let mut total = frame.total(&at_frequency, order);
            for (t, z) in total.iter_mut().zip(&frame.total(&at_zero, order)) {
                add(t, z, -1.0);
            }
            total
        } else {
            let (d1, solved) = frame.driven(&path.geometry.dipoles[a], omega, order, config)?;
            residuals.extend(solved);
            frame.total(&d1, order)
        };
        let scale = if velocity {
            1.0 / (omega * omega)
        } else {
            -1.0
        };
        for n in 0..=order {
            for b in 0..3 {
                if velocity {
                    polarizability[n][b][a] = scale * contract(&gradient[b], &total, n);
                    rotation[n][a][b] = -0.5 * contract(&angular[b], &total, n) / omega;
                } else {
                    polarizability[n][b][a] = -contract(&path.geometry.dipoles[b], &total, n);
                    rotation[n][a][b] = -0.5 * contract(&angular[b], &total, n);
                }
            }
            let second: [[f64; 3]; 3] = std::array::from_fn(|b| {
                std::array::from_fn(|c| scale * contract(&moments[3 * b + c], &total, n))
            });
            let trace = second[0][0] + second[1][1] + second[2][2];
            for b in 0..3 {
                for c in 0..3 {
                    let isotropic = if b == c { 0.5 * trace } else { 0.0 };
                    quadrupole[n][a][b][c] = 1.5 * second[b][c] - isotropic;
                }
            }
        }
    }
    let acceptance = config.acceptance();
    let converged = report.converged && residuals.iter().all(|r| *r <= acceptance);
    Ok(OpticalTensorPath {
        frequency: omega,
        origin: center,
        velocity,
        polarizability,
        rotation,
        quadrupole,
        residuals,
        converged,
        warnings: report.warnings,
    })
}

/// Taylor coefficients of the reference energy and of one excitation energy
/// along a nuclear direction.
#[derive(Clone, Debug)]
pub struct ExcitationPath {
    /// Energy coefficients of the reference state, Hartree.
    pub ground: Vec<f64>,
    /// Excitation energy coefficients; entry 0 is the vertical energy.
    pub excitation: Vec<f64>,
    /// Relative residual of the eigenvalue equation of the supplied state in
    /// the transported frame.
    pub reference_residual: f64,
    pub residuals: Vec<f64>,
    pub converged: bool,
    pub warnings: Vec<String>,
}

/// Amplitudes of `state` in the solver ordering: `X` for a Tamm-Dancoff
/// state, `(X + Y, X - Y)` for a paired state.
fn start_vector(solver: &TdaSolver, state: &TdaState) -> Result<Vec<f64>> {
    let spins = if state.spin == TdaSpin::SpinResolved {
        2
    } else {
        1
    };
    let mut amplitudes = Vec::with_capacity(solver.dim());
    for s in 0..spins {
        let x = &state.x[s];
        for a in 0..x.rows {
            for i in 0..x.cols {
                amplitudes.push(x[(a, i)]);
            }
        }
    }
    if amplitudes.len() != solver.dim() {
        return Err(DftError::InvalidInput(
            "the excited state does not belong to this reference".into(),
        ));
    }
    if state.tamm_dancoff {
        return Ok(amplitudes);
    }
    if !(state.energy > 0.0) {
        return Err(DftError::InvalidInput(
            "a paired excited state needs a positive excitation energy".into(),
        ));
    }
    // (A + B)(X + Y) = omega (X - Y).
    let image = solver.apb_vec(&amplitudes)?;
    Ok(amplitudes
        .iter()
        .copied()
        .chain(image.iter().map(|v| v / state.energy))
        .collect())
}

/// Excitation energy of `state` along `R(t) = R(0) + t direction` through
/// `order`, by perturbation theory of the response eigenvalue problem in the
/// transported frame. The state must be non-degenerate.
#[allow(clippy::too_many_arguments)]
pub fn excitation_path<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    state: &TdaState,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ExcitationPath> {
    let (report, path) = molecular_taylor_parts(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &MolecularDirection::nuclear(direction.to_vec()),
        order,
        config,
    )?;
    let solver = TdaSolver::new(mol, mb, functional, grid, scf, opts, state.spin)?;
    let start = start_vector(&solver, state)?;
    let beta_sign = if state.spin == TdaSpin::Triplet {
        -1.0
    } else {
        1.0
    };
    let frame = Frame::new(&path, functional, &solver, !state.tamm_dancoff, beta_sign);
    let (excitation, residuals, reference_residual) =
        frame.excited(&start, state.energy, order, config)?;
    let acceptance = config.acceptance();
    let converged = report.converged && residuals.iter().all(|r| *r <= acceptance);
    Ok(ExcitationPath {
        ground: path.energy.clone(),
        excitation,
        reference_residual,
        residuals,
        converged,
        warnings: report.warnings,
    })
}

/// Cartesian gradient and Hessian of the total energy of an excited state.
#[derive(Clone, Debug)]
pub struct ExcitedStateHessian {
    /// Gradient of the excited-state energy, Hartree/Bohr, index
    /// `3 * atom + axis`.
    pub gradient: Vec<f64>,
    /// Hessian of the excited-state energy, Hartree/Bohr^2.
    pub hessian: Matrix,
    /// Its reference-state part.
    pub ground_hessian: Matrix,
    /// Its excitation-energy part.
    pub excitation_hessian: Matrix,
    pub reference_residual: f64,
    pub residual: f64,
    pub converged: bool,
    /// Number of nuclear paths evaluated.
    pub paths: usize,
    pub warnings: Vec<String>,
}

/// Second derivatives of `E_reference + omega` for `state`. The second-order
/// path coefficients `c(u)` of every Cartesian direction and of every pair
/// sum give `H_ii = 2 c(e_i)` and `H_ij = c(e_i + e_j) - c(e_i) - c(e_j)`.
#[allow(clippy::too_many_arguments)]
pub fn excited_state_hessian<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    state: &TdaState,
    config: &ResponseSolverConfig,
) -> Result<ExcitedStateHessian> {
    let n = 3 * mol.len();
    let unit = |index: usize| -> Vec<[f64; 3]> {
        let mut direction = vec![[0.0; 3]; mol.len()];
        direction[index / 3][index % 3] = 1.0;
        direction
    };
    let mut gradient = vec![0.0; n];
    let mut ground = Matrix::zeros(n, n);
    let mut excitation = Matrix::zeros(n, n);
    let mut second = vec![[0.0_f64; 2]; n];
    let mut reference_residual = 0.0_f64;
    let mut residual = 0.0_f64;
    let mut converged = true;
    let mut warnings: Vec<String> = Vec::new();
    let mut paths = 0_usize;
    let mut note = |path: &ExcitationPath| {
        paths += 1;
        converged &= path.converged;
        reference_residual = reference_residual.max(path.reference_residual);
        residual = path.residuals.iter().copied().fold(residual, f64::max);
        for warning in &path.warnings {
            if !warnings.contains(warning) {
                warnings.push(warning.clone());
            }
        }
    };
    for i in 0..n {
        let path = excitation_path(
            mol,
            mb,
            functional,
            grid,
            scf,
            opts,
            state,
            &unit(i),
            2,
            config,
        )?;
        note(&path);
        gradient[i] = path.ground[1] + path.excitation[1];
        second[i] = [path.ground[2], path.excitation[2]];
        ground[(i, i)] = 2.0 * path.ground[2];
        excitation[(i, i)] = 2.0 * path.excitation[2];
    }
    for i in 0..n {
        for j in 0..i {
            let mut direction = unit(i);
            direction[j / 3][j % 3] += 1.0;
            let path = excitation_path(
                mol, mb, functional, grid, scf, opts, state, &direction, 2, config,
            )?;
            note(&path);
            let g = path.ground[2] - second[i][0] - second[j][0];
            let e = path.excitation[2] - second[i][1] - second[j][1];
            ground[(i, j)] = g;
            ground[(j, i)] = g;
            excitation[(i, j)] = e;
            excitation[(j, i)] = e;
        }
    }
    let mut hessian = ground.clone();
    add(&mut hessian, &excitation, 1.0);
    Ok(ExcitedStateHessian {
        gradient,
        hessian,
        ground_hessian: ground,
        excitation_hessian: excitation,
        reference_residual,
        residual,
        converged,
        paths,
        warnings,
    })
}
