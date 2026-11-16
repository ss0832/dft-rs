// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static response to uniform electric and magnetic fields in a basis that
//! may depend on the magnetic field.
//!
//! With London orbitals the overlap, the one-electron matrices and the
//! electron repulsion integrals are power series in the path parameter of
//! `E(t) = t e`, `B(t) = t b` (`integrals::london`), and the result does not
//! depend on the gauge origin. The reference orbitals are transported so
//! that `B(t)^H S(t) B(t) = 1`; in that frame the complex Hermitian density
//! coefficients follow from `D^2 = D` and `[F, D] = 0` exactly as in the
//! common-origin recursion of `magnetic_taylor`: the real occupied-virtual
//! part from the orbital Hessian, the imaginary part from the magnetic
//! Hessian.
//!
//! The semilocal functional is evaluated on the series of the density, of
//! its gradient and of the explicitly selected magnetic kinetic density.
//! The legacy nuclear provider uses mechanical tau; the post-SCF wavefunction
//! provider uses `tau - |j|^2/(2 rho)`. The field-free two-electron integrals
//! act through the integral-direct engines of the self-consistent field and
//! their field coefficients are contracted quartet by quartet, so no
//! integral tensor is stored.

use super::complex::Complex;
use super::dynamic::CasidaOperator;
use super::krylov::{self, LinearOperator, OperatorStructure};
use super::magnetic_taylor::{exchange, Hypermagnetizability};
use super::mixed::polarized_mixed;
use super::tddft::{TdaSolver, TdaSpin};
use super::{Frequency, ResponseSolver, ResponseSolverConfig};
mod learned;
mod nonlocal;
mod retained;
use crate::basis::{n_sph, MolBasis};
use crate::grid::{eval::eval_aos_full, MolecularGrid};
use crate::integrals::london::{
    contracted_repulsion_series, one_electron_series, ComplexMatrix, FieldPath, OneElectronSeries,
};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{JkEngine, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::{Family, MagneticKineticDensity, ResolvedFunctional, XcTaylorInput};
use crate::{DftError, Result};
use learned::LearnedFieldProvider;
pub(crate) use learned::LearnedFiniteField;
use nonlocal::NonlocalFieldProvider;
pub(crate) use nonlocal::NonlocalFiniteField;
use rayon::prelude::*;

/// Exact finite AO phases/gradients shared by the global learned and
/// nonlocal finite-field energy/potential providers.
fn finite_point_series(points: &[Point]) -> Vec<PointSeries> {
    points
        .iter()
        .map(|p| {
            let mut w = Vec::new();
            let mut gw = Vec::new();
            let mut pi = Vec::new();
            for mu in 0..p.phi.len() {
                let e = (p.theta[mu].cos(), -p.theta[mu].sin());
                w.push(vec![mul((p.phi[mu], 0.0), e)]);
                gw.push(std::array::from_fn(|a| {
                    vec![mul((p.grad[mu][a], -p.slope[mu][a] * p.phi[mu]), e)]
                }));
                pi.push(std::array::from_fn(|a| {
                    vec![mul((p.gauge[mu][a] * p.phi[mu], -p.grad[mu][a]), e)]
                }));
            }
            PointSeries { w, gw, pi }
        })
        .collect()
}

type Cm = ComplexMatrix;
type Cx = (f64, f64);

#[cfg(test)]
#[path = "field_path_tests.rs"]
mod complex_reference_tests;

#[cfg(test)]
#[path = "field_current_tests.rs"]
mod current_invariant_tests;

fn mul(a: Cx, b: Cx) -> Cx {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn conj(a: Cx) -> Cx {
    (a.0, -a.1)
}
fn add(target: &mut Matrix, source: &Matrix, weight: f64) {
    for (a, b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *a += weight * b;
    }
}
pub(crate) fn cadd(target: &mut Cm, source: &Cm, weight: f64) {
    add(&mut target.re, &source.re, weight);
    add(&mut target.im, &source.im, weight);
}
fn cmul(a: &Cm, b: &Cm) -> Cm {
    a.matmul_seq(b)
}
pub(crate) fn adjoint(a: &Cm) -> Cm {
    let mut im = a.im.transpose();
    for value in im.as_mut_slice() {
        *value = -*value;
    }
    Cm {
        re: a.re.transpose(),
        im,
    }
}
fn from_real(m: &Matrix) -> Cm {
    Cm {
        re: m.clone(),
        im: Matrix::zeros(m.rows, m.cols),
    }
}
/// Cauchy product of two matrix series through `order`.
pub(crate) fn series_product(a: &[Cm], b: &[Cm], order: usize) -> Vec<Cm> {
    (0..=order)
        .map(|n| {
            let mut v = Cm::zeros(a[0].re.rows, b[0].re.cols);
            for k in 0..=n {
                cadd(&mut v, &cmul(&a[k], &b[n - k]), 1.0);
            }
            v
        })
        .collect()
}
/// `Re Tr[a b]`.
fn trace(a: &Cm, b: &Cm) -> f64 {
    a.real_trace_product(b)
}
fn largest(a: &Cm) -> f64 {
    a.re.as_slice()
        .iter()
        .chain(a.im.as_slice())
        .fold(0.0_f64, |m, x| m.max(x.abs()))
}
fn real_product(a: &[f64], b: &[f64], width: usize) -> Vec<f64> {
    (0..width)
        .map(|n| (0..=n).map(|k| a[k] * b[n - k]).sum())
        .collect()
}
/// `(b x v) / 2`.
fn half_cross(b: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        0.5 * (b[1] * v[2] - b[2] * v[1]),
        0.5 * (b[2] * v[0] - b[0] * v[2]),
        0.5 * (b[0] * v[1] - b[1] * v[0]),
    ]
}

/// Basis functions at one quadrature point with the data of their phases.
struct Point {
    weight: f64,
    phi: Vec<f64>,
    grad: Vec<[f64; 3]>,
    /// Phase per unit path parameter of every function at this point.
    theta: Vec<f64>,
    /// Gradient of that phase.
    slope: Vec<[f64; 3]>,
    /// Vector potential per unit path parameter in the gauge of every
    /// function.
    gauge: Vec<[f64; 3]>,
}

/// Coefficients of exp(-i theta(t)) for an arbitrary real phase series.
pub(crate) fn phase_exponential(theta: &[f64]) -> Vec<Cx> {
    let mut phase = vec![(theta[0].cos(), -theta[0].sin())];
    for n in 1..theta.len() {
        let mut value = (0.0, 0.0);
        for k in 1..=n {
            let term = mul((0.0, -(k as f64) * theta[k]), phase[n - k]);
            value.0 += term.0 / n as f64;
            value.1 += term.1 / n as f64;
        }
        phase.push(value);
    }
    phase
}

/// Taylor coefficients `[function][order]` of the phased functions, of their
/// gradients and of `(-i grad + A)` applied to them.
pub(crate) struct PointSeries {
    pub(crate) w: Vec<Vec<Cx>>,
    pub(crate) gw: Vec<[Vec<Cx>; 3]>,
    pub(crate) pi: Vec<[Vec<Cx>; 3]>,
}

fn point_series(point: &Point, width: usize) -> PointSeries {
    let nao = point.phi.len();
    let mut w = Vec::with_capacity(nao);
    let mut gw = Vec::with_capacity(nao);
    let mut pi = Vec::with_capacity(nao);
    for mu in 0..nao {
        // exp(-i t theta) = sum_k (-i theta)^k / k! t^k.
        let mut theta = vec![0.0; width];
        if width > 1 {
            theta[1] = point.theta[mu];
        }
        let e = phase_exponential(&theta);
        let phi = point.phi[mu];
        w.push(
            e.iter()
                .map(|z| (phi * z.0, phi * z.1))
                .collect::<Vec<Cx>>(),
        );
        let gradient: [Vec<Cx>; 3] = std::array::from_fn(|a| {
            (0..width)
                .map(|k| {
                    let mut value = (point.grad[mu][a] * e[k].0, point.grad[mu][a] * e[k].1);
                    if k > 0 {
                        let extra = mul((0.0, -point.slope[mu][a] * phi), e[k - 1]);
                        value = (value.0 + extra.0, value.1 + extra.1);
                    }
                    value
                })
                .collect()
        });
        let momentum: [Vec<Cx>; 3] = std::array::from_fn(|a| {
            (0..width)
                .map(|k| {
                    let mut value = mul((0.0, -point.grad[mu][a]), e[k]);
                    if k > 0 {
                        let factor = point.gauge[mu][a] * phi;
                        value = (value.0 + factor * e[k - 1].0, value.1 + factor * e[k - 1].1);
                    }
                    value
                })
                .collect()
        });
        gw.push(gradient);
        pi.push(momentum);
    }
    PointSeries { w, gw, pi }
}

/// Series of the density, of its gradient and of the kinetic energy density
/// of one spin for the AO density series `p`.
type PointDensities = (Vec<f64>, [Vec<f64>; 3], Vec<f64>, [Vec<f64>; 3]);
fn point_density(
    series: &PointSeries,
    p: &[Cm],
    width: usize,
    gga: bool,
    mgga: bool,
    want_current: bool,
) -> PointDensities {
    let nao = series.w.len();
    let mut rho = vec![0.0; width];
    let mut grad: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; width]);
    let mut tau = vec![0.0; width];
    let mut current: [Vec<f64>; 3] = std::array::from_fn(|_| {
        if want_current {
            vec![0.0; width]
        } else {
            Vec::new()
        }
    });
    for nu in 0..nao {
        // y[k] = sum_mu sum_j D_j[mu, nu] w_mu[k - j], and the same with the
        // momentum functions.
        let mut y: Vec<Cx> = vec![(0.0, 0.0); width];
        let mut z: [Vec<Cx>; 3] = std::array::from_fn(|_| vec![(0.0, 0.0); width]);
        for mu in 0..nao {
            for j in 0..width {
                let d = (p[j].re[(mu, nu)], p[j].im[(mu, nu)]);
                if d.0 == 0.0 && d.1 == 0.0 {
                    continue;
                }
                for k in j..width {
                    let v = mul(d, series.w[mu][k - j]);
                    y[k] = (y[k].0 + v.0, y[k].1 + v.1);
                    if mgga {
                        for a in 0..3 {
                            let v = mul(d, series.pi[mu][a][k - j]);
                            z[a][k] = (z[a][k].0 + v.0, z[a][k].1 + v.1);
                        }
                    }
                }
            }
        }
        for k in 0..width {
            for j in 0..=k {
                rho[k] += mul(y[j], conj(series.w[nu][k - j])).0;
                if gga {
                    for a in 0..3 {
                        grad[a][k] += 2.0 * mul(y[j], conj(series.gw[nu][a][k - j])).0;
                    }
                }
                if mgga {
                    for a in 0..3 {
                        tau[k] += 0.5 * mul(z[a][j], conj(series.pi[nu][a][k - j])).0;
                        if want_current {
                            current[a][k] += mul(z[a][j], conj(series.w[nu][k - j])).0;
                        }
                    }
                }
            }
        }
    }
    (rho, grad, tau, current)
}

/// Weighted exchange-correlation energy series of one point through order
/// `n` and, on request, the order-`n` coefficient of its matrices per spin
/// as flat real and imaginary arrays.
pub(crate) type PointMatrices = [(Vec<f64>, Vec<f64>); 2];

fn point_xc(
    point: &Point,
    functional: &ResolvedFunctional,
    p: &[Vec<Cm>; 2],
    n: usize,
    want_matrix: bool,
    model: MagneticKineticDensity,
) -> Result<(Vec<f64>, Option<PointMatrices>)> {
    let series = point_series(point, n + 1);
    let mut weights = vec![0.0; n + 1];
    weights[0] = point.weight;
    point_xc_series_model(&series, &weights, functional, p, n, want_matrix, model)
}

/// Semilocal complex AO energy/potential series with moving quadrature.
pub(crate) fn point_xc_series(
    series: &PointSeries,
    weights: &[f64],
    functional: &ResolvedFunctional,
    p: &[Vec<Cm>; 2],
    n: usize,
    want_matrix: bool,
) -> Result<(Vec<f64>, Option<PointMatrices>)> {
    point_xc_series_model(
        series,
        weights,
        functional,
        p,
        n,
        want_matrix,
        MagneticKineticDensity::Mechanical,
    )
}

#[allow(clippy::too_many_arguments)]
struct KineticPullback {
    velocities: [[Vec<f64>; 3]; 2],
    density_chain: [Vec<f64>; 2],
    active: bool,
}
struct PointDensitySeries {
    variables: [Vec<f64>; 7],
    gradient: [[Vec<f64>; 3]; 2],
    kinetic: KineticPullback,
}
struct PointPotentials<'a> {
    rho: [&'a [f64]; 2],
    gradient: [[&'a [f64]; 3]; 2],
    tau: [&'a [f64]; 2],
}

fn density_fields(
    series: &PointSeries,
    p: &[Vec<Cm>; 2],
    n: usize,
    gga: bool,
    mgga: bool,
    corrected: bool,
) -> Result<PointDensitySeries> {
    let width = n + 1;
    let (rho_a, grad_a, tau_a, current_a) =
        point_density(series, &p[0], width, gga, mgga, corrected);
    let (rho_b, grad_b, tau_b, current_b) =
        point_density(series, &p[1], width, gga, mgga, corrected);
    let currents = [current_a, current_b];
    let grads = [&grad_a, &grad_b];
    let mut variables: [Vec<f64>; 7] = std::array::from_fn(|_| vec![0.0; width]);
    variables[0] = rho_a;
    variables[1] = rho_b;
    variables[5] = tau_a;
    variables[6] = tau_b;
    for index in [0, 1, 5, 6] {
        variables[index][0] = variables[index][0].max(0.0);
    }
    let mut velocities: [[Vec<f64>; 3]; 2] = std::array::from_fn(|_| {
        std::array::from_fn(|_| {
            if corrected {
                vec![0.0; width]
            } else {
                Vec::new()
            }
        })
    });
    let mut density_chain: [Vec<f64>; 2] = std::array::from_fn(|_| {
        if corrected {
            vec![0.0; width]
        } else {
            Vec::new()
        }
    });
    if corrected {
        for spin in 0..2 {
            if variables[spin][0] <= 0.0 {
                if currents[spin].iter().flatten().any(|&x| x != 0.0) {
                    return Err(DftError::InvalidInput("a current-invariant kinetic-density path needs a positive reference density or an exactly inactive spin channel".into()));
                }
                continue;
            }
            for axis in 0..3 {
                // Direct quotient recurrence avoids forming 1/rho, which
                // can overflow in a physically regular far-tail current.
                for k in 0..width {
                    velocities[spin][axis][k] = (currents[spin][axis][k]
                        - (1..=k)
                            .map(|j| variables[spin][j] * velocities[spin][axis][k - j])
                            .sum::<f64>())
                        / variables[spin][0];
                }
                let correction =
                    real_product(&currents[spin][axis], &velocities[spin][axis], width);
                let squared = real_product(&velocities[spin][axis], &velocities[spin][axis], width);
                for k in 0..width {
                    variables[5 + spin][k] -= 0.5 * correction[k];
                    density_chain[spin][k] += 0.5 * squared[k];
                }
            }
        }
    }
    if gga {
        for (ix, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            for a in 0..3 {
                let v = real_product(&grads[s][a], &grads[t][a], width);
                for k in 0..width {
                    variables[2 + ix][k] += v[k];
                }
            }
        }
    }
    Ok(PointDensitySeries {
        variables,
        gradient: [grad_a, grad_b],
        kinetic: KineticPullback {
            velocities,
            density_chain,
            active: corrected,
        },
    })
}

fn point_pullback(
    series: &PointSeries,
    weights: &[f64],
    potentials: &PointPotentials<'_>,
    kinetic: &KineticPullback,
    n: usize,
    gga: bool,
    mgga: bool,
) -> PointMatrices {
    let width = n + 1;
    let nao = series.w.len();
    let corrected = kinetic.active;
    let velocities = &kinetic.velocities;
    let density_chain = &kinetic.density_chain;
    let mut out: PointMatrices =
        std::array::from_fn(|_| (vec![0.0; nao * nao], vec![0.0; nao * nao]));
    for s in 0..2 {
        let b = &potentials.gradient[s];
        let v_rho: std::borrow::Cow<'_, [f64]> = if corrected {
            let density_extra = real_product(potentials.tau[s], &density_chain[s], width);
            std::borrow::Cow::Owned(
                potentials.rho[s]
                    .iter()
                    .zip(density_extra)
                    .map(|(a, b)| a + b)
                    .collect(),
            )
        } else {
            std::borrow::Cow::Borrowed(potentials.rho[s])
        };
        let v_tau = potentials.tau[s];
        let v_current: [Vec<f64>; 3] = std::array::from_fn(|a| {
            if corrected {
                real_product(v_tau, &velocities[s][a], width)
                    .into_iter()
                    .map(|x| -x)
                    .collect()
            } else {
                Vec::new()
            }
        });
        for nu in 0..nao {
            // Right factors of the ket function through order n.
            let mut right: Vec<Cx> = vec![(0.0, 0.0); width];
            let mut slope: [Vec<Cx>; 3] = std::array::from_fn(|_| vec![(0.0, 0.0); width]);
            let mut kinetic: [Vec<Cx>; 3] = std::array::from_fn(|_| vec![(0.0, 0.0); width]);
            for k in 0..width {
                for j in 0..=k {
                    let value = series.w[nu][k - j];
                    right[k].0 += v_rho[j] * value.0;
                    right[k].1 += v_rho[j] * value.1;
                    for a in 0..3 {
                        if gga {
                            let g = series.gw[nu][a][k - j];
                            right[k].0 += b[a][j] * g.0;
                            right[k].1 += b[a][j] * g.1;
                            slope[a][k].0 += b[a][j] * value.0;
                            slope[a][k].1 += b[a][j] * value.1;
                        }
                        if mgga {
                            let m = series.pi[nu][a][k - j];
                            kinetic[a][k].0 += 0.5 * v_tau[j] * m.0;
                            kinetic[a][k].1 += 0.5 * v_tau[j] * m.1;
                            if corrected {
                                right[k].0 += 0.5 * v_current[a][j] * m.0;
                                right[k].1 += 0.5 * v_current[a][j] * m.1;
                                kinetic[a][k].0 += 0.5 * v_current[a][j] * value.0;
                                kinetic[a][k].1 += 0.5 * v_current[a][j] * value.1;
                            }
                        }
                    }
                }
            }
            for mu in 0..nao {
                let mut value: Cx = (0.0, 0.0);
                for degree in 0..=n {
                    let weight = weights[n - degree];
                    if weight == 0.0 {
                        continue;
                    }
                    let mut coefficient: Cx = (0.0, 0.0);
                    for j in 0..=degree {
                        let v = mul(conj(series.w[mu][j]), right[degree - j]);
                        coefficient = (coefficient.0 + v.0, coefficient.1 + v.1);
                        for a in 0..3 {
                            if gga {
                                let v = mul(conj(series.gw[mu][a][j]), slope[a][degree - j]);
                                coefficient = (coefficient.0 + v.0, coefficient.1 + v.1);
                            }
                            if mgga {
                                let v = mul(conj(series.pi[mu][a][j]), kinetic[a][degree - j]);
                                coefficient = (coefficient.0 + v.0, coefficient.1 + v.1);
                            }
                        }
                    }
                    value.0 += weight * coefficient.0;
                    value.1 += weight * coefficient.1;
                }
                out[s].0[mu * nao + nu] = value.0;
                out[s].1[mu * nao + nu] = value.1;
            }
        }
    }
    out
}

fn point_xc_series_model(
    series: &PointSeries,
    weights: &[f64],
    functional: &ResolvedFunctional,
    p: &[Vec<Cm>; 2],
    n: usize,
    want_matrix: bool,
    model: MagneticKineticDensity,
) -> Result<(Vec<f64>, Option<PointMatrices>)> {
    let width = n + 1;
    let gga = functional.family != Family::Lda;
    let mgga = functional.family == Family::MetaGga;
    let corrected = mgga && model == MagneticKineticDensity::CurrentInvariant;
    let PointDensitySeries {
        variables,
        gradient,
        kinetic,
    } = density_fields(series, p, n, gga, mgga, corrected)?;
    let xc = functional.semilocal_taylor_potentials(&XcTaylorInput {
        variables,
        parameters: None,
    })?;
    let energy = real_product(weights, &xc.energy, width);
    if !want_matrix {
        return Ok((energy, None));
    }
    let gradient_potentials: [[Vec<f64>; 3]; 2] = std::array::from_fn(|s| {
        let own = if s == 0 { 2 } else { 4 };
        std::array::from_fn(|a| {
            let first = real_product(&xc.potentials[own], &gradient[s][a], width);
            let second = real_product(&xc.potentials[3], &gradient[1 - s][a], width);
            (0..width).map(|k| 2.0 * first[k] + second[k]).collect()
        })
    });
    let potentials = PointPotentials {
        rho: std::array::from_fn(|s| xc.potentials[s].as_slice()),
        gradient: std::array::from_fn(|s| {
            std::array::from_fn(|a| gradient_potentials[s][a].as_slice())
        }),
        tau: std::array::from_fn(|s| xc.potentials[5 + s].as_slice()),
    };
    Ok((
        energy,
        Some(point_pullback(
            series,
            weights,
            &potentials,
            &kinetic,
            n,
            gga,
            mgga,
        )),
    ))
}

fn make_points(mb: &MolBasis, grid: &MolecularGrid, path: &FieldPath) -> Vec<Point> {
    let nao = mb.nao_sph;
    let mut centers = vec![[0.0; 3]; nao];
    for shell in &mb.shells {
        for k in 0..n_sph(shell.l) {
            centers[shell.sph_off + k] = shell.center.to_array();
        }
    }
    let mut points = Vec::new();
    for start in (0..grid.len()).step_by(128) {
        let end = (start + 128).min(grid.len());
        let batch = eval_aos_full(mb, &grid.points[start..end], true, false);
        let gradients = batch.grad.as_ref().expect("gradients were requested");
        for i in 0..end - start {
            let r = grid.points[start + i].to_array();
            let mut point = Point {
                weight: grid.weights[start + i],
                phi: Vec::with_capacity(nao),
                grad: Vec::with_capacity(nao),
                theta: Vec::with_capacity(nao),
                slope: Vec::with_capacity(nao),
                gauge: Vec::with_capacity(nao),
            };
            for mu in 0..nao {
                point.phi.push(batch.phi[(i, mu)]);
                point
                    .grad
                    .push(std::array::from_fn(|a| gradients[a][(i, mu)]));
                if path.london {
                    let slope = half_cross(path.magnetic, centers[mu]);
                    point
                        .theta
                        .push(slope[0] * r[0] + slope[1] * r[1] + slope[2] * r[2]);
                    point.slope.push(slope);
                    point.gauge.push(half_cross(
                        path.magnetic,
                        std::array::from_fn(|a| r[a] - centers[mu][a]),
                    ));
                } else {
                    point.theta.push(0.0);
                    point.slope.push([0.0; 3]);
                    point.gauge.push(half_cross(
                        path.magnetic,
                        std::array::from_fn(|a| r[a] - path.origin[a]),
                    ));
                }
            }
            points.push(point);
        }
    }
    points
}

struct Geometry {
    learned: Option<LearnedFieldProvider>,
    nonlocal: Option<(f64, NonlocalFieldProvider)>,
    kinetic_model: MagneticKineticDensity,
    nao: usize,
    one: OneElectronSeries,
    mb: MolBasis,
    path: FieldPath,
    /// Integral-direct engines of the field-free integrals.
    jk: JkEngine,
    jk_lr: Option<JkEngine>,
    omega: f64,
    /// First-order one-electron term per spin, for spin-dependent operators.
    core_path: Option<Vec<[Cm; 2]>>,
    cx: f64,
    clr: f64,
    points: Vec<Point>,
    /// Nuclear repulsion and the coefficient of the nuclear field energy.
    nuclear: [f64; 2],
    nuclear_dipole: [f64; 3],
}

impl Geometry {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mol: &Molecule,
        mb: &MolBasis,
        functional: &ResolvedFunctional,
        grid: Option<&MolecularGrid>,
        opts: &ScfOptions,
        path: &FieldPath,
        core_path: Option<Vec<[Cm; 2]>>,
        kinetic_model: MagneticKineticDensity,
    ) -> Result<Self> {
        let nao = mb.nao_sph;
        if core_path.as_ref().is_some_and(|p| {
            p.len() > path.order.saturating_add(1)
                || p.iter().flatten().any(|m| {
                    m.re.rows != nao
                        || m.re.cols != nao
                        || m.im.rows != nao
                        || m.im.cols != nao
                        || m.re
                            .as_slice()
                            .iter()
                            .chain(m.im.as_slice())
                            .any(|v| !v.is_finite())
                        || (0..nao).any(|i| {
                            (0..nao).any(|j| {
                                (m.re[(i, j)] - m.re[(j, i)]).abs() > 1e-12
                                    || (m.im[(i, j)] + m.im[(j, i)]).abs() > 1e-12
                            })
                        })
                })
        }) {
            return Err(DftError::InvalidInput(
                "explicit complex core coefficients must be finite Hermitian matrices matching the field path".into(),
            ));
        }

        let mut one = one_electron_series(mol, mb, path)?;
        if let Some(field) = opts.external_field {
            for n in 0..=path.order {
                for a in 0..3 {
                    cadd(&mut one.core[n], &one.dipole[a][n], field.get(a));
                }
            }
        }
        let jk = JkEngine::from_options(mol, mb, opts)?;
        let jk_lr = if functional.hybrid.clr != 0.0 {
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
        let points = if !functional.parts.is_empty() || opts.skala.is_some() {
            make_points(
                mb,
                grid.ok_or_else(|| {
                    DftError::InvalidInput("the field path needs the molecular grid".into())
                })?,
                path,
            )
        } else {
            Vec::new()
        };
        let charges = mb.effective_charges(mol);
        let mut nuclear_dipole = [0.0; 3];
        for (atom, z) in mol.atoms.iter().zip(&charges) {
            let r = atom.position.to_array();
            for a in 0..3 {
                nuclear_dipole[a] += z * r[a];
            }
        }
        let field: f64 = (0..3).map(|a| -nuclear_dipole[a] * path.electric[a]).sum();
        let nonlocal = opts
            .vv10
            .map(|parameters| {
                NonlocalFieldProvider::new(mol, mb, parameters, opts, path).map(|p| (1.0, p))
            })
            .transpose()?;
        Ok(Self {
            nonlocal,
            learned: opts.skala.as_ref().map(|model| {
                LearnedFieldProvider::new(
                    model,
                    mol,
                    grid.expect("the learned field grid was checked"),
                )
            }),
            kinetic_model,
            nao,
            one,
            mb: mb.clone(),
            path: *path,
            jk,
            jk_lr,
            omega: functional.hybrid.omega,
            core_path,
            cx: functional.hybrid.cx,
            clr: functional.hybrid.clr,
            points,
            nuclear: [
                mol.nuclear_repulsion()
                    - opts.external_field.map_or(0.0, |f| {
                        (0..3).map(|a| f.get(a) * nuclear_dipole[a]).sum::<f64>()
                    }),
                field,
            ],
            nuclear_dipole,
        })
    }

    /// Order-`n` coefficients of the Coulomb matrix and of the exchange
    /// matrices of the full-range and long-range kernels per spin. The
    /// field-free integrals act on the order-`n` density through the
    /// integral-direct engines; the field coefficients of the integrals over
    /// London orbitals are contracted quartet by quartet.
    fn coulomb_exchange(&self, p: &[Vec<Cm>; 2], n: usize) -> Result<(Cm, [Cm; 2], [Cm; 2])> {
        let nao = self.nao;
        let want_k = self.cx != 0.0;
        let mut total = p[0][n].clone();
        cadd(&mut total, &p[1][n], 1.0);
        // Real and imaginary parts of the density through the real engines.
        let part = |total: &Matrix, spin: [&Matrix; 2]| -> Result<(Matrix, [Matrix; 2])> {
            if spin.iter().all(|m| m.as_slice().iter().all(|x| *x == 0.0)) {
                return Ok((
                    Matrix::zeros(nao, nao),
                    [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)],
                ));
            }
            self.jk.build(&self.mb, total, spin, want_k)
        };
        let (j_re, [k_re0, k_re1]) = part(&total.re, [&p[0][n].re, &p[1][n].re])?;
        let (j_im, [k_im0, k_im1]) = part(&total.im, [&p[0][n].im, &p[1][n].im])?;
        let mut j = Cm { re: j_re, im: j_im };
        let mut k = [
            Cm {
                re: k_re0,
                im: k_im0,
            },
            Cm {
                re: k_re1,
                im: k_im1,
            },
        ];
        let mut klr = [Cm::zeros(nao, nao), Cm::zeros(nao, nao)];
        if let Some(engine) = &self.jk_lr {
            let [re0, re1] = engine.build_k_only(&self.mb, [&p[0][n].re, &p[1][n].re])?;
            let [im0, im1] = engine.build_k_only(&self.mb, [&p[0][n].im, &p[1][n].im])?;
            klr = [Cm { re: re0, im: im0 }, Cm { re: re1, im: im1 }];
        }
        if self.path.london && n > 0 {
            let density = [&p[0][..], &p[1][..]];
            let (dj, dk) =
                contracted_repulsion_series(&self.mb, &self.path, None, density, n, true, want_k)?;
            cadd(&mut j, &dj, 1.0);
            for s in 0..2 {
                cadd(&mut k[s], &dk[s], 1.0);
            }
            if self.jk_lr.is_some() {
                let (_, dk) = contracted_repulsion_series(
                    &self.mb,
                    &self.path,
                    Some(self.omega),
                    density,
                    n,
                    false,
                    true,
                )?;
                for s in 0..2 {
                    cadd(&mut klr[s], &dk[s], 1.0);
                }
            }
        }
        Ok((j, k, klr))
    }

    /// Order-`n` coefficient of the Kohn-Sham matrices of the density series.
    fn fock_coefficient(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Cm>; 2],
        n: usize,
    ) -> Result<[Cm; 2]> {
        let (j, k, klr) = self.coulomb_exchange(p, n)?;
        let mut out = [self.one.core[n].clone(), self.one.core[n].clone()];
        for s in 0..2 {
            cadd(&mut out[s], &j, 1.0);
            cadd(&mut out[s], &k[s], -self.cx);
            cadd(&mut out[s], &klr[s], -self.clr);
            if let Some(extra) = self.core_path.as_ref().and_then(|path| path.get(n)) {
                cadd(&mut out[s], &extra[s], 1.0);
            }
        }
        self.add_functional_fock(functional, p, n, &mut out)?;
        Ok(out)
    }

    /// Shared nonlinear Fock contribution, separated from the linear
    /// one-electron/Coulomb/exchange contraction for retained arithmetic.
    fn add_functional_fock(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Cm>; 2],
        n: usize,
        out: &mut [Cm; 2],
    ) -> Result<()> {
        let nao = self.nao;
        if !functional.parts.is_empty() && !self.points.is_empty() {
            let zero = || -> PointMatrices {
                std::array::from_fn(|_| (vec![0.0; nao * nao], vec![0.0; nao * nao]))
            };
            let total = self
                .points
                .par_iter()
                .try_fold(zero, |mut acc, point| -> Result<PointMatrices> {
                    let (_, matrices) =
                        point_xc(point, functional, p, n, true, self.kinetic_model)?;
                    let matrices = matrices.expect("matrices were requested");
                    for s in 0..2 {
                        for (t, v) in acc[s].0.iter_mut().zip(&matrices[s].0) {
                            *t += *v;
                        }
                        for (t, v) in acc[s].1.iter_mut().zip(&matrices[s].1) {
                            *t += *v;
                        }
                    }
                    Ok(acc)
                })
                .try_reduce(zero, |mut a, b| {
                    for s in 0..2 {
                        for (t, v) in a[s].0.iter_mut().zip(&b[s].0) {
                            *t += *v;
                        }
                        for (t, v) in a[s].1.iter_mut().zip(&b[s].1) {
                            *t += *v;
                        }
                    }
                    Ok(a)
                })?;
            for s in 0..2 {
                for (t, v) in out[s].re.as_mut_slice().iter_mut().zip(&total[s].0) {
                    *t += *v;
                }
                for (t, v) in out[s].im.as_mut_slice().iter_mut().zip(&total[s].1) {
                    *t += *v;
                }
            }
        }
        if let Some(provider) = &self.learned {
            let series: Vec<_> = self
                .points
                .par_iter()
                .map(|point| point_series(point, n + 1))
                .collect();
            let (_, fock) = provider.evaluate(&series, p, n, true, self.kinetic_model)?;
            let fock = fock.expect("global field Fock was requested");
            for s in 0..2 {
                cadd(&mut out[s], &fock[s], 1.0);
            }
        }
        if let Some((coefficient, provider)) = &self.nonlocal {
            let (_, fock) = provider.evaluate(p, n, true)?;
            let fock = fock.expect("the nonlocal field potential was requested");
            for s in 0..2 {
                cadd(&mut out[s], &fock[s], *coefficient);
            }
        }
        Ok(())
    }

    /// Energy coefficients through `order` of the density series.
    fn energy(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Cm>; 2],
        order: usize,
    ) -> Result<Vec<f64>> {
        let mut energy = vec![0.0; order + 1];
        energy[0] = self.nuclear[0];
        if order > 0 {
            energy[1] = self.nuclear[1];
        }
        let terms: Vec<(Cm, [Cm; 2], [Cm; 2])> = (0..=order)
            .map(|k| self.coulomb_exchange(p, k))
            .collect::<Result<_>>()?;
        for n in 0..=order {
            for m in 0..=n {
                let mut total = p[0][m].clone();
                cadd(&mut total, &p[1][m], 1.0);
                energy[n] += trace(&total, &self.one.core[n - m]);
                if let Some(extra) = self.core_path.as_ref().and_then(|path| path.get(n - m)) {
                    for s in 0..2 {
                        energy[n] += trace(&p[s][m], &extra[s]);
                    }
                }
                energy[n] += 0.5 * trace(&total, &terms[n - m].0);
                for s in 0..2 {
                    energy[n] -= 0.5 * self.cx * trace(&p[s][m], &terms[n - m].1[s]);
                    energy[n] -= 0.5 * self.clr * trace(&p[s][m], &terms[n - m].2[s]);
                }
            }
        }
        if !functional.parts.is_empty() && !self.points.is_empty() {
            let parts: Result<Vec<Vec<f64>>> = self
                .points
                .par_iter()
                .map(|point| {
                    Ok(point_xc(point, functional, p, order, false, self.kinetic_model)?.0)
                })
                .collect();
            for part in parts? {
                for n in 0..=order {
                    energy[n] += part[n];
                }
            }
        }
        if let Some(provider) = &self.learned {
            let series: Vec<_> = self
                .points
                .par_iter()
                .map(|point| point_series(point, order + 1))
                .collect();
            let (extra, _) = provider.evaluate(&series, p, order, false, self.kinetic_model)?;
            for (a, b) in energy.iter_mut().zip(extra) {
                *a += b;
            }
        }
        if let Some((coefficient, provider)) = &self.nonlocal {
            let (extra, _) = provider.evaluate(p, order, false)?;
            for (a, b) in energy.iter_mut().zip(extra) {
                *a += coefficient * b;
            }
        }
        Ok(energy)
    }

    fn nonlocal_energy(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<Vec<f64>> {
        match &self.nonlocal {
            None => Ok(vec![0.0; order + 1]),
            Some((coefficient, provider)) => Ok(provider
                .evaluate(p, order, false)?
                .0
                .into_iter()
                .map(|e| coefficient * e)
                .collect()),
        }
    }

    /// Dipole coefficients through `order` of the density series.
    fn dipole(&self, p: &[Vec<Cm>; 2], order: usize) -> Vec<[f64; 3]> {
        (0..=order)
            .map(|n| {
                std::array::from_fn(|axis| {
                    let mut value = if n == 0 {
                        self.nuclear_dipole[axis]
                    } else {
                        0.0
                    };
                    for m in 0..=n {
                        for s in 0..2 {
                            value -= trace(&p[s][m], &self.one.dipole[axis][n - m]);
                        }
                    }
                    value
                })
            })
            .collect()
    }
}

/// Reference orbitals transported so that `B^H S B = 1` at every order.
pub(crate) fn transport(c: &Cm, overlap: &[Cm], order: usize) -> Vec<Cm> {
    let nmo = c.re.cols;
    let ct = adjoint(c);
    let mut x: Vec<Cm> = overlap.iter().map(|s| cmul(&cmul(&ct, s), c)).collect();
    x[0] = Cm::zeros(nmo, nmo);
    let mut power = vec![Cm::zeros(nmo, nmo); order + 1];
    power[0] = from_real(&Matrix::identity(nmo));
    let mut inverse = power.clone();
    let mut binomial = 1.0;
    for k in 1..=order {
        power = series_product(&power, &x, order);
        binomial *= (-0.5 - (k - 1) as f64) / k as f64;
        for n in 1..=order {
            cadd(&mut inverse[n], &power[n], binomial);
        }
    }
    inverse.iter().map(|m| cmul(c, m)).collect()
}

/// Taylor coefficients of the energy and of the dipole along a field path.
#[derive(Clone, Debug)]
pub struct FieldPathTaylor {
    /// `E(t) = sum_n energy[n] t^n`, Hartree.
    pub energy: Vec<f64>,
    /// `mu(t) = sum_n dipole[n] t^n`, atomic units.
    pub dipole: Vec<[f64; 3]>,
    /// Largest deviation of the zeroth-order Kohn-Sham matrices of the
    /// integral series from those of the reference.
    pub reference_defect: f64,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub residuals: Vec<f64>,
}

/// Energy and dipole of an integer-occupation reference along
/// `E(t) = t electric`, `B(t) = t magnetic` through `order`, with London
/// orbitals (`london`) or with the common gauge origin `origin`.
#[allow(clippy::too_many_arguments)]
pub fn field_path_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    electric: [f64; 3],
    magnetic: [f64; 3],
    origin: Vec3,
    london: bool,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<FieldPathTaylor> {
    Ok(field_path_state(
        mol, mb, functional, grid, scf, opts, electric, magnetic, origin, london, order, None,
        config,
    )?
    .0)
}

/// Complete moving orbital data for noncanonical amplitude response.
pub struct FieldWavefunctionPath {
    pub kinetic_density_model: MagneticKineticDensity,
    pub parent: FieldPathTaylor,
    pub target_energy: Vec<f64>,
    /// Explicit auxiliary term decomposition, including its energy coefficient.
    /// Factorial-normalized coefficients in Hartree on the same path.
    pub parent_nonlocal_energy: Vec<f64>,
    pub target_nonlocal_energy: Vec<f64>,
    /// Raw Tr[S(t) P_s(t)] minus the constant spin populations; electrons.
    pub particle_number_coefficients: Vec<[f64; 2]>,
    /// Recomputed max coefficient defects of C_s(t)^dagger S(t) C_s(t)=I.
    pub orbital_metric_defects: Vec<f64>,
    /// Physical AO-to-MO coefficients in a parallel occupied/virtual frame.
    pub orbitals: [Vec<Cm>; 2],
    /// Kohn-Sham matrices in precisely the same moving MO frame.
    pub mo_fock: [Vec<Cm>; 2],
    pub occupied: [usize; 2],
    pub orbital_frame_defect: f64,
}

/// Explicit energy model evaluated on a stationary parent density path.
/// Auxiliary terms belong to this target, never implicitly to its parent.
#[derive(Clone)]
pub struct FieldEnergyModel<'a> {
    functional: &'a ResolvedFunctional,
    nonlocal: Option<(f64, crate::nlc::Vv10Params)>,
    learned: Option<std::sync::Arc<crate::skala::SkalaModel>>,
}
impl<'a> FieldEnergyModel<'a> {
    /// Semilocal and exact-exchange components of an explicit target.
    pub fn new(functional: &'a ResolvedFunctional) -> Self {
        Self {
            functional,
            nonlocal: None,
            learned: None,
        }
    }
    /// Add a variational two-endpoint term with its exact energy coefficient.
    pub fn with_vv10(
        mut self,
        coefficient: f64,
        parameters: crate::nlc::Vv10Params,
    ) -> Result<Self> {
        if !coefficient.is_finite()
            || !parameters.b.is_finite()
            || parameters.b <= 0.0
            || !parameters.c.is_finite()
            || parameters.c < 0.0
        {
            return Err(DftError::InvalidInput(
                "invalid target VV10 coefficient or parameters".into(),
            ));
        }
        self.nonlocal = Some((coefficient, parameters));
        Ok(self)
    }
    /// Add the complete supplied global learned graph to this target.
    pub fn with_learned(mut self, model: std::sync::Arc<crate::skala::SkalaModel>) -> Self {
        self.learned = Some(model);
        self
    }
    pub fn functional(&self) -> &'a ResolvedFunctional {
        self.functional
    }

    /// The exact auxiliary energy coefficient and two-endpoint parameters.
    pub fn nonlocal(&self) -> Option<(f64, crate::nlc::Vv10Params)> {
        self.nonlocal
    }

    /// An explicitly supplied graph, shared with moving-geometry targets.
    pub fn learned(&self) -> Option<&std::sync::Arc<crate::skala::SkalaModel>> {
        self.learned.as_ref()
    }
}

/// Shared stationary field path with an explicit analytic one-electron
/// coefficient provider. The extra source may contain arbitrary mixed
/// operators, including nuclear magnetic moments and their B contact terms.
/// It must describe the same zero-order reference Hamiltonian.
#[allow(clippy::too_many_arguments)]
pub fn field_wavefunction_path(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    target: Option<&ResolvedFunctional>,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    path: &FieldPath,
    core_coefficients: Option<&[[Cm; 2]]>,
    config: &ResponseSolverConfig,
) -> Result<FieldWavefunctionPath> {
    field_wavefunction_path_for_model(
        mol,
        mb,
        functional,
        target.map(FieldEnergyModel::new),
        grid,
        scf,
        opts,
        path,
        core_coefficients,
        config,
    )
}

/// The same full orbital path with explicitly composed target auxiliaries.
#[allow(clippy::too_many_arguments)]
pub fn field_wavefunction_path_for_model(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    path: &FieldPath,
    core_coefficients: Option<&[[Cm; 2]]>,
    config: &ResponseSolverConfig,
) -> Result<FieldWavefunctionPath> {
    if scf.electronic_temperature != 0.0 || scf.fixed_occupation_residual.is_some() {
        return Err(DftError::InvalidInput(
            "complex orbital field paths require an integer zero-temperature reference".into(),
        ));
    }
    if target
        .as_ref()
        .is_some_and(|model| match model.functional.required_nonlocal_kind() {
            None => false,
            Some(crate::xc::RequiredNonlocalTerm::Vv10) => model.nonlocal.is_none(),
            Some(_) => true,
        })
    {
        return Err(DftError::Unsupported {
            feature:"a field target without its required auxiliary provider".into(),
            missing:"supply the explicit VV10 parameters/coefficient; other nonlocal kinds require their own field provider".into(),
        });
    }
    let (parent, state) = field_path_state_with_core(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        path.electric,
        path.magnetic,
        Vec3::new(path.origin[0], path.origin[1], path.origin[2]),
        path.london,
        path.order,
        core_coefficients.map(<[_]>::to_vec),
        MagneticKineticDensity::CurrentInvariant,
        config,
    )?;
    let occupied = [scf.n_alpha, scf.n_beta];
    let mut defect = 0.0_f64;
    let mut orbitals: [Vec<Cm>; 2] = [Vec::new(), Vec::new()];
    let mut mo_fock: [Vec<Cm>; 2] = [Vec::new(), Vec::new()];
    for spin in 0..2 {
        if let Some(frames) = &state.retained_orbital_frames {
            orbitals[spin] = frames[spin].0.clone();
            mo_fock[spin] = frames[spin].1.clone();
            defect = defect.max(frames[spin].2);
            continue;
        }
        let (frame, error) = crate::properties::pt2_response::complex_projector_frames(
            &state.d[spin],
            occupied[spin],
        )?;
        defect = defect.max(error);
        orbitals[spin] = series_product(&state.b[spin], &frame, path.order);
        let dagger: Vec<_> = frame.iter().map(adjoint).collect();
        mo_fock[spin] = series_product(
            &series_product(&dagger, &state.f[spin], path.order),
            &frame,
            path.order,
        );
    }
    let parent_nonlocal_energy = state.geometry.nonlocal_energy(&state.p, path.order)?;
    let mut orbital_metric_defects = vec![0.0_f64; path.order + 1];
    for coefficients in &orbitals {
        let daggers: Vec<_> = coefficients.iter().map(adjoint).collect();
        let gram = series_product(
            &daggers,
            &series_product(&state.geometry.one.overlap, coefficients, path.order),
            path.order,
        );
        for (n, g) in gram.iter().enumerate() {
            for i in 0..g.re.rows {
                for j in 0..g.re.cols {
                    let exact = if n == 0 && i == j { 1.0 } else { 0.0 };
                    orbital_metric_defects[n] = orbital_metric_defects[n]
                        .max((g.re[(i, j)] - exact).abs())
                        .max(g.im[(i, j)].abs());
                }
            }
        }
    }
    let particle_number_coefficients = (0..=path.order)
        .map(|n| {
            std::array::from_fn(|s| {
                (0..=n)
                    .map(|k| trace(&state.p[s][k], &state.geometry.one.overlap[n - k]))
                    .sum::<f64>()
                    - if n == 0 { occupied[s] as f64 } else { 0.0 }
            })
        })
        .collect();
    let (target_energy, target_nonlocal_energy) = match target {
        None => (parent.energy.clone(), parent_nonlocal_energy.clone()),
        Some(target) => {
            let mut target_options = opts.clone();
            target_options.vv10 = target.nonlocal.map(|v| v.1);
            target_options.skala = target.learned.clone();
            let mut geometry = Geometry::new(
                mol,
                mb,
                target.functional,
                grid,
                &target_options,
                path,
                core_coefficients.map(<[_]>::to_vec),
                MagneticKineticDensity::CurrentInvariant,
            )?;
            if let Some((coefficient, _)) = target.nonlocal {
                geometry
                    .nonlocal
                    .as_mut()
                    .expect("explicit target VV10 was constructed")
                    .0 = coefficient;
            }
            (
                geometry.energy(target.functional, &state.p, path.order)?,
                geometry.nonlocal_energy(&state.p, path.order)?,
            )
        }
    };
    Ok(FieldWavefunctionPath {
        kinetic_density_model: MagneticKineticDensity::CurrentInvariant,
        parent,
        target_energy,
        parent_nonlocal_energy,
        target_nonlocal_energy,
        particle_number_coefficients,
        orbital_metric_defects,
        orbitals,
        mo_fock,
        occupied,
        orbital_frame_defect: defect,
    })
}

/// Reference solution of a field path in its transported frame.
struct PathState {
    geometry: Geometry,
    /// Transported orbitals and their adjoints.
    b: [Vec<Cm>; 2],
    bt: [Vec<Cm>; 2],
    /// Density and Kohn-Sham matrix coefficients in the transported frame.
    d: [Vec<Cm>; 2],
    f: [Vec<Cm>; 2],
    /// Density coefficients in the AO basis.
    p: [Vec<Cm>; 2],
    retained_orbital_frames: Option<[(Vec<Cm>, Vec<Cm>, f64); 2]>,
}

/// Frozen-reference kernel and complete moving-geometry coefficient provider.
/// Every kernel must be the derivative of the same stationary energy/Fock
/// model used by its coefficient and energy methods.
pub(crate) trait ComplexPathBackend {
    fn overlap(&self) -> &[Cm];
    fn fock_coefficient(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<[Cm; 2]>;
    fn linear_kernel(&self, direction: &[Cm; 2]) -> Result<[Cm; 2]>;
    fn energy(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<Vec<f64>>;
    fn dipole(&self, p: &[Vec<Cm>; 2], order: usize) -> Vec<[f64; 3]>;
}

/// Complete physical Fock coefficients for a retained high/low AO density.
/// A backend must supply its actual moving integral and functional tangent;
/// a rounded Fock followed by diagnostic-only compensation is insufficient.
pub(crate) trait RetainedComplexBackend: ComplexPathBackend {
    fn retained_fock_coefficient(
        &self,
        high: &[Vec<Cm>; 2],
        low: &[Vec<Cm>; 2],
        order: usize,
    ) -> Result<([Cm; 2], [Cm; 2])>;
}

pub(crate) struct ComplexPathReference {
    pub coefficients: [Cm; 2],
    pub energies: [Vec<f64>; 2],
    pub occupations: [usize; 2],
    pub density: [Cm; 2],
    pub fock: [Cm; 2],
    pub unrestricted: bool,
}
impl ComplexPathReference {
    fn spins(&self) -> usize {
        if self.unrestricted {
            2
        } else {
            1
        }
    }
}

pub(crate) struct ComplexPathState {
    pub b: [Vec<Cm>; 2],
    pub bt: [Vec<Cm>; 2],
    pub d: [Vec<Cm>; 2],
    pub f: [Vec<Cm>; 2],
    pub p: [Vec<Cm>; 2],
    pub retained_number_defects: Option<[Vec<f64>; 2]>,
    pub retained_orbital_frames: Option<[(Vec<Cm>, Vec<Cm>, f64); 2]>,
}

struct FieldPathBackend<'a, 'b> {
    geometry: &'a Geometry,
    functional: &'a ResolvedFunctional,
    solver: &'a ResponseSolver<'b>,
    current: Option<crate::properties::giao_xc::MetaCurrentKernel>,
    grid: Option<&'a MolecularGrid>,
}
impl ComplexPathBackend for FieldPathBackend<'_, '_> {
    fn overlap(&self) -> &[Cm] {
        &self.geometry.one.overlap
    }
    fn fock_coefficient(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<[Cm; 2]> {
        self.geometry.fock_coefficient(self.functional, p, order)
    }
    fn linear_kernel(&self, direction: &[Cm; 2]) -> Result<[Cm; 2]> {
        let real = self
            .solver
            .fock_response(&direction.each_ref().map(|p| p.re.clone()))?;
        let mut imaginary = exchange(self.solver, &direction.each_ref().map(|p| p.im.clone()))?;
        if let Some(kernel) = &self.current {
            let grid = self
                .grid
                .expect("a current kernel owns a molecular quadrature");
            for s in 0..2 {
                add(
                    &mut imaginary[s],
                    &kernel.response_fock(&self.geometry.mb, grid, &direction[s].im, s)?,
                    1.0,
                );
            }
        }
        Ok(std::array::from_fn(|s| Cm {
            re: real[s].clone(),
            im: imaginary[s].clone(),
        }))
    }
    fn energy(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<Vec<f64>> {
        self.geometry.energy(self.functional, p, order)
    }
    fn dipole(&self, p: &[Vec<Cm>; 2], order: usize) -> Vec<[f64; 3]> {
        self.geometry.dipole(p, order)
    }
}

impl RetainedComplexBackend for FieldPathBackend<'_, '_> {
    fn retained_fock_coefficient(
        &self,
        high: &[Vec<Cm>; 2],
        low: &[Vec<Cm>; 2],
        order: usize,
    ) -> Result<([Cm; 2], [Cm; 2])> {
        if self.geometry.nonlocal.is_some() || self.geometry.learned.is_some() {
            return Err(DftError::Unsupported {
                feature: "retained complex field coefficients with auxiliary XC providers".into(),
                missing: "retained physical nonlocal or learned density tangents".into(),
            });
        }
        let geometry = self.geometry;
        let (mut h, mut l) = crate::integrals::london::retained_repulsion_fock(
            &geometry.mb,
            &geometry.path,
            None,
            high,
            low,
            order,
            1.0,
            geometry.cx,
        )?;
        let long = if geometry.clr != 0.0 {
            Some(crate::integrals::london::retained_repulsion_fock(
                &geometry.mb,
                &geometry.path,
                Some(geometry.omega),
                high,
                low,
                order,
                0.0,
                geometry.clr,
            )?)
        } else {
            None
        };
        let mut xc = [
            Cm::zeros(geometry.nao, geometry.nao),
            Cm::zeros(geometry.nao, geometry.nao),
        ];
        geometry.add_functional_fock(self.functional, high, order, &mut xc)?;
        let scale = low.iter().flatten().map(largest).fold(0.0_f64, f64::max);
        let mut tangent = [
            Cm::zeros(geometry.nao, geometry.nao),
            Cm::zeros(geometry.nao, geometry.nao),
        ];
        if scale > 0.0 {
            let normalized = std::array::from_fn(|s| {
                low[s]
                    .iter()
                    .cloned()
                    .map(|mut p| {
                        for part in [&mut p.re, &mut p.im] {
                            for x in part.as_mut_slice() {
                                *x /= scale;
                            }
                        }
                        p
                    })
                    .collect()
            });
            geometry.add_functional_tangent(
                self.functional,
                high,
                &normalized,
                order,
                &mut tangent,
            )?;
        }
        for s in 0..2 {
            for imaginary in [false, true] {
                let part = |m: &Cm, i, j| {
                    if imaginary {
                        m.im[(i, j)]
                    } else {
                        m.re[(i, j)]
                    }
                };
                for i in 0..geometry.nao {
                    for j in 0..geometry.nao {
                        let mut sum = crate::linalg::AccurateSum::default();
                        for value in [
                            part(&h[s], i, j),
                            part(&l[s], i, j),
                            part(&geometry.one.core[order], i, j),
                            part(&xc[s], i, j),
                        ] {
                            sum.add_product(value, 1.0);
                        }
                        if let Some(extra) = geometry.core_path.as_ref().and_then(|v| v.get(order))
                        {
                            sum.add_product(part(&extra[s], i, j), 1.0);
                        }
                        if let Some((a, b)) = &long {
                            sum.add_product(part(&a[s], i, j), 1.0);
                            sum.add_product(part(&b[s], i, j), 1.0);
                        }
                        sum.add_product(part(&tangent[s], i, j), scale);
                        if imaginary {
                            (h[s].im[(i, j)], l[s].im[(i, j)]) = sum.parts();
                        } else {
                            (h[s].re[(i, j)], l[s].re[(i, j)]) = sum.parts();
                        }
                    }
                }
            }
        }
        // For a real field and Hermitian density variation, dE = Re Tr(F dP)
        // contains only the Hermitian part of F. Independently evaluated
        // ordered integrals differ by roundoff; that anti-Hermitian part is
        // not an orbital gradient and cannot be removed by orbital rotations.
        // Project the physical gradient before transport, retaining the low
        // parts of both entries rather than rounding an average afterwards.
        for s in 0..2 {
            for i in 0..geometry.nao {
                for j in i..geometry.nao {
                    let mut re = crate::linalg::AccurateSum::default();
                    let mut im = crate::linalg::AccurateSum::default();
                    for matrix in [&h[s], &l[s]] {
                        re.add_product(matrix.re[(i, j)], 0.5);
                        re.add_product(matrix.re[(j, i)], 0.5);
                        im.add_product(matrix.im[(i, j)], 0.5);
                        im.add_product(matrix.im[(j, i)], -0.5);
                    }
                    let (rh, rl) = re.parts();
                    let (ih, il) = im.parts();
                    h[s].re[(i, j)] = rh;
                    h[s].re[(j, i)] = rh;
                    l[s].re[(i, j)] = rl;
                    l[s].re[(j, i)] = rl;
                    h[s].im[(i, j)] = ih;
                    h[s].im[(j, i)] = -ih;
                    l[s].im[(i, j)] = il;
                    l[s].im[(j, i)] = -il;
                }
            }
        }
        Ok((h, l))
    }
}

/// A static complex orbital Hessian in independent real and imaginary
/// occupied/virtual coordinates. The two parts may couple at a finite field.
struct ComplexOrbitalAction<'a, B> {
    backend: &'a B,
    reference: &'a ComplexPathReference,
}
impl<B: ComplexPathBackend> ComplexOrbitalAction<'_, B> {
    fn density_real(&self, x: &[f64]) -> [Cm; 2] {
        let mut offset = 0;
        let mut mo: [Cm; 2] = std::array::from_fn(|s| {
            let n = self.reference.coefficients[s].re.cols;
            Cm::zeros(n, n)
        });
        for s in 0..self.reference.spins() {
            let no = self.reference.occupations[s];
            for a in no..mo[s].re.rows {
                for i in 0..no {
                    mo[s].re[(a, i)] = x[offset];
                    mo[s].re[(i, a)] = x[offset];
                    mo[s].im[(a, i)] = x[offset + 1];
                    mo[s].im[(i, a)] = -x[offset + 1];
                    offset += 2;
                }
            }
        }
        if self.reference.spins() == 1 {
            mo[1] = mo[0].clone();
        }
        std::array::from_fn(|s| {
            let c = &self.reference.coefficients[s];
            cmul(&cmul(c, &mo[s]), &adjoint(c))
        })
    }
    fn apply_real(&self, x: &[f64]) -> Result<Vec<f64>> {
        let induced = self.backend.linear_kernel(&self.density_real(x))?;
        let mut out = Vec::with_capacity(x.len());
        for s in 0..self.reference.spins() {
            let c = &self.reference.coefficients[s];
            let g = cmul(&cmul(&adjoint(c), &induced[s]), c);
            let no = self.reference.occupations[s];
            for a in no..c.re.cols {
                for i in 0..no {
                    let gap = self.reference.energies[s][a] - self.reference.energies[s][i];
                    let offset = out.len();
                    out.push(g.re[(a, i)] + gap * x[offset]);
                    out.push(g.im[(a, i)] + gap * x[offset + 1]);
                }
            }
        }
        Ok(out)
    }
}
impl<B: ComplexPathBackend> LinearOperator for ComplexOrbitalAction<'_, B> {
    fn dimension(&self) -> usize {
        (0..self.reference.spins())
            .map(|s| {
                2 * self.reference.occupations[s]
                    * (self.reference.coefficients[s].re.cols - self.reference.occupations[s])
            })
            .sum()
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::Hermitian
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| self.apply_real(v))
    }
}

/// `field_path_taylor` together with the transported reference solution.
#[allow(clippy::too_many_arguments)]
fn field_path_state(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    electric: [f64; 3],
    magnetic: [f64; 3],
    origin: Vec3,
    london: bool,
    order: usize,
    spin_core: Option<[Cm; 2]>,
    config: &ResponseSolverConfig,
) -> Result<(FieldPathTaylor, PathState)> {
    let width = order
        .checked_add(1)
        .ok_or_else(|| DftError::InvalidInput("field path order overflows".into()))?;
    let core_path = spin_core.map(|linear| {
        let zero = || {
            [
                Cm::zeros(mb.nao_sph, mb.nao_sph),
                Cm::zeros(mb.nao_sph, mb.nao_sph),
            ]
        };
        let mut path = vec![zero(); width];
        if order > 0 {
            path[1] = linear;
        }
        path
    });
    field_path_state_with_core(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        electric,
        magnetic,
        origin,
        london,
        order,
        core_path,
        MagneticKineticDensity::Mechanical,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn field_path_state_with_core(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    electric: [f64; 3],
    magnetic: [f64; 3],
    origin: Vec3,
    london: bool,
    order: usize,
    core_path: Option<Vec<[Cm; 2]>>,
    kinetic_model: MagneticKineticDensity,
    config: &ResponseSolverConfig,
) -> Result<(FieldPathTaylor, PathState)> {
    config.validate()?;
    if order == usize::MAX
        || electric
            .iter()
            .chain(magnetic.iter())
            .any(|x| !x.is_finite())
        || !scf.converged
    {
        return Err(DftError::InvalidInput(
            "the field path needs finite directions and a converged reference".into(),
        ));
    }
    if !mb.ecps.is_empty()
        || opts.pcm.is_some()
        || functional.required_nonlocal_kind().is_some_and(|kind| {
            kind != crate::xc::RequiredNonlocalTerm::Vv10 || opts.vv10.is_none()
        })
        || opts.density_fit.is_some()
        || opts.extra_hcore.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "field path with additional Hamiltonian terms".into(),
            missing: "field series of effective core potentials, solvent, nonlocal correlation, \
                      density fitting and additional one-electron operators"
                .into(),
        });
    }
    let path = FieldPath {
        electric,
        magnetic,
        origin: origin.to_array(),
        london,
        order,
    };
    // A spin-dependent analytic source opens the two independent collinear
    // response channels even when its zero-field stationary seed is RKS.
    // The seed, occupations and energy functional are otherwise identical.
    let spin_dependent = core_path.as_ref().is_some_and(|coefficients| {
        coefficients.iter().skip(1).any(|h| {
            h[0].re.as_slice() != h[1].re.as_slice() || h[0].im.as_slice() != h[1].im.as_slice()
        })
    });
    let spin_reference = if spin_dependent && !scf.unrestricted {
        let mut open = scf.clone();
        open.unrestricted = true;
        Some(open)
    } else {
        None
    };
    let scf = spin_reference.as_ref().unwrap_or(scf);
    let geometry = Geometry::new(
        mol,
        mb,
        functional,
        grid,
        opts,
        &path,
        core_path,
        kinetic_model,
    )?;
    let solver = ResponseSolver::new(mol, mb, functional, grid, scf, opts)?;
    let reference = ComplexPathReference {
        coefficients: std::array::from_fn(|s| from_real(&scf.mo_coeff[s])),
        energies: scf.mo_energy.clone(),
        density: std::array::from_fn(|s| from_real(&scf.density[s])),
        fock: std::array::from_fn(|s| from_real(&scf.fock[s])),
        occupations: solver.dims.n_occ,
        unrestricted: scf.unrestricted,
    };
    let backend = FieldPathBackend {
        geometry: &geometry,
        functional,
        solver: &solver,
        current: if kinetic_model == MagneticKineticDensity::CurrentInvariant {
            crate::properties::giao_xc::MetaCurrentKernel::for_reference(
                mol, mb, functional, grid, scf, opts,
            )?
        } else {
            None
        },
        grid,
    };
    // High-order projector products can lose absolute accuracy while each
    // linear solve already has a tiny relative residual. Retain the low
    // density parts in the physical Fock, metric and orbital frame as well.
    // Auxiliary nonlocal/learned tangents need their own retained providers.
    let (taylor, state) = if order >= 3 && geometry.nonlocal.is_none() && geometry.learned.is_none()
    {
        stationary_retained_complex_path(&backend, &reference, order, config)?
    } else {
        stationary_complex_path(&backend, &reference, order, config)?
    };
    Ok((
        taylor,
        PathState {
            geometry,
            b: state.b,
            bt: state.bt,
            d: state.d,
            f: state.f,
            p: state.p,
            retained_orbital_frames: state.retained_orbital_frames,
        },
    ))
}

/// Shared projector/stationarity recursion for a complex stationary reference.
fn projector_diagonal(d: &[Cm], occupied: usize, order: usize) -> (Cm, Cm) {
    let size = d[0].re.rows;
    let mut products = Cm::zeros(size, size);
    for m in 1..order {
        cadd(&mut products, &cmul(&d[m], &d[order - m]), 1.0);
    }
    let mut diagonal = Cm::zeros(size, size);
    for i in 0..size {
        for j in 0..size {
            let sign = if i < occupied && j < occupied {
                -1.0
            } else if i >= occupied && j >= occupied {
                1.0
            } else {
                0.0
            };
            diagonal.re[(i, j)] = sign * products.re[(i, j)];
            diagonal.im[(i, j)] = sign * products.im[(i, j)];
        }
    }
    (diagonal, products)
}

/// Static stationary energy only: density response through k fixes energy
/// through 2k+1. Higher projector coefficients have free OV blocks set to
/// zero; metric transport and the energy still retain their complete order.
/// Dipole/density derivatives are deliberately absent from this result.
pub(crate) struct ComplexEnergyPath {
    pub energy: Vec<f64>,
    pub solved_order: usize,
    pub reference_defect: f64,
    pub residuals: Vec<f64>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
}
pub(crate) fn stationary_complex_energy_path<B: ComplexPathBackend>(
    backend: &B,
    reference: &ComplexPathReference,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ComplexEnergyPath> {
    if order == 0 || backend.overlap().len() <= order {
        return Err(DftError::InvalidInput(
            "stationary complex energy needs positive order and complete overlap coefficients"
                .into(),
        ));
    }
    let solved_order = order / 2;
    let (report, state) = stationary_complex_path(backend, reference, solved_order, config)?;
    let mut d = state.d;
    for n in solved_order + 1..=order {
        for s in 0..2 {
            let (diagonal, _) = projector_diagonal(&d[s], reference.occupations[s], n);
            d[s].push(diagonal);
        }
    }
    let b: [Vec<Cm>; 2] =
        std::array::from_fn(|s| transport(&reference.coefficients[s], backend.overlap(), order));
    let bt: [Vec<Cm>; 2] = std::array::from_fn(|s| b[s].iter().map(adjoint).collect());
    let p: [Vec<Cm>; 2] = std::array::from_fn(|s| {
        series_product(&series_product(&b[s], &d[s], order), &bt[s], order)
    });
    let mut defects = Vec::with_capacity(order);
    for n in 1..=order {
        let mut defect = 0.0_f64;
        for channel in &d {
            let mut idempotency = Cm::zeros(channel[0].re.rows, channel[0].re.cols);
            for k in 0..=n {
                cadd(&mut idempotency, &cmul(&channel[k], &channel[n - k]), 1.0);
            }
            cadd(&mut idempotency, &channel[n], -1.0);
            defect = defect.max(largest(&idempotency));
        }
        defects.push(defect);
    }
    Ok(ComplexEnergyPath {
        energy: backend.energy(&p, order)?,
        solved_order,
        reference_defect: report.reference_defect,
        residuals: report.residuals,
        projector_defects: defects,
        stationarity_defects: report.stationarity_defects,
    })
}

/// Retained metric, projector, physical Fock and stationarity energy path.
pub(crate) fn stationary_retained_complex_energy_path<B: RetainedComplexBackend>(
    backend: &B,
    reference: &ComplexPathReference,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ComplexEnergyPath> {
    Ok(retained_complex_response(backend, reference, order, false, config)?.0)
}

/// Retained orbital and Fock series for nonvariational target and PT2 terms.
/// The variational energy shortcut cannot supply their high-order density.
pub(crate) fn stationary_retained_complex_path<B: RetainedComplexBackend>(
    backend: &B,
    reference: &ComplexPathReference,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<(FieldPathTaylor, ComplexPathState)> {
    let (value, state) = retained_complex_response(backend, reference, order, true, config)?;
    let state = state.expect("the complete path retains orbital coefficients");
    let report = FieldPathTaylor {
        energy: value.energy,
        dipole: backend.dipole(&state.p, order),
        reference_defect: value.reference_defect,
        projector_defects: value.projector_defects,
        stationarity_defects: value.stationarity_defects,
        residuals: value.residuals,
    };
    Ok((report, state))
}

fn retained_complex_response<B: RetainedComplexBackend>(
    backend: &B,
    reference: &ComplexPathReference,
    order: usize,
    full: bool,
    config: &ResponseSolverConfig,
) -> Result<(ComplexEnergyPath, Option<ComplexPathState>)> {
    if order == 0 || backend.overlap().len() <= order {
        return Err(DftError::InvalidInput(
            "stationary complex energy needs positive order and complete overlap coefficients"
                .into(),
        ));
    }
    config.validate()?;
    if (0..2).any(|s| {
        let n = reference.coefficients[s].re.cols;
        reference.occupations[s] > n
            || reference.energies[s].len() != n
            || reference.coefficients[s].re.rows != backend.overlap()[0].re.rows
    }) {
        return Err(DftError::InvalidInput(
            "retained complex reference dimensions disagree".into(),
        ));
    }
    krylov::with_physical_acceptance(config, |inner| {
        let value = retained::evaluate_path(backend, reference, order, full, inner)?;
        let defect = value
            .0
            .projector_defects
            .iter()
            .chain(&value.0.stationarity_defects)
            .copied()
            .fold(0.0_f64, |a, b| {
                if b.is_finite() {
                    a.max(b)
                } else {
                    f64::INFINITY
                }
            });
        Ok((value, defect))
    })
}

/// Shared projector/stationarity recursion for a complex stationary reference.
pub(crate) fn stationary_complex_path<B: ComplexPathBackend>(
    backend: &B,
    reference: &ComplexPathReference,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<(FieldPathTaylor, ComplexPathState)> {
    krylov::with_physical_acceptance(config, |inner| {
        let value = stationary_complex_path_once(backend, reference, order, inner)?;
        let defect = value
            .0
            .projector_defects
            .iter()
            .chain(&value.0.stationarity_defects)
            .copied()
            .fold(0.0_f64, |a, b| {
                if b.is_finite() {
                    a.max(b)
                } else {
                    f64::INFINITY
                }
            });
        Ok((value, defect))
    })
}

fn stationary_complex_path_once<B: ComplexPathBackend>(
    backend: &B,
    reference: &ComplexPathReference,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<(FieldPathTaylor, ComplexPathState)> {
    config.validate()?;
    if backend.overlap().len() <= order
        || (0..2).any(|s| {
            let n = reference.coefficients[s].re.cols;
            reference.occupations[s] > n
                || reference.energies[s].len() != n
                || reference.coefficients[s].re.rows != backend.overlap()[0].re.rows
        })
    {
        return Err(DftError::InvalidInput(
            "complex path reference and overlap dimensions/order disagree".into(),
        ));
    }
    let n_occ = reference.occupations;
    let nmo = reference.coefficients.each_ref().map(|c| c.re.cols);
    let n_vir = std::array::from_fn::<_, 2, _>(|s| nmo[s] - n_occ[s]);
    let b: [Vec<Cm>; 2] =
        std::array::from_fn(|s| transport(&reference.coefficients[s], backend.overlap(), order));
    let bt: [Vec<Cm>; 2] = std::array::from_fn(|s| b[s].iter().map(adjoint).collect());
    let mut d: [Vec<Cm>; 2] = std::array::from_fn(|s| {
        let mut start = Cm::zeros(nmo[s], nmo[s]);
        for i in 0..n_occ[s] {
            start.re[(i, i)] = 1.0;
        }
        vec![start]
    });
    let mut f: [Vec<Cm>; 2] = std::array::from_fn(|s| {
        let mut start = Cm::zeros(nmo[s], nmo[s]);
        for (i, &e) in reference.energies[s].iter().enumerate() {
            start.re[(i, i)] = e;
        }
        vec![start]
    });
    let mut p: [Vec<Cm>; 2] = std::array::from_fn(|s| vec![reference.density[s].clone()]);
    let start = backend.fock_coefficient(&p, 0)?;
    let reference_defect = (0..2)
        .map(|s| {
            let mut difference = start[s].clone();
            cadd(&mut difference, &reference.fock[s], -1.0);
            largest(&difference)
        })
        .fold(0.0_f64, f64::max);
    let mut fock: [Vec<Cm>; 2] = std::array::from_fn(|s| vec![reference.fock[s].clone()]);
    let mut residuals = Vec::new();
    let mut projector_defects = Vec::new();
    let mut stationarity_defects = Vec::new();
    for n in 1..=order {
        // Idempotency fixes the occupied and virtual diagonal blocks.
        let mut products = [Cm::zeros(nmo[0], nmo[0]), Cm::zeros(nmo[1], nmo[1])];
        for s in 0..2 {
            let (dn, product) = projector_diagonal(&d[s], n_occ[s], n);
            products[s] = product;
            d[s].push(dn);
            let trial = series_product(&series_product(&b[s], &d[s], n), &bt[s], n)[n].clone();
            p[s].push(trial);
        }
        let trial_fock = backend.fock_coefficient(&p, n)?;
        let mut fn_mo = [Cm::zeros(nmo[0], nmo[0]), Cm::zeros(nmo[1], nmo[1])];
        let mut commutators = [Cm::zeros(nmo[0], nmo[0]), Cm::zeros(nmo[1], nmo[1])];
        for s in 0..2 {
            fock[s].push(trial_fock[s].clone());
            fn_mo[s] = series_product(&series_product(&bt[s], &fock[s], n), &b[s], n)[n].clone();
            for m in 1..n {
                cadd(&mut commutators[s], &cmul(&f[s][m], &d[s][n - m]), 1.0);
                cadd(&mut commutators[s], &cmul(&d[s][n - m], &f[s][m]), -1.0);
            }
        }
        let mut source_re = Vec::new();
        let mut source_im = Vec::new();
        for s in 0..reference.spins() {
            let no = n_occ[s];
            for a in 0..n_vir[s] {
                for i in 0..no {
                    source_re.push(fn_mo[s].re[(no + a, i)] + commutators[s].re[(no + a, i)]);
                    source_im.push(fn_mo[s].im[(no + a, i)] + commutators[s].im[(no + a, i)]);
                }
            }
        }
        let action = ComplexOrbitalAction { backend, reference };
        let source: Vec<_> = source_re
            .iter()
            .zip(&source_im)
            .flat_map(|(re, im)| [Complex::from(-*re), Complex::from(-*im)])
            .collect();
        let source_norm = source
            .iter()
            .fold(0.0_f64, |value, z| value.hypot(z.re).hypot(z.im));
        let solve_config = config.for_absolute_residual(source_norm)?;
        let solution = krylov::solve(&action, &source, None, &solve_config)?;
        let coordinates: Vec<_> = solution.amplitudes.iter().map(|z| z.re).collect();
        let x: Vec<_> = coordinates.chunks_exact(2).map(|v| v[0]).collect();
        let y: Vec<_> = coordinates.chunks_exact(2).map(|v| v[1]).collect();
        let off = action.density_real(&coordinates);
        let df = backend.linear_kernel(&off)?;
        let mut offset = 0;
        for s in 0..reference.spins() {
            let no = n_occ[s];
            for a in 0..n_vir[s] {
                for i in 0..no {
                    d[s][n].re[(no + a, i)] = x[offset];
                    d[s][n].re[(i, no + a)] = x[offset];
                    d[s][n].im[(no + a, i)] = y[offset];
                    d[s][n].im[(i, no + a)] = -y[offset];
                    offset += 1;
                }
            }
        }
        if reference.spins() == 1 {
            let copy = d[0][n].clone();
            d[1][n] = copy;
        }
        let mut defect = 0.0_f64;
        let mut stationary = 0.0_f64;
        for s in 0..2 {
            let c = &reference.coefficients[s];
            let ct = adjoint(c);
            let response = df[s].clone();
            cadd(&mut fock[s][n], &response, 1.0);
            cadd(&mut p[s][n], &off[s], 1.0);
            cadd(&mut fn_mo[s], &cmul(&cmul(&ct, &response), c), 1.0);
            f[s].push(fn_mo[s].clone());
            let mut idempotency = products[s].clone();
            cadd(&mut idempotency, &cmul(&d[s][0], &d[s][n]), 1.0);
            cadd(&mut idempotency, &cmul(&d[s][n], &d[s][0]), 1.0);
            cadd(&mut idempotency, &d[s][n], -1.0);
            defect = defect.max(largest(&idempotency));
            let mut commutator = commutators[s].clone();
            cadd(&mut commutator, &cmul(&f[s][0], &d[s][n]), 1.0);
            cadd(&mut commutator, &cmul(&f[s][n], &d[s][0]), 1.0);
            cadd(&mut commutator, &cmul(&d[s][n], &f[s][0]), -1.0);
            cadd(&mut commutator, &cmul(&d[s][0], &f[s][n]), -1.0);
            stationary = stationary.max(largest(&commutator));
        }
        projector_defects.push(defect);
        stationarity_defects.push(stationary);
        residuals.push(solution.relative_residual);
    }
    let taylor = FieldPathTaylor {
        energy: backend.energy(&p, order)?,
        dipole: backend.dipole(&p, order),
        reference_defect,
        projector_defects,
        stationarity_defects,
        residuals,
    };
    Ok((
        taylor,
        ComplexPathState {
            b,
            bt,
            d,
            f,
            p,
            retained_number_defects: None,
            retained_orbital_frames: None,
        },
    ))
}

/// Static polarizability, magnetizability and hypermagnetizability of the
/// field path response, with London orbitals or a common gauge origin.
///
/// The magnetizability is `xi_cd = -d2E/(dB_c dB_d)` from the second-order
/// energy coefficients of the directions `e_c` and `e_c + e_d`; the
/// hypermagnetizability is the mixed third derivative of the induced dipole
/// by the polarization identity of combined field directions.
#[allow(clippy::too_many_arguments)]
pub fn field_hypermagnetizability(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    origin: Vec3,
    london: bool,
    config: &ResponseSolverConfig,
) -> Result<Hypermagnetizability> {
    let mut residual = 0.0_f64;
    let mut evaluations = 0_usize;
    let mut run =
        |electric: [f64; 3], magnetic: [f64; 3], order: usize| -> Result<FieldPathTaylor> {
            let path = field_path_taylor(
                mol, mb, functional, grid, scf, opts, electric, magnetic, origin, london, order,
                config,
            )?;
            residual = path.residuals.iter().copied().fold(residual, f64::max);
            evaluations += 1;
            Ok(path)
        };
    let mut polarizability = [[0.0; 3]; 3];
    let mut magnetizability = [[0.0; 3]; 3];
    let mut second = [0.0; 3];
    for b in 0..3 {
        let mut unit = [0.0; 3];
        unit[b] = 1.0;
        let electric = run(unit, [0.0; 3], 1)?;
        for a in 0..3 {
            polarizability[a][b] = electric.dipole[1][a];
        }
        second[b] = run([0.0; 3], unit, 2)?.energy[2];
        magnetizability[b][b] = -2.0 * second[b];
    }
    for c in 0..3 {
        for d in 0..c {
            let mut direction = [0.0; 3];
            direction[c] = 1.0;
            direction[d] = 1.0;
            let mixed = run([0.0; 3], direction, 2)?.energy[2];
            let value = -(mixed - second[c] - second[d]);
            magnetizability[c][d] = value;
            magnetizability[d][c] = value;
        }
    }
    let mut cache: Vec<(Vec<f64>, [f64; 3])> = Vec::new();
    let mut tensor = [[[[0.0; 3]; 3]; 3]; 3];
    let mut roundoff = 0.0_f64;
    for b in 0..3 {
        for c in 0..3 {
            for d in c..3 {
                let mut vectors = vec![vec![0.0; 6]; 3];
                vectors[0][b] = 1.0;
                vectors[1][3 + c] = 1.0;
                vectors[2][3 + d] = 1.0;
                let mixed = super::mixed::polarized_mixed(6, &vectors, |direction, n| {
                    if let Some((_, value)) =
                        cache.iter().find(|(key, _)| key.as_slice() == direction)
                    {
                        return Ok(value.to_vec());
                    }
                    let path = run(
                        [direction[0], direction[1], direction[2]],
                        [direction[3], direction[4], direction[5]],
                        n,
                    )?;
                    let value = path.dipole[n];
                    cache.push((direction.to_vec(), value));
                    Ok(value.to_vec())
                })?;
                for a in 0..3 {
                    tensor[a][b][c][d] = mixed.value[a];
                    tensor[a][b][d][c] = mixed.value[a];
                    roundoff = roundoff.max(mixed.accumulation_roundoff[a]);
                }
            }
        }
    }
    Ok(Hypermagnetizability {
        origin: origin.to_array(),
        polarizability,
        magnetizability,
        tensor,
        accumulation_roundoff: roundoff,
        directional_evaluations: evaluations,
        residual,
    })
}

/// Magnetizability `xi_cd = -d2E/(dB_c dB_d)` of the field path response and
/// the largest solver residual.
#[allow(clippy::too_many_arguments)]
pub fn field_magnetizability(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    origin: Vec3,
    london: bool,
    config: &ResponseSolverConfig,
) -> Result<([[f64; 3]; 3], f64)> {
    let mut residual = 0.0_f64;
    let mut run = |direction: [f64; 3]| -> Result<f64> {
        let path = field_path_taylor(
            mol, mb, functional, grid, scf, opts, [0.0; 3], direction, origin, london, 2, config,
        )?;
        residual = path.residuals.iter().copied().fold(residual, f64::max);
        Ok(path.energy[2])
    };
    let mut second = [0.0; 3];
    let mut tensor = [[0.0; 3]; 3];
    for c in 0..3 {
        let mut unit = [0.0; 3];
        unit[c] = 1.0;
        second[c] = run(unit)?;
        tensor[c][c] = -2.0 * second[c];
    }
    for c in 0..3 {
        for d in 0..c {
            let mut direction = [0.0; 3];
            direction[c] = 1.0;
            direction[d] = 1.0;
            let value = -(run(direction)? - second[c] - second[d]);
            tensor[c][d] = value;
            tensor[d][c] = value;
        }
    }
    Ok((tensor, residual))
}

/// Exact density tangent of one semilocal field-potential coefficient.
/// The provider's mixed Taylor contract supplies the functional Hessian;
/// current-invariant kinetic density and its AO pullback are differentiated
/// on the same path. No finite displacement or registered-name dispatch.
fn point_xc_linear(
    point: &Point,
    functional: &ResolvedFunctional,
    p: &[Vec<Cm>; 2],
    q: &[Vec<Cm>; 2],
    n: usize,
    model: MagneticKineticDensity,
) -> Result<PointMatrices> {
    let width = n + 1;
    let gga = functional.family != Family::Lda;
    let mgga = functional.family == Family::MetaGga;
    let corrected = mgga && model == MagneticKineticDensity::CurrentInvariant;
    let series = point_series(point, width);
    let base = density_fields(&series, p, n, gga, mgga, corrected)?;
    let delta = std::array::from_fn::<_, 2, _>(|s| {
        point_density(&series, &q[s], width, gga, mgga, corrected)
    });
    let mut direction: [Vec<f64>; 7] = std::array::from_fn(|_| vec![0.0; width]);
    for s in 0..2 {
        direction[s] = delta[s].0.clone();
        direction[5 + s] = delta[s].2.clone();
    }
    if gga {
        for (ix, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            for a in 0..3 {
                let left = real_product(&delta[s].1[a], &base.gradient[t][a], width);
                let right = real_product(&base.gradient[s][a], &delta[t].1[a], width);
                for k in 0..width {
                    direction[2 + ix][k] += left[k] + right[k];
                }
            }
        }
    }
    let mut kinetic_delta = KineticPullback {
        velocities: std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; width])),
        density_chain: std::array::from_fn(|_| vec![0.0; width]),
        active: corrected,
    };
    if corrected {
        for s in 0..2 {
            let rho = &base.variables[s];
            if rho[0] <= 0.0 {
                if delta[s].3.iter().flatten().any(|&v| v != 0.0) {
                    return Err(DftError::InvalidInput(
                        "current-density tangent requires a positive reference density or an inactive spin channel".into()));
                }
                continue;
            }
            let density_term = real_product(&base.kinetic.density_chain[s], &direction[s], width);
            for (tau, v) in direction[5 + s].iter_mut().zip(density_term) {
                *tau += v;
            }
            for a in 0..3 {
                let velocity = &base.kinetic.velocities[s][a];
                let current_term = real_product(velocity, &delta[s].3[a], width);
                for (tau, v) in direction[5 + s].iter_mut().zip(current_term) {
                    *tau -= v;
                }
                let density_term = real_product(velocity, &direction[s], width);
                let dv = &mut kinetic_delta.velocities[s][a];
                // Differentiate j/rho by polynomial division, without a
                // reciprocal density that could overflow in the grid tail.
                for k in 0..width {
                    dv[k] = (delta[s].3[a][k]
                        - density_term[k]
                        - (1..=k).map(|j| rho[j] * dv[k - j]).sum::<f64>())
                        / rho[0];
                }
                let term = real_product(velocity, dv, width);
                for (v, t) in kinetic_delta.density_chain[s].iter_mut().zip(term) {
                    *v += t;
                }
            }
        }
    }
    let xc = functional.semilocal_taylor_directional_potentials(
        &XcTaylorInput {
            variables: base.variables,
            parameters: None,
        },
        &direction,
    )?;
    let gradient: [[Vec<f64>; 3]; 2] = std::array::from_fn(|s| {
        let own = if s == 0 { 2 } else { 4 };
        std::array::from_fn(|a| {
            let terms = [
                (
                    real_product(&xc.directional.potentials[own], &base.gradient[s][a], width),
                    2.0,
                ),
                (
                    real_product(&xc.value.potentials[own], &delta[s].1[a], width),
                    2.0,
                ),
                (
                    real_product(
                        &xc.directional.potentials[3],
                        &base.gradient[1 - s][a],
                        width,
                    ),
                    1.0,
                ),
                (
                    real_product(&xc.value.potentials[3], &delta[1 - s].1[a], width),
                    1.0,
                ),
            ];
            (0..width)
                .map(|k| terms.iter().map(|(v, c)| c * v[k]).sum())
                .collect()
        })
    });
    let potentials = PointPotentials {
        rho: std::array::from_fn(|s| xc.directional.potentials[s].as_slice()),
        gradient: std::array::from_fn(|s| std::array::from_fn(|a| gradient[s][a].as_slice())),
        tau: std::array::from_fn(|s| xc.directional.potentials[5 + s].as_slice()),
    };
    let mut weights = vec![0.0; width];
    weights[0] = point.weight;
    let mut out = point_pullback(&series, &weights, &potentials, &base.kinetic, n, gga, mgga);
    if corrected {
        // The remaining chain term changes the kinetic pullback itself.
        // Subtract its unchanged mechanical-tau contribution analytically;
        // only the density/current chain terms survive.
        let zero = vec![0.0; width];
        let tau = PointPotentials {
            rho: [&zero, &zero],
            gradient: [[&zero, &zero, &zero], [&zero, &zero, &zero]],
            tau: std::array::from_fn(|s| xc.value.potentials[5 + s].as_slice()),
        };
        let plain = KineticPullback {
            velocities: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            density_chain: std::array::from_fn(|_| Vec::new()),
            active: false,
        };
        let changed = point_pullback(&series, &weights, &tau, &kinetic_delta, n, gga, mgga);
        let mechanical = point_pullback(&series, &weights, &tau, &plain, n, gga, mgga);
        for s in 0..2 {
            let out = &mut out[s];
            for (to, a, b) in [
                (&mut out.0, &changed[s].0, &mechanical[s].0),
                (&mut out.1, &changed[s].1, &mechanical[s].1),
            ] {
                for ((to, a), b) in to.iter_mut().zip(a).zip(b) {
                    *to += a - b;
                }
            }
        }
    }
    Ok(out)
}

impl Geometry {
    /// Order-`n` coefficient of the change of the Kohn-Sham matrices caused
    /// by the density change series `q`, which need not be Hermitian, on the
    /// density series `p`.
    fn response_coefficient(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Cm>; 2],
        q: &[Vec<Cm>; 2],
        n: usize,
    ) -> Result<[Cm; 2]> {
        let (j, k, klr) = self.coulomb_exchange(q, n)?;
        let mut out = [j.clone(), j];
        for s in 0..2 {
            cadd(&mut out[s], &k[s], -self.cx);
            cadd(&mut out[s], &klr[s], -self.clr);
        }
        self.add_functional_tangent(functional, p, q, n, &mut out)?;
        Ok(out)
    }

    fn add_functional_tangent(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Cm>; 2],
        q: &[Vec<Cm>; 2],
        n: usize,
        out: &mut [Cm; 2],
    ) -> Result<()> {
        let nao = self.nao;
        if self.points.is_empty() {
            return Ok(());
        }
        // q = h + i a with Hermitian h and a: the functional responds to
        // each of the two real density changes.
        let hermitian: [Vec<Cm>; 2] = std::array::from_fn(|s| {
            q[s][..=n]
                .iter()
                .map(|m| {
                    let mut h = m.clone();
                    cadd(&mut h, &adjoint(m), 1.0);
                    for value in h.re.as_mut_slice().iter_mut().chain(h.im.as_mut_slice()) {
                        *value *= 0.5;
                    }
                    h
                })
                .collect()
        });
        let anti: [Vec<Cm>; 2] = std::array::from_fn(|s| {
            q[s][..=n]
                .iter()
                .map(|m| {
                    // (q - q^H) / (2 i).
                    let mut difference = m.clone();
                    cadd(&mut difference, &adjoint(m), -1.0);
                    let mut re = difference.im;
                    let mut im = difference.re;
                    for value in re.as_mut_slice() {
                        *value *= 0.5;
                    }
                    for value in im.as_mut_slice() {
                        *value *= -0.5;
                    }
                    Cm { re, im }
                })
                .collect()
        });
        for (part, imaginary) in [(&hermitian, false), (&anti, true)] {
            if part.iter().flatten().all(|m| largest(m) == 0.0) {
                continue;
            }
            let zero = || -> PointMatrices {
                std::array::from_fn(|_| (vec![0.0; nao * nao], vec![0.0; nao * nao]))
            };
            let total = self
                .points
                .par_iter()
                .try_fold(zero, |mut acc, point| -> Result<PointMatrices> {
                    let matrices =
                        point_xc_linear(point, functional, p, part, n, self.kinetic_model)?;
                    for s in 0..2 {
                        for (t, v) in acc[s].0.iter_mut().zip(&matrices[s].0) {
                            *t += *v;
                        }
                        for (t, v) in acc[s].1.iter_mut().zip(&matrices[s].1) {
                            *t += *v;
                        }
                    }
                    Ok(acc)
                })
                .try_reduce(zero, |mut a, b| {
                    for s in 0..2 {
                        for (t, v) in a[s].0.iter_mut().zip(&b[s].0) {
                            *t += *v;
                        }
                        for (t, v) in a[s].1.iter_mut().zip(&b[s].1) {
                            *t += *v;
                        }
                    }
                    Ok(a)
                })?;
            for s in 0..2 {
                let (re, im) = (&total[s].0, &total[s].1);
                if imaginary {
                    // i (x + i y) = -y + i x.
                    for (t, v) in out[s].re.as_mut_slice().iter_mut().zip(im) {
                        *t -= *v;
                    }
                    for (t, v) in out[s].im.as_mut_slice().iter_mut().zip(re) {
                        *t += *v;
                    }
                } else {
                    for (t, v) in out[s].re.as_mut_slice().iter_mut().zip(re) {
                        *t += *v;
                    }
                    for (t, v) in out[s].im.as_mut_slice().iter_mut().zip(im) {
                        *t += *v;
                    }
                }
            }
        }
        Ok(())
    }
}

/// `Tr[a b]` as real and imaginary parts.
pub(crate) fn complex_trace(a: &Cm, b: &Cm) -> Cx {
    let mut sum = (0.0, 0.0);
    for i in 0..a.re.rows {
        for j in 0..a.re.cols {
            sum.0 += a.re[(i, j)] * b.re[(j, i)] - a.im[(i, j)] * b.im[(j, i)];
            sum.1 += a.re[(i, j)] * b.im[(j, i)] + a.im[(i, j)] * b.re[(j, i)];
        }
    }
    sum
}

/// Taylor coefficients in a magnetic field of the frequency-dependent
/// polarizability.
#[derive(Clone, Debug)]
pub struct FieldPolarizabilityPath {
    /// Photon energy, Hartree.
    pub frequency: f64,
    /// Real parts `[order][a][b]` of the coefficients of `alpha_ab(w; t)`:
    /// observable axis `a`, field axis `b`.
    pub real: Vec<[[f64; 3]; 3]>,
    /// Imaginary parts.
    pub imaginary: Vec<[[f64; 3]; 3]>,
    pub reference_defect: f64,
    pub residuals: Vec<f64>,
}

/// Polarizability `alpha_ab(w)` of the paired response along the magnetic
/// path `B(t) = t magnetic`.
///
/// In the transported frame of the field path the reference density `D(t)`
/// is a complex projector that commutes with the Kohn-Sham matrix `F(t)`,
/// and the first-order density of `H' = +E.r` at the frequency `w` obeys
/// `w d = [F, d] + [r + G[d], D]`, `d = D d + d D`, order by order. The
/// projector condition fixes the occupied-occupied and virtual-virtual
/// blocks; the real and the imaginary part of the occupied-virtual blocks
/// each solve the paired equations of the reference. The first-order
/// coefficient is the Faraday tensor (imaginary, antisymmetric), the
/// second-order one the hypermagnetizability at the frequency.
#[allow(clippy::too_many_arguments)]
pub fn field_polarizability_path<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    magnetic: [f64; 3],
    origin: Vec3,
    london: bool,
    order: usize,
    omega: f64,
    config: &ResponseSolverConfig,
) -> Result<FieldPolarizabilityPath> {
    if omega < 0.0 {
        return Err(DftError::InvalidInput(
            "the polarizability needs a finite non-negative photon energy".into(),
        ));
    }
    causal_field_polarizability_path(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        magnetic,
        origin,
        london,
        order,
        Frequency::Retarded {
            omega,
            damping: 0.0,
        },
        config,
    )
}

/// Magnetic Taylor coefficients of the causal electric dipole response.
/// The frequency follows the exp(-i omega t) convention and the real and
/// imaginary parts are retained at each magnetic order.
#[allow(clippy::too_many_arguments)]
fn causal_field_polarizability_path<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    magnetic: [f64; 3],
    origin: Vec3,
    london: bool,
    order: usize,
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<FieldPolarizabilityPath> {
    polarizability_path(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        magnetic,
        origin,
        london,
        None,
        order,
        frequency.energy()?,
        config,
    )
}

/// `field_polarizability_path` with an additional first-order one-electron
/// term per spin in the static perturbation.
#[allow(clippy::too_many_arguments)]
fn polarizability_path<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    magnetic: [f64; 3],
    origin: Vec3,
    london: bool,
    spin_core: Option<[Cm; 2]>,
    order: usize,
    frequency: Complex,
    config: &ResponseSolverConfig,
) -> Result<FieldPolarizabilityPath> {
    let (taylor, state) = field_path_state(
        mol, mb, functional, grid, scf, opts, [0.0; 3], magnetic, origin, london, order, spin_core,
        config,
    )?;
    let spin = if scf.unrestricted {
        TdaSpin::SpinResolved
    } else {
        TdaSpin::Singlet
    };
    let tda = TdaSolver::new(mol, mb, functional, grid, scf, opts, spin)?;
    let dim = tda.dim();
    let operator = CasidaOperator {
        solver: tda,
        frequency,
    };
    let spins = if scf.unrestricted { 2 } else { 1 };
    let n_occ = [scf.n_alpha, scf.n_beta];
    let nmo = [scf.mo_coeff[0].cols, scf.mo_coeff[1].cols];
    let mut real = vec![[[0.0; 3]; 3]; order + 1];
    let mut imaginary = vec![[[0.0; 3]; 3]; order + 1];
    let mut residuals = taylor.residuals.clone();
    let density = |d1: &[Vec<Cm>; 2], n: usize| -> [Vec<Cm>; 2] {
        std::array::from_fn(|s| {
            series_product(&series_product(&state.b[s], &d1[s], n), &state.bt[s], n)
        })
    };
    for a in 0..3 {
        let mut d1: [Vec<Cm>; 2] = [Vec::new(), Vec::new()];
        // Kohn-Sham matrix changes of the first-order density, AO basis.
        let mut g: [Vec<Cm>; 2] = [Vec::new(), Vec::new()];
        for n in 0..=order {
            for s in 0..2 {
                let mut fixed = Cm::zeros(nmo[s], nmo[s]);
                for k in 1..=n {
                    cadd(&mut fixed, &cmul(&state.d[s][k], &d1[s][n - k]), 1.0);
                    cadd(&mut fixed, &cmul(&d1[s][n - k], &state.d[s][k]), 1.0);
                }
                let no = n_occ[s];
                let mut start = Cm::zeros(nmo[s], nmo[s]);
                for i in 0..nmo[s] {
                    for j in 0..nmo[s] {
                        let sign = if i < no && j < no {
                            -1.0
                        } else if i >= no && j >= no {
                            1.0
                        } else {
                            0.0
                        };
                        start.re[(i, j)] = sign * fixed.re[(i, j)];
                        start.im[(i, j)] = sign * fixed.im[(i, j)];
                    }
                }
                d1[s].push(start);
            }
            let trial =
                state
                    .geometry
                    .response_coefficient(functional, &state.p, &density(&d1, n), n)?;
            for s in 0..2 {
                g[s].push(trial[s].clone());
            }
            let mut source = vec![Complex::ZERO; 2 * dim];
            let mut offset = 0;
            for s in 0..spins {
                let total: Vec<Cm> = (0..=n)
                    .map(|k| {
                        let mut m = state.geometry.one.dipole[a][k].clone();
                        cadd(&mut m, &g[s][k], 1.0);
                        m
                    })
                    .collect();
                let w = series_product(&series_product(&state.bt[s], &total, n), &state.b[s], n);
                let mut r = Cm::zeros(nmo[s], nmo[s]);
                for k in 0..=n {
                    cadd(&mut r, &cmul(&state.f[s][k], &d1[s][n - k]), 1.0);
                    cadd(&mut r, &cmul(&d1[s][n - k], &state.f[s][k]), -1.0);
                    cadd(&mut r, &cmul(&w[k], &state.d[s][n - k]), 1.0);
                    cadd(&mut r, &cmul(&state.d[s][n - k], &w[k]), -1.0);
                }
                // The frequency term on the prescribed diagonal blocks
                // has no occupied-virtual projection.
                let no = n_occ[s];
                let nv = nmo[s] - no;
                for v in 0..nv {
                    for i in 0..no {
                        let k = offset + v * no + i;
                        let lower = Complex::new(r.re[(no + v, i)], r.im[(no + v, i)]);
                        let upper = Complex::new(r.re[(i, no + v)], r.im[(i, no + v)]);
                        source[k] = (lower - upper) * -0.5;
                        source[dim + k] = (lower + upper) * -0.5;
                    }
                }
                offset += no * nv;
            }
            let amplitudes = if frequency.im == 0.0 {
                // Keep the two real-frequency residuals and the real
                // Hermitian solves of the existing public calculation.
                let real: Vec<_> = source.iter().map(|z| Complex::from(z.re)).collect();
                let imag: Vec<_> = source.iter().map(|z| Complex::from(z.im)).collect();
                let x = krylov::solve(&operator, &real, None, config)?;
                let y = krylov::solve(&operator, &imag, None, config)?;
                residuals.extend([x.relative_residual, y.relative_residual]);
                x.amplitudes
                    .into_iter()
                    .zip(y.amplitudes)
                    .map(|(x, y)| Complex::new(x.re, y.re))
                    .collect()
            } else {
                let solution = krylov::solve(&operator, &source, None, config)?;
                residuals.push(solution.relative_residual);
                solution.amplitudes
            };
            let mut offset = 0;
            for s in 0..spins {
                let no = n_occ[s];
                let nv = nmo[s] - no;
                for v in 0..nv {
                    for i in 0..no {
                        let k = offset + v * no + i;
                        let lower = amplitudes[k] + amplitudes[dim + k];
                        let upper = amplitudes[k] - amplitudes[dim + k];
                        d1[s][n].re[(no + v, i)] = lower.re;
                        d1[s][n].re[(i, no + v)] = upper.re;
                        d1[s][n].im[(no + v, i)] = lower.im;
                        d1[s][n].im[(i, no + v)] = upper.im;
                    }
                }
                offset += no * nv;
            }
            if spins == 1 {
                let copy = d1[0][n].clone();
                d1[1][n] = copy;
            }
            let updated =
                state
                    .geometry
                    .response_coefficient(functional, &state.p, &density(&d1, n), n)?;
            for s in 0..2 {
                g[s][n] = updated[s].clone();
            }
        }
        let q = density(&d1, order);
        for n in 0..=order {
            for b in 0..3 {
                let mut value = (0.0, 0.0);
                for m in 0..=n {
                    for s in 0..2 {
                        let t = complex_trace(&q[s][m], &state.geometry.one.dipole[b][n - m]);
                        value = (value.0 + t.0, value.1 + t.1);
                    }
                }
                real[n][b][a] = -value.0;
                imaginary[n][b][a] = -value.1;
            }
        }
    }
    Ok(FieldPolarizabilityPath {
        frequency: frequency.re,
        real,
        imaginary,
        reference_defect: taylor.reference_defect,
        residuals,
    })
}

/// Polarizability at a frequency with its first and second magnetic-field
/// derivatives.
#[derive(Clone, Debug)]
pub struct DynamicHypermagnetizability {
    /// Photon energy, Hartree.
    pub frequency: f64,
    /// `alpha[a][b]` at the frequency.
    pub polarizability: [[f64; 3]; 3],
    /// `faraday[a][b][c] = Im d alpha_ab / dB_c`.
    pub faraday: [[[f64; 3]; 3]; 3],
    /// `eta[a][b][c][d] = d2 Re alpha_ab / (dB_c dB_d)`.
    pub tensor: [[[[f64; 3]; 3]; 3]; 3],
    pub residual: f64,
    pub directional_evaluations: usize,
}

/// `d alpha_ab(w) / dB_c` and `d2 alpha_ab(w) / (dB_c dB_d)` from the
/// second-order magnetic paths of every Cartesian direction and pair sum.
#[allow(clippy::too_many_arguments)]
pub fn dynamic_hypermagnetizability<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    origin: Vec3,
    london: bool,
    omega: f64,
    config: &ResponseSolverConfig,
) -> Result<DynamicHypermagnetizability> {
    let mut residual = 0.0_f64;
    let mut evaluations = 0_usize;
    let mut run = |direction: [f64; 3]| -> Result<FieldPolarizabilityPath> {
        let path = field_polarizability_path(
            mol, mb, functional, grid, scf, opts, direction, origin, london, 2, omega, config,
        )?;
        residual = path.residuals.iter().copied().fold(residual, f64::max);
        evaluations += 1;
        Ok(path)
    };
    let mut polarizability = [[0.0; 3]; 3];
    let mut faraday = [[[0.0; 3]; 3]; 3];
    let mut tensor = [[[[0.0; 3]; 3]; 3]; 3];
    let mut second = Vec::with_capacity(3);
    for c in 0..3 {
        let mut unit = [0.0; 3];
        unit[c] = 1.0;
        let path = run(unit)?;
        polarizability = path.real[0];
        for a in 0..3 {
            for b in 0..3 {
                faraday[a][b][c] = path.imaginary[1][a][b];
                tensor[a][b][c][c] = 2.0 * path.real[2][a][b];
            }
        }
        second.push(path.real[2]);
    }
    for c in 0..3 {
        for d in 0..c {
            let mut direction = [0.0; 3];
            direction[c] = 1.0;
            direction[d] = 1.0;
            let path = run(direction)?;
            for a in 0..3 {
                for b in 0..3 {
                    let value = path.real[2][a][b] - second[c][a][b] - second[d][a][b];
                    tensor[a][b][c][d] = value;
                    tensor[a][b][d][c] = value;
                }
            }
        }
    }
    Ok(DynamicHypermagnetizability {
        frequency: omega,
        polarizability,
        faraday,
        tensor,
        residual,
        directional_evaluations: evaluations,
    })
}

/// Polarizability at a frequency and its first magnetic-field derivative,
/// the Faraday tensor `Im d alpha_ab / dB_c`, with the largest residual.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn faraday_tensor<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    origin: Vec3,
    london: bool,
    omega: f64,
    config: &ResponseSolverConfig,
) -> Result<([[f64; 3]; 3], [[[f64; 3]; 3]; 3], f64)> {
    if omega < 0.0 {
        return Err(DftError::InvalidInput(
            "the polarizability needs a finite non-negative photon energy".into(),
        ));
    }
    let response = causal_faraday_tensor(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        origin,
        london,
        Frequency::Retarded {
            omega,
            damping: 0.0,
        },
        config,
    )?;
    Ok((
        response.polarizability.map(|row| row.map(|z| z.re)),
        response
            .derivative
            .map(|row| row.map(|entry| entry.map(|z| z.im))),
        response.residual,
    ))
}

/// Complex electric susceptibility and its first magnetic perturbation.
#[derive(Clone, Debug)]
pub struct CausalMagnetoOpticalResponse {
    pub polarizability: [[Complex; 3]; 3],
    /// Electric observable, electric source, magnetic perturbation axis.
    pub derivative: [[[Complex; 3]; 3]; 3],
    pub residual: f64,
    pub reference_defect: f64,
}

fn assemble_magneto_optical(
    mut path: impl FnMut(usize) -> Result<FieldPolarizabilityPath>,
) -> Result<CausalMagnetoOpticalResponse> {
    let mut result = CausalMagnetoOpticalResponse {
        polarizability: [[Complex::ZERO; 3]; 3],
        derivative: [[[Complex::ZERO; 3]; 3]; 3],
        residual: 0.0,
        reference_defect: 0.0,
    };
    for c in 0..3 {
        let value = path(c)?;
        result.residual = value
            .residuals
            .iter()
            .copied()
            .fold(result.residual, f64::max);
        result.reference_defect = result.reference_defect.max(value.reference_defect);
        for a in 0..3 {
            for b in 0..3 {
                result.polarizability[a][b] =
                    Complex::new(value.real[0][a][b], value.imaginary[0][a][b]);
                result.derivative[a][b][c] =
                    Complex::new(value.real[1][a][b], value.imaginary[1][a][b]);
            }
        }
    }
    Ok(result)
}

/// Causal Faraday response with both parts of the magnetic derivative.
#[allow(clippy::too_many_arguments)]
pub fn causal_faraday_tensor<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    origin: Vec3,
    london: bool,
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<CausalMagnetoOpticalResponse> {
    assemble_magneto_optical(|c| {
        let mut unit = [0.0; 3];
        unit[c] = 1.0;
        causal_field_polarizability_path(
            mol, mb, functional, grid, scf, opts, unit, origin, london, 1, frequency, config,
        )
    })
}

/// First-order change of the frequency-dependent polarizability under a
/// one-electron spin-orbit operator on a collinear open-shell reference.
#[derive(Clone, Debug)]
pub struct SpinOrbitPolarizability {
    /// Photon energy, Hartree.
    pub frequency: f64,
    /// `alpha[a][b]` without spin-orbit coupling.
    pub polarizability: [[f64; 3]; 3],
    /// `Im d alpha_ab / d lambda` for the spin axis `k`, as `[a][b][k]`.
    pub imaginary: [[[f64; 3]; 3]; 3],
    /// `Re d alpha_ab / d lambda`, which vanishes.
    pub real: [[[f64; 3]; 3]; 3],
    pub residual: f64,
}

/// Magneto-optical tensor of a collinear open-shell reference to first order
/// in the spin-orbit coupling. With the spins along the axis `k` only the
/// component `k` of `H_SO = sum_k (-i O_k) s_k` acts in first order, as
/// `-(i/2) O_k` on the alpha and `+(i/2) O_k` on the beta orbitals, where
/// `orbital[k]` is the real antisymmetric matrix `O_k`. The first-order
/// coefficient of the polarizability in that perturbation is imaginary and
/// antisymmetric in its two indices: the tensor behind the Faraday rotation
/// and the magneto-optical Kerr effect of a magnetized system.
#[allow(clippy::too_many_arguments)]
pub fn spin_orbit_polarizability<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    orbital: &[Matrix; 3],
    omega: f64,
    config: &ResponseSolverConfig,
) -> Result<SpinOrbitPolarizability> {
    if omega < 0.0 {
        return Err(DftError::InvalidInput(
            "the polarizability needs a finite non-negative photon energy".into(),
        ));
    }
    let response = causal_spin_orbit_polarizability(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        orbital,
        Frequency::Retarded {
            omega,
            damping: 0.0,
        },
        config,
    )?;
    Ok(SpinOrbitPolarizability {
        frequency: omega,
        polarizability: response.polarizability.map(|row| row.map(|z| z.re)),
        imaginary: response
            .derivative
            .map(|row| row.map(|entry| entry.map(|z| z.im))),
        real: response
            .derivative
            .map(|row| row.map(|entry| entry.map(|z| z.re))),
        residual: response.residual,
    })
}

/// Causal electric susceptibility and its derivative with respect to the
/// one-electron spin-orbit perturbation on a collinear open-shell reference.
#[allow(clippy::too_many_arguments)]
pub fn causal_spin_orbit_polarizability<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    orbital: &[Matrix; 3],
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<CausalMagnetoOpticalResponse> {
    if !scf.unrestricted {
        return Err(DftError::InvalidInput(
            "the spin-orbit magneto-optical response needs an unrestricted open-shell reference"
                .into(),
        ));
    }
    let nao = mb.nao_sph;
    let energy = frequency.energy()?;
    assemble_magneto_optical(|k| {
        let mut up = Cm::zeros(nao, nao);
        let mut down = Cm::zeros(nao, nao);
        add(&mut up.im, &orbital[k], -0.5);
        add(&mut down.im, &orbital[k], 0.5);
        polarizability_path(
            mol,
            mb,
            functional,
            grid,
            scf,
            opts,
            [0.0; 3],
            Vec3::zero(),
            false,
            Some([up, down]),
            1,
            energy,
            config,
        )
    })
}

/// Linear magnetoelectric response of a collinear open-shell reference to
/// first order in a one-electron spin-orbit operator.
#[derive(Clone, Debug)]
pub struct SpinOrbitMagnetoelectric {
    /// `d mu_a / dB_b` per unit spin-orbit strength with the spins along
    /// the axis `k`, as `[a][b][k]`, atomic units.
    pub tensor: [[[f64; 3]; 3]; 3],
    /// Number of field paths that were run.
    pub evaluations: usize,
    pub accumulation_roundoff: f64,
    pub residual: f64,
}

/// Electric dipole induced by a magnetic field on a collinear open-shell
/// reference, to first order in the spin-orbit coupling: minus the mixed
/// third derivative of the energy in an electric field component, a
/// magnetic field component and the strength of `H_SO` for the spins along
/// one axis. Every combined direction is one field path through third
/// order, where the spin-orbit operator enters as the first-order
/// one-electron term `-(i/2) O_k` on the alpha and `+(i/2) O_k` on the beta
/// orbitals; the mixed derivative follows from the polarization identity.
/// With `london` the result does not depend on the gauge origin.
#[allow(clippy::too_many_arguments)]
pub fn spin_orbit_magnetoelectric(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    orbital: &[Matrix; 3],
    origin: Vec3,
    london: bool,
    config: &ResponseSolverConfig,
) -> Result<SpinOrbitMagnetoelectric> {
    if !scf.unrestricted {
        return Err(DftError::InvalidInput(
            "the spin-orbit magnetoelectric response needs an unrestricted open-shell reference"
                .into(),
        ));
    }
    let nao = mb.nao_sph;
    // Coordinates: electric field (3), magnetic field (3), spin-orbit
    // strength for the three spin axes (3).
    let mut cache: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut evaluations = 0_usize;
    let mut residual = 0.0_f64;
    let mut roundoff = 0.0_f64;
    let mut tensor = [[[0.0; 3]; 3]; 3];
    for k in 0..3 {
        for a in 0..3 {
            for b in 0..3 {
                let vectors: Vec<Vec<f64>> = [a, 3 + b, 6 + k]
                    .iter()
                    .map(|&index| {
                        let mut v = vec![0.0; 9];
                        v[index] = 1.0;
                        v
                    })
                    .collect();
                let mixed = polarized_mixed(9, &vectors, |direction, total| {
                    if let Some((_, energy)) =
                        cache.iter().find(|(key, _)| key.as_slice() == direction)
                    {
                        return Ok(vec![energy[total]]);
                    }
                    let weights = &direction[6..9];
                    let spin_core = if weights.iter().all(|w| *w == 0.0) {
                        None
                    } else {
                        let mut up = Cm::zeros(nao, nao);
                        let mut down = Cm::zeros(nao, nao);
                        for (axis, w) in weights.iter().enumerate() {
                            add(&mut up.im, &orbital[axis], -0.5 * w);
                            add(&mut down.im, &orbital[axis], 0.5 * w);
                        }
                        Some([up, down])
                    };
                    let (taylor, _) = field_path_state(
                        mol,
                        mb,
                        functional,
                        grid,
                        scf,
                        opts,
                        [direction[0], direction[1], direction[2]],
                        [direction[3], direction[4], direction[5]],
                        origin,
                        london,
                        3,
                        spin_core,
                        config,
                    )?;
                    evaluations += 1;
                    residual = taylor.residuals.iter().copied().fold(residual, f64::max);
                    let value = taylor.energy[total];
                    cache.push((direction.to_vec(), taylor.energy));
                    Ok(vec![value])
                })?;
                roundoff = roundoff.max(mixed.accumulation_roundoff[0]);
                // mu = -dE/dF.
                tensor[a][b][k] = -mixed.value[0];
            }
        }
    }
    Ok(SpinOrbitMagnetoelectric {
        tensor,
        evaluations,
        accumulation_roundoff: roundoff,
        residual,
    })
}
