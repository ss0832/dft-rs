// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Global spin projection of generic semilocal providers. The spin scaling
//! derivative restores the collinear limit and avoids division by |m|.
//! Native second derivatives and nested duals give the variational potential.

use crate::dual::{DualN, Scalar};
use crate::grid::gauss_legendre;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Maximum verified tensor-product spin-sphere quadrature order.
/// Each caller still checks consecutive rules against its requested tolerance.
pub const MAX_SPIN_ANGULAR_ORDER: usize = 128;

mod collinear_grid;
mod directional;
mod grid_kernel;
mod kernel;
pub(super) use grid_kernel::GridKernel;
mod taylor;
pub use directional::{spinor_semilocal_point_directional_taylor, SpinorXcDirectionalTaylor};
pub use kernel::{spinor_semilocal_point_kernel, SpinorXcPointKernel};
pub(crate) use taylor::lda_endpoint_taylor;
pub use taylor::{
    spinor_semilocal_energy_taylor, spinor_semilocal_point_taylor, SpinorXcPointTaylor,
};

/// Pauli descriptors: number/spin density, their spatial gradients, kinetic
/// density and current. Kinetic/current descriptors must use the same gauge
/// frame. Their combination tau-j^2/(2 rho) is invariant under U(1).
#[derive(Clone, Debug)]
pub struct SpinorXcPoint {
    pub density: [f64; 4],
    pub gradient: [[f64; 3]; 4],
    pub tau: [f64; 4],
    pub current: [[f64; 3]; 4],
}
impl SpinorXcPoint {
    /// Density[4], gradient[4][3], tau[4], current[4][3].
    pub fn descriptors(&self) -> [f64; 32] {
        std::array::from_fn(|i| match i {
            0..=3 => self.density[i],
            4..=15 => self.gradient[(i - 4) / 3][(i - 4) % 3],
            16..=19 => self.tau[i - 16],
            _ => self.current[(i - 20) / 3][(i - 20) % 3],
        })
    }
    pub fn from_descriptors(x: [f64; 32]) -> Self {
        Self {
            density: std::array::from_fn(|a| x[a]),
            gradient: std::array::from_fn(|a| std::array::from_fn(|d| x[4 + 3 * a + d])),
            tau: std::array::from_fn(|a| x[16 + a]),
            current: std::array::from_fn(|a| std::array::from_fn(|d| x[20 + 3 * a + d])),
        }
    }
}

/// Actual quadrature result; angular convergence is a separate measurement.
#[derive(Clone)]
pub struct SpinorXcPointResult {
    pub energy: f64,
    pub gradient: [f64; 32],
    pub component_energies: Vec<f64>,
    pub angular_order: usize,
    pub directions: usize,
}
type Inner = DualN<f64, 1>;
fn spin_descriptor(i: usize) -> bool {
    match i {
        0..=3 => i != 0,
        4..=15 => (i - 4) / 3 != 0,
        16..=19 => i != 16,
        _ => (i - 20) / 3 != 0,
    }
}
fn project<S: Scalar>(
    x: &[S; 32],
    direction: [f64; 3],
    family: Family,
    inactive_limit: bool,
) -> Result<[S; 7]> {
    let sum = |start: usize, stride: usize| {
        (0..3).fold(S::from_f64(0.0), |a, s| {
            a + x[start + stride * (s + 1)].scale(direction[s])
        })
    };
    let magnetization = sum(0, 1);
    let rho = [
        (x[0] + magnetization).scale(0.5),
        (x[0] - magnetization).scale(0.5),
    ];
    let grad: [[S; 3]; 2] = std::array::from_fn(|spin| {
        std::array::from_fn(|d| {
            let magnetic = sum(4 + d, 3);
            (x[4 + d] + magnetic.scale(if spin == 0 { 1.0 } else { -1.0 })).scale(0.5)
        })
    });
    let contraction =
        |a: usize, b: usize| (0..3).fold(S::from_f64(0.0), |v, d| v + grad[a][d] * grad[b][d]);
    let sigma = [contraction(0, 0), contraction(0, 1), contraction(1, 1)];
    let mut tau = [S::from_f64(0.0); 2];
    if family == Family::MetaGga {
        let spin_tau = sum(16, 1);
        for spin in 0..2 {
            let sign = if spin == 0 { 1.0 } else { -1.0 };
            tau[spin] = (x[16] + spin_tau.scale(sign)).scale(0.5);
            if rho[spin].value() <= 0.0 {
                // Each projected descriptor subtracts two separately accumulated
                // contractions. In a collinear empty channel those contractions
                // can differ by roundoff. Keep the zero-density limit only when
                // that bound also accounts for the kinetic density and current.
                let cancellation = |projected: S, charge: S, polarized: S| {
                    projected.value().abs()
                        <= 64.0 * f64::EPSILON * charge.value().abs().max(polarized.value().abs())
                };
                // The real part of the AO momentum contraction can itself
                // cancel at a stationary point. The charge-density and
                // kinetic-density bound on that contraction retains its scale
                // before cancellation, including for London orbitals.
                let current_scale = std::f64::consts::SQRT_2
                    * x[0].value().abs().sqrt()
                    * x[16].value().abs().sqrt();
                if inactive_limit
                    && cancellation(rho[spin], x[0], magnetization)
                    && cancellation(tau[spin], x[16], spin_tau)
                    && (0..3).all(|d| {
                        let charge = x[20 + d];
                        let polarized = sum(20 + d, 3);
                        let residual = (charge + polarized.scale(sign)).scale(0.5);
                        residual.value().abs()
                            <= 64.0
                                * f64::EPSILON
                                * current_scale
                                    .max(charge.value().abs())
                                    .max(polarized.value().abs())
                    })
                {
                    tau[spin] = S::from_f64(0.0);
                    continue;
                }
                return Err(DftError::InvalidInput(
                    "current-corrected meta-GGA spin projection requires a positive \
                    projected density; resolve exact nodal projections with an angular rule \
                    that does not sample the node"
                        .into(),
                ));
            }
            let mut current_square = S::from_f64(0.0);
            for d in 0..3 {
                let j = (x[20 + d] + sum(20 + d, 3).scale(sign)).scale(0.5);
                current_square = current_square + j * j;
            }
            let flow = current_square / rho[spin].scale(2.0);
            let corrected = tau[spin] - flow;
            let von_weizsaecker = sigma[2 * spin] / rho[spin].scale(8.0);
            // At the one-orbital boundary the corrected kinetic density
            // equals the von Weizsaecker value. Subtracting the flow can
            // lose that equality when the phase gradient dominates the
            // amplitude gradient. Resolve roundoff using the scales before
            // subtraction and keep the analytic variation of tau - flow.
            let difference = von_weizsaecker.value() - corrected.value();
            let scale = tau[spin].value().abs().max(flow.value().abs());
            tau[spin] = if difference.abs() <= 64.0 * f64::EPSILON * scale {
                corrected + S::from_f64(difference)
            } else {
                corrected
            };
        }
    }
    Ok([rho[0], rho[1], sigma[0], sigma[1], sigma[2], tau[0], tau[1]])
}

// Exact static GGA projection: densities are linear and gradient contractions
// quadratic in the independent Pauli descriptors. Their mixed spin-scaling
// Jacobians have no density/gradient cross terms. The generic projection is
// retained for every other family and all high-order paths.
fn project_gga_sparse<const N: usize>(values: &[f64; 32], axis: [f64; 3]) -> [DualN<Inner, N>; 7] {
    debug_assert_eq!(N, 16);
    let spin = (0..3).fold(0.0, |sum, a| sum + values[a + 1] * axis[a]);
    let magnetic_gradient: [f64; 3] = std::array::from_fn(|d| {
        (0..3).fold(0.0, |sum, a| sum + values[4 + 3 * (a + 1) + d] * axis[a])
    });
    let grad: [[f64; 3]; 2] = std::array::from_fn(|s| {
        std::array::from_fn(|d| {
            (values[4 + d]
                + if s == 0 {
                    magnetic_gradient[d]
                } else {
                    -magnetic_gradient[d]
                })
                * 0.5
        })
    });
    let tangent: [[f64; 3]; 2] = std::array::from_fn(|s| {
        std::array::from_fn(|d| {
            (if s == 0 {
                magnetic_gradient[d]
            } else {
                -magnetic_gradient[d]
            }) * 0.5
        })
    });
    let mut q = [DualN::<Inner, N>::constant(Inner::constant(0.0)); 7];
    for (s, density) in q.iter_mut().take(2).enumerate() {
        let sign = if s == 0 { 1.0 } else { -1.0 };
        density.re = Inner {
            re: (values[0] + sign * spin) * 0.5,
            im: [sign * spin * 0.5],
        };
        for (i, partial) in density.im.iter_mut().take(4).enumerate() {
            let j = if i == 0 {
                0.5
            } else {
                sign * axis[i - 1] * 0.5
            };
            *partial = Inner {
                re: j,
                im: [if i == 0 { 0.0 } else { j }],
            };
        }
    }
    for (k, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        let r = &mut q[k + 2];
        r.re.re = (0..3).fold(0.0, |v, d| v + grad[s][d] * grad[t][d]);
        r.re.im[0] = (0..3).fold(0.0, |v, d| {
            v + (tangent[s][d] * grad[t][d] + grad[s][d] * tangent[t][d])
        });
        for (i, partial) in r.im.iter_mut().enumerate().skip(4) {
            let d = (i - 4) % 3;
            let component = (i - 4) / 3;
            let coefficient = |spin: usize| {
                if component == 0 {
                    0.5
                } else {
                    (if spin == 0 { 1.0 } else { -1.0 }) * axis[component - 1] * 0.5
                }
            };
            let a = coefficient(s);
            let b = coefficient(t);
            let da = if component == 0 { 0.0 } else { a };
            let db = if component == 0 { 0.0 } else { b };
            partial.re = a * grad[t][d] + grad[s][d] * b;
            partial.im[0] =
                (da * grad[t][d] + a * tangent[t][d]) + (tangent[s][d] * b + grad[s][d] * db);
        }
    }
    q
}

/// Global angular average of f + D_spin f. Differentiation includes the
/// density dependence of the spin-scaling direction, all density gradients
/// and the current correction to tau. No registered functional-name branch.
/// The explicit angular order is not a claim of angular convergence.
pub fn spinor_semilocal_point(
    functional: &ResolvedFunctional,
    point: &SpinorXcPoint,
    angular_order: usize,
) -> Result<SpinorXcPointResult> {
    match functional.family {
        Family::Lda => spinor_semilocal_point_active::<4>(functional, point, angular_order),
        Family::Gga => spinor_semilocal_point_active::<16>(functional, point, angular_order),
        Family::MetaGga => spinor_semilocal_point_active::<32>(functional, point, angular_order),
    }
}

fn spinor_semilocal_point_active<const N: usize>(
    functional: &ResolvedFunctional,
    point: &SpinorXcPoint,
    angular_order: usize,
) -> Result<SpinorXcPointResult> {
    let values = point.descriptors();
    if !(4..=MAX_SPIN_ANGULAR_ORDER).contains(&angular_order)
        || angular_order % 2 != 0
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "global spin projection requires finite descriptors and even angular order 4..128"
                .into(),
        ));
    }
    if !density_is_psd_to_roundoff(point.density) {
        return Err(DftError::InvalidInput(
            "spin projection requires a positive-semidefinite local spin density".into(),
        ));
    }
    let x: [DualN<Inner, N>; 32] = std::array::from_fn(|i| {
        let inner = Inner {
            re: values[i],
            im: [if spin_descriptor(i) { values[i] } else { 0.0 }],
        };
        let mut jet = if i < N {
            DualN::<Inner, N>::variable(inner, i)
        } else {
            DualN::<Inner, N>::constant(inner)
        };
        if i < N && spin_descriptor(i) {
            jet.im[i].im[0] = 1.0;
        }
        jet
    });
    // Tile only temporary descriptors. Directions, providers, angular weights
    // and the scalar accumulation order are identical to the untiled path.
    const TILE: usize = 256;
    let mut energy = 0.0;
    let mut gradient = [0.0; 32];
    let mut components = vec![0.0; functional.parts.len()];
    let mut integrate = |jets: &[[DualN<Inner, N>; 7]], quadrature: &[f64]| -> Result<()> {
        let data: [Vec<f64>; 7] =
            std::array::from_fn(|a| jets.iter().map(|q| q[a].value()).collect());
        let vars = GridVars {
            rho: [&data[0], &data[1]],
            sigma: (functional.family != Family::Lda).then_some([&data[2], &data[3], &data[4]]),
            tau: (functional.family == Family::MetaGga).then_some([&data[5], &data[6]]),
        };
        let mut out = XcOut::default();
        functional.eval(&vars, 2, &mut out)?;
        let invariants = match functional.family {
            Family::Lda => 2,
            Family::Gga => 5,
            Family::MetaGga => 7,
        };
        for (g, (q, w)) in jets.iter().zip(quadrature).enumerate() {
            let (potential, kernel) = out.potential_and_kernel(g, functional.family)?;
            let effective = out.e[g]
                + (0..invariants)
                    .map(|a| potential[a] * q[a].re.im[0])
                    .sum::<f64>();
            energy += w * effective;
            for (i, derivative) in gradient.iter_mut().enumerate().take(N) {
                let mut value = 0.0;
                for a in 0..invariants {
                    value += potential[a] * (q[a].im[i].re + q[a].im[i].im[0]);
                    // The kernel was checked for finite entries above. This
                    // exact-zero sparsity introduces no screening tolerance.
                    if q[a].im[i].re != 0.0 {
                        for b in 0..invariants {
                            value += kernel[a][b] * q[a].im[i].re * q[b].re.im[0];
                        }
                    }
                }
                *derivative += w * value;
            }
        }
        for (part_index, (coefficient, part)) in functional.parts.iter().enumerate() {
            let mut output = XcOut::default();
            output.prepare(jets.len(), 1, functional.family);
            part.eval_add(&vars, 1, *coefficient, &mut output)?;
            let mut value = components[part_index];
            for (g, (q, w)) in jets.iter().zip(quadrature).enumerate() {
                let mut derivative =
                    output.vrho[0][g] * q[0].re.im[0] + output.vrho[1][g] * q[1].re.im[0];
                if functional.family != Family::Lda {
                    for a in 0..3 {
                        derivative += output.vsigma[a][g] * q[a + 2].re.im[0];
                    }
                }
                if functional.family == Family::MetaGga {
                    for a in 0..2 {
                        derivative += output.vtau[a][g] * q[a + 5].re.im[0];
                    }
                }
                value += w * (output.e[g] + derivative);
            }
            components[part_index] = value;
        }
        Ok(())
    };
    let (nodes, weights) = gauss_legendre(angular_order);
    let azimuthal = 2 * angular_order;
    let mut jets = Vec::with_capacity(TILE);
    let mut quadrature = Vec::with_capacity(TILE);
    for (z, w) in nodes.into_iter().zip(weights) {
        let transverse = (1.0 - z * z).sqrt();
        for j in 0..azimuthal {
            let angle = 2.0 * PI * (j as f64 + 0.5) / azimuthal as f64;
            let axis = [transverse * angle.cos(), transverse * angle.sin(), z];
            jets.push(if functional.family == Family::Gga {
                project_gga_sparse(&values, axis)
            } else {
                project(&x, axis, functional.family, false)?
            });
            quadrature.push(w / (2.0 * azimuthal as f64));
            if jets.len() == TILE {
                integrate(&jets, &quadrature)?;
                jets.clear();
                quadrature.clear();
            }
        }
    }
    if !jets.is_empty() {
        integrate(&jets, &quadrature)?;
    }
    if !energy.is_finite()
        || gradient.iter().any(|v| !v.is_finite())
        || components.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "non-finite global spin-projection energy or variational potential".into(),
        ));
    }
    Ok(SpinorXcPointResult {
        energy,
        gradient,
        component_energies: components,
        angular_order,
        directions: 2 * angular_order * angular_order,
    })
}

/// Integrated generic semilocal energy and the actual variational spinor
/// operator on supplied complex orbitals and covariant momentum values.
/// Spatial and angular convergence must be measured independently by callers.
pub fn spinor_semilocal_grid(
    functional: &ResolvedFunctional,
    weights: &[f64],
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    density: &[[super::BlochMatrix; 4]],
    angular_order: usize,
) -> Result<(f64, Vec<super::BlochMatrix>, Vec<f64>)> {
    let value = spinor_semilocal_grid_response(
        functional,
        weights,
        orbitals,
        momentum,
        density,
        angular_order,
    )?;
    Ok((value.energy, value.potential, value.component_energies))
}

pub fn spinor_semilocal_grid_response(
    functional: &ResolvedFunctional,
    weights: &[f64],
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    density: &[[super::BlochMatrix; 4]],
    angular_order: usize,
) -> Result<SpinorXcGridResponse> {
    let points = prepare_grid_points(functional, weights, orbitals, momentum, density)?;
    let values = points.evaluate(functional, angular_order)?;
    integrate_spinor_grid(functional, weights, orbitals, momentum, values)
}

enum GridPoints {
    Collinear(Vec<SpinorXcPointResult>),
    Projected {
        localities: Vec<SpinorXcPoint>,
        exact: Vec<Option<SpinorXcPointResult>>,
    },
}

impl GridPoints {
    fn count_needing_angular_rule(&self) -> usize {
        match self {
            Self::Collinear(_) => 0,
            Self::Projected { exact, .. } => exact.iter().filter(|p| p.is_none()).count(),
        }
    }

    fn evaluate(
        &self,
        functional: &ResolvedFunctional,
        angular_order: usize,
    ) -> Result<Vec<SpinorXcPointResult>> {
        match self {
            Self::Collinear(points) => Ok(points.clone()),
            Self::Projected { localities, exact } => {
                evaluate_angular_points(functional, localities, exact, angular_order)
            }
        }
    }
}

fn prepare_grid_points(
    functional: &ResolvedFunctional,
    weights: &[f64],
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    density: &[[super::BlochMatrix; 4]],
) -> Result<GridPoints> {
    let nk = density.len();
    if nk == 0
        || orbitals.len() != nk
        || momentum.len() != nk
        || weights.iter().any(|w| !w.is_finite())
    {
        return Err(DftError::InvalidInput(
            "spinor XC grid requires matching density/orbital/momentum sectors and finite weights"
                .into(),
        ));
    }
    let n = density[0][0].rows();
    let ng = weights.len();
    if orbitals.iter().any(|p| p.rows() != ng || p.cols() != n)
        || momentum
            .iter()
            .any(|p| p.iter().any(|d| d.rows() != ng || d.cols() != n))
        || density.iter().any(|p| {
            p.iter()
                .any(|d| d.rows() != n || d.cols() != n || d.hermiticity_error() > 1e-10)
        })
    {
        return Err(DftError::InvalidInput(
            "spinor XC grid matrices have inconsistent dimensions or non-Hermitian density".into(),
        ));
    }
    let axis = common_ao_spin_axis(density);
    if functional.family == Family::MetaGga {
        if let Some(axis) = axis {
            if let Some(points) =
                collinear_grid::amplitude_points(orbitals, momentum, density, axis)?
            {
                let points: Vec<_> = points.into_iter().map(|point| (point, axis)).collect();
                return Ok(GridPoints::Collinear(collinear_batch(functional, &points)?));
            }
        }
    }
    let localities = spinor_grid_descriptors(orbitals, momentum, density);
    let exact = exact_collinear_points(functional, &localities, axis)?;
    Ok(GridPoints::Projected { localities, exact })
}

fn spinor_grid_descriptors(
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    density: &[[super::BlochMatrix; 4]],
) -> Vec<SpinorXcPoint> {
    use super::BlochMatrix;
    use crate::response::complex::Complex;
    use rayon::prelude::*;
    let nk = density.len();
    let n = density[0][0].rows();
    let z = |m: &BlochMatrix, i: usize, j: usize| Complex::new(m.re[(i, j)], m.im[(i, j)]);
    (0..orbitals[0].rows())
        .into_par_iter()
        .map(|g| {
            let mut local = SpinorXcPoint {
                density: [0.0; 4],
                gradient: [[0.0; 3]; 4],
                tau: [0.0; 4],
                current: [[0.0; 3]; 4],
            };
            for k in 0..nk {
                for a in 0..n {
                    for b in 0..n {
                        let phi = z(orbitals[k], g, a);
                        let conjugate = z(orbitals[k], g, b).conj();
                        for component in 0..4 {
                            let coefficient = z(&density[k][component], a, b) / (nk as f64);
                            local.density[component] += (coefficient * phi * conjugate).re;
                            for d in 0..3 {
                                let ma = z(&momentum[k][d], g, a);
                                let mb = z(&momentum[k][d], g, b);
                                let velocity = coefficient * ma * conjugate;
                                local.gradient[component][d] -= 2.0 * velocity.im;
                                local.current[component][d] += velocity.re;
                                local.tau[component] += 0.5 * (coefficient * ma * mb.conj()).re;
                            }
                        }
                    }
                }
            }
            local
        })
        .collect()
}

fn integrate_spinor_grid(
    functional: &ResolvedFunctional,
    weights: &[f64],
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    values: Vec<SpinorXcPointResult>,
) -> Result<SpinorXcGridResponse> {
    let energy = values.iter().zip(weights).map(|(v, w)| v.energy * w).sum();
    let parts = (0..functional.parts.len())
        .map(|a| {
            values
                .iter()
                .zip(weights)
                .map(|(v, w)| v.component_energies[a] * w)
                .sum()
        })
        .collect();
    let potential =
        integrate_spinor_grid_gradient(weights, orbitals, momentum, |g| values[g].gradient)?;
    Ok(SpinorXcGridResponse {
        energy,
        potential,
        component_energies: parts,
        point_response: values,
    })
}

/// Apply the adjoint of the grid descriptor map to either the XC potential
/// or its first variation, in the same complex AO basis as the SCF.
fn integrate_spinor_grid_gradient(
    weights: &[f64],
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    gradient: impl Fn(usize) -> [f64; 32] + Sync,
) -> Result<Vec<super::BlochMatrix>> {
    use super::spinor_scf::from_pauli;
    use super::BlochMatrix;
    use crate::response::complex::Complex;
    use rayon::prelude::*;
    let n = orbitals[0].cols();
    let z = |m: &BlochMatrix, i: usize, j: usize| Complex::new(m.re[(i, j)], m.im[(i, j)]);
    (0..orbitals.len())
        .into_par_iter()
        .map(|k| {
            let mut pauli = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
            let imaginary = Complex::new(0.0, 1.0);
            for (g, &weight) in weights.iter().enumerate() {
                let v = gradient(g);
                for a in 0..n {
                    for b in 0..n {
                        let left = z(orbitals[k], g, a).conj();
                        let right = z(orbitals[k], g, b);
                        for c in 0..4 {
                            let mut value = left * right * v[c];
                            for d in 0..3 {
                                let ml = z(&momentum[k][d], g, a).conj();
                                let mr = z(&momentum[k][d], g, b);
                                value += imaginary * (left * mr - ml * right) * v[4 + 3 * c + d];
                                value += ml * mr * (0.5 * v[16 + c]);
                                value += (left * mr + ml * right) * (0.5 * v[20 + 3 * c + d]);
                            }
                            let old = z(&pauli[c], a, b);
                            let sum = old + value * (2.0 * weight);
                            pauli[c].re[(a, b)] = sum.re;
                            pauli[c].im[(a, b)] = sum.im;
                        }
                    }
                }
            }
            from_pauli(&pauli)
        })
        .collect()
}

/// Evaluate common spin descriptors assembled by the caller's own basis.
pub(crate) fn evaluate_points(
    functional: &ResolvedFunctional,
    localities: &[SpinorXcPoint],
    angular_order: usize,
) -> Result<Vec<SpinorXcPointResult>> {
    let exact = exact_collinear_points(functional, localities, None)?;
    evaluate_angular_points(functional, localities, &exact, angular_order)
}

fn exact_collinear_points(
    functional: &ResolvedFunctional,
    localities: &[SpinorXcPoint],
    common_axis: Option<[f64; 3]>,
) -> Result<Vec<Option<SpinorXcPointResult>>> {
    let ng = localities.len();
    let axes: Vec<_> = localities
        .iter()
        .map(|point| {
            common_axis
                .filter(|_| {
                    density_is_psd_to_roundoff(point.density)
                        && point.descriptors().iter().all(|x| x.is_finite())
                })
                .or_else(|| collinear_axis(point, functional.family))
        })
        .collect();
    let indices: Vec<_> = (0..ng).filter(|g| axes[*g].is_some()).collect();
    let points = indices
        .iter()
        .map(|&g| {
            let axis = axes[g].expect("selected only collinear points");
            Ok((
                collinear_point(&localities[g], axis, functional.family)?,
                axis,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let collinear = collinear_batch(functional, &points)?;
    let mut exact: Vec<Option<SpinorXcPointResult>> = (0..ng).map(|_| None).collect();
    for (g, value) in indices.into_iter().zip(collinear) {
        exact[g] = Some(value);
    }
    Ok(exact)
}

fn evaluate_angular_points(
    functional: &ResolvedFunctional,
    localities: &[SpinorXcPoint],
    exact: &[Option<SpinorXcPointResult>],
    angular_order: usize,
) -> Result<Vec<SpinorXcPointResult>> {
    use rayon::prelude::*;
    exact
        .par_iter()
        .zip(localities)
        .map(|(value, local)| match value {
            Some(value) => Ok(value.clone()),
            None => spinor_semilocal_point(functional, local, angular_order),
        })
        .collect()
}

// Relative arithmetic allowance for a PSD boundary, with no density floor.
// A normalized rank-one spinor can acquire a negative eigenvalue of roundoff
// size when its Pauli components and Euclidean norm are evaluated separately.
fn density_is_psd_to_roundoff(density: [f64; 4]) -> bool {
    let rho = density[0];
    let magnitude = density[1].hypot(density[2]).hypot(density[3]);
    rho.is_finite()
        && rho >= 0.0
        && magnitude.is_finite()
        && (magnitude <= rho || magnitude - rho <= 64.0 * f64::EPSILON * magnitude.max(rho))
}

fn spin_norm(v: [f64; 3]) -> f64 {
    v[0].hypot(v[1]).hypot(v[2])
}

fn transverse_spin_norm(axis: [f64; 3], vector: [f64; 3]) -> f64 {
    let along = (0..3).map(|s| axis[s] * vector[s]).sum::<f64>();
    let residual = std::array::from_fn(|s| vector[s] - along * axis[s]);
    spin_norm(residual)
}

fn spin_aligned_to_roundoff(axis: [f64; 3], vector: [f64; 3]) -> bool {
    transverse_spin_norm(axis, vector) <= 64.0 * f64::EPSILON * spin_norm(vector)
}

/// Every grid spin descriptor is a linear contraction of the same Pauli
/// AO vectors. Establish a common axis before cancellations in those
/// contractions amplify their relative roundoff near stationary points.
fn common_ao_spin_axis(density: &[[super::BlochMatrix; 4]]) -> Option<[f64; 3]> {
    let entries = || {
        density.iter().flat_map(|block| {
            (0..block[0].re.as_slice().len()).map(move |i| {
                let re = std::array::from_fn(|s| block[s + 1].re.as_slice()[i]);
                let im = std::array::from_fn(|s| block[s + 1].im.as_slice()[i]);
                let charge = block[0].re.as_slice()[i].hypot(block[0].im.as_slice()[i]);
                (re, im, charge)
            })
        })
    };
    let largest = entries()
        .flat_map(|(re, im, _)| [re, im])
        .max_by(|&a, &b| spin_norm(a).total_cmp(&spin_norm(b)))?;
    let length = spin_norm(largest);
    if !length.is_finite() {
        return None;
    }
    let axis = if length == 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        largest.map(|x| x / length)
    };
    entries()
        .all(|(re, im, charge)| {
            // A real seed rotated by a complex eigensolver acquires imaginary
            // noise on the scale of the entire complex entry. Use a single
            // arithmetic bound for the two parts of that entry.
            let scale = spin_norm(re).hypot(spin_norm(im)).max(charge);
            let transverse = transverse_spin_norm(axis, re).hypot(transverse_spin_norm(axis, im));
            transverse <= 64.0 * f64::EPSILON * scale
        })
        .then_some(axis)
}

// The global projection has an exact collinear limit. Resolve that limit
// algebraically before angular integration, including the unpolarized point.
// Only floating-point collinearity residuals are admitted, independently of
// the requested physical convergence tolerance.
fn collinear_axis(point: &SpinorXcPoint, family: Family) -> Option<[f64; 3]> {
    let x = point.descriptors();
    if x.iter().any(|value| !value.is_finite()) || !density_is_psd_to_roundoff(point.density) {
        return None;
    }
    let starts = [
        (0, 1),
        (4, 3),
        (5, 3),
        (6, 3),
        (16, 1),
        (20, 3),
        (21, 3),
        (22, 3),
    ];
    let active = match family {
        Family::Lda => 1,
        Family::Gga => 4,
        Family::MetaGga => 8,
    };
    let vectors: Vec<[f64; 3]> = starts
        .iter()
        .take(active)
        .map(|&(start, stride)| std::array::from_fn(|s| x[start + (s + 1) * stride]))
        .collect();
    let largest = vectors
        .iter()
        .max_by(|&&a, &&b| spin_norm(a).total_cmp(&spin_norm(b)))?;
    let length = spin_norm(*largest);
    if length == 0.0 {
        return Some([0.0, 0.0, 1.0]);
    }
    let axis = largest.map(|v| v / length);
    vectors
        .into_iter()
        .all(|vector| spin_aligned_to_roundoff(axis, vector))
        .then_some(axis)
}

struct CollinearPoint {
    invariants: [f64; 7],
    gradients: [[f64; 3]; 2],
    currents: [[f64; 3]; 2],
}

fn collinear_point(
    point: &SpinorXcPoint,
    axis: [f64; 3],
    family: Family,
) -> Result<CollinearPoint> {
    let values = point.descriptors();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "collinear XC descriptors must be finite".into(),
        ));
    }
    let mut invariants = project(&values, axis, family, true)?;
    for spin in 0..2 {
        // A negative roundoff eigenvalue can be set to zero without dropping
        // a resolved positive minority contribution.
        if invariants[spin] < 0.0 {
            if -invariants[spin] > 64.0 * f64::EPSILON * values[0] {
                return Err(DftError::InvalidInput(
                    "collinear XC projection requires nonnegative spin densities".into(),
                ));
            }
            invariants[spin] = 0.0;
            invariants[1 - spin] = values[0];
        }
    }
    let component = |start: usize, spin: usize| {
        let aligned = (0..3)
            .map(|a| axis[a] * values[start + (a + 1) * 3])
            .sum::<f64>();
        0.5 * (values[start] + if spin == 0 { aligned } else { -aligned })
    };
    Ok(CollinearPoint {
        invariants,
        gradients: std::array::from_fn(|s| std::array::from_fn(|d| component(4 + d, s))),
        currents: std::array::from_fn(|s| std::array::from_fn(|d| component(20 + d, s))),
    })
}

fn collinear_batch(
    functional: &ResolvedFunctional,
    points: &[(CollinearPoint, [f64; 3])],
) -> Result<Vec<SpinorXcPointResult>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }
    let data: [Vec<f64>; 7] =
        std::array::from_fn(|a| points.iter().map(|(p, _)| p.invariants[a]).collect());
    let vars = GridVars {
        rho: [&data[0], &data[1]],
        sigma: (functional.family != Family::Lda).then_some([&data[2], &data[3], &data[4]]),
        tau: (functional.family == Family::MetaGga).then_some([&data[5], &data[6]]),
    };
    let mut out = XcOut::default();
    functional.eval(&vars, 1, &mut out)?;
    let parts = functional
        .parts
        .iter()
        .map(|(coefficient, part)| {
            let mut value = XcOut::default();
            value.prepare(points.len(), 0, functional.family);
            part.eval_add(&vars, 0, *coefficient, &mut value)?;
            Ok(value.e)
        })
        .collect::<Result<Vec<_>>>()?;
    (0..points.len())
        .map(|g| {
            let (point, axis) = &points[g];
            let q = point.invariants;
            let mut v = [0.0; 7];
            for a in 0..2 {
                v[a] = out.vrho[a][g];
            }
            if functional.family != Family::Lda {
                for a in 0..3 {
                    v[a + 2] = out.vsigma[a][g];
                }
            }
            if functional.family == Family::MetaGga {
                for a in 0..2 {
                    v[a + 5] = out.vtau[a][g];
                }
            }
            // Exact analytic chain rule in the collinear limit. Native first
            // derivatives suffice; no Hessian or angular endpoint kernel is needed.
            let gradients = point.gradients;
            let currents = point.currents;
            let mut vrho = [v[0], v[1]];
            let mut vcurrent = [[0.0; 3]; 2];
            if functional.family == Family::MetaGga {
                for s in 0..2 {
                    if q[s] > 0.0 {
                        let velocity = currents[s].map(|j| j / q[s]);
                        vrho[s] += 0.5 * v[s + 5] * velocity.iter().map(|a| a * a).sum::<f64>();
                        for d in 0..3 {
                            vcurrent[s][d] = -v[s + 5] * velocity[d];
                        }
                    }
                }
            }
            let mut gradient = [0.0; 32];
            let mut add = |start: usize, stride: usize, channel: [f64; 2]| {
                gradient[start] += 0.5 * (channel[0] + channel[1]);
                for a in 0..3 {
                    gradient[start + (a + 1) * stride] += 0.5 * axis[a] * (channel[0] - channel[1]);
                }
            };
            add(0, 1, vrho);
            if functional.family != Family::Lda {
                for d in 0..3 {
                    add(
                        4 + d,
                        3,
                        [
                            2.0 * v[2] * gradients[0][d] + v[3] * gradients[1][d],
                            2.0 * v[4] * gradients[1][d] + v[3] * gradients[0][d],
                        ],
                    );
                }
            }
            if functional.family == Family::MetaGga {
                add(16, 1, [v[5], v[6]]);
                for d in 0..3 {
                    add(20 + d, 3, [vcurrent[0][d], vcurrent[1][d]]);
                }
            }

            let components: Vec<_> = parts.iter().map(|part| part[g]).collect();
            if !out.e[g].is_finite() || gradient.iter().chain(&components).any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput(
                    "non-finite exact collinear XC energy or potential".into(),
                ));
            }
            Ok(SpinorXcPointResult {
                energy: out.e[g],
                gradient,
                component_energies: components,
                angular_order: 0,
                directions: 1,
            })
        })
        .collect()
}

/// Actual angular convergence measurements for a semilocal spinor operator.
#[derive(Clone, Debug)]
pub struct SpinorXcDiagnostics {
    pub angular_order: usize,
    pub angular_energy_defect: f64,
    pub angular_operator_defect: f64,
    pub angular_refinement_history: Vec<[f64; 3]>,
    pub hermiticity_defect: f64,
    pub component_energy_defect: f64,
    pub converged: bool,
}

/// Refine the actual integrated energy and Fock, retaining every failed rule.
/// Algebraically collinear grid points use the exact native limit on all rules.
pub fn spinor_semilocal_grid_converged(
    functional: &ResolvedFunctional,
    weights: &[f64],
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    density: &[[super::BlochMatrix; 4]],
    tolerance: f64,
) -> Result<(f64, Vec<super::BlochMatrix>, Vec<f64>, SpinorXcDiagnostics)> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(DftError::InvalidInput(
            "spinor XC convergence tolerance must be positive and finite".into(),
        ));
    }
    // The Pauli descriptors and native collinear XC are independent of the
    // angular rule. Reuse them throughout refinement of this same density.
    let points = prepare_grid_points(functional, weights, orbitals, momentum, density)?;
    let angular_count = points.count_needing_angular_rule();
    crate::logger::debug(&format!(
        "Spinor XC grid: {} points, {} exact collinear, {angular_count} angular",
        weights.len(),
        weights.len() - angular_count,
    ));
    let evaluate = |order| {
        let values = points.evaluate(functional, order)?;
        let value = integrate_spinor_grid(functional, weights, orbitals, momentum, values)?;
        Ok::<_, DftError>((value.energy, value.potential, value.component_energies))
    };
    let mut previous = evaluate(4)?;
    let mut history = Vec::new();
    for order in [8, 16, 32, 64, MAX_SPIN_ANGULAR_ORDER] {
        // When every point took the exact collinear path, any angular rule
        // yields the same Fock and energy. Still check their other defects.
        let next = if angular_count == 0 {
            None
        } else {
            Some(evaluate(order)?)
        };
        let current = next.as_ref().unwrap_or(&previous);
        let energy_defect = (current.0 - previous.0).abs();
        let operator_defect = current
            .1
            .iter()
            .zip(&previous.1)
            .map(|(a, b)| a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im)))
            .fold(0.0_f64, f64::max);
        let hermiticity_defect = current
            .1
            .iter()
            .map(|m| m.hermiticity_error())
            .fold(0.0_f64, f64::max);
        let component_energy_defect = (current.0 - current.2.iter().sum::<f64>()).abs();
        history.push([order as f64, energy_defect, operator_defect]);
        let converged = [
            energy_defect,
            operator_defect,
            hermiticity_defect,
            component_energy_defect,
        ]
        .iter()
        .all(|v| *v <= tolerance);
        if converged || order == MAX_SPIN_ANGULAR_ORDER {
            let diagnostics = SpinorXcDiagnostics {
                angular_order: order,
                angular_energy_defect: energy_defect,
                angular_operator_defect: operator_defect,
                angular_refinement_history: history,
                hermiticity_defect,
                component_energy_defect,
                converged,
            };
            let (energy, potential, parts) = next.unwrap_or(previous);
            return Ok((energy, potential, parts, diagnostics));
        }
        if let Some(next) = next {
            previous = next;
        }
    }
    unreachable!()
}

/// The same grid evaluation used by the Fock builder, with its point response.
/// Point gradients refer to the 32 Pauli descriptors before quadrature weights.
pub struct SpinorXcGridResponse {
    pub energy: f64,
    pub potential: Vec<super::BlochMatrix>,
    pub component_energies: Vec<f64>,
    pub point_response: Vec<SpinorXcPointResult>,
}

/// A geometry tangent at fixed AO density and fixed reduced Bloch sectors.
/// Momentum is covariant momentum, so its tangent includes any moving gauge.
pub struct SpinorXcGridTangent<'a> {
    pub weights: &'a [f64],
    pub orbitals: &'a [&'a super::BlochMatrix],
    pub momentum: &'a [&'a [super::BlochMatrix; 3]],
}

/// Analytic first variation of the actual grid model, including moving weights.
/// This uses the same exact-collinear/angular classification as the SCF Fock.
/// It is not a density response solve or a complete nuclear force by itself.
/// Immutable geometry/Pauli-density frame; no cross-reference cache reuse.
pub struct SpinorXcGridContext<'a> {
    weights: &'a [f64],
    orbitals: &'a [&'a super::BlochMatrix],
    momentum: &'a [&'a [super::BlochMatrix; 3]],
    density: &'a [[super::BlochMatrix; 4]],
    value: SpinorXcGridResponse,
}
impl<'a> SpinorXcGridContext<'a> {
    pub fn new(
        functional: &ResolvedFunctional,
        weights: &'a [f64],
        orbitals: &'a [&'a super::BlochMatrix],
        momentum: &'a [&'a [super::BlochMatrix; 3]],
        density: &'a [[super::BlochMatrix; 4]],
        angular_order: usize,
    ) -> Result<Self> {
        let value = spinor_semilocal_grid_response(
            functional,
            weights,
            orbitals,
            momentum,
            density,
            angular_order,
        )?;
        Ok(Self {
            weights,
            orbitals,
            momentum,
            density,
            value,
        })
    }
    pub fn energy(&self) -> f64 {
        self.value.energy
    }
    pub fn differential(&self, tangent: SpinorXcGridTangent<'_>) -> Result<f64> {
        grid_differential_from_response(
            &self.value,
            self.weights,
            self.orbitals,
            self.momentum,
            self.density,
            tangent,
        )
    }
}
pub fn spinor_semilocal_grid_differential(
    functional: &ResolvedFunctional,
    weights: &[f64],
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    density: &[[super::BlochMatrix; 4]],
    angular_order: usize,
    tangent: SpinorXcGridTangent<'_>,
) -> Result<(f64, f64)> {
    let context = SpinorXcGridContext::new(
        functional,
        weights,
        orbitals,
        momentum,
        density,
        angular_order,
    )?;
    Ok((context.energy(), context.differential(tangent)?))
}
fn grid_differential_from_response(
    value: &SpinorXcGridResponse,
    weights: &[f64],
    orbitals: &[&super::BlochMatrix],
    momentum: &[&[super::BlochMatrix; 3]],
    density: &[[super::BlochMatrix; 4]],
    tangent: SpinorXcGridTangent<'_>,
) -> Result<f64> {
    use crate::response::complex::Complex;
    let nk = density.len();
    let ng = weights.len();
    let n = density.first().map_or(0, |p| p[0].rows());
    let valid = |a: &super::BlochMatrix| {
        a.rows() == ng
            && a.cols() == n
            && a.re
                .as_slice()
                .iter()
                .chain(a.im.as_slice())
                .all(|v| v.is_finite())
    };
    if nk == 0
        || tangent.weights.len() != ng
        || tangent.weights.iter().any(|v| !v.is_finite())
        || tangent.orbitals.len() != nk
        || tangent.momentum.len() != nk
        || tangent.orbitals.iter().any(|p| !valid(p))
        || tangent.momentum.iter().any(|p| p.iter().any(|a| !valid(a)))
    {
        return Err(DftError::InvalidInput(
            "spinor XC geometry tangent requires finite, matching weight/AO/momentum arrays".into(),
        ));
    }
    let z = |m: &super::BlochMatrix, i: usize, j: usize| Complex::new(m.re[(i, j)], m.im[(i, j)]);
    let mut derivative = 0.0;
    for g in 0..ng {
        let mut response = [0.0; 32];
        for k in 0..nk {
            for a in 0..n {
                for b in 0..n {
                    let phi = z(orbitals[k], g, a);
                    let conjugate = z(orbitals[k], g, b).conj();
                    let dphi = z(tangent.orbitals[k], g, a);
                    let dconjugate = z(tangent.orbitals[k], g, b).conj();
                    for c in 0..4 {
                        let coefficient = z(&density[k][c], a, b) / nk as f64;
                        response[c] += (coefficient * (dphi * conjugate + phi * dconjugate)).re;
                        for d in 0..3 {
                            let ma = z(&momentum[k][d], g, a);
                            let mb = z(&momentum[k][d], g, b);
                            let dma = z(&tangent.momentum[k][d], g, a);
                            let dmb = z(&tangent.momentum[k][d], g, b);
                            let velocity = coefficient * (dma * conjugate + ma * dconjugate);
                            response[4 + 3 * c + d] -= 2.0 * velocity.im;
                            response[20 + 3 * c + d] += velocity.re;
                            response[16 + c] +=
                                0.5 * (coefficient * (dma * mb.conj() + ma * dmb.conj())).re;
                        }
                    }
                }
            }
        }
        derivative += tangent.weights[g] * value.point_response[g].energy
            + weights[g]
                * value.point_response[g]
                    .gradient
                    .iter()
                    .zip(response)
                    .map(|(v, r)| v * r)
                    .sum::<f64>();
    }
    if !derivative.is_finite() {
        return Err(DftError::InvalidInput(
            "nonfinite spinor XC geometry differential".into(),
        ));
    }
    Ok(derivative)
}

#[cfg(test)]
pub(crate) use crate::acceptance::components;

#[cfg(test)]
pub(crate) use crate::acceptance::stencils;

#[cfg(test)]
#[path = "spinor_xc/geometry_differential_tests.rs"]
mod geometry_differential_tests;
