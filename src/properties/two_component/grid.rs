// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The density and its adjoint Fock use the same balanced spinor basis.

use crate::basis::MolBasis;
use crate::dual::{DualN, Scalar};
use crate::grid::eval::{eval_aos_full, second_index, AoBatch};
use crate::grid::MolecularGrid;
use crate::math::Vec3;
use crate::nlc::{vv10_density_fields, Vv10Params};
use crate::periodic::spinor_xc::{
    evaluate_points, spinor_semilocal_point_directional_taylor, spinor_semilocal_point_kernel,
    SpinorXcDiagnostics, SpinorXcPoint, MAX_SPIN_ANGULAR_ORDER,
};
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use crate::{DftError, Result};
use rayon::prelude::*;

mod field;
pub(super) use field::FieldXcCache;

const I: Complex = Complex::new(0.0, 1.0);
const ONE: Complex = Complex::new(1.0, 0.0);
const ZERO: Complex = Complex::ZERO;
const PAULI: [[[Complex; 2]; 2]; 4] = [
    [[ONE, ZERO], [ZERO, ONE]],
    [[ZERO, ONE], [ONE, ZERO]],
    [[ZERO, Complex::new(0.0, -1.0)], [I, ZERO]],
    [[ONE, ZERO], [ZERO, Complex::new(-1.0, 0.0)]],
];

fn z(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}

/// Cartesian factors in the balanced basis, retaining every spin component.
/// The scalar representation separates the three small-component Cartesian
/// channels. Their sum reproduces half the Pauli trace of the charge operator.
pub(super) struct BalancedGrid<'a> {
    pub grid: &'a MolecularGrid,
    ao: AoBatch,
    inverse_2c: f64,
    spin_orbit: bool,
    magnetic: Option<MagneticBasis>,
}

struct MagneticBasis {
    centers: Vec<Vec3>,
    half_field: Vec3,
    vector_derivatives: [[f64; 3]; 3],
}

type SpinorValues = [[Complex; 2]; 4];
type BasisRows = [Vec<SpinorValues>; 4];

impl<'a> BalancedGrid<'a> {
    pub(super) fn new(
        basis: &MolBasis,
        grid: &'a MolecularGrid,
        speed: f64,
        spin_orbit: bool,
        family: Family,
    ) -> Self {
        Self {
            grid,
            ao: eval_aos_full(basis, &grid.points, true, family != Family::Lda),
            inverse_2c: 0.5 / speed,
            spin_orbit,
            magnetic: None,
        }
    }

    pub(super) fn with_uniform_magnetic_basis(mut self, basis: &MolBasis, field: [f64; 3]) -> Self {
        let [x, y, z] = field;
        let half_field = Vec3::new(x, y, z) * 0.5;
        let mut centers = vec![Vec3::zero(); self.ao.phi.cols];
        for shell in &basis.shells {
            let end = shell.sph_off + crate::basis::n_sph(shell.l);
            centers[shell.sph_off..end].fill(shell.center);
        }
        let vector_derivatives = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
        .map(|e| half_field.cross(e).to_array());
        self.magnetic = Some(MagneticBasis {
            centers,
            half_field,
            vector_derivatives,
        });
        self
    }

    fn factors(&self, g: usize, count: usize) -> BasisRows {
        let n = self.ao.phi.cols;
        let gradients = self.ao.grad.as_ref().expect("balanced grid gradients");
        let mut out = std::array::from_fn(|_| vec![[[ZERO; 2]; 4]; 2 * n]);
        for i in 0..n {
            let value = self.ao.phi[(g, i)];
            let g0 = self.grid.points[g];
            let (phase, vector, derivative) = if let Some(model) = &self.magnetic {
                let center = model.centers[i];
                let angle = -model.half_field.cross(center).dot(g0);
                let (im, re) = angle.sin_cos();
                (
                    Complex::new(re, im),
                    model.half_field.cross(g0 - center).to_array(),
                    model.vector_derivatives,
                )
            } else {
                (ONE, [0.0; 3], [[0.0; 3]; 3])
            };
            let gradient = std::array::from_fn::<_, 3, _>(|a| gradients[a][(g, i)]);
            let momentum = std::array::from_fn::<_, 3, _>(|a| {
                phase * Complex::new(vector[a] * value, -gradient[a])
            });
            for s in 0..2 {
                let index = s * n + i;
                out[0][index][0][s] = phase * value;
                for (d, p_d) in momentum.iter().enumerate() {
                    let coefficient = *p_d * self.inverse_2c;
                    if self.spin_orbit {
                        for p in 0..2 {
                            out[1][index][0][p] += PAULI[d + 1][p][s] * coefficient;
                        }
                    } else {
                        out[d + 1][index][0][s] = coefficient;
                    }
                }
                if count == 4 {
                    continue;
                }
                let hessian = self.ao.hess.as_ref().expect("balanced grid Hessians");
                for (a, p_a) in momentum.iter().enumerate() {
                    out[0][index][a + 1][s] = *p_a;
                    for d in 0..3 {
                        // Covariant derivative of the RMB small component.
                        // The local vector potential differentiates alongside
                        // the Gaussian derivative and the plane-wave phase.
                        let coefficient = phase
                            * Complex::new(
                                -hessian[second_index(a, d)][(g, i)]
                                    + vector[a] * vector[d] * value,
                                -derivative[a][d] * value
                                    - vector[d] * gradient[a]
                                    - vector[a] * gradient[d],
                            )
                            * self.inverse_2c;
                        if self.spin_orbit {
                            for p in 0..2 {
                                out[1][index][a + 1][p] += PAULI[d + 1][p][s] * coefficient;
                            }
                        } else {
                            out[d + 1][index][a + 1][s] = coefficient;
                        }
                    }
                }
            }
        }
        out
    }

    fn for_point(
        &self,
        g: usize,
        count: usize,
        mut action: impl FnMut(usize, usize, [Complex; 32]),
    ) {
        let rows = self.factors(g, count);
        let n = rows[0].len();
        for (channel, values) in rows.iter().enumerate() {
            if self.spin_orbit && channel > 1 {
                break;
            }
            let start = if channel == 0 { 0 } else { n };
            for i in 0..n {
                for j in 0..=i {
                    let mut vertex = [ZERO; 32];
                    for (c, sigma) in PAULI.iter().enumerate() {
                        let inner = |a: usize, b: usize| {
                            let [left_up, left_down] = values[i][a].map(Complex::conj);
                            let [right_up, right_down] = values[j][b];
                            left_up * (sigma[0][0] * right_up + sigma[0][1] * right_down)
                                + left_down * (sigma[1][0] * right_up + sigma[1][1] * right_down)
                        };
                        vertex[c] = inner(0, 0);
                        if count >= 16 {
                            for d in 0..3 {
                                let right = inner(0, d + 1);
                                let left = inner(d + 1, 0);
                                vertex[4 + 3 * c + d] = I * (right - left);
                                if count == 32 {
                                    vertex[16 + c] += inner(d + 1, d + 1) * 0.5;
                                    vertex[20 + 3 * c + d] = (right + left) * 0.5;
                                }
                            }
                        }
                    }
                    action(start + i, start + j, vertex);
                }
            }
        }
    }

    fn local(&self, density: &BlochMatrix, count: usize) -> Vec<SpinorXcPoint> {
        (0..self.grid.len())
            .into_par_iter()
            .map(|g| {
                let mut x = [0.0; 32];
                self.for_point(g, count, |i, j, vertex| {
                    let density = z(density, j, i);
                    let factor = if i == j { 1.0 } else { 2.0 };
                    for a in 0..count {
                        x[a] += (density * vertex[a]).re * factor;
                    }
                });
                SpinorXcPoint::from_descriptors(x)
            })
            .collect()
    }

    fn potential(&self, fields: &[[f64; 32]], count: usize) -> BlochMatrix {
        let n = 4 * self.ao.phi.cols;
        const BATCH: usize = 64;
        let mut matrix = (0..self.grid.len().div_ceil(BATCH))
            .into_par_iter()
            .map(|chunk| {
                let mut matrix = BlochMatrix::zeros(n, n);
                let start = chunk * BATCH;
                let end = (start + BATCH).min(self.grid.len());
                for (g, field) in fields.iter().enumerate().take(end).skip(start) {
                    let weight = self.grid.weights[g];
                    self.for_point(g, count, |i, j, vertex| {
                        let mut value = ZERO;
                        for a in 0..count {
                            value += vertex[a] * (field[a] * weight);
                        }
                        matrix.re[(i, j)] += value.re;
                        matrix.im[(i, j)] += value.im;
                    });
                }
                matrix
            })
            .reduce(
                || BlochMatrix::zeros(n, n),
                |mut left, right| {
                    left.add_scaled(&right, 1.0);
                    left
                },
            );
        for i in 0..n {
            for j in 0..i {
                matrix.re[(j, i)] = matrix.re[(i, j)];
                matrix.im[(j, i)] = -matrix.im[(i, j)];
            }
        }
        matrix
    }

    pub(super) fn semilocal(
        &self,
        functional: &ResolvedFunctional,
        density: &BlochMatrix,
        tolerance: f64,
    ) -> Result<(f64, BlochMatrix, Vec<f64>, SpinorXcDiagnostics)> {
        let count = active(functional.family);
        let points = self.local(density, count);
        // The spin projection retains spin currents in the meta-GGA even
        // when time reversal makes the local spin densities zero.
        let evaluate = |order| -> Result<_> {
            let data = evaluate_points(functional, &points, order)?;
            let energy = data
                .iter()
                .zip(&self.grid.weights)
                .map(|(v, w)| v.energy * w)
                .sum::<f64>();
            let parts: Vec<f64> = (0..functional.parts.len())
                .map(|a| {
                    data.iter()
                        .zip(&self.grid.weights)
                        .map(|(v, w)| v.component_energies[a] * w)
                        .sum()
                })
                .collect();
            let fields: Vec<_> = data.iter().map(|v| v.gradient).collect();
            Ok((energy, self.potential(&fields, count), parts))
        };
        let mut previous = evaluate(4)?;
        let mut history = Vec::new();
        for order in [8, 16, 32, 64, MAX_SPIN_ANGULAR_ORDER] {
            let (energy, potential, parts) = evaluate(order)?;
            let energy_defect = (energy - previous.0).abs();
            let operator_defect = potential
                .re
                .max_abs_diff(&previous.1.re)
                .max(potential.im.max_abs_diff(&previous.1.im));
            let component_defect = (energy - parts.iter().sum::<f64>()).abs();
            let hermiticity = potential.hermiticity_error();
            let converged = [
                energy_defect,
                operator_defect,
                component_defect,
                hermiticity,
            ]
            .iter()
            .all(|r| *r <= tolerance);
            history.push([order as f64, energy_defect, operator_defect]);
            if converged || order == MAX_SPIN_ANGULAR_ORDER {
                return Ok((
                    energy,
                    potential,
                    parts,
                    SpinorXcDiagnostics {
                        angular_order: order,
                        angular_energy_defect: energy_defect,
                        angular_operator_defect: operator_defect,
                        angular_refinement_history: history,
                        hermiticity_defect: hermiticity,
                        component_energy_defect: component_defect,
                        converged,
                    },
                ));
            }
            previous = (energy, potential, parts);
        }
        unreachable!()
    }

    pub(super) fn semilocal_kernel(
        &self,
        functional: &ResolvedFunctional,
        density: &BlochMatrix,
        direction: &BlochMatrix,
        angular_order: usize,
    ) -> Result<BlochMatrix> {
        let count = active(functional.family);
        let base = self.local(density, count);
        let tangent = self.local(direction, count);
        if !self.spin_orbit {
            return Ok(self.potential(&restricted_kernel(functional, &base, &tangent)?, count));
        }
        let derivatives = base
            .par_iter()
            .zip(&tangent)
            .map(|(point, tangent)| {
                let d = tangent.descriptors();
                let x = point.descriptors();
                if x[0] < crate::xc::RHO_CUTOFF {
                    return Ok([0.0; 32]);
                }
                if x[0] < x[1].hypot(x[2]).hypot(x[3]) {
                    return Err(DftError::Unsupported {
                        feature: "picture-changed spinor XC response at the PSD boundary".into(),
                        missing: "an analytic one-sided collinear density kernel".into(),
                    });
                }
                if let Some(kernel) =
                    spinor_semilocal_point_kernel(functional, point, angular_order)?
                {
                    return Ok(std::array::from_fn(|a| {
                        if a < count {
                            (0..count)
                                .map(|b| kernel.kernel[a * count + b] * d[b])
                                .sum()
                        } else {
                            0.0
                        }
                    }));
                }
                let descriptors = x.map(|x| vec![x]);
                let direction = d.map(|x| vec![x]);
                let result = spinor_semilocal_point_directional_taylor(
                    functional,
                    &descriptors,
                    &direction,
                    angular_order,
                )?;
                Ok(result.directional.gradient.map(|values| values[0]))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.potential(&derivatives, count))
    }

    pub(super) fn vv10(
        &self,
        density: &BlochMatrix,
        direction: Option<&BlochMatrix>,
        params: Vv10Params,
    ) -> Result<(f64, BlochMatrix)> {
        type Jet = DualN<f64, 1>;
        let base = self.local(density, 16);
        let tangent = direction.map(|d| self.local(d, 16));
        let value = |jet: Jet| {
            if direction.is_some() {
                jet.im[0]
            } else {
                jet.re
            }
        };
        let rho: Vec<_> = base
            .iter()
            .enumerate()
            .map(|(i, p)| Jet {
                re: p.density[0],
                im: [tangent.as_ref().map_or(0.0, |d| d[i].density[0])],
            })
            .collect();
        let gradient: Vec<[Jet; 3]> = base
            .iter()
            .enumerate()
            .map(|(i, p)| {
                std::array::from_fn(|a| Jet {
                    re: p.gradient[0][a],
                    im: [tangent.as_ref().map_or(0.0, |d| d[i].gradient[0][a])],
                })
            })
            .collect();
        let sigma: Vec<_> = gradient
            .iter()
            .map(|d| d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            .collect();
        let fields = vv10_density_fields(self.grid, &rho, &sigma, params)?;
        let derivatives: Vec<_> = gradient
            .iter()
            .enumerate()
            .map(|(g, gradient)| {
                let mut d = [0.0; 32];
                d[0] = value(fields.d_rho[g]);
                for a in 0..3 {
                    d[4 + a] = 2.0 * value(fields.d_sigma[g] * gradient[a]);
                }
                d
            })
            .collect();
        if derivatives.iter().flatten().any(|v| !v.is_finite()) {
            return Err(DftError::LinearAlgebra(
                "nonfinite picture-changed VV10 endpoint potential".into(),
            ));
        }
        Ok((value(fields.energy), self.potential(&derivatives, 16)))
    }
}

/// Differentiate the exact unpolarized semilocal limit. Both spin channels
/// have the same rho, gradient and current-corrected kinetic density.
fn restricted_kernel(
    functional: &ResolvedFunctional,
    base: &[SpinorXcPoint],
    tangent: &[SpinorXcPoint],
) -> Result<Vec<[f64; 32]>> {
    type Jet = DualN<f64, 1>;
    type Seed = DualN<Jet, 8>;
    let seeds: Vec<[Seed; 7]> = base
        .iter()
        .zip(tangent)
        .map(|(p, d)| {
            let input = [
                (p.density[0], d.density[0]),
                (p.gradient[0][0], d.gradient[0][0]),
                (p.gradient[0][1], d.gradient[0][1]),
                (p.gradient[0][2], d.gradient[0][2]),
                (p.tau[0], d.tau[0]),
                (p.current[0][0], d.current[0][0]),
                (p.current[0][1], d.current[0][1]),
                (p.current[0][2], d.current[0][2]),
            ];
            let [rho, gx, gy, gz, tau, jx, jy, jz] = std::array::from_fn(|a| {
                let (re, derivative) = input[a];
                Seed::variable(
                    Jet {
                        re,
                        im: [derivative],
                    },
                    a,
                )
            });
            let number = rho.scale(0.5);
            let sigma = (gx * gx + gy * gy + gz * gz).scale(0.25);
            let tau = if functional.family == Family::MetaGga && rho.re.re > 0.0 {
                tau.scale(0.5) - (jx * jx + jy * jy + jz * jz) / rho.scale(4.0)
            } else {
                tau.scale(0.5)
            };
            [number, number, sigma, sigma, sigma, tau, tau]
        })
        .collect();
    let values: [Vec<f64>; 7] = std::array::from_fn(|a| seeds.iter().map(|q| q[a].re.re).collect());
    let vars = GridVars {
        rho: [&values[0], &values[1]],
        sigma: (functional.family != Family::Lda).then_some([&values[2], &values[3], &values[4]]),
        tau: (functional.family == Family::MetaGga).then_some([&values[5], &values[6]]),
    };
    let mut xc = XcOut::default();
    functional.eval(&vars, 2, &mut xc)?;
    seeds
        .iter()
        .enumerate()
        .map(|(g, q)| {
            let mut d = [0.0; 32];
            let (potential, kernel) = xc.potential_and_kernel(g, functional.family)?;
            for (j, descriptor) in [0, 4, 5, 6, 16, 20, 21, 22].into_iter().enumerate() {
                d[descriptor] = (0..7)
                    .map(|a| {
                        potential[a] * q[a].im[j].im[0]
                            + q[a].im[j].re
                                * (0..7).map(|b| kernel[a][b] * q[b].re.im[0]).sum::<f64>()
                    })
                    .sum();
            }
            Ok(d)
        })
        .collect()
}

fn active(family: Family) -> usize {
    match family {
        Family::Lda => 4,
        Family::Gga => 16,
        Family::MetaGga => 32,
    }
}
