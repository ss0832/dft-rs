// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit uniform-field derivatives of balanced orbital grid vertices.

use super::{
    active, evaluate_points, second_index, spinor_semilocal_point_directional_taylor,
    spinor_semilocal_point_kernel, vv10_density_fields, z, BalancedGrid, BlochMatrix, Complex,
    ResolvedFunctional, Result, SpinorXcPoint, Vv10Params, I, PAULI, ZERO,
};
use crate::dual::Taylor;
use crate::math::Vec3;
use crate::periodic::spinor_xc::SpinorXcPointKernel;
use crate::resources::{checked_bytes, reserve_cache, MemoryReservation};
use crate::DftError;
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct PhaseJet([Complex; 3]);

impl PhaseJet {
    const ZERO: Self = Self([ZERO; 3]);

    fn multiply(self, other: Self) -> Self {
        Self(std::array::from_fn(|n| {
            (0..=n).fold(ZERO, |total, k| total + self.0[k] * other.0[n - k])
        }))
    }

    fn conj(self) -> Self {
        Self(self.0.map(Complex::conj))
    }

    fn scale(self, factor: Complex) -> Self {
        Self(self.0.map(|p| p * factor))
    }

    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|n| self.0[n] + other.0[n]))
    }
}

type FieldSpinor = [[PhaseJet; 2]; 4];
type Coefficients = [f64; 3];
type Descriptors = [Coefficients; 32];

struct XcPoint {
    source: SpinorXcPoint,
    energy: f64,
    gradient: [f64; 32],
    kernel: Option<SpinorXcPointKernel>,
}

pub(in crate::properties::two_component) struct FieldXcCache {
    points: Vec<XcPoint>,
    angular: usize,
    _memory: Option<MemoryReservation>,
}

impl BalancedGrid<'_> {
    fn field_factors(&self, g: usize, count: usize, direction: Vec3) -> [Vec<FieldSpinor>; 2] {
        let model = self
            .magnetic
            .as_ref()
            .expect("London magnetic reference AO centres");
        let nao = self.ao.phi.cols;
        let point = self.grid.points[g];
        let gradients = self
            .ao
            .grad
            .as_ref()
            .expect("balanced Gaussian derivatives");
        let basis = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let half = direction * 0.5;
        let da = basis.map(|e| half.cross(e).to_array());
        let mut result = std::array::from_fn(|_| vec![[[PhaseJet::ZERO; 2]; 4]; 2 * nao]);
        for i in 0..nao {
            let center = model.centers[i];
            let angle = -model.half_field.cross(center).dot(point);
            let slope = -half.cross(center).dot(point);
            let (sin, cos) = angle.sin_cos();
            let start = Complex::new(cos, sin);
            let phase = PhaseJet([start, I * start * slope, start * (-0.5 * slope * slope)]);
            let value = self.ao.phi[(g, i)];
            let grad = std::array::from_fn::<_, 3, _>(|a| gradients[a][(g, i)]);
            let a = [
                model.half_field.cross(point - center),
                half.cross(point - center),
            ]
            .map(Vec3::to_array);
            let momentum = std::array::from_fn::<_, 3, _>(|k| {
                phase.multiply(PhaseJet([
                    Complex::new(a[0][k] * value, -grad[k]),
                    Complex::from(a[1][k] * value),
                    ZERO,
                ]))
            });
            for spin in 0..2 {
                let column = spin * nao + i;
                result[0][column][0][spin] = phase.scale(Complex::from(value));
                for (k, pi) in momentum.iter().enumerate() {
                    for (s, row) in PAULI[k + 1].iter().enumerate() {
                        let small = pi.scale(row[spin] * self.inverse_2c);
                        result[1][column][0][s] = result[1][column][0][s].add(small);
                    }
                }
                if count == 4 {
                    continue;
                }
                let hessian = self.ao.hess.as_ref().expect("balanced Gaussian Hessians");
                for (j, pj) in momentum.iter().enumerate() {
                    result[0][column][j + 1][spin] = *pj;
                    for k in 0..3 {
                        let base = Complex::new(
                            -hessian[second_index(j, k)][(g, i)] + a[0][j] * a[0][k] * value,
                            -model.vector_derivatives[j][k] * value
                                - a[0][j] * grad[k]
                                - a[0][k] * grad[j],
                        );
                        let first = Complex::new(
                            (a[0][j] * a[1][k] + a[1][j] * a[0][k]) * value,
                            -da[j][k] * value - a[1][j] * grad[k] - a[1][k] * grad[j],
                        );
                        let second = Complex::from(a[1][j] * a[1][k] * value);
                        let mixed = phase.multiply(PhaseJet([base, first, second]));
                        for (s, row) in PAULI[k + 1].iter().enumerate() {
                            let small = mixed.scale(row[spin] * self.inverse_2c);
                            result[1][column][j + 1][s] = result[1][column][j + 1][s].add(small);
                        }
                    }
                }
            }
        }
        result
    }

    fn for_field_vertex(
        &self,
        g: usize,
        count: usize,
        direction: Vec3,
        mut action: impl FnMut(usize, usize, [PhaseJet; 32]),
    ) {
        let factors = self.field_factors(g, count, direction);
        for (block, factors) in factors.iter().enumerate() {
            let offset = block * factors.len();
            for (i, left) in factors.iter().enumerate() {
                let left = left.map(|p| p.map(PhaseJet::conj));
                for (j, right) in factors.iter().enumerate().take(i + 1) {
                    let mut v = [PhaseJet::ZERO; 32];
                    for (c, sigma) in PAULI.iter().enumerate() {
                        let inner = |a: usize, b: usize| {
                            let r = std::array::from_fn::<_, 2, _>(|s| {
                                right[b][0]
                                    .scale(sigma[s][0])
                                    .add(right[b][1].scale(sigma[s][1]))
                            });
                            left[a][0].multiply(r[0]).add(left[a][1].multiply(r[1]))
                        };
                        v[c] = inner(0, 0);
                        if count >= 16 {
                            for d in 0..3 {
                                let ket = inner(0, d + 1);
                                let bra = inner(d + 1, 0);
                                v[4 + 3 * c + d] = ket.add(bra.scale(Complex::from(-1.0))).scale(I);
                                if count == 32 {
                                    v[16 + c] = v[16 + c]
                                        .add(inner(d + 1, d + 1).scale(Complex::from(0.5)));
                                    v[20 + 3 * c + d] = ket.add(bra).scale(Complex::from(0.5));
                                }
                            }
                        }
                    }
                    action(offset + i, offset + j, v);
                }
            }
        }
    }

    fn field_descriptors(
        &self,
        g: usize,
        count: usize,
        direction: Vec3,
        density: &[BlochMatrix],
    ) -> Descriptors {
        let mut result = [[0.0; 3]; 32];
        self.for_field_vertex(g, count, direction, |i, j, data| {
            let p = PhaseJet(std::array::from_fn(|k| z(&density[k], j, i)));
            let factor = if i == j { 1.0 } else { 2.0 };
            for (dst, v) in result.iter_mut().zip(data).take(count) {
                for (out, part) in dst.iter_mut().zip(p.multiply(v).0) {
                    *out += factor * part.re;
                }
            }
        });
        result
    }

    fn field_operator(
        &self,
        count: usize,
        direction: Vec3,
        gradients: &[[[f64; 2]; 32]],
    ) -> [BlochMatrix; 2] {
        let n = 4 * self.ao.phi.cols;
        let mut output = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
        for (g, field) in gradients.iter().enumerate() {
            self.for_field_vertex(g, count, direction, |i, j, vertices| {
                let mut terms = [ZERO; 2];
                for (v, x) in vertices.into_iter().zip(field).take(count) {
                    terms[0] += v.0[0] * x[0];
                    terms[1] += v.0[1] * x[0] + v.0[0] * x[1];
                }
                let weight = self.grid.weights[g];
                for (mat, value) in output.iter_mut().zip(terms) {
                    mat.re[(i, j)] += value.re * weight;
                    mat.im[(i, j)] += value.im * weight;
                }
            });
        }
        for matrix in &mut output {
            for i in 0..n {
                for j in 0..i {
                    matrix.re[(j, i)] = matrix.re[(i, j)];
                    matrix.im[(j, i)] = -matrix.im[(i, j)];
                }
            }
        }
        output
    }

    pub(in crate::properties::two_component) fn prepare_field_xc(
        &self,
        functional: &ResolvedFunctional,
        density: &BlochMatrix,
        angular: usize,
        budget: usize,
    ) -> Result<FieldXcCache> {
        let count = active(functional.family);
        let points = self.local(density, count);
        let values = evaluate_points(functional, &points, angular)?;
        let operation = "balanced magnetic XC descriptor Hessians";
        let bytes = checked_bytes(
            &[points.len(), count, count, std::mem::size_of::<f64>()],
            operation,
        )?;
        let memory = if bytes <= budget.saturating_mul(1024 * 1024) {
            reserve_cache(bytes, operation)?
        } else {
            None
        };
        let points = points
            .into_par_iter()
            .zip(values)
            .map(|(point, result)| {
                let kernel = if memory.is_some() && point.density[0] >= crate::xc::RHO_CUTOFF {
                    spinor_semilocal_point_kernel(functional, &point, angular)?
                } else {
                    None
                };
                Ok(XcPoint {
                    source: point,
                    energy: result.energy,
                    gradient: result.gradient,
                    kernel,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(FieldXcCache {
            points,
            angular,
            _memory: memory,
        })
    }

    pub(in crate::properties::two_component) fn semilocal_field_path(
        &self,
        functional: &ResolvedFunctional,
        density: &[BlochMatrix],
        direction: Vec3,
        cache: &FieldXcCache,
    ) -> Result<(Coefficients, [BlochMatrix; 2])> {
        let count = active(functional.family);
        let values = cache
            .points
            .par_iter()
            .enumerate()
            .map(|(g, p)| {
                let x = self.field_descriptors(g, count, direction, density);
                let base = p.source.descriptors();
                if x.iter()
                    .zip(base)
                    .any(|(x, b)| (x[0] - b).abs() > 1e-10 * (1.0 + b.abs()))
                {
                    return Err(DftError::LinearAlgebra(
                        "magnetic XC descriptor path does not match the stationary grid".into(),
                    ));
                }
                let mut slope = [0.0; 32];
                if base[0] >= crate::xc::RHO_CUTOFF {
                    slope = if let Some(kernel) = &p.kernel {
                        std::array::from_fn(|a| {
                            if a < count {
                                (0..count)
                                    .map(|b| kernel.kernel[a * count + b] * x[b][1])
                                    .sum()
                            } else {
                                0.0
                            }
                        })
                    } else {
                        let descriptors = base.map(|value| vec![value]);
                        let tangent = x.map(|values| vec![values[1]]);
                        spinor_semilocal_point_directional_taylor(
                            functional,
                            &descriptors,
                            &tangent,
                            cache.angular,
                        )?
                        .directional
                        .gradient
                        .map(|values| values[0])
                    };
                }
                let energy = [
                    p.energy,
                    (0..count).map(|a| p.gradient[a] * x[a][1]).sum(),
                    (0..count)
                        .map(|a| p.gradient[a] * x[a][2] + 0.5 * slope[a] * x[a][1])
                        .sum(),
                ];
                let potential = std::array::from_fn(|a| [p.gradient[a], slope[a]]);
                Ok((energy, potential))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut energy = [0.; 3];
        let mut gradients = Vec::with_capacity(values.len());
        for ((local, fields), weight) in values.into_iter().zip(&self.grid.weights) {
            for (dst, v) in energy.iter_mut().zip(local) {
                *dst += weight * v;
            }
            gradients.push(fields);
        }
        Ok((energy, self.field_operator(count, direction, &gradients)))
    }

    pub(in crate::properties::two_component) fn vv10_field_path(
        &self,
        density: &[BlochMatrix],
        direction: Vec3,
        params: Vv10Params,
    ) -> Result<(Coefficients, [BlochMatrix; 2])> {
        let descriptors: Vec<_> = (0..self.grid.len())
            .into_par_iter()
            .map(|g| self.field_descriptors(g, 16, direction, density))
            .collect();
        let rho: Vec<_> = descriptors
            .iter()
            .map(|p| Taylor::<3> { c: p[0] })
            .collect();
        let gradient: Vec<_> = descriptors
            .into_iter()
            .map(|p| [p[4], p[5], p[6]].map(|c| Taylor::<3> { c }))
            .collect();
        let sigma: Vec<_> = gradient
            .iter()
            .map(|[x, y, z]| *x * *x + *y * *y + *z * *z)
            .collect();
        let fields = vv10_density_fields(self.grid, &rho, &sigma, params)?;
        let potential = gradient
            .iter()
            .enumerate()
            .map(|(i, &gradient)| {
                let mut out = [[0.; 2]; 32];
                out[0].copy_from_slice(&fields.d_rho[i].c[..2]);
                for (axis, derivative) in gradient.into_iter().enumerate() {
                    let value = fields.d_sigma[i] * derivative;
                    out[4 + axis] = [2.0 * value.c[0], 2.0 * value.c[1]];
                }
                out
            })
            .collect::<Vec<_>>();
        Ok((
            fields.energy.c,
            self.field_operator(16, direction, &potential),
        ))
    }
}
