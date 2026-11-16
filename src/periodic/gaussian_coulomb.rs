// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Dimensional Gaussian Coulomb derivatives and Hermite primitive integrals.
//! The same Green function and gauge serve nuclear attraction and Hartree.

use super::electrostatics::{DimensionalCoulomb, ElectrostaticComponents};
use super::ewald::EwaldParams;
use super::PeriodicGeometry;
use crate::basis::primitive_norm;
use crate::integrals::analytic::boys_values;
use crate::integrals::hermite::{ETable, RTensor};
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::{DftError, Result};
use std::f64::consts::PI;

#[cfg(test)]
mod spin_orbit_derivative_tests {
    use super::*;
    use crate::periodic::{features::PeriodicBoundary, Lattice};

    #[test]
    fn uniform_contact_has_four_step_independent_source_derivative_fits() {
        let gaussians = [
            CartesianGaussian {
                center: Vec3::new(0.13, -0.1, 0.2),
                exponent: 0.7,
                angular: [0, 0, 1],
            },
            CartesianGaussian {
                center: Vec3::new(0.25, 0.4, -0.15),
                exponent: 0.5,
                angular: [0; 3],
            },
        ];
        let source = GaussianNucleus {
            center: Vec3::new(0.5, 0.7, -0.2),
            charge: 1.0,
            width: 0.2,
        };
        let linear = [0.13, 0.21, -0.17];
        let axis = 0;
        let steps = [0.04_f64, 0.02, 0.01, 0.005];
        for axes in [[true, false, false], [true, true, false]] {
            let geometry = PeriodicGeometry::new(
                Lattice::new(
                    Vec3::new(5.0, 0.0, 0.0),
                    Vec3::new(0.3, 5.3, 0.0),
                    Vec3::zero(),
                ),
                PeriodicBoundary::new(axes),
            )
            .unwrap();
            let kernel = DimensionalCoulomb::new(
                geometry,
                EwaldParams {
                    alpha: 0.55,
                    rcut: 14.0,
                    gcut: 8.0,
                },
                192,
            )
            .unwrap();
            let tensor = prepare_integral_tensor(&kernel, &gaussians, Some(&source), 3).unwrap();
            let norm = gaussians
                .iter()
                .map(|g| g.normalization().unwrap())
                .product::<f64>();
            let analytic = -source.charge
                * norm
                * linear_gradient_integral(&tensor, &gaussians, axis, linear).unwrap();
            // Independent source displacement of normalized attraction integrals.
            // Each polynomial coordinate is reconstructed before differentiating.
            let weighted = |s: &GaussianNucleus| {
                let mut value = (0..3)
                    .map(|j| linear[j] * gaussians[0].center.get(j))
                    .sum::<f64>()
                    * evaluate_integral(&kernel, &gaussians, Some(s))
                        .unwrap()
                        .0
                        .iter()
                        .sum::<f64>();
                for j in 0..3 {
                    let mut raised = gaussians.clone();
                    raised[0].angular[j] += 1;
                    let raised_norm = raised
                        .iter()
                        .map(|g| g.normalization().unwrap())
                        .product::<f64>();
                    value += linear[j] * norm / raised_norm
                        * evaluate_integral(&kernel, &raised, Some(s))
                            .unwrap()
                            .0
                            .iter()
                            .sum::<f64>();
                }
                value
            };
            let mut errors = Vec::new();
            let mut estimates = Vec::new();
            for h in steps {
                let mut plus = source.clone();
                let mut minus = source.clone();
                plus.center.x += h;
                minus.center.x -= h;
                let estimate = -(weighted(&plus) - weighted(&minus)) / (2.0 * h);
                estimates.push(estimate);
                errors.push((estimate - analytic).abs());
            }
            assert!(errors.iter().all(|v| v.is_finite() && *v > 0.0));
            let xs = steps.map(f64::ln);
            let ys: Vec<_> = errors.iter().map(|v| v.ln()).collect();
            let mx = xs.iter().sum::<f64>() / 4.0;
            let my = ys.iter().sum::<f64>() / 4.0;
            let xx = xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
            let yy = ys.iter().map(|y| (y - my).powi(2)).sum::<f64>();
            let xy = xs
                .iter()
                .zip(&ys)
                .map(|(x, y)| (x - mx) * (y - my))
                .sum::<f64>();
            let p = xy / xx;
            let r2 = xy * xy / (xx * yy);
            let rich = (4.0 * estimates[3] - estimates[2]) / 3.0;
            println!(
                "uniform nuclear contact {axes:?}: steps={steps:?} errors={errors:?} \
                p={p:.9} R2={r2:.10} Richardson={:.3e}",
                (rich - analytic).abs()
            );
            assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
            assert!((rich - analytic).abs() < 1e-8);
        }
    }

    #[test]
    fn mixed_basis_derivatives_match_four_step_independent_green_integrals() {
        let gaussians = [
            CartesianGaussian {
                center: Vec3::new(0.13, -0.1, 0.2),
                exponent: 0.7,
                angular: [0; 3],
            },
            CartesianGaussian {
                center: Vec3::new(0.25, 0.4, -0.15),
                exponent: 0.5,
                angular: [0, 0, 1],
            },
        ];
        let source = GaussianNucleus {
            center: Vec3::new(0.5, 0.7, -0.2),
            charge: 1.0,
            width: 0.2,
        };
        let steps = [0.08_f64, 0.04, 0.02, 0.01];
        for axes in [[true, false, false], [true, true, false], [true; 3]] {
            let geometry = PeriodicGeometry::new(
                Lattice::new(
                    Vec3::new(5.0, 0.0, 0.0),
                    Vec3::new(0.3, 5.3, 0.0),
                    Vec3::new(0.2, 0.4, 5.8),
                ),
                PeriodicBoundary::new(axes),
            )
            .unwrap();
            let kernel = DimensionalCoulomb::new(
                geometry,
                EwaldParams {
                    alpha: 0.55,
                    rcut: 14.0,
                    gcut: 8.0,
                },
                192,
            )
            .unwrap();
            let tensor = prepare_integral_tensor(&kernel, &gaussians, Some(&source), 3).unwrap();
            let normalization = gaussians
                .iter()
                .map(|g| g.normalization().unwrap())
                .product::<f64>()
                * -source.charge;
            let analytic =
                mixed_basis_derivative(&tensor, &gaussians, 0, 1).unwrap() * normalization;
            let mut errors = Vec::new();
            let mut estimates = Vec::new();
            for h in steps {
                let mut estimate = 0.0;
                for sa in [1.0, -1.0] {
                    for sb in [1.0, -1.0] {
                        let mut moved = gaussians.clone();
                        moved[0].center.x += sa * h;
                        moved[1].center.y += sb * h;
                        let value = evaluate_integral(&kernel, &moved, Some(&source))
                            .unwrap()
                            .0
                            .iter()
                            .sum::<f64>();
                        estimate += sa * sb * value / (4.0 * h * h);
                    }
                }
                estimates.push(estimate);
                errors.push((estimate - analytic).abs());
            }
            assert!(errors.iter().all(|e| e.is_finite() && *e > 0.0));
            let xs = steps.map(f64::ln);
            let ys: Vec<_> = errors.iter().map(|e| e.ln()).collect();
            let mx = xs.iter().sum::<f64>() / 4.0;
            let my = ys.iter().sum::<f64>() / 4.0;
            let xx = xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
            let yy = ys.iter().map(|y| (y - my).powi(2)).sum::<f64>();
            let xy = xs
                .iter()
                .zip(&ys)
                .map(|(x, y)| (x - mx) * (y - my))
                .sum::<f64>();
            let p = xy / xx;
            let r2 = xy * xy / (xx * yy);
            let richardson = (4.0 * estimates[3] - estimates[2]) / 3.0;
            println!(
                "nuclear SOC mixed basis {axes:?}: steps={steps:?} errors={errors:?} \
                p={p:.9} R2={r2:.10} Richardson={:.3e}",
                (richardson - analytic).abs()
            );
            assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
            assert!((richardson - analytic).abs() < 1e-8);
        }
    }
}

/// Actual Cartesian derivatives, with no factorial normalization.
/// Entries outside total order are not part of the tensor.
pub struct CoulombDerivativeTensor {
    order: usize,
    dim: usize,
    real: Vec<f64>,
    nonzero: Vec<f64>,
    zero: Vec<f64>,
}
impl CoulombDerivativeTensor {
    pub fn order(&self) -> usize {
        self.order
    }
    fn index(&self, m: [usize; 3]) -> usize {
        (m[0] * self.dim + m[1]) * self.dim + m[2]
    }
    pub fn components(&self, m: [usize; 3]) -> Result<[f64; 3]> {
        if m.iter()
            .try_fold(0usize, |s, &n| s.checked_add(n))
            .is_none_or(|n| n > self.order)
        {
            return Err(DftError::InvalidInput(
                "Coulomb derivative index exceeds tensor order".into(),
            ));
        }
        let i = self.index(m);
        Ok([self.real[i], self.nonzero[i], self.zero[i]])
    }
    pub fn get(&self, m: [usize; 3]) -> Result<f64> {
        Ok(self.components(m)?.into_iter().sum())
    }
}
pub(super) fn indices(order: usize) -> impl Iterator<Item = [usize; 3]> {
    (0..=order).flat_map(move |x| {
        (0..=order - x).flat_map(move |y| (0..=order - x - y).map(move |z| [x, y, z]))
    })
}
pub(super) fn tensor_size(order: usize) -> Result<(usize, usize)> {
    let dim = order
        .checked_add(1)
        .ok_or_else(|| DftError::InvalidInput("Coulomb derivative order overflow".into()))?;
    let size = dim
        .checked_pow(3)
        .ok_or_else(|| DftError::InvalidInput("Coulomb derivative tensor size overflow".into()))?;
    if size > 1_000_000 {
        return Err(DftError::InvalidInput(
            "Coulomb derivative tensor exceeds the direct-recurrence work limit".into(),
        ));
    }
    Ok((dim, size))
}

/// Cartesian derivatives of exp(i G.r - t^2 |r_open|^2).
/// This multivariate Hermite recurrence retains a tilted open projector.
pub(super) fn exponential_derivatives(
    order: usize,
    dim: usize,
    r_open: Vec3,
    projector: [[f64; 3]; 3],
    g: Vec3,
    t: f64,
    scratch: &mut [Complex],
) {
    let id = |m: [usize; 3]| (m[0] * dim + m[1]) * dim + m[2];
    let r = [r_open.x, r_open.y, r_open.z];
    let wave = [g.x, g.y, g.z];
    let b = std::array::from_fn::<_, 3, _>(|i| Complex::new(-2.0 * t * t * r[i], wave[i]));
    scratch.fill(Complex::ZERO);
    scratch[0] = Complex::ONE;
    for m in indices(order) {
        if m == [0; 3] {
            continue;
        }
        let axis = (0..3).find(|&i| m[i] > 0).unwrap();
        let mut n = m;
        n[axis] -= 1;
        let mut value = b[axis] * scratch[id(n)];
        for j in 0..3 {
            if n[j] > 0 {
                let mut lower = n;
                lower[j] -= 1;
                value += scratch[id(lower)] * (-2.0 * t * t * projector[axis][j] * n[j] as f64);
            }
        }
        scratch[id(m)] = value;
    }
}

impl DimensionalCoulomb {
    /// All mixed position derivatives of a finite-width Gaussian pair.
    /// Width is sqrt(width_a^2+width_b^2); point-self derivatives are not used
    /// for electronic integrals. Cutoff memberships are held fixed.
    pub fn gaussian_derivatives(
        &self,
        r: Vec3,
        width: f64,
        order: usize,
    ) -> Result<CoulombDerivativeTensor> {
        if !r.norm().is_finite() || !width.is_finite() || width <= 0.0 {
            return Err(DftError::InvalidInput(
                "Gaussian Coulomb derivatives require finite displacement and positive width"
                    .into(),
            ));
        }
        let (dim, size) = tensor_size(order)?;
        let mut out = CoulombDerivativeTensor {
            order,
            dim,
            real: vec![0.0; size],
            nonzero: vec![0.0; size],
            zero: vec![0.0; size],
        };
        let alpha = self.params.alpha;
        let inverse = width.recip();
        if alpha != inverse {
            for index in self
                .geometry
                .nearby_translation_indices(r, self.params.rcut)?
            {
                let delta = r + self.geometry.lattice().translation(index);
                let a = inverse * inverse;
                let b = alpha * alpha;
                let ra = RTensor::build(order, a, delta, &boys_values(order, a * delta.norm2())?);
                let rb = RTensor::build(order, b, delta, &boys_values(order, b * delta.norm2())?);
                for m in indices(order) {
                    let i = out.index(m);
                    out.real[i] += 2.0 / PI.sqrt()
                        * (inverse * ra.get(m[0], m[1], m[2]) - alpha * rb.get(m[0], m[1], m[2]));
                }
            }
        }
        let d = self.geometry.dimension();
        let measure = self.geometry.measure();
        let open = self.geometry.perpendicular(r);
        let axes = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let columns = axes.map(|v| self.geometry.perpendicular(v));
        let projector = std::array::from_fn(|i| std::array::from_fn(|j| axes[i].dot(columns[j])));
        let factor = 2.0 * PI.powf((d as f64 - 1.0) * 0.5) / measure;
        let mut scratch = vec![Complex::ZERO; size];
        for &g in &self.reciprocal {
            let phase = g.dot(r);
            let trig = Complex::new(phase.cos(), phase.sin());
            if d == 3 {
                exponential_derivatives(order, dim, open, projector, g, 0.0, &mut scratch);
                let scale =
                    4.0 * PI / measure * (-g.norm2() / (4.0 * alpha * alpha)).exp() / g.norm2();
                for m in indices(order) {
                    let i = out.index(m);
                    out.nonzero[i] += scale * (trig * scratch[i]).re;
                }
            } else {
                for (&x, &w) in self.nodes.iter().zip(&self.weights) {
                    let t = alpha * x;
                    exponential_derivatives(order, dim, open, projector, g, t, &mut scratch);
                    let scale = factor
                        * w
                        * alpha
                        * t.powi(-(d as i32))
                        * (-g.norm2() / (4.0 * t * t) - open.norm2() * t * t).exp();
                    for m in indices(order) {
                        let i = out.index(m);
                        out.nonzero[i] += scale * (trig * scratch[i]).re;
                    }
                }
            }
        }
        match d {
            3 => out.zero[0] = -PI / (measure * alpha * alpha) + PI * width * width / measure,
            2 => {
                let normal = columns
                    .iter()
                    .max_by(|a, b| a.norm2().total_cmp(&b.norm2()))
                    .unwrap()
                    .normalized();
                let z = normal.dot(r);
                let n = [normal.x, normal.y, normal.z];
                let radial = slab_zero_radial_derivatives(z, alpha, measure, order);
                for m in indices(order) {
                    let degree = m.iter().sum::<usize>();
                    let direction = (0..3).map(|i| n[i].powi(m[i] as i32)).product::<f64>();
                    let i = out.index(m);
                    out.zero[i] = radial[degree] * direction;
                }
            }
            _ => {
                out.zero[0] = 2.0 / measure * alpha.ln();
                for (&x, &w) in self.nodes.iter().zip(&self.weights) {
                    let t = alpha * x;
                    exponential_derivatives(
                        order,
                        dim,
                        open,
                        projector,
                        Vec3::zero(),
                        t,
                        &mut scratch,
                    );
                    let scale = 2.0 / measure * w / x;
                    let exponent = -open.norm2() * t * t;
                    out.zero[0] += scale * exponent.exp_m1();
                    for m in indices(order) {
                        if m != [0; 3] {
                            let i = out.index(m);
                            out.zero[i] += scale * exponent.exp() * scratch[i].re;
                        }
                    }
                }
            }
        }
        if out
            .real
            .iter()
            .chain(&out.nonzero)
            .chain(&out.zero)
            .any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "Coulomb derivatives exceed finite numerical range".into(),
            ));
        }
        Ok(out)
    }
}

/// Isolated-slab zero mode and its derivatives along a unit normal.
fn slab_zero_radial_derivatives(z: f64, alpha: f64, area: f64, order: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(order + 1);
    let gaussian = (-alpha * alpha * z * z).exp();
    let erf = libm::erf(alpha * z);
    let scale = -2.0 * PI / area;
    values.push(scale * (z * erf + gaussian / (alpha * PI.sqrt())));
    if order > 0 {
        values.push(scale * erf);
    }
    if order > 1 {
        values.push(gaussian);
    }
    if order > 2 {
        values.push(-2.0 * alpha * alpha * z * values[2]);
    }
    for n in 4..=order {
        values.push(-2.0 * alpha * alpha * (z * values[n - 1] + (n - 3) as f64 * values[n - 2]));
    }
    for value in values.iter_mut().skip(2) {
        *value *= -4.0 * PI.sqrt() * alpha / area;
    }
    values
}

#[derive(Clone, Debug)]
pub struct CartesianGaussian {
    pub center: Vec3,
    pub exponent: f64,
    pub angular: [usize; 3],
}
impl CartesianGaussian {
    fn normalization(&self) -> Result<f64> {
        let order = self
            .angular
            .iter()
            .try_fold(0usize, |s, &l| s.checked_add(l))
            .ok_or_else(|| {
                DftError::InvalidInput("Cartesian Gaussian angular order overflow".into())
            })?;
        tensor_size(order)?;
        if !self.center.norm().is_finite()
            || !self.exponent.is_finite()
            || self.exponent <= 0.0
            || self.angular.iter().any(|&l| l > i32::MAX as usize / 2)
        {
            return Err(DftError::InvalidInput(
                "Cartesian Gaussian needs finite centre, positive exponent and \
                representable angular orders"
                    .into(),
            ));
        }
        let n = primitive_norm(
            self.exponent,
            (
                self.angular[0] as i32,
                self.angular[1] as i32,
                self.angular[2] as i32,
            ),
        );
        if !n.is_finite() || n <= 0.0 {
            return Err(DftError::InvalidInput(
                "Cartesian Gaussian normalization exceeds finite numerical range".into(),
            ));
        }
        Ok(n)
    }
}
#[derive(Clone, Debug)]
pub struct GaussianNucleus {
    pub center: Vec3,
    pub charge: f64,
    /// Point nucleus at zero; normalized Gaussian charge for positive width.
    pub width: f64,
}
#[derive(Clone, Debug)]
pub struct GaussianIntegralOptions {
    pub alpha: Option<f64>,
    pub real_cutoff: Option<f64>,
    pub reciprocal_cutoff: Option<f64>,
    pub quadrature_order: usize,
    pub tolerance: f64,
}
impl Default for GaussianIntegralOptions {
    fn default() -> Self {
        Self {
            alpha: None,
            real_cutoff: None,
            reciprocal_cutoff: None,
            quadrature_order: 96,
            tolerance: 1e-11,
        }
    }
}
pub struct GaussianIntegralReport {
    pub value: f64,
    pub components: ElectrostaticComponents,
    /// Gaussian centres first; nuclear source centre last for attraction.
    pub center_gradients: Vec<Vec3>,
    pub quadrature_value_defect: f64,
    pub quadrature_gradient_defect: f64,
    pub translation_defect: f64,
    pub quadrature_order: usize,
    pub params: EwaldParams,
    pub converged: bool,
    pub kind: &'static str,
    pub units: &'static str,
    pub model: &'static str,
    pub warnings: Vec<String>,
}
pub(super) fn pair_center(a: &CartesianGaussian, b: &CartesianGaussian) -> Vec3 {
    (a.center * a.exponent + b.center * b.exponent) / (a.exponent + b.exponent)
}
pub(super) fn pair_coefficients(a: &CartesianGaussian, b: &CartesianGaussian) -> [Vec<f64>; 3] {
    let delta = a.center - b.center;
    let r = [delta.x, delta.y, delta.z];
    std::array::from_fn(|axis| {
        let (i, j) = (a.angular[axis], b.angular[axis]);
        let table = ETable::build(i, j, r[axis], a.exponent, b.exponent);
        (0..=i + j).map(|t| table.get(i, j, t)).collect()
    })
}
fn contract_components(
    components: &mut impl FnMut([usize; 3]) -> Result<[f64; 3]>,
    gaussians: &[CartesianGaussian],
    extra: [usize; 3],
) -> Result<[f64; 3]> {
    let a = pair_coefficients(&gaussians[0], &gaussians[1]);
    let p = gaussians[0].exponent + gaussians[1].exponent;
    let (b, scale) = if gaussians.len() == 4 {
        let q = gaussians[2].exponent + gaussians[3].exponent;
        (
            pair_coefficients(&gaussians[2], &gaussians[3]),
            (PI / p).powf(1.5) * (PI / q).powf(1.5),
        )
    } else {
        ([vec![1.0], vec![1.0], vec![1.0]], (PI / p).powf(1.5))
    };
    let mut c: [Vec<f64>; 3] = std::array::from_fn(|i| vec![0.0; a[i].len() + b[i].len() - 1]);
    for axis in 0..3 {
        for (i, &x) in a[axis].iter().enumerate() {
            for (j, &y) in b[axis].iter().enumerate() {
                c[axis][i + j] += x * y * if j % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    let mut result = [0.0; 3];
    for (i, &x) in c[0].iter().enumerate() {
        for (j, &y) in c[1].iter().enumerate() {
            for (k, &z) in c[2].iter().enumerate() {
                if x * y * z == 0.0 {
                    continue;
                }
                let parts = components([i + extra[0], j + extra[1], k + extra[2]])?;
                for component in 0..3 {
                    result[component] += scale * x * y * z * parts[component];
                }
            }
        }
    }
    Ok(result)
}
fn contract(
    tensor: &CoulombDerivativeTensor,
    gaussians: &[CartesianGaussian],
    extra: [usize; 3],
) -> Result<[f64; 3]> {
    contract_components(&mut |m| tensor.components(m), gaussians, extra)
}
pub(crate) fn prepare_integral_tensor(
    kernel: &DimensionalCoulomb,
    gaussians: &[CartesianGaussian],
    source: Option<&GaussianNucleus>,
    order: usize,
) -> Result<CoulombDerivativeTensor> {
    let (a, b) = (&gaussians[0], &gaussians[1]);
    let p = a.exponent + b.exponent;
    let (r, width) = if let Some(source) = source {
        (
            pair_center(a, b) - source.center,
            (p.recip() + source.width.powi(2)).sqrt(),
        )
    } else {
        let q = gaussians[2].exponent + gaussians[3].exponent;
        (
            pair_center(a, b) - pair_center(&gaussians[2], &gaussians[3]),
            (p.recip() + q.recip()).sqrt(),
        )
    };
    kernel.gaussian_derivatives(r, width, order)
}

/// Unnormalized polynomial contraction for pre-normalized spherical shells.
/// One tensor is shared by all Cartesian components of a primitive quartet.
pub(crate) fn contract_integral_tensor(
    tensor: &CoulombDerivativeTensor,
    gaussians: &[CartesianGaussian],
    nuclear: bool,
) -> Result<([f64; 3], Vec<Vec3>)> {
    let components = contract(tensor, gaussians, [0; 3])?;
    let mut gradients = vec![Vec3::zero(); gaussians.len() + usize::from(nuclear)];
    for (center, gaussian) in gaussians.iter().enumerate() {
        let mut values = [0.0; 3];
        for axis in 0..3 {
            let mut raised = gaussians.to_vec();
            raised[center].angular[axis] += 1;
            let mut derivative = 2.0
                * gaussian.exponent
                * contract(tensor, &raised, [0; 3])?.into_iter().sum::<f64>();
            if gaussian.angular[axis] > 0 {
                let mut lowered = gaussians.to_vec();
                lowered[center].angular[axis] -= 1;
                derivative -= gaussian.angular[axis] as f64
                    * contract(tensor, &lowered, [0; 3])?.into_iter().sum::<f64>();
            }
            values[axis] = derivative;
        }
        gradients[center] = Vec3::new(values[0], values[1], values[2]);
    }
    if nuclear {
        let mut values = [0.0; 3];
        for axis in 0..3 {
            let mut extra = [0; 3];
            extra[axis] = 1;
            values[axis] = -contract(tensor, gaussians, extra)?.into_iter().sum::<f64>();
        }
        gradients[gaussians.len()] = Vec3::new(values[0], values[1], values[2]);
    }
    Ok((components, gradients))
}

/// Mixed bra/ket spatial derivatives of a contracted Green integral.
/// Raising/lowering acts on the unnormalized polynomial factors; the caller
/// retains the original shell's primitive normalization coefficients.
pub(crate) fn mixed_basis_derivative(
    tensor: &CoulombDerivativeTensor,
    gaussians: &[CartesianGaussian],
    bra_axis: usize,
    ket_axis: usize,
) -> Result<f64> {
    let a = &gaussians[0];
    let b = &gaussians[1];
    let mut raised = gaussians.to_vec();
    raised[0].angular[bra_axis] += 1;
    raised[1].angular[ket_axis] += 1;
    let mut value =
        4.0 * a.exponent * b.exponent * contract(tensor, &raised, [0; 3])?.iter().sum::<f64>();
    if b.angular[ket_axis] > 0 {
        let mut term = gaussians.to_vec();
        term[0].angular[bra_axis] += 1;
        term[1].angular[ket_axis] -= 1;
        value -= 2.0
            * a.exponent
            * b.angular[ket_axis] as f64
            * contract(tensor, &term, [0; 3])?.iter().sum::<f64>();
    }
    if a.angular[bra_axis] > 0 {
        let mut term = gaussians.to_vec();
        term[0].angular[bra_axis] -= 1;
        term[1].angular[ket_axis] += 1;
        value -= 2.0
            * b.exponent
            * a.angular[bra_axis] as f64
            * contract(tensor, &term, [0; 3])?.iter().sum::<f64>();
    }
    if a.angular[bra_axis] > 0 && b.angular[ket_axis] > 0 {
        let mut term = gaussians.to_vec();
        term[0].angular[bra_axis] -= 1;
        term[1].angular[ket_axis] -= 1;
        value += a.angular[bra_axis] as f64
            * b.angular[ket_axis] as f64
            * contract(tensor, &term, [0; 3])?.iter().sum::<f64>();
    }
    Ok(value)
}

/// Integral of (partial_axis Green) times a linear coordinate potential.
/// Integration by parts differentiates both Gaussian factors and the explicit
/// coordinate. The original primitive normalization stays with the caller.
pub(crate) fn linear_gradient_integral(
    tensor: &CoulombDerivativeTensor,
    gaussians: &[CartesianGaussian],
    axis: usize,
    linear: [f64; 3],
) -> Result<f64> {
    let weighted = |gs: &[CartesianGaussian]| -> Result<f64> {
        let mut value = (0..3).map(|j| linear[j] * gs[0].center.get(j)).sum::<f64>()
            * contract(tensor, gs, [0; 3])?.iter().sum::<f64>();
        for j in 0..3 {
            if linear[j] != 0.0 {
                let mut raised = gs.to_vec();
                raised[0].angular[j] += 1;
                value += linear[j] * contract(tensor, &raised, [0; 3])?.iter().sum::<f64>();
            }
        }
        Ok(value)
    };
    let mut value = -linear[axis] * contract(tensor, gaussians, [0; 3])?.iter().sum::<f64>();
    for center in 0..2 {
        let g = &gaussians[center];
        let mut raised = gaussians.to_vec();
        raised[center].angular[axis] += 1;
        value += 2.0 * g.exponent * weighted(&raised)?;
        if g.angular[axis] > 0 {
            let mut lowered = gaussians.to_vec();
            lowered[center].angular[axis] -= 1;
            value -= g.angular[axis] as f64 * weighted(&lowered)?;
        }
    }
    Ok(value)
}

fn evaluate_integral(
    kernel: &DimensionalCoulomb,
    gaussians: &[CartesianGaussian],
    source: Option<&GaussianNucleus>,
) -> Result<([f64; 3], Vec<Vec3>)> {
    let order = gaussians
        .iter()
        .flat_map(|g| g.angular)
        .try_fold(1usize, |s, l| s.checked_add(l))
        .ok_or_else(|| DftError::InvalidInput("Gaussian integral angular order overflow".into()))?;
    let tensor = prepare_integral_tensor(kernel, gaussians, source, order)?;
    let normalization = gaussians
        .iter()
        .map(CartesianGaussian::normalization)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .product::<f64>()
        * source.map_or(1.0, |s| -s.charge);
    let (parts, gradients) = contract_integral_tensor(&tensor, gaussians, source.is_some())?;
    Ok((
        parts.map(|v| v * normalization),
        gradients.into_iter().map(|v| v * normalization).collect(),
    ))
}

/// Two normalized Cartesian Gaussians plus a point/Gaussian nuclear source,
/// or four normalized Gaussians for an electron-repulsion integral.
/// This is the periodic Green integral, before AO/Bloch/density contraction.
pub fn periodic_gaussian_integral(
    geometry: &PeriodicGeometry,
    gaussians: &[CartesianGaussian],
    source: Option<&GaussianNucleus>,
    options: &GaussianIntegralOptions,
) -> Result<GaussianIntegralReport> {
    if geometry.dimension() == 0
        || !matches!((gaussians.len(), source.is_some()), (2, true) | (4, false))
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || !(16..=1024).contains(&options.quadrature_order)
        || [
            options.alpha,
            options.real_cutoff,
            options.reciprocal_cutoff,
        ]
        .iter()
        .flatten()
        .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(DftError::InvalidInput(
            "periodic Gaussian integral requires 1D/2D/3D geometry, two Gaussians \
            plus source or four Gaussians, positive finite controls and quadrature \
            order 16..1024"
                .into(),
        ));
    }
    for gaussian in gaussians {
        gaussian.normalization()?;
    }
    if source.is_some_and(|s| {
        !s.center.norm().is_finite()
            || !s.charge.is_finite()
            || !s.width.is_finite()
            || s.width < 0.0
    }) {
        return Err(DftError::InvalidInput(
            "Gaussian nuclear source needs finite centre/charge and non-negative width".into(),
        ));
    }
    let p = gaussians[0].exponent + gaussians[1].exponent;
    let width = if let Some(s) = source {
        (p.recip() + s.width.powi(2)).sqrt()
    } else {
        (p.recip() + (gaussians[2].exponent + gaussians[3].exponent).recip()).sqrt()
    };
    let order = gaussians
        .iter()
        .flat_map(|g| g.angular)
        .try_fold(1usize, |s, l| s.checked_add(l))
        .ok_or_else(|| DftError::InvalidInput("Gaussian angular order overflow".into()))?;
    tensor_size(order)?;
    let alpha = options
        .alpha
        .unwrap_or(PI.sqrt() / geometry.measure().powf(1.0 / geometry.dimension() as f64));
    let logarithm = (-options.tolerance.min(1e-12).ln()
        + 2.0 * order as f64 * ((order + 2) as f64).ln())
    .sqrt();
    let params = EwaldParams {
        alpha,
        rcut: options
            .real_cutoff
            .unwrap_or(logarithm * alpha.recip().max(width)),
        gcut: options.reciprocal_cutoff.unwrap_or(2.0 * alpha * logarithm),
    };
    let coarse = DimensionalCoulomb::new(geometry.clone(), params, options.quadrature_order)?;
    let fine = DimensionalCoulomb::new(geometry.clone(), params, 2 * options.quadrature_order)?;
    let (old, old_gradients) = evaluate_integral(&coarse, gaussians, source)?;
    let (values, center_gradients) = evaluate_integral(&fine, gaussians, source)?;
    let value = values.into_iter().sum::<f64>();
    let quadrature_value_defect = (value - old.into_iter().sum::<f64>()).abs();
    let quadrature_gradient_defect = center_gradients
        .iter()
        .zip(old_gradients)
        .map(|(a, b)| (*a - b).norm())
        .fold(0.0, f64::max);
    let translation_defect = center_gradients
        .iter()
        .fold(Vec3::zero(), |s, &g| s + g)
        .norm();
    let converged = value.is_finite()
        && center_gradients.iter().all(|g| g.norm().is_finite())
        && quadrature_value_defect <= options.tolerance
        && quadrature_gradient_defect <= options.tolerance
        && translation_defect <= options.tolerance;
    let mut warnings = vec![
        "quadrature agreement is measured; independently converge Ewald cutoffs \
        and splitting parameter"
            .into(),
        "primitive periodic Green integrals are components of a jointly \
            compensated electronic problem, not self-consistent SCF energies"
            .into(),
    ];
    if !converged {
        warnings.push(
            "Gaussian integral quadrature or translation defect exceeds the requested tolerance"
                .into(),
        );
    }
    Ok(GaussianIntegralReport {
        value,
        components: ElectrostaticComponents {
            real_space: values[0],
            reciprocal_nonzero: values[1],
            reciprocal_zero: values[2],
            point_self: 0.0,
        },
        center_gradients,
        quadrature_value_defect,
        quadrature_gradient_defect,
        translation_defect,
        quadrature_order: 2 * options.quadrature_order,
        params,
        converged,
        kind: if source.is_some() {
            "nuclear-attraction"
        } else {
            "electron-repulsion"
        },
        units: "normalized Cartesian primitive integral Hartree; centre gradients \
            Hartree/Bohr",
        model: "dimensional periodic Coulomb Green integral with normalized Cartesian \
            Gaussians; zero-average 3D, isolated slab/wire gauge",
        warnings,
    })
}

pub(crate) mod lattice_green;

/// Unnormalized Hermite contraction of fixed-centre lattice partials.
/// Cartesian/spherical shell normalization remains with the original caller.
pub(crate) fn contract_integral_lattice(
    kernel: &DimensionalCoulomb,
    gs: &[CartesianGaussian],
    source: Option<&GaussianNucleus>,
    order: usize,
) -> Result<[[f64; 3]; 9]> {
    let p = gs[0].exponent + gs[1].exponent;
    let (r, width) = if let Some(s) = source {
        (
            pair_center(&gs[0], &gs[1]) - s.center,
            (p.recip() + s.width.powi(2)).sqrt(),
        )
    } else {
        let q = gs[2].exponent + gs[3].exponent;
        (
            pair_center(&gs[0], &gs[1]) - pair_center(&gs[2], &gs[3]),
            (p.recip() + q.recip()).sqrt(),
        )
    };
    let tensor = lattice_green::fixed_displacement_cell_response(
        kernel,
        &kernel.reciprocal,
        r,
        width,
        order,
    )?;
    let mut result = [[0.0; 3]; 9];
    for (c, row) in result.iter_mut().enumerate() {
        *row = contract_components(&mut |m| Ok(tensor.components(m)?[c]), gs, [0; 3])?;
    }
    Ok(result)
}
