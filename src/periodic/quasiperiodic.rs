// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex dimensional Coulomb kernels for explicitly labelled Bloch transfers.
//! Fractional transfer coordinates distinguish the uniform channel from an
//! arbitrarily small nonzero wave without a denominator or wave-vector floor.

use super::electrostatics::DimensionalCoulomb;
use super::gaussian_coulomb::{
    exponential_derivatives, indices, pair_center, pair_coefficients, tensor_size,
    CartesianGaussian,
};
use super::PeriodicGeometry;
use crate::integrals::analytic::boys_values;
use crate::integrals::hermite::RTensor;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::{DftError, Result};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct PeriodicTransfer {
    geometry: PeriodicGeometry,
    fractional: [f64; 3],
    reduced: [f64; 3],
}
impl PeriodicTransfer {
    pub fn new(geometry: &PeriodicGeometry, fractional: [f64; 3]) -> Result<Self> {
        if geometry.dimension() == 0
            || fractional.iter().any(|f| !f.is_finite() || f.abs() > 1e12)
            || (0..3).any(|i| !geometry.boundary().axes[i] && fractional[i] != 0.0)
        {
            return Err(DftError::InvalidInput("Bloch transfer needs finite fractional coordinates on periodic axes; open coordinates must be zero".into()));
        }
        let reduced = fractional.map(|v| v - v.round());
        let value = Self {
            geometry: geometry.clone(),
            fractional,
            reduced,
        };
        if !value.cartesian().norm().is_finite() || !value.reduced_cartesian().norm().is_finite() {
            return Err(DftError::InvalidInput(
                "Bloch transfer exceeds finite Cartesian range".into(),
            ));
        }
        Ok(value)
    }
    pub fn fractional(&self) -> [f64; 3] {
        self.fractional
    }
    pub fn reduced_fractional(&self) -> [f64; 3] {
        self.reduced
    }
    fn vector(&self, coordinates: [f64; 3]) -> Vec3 {
        self.geometry
            .reciprocal()
            .into_iter()
            .zip(coordinates)
            .fold(Vec3::zero(), |v, (b, f)| v + b * f)
    }
    pub fn cartesian(&self) -> Vec3 {
        self.vector(self.fractional)
    }
    pub fn reduced_cartesian(&self) -> Vec3 {
        self.vector(self.reduced)
    }
    pub fn is_uniform(&self) -> bool {
        self.reduced.iter().all(|&v| v == 0.0)
    }
    pub fn matches(&self, geometry: &PeriodicGeometry) -> bool {
        self.geometry.boundary() == geometry.boundary()
            && self.geometry.lattice().a == geometry.lattice().a
    }
    pub(crate) fn reciprocal_waves(&self, cutoff: f64) -> Result<Vec<Vec3>> {
        let q = self.reduced_cartesian();
        let mut waves = self.geometry.reciprocal_within(cutoff + q.norm())?;
        waves.push(Vec3::zero());
        let mut waves: Vec<_> = waves
            .into_iter()
            .map(|g| g + q)
            .filter(|g| g.norm() <= cutoff)
            .collect();
        waves.sort_by(|a, b| {
            a.norm2()
                .total_cmp(&b.norm2())
                .then_with(|| a.x.total_cmp(&b.x))
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| a.z.total_cmp(&b.z))
        });
        if waves.is_empty() {
            return Err(DftError::InvalidInput(
                "finite-q reciprocal cutoff contains no modes".into(),
            ));
        }
        Ok(waves)
    }
}

pub(crate) struct ComplexCoulombTensor {
    order: usize,
    dim: usize,
    parts: [Vec<Complex>; 3],
}
impl ComplexCoulombTensor {
    pub(crate) fn components(&self, m: [usize; 3]) -> Result<[Complex; 3]> {
        if m.iter()
            .try_fold(0usize, |s, &v| s.checked_add(v))
            .is_none_or(|v| v > self.order)
        {
            return Err(DftError::InvalidInput(
                "complex Coulomb derivative exceeds tensor order".into(),
            ));
        }
        let i = (m[0] * self.dim + m[1]) * self.dim + m[2];
        Ok(std::array::from_fn(|c| self.parts[c][i]))
    }
}

/// Derivatives of sum_R exp(-i q.R) erf(|r+R|/width)/|r+R|.
/// Reciprocal waves are G+q in the physical periodic subspace. The exact
/// folded uniform channel retains the dimensional open-direction zero mode.
pub(crate) fn gaussian_tensor(
    kernel: &DimensionalCoulomb,
    transfer: &PeriodicTransfer,
    r: Vec3,
    width: f64,
    order: usize,
) -> Result<ComplexCoulombTensor> {
    if !transfer.matches(kernel.geometry())
        || !r.norm().is_finite()
        || !width.is_finite()
        || width <= 0.0
    {
        return Err(DftError::InvalidInput("quasiperiodic Gaussian Coulomb needs matching geometry, finite displacement and positive width".into()));
    }
    let (dim, size) = tensor_size(order)?;
    let mut result = ComplexCoulombTensor {
        order,
        dim,
        parts: std::array::from_fn(|_| vec![Complex::ZERO; size]),
    };
    let id = |m: [usize; 3]| (m[0] * dim + m[1]) * dim + m[2];
    if transfer.is_uniform() {
        let tensor = kernel.gaussian_derivatives(r, width, order)?;
        for m in indices(order) {
            let values = tensor.components(m)?;
            for c in 0..3 {
                result.parts[c][id(m)] = Complex::from(values[c]);
            }
        }
        return Ok(result);
    }
    let q = transfer.reduced_cartesian();
    let alpha = kernel.params().alpha;
    let inverse = width.recip();
    if alpha != inverse {
        for index in kernel
            .geometry()
            .nearby_translation_indices(r, kernel.params().rcut)?
        {
            let translation = kernel.geometry().lattice().translation(index);
            let delta = r + translation;
            let phase = -q.dot(translation);
            let phase = Complex::new(phase.cos(), phase.sin());
            let a = inverse * inverse;
            let b = alpha * alpha;
            let ra = RTensor::build(order, a, delta, &boys_values(order, a * delta.norm2())?);
            let rb = RTensor::build(order, b, delta, &boys_values(order, b * delta.norm2())?);
            for m in indices(order) {
                result.parts[0][id(m)] += phase
                    * (2.0 / PI.sqrt()
                        * (inverse * ra.get(m[0], m[1], m[2]) - alpha * rb.get(m[0], m[1], m[2])));
            }
        }
    }
    let geometry = kernel.geometry();
    let d = geometry.dimension();
    let open = geometry.perpendicular(r);
    let axes = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let columns = axes.map(|v| geometry.perpendicular(v));
    let projector = std::array::from_fn(|i| std::array::from_fn(|j| axes[i].dot(columns[j])));
    let factor = 2.0 * PI.powf((d as f64 - 1.0) * 0.5) / geometry.measure();
    let mut scratch = vec![Complex::ZERO; size];
    for g in transfer.reciprocal_waves(kernel.params().gcut)? {
        let g2 = g.norm2();
        if !g2.is_finite() || g2 == 0.0 {
            return Err(DftError::LinearAlgebra(
                "nonuniform Fourier wave lies outside finite squared-norm range".into(),
            ));
        }
        let angle = g.dot(r);
        let phase = Complex::new(angle.cos(), angle.sin());
        if d == 3 {
            exponential_derivatives(order, dim, open, projector, g, 0.0, &mut scratch);
            let scale = 4.0 * PI / geometry.measure() * (-g2 / (4.0 * alpha * alpha)).exp() / g2;
            for m in indices(order) {
                result.parts[1][id(m)] += phase * scratch[id(m)] * scale;
            }
        } else {
            // t=alpha exp(-s) resolves a small nonzero |G+q| without a
            // floor. The omitted tail begins beyond u=|G+q|/(2t)=12,
            // where exp(-u^2) suppresses it, independent of node order.
            let length = ((2.0 * alpha / g2.sqrt()).ln() + 12.0_f64.ln()).max(1.0);
            for (&x, &w) in kernel.nodes.iter().zip(&kernel.weights) {
                let t = alpha * (-length * x).exp();
                exponential_derivatives(order, dim, open, projector, g, t, &mut scratch);
                let scale = factor
                    * w
                    * length
                    * t.powi(1 - d as i32)
                    * (-g2 / (4.0 * t * t) - open.norm2() * t * t).exp();
                for m in indices(order) {
                    result.parts[1][id(m)] += phase * scratch[id(m)] * scale;
                }
            }
        }
    }
    if result.parts.iter().flatten().any(|v| !v.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "quasiperiodic Coulomb tensor exceeds finite numerical range".into(),
        ));
    }
    Ok(result)
}

pub(crate) fn prepare_quartet(
    kernel: &DimensionalCoulomb,
    transfer: &PeriodicTransfer,
    gaussians: &[CartesianGaussian],
    order: usize,
) -> Result<ComplexCoulombTensor> {
    if gaussians.len() != 4 {
        return Err(DftError::InvalidInput(
            "quasiperiodic electronic Coulomb needs four Gaussians".into(),
        ));
    }
    let p = gaussians[0].exponent + gaussians[1].exponent;
    let s = gaussians[2].exponent + gaussians[3].exponent;
    gaussian_tensor(
        kernel,
        transfer,
        pair_center(&gaussians[0], &gaussians[1]) - pair_center(&gaussians[2], &gaussians[3]),
        (p.recip() + s.recip()).sqrt(),
        order,
    )
}
pub(crate) fn contract_quartet(
    tensor: &ComplexCoulombTensor,
    gaussians: &[CartesianGaussian],
) -> Result<Complex> {
    let a = pair_coefficients(&gaussians[0], &gaussians[1]);
    let b = pair_coefficients(&gaussians[2], &gaussians[3]);
    let mut coefficients: [Vec<f64>; 3] =
        std::array::from_fn(|i| vec![0.0; a[i].len() + b[i].len() - 1]);
    for axis in 0..3 {
        for (i, &x) in a[axis].iter().enumerate() {
            for (j, &y) in b[axis].iter().enumerate() {
                coefficients[axis][i + j] += x * y * if j % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    let p = gaussians[0].exponent + gaussians[1].exponent;
    let s = gaussians[2].exponent + gaussians[3].exponent;
    let scale = (PI / p).powf(1.5) * (PI / s).powf(1.5);
    let mut result = Complex::ZERO;
    for (i, &x) in coefficients[0].iter().enumerate() {
        for (j, &y) in coefficients[1].iter().enumerate() {
            for (k, &z) in coefficients[2].iter().enumerate() {
                if x * y * z != 0.0 {
                    result += tensor
                        .components([i, j, k])?
                        .into_iter()
                        .fold(Complex::ZERO, |s, v| s + v)
                        * (scale * x * y * z);
                }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::{ewald::EwaldParams, features::PeriodicBoundary, Lattice};

    fn geometry(mask: usize) -> PeriodicGeometry {
        PeriodicGeometry::new(
            Lattice::cubic(7.0),
            PeriodicBoundary {
                axes: std::array::from_fn(|a| mask & (1 << a) != 0),
            },
        )
        .unwrap()
    }
    fn kernel(g: &PeriodicGeometry, alpha: f64, nodes: usize) -> DimensionalCoulomb {
        DimensionalCoulomb::new(
            g.clone(),
            EwaldParams {
                alpha,
                rcut: 18.0,
                gcut: 3.7,
            },
            nodes,
        )
        .unwrap()
    }
    fn value(
        k: &DimensionalCoulomb,
        q: &PeriodicTransfer,
        r: Vec3,
        width: f64,
        m: [usize; 3],
    ) -> Complex {
        gaussian_tensor(k, q, r, width, m.iter().sum())
            .unwrap()
            .components(m)
            .unwrap()
            .into_iter()
            .fold(Complex::ZERO, |s, v| s + v)
    }
    fn transfer(g: &PeriodicGeometry, f: f64) -> PeriodicTransfer {
        let mut coordinates = [0.0; 3];
        coordinates[g.boundary().axes.iter().position(|&a| a).unwrap()] = f;
        PeriodicTransfer::new(g, coordinates).unwrap()
    }
    #[test]
    fn all_masks_gamma_conjugation_and_exact_transfer_labels() {
        let r = Vec3::new(0.43, 0.37, -0.29);
        for mask in 1..8 {
            let g = geometry(mask);
            let k = kernel(&g, 0.65, 192);
            let gamma = transfer(&g, 1.0);
            assert!(gamma.is_uniform());
            let real = k.gaussian_derivatives(r, 1.1, 4).unwrap();
            let folded = gaussian_tensor(&k, &gamma, r, 1.1, 4).unwrap();
            let plus = transfer(&g, 0.231);
            let minus = transfer(&g, -0.231);
            assert!(!transfer(&g, 1e-14).is_uniform());
            for m in indices(4) {
                assert_eq!(
                    folded.components(m).unwrap(),
                    real.components(m).unwrap().map(Complex::from)
                );
                let a = value(&k, &plus, r, 1.1, m);
                let b = value(&k, &minus, r, 1.1, m);
                assert!((a.conj() - b).abs() < 3e-12);
            }
        }
        let g = geometry(1);
        assert!(PeriodicTransfer::new(&g, [0.1, 0.01, 0.0]).is_err());
        assert!(PeriodicTransfer::new(&g, [f64::NAN, 0.0, 0.0]).is_err());
    }
    #[test]
    fn rational_transfer_matches_independent_supercell_folding() {
        let r = Vec3::new(0.43, 0.37, -0.29);
        for mask in 1..8 {
            let g = geometry(mask);
            let a = g.boundary().axes.iter().position(|&a| a).unwrap();
            let mut lattice = g.lattice().clone();
            lattice.a[a] = lattice.a[a] * 3.0;
            let supercell = PeriodicGeometry::new(lattice, g.boundary()).unwrap();
            let k = kernel(&g, 0.65, 192);
            let sk = kernel(&supercell, 0.65, 192);
            let q = transfer(&g, 1.0 / 3.0);
            let translation = g.lattice().a[a];
            for m in indices(4) {
                let expected = (0..3).fold(Complex::ZERO, |sum, s| {
                    let t = translation * s as f64;
                    let phase = -q.reduced_cartesian().dot(t);
                    let real: f64 = sk
                        .gaussian_derivatives(r + t, 1.1, 4)
                        .unwrap()
                        .components(m)
                        .unwrap()
                        .into_iter()
                        .sum();
                    sum + Complex::new(phase.cos(), phase.sin()) * real
                });
                let actual = value(&k, &q, r, 1.1, m);
                assert!(
                    (actual - expected).abs() < 5e-11,
                    "mask={mask} m={m:?} actual={actual:?} expected={expected:?}"
                );
                let phase = q.reduced_cartesian().dot(translation);
                let shifted = value(&k, &q, r + translation, 1.1, m);
                assert!((shifted - actual * Complex::new(phase.cos(), phase.sin())).abs() < 5e-11);
            }
        }
    }
    fn e1(x: f64) -> f64 {
        assert!(x > 0.0 && x < 1.0);
        let mut factorial_term = 1.0;
        let mut sum = 0.0;
        for n in 1..120 {
            factorial_term *= -x / n as f64;
            sum += factorial_term / n as f64;
        }
        -0.5772156649015329 - x.ln() - sum
    }
    #[test]
    fn small_nonzero_transfer_matches_closed_wire_and_slab_oracles() {
        for mask in [1, 3] {
            let g = geometry(mask);
            let width: f64 = 0.5;
            for f in [0.231, 1e-8, 1e-12] {
                let q = transfer(&g, f);
                let k = kernel(&g, width.recip(), 384);
                let actual = value(&k, &q, Vec3::zero(), width, [0; 3]);
                let expected = q
                    .reciprocal_waves(k.params().gcut)
                    .unwrap()
                    .iter()
                    .map(|wave| {
                        let w = wave.norm();
                        if g.dimension() == 1 {
                            e1((width * w * 0.5).powi(2)) / g.measure()
                        } else {
                            2.0 * PI / g.measure() / w * libm::erfc(width * w * 0.5)
                        }
                    })
                    .sum::<f64>();
                let relative = (actual - Complex::from(expected)).abs() / expected.abs();
                assert!(
                    relative < 3e-12,
                    "mask={mask} f={f:e} relative={relative:e}"
                );
                let coarse = value(
                    &kernel(&g, width.recip(), 192),
                    &q,
                    Vec3::zero(),
                    width,
                    [0; 3],
                );
                assert!((actual - coarse).abs() / actual.abs() < 3e-12);
            }
        }
    }
    fn fit(label: &str, h: &[f64], errors: &[f64]) {
        assert!(errors.iter().all(|e| e.is_finite() && *e > 0.0));
        let x: Vec<_> = h.iter().map(|h| h.ln()).collect();
        let y: Vec<_> = errors.iter().map(|e| e.ln()).collect();
        let xm = x.iter().sum::<f64>() / x.len() as f64;
        let ym = y.iter().sum::<f64>() / y.len() as f64;
        let p = x
            .iter()
            .zip(&y)
            .map(|(x, y)| (x - xm) * (y - ym))
            .sum::<f64>()
            / x.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
        let r2 = 1.0
            - x.iter()
                .zip(&y)
                .map(|(x, y)| (y - ym - p * (x - xm)).powi(2))
                .sum::<f64>()
                / y.iter().map(|y| (y - ym).powi(2)).sum::<f64>();
        println!("dimensional complex Coulomb fit {label}: h={h:?} errors={errors:?} p={p:.12} R2={r2:.12}");
        assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
        for e in errors.windows(2) {
            assert!((e[0] / e[1] / 4.0 - 1.0).abs() < 0.03);
        }
    }
    #[test]
    fn first_through_fourth_derivatives_have_four_step_quadratic_error_fits() {
        let r = Vec3::new(0.43, 0.37, -0.29);
        let hs = [0.04, 0.02, 0.01, 0.005];
        for mask in [1, 3, 7] {
            let g = geometry(mask);
            let k = kernel(&g, 0.65, 192);
            let q = transfer(&g, 0.231);
            for order in 1..=4 {
                let mut m = [0; 3];
                m[0] = order;
                let exact = value(&k, &q, r, 1.1, m);
                m[0] -= 1;
                let approx: Vec<_> = hs
                    .iter()
                    .map(|&h| {
                        let step = Vec3::new(h, 0.0, 0.0);
                        (value(&k, &q, r + step, 1.1, m) - value(&k, &q, r - step, 1.1, m))
                            / (2.0 * h)
                    })
                    .collect();
                let errors: Vec<_> = approx.iter().map(|&a| (a - exact).abs()).collect();
                fit(&format!("mask={mask} order={order}"), &hs, &errors);
                assert!(((approx[3] * 4.0 - approx[2]) / 3.0 - exact).abs() < 1e-8);
            }
        }
    }
}
