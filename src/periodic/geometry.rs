// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Geometry of a periodic subspace embedded in three-dimensional space.
//! Open directions have no images, reciprocal vectors, wrapping or k mesh.

use super::features::PeriodicBoundary;
use super::Lattice;
use crate::math::Vec3;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct PeriodicGeometry {
    lattice: Lattice,
    boundary: PeriodicBoundary,
    reciprocal: [Vec3; 3],
    measure: f64,
}
impl PeriodicGeometry {
    pub fn new(lattice: Lattice, boundary: PeriodicBoundary) -> Result<Self> {
        if lattice
            .a
            .iter()
            .any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite())
        {
            return Err(DftError::InvalidInput(
                "periodic geometry requires finite lattice vectors".into(),
            ));
        }
        let axes: Vec<_> = (0..3).filter(|&i| boundary.axes[i]).collect();
        let mut reciprocal = [Vec3::zero(); 3];
        let tau = 2.0 * std::f64::consts::PI;
        let measure = match axes.len() {
            0 => 1.0,
            1 => {
                let i = axes[0];
                let a = lattice.a[i];
                if a.norm() <= 1e-12 {
                    return Err(DftError::InvalidInput(
                        "periodic vector must have positive length".into(),
                    ));
                }
                reciprocal[i] = a * (tau / a.norm2());
                a.norm()
            }
            2 => {
                let (i, j) = (axes[0], axes[1]);
                let (a, b) = (lattice.a[i], lattice.a[j]);
                let (aa, ab, bb) = (a.norm2(), a.dot(b), b.norm2());
                let det = aa * bb - ab * ab;
                if aa <= 1e-24 || bb <= 1e-24 || det <= 1e-12 * aa * bb {
                    return Err(DftError::InvalidInput(
                        "periodic plane vectors must be independent".into(),
                    ));
                }
                reciprocal[i] = (a * bb - b * ab) * (tau / det);
                reciprocal[j] = (b * aa - a * ab) * (tau / det);
                det.sqrt()
            }
            _ => {
                let v = lattice.volume().abs();
                let norm_product: f64 = lattice.a.iter().map(|v| v.norm()).product();
                if v <= 1e-12 * norm_product || norm_product <= 1e-36 {
                    return Err(DftError::InvalidInput(
                        "three periodic vectors must be independent".into(),
                    ));
                }
                reciprocal = lattice.reciprocal();
                v
            }
        };
        if !measure.is_finite() || reciprocal.iter().any(|v| !v.norm().is_finite()) {
            return Err(DftError::InvalidInput(
                "periodic geometry exceeds finite numerical range".into(),
            ));
        }
        Ok(Self {
            lattice,
            boundary,
            reciprocal,
            measure,
        })
    }
    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }
    pub fn boundary(&self) -> PeriodicBoundary {
        self.boundary
    }
    pub fn dimension(&self) -> usize {
        self.boundary.dimension()
    }
    /// Length, area or volume in Bohr^dimension; 1 for an isolated system.
    pub fn measure(&self) -> f64 {
        self.measure
    }
    /// Subspace dual: a_i.b_j=2*pi*delta_ij on periodic axes; open entries zero.
    pub fn reciprocal(&self) -> [Vec3; 3] {
        self.reciprocal
    }
    pub fn fractional(&self, r: Vec3) -> [f64; 3] {
        std::array::from_fn(|i| r.dot(self.reciprocal[i]) / (2.0 * std::f64::consts::PI))
    }
    pub fn parallel(&self, r: Vec3) -> Vec3 {
        let f = self.fractional(r);
        (0..3).fold(Vec3::zero(), |v, i| v + self.lattice.a[i] * f[i])
    }
    pub fn perpendicular(&self, r: Vec3) -> Vec3 {
        r - self.parallel(r)
    }
    /// Wrap only the periodic projection; preserve the open Cartesian component.
    pub fn wrap(&self, r: Vec3) -> Vec3 {
        let f = self.fractional(r);
        r - (0..3).fold(Vec3::zero(), |v, i| v + self.lattice.a[i] * f[i].floor())
    }
    fn enumerate(
        &self,
        vectors: [Vec3; 3],
        dual: [Vec3; 3],
        cutoff: f64,
        include_zero: bool,
    ) -> Result<Vec<Vec3>> {
        if !cutoff.is_finite() || cutoff < 0.0 {
            return Err(DftError::InvalidInput(
                "periodic enumeration cutoff must be finite and non-negative".into(),
            ));
        }
        let mut bounds = [0i64; 3];
        let mut size = 1usize;
        for i in 0..3 {
            if self.boundary.axes[i] {
                let b = (cutoff * dual[i].norm() / (2.0 * std::f64::consts::PI)).ceil();
                if b > 1_000_000.0 {
                    return Err(DftError::InvalidInput("periodic enumeration is too large; reduce the cutoff or use a better conditioned basis".into()));
                }
                bounds[i] = b as i64 + 1;
                size = size
                    .checked_mul((2 * bounds[i] + 1) as usize)
                    .ok_or_else(|| {
                        DftError::InvalidInput("periodic enumeration size overflow".into())
                    })?;
            }
        }
        if size > 20_000_000 {
            return Err(DftError::InvalidInput(
                "periodic enumeration exceeds the direct-sum work limit".into(),
            ));
        }
        let mut out = Vec::new();
        for i in -bounds[0]..=bounds[0] {
            for j in -bounds[1]..=bounds[1] {
                for k in -bounds[2]..=bounds[2] {
                    let r =
                        vectors[0] * (i as f64) + vectors[1] * (j as f64) + vectors[2] * (k as f64);
                    if r.norm() <= cutoff && (include_zero || i != 0 || j != 0 || k != 0) {
                        out.push(r);
                    }
                }
            }
        }
        out.sort_by(|a, b| {
            a.norm2()
                .total_cmp(&b.norm2())
                .then_with(|| a.x.total_cmp(&b.x))
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| a.z.total_cmp(&b.z))
        });
        Ok(out)
    }
    pub fn translations_within(&self, cutoff: f64) -> Result<Vec<Vec3>> {
        self.enumerate(self.lattice.a, self.reciprocal, cutoff, true)
    }
    pub fn reciprocal_within(&self, cutoff: f64) -> Result<Vec<Vec3>> {
        let dual = std::array::from_fn(|i| {
            if self.boundary.axes[i] {
                self.lattice.a[i]
            } else {
                Vec3::zero()
            }
        });
        self.enumerate(self.reciprocal, dual, cutoff, false)
    }
    /// Integer images satisfying |displacement + A n| <= cutoff.
    /// Bounds use the periodic subspace dual and exclude every open-axis image.
    /// A local integer origin avoids work proportional to atom cell labels.
    pub fn nearby_translation_indices(
        &self,
        displacement: Vec3,
        cutoff: f64,
    ) -> Result<Vec<[i64; 3]>> {
        if !displacement.norm().is_finite() || !cutoff.is_finite() || cutoff < 0.0 {
            return Err(DftError::InvalidInput(
                "image displacement and non-negative cutoff must be finite".into(),
            ));
        }
        let f = self.fractional(displacement);
        let mut origin = [0i64; 3];
        for i in 0..3 {
            if self.boundary.axes[i] {
                if !f[i].is_finite() || f[i].abs() > 1e15 {
                    return Err(DftError::InvalidInput(
                        "image cell label exceeds the exact integer range".into(),
                    ));
                }
                origin[i] = -(f[i].round() as i64);
            }
        }
        let residual = displacement + self.lattice.translation(origin);
        let local = self.fractional(residual);
        let radius = cutoff + 64.0 * f64::EPSILON * (1.0 + cutoff);
        let mut lower = [0i64; 3];
        let mut upper = [0i64; 3];
        let mut size = 1usize;
        for i in 0..3 {
            if self.boundary.axes[i] {
                let bound = radius * self.reciprocal[i].norm() / (2.0 * std::f64::consts::PI)
                    + 64.0 * f64::EPSILON;
                if !bound.is_finite() || bound > 1_000_000.0 {
                    return Err(DftError::InvalidInput(
                        "image enumeration bound is too large".into(),
                    ));
                }
                lower[i] = (-local[i] - bound).ceil() as i64;
                upper[i] = (-local[i] + bound).floor() as i64;
                if upper[i] < lower[i] {
                    return Ok(Vec::new());
                }
                size = size
                    .checked_mul((upper[i] - lower[i] + 1) as usize)
                    .ok_or_else(|| {
                        DftError::InvalidInput("image enumeration size overflow".into())
                    })?;
            }
        }
        if size > 20_000_000 {
            return Err(DftError::InvalidInput(
                "image enumeration exceeds the direct-sum work limit".into(),
            ));
        }
        let mut out = Vec::new();
        for i in lower[0]..=upper[0] {
            for j in lower[1]..=upper[1] {
                for k in lower[2]..=upper[2] {
                    if (residual + self.lattice.translation([i, j, k])).norm() <= radius {
                        out.push([origin[0] + i, origin[1] + j, origin[2] + k]);
                    }
                }
            }
        }
        Ok(out)
    }
    /// Exact closest periodic image, including skew or unreduced bases.
    pub fn minimum_image(&self, r: Vec3) -> Result<Vec3> {
        if !r.norm().is_finite() {
            return Err(DftError::InvalidInput(
                "minimum-image displacement must be finite".into(),
            ));
        }
        let f = self.fractional(r);
        let mut nearest = Vec3::zero();
        for i in 0..3 {
            nearest += self.lattice.a[i] * f[i].round();
        }
        let mut best = r - nearest;
        let radius = self.parallel(best).norm();
        // Every improving translation lies within 2*radius of this seed.
        for t in self.translations_within(2.0 * radius + 1e-12)? {
            let trial = r - nearest - t;
            if trial.norm2() < best.norm2() {
                best = trial;
            }
        }
        Ok(best)
    }
    /// Uniform unshifted mesh, with one point on every open axis.
    pub fn k_points(&self, mesh: [usize; 3]) -> Result<Vec<Vec3>> {
        self.boundary.validate_k_mesh(mesh)?;
        let size = mesh
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| DftError::InvalidInput("k mesh size overflow".into()))?;
        if size > 20_000_000 {
            return Err(DftError::InvalidInput(
                "k mesh exceeds the direct-sum work limit".into(),
            ));
        }
        let mut out = Vec::with_capacity(size);
        for i in 0..mesh[0] {
            for j in 0..mesh[1] {
                for k in 0..mesh[2] {
                    // Preserve the existing public unshifted [0,1) order.
                    let f = |n: usize, m: usize| n as f64 / m as f64;
                    out.push(
                        self.reciprocal[0] * f(i, mesh[0])
                            + self.reciprocal[1] * f(j, mesh[1])
                            + self.reciprocal[2] * f(k, mesh[2]),
                    );
                }
            }
        }
        Ok(out)
    }
}
