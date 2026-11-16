// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Continuum uniform magnetic fields with a periodic linear gauge.
//! A wire admits every field direction. A slab admits fields in its plane
//! with ordinary Bloch translations; normal flux needs magnetic translations.
//! The gauge-origin change is a common unitary AO phase, not a Peierls model.

use super::magnetic_scf::StaticFieldOperators;
use super::{BlochMatrix, PeriodicGeometry};
use crate::basis::{aos::sph_block, cart_components, MolBasis, MolShell};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::{DftError, Result};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct UniformOpenField {
    /// Curl A in reduced atomic units; the applied field is amplitude times this.
    pub magnetic_field: Vec3,
    /// Bohr. The basis includes exp(i amplitude (M origin).r).
    pub origin: Vec3,
    pub amplitude: f64,
}

fn apply(m: &[[f64; 3]; 3], r: Vec3) -> Vec3 {
    Vec3::new(
        (0..3).map(|j| m[0][j] * r.get(j)).sum(),
        (0..3).map(|j| m[1][j] * r.get(j)).sum(),
        (0..3).map(|j| m[2][j] * r.get(j)).sum(),
    )
}

impl UniformOpenField {
    /// A=M(r-origin), with M t=0 on every periodic translation and div A=0.
    /// No component of the supplied field is silently projected away.
    pub fn gauge_matrix(&self, geometry: &PeriodicGeometry) -> Result<[[f64; 3]; 3]> {
        if !self.magnetic_field.norm().is_finite()
            || !self.origin.norm().is_finite()
            || !self.amplitude.is_finite()
        {
            return Err(DftError::InvalidInput(
                "uniform open-boundary field requires finite field, origin and amplitude".into(),
            ));
        }
        let axes: Vec<_> = (0..3).filter(|&a| geometry.boundary().axes[a]).collect();
        let b = self.magnetic_field;
        let matrix = match *axes.as_slice() {
            [a] => {
                let t = geometry.lattice().a[a] / geometry.lattice().a[a].norm();
                let transverse = t.cross(b);
                let parallel = t * t.dot(b);
                std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        let e = match j {
                            0 => Vec3::new(1.0, 0.0, 0.0),
                            1 => Vec3::new(0.0, 1.0, 0.0),
                            _ => Vec3::new(0.0, 0.0, 1.0),
                        };
                        t.get(i) * transverse.get(j) + 0.5 * parallel.cross(e).get(i)
                    })
                })
            }
            [a, c] => {
                let n = geometry.lattice().a[a].cross(geometry.lattice().a[c]);
                let n = n / n.norm();
                let normal = n.dot(b);
                if normal.abs() > 1e-12 * (1.0 + b.norm()) {
                    return Err(DftError::InvalidInput(
                        "ordinary slab Bloch translations require B.normal=0; nonzero normal \
                        flux requires a flux-compatible magnetic Bloch cell with open-direction \
                        Coulomb boundary"
                            .into(),
                    ));
                }
                let v = b.cross(n);
                std::array::from_fn(|i| std::array::from_fn(|j| v.get(i) * n.get(j)))
            }
            _ => {
                return Err(DftError::InvalidInput(
                    "a periodic linear uniform-field gauge requires a 1D wire or 2D slab; 3D \
                uniform flux uses magnetic Bloch cells"
                        .into(),
                ))
            }
        };
        Ok(matrix)
    }
    /// The common phase wave vector is essential when evaluating real-space AOs.
    pub fn origin_phase_wave(&self, geometry: &PeriodicGeometry) -> Result<Vec3> {
        Ok(apply(&self.gauge_matrix(geometry)?, self.origin) * self.amplitude)
    }
    pub fn vector_potential(&self, geometry: &PeriodicGeometry, r: Vec3) -> Result<Vec3> {
        Ok(apply(&self.gauge_matrix(geometry)?, r - self.origin) * self.amplitude)
    }
    pub fn operators(
        &self,
        basis: &MolBasis,
        geometry: &PeriodicGeometry,
        points: &[Vec3],
        cutoff: f64,
    ) -> Result<StaticFieldOperators> {
        let m = self.gauge_matrix(geometry)?;
        if basis.nao_sph == 0
            || points.is_empty()
            || !cutoff.is_finite()
            || cutoff <= 0.0
            || points.iter().any(|k| {
                !k.norm().is_finite()
                    || geometry.perpendicular(*k).norm() > 1e-12 * (1.0 + k.norm())
            })
        {
            return Err(DftError::InvalidInput("uniform field operators require AOs, physical finite Bloch points and positive cutoff".into()));
        }
        let mut pairs = Vec::new();
        for (i, a) in basis.shells.iter().enumerate() {
            for (j, b) in basis.shells.iter().enumerate() {
                for index in geometry.nearby_translation_indices(b.center - a.center, cutoff)? {
                    pairs.push((i, j, geometry.lattice().translation(index)));
                }
            }
        }
        let blocks: Vec<_> = pairs
            .par_iter()
            .map(|&(i, j, t)| {
                let a = &basis.shells[i];
                let mut b = basis.shells[j].clone();
                b.center += t;
                shell_operators(a, &b, &m)
            })
            .collect();
        let n = basis.nao_sph;
        let mut linear = vec![BlochMatrix::zeros(n, n); points.len()];
        let mut quadratic = linear.clone();
        for ((i, j, t), (li, qu)) in pairs.iter().zip(&blocks) {
            let a = &basis.shells[*i];
            let b = &basis.shells[*j];
            for (ik, k) in points.iter().enumerate() {
                let (s, c) = k.dot(*t).sin_cos();
                for r in 0..li.rows {
                    for col in 0..li.cols {
                        let row = a.sph_off + r;
                        let column = b.sph_off + col;
                        linear[ik].re[(row, column)] -= li[(r, col)] * s;
                        linear[ik].im[(row, column)] += li[(r, col)] * c;
                        quadratic[ik].re[(row, column)] += qu[(r, col)] * c;
                        quadratic[ik].im[(row, column)] += qu[(r, col)] * s;
                    }
                }
            }
        }
        let mut hamiltonian = linear.clone();
        let mut first_derivative = linear;
        for ik in 0..points.len() {
            hamiltonian[ik].scale(self.amplitude);
            hamiltonian[ik].add_scaled(&quadratic[ik], 0.5 * self.amplitude * self.amplitude);
            first_derivative[ik].add_scaled(&quadratic[ik], self.amplitude);
        }
        let raw = hamiltonian
            .iter()
            .chain(&first_derivative)
            .chain(&quadratic)
            .map(BlochMatrix::hermiticity_error)
            .fold(0.0_f64, f64::max);
        if !raw.is_finite()
            || raw > 1e-10
            || hamiltonian
                .iter()
                .chain(&first_derivative)
                .chain(&quadratic)
                .any(|h| {
                    h.re.as_slice()
                        .iter()
                        .chain(h.im.as_slice())
                        .any(|v| !v.is_finite())
                })
        {
            return Err(DftError::InvalidInput(
                "uniform continuum field operator violates Hermiticity before projection \
                or is non-finite; refine the Gaussian image cutoff"
                    .into(),
            ));
        }
        Ok(StaticFieldOperators {
            hamiltonian,
            first_derivative,
            second_derivative: quadratic,
            raw_hermitian_residual: raw,
        })
    }
}

fn derivative(l: [usize; 3], exponent: f64, axis: usize) -> Vec<([usize; 3], f64)> {
    let mut raised = l;
    raised[axis] += 1;
    let mut out = vec![(raised, 2.0 * exponent)];
    if l[axis] > 0 {
        let mut lowered = l;
        lowered[axis] -= 1;
        out.push((lowered, -(l[axis] as f64)));
    }
    out
}
fn potential(
    l: [usize; 3],
    center: Vec3,
    m: &[[f64; 3]; 3],
    axis: usize,
) -> Vec<([usize; 3], f64)> {
    let mut out = vec![(l, apply(m, center).get(axis))];
    for j in 0..3 {
        if m[axis][j] != 0.0 {
            let mut raised = l;
            raised[j] += 1;
            out.push((raised, m[axis][j]));
        }
    }
    out
}
/// Imaginary coefficient of (p.A+A.p)/2, and the real A.A contact.
fn shell_operators(a: &MolShell, b: &MolShell, m: &[[f64; 3]; 3]) -> (Matrix, Matrix) {
    let ca = cart_components(a.l);
    let cb = cart_components(b.l);
    let mut li = Matrix::zeros(ca.len(), cb.len());
    let mut qu = li.clone();
    for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
            for (i, ia) in ca.iter().enumerate() {
                for (j, ib) in cb.iter().enumerate() {
                    let la = [ia.0 as usize, ia.1 as usize, ia.2 as usize];
                    let lb = [ib.0 as usize, ib.1 as usize, ib.2 as usize];
                    let overlap = |x, y| {
                        super::planewave::plane_wave_primitive(
                            x,
                            y,
                            ea,
                            eb,
                            a.center,
                            b.center,
                            Vec3::zero(),
                        )
                        .0
                    };
                    for axis in 0..3 {
                        let da = derivative(la, ea, axis);
                        let db = derivative(lb, eb, axis);
                        let aa = potential(la, a.center, m, axis);
                        let ab = potential(lb, b.center, m, axis);
                        for &(x, c) in &da {
                            for &(y, d) in &ab {
                                li[(i, j)] -= 0.5 * wa * wb * c * d * overlap(x, y);
                            }
                        }
                        for &(x, c) in &aa {
                            for &(y, d) in &db {
                                li[(i, j)] += 0.5 * wa * wb * c * d * overlap(x, y);
                            }
                        }
                        for &(x, c) in &aa {
                            for &(y, d) in &ab {
                                qu[(i, j)] += wa * wb * c * d * overlap(x, y);
                            }
                        }
                    }
                }
            }
        }
    }
    (sph_block(a.l, b.l, &li), sph_block(a.l, b.l, &qu))
}
