// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent complex orbital references in real static periodic fields.
//! A(r)=A_q exp(i q.r)+A_q* exp(-i q.r), in (p+A)^2/2 atomic units.
//! This zero-net-flux field is distinct from a uniform-flux magnetic cell.

use super::{BlochMatrix, PeriodicGeometry};
use crate::basis::MolBasis;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::{DftError, Result};

pub(crate) mod geometry;

#[derive(Clone, Debug)]
pub struct StaticPeriodicField {
    pub wave: Vec3,
    pub vector_potential: [Complex; 3],
    /// Electron potential energy, V(r)=V_q exp(i q.r)+conjugate.
    pub scalar_potential: Complex,
    pub amplitude: f64,
}

pub struct StaticFieldOperators {
    /// Field Hamiltonian lambda H1 + lambda^2 H2/2, including all contact terms.
    pub hamiltonian: Vec<BlochMatrix>,
    pub first_derivative: Vec<BlochMatrix>,
    pub second_derivative: Vec<BlochMatrix>,
    pub raw_hermitian_residual: f64,
}

pub(super) fn add(target: &mut BlochMatrix, source: &BlochMatrix, coefficient: Complex) {
    for i in 0..target.rows() {
        for j in 0..target.cols() {
            let z = Complex::new(source.re[(i, j)], source.im[(i, j)]) * coefficient;
            target.re[(i, j)] += z.re;
            target.im[(i, j)] += z.im;
        }
    }
}

/// Reproducible transverse electric-vector and magnetic-field axes.
pub(crate) fn transverse_frame(wave: Vec3) -> ([Vec3; 2], [Vec3; 2]) {
    let weakest = (0..3)
        .min_by(|a, b| wave.get(*a).abs().total_cmp(&wave.get(*b).abs()))
        .expect("three Cartesian axes");
    let trial = Vec3::new(
        f64::from(weakest == 0),
        f64::from(weakest == 1),
        f64::from(weakest == 2),
    );
    let first = wave.cross(trial);
    let first = first / first.norm();
    let along = wave / wave.norm();
    let second = along.cross(first);
    ([first, second], [second, along.cross(second)])
}

impl StaticPeriodicField {
    fn linear_coefficients(&self, positive: bool) -> (Complex, [Complex; 3]) {
        let mode = |z: Complex| if positive { z } else { z.conj() };
        (mode(self.scalar_potential), self.vector_potential.map(mode))
    }

    fn quadratic_coefficients(&self) -> (Complex, f64) {
        let square = self
            .vector_potential
            .iter()
            .map(|z| *z * *z)
            .fold(Complex::ZERO, |a, b| a + b);
        let norm = self
            .vector_potential
            .iter()
            .map(|z| z.re * z.re + z.im * z.im)
            .sum();
        (square, norm)
    }

    pub fn operators(
        &self,
        basis: &MolBasis,
        geometry: &PeriodicGeometry,
        points: &[Vec3],
        cutoff: f64,
    ) -> Result<StaticFieldOperators> {
        if !self.wave.norm().is_finite()
            || !self.amplitude.is_finite()
            || !self.scalar_potential.is_finite()
            || self.vector_potential.iter().any(|z| !z.is_finite())
            || points.is_empty()
            || points.iter().any(|k| !k.norm().is_finite())
            || !cutoff.is_finite()
            || cutoff <= 0.0
            || basis.nao_sph == 0
        {
            return Err(DftError::InvalidInput("static periodic field requires finite amplitudes, non-empty finite k points, AOs and a positive cutoff".into()));
        }
        for axis in 0..3 {
            if geometry.boundary().axes[axis] {
                let cycles =
                    self.wave.dot(geometry.lattice().a[axis]) / (2.0 * std::f64::consts::PI);
                if (cycles - cycles.round()).abs() > 1e-12 * (1.0 + cycles.abs()) {
                    return Err(DftError::InvalidInput(
                        "a static field in one Bloch cell must be periodic: q.a_i/(2 pi) must be \
                        an integer on every periodic axis; use a compatible supercell"
                            .into(),
                    ));
                }
            }
        }
        let n = basis.nao_sph;
        let mut linear = vec![BlochMatrix::zeros(n, n); points.len()];
        let mut quadratic = linear.clone();
        let (aa, norm) = self.quadratic_coefficients();
        for sign in [1.0, -1.0] {
            let q = self.wave * sign;
            let (density, current) =
                super::field::vertices(basis, geometry, points, q, cutoff, true)?;
            let (contact, _) =
                super::field::vertices(basis, geometry, points, q * 2.0, cutoff, false)?;
            let (scalar, vector) = self.linear_coefficients(sign > 0.0);
            for ik in 0..points.len() {
                add(&mut linear[ik], &density[ik], scalar);
                for (a, z) in vector.into_iter().enumerate() {
                    add(&mut linear[ik], &current[a][ik], z);
                }
                add(
                    &mut quadratic[ik],
                    &contact[ik],
                    if sign > 0.0 { aa } else { aa.conj() },
                );
            }
        }
        let (constant, _) =
            super::field::vertices(basis, geometry, points, Vec3::zero(), cutoff, false)?;
        let mut hamiltonian = Vec::new();
        let mut first_derivative = Vec::new();
        for ik in 0..points.len() {
            add(
                &mut quadratic[ik],
                &constant[ik],
                Complex::new(2.0 * norm, 0.0),
            );
            let mut h = linear[ik].clone();
            h.scale(self.amplitude);
            h.add_scaled(&quadratic[ik], 0.5 * self.amplitude * self.amplitude);
            let mut first = linear[ik].clone();
            first.add_scaled(&quadratic[ik], self.amplitude);
            hamiltonian.push(h);
            first_derivative.push(first);
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
                .any(|m| {
                    m.re.as_slice()
                        .iter()
                        .chain(m.im.as_slice())
                        .any(|x| !x.is_finite())
                })
        {
            return Err(DftError::InvalidInput(format!(
                "static real field Hamiltonian is non-finite or non-Hermitian; raw \
                residual {raw:e}; no symmetry projection was applied"
            )));
        }
        Ok(StaticFieldOperators {
            hamiltonian,
            first_derivative,
            second_derivative: quadratic,
            raw_hermitian_residual: raw,
        })
    }
}
