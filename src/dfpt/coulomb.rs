// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic Gaussian plane-wave matrix elements between `(k+q,k)` spaces
//! and the three-dimensional reciprocal Hartree response. Reciprocal and
//! localized AO cutoffs are explicit convergence parameters.

use crate::basis::n_sph;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::integrals::nearby_translation_indices;
use crate::periodic::planewave::plane_wave_shell_pair;
use crate::periodic::Cell;
use crate::response::complex::Complex;
use crate::{DftError, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;

fn finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// `<Phi_mu(k+q)|exp(i Q.r)|Phi_nu(k)>`, with `Q=q+G` and reciprocal G.
/// Translating both centres cancels the Bloch/plane-wave phase. The remaining
/// all-space image integral has the ket phase `exp(i k.T)`.
pub fn plane_wave_pair(
    cell: &Cell,
    k: Vec3,
    q: Vec3,
    wave: Vec3,
    ao_cutoff: f64,
) -> Result<BlochMatrix> {
    if !finite(k) || !finite(q) || !finite(wave) || !ao_cutoff.is_finite() || ao_cutoff <= 0.0 {
        return Err(DftError::InvalidInput(
            "invalid finite-q plane-wave geometry or AO cutoff".into(),
        ));
    }
    // Reciprocal coordinates are Q.a_i/(2*pi), rather than the real-space
    // fractional-coordinate map.
    if cell.lattice.a.iter().any(|a| {
        let x = (wave - q).dot(*a) / (2.0 * std::f64::consts::PI);
        (x - x.round()).abs() > 1e-10 * (1.0 + x.abs())
    }) {
        return Err(DftError::InvalidInput(
            "plane-wave transfer must equal q plus a reciprocal lattice vector".into(),
        ));
    }
    let mut images: BTreeMap<[i64; 3], Vec<(usize, usize)>> = BTreeMap::new();
    for (a, left) in cell.basis.shells.iter().enumerate() {
        for (b, right) in cell.basis.shells.iter().enumerate() {
            for index in
                nearby_translation_indices(&cell.lattice, right.center - left.center, ao_cutoff)
            {
                images.entry(index).or_default().push((a, b));
            }
        }
    }
    let nao = cell.nao();
    let mut output = BlochMatrix::zeros(nao, nao);
    for (index, pairs) in images {
        let translation = cell.lattice.translation(index);
        let phase = Complex::new(k.dot(translation).cos(), k.dot(translation).sin());
        for (a, b) in pairs {
            let left = &cell.basis.shells[a];
            let right = &cell.basis.shells[b];
            let mut image = right.clone();
            image.center += translation;
            let (re, im) = plane_wave_shell_pair(left, &image, wave);
            for i in 0..n_sph(left.l) {
                for j in 0..n_sph(right.l) {
                    let (mu, nu) = (left.sph_off + i, right.sph_off + j);
                    let value = Complex::new(re[(i, j)], im[(i, j)]) * phase;
                    output.re[(mu, nu)] += value.re;
                    output.im[(mu, nu)] += value.im;
                }
            }
        }
    }
    Ok(output)
}

/// Complete spherical reciprocal set satisfying `|G+q| <= cutoff`.
/// The zero mode is omitted as the constant-potential gauge of a neutral
/// three-dimensional cell. At nonzero q no long-wavelength mode is dropped.
pub fn reciprocal_waves(cell: &Cell, q: Vec3, cutoff: f64) -> Result<Vec<Vec3>> {
    if !finite(q) || !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(DftError::InvalidInput(
            "reciprocal response cutoff must be finite and positive".into(),
        ));
    }
    let reciprocal = cell.lattice.reciprocal();
    let bounds: [(i64, i64); 3] = std::array::from_fn(|i| {
        let shift = q.dot(cell.lattice.a[i]) / (2.0 * std::f64::consts::PI);
        let radius = cutoff * cell.lattice.a[i].norm() / (2.0 * std::f64::consts::PI);
        (
            (-radius - shift - 1e-12).ceil() as i64,
            (radius - shift + 1e-12).floor() as i64,
        )
    });
    let mut waves = Vec::new();
    for a in bounds[0].0..=bounds[0].1 {
        for b in bounds[1].0..=bounds[1].1 {
            for c in bounds[2].0..=bounds[2].1 {
                let wave = reciprocal[0] * (a as f64)
                    + reciprocal[1] * (b as f64)
                    + reciprocal[2] * (c as f64)
                    + q;
                if wave.norm2() != 0.0 && wave.norm() <= cutoff * (1.0 + 1e-12) {
                    waves.push(wave);
                }
            }
        }
    }
    waves.sort_by(|a, b| {
        a.norm2()
            .total_cmp(&b.norm2())
            .then_with(|| a.x.total_cmp(&b.x))
            .then_with(|| a.y.total_cmp(&b.y))
            .then_with(|| a.z.total_cmp(&b.z))
    });
    if waves.is_empty() {
        return Err(DftError::InvalidInput(
            "reciprocal response set contains no nonzero modes".into(),
        ));
    }
    Ok(waves)
}

/// Cached density Fourier forms and induced Hartree potentials. Density
/// coefficients are per unit cell volume and include both spin channels.
pub struct FiniteQHartree {
    pub waves: Vec<Vec3>,
    forms: Vec<Vec<BlochMatrix>>,
    volume: f64,
    nk: usize,
    nao: usize,
}

impl FiniteQHartree {
    pub fn new(
        cell: &Cell,
        k_points: &[Vec3],
        q: Vec3,
        reciprocal_cutoff: f64,
        ao_cutoff: f64,
    ) -> Result<Self> {
        if k_points.is_empty()
            || k_points.iter().any(|&k| !finite(k))
            || !cell.lattice.volume().is_finite()
            || cell.lattice.volume().abs() == 0.0
        {
            return Err(DftError::InvalidInput(
                "finite-q Hartree requires a finite Bloch mesh and nonzero cell volume".into(),
            ));
        }
        let waves = reciprocal_waves(cell, q, reciprocal_cutoff)?;
        let forms = waves
            .par_iter()
            .map(|&g| {
                k_points
                    .iter()
                    .map(|&k| plane_wave_pair(cell, k, q, g, ao_cutoff))
                    .collect()
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            waves,
            forms,
            volume: cell.lattice.volume().abs(),
            nk: k_points.len(),
            nao: cell.nao(),
        })
    }
    pub fn forms(&self, mode: usize) -> Result<&[BlochMatrix]> {
        self.forms
            .get(mode)
            .map(|v| v.as_slice())
            .ok_or_else(|| DftError::InvalidInput("reciprocal mode index out of range".into()))
    }
    pub fn density_fourier(&self, density: &[Vec<BlochMatrix>; 2]) -> Result<Vec<Complex>> {
        if density.iter().any(|s| {
            s.len() != self.nk
                || s.iter().any(|d| {
                    d.rows() != self.nao
                        || d.cols() != self.nao
                        || d.im.rows != self.nao
                        || d.im.cols != self.nao
                        || d.re
                            .as_slice()
                            .iter()
                            .chain(d.im.as_slice())
                            .any(|v| !v.is_finite())
                })
        }) {
            return Err(DftError::InvalidInput(
                "invalid finite-q Hartree AO density".into(),
            ));
        }
        let scale = 1.0 / (self.volume * self.nk as f64);
        Ok(self
            .forms
            .iter()
            .map(|mesh| {
                let mut sum = Complex::ZERO;
                for spin in density {
                    for (form, d) in mesh.iter().zip(spin) {
                        for i in 0..self.nao {
                            for j in 0..self.nao {
                                sum += Complex::new(form.re[(i, j)], -form.im[(i, j)])
                                    * Complex::new(d.re[(i, j)], d.im[(i, j)]);
                            }
                        }
                    }
                }
                sum * scale
            })
            .collect())
    }
    pub fn apply(&self, density: &[Vec<BlochMatrix>; 2]) -> Result<[Vec<BlochMatrix>; 2]> {
        let rho = self.density_fourier(density)?;
        let mut potential: Vec<_> = (0..self.nk)
            .map(|_| BlochMatrix::zeros(self.nao, self.nao))
            .collect();
        for (mode, (coefficient, mesh)) in rho.iter().zip(&self.forms).enumerate() {
            let amplitude = *coefficient * (4.0 * std::f64::consts::PI / self.waves[mode].norm2());
            for (output, form) in potential.iter_mut().zip(mesh) {
                for i in 0..self.nao {
                    for j in 0..self.nao {
                        let term = Complex::new(form.re[(i, j)], form.im[(i, j)]) * amplitude;
                        output.re[(i, j)] += term.re;
                        output.im[(i, j)] += term.im;
                    }
                }
            }
        }
        Ok([potential.clone(), potential])
    }
}
