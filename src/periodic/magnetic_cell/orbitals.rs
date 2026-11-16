// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Pointwise London AO values and continuum kinetic momentum.

use super::{phase, BlochMatrix, MagneticCell};
use crate::basis::aos::{cart_to_sph, MAX_L};
use crate::basis::{cart_components, MolBasis};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::{DftError, Result};
use rayon::prelude::*;

impl MagneticCell {
    /// Magnetic Bloch orbitals on arbitrary points, for quadrature and raw
    /// boundary and gauge checks. The image cutoff is a physical distance.
    pub fn orbital_values(
        &self,
        basis: &MolBasis,
        points: &[Vec3],
        cutoff: f64,
    ) -> Result<BlochMatrix> {
        self.orbital_component(basis, points, cutoff, None)
    }

    /// The continuum kinetic momentum (p+A) acting on each magnetic Bloch
    /// orbital. Spatial differentiation includes the London phase exactly.
    pub fn orbital_momentum_values(
        &self,
        basis: &MolBasis,
        points: &[Vec3],
        cutoff: f64,
        axis: usize,
    ) -> Result<BlochMatrix> {
        if axis >= 3 {
            return Err(DftError::InvalidInput(
                "kinetic momentum axis must be 0..2".into(),
            ));
        }
        self.orbital_component(basis, points, cutoff, Some(axis))
    }

    /// Collocate the London AOs and all three Cartesian components of
    /// (p+A) in one pass. Points and the image cutoff are in Bohr. Each
    /// matrix has one row per point and one column per spherical AO.
    pub fn orbital_values_with_momentum(
        &self,
        basis: &MolBasis,
        points: &[Vec3],
        cutoff: f64,
    ) -> Result<(BlochMatrix, [BlochMatrix; 3])> {
        let grid = LondonGrid::new(self, basis, points, cutoff)?;
        let n = basis.nao_sph;
        let [mut values, mut x, mut y, mut z] =
            std::array::from_fn(|_| BlochMatrix::zeros(points.len(), n));
        (
            matrix_rows(&mut values, n),
            matrix_rows(&mut x, n),
            matrix_rows(&mut y, n),
            matrix_rows(&mut z, n),
        )
            .into_par_iter()
            .zip(points.par_iter())
            .try_for_each_init(Vec::new, |scratch, ((values, x, y, z), &point)| {
                grid.fill_row(
                    point,
                    [None, Some(0), Some(1), Some(2)],
                    scratch,
                    [values, x, y, z],
                )
            })?;
        Ok((values, [x, y, z]))
    }

    fn orbital_component(
        &self,
        basis: &MolBasis,
        points: &[Vec3],
        cutoff: f64,
        axis: Option<usize>,
    ) -> Result<BlochMatrix> {
        let grid = LondonGrid::new(self, basis, points, cutoff)?;
        let mut out = BlochMatrix::zeros(points.len(), basis.nao_sph);
        matrix_rows(&mut out, basis.nao_sph)
            .zip(points.par_iter())
            .try_for_each_init(Vec::new, |scratch, (rows, &point)| {
                grid.fill_row(point, [axis], scratch, [rows])
            })?;
        Ok(out)
    }
}

fn matrix_rows(
    matrix: &mut BlochMatrix,
    n: usize,
) -> impl IndexedParallelIterator<Item = (&mut [f64], &mut [f64])> {
    matrix
        .re
        .as_mut_slice()
        .par_chunks_mut(n)
        .zip(matrix.im.as_mut_slice().par_chunks_mut(n))
}

struct ShellTransform {
    angular: Vec<[usize; 3]>,
    spherical: Matrix,
}

struct LondonGrid<'a> {
    cell: &'a MagneticCell,
    basis: &'a MolBasis,
    cutoff: f64,
    transforms: Vec<ShellTransform>,
}

impl<'a> LondonGrid<'a> {
    fn new(
        cell: &'a MagneticCell,
        basis: &'a MolBasis,
        points: &[Vec3],
        cutoff: f64,
    ) -> Result<Self> {
        if basis.nao_sph == 0
            || !cutoff.is_finite()
            || cutoff <= 0.0
            || points.iter().any(|p| !p.norm().is_finite())
        {
            return Err(DftError::InvalidInput(
                "magnetic orbitals require nonempty AOs, finite points and a positive cutoff"
                    .into(),
            ));
        }
        let transforms = basis
            .shells
            .iter()
            .map(|shell| ShellTransform {
                angular: cart_components(shell.l)
                    .into_iter()
                    .map(|(x, y, z)| [x as usize, y as usize, z as usize])
                    .collect(),
                spherical: cart_to_sph(shell.l).transpose(),
            })
            .collect();
        Ok(Self {
            cell,
            basis,
            cutoff,
            transforms,
        })
    }

    fn fill_row<const N: usize>(
        &self,
        point: Vec3,
        axes: [Option<usize>; N],
        cartesian: &mut Vec<[Complex; N]>,
        mut rows: [(&mut [f64], &mut [f64]); N],
    ) -> Result<()> {
        let needs_momentum = axes.iter().any(Option::is_some);
        for (shell, transform) in self.basis.shells.iter().zip(&self.transforms) {
            cartesian.resize(transform.angular.len(), [Complex::ZERO; N]);
            for index in self
                .cell
                .geometry
                .nearby_translation_indices(shell.center - point, self.cutoff)?
            {
                let center = shell.center + self.cell.geometry.lattice().translation(index);
                let delta = point - center;
                let r2 = delta.norm2();
                let mut radial = 0.0;
                let mut radial_slope = 0.0;
                for (&exponent, &coefficient) in shell.exps.iter().zip(&shell.coefs) {
                    let primitive = coefficient * (-exponent * r2).exp();
                    radial += primitive;
                    if needs_momentum {
                        radial_slope += 2.0 * exponent * primitive;
                    }
                }
                let mut powers = [[1.0; MAX_L as usize + 1]; 3];
                for (axis, power) in powers.iter_mut().enumerate() {
                    for l in 1..=shell.l as usize {
                        power[l] = power[l - 1] * delta.get(axis);
                    }
                }
                // The London phase combines with the vector potential to
                // give A(r) + k_L = B × (r - center) / 2.
                let vector_potential = if needs_momentum {
                    self.cell.field.cross(delta) * 0.5
                } else {
                    Vec3::zero()
                };
                let monomial = |[x, y, z]: [usize; 3]| powers[0][x] * powers[1][y] * powers[2][z];
                for (&angular, channels) in transform.angular.iter().zip(cartesian.iter_mut()) {
                    let polynomial = monomial(angular);
                    let value = radial * polynomial;
                    for (channel, axis) in channels.iter_mut().zip(axes) {
                        *channel = if let Some(axis) = axis {
                            // Differentiate the angular monomial and Gaussian
                            // radial contraction before applying -i ∇.
                            let mut gradient = -delta.get(axis) * radial_slope * polynomial;
                            if angular[axis] != 0 {
                                let mut lower = angular;
                                lower[axis] -= 1;
                                gradient += angular[axis] as f64 * radial * monomial(lower);
                            }
                            Complex::new(vector_potential.get(axis) * value, -gradient)
                        } else {
                            Complex::from(value)
                        };
                    }
                }
                let coefficient = self.cell.image_phase(shell.center, index)
                    * phase(self.cell.london_wave(center).dot(point));
                for i in 0..transform.spherical.rows {
                    let mut spherical = [Complex::ZERO; N];
                    for (cart, values) in cartesian.iter().enumerate() {
                        let weight = transform.spherical[(i, cart)];
                        for (sum, &value) in spherical.iter_mut().zip(values) {
                            *sum += value * weight;
                        }
                    }
                    let ao = shell.sph_off + i;
                    for ((real, imag), value) in rows.iter_mut().zip(spherical) {
                        let contribution = coefficient * value;
                        real[ao] += contribution.re;
                        imag[ao] += contribution.im;
                    }
                }
            }
        }
        Ok(())
    }
}
