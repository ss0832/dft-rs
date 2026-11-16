// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-displacement lattice response of the dimensional Gaussian Green gauge.
use crate::integrals::analytic::boys_values;
use crate::integrals::hermite::RTensor;
use crate::math::Vec3;
use crate::periodic::electrostatics::DimensionalCoulomb;
use crate::resources::{checked_bytes, reserve, MemoryReservation};
use crate::response::complex::Complex;
use crate::{DftError, Result};
use std::f64::consts::PI;

pub(in crate::periodic) mod lower;

pub struct CellGreenResponse {
    order: usize,
    dim: usize,
    values: Vec<[[f64; 3]; 9]>,
    _output_memory: MemoryReservation,
}
impl CellGreenResponse {
    pub fn components(&self, m: [usize; 3]) -> Result<[[f64; 3]; 9]> {
        if m.iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .is_none_or(|n| n > self.order)
        {
            return Err(DftError::InvalidInput(
                "Gaussian lattice derivative index exceeds its order".into(),
            ));
        }
        Ok(self.values[(m[0] * self.dim + m[1]) * self.dim + m[2]])
    }
}
fn indices(order: usize) -> impl Iterator<Item = [usize; 3]> {
    (0..=order).flat_map(move |x| {
        (0..=order - x).flat_map(move |y| (0..=order - x - y).map(move |z| [x, y, z]))
    })
}
fn plane(m: [usize; 3], g: [f64; 3]) -> Complex {
    let mut value = Complex::ONE;
    for axis in 0..3 {
        for _ in 0..m[axis] {
            value = value * Complex::new(0., g[axis]);
        }
    }
    value
}

/// `reciprocal` is the unchanged canonical nonzero list of this kernel.
/// This internal primitive holds r, width, alpha and cutoff memberships fixed;
/// moving AO/nuclear centres and Pulay terms are separate strain contributions.
pub fn fixed_displacement_cell_response(
    kernel: &DimensionalCoulomb,
    reciprocal: &[Vec3],
    r: Vec3,
    width: f64,
    order: usize,
) -> Result<CellGreenResponse> {
    if !r.norm().is_finite() || !width.is_finite() || width <= 0. {
        return Err(DftError::InvalidInput(
            "Gaussian lattice response requires finite r and positive width".into(),
        ));
    }
    let dim = order.checked_add(1).ok_or_else(|| {
        DftError::InvalidInput("Gaussian lattice derivative order overflow".into())
    })?;
    let size = dim.checked_pow(3).ok_or_else(|| {
        DftError::InvalidInput("Gaussian lattice derivative size overflow".into())
    })?;
    let entries = size.checked_mul(27).ok_or_else(|| {
        DftError::InvalidInput("Gaussian lattice derivative work overflow".into())
    })?;
    if entries > 1_000_000 {
        return Err(DftError::InvalidInput(
            "Gaussian lattice derivative exceeds direct work limit".into(),
        ));
    }
    let memory = reserve(
        checked_bytes(&[entries, 8], "Gaussian lattice output")?,
        "Gaussian lattice output",
    )?;
    let scratch_dim = dim + 1;
    let _scratch_memory = reserve(
        checked_bytes(
            &[2, scratch_dim, scratch_dim, scratch_dim, scratch_dim, 8],
            "Gaussian lattice Hermite scratch",
        )?,
        "Gaussian lattice Hermite scratch",
    )?;
    let mut output = CellGreenResponse {
        order,
        dim,
        values: vec![[[0.; 3]; 9]; size],
        _output_memory: memory,
    };
    let index = |m: [usize; 3]| (m[0] * dim + m[1]) * dim + m[2];
    let geometry = kernel.geometry();
    let params = kernel.params();
    let alpha = params.alpha;
    let inverse = width.recip();
    if alpha != inverse {
        for image in geometry.nearby_translation_indices(r, params.rcut)? {
            let translation = geometry.lattice().translation(image);
            let delta = r + translation;
            if delta.norm() == params.rcut {
                return Err(DftError::InvalidInput(
                    "Gaussian affine derivative is undefined at an exact hard real cutoff".into(),
                ));
            }

            let ra = RTensor::build(
                order + 1,
                inverse * inverse,
                delta,
                &boys_values(order + 1, inverse * inverse * delta.norm2())?,
            );
            let rb = RTensor::build(
                order + 1,
                alpha * alpha,
                delta,
                &boys_values(order + 1, alpha * alpha * delta.norm2())?,
            );
            let t = translation.to_array();
            for m in indices(order) {
                for a in 0..3 {
                    let mut raised = m;
                    raised[a] += 1;
                    let gradient = 2. / PI.sqrt()
                        * (inverse * ra.get(raised[0], raised[1], raised[2])
                            - alpha * rb.get(raised[0], raised[1], raised[2]));
                    for b in 0..3 {
                        output.values[index(m)][3 * a + b][0] += gradient * t[b];
                    }
                }
            }
        }
    }
    if geometry.dimension() != 3 {
        check_reciprocal_membership(reciprocal, params.gcut)?;
        lower::reciprocal_response(kernel, reciprocal, r, order, dim, &mut output.values);
        return Ok(output);
    }
    let position = r.to_array();
    let measure = geometry.measure();
    for &wave in reciprocal {
        let square = wave.norm2();
        if !square.is_finite() || square <= 0. {
            return Err(DftError::InvalidInput(
                "Gaussian lattice response needs finite nonzero reciprocal vectors".into(),
            ));
        }
        if wave.norm() == params.gcut {
            return Err(DftError::InvalidInput(
                "Gaussian affine derivative is undefined at an exact hard reciprocal cutoff".into(),
            ));
        }
        let g = wave.to_array();
        let phase = wave.dot(r);
        let trig = Complex::new(phase.cos(), phase.sin());
        let scale = 4. * PI / measure * (-square / (4. * alpha * alpha)).exp() / square;
        for m in indices(order) {
            let polynomial = plane(m, g);
            for a in 0..3 {
                for b in 0..3 {
                    let log_derivative = -f64::from(a == b)
                        + g[a] * g[b] * (2. / square + 1. / (2. * alpha * alpha));
                    let mut polynomial_derivative = Complex::ZERO;
                    if m[b] > 0 {
                        let mut lower = m;
                        lower[b] -= 1;
                        polynomial_derivative =
                            plane(lower, g) * Complex::new(0., -g[a] * m[b] as f64);
                    }
                    let phase_derivative = polynomial * Complex::new(0., -g[a] * position[b]);
                    output.values[index(m)][3 * a + b][1] += scale
                        * (trig
                            * (polynomial * log_derivative
                                + polynomial_derivative
                                + phase_derivative))
                            .re;
                }
            }
        }
    }
    let zero = -PI / (measure * alpha * alpha) + PI * width * width / measure;
    for a in 0..3 {
        output.values[0][3 * a + a][2] = -zero;
    }
    Ok(output)
}

pub(in crate::periodic) fn check_reciprocal_membership(
    reciprocal: &[Vec3],
    cutoff: f64,
) -> Result<()> {
    for &wave in reciprocal {
        let square = wave.norm2();
        if !square.is_finite() || square <= 0.0 {
            return Err(DftError::InvalidInput(
                "cell response needs finite nonzero reciprocal vectors".into(),
            ));
        }
        if wave.norm() == cutoff {
            return Err(DftError::InvalidInput(
                "affine derivative is undefined at an exact hard reciprocal cutoff".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod cutoff_boundary_tests {
    use super::*;
    use crate::periodic::ewald::EwaldParams;
    use crate::periodic::features::PeriodicBoundary;
    use crate::periodic::{Lattice, PeriodicGeometry};
    fn kernel(real: f64, reciprocal: f64) -> DimensionalCoulomb {
        let geometry = PeriodicGeometry::new(
            Lattice::new(
                Vec3::new(6.1, 0.21, 0.17),
                Vec3::new(0.13, 6.3, 0.19),
                Vec3::new(0.23, 0.09, 6.5),
            ),
            PeriodicBoundary::FULL_3D,
        )
        .unwrap();
        DimensionalCoulomb::new(
            geometry,
            EwaldParams {
                alpha: 0.37,
                rcut: real,
                gcut: reciprocal,
            },
            16,
        )
        .unwrap()
    }
    #[test]
    fn exact_real_and_reciprocal_cutoff_derivatives_are_rejected() {
        let k = kernel(0.25, 1.22);
        let r = Vec3::new(0.25, 0.0, 0.0);
        assert!(fixed_displacement_cell_response(&k, &k.reciprocal, r, 0.71, 0).is_err());
        let generous = kernel(7.6, 1.22);
        let wave = generous.reciprocal[0];
        let k = kernel(7.6, wave.norm());
        assert!(fixed_displacement_cell_response(
            &k,
            &[wave],
            Vec3::new(0.13, -0.17, 0.11),
            0.71,
            0
        )
        .is_err());
    }
}
