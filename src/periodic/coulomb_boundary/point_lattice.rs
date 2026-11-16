// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Private same-gauge dimensional point Green lattice partials.
use crate::math::Vec3;
use crate::periodic::electrostatics::DimensionalCoulomb;
use crate::periodic::gaussian_coulomb::lattice_green::{check_reciprocal_membership, lower};
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Hold physical displacement fixed while differentiating the lattice.
/// The caller supplies this kernel's canonical reciprocal list, unchanged.
/// The bare point self constant has zero lattice derivative at fixed alpha.
pub fn fixed_point_cell_response(
    kernel: &DimensionalCoulomb,
    reciprocal: &[Vec3],
    r: Vec3,
    exclude_home: bool,
) -> Result<[[f64; 3]; 9]> {
    if !r.norm().is_finite() || (exclude_home && r.norm2() != 0.0) {
        return Err(DftError::InvalidInput(
            "point self exclusion requires zero displacement; all inputs must be finite".into(),
        ));
    }
    let mut output = [[0.0; 3]; 9];
    let geometry = kernel.geometry();
    let params = kernel.params();
    let alpha = params.alpha;
    for image in geometry.nearby_translation_indices(r, params.rcut)? {
        if exclude_home && image == [0; 3] {
            continue;
        }
        let translation = geometry.lattice().translation(image);
        let delta = r + translation;
        let square = delta.norm2();
        if square == 0.0 {
            return Err(DftError::InvalidInput(
                "singular point pair without self exclusion".into(),
            ));
        }
        let distance = square.sqrt();
        if distance == params.rcut {
            return Err(DftError::InvalidInput(
                "point affine derivative is undefined at an exact hard real cutoff".into(),
            ));
        }

        let radial = -(libm::erfc(alpha * distance) / (square * distance)
            + 2.0 * alpha / PI.sqrt() * (-alpha * alpha * square).exp() / square);
        let t = translation.to_array();
        let d = delta.to_array();
        for a in 0..3 {
            for b in 0..3 {
                output[3 * a + b][0] += radial * d[a] * t[b];
            }
        }
    }
    if geometry.dimension() != 3 {
        check_reciprocal_membership(reciprocal, params.gcut)?;
        lower::reciprocal_response(
            kernel,
            reciprocal,
            r,
            0,
            1,
            std::slice::from_mut(&mut output),
        );
        return Ok(output);
    }
    let position = r.to_array();
    for &wave in reciprocal {
        let square = wave.norm2();
        if !square.is_finite() || square <= 0.0 {
            return Err(DftError::InvalidInput(
                "nonfinite or zero reciprocal vector".into(),
            ));
        }
        if wave.norm() == params.gcut {
            return Err(DftError::InvalidInput(
                "point affine derivative is undefined at an exact hard reciprocal cutoff".into(),
            ));
        }
        let g = wave.to_array();
        let phase = wave.dot(r);
        let scale =
            4.0 * PI / geometry.measure() * (-square / (4.0 * alpha * alpha)).exp() / square;
        for a in 0..3 {
            for b in 0..3 {
                let log_derivative =
                    -f64::from(a == b) + g[a] * g[b] * (2.0 / square + 1.0 / (2.0 * alpha * alpha));
                output[3 * a + b][1] +=
                    scale * (phase.cos() * log_derivative + phase.sin() * g[a] * position[b]);
            }
        }
    }
    for a in 0..3 {
        output[3 * a + a][2] = PI / (geometry.measure() * alpha * alpha);
    }
    Ok(output)
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
        assert!(fixed_point_cell_response(&k, &k.reciprocal, r, false).is_err());
        let generous = kernel(7.6, 1.22);
        let wave = generous.reciprocal[0];
        let k = kernel(7.6, wave.norm());
        assert!(
            fixed_point_cell_response(&k, &[wave], Vec3::new(0.13, -0.17, 0.11), false).is_err()
        );
    }
}
