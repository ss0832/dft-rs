// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic VV10 pair functional. Both pair endpoints belong to the
//! variational density; the partner is summed over the physical periodic
//! subspace. Scalar-generic positions, lattice vectors, weights and density
//! fields carry metric, strain and nuclear derivatives through this kernel.

use super::PeriodicGeometry;
use crate::dual::{DualN, Scalar};
use crate::math::Vec3;
use crate::nlc::{vv10_prepared_pair, Vv10Endpoint, Vv10Params, RHO_CUT};
use crate::{DftError, Result};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct PeriodicVv10Options {
    pub parameters: Vv10Params,
    /// Physical pair distance in Bohr. This is an explicit convergence
    /// parameter of the finite direct lattice sum, not a density cutoff.
    pub cutoff: f64,
}

impl Default for PeriodicVv10Options {
    fn default() -> Self {
        Self {
            parameters: Vv10Params::ORIGINAL,
            cutoff: 40.0,
        }
    }
}

impl PeriodicVv10Options {
    pub fn validate(self) -> Result<()> {
        if !self.parameters.b.is_finite()
            || self.parameters.b <= 0.0
            || !self.parameters.c.is_finite()
            || self.parameters.c < 0.0
            || !self.cutoff.is_finite()
            || self.cutoff <= 0.0
        {
            return Err(DftError::InvalidInput(
                "periodic VV10 requires finite b>0, C>=0 and a positive physical pair cutoff"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Quadrature derivatives already include the weight at the target point.
/// `sigma` is the squared gradient of the total electron density.
pub struct PeriodicVv10Fields<S> {
    pub energy: S,
    pub d_rho: Vec<S>,
    pub d_sigma: Vec<S>,
    pub active_points: usize,
    pub pair_images: usize,
}

pub(crate) struct Vv10GeometryFields<S> {
    pub d_weight: Vec<S>,
    pub d_position: Vec<[S; 3]>,
    /// Affine derivative from translating the partner images, excluding
    /// displacement of the quadrature points in the reference cell.
    pub image_virial: [[S; 3]; 3],
}

/// Infinitely differentiable switch from zero to one. No denominator or
/// physical density is clipped. Compact support makes terms entering a
/// neighbor/vacuum list have vanishing derivatives at every order.
fn smooth_unit<S: Scalar>(x: S) -> S {
    if x.value() <= 0.0 {
        return S::from_f64(0.0);
    }
    if x.value() >= 1.0 {
        return S::from_f64(1.0);
    }
    let left = (-S::from_f64(1.0) / x).exp();
    let right = (-S::from_f64(1.0) / (S::from_f64(1.0) - x)).exp();
    left / (left + right)
}

fn vacuum_switch<S: Scalar>(rho: S) -> S {
    smooth_unit((rho - S::from_f64(RHO_CUT)) / S::from_f64(RHO_CUT))
}

/// Direct periodic sum per primitive cell, for 1D, 2D or 3D. Retained image
/// topology and vacuum screening are determined at the reference point;
/// every retained term is differentiated exactly. No minimum-image
/// replacement of the nonlocal interaction is made.
#[allow(clippy::too_many_arguments)]
pub fn vv10_fields<S: Scalar + Send + Sync>(
    geometry: &PeriodicGeometry,
    reference_points: &[Vec3],
    positions: &[[S; 3]],
    lattice: [[S; 3]; 3],
    weights: &[S],
    rho: &[S],
    sigma: &[S],
    options: PeriodicVv10Options,
) -> Result<PeriodicVv10Fields<S>> {
    Ok(evaluate::<S, 2>(
        geometry,
        reference_points,
        positions,
        lattice,
        weights,
        rho,
        sigma,
        options,
    )?
    .0)
}

pub(crate) fn vv10_geometry_fields(
    geometry: &PeriodicGeometry,
    points: &[Vec3],
    weights: &[f64],
    rho: &[f64],
    sigma: &[f64],
    options: PeriodicVv10Options,
) -> Result<(PeriodicVv10Fields<f64>, Vv10GeometryFields<f64>)> {
    let positions: Vec<_> = points.iter().map(|p| p.to_array()).collect();
    let lattice = geometry.lattice().a.map(|a| a.to_array());
    let (density, geometry) = evaluate::<f64, 3>(
        geometry, points, &positions, lattice, weights, rho, sigma, options,
    )?;
    Ok((
        density,
        geometry.expect("third slot is the squared-distance derivative"),
    ))
}

type Vv10Evaluation<S> = (PeriodicVv10Fields<S>, Option<Vv10GeometryFields<S>>);

fn evaluate<S: Scalar + Send + Sync, const N: usize>(
    geometry: &PeriodicGeometry,
    reference_points: &[Vec3],
    positions: &[[S; 3]],
    lattice: [[S; 3]; 3],
    weights: &[S],
    rho: &[S],
    sigma: &[S],
    options: PeriodicVv10Options,
) -> Result<Vv10Evaluation<S>> {
    options.validate()?;
    let n = reference_points.len();
    if n == 0
        || positions.len() != n
        || weights.len() != n
        || rho.len() != n
        || sigma.len() != n
        || reference_points.iter().any(|p| !p.norm().is_finite())
        || positions
            .iter()
            .flatten()
            .chain(lattice.iter().flatten())
            .chain(weights)
            .chain(rho)
            .chain(sigma)
            .any(|v| !v.value().is_finite())
        || sigma.iter().any(|v| v.value() < 0.0)
    {
        return Err(DftError::InvalidInput(
            "invalid periodic VV10 grid, scalar geometry or density fields".into(),
        ));
    }
    let beta = (1.0 / 32.0) * (3.0 / options.parameters.b.powi(2)).powf(0.75);
    let active: Vec<_> = (0..n)
        .filter(|i| rho[*i].value() >= RHO_CUT && (N == 3 || weights[*i].value() != 0.0))
        .collect();
    let partners: Vec<_> = active
        .iter()
        .map(|&j| {
            (
                Vv10Endpoint::new(rho[j], sigma[j], options.parameters),
                vacuum_switch(rho[j]),
            )
        })
        .collect();
    let outer = options.cutoff.powi(2);
    let inner = (0.8 * options.cutoff).powi(2);
    let zero = S::from_f64(0.0);
    let rows: Result<Vec<_>> = active
        .par_iter()
        .map(|&i| {
            let target_rho = DualN::<S, N>::variable(rho[i], 0);
            let target_switch = vacuum_switch(target_rho);
            let target = Vv10Endpoint::new(
                target_rho,
                DualN::<S, N>::variable(sigma[i], 1),
                options.parameters,
            );
            let local = DualN::from_f64(beta) * target_rho * target_switch;
            let mut energy = local.re;
            let mut vrho = local.im[0];
            let mut vsigma = S::from_f64(0.0);
            let mut point_gradient = [zero; 3];
            let mut image_virial = [[zero; 3]; 3];
            let mut count = 0usize;
            for (partner_index, &j) in active.iter().enumerate() {
                if N == 3 && weights[j].value() == 0.0 {
                    continue;
                }
                let (partner, partner_switch) = partners[partner_index];
                let partner = partner.map(DualN::constant);
                let images = geometry.nearby_translation_indices(
                    reference_points[i] - reference_points[j],
                    options.cutoff,
                )?;
                for image in images {
                    let translation: [S; 3] = std::array::from_fn(|axis| {
                        (0..3)
                            .map(|a| lattice[a][axis] * S::from_f64(image[a] as f64))
                            .fold(zero, |a, b| a + b)
                    });
                    let r: [S; 3] =
                        std::array::from_fn(|a| positions[i][a] - positions[j][a] + translation[a]);
                    let r2 = r.iter().fold(S::from_f64(0.0), |sum, x| sum + *x * *x);
                    // Exact VV10 below 0.8 Rcut, C-infinity taper to zero at
                    // Rcut. The omitted tail vanishes as Rcut increases.
                    let distance = if N == 3 {
                        DualN::variable(r2, 2)
                    } else {
                        DualN::constant(r2)
                    };
                    let taper = if N == 3 {
                        DualN::from_f64(1.0)
                            - smooth_unit(
                                (distance - DualN::from_f64(inner))
                                    / DualN::from_f64(outer - inner),
                            )
                    } else {
                        DualN::constant(
                            S::from_f64(1.0)
                                - smooth_unit(
                                    (r2 - S::from_f64(inner)) / S::from_f64(outer - inner),
                                ),
                        )
                    };
                    let partner_weight = if N == 3 {
                        DualN::constant(partner_switch) * taper
                    } else {
                        DualN::constant(partner_switch * taper.re)
                    };
                    let pair = vv10_prepared_pair(target, partner, distance)
                        * target_switch
                        * partner_weight;
                    energy = energy + S::from_f64(0.5) * weights[j] * pair.re;
                    vrho = vrho + weights[j] * pair.im[0];
                    vsigma = vsigma + weights[j] * pair.im[1];
                    if N == 3 {
                        // The point appears at both endpoints of the
                        // ordered sum. Each image derivative retains 1/2.
                        let radial = weights[i] * weights[j] * pair.im[2];
                        for a in 0..3 {
                            let derivative = radial * r[a];
                            point_gradient[a] = point_gradient[a] + S::from_f64(2.0) * derivative;
                            for b in 0..3 {
                                image_virial[a][b] =
                                    image_virial[a][b] + derivative * translation[b];
                            }
                        }
                    }
                    if N != 3 || weights[i].value() != 0.0 {
                        count = count.checked_add(1).ok_or_else(|| {
                            DftError::InvalidInput("periodic VV10 pair count overflow".into())
                        })?;
                    }
                }
            }
            Ok((
                i,
                weights[i] * energy,
                weights[i] * vrho,
                weights[i] * vsigma,
                count,
                (N == 3).then(|| {
                    (
                        S::from_f64(2.0) * energy - local.re,
                        point_gradient,
                        image_virial,
                    )
                }),
            ))
        })
        .collect();
    let mut result = PeriodicVv10Fields {
        energy: S::from_f64(0.0),
        d_rho: vec![S::from_f64(0.0); n],
        d_sigma: vec![S::from_f64(0.0); n],
        active_points: if N == 3 {
            active
                .iter()
                .filter(|&&i| weights[i].value() != 0.0)
                .count()
        } else {
            active.len()
        },
        pair_images: 0,
    };
    let mut geometry = (N == 3).then(|| Vv10GeometryFields {
        d_weight: vec![zero; n],
        d_position: vec![[zero; 3]; n],
        image_virial: [[zero; 3]; 3],
    });
    for (i, energy, vrho, vsigma, count, geometry_row) in rows? {
        result.energy = result.energy + energy;
        result.d_rho[i] = vrho;
        result.d_sigma[i] = vsigma;
        if let (Some(geometry), Some((weight, point, virial))) = (&mut geometry, geometry_row) {
            geometry.d_weight[i] = weight;
            geometry.d_position[i] = point;
            for (target, row) in geometry.image_virial.iter_mut().zip(virial) {
                for (a, b) in target.iter_mut().zip(row) {
                    *a = *a + b;
                }
            }
        }
        result.pair_images = result.pair_images.checked_add(count).ok_or_else(|| {
            DftError::InvalidInput("periodic VV10 total pair count overflow".into())
        })?;
    }
    if !result.energy.value().is_finite()
        || result
            .d_rho
            .iter()
            .chain(&result.d_sigma)
            .any(|v| !v.value().is_finite())
        || geometry.as_ref().is_some_and(|fields| {
            fields
                .d_weight
                .iter()
                .chain(fields.d_position.iter().flatten())
                .chain(fields.image_virial.iter().flatten())
                .any(|v| !v.value().is_finite())
        })
    {
        return Err(DftError::InvalidInput(
            "periodic VV10 produced a nonfinite functional or kernel".into(),
        ));
    }
    Ok((result, geometry))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use crate::periodic::{features::PeriodicBoundary, Lattice};

    #[test]
    fn vv10_point_weight_and_image_virial_match_independent_four_step_energy() {
        let rows = [[6.7, 0.16, -0.21], [1.1, 7.3, 0.28], [-0.8, 0.6, 8.1]];
        let points = [
            [0.13, -0.21, 0.31],
            [1.17, 0.42, -0.53],
            [-0.72, 1.29, 0.84],
        ];
        let weights = [0.7, 1.1, 0.9];
        let rho = [0.14, 0.21, 0.08];
        let sigma = [0.013, 0.007, 0.019];
        let settings = PeriodicVv10Options {
            parameters: Vv10Params::WB97XV,
            cutoff: 13.7,
        };
        let vector = |[x, y, z]: [f64; 3]| Vec3::new(x, y, z);
        let g = PeriodicGeometry::new(
            Lattice::new(vector(rows[0]), vector(rows[1]), vector(rows[2])),
            PeriodicBoundary::FULL_3D,
        )
        .unwrap();
        let (fields, explicit) =
            vv10_geometry_fields(&g, &points.map(vector), &weights, &rho, &sigma, settings)
                .unwrap();
        let mut zero_weight = weights;
        zero_weight[2] = 0.0;
        let (_, zero_fields) = vv10_geometry_fields(
            &g,
            &points.map(vector),
            &zero_weight,
            &rho,
            &sigma,
            settings,
        )
        .unwrap();
        let sample = |h| {
            let mut w = zero_weight;
            w[2] = h;
            oracle::energy(rows, [true; 3], &points, &rho, &sigma, &w, settings)
        };
        // The weight covector remains defined at a zero-weight endpoint.
        let slope = (sample(0.002) - sample(-0.002)) / 0.004;
        assert!((slope - zero_fields.d_weight[2]).abs() < 1e-13);
        let expected: Vec<_> = explicit
            .d_weight
            .iter()
            .chain(explicit.d_position.iter().flatten())
            .chain(explicit.image_virial.iter().flatten())
            .copied()
            .collect();
        let energy = |i: usize, h: f64| {
            let (mut r, mut p, mut w) = (rows, points, weights);
            if i < 3 {
                // A nonlinear path gives the direct weight derivative a
                // resolvable centered-stencil error (energy is quadratic in w).
                w[i] += h + 0.4 * h * h;
            } else if i < 12 {
                p[(i - 3) / 3][(i - 3) % 3] += h;
            } else {
                let (a, b) = ((i - 12) / 3, (i - 12) % 3);
                for row in &mut r {
                    row[a] += h * row[b];
                }
            }
            oracle::energy(r, [true; 3], &p, &rho, &sigma, &w, settings)
        };
        assert!((energy(0, 0.0) - fields.energy).abs() < 1e-13);
        for a in 0..3 {
            assert!(explicit.d_position.iter().map(|p| p[a]).sum::<f64>().abs() < 1e-15);
        }
        println!(
            "periodic VV10 fixed descriptors energy={:.15e} partials={expected:?}",
            fields.energy
        );
        let names: Vec<_> = (0..expected.len())
            .map(|i| format!("VV10 geometry partial {i}"))
            .collect();
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.018, 0.013, 0.009, 0.006], stencils::Stencil::Second),
                ([0.009, 0.006, 0.004, 0.0025], stencils::Stencil::Second),
                ([0.025, 0.019, 0.014, 0.010], stencils::Stencil::Fourth),
                // The xz image coefficient has competing h^2 and h^4
                // contributions at the coarser taper intervals.
                ([0.002, 0.0014, 0.0010, 0.0007], stencils::Stencil::Second),
            ],
            |steps, stencil| {
                expected
                    .iter()
                    .enumerate()
                    .map(|(i, &target)| {
                        steps.map(|h| {
                            let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                                (energy(i, 2.0 * h), energy(i, -2.0 * h))
                            } else {
                                (0.0, 0.0)
                            };
                            (stencil.first(energy(i, h), energy(i, -h), p2, m2, h) - target).abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
        components::assert_no_failures("periodic VV10 geometry partials", &failures);
    }

    mod oracle {
        use super::PeriodicVv10Options;
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/nonlocal_oracle.rs"
        ));
    }
}
