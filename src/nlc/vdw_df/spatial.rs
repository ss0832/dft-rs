// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Direct molecular spatial quadrature of the Dion double-density integral.

use super::{
    unpolarized_local_scale, InternalExchange, MolecularVdwDfEnergy, PairKernelQuadrature, Variant,
};
use crate::basis::MolBasis;
use crate::grid::{
    angular_rule, eval::eval_aos_full, gauss_legendre, AngularScheme, MolecularGrid,
};
use crate::linalg::{AccurateSum, Matrix};
use crate::math::Vec3;
use crate::{DftError, Result};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Total density (Bohr⁻³) and its squared gradient (Bohr⁻⁸).
#[derive(Clone, Copy)]
pub struct LocalDensity {
    pub density: f64,
    pub gradient_squared: f64,
}

/// An outer molecular quadrature point with its sampled density.
#[derive(Clone, Copy)]
pub struct WeightedDensity {
    pub position: Vec3,
    pub volume: f64,
    pub local: LocalDensity,
}

#[derive(Clone, Copy)]
struct Shift {
    displacement: Vec3,
    radius: f64,
    volume: f64,
}

/// An inner spherical rule centered separately on each outer density point.
/// Both spatial quadratures and the independent momentum kernel can be
/// refined to control the molecular nonlocal correlation energy.
pub struct MolecularPairGrid {
    shifts: Vec<Shift>,
    density_cutoff: f64,
}

impl MolecularPairGrid {
    /// Map a radial Legendre rule on [-1, 1] to [0, infinity). The radial
    /// scale is in Bohr and multiplies (1+x)/(1-x).
    pub fn new(
        radial_order: usize,
        angular_degree: usize,
        radial_scale: f64,
        density_cutoff: f64,
    ) -> Result<Self> {
        if !(2..=256).contains(&radial_order)
            || !(3..=131).contains(&angular_degree)
            || !radial_scale.is_finite()
            || radial_scale <= 0.0
            || !density_cutoff.is_finite()
            || density_cutoff <= 0.0
        {
            return Err(DftError::InvalidInput(
                "invalid vdW-DF molecular pair quadrature settings".into(),
            ));
        }
        let (nodes, weights) = gauss_legendre(radial_order);
        let angular = angular_rule(AngularScheme::Lebedev, angular_degree);
        let mut shifts = Vec::with_capacity(radial_order * angular.len());
        for (x, w) in nodes.into_iter().zip(weights) {
            let radius = radial_scale * (1.0 + x) / (1.0 - x);
            // The radial Jacobian makes the logarithmic coincidence
            // integrable; the quadrature samples strictly positive radii.
            let radial_volume = 2.0 * radial_scale * w * (radius / (1.0 - x)).powi(2);
            shifts.extend(angular.iter().map(|&(direction, weight)| Shift {
                displacement: direction * radius,
                radius,
                volume: radial_volume * weight,
            }));
        }
        Ok(Self {
            shifts,
            density_cutoff,
        })
    }

    /// Nonlocal correlation in Hartree for a supplied unpolarized AO density.
    /// The direct momentum quadrature is useful as a reference calculation.
    pub fn energy_on_density(
        &self,
        outer_grid: &MolecularGrid,
        basis: &MolBasis,
        density: &Matrix,
        model: InternalExchange,
        kernel: &PairKernelQuadrature,
    ) -> Result<f64> {
        let outer = collocate_outer(outer_grid, basis, density)?;
        self.energy_on_samples(
            &outer,
            |points| collocate(basis, density, points),
            model,
            |d, dp| kernel.evaluate(d, dp),
        )
    }

    /// Exchange, local correlation and nonlocal correlation in Hartree at a
    /// supplied unpolarized molecular density. The selected outer exchange
    /// and nonlocal kernel are the distinct parts of the named variant.
    pub fn xc_on_density(
        &self,
        outer_grid: &MolecularGrid,
        basis: &MolBasis,
        density: &Matrix,
        variant: Variant,
        kernel: impl Fn(f64, f64) -> Result<f64> + Sync,
    ) -> Result<MolecularVdwDfEnergy> {
        let outer = collocate_outer(outer_grid, basis, density)?;
        let mut exchange = AccurateSum::default();
        let mut local_correlation = AccurateSum::default();
        for point in &outer {
            let rho = point.local.density;
            if rho < crate::xc::RHO_CUTOFF {
                continue;
            }
            let kf = (3.0 * PI * PI * rho).cbrt();
            let s = point.local.gradient_squared.sqrt() / (2.0 * kf * rho);
            let eps_c = crate::xc::lda::pw92_epsilon_original(
                0.5 * rho,
                0.5 * rho,
                &crate::xc::lda::PW92_PARAMS,
            );
            let wrho = point.volume * rho;
            exchange.add(-0.75 * kf / PI * wrho * variant.exchange_enhancement(s)?);
            local_correlation.add(wrho * eps_c);
        }
        let nonlocal_correlation = self.energy_on_samples(
            &outer,
            |points| collocate(basis, density, points),
            variant.internal_exchange(),
            kernel,
        )?;
        Ok(MolecularVdwDfEnergy {
            exchange: exchange.value(),
            local_correlation: local_correlation.value(),
            nonlocal_correlation,
        })
    }

    /// Nonlocal correlation in Hartree on a supplied outer quadrature.
    /// The density provider samples each batch of shifted positions. The
    /// pair evaluator may use direct momentum quadrature or a converged table.
    pub fn energy_on_samples(
        &self,
        outer: &[WeightedDensity],
        density: impl Fn(&[Vec3]) -> Vec<LocalDensity> + Sync,
        model: InternalExchange,
        kernel: impl Fn(f64, f64) -> Result<f64> + Sync,
    ) -> Result<f64> {
        let values: Vec<_> = outer
            .par_iter()
            .map(|point| {
                if !point.position.norm2().is_finite() || !point.volume.is_finite() {
                    return Err(DftError::InvalidInput(
                        "vdW-DF outer quadrature contains a nonfinite position or weight".into(),
                    ));
                }
                let Some(q) = self.scale(point.local, model)? else {
                    return Ok(0.0);
                };
                let mut integral = AccurateSum::default();
                for shifts in self.shifts.chunks(512) {
                    let positions: Vec<_> = shifts
                        .iter()
                        .map(|s| point.position + s.displacement)
                        .collect();
                    let samples = density(&positions);
                    if samples.len() != positions.len() {
                        return Err(DftError::InvalidInput(
                            "vdW-DF density provider returned the wrong number of inner samples"
                                .into(),
                        ));
                    }
                    for (shift, sample) in shifts.iter().zip(samples) {
                        if let Some(qp) = self.scale(sample, model)? {
                            integral.add(
                                shift.volume
                                    * sample.density
                                    * kernel(q * shift.radius, qp * shift.radius)?,
                            );
                        }
                    }
                }
                Ok(0.5 * point.volume * point.local.density * integral.value())
            })
            .collect::<Result<_>>()?;
        let mut sum = AccurateSum::default();
        for value in values {
            sum.add(value);
        }
        let energy = sum.value();
        if !energy.is_finite() {
            return Err(DftError::InvalidInput(
                "vdW-DF nonlocal spatial energy is nonfinite".into(),
            ));
        }
        Ok(energy)
    }

    fn scale(&self, local: LocalDensity, model: InternalExchange) -> Result<Option<f64>> {
        if !local.density.is_finite()
            || local.density < 0.0
            || !local.gradient_squared.is_finite()
            || local.gradient_squared < 0.0
        {
            return Err(DftError::InvalidInput(
                "invalid vdW-DF spatial density or squared gradient".into(),
            ));
        }
        if local.density < self.density_cutoff {
            return Ok(None);
        }
        unpolarized_local_scale(local.density, local.gradient_squared, model).map(Some)
    }
}

fn collocate_outer(
    grid: &MolecularGrid,
    basis: &MolBasis,
    density: &Matrix,
) -> Result<Vec<WeightedDensity>> {
    if density.rows != basis.nao_sph
        || density.cols != basis.nao_sph
        || density.as_slice().iter().any(|v| !v.is_finite())
        || grid.weights.len() != grid.len()
    {
        return Err(DftError::InvalidInput(
            "vdW-DF needs a finite molecular AO density and a valid outer grid".into(),
        ));
    }
    Ok(grid
        .points
        .chunks(512)
        .zip(grid.weights.chunks(512))
        .flat_map(|(points, weights)| {
            points
                .iter()
                .zip(weights)
                .zip(collocate(basis, density, points))
                .map(|((&position, &volume), local)| WeightedDensity {
                    position,
                    volume,
                    local,
                })
        })
        .collect())
}

fn collocate(basis: &MolBasis, density: &Matrix, points: &[Vec3]) -> Vec<LocalDensity> {
    let ao = eval_aos_full(basis, points, true, false);
    let grad = ao.grad.as_ref().expect("AO gradients were requested");
    let dp = ao.phi.matmul_seq(density);
    (0..points.len())
        .map(|p| {
            let dot = |left: &Matrix| {
                (0..basis.nao_sph)
                    .map(|mu| left[(p, mu)] * dp[(p, mu)])
                    .sum::<f64>()
            };
            LocalDensity {
                density: dot(&ao.phi).max(0.0),
                gradient_squared: grad.iter().map(|g| (2.0 * dot(g)).powi(2)).sum(),
            }
        })
        .collect()
}
