// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Coulomb energy and analytic forces with one, two or three periodic axes.
//! The reciprocal transform is discrete only in the periodic subspace.
//! Open components remain continuous, including the nonconstant zero mode.
//! Gaussian countercharges are actual charge densities, not a dropped mode.

use super::ewald::EwaldParams;
use super::PeriodicGeometry;
use crate::dual::Scalar;
use crate::grid::gauss_legendre;

mod scalar;
use crate::math::Vec3;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct ElectrostaticSource {
    pub position: Vec3,
    pub charge: f64,
    /// Zero: point charge (bare self excluded). Positive: normalized
    /// rho = q/(pi*width^2)^(3/2) exp(-|r-R|^2/width^2), including its
    /// physical finite self energy. Width is in Bohr.
    pub width: f64,
}

#[derive(Clone, Debug, Default)]
pub enum ChargeCompensation {
    #[default]
    RequireNeutral,
    /// Uniform 3D compensating density and zero average potential.
    UniformBackground,
    /// A normalized Gaussian with charge minus the supplied net charge,
    /// repeated only along periodic axes. The center is fixed in space.
    GaussianCountercharge { center: Vec3, width: f64 },
}

#[derive(Clone, Debug)]
pub struct ElectrostaticOptions {
    pub compensation: ChargeCompensation,
    pub alpha: Option<f64>,
    pub real_cutoff: Option<f64>,
    pub reciprocal_cutoff: Option<f64>,
    pub quadrature_order: usize,
    /// Absolute agreement between the order-n and order-2n Fourier
    /// integrals, for both energy and forces in atomic units. Lattice-sum
    /// truncation must independently be converged with the cutoff controls.
    pub tolerance: f64,
}
impl Default for ElectrostaticOptions {
    fn default() -> Self {
        Self {
            compensation: ChargeCompensation::RequireNeutral,
            alpha: None,
            real_cutoff: None,
            reciprocal_cutoff: None,
            quadrature_order: 96,
            tolerance: 1e-11,
        }
    }
}
#[derive(Clone, Debug)]
pub struct ElectrostaticComponents<S = f64> {
    pub real_space: S,
    pub reciprocal_nonzero: S,
    pub reciprocal_zero: S,
    pub point_self: S,
}
impl<S: Scalar> Default for ElectrostaticComponents<S> {
    fn default() -> Self {
        Self {
            real_space: S::from_f64(0.0),
            reciprocal_nonzero: S::from_f64(0.0),
            reciprocal_zero: S::from_f64(0.0),
            point_self: S::from_f64(0.0),
        }
    }
}
impl<S: Scalar> ElectrostaticComponents<S> {
    pub fn total(&self) -> S {
        self.real_space + self.reciprocal_nonzero + self.reciprocal_zero + self.point_self
    }
}
#[derive(Clone, Debug)]
pub struct ElectrostaticReport {
    pub energy: f64,
    pub components: ElectrostaticComponents,
    pub forces: Vec<Vec3>,
    /// Charge-averaged site potentials (Gaussian sites use their own
    /// width); 1/2 sum q_i phi_i includes the countercharge and equals E.
    pub site_potentials: Vec<f64>,
    pub compensation_force: Option<Vec3>,
    pub compensation_potential: Option<f64>,
    pub supplied_charge: f64,
    pub compensating_charge: f64,
    pub boundary: super::features::PeriodicBoundary,
    pub measure: f64,
    pub params: EwaldParams,
    pub quadrature_order: usize,
    pub quadrature_energy_defect: f64,
    pub quadrature_force_defect: f64,
    pub force_sum_defect: f64,
    pub converged: bool,
    pub units: &'static str,
    pub model: &'static str,
    pub potential_convention: &'static str,
    pub warnings: Vec<String>,
}

/// A reusable Green operator for point or Gaussian source pairs. All
/// returned gradients are derivatives with respect to the first position.
pub struct DimensionalCoulomb {
    pub(super) geometry: PeriodicGeometry,
    pub(super) params: EwaldParams,
    pub(super) reciprocal: Vec<Vec3>,
    pub(super) nodes: Vec<f64>,
    pub(super) weights: Vec<f64>,
}
impl DimensionalCoulomb {
    pub fn new(
        geometry: PeriodicGeometry,
        params: EwaldParams,
        quadrature_order: usize,
    ) -> Result<Self> {
        if geometry.dimension() == 0
            || !params.alpha.is_finite()
            || params.alpha <= 0.0
            || !params.rcut.is_finite()
            || params.rcut <= 0.0
            || !params.gcut.is_finite()
            || params.gcut <= 0.0
            || !(16..=2048).contains(&quadrature_order)
        {
            return Err(DftError::InvalidInput(
                "dimensional Coulomb requires 1D/2D/3D geometry, positive finite Ewald \
                parameters and quadrature order 16..2048"
                    .into(),
            ));
        }
        let reciprocal = geometry.reciprocal_within(params.gcut)?;
        let (nodes, weights) = gauss_legendre(quadrature_order);
        let nodes = nodes.into_iter().map(|v| (v + 1.0) * 0.5).collect();
        let weights = weights.into_iter().map(|v| v * 0.5).collect();
        Ok(Self {
            geometry,
            params,
            reciprocal,
            nodes,
            weights,
        })
    }
    pub fn geometry(&self) -> &PeriodicGeometry {
        &self.geometry
    }
    pub fn params(&self) -> EwaldParams {
        self.params
    }
    /// Full periodic Coulomb pair, including the selected zero-mode gauge.
    /// `width` is sqrt(width_i^2+width_j^2). `exclude_home` is valid only
    /// for the self interaction of one point charge at displacement zero.
    pub fn pair(&self, r: Vec3, width: f64, exclude_home: bool) -> Result<(f64, Vec3)> {
        let (components, gradient) =
            self.scalar_pair(r.to_array(), width, exclude_home, |x: f64| {
                let (s, c) = x.sin_cos();
                (c, s)
            })?;
        Ok((
            components.total(),
            Vec3::new(gradient[0], gradient[1], gradient[2]),
        ))
    }
    pub(crate) fn evaluate(
        &self,
        sources: &[ElectrostaticSource],
    ) -> Result<(ElectrostaticComponents, Vec<Vec3>, Vec<f64>)> {
        let mut components = ElectrostaticComponents::default();
        let mut forces = vec![Vec3::zero(); sources.len()];
        let mut potentials = vec![0.0; sources.len()];
        for i in 0..sources.len() {
            for j in 0..=i {
                let r = sources[i].position - sources[j].position;
                let width = sources[i].width.hypot(sources[j].width);
                let point_self = i == j && width == 0.0;
                let (pair, derivative) =
                    self.scalar_pair(r.to_array(), width, point_self, |x: f64| {
                        let (s, c) = x.sin_cos();
                        (c, s)
                    })?;
                let q = sources[i].charge * sources[j].charge;
                let factor = if i == j { 0.5 } else { 1.0 };
                components.real_space += factor * q * pair.real_space;
                components.reciprocal_nonzero += factor * q * pair.reciprocal_nonzero;
                components.reciprocal_zero += factor * q * pair.reciprocal_zero;
                components.point_self += factor * q * pair.point_self;
                let potential = pair.total();
                potentials[i] += sources[j].charge * potential;
                if i != j {
                    potentials[j] += sources[i].charge * potential;
                    let gradient = Vec3::new(derivative[0], derivative[1], derivative[2]) * q;
                    forces[i] -= gradient;
                    forces[j] += gradient;
                }
            }
        }
        Ok((components, forces, potentials))
    }
}

/// Complete electrostatic source calculation. This is a Coulomb foundation
/// driver, not a self-consistent electronic energy or a point-charge DFT model.
pub fn periodic_electrostatics(
    geometry: &PeriodicGeometry,
    sources: &[ElectrostaticSource],
    options: &ElectrostaticOptions,
) -> Result<ElectrostaticReport> {
    if sources.is_empty()
        || sources.iter().any(|s| {
            !s.position.norm().is_finite()
                || !s.charge.is_finite()
                || !s.width.is_finite()
                || s.width < 0.0
        })
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || !(16..=1024).contains(&options.quadrature_order)
        || [
            options.alpha,
            options.real_cutoff,
            options.reciprocal_cutoff,
        ]
        .iter()
        .flatten()
        .any(|v| !v.is_finite() || *v <= 0.0)
        || geometry.dimension() == 0
    {
        return Err(DftError::InvalidInput(
            "periodic electrostatics requires finite nonempty sources, non-negative \
            widths, positive tolerance/cutoffs and 1D/2D/3D periodicity; quadrature \
            order is 16..1024"
                .into(),
        ));
    }
    let supplied_charge: f64 = sources.iter().map(|s| s.charge).sum();
    let mut all = sources.to_vec();
    let mut compensating_charge = 0.0;
    let (model, convention) = match options.compensation {
        ChargeCompensation::RequireNeutral => {
            let magnitude: f64 = sources.iter().map(|s| s.charge.abs()).sum();
            if supplied_charge.abs() > 64.0 * f64::EPSILON * magnitude.max(1.0) {
                return Err(DftError::InvalidInput(
                    "non-neutral periodic sources require an explicit charge compensation model"
                        .into(),
                ));
            }
            (
                "neutral dimensional Coulomb; open directions isolated",
                "3D zero-average/tin-foil; 2D isolated slab zero mode; 1D logarithmic \
                gauge at 1 Bohr, neutral energy gauge invariant",
            )
        }
        ChargeCompensation::UniformBackground => {
            if geometry.dimension() != 3 {
                return Err(DftError::InvalidInput(
                    "uniform background requires 3D periodicity; a charged wire/slab needs a \
                    specified finite transverse countercharge"
                        .into(),
                ));
            }
            compensating_charge = -supplied_charge;
            (
                "3D Coulomb with a uniform compensating background; no isolated-defect correction",
                "zero average potential; G=0 background -pi Q^2/(2 alpha^2 V); tin-foil boundary",
            )
        }
        ChargeCompensation::GaussianCountercharge { center, width } => {
            if !center.norm().is_finite() || !width.is_finite() || width <= 0.0 {
                return Err(DftError::InvalidInput(
                    "Gaussian countercharge requires a finite center and positive width in Bohr"
                        .into(),
                ));
            }
            compensating_charge = -supplied_charge;
            all.push(ElectrostaticSource {
                position: center,
                charge: compensating_charge,
                width,
            });
            (
                "dimensional Coulomb with a co-periodic normalized Gaussian \
                countercharge of specified center/width; open directions isolated",
                "same dimensional Green function for sources and countercharge; finite \
                Gaussian self energy included; neutral total energy gauge invariant",
            )
        }
    };
    let scale = geometry.measure().powf(1.0 / geometry.dimension() as f64);
    let alpha = options.alpha.unwrap_or(std::f64::consts::PI.sqrt() / scale);
    let widest = all.iter().map(|s| s.width).fold(0.0_f64, f64::max);
    let shortest_inverse = if widest == 0.0 {
        alpha
    } else {
        alpha.min(1.0 / (2.0_f64.sqrt() * widest))
    };
    let logarithm = (1.0 / options.tolerance.min(1e-12)).ln();
    let params = EwaldParams {
        alpha,
        rcut: options
            .real_cutoff
            .unwrap_or(logarithm.sqrt() / shortest_inverse),
        gcut: options
            .reciprocal_cutoff
            .unwrap_or(2.0 * alpha * logarithm.sqrt()),
    };
    let coarse = DimensionalCoulomb::new(geometry.clone(), params, options.quadrature_order)?;
    let fine = DimensionalCoulomb::new(geometry.clone(), params, 2 * options.quadrature_order)?;
    let (old, old_forces, _) = coarse.evaluate(&all)?;
    let (components, mut forces, mut potentials) = fine.evaluate(&all)?;
    let energy = components.total();
    let quadrature_energy_defect = (energy - old.total()).abs();
    let quadrature_force_defect = forces
        .iter()
        .zip(&old_forces)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);
    let force_sum_defect = forces.iter().fold(Vec3::zero(), |a, &b| a + b).norm();
    let (compensation_force, compensation_potential) = if all.len() > sources.len() {
        (forces.pop(), potentials.pop())
    } else {
        (None, None)
    };
    let converged = energy.is_finite()
        && forces.iter().all(|f| f.norm().is_finite())
        && quadrature_energy_defect <= options.tolerance
        && quadrature_force_defect <= options.tolerance;
    let mut warnings = vec![
        "Fourier quadrature agreement is measured independently; converge \
        real/reciprocal cutoffs and splitting parameter for the requested \
        source distribution"
            .into(),
    ];
    if !converged {
        warnings.push("Fourier quadrature defects exceed the requested absolute tolerance".into());
    }
    if compensating_charge != 0.0 {
        warnings.push(
            "charged-cell energies depend on the stated compensation model; no \
            Makov-Payne or other isolated-defect correction is applied by this \
            driver"
                .into(),
        );
    }
    Ok(ElectrostaticReport {
        energy,components,forces,site_potentials:potentials,compensation_force,compensation_potential,
        supplied_charge,compensating_charge,boundary:geometry.boundary(),measure:geometry.measure(),params,
        quadrature_order:2*options.quadrature_order,quadrature_energy_defect,quadrature_force_defect,
        force_sum_defect,converged,units:"energy Hartree; forces Hartree/Bohr; potentials Hartree/e; charge e; lengths Bohr; reciprocal vectors 1/Bohr",
        model,potential_convention:convention,warnings,
    })
}
