// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Staged affine coefficient chart for a fixed static-core model.
//!
//! This module is deliberately unregistered until its parent patch and every
//! independent parameter derivative pass acceptance. Fifty tangent coordinates
//! represent the neutral subspace of the common51 coefficient layout. The
//! caller supplies actual unit operator/classical/moment/geometry derivatives;
//! this type does not infer them from an arbitrary fitted coefficient record.

use super::{ContinuousSpeciesModel, Fingerprint, SpeciesState};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::{DftError, Result};
use std::ops::Deref;

pub use super::static_core::{
    static_core_correction as build_static_core_contribution,
    StaticCoreCorrection as CoreParameterContribution,
    StaticCoreParameters as CoreParameterDefinition,
};

pub const AMBIENT_COEFFICIENT_COUNT: usize = 51;
pub const NEUTRAL_TANGENT_COUNT: usize = 50;

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

fn same_vector(a: Vec3, b: Vec3) -> bool {
    (0..3).all(|i| a.get(i).to_bits() == b.get(i).to_bits())
}

fn hash_core(identity: &mut Fingerprint, core: &CoreParameterContribution) {
    identity.number(core.operator.rows);
    identity.number(core.operator.cols);
    for &x in core.operator.as_slice() { identity.float(x); }
    for atom in &core.atom_derivative_matrices {
        for matrix in atom { for &x in matrix.as_slice() { identity.float(x); } }
    }
    identity.float(core.classical_energy);
    for &gradient in &core.classical_gradients { identity.vector(gradient); }
    identity.vector(core.permanent_dipole);
    for atom in &core.permanent_dipole_derivatives {
        for row in atom { for &x in row { identity.float(x); } }
    }
    identity.float(core.charge_correction);
    identity.vector(core.origin);
    for site in &core.electrostatic_sites {
        identity.vector(site.center); identity.float(site.exponent);
        identity.float(site.charge); identity.vector(site.dipole);
    }
}

fn copy_core(base: &CoreParameterContribution) -> CoreParameterContribution {
    CoreParameterContribution {
        operator: base.operator.clone(),
        atom_derivative_matrices: base.atom_derivative_matrices.clone(),
        classical_energy: base.classical_energy,
        classical_gradients: base.classical_gradients.clone(),
        permanent_dipole: base.permanent_dipole,
        permanent_dipole_derivatives: base.permanent_dipole_derivatives.clone(),
        charge_correction: base.charge_correction,
        electrostatic_sites: base.electrostatic_sites.clone(),
        origin: base.origin,
    }
}

fn add_matrix(target: &mut Matrix, source: &Matrix, amount: f64) {
    for (x, &y) in target.as_mut_slice().iter_mut().zip(source.as_slice()) { *x += amount * y; }
}

fn add_core(target: &mut CoreParameterContribution, direction: &CoreParameterContribution, q: f64) {
    add_matrix(&mut target.operator, &direction.operator, q);
    for (target, direction) in target.atom_derivative_matrices.iter_mut().zip(&direction.atom_derivative_matrices) {
        for axis in 0..3 { add_matrix(&mut target[axis], &direction[axis], q); }
    }
    target.classical_energy += q * direction.classical_energy;
    for (x, &y) in target.classical_gradients.iter_mut().zip(&direction.classical_gradients) { *x += y * q; }
    target.permanent_dipole += direction.permanent_dipole * q;
    for (target, direction) in target.permanent_dipole_derivatives.iter_mut().zip(&direction.permanent_dipole_derivatives) {
        for i in 0..3 { for j in 0..3 { target[i][j] += q * direction[i][j]; } }
    }
    target.charge_correction += q * direction.charge_correction;
    // Coalesce amplitudes on exactly the same three Gaussian sites. Never
    // manufacture three extra sites for every parameter direction.
    for (target, direction) in target.electrostatic_sites.iter_mut().zip(&direction.electrostatic_sites) {
        target.charge += q * direction.charge;
        target.dipole += direction.dipole * q;
    }
}

/// Validated affine data, not a generation or transferability certificate.
/// Every direction includes the complete scalar-energy and permanent-moment
/// derivative matching its operator. Local/nonlocal matrices may both occur.
pub struct AffineCoreDirections {
    base: CoreParameterContribution,
    directions: Vec<CoreParameterContribution>,
    provenance: String,
}

impl AffineCoreDirections {
    pub fn new(base: CoreParameterContribution, directions: Vec<CoreParameterContribution>, provenance: String) -> Result<Self> {
        let nao = base.operator.rows;
        let atoms = base.atom_derivative_matrices.len();
        if nao == 0 || atoms == 0 || directions.len() != NEUTRAL_TANGENT_COUNT || provenance.trim().is_empty() {
            return invalid("affine core requires a fixed nonempty layout,50 neutral tangent directions and explicit generation provenance");
        }
        base.validate(nao, atoms)?;
        for direction in &directions {
            direction.validate(nao, atoms)?;
            if !same_vector(direction.origin, base.origin)
                || direction.electrostatic_sites.iter().zip(&base.electrostatic_sites).any(|(a,b)| {
                    !same_vector(a.center,b.center) || a.exponent.to_bits() != b.exponent.to_bits()
                })
            {
                return invalid("affine core directions must keep the exact origin,three Gaussian site centers and fixed exponents");
            }
        }
        Ok(Self { base, directions, provenance })
    }

    pub fn ambient_coefficient_count(&self) -> usize { AMBIENT_COEFFICIENT_COUNT }
    pub fn tangent_count(&self) -> usize { self.directions.len() }
    pub fn provenance(&self) -> &str { &self.provenance }
}

pub(super) enum CoreAtPoint<'a> {
    Borrowed(&'a CoreParameterContribution),
    Affine(CoreParameterContribution),
}

impl Deref for CoreAtPoint<'_> {
    type Target = CoreParameterContribution;
    fn deref(&self) -> &Self::Target {
        match self { Self::Borrowed(value) => value, Self::Affine(value) => value }
    }
}

impl ContinuousSpeciesModel {
    /// Attach one immutable base and actual50 unit perturbations. q is not
    /// part of this fingerprint; a previous state remains legitimate within
    /// this single chart. Altering any base/direction/layout changes identity.
    pub fn with_affine_core_directions(mut self, provider: AffineCoreDirections) -> Result<Self> {
        if self.chart.parameters.len() != NEUTRAL_TANGENT_COUNT
            || !self.static_core_directions.is_empty()
            || self.chart.parameters.iter().any(|p| p.charge_direction.iter().any(|&x| x != 0.0))
        {
            return invalid("affine coefficient response requires one fixed50-coordinate chart with zero ionic-charge directions");
        }
        let AffineCoreDirections { base, directions, provenance } = provider;
        if !same_vector(base.origin, self.origin) {
            return invalid("affine core origin differs from the fixed chart origin");
        }
        base.validate(self.basis.nao_sph, self.molecule.len())?;
        for direction in &directions { direction.validate(self.basis.nao_sph,self.molecule.len())?; }
        self = self.with_static_core_correction(base)?;
        let mut identity = Fingerprint(self.fingerprints.chart);
        identity.string("neutral-affine51-tangent50-static-core-v1");
        identity.string(&provenance);
        identity.number(AMBIENT_COEFFICIENT_COUNT);
        identity.number(directions.len());
        for (operator,direction) in self.parameter_operators.iter_mut().zip(&directions) {
            add_matrix(operator,&direction.operator,1.0);
            hash_core(&mut identity,direction);
        }
        self.static_core_directions = directions;
        self.fingerprints.chart = identity.0;
        Ok(self)
    }

    pub(super) fn affine_core_at(&self, q: &[f64]) -> Result<Option<CoreAtPoint<'_>>> {
        let Some(base) = &self.static_core else {
            if self.static_core_directions.is_empty() { return Ok(None); }
            return invalid("affine core directions have no fixed base");
        };
        if self.static_core_directions.is_empty() { return Ok(Some(CoreAtPoint::Borrowed(base))); }
        if q.len() != self.static_core_directions.len() || q.iter().any(|x| !x.is_finite()) {
            return invalid("affine core coordinate dimensions or finite values disagree");
        }
        let mut core = copy_core(base);
        for (&value,direction) in q.iter().zip(&self.static_core_directions) { add_core(&mut core,direction,value); }
        core.validate(self.basis.nao_sph,self.molecule.len())?;
        Ok(Some(CoreAtPoint::Affine(core)))
    }

    pub fn affine_core_parameter_count(&self) -> usize { self.static_core_directions.len() }

    /// This third mixed derivative needs its own high-order Lagrangian/XC
    /// acceptance; ordinary alpha does not establish alpha_q.
    pub fn coefficient_polarizability_gradient(&self, state: &SpeciesState) -> Result<Vec<[[f64;3];3]>> {
        self.check_state(state)?;
        Err(DftError::Unsupported { feature: "affine coefficient polarizability gradient".into(),
            missing: "independent high-order coefficient/field response provider and acceptance".into() })
    }
}
