// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit local electronic and classical residuals of a generalized site.
use super::frame::{AttachedFrame, FramePoint};
use super::gaussian::{gaussian_potential, GaussianPotential};
use super::steric::{steric_overlap, StericAtom, StericOverlap};
use super::{ContinuousSpeciesModel, Fingerprint};
use crate::math::Vec3;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub enum LocalCenter {
    /// Spherical term at a real template atom; no extra basis is added.
    Atom(usize),
    /// Laboratory-fixed external center; not a molecular substituent frame.
    Fixed(Vec3),
    Attached(AttachedFrame),
}

/// Affine amplitudes use the same dimensionless q and parameter order as the
/// charge chart. A zero charge column is legitimate for a purely local term.
#[derive(Clone, Debug)]
pub struct LocalGaussianSpec {
    pub center: LocalCenter,
    pub exponent: f64,
    pub reference_amplitude: f64,
    pub amplitude_direction: Vec<f64>,
    pub reference_steric_strength: f64,
    pub steric_direction: Vec<f64>,
    /// Fixed atom-centered environment Gaussians, positions supplied by model.
    pub environment_exponents: Vec<f64>,
    pub environment_hardness: Vec<f64>,
    pub excluded_atoms: Vec<bool>,
}

pub(super) struct LocalTerm {
    pub spec: LocalGaussianSpec,
    pub potential: GaussianPotential,
    pub steric: StericOverlap,
    pub frame: FramePoint,
}

impl LocalTerm {
    pub(super) fn amplitudes(&self, q: &[f64]) -> Result<(f64, f64)> {
        let c = self.spec.reference_amplitude
            + self
                .spec
                .amplitude_direction
                .iter()
                .zip(q)
                .map(|(a, b)| a * b)
                .sum::<f64>();
        let g = self.spec.reference_steric_strength
            + self
                .spec
                .steric_direction
                .iter()
                .zip(q)
                .map(|(a, b)| a * b)
                .sum::<f64>();
        if !c.is_finite() || !g.is_finite() || g < 0.0 {
            return Err(DftError::InvalidInput("local amplitudes must be finite and steric strengths nonnegative throughout the evaluated chart".into()));
        }
        Ok((c, g))
    }
}

pub(super) fn from_controls(
    controls: &crate::driver::Controls,
    atoms: usize,
    parameters: usize,
) -> Result<Vec<LocalGaussianSpec>> {
    let centers = controls.opt_usize_list("local_potential_atoms")?;
    let exponents = controls.opt_f64_list("local_potential_exponents")?;
    let amplitudes = controls.opt_f64_list("local_potential_reference_amplitudes")?;
    let directions = controls.opt_rows("local_potential_directions")?;
    let strengths = controls.opt_f64_list("steric_reference_strengths")?;
    let steric_directions = controls.opt_rows("steric_directions")?;
    let environment_exponents = controls.opt_f64_list("steric_environment_exponents")?;
    let hardness = controls.opt_f64_list("steric_environment_hardness")?;
    let masks = controls.opt_rows("steric_pair_mask")?;
    let anchors = controls.opt_usize_list("local_frame_anchors")?;
    let offsets = controls.opt_rows("local_frame_offsets")?;
    let Some(centers) = centers else {
        if exponents.is_some()
            || amplitudes.is_some()
            || directions.is_some()
            || strengths.is_some()
            || steric_directions.is_some()
            || environment_exponents.is_some()
            || hardness.is_some()
            || masks.is_some()
            || anchors.is_some()
            || offsets.is_some()
        {
            return Err(DftError::InvalidInput(
                "local potential controls require explicit local_potential_atoms".into(),
            ));
        }
        return Ok(Vec::new());
    };
    let n = centers.len();
    let exponents = exponents
        .ok_or_else(|| DftError::InvalidInput("local_potential_exponents is required".into()))?;
    let amplitudes = amplitudes.unwrap_or_else(|| vec![0.0; n]);
    let directions = directions.unwrap_or_else(|| vec![vec![0.0; parameters]; n]);
    let strengths = strengths.unwrap_or_else(|| vec![0.0; n]);
    let steric_directions = steric_directions.unwrap_or_else(|| vec![vec![0.0; parameters]; n]);
    if n == 0
        || [
            exponents.len(),
            amplitudes.len(),
            directions.len(),
            strengths.len(),
            steric_directions.len(),
        ]
        .iter()
        .any(|&len| len != n)
        || centers.iter().any(|&a| a >= atoms)
    {
        return Err(DftError::InvalidInput(
            "local potential arrays must contain one valid row per explicit center".into(),
        ));
    }
    let active_steric = strengths.iter().any(|&g| g != 0.0)
        || steric_directions.iter().flatten().any(|&g| g != 0.0);
    if active_steric && (environment_exponents.is_none() || hardness.is_none() || masks.is_none()) {
        return Err(DftError::InvalidInput("active steric residuals require explicit environment exponents, hardness and complete binary pair masks".into()));
    }
    let environment_exponents = environment_exponents.unwrap_or_else(|| vec![1.0; atoms]);
    let hardness = hardness.unwrap_or_else(|| vec![0.0; atoms]);
    let masks = masks.unwrap_or_else(|| vec![vec![1.0; atoms]; n]);
    if environment_exponents.len() != atoms
        || hardness.len() != atoms
        || masks.len() != n
        || masks
            .iter()
            .any(|row| row.len() != atoms || row.iter().any(|&v| v != 0.0 && v != 1.0))
    {
        return Err(DftError::InvalidInput("steric environment arrays must match atoms and mask rows must contain only 0 (included) or 1 (excluded)".into()));
    }
    if anchors.is_some() != offsets.is_some()
        || anchors.as_ref().is_some_and(|a| a.len() != 3 * n)
        || offsets
            .as_ref()
            .is_some_and(|a| a.len() != n || a.iter().any(|r| r.len() != 3))
    {
        return Err(DftError::InvalidInput("attached local frames require three flattened anchor indices and one three-vector offset per local center".into()));
    }
    let mut specs = Vec::with_capacity(n);
    for i in 0..n {
        let center = match (&anchors, &offsets) {
            (Some(a), Some(offsets)) => LocalCenter::Attached(AttachedFrame {
                anchors: [a[3 * i], a[3 * i + 1], a[3 * i + 2]],
                offset: Vec3::new(offsets[i][0], offsets[i][1], offsets[i][2]),
                min_anchor_distance: 1e-8,
                min_sine_angle: 1e-6,
            }),
            _ => LocalCenter::Atom(centers[i]),
        };
        specs.push(LocalGaussianSpec {
            center,
            exponent: exponents[i],
            reference_amplitude: amplitudes[i],
            amplitude_direction: directions[i].clone(),
            reference_steric_strength: strengths[i],
            steric_direction: steric_directions[i].clone(),
            environment_exponents: environment_exponents.clone(),
            environment_hardness: hardness.clone(),
            excluded_atoms: masks[i].iter().map(|&v| v == 1.0).collect(),
        });
    }
    Ok(specs)
}

impl ContinuousSpeciesModel {
    /// Consume an immutable model and attach explicit additional operators.
    /// Existing states become incompatible through a new content fingerprint.
    /// There are no extra electrons, nuclear charges, basis functions or
    /// density-dependent parameters attached to these local terms.
    pub fn with_local_potentials(mut self, specs: Vec<LocalGaussianSpec>) -> Result<Self> {
        if specs.is_empty() {
            return Ok(self);
        }
        if !self.local_terms.is_empty() {
            return Err(DftError::InvalidInput(
                "local potentials must be attached in one complete immutable specification".into(),
            ));
        }
        let positions: Vec<_> = self.molecule.atoms.iter().map(|a| a.position).collect();
        let atoms = positions.len();
        let nq = self.chart.parameters.len();
        let mut identity = Fingerprint(self.fingerprints.chart);
        identity.string("local-electronic-and-steric-v1");
        identity.number(specs.len());
        for spec in specs {
            if spec.amplitude_direction.len() != nq
                || spec.steric_direction.len() != nq
                || spec.environment_exponents.len() != atoms
                || spec.environment_hardness.len() != atoms
                || spec.excluded_atoms.len() != atoms
                || !spec.reference_amplitude.is_finite()
                || !spec.reference_steric_strength.is_finite()
                || spec.reference_steric_strength < 0.0
                || spec
                    .amplitude_direction
                    .iter()
                    .chain(&spec.steric_direction)
                    .any(|x| !x.is_finite())
            {
                return Err(DftError::InvalidInput("local potential dimensions, finite affine amplitudes and explicit atom environment/mask are required".into()));
            }
            let frame = match &spec.center {
                LocalCenter::Fixed(center) => FramePoint {
                    center: *center,
                    atom_jacobians: vec![[[0.0; 3]; 3]; atoms],
                },
                LocalCenter::Atom(atom) => {
                    if *atom >= atoms {
                        return Err(DftError::InvalidInput(
                            "local center atom is outside the template".into(),
                        ));
                    }
                    let mut jacobians = vec![[[0.0; 3]; 3]; atoms];
                    for axis in 0..3 {
                        jacobians[*atom][axis][axis] = 1.0;
                    }
                    FramePoint {
                        center: positions[*atom],
                        atom_jacobians: jacobians,
                    }
                }
                LocalCenter::Attached(frame) => frame.evaluate(&self.molecule)?,
            };
            let potential = gaussian_potential(&self.basis, atoms, frame.center, spec.exponent)?;
            let environment: Vec<_> = (0..atoms)
                .map(|a| StericAtom {
                    position: positions[a],
                    exponent: spec.environment_exponents[a],
                    hardness: spec.environment_hardness[a],
                })
                .collect();
            let steric = steric_overlap(
                frame.center,
                spec.exponent,
                &environment,
                &spec.excluded_atoms,
            )?;
            for (operator, &coefficient) in self
                .parameter_operators
                .iter_mut()
                .zip(&spec.amplitude_direction)
            {
                for (value, &element) in operator
                    .as_mut_slice()
                    .iter_mut()
                    .zip(potential.operator.as_slice())
                {
                    *value += coefficient * element;
                }
            }
            identity.float(spec.exponent);
            identity.float(spec.reference_amplitude);
            identity.float(spec.reference_steric_strength);
            identity.vector(frame.center);
            for &x in spec
                .amplitude_direction
                .iter()
                .chain(&spec.steric_direction)
                .chain(&spec.environment_exponents)
                .chain(&spec.environment_hardness)
            {
                identity.float(x);
            }
            for &excluded in &spec.excluded_atoms {
                identity.number(usize::from(excluded));
            }
            for jacobian in &frame.atom_jacobians {
                for row in jacobian {
                    for &x in row {
                        identity.float(x);
                    }
                }
            }
            self.local_terms.push(LocalTerm {
                spec,
                potential,
                steric,
                frame,
            });
        }
        self.fingerprints.chart = identity.0;
        Ok(self)
    }

    pub(super) fn local_gradient(
        &self,
        state: &super::SpeciesState,
        gradient: &mut [Vec3],
    ) -> Result<()> {
        let density = state.scf.p_total();
        for term in &self.local_terms {
            let (c, g) = term.amplitudes(&state.coordinates)?;
            let center_gradient = std::array::from_fn::<_, 3, _>(|axis| {
                c * density.frobenius_dot(&term.potential.center_derivatives[axis])
                    + g * term.steric.center_gradient.get(axis)
            });
            for (atom, value) in gradient.iter_mut().enumerate() {
                let mut derivative = [0.0; 3];
                for axis in 0..3 {
                    derivative[axis] = c * density
                        .frobenius_dot(&term.potential.atom_derivatives[atom][axis])
                        + g * term.steric.atom_gradients[atom].get(axis)
                        + (0..3)
                            .map(|i| center_gradient[i] * term.frame.atom_jacobians[atom][i][axis])
                            .sum::<f64>();
                }
                *value += Vec3::new(derivative[0], derivative[1], derivative[2]);
            }
        }
        Ok(())
    }
}
