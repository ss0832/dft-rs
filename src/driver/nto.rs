// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::{Controls, Value};
use crate::periodic::BlochMatrix;
use crate::properties::natural_transition_orbitals::{columns, decompose, real};
use crate::{linalg::Matrix, Result};

#[derive(Clone, Copy, Default)]
pub(super) struct Request {
    nto: bool,
    difference: bool,
}

impl Request {
    pub(super) fn from_controls(controls: &Controls) -> Result<Self> {
        Ok(Self {
            nto: controls.bool("td_nto", false)?,
            difference: controls.bool("td_difference_density", false)?,
        })
    }

    pub(super) fn enabled(self) -> bool {
        self.nto || self.difference
    }

    pub(super) fn needs_difference(self) -> bool {
        self.difference
    }

    pub(super) fn attach(self, state: &mut Value, channels: Vec<(Value, Value)>) {
        let (transition, difference): (Vec<_>, Vec<_>) = channels.into_iter().unzip();
        if self.nto {
            state.set("natural_transition_orbitals", transition);
        }
        if self.difference {
            state.set("natural_difference_orbitals", difference);
        }
    }
}

pub(super) fn channel(
    transition: &BlochMatrix,
    particles: &BlochMatrix,
    holes: &BlochMatrix,
    name: &str,
    transition_scale: f64,
    request: Request,
) -> Result<(Value, Value)> {
    let r = decompose(transition, particles, holes)?;
    let weights: Vec<_> = r
        .singular_values
        .iter()
        .map(|sigma| sigma * sigma)
        .collect();
    let common = Value::object()
        .with("channel", name)
        .with("particle_coefficients_real", r.particle_coefficients.re)
        .with(
            "particle_coefficients_imaginary",
            r.particle_coefficients.im,
        )
        .with("hole_coefficients_real", r.hole_coefficients.re)
        .with("hole_coefficients_imaginary", r.hole_coefficients.im);
    let transition_report = if request.nto {
        let scale_squared = transition_scale.powi(2);
        common
            .clone()
            .with(
                "weights",
                weights
                    .iter()
                    .map(|w| w * scale_squared)
                    .collect::<Vec<_>>(),
            )
            .with(
                "singular_values",
                r.singular_values
                    .iter()
                    .map(|w| w * transition_scale)
                    .collect::<Vec<_>>(),
            )
            .with("transition_weight", r.transition_weight * scale_squared)
            .with(
                "reconstruction_residual",
                r.reconstruction_residual * transition_scale,
            )
    } else {
        Value::Null
    };
    let difference_report = if request.difference {
        // In CIS/TDA the unrelaxed excess and deficit blocks are
        // T T^H and -T^H T. A normalized rank-one excitation transfers
        // one electron, including the spin-adapted restricted channels.
        let attached = weights.iter().sum::<f64>();
        common
            .with("attachment_occupations", weights.clone())
            .with(
                "detachment_occupations",
                weights.iter().map(|&w| -w).collect::<Vec<_>>(),
            )
            .with("attachment_electrons", attached)
            .with("detachment_electrons", attached)
            .with(
                "model",
                "unrelaxed ground-to-TDA-state one-particle density difference",
            )
    } else {
        Value::Null
    };
    Ok((transition_report, difference_report))
}

pub(super) fn real_excitation(
    transition: &Matrix,
    orbitals: &Matrix,
    name: &str,
    scale: f64,
    request: Request,
) -> Result<(Value, Value)> {
    let transition = real(transition.clone());
    let frame = real(orbitals.clone());
    let nv = transition.rows();
    let no = transition.cols();
    channel(
        &transition,
        &columns(&frame, no, nv),
        &columns(&frame, 0, no),
        name,
        scale,
        request,
    )
}

pub(super) fn add_units(units: &mut Value, request: Request) {
    for key in [
        "states.natural_transition_orbitals.weights",
        "states.natural_transition_orbitals.singular_values",
        "states.natural_transition_orbitals.transition_weight",
        "states.natural_transition_orbitals.reconstruction_residual",
        "states.natural_transition_orbitals.particle_coefficients_real",
        "states.natural_transition_orbitals.particle_coefficients_imaginary",
        "states.natural_transition_orbitals.hole_coefficients_real",
        "states.natural_transition_orbitals.hole_coefficients_imaginary",
    ]
    .into_iter()
    .filter(|_| request.nto)
    {
        units.set(key, "dimensionless");
    }
    if request.difference {
        for (key, unit) in [
            ("attachment_occupations", "electrons"),
            ("detachment_occupations", "electrons"),
            ("attachment_electrons", "electrons"),
            ("detachment_electrons", "electrons"),
            ("particle_coefficients_real", "dimensionless"),
            ("particle_coefficients_imaginary", "dimensionless"),
            ("hole_coefficients_real", "dimensionless"),
            ("hole_coefficients_imaginary", "dimensionless"),
        ] {
            units.set(&format!("states.natural_difference_orbitals.{key}"), unit);
        }
    }
}
