// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Functional Taylor input: spin-resolved rho, sigma and tau in the same
//! ordering and conventions as GridVars. This is the semilocal energy
//! density only; Hartree, exchange and nonlocal terms belong to their own
//! physical response backends.

use crate::dual::{DualN, Scalar};
use crate::response::series::{evaluate_series, evaluate_series_many, SeriesScalar};
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct XcTaylorInput {
    /// rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b.
    pub variables: [Vec<f64>; 7],
    /// Optional parameter Taylor arrays, in the functional's params order.
    pub parameters: Option<Vec<Vec<f64>>>,
}

/// Energy and seven first partial derivatives, each evaluated along the
/// supplied density/parameter path to the full requested Taylor order.
#[derive(Clone, Debug)]
pub struct XcTaylorOutput {
    pub energy: Vec<f64>,
    pub potentials: [Vec<f64>; 7],
}

/// A potential path and its exact mixed directional derivative. The direction
/// may itself vary along the Taylor path. It is independent of that path,
/// rather than a finite displacement of its leading coefficient.
#[derive(Clone, Debug)]
pub struct XcTaylorDirectionalOutput {
    pub value: XcTaylorOutput,
    pub directional: XcTaylorOutput,
}

/// Seven density-variable tangents, one independent directional tangent,
/// all over the supplied arbitrary-order Taylor scalar.
pub type DirectionalTaylorScalar = ActiveDirectionalTaylorScalar<7>;

/// Only the functional body's active descriptor prefix is differentiated.
pub(crate) type ActiveDirectionalTaylorScalar<const N: usize> = DualN<DualN<SeriesScalar, N>, 1>;

impl XcTaylorOutput {
    pub fn zeros(width: usize) -> Self {
        Self {
            energy: vec![0.0; width],
            potentials: std::array::from_fn(|_| vec![0.0; width]),
        }
    }
}

impl XcTaylorInput {
    /// Evaluate an external scalar-generic functional body using the shared
    /// mixed-directional/Taylor machinery and optional parameter paths.
    pub fn evaluate_directional_potentials(
        &self,
        direction: &[Vec<f64>; 7],
        defaults: &[f64],
        expression: impl FnOnce(
            &[DirectionalTaylorScalar],
            &[DirectionalTaylorScalar],
        ) -> DirectionalTaylorScalar,
    ) -> Result<XcTaylorDirectionalOutput> {
        self.evaluate_directional_potentials_for::<7>(direction, defaults, expression)
    }

    pub(crate) fn evaluate_directional_potentials_for<const N: usize>(
        &self,
        direction: &[Vec<f64>; 7],
        defaults: &[f64],
        expression: impl FnOnce(
            &[ActiveDirectionalTaylorScalar<N>],
            &[ActiveDirectionalTaylorScalar<N>],
        ) -> ActiveDirectionalTaylorScalar<N>,
    ) -> Result<XcTaylorDirectionalOutput> {
        self.validate()?;
        let width = self.order() + 1;
        if !(2..=7).contains(&N) {
            return Err(DftError::InvalidInput(
                "active descriptor count must be two through seven".into(),
            ));
        }
        if direction
            .iter()
            .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
        {
            return Err(DftError::InvalidInput(
                "XC potential directions must have finite matching Taylor coefficients".into(),
            ));
        }
        let parameters = match &self.parameters {
            Some(p) if p.len() == defaults.len() => p.clone(),
            Some(_) => {
                return Err(DftError::InvalidInput(
                    "XC Taylor parameter count does not match the functional".into(),
                ))
            }
            None => defaults
                .iter()
                .map(|&p| {
                    let mut v = vec![0.0; width];
                    v[0] = p;
                    v
                })
                .collect(),
        };
        let inputs: Vec<_> = self.variables[..N]
            .iter()
            .chain(&direction[..N])
            .cloned()
            .chain(parameters)
            .collect();
        let outputs = evaluate_series_many(&inputs, |values| {
            let variables: Vec<_> = (0..N)
                .map(|i| ActiveDirectionalTaylorScalar::<N> {
                    re: DualN::variable(values[i], i),
                    im: [DualN::constant(values[i + N])],
                })
                .collect();
            let parameters: Vec<_> = values[2 * N..]
                .iter()
                .map(|&v| ActiveDirectionalTaylorScalar::<N>::constant(DualN::constant(v)))
                .collect();
            let e = if values[0].value() + values[1].value() < 1e-13 {
                ActiveDirectionalTaylorScalar::<N>::from_f64(0.0)
            } else {
                expression(&variables, &parameters)
            };
            std::iter::once(e.re.re)
                .chain(e.re.im)
                .chain(std::iter::once(e.im[0].re))
                .chain(e.im[0].im)
                .collect()
        })?;
        Ok(XcTaylorDirectionalOutput {
            value: XcTaylorOutput {
                energy: outputs[0].clone(),
                potentials: std::array::from_fn(|i| {
                    if i < N {
                        outputs[i + 1].clone()
                    } else {
                        vec![0.0; width]
                    }
                }),
            },
            directional: XcTaylorOutput {
                energy: outputs[N + 1].clone(),
                potentials: std::array::from_fn(|i| {
                    if i < N {
                        outputs[i + N + 2].clone()
                    } else {
                        vec![0.0; width]
                    }
                }),
            },
        })
    }
    pub fn linear(values: [f64; 7], directions: [f64; 7], order: usize) -> Result<Self> {
        if order == usize::MAX {
            return Err(DftError::InvalidInput("Taylor order overflow".into()));
        }
        let variables = std::array::from_fn(|i| {
            let mut v = vec![0.0; order + 1];
            v[0] = values[i];
            if order > 0 {
                v[1] = directions[i];
            }
            v
        });
        let result = Self {
            variables,
            parameters: None,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn order(&self) -> usize {
        self.variables[0].len().saturating_sub(1)
    }
    pub fn validate(&self) -> Result<()> {
        let width = self.variables[0].len();
        if width == 0
            || self
                .variables
                .iter()
                .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
            || self.parameters.as_ref().is_some_and(|p| {
                p.iter()
                    .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
            })
        {
            return Err(DftError::InvalidInput(
                "XC Taylor arrays must be finite, nonempty and have equal lengths".into(),
            ));
        }
        if self.variables[0][0] < 0.0 || self.variables[1][0] < 0.0 {
            return Err(DftError::InvalidInput(
                "XC Taylor base densities must be non-negative".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn evaluate(
        &self,
        defaults: &[f64],
        expression: impl FnOnce(&[SeriesScalar], &[SeriesScalar]) -> SeriesScalar,
    ) -> Result<Vec<f64>> {
        self.validate()?;
        let width = self.order() + 1;
        let parameters = match &self.parameters {
            Some(p) if p.len() == defaults.len() => p.clone(),
            Some(_) => {
                return Err(DftError::InvalidInput(
                    "XC Taylor parameter count does not match the functional".into(),
                ))
            }
            None => defaults
                .iter()
                .map(|&p| {
                    let mut v = vec![0.0; width];
                    v[0] = p;
                    v
                })
                .collect(),
        };
        let inputs: Vec<_> = self.variables.iter().cloned().chain(parameters).collect();
        evaluate_series(&inputs, |values| expression(&values[..7], &values[7..]))
    }

    pub(crate) fn evaluate_potentials(
        &self,
        defaults: &[f64],
        expression: impl FnOnce(
            &[DualN<SeriesScalar, 7>],
            &[DualN<SeriesScalar, 7>],
        ) -> DualN<SeriesScalar, 7>,
    ) -> Result<XcTaylorOutput> {
        self.evaluate_potentials_for::<7>(defaults, expression)
    }

    pub(crate) fn evaluate_potentials_for<const N: usize>(
        &self,
        defaults: &[f64],
        expression: impl FnOnce(
            &[DualN<SeriesScalar, N>],
            &[DualN<SeriesScalar, N>],
        ) -> DualN<SeriesScalar, N>,
    ) -> Result<XcTaylorOutput> {
        self.validate()?;
        let width = self.order() + 1;
        if !(2..=7).contains(&N) {
            return Err(DftError::InvalidInput(
                "active descriptor count must be two through seven".into(),
            ));
        }
        let parameters = match &self.parameters {
            Some(p) if p.len() == defaults.len() => p.clone(),
            Some(_) => {
                return Err(DftError::InvalidInput(
                    "XC Taylor parameter count does not match the functional".into(),
                ))
            }
            None => defaults
                .iter()
                .map(|&p| {
                    let mut v = vec![0.0; width];
                    v[0] = p;
                    v
                })
                .collect(),
        };
        let inputs: Vec<_> = self.variables[..N]
            .iter()
            .cloned()
            .chain(parameters)
            .collect();
        let outputs = evaluate_series_many(&inputs, |values| {
            let variables: Vec<_> = values[..N]
                .iter()
                .enumerate()
                .map(|(i, &v)| DualN::<SeriesScalar, N>::variable(v, i))
                .collect();
            let parameters: Vec<_> = values[N..]
                .iter()
                .copied()
                .map(DualN::<SeriesScalar, N>::constant)
                .collect();
            let e = if values[0].value() + values[1].value() < 1e-13 {
                DualN::<SeriesScalar, N>::constant(SeriesScalar::from_f64(0.0))
            } else {
                expression(&variables, &parameters)
            };
            std::iter::once(e.re).chain(e.im).collect()
        })?;
        Ok(XcTaylorOutput {
            energy: outputs[0].clone(),
            potentials: std::array::from_fn(|i| {
                if i < N {
                    outputs[i + 1].clone()
                } else {
                    vec![0.0; width]
                }
            }),
        })
    }
}

#[cfg(test)]
#[path = "taylor_tests.rs"]
mod tests;
