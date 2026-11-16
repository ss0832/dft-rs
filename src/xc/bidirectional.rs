// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Two independent nilpotent density directions over a nuclear Taylor path.
//! This is analytic mixed differentiation, without a finite displacement or
//! a higher-order univariate encoding that asks for unnecessary derivatives.

use super::taylor::{XcTaylorInput, XcTaylorOutput};
use crate::dual::{DualN, Scalar};
use crate::response::series::{evaluate_series_many, SeriesScalar};
use crate::{DftError, Result};

pub type BidirectionalTaylorScalar = DualN<DualN<DualN<SeriesScalar, 7>, 1>, 1>;

#[derive(Clone, Debug)]
pub struct XcTaylorBidirectionalOutput {
    pub value: XcTaylorOutput,
    pub first: XcTaylorOutput,
    pub second: XcTaylorOutput,
    pub mixed: XcTaylorOutput,
}

impl XcTaylorBidirectionalOutput {
    pub fn zeros(width: usize) -> Self {
        Self {
            value: XcTaylorOutput::zeros(width),
            first: XcTaylorOutput::zeros(width),
            second: XcTaylorOutput::zeros(width),
            mixed: XcTaylorOutput::zeros(width),
        }
    }
}

impl super::ResolvedFunctional {
    /// Weighted semilocal composition of two independent mixed tangent
    /// paths. Exact exchange, nonlocal tails and PT2 are separate backends.
    pub fn semilocal_taylor_bidirectional_potentials(
        &self,
        input: &XcTaylorInput,
        first: &[Vec<f64>; 7],
        second: &[Vec<f64>; 7],
        mixed: &[Vec<f64>; 7],
    ) -> Result<XcTaylorBidirectionalOutput> {
        input.validate()?;
        let width = input.order() + 1;
        if input.parameters.is_some() {
            return Err(DftError::InvalidInput(
                "composite parameter derivatives require per-part Taylor inputs".into(),
            ));
        }
        if first
            .iter()
            .chain(second)
            .chain(mixed)
            .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
        {
            return Err(DftError::InvalidInput(
                "bidirectional XC tangents require finite matching Taylor coefficients".into(),
            ));
        }
        let mut result = XcTaylorBidirectionalOutput::zeros(width);
        for (weight, part) in &self.parts {
            let output = part.taylor_bidirectional_potentials(input, first, second, mixed)?;
            for (target, source) in [
                &mut result.value,
                &mut result.first,
                &mut result.second,
                &mut result.mixed,
            ]
            .into_iter()
            .zip([output.value, output.first, output.second, output.mixed])
            {
                if source.energy.len() != width
                    || source.potentials.iter().any(|v| v.len() != width)
                {
                    return Err(DftError::InvalidInput(
                        "functional mixed Taylor output has an incompatible order".into(),
                    ));
                }
                for (a, b) in target.energy.iter_mut().zip(source.energy) {
                    *a += weight * b;
                }
                for (target, source) in target.potentials.iter_mut().zip(source.potentials) {
                    for (a, b) in target.iter_mut().zip(source) {
                        *a += weight * b;
                    }
                }
            }
        }
        Ok(result)
    }
}

impl XcTaylorInput {
    /// Evaluate at x(t)+u a(t)+v b(t)+uv c(t), extracting value, u, v and
    /// uv energy/potential paths. `mixed` is the independent input c(t);
    /// it includes a changing spin-scaling tangent when needed. Nuclear
    /// coefficients are factorial normalized; u/v coefficients equal their
    /// mixed partial derivatives because both directions are nilpotent.
    /// Potential paths of order N require scalar-body derivatives through
    /// N+3. Parameter paths have no independent u/v perturbation here.
    pub fn evaluate_bidirectional_potentials(
        &self,
        first: &[Vec<f64>; 7],
        second: &[Vec<f64>; 7],
        mixed: &[Vec<f64>; 7],
        defaults: &[f64],
        expression: impl FnOnce(
            &[BidirectionalTaylorScalar],
            &[BidirectionalTaylorScalar],
        ) -> BidirectionalTaylorScalar,
    ) -> Result<XcTaylorBidirectionalOutput> {
        self.validate()?;
        let width = self.order() + 1;
        if first
            .iter()
            .chain(second)
            .chain(mixed)
            .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
        {
            return Err(DftError::InvalidInput(
                "bidirectional XC tangents require finite matching Taylor coefficients".into(),
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
        let inputs = self
            .variables
            .iter()
            .chain(first)
            .chain(second)
            .chain(mixed)
            .cloned()
            .chain(parameters)
            .collect::<Vec<_>>();
        let outputs = evaluate_series_many(&inputs, |values| {
            let variables = (0..7)
                .map(|i| BidirectionalTaylorScalar {
                    re: DualN {
                        re: DualN::variable(values[i], i),
                        im: [DualN::constant(values[i + 7])],
                    },
                    im: [DualN {
                        re: DualN::constant(values[i + 14]),
                        im: [DualN::constant(values[i + 21])],
                    }],
                })
                .collect::<Vec<_>>();
            let parameters = values[28..]
                .iter()
                .map(|&p| BidirectionalTaylorScalar::constant(DualN::constant(DualN::constant(p))))
                .collect::<Vec<_>>();
            let e = if values[0].value() + values[1].value() < 1e-13 {
                BidirectionalTaylorScalar::from_f64(0.0)
            } else {
                expression(&variables, &parameters)
            };
            [e.re.re, e.re.im[0], e.im[0].re, e.im[0].im[0]]
                .into_iter()
                .flat_map(|block| std::iter::once(block.re).chain(block.im))
                .collect()
        })?;
        let block = |offset: usize| XcTaylorOutput {
            energy: outputs[offset].clone(),
            potentials: std::array::from_fn(|i| outputs[offset + 1 + i].clone()),
        };
        Ok(XcTaylorBidirectionalOutput {
            value: block(0),
            first: block(8),
            second: block(16),
            mixed: block(24),
        })
    }
}
