// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Bounded local interpolation of the direct Dion momentum quadrature.

use super::PairKernelQuadrature;
use crate::{DftError, Result};
use rayon::prelude::*;

/// Cubically interpolate the symmetric pair kernel in the logarithms of the
/// two dimensionless distances. The samples are generated from the direct
/// momentum quadrature; no external parameter table is needed.
pub struct PairKernelTable {
    minimum: f64,
    maximum: f64,
    log_minimum: f64,
    log_step: f64,
    points: usize,
    values: Vec<f64>,
}

impl PairKernelTable {
    /// Tabulate the kernel on the specified finite interval of positive
    /// distances. Independently increase the momentum cutoff and table size.
    pub fn new(
        kernel: &PairKernelQuadrature,
        minimum: f64,
        maximum: f64,
        points: usize,
    ) -> Result<Self> {
        if !minimum.is_finite()
            || minimum <= 0.0
            || !maximum.is_finite()
            || maximum <= minimum
            || !(4..=384).contains(&points)
        {
            return Err(DftError::InvalidInput(
                "vdW-DF table needs positive ordered bounds and 4..=384 points".into(),
            ));
        }
        let log_minimum = minimum.ln();
        let log_step = (maximum.ln() - log_minimum) / (points - 1) as f64;
        if !log_step.is_finite() || log_step <= 0.0 {
            return Err(DftError::InvalidInput(
                "vdW-DF table bounds must remain distinct on a logarithmic scale".into(),
            ));
        }
        let distances: Vec<_> = (0..points)
            .map(|i| (log_minimum + i as f64 * log_step).exp())
            .collect();
        let values: Vec<Vec<f64>> = distances
            .par_iter()
            .enumerate()
            .map(|(i, &d)| {
                distances[..=i]
                    .iter()
                    .map(|&dp| kernel.evaluate(d, dp))
                    .collect::<Result<_>>()
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            minimum,
            maximum,
            log_minimum,
            log_step,
            points,
            values: values.into_iter().flatten().collect(),
        })
    }

    /// Interpolate the pair energy in Hartree within the represented domain.
    /// The caller can use the direct quadrature outside this domain.
    pub fn interpolate(&self, d: f64, dp: f64) -> Option<f64> {
        let in_domain = |x| (self.minimum..=self.maximum).contains(&x);
        if !in_domain(d) || !in_domain(dp) {
            return None;
        }
        // Canonicalize the order so exchanging endpoints retains the same
        // summation order and gives the same rounded result.
        let (i, a) = self.coefficients(d.min(dp));
        let (j, b) = self.coefficients(d.max(dp));
        let mut value = 0.0;
        for (u, left) in a.into_iter().enumerate() {
            for (v, right) in b.into_iter().enumerate() {
                let lower = (i + u).min(j + v);
                let upper = (i + u).max(j + v);
                value += left * right * self.values[upper * (upper + 1) / 2 + lower];
            }
        }
        Some(value)
    }

    fn coefficients(&self, distance: f64) -> (usize, [f64; 4]) {
        let coordinate = (distance.ln() - self.log_minimum) / self.log_step;
        let start = (coordinate.floor() as usize)
            .saturating_sub(1)
            .min(self.points - 4);
        let x = coordinate - start as f64;
        (
            start,
            [
                -(x - 1.0) * (x - 2.0) * (x - 3.0) / 6.0,
                x * (x - 2.0) * (x - 3.0) / 2.0,
                -x * (x - 1.0) * (x - 3.0) / 2.0,
                x * (x - 1.0) * (x - 2.0) / 6.0,
            ],
        )
    }
}
