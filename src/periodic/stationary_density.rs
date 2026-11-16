// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Borrowed spin densities and total Lagrangian multipliers for analytic forces.
//! This is internal contraction data, not a claim of unrestricted stationarity.

use super::bloch::BlochMatrix;
use super::scf::PeriodicScfResult;
use crate::math::Vec3;

pub(crate) struct StationaryDensity<'a> {
    pub k_points: &'a [Vec3],
    pub density: &'a [Vec<BlochMatrix>; 2],
    pub energy_weighted_density: &'a [Vec<BlochMatrix>; 2],
    pub distinct_spin_channels: bool,
}

impl<'a> StationaryDensity<'a> {
    pub fn from_scf(result: &'a PeriodicScfResult) -> Self {
        Self {
            k_points: &result.k_points,
            density: &result.density,
            energy_weighted_density: &result.energy_weighted_density,
            distinct_spin_channels: result.unrestricted,
        }
    }

    pub fn total_density(&self) -> Vec<BlochMatrix> {
        self.density[0]
            .iter()
            .zip(&self.density[1])
            .map(|(a, b)| {
                let mut total = a.clone();
                total.add_scaled(b, 1.0);
                total
            })
            .collect()
    }
}
