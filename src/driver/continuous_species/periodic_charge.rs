// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Classical Ewald contract for a neutral fixed-cell real-charge chart.
//! This module implements no periodic electronic SCF, force or response route.

use crate::driver::continuous_species::ChargeChart;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::ewald::{ewald_energy, ewald_potential, EwaldParams};
use crate::periodic::Lattice;
use crate::{DftError, Result};

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}
fn finite(point: Vec3) -> bool {
    (0..3).all(|axis| point.get(axis).is_finite())
}

/// Integer identities are stored independently of the physical real charges.
/// The cached matrix contains real, reciprocal, self and background terms.
pub struct NeutralEwaldChart {
    lattice: Lattice,
    positions: Vec<Vec3>,
    atomic_numbers: Vec<u8>,
    fixed_electrons: usize,
    chart: ChargeChart,
    params: EwaldParams,
    charge_matrix: Matrix,
    hessian: Matrix,
    raw_matrix_asymmetry: f64,
}

pub struct EwaldChartEvaluation {
    /// Ewald source energy per primitive cell, not the total DFT energy.
    pub energy: f64,
    pub physical_charges: Vec<f64>,
    pub physical_total_charge: f64,
    pub parameter_gradient: Vec<f64>,
    pub parameter_hessian: Matrix,
    pub raw_matrix_asymmetry: f64,
}

impl NeutralEwaldChart {
    pub fn new(
        lattice: Lattice,
        positions: Vec<Vec3>,
        atomic_numbers: Vec<u8>,
        fixed_electrons: usize,
        chart: ChargeChart,
        params: EwaldParams,
    ) -> Result<Self> {
        chart.validate(positions.len())?;
        let n = positions.len();
        if n == 0
            || atomic_numbers.len() != n
            || atomic_numbers.contains(&0)
            || fixed_electrons == 0
            || !chart.conserve_total_charge
            || positions.iter().any(|&p| !finite(p))
            || lattice.a.iter().any(|&v| !finite(v))
            || !lattice.volume().is_finite()
            || lattice.volume().abs() <= 1e-12
            || [params.alpha, params.rcut, params.gcut]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return invalid("neutral Ewald charts require fixed positive integer labels/electrons, finite distinct sources, a nonsingular cell and positive Ewald parameters");
        }
        let total: f64 = chart.reference_charges.iter().sum();
        if (total - fixed_electrons as f64).abs() > 64.0 * f64::EPSILON * (fixed_electrons as f64) {
            return invalid(
                "Ewald chart reference must be physically neutral at its fixed electron number",
            );
        }
        for i in 0..n {
            for j in 0..i {
                if lattice.minimum_image(positions[i] - positions[j]).norm() < 1e-8 {
                    return invalid("periodic charge sites coincide modulo the lattice");
                }
            }
        }
        // Every charge is strictly positive throughout the declared box.
        for atom in 0..n {
            let minimum = chart.reference_charges[atom]
                + chart
                    .parameters
                    .iter()
                    .map(|p| {
                        (p.charge_direction[atom] * p.bounds[0])
                            .min(p.charge_direction[atom] * p.bounds[1])
                    })
                    .sum::<f64>();
            if !minimum.is_finite() || minimum <= 0.0 {
                return invalid("Ewald chart bounds leave the positive real-charge domain");
            }
        }
        let mut a = Matrix::zeros(n, n);
        for j in 0..n {
            let mut unit = vec![0.0; n];
            unit[j] = 1.0;
            for i in 0..n {
                a[(i, j)] = ewald_potential(&lattice, &positions, &unit, positions[i], params);
            }
        }
        if a.as_slice().iter().any(|x| !x.is_finite()) {
            return invalid("Ewald matrix contains a nonfinite entry");
        }
        let mut asymmetry = 0.0_f64;
        for i in 0..n {
            for j in 0..i {
                asymmetry = asymmetry.max((a[(i, j)] - a[(j, i)]).abs());
            }
        }
        let scale = a.as_slice().iter().map(|x| x.abs()).fold(1.0, f64::max);
        if asymmetry > 256.0 * f64::EPSILON * scale * n as f64 {
            return invalid("Ewald potential matrix is not numerically symmetric; no averaging repair is applied");
        }
        let p = chart.parameters.len();
        let mut hessian = Matrix::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                for atom in 0..n {
                    for other in 0..n {
                        hessian[(i, j)] += chart.parameters[i].charge_direction[atom]
                            * a[(atom, other)]
                            * chart.parameters[j].charge_direction[other];
                    }
                }
            }
        }
        Ok(Self {
            lattice,
            positions,
            atomic_numbers,
            fixed_electrons,
            chart,
            params,
            charge_matrix: a,
            hessian,
            raw_matrix_asymmetry: asymmetry,
        })
    }

    pub fn atomic_numbers(&self) -> &[u8] {
        &self.atomic_numbers
    }
    pub fn fixed_electrons(&self) -> usize {
        self.fixed_electrons
    }
    pub fn charge_matrix(&self) -> &Matrix {
        &self.charge_matrix
    }
    pub fn chart(&self) -> &ChargeChart {
        &self.chart
    }
    pub fn scope(&self) -> &'static str {
        "classical-neutral-fixed-cell-ewald-only"
    }

    pub fn evaluate(&self, q: &[f64]) -> Result<EwaldChartEvaluation> {
        let z = self.chart.charges(q)?;
        let net = z.iter().sum::<f64>() - self.fixed_electrons as f64;
        if net.abs() > 128.0 * f64::EPSILON * self.fixed_electrons as f64 {
            return invalid("real-charge chart violates its neutral fixed-electron sector");
        }
        let potential: Vec<f64> = (0..z.len())
            .map(|i| {
                (0..z.len())
                    .map(|j| self.charge_matrix[(i, j)] * z[j])
                    .sum()
            })
            .collect();
        let energy = 0.5 * z.iter().zip(&potential).map(|(q, v)| q * v).sum::<f64>();
        let gradient = self
            .chart
            .parameters
            .iter()
            .map(|p| {
                p.charge_direction
                    .iter()
                    .zip(&potential)
                    .map(|(b, v)| b * v)
                    .sum()
            })
            .collect();
        Ok(EwaldChartEvaluation {
            energy,
            physical_charges: z,
            physical_total_charge: net,
            parameter_gradient: gradient,
            parameter_hessian: self.hessian.clone(),
            raw_matrix_asymmetry: self.raw_matrix_asymmetry,
        })
    }

    /// Independent direct Ewald energy, retained as an oracle/diagnostic.
    pub fn direct_energy(&self, q: &[f64]) -> Result<f64> {
        let state = self.evaluate(q)?;
        Ok(ewald_energy(
            &self.lattice,
            &self.positions,
            &state.physical_charges,
            self.params,
        ))
    }
}
