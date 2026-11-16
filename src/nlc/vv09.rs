// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! VV09 nonlocal correlation in direct real space, Vydrov and Van Voorhis,
//! Phys. Rev. Lett. 103, 063004 (2009), Eqs. (8)–(17).
//! The full VV09 correlation adds PW92 local spin-density correlation.

use crate::basis::MolBasis;
use crate::grid::{eval::eval_aos_full, MolecularGrid};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::{DftError, Result};
use rayon::prelude::*;

const GAP: f64 = 0.0089;

// Scaled damping at coincidence. B(K)/K^3 is a Gaussian moment on
// [0, 1]. Its short series avoids cancellation between erf(K) and A(K).
fn damping_over_r6(r2: f64, k2_over_r2: f64) -> f64 {
    let k2 = r2 * k2_over_r2;
    let a_over_k = 2.0 * libm::exp(-k2) / std::f64::consts::PI.sqrt();
    if k2 < 0.25 {
        let b_over_k3 = crate::math::scaled_erf_difference(k2.sqrt());
        k2_over_r2.powi(3) * b_over_k3 * ((4.0 / 3.0) * a_over_k - b_over_k3)
    } else {
        let k = k2.sqrt();
        let a = k * a_over_k;
        let b = libm::erf(k) - a;
        b * ((4.0 / 3.0) * k2 * a - b) / r2.powi(3)
    }
}

#[derive(Clone, Copy)]
struct Endpoint {
    position: Vec3,
    weighted_rho: f64,
    omega: f64,
    screening: f64,
}

impl Endpoint {
    fn pair(self, partner: Self) -> f64 {
        let r2 = (self.position - partner.position).norm2();
        let k2_over_r2 =
            0.25 * self.screening * partner.screening / (self.screening + partner.screening);
        1.5 * (self.weighted_rho / self.omega)
            * (partner.weighted_rho / partner.omega)
            * damping_over_r6(r2, k2_over_r2)
            / (self.omega + partner.omega)
    }
}

/// VV09 nonlocal correlation in Hartree on a supplied molecular quadrature.
/// `sigma` is the squared gradient of the total density and `polarization`
/// is `(rho_up-rho_down)/rho` or the magnitude of the spin-density vector
/// divided by `rho`. Vacuum points use the same density cutoff as VV10.
pub fn energy_on_grid(
    grid: &MolecularGrid,
    rho: &[f64],
    sigma: &[f64],
    polarization: &[f64],
) -> Result<f64> {
    let n = grid.len();
    if [
        rho.len(),
        sigma.len(),
        polarization.len(),
        grid.weights.len(),
    ]
    .into_iter()
    .any(|len| len != n)
        || rho.iter().chain(sigma).any(|v| !v.is_finite() || *v < 0.0)
        || polarization.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
        || grid.weights.iter().any(|v| !v.is_finite())
        || grid.points.iter().any(|v| !v.norm2().is_finite())
    {
        return Err(DftError::InvalidInput(
            "invalid VV09 molecular grid density or spin polarization".into(),
        ));
    }
    let pi = std::f64::consts::PI;
    let active: Vec<_> = (0..n)
        .filter(|&i| rho[i] >= super::RHO_CUT)
        .map(|i| {
            let z = polarization[i];
            let phi = 0.5 * ((1.0 + z).powf(2.0 / 3.0) + (1.0 - z).powf(2.0 / 3.0));
            let kf = (3.0 * pi * pi * rho[i]).cbrt();
            Endpoint {
                position: grid.points[i],
                weighted_rho: grid.weights[i] * rho[i],
                omega: super::local_frequency(rho[i], sigma[i], GAP),
                screening: 4.0 * kf * phi * phi / pi,
            }
        })
        .collect();
    // Integrate each off-diagonal pair once. The diagonal has half weight
    // in the symmetric double integral and uses the analytic coincidence limit.
    let energy = active
        .par_iter()
        .enumerate()
        .map(|(i, &point)| {
            0.5 * point.pair(point)
                + active[..i]
                    .iter()
                    .map(|&partner| point.pair(partner))
                    .sum::<f64>()
        })
        .sum::<f64>();
    if !energy.is_finite() {
        return Err(DftError::InvalidInput(
            "VV09 pair energy is nonfinite".into(),
        ));
    }
    Ok(energy)
}

/// Nonlocal VV09 energy in Hartree at fixed AO density and magnetic density.
/// Supply no spin matrix for a restricted reference, the single magnetization
/// matrix for a collinear reference, or all three real Pauli matrices for
/// a general spinor density. Use PW92 local correlation for the full VV09 model.
pub fn energy_on_density(
    basis: &MolBasis,
    grid: &MolecularGrid,
    density: &Matrix,
    spin_density: &[&Matrix],
) -> Result<f64> {
    let nao = basis.nao_sph;
    if !matches!(spin_density.len(), 0 | 1 | 3)
        || std::iter::once(density)
            .chain(spin_density.iter().copied())
            .any(|m| m.rows != nao || m.cols != nao || m.as_slice().iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "VV09 needs a finite total AO matrix and zero, one, or three spin-density matrices"
                .into(),
        ));
    }
    let mut rho = Vec::with_capacity(grid.len());
    let mut sigma = Vec::with_capacity(grid.len());
    let mut polarization = Vec::with_capacity(grid.len());
    for points in grid.points.chunks(512) {
        let ao = eval_aos_full(basis, points, true, false);
        let grad = ao.grad.as_ref().expect("AO derivatives were requested");
        let dp = ao.phi.matmul_seq(density);
        let sp: Vec<_> = spin_density.iter().map(|p| ao.phi.matmul_seq(p)).collect();
        for i in 0..points.len() {
            let product = |left: &Matrix, right: &Matrix| {
                (0..nao)
                    .map(|mu| left[(i, mu)] * right[(i, mu)])
                    .sum::<f64>()
            };
            let n = product(&ao.phi, &dp).max(0.0);
            rho.push(n);
            sigma.push(
                grad.iter()
                    .map(|derivative| (2.0 * product(derivative, &dp)).powi(2))
                    .sum(),
            );
            let magnetization = sp
                .iter()
                .map(|p| product(&ao.phi, p).powi(2))
                .sum::<f64>()
                .sqrt();
            polarization.push(if n < super::RHO_CUT {
                0.0
            } else {
                (magnetization / n).min(1.0)
            });
        }
    }
    energy_on_grid(grid, &rho, &sigma, &polarization)
}
