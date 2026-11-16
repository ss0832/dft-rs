// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Third-order Birch-Murnaghan equation of state.
//!
//! The Birch-Murnaghan energy is a cubic polynomial in the Eulerian strain
//! measure `(V0/V)^(2/3)`, so with `s = (V/V_r)^(-2/3)` for a reference
//! volume `V_r` the fit `E(V) = a0 + a1 s + a2 s^2 + a3 s^3` is linear least
//! squares and exactly the third-order form. With `P_k = d^k E/ds^k` and
//! `s' = -2s/(3V)`, `s'' = 10s/(9V^2)`, `s''' = -80s/(27V^3)`,
//!
//! ```text
//! E_V   = P1 s'
//! E_VV  = P2 s'^2 + P1 s''
//! E_VVV = P3 s'^3 + 3 P2 s' s'' + P1 s'''
//! p = -E_V,  B = V E_VV,  B' = dB/dp = -1 - V E_VVV / E_VV
//! ```
//!
//! and the equilibrium volume solves `P1(s) = 0`.

use crate::error::{DftError, Result};

/// A fitted third-order Birch-Murnaghan equation of state (atomic units).
#[derive(Clone, Debug)]
pub struct BirchMurnaghan {
    /// Reference volume of the strain measure (Bohr^3).
    pub reference_volume: f64,
    /// Polynomial coefficients in `s = (V/V_r)^(-2/3)` (Hartree).
    pub coefficients: [f64; 4],
    /// Equilibrium volume (Bohr^3), energy (Hartree), bulk modulus
    /// (Hartree/Bohr^3) and its pressure derivative.
    pub v0: f64,
    pub e0: f64,
    pub b0: f64,
    pub b0_prime: f64,
    /// Root-mean-square energy residual of the fit (Hartree).
    pub residual_rms: f64,
}

impl BirchMurnaghan {
    fn s(&self, v: f64) -> f64 {
        (v / self.reference_volume).powf(-2.0 / 3.0)
    }

    fn polynomial(&self, s: f64) -> [f64; 4] {
        let a = self.coefficients;
        [
            a[0] + a[1] * s + a[2] * s * s + a[3] * s * s * s,
            a[1] + 2.0 * a[2] * s + 3.0 * a[3] * s * s,
            2.0 * a[2] + 6.0 * a[3] * s,
            6.0 * a[3],
        ]
    }

    /// `E`, `dE/dV`, `d2E/dV2`, `d3E/dV3` at `v`.
    pub fn derivatives(&self, v: f64) -> [f64; 4] {
        let s = self.s(v);
        let p = self.polynomial(s);
        let (s1, s2, s3) = (
            -2.0 * s / (3.0 * v),
            10.0 * s / (9.0 * v * v),
            -80.0 * s / (27.0 * v * v * v),
        );
        [
            p[0],
            p[1] * s1,
            p[2] * s1 * s1 + p[1] * s2,
            p[3] * s1 * s1 * s1 + 3.0 * p[2] * s1 * s2 + p[1] * s3,
        ]
    }

    pub fn energy(&self, v: f64) -> f64 {
        self.derivatives(v)[0]
    }

    /// `p = -dE/dV` (Hartree/Bohr^3).
    pub fn pressure(&self, v: f64) -> f64 {
        -self.derivatives(v)[1]
    }

    /// `B = V d2E/dV2` (Hartree/Bohr^3).
    pub fn bulk_modulus(&self, v: f64) -> f64 {
        v * self.derivatives(v)[2]
    }
}

/// Least-squares third-order Birch-Murnaghan fit of energies at volumes
/// (at least four distinct volumes; five or more give a residual). The
/// fitted curve must have a minimum.
pub fn fit_birch_murnaghan(volumes: &[f64], energies: &[f64]) -> Result<BirchMurnaghan> {
    let fit = fit_birch_murnaghan_isotherm(volumes, energies)?;
    if fit.v0.is_nan() {
        return Err(DftError::InvalidInput(
            "the fitted equation of state has no minimum".into(),
        ));
    }
    Ok(fit)
}

/// The same fit without the requirement of a minimum, for an isotherm that
/// is evaluated at a pressure: `v0`, `e0`, `b0` and `b0_prime` are NaN when
/// the fitted curve has no minimum.
pub fn fit_birch_murnaghan_isotherm(volumes: &[f64], energies: &[f64]) -> Result<BirchMurnaghan> {
    let n = volumes.len();
    if n < 4 || energies.len() != n || volumes.iter().any(|v| *v <= 0.0) {
        return Err(DftError::InvalidInput(
            "a Birch-Murnaghan fit needs at least four positive volumes with one energy each"
                .into(),
        ));
    }
    let reference_volume = volumes.iter().sum::<f64>() / n as f64;
    let shift = energies.iter().sum::<f64>() / n as f64;
    let s: Vec<f64> = volumes
        .iter()
        .map(|v| (v / reference_volume).powf(-2.0 / 3.0))
        .collect();
    // Normal equations in the centred variable x = s - 1 for conditioning.
    let mut normal = crate::linalg::Matrix::zeros(4, 4);
    let mut rhs = vec![0.0; 4];
    for (si, ei) in s.iter().zip(energies) {
        let x = si - 1.0;
        let powers = [1.0, x, x * x, x * x * x];
        for i in 0..4 {
            rhs[i] += powers[i] * (ei - shift);
            for j in 0..4 {
                normal[(i, j)] += powers[i] * powers[j];
            }
        }
    }
    let c = crate::linalg::solve_linear(&normal, &rhs)?;
    // Back to powers of s: sum c_k (s - 1)^k.
    let coefficients = [
        shift + c[0] - c[1] + c[2] - c[3],
        c[1] - 2.0 * c[2] + 3.0 * c[3],
        c[2] - 3.0 * c[3],
        c[3],
    ];
    let mut fit = BirchMurnaghan {
        reference_volume,
        coefficients,
        v0: 0.0,
        e0: 0.0,
        b0: 0.0,
        b0_prime: 0.0,
        residual_rms: 0.0,
    };
    // Stationary points: a1 + 2 a2 s + 3 a3 s^2 = 0; keep the minimum
    // closest to the sampled range.
    let a = coefficients;
    let mut roots = Vec::new();
    if a[3].abs() < 1e-300 {
        if a[2] != 0.0 {
            roots.push(-a[1] / (2.0 * a[2]));
        }
    } else {
        let disc = 4.0 * a[2] * a[2] - 12.0 * a[3] * a[1];
        if disc >= 0.0 {
            roots.push((-2.0 * a[2] + disc.sqrt()) / (6.0 * a[3]));
            roots.push((-2.0 * a[2] - disc.sqrt()) / (6.0 * a[3]));
        }
    }
    let minimum = roots
        .into_iter()
        .filter(|r| *r > 0.0)
        .map(|r| reference_volume * r.powf(-1.5))
        .filter(|v| fit.derivatives(*v)[2] > 0.0)
        .min_by(|x, y| {
            (x - reference_volume)
                .abs()
                .total_cmp(&(y - reference_volume).abs())
        });
    match minimum {
        Some(minimum) => {
            let d = fit.derivatives(minimum);
            fit.v0 = minimum;
            fit.e0 = d[0];
            fit.b0 = minimum * d[2];
            fit.b0_prime = -1.0 - minimum * d[3] / d[2];
        }
        None => {
            fit.v0 = f64::NAN;
            fit.e0 = f64::NAN;
            fit.b0 = f64::NAN;
            fit.b0_prime = f64::NAN;
        }
    }
    fit.residual_rms = (volumes
        .iter()
        .zip(energies)
        .map(|(v, e)| (fit.energy(*v) - e).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    Ok(fit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed-form third-order Birch-Murnaghan energy.
    fn bm3(v: f64, e0: f64, v0: f64, b0: f64, bp: f64) -> f64 {
        let x = (v0 / v).powf(2.0 / 3.0) - 1.0;
        e0 + 9.0 * v0 * b0 / 16.0
            * (x * x * x * bp + x * x * (6.0 - 4.0 * (v0 / v).powf(2.0 / 3.0)))
    }

    #[test]
    fn fit_recovers_the_closed_form_parameters_and_pressure() {
        let (e0, v0, b0, bp) = (-12.3, 270.0, 0.0034, 4.3);
        let volumes: Vec<f64> = (0..7).map(|i| v0 * (0.92 + 0.03 * i as f64)).collect();
        let energies: Vec<f64> = volumes.iter().map(|v| bm3(*v, e0, v0, b0, bp)).collect();
        let fit = fit_birch_murnaghan(&volumes, &energies).unwrap();
        assert!((fit.v0 - v0).abs() < 1e-8 * v0, "{}", fit.v0);
        assert!((fit.e0 - e0).abs() < 1e-12);
        assert!((fit.b0 - b0).abs() < 1e-9 * b0, "{}", fit.b0);
        assert!((fit.b0_prime - bp).abs() < 1e-7, "{}", fit.b0_prime);
        assert!(fit.residual_rms < 1e-13);
        // p(V) = 3B0/2 [y^7 - y^5] [1 + 3/4 (B0' - 4)(y^2 - 1)], y = (V0/V)^(1/3).
        for v in [0.95 * v0, 1.07 * v0] {
            let y = (v0 / v).cbrt();
            let p = 1.5 * b0 * (y.powi(7) - y.powi(5)) * (1.0 + 0.75 * (bp - 4.0) * (y * y - 1.0));
            assert!((fit.pressure(v) - p).abs() < 1e-9 * b0);
        }
    }

    #[test]
    fn isotherm_fit_accepts_a_branch_without_a_minimum() {
        // The compressed branch of a Birch-Murnaghan curve: monotonic.
        let (e0, v0, b0, bp) = (-3.0, 120.0, 0.005, 4.2);
        let volumes: Vec<f64> = (0..6).map(|i| v0 * (0.80 + 0.01 * i as f64)).collect();
        let energies: Vec<f64> = volumes.iter().map(|v| bm3(*v, e0, v0, b0, bp)).collect();
        assert!(fit_birch_murnaghan(&volumes, &energies).is_ok());
        // Shifting the curve by a pressure term removes the stationary point
        // from the fitted cubic inside any reasonable range; a linear term in
        // V is not of the fitted form, so only the pressure is compared.
        let fit = fit_birch_murnaghan_isotherm(&volumes, &energies).unwrap();
        let ratio = (v0 / volumes[2]).cbrt();
        let exact = 1.5
            * b0
            * (ratio.powi(7) - ratio.powi(5))
            * (1.0 + 0.75 * (bp - 4.0) * (ratio * ratio - 1.0));
        println!(
            "pressure at V = {}: fit {:e}, closed form {exact:e}, v0 {}",
            volumes[2],
            fit.pressure(volumes[2]),
            fit.v0
        );
        assert!((fit.pressure(volumes[2]) - exact).abs() < 1e-10);
        // A strictly decreasing line has no minimum at all.
        let line: Vec<f64> = volumes.iter().map(|v| -1e-4 * v).collect();
        assert!(fit_birch_murnaghan(&volumes, &line).is_err());
        let open = fit_birch_murnaghan_isotherm(&volumes, &line).unwrap();
        assert!(open.v0.is_nan() && open.b0.is_nan());
        assert!((open.pressure(volumes[3]) - 1e-4).abs() < 1e-8);
    }
}
