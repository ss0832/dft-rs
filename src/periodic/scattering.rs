// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Energy-dependent scalar relaxation-time transport on rigid bands.
//!
//! Moments contain tau(E); the ordinary Hall vertex contains tau(E)^2.
//! No derivative of tau contributes to the Lorentz vertex because
//! (v cross B).grad_k E = 0. At frequency omega replace tau by
//! tau/(1-i omega tau), with the convention exp(-i omega t).
//! This is an intraband relaxation-time model, not a microscopic collision
//! operator or an interband-coherence approximation.

use super::band_transport::{
    self as bands, FrequencyTransport, HallState, HallTensor, TransportIntegrals, TransportState,
};
use crate::error::{DftError, Result};
use crate::response::complex::Complex;

/// Positive lifetime table: rows [absolute band energy, lifetime], in Hartree
/// and hbar/Hartree. Log-linear interpolation preserves positivity without a
/// floor. Outside the knot interval the nearest endpoint lifetime is used.
#[derive(Clone, Debug)]
pub struct RelaxationTimeTable {
    knots: Vec<[f64; 2]>,
}

impl RelaxationTimeTable {
    pub fn new(knots: Vec<[f64; 2]>) -> Result<Self> {
        if knots.is_empty()
            || knots
                .iter()
                .any(|r| !r[0].is_finite() || !r[1].is_finite() || r[1] <= 0.0)
            || knots.windows(2).any(|r| r[1][0] <= r[0][0])
        {
            return Err(DftError::InvalidInput(
                "scattering table needs strictly increasing finite energies and positive finite lifetimes".into(),
            ));
        }
        Ok(Self { knots })
    }

    pub fn lifetime(&self, energy: f64) -> Result<f64> {
        if !energy.is_finite() {
            return Err(DftError::InvalidInput(
                "scattering energy must be finite".into(),
            ));
        }
        let upper = self.knots.partition_point(|r| r[0] < energy);
        if upper == 0 {
            return Ok(self.knots[0][1]);
        }
        if upper == self.knots.len() {
            return Ok(self.knots[upper - 1][1]);
        }
        let [lo, a] = self.knots[upper - 1];
        let [hi, b] = self.knots[upper];
        let fraction = (energy - lo) / (hi - lo);
        positive_lifetime(((1.0 - fraction) * a.ln() + fraction * b.ln()).exp())
    }
}

fn positive_lifetime(tau: f64) -> Result<f64> {
    if tau.is_finite() && tau > 0.0 {
        Ok(tau)
    } else {
        Err(DftError::InvalidInput(
            "scattering provider must return a positive finite lifetime".into(),
        ))
    }
}

/// K_n = g/(V Nk) sum tau(E) v_a v_b (E-mu)^n (-df/dE), n=0,1,2.
/// Particle number uses the unweighted Fermi occupations. A provider may
/// capture temperature and chemical potential; no functional name is used.
pub fn transport_integrals(
    states: &[TransportState],
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
    lifetime: &(impl Fn(f64) -> Result<f64> + Sync),
) -> Result<TransportIntegrals> {
    let (l, electron_density) =
        bands::weighted_transport_integrals(states, volume, nk, g, mu, kt, window, &|e| {
            positive_lifetime(lifetime(e)?)
        })?;
    Ok(TransportIntegrals {
        l0: l[0],
        l1: l[1],
        l2: l[2],
        electron_density,
    })
}

/// Physical coefficients with lifetime included, in atomic units. Seebeck
/// is in k_B/e. Singular K0 returns None for Seebeck and open-circuit heat
/// conduction, with the same inverse condition as constant-time transport.
#[derive(Clone, Debug)]
pub struct ScatteringCoefficients {
    pub conductivity: [[f64; 3]; 3],
    pub seebeck: Option<[[f64; 3]; 3]>,
    pub thermal_conductivity: Option<[[f64; 3]; 3]>,
}

pub fn transport_coefficients(moments: &TransportIntegrals, kt: f64) -> ScatteringCoefficients {
    let c = bands::transport_coefficients(moments, kt);
    ScatteringCoefficients {
        conductivity: c.conductivity_over_tau,
        seebeck: c.seebeck,
        thermal_conductivity: c.thermal_conductivity_over_tau,
    }
}

/// Weak-field ordinary Hall response, with lifetime squared included and
/// without an antisymmetric projection of the quadrature.
#[derive(Clone, Debug)]
pub struct ScatteringHall {
    pub conductivity_derivative: HallTensor,
    pub raw_onsager_residual: f64,
    pub relative_onsager_residual: Option<f64>,
}

impl ScatteringHall {
    pub(crate) fn quadrature(&self) -> bands::HallIntegrals {
        bands::HallIntegrals {
            conductivity_derivative_over_tau_squared: self.conductivity_derivative,
            raw_onsager_residual: self.raw_onsager_residual,
            relative_onsager_residual: self.relative_onsager_residual,
        }
    }
}

pub fn hall_integrals(
    states: &[HallState],
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
    lifetime: &(impl Fn(f64) -> Result<f64> + Sync),
) -> Result<ScatteringHall> {
    let h = bands::hall_diagnostics(bands::weighted_hall_integrals(
        states,
        volume,
        nk,
        g,
        mu,
        kt,
        window,
        &|e| {
            let tau = positive_lifetime(lifetime(e)?)?;
            Ok(tau * tau)
        },
    )?);
    Ok(ScatteringHall {
        conductivity_derivative: h.conductivity_derivative_over_tau_squared,
        raw_onsager_residual: h.raw_onsager_residual,
        relative_onsager_residual: h.relative_onsager_residual,
    })
}

/// d rho_ab / d B_c = -K0^-1 H_c K0^-1. Lifetime does not factor out for
/// energy-dependent scattering. Axes are [electric field, current, B].
pub fn hall_coefficient(moments: &TransportIntegrals, hall: &ScatteringHall) -> Option<HallTensor> {
    bands::hall_coefficient(moments, &hall.quadrature())
}

/// Retarded electric and ordinary Hall conductivities. Each state carries
/// its own causal propagator; averaging tau before solving is incorrect.
/// Hall states must be isolated bands; zero-field states may be clusters.
pub fn frequency_transport(
    states: &[TransportState],
    hall_states: Option<&[HallState]>,
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
    omega: f64,
    lifetime: &(impl Fn(f64) -> Result<f64> + Sync),
) -> Result<FrequencyTransport> {
    let (moments, hall_conductivity_derivative) = frequency_moments(
        states,
        hall_states,
        volume,
        nk,
        g,
        mu,
        kt,
        window,
        omega,
        lifetime,
    )?;
    Ok(FrequencyTransport {
        conductivity: moments[0],
        hall_conductivity_derivative,
    })
}

fn frequency_moments(
    states: &[TransportState],
    hall_states: Option<&[HallState]>,
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
    omega: f64,
    lifetime: &(impl Fn(f64) -> Result<f64> + Sync),
) -> Result<([[[Complex; 3]; 3]; 3], Option<[[[Complex; 3]; 3]; 3]>)> {
    if !omega.is_finite() {
        return Err(DftError::InvalidInput(
            "scattering frequency must be finite".into(),
        ));
    }
    let propagator = |e| {
        let tau = positive_lifetime(lifetime(e)?)?;
        if !(omega * tau).is_finite() {
            return Err(DftError::InvalidInput(
                "scattering frequency times lifetime exceeds the finite floating-point range"
                    .into(),
            ));
        }
        Ok(Complex::from(tau) / Complex::new(1.0, -omega * tau))
    };
    let (moments, _) =
        bands::weighted_transport_integrals(states, volume, nk, g, mu, kt, window, &propagator)?;
    let hall_conductivity_derivative = hall_states
        .map(|s| {
            bands::weighted_hall_integrals(s, volume, nk, g, mu, kt, window, &|e| {
                let p = propagator(e)?;
                Ok(p * p)
            })
        })
        .transpose()?;
    Ok((moments, hall_conductivity_derivative))
}

/// Retarded thermoelectric tensors of the scalar relaxation-time kinetic
/// equation at zero magnetic field, plus its ordinary electric Hall vertex.
/// Seebeck is in k_B/e, Peltier in Hartree/e, heat conductivity in atomic
/// units. The imposed temperature-gradient drive uses a local equilibrium
/// bath; this is not an energy-conserving hydrodynamic collision model.
#[derive(Clone, Debug)]
pub struct FrequencyThermoelectric {
    pub electric: FrequencyTransport,
    /// Causal K0, K1, K2, with the lifetime propagator inside each sum.
    pub moments: [[[Complex; 3]; 3]; 3],
    pub seebeck: Option<[[Complex; 3]; 3]>,
    pub peltier: Option<[[Complex; 3]; 3]>,
    pub thermal_conductivity: Option<[[Complex; 3]; 3]>,
    /// max |(K0/scale) inverse(K0/scale) - I|, without projection.
    pub inverse_residual: Option<f64>,
    /// max |Peltier - kT Seebeck^T|, in Hartree/e, before projection.
    pub kelvin_residual: Option<f64>,
}

fn complex_product(a: &[[Complex; 3]; 3], b: &[[Complex; 3]; 3]) -> [[Complex; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).fold(Complex::ZERO, |s, k| s + a[i][k] * b[k][j]))
    })
}

pub fn thermoelectric_frequency_transport(
    states: &[TransportState],
    hall_states: Option<&[HallState]>,
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
    omega: f64,
    lifetime: &(impl Fn(f64) -> Result<f64> + Sync),
) -> Result<FrequencyThermoelectric> {
    let (moments, hall_conductivity_derivative) = frequency_moments(
        states,
        hall_states,
        volume,
        nk,
        g,
        mu,
        kt,
        window,
        omega,
        lifetime,
    )?;
    let mut result = FrequencyThermoelectric {
        electric: FrequencyTransport {
            conductivity: moments[0],
            hall_conductivity_derivative,
        },
        moments,
        seebeck: None,
        peltier: None,
        thermal_conductivity: None,
        inverse_residual: None,
        kelvin_residual: None,
    };
    if let Some((inverse, scale)) = crate::linalg::normalized_complex3_inverse(&moments[0])? {
        let k = moments.map(|m| m.map(|row| row.map(|z| z / scale)));
        let left = complex_product(&inverse, &k[1]);
        let right = complex_product(&k[1], &inverse);
        let correction = complex_product(&k[1], &left);
        let seebeck = left.map(|row| row.map(|z| -z / kt));
        let peltier = right.map(|row| row.map(|z| -z));
        let thermal: [[Complex; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (k[2][i][j] - correction[i][j]) * (scale / kt))
        });
        if seebeck
            .iter()
            .chain(peltier.iter())
            .chain(thermal.iter())
            .flatten()
            .any(|z| !z.is_finite())
        {
            return Err(DftError::InvalidInput(
                "thermoelectric frequency tensor exceeds the finite floating-point range".into(),
            ));
        }
        let identity = complex_product(&k[0], &inverse);
        let mut inverse_residual = 0.0_f64;
        let mut kelvin_residual = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                inverse_residual = inverse_residual
                    .max((identity[i][j] - Complex::from(if i == j { 1.0 } else { 0.0 })).abs());
                kelvin_residual = kelvin_residual.max((peltier[i][j] - seebeck[j][i] * kt).abs());
            }
        }
        result.seebeck = Some(seebeck);
        result.peltier = Some(peltier);
        result.thermal_conductivity = Some(thermal);
        result.inverse_residual = Some(inverse_residual);
        result.kelvin_residual = Some(kelvin_residual);
    }
    Ok(result)
}
