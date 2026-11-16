// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Interspecies correlation providers.
//!
//! A provider couples the total electron density to the summed density of
//! one channel of quantum particles (quantum protons, positrons, ...). The
//! correlation of a calculation is a [`CorrelationModel`]: one provider per
//! channel, whose energies add.

use crate::{DftError, Result};

/// Energy per volume and both functional derivatives on the same grid point.
#[derive(Clone, Copy, Debug)]
pub struct CorrelationPoint {
    pub energy: f64,
    pub electron_potential: f64,
    /// Derivative with respect to the partner (particle-channel) density.
    pub partner_potential: f64,
}

/// A local interspecies functional, independent of the electronic XC provider.
/// Densities are total number densities in Bohr^-3 (electron, partner);
/// potentials are Hartree.
pub trait InterspeciesCorrelation: Sync {
    fn description(&self) -> &str;
    /// Only an identically zero provider may omit numerical integration.
    fn is_zero(&self) -> bool {
        false
    }
    fn evaluate(&self, electron: f64, partner: f64) -> Result<CorrelationPoint>;
    /// Analytic action of the symmetric two-density kernel on
    /// `[d rho_e, d rho_partner]`.
    fn kernel_action(&self, electron: f64, partner: f64, direction: [f64; 2]) -> Result<[f64; 2]> {
        let _ = (electron, partner, direction);
        Err(DftError::Unsupported {
            feature: "interspecies density response".into(),
            missing: "an analytic two-density kernel supplied by the interspecies provider".into(),
        })
    }
}

/// Density tangents of an orbital product where the density itself
/// underflowed. Far from a quantum particle its orbital is ~1e-170, so
/// `rho = psi^2` rounds to exactly zero while `d rho = 2 psi d psi` is still a
/// tiny nonzero number; the exact tangent of a density that vanishes is zero
/// (`psi = 0`), and nothing below the smallest representable density carries
/// energy. A density that is exactly zero therefore gets a zero tangent.
pub(crate) fn representable_tangent(density: [f64; 2], direction: [f64; 2]) -> [f64; 2] {
    std::array::from_fn(|k| if density[k] == 0.0 { 0.0 } else { direction[k] })
}

/// Explicit mean-field model, without interspecies dynamical correlation.
pub struct NoCorrelation;

impl InterspeciesCorrelation for NoCorrelation {
    fn description(&self) -> &str {
        "no electron-proton correlation; quantum Hartree product"
    }
    fn is_zero(&self) -> bool {
        true
    }
    fn evaluate(&self, electron: f64, proton: f64) -> Result<CorrelationPoint> {
        validate_density(electron, proton)?;
        Ok(CorrelationPoint {
            energy: 0.0,
            electron_potential: 0.0,
            partner_potential: 0.0,
        })
    }
    fn kernel_action(&self, electron: f64, proton: f64, direction: [f64; 2]) -> Result<[f64; 2]> {
        validate_density(electron, proton)?;
        if direction.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "interspecies density directions must be finite".into(),
            ));
        }
        Ok([0.0; 2])
    }
}

/// EPC17, Eq. (13) of doi:10.1063/1.4996038.
/// Parameters 2.35, 2.4, 3.2 were fitted to proton densities, not energies.
#[derive(Clone, Copy, Debug)]
pub struct Epc17 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Epc17 {
    /// EPC17-2: the parameters fitted to proton affinities (c = 6.6).
    pub fn epc17_2() -> Self {
        Self {
            a: 2.35,
            b: 2.4,
            c: 6.6,
        }
    }
}

/// Electron-proton providers by name.
pub fn interspecies_correlation(name: &str) -> Result<Box<dyn InterspeciesCorrelation>> {
    match name {
        "epc17" | "epc17-1" => Ok(Box::new(Epc17::default())),
        "epc17-2" => Ok(Box::new(Epc17::epc17_2())),
        "none" => Ok(Box::new(NoCorrelation)),
        _ => Err(DftError::InvalidInput(
            "interspecies_correlation must be epc17 (epc17-1), epc17-2 or none".into(),
        )),
    }
}

/// Any interspecies provider by name: the electron-proton functionals
/// (`epc17`, `epc17-1`, `epc17-2`), the electron-positron functionals
/// (`drummond-2011`) and `none`.
pub fn by_name(name: &str) -> Result<Box<dyn InterspeciesCorrelation>> {
    interspecies_correlation(name)
        .or_else(|_| positron_correlation(name))
        .map_err(|_| {
            DftError::InvalidInput(format!(
                "unknown interspecies correlation `{name}`; use epc17 (epc17-1), epc17-2, \
                 drummond-2011 (qmc-2011) or none"
            ))
        })
}

/// Electron-positron providers by name.
pub fn positron_correlation(name: &str) -> Result<Box<dyn InterspeciesCorrelation>> {
    match name {
        "drummond-2011" | "qmc-2011" => Ok(Box::new(PositronRelaxation::drummond_2011())),
        "none" => Ok(Box::new(NoCorrelation)),
        _ => Err(DftError::InvalidInput(
            "positron_correlation must be drummond-2011 (qmc-2011) or none".into(),
        )),
    }
}

/// The interspecies correlation of a calculation: provider `c` couples the
/// electron density to the summed density of the particles whose
/// `correlation_channel` is `c`, and the channel energies add.
pub struct CorrelationModel {
    channels: Vec<Box<dyn InterspeciesCorrelation>>,
}

impl CorrelationModel {
    pub fn new(channels: Vec<Box<dyn InterspeciesCorrelation>>) -> Self {
        Self { channels }
    }
    /// No interspecies correlation.
    pub fn none() -> Self {
        Self::new(Vec::new())
    }
    /// One channel.
    pub fn single(provider: Box<dyn InterspeciesCorrelation>) -> Self {
        Self::new(vec![provider])
    }
    pub fn channels(&self) -> &[Box<dyn InterspeciesCorrelation>] {
        &self.channels
    }
    pub fn len(&self) -> usize {
        self.channels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
    /// Every channel is identically zero.
    pub fn is_zero(&self) -> bool {
        self.channels.iter().all(|c| c.is_zero())
    }
    pub fn descriptions(&self) -> Vec<String> {
        self.channels
            .iter()
            .map(|c| c.description().to_string())
            .collect()
    }
}

/// A positron in the electron gas in the zero-positron-density limit,
/// `E = int n_+ Delta(r_s(n_-))`, with the relaxation (correlation) energy
/// per positron
///
/// ```text
/// Delta(r_s) = (A_-1 / r_s + A_0 + A_1 r_s - D B_2 r_s^2) / (1 + B_1 r_s + B_2 r_s^2)
/// ```
///
/// in Hartree, fitted to diffusion Monte Carlo energies by Drummond, Lopez
/// Rios, Needs and Pickard, Phys. Rev. Lett. 107, 207402 (2011), Eq. (2):
/// `A_-1 = -0.260361, A_0 = -0.261762, A_1 = 0.00375534, B_1 = 0.113718,
/// B_2 = 0.0270912`, and `D = 0.262005`, the energy of the Ps- ion and the
/// low-density limit. The form is a smooth function of the electron density.
#[derive(Clone, Copy, Debug)]
pub struct PositronRelaxation {
    pub a_minus1: f64,
    pub a0: f64,
    pub a1: f64,
    pub b1: f64,
    pub b2: f64,
    pub low_density_limit: f64,
}

impl PositronRelaxation {
    pub fn drummond_2011() -> Self {
        Self {
            a_minus1: -0.260361,
            a0: -0.261762,
            a1: 0.00375534,
            b1: 0.113718,
            b2: 0.0270912,
            low_density_limit: 0.262005,
        }
    }

    /// `Delta`, `dDelta/dr_s` and `d2Delta/dr_s2` in Hartree.
    pub fn relaxation(&self, rs: f64) -> [f64; 3] {
        let d = self.low_density_limit;
        let numerator = self.a_minus1 / rs + self.a0 + self.a1 * rs - d * self.b2 * rs * rs;
        let dn = -self.a_minus1 / (rs * rs) + self.a1 - 2.0 * d * self.b2 * rs;
        let d2n = 2.0 * self.a_minus1 / rs.powi(3) - 2.0 * d * self.b2;
        let denominator = 1.0 + self.b1 * rs + self.b2 * rs * rs;
        let dd = self.b1 + 2.0 * self.b2 * rs;
        let d2d = 2.0 * self.b2;
        let value = numerator / denominator;
        let first = (dn - value * dd) / denominator;
        let second = (d2n - 2.0 * first * dd - value * d2d) / denominator;
        [value, first, second]
    }

    /// `Delta`, `n dDelta/dn` and `n^2 d2Delta/dn2`.
    /// The scaled derivatives remain representable in dilute AO tails even
    /// when the uncontracted second density derivative would overflow.
    fn density_scaled(&self, n: f64) -> [f64; 3] {
        let t = (4.0 * std::f64::consts::PI / 3.0).cbrt() * n.cbrt();
        if t <= 1.0 {
            // Divide the published rational form by r_s^2 before evaluating
            // it. This also retains the Ps- limit without infinity/infinity.
            let numerator = -self.low_density_limit * self.b2
                + t * (self.a1 + t * (self.a0 + t * self.a_minus1));
            let dn = self.a1 + t * (2.0 * self.a0 + 3.0 * t * self.a_minus1);
            let d2n = 2.0 * self.a0 + 6.0 * t * self.a_minus1;
            let denominator = self.b2 + t * (self.b1 + t);
            let dd = self.b1 + 2.0 * t;
            let value = numerator / denominator;
            let first = (dn - value * dd) / denominator;
            let second = (d2n - 2.0 * first * dd - 2.0 * value) / denominator;
            [
                value,
                t * first / 3.0,
                (t * t * second - 2.0 * t * first) / 9.0,
            ]
        } else {
            let rs = 1.0 / t;
            let rs2 = rs * rs;
            let denominator = 1.0 + self.b1 * rs + self.b2 * rs2;
            let value = (self.a_minus1 / rs + self.a0 + self.a1 * rs
                - self.low_density_limit * self.b2 * rs2)
                / denominator;
            let dd = self.b1 * rs + 2.0 * self.b2 * rs2;
            let first = (-self.a_minus1 / rs + self.a1 * rs
                - 2.0 * self.low_density_limit * self.b2 * rs2
                - value * dd)
                / denominator;
            let second = (2.0 * self.a_minus1 / rs
                - 2.0 * self.low_density_limit * self.b2 * rs2
                - 2.0 * first * dd
                - 2.0 * value * self.b2 * rs2)
                / denominator;
            [value, -first / 3.0, (second + 4.0 * first) / 9.0]
        }
    }
}

impl InterspeciesCorrelation for PositronRelaxation {
    fn description(&self) -> &str {
        "electron-positron relaxation energy in the zero-positron-density limit (DMC fit, 2011)"
    }
    fn evaluate(&self, electron: f64, partner: f64) -> Result<CorrelationPoint> {
        validate_density(electron, partner)?;
        if electron == 0.0 {
            // Where the electron density underflowed the orbital products
            // vanish and carry no tangent (`representable_tangent`).
            return Ok(CorrelationPoint {
                energy: 0.0,
                electron_potential: 0.0,
                partner_potential: 0.0,
            });
        }
        let [value, first, _] = self.density_scaled(electron);
        let point = CorrelationPoint {
            energy: partner * value,
            electron_potential: density_product(first, partner, 1.0, electron, 1),
            partner_potential: value,
        };
        if [
            point.energy,
            point.electron_potential,
            point.partner_potential,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "non-representable electron-positron correlation energy or potential".into(),
            ));
        }
        Ok(point)
    }
    fn kernel_action(&self, electron: f64, partner: f64, direction: [f64; 2]) -> Result<[f64; 2]> {
        validate_density(electron, partner)?;
        if direction.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "interspecies density directions must be finite".into(),
            ));
        }
        if electron == 0.0 {
            return Ok([0.0; 2]);
        }
        let [_, first, second] = self.density_scaled(electron);
        let [de, dp] = direction;
        let action = [
            density_product(second, partner, de, electron, 2)
                + density_product(first, dp, 1.0, electron, 1),
            density_product(first, de, 1.0, electron, 1),
        ];
        if action.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "non-representable electron-positron density kernel action".into(),
            ));
        }
        Ok(action)
    }
}

/// `value * left * right / density^power`, without transient underflow or
/// overflow. Binary exponent bookkeeping changes no physical density or
/// kernel; only an unrepresentable final result rounds to zero or infinity.
fn density_product(value: f64, left: f64, right: f64, density: f64, power: i32) -> f64 {
    if value == 0.0 || left == 0.0 || right == 0.0 {
        return 0.0;
    }
    let (v, ve) = libm::frexp(value);
    let (l, le) = libm::frexp(left);
    let (r, re) = libm::frexp(right);
    let (n, ne) = libm::frexp(density);
    libm::scalbn(v * l * r / n.powi(power), ve + le + re - power * ne)
}

impl Default for Epc17 {
    fn default() -> Self {
        Self {
            a: 2.35,
            b: 2.4,
            c: 3.2,
        }
    }
}

impl InterspeciesCorrelation for Epc17 {
    fn description(&self) -> &str {
        "EPC17 local total-electron/total-proton density correlation"
    }
    fn evaluate(&self, electron: f64, proton: f64) -> Result<CorrelationPoint> {
        validate_density(electron, proton)?;
        if !self.a.is_finite()
            || !self.b.is_finite()
            || !self.c.is_finite()
            || self.a <= 0.0
            || self.c <= 0.0
            || (self.b > 0.0 && self.b * self.b >= 4.0 * self.a * self.c)
        {
            return Err(DftError::InvalidInput("EPC parameters must have a strictly positive denominator for every non-negative density".into()));
        }
        let x = electron * proton;
        if !x.is_finite() {
            return Err(DftError::InvalidInput(
                "electron-proton density product overflow".into(),
            ));
        }
        let root = x.sqrt();
        if x == 0.0 {
            return Ok(CorrelationPoint {
                energy: 0.0,
                electron_potential: -proton / self.a,
                partner_potential: -electron / self.a,
            });
        }
        let denominator = self.a - self.b * root + self.c * x;
        // d[-x/(a-b sqrt(x)+cx)]/dx, including its exact x -> 0 limit.
        let derivative = (-self.a + 0.5 * self.b * root) / denominator.powi(2);
        Ok(CorrelationPoint {
            energy: -x / denominator,
            electron_potential: proton * derivative,
            partner_potential: electron * derivative,
        })
    }
    fn kernel_action(&self, electron: f64, proton: f64, direction: [f64; 2]) -> Result<[f64; 2]> {
        self.evaluate(electron, proton)?;
        if direction.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "interspecies density directions must be finite".into(),
            ));
        }
        let [de, dp] = direction;
        let x = electron * proton;
        if electron > 0.0 && proton > 0.0 && x < f64::MIN_POSITIVE {
            // A positive product can be subnormal or underflow without
            // either density being nodal. Form sqrt(e*p) separately and
            // evaluate root*f'' before the density ratios, rather than
            // discarding finite diagonal kernel terms or dividing by x.
            let root_e = electron.sqrt();
            let root_p = proton.sqrt();
            let root = root_e * root_p;
            let denominator = self.a - self.b * root + (self.c * root) * root;
            let a_over_d = self.a / denominator;
            let b_over_d = self.b / denominator;
            let br_over_d = (self.b * root) / denominator;
            let cr_over_d = (self.c * root) / denominator;
            let first = (-a_over_d + 0.5 * br_over_d) / denominator;
            let scaled_second = (br_over_d * b_over_d + 8.0 * a_over_d * cr_over_d
                - 3.0 * b_over_d * (a_over_d + cr_over_d * root))
                / (4.0 * denominator);
            let mixed = first + root * scaled_second;
            let electron_diagonal = (proton * (root_p / root_e)) * scaled_second;
            let partner_diagonal = (electron * (root_e / root_p)) * scaled_second;
            let result = [
                electron_diagonal * de + mixed * dp,
                mixed * de + partner_diagonal * dp,
            ];
            if !denominator.is_finite()
                || denominator <= 0.0
                || !scaled_second.is_finite()
                || result.iter().any(|value| !value.is_finite())
            {
                return Err(DftError::InvalidInput(
                    "non-finite EPC17 density kernel action".into(),
                ));
            }
            return Ok(result);
        }
        if x == 0.0 {
            // At a nodal density the own-density Hessian diverges. A physical
            // orbital tangent has d rho=2 psi d psi=0 at that node; its finite
            // kernel action is the mixed limit. Reject an undefined density
            // tangent instead of regularizing the physical kernel.
            if electron == 0.0 && proton > 0.0 && de != 0.0
                || proton == 0.0 && electron > 0.0 && dp != 0.0
            {
                return Err(DftError::InvalidInput("EPC17 own-density kernel is singular at a zero density with a nonzero density tangent; use a physical orbital tangent".into()));
            }
            return Ok([-dp / self.a, -de / self.a]);
        }
        let root = x.sqrt();
        let denominator = self.a - self.b * root + self.c * x;
        let first = (-self.a + 0.5 * self.b * root) / denominator.powi(2);
        let second = (self.b * self.b + 8.0 * self.a * self.c
            - 3.0 * self.b * (self.a / root + self.c * root))
            / (4.0 * denominator.powi(3));
        let dx = proton * de + electron * dp;
        let result = [
            first * dp + proton * second * dx,
            first * de + electron * second * dx,
        ];
        if result.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "non-finite EPC17 density kernel action".into(),
            ));
        }
        Ok(result)
    }
}

fn validate_density(electron: f64, partner: f64) -> Result<()> {
    if [electron, partner]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(DftError::InvalidInput(
            "interspecies correlation requires finite non-negative number densities".into(),
        ));
    }
    Ok(())
}
