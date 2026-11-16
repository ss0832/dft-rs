// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static, singlet, normal-band SCDFT phonon pairing and mass kernels.
//!
//! The electron-phonon term is LM2005 Eq. (74), with its thermal convolution
//! from Eq. (55). The supplied screened Coulomb matrix is kept separately.
//! Energies, kT and phonon energies are Hartree; squared electron-phonon
//! vertices are Hartree squared and include the caller's mode/q weights.
//! Band quadrature weights are not included in the kernel matrix.
//!
//! The mass follows the LM2005 convention clarified by Eq. (10) of
//! Phys. Rev. B 88, 014514 (2013): Z=-sum |g|^2 (J(x,y)+J(x,-y))/tanh(x/2kT).
//! Confluent Fermi divided differences fix the derivative product and its
//! removable resonance without copying a singular printed formula.
//! Gap/number/phase equations live in response::superconducting; this kernel
//! does not include normal-density backreaction or retarded external response.
//! All inputs are scalar-generic, including the thermal and interaction
//! parameters. Removable resonances use analytic divided-difference integrals;
//! Fermi-surface limits use derivatives of the same convolution. No physical
//! denominator is clipped and no finite differences enter production.

use crate::dual::{DualN, Scalar};
use crate::{DftError, Result};
use rayon::prelude::*;

/// Normal-band data for the off-diagonal singlet pairing kernel. A mode row
/// contains `n*n` squared vertices in `[initial band][final band]` order.
#[derive(Clone, Debug)]
pub struct PairingInput<S> {
    pub energies: Vec<S>,
    pub chemical_potential: S,
    pub temperature_energy: S,
    pub phonon_energies: Vec<S>,
    pub coupling_squared: Vec<Vec<S>>,
    pub screened_coulomb: Vec<S>,
}

/// Row-major matrices in Hartree. The Coulomb and phonon pieces are additive.
#[derive(Clone, Debug)]
pub struct PairingKernel<S> {
    pub bands: usize,
    pub phonon: Vec<S>,
    pub coulomb: Vec<S>,
    pub total: Vec<S>,
}

/// Reusable Gauss-Legendre rule for analytic removable-singularity integrals.
/// Order 32 is the default; a second order can independently assess quadrature.
#[derive(Clone, Debug)]
pub struct Lm2005 {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

/// Compatibility name for the original off-diagonal-only public component.
pub type Lm2005OffDiagonal = Lm2005;

impl Default for Lm2005 {
    fn default() -> Self {
        let (nodes, weights) = crate::grid::gauss_legendre(32);
        Self { nodes, weights }
    }
}

impl Lm2005 {
    pub fn new(quadrature_order: usize) -> Result<Self> {
        if !(8..=128).contains(&quadrature_order) {
            return Err(DftError::InvalidInput(
                "pairing quadrature order must be between 8 and 128".into(),
            ));
        }
        let (nodes, weights) = crate::grid::gauss_legendre(quadrature_order);
        Ok(Self { nodes, weights })
    }

    /// Thermal convolution before the anomalous-density factors, in 1/Hartree.
    /// `x` and `y` are measured from the chemical potential; `omega` and `kt`
    /// must be strictly positive. The same function handles `x-y=+-omega`.
    pub fn convolution<S: Scalar>(&self, x: S, y: S, omega: S, kt: S) -> Result<S> {
        validate_thermal(x, y, omega, kt)?;
        Ok(self.convolution_value(x, y, omega, kt))
    }

    /// LM2005 phonon pairing factor per unit squared vertex, in 1/Hartree.
    /// Fermi-surface limits preserve differentiation of all four arguments.
    pub fn phonon_factor<S: Scalar>(&self, x: S, y: S, omega: S, kt: S) -> Result<S> {
        validate_thermal(x, y, omega, kt)?;
        let small_x = (x.value() / kt.value()).abs() < 0.5;
        let small_y = (y.value() / kt.value()).abs() < 0.5;
        let value = match (small_x, small_y) {
            (false, false) => {
                (self.convolution_value(x, y, omega, kt) - self.convolution_value(x, -y, omega, kt))
                    .scale(2.0)
                    / (tanh_half(x / kt) * tanh_half(y / kt))
            }
            (true, false) => {
                let integral = self.integrate(|t| {
                    let seeded = DualN {
                        re: x.scale(t),
                        im: [S::from_f64(1.0)],
                    };
                    let at = |y| {
                        self.convolution_value(
                            seeded,
                            DualN::constant(y),
                            DualN::constant(omega),
                            DualN::constant(kt),
                        )
                        .im[0]
                    };
                    at(y) - at(-y)
                });
                self.energy_over_tanh(x, kt) * integral / tanh_half(y / kt)
            }
            (false, true) => self.phonon_factor(y, x, omega, kt)?,
            (true, true) => {
                let integral = self.integrate(|t| {
                    self.integrate(|u| {
                        let one = S::from_f64(1.0);
                        let zero = S::from_f64(0.0);
                        let a = DualN {
                            re: DualN {
                                re: x.scale(t),
                                im: [one],
                            },
                            im: [DualN {
                                re: zero,
                                im: [zero],
                            }],
                        };
                        let b = DualN {
                            re: DualN {
                                re: y.scale(u),
                                im: [zero],
                            },
                            im: [DualN {
                                re: one,
                                im: [zero],
                            }],
                        };
                        self.convolution_value(
                            a,
                            b,
                            DualN::constant(DualN::constant(omega)),
                            DualN::constant(DualN::constant(kt)),
                        )
                        .im[0]
                            .im[0]
                    })
                });
                self.energy_over_tanh(x, kt) * self.energy_over_tanh(y, kt) * integral
            }
        };
        if !value.value().is_finite() {
            return Err(DftError::LinearAlgebra(
                "non-finite phonon pairing kernel".into(),
            ));
        }
        Ok(value)
    }

    /// Diagonal phonon mass factor per squared vertex, Hartree^-2.
    /// This is even in x and y within the LM2005 particle-hole-symmetric mass
    /// approximation. Resonances x-y=+-omega and x=0 use analytic limits;
    /// no physical energy is shifted and no denominator floor is introduced.
    pub fn mass_factor<S: Scalar>(&self, x: S, y: S, omega: S, kt: S) -> Result<S> {
        validate_thermal(x, y, omega, kt)?;
        let value = if (x.value() / kt.value()).abs() < 0.5 {
            let derivative = self
                .integrate(|node| {
                    let t = 0.5 * (node + 1.0);
                    let seeded = DualN {
                        re: x.scale(t),
                        im: [S::from_f64(1.0)],
                    };
                    self.mass_numerator(
                        seeded,
                        DualN::constant(y),
                        DualN::constant(omega),
                        DualN::constant(kt),
                    )
                    .im[0]
                })
                .scale(0.5);
            self.energy_over_tanh(x, kt) * derivative
        } else {
            self.mass_numerator(x, y, omega, kt) / tanh_half(x / kt)
        };
        if !value.value().is_finite() {
            return Err(DftError::LinearAlgebra(
                "non-finite phonon mass kernel".into(),
            ));
        }
        Ok(value)
    }

    /// Z_i=sum_j,nu w_j |g_ij,nu|^2 mass_factor(xi_i,xi_j,Omega_nu,kT).
    /// Vertices retain mode/q weights and band weights are applied once here.
    /// Every entry of K remains unweighted, as in evaluate().
    pub fn mass<S: Scalar + Send + Sync>(
        &self,
        input: &PairingInput<S>,
        band_weights: &[S],
    ) -> Result<Vec<S>> {
        input.validate()?;
        let n = input.energies.len();
        if band_weights.len() != n
            || band_weights
                .iter()
                .any(|w| !w.value().is_finite() || w.value() <= 0.0)
        {
            return Err(DftError::InvalidInput(
                "phonon mass needs one finite positive band weight per band".into(),
            ));
        }
        (0..n)
            .into_par_iter()
            .map(|i| {
                let x = input.energies[i] - input.chemical_potential;
                let mut sum = S::from_f64(0.0);
                for (omega, vertices) in input.phonon_energies.iter().zip(&input.coupling_squared) {
                    for (j, &weight) in band_weights.iter().enumerate() {
                        let y = input.energies[j] - input.chemical_potential;
                        sum = sum
                            + weight
                                * vertices[i * n + j]
                                * self.mass_factor(x, y, *omega, input.temperature_energy)?;
                    }
                }
                if !sum.value().is_finite() {
                    return Err(DftError::LinearAlgebra("phonon mass sum overflow".into()));
                }
                Ok(sum)
            })
            .collect()
    }

    /// Compute each matrix entry independently, including both triangles.
    /// The reduction over physical modes has a deterministic serial order.
    pub fn evaluate<S: Scalar + Send + Sync>(
        &self,
        input: &PairingInput<S>,
    ) -> Result<PairingKernel<S>> {
        input.validate()?;
        let n = input.energies.len();
        let phonon: Result<Vec<_>> = (0..n * n)
            .into_par_iter()
            .map(|ij| {
                let x = input.energies[ij / n] - input.chemical_potential;
                let y = input.energies[ij % n] - input.chemical_potential;
                let mut sum = S::from_f64(0.0);
                for (omega, vertices) in input.phonon_energies.iter().zip(&input.coupling_squared) {
                    sum = sum
                        + vertices[ij]
                            * self.phonon_factor(x, y, *omega, input.temperature_energy)?;
                }
                Ok(sum)
            })
            .collect();
        let phonon = phonon?;
        let total = phonon
            .iter()
            .zip(&input.screened_coulomb)
            .map(|(&a, &b)| a + b)
            .collect();
        if phonon
            .iter()
            .chain(&input.screened_coulomb)
            .chain(&total)
            .any(|x| !x.value().is_finite())
        {
            return Err(DftError::LinearAlgebra(
                "non-finite off-diagonal pairing component".into(),
            ));
        }
        Ok(PairingKernel {
            bands: n,
            phonon,
            coulomb: input.screened_coulomb.clone(),
            total,
        })
    }

    fn integrate<S: Scalar>(&self, f: impl Fn(f64) -> S) -> S {
        self.nodes
            .iter()
            .zip(&self.weights)
            .fold(S::from_f64(0.0), |sum, (&node, &weight)| {
                sum + f(node).scale(weight)
            })
    }

    fn fermi_difference<S: Scalar>(&self, a: S, b: S, kt: S) -> S {
        if ((a.value() - b.value()) / kt.value()).abs() < 1.0 {
            // (f(a)-f(b))/(a-b) = integral_0^1 f'(b+t(a-b)) dt.
            self.integrate(|t| fermi_prime(b + (a - b).scale(0.5 * (t + 1.0)), kt))
                .scale(0.5)
        } else if a.value() < 0.0 && b.value() < 0.0 {
            (fermi(-b / kt) - fermi(-a / kt)) / (a - b)
        } else {
            (fermi(a / kt) - fermi(b / kt)) / (a - b)
        }
    }

    /// f[b,b,y]=integral_0^1 t f''(y+t(b-y)) dt, including b=y exactly.
    fn fermi_confluent_second<S: Scalar>(&self, b: S, y: S, kt: S) -> S {
        if ((b.value() - y.value()) / kt.value()).abs() < 1.0 {
            self.integrate(|node| {
                let t = 0.5 * (node + 1.0);
                fermi_second(y + (b - y).scale(t), kt).scale(t)
            })
            .scale(0.5)
        } else {
            (fermi_prime(b, kt) - self.fermi_difference(b, y, kt)) / (b - y)
        }
    }

    fn mass_numerator<S: Scalar>(&self, x: S, y: S, omega: S, kt: S) -> S {
        let n = bose(omega, kt);
        let emission = n + fermi(x / kt);
        let absorption = n + fermi(-x / kt);
        let term = |y| {
            emission * self.fermi_confluent_second(x - omega, y, kt)
                + absorption * self.fermi_confluent_second(x + omega, y, kt)
        };
        term(y) + term(-y)
    }

    fn convolution_value<S: Scalar>(&self, x: S, y: S, omega: S, kt: S) -> S {
        let bose = bose(omega, kt);
        -(bose + fermi(x / kt)) * self.fermi_difference(y, x - omega, kt)
            - (bose + fermi(y / kt)) * self.fermi_difference(x, y - omega, kt)
    }

    fn energy_over_tanh<S: Scalar>(&self, x: S, kt: S) -> S {
        // x/tanh(x/2kT) = 2kT / integral_0^1 sech^2(tx/2kT) dt.
        let integral = self
            .integrate(|t| {
                let f = fermi((x / kt).scale(0.5 * (t + 1.0)));
                (f * (S::from_f64(1.0) - f)).scale(4.0)
            })
            .scale(0.5);
        kt.scale(2.0) / integral
    }
}

impl<S: Scalar> PairingInput<S> {
    pub fn validate(&self) -> Result<()> {
        let n = self.energies.len();
        if n == 0
            || self.phonon_energies.len() != self.coupling_squared.len()
            || self.screened_coulomb.len() != n * n
            || self.coupling_squared.iter().any(|g| g.len() != n * n)
            || self
                .energies
                .iter()
                .chain(&self.screened_coulomb)
                .any(|x| !x.value().is_finite())
            || !self.chemical_potential.value().is_finite()
            || !self.temperature_energy.value().is_finite()
            || self.temperature_energy.value() <= 0.0
            || self
                .phonon_energies
                .iter()
                .any(|x| !x.value().is_finite() || x.value() <= 0.0)
            || self
                .coupling_squared
                .iter()
                .flatten()
                .any(|x| !x.value().is_finite() || x.value() < 0.0)
        {
            return Err(DftError::InvalidInput(
                "pairing data need finite band energies and Coulomb n*n entries, positive kT and phonon energies, and one non-negative n*n squared-vertex row per mode".into()));
        }
        Ok(())
    }
}

fn validate_thermal<S: Scalar>(x: S, y: S, omega: S, kt: S) -> Result<()> {
    if !x.value().is_finite()
        || !y.value().is_finite()
        || !omega.value().is_finite()
        || !kt.value().is_finite()
        || omega.value() <= 0.0
        || kt.value() <= 0.0
    {
        return Err(DftError::InvalidInput(
            "pairing convolution requires finite energies and positive phonon energy and kT".into(),
        ));
    }
    Ok(())
}

fn fermi<S: Scalar>(x: S) -> S {
    let one = S::from_f64(1.0);
    if x.value() >= 0.0 {
        let e = (-x).exp();
        e / (one + e)
    } else {
        (one + x.exp()).recip()
    }
}

fn fermi_prime<S: Scalar>(x: S, kt: S) -> S {
    let u = x / kt;
    let e = if u.value() >= 0.0 {
        (-u).exp()
    } else {
        u.exp()
    };
    -e / (kt * (S::from_f64(1.0) + e).powi(2))
}

fn fermi_second<S: Scalar>(x: S, kt: S) -> S {
    fermi_prime(x, kt) * (fermi(x / kt).scale(2.0) - S::from_f64(1.0)) / kt
}

fn bose<S: Scalar>(omega: S, kt: S) -> S {
    let u = omega / kt;
    if u.value() < 0.5 {
        exp_minus_one(u).recip()
    } else {
        let e = (-u).exp();
        e / (S::from_f64(1.0) - e)
    }
}

fn tanh_half<S: Scalar>(x: S) -> S {
    // Called only away from zero, so subtraction cannot remove the limit.
    S::from_f64(1.0) - fermi(x).scale(2.0)
}

fn exp_minus_one<S: Scalar>(x: S) -> S {
    if x.value().abs() < 0.5 {
        let mut term = x;
        let mut sum = x;
        for k in 2..=24 {
            term = term * x.scale(1.0 / k as f64);
            sum = sum + term;
        }
        sum
    } else {
        x.exp() - S::from_f64(1.0)
    }
}
