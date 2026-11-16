// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Two-photon annihilation rate of a positron density with the electrons.
//!
//! In Hartree atomic units the spin-averaged rate is
//!
//! ```text
//! lambda = pi r_e^2 c int n_+ n_- gamma(n_-) = (pi / c^3) int n_+ n_- gamma,
//! ```
//!
//! with the classical electron radius `r_e = 1/c^2`, the speed of light
//! `c = 1/alpha` and the enhancement factor `gamma` (the contact value of
//! the electron-positron pair-correlation function in the electron gas at
//! the local density parameter `r_s`). For a positron in the electron gas
//! this is `3 gamma / (4 c^3 r_s^3)`. The lifetime is `1 / lambda`.
//!
//! Enhancement factors:
//!
//! * `ipm`: the independent-particle model, `gamma = 1`.
//! * `drummond-2011`: diffusion Monte Carlo contact values, Drummond, Lopez
//!   Rios, Needs and Pickard, Phys. Rev. Lett. 107, 207402 (2011), Eq. (4):
//!   `1 + 1.23 r_s - 3.38208 r_s^(3/2) + 8.6957 r_s^2 - 7.37037 r_s^(7/3)
//!   + 1.75648 r_s^(8/3) + 0.173694 r_s^3`.
//! * `barbiellini-ap`, `barbiellini-sl`, `barbiellini-qmc`: the
//!   one-parameter form `1 + 1.23 r_s + p r_s^2 + r_s^3/3` of Barbiellini,
//!   Puska, Torsti and Nieminen, Phys. Rev. B 51, 7341 (1995), with
//!   `p = -0.0742` (Arponen-Pajanne data), `-0.1375` (Stachowiak-Lach) and
//!   `-0.22` (quantum Monte Carlo), as quoted by Kuriplach and Barbiellini
//!   (2014).
//!
//! With `gradient_alpha` the gradient correction of Barbiellini et al.,
//! `gamma = 1 + (gamma_LDA - 1) exp(-alpha epsilon)`, with
//! `epsilon = |grad n_-|^2 / (n_- q_TF)^2` and the Thomas-Fermi screening
//! wavevector `q_TF^2 = 4 k_F / pi`, `k_F = (3 pi^2 n_-)^(1/3)`, is applied
//! (`alpha = 0.22` for the Arponen-Pajanne enhancement, 0.05 for the
//! quantum Monte Carlo one, 0.10 for Stachowiak-Lach).

use crate::basis::MolBasis;
use crate::grid::{eval::eval_aos, MolecularGrid};
use crate::linalg::Matrix;
use crate::{DftError, Result};

/// The contact enhancement factor `gamma(r_s)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Enhancement {
    IndependentParticle,
    Drummond2011,
    /// `1 + 1.23 r_s + p r_s^2 + r_s^3/3`.
    Barbiellini {
        p: f64,
    },
}

impl Enhancement {
    pub fn by_name(name: &str) -> Result<Self> {
        Ok(match name {
            "ipm" | "none" => Self::IndependentParticle,
            "drummond-2011" | "qmc-2011" => Self::Drummond2011,
            "barbiellini-ap" => Self::Barbiellini { p: -0.0742 },
            "barbiellini-sl" => Self::Barbiellini { p: -0.1375 },
            "barbiellini-qmc" => Self::Barbiellini { p: -0.22 },
            _ => {
                return Err(DftError::InvalidInput(
                    "annihilation_enhancement must be ipm, drummond-2011, barbiellini-ap, \
                     barbiellini-sl or barbiellini-qmc"
                        .into(),
                ))
            }
        })
    }

    /// Powers and coefficients of gamma-1. The contact factor and its
    /// density-weighted evaluation share exactly the same physical model.
    fn terms(&self) -> [(f64, f64); 6] {
        match *self {
            Self::IndependentParticle => [(1.0, 0.0); 6],
            Self::Drummond2011 => [
                (1.0, 1.23),
                (1.5, -3.38208),
                (2.0, 8.6957),
                (7.0 / 3.0, -7.37037),
                (8.0 / 3.0, 1.75648),
                (3.0, 0.173694),
            ],
            Self::Barbiellini { p } => [
                (1.0, 1.23),
                (2.0, p),
                (3.0, 1.0 / 3.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
            ],
        }
    }

    pub fn factor(&self, rs: f64) -> f64 {
        self.terms()
            .iter()
            .filter(|(_, c)| *c != 0.0)
            .fold(1.0, |v, (power, coefficient)| {
                v + coefficient * rs.powf(*power)
            })
    }

    /// n*gamma at n>0. Individual gamma terms can overflow while this
    /// physical contact density is finite, including its dilute limit.
    fn weighted_density(&self, n: f64) -> f64 {
        let sphere = 3.0 / (4.0 * std::f64::consts::PI);
        self.terms()
            .iter()
            .filter(|(_, c)| *c != 0.0)
            .fold(n, |v, (power, coefficient)| {
                v + coefficient * sphere.powf(power / 3.0) * n.powf(1.0 - power / 3.0)
            })
    }
}

/// The annihilation of one positron density.
#[derive(Clone, Copy, Debug)]
pub struct Annihilation {
    /// `int n_+ n_-` (Bohr^-3).
    pub contact_density: f64,
    /// `int n_+ n_- gamma` (Bohr^-3).
    pub enhanced_contact_density: f64,
    /// Rate in inverse atomic units of time.
    pub rate: f64,
}

impl Annihilation {
    /// Rate in inverse nanoseconds.
    pub fn rate_per_ns(&self) -> f64 {
        self.rate / (crate::constants::AU_TIME_SECONDS * 1e9)
    }
}

/// Total electron density (and its gradient, when asked for) on the grid.
fn electron_density(
    basis: &MolBasis,
    density: &[Matrix; 2],
    grid: &MolecularGrid,
    gradient: bool,
) -> (Vec<f64>, Option<Vec<[f64; 3]>>) {
    let mut total = density[0].clone();
    for (a, b) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
        *a += b;
    }
    let aos = eval_aos(basis, &grid.points, gradient);
    let pd = aos.phi.matmul_seq(&total);
    let n = grid.points.len();
    let values = (0..n)
        .map(|g| {
            (0..basis.nao_sph)
                .map(|m| pd[(g, m)] * aos.phi[(g, m)])
                .sum()
        })
        .collect();
    let gradients = aos.grad.as_ref().map(|grad| {
        (0..n)
            .map(|g| {
                std::array::from_fn(|k| {
                    2.0 * (0..basis.nao_sph)
                        .map(|m| pd[(g, m)] * grad[k][(g, m)])
                        .sum::<f64>()
                })
            })
            .collect()
    });
    (values, gradients)
}

/// The annihilation rate of a positron with the AO density `positron` in
/// `positron_basis`, against the electronic spin densities `electronic`.
pub fn annihilation(
    electronic_basis: &MolBasis,
    electronic: &[Matrix; 2],
    positron_basis: &MolBasis,
    positron: &Matrix,
    grid: &MolecularGrid,
    enhancement: Enhancement,
    gradient_alpha: Option<f64>,
) -> Result<Annihilation> {
    if gradient_alpha.is_some_and(|a| !a.is_finite() || a < 0.0) {
        return Err(DftError::InvalidInput(
            "annihilation_gradient_alpha must be finite and non-negative".into(),
        ));
    }
    let (electron, gradient) =
        electron_density(electronic_basis, electronic, grid, gradient_alpha.is_some());
    let aos = eval_aos(positron_basis, &grid.points, false);
    let pd = aos.phi.matmul_seq(positron);
    let c = 1.0 / crate::constants::FINE_STRUCTURE;
    let mut contact = 0.0;
    let mut enhanced = 0.0;
    for (g, &n) in electron.iter().enumerate() {
        if n <= 0.0 {
            continue;
        }
        let np: f64 = (0..positron_basis.nao_sph)
            .map(|m| pd[(g, m)] * aos.phi[(g, m)])
            .sum();
        let mut weighted = enhancement.weighted_density(n);
        if let (Some(alpha), Some(gradient)) = (gradient_alpha, gradient.as_ref()) {
            // alpha=0 is exactly the LDA model, even when epsilon itself
            // overflows. Form grad(n)/n before its norm, avoiding n*n.
            if alpha != 0.0 {
                let k_f = (3.0 * std::f64::consts::PI.powi(2) * n).cbrt();
                let q_tf2 = 4.0 * k_f / std::f64::consts::PI;
                let relative = gradient[g].iter().fold(0.0_f64, |v, x| v.hypot(x / n));
                let epsilon = relative * relative / q_tf2;
                let damping = (-alpha * epsilon).exp();
                weighted = n + (weighted - n) * damping;
            }
        }
        contact += grid.weights[g] * np * n;
        enhanced += grid.weights[g] * np * weighted;
    }
    if !contact.is_finite() || !enhanced.is_finite() {
        return Err(DftError::InvalidInput(
            "non-representable positron annihilation contact density".into(),
        ));
    }
    Ok(Annihilation {
        contact_density: contact,
        enhanced_contact_density: enhanced,
        rate: std::f64::consts::PI / c.powi(3) * enhanced,
    })
}
