// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Published exchange partners of the Dion and Lee nonlocal density kernels.

use super::InternalExchange;
use crate::xc::gga::{pbe_enhancement, pw86_enhancement, rpbe_enhancement};
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Exchange and nonlocal correlation pairing of the vdW-DF family.
#[derive(Clone, Copy, Debug)]
pub enum Variant {
    /// Dion et al. (2004), revPBE outer exchange.
    VdwDf,
    /// Lee et al. (2010), revised PW86 outer exchange.
    VdwDf2,
    /// Klimeš, Bowler and Michaelides (2010), published PBE/RPBE mixture.
    OptPbe,
    /// Klimeš, Bowler and Michaelides (2010), optimized B88 enhancement.
    OptB88,
    /// Klimeš, Bowler and Michaelides (2011), optimized B86b enhancement.
    OptB86b,
    /// Berland and Hyldgaard (2014), Langreth–Vosko to PW86r exchange.
    Cx,
    /// Hamada (2014), revised B86b exchange and the Lee kernel.
    RevVdwDf2,
}

fn b86b(s2: f64, mu: f64, kappa: f64) -> f64 {
    let gradient = mu * s2;
    1.0 + gradient / (1.0 + gradient / kappa).powf(0.8)
}

impl Variant {
    /// The canonical molecular fixed-density energy selector.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "vdw-df" => Some(Self::VdwDf),
            "vdw-df2" => Some(Self::VdwDf2),
            "optpbe-vdw" => Some(Self::OptPbe),
            "optb88-vdw" => Some(Self::OptB88),
            "optb86b-vdw" => Some(Self::OptB86b),
            "vdw-df-cx" => Some(Self::Cx),
            "rev-vdw-df2" => Some(Self::RevVdwDf2),
            _ => None,
        }
    }

    /// The canonical name of the selected XC variant.
    pub fn name(self) -> &'static str {
        match self {
            Self::VdwDf => "vdw-df",
            Self::VdwDf2 => "vdw-df2",
            Self::OptPbe => "optpbe-vdw",
            Self::OptB88 => "optb88-vdw",
            Self::OptB86b => "optb86b-vdw",
            Self::Cx => "vdw-df-cx",
            Self::RevVdwDf2 => "rev-vdw-df2",
        }
    }

    /// Inner exchange defining the family member's nonlocal kernel.
    pub fn internal_exchange(self) -> InternalExchange {
        match self {
            Self::VdwDf2 | Self::RevVdwDf2 => InternalExchange::Lee,
            _ => InternalExchange::Dion,
        }
    }

    /// Outer exchange enhancement at a nonnegative reduced gradient.
    pub fn exchange_enhancement(self, s: f64) -> Result<f64> {
        if !s.is_finite() || s < 0.0 {
            return Err(DftError::InvalidInput(
                "vdW-DF outer exchange requires a finite nonnegative reduced gradient".into(),
            ));
        }
        let s2 = s * s;
        let pw86r = || pw86_enhancement(s2, [1.851, 17.33, 0.163]);
        let enhancement = match self {
            Self::VdwDf => pbe_enhancement(s2, 1.245, 0.2195149727645171),
            Self::VdwDf2 => pw86r(),
            Self::OptPbe => {
                // Use the final publication's constants instead of the rounded preprint fit.
                let (k, mu, pbe) = (1.04804, 0.175519, 0.945268);
                pbe * pbe_enhancement(s2, k, mu) + (1.0 - pbe) * rpbe_enhancement(s2, k, mu)
            }
            Self::OptB88 => {
                let mu = 0.22;
                let c = (48.0 * PI * PI).cbrt();
                1.0 + mu * s2 / (1.0 + (mu / 1.2) * s * (c * s).asinh())
            }
            Self::OptB86b => b86b(s2, 0.1234, 1.0),
            Self::Cx => {
                let tuning = 0.02178 * s2 * s2 * s2;
                (1.0 + (0.8491 / 9.0) * s2) / (1.0 + tuning) + tuning / (1.15 + tuning) * pw86r()
            }
            Self::RevVdwDf2 => b86b(s2, 10.0 / 81.0, 0.7114),
        };
        if !enhancement.is_finite() {
            return Err(DftError::InvalidInput(
                "vdW-DF outer exchange enhancement is nonfinite".into(),
            ));
        }
        Ok(enhancement)
    }
}

/// The three directly integrated vdW-DF XC terms at a fixed unpolarized density.
#[derive(Clone, Copy, Debug)]
pub struct MolecularVdwDfEnergy {
    /// Outer exchange (Hartree).
    pub exchange: f64,
    /// PW92 local correlation (Hartree).
    pub local_correlation: f64,
    /// Dion-family nonlocal correlation (Hartree).
    pub nonlocal_correlation: f64,
}

impl MolecularVdwDfEnergy {
    /// The total exchange-correlation energy (Hartree).
    pub fn total(self) -> f64 {
        self.exchange + self.local_correlation + self.nonlocal_correlation
    }
}
