// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! PKZB meta-GGA: Perdew, Kurth, Zupan and Blaha, PRL 82, 2544
//! (1999), doi:10.1103/PhysRevLett.82.2544; erratum 82, 5179.
//!
//! Exchange uses the fourth-order gradient expansion with D=0.113.
//! Correlation uses the original spin-channel kinetic indicators, not the
//! total-density indicator or max construction of revised PKZB in TPSS.

use super::common::RHO_CUTOFF;
use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

fn exchange<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    let one = S::from_f64(1.0);
    let kappa = th[0];
    let mu = S::from_f64(10.0 / 81.0);
    let kf2 = (6.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0);
    let mut energy = S::from_f64(0.0);
    for (rho, sigma, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let p = sigma / (S::from_f64(4.0 * kf2) * rho.powf(8.0 / 3.0));
        let tau_uniform = S::from_f64(0.3 * kf2) * rho.powf(5.0 / 3.0);
        let q = S::from_f64(9.0 / 20.0) * (tau / tau_uniform - one) - p * S::from_f64(1.0 / 12.0);
        let x = mu * p + S::from_f64(146.0 / 2025.0) * q * q - S::from_f64(73.0 / 405.0) * q * p
            + (th[1] + mu * mu / kappa) * p * p;
        let enhancement = one + kappa - kappa / (one + x / kappa);
        energy =
            energy - S::from_f64(super::common::X_FACTOR_C) * rho.powf(4.0 / 3.0) * enhancement;
    }
    energy
}

fn correlation<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return zero;
    }
    let (c, beta, gamma) = (th[0], th[1], th[2]);
    let pw = &th[3..];
    let have_a = ra.value() >= RHO_CUTOFF;
    let have_b = rb.value() >= RHO_CUTOFF;
    if !have_a && !have_b {
        return zero;
    }
    let eps = if !have_b {
        super::gga::pbe_c_eps_ferro(ra, saa, beta, gamma, pw)
    } else if !have_a {
        super::gga::pbe_c_eps_ferro(rb, sbb, beta, gamma, pw)
    } else {
        super::gga::pbe_c_eps(ra, rb, saa, sab, sbb, beta, gamma, pw)
    };
    let mut tau_w_sum = zero;
    let mut parallel = zero;
    let mut tau_sum = zero;
    for (density, sigma, kinetic) in [(ra, saa, ta), (rb, sbb, tb)] {
        if density.value() < RHO_CUTOFF {
            continue;
        }
        let kinetic = if kinetic.value() < 1e-20 {
            S::from_f64(1e-20)
        } else {
            kinetic
        };
        let tau_w = sigma / (S::from_f64(8.0) * density);
        let indicator = tau_w / kinetic;
        let eps_spin = super::gga::pbe_c_eps_ferro(density, sigma, beta, gamma, pw);
        tau_w_sum = tau_w_sum + tau_w;
        tau_sum = tau_sum + kinetic;
        parallel = parallel + density * indicator * indicator * eps_spin;
    }
    let indicator = tau_w_sum / tau_sum;
    rho * eps * (one + c * indicator * indicator) - (one + c) * parallel
}

/// Original PKZB exchange, with differentiable kappa and D parameters.
pub fn pkzb_x() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "pkzb",
        Kind::Exchange,
        vec![0.804, 0.113],
        &["kappa", "D"],
        exchange,
        true
    )
}

/// Original PKZB correlation, using the PBE/PW92-MOD convention.
pub fn pkzb_c() -> Box<dyn XcFunctional> {
    let mut params = vec![0.53, 0.06672455060314922, 0.031090690869654895];
    params.extend_from_slice(&super::lda::PW92_MOD_PARAMS);
    mgga_driver!(
        "pkzb",
        Kind::Correlation,
        params,
        &[
            "C", "beta", "gamma", "A0", "alpha0", "beta00", "beta01", "beta02", "beta03", "A1",
            "alpha1", "beta10", "beta11", "beta12", "beta13", "A2", "alpha2", "beta20", "beta21",
            "beta22", "beta23",
        ],
        correlation,
        true
    )
}
