// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Tao-Mo meta-GGA exchange and correlation.
//!
//! Tao and Mo, Phys. Rev. Lett. 117, 073001 (2016), equations (7), (9),
//! (10)--(12), combine density-matrix expansion (DME) exchange with a
//! fourth-order gradient expansion and modify the TPSS correlation
//! spin-dependent factor. Jana, Sharma and Samal, J. Phys. Chem. A 123,
//! 6356 (2019), doi:10.1021/acs.jpca.9b02921, replace the Laplacian proxy
//! in exchange and use the revTPSS density-dependent beta in correlation.

use super::common::{K_FACTOR_C, RHO_CUTOFF, X2S, X_FACTOR_C};
use super::tpss::tpss_c_e;
use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

const EXCHANGE_PARAMETERS: &[&str] = &["lambda", "beta", "q_b"];

// Exact exchange spin scaling permits the published unpolarized formula
// to be evaluated separately at each doubled spin density. The per-spin
// uniform kinetic-energy and Slater coefficients already include that scaling.
fn exchange_spin<S: Scalar>(rho: S, sigma: S, tau: S, params: &[S]) -> S {
    let one = S::from_f64(1.0);
    let p = sigma * S::from_f64(X2S * X2S) / rho.powf(8.0 / 3.0);
    let t = tau / (S::from_f64(K_FACTOR_C) * rho.powf(5.0 / 3.0));
    let t_w = S::from_f64(5.0 / 3.0) * p;
    let alpha = t - t_w;
    // TM has an order-of-limits ambiguity at p = tau = 0. At that single
    // point use the continuation along p = 0. This body adds no kinetic floor.
    let z = if tau.value() == 0.0 {
        S::from_f64(0.0)
    } else {
        t_w / t
    };
    let z2 = z * z;
    let z3 = z2 * z;
    let den = one + z3;
    let w = (z2 + S::from_f64(3.0) * z3) / (den * den);

    let lambda = params[0];
    let two_lambda_minus_one = lambda + lambda - one;
    let y = two_lambda_minus_one * two_lambda_minus_one * p;
    let inverse_f2 = (one + S::from_f64(700.0 / 27.0) * y + params[1] * y * y).powf(-0.2);
    let transformed = S::from_f64(3.0) * (lambda * lambda - lambda + S::from_f64(0.5));
    // The term 5p/27 is |grad n|^2 / (72 n tau_unif).
    let r = one + S::from_f64(595.0 / 54.0) * y - t
        + transformed * (t - one - S::from_f64(5.0 / 27.0) * p);
    let dme = inverse_f2 * (one + S::from_f64(7.0 / 9.0) * r * inverse_f2);

    let q = S::from_f64(9.0 / 20.0) * (alpha - one)
        / (one + params[2] * alpha * (alpha - one)).sqrt()
        + S::from_f64(2.0 / 3.0) * p;
    let gradient_expansion = (S::from_f64(10.0 / 81.0) + S::from_f64(50.0 / 729.0) * p) * p
        + S::from_f64(146.0 / 2025.0) * q * q
        - S::from_f64(73.0 / 675.0) * q * z * (one - z);
    let slowly_varying = (one + S::from_f64(10.0) * gradient_expansion).powf(0.1);
    -S::from_f64(X_FACTOR_C) * rho.powf(4.0 / 3.0) * (w * dme + (one - w) * slowly_varying)
}

fn exchange<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    [(ra, saa, ta), (rb, sbb, tb)]
        .into_iter()
        .filter(|(rho, _, _)| rho.value() >= RHO_CUTOFF)
        .fold(S::from_f64(0.0), |energy, (rho, sigma, tau)| {
            energy + exchange_spin(rho, sigma, tau, th)
        })
}

fn exchange_model(name: &'static str, q_b: f64) -> Box<dyn XcFunctional> {
    mgga_driver!(
        name,
        Kind::Exchange,
        vec![0.6866, 79.873, q_b],
        EXCHANGE_PARAMETERS,
        exchange,
        true
    )
}

fn correlation_model(name: &'static str, beta_rs: [f64; 2]) -> Box<dyn XcFunctional> {
    let tpss = super::tpss::tpss_c();
    let mut params = tpss.params();
    // The first 21 TPSS parameters define d, the PBE correlation coefficients
    // and the fitted electron gas. The last six define C(zeta, 0) and beta(rs).
    params[21..].copy_from_slice(&[0.0, 0.1, 0.32, 0.0, beta_rs[0], beta_rs[1]]);
    mgga_driver!(
        name,
        Kind::Correlation,
        params,
        tpss.param_names(),
        tpss_c_e,
        true
    )
}

/// Original TM exchange, with the unmodified fourth-order Laplacian proxy.
pub fn tm_x() -> Box<dyn XcFunctional> {
    exchange_model("tm", 0.0)
}

/// Revised TM exchange, with the b = 0.4 Laplacian proxy.
pub fn revtm_x() -> Box<dyn XcFunctional> {
    exchange_model("revtm", 0.4)
}

/// Original TM correlation, with the PBE constant beta.
pub fn tm_c() -> Box<dyn XcFunctional> {
    correlation_model("tm", [0.0, 0.0])
}

/// Revised TM correlation, with the revTPSS beta(rs).
pub fn revtm_c() -> Box<dyn XcFunctional> {
    correlation_model("revtm", [0.1, 0.1778])
}
