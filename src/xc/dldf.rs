// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Dispersionless meta-GGA hybrid of Pernal, Podeszwa, Patkowski and
//! Szalewicz, PRL 103, 263201 (2009). This is the local part of dlDF;
//! the separately declared global HF fraction is 0.6144129. No empirical
//! intermolecular dispersion function is included in this XC definition.

use super::common::{RHO_CUTOFF, X2S, X_FACTOR_C};
use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

fn polynomial<S: Scalar>(u: S, coefficients: &[S]) -> S {
    coefficients
        .iter()
        .rev()
        .fold(S::from_f64(0.0), |sum, &c| sum * u + c)
}

pub(crate) fn exchange_energy<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    let mut energy = S::from_f64(0.0);
    for (rho, sigma, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let r53 = rho.powf(5.0 / 3.0);
        let x2 = sigma / (r53 * rho);
        let reduced = x2 * S::from_f64(X2S * X2S);
        let enhancement =
            S::from_f64(1.0) + th[1] - th[1] / (S::from_f64(1.0) + th[2] * reduced / th[1]);
        let uniform_tau =
            S::from_f64(0.3 * (6.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0)) * r53;
        let w = (uniform_tau - tau) / (uniform_tau + tau);
        energy = energy
            - S::from_f64(X_FACTOR_C)
                * rho.powf(4.0 / 3.0)
                * th[0]
                * enhancement
                * polynomial(w, &th[3..8]);
    }
    energy
}

/// Analytic removable limit of `(1-x2/(8*t))*(1-exp(-4*t^2/eta^2))`.
/// Rewriting the exponential quotient preserves all seeded derivatives at
/// t=0. The small-q series is numerical evaluation, not a physical cutoff.
fn corrected_curvature<S: Scalar>(x2: S, t: S, eta: S) -> S {
    let coefficient = S::from_f64(4.0) / (eta * eta);
    let q = coefficient * t * t;
    if q.value() <= 0.01 {
        // phi(q)=(1-exp(-q))/q. At q<=.01 the first omitted term is
        // bounded by .01^13/14!, far below binary64 precision.
        let mut term = S::from_f64(1.0);
        let mut phi = term;
        for k in 1..=12 {
            term = term * (-q) / S::from_f64((k + 1) as f64);
            phi = phi + term;
        }
        (t - x2 / S::from_f64(8.0)) * coefficient * t * phi
    } else {
        (S::from_f64(1.0) - x2 / (S::from_f64(8.0) * t)) * (S::from_f64(1.0) - (-q).exp())
    }
}

pub(crate) fn correlation_energy<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let mut energy = zero;
    let mut x2 = [zero; 2];
    let mut parallel = [zero; 2];
    let mut have = [false; 2];
    let pw = super::lda::PW92_MOD_PARAMS.map(S::from_f64);
    for (i, (rho, sigma, tau)) in [(ra, saa, ta), (rb, sbb, tb)].into_iter().enumerate() {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        have[i] = true;
        let r53 = rho.powf(5.0 / 3.0);
        x2[i] = sigma / (r53 * rho);
        parallel[i] = rho * super::lda::pw92_epsilon_ferro(rho, &pw);
        let u = th[0] * x2[i] / (one + th[0] * x2[i]);
        energy = energy
            + parallel[i]
                * polynomial(u, &th[2..7])
                * corrected_curvature(x2[i], tau / r53, th[12]);
    }
    if have[0] && have[1] {
        let perpendicular =
            (ra + rb) * super::lda::pw92_epsilon(ra, rb, &pw) - parallel[0] - parallel[1];
        let scaled = th[1] * (x2[0] + x2[1]);
        energy = energy + perpendicular * polynomial(scaled / (one + scaled), &th[7..12]);
    }
    energy
}

/// The fitted local exchange already contains its semilocal fraction.
/// Use the shared value-only Fermi-hole input convention outside the
/// physical sigma <= 8*rho*tau domain; derivative seeds remain independent.
pub fn dldf_x() -> Box<dyn XcFunctional> {
    let params = vec![
        1.0 - 0.6144129,
        4.8827323,
        0.3511128,
        1.0,
        -0.1637571,
        -0.1880028,
        -0.4490609,
        -0.0082359,
    ];
    let names = &[
        "local_fraction",
        "kappa",
        "mu",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
    ];
    mgga_driver!(
        "dldfx",
        Kind::Exchange,
        params,
        names,
        exchange_energy,
        true
    )
}

/// PW92-MOD spin decomposition and the corrected Fermi-hole factor.
/// The shared driver supplies the same Fermi-hole input convention as X.
pub fn dldf_c() -> Box<dyn XcFunctional> {
    let params = vec![
        0.06,
        0.0031,
        1.0,
        -2.5960897,
        2.2233793,
        0.0,
        0.0,
        1.0,
        5.9515308,
        -11.1602877,
        0.0,
        0.0,
        1e-10,
    ];
    let names = &[
        "gamma_ss",
        "gamma_ab",
        "css0",
        "css1",
        "css2",
        "css3",
        "css4",
        "cab0",
        "cab1",
        "cab2",
        "cab3",
        "cab4",
        "fermi_regularizer",
    ];
    mgga_driver!(
        "dldfc",
        Kind::Correlation,
        params,
        names,
        correlation_energy,
        true
    )
}
