// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Reoptimized exchange and kinetic-energy-density correlation ingredients
//! of PWPB95 (Goerigk and Grimme, JCTC 7, 291 (2011)). Parameters are
//! exposed through the same analytic and Taylor interfaces as other XC parts.

use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

use super::common::{X2S, X_FACTOR_C as CX};

/// The generalized PW91 exchange
/// `F = (1 + a s asinh(s/X2S) + (c + d e^(-alpha x^2)) s^2) / (1 + a s asinh(s/X2S) + 1e-6 x^n / CX)`
/// with `a, c, d` fixed by `bt` (PW91, mPW91, PWPB95).
/// theta = [bt, alpha (in x^2), n].
pub(crate) fn pw_exchange<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, p: &[S]) -> S {
    let one = S::from_f64(1.0);
    let beta = S::from_f64(5.0 * (36.0 * std::f64::consts::PI).powf(-5.0 / 3.0));
    let bt = p[0];
    let a = bt.scale(6.0 / X2S);
    let c = bt.scale(1.0 / (CX * X2S * X2S));
    let d = -(bt - beta).scale(1.0 / (CX * X2S * X2S));
    let alpha = p[1].scale(1.0 / (X2S * X2S));
    let mut energy = S::from_f64(0.0);
    for (rho, sigma) in [(ra, saa), (rb, sbb)] {
        if rho.value() < 1e-13 {
            continue;
        }
        let x2 = sigma / rho.powf(8.0 / 3.0);
        let s2 = x2.scale(X2S * X2S);
        // s*asinh(s/X2S) is analytic in s^2 at zero. This local series
        // avoids differentiating sqrt(0) without changing the gradient.
        let q = x2;
        let sasinh = if q.value() < 1e-8 {
            (q - q * q.scale(1.0 / 6.0) + q.powi(3).scale(3.0 / 40.0)
                - q.powi(4).scale(5.0 / 112.0))
            .scale(X2S)
        } else {
            let x = q.sqrt();
            x.scale(X2S) * x.asinh()
        };
        let power = if super::gga::sigma_is_absent(sigma) {
            // The power x^n (n < 4) has a divergent second sigma derivative
            // at sigma = 0: below the gradient cutoff the term is absent (the
            // declared absent-gradient convention, value exactly zero).
            S::from_f64(0.0)
        } else {
            (p[2] * x2.ln().scale(0.5)).exp()
        };
        let tail = power.scale(1e-6 / CX);
        let common = one + a * sasinh;
        let numerator = common + (c + d * (-alpha * s2).exp()) * s2;
        let enhancement = numerator / (common + tail);
        energy = energy - rho.powf(4.0 / 3.0).scale(CX) * enhancement;
    }
    energy
}

fn b95_correlation<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, ta: S, tb: S, p: &[S]) -> S {
    let one = S::from_f64(1.0);
    let zero = S::from_f64(0.0);
    let k = 0.3 * (6.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0);
    let pw = super::m06::pw_mod_params::<S>();
    let mut e = zero;
    let mut parallel = zero;
    let mut x = [zero; 2];
    for (i, (rho, sigma, tau)) in [(ra, saa, ta), (rb, sbb, tb)].into_iter().enumerate() {
        if rho.value() < 1e-13 {
            continue;
        }
        let x2 = sigma / rho.powf(8.0 / 3.0);
        x[i] = x2;
        let t = tau / rho.powf(5.0 / 3.0);
        let ec = rho * super::lda::pw92_epsilon_ferro(rho, &pw);
        parallel = parallel + ec;
        // t*D = t-x^2/8: no division by tau and exact one-orbital limit.
        let factor = (t - x2.scale(0.125)).scale(1.0 / k) / (one + p[0] * x2).powi(2);
        e = e + ec * factor;
    }
    if ra.value() >= 1e-13 && rb.value() >= 1e-13 {
        let antiparallel = (ra + rb) * super::lda::pw92_epsilon(ra, rb, &pw) - parallel;
        e = e + antiparallel / (one + p[1] * (x[0] + x[1]));
    }
    e
}

/// A generalized PW91 exchange with parameters [bt, alpha, n].
pub(crate) fn pw_form_exchange(name: &'static str, params: [f64; 3]) -> Box<dyn XcFunctional> {
    gga_driver!(
        name,
        Kind::Exchange,
        params.to_vec(),
        &["beta", "c", "exponent"],
        pw_exchange
    )
}

/// Reoptimized PW exchange; parameters are beta, c and the fitted exponent.
pub fn pwpb95_x() -> Box<dyn XcFunctional> {
    pw_form_exchange("pwpb95_x", [0.00444, 0.32620, 3.7868])
}

/// B95 correlation with PWPB95's separately fitted spin-channel parameters.
pub fn pwpb95_c() -> Box<dyn XcFunctional> {
    b95_with("pwpb95_c", [0.03241, 0.00250])
}

/// B95 correlation with Becke's parameters (same spin 0.038, opposite
/// spin 0.0031).
pub fn b95_c() -> Box<dyn XcFunctional> {
    b95_with("b95", [0.038, 0.0031])
}

/// PW6B95 (Zhao-Truhlar, JPCA 109, 5656 (2005)) exchange and correlation.
pub fn pw6b95_parts() -> (Box<dyn XcFunctional>, Box<dyn XcFunctional>) {
    (
        pw_form_exchange("pw6b95_x", [0.00538, 1.7382, 3.8901]),
        b95_with("pw6b95_c", [0.03668, 0.00262]),
    )
}

/// PWB6K (Zhao-Truhlar 2005) exchange and correlation.
pub fn pwb6k_parts() -> (Box<dyn XcFunctional>, Box<dyn XcFunctional>) {
    (
        pw_form_exchange("pwb6k_x", [0.00539, 1.7077, 4.0876]),
        b95_with("pwb6k_c", [0.04120, 0.00353]),
    )
}

/// B95 correlation (Becke, JCP 104, 1040 (1996)) with the same-spin and
/// opposite-spin damping parameters.
pub(crate) fn b95_with(name: &'static str, params: [f64; 2]) -> Box<dyn XcFunctional> {
    mgga_driver!(
        name,
        Kind::Correlation,
        params.to_vec(),
        &["same_spin_damping", "opposite_spin_damping"],
        b95_correlation,
        true
    )
}
