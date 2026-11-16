// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Pieces shared by every functional body: the density cutoff, the
//! Wigner-Seitz radius, the spin-interpolation function and the macros that
//! instantiate a generic energy density for the seven scalar types of the
//! drivers (value, Taylor series, Taylor potentials, dense orders 1-3 and
//! the parameter tangent). A new functional is one generic body plus one
//! macro call.

use crate::dual::Scalar;

/// `1 / (2 (6 pi^2)^(1/3))`: the per-spin reduced gradient
/// `x = |grad rho_s| / rho_s^(4/3)` times this is the `s` of the
/// spin-scaled density.
pub(crate) const X2S: f64 = 0.128_278_243_853_042_2;
/// Minus the per-spin LDA exchange prefactor, `3/8 (3/pi)^(1/3) 4^(2/3)`.
pub(crate) const X_FACTOR_C: f64 = 0.930_525_736_349_100_1;
/// `3/10 (6 pi^2)^(2/3)`: `tau_unif = K_FACTOR_C rho_s^(5/3)` per spin.
pub(crate) const K_FACTOR_C: f64 = 4.557_799_872_345_597;

/// Densities below this are treated as absent.
pub(crate) const RHO_CUTOFF: f64 = 1e-13;

/// Wigner-Seitz radius `rs = (3 / (4 pi rho))^(1/3)`.
pub(crate) fn wigner_seitz<S: Scalar>(rho: S) -> S {
    (S::from_f64(3.0 / (4.0 * std::f64::consts::PI)) / rho).cbrt()
}

/// Spin-interpolation function
/// `f(zeta) = ((1 + zeta)^(4/3) + (1 - zeta)^(4/3) - 2) / (2^(4/3) - 2)`.
pub(crate) fn f_zeta<S: Scalar>(z: S) -> S {
    let one = S::from_f64(1.0);
    ((one + z).powf(4.0 / 3.0) + (one - z).powf(4.0 / 3.0) - S::from_f64(2.0))
        * S::from_f64(1.0 / (2.0_f64.powf(4.0 / 3.0) - 2.0))
}

/// Principal branch of the Lambert W function, `W e^W = y` (y >= 0), for
/// every scalar type: the value from Newton iterations on f64 and then
/// Newton steps in `S`, each of which doubles the number of exact Taylor
/// orders of the result.
pub(crate) fn lambert_w<S: Scalar>(y: S) -> S {
    let target = y.value();
    let mut w = if target < 1.0 { target } else { target.ln() };
    for _ in 0..100 {
        let e = w.exp();
        let step = (w * e - target) / (e * (1.0 + w));
        w -= step;
        if step.abs() <= 1e-16 * (1.0 + w.abs()) {
            break;
        }
    }
    let mut ws = S::from_f64(w);
    for _ in 0..6 {
        let e = ws.exp();
        ws = ws - (ws * e - y) / (e * (S::from_f64(1.0) + ws));
    }
    ws
}

/// A GGA driver over the generic body `$body(ra, rb, saa, sab, sbb, theta)`.
macro_rules! gga_driver {
    ($name:expr, $kind:expr, $params:expr, $names:expr, $body:ident) => {
        gga_driver!($name, $kind, $params, $names, $body; scaling = &[])
    };
    ($name:expr, $kind:expr, $params:expr, $names:expr, $body:ident; scaling = $scaling:expr) => {
        Box::new($crate::xc::gga::GgaDriver {
            param_scaling: $scaling,
            name: $name,
            kind: $kind,
            params: $params,
            param_names: $names,
            f0: $body::<f64>,
            fj: $body::<$crate::response::series::SeriesScalar>,
            fjt: $body::<$crate::dual::DualN<$crate::response::series::SeriesScalar, 5>>,
            fjv: $body::<$crate::xc::taylor::ActiveDirectionalTaylorScalar<5>>,
            fjb: $body::<$crate::xc::bidirectional::BidirectionalTaylorScalar>,
            f1: $body::<$crate::dual::DualN<f64, 5>>,
            f2: $body::<$crate::dual::DualN<$crate::dual::DualN<f64, 5>, 5>>,
            f3: $body::<
                $crate::dual::DualN<$crate::dual::DualN<$crate::dual::DualN<f64, 5>, 5>, 5>,
            >,
            fp: $body::<$crate::dual::DualN<f64, 1>>,
        })
    };
}

/// A meta-GGA driver over `$body(ra, rb, saa, sab, sbb, ta, tb, theta)`;
/// `$fhc` enforces the Fermi-hole-curvature bound on the inputs.
macro_rules! mgga_driver {
    ($name:expr, $kind:expr, $params:expr, $names:expr, $body:ident, $fhc:expr) => {
        mgga_driver!($name, $kind, $params, $names, $body, $fhc; scaling = &[])
    };
    ($name:expr, $kind:expr, $params:expr, $names:expr, $body:ident, $fhc:expr; scaling = $scaling:expr) => {
        Box::new($crate::xc::mgga::MggaDriver {
            param_scaling: $scaling,
            name: $name,
            kind: $kind,
            params: $params,
            param_names: $names,
            f0: $body::<f64>,
            fj: $body::<$crate::response::series::SeriesScalar>,
            fjt: $body::<$crate::dual::DualN<$crate::response::series::SeriesScalar, 7>>,
            fjv: $body::<$crate::xc::taylor::DirectionalTaylorScalar>,
            fjb: $body::<$crate::xc::bidirectional::BidirectionalTaylorScalar>,
            f1: $body::<$crate::dual::DualN<f64, 7>>,
            f2: $body::<$crate::dual::DualN<$crate::dual::DualN<f64, 7>, 7>>,
            f3: $body::<
                $crate::dual::DualN<$crate::dual::DualN<$crate::dual::DualN<f64, 7>, 7>, 7>,
            >,
            fp: $body::<$crate::dual::DualN<f64, 1>>,
            clamp_fhc: $fhc,
        })
    };
}
