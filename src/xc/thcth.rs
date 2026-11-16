// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Shared kinetic exchange of tau-HCTH, hybrid tau-HCTH and BMK.
//! Boese-Handy, JCP116,9559 (2002), Eq.29.
//! The fitted local exchange is already weighted.
use super::common::{K_FACTOR_C, RHO_CUTOFF, X_FACTOR_C};
use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

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
    let one = S::from_f64(1.0);
    let mut energy = S::from_f64(0.0);
    for (rho, sigma, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let r43 = rho.powf(4.0 / 3.0);
        let reduced = th[0] * sigma / (r43 * r43);
        let u = reduced / (one + reduced);
        let uniform_tau = S::from_f64(K_FACTOR_C) * rho.powf(5.0 / 3.0);
        let w = (uniform_tau - tau) / (uniform_tau + tau);
        let complement = one - w * w;
        let kinetic = w * complement * complement;
        let local = th[1] + u * (th[2] + u * (th[3] + u * th[4]));
        let kinetic_coefficient = th[5] + u * (th[6] + u * (th[7] + u * th[8]));
        energy = energy - S::from_f64(X_FACTOR_C) * r43 * (local + kinetic_coefficient * kinetic);
    }
    energy
}

fn exchange(name: &'static str, local: [f64; 4], kinetic: [f64; 4]) -> Box<dyn XcFunctional> {
    let mut params = vec![0.004];
    params.extend(local);
    params.extend(kinetic);
    let names = &[
        "gamma_x", "local0", "local1", "local2", "local3", "kinetic0", "kinetic1", "kinetic2",
        "kinetic3",
    ];
    mgga_driver!(name, Kind::Exchange, params, names, exchange_energy, true)
}
pub fn thcth_x() -> Box<dyn XcFunctional> {
    exchange(
        "thcthx",
        [1.10734, -1.0534, 6.3491, -2.5531],
        [0.00110, -0.3041, 6.9543, -0.7235],
    )
}
pub fn thcth_hybrid_x() -> Box<dyn XcFunctional> {
    exchange(
        "thcthhybx",
        [0.86735, 0.3008, 1.2208, 0.1574],
        [-0.00230, -0.2849, 5.4146, -10.909],
    )
}

/// Boese-Martin kinetic exchange, JCP 121, 3405 (2004), Eqs. 2--8.
/// Published Table II gives kinetic1=4.73936; the early preprint differs.
/// The local coefficients already include their fitted semilocal weight.
/// The recipe must add the separate 0.42 fraction of full exact exchange.
pub fn bmk_x() -> Box<dyn XcFunctional> {
    exchange(
        "bmkx",
        [0.474302, 2.77701, -11.4230, 11.7167],
        [-0.192212, 4.73936, -26.6188, 22.4891],
    )
}
