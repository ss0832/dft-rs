// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The original ωB97X-D3 fit of Lin, Li, Mao and Chai (2013).
//! Semilocal and exact exchange coefficients are from Table 1 of
//! DOI 10.1021/ct300715s. Equations (21)–(25) define the unscaled
//! zero-damped pair correction with no three-body contribution.

use super::b97::{wb97_boxed, wb97_params};
use super::{HybridParams, XcFunctional};
use crate::dispersion::D3ZeroParams;

const SHORT_RANGE_HF: f64 = 0.195728;
const OMEGA: f64 = 0.25;

pub(crate) const HYBRID: HybridParams = HybridParams {
    cx: SHORT_RANGE_HF,
    clr: 1.0 - SHORT_RANGE_HF,
    omega: OMEGA,
};

pub(crate) const PAIR_PARAMETERS: D3ZeroParams = D3ZeroParams {
    s6: 1.0,
    s8: 1.0,
    rs6: 1.281,
    rs8: 1.094,
    alp6: 14.0,
    alp8: 16.0,
};

/// Semilocal part of the original ωB97X-D3 range-separated hybrid.
pub(crate) fn semilocal() -> Box<dyn XcFunctional> {
    // The common ωB97 driver evaluates short-range exchange and the two
    // PW92 correlation channels with the B97 polynomial enhancements.
    wb97_boxed(
        "wb97x-d3",
        wb97_params(
            OMEGA,
            &[
                1.0 - SHORT_RANGE_HF,
                0.698900,
                0.508940,
                -3.744903,
                10.060790,
            ],
            &[1.0, -4.868902, 21.295726, -36.020866, 19.177018],
            &[1.0, 2.433266, -15.446008, 17.644390, -8.879494],
        ),
    )
}
