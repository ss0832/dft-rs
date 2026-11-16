// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::dual::Scalar;

// Direct contour-sum closed form, algebraically independent of the
// production divided-difference and analytic-limit integrals (Eq. 55).
pub fn convolution<S: Scalar>(x: S, y: S, w: S, kt: S) -> S {
    let one = S::from_f64(1.0);
    let ex = (x / kt).exp();
    let ey = (y / kt).exp();
    let ew = (w / kt).exp();
    ((ex - ey * ew) / (x - y - w) - (ey - ex * ew) / (x - y + w))
        / ((one + ex) * (one + ey) * (ew - one))
}

pub fn factor<S: Scalar>(x: S, y: S, w: S, kt: S) -> S {
    let one = S::from_f64(1.0);
    let t = |x: S| {
        let e = (x / kt).exp();
        (e - one) / (e + one)
    };
    (convolution(x, y, w, kt) - convolution(x, -y, w, kt)).scale(2.0) / (t(x) * t(y))
}
