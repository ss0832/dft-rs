// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! A fixed Hermitian source with nonzero transverse real/imaginary spin blocks.
pub fn source(spatial: usize) -> dft_rs::periodic::BlochMatrix {
    let mut operator = dft_rs::periodic::BlochMatrix::zeros(2 * spatial, 2 * spatial);
    for i in 0..spatial {
        operator.re[(i, i)] = -0.013 * (i + 1) as f64;
        operator.re[(spatial + i, spatial + i)] = -operator.re[(i, i)];
        operator.re[(i, spatial + i)] = 0.007 * (i + 1) as f64;
        operator.re[(spatial + i, i)] = operator.re[(i, spatial + i)];
        operator.im[(i, spatial + i)] = 0.009 * (spatial - i) as f64;
        operator.im[(spatial + i, i)] = -operator.im[(i, spatial + i)];
    }
    operator
}
