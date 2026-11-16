// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared C1 all-electron and nonlocal-ECP s/p/d/f nuclear fixtures.

pub fn fixture(
    ecp: bool,
    unrestricted: bool,
    xc: &str,
) -> (dft_rs::driver::System, dft_rs::driver::Value) {
    let central = [0.17, -0.11, 0.23];
    let mut positions = vec![
        central,
        [1.77, 0.18, -0.53],
        [-0.31, 1.74, -0.45],
        [-1.20, -1.31, -0.61],
    ];
    if ecp {
        for p in &mut positions[1..] {
            for a in 0..3 {
                p[a] = central[a] + 1.4 * (p[a] - central[a]);
            }
        }
    }
    let basis =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/c1_spdf_model.g94");
    let system =
        dft_rs::driver::System::molecule(vec![if ecp { 15 } else { 7 }, 1, 1, 1], positions);
    let controls = dft_rs::driver::Value::object()
        .with("xc", xc)
        .with("basis", basis.to_str().unwrap())
        .with("reference", if unrestricted { "uks" } else { "rks" })
        .with("unrestricted", unrestricted)
        .with("charge", if unrestricted { 1i32 } else { 0i32 })
        .with("multiplicity", if unrestricted { 2usize } else { 1usize })
        .with("dispersion", "none")
        .with("grid_level", 0usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-14)
        .with("conv_tol_dp", 1e-14)
        .with("max_iter", 500usize)
        .with("guess", "core");
    (system, controls)
}
