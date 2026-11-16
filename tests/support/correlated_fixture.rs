// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::driver::{System, Value};

pub fn fixture(model: &str, u: bool, dispersion: &str) -> (System, Value) {
    let mut positions = vec![
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
    ];
    if u {
        positions.push([1.41, 1.83, 1.59]);
    }
    let system = System::molecule(vec![1; positions.len()], positions);
    let mut options = Value::object()
        .with("xc", model)
        .with("basis", "sto-3g")
        .with("reference", if u { "uks" } else { "rks" })
        .with("multiplicity", if u { 2usize } else { 1usize })
        .with("dispersion", dispersion)
        .with("grid_level", 0usize)
        .with("vv10_grid_level", 0usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-14)
        .with("conv_tol_dp", 1e-14)
        .with("max_iter", 500usize)
        .with("guess", "core")
        .with("response_tolerance", 1e-12)
        .with("pt2_same_spin", 0.31)
        .with("pt2_opposite_spin", 0.47)
        .with("reference_field", vec![0.003, -0.002, 0.004]);
    if model == "hf" {
        options.set("double_hybrid_target_xc", "pbe");
    }
    (system, options)
}
