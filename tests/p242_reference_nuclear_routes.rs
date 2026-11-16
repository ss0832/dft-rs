// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Canonical ROKS/ROHF gradient and Hessian routes. Independent four-step
//! fits cover all 12 gradient entries and all 144 Hessian entries, without
//! symmetry copying, on a C1 four-centre ion with no quadrature pruning.
//! All 156 entries pass after resolving retained coarse-stage failures:
//! p=1.807366..2.199687, minimum R2=0.99905866. See response.md for intervals.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_reference.rs"]
mod nuclear_reference;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::{run, Controls, System, Value};

fn system() -> System {
    System {
        numbers: vec![1; 4],
        positions: vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
        cell: None,
        pbc: [false; 3],
    }
}
fn controls(reference: bool) -> Controls {
    let mut v = Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("charge", 1)
        .with("multiplicity", 2)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_grad", 1e-12)
        .with("conv_tol_dp", 1e-12)
        .with("max_iter", 300)
        .with("guess", "core");
    if reference {
        v.set("reference", "rohf");
    }
    Controls::from_value(&v).unwrap()
}
#[test]
fn complete_roks_nuclear_derivatives_on_the_canonical_api() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    nuclear_reference::fit(
        "ROHF canonical",
        &system(),
        &controls(true),
        &[
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
            ([0.002, 0.0015, 0.001, 0.00075], stencils::Stencil::Second),
            (
                [0.001, 0.00075, 0.0005, 0.000375],
                stencils::Stencil::Second,
            ),
            (
                [0.0005, 0.000375, 0.00025, 0.0001875],
                stencils::Stencil::Second,
            ),
            ([0.002, 0.0015, 0.001, 0.00075], stencils::Stencil::Fourth),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
        ],
        &[""],
        |feature, system, controls| run(feature, system, controls).unwrap(),
    );
}
