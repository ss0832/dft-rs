// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic complex GHF nuclear gradients and Hessians, all independent
//! components against four-step differences of separately converged states.
//! All 156 entries pass. Hessian (2,9) resolves a small quadratic coefficient
//! with a fourth-order stencil: p=3.986873, R2=0.99999742. All failed stages
//! and the resolved four-step interval remain recorded in response.md.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_reference.rs"]
mod nuclear_reference;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::{run, Controls, System, Value};

#[test]
fn every_complex_ghf_nuclear_derivative_component_passes_a_fit() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    let system = System {
        numbers: vec![1; 4],
        positions: vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
        cell: None,
        pbc: [false; 3],
    };
    let source = spinor_operator::source(4);
    let controls = Controls::from_value(
        &Value::object()
            .with("reference", "ghf")
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("conv_tol_grad", 1e-11)
            .with("conv_tol_dp", 1e-12)
            .with("max_iter", 300)
            .with("guess", "core")
            .with("response_tolerance", 1e-12)
            .with("gks_operator_real", &source.re)
            .with("gks_operator_imaginary", &source.im),
    )
    .unwrap();
    nuclear_reference::fit(
        "complex GHF nuclear",
        &system,
        &controls,
        &[
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
        ],
        &[""],
        |feature, system, controls| run(feature, system, controls).unwrap(),
    );
}
