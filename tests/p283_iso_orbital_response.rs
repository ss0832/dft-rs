// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Inward iso-orbital endpoint convention: all seven point-potential entries,
//! and every water nuclear/electric tensor entry in real RKS and UKS.
//! All failed intervals are retained by the shared component/stencil helpers.
//! Measured: all 266 fits pass (14 point, 18 gradient, 162 Hessian, 18 alpha,
//! 54 beta), p=1.955170..2.013108 and minimum R2=0.99989964. Four failed
//! larger-step Hessian intervals are retained. Nuclear intervals start at
//! [0.02,0.01,0.005,0.0025] Bohr; the cancellation-limited mixed Hessian
//! uses [0.005,0.0025,0.00125,0.000625]. Electric steps are
//! [0.016,0.008,0.004,0.002] atomic units. This is a finite-grid molecular
//! subset; other functional/reference/periodic combinations are separate.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/molecular_fits.rs"]
mod molecular_fits;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{Controls, Value};
use dft_rs::xc::{resolve, FunctionalSpec, GridVars, ResolvedFunctional, XcOut};

fn evaluate(functional: &ResolvedFunctional, x: [f64; 7], order: usize) -> XcOut {
    let data = x.map(|v| [v]);
    let vars = GridVars {
        rho: [&data[0], &data[1]],
        sigma: Some([&data[2], &data[3], &data[4]]),
        tau: Some([&data[5], &data[6]]),
    };
    let mut out = XcOut::default();
    functional.eval(&vars, order, &mut out).unwrap();
    out
}

#[test]
fn every_point_potential_retains_its_independent_four_step_energy_fit() {
    for name in ["tpss", "revtpss"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let x = [0.23, 0.16, 0.08, 0.02, 0.11, 0.12, 0.09];
        let point = evaluate(&functional, x, 1);
        let exact = [
            point.vrho[0][0],
            point.vrho[1][0],
            point.vsigma[0][0],
            point.vsigma[1][0],
            point.vsigma[2][0],
            point.vtau[0][0],
            point.vtau[1][0],
        ];
        let names: Vec<_> = (0..7)
            .map(|i| format!("{name} point potential {i}"))
            .collect();
        let mut failures = Vec::new();
        components::fit_components(
            &names,
            &[[0.004, 0.002, 0.001, 0.0005]],
            |steps| {
                (0..7)
                    .map(|i| {
                        std::array::from_fn(|j| {
                            let mut plus = x;
                            let mut minus = x;
                            plus[i] += steps[j];
                            minus[i] -= steps[j];
                            let fd = (evaluate(&functional, plus, 0).e[0]
                                - evaluate(&functional, minus, 0).e[0])
                                / (2.0 * steps[j]);
                            (fd - exact[i]).abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
        components::assert_no_failures("point potentials", &failures);
    }
}

fn controls(unrestricted: bool, extra: &[(&str, Value)]) -> Controls {
    let mut request = Value::object()
        .with("xc", "tpss")
        .with("unrestricted", unrestricted)
        .with("basis", "sto-3g")
        .with("dispersion", "none")
        .with("grid_level", 3usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-12)
        .with("conv_tol_dp", 1e-12)
        .with("max_iter", 500usize);
    for (name, value) in extra {
        request.set(name, value.clone());
    }
    Controls::from_value(&request).unwrap()
}

#[test]
fn every_real_nuclear_gradient_and_hessian_component_has_an_independent_fit() {
    let mut failures = Vec::new();
    for unrestricted in [false, true] {
        molecular_fits::gradient_and_hessian(
            &format!("TPSS unrestricted={unrestricted}"),
            &|extra| controls(unrestricted, extra),
            &mut failures,
        );
    }
    components::assert_no_failures("nuclear response", &failures);
}

#[test]
fn every_real_alpha_and_beta_component_has_an_independent_fit() {
    let mut failures = Vec::new();
    for unrestricted in [false, true] {
        molecular_fits::polarizability_and_hyperpolarizability(
            &format!("TPSS unrestricted={unrestricted}"),
            &|extra| controls(unrestricted, extra),
            &mut failures,
        );
    }
    components::assert_no_failures("electric response", &failures);
}
