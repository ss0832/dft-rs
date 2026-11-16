// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All Cartesian nuclear derivatives of constrained open-shell references
//! across semilocal and hybrid classes. No symmetry-copy validation.
//! The bounded C1 H4+/STO-3G rerun passes all 624 gradient/Hessian fits over
//! SVWN5/PBE/TPSS/PBE0: p=1.811722327531..2.193220936186,
//! minimum R2=0.999138249778. Four failed coarse intervals are retained and
//! resolved on the existing finer interval; all five raw convergence gates
//! pass at 1e-12. This does not certify other geometries or reference types.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_reference.rs"]
mod nuclear_reference;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::{run, Controls, System, Value};

fn fit(xc: &str) {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
    );
    let controls = Controls::from_value(
        &Value::object()
            .with("reference", "roks")
            .with("xc", xc)
            .with("basis", "sto-3g")
            .with("charge", 1)
            .with("multiplicity", 2)
            .with("grid_level", 0)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("conv_tol_grad", 1e-12)
            .with("conv_tol_dp", 1e-12)
            .with("response_tolerance", 1e-12)
            .with("max_iter", 300)
            .with("guess", "core"),
    )
    .unwrap();
    nuclear_reference::fit(
        &format!("ROKS {xc} nuclear"),
        &system,
        &controls,
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
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
        ],
        &[""],
        |feature, system, controls| {
            let report = run(feature, system, controls).unwrap();
            if feature == "hessian" {
                if let Ok(directory) = std::env::var("DFT_RS_TEST_REPORT_DIRECTORY") {
                    let path = std::path::Path::new(&directory).join(format!("{xc}-hessian.json"));
                    std::fs::write(path, report.to_json()).unwrap();
                }
                assert_eq!(
                    report.get("response_converged").unwrap().as_bool(),
                    Some(true)
                );
                assert_eq!(report.get("converged").unwrap().as_bool(), Some(true));
                for key in [
                    "response_residual",
                    "flag_defect",
                    "stationarity_defect",
                    "reference_stationarity",
                    "reference_energy_error",
                ] {
                    let value = report.get(key).unwrap().as_f64().unwrap();
                    println!("ROKS {xc} {key}={value:.16e}");
                    assert!(value.is_finite() && value.abs() <= 1e-12, "{key}={value}");
                }
            }
            report
        },
    );
}
#[test]
fn lda_open_shell_full_nuclear_tensors() {
    fit("svwn5");
}
#[test]
fn gga_open_shell_full_nuclear_tensors() {
    fit("pbe");
}
#[test]
fn meta_gga_open_shell_full_nuclear_tensors() {
    fit("tpss");
}
#[test]
fn hybrid_open_shell_full_nuclear_tensors() {
    fit("pbe0");
}
