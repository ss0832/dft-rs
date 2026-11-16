// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Restricted and genuinely open-shell unrestricted nuclear Hessians.
//! Every Cartesian gradient/Hessian component is fitted independently;
//! the shared harness retains all failed intervals without an error floor.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_reference.rs"]
mod nuclear_reference;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::{run, Controls, System, Value};

fn fit(xc: &str, unrestricted: bool) {
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
    let reference = if unrestricted { "uks" } else { "rks" };
    let controls = Controls::from_value(
        &Value::object()
            .with("reference", reference)
            .with("xc", xc)
            .with("basis", "sto-3g")
            .with("dispersion", "none")
            .with("charge", usize::from(unrestricted))
            .with("multiplicity", if unrestricted { 2usize } else { 1usize })
            .with("grid_level", 0usize)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("conv_tol_grad", 1e-12)
            .with("conv_tol_dp", 1e-12)
            .with("max_iter", 400usize)
            .with("guess", "core"),
    )
    .unwrap();
    let result = nuclear_reference::fit(
        &format!("{reference} {xc} canonical nuclear"),
        &system,
        &controls,
        &[
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
            ([0.002, 0.0015, 0.001, 0.00075], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
        ],
        &[""],
        |feature, system, controls| {
            let report = run(feature, system, controls).unwrap();
            assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
            assert_eq!(
                report.get("unrestricted").and_then(Value::as_bool),
                Some(unrestricted)
            );
            if feature == "hessian" {
                let (shape, gaps) = report
                    .get("occupied_virtual_gaps")
                    .unwrap()
                    .numeric_array()
                    .unwrap();
                assert_eq!(shape, vec![2]);
                assert!(gaps.iter().all(|gap| gap.is_finite()));
                assert_eq!(
                    report
                        .get("units")
                        .unwrap()
                        .get("occupied_virtual_gaps")
                        .and_then(Value::as_str),
                    Some("hartree")
                );
                assert_eq!(
                    report.get("provenance").and_then(Value::as_str),
                    Some("analytic")
                );
                let (_, residuals) = report
                    .get("orbital_response_residuals")
                    .unwrap()
                    .numeric_array()
                    .unwrap();
                assert_eq!(residuals.len(), 12);
                assert!(residuals.iter().all(|r| r.is_finite() && *r <= 1e-12));
                let reference_residual = report
                    .get("scf_orbital_residual")
                    .unwrap()
                    .as_f64()
                    .unwrap();
                assert!(reference_residual.is_finite() && reference_residual <= 1e-10);
                for key in ["raw_asymmetry", "translation_residual"] {
                    let residual = report.get(key).unwrap().as_f64().unwrap();
                    assert!(residual.is_finite() && residual < 1e-9, "{key}={residual}");
                }
                println!("{reference} {xc} energy={} fresh CPKS residuals={residuals:?} reference={reference_residual:.16e}", report.get("energy").unwrap().as_f64().unwrap());
                if let Ok(directory) = std::env::var("DFT_RS_TEST_REPORT_DIRECTORY") {
                    std::fs::write(
                        std::path::Path::new(&directory)
                            .join(format!("{reference}-{xc}-hessian.json")),
                        report.to_json_pretty(),
                    )
                    .unwrap();
                }
            }
            report
        },
    );
    assert_eq!(result.get("n_alpha").and_then(Value::as_i64), Some(2));
    assert_eq!(
        result.get("n_beta").and_then(Value::as_i64),
        Some(if unrestricted { 1 } else { 2 })
    );
}

macro_rules! cases {
    ($(($name:ident, $xc:expr, $unrestricted:expr)),* $(,)?) => {$ (
        #[test]
        fn $name() { fit($xc, $unrestricted); }
    )*};
}
cases! {
    (restricted_hf, "hf", false), (unrestricted_hf, "hf", true),
    (restricted_lda, "svwn5", false), (unrestricted_lda, "svwn5", true),
    (restricted_gga, "pbe", false), (unrestricted_gga, "pbe", true),
    (restricted_meta_gga, "tpss", false), (unrestricted_meta_gga, "tpss", true),
    (restricted_hybrid, "pbe0", false), (unrestricted_hybrid, "pbe0", true),
}
