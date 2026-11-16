// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact-source aggregation regressions, not molecular SCF convergence proof.
pub use dft_rs::periodic;
#[path = "../src/periodic/spinor_scf/acceptance.rs"]
mod acceptance;
use dft_rs::periodic::spinor_xc::SpinorXcDiagnostics;

#[test]
fn stationary_reference_cannot_hide_a_failed_operator() {
    assert!(acceptance::reference_converged(true, true, None));
    assert!(!acceptance::reference_converged(true, false, None));
    assert!(!acceptance::reference_converged(false, true, None));
    assert!(!acceptance::reference_converged(false, false, None));
}

#[test]
fn contradictory_xc_flags_reject_without_mutating_the_diagnostics() {
    let mut xc = SpinorXcDiagnostics {
        angular_order: 128,
        angular_energy_defect: 2e-5,
        angular_operator_defect: 1e-4,
        angular_refinement_history: vec![[128., 2e-5, 1e-4]],
        hermiticity_defect: 0.0,
        component_energy_defect: 0.0,
        converged: false,
    };
    assert!(!acceptance::reference_converged(true, true, Some(&xc)));
    assert!(!xc.converged);
    assert_eq!(xc.angular_operator_defect, 1e-4);
    xc.angular_energy_defect = 0.0;
    xc.angular_operator_defect = 0.0;
    xc.converged = true;
    assert!(acceptance::reference_converged(true, true, Some(&xc)));
    assert!(!acceptance::reference_converged(true, false, Some(&xc)));
    assert!(!acceptance::reference_converged(false, true, Some(&xc)));
}
