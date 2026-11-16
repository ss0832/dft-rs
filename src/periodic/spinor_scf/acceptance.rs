// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete-reference acceptance, separate from SCF stationarity diagnostics.
use crate::periodic::spinor_xc::SpinorXcDiagnostics;

/// Aggregate the original SCF and provider convergence decisions.
///
/// A stationary density does not certify a converged numerical operator.
/// `None` XC diagnostics are valid for exact providers such as HF. This
/// function neither changes provider flags nor recomputes or relaxes their
/// tolerances; all raw decisions remain available to the caller.
pub fn reference_converged(
    scf_stationary: bool,
    operator_converged: bool,
    xc: Option<&SpinorXcDiagnostics>,
) -> bool {
    scf_stationary && operator_converged && xc.is_none_or(|diagnostics| diagnostics.converged)
}
