// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Error type for the whole DFT pipeline.

use std::fmt;

pub type Result<T> = std::result::Result<T, DftError>;

/// Errors raised across the dft-rs pipeline.
///
/// `#[non_exhaustive]`: this crate is 0.x and pre-publication, but the enum
/// is part of the public API and naming a missing capability is something
/// this project does often, so adding a variant should not be a breaking
/// change for downstream matches.
#[derive(Debug)]
#[non_exhaustive]
pub enum DftError {
    Io(std::io::Error),
    Parse {
        line: usize,
        message: String,
    },
    InvalidInput(String),
    /// No basis set entry exists for this atomic number in the selected set.
    MissingElement {
        basis: String,
        z: u8,
    },
    /// The named basis set is not bundled and no file path was given.
    UnknownBasis(String),
    /// Functional-name resolution failed (unknown name, or an XC-only
    /// functional was specified component-wise with mismatching X and C).
    FunctionalResolution(String),
    LinearAlgebra(String),
    /// A requested calculation would exceed its configured soft workspace limit.
    ResourceLimit {
        operation: &'static str,
        required_mb: usize,
        limit_mb: usize,
    },
    /// The SCF loop hit `max_iter` without reaching the energy/density tolerance.
    ScfNotConverged {
        iterations: usize,
        error: f64,
    },
    /// A coupled-perturbed (CPKS) response solve hit its iteration limit.
    /// Returning the unconverged iterate would silently corrupt the analytic
    /// Hessian / response property, so it is an error.
    ResponseNotConverged {
        iterations: usize,
        residual: f64,
    },
    /// An analytic derivative of the requested order is not implemented for
    /// this functional (and strict mode forbids the numerical fallback).
    DerivOrderUnsupported {
        functional: String,
        requested: usize,
        max_analytic: usize,
    },
    /// A capability is not implemented for this combination of inputs.
    ///
    /// Distinct from [`Self::InvalidInput`], which means the CALLER made a
    /// mistake. This one means the caller asked for something reasonable
    /// that this program cannot yet deliver, and it is therefore the variant
    /// a fallback path may legitimately catch.
    ///
    /// The distinction is not cosmetic. Refusals of this kind used to be
    /// free-text `InvalidInput`, so callers could only tell them apart by
    /// parsing the message — which `docs/rust-api.md` explicitly tells users
    /// not to do. The concrete cost: `analytic_hessian_full` reached
    /// `pcm::require_k_is_s`, got an `InvalidInput` back, and the CLI's
    /// numerical fallback (keyed on `DerivOrderUnsupported`) did not engage,
    /// so `frequencies --solvent ief:water` aborted instead of falling back.
    Unsupported {
        /// What was asked for, e.g. "analytic Hessian with IEF-PCM".
        feature: String,
        /// What is missing that would make it possible.
        missing: String,
    },
}

impl fmt::Display for DftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse { line, message } => write!(f, "parse error at line {line}: {message}"),
            Self::InvalidInput(msg) => write!(f, "{msg}"),
            Self::MissingElement { basis, z } => {
                write!(f, "basis set `{basis}` has no entry for Z={z}")
            }
            Self::UnknownBasis(name) => write!(
                f,
                "unknown basis set `{name}` (not bundled; pass a Gaussian94 file path instead)"
            ),
            Self::FunctionalResolution(msg) => write!(f, "functional resolution error: {msg}"),
            Self::LinearAlgebra(msg) => write!(f, "linear algebra error: {msg}"),
            Self::ResourceLimit {
                operation,
                required_mb,
                limit_mb,
            } => write!(
                f,
                "{operation} requires approximately {required_mb} MiB, exceeding the configured {limit_mb} MiB limit"
            ),
            Self::ScfNotConverged { iterations, error } => write!(
                f,
                "SCF did not converge after {iterations} iterations (error={error:.3e})"
            ),
            Self::ResponseNotConverged {
                iterations,
                residual,
            } => write!(
                f,
                "CPKS response did not converge after {iterations} iterations (residual={residual:.3e}); \
                 the result would be unreliable — try a tighter SCF or a numerical derivative path"
            ),
            Self::DerivOrderUnsupported {
                functional,
                requested,
                max_analytic,
            } => write!(
                f,
                "functional `{functional}` implements analytic derivatives up to order {max_analytic}, \
                 but order {requested} was requested (enable the numerical fallback or lower the order)"
            ),
            Self::Unsupported { feature, missing } => {
                write!(f, "{feature} is not implemented: {missing}")
            }
        }
    }
}

impl DftError {
    /// Does this error mean "not implemented" rather than "you asked wrong"?
    ///
    /// The predicate a numerical-fallback path should branch on. Anything
    /// else — a malformed input, a failed SCF, a resource limit — must
    /// propagate, because retrying it by a different route would either fail
    /// the same way or quietly produce a wrong answer.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::Unsupported { .. } | Self::DerivOrderUnsupported { .. }
        )
    }
}

impl std::error::Error for DftError {}

impl From<std::io::Error> for DftError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
