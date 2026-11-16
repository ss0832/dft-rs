// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
#![forbid(unsafe_code)]
// `!(x > 0.0)` is used on purpose throughout input validation: unlike
// `x <= 0.0` it also rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

//! # dft-rs
//!
//! Pure-Rust Gaussian-basis Kohn–Sham DFT (and Hartree–Fock): energies,
//! analytic nuclear gradients, Hessians (numerical, then analytic CPKS),
//! vibrational/IR/Raman analysis, molecular orbitals and properties.
//!
//! Design constraints:
//! - Dependencies: `libm`, `rayon`, `faer` only (optional `pyo3` behind the
//!   `python` feature). No libxc, no libint, no BLAS/LAPACK, no FFI.
//! - `#![forbid(unsafe_code)]`: the crate contains no unsafe Rust.
//! - Exchange–correlation functionals are implemented from the primary
//!   literature; external packages (PySCF, ...) are used only as *numerical
//!   oracles* to verify agreement of energies and derivatives.
//!
//! Internal unit policy: **Hartree** energies and **Bohr** distances (atomic
//! units) throughout the computational core. XYZ input/output is Ångström.

#[cfg(test)]
mod acceptance;

// The command-line frontend uses `dft_rs::` paths so that the same source
// compiles as a library module (shared by the executables and the Python
// `cli_main`) and inside the CLI contract tests.
extern crate self as dft_rs;

pub mod basis;
/// The `dft-rs` command line as a library entry point:
/// [`cli::run_arguments`] takes the arguments without the program name and
/// returns the exit status.
#[path = "bin/dft_rs.rs"]
pub mod cli;
pub mod constants;
pub mod dfpt;
pub mod dispersion;
pub mod driver;
pub mod dual;
pub mod error;
pub mod geomech;
pub mod gradient;
pub mod grid;
pub mod hessian;
pub mod indices;
pub mod integrals;
pub mod ks;
pub mod linalg;
pub mod logger;
pub mod math;
pub mod mcscf;
pub mod multicomponent;
pub mod nlc;
pub mod optimizer;
pub mod pcm;
pub mod periodic;
pub mod properties;
pub mod pseudo_sub;
pub mod quantities;
pub mod resources;
pub mod response;
pub mod saddle;
pub mod sapt;
pub mod scf;
pub mod skala;
pub mod system;
pub mod xc;

#[cfg(feature = "python")]
pub mod python;

pub use error::{DftError, Result};
pub use linalg::{ensure_parallelism, symmetric_eigen, Matrix, Orthogonalizer};
pub use math::{Mat3, Vec3};
pub use scf::{run_hf, run_scf, Reference, ScfOptions, ScfResult};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule};
pub use xc::{resolve, FunctionalSpec, ResolvedFunctional};
