// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Gaussian integral engine (McMurchie–Davidson / Hermite expansion).
//!
//! Layering:
//! - [`boys`]: the Boys function (three-regime scalar core),
//! - [`kernels`]: interchangeable interelectronic operators
//!   (1/r₁₂ · erf(ωr)/r₁₂ · erfc(ωr)/r₁₂) entering only the R-tensor base,
//! - [`hermite`]: E-coefficient tables and the flat-array R-tensor,
//! - [`one_electron`]: S, T, V, arbitrary-order multipoles,
//! - [`eri`]: shell-quartet (ab|cd), Schwarz screening, in-core store.
//!
//! Design commitments (see the project plan): derivative order is
//! generalized through the Hermite recursions (arbitrary-order center
//! derivatives arrive with the gradient phase), and the scalar kernels will
//! be made generic over the forward-AD `Scalar` type so that derivatives
//! with respect to basis-set parameters (contraction coefficients,
//! exponents) flow through the same code path.

pub mod analytic;
pub mod analytic_basis;
pub mod boys;
pub mod deriv;
pub mod deriv2;
pub mod deriv_mat;
pub mod dirac;
pub mod ecp;
pub mod eri;
pub mod hermite;
pub mod kernels;
pub mod london;
pub mod one_electron;

pub use ecp::ecp_matrix;
pub use eri::{
    eri_2c, eri_3c, eri_incore, eri_quartet, eri_quartet_masked, eri_shell_quartet, schwarz_matrix,
    EriTensor, Quartet,
};
pub use kernels::CoulombKernel;
pub use one_electron::{
    dipole_matrices, kinetic_matrix, moment_matrices, nuclear_matrix, overlap_matrix,
};
