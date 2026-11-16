// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Multiconfigurational active-space wavefunctions and multiconfiguration
//! pair-density functional theory: determinant CI in an active space,
//! state-specific and state-averaged complete-active-space SCF, translated
//! on-top functionals of the density and the on-top pair density, and the
//! linearized, compressed and extended multistate forms, and analytic
//! nuclear gradients and Hessians from nuclear Taylor paths.

pub mod casscf;
pub mod ci;
pub mod path;
pub mod pdft;
