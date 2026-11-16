// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Normal-state thermodynamic and finite-momentum density-response
//! primitives. Spectral solvers require the physical reference and kernel
//! from their caller. The coupled field driver provides zero-field orbital
//! electric/magnetic density and current; phonon, spin and finite-field
//! magnetic reference extensions retain their separate physical kernels.

pub mod coulomb;
pub mod density;
pub mod exchange;
pub mod field;
pub mod field_dimensional;
pub mod orbitals;
pub mod spectral;
pub mod spin;
pub mod thermodynamic;
pub mod xc;
