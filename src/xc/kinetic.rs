// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit magnetic extensions of a semilocal kinetic-density functional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagneticKineticDensity {
    /// Mechanical kinetic density, including the external vector potential.
    Mechanical,
    /// `tau - |j|^2/(2 rho)` with mechanical tau and current. The vector
    /// potential cancels, giving the current-invariant orbital functional.
    CurrentInvariant,
}

impl MagneticKineticDensity {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mechanical => "mechanical",
            Self::CurrentInvariant => "current-invariant",
        }
    }
}
