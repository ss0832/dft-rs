// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared component-fit helpers for in-crate derivative acceptance.
#[path = "../tests/support/components.rs"]
pub(crate) mod components;
#[path = "../tests/support/convergence.rs"]
mod convergence;
#[path = "../tests/support/molecular_fits.rs"]
pub(crate) mod molecular_fits;
#[path = "../tests/support/stencils.rs"]
pub(crate) mod stencils;
#[path = "../tests/support/symmetric_cartesian_directions.rs"]
pub(crate) mod symmetric_cartesian_directions;

#[path = "../tests/support/c1_geometry.rs"]
pub(crate) mod c1_geometry;
