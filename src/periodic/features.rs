// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit feature contracts for periodic calculations.
//!
//! Representing a requested boundary/operator is useful even before its
//! kernel is complete: callers can discover support and an unsupported
//! request is rejected before a plausible but physically different 3D
//! calculation can be run.

use crate::error::{DftError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeriodicBoundary {
    pub axes: [bool; 3],
}

impl PeriodicBoundary {
    pub const FULL_3D: Self = Self { axes: [true; 3] };
    pub const ISOLATED: Self = Self { axes: [false; 3] };

    pub const fn new(axes: [bool; 3]) -> Self {
        Self { axes }
    }

    pub fn dimension(self) -> usize {
        self.axes.iter().filter(|&&axis| axis).count()
    }

    pub fn validate_k_mesh(self, mesh: [usize; 3]) -> Result<()> {
        if mesh.contains(&0) {
            return Err(DftError::InvalidInput(
                "periodic k-mesh dimensions must all be positive".to_string(),
            ));
        }
        for axis in 0..3 {
            if !self.axes[axis] && mesh[axis] != 1 {
                return Err(DftError::InvalidInput(format!(
                    "non-periodic axis {axis} requires k_mesh[{axis}] = 1"
                )));
            }
        }
        Ok(())
    }
}

impl Default for PeriodicBoundary {
    fn default() -> Self {
        Self::FULL_3D
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeriodicNonlocalKernel {
    Vv10,
    Chg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeriodicCapabilities {
    pub finite_k: bool,
    pub exact_exchange: bool,
    pub meta_gga: bool,
    pub analytic_forces: bool,
    /// Atomic-position optimization with immutable lattice vectors.
    pub fixed_cell_optimization: bool,
    pub fermi_dirac_occupations: bool,
    pub periodic_ecp_energy: bool,
    pub periodic_ecp_forces: bool,
    pub partial_boundaries: bool,
    /// Masked geometry and dimensional Coulomb source calculations.
    /// These do not imply that all electronic SCF/DFPT operators are connected.
    pub partial_geometry: bool,
    pub dimensional_basis_operators: bool,
    /// Primitive nuclear/Hartree Green integrals before Bloch contraction.
    pub dimensional_gaussian_integrals: bool,
    /// Contracted spherical nuclear/Hartree operators with supplied Bloch density.
    pub dimensional_coulomb_operators: bool,
    pub dimensional_xc_grid: bool,
    pub dimensional_xc_operators: bool,
    pub dimensional_electrostatics: bool,
    pub charged_lower_dimensional_electrostatics: bool,
    pub vv10: bool,
    pub chg: bool,
}

impl PeriodicCapabilities {
    pub const fn current() -> Self {
        Self {
            finite_k: true,
            exact_exchange: true,
            meta_gga: true,
            analytic_forces: true,
            fixed_cell_optimization: true,
            fermi_dirac_occupations: true,
            periodic_ecp_energy: true,
            periodic_ecp_forces: true,
            partial_boundaries: false,
            partial_geometry: true,
            dimensional_basis_operators: true,
            dimensional_gaussian_integrals: true,
            dimensional_coulomb_operators: true,
            dimensional_xc_grid: true,
            dimensional_xc_operators: true,
            dimensional_electrostatics: true,
            charged_lower_dimensional_electrostatics: true,
            vv10: false,
            chg: false,
        }
    }

    pub fn validate_boundary(self, boundary: PeriodicBoundary) -> Result<()> {
        if boundary.dimension() != 3 && !self.partial_boundaries {
            return Err(DftError::Unsupported {
                feature: format!("{}D periodic boundary conditions", boundary.dimension()),
                missing: "the complete electronic SCF/DFPT connection to dimensional contracted Coulomb operators and grids".into(),
            });
        }
        Ok(())
    }

    pub fn validate_nonlocal(self, kernel: PeriodicNonlocalKernel) -> Result<()> {
        let available = match kernel {
            PeriodicNonlocalKernel::Vv10 => self.vv10,
            PeriodicNonlocalKernel::Chg => self.chg,
        };
        if available {
            Ok(())
        } else {
            Err(DftError::Unsupported {
                feature: format!("periodic {kernel:?}"),
                missing: "the lattice-summed nonlocal kernel".into(),
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicEcpPlan {
    pub centers_per_cell: usize,
    pub core_electrons_per_cell: u32,
    pub max_projector_l: usize,
    pub image_cutoff: f64,
}

impl PeriodicEcpPlan {
    pub fn from_cell(cell: &crate::periodic::Cell) -> Option<Self> {
        if cell.basis.ecps.is_empty() {
            return None;
        }
        Some(Self {
            centers_per_cell: cell.basis.ecps.len(),
            core_electrons_per_cell: cell.basis.n_core_electrons,
            max_projector_l: cell
                .basis
                .ecps
                .iter()
                .map(|ecp| ecp.potential.lmax)
                .max()
                .unwrap_or(0),
            image_cutoff: cell.overlap_cutoff,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_boundary_contract_rejects_only_the_missing_physics() {
        let slab = PeriodicBoundary::new([true, true, false]);
        slab.validate_k_mesh([4, 3, 1]).unwrap();
        assert!(slab.validate_k_mesh([4, 3, 2]).is_err());
        let error = PeriodicCapabilities::current()
            .validate_boundary(slab)
            .unwrap_err();
        assert!(error.is_unsupported());
        assert!(error
            .to_string()
            .contains("2D periodic boundary conditions"));
        PeriodicCapabilities::current()
            .validate_boundary(PeriodicBoundary::FULL_3D)
            .unwrap();
    }

    #[test]
    fn nonlocal_requirements_are_typed_and_discoverable() {
        let capabilities = PeriodicCapabilities::current();
        assert!(capabilities
            .validate_nonlocal(PeriodicNonlocalKernel::Vv10)
            .unwrap_err()
            .is_unsupported());
        assert!(capabilities
            .validate_nonlocal(PeriodicNonlocalKernel::Chg)
            .unwrap_err()
            .is_unsupported());
    }
}
