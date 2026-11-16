// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Density-functional terms of the periodic response at fixed density
//! images, summed over the parts of a calculation: the semilocal functional,
//! evaluated by its analytic batch code, and a learned functional, evaluated
//! through its reverse pass. Callers ask for a term once and do not
//! distinguish the parts.

use crate::dfpt::xc::{FiniteQXc, LearnedXc};
use crate::error::Result;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::integrals::ImageBlock;
use crate::periodic::learned_response as learned;
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::strain::StrainSecond;
use crate::periodic::strain_fock::{merge_strain_blocks, StrainBlock};
use crate::periodic::xc::{AoDerivativeOrder, PeriodicXcGrid};
use crate::periodic::xc_hessian::{periodic_xc_hessian, PeriodicXcHessian};
use crate::periodic::xc_strain;
use crate::periodic::Cell;
use crate::xc::ResolvedFunctional;

/// Fixed density images of both spins (they must reach twice `ao_cutoff`)
/// and the cutoff of the Bloch orbitals on the grid.
#[derive(Clone, Copy)]
pub(crate) struct Density<'a> {
    pub(crate) images: [&'a [ImageBlock]; 2],
    pub(crate) ao_cutoff: f64,
}

/// Gamma-point displacement terms: second derivatives of the energy and
/// derivatives of the Kohn-Sham image blocks. `None` without any
/// density-functional part.
pub(crate) fn displacement_terms(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    density: Density<'_>,
) -> Result<Option<PeriodicXcHessian>> {
    let mut total = if functional.parts.is_empty() {
        None
    } else {
        Some(periodic_xc_hessian(
            cell,
            functional,
            density.images,
            &options.grid,
            options.grid_shell,
            density.ao_cutoff,
        )?)
    };
    if let Some(model) = &options.skala {
        let part = learned::learned_displacement_terms(
            cell,
            model,
            density.images,
            &options.grid,
            options.grid_shell,
            density.ao_cutoff,
        )?;
        total = Some(match total {
            Some(mut sum) => {
                for (target, value) in sum
                    .skeleton
                    .as_mut_slice()
                    .iter_mut()
                    .zip(part.skeleton.as_slice())
                {
                    *target += *value;
                }
                for (targets, sources) in sum.vxc_derivative.iter_mut().zip(part.vxc_derivative) {
                    for (target, source) in targets.iter_mut().zip(sources) {
                        target.extend(source);
                    }
                }
                sum
            }
            None => part,
        });
    }
    Ok(total)
}

/// Strain-strain and strain-displacement second derivatives of the energy.
pub(crate) fn strain_second(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    density: Density<'_>,
) -> Result<Option<StrainSecond>> {
    let mut total = if functional.parts.is_empty() {
        None
    } else {
        Some(
            xc_strain::xc_strain_second(
                cell,
                functional,
                density.images,
                &options.grid,
                options.grid_shell,
                density.ao_cutoff,
            )?
            .0,
        )
    };
    if let Some(model) = &options.skala {
        let part = learned::learned_strain_second(
            cell,
            model,
            density.images,
            &options.grid,
            options.grid_shell,
            density.ao_cutoff,
        )?;
        total = Some(match total {
            Some(mut sum) => {
                sum.add(&part);
                sum
            }
            None => part,
        });
    }
    Ok(total)
}

/// First-order strain matrices of the Kohn-Sham image blocks.
pub(crate) fn strain_fock(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    density: Density<'_>,
) -> Result<Option<[Vec<StrainBlock>; 2]>> {
    let mut total = if functional.parts.is_empty() {
        None
    } else {
        Some(xc_strain::xc_strain_fock(
            cell,
            functional,
            density.images,
            &options.grid,
            options.grid_shell,
            density.ao_cutoff,
        )?)
    };
    if let Some(model) = &options.skala {
        let part = learned::learned_strain_fock(
            cell,
            model,
            density.images,
            &options.grid,
            options.grid_shell,
            density.ao_cutoff,
        )?;
        total = Some(match total {
            Some(mut sum) => {
                for (target, source) in sum.iter_mut().zip(part) {
                    target.extend(source);
                }
                sum.map(merge_strain_blocks)
            }
            None => part,
        });
    }
    Ok(total)
}

/// Kohn-Sham image blocks `<m(0)| v_s |n(T)>` of both spins; empty lists
/// without any density-functional part.
pub(crate) fn potential_blocks(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    density: Density<'_>,
) -> Result<[Vec<ImageBlock>; 2]> {
    let mut total = if functional.parts.is_empty() {
        [Vec::new(), Vec::new()]
    } else {
        xc_strain::xc_potential_blocks(
            cell,
            functional,
            density.images,
            &options.grid,
            options.grid_shell,
            density.ao_cutoff,
        )?
    };
    if let Some(model) = &options.skala {
        let part = learned::learned_potential_blocks(
            cell,
            model,
            density.images,
            &options.grid,
            options.grid_shell,
            density.ao_cutoff,
        )?;
        for (target, source) in total.iter_mut().zip(part) {
            target.extend(source);
        }
    }
    Ok(total)
}

/// Change of the Kohn-Sham matrices with the density at `q = 0`, summed
/// over the parts.
pub(crate) struct ResponseKernel {
    semilocal: Option<FiniteQXc>,
    learned: Option<LearnedXc>,
}

impl ResponseKernel {
    /// The kernel of the converged reference on its k points; `None`
    /// without any density-functional part.
    pub(crate) fn new(
        cell: &Cell,
        functional: &ResolvedFunctional,
        result: &PeriodicScfResult,
        options: &PeriodicScfOptions,
        k_points: &[Vec3],
        ao_cutoff: f64,
    ) -> Result<Option<Self>> {
        if functional.parts.is_empty() && options.skala.is_none() {
            return Ok(None);
        }
        let grid = PeriodicXcGrid::build(
            cell,
            k_points,
            &options.grid,
            options.grid_shell,
            ao_cutoff,
            AoDerivativeOrder::for_scf(functional.family),
        )?;
        let semilocal = if functional.parts.is_empty() {
            None
        } else {
            Some(FiniteQXc::new(functional, &grid, &grid, &result.density)?)
        };
        let learned = match &options.skala {
            Some(model) => Some(LearnedXc::new(
                &grid.with_learned(model.clone(), cell),
                &result.density,
            )?),
            None => None,
        };
        Ok(Some(Self { semilocal, learned }))
    }

    pub(crate) fn apply(&self, density: &[Vec<BlochMatrix>; 2]) -> Result<[Vec<BlochMatrix>; 2]> {
        let mut parts = Vec::with_capacity(2);
        if let Some(kernel) = &self.semilocal {
            parts.push(kernel.apply(density)?);
        }
        if let Some(kernel) = &self.learned {
            parts.push(kernel.apply(density)?);
        }
        let mut total = parts.pop().expect("a kernel has at least one part");
        for part in parts {
            for (target, value) in total.iter_mut().flatten().zip(part.iter().flatten()) {
                target.add_scaled(value, 1.0);
            }
        }
        Ok(total)
    }
}
