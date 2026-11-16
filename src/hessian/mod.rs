// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Numerical Cartesian Hessian from central differences of the ANALYTIC
//! gradient, plus dipole derivatives (atomic polar tensor) harvested from
//! the same displaced calculations. The analytic CPKS Hessian supersedes
//! this for production sizes; this path stays as its permanent cross-check.

pub mod analytic;
pub mod fixed_density;
pub mod mgga_terms;
pub mod nlc_terms;
pub mod skala_terms;
pub mod xc_terms;
pub use analytic::{analytic_hessian, analytic_hessian_full, AnalyticHessianResult};

use crate::basis::{BasisSet, MolBasis};
use crate::dispersion::{dispersion_gradient, dispersion_hessian, Dispersion};
use crate::error::Result;
use crate::gradient::assemble_gradient;
use crate::grid::{GridOptions, MolecularGrid};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::dipole_moment;
use crate::scf::{run_scf, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use rayon::prelude::*;

pub struct HessianResult {
    /// 3N × 3N Cartesian Hessian (Hartree/Bohr²), symmetrized.
    pub hessian: Matrix,
    /// Atomic polar tensor ∂μ_ax/∂R_k (3 × 3N, a.u.).
    pub dipole_derivatives: Matrix,
}

/// Which implementation produced a Cartesian Hessian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessianMethod {
    /// Analytic CPKS electronic Hessian plus any analytic empirical-
    /// dispersion correction.
    Analytic,
    /// Central differences of the complete analytic gradient.
    Numerical,
}

impl HessianMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Analytic => "analytic",
            Self::Numerical => "numerical",
        }
    }
}

/// Evaluate the best complete Hessian available for one already-converged
/// molecular state.
///
/// The analytic path is tried first and receives the analytic D2/D3/CHG
/// correction when requested.  A typed capability refusal (RI-J, a response
/// term that is still staged, and similar cases) falls back to central
/// differences of the complete analytic gradient.  Invalid input and real
/// numerical failures are never hidden by the fallback.
#[allow(clippy::too_many_arguments)]
pub fn hessian_with_fallback(
    mol: &Molecule,
    basis_set: &BasisSet,
    mb: &MolBasis,
    scf_opts: &ScfOptions,
    functional: &ResolvedFunctional,
    functional_name: &str,
    grid_opts: Option<&GridOptions>,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    dispersion: &Dispersion,
    step: f64,
) -> Result<(HessianResult, HessianMethod)> {
    match analytic_hessian_full(mol, mb, scf_opts, functional, scf, grid) {
        Ok(full) => {
            let mut hessian = full.hessian;
            if *dispersion != Dispersion::None {
                let correction = dispersion_hessian(mol, dispersion, functional_name)?;
                for (total, delta) in hessian.as_mut_slice().iter_mut().zip(correction.as_slice()) {
                    *total += *delta;
                }
            }
            Ok((
                HessianResult {
                    hessian,
                    dipole_derivatives: full.dipole_derivatives,
                },
                HessianMethod::Analytic,
            ))
        }
        Err(error) if error.is_unsupported() => Ok((
            numerical_hessian(
                mol,
                basis_set,
                scf_opts,
                functional,
                functional_name,
                grid_opts,
                dispersion,
                step,
            )?,
            HessianMethod::Numerical,
        )),
        Err(error) => Err(error),
    }
}

/// Central-difference Hessian (and APT) with geometry-rebuilt basis/grid.
#[allow(clippy::too_many_arguments)]
pub fn numerical_hessian(
    mol: &Molecule,
    basis_set: &BasisSet,
    scf_opts: &ScfOptions,
    functional: &ResolvedFunctional,
    functional_name: &str,
    grid_opts: Option<&GridOptions>,
    dispersion: &Dispersion,
    step: f64,
) -> Result<HessianResult> {
    let natoms = mol.len();
    let dim = 3 * natoms;

    let eval = |m: &Molecule| -> Result<(Vec<Vec3>, Vec3)> {
        let mb = MolBasis::build(m, basis_set)?;
        let grid = match grid_opts {
            Some(go) => Some(MolecularGrid::build(m, go)?),
            None => None,
        };
        let scf = run_scf(m, &mb, scf_opts, functional, grid.as_ref())?;
        let mut grad = if scf_opts.density_fit.is_some() {
            crate::gradient::numerical_gradient(
                m, basis_set, scf_opts, functional, grid_opts, 1e-4,
            )?
        } else {
            assemble_gradient(m, &mb, functional, grid.as_ref(), &scf, scf_opts)?
        };
        if *dispersion != Dispersion::None {
            let gd = dispersion_gradient(m, dispersion, functional_name)?;
            for (g, d) in grad.iter_mut().zip(gd) {
                *g += d;
            }
        }
        let mu = dipole_moment(m, &mb, &scf);
        Ok((grad, mu))
    };

    let columns: Result<Vec<(Vec<f64>, Vec3)>> = (0..dim)
        .into_par_iter()
        .map(|k| {
            let (atom, ax) = (k / 3, k % 3);
            let displace = |sign: f64| -> Molecule {
                let mut m = mol.clone();
                match ax {
                    0 => m.atoms[atom].position.x += sign * step,
                    1 => m.atoms[atom].position.y += sign * step,
                    _ => m.atoms[atom].position.z += sign * step,
                }
                m
            };
            let (gp, mup) = eval(&displace(1.0))?;
            let (gm, mum) = eval(&displace(-1.0))?;
            let col: Vec<f64> = (0..dim)
                .map(|l| {
                    let (b, bx) = (l / 3, l % 3);
                    (gp[b].get(bx) - gm[b].get(bx)) / (2.0 * step)
                })
                .collect();
            let dmu = (mup - mum) / (2.0 * step);
            Ok((col, dmu))
        })
        .collect();

    let columns = columns?;
    let mut h = Matrix::zeros(dim, dim);
    let mut apt = Matrix::zeros(3, dim);
    for (k, (col, dmu)) in columns.into_iter().enumerate() {
        for l in 0..dim {
            h[(l, k)] = col[l];
        }
        apt[(0, k)] = dmu.x;
        apt[(1, k)] = dmu.y;
        apt[(2, k)] = dmu.z;
    }
    // Symmetrize.
    for i in 0..dim {
        for j in 0..i {
            let avg = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = avg;
            h[(j, i)] = avg;
        }
    }
    Ok(HessianResult {
        hessian: h,
        dipole_derivatives: apt,
    })
}
