// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic static polarizability from electric-field CPKS, and Raman
//! activities from numerical derivatives of the ANALYTIC polarizability
//! along normal modes.

use crate::basis::{BasisSet, MolBasis};
use crate::constants::BOHR_TO_ANGSTROM;
use crate::error::Result;
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::dipole_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::ResponseSolver;
use crate::scf::{run_scf, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

/// Static dipole polarizability α (a.u., 3×3) via CPKS:
/// `α_xy = −Tr[ΔP_y D_x]` with the response to `H' = +E_y·D_y`.
pub fn polarizability(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
) -> Result<[[f64; 3]; 3]> {
    let solver = ResponseSolver::new(mol, mb, functional, grid, scf, opts)?;
    let dips = dipole_matrices(mb, Vec3::zero());
    let responses = solver.solve(&dips)?;
    let mut alpha = [[0.0; 3]; 3];
    for (y, dp) in responses.iter().enumerate() {
        let mut p_tot = dp[0].clone();
        for (t, b) in p_tot.as_mut_slice().iter_mut().zip(dp[1].as_slice()) {
            *t += *b;
        }
        for (x, dip) in dips.iter().enumerate() {
            alpha[x][y] = -p_tot.frobenius_dot(dip);
        }
    }
    Ok(alpha)
}

/// Raman activities (Å⁴/amu) per vibrational mode: central differences of
/// the analytic α along the (mass-weighted) normal modes.
#[allow(clippy::too_many_arguments)]
pub fn raman_activities(
    mol: &Molecule,
    basis_set: &BasisSet,
    scf_opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid_opts: Option<&GridOptions>,
    modes: &Matrix,
    step: f64,
) -> Result<Vec<f64>> {
    let natoms = mol.len();
    let n_modes = modes.cols;
    let alpha_at = |geom: &Molecule| -> Result<[[f64; 3]; 3]> {
        let mb = MolBasis::build(geom, basis_set)?;
        let grid = match grid_opts {
            Some(go) => Some(MolecularGrid::build(geom, go)?),
            None => None,
        };
        let scf = run_scf(geom, &mb, scf_opts, functional, grid.as_ref())?;
        polarizability(geom, &mb, functional, grid.as_ref(), &scf, scf_opts)
    };
    let mut activities = Vec::with_capacity(n_modes);
    // dα/dQ with Q in √amu·Bohr: displace along L_k/√m_k (Cartesian).
    let sqrt_m_amu: Vec<f64> = (0..natoms)
        .flat_map(|a| {
            let m = crate::constants::ATOMIC_MASS_AMU
                .get(mol.atoms[a].z as usize)
                .copied()
                .unwrap_or(2.5 * mol.atoms[a].z as f64)
                .sqrt();
            [m, m, m]
        })
        .collect();
    for k in 0..n_modes {
        let displace = |sign: f64| -> Molecule {
            let mut g = mol.clone();
            for a in 0..natoms {
                for ax in 0..3 {
                    let d = sign * step * modes[(3 * a + ax, k)] / sqrt_m_amu[3 * a + ax];
                    match ax {
                        0 => g.atoms[a].position.x += d,
                        1 => g.atoms[a].position.y += d,
                        _ => g.atoms[a].position.z += d,
                    }
                }
            }
            g
        };
        let ap = alpha_at(&displace(1.0))?;
        let am = alpha_at(&displace(-1.0))?;
        let mut da = [[0.0; 3]; 3];
        for x in 0..3 {
            for y in 0..3 {
                da[x][y] = (ap[x][y] - am[x][y]) / (2.0 * step);
            }
        }
        // a.u. → Å²/√amu: α[a0³] × a0²[Å²]... net factor a0² per (a0·√amu).
        let conv = BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM;
        for row in da.iter_mut() {
            for v in row.iter_mut() {
                *v *= conv;
            }
        }
        let mean = (da[0][0] + da[1][1] + da[2][2]) / 3.0;
        let gamma2 = 0.5
            * ((da[0][0] - da[1][1]).powi(2)
                + (da[1][1] - da[2][2]).powi(2)
                + (da[2][2] - da[0][0]).powi(2))
            + 3.0 * (da[0][1].powi(2) + da[1][2].powi(2) + da[0][2].powi(2));
        activities.push(45.0 * mean * mean + 7.0 * gamma2);
    }
    Ok(activities)
}
