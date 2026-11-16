// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Energy coefficients through order `2k + 1` from the density through
//! order `k`.
//!
//! The energy is stationary in the density under the projector condition. A
//! trial density path that is exact through order `k` and a projector at
//! every order therefore reproduces the energy through order `2k + 1`. The
//! trial path keeps the solved coefficients of the transported-frame density
//! and continues with the blocks that the projector condition alone
//! determines, without a further response solve: one first-order solve gives
//! the third derivative, two give the fifth.

use super::geometry::GeometryTaylor;
use super::{
    add, density_path, metric_transport, molecular_taylor_parts_on, MolecularDirection,
    NuclearEnergyCoefficient,
};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::response::krylov::ResponseSolverConfig;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Energy coefficients along a nuclear path from a low-order response.
#[derive(Clone, Debug)]
pub struct ReducedTaylor {
    /// `E(t) = sum_n energy[n] t^n` through `n = 2 solved + 1`, Hartree.
    pub energy: Vec<f64>,
    /// Highest order of the response equations that was solved.
    pub solved: usize,
    pub converged: bool,
    pub residuals: Vec<f64>,
    pub warnings: Vec<String>,
}

/// Energy coefficients through order `2 solved + 1` along
/// `R(t) = R(0) + t direction` with response solves through order `solved`.
#[allow(clippy::too_many_arguments)]
pub fn reduced_order_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    direction: &[[f64; 3]],
    solved: usize,
    config: &ResponseSolverConfig,
) -> Result<ReducedTaylor> {
    let total = solved
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| DftError::InvalidInput("reduced Taylor order overflow".into()))?;
    // One series through order 2 solved + 1 serves the response solve and
    // the energy evaluation.
    let reference_field = opts.external_field.map_or([0.0; 3], |f| [f.x, f.y, f.z]);
    let geometry = GeometryTaylor::new(
        mol,
        mb,
        grid,
        functional,
        opts.skala.as_ref(),
        direction,
        reference_field,
        [0.0; 3],
        total,
        opts.incore_mb,
    )?;
    let (report, path) = molecular_taylor_parts_on(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &MolecularDirection::nuclear(direction.to_vec()),
        solved,
        config,
        geometry,
    )?;
    let geometry = path.geometry;
    let n_occ = path.n_occ;
    let mut d = path.d;
    for s in 0..2 {
        let nmo = d[s][0].rows;
        for n in solved + 1..=total {
            // The projector condition alone: occupied and virtual diagonal
            // blocks from the lower orders, no occupied-virtual block.
            let mut lower = Matrix::zeros(nmo, nmo);
            for k in 1..n {
                add(&mut lower, &d[s][k].matmul_seq(&d[s][n - k]), 1.0);
            }
            let mut next = Matrix::zeros(nmo, nmo);
            for i in 0..nmo {
                for j in 0..nmo {
                    if i < n_occ[s] && j < n_occ[s] {
                        next[(i, j)] = -lower[(i, j)];
                    }
                    if i >= n_occ[s] && j >= n_occ[s] {
                        next[(i, j)] = lower[(i, j)];
                    }
                }
            }
            d[s].push(next);
        }
    }
    let p: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        density_path(
            &metric_transport(&scf.mo_coeff[s], &geometry.overlap, total),
            &d[s],
            total,
        )
    });
    let evaluation = geometry.evaluate(functional, &p, total)?;
    Ok(ReducedTaylor {
        energy: evaluation
            .energy
            .iter()
            .map(NuclearEnergyCoefficient::total)
            .collect(),
        solved,
        converged: report.converged,
        residuals: report.residuals,
        warnings: report.warnings,
    })
}

/// Cartesian third derivatives of the energy.
#[derive(Clone, Debug)]
pub struct CubicForceConstants {
    /// `d3E / (dx_i dx_j dx_k)` as `[(i n + j) n + k]` with `n = 3 atoms`,
    /// Hartree/Bohr^3.
    pub tensor: Vec<f64>,
    pub dimension: usize,
    /// Number of nuclear paths, each with one first-order response solve.
    pub paths: usize,
    pub accumulation_roundoff: f64,
    pub converged: bool,
    pub warnings: Vec<String>,
}

/// All Cartesian third derivatives from first-order responses only: the
/// third-order energy coefficient of every combined direction follows from
/// its first-order density, and the mixed derivatives from the polarization
/// identity of those coefficients.
#[allow(clippy::too_many_arguments)]
pub fn cubic_force_constants_reduced(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    config: &ResponseSolverConfig,
) -> Result<CubicForceConstants> {
    let n = 3 * mol.len();
    let mut cache: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut tensor = vec![0.0; n * n * n];
    let mut paths = 0_usize;
    let mut converged = true;
    let mut warnings: Vec<String> = Vec::new();
    let mut roundoff = 0.0_f64;
    for i in 0..n {
        for j in i..n {
            for k in j..n {
                let vectors: Vec<Vec<f64>> = [i, j, k]
                    .iter()
                    .map(|&coordinate| {
                        let mut v = vec![0.0; n];
                        v[coordinate] = 1.0;
                        v
                    })
                    .collect();
                let mixed =
                    crate::response::mixed::polarized_mixed(n, &vectors, |direction, order| {
                        if let Some((_, value)) =
                            cache.iter().find(|(key, _)| key.as_slice() == direction)
                        {
                            return Ok(vec![*value]);
                        }
                        let displacement: Vec<[f64; 3]> = direction
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        let path = reduced_order_taylor(
                            mol,
                            mb,
                            functional,
                            grid,
                            scf,
                            opts,
                            &displacement,
                            1,
                            config,
                        )?;
                        paths += 1;
                        converged &= path.converged;
                        for warning in path.warnings {
                            if !warnings.contains(&warning) {
                                warnings.push(warning);
                            }
                        }
                        let value = path.energy[order];
                        cache.push((direction.to_vec(), value));
                        Ok(vec![value])
                    })?;
                roundoff = roundoff.max(mixed.accumulation_roundoff[0]);
                for (a, b, c) in [
                    (i, j, k),
                    (i, k, j),
                    (j, i, k),
                    (j, k, i),
                    (k, i, j),
                    (k, j, i),
                ] {
                    tensor[(a * n + b) * n + c] = mixed.value[0];
                }
            }
        }
    }
    Ok(CubicForceConstants {
        tensor,
        dimension: n,
        paths,
        accumulation_roundoff: roundoff,
        converged,
        warnings,
    })
}

/// Force constants contracted with displacement patterns on their leading
/// indices.
#[derive(Clone, Debug)]
pub struct ContractedForceConstants {
    /// `sum u_i (v_j) d^nE / (dx_i (dx_j) dx_k dx_l)` over the remaining
    /// pair of Cartesian coordinates, Hartree/Bohr^n per unit pattern.
    pub matrix: Matrix,
    pub paths: usize,
    /// Fresh residuals of the solved density equations on every distinct path.
    pub response_residuals: Vec<f64>,
    pub accumulation_roundoff: f64,
    pub converged: bool,
    pub warnings: Vec<String>,
}

/// Derivatives of the energy of order `probes.len() + 2`, contracted with
/// the probes on the leading indices, for every pair of the remaining
/// Cartesian coordinates. The response is solved through first order for a
/// third derivative and through second order for a fourth one.
#[allow(clippy::too_many_arguments)]
fn contracted(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    probes: &[Vec<f64>],
    config: &ResponseSolverConfig,
) -> Result<ContractedForceConstants> {
    contracted_with(3 * mol.len(), probes, |direction, solved| {
        reduced_order_taylor(
            mol, mb, functional, grid, scf, opts, direction, solved, config,
        )
    })
}

/// [`contracted`] for any energy path over `n` Cartesian coordinates:
/// `energy_path(direction, solved)` gives the energy coefficients through
/// order `2 solved + 1` along `direction`.
pub(super) fn contracted_with(
    n: usize,
    probes: &[Vec<f64>],
    mut energy_path: impl FnMut(&[[f64; 3]], usize) -> Result<ReducedTaylor>,
) -> Result<ContractedForceConstants> {
    if probes.is_empty()
        || probes
            .iter()
            .any(|p| p.len() != n || p.iter().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "a displacement pattern holds one finite value per Cartesian coordinate".into(),
        ));
    }
    let order = probes.len() + 2;
    // The energy through order 2k + 1 needs the response through order k.
    let solved = order / 2;
    let mut cache: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut matrix = Matrix::zeros(n, n);
    let mut paths = 0_usize;
    let mut response_residuals = Vec::new();
    let mut converged = true;
    let mut warnings: Vec<String> = Vec::new();
    let mut roundoff = 0.0_f64;
    for j in 0..n {
        for k in j..n {
            let mut vectors: Vec<Vec<f64>> = probes.to_vec();
            for coordinate in [j, k] {
                let mut v = vec![0.0; n];
                v[coordinate] = 1.0;
                vectors.push(v);
            }
            let mixed =
                crate::response::mixed::polarized_mixed(n, &vectors, |direction, total| {
                    if let Some((_, value)) =
                        cache.iter().find(|(key, _)| key.as_slice() == direction)
                    {
                        return Ok(vec![*value]);
                    }
                    let displacement: Vec<[f64; 3]> = direction
                        .chunks_exact(3)
                        .map(|c| [c[0], c[1], c[2]])
                        .collect();
                    let path = energy_path(&displacement, solved)?;
                    response_residuals.extend(path.residuals);
                    paths += 1;
                    converged &= path.converged;
                    for warning in path.warnings {
                        if !warnings.contains(&warning) {
                            warnings.push(warning);
                        }
                    }
                    let value = path.energy[total];
                    cache.push((direction.to_vec(), value));
                    Ok(vec![value])
                })?;
            roundoff = roundoff.max(mixed.accumulation_roundoff[0]);
            matrix[(j, k)] = mixed.value[0];
            matrix[(k, j)] = mixed.value[0];
        }
    }
    Ok(ContractedForceConstants {
        matrix,
        paths,
        response_residuals,
        accumulation_roundoff: roundoff,
        converged,
        warnings,
    })
}

/// Third derivatives contracted with one displacement pattern,
/// `Phi(u)_jk = sum_i u_i d3E / (dx_i dx_j dx_k)`: the derivative of the
/// Hessian along `u`, from first-order responses.
#[allow(clippy::too_many_arguments)]
pub fn cubic_force_constants_along(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    probe: &[f64],
    config: &ResponseSolverConfig,
) -> Result<ContractedForceConstants> {
    contracted(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &[probe.to_vec()],
        config,
    )
}

/// Fourth derivatives contracted with two displacement patterns,
/// `Phi(u, v)_kl = sum_ij u_i v_j d4E / (dx_i dx_j dx_k dx_l)`, from
/// responses through second order.
#[allow(clippy::too_many_arguments)]
pub fn quartic_force_constants_along(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    first: &[f64],
    second: &[f64],
    config: &ResponseSolverConfig,
) -> Result<ContractedForceConstants> {
    contracted(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &[first.to_vec(), second.to_vec()],
        config,
    )
}
