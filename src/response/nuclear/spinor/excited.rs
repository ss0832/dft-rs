// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full-spin TDA poles from the same physical kernel as the molecular
//! spinor reference. Exact spin labels are deliberately absent.

use super::*;
use crate::response::spinor_tda::{spinor_tda_states_with_kernel, SpinorTdaConfig, SpinorTdaState};

pub struct SpinorExcitationReport {
    pub states: Vec<SpinorTdaState>,
    pub reference_fock_defect: f64,
    pub angular_history: Vec<[f64; 3]>,
    pub cache_bound_history: Vec<[f64; 3]>,
}

/// The complex-linear extension G[H+iA]=G[H]+iG[A] of the real energy
/// Hessian, with H and A independently Hermitian. A transition density is
/// not Hermitian and must not be symmetrized away.
fn transition_kernel(backend: &SpinorGeometry<'_>, direction: &BlochMatrix) -> Result<BlochMatrix> {
    let mut h = Cm::zeros(direction.rows(), direction.cols());
    let mut a = h.clone();
    for i in 0..direction.rows() {
        for j in 0..direction.cols() {
            h.re[(i, j)] = 0.5 * (direction.re[(i, j)] + direction.re[(j, i)]);
            h.im[(i, j)] = 0.5 * (direction.im[(i, j)] - direction.im[(j, i)]);
            a.re[(i, j)] = 0.5 * (direction.im[(i, j)] + direction.im[(j, i)]);
            a.im[(i, j)] = -0.5 * (direction.re[(i, j)] - direction.re[(j, i)]);
        }
    }
    let zero = Cm::zeros(direction.rows(), direction.cols());
    let vh = backend.linear_kernel(&[h, zero.clone()])?;
    let va = backend.linear_kernel(&[a, zero])?;
    let mut out = BlochMatrix {
        re: vh[0].re.clone(),
        im: vh[0].im.clone(),
    };
    for i in 0..out.rows() {
        for j in 0..out.cols() {
            out.re[(i, j)] -= va[0].im[(i, j)];
            out.im[(i, j)] += va[0].re[(i, j)];
        }
    }
    Ok(out)
}

/// Complete occupied/empty spinor TDA spectrum with the reference's
/// spin-covariant functional kernel. This includes transverse channels;
/// it is not the spin-diagonal approximation of a collinear response.
/// `operator` has the same fixed-AO convention as the spinor reference.
#[allow(clippy::too_many_arguments)]
pub fn spinor_excitations(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    tda_config: &SpinorTdaConfig,
    config: &ResponseSolverConfig,
) -> Result<SpinorExcitationReport> {
    let (mut backend, state) = prepare_spinor_path(
        molecule,
        basis,
        grid,
        functional,
        reference,
        options,
        operator,
        &vec![[0.0; 3]; molecule.len()],
        [0.0; 3],
        1,
        config,
    )?;
    backend.cache = descriptor_cache(&backend, 2, options.incore_mb)?;
    let zero = Cm::zeros(state.density[0].re.rows, state.density[0].re.cols);
    let actual = backend.fock_coefficient(&[vec![state.density[0].clone()], vec![zero]], 0)?;
    let reference_fock_defect = actual[0]
        .re
        .max_abs_diff(&reference.fields.fock[0].re)
        .max(actual[0].im.max_abs_diff(&reference.fields.fock[0].im));
    if reference_fock_defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: reference_fock_defect,
        });
    }
    let overlap = BlochMatrix {
        re: backend.overlap[0].re.clone(),
        im: backend.overlap[0].im.clone(),
    };
    let states = spinor_tda_states_with_kernel(
        &reference.bands[0].coefficients,
        &reference.bands[0].energies,
        state.occupations[0],
        &overlap,
        &|density| transition_kernel(&backend, density),
        tda_config,
    )?;
    Ok(SpinorExcitationReport {
        states,
        reference_fock_defect,
        angular_history: backend.angular_history.into_inner(),
        cache_bound_history: backend
            .cache
            .as_ref()
            .map_or_else(Vec::new, |c| c.angular_history.clone()),
    })
}
