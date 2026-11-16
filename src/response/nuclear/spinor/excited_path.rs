// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic isolated-pole derivatives in the full complex spinor space.
use super::*;
use crate::properties::pt2_response::complex_projector_frames;
use crate::response::field_path::{adjoint, series_product};
use crate::response::spinor_tda::{spinor_tda_pole_taylor, SpinorTdaConfig};

pub struct SpinorExcitationPath {
    pub ground: Vec<f64>,
    pub excitation: Vec<f64>,
    pub reference_fock_defect: f64,
    pub frame_defect: f64,
    /// Largest reconstructed stationarity defect, Hartree.
    pub stationarity_defect: f64,
    /// Largest projector identity defect, dimensionless.
    pub projector_defect: f64,
    /// Largest relative linear-solve residual, dimensionless.
    pub residual: f64,
}

fn block(matrix: &Cm, first: usize, count: usize) -> Cm {
    let mut result = Cm::zeros(matrix.re.rows, count);
    for i in 0..matrix.re.rows {
        for j in 0..count {
            result.re[(i, j)] = matrix.re[(i, first + j)];
            result.im[(i, j)] = matrix.im[(i, first + j)];
        }
    }
    result
}

/// Nuclear Taylor coefficients of an isolated full-spin TDA pole through
/// order two. Coulomb and full/range-exchange derivatives include every
/// moving AO/metric/orbital endpoint. A fixed AO `operator` has zero explicit
/// nuclear derivative. Semilocal terms use the moving transverse mixed
/// kernel and its independently converged spin-angular quadrature.
#[allow(clippy::too_many_arguments)]
pub fn spinor_excitation_path(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    direction: &[[f64; 3]],
    root: usize,
    order: usize,
    tda_config: &SpinorTdaConfig,
    config: &ResponseSolverConfig,
) -> Result<SpinorExcitationPath> {
    excitation_path_with_cache(
        molecule, basis, grid, functional, reference, options, operator, direction, root, order,
        tda_config, config, None,
    )
}

#[allow(clippy::too_many_arguments)]
fn excitation_path_with_cache(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    direction: &[[f64; 3]],
    root: usize,
    order: usize,
    tda_config: &SpinorTdaConfig,
    config: &ResponseSolverConfig,
    cache: Option<&Option<Arc<semilocal::GridCache>>>,
) -> Result<SpinorExcitationPath> {
    if !(1..=2).contains(&order) {
        return Err(DftError::InvalidInput(
            "spinor excitation nuclear derivatives support orders one and two".into(),
        ));
    }
    let (mut backend, reference_state) = prepare_spinor_path(
        molecule, basis, grid, functional, reference, options, operator, direction, [0.0; 3],
        order, config,
    )?;
    // The cache is bound to this exact reference and base geometry. Only
    // already-supported lower-order contractions use it; higher mixed jets
    // retain the original direct provider and independent angular check.
    backend.cache = match cache {
        Some(value) => value.clone(),
        None => descriptor_cache(&backend, 2, options.incore_mb)?,
    };
    let (ground, state) = stationary_complex_path(&backend, &reference_state, order, config)?;
    let occupied = reference_state.occupations[0];
    let spinors = reference_state.coefficients[0].re.cols;
    let empty = spinors - occupied;
    let dim = occupied
        .checked_mul(empty)
        .ok_or_else(|| DftError::InvalidInput("spinor excitation dimension overflow".into()))?;
    let required_mb = dim
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(512 * (order + 1)))
        .ok_or_else(|| DftError::InvalidInput("spinor excitation path workspace overflow".into()))?
        .div_ceil(1024 * 1024);
    if dim == 0 || root >= dim {
        return Err(DftError::InvalidInput(
            "the selected spinor excitation root is outside the occupied/empty singles space"
                .into(),
        ));
    }
    if required_mb > tda_config.memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "spinor excitation matrix jets",
            required_mb,
            limit_mb: tda_config.memory_mb,
        });
    }
    let (frame, frame_defect) = complex_projector_frames(&state.d[0], occupied)?;
    if frame_defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: frame_defect,
        });
    }
    let orbitals = series_product(&state.b[0], &frame, order);
    let dagger = frame.iter().map(adjoint).collect::<Vec<_>>();
    let fock = series_product(&series_product(&dagger, &state.f[0], order), &frame, order);
    let co = orbitals
        .iter()
        .map(|c| block(c, 0, occupied))
        .collect::<Vec<_>>();
    let cv = orbitals
        .iter()
        .map(|c| block(c, occupied, empty))
        .collect::<Vec<_>>();
    let co_dagger = co.iter().map(adjoint).collect::<Vec<_>>();
    let cv_dagger = cv.iter().map(adjoint).collect::<Vec<_>>();
    let mut matrices = vec![BlochMatrix::zeros(dim, dim); order + 1];
    let active = match functional.family {
        crate::xc::Family::Lda => 4,
        crate::xc::Family::Gga => 16,
        crate::xc::Family::MetaGga => 32,
    };
    // Both first rules are evaluated for every column. Cache only when fewer
    // unit descriptor directions are needed than actual Hermitian queries;
    // rare finer-rule queries retain the original direct implementation.
    let mut mixed_cache = if dim.saturating_mul(2) > active {
        semilocal::KernelCache::new(
            backend.functional,
            backend.geometry.grid_points(),
            &state.p[0],
            order,
            tda_config.memory_mb.saturating_sub(required_mb),
            backend.angular_order.min(32),
        )?
    } else {
        None
    };
    for column in 0..dim {
        let mut unit = Cm::zeros(empty, occupied);
        unit.re[(column / occupied, column % occupied)] = 1.0;
        let mut units = vec![Cm::zeros(empty, occupied); order + 1];
        units[0] = unit;
        let density = series_product(&series_product(&cv, &units, order), &co_dagger, order);
        let induced =
            backend.transition_interaction(&state.p[0], &density, order, mixed_cache.as_mut())?;
        let projected = series_product(&series_product(&cv_dagger, &induced, order), &co, order);
        let b = column / occupied;
        let j = column % occupied;
        for k in 0..=order {
            for row in 0..dim {
                let a = row / occupied;
                let i = row % occupied;
                let mut re = projected[k].re[(a, i)];
                let mut im = projected[k].im[(a, i)];
                if i == j {
                    re += fock[k].re[(occupied + a, occupied + b)];
                    im += fock[k].im[(occupied + a, occupied + b)];
                }
                if a == b {
                    re -= fock[k].re[(j, i)];
                    im -= fock[k].im[(j, i)];
                }
                matrices[k].re[(row, column)] = re;
                matrices[k].im[(row, column)] = im;
            }
        }
    }
    let excitation = spinor_tda_pole_taylor(&matrices, root, tda_config)?;
    let residual = ground.residuals.iter().copied().fold(0.0_f64, f64::max);
    let projector_defect = ground
        .projector_defects
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let stationarity_defect = ground
        .stationarity_defects
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let defect = residual
        .max(projector_defect)
        .max(stationarity_defect)
        .max(ground.reference_defect);
    if defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: config.max_iterations,
            residual: defect,
        });
    }
    Ok(SpinorExcitationPath {
        ground: ground.energy,
        excitation,
        reference_fock_defect: ground.reference_defect,
        frame_defect,
        stationarity_defect,
        projector_defect,
        residual,
    })
}

/// All Cartesian entries of the analytic total excited-state Hessian.
/// Pair sums recover the mixed coefficients by polarization, with one
/// immutable isolated-pole model throughout. `root` is zero based.
#[allow(clippy::too_many_arguments)]
pub fn spinor_excited_state_hessian(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    root: usize,
    tda_config: &SpinorTdaConfig,
    config: &ResponseSolverConfig,
) -> Result<crate::response::nuclear::ExcitedStateHessian> {
    let n = molecule
        .len()
        .checked_mul(3)
        .ok_or_else(|| DftError::InvalidInput("spinor Hessian dimension overflow".into()))?;
    let required_mb = n
        .checked_mul(n)
        .and_then(|v| v.checked_mul(8 * 4))
        .ok_or_else(|| DftError::InvalidInput("spinor Hessian storage overflow".into()))?
        .div_ceil(1024 * 1024);
    if required_mb > tda_config.memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "Cartesian spinor excitation Hessian",
            required_mb,
            limit_mb: tda_config.memory_mb,
        });
    }
    let direction = |i: usize| {
        let mut d = vec![[0.0; 3]; molecule.len()];
        d[i / 3][i % 3] = 1.0;
        d
    };
    // Share the immutable base-density descriptor Hessian across every
    // Cartesian path, as in the stationary nuclear context. Geometry and
    // orbital jets remain separately rebuilt for each physical direction.
    let cache = if functional.parts.is_empty() {
        None
    } else {
        let (prepared, _) = prepare_spinor_path(
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
        descriptor_cache(&prepared, 2, options.incore_mb)?
    };
    let path = |d: &[[f64; 3]]| {
        excitation_path_with_cache(
            molecule,
            basis,
            grid,
            functional,
            reference,
            options,
            operator,
            d,
            root,
            2,
            tda_config,
            config,
            Some(&cache),
        )
    };
    let mut gradient = vec![0.0; n];
    let mut ground = Matrix::zeros(n, n);
    let mut excitation = Matrix::zeros(n, n);
    let mut diagonal = vec![[0.0; 2]; n];
    let mut residual = 0.0_f64;
    let mut reference_residual = 0.0_f64;
    let mut paths = 0;
    for i in 0..n {
        let p = path(&direction(i))?;
        gradient[i] = p.ground[1] + p.excitation[1];
        diagonal[i] = [p.ground[2], p.excitation[2]];
        ground[(i, i)] = 2.0 * p.ground[2];
        excitation[(i, i)] = 2.0 * p.excitation[2];
        residual = residual.max(p.residual);
        reference_residual = reference_residual.max(p.reference_fock_defect);
        paths += 1;
    }
    for i in 0..n {
        for j in 0..i {
            let mut d = direction(i);
            d[j / 3][j % 3] += 1.0;
            let p = path(&d)?;
            ground[(i, j)] = p.ground[2] - diagonal[i][0] - diagonal[j][0];
            ground[(j, i)] = ground[(i, j)];
            excitation[(i, j)] = p.excitation[2] - diagonal[i][1] - diagonal[j][1];
            excitation[(j, i)] = excitation[(i, j)];
            residual = residual.max(p.residual);
            reference_residual = reference_residual.max(p.reference_fock_defect);
            paths += 1;
        }
    }
    let mut hessian = ground.clone();
    for (a, b) in hessian.as_mut_slice().iter_mut().zip(excitation.as_slice()) {
        *a += b;
    }
    Ok(crate::response::nuclear::ExcitedStateHessian {
        gradient,
        hessian,
        ground_hessian: ground,
        excitation_hessian: excitation,
        reference_residual,
        residual,
        converged: true,
        paths,
        warnings: Vec::new(),
    })
}
