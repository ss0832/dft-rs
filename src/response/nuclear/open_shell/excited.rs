// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! TDA of the defined time-dependent common-orbital flag model. These are
//! spin-conserving constrained rotations, not transverse or exact-S states.
use super::*;

pub struct OpenShellTdaState {
    pub energy: f64,
    /// Closed-open, closed-virtual, open-virtual rotations, with X^T M X=1.
    pub rotations: Vec<f64>,
    pub residual: f64,
    pub nearest_gap: f64,
}

pub struct OpenShellExcitations {
    pub states: Vec<OpenShellTdaState>,
    pub berry_weights: Vec<f64>,
    pub reference_stationarity: f64,
    pub metric_defect: f64,
}

/// The rotating-wave/Tamm-Dancoff operator of the existing paired flag
/// response is M^-1/2 (H_even+H_odd)/2 M^-1/2. Its closed-shell limit is the
/// spin-adapted singlet operator; common open orbitals retain Berry weights
/// (1,2,1), not the unrestricted particle-hole metric. No exact total-spin
/// assignment is made, and no transverse sector is inferred.
#[allow(clippy::too_many_arguments)]
pub fn open_shell_tda_states(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    nstates: usize,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<OpenShellExcitations> {
    config.validate()?;
    if !scf.converged
        || scf.n_alpha < scf.n_beta
        || scf.n_alpha > orbitals.cols
        || orbitals.rows != mb.nao_sph
        || !scf.has_integer_occupations()
        || scf.electronic_temperature != 0.0
        || nstates == 0
    {
        return Err(DftError::InvalidInput("common-orbital excitations require finite converged integer occupations and matching closed/open/virtual orbitals".into()));
    }
    let classes = Classes {
        closed: scf.n_beta,
        open: scf.n_alpha - scf.n_beta,
        total: orbitals.cols,
    };
    let dim = classes.dimension();
    if dim == 0 {
        return Err(DftError::InvalidInput(
            "the common-orbital flag has no interclass excitations".into(),
        ));
    }
    let required_mb = dim
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(8 * 12))
        .ok_or_else(|| {
            DftError::InvalidInput("common-orbital excitation workspace overflow".into())
        })?
        .div_ceil(1024 * 1024);
    if required_mb > memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "common-orbital TDA spectrum",
            required_mb,
            limit_mb: memory_mb,
        });
    }
    let ct = orbitals.transpose();
    let mut metric = ct.matmul_seq(&scf.overlap).matmul_seq(orbitals);
    for i in 0..metric.rows {
        metric[(i, i)] -= 1.0;
    }
    let mut metric_defect = max_abs(&metric);
    let mut projectors = [
        Matrix::zeros(orbitals.cols, orbitals.cols),
        Matrix::zeros(orbitals.cols, orbitals.cols),
    ];
    for s in 0..2 {
        for i in 0..[scf.n_alpha, scf.n_beta][s] {
            projectors[s][(i, i)] = 1.0;
        }
        let mut density = orbitals.matmul_seq(&projectors[s]).matmul_seq(&ct);
        add(&mut density, &scf.density[s], -1.0);
        metric_defect = metric_defect.max(max_abs(&density));
    }
    if metric_defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: metric_defect,
        });
    }
    let fock = std::array::from_fn(|s| ct.matmul_seq(&scf.fock[s]).matmul_seq(orbitals));
    let mut commutator_sum = Matrix::zeros(orbitals.cols, orbitals.cols);
    for s in 0..2 {
        commutator(&mut commutator_sum, &fock[s], &projectors[s]);
    }
    let reference_stationarity = classes
        .blocks(&commutator_sum)
        .iter()
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    if reference_stationarity > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: reference_stationarity,
        });
    }
    let kernel = ResponseSolver::kernel_only(mol, mb, functional, grid, scf, opts)?;
    let rotation = RotationOperator {
        kernel: &kernel,
        orbitals,
        fock: &fock,
        classes,
    };
    let nc = classes.closed;
    let no = classes.open;
    let nv = classes.virtuals();
    let berry_weights: Vec<f64> =
        [vec![1.0; nc * no], vec![2.0; nc * nv], vec![1.0; no * nv]].concat();
    let mut matrix = Matrix::zeros(dim, dim);
    for column in 0..dim {
        let mut unit = vec![0.0; dim];
        unit[column] = 1.0;
        let even = rotation.image_parity(&unit, false)?;
        let odd = rotation.image_parity(&unit, true)?;
        for row in 0..dim {
            matrix[(row, column)] =
                0.5 * (even[row] + odd[row]) / (berry_weights[row] * berry_weights[column]).sqrt();
        }
    }
    if (0..dim)
        .any(|i| (0..dim).any(|j| (matrix[(i, j)] - matrix[(j, i)]).abs() > config.acceptance()))
    {
        return Err(DftError::LinearAlgebra(
            "the common-orbital TDA operator is not symmetric".into(),
        ));
    }
    let (energies, vectors) = crate::linalg::symmetric_eigen(&matrix)?;
    let mut states = Vec::new();
    for root in 0..nstates.min(dim) {
        let rotations = (0..dim)
            .map(|i| vectors[(i, root)] / berry_weights[i].sqrt())
            .collect::<Vec<_>>();
        let even = rotation.image_parity(&rotations, false)?;
        let odd = rotation.image_parity(&rotations, true)?;
        let residual = (0..dim).fold(0.0_f64, |n, i| {
            n.hypot(
                (0.5 * (even[i] + odd[i]) - energies[root] * berry_weights[i] * rotations[i])
                    / berry_weights[i].sqrt(),
            )
        });
        if residual > config.acceptance() {
            return Err(DftError::ResponseNotConverged {
                iterations: 0,
                residual,
            });
        }
        let nearest_gap = energies
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != root)
            .map(|(_, e)| (e - energies[root]).abs())
            .fold(f64::INFINITY, f64::min);
        states.push(OpenShellTdaState {
            energy: energies[root],
            rotations,
            residual,
            nearest_gap,
        });
    }
    Ok(OpenShellExcitations {
        states,
        berry_weights,
        reference_stationarity,
        metric_defect,
    })
}
