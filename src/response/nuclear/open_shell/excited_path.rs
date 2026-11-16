// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Genuine analytic pole jets on the transported common-orbital flag.
use super::*;
use crate::periodic::BlochMatrix;
use crate::response::spinor_tda::{spinor_tda_pole_taylor, SpinorTdaConfig};

pub struct OpenShellExcitationPath {
    pub ground: Vec<f64>,
    pub excitation: Vec<f64>,
    pub frame_defect: f64,
    pub flag_defect: f64,
    pub stationarity_defect: f64,
    pub residual: f64,
}

fn transpose(path: &[Matrix]) -> Vec<Matrix> {
    path.iter().map(Matrix::transpose).collect()
}

fn flag_frame(path: &OpenShellPath, order: usize) -> (Vec<Matrix>, f64) {
    let n = path.b[0].cols;
    let mut raw = vec![Matrix::zeros(n, n); order + 1];
    for k in 0..=order {
        for i in 0..n {
            for j in 0..n {
                raw[k][(i, j)] = if j < path.counts[0] {
                    path.closed[k][(i, j)]
                } else if j < path.counts[0] + path.counts[1] {
                    path.open[k][(i, j)]
                } else {
                    (if k == 0 && i == j { 1.0 } else { 0.0 })
                        - path.closed[k][(i, j)]
                        - path.open[k][(i, j)]
                };
            }
        }
    }
    let gram = matrix_product(&transpose(&raw), &raw, order);
    let normalizer = metric_transport(&Matrix::identity(n), &gram, order);
    let frame = matrix_product(&raw, &normalizer, order);
    let mut metric = matrix_product(&transpose(&frame), &frame, order);
    for i in 0..n {
        metric[0][(i, i)] -= 1.0;
    }
    let mut defect = metric.iter().map(max_abs).fold(0.0_f64, f64::max);
    for class in 0..2 {
        let mut selector = vec![Matrix::zeros(n, n); order + 1];
        let start = if class == 0 { 0 } else { path.counts[0] };
        for i in start..start + path.counts[class] {
            selector[0][(i, i)] = 1.0;
        }
        let projected = matrix_product(
            &matrix_product(&frame, &selector, order),
            &transpose(&frame),
            order,
        );
        for k in 0..=order {
            defect = defect.max(projected[k].max_abs_diff(if class == 0 {
                &path.closed[k]
            } else {
                &path.open[k]
            }));
        }
    }
    (frame, defect)
}

/// Isolated reference-spin TDA pole through nuclear order one or two.
/// All reference rotation equations are solved through the requested order;
/// the ground-state 2n+1 shortcut cannot replace those nonvariational jets.
#[allow(clippy::too_many_arguments)]
pub fn open_shell_excitation_path(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    root: usize,
    order: usize,
    tda: &SpinorTdaConfig,
    config: &ResponseSolverConfig,
) -> Result<OpenShellExcitationPath> {
    if !(1..=2).contains(&order) {
        return Err(DftError::InvalidInput(
            "common-orbital excited nuclear paths support orders one and two".into(),
        ));
    }
    // This also checks the static reference metric, density and flag.
    open_shell_tda_states(
        mol,
        mb,
        functional,
        grid,
        scf,
        orbitals,
        opts,
        1,
        tda.memory_mb,
        config,
    )?;
    let classes = Classes {
        closed: scf.n_beta,
        open: scf.n_alpha - scf.n_beta,
        total: orbitals.cols,
    };
    let dim = classes.dimension();
    if root >= dim {
        return Err(DftError::InvalidInput(
            "common-orbital excitation root is outside the flag space".into(),
        ));
    }
    let required_mb = dim
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(128))
        .and_then(|v| {
            orbitals
                .cols
                .checked_mul(orbitals.cols)
                .and_then(|n| n.checked_mul(8 * 24))
                .and_then(|n| v.checked_add(n))
        })
        .and_then(|v| {
            orbitals
                .rows
                .checked_mul(orbitals.rows)
                .and_then(|n| n.checked_mul(8 * 16))
                .and_then(|n| v.checked_add(n))
        })
        .and_then(|v| v.checked_mul(order + 1))
        .ok_or_else(|| DftError::InvalidInput("common-orbital pole workspace overflow".into()))?
        .div_ceil(1024 * 1024);
    if required_mb > tda.memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "common-orbital excitation matrix jets",
            required_mb,
            limit_mb: tda.memory_mb,
        });
    }
    let (ground, path) = open_shell_path(
        mol, mb, functional, grid, scf, orbitals, opts, direction, order, order, config,
    )?;
    if !ground.converged {
        let defect = ground
            .flag_defects
            .iter()
            .chain(&ground.stationarity_defects)
            .copied()
            .fold(
                ground
                    .reference_stationarity
                    .max(ground.reference_energy_error.abs()),
                f64::max,
            );
        return Err(DftError::ResponseNotConverged {
            iterations: config.max_iterations,
            residual: defect,
        });
    }
    let (frame, frame_defect) = flag_frame(&path, order);
    if frame_defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: frame_defect,
        });
    }
    let physical = matrix_product(&path.b, &frame, order);
    let physical_t = transpose(&physical);
    let frame_t = transpose(&frame);
    let fock: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        matrix_product(&matrix_product(&frame_t, &path.f[s], order), &frame, order)
    });
    let berry: Vec<f64> = [
        vec![1.0; classes.closed * classes.open],
        vec![2.0; classes.closed * classes.virtuals()],
        vec![1.0; classes.open * classes.virtuals()],
    ]
    .concat();
    let mut matrices = vec![BlochMatrix::zeros(dim, dim); order + 1];
    for column in 0..dim {
        let mut unit = vec![0.0; dim];
        unit[column] = 1.0;
        let [closed, open] = classes.rotation(&unit);
        let mut alpha = closed.clone();
        add(&mut alpha, &open, 1.0);
        for odd in [false, true] {
            let mut change = [alpha.clone(), closed.clone()];
            if odd {
                for p in &mut change {
                    for i in 0..p.rows {
                        for j in i + 1..p.cols {
                            p[(i, j)] = -p[(i, j)];
                        }
                    }
                }
            }
            let q: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
                let mut constant = vec![Matrix::zeros(classes.total, classes.total); order + 1];
                constant[0] = change[s].clone();
                matrix_product(
                    &matrix_product(&physical, &constant, order),
                    &physical_t,
                    order,
                )
            });
            let response = path
                .geometry
                .evaluate_linear(functional, &path.p, &q, order)?;
            let response: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
                matrix_product(
                    &matrix_product(&physical_t, &response[s], order),
                    &physical,
                    order,
                )
            });
            for k in 0..=order {
                let mut image = Matrix::zeros(classes.total, classes.total);
                for s in 0..2 {
                    commutator(&mut image, &fock[s][k], &change[s]);
                    let occupied = if s == 0 { scf.n_alpha } else { scf.n_beta };
                    for i in 0..classes.total {
                        for j in 0..classes.total {
                            if j < occupied && i >= occupied {
                                image[(i, j)] += response[s][k][(i, j)];
                            }
                            if i < occupied && j >= occupied {
                                image[(i, j)] -= response[s][k][(i, j)];
                            }
                        }
                    }
                }
                for (row, v) in classes.blocks(&image).into_iter().enumerate() {
                    matrices[k].re[(row, column)] += 0.5 * v / (berry[row] * berry[column]).sqrt();
                }
            }
        }
    }
    let excitation = spinor_tda_pole_taylor(&matrices, root, tda)?;
    Ok(OpenShellExcitationPath {
        ground: ground.energy,
        excitation,
        frame_defect,
        flag_defect: ground.flag_defects.iter().copied().fold(0.0_f64, f64::max),
        stationarity_defect: ground
            .stationarity_defects
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        residual: ground.residuals.iter().copied().fold(0.0_f64, f64::max),
    })
}

/// All Cartesian entries of the analytic total excited-state Hessian.
/// Pair sums recover the mixed coefficients by polarization, with one
/// immutable isolated-pole model throughout. `root` is zero based.
#[allow(clippy::too_many_arguments)]
pub fn open_shell_excited_state_hessian(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &ScfResult,
    orbitals: &Matrix,
    options: &ScfOptions,
    root: usize,
    tda_config: &SpinorTdaConfig,
    config: &ResponseSolverConfig,
) -> Result<crate::response::nuclear::ExcitedStateHessian> {
    let n = molecule.len().checked_mul(3).ok_or_else(|| {
        DftError::InvalidInput("common-orbital Hessian dimension overflow".into())
    })?;
    let required_mb = n
        .checked_mul(n)
        .and_then(|v| v.checked_mul(8 * 4))
        .ok_or_else(|| DftError::InvalidInput("common-orbital Hessian storage overflow".into()))?
        .div_ceil(1024 * 1024);
    if required_mb > tda_config.memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "Cartesian common-orbital excitation Hessian",
            required_mb,
            limit_mb: tda_config.memory_mb,
        });
    }
    let direction = |i: usize| {
        let mut d = vec![[0.0; 3]; molecule.len()];
        d[i / 3][i % 3] = 1.0;
        d
    };
    let path = |d: &[[f64; 3]]| {
        open_shell_excitation_path(
            molecule,
            basis,
            functional,
            grid,
            reference,
            orbitals,
            options,
            &MolecularDirection::nuclear(d.to_vec()),
            root,
            2,
            tda_config,
            config,
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
        reference_residual = reference_residual.max(p.stationarity_defect);
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
            reference_residual = reference_residual.max(p.stationarity_defect);
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
