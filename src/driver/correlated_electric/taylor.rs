// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static correlated energy Hessian from the common parent density series.
use super::{add, set_component, target_difference};
use crate::driver::method::{MolecularReference, MolecularSetup};
use crate::linalg::Matrix;
use crate::nlc::Vv10Params;
use crate::properties::pt2::Pt2Problem;
use crate::properties::pt2_response::{projector_frames, taylor_response, transform_path};
use crate::response::nonlinear::one_body_taylor;
use crate::response::nuclear::matrix_product;
use crate::response::{ResponseSolver, ResponseSolverConfig};
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub(crate) struct CorrelatedPolarizability<const N: usize> {
    pub tensor: [[f64; N]; N],
    pub parent: [[f64; N]; N],
    pub target: [[f64; N]; N],
    pub pt2: [[f64; N]; N],
    pub channels: [[[f64; N]; N]; 3],
    pub response_residuals: Vec<f64>,
    pub response_iterations: Vec<usize>,
    pub amplitude_residuals: Vec<f64>,
    pub number_residuals: Vec<f64>,
    pub projector_residuals: Vec<f64>,
    pub stationarity_residuals: Vec<f64>,
    pub frame_residuals: Vec<f64>,
}

/// Direct rotation Q (Q^T Q)^-1/2 between the reference and perturbed
/// projectors. Reuse the existing noncommuting metric-transport series;
/// rotations inside occupied/virtual degenerate subspaces are unnecessary.
fn orbital_frames(
    reference: &MolecularReference,
    density: &[[Matrix; 2]],
) -> ([Vec<Matrix>; 2], f64) {
    let scf = &reference.scf;
    let mut defect = 0.0_f64;
    let frames = std::array::from_fn(|spin| {
        let c = &scf.mo_coeff[spin];
        let sc = scf.overlap.matmul_seq(c);
        let projectors: Vec<_> = density
            .iter()
            .map(|p| sc.transpose_matmul(&p[spin].matmul_seq(&sc)))
            .collect();
        let occupied = if spin == 0 { scf.n_alpha } else { scf.n_beta };
        let (frame, residual) = projector_frames(&projectors, occupied);
        defect = defect.max(residual);
        frame
    });
    (frames, defect)
}

/// All AO-to-MO derivatives use the existing scalar-generic transformation.
fn moving_integrals<const W: usize>(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    frames: &[Vec<Matrix>; 2],
    tensor: &[f64],
    problem: &Pt2Problem<f64>,
) -> Result<Vec<[Vec<f64>; 3]>> {
    let order = frames[0].len() - 1;
    if W != order + 1 {
        return Err(DftError::InvalidInput(
            "PT2 integral scalar width differs from its orbital path".into(),
        ));
    }
    let orbitals: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        frames[s]
            .iter()
            .map(|u| reference.scf.mo_coeff[s].matmul_seq(u))
            .collect()
    });
    let mut integrals = vec![vec![0.0; tensor.len()]; W];
    integrals[0] = tensor.to_vec();
    let transformed = transform_path::<W>(
        setup.basis.nao_sph,
        &integrals,
        &orbitals,
        &problem.energies,
        problem.occupied,
    )?;
    Ok(transformed.into_iter().skip(1).map(|p| p.ovov).collect())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn polarizability<const N: usize>(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    target: Option<&ResolvedFunctional>,
    target_vv10: Option<(f64, Vv10Params)>,
    problem: &Pt2Problem<f64>,
    eri: &[f64],
    coefficients: Pt2Coefficients,
    operators: &[Matrix; N],
    config: &ResponseSolverConfig,
) -> Result<CorrelatedPolarizability<N>> {
    if setup.options.skala.is_some() {
        return Err(DftError::Unsupported {
            feature: "correlated polarizability with a learned orbital parent".into(),
            missing:
                "the nonlinear learned-potential source of the parent second-order density response"
                    .into(),
        });
    }
    let scf = &reference.scf;
    let parent_kernel = ResponseSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
    )?;
    let delta = target_difference(setup, reference, target, target_vv10)?;
    // VV10 target scaling is explicit; all common Hartree/continuum terms
    // cancel between these two density kernels.
    let mut target_options = setup.options.clone();
    target_options.vv10 = None;
    target_options.skala = None;
    let target_kernel = target
        .map(|functional| {
            ResponseSolver::kernel_only(
                &setup.molecule,
                &setup.basis,
                functional,
                setup.grid.as_ref(),
                scf,
                &target_options,
            )
        })
        .transpose()?;
    let target_nlc = target_vv10
        .map(|(scale, params)| {
            setup
                .options
                .nonlocal_grid(&setup.molecule)
                .map(|grid| (scale, params, grid))
        })
        .transpose()?;
    let mut result = CorrelatedPolarizability {
        tensor: [[0.0; N]; N],
        parent: [[0.0; N]; N],
        target: [[0.0; N]; N],
        pt2: [[0.0; N]; N],
        channels: [[[0.0; N]; N]; 3],
        response_residuals: Vec::new(),
        response_iterations: Vec::new(),
        amplitude_residuals: Vec::new(),
        number_residuals: Vec::new(),
        projector_residuals: Vec::new(),
        stationarity_residuals: Vec::new(),
        frame_residuals: Vec::new(),
    };
    let directions = (0..N)
        .map(|a| (a, a))
        .chain((0..N).flat_map(|a| (a + 1..N).map(move |b| (a, b))));
    for (a, b) in directions {
        let terms = [(&operators[a], 1.0), (&operators[b], 1.0)];
        let terms = &terms[..if a == b { 1 } else { 2 }];
        let path = one_body_taylor(&parent_kernel, terms, 2, config)?;
        let p1 = &path.density_coefficients[1];
        let parent = 0.5
            * terms
                .iter()
                .map(|(operator, weight)| {
                    weight * (p1[0].frobenius_dot(operator) + p1[1].frobenius_dot(operator))
                })
                .sum::<f64>();
        set_component(&mut result.parent, a, b, parent);
        let (frames, frame_defect) = orbital_frames(reference, &path.density_coefficients);
        let integrals = moving_integrals::<3>(setup, reference, &frames, eri, problem)?;
        let mut moving: [Vec<Matrix>; 2] = std::array::from_fn(|_| Vec::new());
        for s in 0..2 {
            let f: Vec<_> = path
                .mo_fock_coefficients
                .iter()
                .map(|f| f[s].clone())
                .collect();
            let ut: Vec<_> = frames[s].iter().map(Matrix::transpose).collect();
            moving[s] = matrix_product(&matrix_product(&ut, &f, 2), &frames[s], 2);
        }
        let fock: Vec<_> = (1..=2)
            .map(|n| [moving[0][n].clone(), moving[1][n].clone()])
            .collect();
        let pair = taylor_response(problem, &integrals, &fock, coefficients)?;
        set_component(&mut result.pt2, a, b, pair.correlation[2]);
        for (s, energy) in [
            &pair.same_spin_alpha,
            &pair.opposite_spin,
            &pair.same_spin_beta,
        ]
        .into_iter()
        .enumerate()
        {
            set_component(&mut result.channels[s], a, b, energy[2]);
        }
        if let Some(target_kernel) = &target_kernel {
            let mut change = target_kernel.fock_response(p1)?;
            let parent_change = parent_kernel.fock_response(p1)?;
            for s in 0..2 {
                add(&mut change[s], &parent_change[s], -1.0);
            }
            if let Some((scale, params, grid)) = &target_nlc {
                let mut dp = p1[0].clone();
                add(&mut dp, &p1[1], 1.0);
                let nlc = crate::nlc::vv10_fxc_contraction(
                    &setup.basis,
                    grid,
                    &scf.p_total(),
                    &dp,
                    *params,
                )?;
                for value in &mut change {
                    add(value, &nlc, *scale);
                }
            }
            let target = (0..2)
                .map(|s| {
                    delta[s].frobenius_dot(&path.density_coefficients[2][s])
                        + 0.5 * change[s].frobenius_dot(&p1[s])
                })
                .sum();
            set_component(&mut result.target, a, b, target);
        }
        result.tensor[a][b] = result.parent[a][b] + result.target[a][b] + result.pt2[a][b];
        result.tensor[b][a] = result.tensor[a][b];
        result.response_residuals.extend(path.residuals);
        result.response_iterations.extend(path.iterations);
        result.amplitude_residuals.extend(pair.residuals);
        result
            .number_residuals
            .extend(path.particle_number_derivatives.iter().flatten());
        result.projector_residuals.extend(&path.projector_defects);
        result
            .stationarity_residuals
            .extend(&path.stationarity_defects);
        result.frame_residuals.push(frame_defect);
    }
    Ok(result)
}
