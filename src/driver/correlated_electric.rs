// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Correlated molecular electric-field derivatives on a generic orbital
//! reference, including distinct-target xDH and noncanonical PT2 response.

use super::method::{MolecularReference, MolecularSetup};
use crate::dual::DualN;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::nlc::Vv10Params;
use crate::properties::pt2::{from_ao, OrbitalSpace, Pt2Problem};
use crate::properties::pt2_response::linear_response;
use crate::response::{ResponseSolver, ResponseSolverConfig};
use crate::scf::fock::JkEngine;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::xc::ResolvedFunctional;
use crate::Result;

#[derive(Default)]
pub(super) struct MomentResponse {
    pub pt2: f64,
    pub same_spin_alpha: f64,
    pub same_spin_beta: f64,
    pub opposite_spin: f64,
    pub target: f64,
    pub orbital_residual: f64,
    pub amplitude_residual: f64,
    pub number_residual: f64,
}

fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (a, b) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *a += scale * b;
    }
}

/// Functional derivative of E_target-E_parent at the reference density.
/// Common one-electron, Hartree and continuum terms cancel identically.
pub(super) fn target_difference(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    target: Option<&ResolvedFunctional>,
    target_vv10: Option<(f64, Vv10Params)>,
) -> Result<[Matrix; 2]> {
    let n = setup.basis.nao_sph;
    let mut delta = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
    let Some(target) = target else {
        return Ok(delta);
    };
    let scf = &reference.scf;
    let spin = [&scf.density[0], &scf.density[1]];
    if let Some(grid) = &setup.grid {
        let parent = crate::ks::xc_contribution(&setup.functional, &setup.basis, grid, spin)?;
        let target = crate::ks::xc_contribution(target, &setup.basis, grid, spin)?;
        for s in 0..2 {
            add(&mut delta[s], &target.vxc[s], 1.);
            add(&mut delta[s], &parent.vxc[s], -1.);
        }
    }
    let engine = JkEngine::from_options(&setup.molecule, &setup.basis, &setup.options)?;
    let k = engine.build_k_only(&setup.basis, spin)?;
    for s in 0..2 {
        add(
            &mut delta[s],
            &k[s],
            setup.functional.hybrid.cx - target.hybrid.cx,
        );
    }
    for (hybrid, sign) in [(setup.functional.hybrid, 1.), (target.hybrid, -1.)] {
        if hybrid.clr == 0. {
            continue;
        }
        let engine = JkEngine::new(
            &setup.basis,
            setup.options.eri_mode,
            CoulombKernel::Erf {
                omega: hybrid.omega,
            },
        )?;
        let k = engine.build_k_only(&setup.basis, spin)?;
        for s in 0..2 {
            add(&mut delta[s], &k[s], sign * hybrid.clr);
        }
    }
    if setup.options.vv10.is_some() || target_vv10.is_some() {
        let grid = setup.options.nonlocal_grid(&setup.molecule)?;
        let p = scf.p_total();
        for item in [setup.options.vv10.map(|p| (-1., p)), target_vv10]
            .into_iter()
            .flatten()
        {
            let (_, v) = crate::nlc::vv10_potential(&setup.basis, &grid, &p, item.1)?;
            for d in &mut delta {
                add(d, &v, item.0);
            }
        }
    }
    Ok(delta)
}

/// Corrections to the negative energy derivative for each positive
/// one-electron perturbation. The unweighted PT2 channels retain their signs.
pub(super) fn moment_response(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    target: Option<&ResolvedFunctional>,
    target_vv10: Option<(f64, Vv10Params)>,
    problem: &Pt2Problem<f64>,
    tensor: &[f64],
    coefficients: Pt2Coefficients,
    operators: &[Matrix],
    config: &ResponseSolverConfig,
) -> Result<Vec<MomentResponse>> {
    if operators.is_empty() {
        return Ok(Vec::new());
    }
    let scf = &reference.scf;
    let solver = ResponseSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
    )?;
    let (occupied, virtuals) = solver.ov_dims();
    let spins = if scf.unrestricted { 2 } else { 1 };
    let size = (0..spins).map(|s| occupied[s] * virtuals[s]).sum();
    let target_delta = target_difference(setup, reference, target, target_vv10)?;
    let dual_integrals: Vec<_> = tensor
        .iter()
        .map(|v| DualN::<f64, 1>::constant(*v))
        .collect();
    let mut result = Vec::with_capacity(operators.len());
    for operator in operators {
        let mut source = vec![0.; size];
        solver.project_ov([operator, operator], &mut source);
        let solved = solver.solve_ov_report_with_structure(
            &source,
            config,
            crate::response::krylov::OperatorStructure::Hermitian,
        )?;
        let amplitudes: Vec<_> = solved.amplitudes.iter().map(|z| z.re).collect();
        let dp = solver.delta_density(&amplitudes);
        let induced = solver.fock_response(&dp)?;
        let mut orbital_directions = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        let mut fock_directions = orbital_directions.clone();
        let mut number_residual = 0.0_f64;
        for s in 0..2 {
            let c = &scf.mo_coeff[s];
            let sc = scf.overlap.matmul_seq(c);
            let d = sc.transpose_matmul(&dp[s].matmul_seq(&sc));
            let mut rotation = Matrix::zeros(c.cols, c.cols);
            for i in 0..occupied[s] {
                for a in occupied[s]..c.cols {
                    rotation[(a, i)] = d[(a, i)];
                    rotation[(i, a)] = -d[(i, a)];
                }
            }
            let dc = c.matmul_seq(&rotation);
            let mut df = induced[s].clone();
            add(&mut df, operator, 1.);
            let mut moving = c.transpose_matmul(&df.matmul_seq(c));
            add(
                &mut moving,
                &dc.transpose_matmul(&scf.fock[s].matmul_seq(c)),
                1.,
            );
            add(
                &mut moving,
                &c.transpose_matmul(&scf.fock[s].matmul_seq(&dc)),
                1.,
            );
            orbital_directions[s] = dc;
            fock_directions[s] = moving;
            number_residual = number_residual.max(dp[s].frobenius_dot(&scf.overlap).abs());
        }
        let orbitals = std::array::from_fn(|s| OrbitalSpace {
            energies: scf.mo_energy[s]
                .iter()
                .map(|e| DualN::constant(*e))
                .collect(),
            coefficients: scf.mo_coeff[s]
                .as_slice()
                .iter()
                .zip(orbital_directions[s].as_slice())
                .map(|(c, dc)| DualN { re: *c, im: [*dc] })
                .collect(),
            occupied: occupied[s],
        });
        let derivative = from_ao(setup.basis.nao_sph, &dual_integrals, &orbitals)?;
        let ovov = derivative.ovov.map(|b| b.iter().map(|x| x.im[0]).collect());
        let response = linear_response(problem, &ovov, &fock_directions, coefficients)?;
        result.push(MomentResponse {
            pt2: -response.correlation,
            same_spin_alpha: -response.same_spin_alpha,
            same_spin_beta: -response.same_spin_beta,
            opposite_spin: -response.opposite_spin,
            target: -(0..2)
                .map(|s| target_delta[s].frobenius_dot(&dp[s]))
                .sum::<f64>(),
            orbital_residual: solved.relative_residual,
            amplitude_residual: response.residual,
            number_residual,
        });
    }
    Ok(result)
}

pub(super) mod frequency;
pub(super) mod spinor;
pub(super) mod spinor_high_order;
mod taylor;
pub(super) use taylor::{polarizability, CorrelatedPolarizability};

/// Convert a directional quadratic energy coefficient to a component of
/// the negative Hessian. Mixed directions follow the two diagonal paths.
pub(super) fn set_component<const N: usize>(
    tensor: &mut [[f64; N]; N],
    a: usize,
    b: usize,
    e2: f64,
) {
    let value = if a == b {
        -2.0 * e2
    } else {
        -(e2 + 0.5 * tensor[a][a] + 0.5 * tensor[b][b])
    };
    tensor[a][b] = value;
    tensor[b][a] = value;
}
