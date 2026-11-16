// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Variational 2n+1 beta from three first-order field responses. The chart
//! D(lambda)=exp(K(lambda)) D0 exp(-K(lambda)) preserves the density spectrum, so its
//! cubic energy derivative needs no second-order orbital response.

use super::{complete_tensor, StaticElectricTensor};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::dipole_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::{ResponseReport, ResponseSolver, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::Result;

fn add(a: &mut Matrix, b: &Matrix, weight: f64) {
    for (a, b) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *a += weight * b;
    }
}
fn commutator(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = a.matmul_seq(b);
    add(&mut out, &b.matmul_seq(a), -1.0);
    out
}
fn mo(c: &Matrix, f: &Matrix) -> Matrix {
    c.transpose().matmul_seq(f).matmul_seq(c)
}

#[allow(clippy::too_many_arguments)]
pub fn static_beta_variational(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<StaticElectricTensor>> {
    config.validate()?;
    let solver = if scf.fixed_occupation_residual.is_some() {
        ResponseSolver::kernel_only(mol, mb, functional, grid, scf, opts)?
    } else {
        ResponseSolver::new(mol, mb, functional, grid, scf, opts)?
    };
    let dipoles = dipole_matrices(mb, Vec3::zero());
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    let mut projector_defect = 0.0_f64;
    let mut stationarity_defect = 0.0_f64;
    let mut particle_number_defect = 0.0_f64;
    let mut fixed_population_defect: Option<f64> = None;
    let mut densities = Vec::new();
    let mut generators = Vec::<[Matrix; 2]>::new();
    let mut first_focks = Vec::<[Matrix; 2]>::new();
    let d0: [Matrix; 2] = std::array::from_fn(|s| {
        let mut d = Matrix::zeros(scf.mo_coeff[s].cols, scf.mo_coeff[s].cols);
        for (i, &f) in scf.occupations[s].iter().enumerate() {
            d[(i, i)] = f;
        }
        d
    });
    let f0: [Matrix; 2] = std::array::from_fn(|s| mo(&scf.mo_coeff[s], &scf.fock[s]));
    for axis in 0..3 {
        let mut direction = [0.0; 3];
        direction[axis] = 1.0;
        let report = crate::response::electric_field_taylor(
            mol, mb, functional, grid, scf, opts, direction, 1, config,
        )?;
        converged &= report.converged;
        residuals.extend(report.residuals);
        iterations.extend(report.iterations);
        warnings.extend(report.warnings);
        if let Some(d) = &report.value.fixed_population {
            fixed_population_defect = Some(
                fixed_population_defect
                    .unwrap_or(0.0)
                    .max(d.maximum_defect()),
            );
        }
        projector_defect = projector_defect.max(report.value.projector_defects[0]);
        stationarity_defect = stationarity_defect.max(report.value.stationarity_defects[0]);
        for &v in &report.value.particle_number_derivatives[0] {
            particle_number_defect = particle_number_defect.max(v.abs());
        }
        let dp = report.value.density_coefficients[1].clone();
        first_focks.push(report.value.mo_fock_coefficients[1].clone());
        generators.push(std::array::from_fn(|s| {
            let sc = scf.overlap.matmul_seq(&scf.mo_coeff[s]);
            let da = sc.transpose().matmul_seq(&dp[s]).matmul_seq(&sc);
            let mut k = Matrix::zeros(da.rows, da.cols);
            if scf.fixed_occupation_residual.is_some() {
                for i in 0..da.rows {
                    for j in i + 1..da.cols {
                        if scf.occupations[s][i] != scf.occupations[s][j] {
                            let value = -first_focks[axis][s][(i, j)]
                                / (scf.mo_energy[s][i] - scf.mo_energy[s][j]);
                            k[(i, j)] = value;
                            k[(j, i)] = -value;
                        }
                    }
                }
                return k;
            }
            let no = [scf.n_alpha, scf.n_beta][s];
            for a in no..da.rows {
                for i in 0..no {
                    k[(a, i)] = da[(a, i)];
                    k[(i, a)] = -da[(i, a)];
                }
            }
            k
        }));
        densities.push(dp);
    }
    let mut tensor = vec![0.0; 27];
    let mut roundoff = vec![0.0; 27];
    let mut directional_evaluations = 3;
    for b in 0..3 {
        for c in 0..3 {
            let mut db = vec![0.0; 3];
            db[b] = 1.0;
            let mut dc = vec![0.0; 3];
            dc[c] = 1.0;
            let nonlinear =
                crate::response::mixed::polarized_mixed(3, &[db, dc], |direction, _| {
                    let mut dp = [
                        Matrix::zeros(mb.nao_sph, mb.nao_sph),
                        Matrix::zeros(mb.nao_sph, mb.nao_sph),
                    ];
                    for a in 0..3 {
                        for s in 0..2 {
                            add(&mut dp[s], &densities[a][s], direction[a]);
                        }
                    }
                    let zero = [
                        Matrix::zeros(mb.nao_sph, mb.nao_sph),
                        Matrix::zeros(mb.nao_sph, mb.nao_sph),
                    ];
                    let f2 = crate::response::nonlinear::field_fock_coefficient(
                        &solver,
                        &[scf.density.clone(), dp.clone(), zero],
                        &dipoles,
                        [0.0; 3],
                        2,
                    )?;
                    Ok(f2
                        .iter()
                        .flat_map(|f| f.as_slice().iter().copied())
                        .collect())
                })?;
            directional_evaluations += nonlinear.directional_evaluations;
            let width = mb.nao_sph * mb.nao_sph;
            let nonlinear_fock: [Matrix; 2] = std::array::from_fn(|s| {
                Matrix::from_row_major(
                    mb.nao_sph,
                    mb.nao_sph,
                    nonlinear.value[s * width..(s + 1) * width].to_vec(),
                )
            });
            for a in 0..3 {
                let flat = (a * 3 + b) * 3 + c;
                let mut third = 0.0;
                let mut magnitude = 0.0;
                for s in 0..2 {
                    let pair = |i: usize, j: usize| {
                        let mut out =
                            commutator(&generators[i][s], &commutator(&generators[j][s], &d0[s]));
                        add(
                            &mut out,
                            &commutator(&generators[j][s], &commutator(&generators[i][s], &d0[s])),
                            1.0,
                        );
                        for v in out.as_mut_slice() {
                            *v *= 0.5;
                        }
                        out
                    };
                    let mut dabc = Matrix::zeros(d0[s].rows, d0[s].cols);
                    for [i, j, k] in [
                        [a, b, c],
                        [a, c, b],
                        [b, a, c],
                        [b, c, a],
                        [c, a, b],
                        [c, b, a],
                    ] {
                        add(
                            &mut dabc,
                            &commutator(
                                &generators[i][s],
                                &commutator(
                                    &generators[j][s],
                                    &commutator(&generators[k][s], &d0[s]),
                                ),
                            ),
                            1.0 / 6.0,
                        );
                    }
                    for term in [
                        first_focks[a][s].frobenius_dot(&pair(b, c)),
                        first_focks[b][s].frobenius_dot(&pair(a, c)),
                        first_focks[c][s].frobenius_dot(&pair(a, b)),
                        f0[s].frobenius_dot(&dabc),
                        nonlinear_fock[s].frobenius_dot(&densities[a][s]),
                    ] {
                        third += term;
                        magnitude += term.abs();
                    }
                    roundoff[flat] += nonlinear.accumulation_roundoff[s * width..(s + 1) * width]
                        .iter()
                        .zip(densities[a][s].as_slice())
                        .map(|(error, d)| error * d.abs())
                        .sum::<f64>();
                }
                tensor[flat] = -third;
                roundoff[flat] += 64.0 * f64::EPSILON * magnitude;
            }
        }
    }
    complete_tensor(
        2,
        tensor,
        roundoff,
        9,
        directional_evaluations,
        projector_defect,
        stationarity_defect,
        particle_number_defect,
        fixed_population_defect,
        residuals,
        iterations,
        warnings,
        converged,
        functional,
        scf,
        config,
        if scf.fixed_occupation_residual.is_some() {
            "static clamped-nuclei prescribed-population KS/HF variational 2n+1 response; three first-order isospectral solves; analytic nonlinear functional kernel; raw slot contractions"
        } else {
            "static clamped-nuclei integer-occupation KS/HF variational 2n+1 \
            response; three first-order field solves; analytic nonlinear functional \
            kernel; raw slot contractions"
        },
    )
}
