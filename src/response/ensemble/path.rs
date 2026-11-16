// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Isospectral orbital response in a moving orthonormal frame.
//!
//! U(t)=exp(K(t)), K(t)^T=-K(t), K(0)=0, and D(t)=U(t) f U(t)^T.
//! At order n the new K_n enters D_n only as [K_n,f]. The transformed
//! Fock equation is F_n(source)+[epsilon,K_n]+G[[K_n,f]]=0 on pairs of
//! unequal population. Its full-MO spectral solve is the same one used
//! for first-order ensemble response. Equal-population rotations are a
//! gauge. No occupation difference is inverted to recover K_n.

use super::{EnsembleSolver, FixedPopulationDiagnostics, PopulationLaw};
use crate::linalg::Matrix;
use crate::response::nuclear::{add, matrix_product};
use crate::response::ResponseSolverConfig;
use crate::{DftError, Result};

pub(crate) struct FixedPopulationPath {
    pub density: Vec<[Matrix; 2]>,
    pub density_in_frame: [Vec<Matrix>; 2],
    pub mo_fock: Vec<[Matrix; 2]>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub particle_numbers: Vec<[f64; 2]>,
    pub diagnostics: FixedPopulationDiagnostics,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
}

fn norm(m: &Matrix) -> f64 {
    m.as_slice().iter().fold(0.0_f64, |a, x| {
        if x.is_finite() {
            a.max(x.abs())
        } else {
            f64::INFINITY
        }
    })
}

/// Exact truncated exponential: K has zero constant term, so powers above
/// the requested Taylor order vanish. No exponential tail is approximated.
fn exponential(k: &[Matrix], order: usize) -> Vec<Matrix> {
    let mut unit = vec![Matrix::zeros(k[0].rows, k[0].cols); order + 1];
    unit[0] = Matrix::identity(k[0].rows);
    let mut power = unit.clone();
    let mut out = unit;
    let mut inverse_factorial = 1.0;
    for m in 1..=order {
        power = matrix_product(&power, k, order);
        inverse_factorial /= m as f64;
        for j in m..=order {
            add(&mut out[j], &power[j], inverse_factorial);
        }
    }
    out
}

fn density(
    rotations: &[Vec<Matrix>; 2],
    b: &[Vec<Matrix>; 2],
    populations: &[Vec<f64>; 2],
    order: usize,
) -> [Vec<Matrix>; 2] {
    std::array::from_fn(|s| {
        let orbitals = matrix_product(&b[s], &rotations[s], order);
        let transpose: Vec<_> = orbitals.iter().map(Matrix::transpose).collect();
        let weighted: Vec<_> = orbitals
            .into_iter()
            .map(|mut c| {
                for i in 0..c.rows {
                    for j in 0..c.cols {
                        c[(i, j)] *= populations[s][j];
                    }
                }
                c
            })
            .collect();
        matrix_product(&weighted, &transpose, order)
    })
}

/// All AO Fock coefficients are supplied by the same complete energy model.
/// `b(t)` transports the reference MOs, with b(0)=C and b^T S b=I.
pub(crate) fn fixed_population_path(
    solver: &EnsembleSolver<'_>,
    b: &[Vec<Matrix>; 2],
    overlap: &[Matrix],
    order: usize,
    config: &ResponseSolverConfig,
    coefficient: impl Fn(&[[Matrix; 2]], usize) -> Result<[Matrix; 2]>,
) -> Result<FixedPopulationPath> {
    config.validate()?;
    if !matches!(solver.population_law, PopulationLaw::Fixed)
        || order == usize::MAX
        || overlap.len() <= order
        || b.iter().any(|v| v.len() <= order)
    {
        return Err(DftError::InvalidInput(
            "isospectral response requires fixed populations and complete metric/transport series"
                .into(),
        ));
    }
    let scf = solver.kernel.scf;
    let nao = scf.overlap.rows;
    if overlap
        .iter()
        .any(|m| m.rows != nao || m.cols != nao || m.as_slice().iter().any(|v| !v.is_finite()))
        || (0..2).any(|s| {
            b[s].iter().any(|m| {
                m.rows != nao
                    || m.cols != solver.nmo[s]
                    || m.as_slice().iter().any(|v| !v.is_finite())
            })
        })
    {
        return Err(DftError::InvalidInput(
            "isospectral AO/metric/transport series must be finite with matching dimensions".into(),
        ));
    }
    let bt: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| b[s].iter().map(Matrix::transpose).collect());
    let mut generators: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| vec![Matrix::zeros(solver.nmo[s], solver.nmo[s]); order + 1]);
    let mut rotations: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| exponential(&generators[s], order));
    let initial = density(&rotations, b, &scf.occupations, 0);
    let mut p = vec![[initial[0][0].clone(), initial[1][0].clone()]];
    let mut fock = vec![coefficient(&p, 0)?];
    let mut residuals = Vec::with_capacity(order);
    let mut iterations = Vec::with_capacity(order);
    let offsets = solver.offsets();
    for n in 1..=order {
        rotations = std::array::from_fn(|s| exponential(&generators[s], n));
        let trial = density(&rotations, b, &scf.occupations, n);
        p.push([trial[0][n].clone(), trial[1][n].clone()]);
        fock.push(coefficient(&p, n)?);
        let mut potential = vec![0.0; offsets[2]];
        for s in 0..2 {
            let ao_fock: Vec<_> = fock.iter().map(|a| a[s].clone()).collect();
            let mo = matrix_product(&matrix_product(&bt[s], &ao_fock, n), &b[s], n);
            let ut: Vec<_> = rotations[s].iter().map(Matrix::transpose).collect();
            let rotated = matrix_product(&matrix_product(&ut, &mo, n), &rotations[s], n);
            potential[offsets[s]..offsets[s + 1]].copy_from_slice(rotated[n].as_slice());
        }
        let (solution, _, hamiltonian) = solver.solve_mo(&potential, [0.0; 2], config)?;
        for s in 0..2 {
            let nmo = solver.nmo[s];
            for i in 0..nmo {
                for j in i + 1..nmo {
                    if scf.occupations[s][i] != scf.occupations[s][j] {
                        let gap = scf.mo_energy[s][i] - scf.mo_energy[s][j];
                        let k = -hamiltonian[offsets[s] + i * nmo + j] / gap;
                        generators[s][n][(i, j)] = k;
                        generators[s][n][(j, i)] = -k;
                    }
                }
            }
        }
        rotations = std::array::from_fn(|s| exponential(&generators[s], n));
        let actual = density(&rotations, b, &scf.occupations, n);
        p[n] = [actual[0][n].clone(), actual[1][n].clone()];
        // Reevaluate the actual complete coefficient, rather than measuring
        // stationarity on the linearized source or corrected residual vector.
        fock[n] = coefficient(&p, n)?;
        residuals.push(solution.relative_residual);
        iterations.push(solution.iterations);
    }
    let mo_fock: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        let af: Vec<_> = fock.iter().map(|f| f[s].clone()).collect();
        matrix_product(&matrix_product(&bt[s], &af, order), &b[s], order)
    });
    let mo_density: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        let sp = matrix_product(
            overlap,
            &p.iter().map(|a| a[s].clone()).collect::<Vec<_>>(),
            order,
        );
        matrix_product(
            &matrix_product(&matrix_product(&bt[s], &sp, order), overlap, order),
            &b[s],
            order,
        )
    });
    let mut populations = vec![0.0_f64; order + 1];
    let mut unitary = vec![0.0_f64; order + 1];
    let mut raw_projector = vec![0.0_f64; order];
    let mut commutators = vec![0.0_f64; order];
    let mut relative_commutators = vec![0.0_f64; order];
    let mut numbers = vec![[0.0_f64; 2]; order];
    for s in 0..2 {
        let c = matrix_product(&b[s], &rotations[s], order);
        let ct: Vec<_> = c.iter().map(Matrix::transpose).collect();
        let sc = matrix_product(overlap, &c, order);
        let sct: Vec<_> = sc.iter().map(Matrix::transpose).collect();
        let ao: Vec<_> = p.iter().map(|a| a[s].clone()).collect();
        let mut pop = matrix_product(&matrix_product(&sct, &ao, order), &sc, order);
        let mut metric = matrix_product(&ct, &sc, order);
        for i in 0..solver.nmo[s] {
            pop[0][(i, i)] -= scf.occupations[s][i];
            metric[0][(i, i)] -= 1.0;
        }
        for n in 0..=order {
            populations[n] = populations[n].max(norm(&pop[n]));
            unitary[n] = unitary[n].max(norm(&metric[n]));
        }
        let square = matrix_product(&mo_density[s], &mo_density[s], order);
        let fd = matrix_product(&mo_fock[s], &mo_density[s], order);
        let df = matrix_product(&mo_density[s], &mo_fock[s], order);
        for n in 1..=order {
            let mut idempotency = square[n].clone();
            add(&mut idempotency, &mo_density[s][n], -1.0);
            raw_projector[n - 1] = raw_projector[n - 1].max(norm(&idempotency));
            let mut comm = fd[n].clone();
            add(&mut comm, &df[n], -1.0);
            let raw = norm(&comm);
            let scale = (0..=n)
                .map(|k| {
                    solver.nmo[s] as f64
                        * (norm(&mo_fock[s][k]) * norm(&mo_density[s][n - k])
                            + norm(&mo_density[s][k]) * norm(&mo_fock[s][n - k]))
                })
                .sum::<f64>();
            commutators[n - 1] = commutators[n - 1].max(raw);
            relative_commutators[n - 1] =
                relative_commutators[n - 1].max(if scale == 0.0 { raw } else { raw / scale });
            numbers[n - 1][s] = (0..=n)
                .map(|k| p[k][s].frobenius_dot(&overlap[n - k]))
                .sum();
        }
    }
    let reference_density_defect = (0..2)
        .map(|s| p[0][s].max_abs_diff(&scf.density[s]))
        .fold(0.0_f64, f64::max);
    if p.iter()
        .flatten()
        .chain(mo_fock.iter().flatten())
        .flat_map(Matrix::as_slice)
        .any(|v| !v.is_finite())
        || populations
            .iter()
            .chain(&unitary)
            .chain(&raw_projector)
            .chain(&commutators)
            .chain(&relative_commutators)
            .chain(numbers.iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err(DftError::LinearAlgebra(
            "non-finite fixed-population orbital path".into(),
        ));
    }
    Ok(FixedPopulationPath {
        density: p,
        density_in_frame: mo_density,
        mo_fock: (0..=order)
            .map(|n| [mo_fock[0][n].clone(), mo_fock[1][n].clone()])
            .collect(),
        projector_defects: raw_projector,
        stationarity_defects: commutators,
        particle_numbers: numbers,
        diagnostics: FixedPopulationDiagnostics {
            reference_population_defect: populations[0],
            reference_density_defect,
            population_defects: populations[1..].to_vec(),
            orthogonality_defects: unitary[1..].to_vec(),
            relative_stationarity_defects: relative_commutators,
        },
        residuals,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_rotation_matches_independent_sine_cosine_coefficients() {
        let (a, b) = (0.7_f64, -0.3_f64);
        let mut k = vec![Matrix::zeros(2, 2); 5];
        for (n, value) in [(1, a), (2, b)] {
            k[n][(0, 1)] = -value;
            k[n][(1, 0)] = value;
        }
        let result = exponential(&k, 4);
        let cosine = [1., 0., -a * a / 2., -a * b, -b * b / 2. + a.powi(4) / 24.];
        let sine = [0., a, b, -a.powi(3) / 6., -a * a * b / 2.];
        for n in 0..=4 {
            for (i, j, exact) in [
                (0, 0, cosine[n]),
                (1, 1, cosine[n]),
                (1, 0, sine[n]),
                (0, 1, -sine[n]),
            ] {
                assert!((result[n][(i, j)] - exact).abs() < 2e-16);
            }
        }
        k[1][(0, 1)] = f64::NAN;
        assert!(!norm(&k[1]).is_finite());
    }
}
