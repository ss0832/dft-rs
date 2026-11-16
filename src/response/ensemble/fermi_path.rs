// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-temperature canonical Fermi response in a transported MO frame.

use super::{EnsembleSolver, PopulationLaw};
use crate::linalg::{AccurateSum, Matrix};
use crate::periodic::BlochMatrix;
use crate::response::nuclear::{add, density_path, matrix_congruence};
use crate::response::thermal_matrix::fermi_matrix_taylor;
use crate::response::ResponseSolverConfig;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct FermiPathDiagnostics {
    /// Factorial-normalized chemical-potential coefficients (Hartree).
    pub chemical_potentials: Vec<[f64; 2]>,
    /// Maximum density defect against the Fermi matrix law at each order,
    /// including order zero, evaluated with the actual self-consistent Fock.
    pub density_defects: Vec<f64>,
}

pub(crate) struct CanonicalFermiPath {
    pub density_in_frame: [Vec<Matrix>; 2],
    pub entropy_energy: Vec<f64>,
    pub diagnostics: FermiPathDiagnostics,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
}

fn trace(matrix: &Matrix) -> f64 {
    (0..matrix.rows).map(|i| matrix[(i, i)]).sum()
}

fn restricted_spin_average(alpha: &mut Matrix, beta: &mut Matrix) {
    for (a, b) in alpha.as_mut_slice().iter_mut().zip(beta.as_mut_slice()) {
        let mean = 0.5 * (*a + *b);
        *a = mean;
        *b = mean;
    }
}

fn fermi_series(
    fock: &[Matrix],
    mu: &[[f64; 2]],
    spin: usize,
    temperature: f64,
    config: &ResponseSolverConfig,
) -> Result<Vec<Matrix>> {
    let n = fock[0].rows;
    let hamiltonian: Vec<_> = fock
        .iter()
        .zip(mu)
        .map(|(f, m)| {
            let mut h = f.clone();
            for i in 0..n {
                h[(i, i)] -= m[spin];
                for j in 0..i {
                    let average = 0.5 * (h[(i, j)] + h[(j, i)]);
                    h[(i, j)] = average;
                    h[(j, i)] = average;
                }
            }
            BlochMatrix::from_pair((h, Matrix::zeros(n, n)))
        })
        .collect();
    Ok(fermi_matrix_taylor(
        &hamiltonian,
        &[BlochMatrix::from_pair((
            Matrix::identity(n),
            Matrix::zeros(n, n),
        ))],
        &[temperature],
        fock.len() - 1,
        0.5,
        config,
    )?
    .density
    .into_iter()
    .map(|matrix| matrix.re)
    .collect())
}

/// Advance the coupled Fermi equation order by order. For a new coefficient
/// the source contains the Fermi matrix expansion with zero order-n density
/// and chemical potential. Its feedback and trace shift use the same linear
/// spectral solve as ordinary first-order ensemble response.
pub(crate) fn canonical_fermi_path(
    solver: &EnsembleSolver<'_>,
    frame: &[Vec<Matrix>; 2],
    mut density: [Vec<Matrix>; 2],
    order: usize,
    config: &ResponseSolverConfig,
    fock_for: impl Fn(&[Vec<Matrix>; 2], usize) -> Result<[Vec<Matrix>; 2]>,
) -> Result<CanonicalFermiPath> {
    let PopulationLaw::FermiDirac(temperature) = solver.population_law else {
        return Err(DftError::InvalidInput(
            "canonical Fermi response requires a positive electronic temperature".into(),
        ));
    };
    let scf = solver.kernel.scf;
    let numbers = [scf.n_alpha, scf.n_beta];
    let active: [bool; 2] = std::array::from_fn(|s| numbers[s] > 0 && numbers[s] < solver.nmo[s]);
    let mut mu = vec![[0.0; 2]; order + 1];
    for s in 0..2 {
        if active[s] && solver.channels[s].w_diag_sum == 0.0 {
            return Err(DftError::Unsupported {
                feature: "canonical Fermi Taylor response with unresolved occupation weights".into(),
                missing: "a nonzero derivative of the Fermi population sum with respect to its chemical potential".into(),
            });
        }
        mu[0][s] = scf.chemical_potentials[s].ok_or_else(|| {
            DftError::InvalidInput(
                "the finite-temperature reference needs chemical potentials".into(),
            )
        })?;
    }
    let evaluate = |d: &[Vec<Matrix>; 2], n| -> Result<[Vec<Matrix>; 2]> {
        let ao = std::array::from_fn(|s| density_path(&frame[s], &d[s], n));
        let fock = fock_for(&ao, n)?;
        let mut mo = std::array::from_fn(|s| matrix_congruence(&frame[s], &fock[s], n));
        if solver.restricted {
            let [alpha, beta] = &mut mo;
            for (a, b) in alpha.iter_mut().zip(beta) {
                restricted_spin_average(a, b);
            }
        }
        Ok(mo)
    };
    let offsets = solver.offsets();
    let mut residuals = Vec::with_capacity(order);
    let mut iterations = Vec::with_capacity(order);
    for n in 1..=order {
        for s in 0..2 {
            density[s].push(Matrix::zeros(solver.nmo[s], solver.nmo[s]));
        }
        let fock = evaluate(&density, n)?;
        let mut source = vec![0.0; offsets[2]];
        let mut number_shift = [0.0; 2];
        for s in 0..2 {
            if active[s] {
                density[s][n] = if solver.restricted && s == 1 {
                    // The restricted solve uses a single population law and
                    // therefore a single number target, including in roundoff.
                    density[0][n].clone()
                } else {
                    let bare = fermi_series(&fock[s], &mu, s, temperature, config)?;
                    bare[n].clone()
                };
            }
            source[offsets[s]..offsets[s + 1]].copy_from_slice(density[s][n].as_slice());
            number_shift[s] = -trace(&density[s][n]);
        }
        let induced = solver.kernel_mo(&source)?;
        let (result, coupled, _) = solver.solve_mo(&induced, number_shift, config)?;
        for s in 0..2 {
            for (d, correction) in density[s][n]
                .as_mut_slice()
                .iter_mut()
                .zip(&coupled[offsets[s]..offsets[s + 1]])
            {
                *d += correction;
            }
        }
        mu[n] = result.delta_chemical_potentials;
        if solver.restricted {
            let [alpha, beta] = &mut density;
            restricted_spin_average(&mut alpha[n], &mut beta[n]);
            mu[n] = [0.5 * (mu[n][0] + mu[n][1]); 2];
        }
        residuals.push(result.relative_residual);
        iterations.push(result.iterations);
    }

    let fock = evaluate(&density, order)?;
    let mut defects = vec![0.0_f64; order + 1];
    for s in 0..2 {
        if active[s] {
            let predicted = fermi_series(&fock[s], &mu, s, temperature, config)?;
            for (n, (actual, predicted)) in density[s].iter().zip(predicted.iter()).enumerate() {
                defects[n] = defects[n].max(actual.max_abs_diff(predicted));
            }
        }
    }
    // For D=f(F-mu), the derivative of kT*S/k_B is
    // Tr[(F-mu I) dD/dt]. Integrate this exact Taylor identity in the
    // orthonormal frame, avoiding endpoint logarithms for full/empty levels.
    let mut entropy_energy = vec![-temperature * scf.electronic_entropy];
    for n in 1..=order {
        let mut coefficient = AccurateSum::default();
        for s in 0..2 {
            for k in 0..n {
                let mut shifted_fock = fock[s][k].clone();
                add(
                    &mut shifted_fock,
                    &Matrix::identity(solver.nmo[s]),
                    -mu[k][s],
                );
                let derivative = &density[s][n - k];
                coefficient.add_product(
                    -((n - k) as f64) / n as f64,
                    shifted_fock.frobenius_dot(derivative),
                );
            }
        }
        entropy_energy.push(coefficient.value());
    }
    Ok(CanonicalFermiPath {
        density_in_frame: density,
        entropy_energy,
        diagnostics: FermiPathDiagnostics {
            chemical_potentials: mu,
            density_defects: defects,
        },
        residuals,
        iterations,
    })
}
