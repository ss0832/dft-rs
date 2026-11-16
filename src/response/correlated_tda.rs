// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Second-order TDA pole and transition-residue corrections. The parent
//! adiabatic DFT A operator is retained. Normal-ordered full-Coulomb PT2
//! supplies the second-order singles Hessian/metric, first-order doubles
//! coupling, and electric/magnetic property gradients, including ground
//! second-order singles. This is a perturbative propagator model, not an
//! energy-only shift or a frequency-dependent XC functional.

use super::tddft::{TdaSolver, TdaSpin, TdaState};

mod pole_path;
pub use pole_path::{correlated_pole_taylor, CorrelatedPoleTaylor, PropagatorTaylor};

#[cfg(test)]
#[path = "correlated_tda_tests.rs"]
mod tests;
use crate::basis::MolBasis;
use crate::integrals::{dipole_matrices, eri::EriTensor};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::properties::nmr::{angular_momentum_operator, gradient_operator};
use crate::properties::pt2::{Pt2Problem, Pt2Solution};
use crate::properties::pt2_transition::Pt2Transition;
use crate::scf::ScfResult;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};

/// Correlated poles, actual transition moments and raw equation diagnostics.
pub struct CorrelatedTdaResult {
    pub states: Vec<TdaState>,
    pub parent_energies: Vec<f64>,
    pub energy_corrections: Vec<f64>,
    pub singles_correction_norms: Vec<f64>,
    pub doubles_norms: Vec<f64>,
    pub pole_residuals: Vec<f64>,
    pub normalization_defects: Vec<f64>,
    pub smallest_double_denominators: Vec<f64>,
    pub parent_residual: f64,
    pub model: &'static str,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Symmetric metric transport fixes the within-block normalization gauge
/// without dividing by a degenerate excitation-energy difference.
fn block_metric_transport(vectors: &[Vec<f64>], metric: &Matrix, root: usize) -> Vec<f64> {
    let mut out = vec![0.0; vectors[0].len()];
    for (q, vector) in vectors.iter().enumerate() {
        for (z, v) in out.iter_mut().zip(vector) {
            *z -= 0.5 * metric[(q, root)] * v;
        }
    }
    out
}
fn flatten(state: &TdaState) -> Vec<f64> {
    if state.spin == TdaSpin::SpinResolved {
        state
            .x
            .iter()
            .flat_map(|m| m.as_slice().iter().copied())
            .collect()
    } else {
        state.x[0].as_slice().to_vec()
    }
}
fn expand(u: &[f64], scf: &ScfResult, spin: TdaSpin) -> [Matrix; 2] {
    let mut offset = 0;
    let mut out = std::array::from_fn(|s| {
        Matrix::zeros(
            scf.mo_coeff[s].cols - if s == 0 { scf.n_alpha } else { scf.n_beta },
            if s == 0 { scf.n_alpha } else { scf.n_beta },
        )
    });
    if spin == TdaSpin::SpinResolved {
        for m in &mut out {
            let size = m.as_slice().len();
            m.as_mut_slice().copy_from_slice(&u[offset..offset + size]);
            offset += size;
        }
    } else {
        for s in 0..2 {
            for (x, v) in out[s].as_mut_slice().iter_mut().zip(u) {
                *x = v / std::f64::consts::SQRT_2
                    * if s == 1 && spin == TdaSpin::Triplet {
                        -1.0
                    } else {
                        1.0
                    };
            }
        }
    }
    out
}
fn project(x: &[Matrix; 2], spin: TdaSpin) -> Vec<f64> {
    if spin == TdaSpin::SpinResolved {
        x.iter()
            .flat_map(|m| m.as_slice().iter().copied())
            .collect()
    } else {
        x[0].as_slice()
            .iter()
            .zip(x[1].as_slice())
            .map(|(a, b)| {
                (a + if spin == TdaSpin::Triplet { -b } else { *b }) / std::f64::consts::SQRT_2
            })
            .collect()
    }
}
fn correlation_action(
    model: &Pt2Transition,
    scf: &ScfResult,
    spin: TdaSpin,
    omega: f64,
    u: &[f64],
) -> Result<(Vec<f64>, [Vec<f64>; 3], f64, f64)> {
    let x = expand(u, scf, spin);
    let (a, _) = model.kernels(&x);
    let metric = model.metric(&x);
    let (doubles, norm, minimum) = model.doubles(omega, &x)?;
    let coupled = model.coupling(&doubles);
    let mut action = project(&a, spin);
    let metric = project(&metric, spin);
    let coupled = project(&coupled, spin);
    for k in 0..u.len() {
        action[k] += -omega * metric[k] + coupled[k];
    }
    Ok((action, doubles, norm, minimum))
}

/// Common-origin electric, angular-momentum and momentum transition
/// moments for a spin-resolved or spin-adapted parent. A dense parent
/// eigenspace provides the complete reduced resolvent with a checked
/// memory budget; the correlation kernels themselves are tensor actions.
/// Exactly degenerate parent roots are diagonalized as a whole subspace.
/// Coupled double poles are diagnosed rather than shifted or floored.
pub fn correlated_tda_states(
    parent: &TdaSolver<'_>,
    basis: &MolBasis,
    scf: &ScfResult,
    eri: &EriTensor,
    problem: &Pt2Problem<f64>,
    pt2: &Pt2Solution<f64>,
    coefficients: Pt2Coefficients,
    nstates: usize,
    memory_mb: usize,
) -> Result<CorrelatedTdaResult> {
    if nstates == 0 {
        return Err(DftError::InvalidInput(
            "PT2 transitions require at least one state".into(),
        ));
    }
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let dim = parent.dim();
    let bytes = dim
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(8 * 12))
        .and_then(|v| v.checked_add(Pt2Transition::workspace_bytes(eri.nao).ok()?))
        .ok_or_else(|| DftError::InvalidInput("PT2 parent resolvent workspace overflow".into()))?;
    let required_mb = bytes.div_ceil(1024 * 1024);
    if required_mb > memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "PT2 parent spectral resolvent",
            required_mb,
            limit_mb: memory_mb,
        });
    }
    let parents = parent.solve_dense(dim)?;
    if parents.is_empty() {
        return Err(DftError::InvalidInput(
            "PT2 transition space has no single excitations".into(),
        ));
    }
    let spin = parents[0].spin;
    let vectors: Vec<_> = parents.iter().map(flatten).collect();
    let energies: Vec<_> = parents.iter().map(|s| s.energy).collect();
    let mut parent_residual = 0.0_f64;
    for (u, &e) in vectors.iter().zip(&energies) {
        let applied = parent.a_vec(u)?;
        parent_residual = parent_residual.max(
            applied
                .iter()
                .zip(u)
                .map(|(a, u)| (a - e * u).powi(2))
                .sum::<f64>()
                .sqrt()
                / e.abs(),
        );
    }
    let model =
        Pt2Transition::from_reference(scf, eri, problem, &pt2.amplitudes, coefficients, memory_mb)?;
    let mut operators = Vec::new();
    for ao in dipole_matrices(basis, Vec3::zero())
        .into_iter()
        .chain(angular_momentum_operator(basis, Vec3::zero()))
        .chain(gradient_operator(basis))
    {
        operators.push(std::array::from_fn(|s| {
            scf.mo_coeff[s].transpose_matmul(&ao.matmul(&scf.mo_coeff[s]))
        }));
    }
    let properties: Vec<_> = operators
        .iter()
        .map(|op| {
            let (p2, pd) = model.property(op);
            let p0: [Matrix; 2] = std::array::from_fn(|s| {
                let o = if s == 0 { scf.n_alpha } else { scf.n_beta };
                let n = scf.mo_coeff[s].cols;
                let mut p = Matrix::zeros(n - o, o);
                for a in o..n {
                    for i in 0..o {
                        p[(a - o, i)] = op[s][(a, i)];
                    }
                }
                p
            });
            (project(&p0, spin), project(&p2, spin), pd)
        })
        .collect();
    let mut out=CorrelatedTdaResult{states:Vec::new(),parent_energies:Vec::new(),energy_corrections:Vec::new(),singles_correction_norms:Vec::new(),doubles_norms:Vec::new(),pole_residuals:Vec::new(),normalization_defects:Vec::new(),smallest_double_denominators:Vec::new(),parent_residual,
        model:"second-order normal-ordered full-Coulomb TDA propagator on an adiabatic parent; MP2 ground singles and singles/doubles property gradients; common origin; perturbative pole residues; target functional enters the ground energy only"};
    let mut first = 0;
    while first < dim && out.states.len() < nstates {
        let omega = energies[first];
        let mut last = first + 1;
        let numerical_degeneracy = 64.0 * f64::EPSILON * omega.abs();
        while last < dim && (energies[last] - omega).abs() <= numerical_degeneracy {
            last += 1;
        }
        let count = last - first;
        let mut block = Matrix::zeros(count, count);
        for j in 0..count {
            let (action, _, _, _) =
                correlation_action(&model, scf, spin, omega, &vectors[first + j])?;
            for i in 0..count {
                block[(i, j)] = dot(&vectors[first + i], &action);
            }
        }
        let (shifts, rotation) = symmetric_eigen(&block)?;
        let block_vectors: Vec<Vec<f64>> = (0..count)
            .map(|root| {
                let mut u = vec![0.0; dim];
                for j in 0..count {
                    for k in 0..dim {
                        u[k] += rotation[(j, root)] * vectors[first + j][k];
                    }
                }
                u
            })
            .collect();
        let block_doubles: Vec<_> = block_vectors
            .iter()
            .map(|u| model.doubles(omega, &expand(u, scf, spin)))
            .collect::<Result<_>>()?;
        let block_metrics: Vec<_> = block_vectors
            .iter()
            .map(|u| project(&model.metric(&expand(u, scf, spin)), spin))
            .collect();
        let mut normalization_metric = Matrix::zeros(count, count);
        for p in 0..count {
            for q in 0..count {
                normalization_metric[(p, q)] = dot(&block_vectors[p], &block_metrics[q])
                    + model.pair_dot(&block_doubles[p].0, &block_doubles[q].0);
            }
        }
        for (root, &shift) in shifts.iter().enumerate() {
            let u = block_vectors[root].clone();
            let (action, doubles, dnorm, minimum) =
                correlation_action(&model, scf, spin, omega, &u)?;
            let metric = project(&model.metric(&expand(&u, scf, spin)), spin);
            let snorm = dot(&u, &metric);
            let mut z = block_metric_transport(&block_vectors, &normalization_metric, root);
            for j in 0..dim {
                if (first..last).contains(&j) {
                    continue;
                }
                let factor = dot(&vectors[j], &action) / (omega - energies[j]);
                for k in 0..dim {
                    z[k] += factor * vectors[j][k];
                }
            }
            let az = parent.a_vec(&z)?;
            let residual = az
                .iter()
                .zip(&z)
                .zip(&action)
                .zip(&u)
                .map(|(((az, z), a), u)| (az - omega * z + a - shift * u).powi(2))
                .sum::<f64>()
                .sqrt();
            let normalization = (2.0 * dot(&u, &z) + snorm + dnorm).abs();
            let moments: Vec<_> = properties
                .iter()
                .map(|(p0, p2, pd)| {
                    dot(p0, &u) + dot(p0, &z) + dot(p2, &u) + model.pair_dot(pd, &doubles)
                })
                .collect();
            let energy = omega + shift;
            if !(energy > 0.0) || !energy.is_finite() {
                return Err(DftError::InvalidInput("PT2-corrected TDA pole is nonpositive or nonfinite; the perturbative reference is unstable".into()));
            }
            let corrected: Vec<_> = u.iter().zip(&z).map(|(u, z)| u + z).collect();
            let mut x = expand(&corrected, scf, spin);
            // Public TdaState uses a unit-norm spatial block for restricted
            // spin adaptation; the physical spin tensors above use 1/sqrt(2).
            if spin != TdaSpin::SpinResolved {
                for m in &mut x {
                    for v in m.as_mut_slice() {
                        *v *= std::f64::consts::SQRT_2;
                    }
                }
            }
            let d = [moments[0], moments[1], moments[2]];
            let l = [moments[3], moments[4], moments[5]];
            let p = [moments[6], moments[7], moments[8]];
            let mut state = TdaState {
                energy,
                x,
                spin,
                tamm_dancoff: true,
                transition_dipole: d,
                oscillator_strength: 2.0 / 3.0 * energy * dot(&d, &d),
                transition_angular_momentum: l,
                transition_momentum: p,
                rotatory_strength: -0.5 * dot(&d, &l),
                rotatory_strength_velocity: 0.5 * dot(&p, &l) / energy,
            };
            let uncorrelated: Vec<_> = properties.iter().map(|(p0, _, _)| dot(p0, &u)).collect();
            state.canonicalize_phase(Some([
                [uncorrelated[0], uncorrelated[1], uncorrelated[2]],
                [uncorrelated[6], uncorrelated[7], uncorrelated[8]],
                [uncorrelated[3], uncorrelated[4], uncorrelated[5]],
            ]));
            out.states.push(state);
            out.parent_energies.push(omega);
            out.energy_corrections.push(shift);
            out.singles_correction_norms.push(dot(&z, &z).sqrt());
            out.doubles_norms.push(dnorm);
            out.pole_residuals.push(residual);
            out.normalization_defects.push(normalization);
            out.smallest_double_denominators.push(minimum);
        }
        first = last;
    }
    Ok(out)
}
