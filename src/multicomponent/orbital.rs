// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic quantum-orbital Newton proposals for the variational line search.
//! The complete nuclear density kernel is differentiated at the current
//! orbitals; an energy-flat fixed-point map is never declared converged.

use super::correlation::CorrelationModel;
use super::{add, block, orbital_density, CorrelationGrid, QuantumParticle, QuantumState};
use crate::linalg::{symmetric_eigen, Matrix, Orthogonalizer};
use crate::scf::ScfResult;
use crate::Result;

#[allow(clippy::too_many_arguments)]
pub(super) fn proposal(
    electronic: &ScfResult,
    quantum: &[QuantumState],
    orth: &[Orthogonalizer],
    grid: Option<&CorrelationGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    coulomb: &impl Fn(&Matrix, usize) -> Result<Matrix>,
    offsets: &[usize],
) -> Result<Option<Vec<Vec<f64>>>> {
    let mut frames = Vec::new();
    for (q, o) in quantum.iter().zip(orth) {
        let c = Matrix::from_row_major(q.coefficient.len(), 1, q.coefficient.clone());
        let y = o.x.transpose_matmul_seq(&q.overlap.matmul_seq(&c));
        let mut v: Vec<_> = (0..y.rows)
            .map(|i| if i == 0 { 1.0 - y[(i, 0)] } else { -y[(i, 0)] })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut householder = Matrix::identity(y.rows);
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
            for i in 0..y.rows {
                for j in 0..y.rows {
                    householder[(i, j)] -= 2.0 * v[i] * v[j];
                }
            }
        }
        frames.push(o.x.matmul_seq(&householder));
    }
    let slots: Vec<_> = frames
        .iter()
        .enumerate()
        .flat_map(|(i, f)| (1..f.cols).map(move |a| (i, a)))
        .collect();
    if slots.is_empty() {
        return Ok(None);
    }
    let fock: Vec<_> = frames
        .iter()
        .zip(quantum)
        .map(|(c, q)| c.transpose_matmul_seq(&q.fock).matmul_seq(c))
        .collect();
    let gradient: Vec<_> = slots.iter().map(|&(i, a)| fock[i][(a, 0)]).collect();
    let mut h = Matrix::zeros(slots.len(), slots.len());
    let mut waves = Vec::new();
    let mut electron = Vec::new();
    let mut partner: Vec<Vec<f64>> = Vec::new();
    if let Some(g) = grid.filter(|_| !correlation.is_zero()) {
        electron = orbital_density(
            &g.phi[0],
            &electronic.mo_coeff[0],
            &electronic.occupations[0],
        );
        for (e, b) in electron.iter_mut().zip(orbital_density(
            &g.phi[0],
            &electronic.mo_coeff[1],
            &electronic.occupations[1],
        )) {
            *e += b;
        }
        partner = vec![vec![0.0; electron.len()]; correlation.len()];
        for (i, c) in frames.iter().enumerate() {
            let wave = g.phi[i + 1].matmul_seq(c);
            if let Some(channel) = particles[i].correlation_channel {
                for j in 0..electron.len() {
                    partner[channel][j] += wave[(j, 0)].powi(2);
                }
            }
            waves.push(wave);
        }
    }
    for (column, &(s, b)) in slots.iter().enumerate() {
        let mut dp = Matrix::zeros(frames[s].rows, frames[s].rows);
        for i in 0..dp.rows {
            for j in 0..dp.cols {
                dp[(i, j)] =
                    frames[s][(i, 0)] * frames[s][(j, b)] + frames[s][(i, b)] * frames[s][(j, 0)];
            }
        }
        let j = if particles.len() > 1 {
            Some(coulomb(&dp, offsets[s + 1])?)
        } else {
            None
        };
        let channel = particles[s].correlation_channel;
        let kernel =
            if let (Some(g), Some(channel)) = (grid.filter(|_| !correlation.is_zero()), channel) {
                let provider = &correlation.channels()[channel];
                let partner = &partner[channel];
                Some(
                    (0..g.weights.len())
                        .map(|k| {
                            provider
                                .kernel_action(
                                    electron[k],
                                    partner[k],
                                    super::correlation::representable_tangent(
                                        [electron[k], partner[k]],
                                        [0.0, 2.0 * waves[s][(k, 0)] * waves[s][(k, b)]],
                                    ),
                                )
                                .map(|v| g.weights[k] * v[1])
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                None
            };
        for (row, &(t, a)) in slots.iter().enumerate() {
            let mut value = if s == t {
                fock[s][(a, b)] - if a == b { fock[s][(0, 0)] } else { 0.0 }
            } else {
                0.0
            };
            if s != t {
                if let Some(j) = &j {
                    let v = block(j, offsets[t + 1], frames[t].rows);
                    let v = frames[t].transpose_matmul_seq(&v).matmul_seq(&frames[t]);
                    value += particles[s].charge * particles[t].charge * v[(a, 0)];
                }
            }
            if channel.is_some() && particles[t].correlation_channel == channel {
                if let Some(k) = &kernel {
                    value += k
                        .iter()
                        .enumerate()
                        .map(|(g, k)| k * waves[t][(g, a)] * waves[t][(g, 0)])
                        .sum::<f64>();
                }
            }
            h[(row, column)] = value;
        }
    }
    let (eigenvalues, vectors) = symmetric_eigen(&h)?;
    // Newton is used only in a positive-curvature quantum tangent space.
    // An indefinite or singular space retains the ordinary descent search.
    if eigenvalues.iter().any(|x| *x <= 0.0) {
        return Ok(None);
    }
    let mut step = vec![0.0; slots.len()];
    for (k, e) in eigenvalues.iter().enumerate() {
        let component: f64 = gradient
            .iter()
            .enumerate()
            .map(|(i, g)| vectors[(i, k)] * g)
            .sum();
        for i in 0..step.len() {
            step[i] -= vectors[(i, k)] * component / e;
        }
    }
    let mut next: Vec<_> = quantum
        .iter()
        .map(|q| Matrix::from_row_major(q.coefficient.len(), 1, q.coefficient.clone()))
        .collect();
    for (&(s, a), u) in slots.iter().zip(step) {
        let column = Matrix::from_row_major(
            frames[s].rows,
            1,
            (0..frames[s].rows).map(|i| frames[s][(i, a)]).collect(),
        );
        add(&mut next[s], &column, u);
    }
    Ok(Some(
        next.into_iter().map(|c| c.as_slice().to_vec()).collect(),
    ))
}
