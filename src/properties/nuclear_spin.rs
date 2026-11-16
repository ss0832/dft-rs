// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact isotropic nuclear-spin spectra in conserved total-Iz blocks.
//!
//! H/h = -sum_i (carrier_i + offset_i) Iz_i + sum_ij J_ij I_i.I_j.
//! Transverse transition strengths are normalized to unit integrated area per
//! observed nucleus, with the common high-temperature population factor removed.
//! This is a spin-Hamiltonian postprocessor of arbitrary shielding/J providers;
//! it adds no electronic approximation or functional-name dispatch.

use crate::linalg::{symmetric_eigen, Matrix};
use crate::{DftError, Result};
use std::collections::BTreeSet;

/// Transition offset from the positive observation carrier and integrated area.
#[derive(Clone, Copy, Debug)]
pub struct SpinLine {
    pub offset_hz: f64,
    pub weight: f64,
}

/// Exact finite spin-Hamiltonian result; each line vector is one isotope group.
/// All adjacent-Iz candidates, including exactly dark lines, are retained.
pub struct SpinSpectrum {
    pub lines: Vec<Vec<SpinLine>>,
    pub hilbert_dimension: usize,
    /// Largest component of H V - V E over all total-Iz blocks, in Hz.
    pub eigen_residual_hz: f64,
    /// Largest block residual divided by its largest Hamiltonian entry.
    pub eigen_relative_residual: f64,
    /// Signed defect of the integrated-area sum rule of each observed group.
    pub sum_rule_defects: Vec<f64>,
}

struct Block {
    states: Vec<usize>,
    energy_reference: Vec<usize>,
    energy_shift: Vec<f64>,
    vectors: Matrix,
}

/// Solve the isotropic Hamiltonian, retaining flip-flop terms and coherent
/// detection of equivalent nuclei. Spins are specified as twice I. A group
/// contains distinct nuclei with the same signed carrier; the requested
/// dimension limit bounds resources rather than truncating the Hilbert space.
/// All frequencies and J values are in Hz. No weak-transition floor is used.
pub fn nuclear_spin_spectrum(
    carriers_hz: &[f64],
    offsets_hz: &[f64],
    twice_spins: &[usize],
    couplings_hz: &[(usize, usize, f64)],
    groups: &[Vec<usize>],
    max_dimension: usize,
) -> Result<SpinSpectrum> {
    let invalid = |message: &str| DftError::InvalidInput(message.into());
    let n = carriers_hz.len();
    if n == 0 || offsets_hz.len() != n || twice_spins.len() != n || max_dimension == 0 {
        return Err(invalid("nuclear-spin carriers, offsets and spins need equal nonzero lengths and a positive dimension limit"));
    }
    if carriers_hz.iter().any(|v| !v.is_finite() || *v == 0.0)
        || offsets_hz.iter().any(|v| !v.is_finite())
        || twice_spins.contains(&0)
    {
        return Err(invalid(
            "nuclear-spin carriers must be finite and nonzero, offsets finite and spins positive",
        ));
    }
    let mut strides = Vec::with_capacity(n);
    let mut dimension = 1usize;
    for &spin in twice_spins {
        strides.push(dimension);
        dimension = spin
            .checked_add(1)
            .and_then(|s| dimension.checked_mul(s))
            .ok_or_else(|| {
                invalid("nuclear-spin Hilbert dimension overflows addressable memory")
            })?;
        if dimension > max_dimension {
            return Err(invalid("nuclear-spin Hilbert dimension exceeds nmr_spin_max_dimension; increase the explicit resource limit"));
        }
    }
    let mut pairs = BTreeSet::new();
    for &(i, j, value) in couplings_hz {
        if i >= n || j >= n || i == j || !value.is_finite() || !pairs.insert((i.min(j), i.max(j))) {
            return Err(invalid("nuclear-spin couplings require distinct in-range pairs, finite values and no duplicate pairs"));
        }
    }
    for group in groups {
        let unique: BTreeSet<_> = group.iter().copied().collect();
        if group.is_empty()
            || unique.len() != group.len()
            || group.iter().any(|&i| i >= n)
            || group
                .iter()
                .any(|&i| carriers_hz[i] != carriers_hz[group[0]])
        {
            return Err(invalid(
                "each observed isotope group needs distinct in-range spins with one signed carrier",
            ));
        }
    }
    let levels = |state: usize, i: usize| (state / strides[i]) % (twice_spins[i] + 1);
    let total_spin: usize = twice_spins.iter().sum();
    let mut sectors = vec![Vec::new(); total_spin + 1];
    for state in 0..dimension {
        let sector: usize = (0..n).map(|i| levels(state, i)).sum();
        sectors[sector].push(state);
    }
    let mut rows = vec![0usize; dimension];
    for sector in &sectors {
        for (row, &state) in sector.iter().enumerate() {
            rows[state] = row;
        }
    }
    // One Larmor frequency can be removed within every total-Iz block.
    // For transition offsets the same carrier contraction uses integer
    // product-state differences, so the observed carrier cancels exactly.
    let frame = carriers_hz[0];
    let carrier_difference = |hi: usize, lo: usize, origin: f64| {
        -(0..n)
            .map(|i| (carriers_hz[i] - origin) * (levels(hi, i) as f64 - levels(lo, i) as f64))
            .sum::<f64>()
    };
    let mut residual = 0.0f64;
    let mut relative_residual = 0.0f64;
    let mut blocks = Vec::with_capacity(sectors.len());
    for states in sectors {
        let size = states.len();
        let mut h = Matrix::zeros(size, size);
        let mut small_diagonal = vec![0.0; size];
        let mut carrier_diagonal = vec![0.0; size];
        let mut transverse = Vec::new();
        for (row, &state) in states.iter().enumerate() {
            let m: Vec<_> = (0..n)
                .map(|i| levels(state, i) as f64 - 0.5 * twice_spins[i] as f64)
                .collect();
            carrier_diagonal[row] = -(0..n).map(|i| (carriers_hz[i] - frame) * m[i]).sum::<f64>();
            small_diagonal[row] = -(0..n).map(|i| offsets_hz[i] * m[i]).sum::<f64>();
            for &(i, j, coupling) in couplings_hz {
                small_diagonal[row] += coupling * m[i] * m[j];
                let (di, dj) = (levels(state, i), levels(state, j));
                if di < twice_spins[i] && dj > 0 {
                    let other = state + strides[i] - strides[j];
                    let column = rows[other];
                    let amplitude = 0.5
                        * coupling
                        * (((twice_spins[i] - di) as f64 * (di + 1) as f64)
                            * (dj as f64 * (twice_spins[j] - dj + 1) as f64))
                            .sqrt();
                    h[(row, column)] += amplitude;
                    h[(column, row)] += amplitude;
                    transverse.push((row, column, amplitude));
                }
            }
            h[(row, row)] = carrier_diagonal[row] + small_diagonal[row];
        }
        if h.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "nuclear-spin Hamiltonian exceeds finite floating-point range",
            ));
        }
        let (_, vectors) = symmetric_eigen(&h)?;
        // Reconstruct each Rayleigh eigenvalue as a product-state carrier
        // plus a small correction. Reintroducing the carrier before taking a
        // transition would discard the heteronuclear shift's low bits.
        let mut energy_reference = Vec::with_capacity(size);
        let mut energy_shift = Vec::with_capacity(size);
        let mut energies = Vec::with_capacity(size);
        for j in 0..size {
            let pivot = (0..size)
                .max_by(|&a, &b| vectors[(a, j)].abs().total_cmp(&vectors[(b, j)].abs()))
                .expect("nonempty total-Iz block");
            let reference = states[pivot];
            let mut norm = 0.0;
            let mut correction = 0.0;
            for (i, &state) in states.iter().enumerate() {
                let weight = vectors[(i, j)].powi(2);
                norm += weight;
                correction +=
                    weight * (carrier_difference(state, reference, frame) + small_diagonal[i]);
            }
            for &(i, k, coupling) in &transverse {
                correction += 2.0 * vectors[(i, j)] * coupling * vectors[(k, j)];
            }
            let shift = correction / norm;
            energy_reference.push(reference);
            energy_shift.push(shift);
            energies.push(carrier_diagonal[pivot] + shift);
        }
        let work = h.matmul_seq(&vectors);
        let scale = h.as_slice().iter().map(|v| v.abs()).fold(0.0, f64::max);
        let mut block_residual = 0.0f64;
        for i in 0..size {
            for j in 0..size {
                let error = work[(i, j)] - vectors[(i, j)] * energies[j];
                if !error.is_finite() {
                    return Err(invalid("nuclear-spin eigensystem residual is not finite"));
                }
                block_residual = block_residual.max(error.abs());
            }
        }
        residual = residual.max(block_residual);
        relative_residual = relative_residual.max(if scale == 0.0 {
            0.0
        } else {
            block_residual / scale
        });
        blocks.push(Block {
            states,
            energy_reference,
            energy_shift,
            vectors,
        });
    }
    let mut all_lines = Vec::with_capacity(groups.len());
    let mut defects = Vec::with_capacity(groups.len());
    for group in groups {
        let spin_norm: f64 = group
            .iter()
            .map(|&i| {
                let spin = 0.5 * twice_spins[i] as f64;
                spin * (spin + 1.0)
            })
            .sum();
        let normalization = 6.0 * group.len() as f64 / (dimension as f64 * spin_norm);
        let observed = carriers_hz[group[0]];
        let carrier = observed.abs();
        let mut lines = Vec::new();
        for pair in blocks.windows(2) {
            let (low, high) = (&pair[0], &pair[1]);
            let mut transition = Matrix::zeros(high.states.len(), low.states.len());
            for (column, &state) in low.states.iter().enumerate() {
                for &i in group {
                    let digit = levels(state, i);
                    if digit < twice_spins[i] {
                        let row = rows[state + strides[i]];
                        transition[(row, column)] +=
                            0.5 * (((twice_spins[i] - digit) as f64) * (digit + 1) as f64).sqrt();
                    }
                }
            }
            let amplitudes = high
                .vectors
                .transpose_matmul_seq(&transition.matmul_seq(&low.vectors));
            for i in 0..high.states.len() {
                for j in 0..low.states.len() {
                    let hi = high.energy_reference[i];
                    let lo = low.energy_reference[j];
                    let shift = high.energy_shift[i] - low.energy_shift[j];
                    let delta = carrier_difference(hi, lo, 0.0) + shift;
                    let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
                    // Adjacent total-Iz sectors differ by one total
                    // projection. Subtract the signed observation carrier
                    // inside the product-state contraction.
                    let offset = sign * (carrier_difference(hi, lo, observed) + shift)
                        + (-sign * observed - carrier);
                    let weight = normalization * amplitudes[(i, j)].powi(2);
                    if !offset.is_finite() || !weight.is_finite() {
                        return Err(invalid(
                            "nuclear-spin transition exceeds finite floating-point range",
                        ));
                    }
                    lines.push(SpinLine {
                        offset_hz: offset,
                        weight,
                    });
                }
            }
        }
        lines.sort_by(|a, b| a.offset_hz.total_cmp(&b.offset_hz));
        defects.push(lines.iter().map(|line| line.weight).sum::<f64>() - group.len() as f64);
        all_lines.push(lines);
    }
    Ok(SpinSpectrum {
        lines: all_lines,
        hilbert_dimension: dimension,
        eigen_residual_hz: residual,
        eigen_relative_residual: relative_residual,
        sum_rule_defects: defects,
    })
}
