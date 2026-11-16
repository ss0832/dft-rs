// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! SAPT(0) with the single-pair intermolecular exchange approximation.
//!
//! Each interaction matrix element of (-sum_ij P_ij) is evaluated with
//! the bra's fragment labels held fixed. Overlap products on the other
//! occupied slots supply the spectator factors in Wick contractions.

use super::antisymmetry::{Interaction, Terms};
use crate::linalg::Matrix;
use crate::properties::pt2_density::transform;
use crate::scf::ScfResult;
use crate::{DftError, Result};

struct Slot {
    fragment: usize,
    spin: usize,
    orbital: usize,
}

struct Excitation {
    slot: usize,
    orbital: usize,
    gap: f64,
    induction: f64,
}

struct Operators {
    occupied: Vec<Slot>,
    overlap: Matrix,
    potential: [Matrix; 2],
    eri: Vec<f64>,
    nuclear_minus_electrostatic: f64,
}

impl Operators {
    fn eri(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        let n_occupied = self.occupied.len();
        let n_orbitals = self.overlap.cols;
        self.eri[((a * n_orbitals + b) * n_occupied + c) * n_orbitals + d]
    }

    fn exchange(&self, ket: &[usize], replacements: [Option<usize>; 2]) -> f64 {
        let mut exchange = 0.0;
        let occupied = &self.occupied;
        for (a, oa) in occupied.iter().enumerate().filter(|(_, o)| o.fragment == 0) {
            for (b, _) in occupied
                .iter()
                .enumerate()
                .filter(|(_, o)| o.fragment == 1 && o.spin == oa.spin)
            {
                let swapped = |i| {
                    if i == a {
                        ket[b]
                    } else if i == b {
                        ket[a]
                    } else {
                        ket[i]
                    }
                };
                let overlap_a = self.overlap[(a, ket[b])];
                let overlap_b = self.overlap[(b, ket[a])];
                let spectator = |j, k| {
                    // Spectators outside the swapped slots retain their
                    // monomer orbital. Replacing one by a virtual gives
                    // exact zero by intramonomer orthogonality.
                    if replacements
                        .into_iter()
                        .flatten()
                        .any(|i| i != a && i != b && i != j && i != k)
                    {
                        return 0.0;
                    }
                    let sa = if a == j || a == k { 1.0 } else { overlap_a };
                    let sb = if b == j || b == k { 1.0 } else { overlap_b };
                    sa * sb
                };
                // A single exchange contributes with the negative
                // permutation sign. Subtract the first-order polarization
                // energy times the exchanged state's overlap.
                let spectator_all = spectator(usize::MAX, usize::MAX);
                let mut value = self.nuclear_minus_electrostatic * spectator_all;
                for (i, o) in occupied.iter().enumerate() {
                    value += self.potential[1 - o.fragment][(i, swapped(i))] * spectator(i, i);
                    if o.fragment == 0 {
                        for (j, _) in occupied.iter().enumerate().filter(|(_, o)| o.fragment == 1) {
                            value += self.eri(i, swapped(i), j, swapped(j)) * spectator(i, j);
                        }
                    }
                }
                exchange -= value;
            }
        }
        exchange
    }
}

fn budget(occupied: usize, orbitals: usize, aos: usize, limit_mb: usize) -> Result<()> {
    // Bound the AO tensor and simultaneously retained partial transforms.
    let partial = aos
        .saturating_pow(4)
        .saturating_add(occupied.saturating_mul(aos.saturating_pow(3)));
    let second = occupied
        .saturating_mul(orbitals)
        .saturating_mul(aos.saturating_pow(2));
    let final_size = occupied.saturating_mul(orbitals).saturating_pow(2);
    let bytes = partial
        .saturating_add(second)
        .saturating_add(final_size)
        .saturating_mul(8);
    if bytes > limit_mb.saturating_mul(1024 * 1024) {
        return Err(DftError::InvalidInput(format!(
            "SAPT(0)-S2 orbital integral workspace needs {:.1} MiB, above the {limit_mb} MiB integral limit",
            bytes as f64 / (1024.0 * 1024.0),
        )));
    }
    Ok(())
}

pub(super) fn scaling_ratio(
    exact: f64,
    single_pair: f64,
    exchange_induction: [f64; 2],
    exchange_dispersion: f64,
) -> Result<f64> {
    if single_pair == 0.0
        && exact == 0.0
        && exchange_induction == [0.0; 2]
        && exchange_dispersion == 0.0
    {
        return Ok(1.0);
    }
    let ratio = exact / single_pair;
    if !ratio.is_finite() || ratio <= 0.0 || !ratio.powi(3).is_finite() {
        return Err(DftError::InvalidInput(format!(
            "sSAPT0 needs a finite positive first-order exchange ratio; full {exact:.9e} Hartree, single-pair {single_pair:.9e} Hartree"
        )));
    }
    Ok(ratio)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn terms(
    interaction: &Interaction<'_>,
    monomers: [&ScfResult; 2],
    tangents: [&[Matrix; 2]; 2],
    monomer_overlaps: [&Matrix; 2],
    electrostatic: f64,
    induction: [f64; 2],
    limit_mb: usize,
) -> Result<Terms> {
    let counts = monomers.map(|scf| [scf.n_alpha, scf.n_beta]);
    let n_occ: usize = counts.iter().flatten().sum();
    let n_ao = interaction.overlap.rows;
    let n_mo = monomers
        .iter()
        .map(|scf| {
            scf.mo_coeff[0].cols
                + if scf.unrestricted {
                    scf.mo_coeff[1].cols
                } else {
                    0
                }
        })
        .sum();
    budget(n_occ, n_mo, n_ao, limit_mb)?;
    let mut c_all = Matrix::zeros(n_ao, n_mo);
    let mut c_occ = Matrix::zeros(n_ao, n_occ);
    let mut occupied = Vec::with_capacity(n_occ);
    let mut excitations = [Vec::new(), Vec::new()];
    let mut ao_offset = 0;
    let mut mo_offset = 0;
    for fragment in 0..2 {
        let scf = monomers[fragment];
        let alpha_offset = mo_offset;
        for spin in 0..2 {
            let channel = if scf.unrestricted { spin } else { 0 };
            let mo = &scf.mo_coeff[channel];
            let energies = &scf.mo_energy[channel];
            let offset = if spin == 0 || scf.unrestricted {
                let offset = mo_offset;
                for a in 0..mo.rows {
                    for j in 0..mo.cols {
                        c_all[(ao_offset + a, offset + j)] = mo[(a, j)];
                    }
                }
                mo_offset += mo.cols;
                offset
            } else {
                alpha_offset
            };
            let n = counts[fragment][spin];
            let scm = monomer_overlaps[fragment].matmul(mo);
            let rotation = scm.transpose_matmul(&tangents[fragment][spin].matmul(&scm));
            for i in 0..n {
                let slot = occupied.len();
                occupied.push(Slot {
                    fragment,
                    spin,
                    orbital: offset + i,
                });
                for a in 0..mo.rows {
                    c_occ[(ao_offset + a, slot)] = mo[(a, i)];
                }
                for a in n..mo.cols {
                    let gap = energies[a] - energies[i];
                    if !gap.is_finite() || gap <= 0.0 {
                        return Err(DftError::InvalidInput(
                            "SAPT(0)-S2 requires finite positive Aufbau orbital gaps".into(),
                        ));
                    }
                    excitations[fragment].push(Excitation {
                        slot,
                        orbital: offset + a,
                        gap,
                        induction: rotation[(a, i)],
                    });
                }
            }
        }
        ao_offset += scf.mo_coeff[0].rows;
    }
    let orbital_operator = |op: &Matrix| c_occ.transpose_matmul(&op.matmul(&c_all));
    let operators = Operators {
        occupied,
        overlap: orbital_operator(interaction.overlap),
        potential: interaction.external.map(orbital_operator),
        eri: transform(interaction.eri, [&c_occ, &c_all, &c_occ, &c_all]),
        nuclear_minus_electrostatic: interaction.nuclear - electrostatic,
    };
    let reference: Vec<_> = operators.occupied.iter().map(|o| o.orbital).collect();
    let mut ket = reference.clone();
    let first_order = electrostatic + operators.exchange(&ket, [None, None]);
    let induction_with_exchange = std::array::from_fn(|fragment| {
        let mut value = induction[fragment];
        for x in &excitations[fragment] {
            ket[x.slot] = x.orbital;
            value += x.induction * operators.exchange(&ket, [Some(x.slot), None]);
            ket[x.slot] = reference[x.slot];
        }
        value
    });
    let mut dispersion = 0.0;
    let mut exchange_dispersion = 0.0;
    for a in &excitations[0] {
        ket[a.slot] = a.orbital;
        for b in &excitations[1] {
            ket[b.slot] = b.orbital;
            let v = operators.eri(a.slot, a.orbital, b.slot, b.orbital);
            let amplitude = -v / (a.gap + b.gap);
            dispersion += amplitude * v;
            exchange_dispersion +=
                amplitude * operators.exchange(&ket, [Some(a.slot), Some(b.slot)]);
            ket[b.slot] = reference[b.slot];
        }
        ket[a.slot] = reference[a.slot];
    }
    Ok(Terms {
        first_order,
        induction_with_exchange,
        dispersion,
        exchange_dispersion,
    })
}
