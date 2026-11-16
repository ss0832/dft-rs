// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Single-determinant SAPT(0) exchange at full intermolecular overlap.
//!
//! The normalized one-sided interaction is <AB|V A|A'B'>/<AB|A|A'B'>,
//! where V acts on the electron slots assigned to opposite fragments.
//! Keeping those bra slots tagged in the determinant minors gives the
//! two transition densities below. No projection of a transition density
//! or single-pair expansion of the antisymmetrizer enters this route.

use super::exchange_induction::OccupiedProjector;
use crate::integrals::eri::EriTensor;
use crate::linalg::Matrix;
use crate::scf::ScfResult;
use crate::{DftError, Result};

type TaggedMatrices = [[Matrix; 2]; 2];

pub(super) struct Interaction<'a> {
    pub(super) eri: &'a EriTensor,
    pub(super) overlap: &'a Matrix,
    pub(super) external: [&'a Matrix; 2],
    pub(super) nuclear: f64,
}

pub(super) struct Terms {
    pub(super) first_order: f64,
    pub(super) induction_with_exchange: [f64; 2],
    pub(super) dispersion: f64,
    pub(super) exchange_dispersion: f64,
}

struct Transition<'a> {
    // For fixed bra L, inverse M = (L^T S C)^-1. The row of Z[x]
    // tagged by fragment x gets only left occupied columns from x.
    bra_rows: TaggedMatrices,
    base: TaggedMatrices,
    projectors: &'a [OccupiedProjector; 2],
    overlap: &'a Matrix,
}

struct Excitation {
    spin: usize,
    index: usize,
    gap: f64,
    projected_virtual: Vec<f64>,
    inverse_overlap_column: Vec<f64>,
    occupied: Vec<f64>,
    virtual_orbital: Vec<f64>,
    first: f64,
}

fn add(target: &mut Matrix, source: &Matrix, scale: f64) {
    for (a, &b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *a += scale * b;
    }
}

fn matrix(rows: usize, cols: usize, entry: impl Fn(usize, usize) -> f64) -> Matrix {
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            values.push(entry(row, col));
        }
    }
    Matrix::from_row_major(rows, cols, values)
}

fn direct_field(eri: &EriTensor, p: &Matrix) -> Matrix {
    let n = p.rows;
    let mut j = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..=mu {
            let mut val = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    val += p[(lam, sig)] * eri.get(mu, nu, lam, sig);
                }
            }
            j[(mu, nu)] = val;
            j[(nu, mu)] = val;
        }
    }
    j
}

/// The same-spin exchange contraction retains both matrix endpoints of a
/// non-symmetric transition density. The usual symmetric SCF K build has a
/// different input contract.
fn exchange_field(eri: &EriTensor, p: &Matrix) -> Matrix {
    let n = p.rows;
    let mut k = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut val = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    val += p[(lam, sig)] * eri.get(nu, lam, sig, mu);
                }
            }
            k[(mu, nu)] = val;
        }
    }
    k
}

fn interaction_fields(eri: &EriTensor, densities: &TaggedMatrices) -> TaggedMatrices {
    std::array::from_fn(|fragment| {
        let partner = &densities[1 - fragment];
        let mut total = partner[0].clone();
        add(&mut total, &partner[1], 1.0);
        let j = direct_field(eri, &total);
        std::array::from_fn(|spin| {
            let k = exchange_field(eri, &partner[spin]);
            let mut field = j.clone();
            add(&mut field, &k, -1.0);
            field
        })
    })
}

impl<'a> Transition<'a> {
    fn new(
        projectors: &'a [OccupiedProjector; 2],
        overlap: &'a Matrix,
        occupied_a: [usize; 2],
    ) -> Self {
        let bra_rows = std::array::from_fn(|fragment| {
            std::array::from_fn(|spin| {
                let p = &projectors[spin];
                let c = &p.coefficients;
                let mut rows = c.transpose();
                for i in 0..rows.rows {
                    if usize::from(i >= occupied_a[spin]) != fragment {
                        for j in 0..rows.cols {
                            rows[(i, j)] = 0.0;
                        }
                    }
                }
                p.inverse_metric.matmul(&rows)
            })
        });
        let base = std::array::from_fn(|fragment| {
            std::array::from_fn(|spin| {
                projectors[spin]
                    .coefficients
                    .matmul(&bra_rows[fragment][spin])
            })
        });
        Self {
            bra_rows,
            base,
            projectors,
            overlap,
        }
    }

    fn first_order(&self, interaction: &Interaction<'_>, fields: &mut TaggedMatrices) -> f64 {
        let mut value = interaction.nuclear;
        // Contract the two-electron contribution with just one side.
        for spin in 0..2 {
            value += self.base[0][spin].frobenius_dot(&fields[0][spin]);
            for fragment in 0..2 {
                let v = interaction.external[1 - fragment];
                value += self.base[fragment][spin].frobenius_dot(v);
                add(&mut fields[fragment][spin], v, 1.0);
            }
        }
        value
    }

    fn induction(
        &self,
        tangent: &[Matrix; 2],
        monomer_overlap: &Matrix,
        fields: &TaggedMatrices,
        offset: usize,
    ) -> f64 {
        (0..2)
            .map(|spin| {
                let p = &self.projectors[spin];
                let projected =
                    p.polarization_direction(&tangent[spin], monomer_overlap, self.overlap, offset);
                (0..2)
                    .map(|x| {
                        projected
                            .matmul(&self.bra_rows[x][spin])
                            .frobenius_dot(&fields[x][spin])
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    fn rank_contraction(&self, fields: &TaggedMatrices, mode: &Excitation, index: usize) -> f64 {
        (0..2)
            .map(|x| {
                let operator = &fields[x][mode.spin];
                let z = &self.bra_rows[x][mode.spin];
                mode.projected_virtual
                    .iter()
                    .enumerate()
                    .map(|(i, &g)| {
                        g * (0..operator.cols)
                            .map(|j| operator[(i, j)] * z[(index, j)])
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    fn modes(
        &self,
        scf: &ScfResult,
        offset: usize,
        occupied_start: [usize; 2],
        fields: &TaggedMatrices,
    ) -> Result<Vec<Excitation>> {
        let n = self.overlap.rows;
        let mut modes = Vec::new();
        for (spin, count) in [scf.n_alpha, scf.n_beta].into_iter().enumerate() {
            let channel = if scf.unrestricted { spin } else { 0 };
            let mo = &scf.mo_coeff[channel];
            let p = &self.projectors[spin];
            let inverse_times_bra = p.inverse_metric.matmul(&p.coefficients.transpose());
            let column = |j| {
                (0..n)
                    .map(|i| {
                        if (offset..offset + mo.rows).contains(&i) {
                            mo[(i - offset, j)]
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            };
            for a in count..mo.cols {
                let virtual_orbital = column(a);
                let virtual_as_matrix = Matrix::from_row_major(n, 1, virtual_orbital.clone());
                let overlap_virtual = self.overlap.matmul(&virtual_as_matrix);
                let inverse_overlap_column = inverse_times_bra
                    .matmul(&overlap_virtual)
                    .as_slice()
                    .to_vec();
                let removed = p.density.matmul(&overlap_virtual);
                let projected_virtual = virtual_orbital
                    .iter()
                    .zip(removed.as_slice())
                    .map(|(v, r)| v - r)
                    .collect::<Vec<_>>();
                for i in 0..count {
                    let gap = scf.mo_energy[channel][a] - scf.mo_energy[channel][i];
                    if !gap.is_finite() || gap <= 0.0 {
                        return Err(DftError::InvalidInput(
                            "SAPT(0) requires finite positive Aufbau orbital gaps".into(),
                        ));
                    }
                    let mut mode = Excitation {
                        spin,
                        index: occupied_start[spin] + i,
                        gap,
                        projected_virtual: projected_virtual.clone(),
                        inverse_overlap_column: inverse_overlap_column.clone(),
                        occupied: column(i),
                        virtual_orbital: virtual_orbital.clone(),
                        first: 0.0,
                    };
                    mode.first = self.rank_contraction(fields, &mode, mode.index);
                    modes.push(mode);
                }
            }
        }
        Ok(modes)
    }
}

/// First-order normalized interaction without constructing excited states.
pub(super) fn first_order(
    interaction: &Interaction<'_>,
    projectors: &[OccupiedProjector; 2],
    occupied_a: [usize; 2],
) -> f64 {
    let transition = Transition::new(projectors, interaction.overlap, occupied_a);
    let mut fields = interaction_fields(interaction.eri, &transition.base);
    transition.first_order(interaction, &mut fields)
}

/// Induction including exchange for externally supplied density tangents.
/// The bare interaction in `interaction` determines the final energy
/// contraction even if the tangents were obtained with regularized sources.
pub(super) fn coupled_induction(
    interaction: &Interaction<'_>,
    projectors: &[OccupiedProjector; 2],
    occupied_a: [usize; 2],
    induction_tangents: [&[Matrix; 2]; 2],
    monomer_overlaps: [&Matrix; 2],
    nao_a: usize,
) -> [f64; 2] {
    let transition = Transition::new(projectors, interaction.overlap, occupied_a);
    let mut fields = interaction_fields(interaction.eri, &transition.base);
    transition.first_order(interaction, &mut fields);
    std::array::from_fn(|fragment| {
        transition.induction(
            induction_tangents[fragment],
            monomer_overlaps[fragment],
            &fields,
            [0, nao_a][fragment],
        )
    })
}

/// Normalized second-order term from the mixed transition per occupied-virtual
/// pair, including the two overlap-determinant factors in the normalized SAPT
/// recurrence. Reference and induction come from the same tagged densities.
#[allow(clippy::too_many_arguments)]
pub(super) fn terms(
    interaction: &Interaction<'_>,
    projectors: &[OccupiedProjector; 2],
    monomers: [&ScfResult; 2],
    induction_tangents: [&[Matrix; 2]; 2],
    monomer_overlaps: [&Matrix; 2],
    nao_a: usize,
) -> Result<Terms> {
    let occ_a = [monomers[0].n_alpha, monomers[0].n_beta];
    let trans = Transition::new(projectors, interaction.overlap, occ_a);
    let mut fields = interaction_fields(interaction.eri, &trans.base);
    let first_order = trans.first_order(interaction, &mut fields);
    let induction_with_exchange = [
        trans.induction(induction_tangents[0], monomer_overlaps[0], &fields, 0),
        trans.induction(induction_tangents[1], monomer_overlaps[1], &fields, nao_a),
    ];
    let modes_a = trans.modes(monomers[0], 0, [0; 2], &fields)?;
    let modes_b = trans.modes(monomers[1], nao_a, occ_a, &fields)?;
    let n = interaction.overlap.rows;
    let mut dispersion = 0.0;
    let mut exchange_dispersion = 0.0;
    for a in &modes_a {
        let left_tangent: TaggedMatrices = std::array::from_fn(|fragment| {
            std::array::from_fn(|spin| {
                if spin == a.spin {
                    matrix(n, n, |i, j| {
                        a.projected_virtual[i] * trans.bra_rows[fragment][spin][(a.index, j)]
                    })
                } else {
                    Matrix::zeros(n, n)
                }
            })
        });
        let left_fields = interaction_fields(interaction.eri, &left_tangent);
        let charge_a = matrix(n, n, |i, j| a.virtual_orbital[i] * a.occupied[j]);
        let direct = direct_field(interaction.eri, &charge_a);
        for b in &modes_b {
            let v = b
                .virtual_orbital
                .iter()
                .enumerate()
                .map(|(i, &cv)| {
                    cv * b
                        .occupied
                        .iter()
                        .enumerate()
                        .map(|(j, &co)| direct[(i, j)] * co)
                        .sum::<f64>()
                })
                .sum::<f64>();
            let mut mixed = trans.rank_contraction(&left_fields, b, b.index);
            if a.spin == b.spin {
                mixed -= b.inverse_overlap_column[a.index]
                    * trans.rank_contraction(&fields, a, b.index)
                    + a.inverse_overlap_column[b.index]
                        * trans.rank_contraction(&fields, b, a.index);
            }
            mixed += a.first * b.inverse_overlap_column[b.index]
                + b.first * a.inverse_overlap_column[a.index];
            let amplitude = -v / (a.gap + b.gap);
            dispersion += amplitude * v;
            exchange_dispersion += amplitude * (mixed - v);
        }
    }
    Ok(Terms {
        first_order,
        induction_with_exchange,
        dispersion,
        exchange_dispersion,
    })
}
