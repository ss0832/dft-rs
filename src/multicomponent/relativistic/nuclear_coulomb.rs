// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::BalancedQuantumNucleus;
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::repulsion;
use crate::linalg::Matrix;
use crate::{DftError, Result};

pub(super) struct NuclearPairFields {
    pub first: Matrix,
    pub second: Matrix,
    pub energy: f64,
    pub reciprocity_defect: f64,
}

/// Evaluate both potentials of one unordered pair from the same quartets.
/// Independent Gaussian charge smearing adds the two intrinsic variances.
pub(super) fn pair(
    [first, second]: [&BalancedQuantumNucleus<'_>; 2],
    [left, right]: [&Matrix; 2],
    [left_offsets, right_offsets]: [&[usize]; 2],
) -> Result<NuclearPairFields> {
    let radius = first
        .intrinsic_rms_radius
        .unwrap_or(0.0)
        .hypot(second.intrinsic_rms_radius.unwrap_or(0.0));
    let kernel = if radius == 0.0 {
        GaussianKernel::Coulomb
    } else {
        GaussianKernel::LongRange {
            omega: 1.5_f64.sqrt() / radius,
        }
    };
    let charge = first.charge * second.charge;
    let mut v_first = Matrix::zeros(left.rows, left.cols);
    let mut v_second = Matrix::zeros(right.rows, right.cols);
    for (a, bra_first) in first.shells.iter().enumerate() {
        for (b, ket_first) in first.shells.iter().enumerate() {
            for (c, bra_second) in second.shells.iter().enumerate() {
                for (d, ket_second) in second.shells.iter().enumerate() {
                    let g = repulsion(bra_first, ket_first, bra_second, ket_second, kernel, 0)?;
                    let [na, nb, nc, nd] = g.shape;
                    for i in 0..na {
                        for j in 0..nb {
                            let p = (left_offsets[a] + i, left_offsets[b] + j);
                            for k in 0..nc {
                                for l in 0..nd {
                                    let q = (right_offsets[c] + k, right_offsets[d] + l);
                                    let integral =
                                        charge * g.values[((i * nb + j) * nc + k) * nd + l];
                                    v_first[p] += integral * right[q];
                                    v_second[q] += integral * left[p];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let energy = left.frobenius_dot(&v_first);
    let reverse = right.frobenius_dot(&v_second);
    if !energy.is_finite() || !reverse.is_finite() {
        return Err(DftError::LinearAlgebra(
            "nonfinite quantum nuclear Coulomb pair energy".into(),
        ));
    }
    Ok(NuclearPairFields {
        first: v_first,
        second: v_second,
        energy,
        reciprocity_defect: (energy - reverse).abs(),
    })
}
