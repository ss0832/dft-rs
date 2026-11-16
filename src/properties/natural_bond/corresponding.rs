// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Principal-angle analysis of the occupied alpha and beta spaces.

use crate::linalg::{symmetric_eigen, Matrix};
use crate::properties::natural_transition_orbitals::{decompose, real};
use crate::scf::ScfResult;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct CorrespondingOrbitals {
    pub paired_overlap: Vec<f64>,
    /// AO coefficients of the matched occupied alpha and beta orbitals.
    pub paired_coefficients: [Matrix; 2],
    /// Orthonormal AO complements in the original occupied spaces.
    pub unpaired_coefficients: [Matrix; 2],
    pub spin_squared: f64,
    pub cross_overlap_residual: f64,
}

fn occupied_complement(frame: &Matrix, paired: &Matrix, overlap: &Matrix) -> Result<Matrix> {
    let n = frame.cols;
    let remaining = n - paired.cols;
    if remaining == 0 {
        return Ok(Matrix::zeros(frame.rows, 0));
    }
    if paired.cols == 0 {
        return Ok(frame.clone());
    }
    let p = frame.transpose_matmul(&overlap.matmul(paired));
    let mut projector = Matrix::identity(n);
    let p = p.matmul_transpose_seq(&p);
    for (element, correction) in projector.as_mut_slice().iter_mut().zip(p.as_slice()) {
        *element -= correction;
    }
    let (_, frame_transform) = symmetric_eigen(&projector)?;
    let mut complement = Matrix::zeros(n, remaining);
    for i in 0..n {
        for j in 0..remaining {
            complement[(i, j)] = frame_transform[(i, paired.cols + j)];
        }
    }
    Ok(frame.matmul(&complement))
}

fn occupied_frame(coefficients: &Matrix, populations: &[f64]) -> Result<Matrix> {
    if populations.len() != coefficients.cols {
        return Err(DftError::InvalidInput(
            "corresponding-orbital occupations do not match the coefficient matrix".into(),
        ));
    }
    let occupied: Vec<_> = populations
        .iter()
        .enumerate()
        .filter_map(|(column, &population)| (population > 0.5).then_some(column))
        .collect();
    let mut frame = Matrix::zeros(coefficients.rows, occupied.len());
    for (column, original) in occupied.into_iter().enumerate() {
        for row in 0..coefficients.rows {
            frame[(row, column)] = coefficients[(row, original)];
        }
    }
    Ok(frame)
}

/// Pair the integer-occupied alpha and beta spaces of a converged real
/// determinant. Zero-overlap directions remain in the separate complements.
pub fn corresponding_orbitals(scf: &ScfResult) -> Result<CorrespondingOrbitals> {
    if !scf.converged || !scf.has_integer_occupations() || scf.electronic_temperature != 0.0 {
        return Err(DftError::InvalidInput(
            "corresponding orbitals require a converged integer-occupied collinear reference"
                .into(),
        ));
    }
    let frames = [
        real(occupied_frame(&scf.mo_coeff[0], &scf.occupations[0])?),
        real(occupied_frame(&scf.mo_coeff[1], &scf.occupations[1])?),
    ];
    let counts = [frames[0].cols(), frames[1].cols()];
    let overlap = &scf.overlap;
    let cross = frames[0]
        .re
        .transpose_matmul(&overlap.matmul(&frames[1].re));
    let paired = decompose(&real(cross), &frames[0], &frames[1])?;
    let matched = [paired.particle_coefficients.re, paired.hole_coefficients.re];
    let unpaired = [
        occupied_complement(&frames[0].re, &matched[0], overlap)?,
        occupied_complement(&frames[1].re, &matched[1], overlap)?,
    ];
    let largest = |matrix: &Matrix| {
        matrix
            .as_slice()
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()))
    };
    let mut represented = matched[0].transpose_matmul(&overlap.matmul(&matched[1]));
    for (i, &value) in paired.singular_values.iter().enumerate() {
        represented[(i, i)] -= value;
    }
    let mut defect = largest(&represented);
    for (alpha, beta) in [
        (&unpaired[0], &matched[1]),
        (&matched[0], &unpaired[1]),
        (&unpaired[0], &unpaired[1]),
    ] {
        defect = defect.max(largest(&alpha.transpose_matmul(&overlap.matmul(beta))));
    }
    let sz = 0.5 * (counts[0] as f64 - counts[1] as f64);
    let spin_squared = sz * sz + 0.5 * (counts[0] + counts[1]) as f64
        - paired
            .singular_values
            .iter()
            .map(|sigma| sigma * sigma)
            .sum::<f64>();
    Ok(CorrespondingOrbitals {
        paired_overlap: paired.singular_values,
        paired_coefficients: matched,
        unpaired_coefficients: unpaired,
        spin_squared,
        cross_overlap_residual: defect,
    })
}
