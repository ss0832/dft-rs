// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::matrix::{block, congruence, product, root_and_inverse};
use super::{finish_embedding, spin_scalar_operator, RelativisticSpin, TwoComponentTaylor};
use crate::periodic::BlochMatrix;
use crate::properties::no_pair::no_pair_projector_taylor;
use crate::{DftError, Result};

/// Exact one-electron decoupling through order four on the supplied RKB path.
///
/// If P is the contravariant positive-sector projector and J selects the
/// large AOs, C = J (J† S J)^(-1/2) embeds the orthonormal large AO chart.
/// The direct-rotation embedding is B = P S C (C† S P S C)^(-1/2).
/// Its large block has the positive polar orientation in that chart.
/// This construction differentiates the whole separated sector, including
/// negative-energy mixing, with no derivatives of individual eigenvectors.
pub fn x2c_taylor(
    hamiltonian: &[BlochMatrix],
    overlap: &[BlochMatrix],
    order: usize,
    separator: f64,
    cutoff: f64,
    spin: RelativisticSpin,
) -> Result<TwoComponentTaylor> {
    let spin_free;
    let h = if spin == RelativisticSpin::Scalar {
        spin_free = hamiltonian
            .iter()
            .map(spin_scalar_operator)
            .collect::<Result<Vec<_>>>()?;
        &spin_free
    } else {
        hamiltonian
    };
    let sector = no_pair_projector_taylor(h, overlap, order, separator, cutoff)?;
    let full = overlap[0].rows();
    let large = full / 2;
    if sector.electronic_dimension != large {
        return Err(DftError::InvalidInput(
            "X2C requires equal positive and negative Dirac sectors".into(),
        ));
    }
    let embedding = rotate(&sector.projector, overlap, 0, order, cutoff)?;
    let effective = congruence(&embedding, h, order)?;
    finish_embedding(embedding, effective, h, overlap)
}

/// Direct rotation of either Dirac sector from its original AO chart.
pub(super) fn rotate(
    projector: &[BlochMatrix],
    overlap: &[BlochMatrix],
    offset: usize,
    order: usize,
    cutoff: f64,
) -> Result<Vec<BlochMatrix>> {
    let chart = orthonormal_chart(overlap, offset, order, cutoff)?;
    let source = product(overlap, &chart, order)?;
    let projected = product(projector, &source, order)?;
    let projected_metric = congruence(&source, projector, order)?;
    let (_, inverse_root) = root_and_inverse(&projected_metric, order, cutoff)?;
    product(&projected, &inverse_root, order)
}

pub(super) fn orthonormal_chart(
    overlap: &[BlochMatrix],
    offset: usize,
    order: usize,
    cutoff: f64,
) -> Result<Vec<BlochMatrix>> {
    let full = overlap[0].rows();
    let n = full / 2;
    let metric: Vec<_> = overlap
        .iter()
        .map(|s| block(s, offset, offset, n, n))
        .collect();
    let (_, inverse_root) = root_and_inverse(&metric, order, cutoff)?;
    let start = offset * n;
    Ok(inverse_root
        .into_iter()
        .map(|root| {
            let mut c = BlochMatrix::zeros(full, n);
            c.re.as_mut_slice()[start..start + n * n].copy_from_slice(root.re.as_slice());
            c.im.as_mut_slice()[start..start + n * n].copy_from_slice(root.im.as_slice());
            c
        })
        .collect())
}
