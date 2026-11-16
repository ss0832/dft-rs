// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Covariant one-electron operators in the separated no-pair sector.
//!
//! For contravariant P, the covariant AO result is S P O P S. This retains
//! negative-sector and moving-metric response without choosing an electronic
//! eigenvector gauge. It is bare projection, not X2C decoupling, interacting
//! DKS, two-electron picture change or a conserved-current construction.
use super::{no_pair_projector_taylor, NoPairTaylor};
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

/// Projector diagnostics and factorial-normalized covariant S P O P S.
#[derive(Clone, Debug)]
pub struct NoPairOperatorTaylor {
    pub sector: NoPairTaylor,
    /// Operator coefficients retain the physical units of the supplied O.
    pub operator: Vec<BlochMatrix>,
}

fn product(a: &[BlochMatrix], b: &[BlochMatrix], order: usize) -> Vec<BlochMatrix> {
    let n = a[0].rows();
    let mut out = vec![BlochMatrix::zeros(n, n); order + 1];
    for k in 0..=order {
        for i in 0..=k {
            if let (Some(x), Some(y)) = (a.get(i), b.get(k - i)) {
                for r in 0..n {
                    for c in 0..n {
                        for q in 0..n {
                            out[k].re[(r, c)] +=
                                x.re[(r, q)] * y.re[(q, c)] - x.im[(r, q)] * y.im[(q, c)];
                            out[k].im[(r, c)] +=
                                x.re[(r, q)] * y.im[(q, c)] + x.im[(r, q)] * y.re[(q, c)];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Project a supplied Hermitian covariant operator through order zero to four.
///
/// Coefficients are factorial normalized; omitted higher O coefficients are
/// exactly zero. H/S satisfy the contracts of [`no_pair_projector_taylor`].
/// With P=C C^dagger and C^dagger S C=I, Q=S C(C^dagger O C)C^dagger S;
/// electronic vectors retain their original expectation values. Operator
/// generation, physical balance and units remain the provider's responsibility.
pub fn no_pair_operator_taylor(
    h: &[BlochMatrix],
    s: &[BlochMatrix],
    o: &[BlochMatrix],
    order: usize,
    separator: f64,
    cutoff: f64,
) -> Result<NoPairOperatorTaylor> {
    let sector = no_pair_projector_taylor(h, s, order, separator, cutoff)?;
    let n = h[0].rows();
    if o.is_empty()
        || o.len() > order + 1
        || o.iter().any(|a| {
            a.re.rows != n
                || a.re.cols != n
                || a.im.rows != n
                || a.im.cols != n
                || a.re
                    .as_slice()
                    .iter()
                    .chain(a.im.as_slice())
                    .any(|x| !x.is_finite())
                || a.hermiticity_error() > 1e-10
        })
    {
        return Err(DftError::InvalidInput(
            "projected operator needs finite matching Hermitian covariant coefficients".into(),
        ));
    }
    let sp = product(s, &sector.projector, order);
    let ps = product(&sector.projector, s, order);
    let operator = product(&product(&sp, o, order), &ps, order);
    if operator.iter().any(|a| {
        a.re.as_slice()
            .iter()
            .chain(a.im.as_slice())
            .any(|x| !x.is_finite())
    }) {
        return Err(DftError::LinearAlgebra(
            "projected operator coefficient overflow".into(),
        ));
    }
    Ok(NoPairOperatorTaylor { sector, operator })
}
