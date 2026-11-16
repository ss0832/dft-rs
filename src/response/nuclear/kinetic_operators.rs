// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving kinetic-energy matrices for nuclear property contractions. The
//! basis exponents and normalized contraction coefficients are held fixed;
//! both Gaussian centres follow the supplied nuclear direction.

use crate::basis::{n_sph, MolBasis};
use crate::dual::Taylor;
use crate::integrals::analytic_basis::{kinetic, AnalyticShell};
use crate::linalg::Matrix;
use crate::{DftError, Result};

/// Coefficient k is the nuclear path derivative divided by k!, in Hartree
/// per bohr^k. This supplies only an operator, not a complete g-tensor model.
pub fn nuclear_kinetic_operator_path(
    basis: &MolBasis,
    direction: &[[f64; 3]],
    order: usize,
) -> Result<Vec<Matrix>> {
    if order > 2 {
        return Err(DftError::Unsupported {
            feature: "kinetic nuclear operator order above two".into(),
            missing: "higher-order kinetic-operator acceptance".into(),
        });
    }
    let n = basis.nao_sph;
    if n == 0
        || direction.iter().flatten().any(|value| !value.is_finite())
        || basis.shells.iter().any(|shell| {
            shell.atom >= direction.len()
                || shell
                    .sph_off
                    .checked_add(n_sph(shell.l))
                    .is_none_or(|end| end > n)
        })
    {
        return Err(DftError::InvalidInput(
            "moving kinetic operator requires a complete AO layout and finite nuclear directions"
                .into(),
        ));
    }
    let shells = basis
        .shells
        .iter()
        .map(|shell| {
            let center = std::array::from_fn(|axis| {
                Taylor::<3>::from_coefficients(&[
                    shell.center.get(axis),
                    direction[shell.atom][axis],
                ])
            });
            AnalyticShell::from_placed(shell, center)
        })
        .collect::<Result<Vec<_>>>()?;
    let max_shell = basis
        .shells
        .iter()
        .map(|shell| n_sph(shell.l))
        .max()
        .ok_or_else(|| DftError::InvalidInput("empty kinetic shell space".into()))?;
    let bytes = crate::resources::checked_bytes(
        &[max_shell, max_shell, std::mem::size_of::<Taylor<3>>()],
        "moving kinetic output block",
    )?;
    let _block_workspace = crate::resources::reserve(bytes, "moving kinetic output block")?;
    // Matrix storage is accounted separately; internal Gaussian scratch is
    // not included in this output-block reservation or an RSS guarantee.
    let mut result = vec![Matrix::zeros(n, n); order + 1];
    for (ia, a) in shells.iter().enumerate() {
        for (ib, b) in shells.iter().enumerate() {
            let block = kinetic(a, b)?;
            for i in 0..block.shape[0] {
                for j in 0..block.shape[1] {
                    let value = block.values[i * block.shape[1] + j];
                    for degree in 0..=order {
                        result[degree]
                            [(basis.shells[ia].sph_off + i, basis.shells[ib].sph_off + j)] =
                            value.c[degree];
                    }
                }
            }
        }
    }
    Ok(result)
}
