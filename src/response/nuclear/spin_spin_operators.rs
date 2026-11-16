// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving-basis two-electron spin-dipole operators. Spatial derivatives are
//! obtained by integration by parts, using independent translations of the
//! bra and ket charge distributions. Any outer Scalar derivatives, including
//! nuclear Taylor coefficients, remain analytic.
//!
//! The Coulomb distribution includes its contact term. It must be contracted
//! with the direct-minus-exchange pair density before interpreting the result
//! as a traceless electron spin-spin interaction. This operator alone is not
//! a complete zero-field-splitting model or a spin-orbit contribution.

use crate::dual::{DualN, Scalar};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{repulsion, AnalyticShell};
use crate::Result;

#[derive(Clone, Debug)]
pub struct SpinSpinIntegralBlock<S: Scalar> {
    /// Spherical AO dimensions, with the last shell index varying fastest.
    pub shape: [usize; 4],
    /// Tensor components in Cartesian row-major order, in inverse bohr cubed.
    pub values: Vec<[S; 9]>,
}

fn translated<S: Scalar>(
    shell: &AnalyticShell<S>,
    first_pair: bool,
) -> AnalyticShell<DualN<DualN<S, 3>, 3>> {
    type First<S> = DualN<S, 3>;
    type Both<S> = DualN<First<S>, 3>;
    let center = std::array::from_fn(|axis| {
        if first_pair {
            Both::constant(First::variable(shell.center[axis], axis))
        } else {
            Both::variable(First::constant(shell.center[axis]), axis)
        }
    });
    let constant = |value| Both::constant(First::constant(value));
    AnalyticShell {
        l: shell.l,
        center,
        exponents: shell.exponents.iter().copied().map(constant).collect(),
        coefficients: shell.coefficients.iter().copied().map(constant).collect(),
    }
}

/// Return integrals of `d_a(phi_mu phi_nu) / r12 d_b(phi_kappa phi_lambda)`.
/// `outer_order` is the maximum derivative order already carried by Scalar;
/// the two independent auxiliary translations require two additional orders.
pub fn spin_spin_integrals<S: Scalar>(
    a: &AnalyticShell<S>,
    b: &AnalyticShell<S>,
    c: &AnalyticShell<S>,
    d: &AnalyticShell<S>,
    outer_order: usize,
) -> Result<SpinSpinIntegralBlock<S>> {
    let order = outer_order.checked_add(2).ok_or_else(|| {
        crate::DftError::InvalidInput("spin-spin derivative order overflow".into())
    })?;
    let block = repulsion(
        &translated(a, true),
        &translated(b, true),
        &translated(c, false),
        &translated(d, false),
        GaussianKernel::Coulomb,
        order,
    )?;
    Ok(SpinSpinIntegralBlock {
        shape: block.shape,
        values: block
            .values
            .into_iter()
            .map(|value| std::array::from_fn(|index| value.im[index % 3].im[index / 3]))
            .collect(),
    })
}
