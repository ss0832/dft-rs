// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::matrix::{block, congruence, identity, product, root_and_inverse, validate_hermitian};
use super::x2c::{orthonormal_chart, rotate};
use super::{finish_embedding, spin_scalar_operator, RelativisticSpin, TwoComponentTaylor};
use crate::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use crate::periodic::BlochMatrix;
use crate::properties::no_pair::no_pair_projector_taylor;
use crate::{DftError, Result};

fn sum(a: &[BlochMatrix], b: &[BlochMatrix], scale: f64, order: usize) -> Vec<BlochMatrix> {
    (0..=order)
        .map(|k| {
            let mut value = a
                .get(k)
                .cloned()
                .unwrap_or_else(|| BlochMatrix::zeros(a[0].rows(), a[0].cols()));
            if let Some(v) = b.get(k) {
                value.add_scaled(v, scale);
            }
            value
        })
        .collect()
}

fn blocks(series: &[BlochMatrix], row: usize, col: usize, n: usize) -> Vec<BlochMatrix> {
    series.iter().map(|a| block(a, row, col, n, n)).collect()
}

fn adjoints(series: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
    series.iter().map(adjoint).collect()
}

fn rkb_contract(h: &[BlochMatrix], s: &[BlochMatrix], order: usize, cutoff: f64) -> Result<usize> {
    let n = h.first().map_or(0, BlochMatrix::rows);
    if n % 4 != 0 || order > 4 || !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(DftError::InvalidInput(
            "approximate two-component decoupling needs a four-component RKB basis, positive cutoff and order zero through four"
                .into(),
        ));
    }
    validate_hermitian(h, n, order)?;
    validate_hermitian(s, n, order)?;
    let large = n / 2;
    if s.iter().any(|matrix| {
        (0..large).any(|i| (large..n).any(|j| matrix.re[(i, j)] != 0.0 || matrix.im[(i, j)] != 0.0))
    }) {
        return Err(DftError::InvalidInput(
            "approximate two-component decoupling requires the block-diagonal RKB metric".into(),
        ));
    }
    Ok(large)
}

struct FreeSylvester {
    minus: Vec<BlochMatrix>,
    plus: Vec<BlochMatrix>,
    left: BlochMatrix,
    left_adjoint: BlochMatrix,
    right: BlochMatrix,
    right_adjoint: BlochMatrix,
    denominator: Vec<f64>,
}

impl FreeSylvester {
    fn new(minus: Vec<BlochMatrix>, plus: Vec<BlochMatrix>, cutoff: f64) -> Result<Self> {
        let n = plus[0].rows();
        let identity = identity(n);
        let left = generalized_orbitals(&minus[0], &identity, cutoff)?;
        let right = generalized_orbitals(&plus[0], &identity, cutoff)?;
        let mut denominator = Vec::with_capacity(n * n);
        for a in left.energies {
            for &i in &right.energies {
                let gap = a + i;
                if !gap.is_finite() || gap <= 0.0 {
                    return Err(DftError::LinearAlgebra(
                        "DKH2 free-sector Sylvester equation needs positive energy gaps".into(),
                    ));
                }
                denominator.push(gap);
            }
        }
        Ok(Self {
            minus,
            plus,
            left_adjoint: adjoint(&left.coefficients)?,
            right_adjoint: adjoint(&right.coefficients)?,
            left: left.coefficients,
            right: right.coefficients,
            denominator,
        })
    }

    fn solve(&self, rhs: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        let mut solution = Vec::with_capacity(rhs.len());
        for (k, value) in rhs.iter().enumerate() {
            let mut remaining = value.clone();
            for i in 1..=k {
                remaining.add_scaled(&matmul(&self.minus[i], &solution[k - i])?, -1.0);
                remaining.add_scaled(&matmul(&solution[k - i], &self.plus[i])?, -1.0);
            }
            let mut local = matmul(&self.left_adjoint, &matmul(&remaining, &self.right)?)?;
            for ((re, im), gap) in local
                .re
                .as_mut_slice()
                .iter_mut()
                .zip(local.im.as_mut_slice())
                .zip(&self.denominator)
            {
                *re /= gap;
                *im /= gap;
            }
            solution.push(matmul(&self.left, &matmul(&local, &self.right_adjoint)?)?);
        }
        Ok(solution)
    }
}

/// Second-order Douglas–Kroll–Hess core on a finite-nucleus RKB path.
///
/// The potential is the full balanced electron–nuclear operator, with
/// the electron rest energy excluded from the supplied Hamiltonian. The
/// free positive and negative sectors fix the initial Foldy–Wouthuysen
/// rotation. The core keeps terms through second order in this external
/// potential. Operators and density use its corresponding graph through
/// second order, with positive metric normalization. That normalization
/// preserves the particle number exactly in the chosen approximate space.
pub fn dkh2_taylor(
    hamiltonian: &[BlochMatrix],
    overlap: &[BlochMatrix],
    potential: &[BlochMatrix],
    order: usize,
    separator: f64,
    cutoff: f64,
    spin: RelativisticSpin,
) -> Result<TwoComponentTaylor> {
    let n = rkb_contract(hamiltonian, overlap, order, cutoff)?;
    validate_hermitian(potential, 2 * n, order)?;
    let v = potential
        .iter()
        .map(|a| match spin {
            RelativisticSpin::Scalar => spin_scalar_operator(a),
            RelativisticSpin::SpinOrbit => Ok(a.clone()),
        })
        .collect::<Result<Vec<_>>>()?;
    let h = sum(&sum(hamiltonian, potential, -1.0, order), &v, 1.0, order);
    let free = sum(hamiltonian, potential, -1.0, order);
    let projector = no_pair_projector_taylor(&free, overlap, order, separator, cutoff)?;
    if projector.electronic_dimension != n {
        return Err(DftError::InvalidInput(
            "DKH2 requires equal positive and negative free Dirac sectors".into(),
        ));
    }
    let (_, s_inverse_root) = root_and_inverse(overlap, order, cutoff)?;
    let inverse_metric = product(&s_inverse_root, &s_inverse_root, order)?;
    let minus_projector = sum(&inverse_metric, &projector.projector, -1.0, order);
    let plus = rotate(&projector.projector, overlap, 0, order, cutoff)?;
    let minus = rotate(&minus_projector, overlap, n, order, cutoff)?;
    let e_plus = congruence(&plus, &free, order)?;
    let mut e_minus = congruence(&minus, &free, order)?;
    for coefficient in &mut e_minus {
        coefficient.scale(-1.0);
    }
    let v_plus = congruence(&plus, &v, order)?;
    let v_minus = congruence(&minus, &v, order)?;
    let odd = product(&adjoints(&minus)?, &product(&v, &plus, order)?, order)?;
    let equation = FreeSylvester::new(e_minus, e_plus, cutoff)?;
    let first = equation.solve(&odd)?;
    let first_adjoint = adjoints(&first)?;
    let correction = sum(
        &product(&first_adjoint, &odd, order)?,
        &product(&adjoints(&odd)?, &first, order)?,
        1.0,
        order,
    );
    let effective = sum(
        &sum(&equation.plus, &v_plus, 1.0, order),
        &correction,
        0.5,
        order,
    );
    let next_odd = sum(
        &product(&v_minus, &first, order)?,
        &product(&first, &v_plus, order)?,
        -1.0,
        order,
    );
    let graph = sum(&first, &equation.solve(&next_odd)?, 1.0, order);
    let mut metric = product(&adjoints(&graph)?, &graph, order)?;
    metric[0].add_scaled(&identity(n), 1.0);
    let (_, normalization) = root_and_inverse(&metric, order, cutoff)?;
    let embedding = product(
        &sum(&plus, &product(&minus, &graph, order)?, 1.0, order),
        &normalization,
        order,
    )?;
    finish_embedding(embedding, effective, &h, overlap)
}

/// Finite-basis matrix ZORA from stationary elimination of the balanced
/// small coefficients with the electronic energy omitted from the denominator.
///
/// In the original AO chart X = (-H_SS)^(-1) H_SL and the ordinary
/// unscaled ZORA core is H_LL + H_LS X with the large-component metric.
/// The picture-changed operators use the normalized approximate restored
/// four-component spinors [I; X]. This is the finite-basis inverse, whose
/// value can differ from direct quadrature of the inverse local potential.
pub fn matrix_zora_taylor(
    hamiltonian: &[BlochMatrix],
    overlap: &[BlochMatrix],
    order: usize,
    cutoff: f64,
    spin: RelativisticSpin,
) -> Result<TwoComponentTaylor> {
    let n = rkb_contract(hamiltonian, overlap, order, cutoff)?;
    let scalar;
    let h = if spin == RelativisticSpin::Scalar {
        scalar = hamiltonian
            .iter()
            .map(spin_scalar_operator)
            .collect::<Result<Vec<_>>>()?;
        &scalar
    } else {
        hamiltonian
    };
    let mut denominator = blocks(h, n, n, n);
    for a in &mut denominator {
        a.scale(-1.0);
    }
    let (_, half) = root_and_inverse(&denominator, order, cutoff)?;
    let inverse = product(&half, &half, order)?;
    let graph = product(&inverse, &blocks(h, n, 0, n), order)?;
    let large_core = sum(
        &blocks(h, 0, 0, n),
        &product(&blocks(h, 0, n, n), &graph, order)?,
        1.0,
        order,
    );
    let large_chart = orthonormal_chart(overlap, 0, order, cutoff)?;
    let chart = blocks(&large_chart, 0, 0, n);
    let effective = congruence(&chart, &large_core, order)?;
    let bottom = product(&graph, &chart, order)?;
    let mut restored = large_chart;
    for (b, lower) in restored.iter_mut().zip(bottom) {
        b.re.as_mut_slice()[n * n..].copy_from_slice(lower.re.as_slice());
        b.im.as_mut_slice()[n * n..].copy_from_slice(lower.im.as_slice());
    }
    let metric = congruence(&restored, overlap, order)?;
    let (_, normalization) = root_and_inverse(&metric, order, cutoff)?;
    let embedding = product(&restored, &normalization, order)?;
    finish_embedding(embedding, effective, h, overlap)
}
