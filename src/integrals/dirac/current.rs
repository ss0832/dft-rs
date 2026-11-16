// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Gaunt and frequency-independent Breit operators in the field-free RKB basis.

use super::electronic::{
    contract_orientations, finish_fields, shell_quartets, validate_density, DiracCoulombFields,
};
use crate::dual::DualN;
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{repulsion, repulsion_distance_hessian, AnalyticShell};
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::Result;

const ZERO: Complex = Complex::ZERO;
const ONE: Complex = Complex::ONE;
const NEG: Complex = Complex::new(-1.0, 0.0);
const I: Complex = Complex::new(0.0, 1.0);
const NEG_I: Complex = Complex::new(0.0, -1.0);
const PAULI: [[[Complex; 2]; 2]; 3] = [
    [[ZERO, ONE], [ONE, ZERO]],
    [[ZERO, NEG_I], [I, ZERO]],
    [[ONE, ZERO], [ZERO, NEG]],
];
pub(super) const SYMMETRIC_INDEX: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];

/// Select only the correction to the scalar Coulomb interaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiracCurrentInteraction {
    /// Instantaneous magnetic term -alpha_1 dot alpha_2 / r12.
    Gaunt,
    /// Gaunt plus the zero-frequency gauge term.
    Breit,
}

// One inner derivative acts at either endpoint of the first electron's
// transition current, and one outer derivative does the same for the second.
type PairJet = DualN<f64, 6>;
type CurrentJet = DualN<PairJet, 6>;

fn constant(value: f64) -> CurrentJet {
    CurrentJet::constant(PairJet::constant(value))
}

fn current_shell(
    shell: &AnalyticShell<f64>,
    first_electron: bool,
    bra: bool,
) -> AnalyticShell<CurrentJet> {
    let offset = if bra { 0 } else { 3 };
    AnalyticShell {
        l: shell.l,
        center: std::array::from_fn(|axis| {
            let value = shell.center[axis];
            if first_electron {
                CurrentJet::constant(PairJet::variable(value, offset + axis))
            } else {
                CurrentJet::variable(PairJet::constant(value), offset + axis)
            }
        }),
        exponents: shell.exponents.iter().copied().map(constant).collect(),
        coefficients: shell.coefficients.iter().copied().map(constant).collect(),
    }
}

pub(super) struct CurrentPair {
    pub(super) row: usize,
    pub(super) column: usize,
    pub(super) terms: Vec<(usize, usize, Complex)>,
}

pub(super) fn current_pairs() -> Vec<CurrentPair> {
    let mut pairs = Vec::with_capacity(8);
    for small_on_bra in [false, true] {
        for p in 0..2 {
            for q in 0..2 {
                let mut terms = Vec::new();
                for (axis, alpha) in PAULI.iter().enumerate() {
                    for (d, sigma) in PAULI.iter().enumerate() {
                        // alpha is off diagonal in the large/small blocks.
                        // A ket small component is +i sigma.d_A/(2c) in
                        // center derivatives, and the bra gives its adjoint.
                        let (left, right, factor, offset) = if small_on_bra {
                            (sigma, alpha, NEG_I, 0)
                        } else {
                            (alpha, sigma, I, 3)
                        };
                        let z = factor * (left[p][0] * right[0][q] + left[p][1] * right[1][q]);
                        if z != ZERO {
                            terms.push((axis, offset + d, z));
                        }
                    }
                }
                let (row, column) = if small_on_bra { (2 + p, q) } else { (p, 2 + q) };
                pairs.push(CurrentPair { row, column, terms });
            }
        }
    }
    pairs
}

/// Contract the current-current correction for a Hermitian density in
/// [large up, large down, small up, small down] AO order. The returned J and K
/// include the sign of the selected interaction. The usual HF Fock uses J-K.
/// This is a fixed-basis operator. It carries no nuclear/field derivatives.
pub fn electronic_current_exchange(
    shells: &[AnalyticShell<f64>],
    density: &BlochMatrix,
    speed_of_light: f64,
    interaction: DiracCurrentInteraction,
) -> Result<DiracCoulombFields> {
    let (offsets, n, dimension) = validate_density(shells, density, speed_of_light)?;
    let scale = (0.5 / speed_of_light).powi(2);
    let shells = [0, 1, 2, 3].map(|position| {
        shells
            .iter()
            .map(|shell| current_shell(shell, position < 2, position % 2 == 0))
            .collect::<Vec<_>>()
    });
    let pairs = current_pairs();
    let mut operators = std::array::from_fn(|_| BlochMatrix::zeros(dimension, dimension));
    for [a, b, c, d] in shell_quartets(shells[0].len()) {
        let indices = [a, b, c, d];
        let [first, second, third, fourth] = std::array::from_fn(|i| &shells[i][indices[i]]);
        let gaunt = repulsion(first, second, third, fourth, GaussianKernel::Coulomb, 2)?;
        let gauge = if interaction == DiracCurrentInteraction::Breit {
            let (shape, hessian) = repulsion_distance_hessian([first, second, third, fourth], 2)?;
            debug_assert_eq!(shape, gaunt.shape);
            Some(hessian)
        } else {
            None
        };
        let [na, nb, nc, nd] = gaunt.shape;
        let symmetry = [a != b, c != d, (a, b) != (c, d)];
        for i in 0..na {
            for j in 0..nb {
                for k in 0..nc {
                    for l in 0..nd {
                        let local = ((i * nb + j) * nc + k) * nd + l;
                        let aos = [
                            offsets[a] + i,
                            offsets[b] + j,
                            offsets[c] + k,
                            offsets[d] + l,
                        ];
                        let gaunt = &gaunt.values[local];
                        let gauge = gauge.as_ref().map(|values| &values[local]);
                        for left in &pairs {
                            for right in &pairs {
                                let mut integral = ZERO;
                                let mut left_conjugated = ZERO;
                                for &(u, x, p) in &left.terms {
                                    for &(v, y, q) in &right.terms {
                                        // B_uv = -delta_uv/r + d_u d_v |r|/2.
                                        // The derivatives here are RKB basis
                                        // endpoint derivatives of these kernels.
                                        let mut radial =
                                            if u == v { -gaunt.im[y].im[x] } else { 0.0 };
                                        if let Some(hessian) = gauge {
                                            radial +=
                                                0.5 * hessian[SYMMETRIC_INDEX[u][v]].im[y].im[x];
                                        }
                                        let weight = q * (radial * scale);
                                        integral += p * weight;
                                        left_conjugated += p.conj() * weight;
                                    }
                                }
                                contract_orientations(
                                    &mut operators,
                                    density,
                                    [
                                        left.row * n + aos[0],
                                        left.column * n + aos[1],
                                        right.row * n + aos[2],
                                        right.column * n + aos[3],
                                    ],
                                    integral,
                                    left_conjugated,
                                    symmetry,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    finish_fields(operators, density)
}
