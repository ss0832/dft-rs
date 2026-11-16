// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Coulomb and range-separated exchange coefficients in a London-RMB basis.

use super::electronic::{contract_orientations, finish_fields, shell_quartets, SpinPair, PAIRS};
use super::london::{half_cross, phase_coefficients};
use super::{shell_offsets, DiracCoulombFields};
use crate::dual::{Scalar, Taylor};
use crate::integrals::analytic::{GaussianKernel, HermiteQuartet, Primitive};
use crate::integrals::analytic_basis::{contracted_components, AnalyticShell};
use crate::periodic::BlochMatrix;
use crate::resources::{checked_bytes, reserve, MemoryReservation};
use crate::response::complex::Complex;
use crate::{DftError, Result};
use std::collections::BTreeMap;

type Jet = Taylor<3>;
type Angular = [usize; 3];
type ComplexJet = [Jet; 2];
type PairProducts = BTreeMap<[Angular; 2], ComplexJet>;
const NPAIR: usize = 5;
const NCHANNEL: usize = 4 * NPAIR * NPAIR;

mod current;
pub use current::LondonCurrentTaylor;

fn product([ar, ai]: ComplexJet, [br, bi]: ComplexJet) -> ComplexJet {
    [ar * br - ai * bi, ar * bi + ai * br]
}

fn empty(a: ComplexJet) -> bool {
    a.iter().all(|v| v.c.iter().all(|&c| c == 0.0))
}

fn raised(angular: Angular, axis: usize) -> Angular {
    let mut result = angular;
    result[axis] += 1;
    result
}

/// Polynomial terms of pi_i times one primitive. The left factor is
/// conjugated before its products with the right factor are formed.
fn momentum_terms(
    primitive: &Primitive<Jet>,
    field: [Jet; 3],
    sign: f64,
) -> [Vec<(Angular, ComplexJet)>; 3] {
    let zero = Jet::from_f64(0.0);
    let [bx, by, bz] = field.map(|f| f.scale(0.5));
    let vector = [[zero, -bz, by], [bz, zero, -bx], [-by, bx, zero]];
    std::array::from_fn(|i| {
        let angular = primitive.angular;
        let mut terms = vec![(
            raised(angular, i),
            [zero, primitive.exponent.scale(2.0 * sign)],
        )];
        if angular[i] > 0 {
            let mut lower = angular;
            lower[i] -= 1;
            terms.push((lower, [zero, Jet::from_f64(-sign * angular[i] as f64)]));
        }
        for (j, v) in vector[i].into_iter().enumerate() {
            if !empty([v, zero]) {
                terms.push((raised(angular, j), [v, zero]));
            }
        }
        terms
    })
}

fn accumulate(terms: &mut PairProducts, angular: [Angular; 2], [re, im]: ComplexJet) {
    let zero = Jet::from_f64(0.0);
    let target = terms.entry(angular).or_insert([zero; 2]);
    target[0] = target[0] + re;
    target[1] = target[1] + im;
}

fn about_ket(
    a: &Primitive<Jet>,
    b: &Primitive<Jet>,
    field: [Jet; 3],
    order: usize,
    phase_order: usize,
) -> Vec<(Angular, ComplexJet)> {
    let center = std::array::from_fn(|i| (a.center[i] + b.center[i]).scale(0.5));
    let wave = half_cross(field, std::array::from_fn(|i| a.center[i] - b.center[i]));
    let mut local = BTreeMap::new();
    // Integrate the same midpoint-centred plane-wave expansion used by
    // the one-electron provider. Re-expand the polynomial at the ket
    // centre for the Hermite recurrence. Reversing a pair then conjugates
    // exactly at each finite truncation.
    for ([x, y, z], coefficient) in phase_coefficients(wave, center, order, phase_order) {
        if empty(coefficient) {
            continue;
        }
        for i in 0..=x {
            for j in 0..=y {
                for k in 0..=z {
                    let base = [x, y, z];
                    let shift = [i, j, k];
                    let mut weight = Jet::from_f64(1.0);
                    for axis in 0..3 {
                        let mut choose = 1.0;
                        for m in 0..shift[axis] {
                            choose *= (base[axis] - m) as f64 / (m + 1) as f64;
                        }
                        weight = weight
                            * (b.center[axis] - center[axis])
                                .powi((base[axis] - shift[axis]) as i32)
                                .scale(choose);
                    }
                    let [re, im] = coefficient;
                    let zero = Jet::from_f64(0.0);
                    let target = local.entry(shift).or_insert([zero; 2]);
                    target[0] = target[0] + weight * re;
                    target[1] = target[1] + weight * im;
                }
            }
        }
    }
    local.into_iter().filter(|(_, w)| !empty(*w)).collect()
}

fn pair_products(
    a: &Primitive<Jet>,
    b: &Primitive<Jet>,
    field: [Jet; 3],
    inverse_4c2: f64,
    order: usize,
    phase_order: usize,
) -> [PairProducts; NPAIR] {
    let zero = Jet::from_f64(0.0);
    let phases = about_ket(a, b, field, order, phase_order);
    let left = momentum_terms(a, field, -1.0);
    let right = momentum_terms(b, field, 1.0);
    let mut unphased: [PairProducts; NPAIR] = std::array::from_fn(|_| BTreeMap::new());
    accumulate(
        &mut unphased[0],
        [a.angular, b.angular],
        [Jet::from_f64(1.0), zero],
    );
    let mut add = |channel: usize, i: usize, j: usize, sign: f64| {
        for (ap, av) in &left[i] {
            for (bp, bv) in &right[j] {
                accumulate(
                    &mut unphased[channel],
                    [*ap, *bp],
                    product(*av, *bv).map(|v| v.scale(sign * inverse_4c2)),
                );
            }
        }
    };
    for i in 0..3 {
        add(1, i, i, 1.0);
    }
    for (i, (j, k)) in [(1, 2), (2, 0), (0, 1)].into_iter().enumerate() {
        add(2 + i, j, k, 1.0);
        add(2 + i, k, j, -1.0);
    }
    unphased.map(|terms| apply_phase(terms, &phases))
}

fn apply_phase(terms: PairProducts, phases: &[(Angular, ComplexJet)]) -> PairProducts {
    let mut result = BTreeMap::new();
    for ([a, b], value) in terms {
        for &(phase, weight) in phases {
            let term = product(value, weight);
            if !empty(term) {
                accumulate(
                    &mut result,
                    [a, std::array::from_fn(|i| b[i] + phase[i])],
                    term,
                );
            }
        }
    }
    result
}

fn members(terms: &PairProducts, channel: usize) -> Vec<(Jet, Angular, Angular)> {
    terms
        .iter()
        .filter_map(|([a, b], coefficient)| {
            let weight = coefficient[channel];
            (!empty([weight, weight])).then_some((weight, *a, *b))
        })
        .collect()
}

/// Hermite contractions for both pair orientations, before spin factors.
fn orientations([rr, ri, ir, ii]: [Jet; 4]) -> [Jet; 4] {
    [rr - ii, ri + ir, rr + ii, ri - ir]
}

fn quartet(
    primitives: [Primitive<Jet>; 4],
    kernel: GaussianKernel<Jet>,
    field: [Jet; 3],
    inverse_4c2: f64,
    order: usize,
    phase_order: usize,
) -> Result<[Jet; NCHANNEL]> {
    let [a, b, c, d] = primitives.each_ref();
    let left = pair_products(a, b, field, inverse_4c2, order, phase_order);
    let right = pair_products(c, d, field, inverse_4c2, order, phase_order);
    let raise = [1, phase_order + 1, 1, phase_order + 1];
    let integrals =
        HermiteQuartet::new(primitives.each_ref(), kernel, 0, raise, 2 * phase_order + 4)?;
    let left = left.map(|p| std::array::from_fn::<_, 2, _>(|i| integrals.bra(members(&p, i))));
    let right = right.map(|p| std::array::from_fn::<_, 2, _>(|i| integrals.ket(members(&p, i))));
    let mut result = [Jet::from_f64(0.0); NCHANNEL];
    for (i, [ar, ai]) in left.iter().enumerate() {
        for (j, [br, bi]) in right.iter().enumerate() {
            let rr = integrals.contract(ar, br);
            let ri = integrals.contract(ar, bi);
            let ir = integrals.contract(ai, br);
            let ii = integrals.contract(ai, bi);
            // A single reversed pair conjugates only that pair. Store this
            // contraction independently of the ordinary oriented integral.
            let values = orientations([rr, ri, ir, ii]);
            let index = 4 * (i * NPAIR + j);
            result[index..index + 4].copy_from_slice(&values);
        }
    }
    Ok(result)
}

struct LondonQuartet<const N: usize> {
    indices: [usize; 4],
    symmetry: [bool; 3],
    coefficients: [[[[Complex; N]; N]; 2]; 3],
}

/// Shared retained AO-quartet and orientation contraction machinery.
struct LondonQuartets<const N: usize> {
    n: usize,
    order: usize,
    quartets: Vec<LondonQuartet<N>>,
    _workspace: MemoryReservation,
}

/// Fixed-geometry Taylor coefficients of scalar Coulomb and exchange in
/// a spin-orbit four-component uniformly magnetic basis. The same retained
/// coefficients act on each trial or response density.
pub struct LondonCoulombTaylor(LondonQuartets<NPAIR>);

impl LondonCoulombTaylor {
    /// Basis/field parameters and derivatives are in atomic units. The exact
    /// zero-bias phase Taylor requires phase_order >= order. At finite B
    /// converge the phase order separately. Coefficients include factorials.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shells: &[AnalyticShell<f64>],
        speed: f64,
        kernel: GaussianKernel<f64>,
        field: [f64; 3],
        direction: [f64; 3],
        order: usize,
        phase_order: usize,
        memory_mb: usize,
    ) -> Result<Self> {
        let kernel = match kernel {
            GaussianKernel::Coulomb => GaussianKernel::Coulomb,
            GaussianKernel::LongRange { omega } => GaussianKernel::LongRange {
                omega: Jet::from_f64(omega),
            },
            GaussianKernel::ShortRange { omega } => GaussianKernel::ShortRange {
                omega: Jet::from_f64(omega),
            },
        };
        Ok(Self(LondonQuartets::new(
            shells,
            speed,
            field,
            direction,
            order,
            phase_order,
            memory_mb,
            "London Coulomb Taylor quartets",
            |primitives, magnetic, scale| {
                quartet(primitives, kernel, magnetic, scale, order, phase_order)
            },
        )?))
    }

    /// Apply one Taylor coefficient to the supplied contravariant AO density.
    pub fn fields(&self, order: usize, density: &BlochMatrix) -> Result<DiracCoulombFields> {
        self.0.fields(order, density, &PAIRS)
    }
}

impl<const N: usize> LondonQuartets<N> {
    fn new<const C: usize>(
        shells: &[AnalyticShell<f64>],
        speed: f64,
        field: [f64; 3],
        direction: [f64; 3],
        order: usize,
        phase_order: usize,
        memory_mb: usize,
        operation: &'static str,
        primitive: impl Fn([Primitive<Jet>; 4], [Jet; 3], f64) -> Result<[Jet; C]>,
    ) -> Result<Self> {
        debug_assert_eq!(C, 4 * N * N);
        let offsets = shell_offsets(shells)?;
        let n = *offsets.last().unwrap();
        if n == 0
            || !speed.is_finite()
            || speed <= 0.0
            || order > 2
            || phase_order < order
            || phase_order > 8
            || field.iter().chain(&direction).any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "London interaction coefficients need a nonempty basis, positive c, finite magnetic path and orders 0..=2 with a phase order between the response order and 8".into(),
            ));
        }
        let count = shell_quartets(shells.len()).try_fold(0_usize, |sum, tuple| {
            let widths = tuple.map(|i| offsets[i + 1] - offsets[i]);
            let block = checked_bytes(&widths, operation)?;
            sum.checked_add(block)
                .ok_or_else(|| DftError::InvalidInput(format!("{operation} count overflow")))
        })?;
        let bytes = checked_bytes(&[count, std::mem::size_of::<LondonQuartet<N>>()], operation)?;
        if bytes > memory_mb.saturating_mul(1024 * 1024) {
            return Err(DftError::InvalidInput(format!(
                "{operation} need {:.1} MiB, above the {memory_mb} MiB integral budget",
                bytes as f64 / (1024.0 * 1024.0),
            )));
        }
        let workspace = reserve(bytes, operation)?;
        let mut quartets = Vec::new();
        quartets.try_reserve_exact(count).map_err(|_| {
            DftError::InvalidInput(format!("{operation} exceed available allocation"))
        })?;
        let shells: Vec<_> = shells
            .iter()
            .map(|s| AnalyticShell {
                l: s.l,
                center: s.center.map(Jet::from_f64),
                exponents: s.exponents.iter().copied().map(Jet::from_f64).collect(),
                coefficients: s.coefficients.iter().copied().map(Jet::from_f64).collect(),
            })
            .collect();
        let magnetic = std::array::from_fn(|i| Jet::from_coefficients(&[field[i], direction[i]]));
        let inverse_4c2 = (0.5 / speed).powi(2);
        for tuple @ [a, b, c, d] in shell_quartets(shells.len()) {
            let result = contracted_components(tuple.map(|i| &shells[i]), |primitives| {
                primitive(primitives, magnetic, inverse_4c2)
            })?;
            let shape = result[0].shape;
            let symmetry = [a != b, c != d, (a, b) != (c, d)];
            for index in 0..result[0].values.len() {
                let mut remainder = index;
                let mut indices = [0; 4];
                for s in (0..4).rev() {
                    indices[s] = offsets[tuple[s]] + remainder % shape[s];
                    remainder /= shape[s];
                }
                let mut coefficients = [[[[Complex::ZERO; N]; N]; 2]; 3];
                for (p, order_coefficients) in coefficients.iter_mut().enumerate().take(order + 1) {
                    for (orientation, matrix) in order_coefficients.iter_mut().enumerate() {
                        for (i, row) in matrix.iter_mut().enumerate() {
                            for (j, value) in row.iter_mut().enumerate() {
                                let offset = 4 * (i * N + j) + 2 * orientation;
                                *value = Complex::new(
                                    result[offset].values[index].c[p],
                                    result[offset + 1].values[index].c[p],
                                );
                            }
                        }
                    }
                }
                quartets.push(LondonQuartet {
                    indices,
                    symmetry,
                    coefficients,
                });
            }
        }
        Ok(Self {
            n,
            order,
            quartets,
            _workspace: workspace,
        })
    }

    fn fields(
        &self,
        order: usize,
        density: &BlochMatrix,
        pairs: &[SpinPair],
    ) -> Result<DiracCoulombFields> {
        let dim = 4 * self.n;
        if order > self.order
            || [
                density.re.rows,
                density.re.cols,
                density.im.rows,
                density.im.cols,
            ] != [dim; 4]
            || density
                .re
                .as_slice()
                .iter()
                .chain(density.im.as_slice())
                .any(|v| !v.is_finite())
            || density.hermiticity_error() > 1e-10
        {
            return Err(DftError::InvalidInput(
                "London interaction contraction needs a retained order and a finite Hermitian balanced density".into(),
            ));
        }
        let mut operators = std::array::from_fn(|_| BlochMatrix::zeros(dim, dim));
        for quartet in &self.quartets {
            let [a, b, c, d] = quartet.indices;
            let [forward, reversed] = &quartet.coefficients[order];
            for &(p, q, left) in pairs {
                for &(r, s, right) in pairs {
                    let mut integral = Complex::ZERO;
                    let mut inverse = Complex::ZERO;
                    for (i, x) in left {
                        for (j, y) in right {
                            integral += x * y * forward[i][j];
                            inverse += x.conj() * y * reversed[i][j];
                        }
                    }
                    contract_orientations(
                        &mut operators,
                        density,
                        [
                            p * self.n + a,
                            q * self.n + b,
                            r * self.n + c,
                            s * self.n + d,
                        ],
                        integral,
                        inverse,
                        quartet.symmetry,
                    );
                }
            }
        }
        finish_fields(operators, density)
    }
}
