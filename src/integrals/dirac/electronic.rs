// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Direct scalar two-electron Coulomb and exchange in an RKB basis.

use super::{
    constant, constant_shell, endpoint, kinetic_scalar_channels, shell_offsets, CenterJet,
};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{repulsion, AnalyticShell};
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::{DftError, Result};

pub struct DiracCoulombFields {
    /// Direct potential J and exchange potential K, both in Hartree.
    pub direct: BlochMatrix,
    pub exchange: BlochMatrix,
    pub direct_energy: f64,
    /// Includes the minus sign and factor one half in the HF energy.
    pub exchange_energy: f64,
    pub hermiticity_defect: f64,
}

const ONE: Complex = Complex::new(1.0, 0.0);
const IM: Complex = Complex::new(0.0, 1.0);
const NEG_IM: Complex = Complex::new(0.0, -1.0);
const NEG: Complex = Complex::new(-1.0, 0.0);
const ZERO: Complex = Complex::ZERO;

/// Pair spin rows/columns and coefficients of the five real spatial
/// channels (large scalar, small scalar, small i sigma_x/y/z).
pub(super) type SpinPair = (usize, usize, [(usize, Complex); 2]);
pub(super) const PAIRS: [SpinPair; 6] = [
    (0, 0, [(0, ONE), (0, ZERO)]),
    (1, 1, [(0, ONE), (0, ZERO)]),
    (2, 2, [(1, ONE), (4, IM)]),
    (3, 3, [(1, ONE), (4, NEG_IM)]),
    (2, 3, [(2, IM), (3, ONE)]),
    (3, 2, [(2, IM), (3, NEG)]),
];

fn channels<S: crate::dual::Scalar>(value: CenterJet<S>, scale: S) -> [S; 5] {
    let [scalar, x, y, z] = kinetic_scalar_channels(value, scale);
    [value.re.re, scalar, x, y, z]
}

fn entry(matrix: &BlochMatrix, row: usize, column: usize) -> Complex {
    Complex::new(matrix.re[(row, column)], matrix.im[(row, column)])
}

fn accumulate(matrix: &mut BlochMatrix, row: usize, column: usize, term: Complex) {
    matrix.re[(row, column)] += term.re;
    matrix.im[(row, column)] += term.im;
}

fn contract_eri(
    operators: &mut [BlochMatrix; 2],
    density: &BlochMatrix,
    [p, q, r, s]: [usize; 4],
    integral: Complex,
    exchange_pairs: bool,
) {
    let [direct, exchange] = operators;
    for [p, q, r, s] in [[p, q, r, s], [r, s, p, q]]
        .into_iter()
        .take(1 + usize::from(exchange_pairs))
    {
        accumulate(direct, p, q, integral * entry(density, s, r));
        accumulate(exchange, p, s, integral * entry(density, q, r));
    }
}

fn contract_quartet(
    operators: &mut [BlochMatrix; 2],
    density: &BlochMatrix,
    n: usize,
    [i, j, k, l]: [usize; 4],
    spatial: [[f64; 5]; 5],
    [reverse_left, reverse_right, exchange_pairs]: [bool; 3],
) {
    for (p, q, left) in PAIRS {
        let (p, q) = (p * n + i, q * n + j);
        for (r, s, right) in PAIRS {
            let (r, s) = (r * n + k, s * n + l);
            let mut integral = ZERO;
            let mut left_conjugated = ZERO;
            for (u, cu) in left {
                for (v, cv) in right {
                    let weight = cv * spatial[u][v];
                    integral += cu * weight;
                    left_conjugated += cu.conj() * weight;
                }
            }
            contract_orientations(
                operators,
                density,
                [p, q, r, s],
                integral,
                left_conjugated,
                [reverse_left, reverse_right, exchange_pairs],
            );
        }
    }
}

/// Share the spinor density/exchange contraction and its shell-pair
/// symmetries between scalar charge and current interactions.
pub(super) fn contract_orientations(
    operators: &mut [BlochMatrix; 2],
    density: &BlochMatrix,
    [p, q, r, s]: [usize; 4],
    integral: Complex,
    left_conjugated: Complex,
    [reverse_left, reverse_right, exchange_pairs]: [bool; 3],
) {
    contract_eri(operators, density, [p, q, r, s], integral, exchange_pairs);
    // Exchanging the full bra/ket indices of one transition density
    // conjugates only that density. Swapping both conjugates the ERI.
    if reverse_left {
        contract_eri(
            operators,
            density,
            [q, p, r, s],
            left_conjugated,
            exchange_pairs,
        );
    }
    if reverse_right {
        contract_eri(
            operators,
            density,
            [p, q, s, r],
            left_conjugated.conj(),
            exchange_pairs,
        );
    }
    if reverse_left && reverse_right {
        contract_eri(
            operators,
            density,
            [q, p, s, r],
            integral.conj(),
            exchange_pairs,
        );
    }
}

pub(super) fn shell_quartets(count: usize) -> impl Iterator<Item = [usize; 4]> {
    (0..count).flat_map(|a| {
        (0..=a).flat_map(move |b| {
            (0..=a).flat_map(move |c| (0..=if c == a { b } else { c }).map(move |d| [a, b, c, d]))
        })
    })
}

pub(super) fn validate_density(
    shells: &[AnalyticShell<f64>],
    density: &BlochMatrix,
    speed_of_light: f64,
) -> Result<(Vec<usize>, usize, usize)> {
    let offsets = shell_offsets(shells)?;
    let n = *offsets.last().unwrap();
    let dimension = n.checked_mul(4).ok_or_else(|| {
        DftError::InvalidInput("four-component two-electron dimension overflow".into())
    })?;
    if n == 0
        || !speed_of_light.is_finite()
        || speed_of_light <= 0.0
        || [
            density.re.rows,
            density.re.cols,
            density.im.rows,
            density.im.cols,
        ] != [dimension; 4]
        || density
            .re
            .as_slice()
            .iter()
            .chain(density.im.as_slice())
            .any(|v| !v.is_finite())
        || density.hermiticity_error() > 1e-10
    {
        return Err(DftError::InvalidInput(
            "four-component two-electron interaction requires a finite Hermitian density in a balanced basis and positive finite c".into(),
        ));
    }
    Ok((offsets, n, dimension))
}

/// Scalar Coulomb repulsion for a contravariant Hermitian density ordered
/// [large up, large down, small up, small down]. Each quartet retains
/// LL|LL, LL|SS, SS|LL and SS|SS. The Pauli terms include complex spin
/// mixing, and exchange retains the off-diagonal large/small density.
/// Each unordered shell-pair quartet is evaluated once and expanded by the
/// two charge-density conjugations and interchange of electron coordinates.
pub fn electronic_coulomb_exchange(
    shells: &[AnalyticShell<f64>],
    density: &BlochMatrix,
    speed_of_light: f64,
) -> Result<DiracCoulombFields> {
    electronic_screened_exchange(
        shells,
        density,
        speed_of_light,
        GaussianKernel::Coulomb,
        true,
    )
}

/// The same balanced pair contraction for full or range-separated Coulomb
/// interactions, retaining either the scalar or complete Pauli products.
pub(crate) fn electronic_screened_exchange(
    shells: &[AnalyticShell<f64>],
    density: &BlochMatrix,
    speed_of_light: f64,
    kernel: GaussianKernel<f64>,
    spin_orbit: bool,
) -> Result<DiracCoulombFields> {
    let (offsets, n, dimension) = validate_density(shells, density, speed_of_light)?;
    let mut operators = std::array::from_fn(|_| BlochMatrix::zeros(dimension, dimension));
    visit_quartets(shells, &offsets, speed_of_light, kernel, spin_orbit, |q| {
        contract_quartet(&mut operators, density, n, q.indices, q.spatial, q.symmetry);
    })?;
    finish_fields(operators, density)
}

struct BalancedQuartet {
    indices: [usize; 4],
    spatial: [[f64; 5]; 5],
    symmetry: [bool; 3],
}

fn visit_quartets(
    shells: &[AnalyticShell<f64>],
    offsets: &[usize],
    speed_of_light: f64,
    kernel: GaussianKernel<f64>,
    spin_orbit: bool,
    mut visit: impl FnMut(BalancedQuartet),
) -> Result<()> {
    let scale = (0.5 / speed_of_light).powi(2);
    let bra_left: Vec<_> = shells
        .iter()
        .map(|s| endpoint(&constant_shell(s), true))
        .collect();
    let ket_left: Vec<_> = shells
        .iter()
        .map(|s| endpoint(&constant_shell(s), false))
        .collect();
    let bra_right: Vec<_> = shells
        .iter()
        .map(|s| constant_shell(&endpoint(s, true)))
        .collect();
    let ket_right: Vec<_> = shells
        .iter()
        .map(|s| constant_shell(&endpoint(s, false)))
        .collect();
    for [a, b, c, d] in shell_quartets(shells.len()) {
        let quartet = repulsion(
            &bra_left[a],
            &ket_left[b],
            &bra_right[c],
            &ket_right[d],
            match kernel {
                GaussianKernel::Coulomb => GaussianKernel::Coulomb,
                GaussianKernel::LongRange { omega } => GaussianKernel::LongRange {
                    omega: constant(constant(omega)),
                },
                GaussianKernel::ShortRange { omega } => GaussianKernel::ShortRange {
                    omega: constant(constant(omega)),
                },
            },
            4,
        )?;
        // Equal shells or equal shell pairs already occur in
        // both orientations in the full component loops below.
        let symmetry = [a != b, c != d, (a, b) != (c, d)];
        let [na, nb, nc, nd] = quartet.shape;
        for ia in 0..na {
            for ib in 0..nb {
                for ic in 0..nc {
                    for id in 0..nd {
                        let index = ((ia * nb + ib) * nc + ic) * nd + id;
                        let mut spatial = channels(quartet.values[index], constant(scale))
                            .map(|pair| channels(pair, scale));
                        if !spin_orbit {
                            for (a, row) in spatial.iter_mut().enumerate() {
                                for (b, value) in row.iter_mut().enumerate() {
                                    if a >= 2 || b >= 2 {
                                        *value = 0.0;
                                    }
                                }
                            }
                        }
                        let indices = [
                            offsets[a] + ia,
                            offsets[b] + ib,
                            offsets[c] + ic,
                            offsets[d] + id,
                        ];
                        visit(BalancedQuartet {
                            indices,
                            spatial,
                            symmetry,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// Optional stationary cache of the same exact shell-quartet integrals
/// used by the streaming pair contraction. Density and Fock are never cached.
pub(crate) struct BalancedPairCache {
    n: usize,
    quartets: Vec<BalancedQuartet>,
    _memory: crate::resources::MemoryReservation,
}

impl BalancedPairCache {
    pub(crate) fn new(
        shells: &[AnalyticShell<f64>],
        speed: f64,
        kernel: GaussianKernel<f64>,
        spin_orbit: bool,
        memory_mb: usize,
    ) -> Result<Option<Self>> {
        let offsets = shell_offsets(shells)?;
        let n = *offsets.last().unwrap();
        let width = |shell| offsets[shell + 1] - offsets[shell];
        let operation = "molecular balanced pair cache";
        let (sum, square_sum) = (0..shells.len())
            .flat_map(|a| (0..=a).map(move |b| width(a) as u128 * width(b) as u128))
            .fold((0_u128, 0_u128), |(sum, square_sum), p| {
                (sum + p, square_sum + p * p)
            });
        let Some(count) = sum
            .checked_mul(sum)
            .and_then(|x| x.checked_add(square_sum))
            .and_then(|x| usize::try_from(x / 2).ok())
        else {
            return Ok(None);
        };
        let size = std::mem::size_of::<BalancedQuartet>();
        if count > memory_mb.saturating_mul(1024 * 1024) / size {
            return Ok(None);
        }
        let bytes = crate::resources::checked_bytes(&[count, size], operation)?;
        let Some(memory) = crate::resources::reserve_cache(bytes, operation)? else {
            return Ok(None);
        };
        let mut quartets = Vec::new();
        if quartets.try_reserve_exact(count).is_err() {
            return Ok(None);
        }
        visit_quartets(shells, &offsets, speed, kernel, spin_orbit, |q| {
            quartets.push(q)
        })?;
        Ok(Some(Self {
            n,
            quartets,
            _memory: memory,
        }))
    }

    pub(crate) fn evaluate(&self, density: &BlochMatrix) -> Result<DiracCoulombFields> {
        let dimension = 4 * self.n;
        let mut operators = std::array::from_fn(|_| BlochMatrix::zeros(dimension, dimension));
        for q in &self.quartets {
            contract_quartet(
                &mut operators,
                density,
                self.n,
                q.indices,
                q.spatial,
                q.symmetry,
            );
        }
        finish_fields(operators, density)
    }
}

pub(super) fn finish_fields(
    operators: [BlochMatrix; 2],
    density: &BlochMatrix,
) -> Result<DiracCoulombFields> {
    let [direct, exchange] = operators;
    let direct_energy = 0.5 * density.trace_product(&direct);
    let exchange_energy = -0.5 * density.trace_product(&exchange);
    let hermiticity_defect = direct.hermiticity_error().max(exchange.hermiticity_error());
    if !direct_energy.is_finite()
        || !exchange_energy.is_finite()
        || !hermiticity_defect.is_finite()
        || hermiticity_defect > 1e-10
    {
        return Err(DftError::LinearAlgebra(
            "four-component direct or exchange contraction is nonfinite or non-Hermitian".into(),
        ));
    }
    Ok(DiracCoulombFields {
        direct,
        exchange,
        direct_energy,
        exchange_energy,
        hermiticity_defect,
    })
}

#[cfg(test)]
mod tests;
