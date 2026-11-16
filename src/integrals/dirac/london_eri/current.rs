// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! London derivatives of the relativistic alpha-current interaction.

use super::{
    about_ket, accumulate, apply_phase, members, momentum_terms, orientations, AnalyticShell,
    BlochMatrix, Complex, DiracCoulombFields, GaussianKernel, HermiteQuartet, Jet, LondonQuartets,
    PairProducts, Primitive, Result, Scalar, SpinPair,
};
use crate::integrals::dirac::current::{
    current_pairs, CurrentPair, DiracCurrentInteraction, SYMMETRIC_INDEX,
};

const CURRENT_PAIRS: usize = 8;
const COEFFICIENTS: usize = 4 * CURRENT_PAIRS * CURRENT_PAIRS;
const I: Complex = Complex::new(0.0, 1.0);
const NEG_I: Complex = Complex::new(0.0, -1.0);

fn transition_polynomials(
    bra: &Primitive<Jet>,
    ket: &Primitive<Jet>,
    field: [Jet; 3],
    order: usize,
    phase_order: usize,
) -> [PairProducts; 6] {
    let phase = about_ket(bra, ket, field, order, phase_order);
    let left = momentum_terms(bra, field, -1.0);
    let right = momentum_terms(ket, field, 1.0);
    // Each alpha vertex has precisely one small component. Its spatial
    // basis is one component of the covariant momentum acting on the bra
    // or the ket. The spin matrices are applied after radial contraction.
    std::array::from_fn(|i| {
        let mut polynomial = PairProducts::new();
        if i < 3 {
            for &(angular, weight) in &left[i] {
                accumulate(&mut polynomial, [angular, ket.angular], weight);
            }
        } else {
            for &(angular, weight) in &right[i - 3] {
                accumulate(&mut polynomial, [bra.angular, angular], weight);
            }
        }
        apply_phase(polynomial, &phase)
    })
}

struct Radial {
    coulomb: [Jet; 4],
    distance: [[Jet; 4]; 6],
}

fn quartet(
    primitives: [Primitive<Jet>; 4],
    field: [Jet; 3],
    scale: f64,
    order: usize,
    phase_order: usize,
    interaction: DiracCurrentInteraction,
    pairs: &[CurrentPair],
) -> Result<[Jet; COEFFICIENTS]> {
    let [a, b, c, d] = primitives.each_ref();
    let raise = [1, phase_order + 1, 1, phase_order + 1];
    let total = 2 * phase_order + 2;
    let integrals = HermiteQuartet::new(
        primitives.each_ref(),
        GaussianKernel::Coulomb,
        0,
        raise,
        total,
    )?;
    let distance = (interaction == DiracCurrentInteraction::Breit)
        .then(|| HermiteQuartet::new_distance(primitives.each_ref(), 0, raise, total))
        .transpose()?;
    let left = transition_polynomials(a, b, field, order, phase_order)
        .map(|p| std::array::from_fn::<_, 2, _>(|channel| integrals.bra(members(&p, channel))));
    let right = transition_polynomials(c, d, field, order, phase_order)
        .map(|p| std::array::from_fn::<_, 2, _>(|channel| integrals.ket(members(&p, channel))));
    let zero = Jet::from_f64(0.0);
    let radial = left.each_ref().map(|[ar, ai]| {
        right.each_ref().map(|[br, bi]| {
            let coulomb = orientations([
                integrals.contract(ar, br),
                integrals.contract(ar, bi),
                integrals.contract(ai, br),
                integrals.contract(ai, bi),
            ]);
            // The Hermite pair polynomials are shared: only the radial
            // tensor changes. B_uv = -delta_uv/r + (d_u d_v r)/2.
            let distance = distance.as_ref().map_or([[zero; 4]; 6], |h| {
                let blocks = [
                    h.distance_hessian(ar, br),
                    h.distance_hessian(ar, bi),
                    h.distance_hessian(ai, br),
                    h.distance_hessian(ai, bi),
                ];
                std::array::from_fn(|k| orientations(blocks.map(|values| values[k])))
            });
            Radial { coulomb, distance }
        })
    });
    let mut values = [zero; COEFFICIENTS];
    for (i, p) in pairs.iter().enumerate() {
        for (j, q) in pairs.iter().enumerate() {
            let mut integral = [zero; 4];
            for &(u, x, p) in &p.terms {
                // Momentum polynomials already contain the +/-i factor in
                // the original field-free center-derivative representation.
                let p = p * if x < 3 { I } else { NEG_I };
                for &(v, y, q) in &q.terms {
                    let q = q * if y < 3 { I } else { NEG_I };
                    let r = &radial[x][y];
                    let kernel = |k: usize| {
                        let longitudinal = r.distance[SYMMETRIC_INDEX[u][v]][k].scale(0.5);
                        if u == v {
                            longitudinal - r.coulomb[k]
                        } else {
                            longitudinal
                        }
                    };
                    for (orientation, pair) in integral.chunks_exact_mut(2).enumerate() {
                        let z = (if orientation == 0 { p } else { p.conj() }) * q * scale;
                        let re: Jet = kernel(2 * orientation);
                        let im: Jet = kernel(2 * orientation + 1);
                        pair[0] = pair[0] + re.scale(z.re) - im.scale(z.im);
                        pair[1] = pair[1] + im.scale(z.re) + re.scale(z.im);
                    }
                }
            }
            let start = 4 * (i * CURRENT_PAIRS + j);
            values[start..start + 4].copy_from_slice(&integral);
        }
    }
    Ok(values)
}

/// Gaunt or frequency-independent Breit current coefficients for a fixed
/// London-RMB molecular basis. The scalar Coulomb interaction is separate.
pub struct LondonCurrentTaylor {
    quartets: LondonQuartets<CURRENT_PAIRS>,
    pairs: Vec<SpinPair>,
}

impl LondonCurrentTaylor {
    /// Retain coefficients through order two along a uniform magnetic path.
    /// A zero-bias phase expansion of this order is exact. At finite field,
    /// converge the phase order with respect to the size of the applied field.
    pub fn new(
        shells: &[AnalyticShell<f64>],
        speed: f64,
        interaction: DiracCurrentInteraction,
        field: [f64; 3],
        direction: [f64; 3],
        order: usize,
        phase_order: usize,
        memory_mb: usize,
    ) -> Result<Self> {
        let spin_pairs = current_pairs();
        let pairs = spin_pairs
            .iter()
            .enumerate()
            .map(|(i, pair)| {
                (
                    pair.row,
                    pair.column,
                    [(i, Complex::ONE), (i, Complex::ZERO)],
                )
            })
            .collect();
        let quartets = LondonQuartets::new(
            shells,
            speed,
            field,
            direction,
            order,
            phase_order,
            memory_mb,
            "London Gaunt/Breit Taylor quartets",
            |primitives, field, scale| {
                quartet(
                    primitives,
                    field,
                    scale,
                    order,
                    phase_order,
                    interaction,
                    &spin_pairs,
                )
            },
        )?;
        Ok(Self { quartets, pairs })
    }

    /// Apply a factorial-normalized coefficient to a Hermitian AO density.
    pub fn fields(&self, order: usize, density: &BlochMatrix) -> Result<DiracCoulombFields> {
        self.quartets.fields(order, density, &self.pairs)
    }
}
