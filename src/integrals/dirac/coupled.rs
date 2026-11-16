// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Reciprocal scalar Coulomb fields for a balanced electron and quantum nucleus.

use super::{block, constant, constant_shell, endpoint, kinetic_scalar_product, shell_offsets};
use crate::dual::Scalar;
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{repulsion, AnalyticShell, IntegralBlock};
use crate::{DftError, Result};

/// Densities are contravariant in their respective AO bases. Electronic
/// matrices use [large up, large down, small up, small down] order.
pub struct BalancedCoulombInput<'a, S: Scalar> {
    pub electronic_shells: &'a [AnalyticShell<S>],
    pub nuclear_shells: &'a [AnalyticShell<S>],
    pub electronic_density_real: &'a IntegralBlock<S, 2>,
    pub electronic_density_imaginary: &'a IntegralBlock<S, 2>,
    pub nuclear_density: &'a IntegralBlock<S, 2>,
    pub speed_of_light: S,
    /// Charge of the distinguishable nucleus in positive elementary units.
    pub nuclear_charge: S,
    /// Intrinsic Gaussian charge radius in Bohr, in addition to the width
    /// arising from the nuclear wavefunction. None uses a point charge at
    /// each position of the quantum nucleus.
    pub intrinsic_rms_radius: Option<S>,
    pub order: usize,
}

pub struct BalancedCoulombFields<S: Scalar> {
    pub electronic_real: IntegralBlock<S, 2>,
    pub electronic_imaginary: IntegralBlock<S, 2>,
    pub nuclear: IntegralBlock<S, 2>,
    /// Interaction energy from the electronic contraction, in Hartree.
    pub energy: S,
    /// The independently accumulated contraction with the nuclear density.
    pub nuclear_contracted_energy: S,
    /// Signed electronic minus nuclear contraction, in Hartree.
    pub reciprocity_residual: S,
}

fn matrix_ok<S: Scalar>(matrix: &IntegralBlock<S, 2>, n: usize, transpose_sign: f64) -> bool {
    matrix.shape == [n, n]
        && n.checked_mul(n) == Some(matrix.values.len())
        && matrix.values.iter().all(|x| x.value().is_finite())
        && (0..n).all(|i| {
            (0..=i).all(|j| {
                let a = matrix.values[i * n + j].value();
                let b = matrix.values[j * n + i].value();
                (a - transpose_sign * b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0)
            })
        })
}

/// Electron–nucleus Coulomb interaction in the kinetically balanced basis.
/// The large and small scalar density products are included with their
/// spinor metric. Both fields are derivatives of one bilinear energy.
/// Each shell quartet is contracted once without storing a full pair tensor.
/// Outer analytic coefficients of centres, charges, widths and densities
/// pass through the same construction as the stationary fields.
pub fn electron_quantum_coulomb<S: Scalar>(
    input: BalancedCoulombInput<'_, S>,
) -> Result<BalancedCoulombFields<S>> {
    let positive = |x: S| x.value().is_finite() && x.value() > 0.0;
    if input.electronic_shells.is_empty()
        || input.nuclear_shells.is_empty()
        || !positive(input.speed_of_light)
        || !positive(input.nuclear_charge)
        || input.intrinsic_rms_radius.is_some_and(|r| !positive(r))
        || input.order > 32
    {
        return Err(DftError::InvalidInput("balanced Coulomb fields need nonempty bases, positive finite c, charge and optional intrinsic radius, and AD order at most 32".into()));
    }
    let e_offsets = shell_offsets(input.electronic_shells)?;
    let q_offsets = shell_offsets(input.nuclear_shells)?;
    let n = *e_offsets.last().unwrap();
    let four = n
        .checked_mul(4)
        .ok_or_else(|| DftError::InvalidInput("balanced Coulomb AO dimension overflow".into()))?;
    let q = *q_offsets.last().unwrap();
    let dr = input.electronic_density_real;
    let di = input.electronic_density_imaginary;
    let dq = input.nuclear_density;
    if !matrix_ok(dr, four, 1.0) || !matrix_ok(di, four, -1.0) || !matrix_ok(dq, q, 1.0) {
        return Err(DftError::InvalidInput("balanced Coulomb fields require finite Hermitian electronic and symmetric nuclear AO densities".into()));
    }
    let kernel = input
        .intrinsic_rms_radius
        .map_or(GaussianKernel::Coulomb, |radius| {
            GaussianKernel::LongRange {
                omega: constant(S::from_f64(1.5_f64.sqrt()) / radius),
            }
        });
    let scale = S::from_f64(0.25) / (input.speed_of_light * input.speed_of_light);
    let charge = -input.nuclear_charge;
    let left: Vec<_> = input
        .electronic_shells
        .iter()
        .map(|shell| endpoint(shell, true))
        .collect();
    let right: Vec<_> = input
        .electronic_shells
        .iter()
        .map(|shell| endpoint(shell, false))
        .collect();
    let nuclei: Vec<_> = input.nuclear_shells.iter().map(constant_shell).collect();
    let mut electronic_real = block(four, four)?;
    let mut electronic_imaginary = block(four, four)?;
    let mut nuclear = block(q, q)?;
    for (a, bra) in left.iter().enumerate() {
        for (b, ket) in right.iter().enumerate() {
            for (c, first) in nuclei.iter().enumerate() {
                for (d, second) in nuclei.iter().enumerate() {
                    let quartet = repulsion(bra, ket, first, second, kernel, input.order + 2)?;
                    let [na, nb, nc, nd] = quartet.shape;
                    for ia in 0..na {
                        let i = e_offsets[a] + ia;
                        for ib in 0..nb {
                            let j = e_offsets[b] + ib;
                            for ic in 0..nc {
                                for id in 0..nd {
                                    let qc = (q_offsets[c] + ic) * q + q_offsets[d] + id;
                                    let value =
                                        quartet.values[((ia * nb + ib) * nc + ic) * nd + id];
                                    let ll = charge * value.re.re;
                                    let mut contracted = S::from_f64(0.0);
                                    for spin in 0..2 {
                                        let p = (spin * n + i) * four + spin * n + j;
                                        electronic_real.values[p] =
                                            electronic_real.values[p] + dq.values[qc] * ll;
                                        contracted = contracted + dr.values[p] * ll;
                                    }
                                    for (spin_pair, (real, imag)) in
                                        kinetic_scalar_product(value, scale).into_iter().enumerate()
                                    {
                                        let p = ((2 + spin_pair / 2) * n + i) * four
                                            + (2 + spin_pair % 2) * n
                                            + j;
                                        let vr = charge * real;
                                        let vi = charge * imag;
                                        electronic_real.values[p] =
                                            electronic_real.values[p] + dq.values[qc] * vr;
                                        electronic_imaginary.values[p] =
                                            electronic_imaginary.values[p] + dq.values[qc] * vi;
                                        contracted =
                                            contracted + dr.values[p] * vr + di.values[p] * vi;
                                    }
                                    nuclear.values[qc] = nuclear.values[qc] + contracted;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let dot = |a: &IntegralBlock<S, 2>, b: &IntegralBlock<S, 2>| {
        a.values
            .iter()
            .zip(&b.values)
            .fold(S::from_f64(0.0), |sum, (&x, &y)| sum + x * y)
    };
    let energy = dot(dr, &electronic_real) + dot(di, &electronic_imaginary);
    let nuclear_contracted_energy = dot(dq, &nuclear);
    if !energy.value().is_finite()
        || !nuclear_contracted_energy.value().is_finite()
        || !matrix_ok(&electronic_real, four, 1.0)
        || !matrix_ok(&electronic_imaginary, four, -1.0)
        || !matrix_ok(&nuclear, q, 1.0)
    {
        return Err(DftError::InvalidInput(
            "balanced Coulomb contraction is nonfinite or non-Hermitian".into(),
        ));
    }
    Ok(BalancedCoulombFields {
        electronic_real,
        electronic_imaginary,
        nuclear,
        energy,
        nuclear_contracted_energy,
        reciprocity_residual: energy - nuclear_contracted_energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use crate::dual::DualN;
    use crate::integrals::dirac::{restricted_kinetic_balance, GaussianNucleus};
    use stencils::Stencil;

    const CENTRES: [f64; 9] = [-0.31, 0.26, 0.12, 1.19, -0.41, 0.53, 0.35, 1.27, -0.48];
    const C: f64 = 11.7;
    const Q: f64 = 1.3;
    const RADIUS: f64 = 0.081;
    const ALPHA: f64 = 1.19;
    const DR: [f64; 8] = [0.71, -0.36, 0.18, 0.28, -0.17, 0.21, 0.08, 0.12];
    const DI: [f64; 8] = [0.18, 0.24, -0.23, 0.19, 0.31, 0.04, -0.11, 0.28];

    fn density<S: Scalar>(imaginary: bool) -> IntegralBlock<S, 2> {
        let mut out = block(8, 8).unwrap();
        for i in 0..8 {
            for j in 0..8 {
                out.values[8 * i + j] = S::from_f64(if imaginary {
                    DI[i] * DR[j] - DR[i] * DI[j]
                } else {
                    DR[i] * DR[j] + DI[i] * DI[j]
                });
            }
        }
        out
    }

    fn shells<S: Scalar>(centres: [S; 9]) -> ([AnalyticShell<S>; 2], [AnalyticShell<S>; 1]) {
        let shell = |i: usize, exponent: f64| {
            AnalyticShell::normalized(
                0,
                std::array::from_fn(|a| centres[3 * i + a]),
                vec![S::from_f64(exponent)],
                vec![S::from_f64(1.0)],
            )
            .unwrap()
        };
        ([shell(0, 0.77), shell(1, 0.91)], [shell(2, ALPHA)])
    }

    fn coupling<S: Scalar>(centres: [S; 9], c: f64, order: usize) -> BalancedCoulombFields<S> {
        let (electronic_shells, nuclear_shells) = shells(centres);
        electron_quantum_coulomb(BalancedCoulombInput {
            electronic_shells: &electronic_shells,
            nuclear_shells: &nuclear_shells,
            electronic_density_real: &density(false),
            electronic_density_imaginary: &density(true),
            nuclear_density: &IntegralBlock {
                shape: [1, 1],
                values: vec![S::from_f64(1.0)],
            },
            speed_of_light: S::from_f64(c),
            nuclear_charge: S::from_f64(Q),
            intrinsic_rms_radius: Some(S::from_f64(RADIUS)),
            order,
        })
        .unwrap()
    }

    fn attraction_oracle(centres: [f64; 9]) -> (f64, f64) {
        let (electronic, _) = shells(centres);
        let charge = GaussianNucleus {
            center: [centres[6], centres[7], centres[8]],
            charge: Q,
            rms_radius: (RADIUS * RADIUS + 0.75 / ALPHA).sqrt(),
        };
        let fields = restricted_kinetic_balance(&electronic, &[charge], C, 0).unwrap();
        let re = density::<f64>(false);
        let im = density::<f64>(true);
        let mut large = 0.0;
        let mut small = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                if i / 2 == j / 2 {
                    large += re.values[8 * i + j] * fields.potential.values[2 * (i % 2) + j % 2];
                }
                let index = 8 * (i + 4) + j + 4;
                small += re.values[index] * fields.small_potential_real.values[4 * i + j]
                    + im.values[index] * fields.small_potential_imaginary.values[4 * i + j];
            }
        }
        (large, small)
    }

    #[test]
    fn balanced_quantum_coulomb_reciprocity_convolution_and_all_centre_fits() {
        let fields = coupling(CENTRES, C, 0);
        let (large, small) = attraction_oracle(CENTRES);
        assert!(small.abs() > 1e-6);
        assert!((fields.energy - large - small).abs() < 1e-13);
        assert!(fields.reciprocity_residual.abs() < 1e-13);
        assert!((coupling(CENTRES, 2.0 * C, 0).energy - large - small / 4.0).abs() < 1e-13);
        let ad = coupling(
            std::array::from_fn(|i| DualN::<f64, 9>::variable(CENTRES[i], i)),
            C,
            1,
        );
        let mixed = coupling(
            std::array::from_fn(|i| {
                if i < 6 {
                    DualN::<DualN<f64, 3>, 6>::variable(DualN::constant(CENTRES[i]), i)
                } else {
                    DualN::constant(DualN::variable(CENTRES[i], i - 6))
                }
            }),
            C,
            2,
        );
        for axis in 0..3 {
            assert!(
                (ad.energy.im[axis] + ad.energy.im[axis + 3] + ad.energy.im[axis + 6]).abs()
                    < 1e-12
            );
        }
        assert!(ad.reciprocity_residual.im.iter().all(|v| v.abs() < 1e-12));
        assert!(mixed
            .reciprocity_residual
            .im
            .iter()
            .flat_map(|v| v.im)
            .all(|v| v.abs() < 1e-12));
        println!("Balanced quantum Coulomb energy={:.17e} large={large:.17e} small={small:.17e} nuclear={:.17e} reciprocity={:.17e} gradient={:?} mixed={:?}",
            fields.energy, fields.nuclear_contracted_energy, fields.reciprocity_residual, ad.energy.im,
            mixed.energy.im.map(|dual| dual.im));
        let energy = |coordinates| {
            let (a, b) = attraction_oracle(coordinates);
            a + b
        };
        let names: Vec<_> = (0..9)
            .map(|i| format!("balanced Coulomb dR[{i}]"))
            .chain(
                (0..6)
                    .flat_map(|i| (0..3).map(move |j| format!("balanced Coulomb mixed[{i},{j}]"))),
            )
            .collect();
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.06, 0.04, 0.03, 0.02], Stencil::Second),
                ([0.03, 0.02, 0.01, 0.005], Stencil::Second),
            ],
            |steps, stencil| {
                let first = (0..9).map(|i| {
                    steps.map(|h| {
                        let mut plus = CENTRES;
                        let mut minus = CENTRES;
                        plus[i] += h;
                        minus[i] -= h;
                        (stencil.first(energy(plus), energy(minus), 0.0, 0.0, h) - ad.energy.im[i])
                            .abs()
                    })
                });
                let second = (0..6).flat_map(|i| {
                    (0..3).map(move |j| {
                        steps.map(|h| {
                            let sample = |x: f64, y: f64| {
                                let mut centres = CENTRES;
                                centres[i] += x * h;
                                centres[6 + j] += y * h;
                                energy(centres)
                            };
                            let fd = (sample(1.0, 1.0) - sample(1.0, -1.0) - sample(-1.0, 1.0)
                                + sample(-1.0, -1.0))
                                / (4.0 * h * h);
                            (fd - mixed.energy.im[i].im[j]).abs()
                        })
                    })
                });
                first.chain(second).collect()
            },
            &mut failures,
        );
        components::assert_no_failures(
            "balanced quantum Coulomb gradients and mixed geometry",
            &failures,
        );
    }
}
