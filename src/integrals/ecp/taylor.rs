// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving-centre ECP Taylor coefficients from Gaussian centre recurrences.
//! The local and angular-projector integrals share the existing scalar
//! evaluator. Translational covariance eliminates operator-centre derivatives:
//! D=(v_A-v_C).d_A+(v_B-v_C).d_B. No displaced integral enters production.

use super::*;
use crate::{DftError, Result};
use std::collections::HashMap;

type Key = (u8, usize, u8, usize, usize);
type CentreDerivative = fn(u8, u8, usize, &Matrix, Option<&Matrix>) -> Matrix;

struct Recurrence<'a> {
    shells: [&'a MolShell; 2],
    center: Vec3,
    potential: &'a EcpPotential,
    motion: &'a [[[f64; 3]; 2]],
    // The cache is created inside one worker and never crosses a thread.
    cache: HashMap<Key, Matrix>,
}

impl Recurrence<'_> {
    fn weighted_shell(&self, side: usize, l: u8, power: usize) -> MolShell {
        let mut shell = self.shells[side].clone();
        shell.l = l;
        for (coefficient, &exponent) in shell.coefs.iter_mut().zip(&shell.exps) {
            *coefficient *= (2.0 * exponent).powi(power as i32);
        }
        shell
    }

    fn coefficient(&mut self, key: Key) -> Key {
        if self.cache.contains_key(&key) {
            return key;
        }
        let (la, pa, lb, pb, order) = key;
        let value = if order == 0 {
            ecp_shell_pair(
                &self.weighted_shell(0, la, pa),
                &self.weighted_shell(1, lb, pb),
                self.center,
                self.potential,
            )
        } else {
            let mut out = Matrix::zeros(cart_components(la).len(), cart_components(lb).len());
            // The chain rule for dU/dt is a convolution of the AO centre
            // derivatives with the polynomial velocity. Dividing the
            // coefficient of t^(n-1) by n yields the coefficient of U.
            for k in 1..=order.min(self.motion.len()) {
                let step = self.motion[k - 1];
                let remaining = order - k;
                for (side, motion) in step.into_iter().enumerate() {
                    if motion.iter().all(|&v| v == 0.0) {
                        continue;
                    }
                    let (raised, lowered, combine): (_, _, CentreDerivative) = if side == 0 {
                        let raised = self.coefficient((la + 1, pa + 1, lb, pb, remaining));
                        let lowered =
                            (la > 0).then(|| self.coefficient((la - 1, pa, lb, pb, remaining)));
                        (
                            raised,
                            lowered,
                            crate::integrals::deriv::combine_bra_derivative,
                        )
                    } else {
                        let raised = self.coefficient((la, pa, lb + 1, pb + 1, remaining));
                        let lowered =
                            (lb > 0).then(|| self.coefficient((la, pa, lb - 1, pb, remaining)));
                        (
                            raised,
                            lowered,
                            crate::integrals::deriv2::combine_ket_derivative,
                        )
                    };
                    for (axis, direction) in motion.into_iter().enumerate() {
                        if direction == 0.0 {
                            continue;
                        }
                        let block = combine(
                            la,
                            lb,
                            axis,
                            &self.cache[&raised],
                            lowered.map(|key| &self.cache[&key]),
                        );
                        let weight = direction * (k as f64 / order as f64);
                        for (a, b) in out.as_mut_slice().iter_mut().zip(block.as_slice()) {
                            *a += weight * b;
                        }
                    }
                }
            }
            out
        };
        self.cache.insert(key, value);
        key
    }
}

// Count exactly the reachable raising/lowering states across all requested
// orders. A lowering count cannot exceed the initial angular momentum plus
// preceding raises. The remaining derivative order supplies the final factor.
// This avoids reserving a doubled hash table for unreachable s-shell states.
fn cache_capacity(order: usize, angular: [u8; 2], moving: [bool; 2]) -> usize {
    let mut count = 0;
    for ra in 0..=order {
        for qa in 0..=order - ra {
            if (!moving[0] && ra + qa != 0) || qa > angular[0] as usize + ra {
                continue;
            }
            for rb in 0..=order - ra - qa {
                for qb in 0..=order - ra - qa - rb {
                    if (!moving[1] && rb + qb != 0) || qb > angular[1] as usize + rb {
                        continue;
                    }
                    count += order - ra - qa - rb - qb + 1;
                }
            }
        }
    }
    count
}

/// Accumulate one translated projector using the same factorial-normalized
/// centre recurrence for molecular and lattice paths.
fn accumulate_pair_taylor(
    cart: &mut [Matrix],
    shells: [&MolShell; 2],
    center: Vec3,
    potential: &EcpPotential,
    motion: &[[[f64; 3]; 2]],
) {
    let [sa, sb] = shells;
    let order = cart.len() - 1;
    let moving = std::array::from_fn(|side| {
        motion
            .iter()
            .any(|step| step[side].iter().any(|&v| v != 0.0))
    });
    if moving == [false; 2] {
        let block = ecp_shell_pair(sa, sb, center, potential);
        for (value, integral) in cart[0].as_mut_slice().iter_mut().zip(block.as_slice()) {
            *value += integral;
        }
        return;
    }
    let capacity = cache_capacity(order, [sa.l, sb.l], moving);
    let mut recurrence = Recurrence {
        shells: [sa, sb],
        center,
        potential,
        motion,
        cache: HashMap::with_capacity(capacity),
    };
    for (n, coefficient) in cart.iter_mut().enumerate() {
        let index = recurrence.coefficient((sa.l, 0, sb.l, 0, n));
        let block = &recurrence.cache[&index];
        for (x, y) in coefficient.as_mut_slice().iter_mut().zip(block.as_slice()) {
            *x += y;
        }
    }
}

/// Check the angular capacity before either second-derivative contraction.
pub(crate) fn check_ecp_second_order(
    shells: [&MolShell; 2],
    potential: &EcpPotential,
) -> Result<()> {
    check_pair_order(shells, potential, 2)
}

fn check_pair_order(shells: [&MolShell; 2], potential: &EcpPotential, order: usize) -> Result<()> {
    let [a, b] = shells;
    let maximum = usize::from(a.l.max(b.l));
    if usize::from(a.l) + usize::from(b.l) + order > LAMBDA_CAP
        || maximum + order + potential.semilocal.len().saturating_sub(1) > LAMBDA_CAP
    {
        return Err(DftError::Unsupported {
            feature: format!("ECP derivative order {order} beyond the scalar harmonic capacity"),
            missing: "a larger exact AO/projector angular table".into(),
        });
    }
    Ok(())
}

/// Coefficients of one AO shell pair with motion relative to the projector.
pub(crate) fn ecp_pair_path(
    shells: [&MolShell; 2],
    center: Vec3,
    potential: &EcpPotential,
    motion: &[[[f64; 3]; 2]],
) -> Result<Vec<Matrix>> {
    check_pair_order(shells, potential, motion.len())?;
    let [a, b] = shells;
    let mut cart = vec![
        Matrix::zeros(cart_components(a.l).len(), cart_components(b.l).len());
        motion.len() + 1
    ];
    accumulate_pair_taylor(&mut cart, shells, center, potential, motion);
    Ok(cart
        .iter()
        .map(|block| sph_block(a.l, b.l, block))
        .collect())
}

/// Spherical-AO coefficients U_n in U(t)=sum_n U_n t^n for a straight
/// nuclear path, through order four. Shell centres and each local/nonlocal
/// ECP centre move with their owning atom; rigid translation has identically
/// zero higher coefficients. The contractions retain primitive normalization.
pub fn ecp_nuclear_taylor(
    basis: &MolBasis,
    direction: &[[f64; 3]],
    order: usize,
) -> Result<Vec<Matrix>> {
    if order > 4
        || direction.iter().flatten().any(|v| !v.is_finite())
        || basis.shells.iter().any(|s| s.atom >= direction.len())
        || basis.ecps.iter().any(|p| p.atom >= direction.len())
    {
        return Err(DftError::InvalidInput(
            "ECP nuclear Taylor coefficients require finite per-atom directions and order 0..4"
                .into(),
        ));
    }
    if basis.ecps.is_empty() {
        return Ok(vec![Matrix::zeros(basis.nao_sph, basis.nao_sph); order + 1]);
    }
    if 2 * basis.max_l as usize + order > LAMBDA_CAP
        || basis.ecps.iter().any(|p| {
            basis.max_l as usize + order + p.potential.semilocal.len().saturating_sub(1)
                > LAMBDA_CAP
        })
    {
        return Err(DftError::Unsupported {
            feature: "ECP nuclear Taylor angular order beyond the scalar harmonic capacity".into(),
            missing:
                "a larger exact angular table for the requested AO/projector/derivative orders"
                    .into(),
        });
    }
    let pairs = (0..basis.shells.len())
        .flat_map(|a| (a..basis.shells.len()).map(move |b| (a, b)))
        .collect::<Vec<_>>();
    let blocks = pairs
        .par_iter()
        .map(|&(a, b)| {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            let mut cart =
                vec![
                    Matrix::zeros(cart_components(sa.l).len(), cart_components(sb.l).len());
                    order + 1
                ];
            for ecp in &basis.ecps {
                let velocity = std::array::from_fn(|side| {
                    std::array::from_fn(|axis| {
                        direction[[sa.atom, sb.atom][side]][axis] - direction[ecp.atom][axis]
                    })
                });
                accumulate_pair_taylor(
                    &mut cart,
                    [sa, sb],
                    ecp.center,
                    &ecp.potential,
                    &[velocity],
                );
            }
            cart.iter()
                .map(|m| sph_block(sa.l, sb.l, m))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut output = vec![Matrix::zeros(basis.nao_sph, basis.nao_sph); order + 1];
    for (&(a, b), blocks) in pairs.iter().zip(blocks) {
        let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
        for (n, block) in blocks.iter().enumerate() {
            for i in 0..block.rows {
                for j in 0..block.cols {
                    output[n][(sa.sph_off + i, sb.sph_off + j)] = block[(i, j)];
                    output[n][(sb.sph_off + j, sa.sph_off + i)] = block[(i, j)];
                }
            }
        }
    }
    if output
        .iter()
        .flat_map(|m| m.as_slice())
        .any(|v| !v.is_finite())
    {
        return Err(DftError::LinearAlgebra(
            "non-finite ECP nuclear Taylor coefficient".into(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn allocation_bound_counts_only_reachable_angular_states() {
        fn visit(key: Key, moving: [bool; 2], seen: &mut std::collections::HashSet<Key>) {
            if !seen.insert(key) || key.4 == 0 {
                return;
            }
            let (a, p, b, q, n) = key;
            if moving[0] {
                visit((a + 1, p + 1, b, q, n - 1), moving, seen);
                if a > 0 {
                    visit((a - 1, p, b, q, n - 1), moving, seen);
                }
            }
            if moving[1] {
                visit((a, p, b + 1, q + 1, n - 1), moving, seen);
                if b > 0 {
                    visit((a, p, b - 1, q, n - 1), moving, seen);
                }
            }
        }
        for a in 0..=4 {
            for b in 0..=4 {
                for order in 0..=4 {
                    for moving in [[false, false], [false, true], [true, false], [true, true]] {
                        let mut seen = std::collections::HashSet::new();
                        for n in 0..=order {
                            visit((a, 0, b, 0, n), moving, &mut seen);
                        }
                        assert_eq!(cache_capacity(order, [a, b], moving), seen.len());
                    }
                }
            }
        }
        assert_eq!(cache_capacity(4, [0, 0], [true, false]), 22);
        assert_eq!(cache_capacity(4, [0, 0], [true, true]), 58);
        assert_eq!(cache_capacity(4, [3, 3], [true, true]), 124);
    }
    use crate::system::{Atom, Molecule};

    #[test]
    fn recurrence_reuses_immutable_buffers_across_orders() {
        let molecule = Molecule::new(vec![
            Atom {
                z: 15,
                position: Vec3::new(0.17, -0.11, 0.23),
            },
            Atom {
                z: 1,
                position: Vec3::new(2.41, 0.296, -0.834),
            },
        ]);
        let basis =
            MolBasis::build(&molecule, &crate::basis::data::load("lanl2dz").unwrap()).unwrap();
        let left = &basis.shells[0];
        let right = basis.shells.iter().find(|s| s.atom == 1).unwrap();
        let potential = &basis.ecps[0];
        let motion = [[[0.11, 0.23, 0.37], [-0.14, 0.19, 0.07]]];
        let mut recurrence = Recurrence {
            shells: [left, right],
            center: potential.center,
            potential: &potential.potential,
            motion: &motion,
            cache: HashMap::with_capacity(cache_capacity(4, [left.l, right.l], [true; 2])),
        };
        let key = (left.l, 0, right.l, 0, 2);
        let initial = recurrence.coefficient(key);
        let values = recurrence.cache[&initial].as_slice().to_vec();
        let pointer = recurrence.cache[&initial].as_slice().as_ptr();
        let entries = recurrence.cache.len();
        let repeated = recurrence.coefficient(key);
        assert_eq!(initial, repeated);
        assert!(std::ptr::eq(
            pointer,
            recurrence.cache[&repeated].as_slice().as_ptr()
        ));
        assert_eq!(entries, recurrence.cache.len());
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|v| v.abs() > 1e-14));
        recurrence.coefficient((left.l, 0, right.l, 0, 4));
        let retained = recurrence.coefficient(key);
        assert_eq!(initial, retained);
        assert!(std::ptr::eq(
            pointer,
            recurrence.cache[&retained].as_slice().as_ptr()
        ));
        assert_eq!(values.as_slice(), recurrence.cache[&retained].as_slice());
        assert!(recurrence.cache.len() <= cache_capacity(4, [left.l, right.l], [true; 2]));
        let scalar = recurrence.coefficient((left.l, 0, right.l, 0, 0));
        let direct = ecp_shell_pair(left, right, potential.center, &potential.potential);
        assert_eq!(recurrence.cache[&scalar].as_slice(), direct.as_slice());
    }
}
