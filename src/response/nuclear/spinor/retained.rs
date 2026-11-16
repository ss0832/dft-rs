// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full complex spin-block Fock remainders through the shared physical
//! Coulomb/exchange contraction and moving semilocal density tangent.
use super::*;
use crate::linalg::AccurateSum;
use crate::response::nuclear::precision::{MatrixParts, Series};

impl RetainedComplexBackend for SpinorGeometry<'_> {
    fn retained_fock_coefficient(
        &self,
        high: &[Vec<Cm>; 2],
        low: &[Vec<Cm>; 2],
        order: usize,
    ) -> Result<([Cm; 2], [Cm; 2])> {
        let n = self.geometry.overlap[0].rows;
        let mut h = Cm::zeros(2 * n, 2 * n);
        let mut l = h.clone();
        for imaginary in [false, true] {
            let block = |s: usize, t: usize| -> Series {
                (0..=order)
                    .map(|k| {
                        let mut value = MatrixParts::zeros(n, n);
                        let (a, b) = if imaginary {
                            (&high[0][k].im, &low[0][k].im)
                        } else {
                            (&high[0][k].re, &low[0][k].re)
                        };
                        for i in 0..n {
                            for j in 0..n {
                                value.high[(i, j)] = a[(s * n + i, t * n + j)];
                                value.low[(i, j)] = b[(s * n + i, t * n + j)];
                            }
                        }
                        value
                    })
                    .collect()
            };
            let (_, diagonal) = self.geometry.evaluate_retained(
                &self.linear_functional,
                &[block(0, 0), block(1, 1)],
                order,
            )?;
            let (target, remainder) = if imaginary {
                (&mut h.im, &mut l.im)
            } else {
                (&mut h.re, &mut l.re)
            };
            let mut assign = |s: usize, t: usize, mut value: MatrixParts| {
                // The auxiliary spatial evaluation includes its core once.
                // Remove it in retained arithmetic; only the interaction
                // belongs to each independent spin block here.
                value.add_scaled(
                    &MatrixParts::new(self.geometry.core_coefficients()[order].clone()),
                    -1.0,
                );
                for i in 0..n {
                    for j in 0..n {
                        target[(s * n + i, t * n + j)] = value.high[(i, j)];
                        remainder[(s * n + i, t * n + j)] = value.low[(i, j)];
                    }
                }
            };
            for s in 0..2 {
                assign(s, s, diagonal[s][order].clone());
            }
            for (s, t) in [(0, 1), (1, 0)] {
                let q = block(s, t);
                let mut opposite = q.clone();
                for value in &mut opposite {
                    for part in [&mut value.high, &mut value.low] {
                        for x in part.as_mut_slice() {
                            *x = -*x;
                        }
                    }
                }
                let (_, field) = self.geometry.evaluate_retained(
                    &self.linear_functional,
                    &[q, opposite],
                    order,
                )?;
                assign(s, t, field[0][order].clone());
            }
        }
        let core = self.core(order);
        let nonlocal = self
            .nonlocal
            .as_ref()
            .map(|term| {
                let base = spin_diagonal_real_path(&high[0]);
                let remainder = spin_diagonal_real_path(&low[0]);
                let (_, potential) = term.evaluate(&[], &base, None, order)?;
                // The semilocal tangent below already uses the full
                // transition kernel, including its VV10 charge channel.
                // For an empty semilocal provider add that tangent here.
                let tangent = if self.functional.parts.is_empty() {
                    let (_, tangent) = term.evaluate(&[], &base, Some(&remainder), order)?;
                    Some(scalar(&tangent[0][order]))
                } else {
                    None
                };
                Ok::<_, DftError>((scalar(&potential[0][order]), tangent))
            })
            .transpose()?;
        let (xc, tangent, pair_tangent) = if self.functional.parts.is_empty() {
            (
                Cm::zeros(2 * n, 2 * n),
                Cm::zeros(2 * n, 2 * n),
                Cm::zeros(2 * n, 2 * n),
            )
        } else {
            let (_, xc) = self.semilocal(&high[0], order, true, true)?;
            // Reuse the same Hermitian/non-Hermitian tangent continuation
            // and angular acceptance as excited-state mixed kernels.
            let mut scale = 0.0_f64;
            for value in low[0]
                .iter()
                .flat_map(|p| p.re.as_slice().iter().chain(p.im.as_slice()))
            {
                if !value.is_finite() {
                    return Err(DftError::InvalidInput(
                        "non-finite retained spinor density tangent".into(),
                    ));
                }
                scale = scale.max(value.abs());
            }
            let (mut tangent, mut pair) = if scale == 0.0 {
                (Cm::zeros(2 * n, 2 * n), Cm::zeros(2 * n, 2 * n))
            } else {
                // Test angular convergence on a normalized independent
                // tangent; a sub-ulp input must not accept an unconverged
                // angular kernel solely because its amplitude is tiny.
                let normalized: Vec<_> = low[0]
                    .iter()
                    .cloned()
                    .map(|mut p| {
                        for part in [&mut p.re, &mut p.im] {
                            for x in part.as_mut_slice() {
                                *x /= scale;
                            }
                        }
                        p
                    })
                    .collect();
                let field = self.transition_interaction(&high[0], &normalized, order, None)?;
                (field[order].clone(), self.interaction(&normalized, order)?)
            };
            for value in [&mut tangent, &mut pair] {
                for part in [&mut value.re, &mut value.im] {
                    for x in part.as_mut_slice() {
                        *x *= scale;
                    }
                }
            }
            (xc[order].clone(), tangent, pair)
        };
        for (a, b, c, x, t, k, charge) in [
            (
                &mut h.re,
                &mut l.re,
                &core.re,
                &xc.re,
                &tangent.re,
                &pair_tangent.re,
                nonlocal.as_ref(),
            ),
            (
                &mut h.im,
                &mut l.im,
                &core.im,
                &xc.im,
                &tangent.im,
                &pair_tangent.im,
                None,
            ),
        ] {
            for i in 0..2 * n {
                for j in 0..2 * n {
                    let mut sum = AccurateSum::default();
                    for value in [
                        a[(i, j)],
                        b[(i, j)],
                        c[(i, j)],
                        x[(i, j)],
                        t[(i, j)],
                        -k[(i, j)],
                    ] {
                        sum.add_product(value, 1.0);
                    }
                    if let Some((potential, derivative)) = charge {
                        sum.add_product(potential.re[(i, j)], 1.0);
                        if let Some(derivative) = derivative {
                            sum.add_product(derivative.re[(i, j)], 1.0);
                        }
                    }
                    (a[(i, j)], b[(i, j)]) = sum.parts();
                }
            }
        }
        let zero = Cm::zeros(2 * n, 2 * n);
        Ok(([h, zero.clone()], [l, zero]))
    }
}
