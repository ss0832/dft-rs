// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex-linear continuation of the moving physical density Hessian.
use super::*;

impl SpinorGeometry<'_> {
    pub(super) fn transition_interaction(
        &self,
        reference: &[Cm],
        transition: &[Cm],
        order: usize,
        mut cache: Option<&mut semilocal::KernelCache<'_>>,
    ) -> Result<Vec<Cm>> {
        let mut out = (0..=order)
            .map(|k| self.interaction(transition, k))
            .collect::<Result<Vec<_>>>()?;
        if let Some(nonlocal) = &self.nonlocal {
            let base = spin_diagonal_real_path(reference);
            let real = spin_diagonal_real_path(transition);
            let imaginary = spin_diagonal_path(transition, |p| &p.im);
            // The charge kernel extends complex-linearly to transition
            // densities. The spatial provider symmetrizes each AO channel.
            let (_, real) = nonlocal.evaluate(&[], &base, Some(&real), order)?;
            let (_, imaginary) = nonlocal.evaluate(&[], &base, Some(&imaginary), order)?;
            for k in 0..=order {
                add(
                    &mut out[k],
                    &Cm {
                        re: scalar(&real[0][k]).re,
                        im: scalar(&imaginary[0][k]).re,
                    },
                );
            }
        }
        if self.functional.parts.is_empty() {
            return Ok(out);
        }
        let n = transition[0].re.rows;
        let mut h = vec![Cm::zeros(n, n); order + 1];
        let mut a = h.clone();
        for k in 0..=order {
            for i in 0..n {
                for j in 0..n {
                    h[k].re[(i, j)] = 0.5 * (transition[k].re[(i, j)] + transition[k].re[(j, i)]);
                    h[k].im[(i, j)] = 0.5 * (transition[k].im[(i, j)] - transition[k].im[(j, i)]);
                    a[k].re[(i, j)] = 0.5 * (transition[k].im[(i, j)] + transition[k].im[(j, i)]);
                    a[k].im[(i, j)] = -0.5 * (transition[k].re[(i, j)] - transition[k].re[(j, i)]);
                }
            }
        }
        let mut evaluate = |rule| -> Result<Vec<Cm>> {
            let (mut vh, va) =
                if let Some(cached) = cache.as_deref_mut().filter(|c| c.supports(rule)) {
                    (cached.apply(&h, rule)?, cached.apply(&a, rule)?)
                } else {
                    (
                        semilocal::evaluate_linear(
                            self.geometry.grid_points(),
                            self.functional,
                            reference,
                            &h,
                            order,
                            rule,
                        )?,
                        semilocal::evaluate_linear(
                            self.geometry.grid_points(),
                            self.functional,
                            reference,
                            &a,
                            order,
                            rule,
                        )?,
                    )
                };
            for k in 0..=order {
                for i in 0..n {
                    for j in 0..n {
                        vh[k].re[(i, j)] -= va[k].im[(i, j)];
                        vh[k].im[(i, j)] += va[k].re[(i, j)];
                    }
                }
            }
            Ok(vh)
        };
        let mut rule = self.angular_order.min(32);
        let mut previous = evaluate(rule)?;
        loop {
            rule *= 2;
            let current = evaluate(rule)?;
            let error = current
                .iter()
                .zip(&previous)
                .map(|(a, b)| a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im)))
                .fold(0.0_f64, f64::max);
            self.angular_history
                .borrow_mut()
                .push([rule as f64, 0.0, error]);
            if error <= self.angular_tolerance {
                for k in 0..=order {
                    add(&mut out[k], &current[k]);
                }
                return Ok(out);
            }
            if rule == crate::periodic::spinor_xc::MAX_SPIN_ANGULAR_ORDER {
                return Err(DftError::ResponseNotConverged {
                    iterations: rule,
                    residual: error,
                });
            }
            previous = current;
        }
    }
}
