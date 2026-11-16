// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving-grid contraction of an independent Hermitian density tangent.
use super::*;
use crate::periodic::spinor_xc::spinor_semilocal_point_directional_taylor;

#[cfg(test)]
#[path = "mixed_tests.rs"]
mod tests;

pub(in crate::response::nuclear::spinor) fn evaluate_linear(
    points: &[GridPointTaylor],
    functional: &ResolvedFunctional,
    density: &[Cm],
    tangent: &[Cm],
    order: usize,
    angular_order: usize,
) -> Result<Vec<Cm>> {
    if density.len() <= order || tangent.len() <= order {
        return Err(DftError::InvalidInput(
            "spinor mixed grid density paths are shorter than the requested order".into(),
        ));
    }
    let n = density[0].re.rows / 2;
    let pauli = |path: &[Cm]| {
        path[..=order]
            .iter()
            .map(|p| {
                pauli_components(&BlochMatrix {
                    re: p.re.clone(),
                    im: p.im.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()
    };
    let base = pauli(density)?;
    let direction = pauli(tangent)?;
    let zero = || vec![Cm::zeros(2 * n, 2 * n); order + 1];
    points
        .par_iter()
        .try_fold(zero, |mut out, point| {
            // Every AO derivative and moving Becke weight participates in both
            // descriptor maps and the final potential contraction.
            let coefficients = |v: &[f64]| {
                (0..=order)
                    .map(|k| v.get(k).copied().unwrap_or(0.0))
                    .collect::<Vec<_>>()
            };
            let aos: Vec<[Vec<f64>; 4]> = point
                .aos
                .iter()
                .map(|a| std::array::from_fn(|d| coefficients(&a[d])))
                .collect();
            let x = descriptors(&aos, &base, order);
            let y = descriptors(&aos, &direction, order);
            let xc = spinor_semilocal_point_directional_taylor(functional, &x, &y, angular_order)?;
            let weight = coefficients(&point.weight);
            let potentials =
                std::array::from_fn(|i| product(&weight, &xc.directional.gradient[i], order));
            let fields = potential_matrices(&aos, &potentials, order)?;
            for k in 0..=order {
                super::super::add(&mut out[k], &fields[k]);
            }
            Ok(out)
        })
        .try_reduce(zero, |mut a, b| {
            for k in 0..=order {
                super::super::add(&mut a[k], &b[k]);
            }
            Ok(a)
        })
}

/// Reference-path-bound local Hessian jets reused by all transition columns.
/// This is valid for semilocal providers only; nonlocal learned kernels have
/// separate grid-cross couplings and must never use this pointwise cache.
pub(in crate::response::nuclear::spinor) struct KernelCache<'a> {
    functional: &'a ResolvedFunctional,
    points: Vec<PointPath>,
    order: usize,
    active: usize,
    dimension: usize,
    levels: Vec<(usize, Vec<Vec<f64>>)>,
    first_rule: usize,
}
struct PointPath {
    aos: Vec<[Vec<f64>; 4]>,
    descriptors: [Vec<f64>; 32],
    weight: Vec<f64>,
}
impl<'a> KernelCache<'a> {
    pub(in crate::response::nuclear::spinor) fn new(
        functional: &'a ResolvedFunctional,
        points: &[GridPointTaylor],
        density: &[Cm],
        order: usize,
        memory_mb: usize,
        first_rule: usize,
    ) -> Result<Option<Self>> {
        if functional.parts.is_empty() || points.is_empty() {
            return Ok(None);
        }
        if density.len() <= order {
            return Err(DftError::InvalidInput(
                "mixed spinor cache needs the complete reference density path".into(),
            ));
        }
        let active: usize = match functional.family {
            Family::Lda => 4,
            Family::Gga => 16,
            Family::MetaGga => 32,
        };
        let width = order + 1;
        let vectors = points
            .iter()
            .try_fold(0usize, |sum, p| {
                sum.checked_add(
                    p.aos
                        .len()
                        .checked_mul(4)?
                        .checked_mul(std::mem::size_of::<Vec<f64>>() + 8 * width)?,
                )
            })
            .and_then(|v| {
                v.checked_add(
                    points
                        .len()
                        .checked_mul(std::mem::size_of::<PointPath>() + 33 * 8 * width)?,
                )
            });
        let persistent = active
            .checked_mul(active)
            .and_then(|v| v.checked_mul(width))
            .and_then(|v| v.checked_mul(8))
            .and_then(|v| v.checked_add(std::mem::size_of::<Vec<f64>>()))
            .and_then(|v| v.checked_mul(points.len()))
            .and_then(|v| v.checked_mul(2));
        let additional = vectors
            .and_then(|a| persistent.and_then(|b| a.checked_add(b)))
            .and_then(|v| v.checked_mul(2));
        // Conservative allocator-capacity/scratch allowance. If it cannot fit,
        // preserve the original direct algorithm rather than reject a model.
        if additional.is_none_or(|v| v.div_ceil(1024 * 1024) > memory_mb) {
            return Ok(None);
        }
        let base = density[..=order]
            .iter()
            .map(|p| {
                pauli_components(&BlochMatrix {
                    re: p.re.clone(),
                    im: p.im.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let coefficients = |v: &[f64]| {
            (0..=order)
                .map(|k| v.get(k).copied().unwrap_or(0.0))
                .collect::<Vec<_>>()
        };
        let paths = points
            .iter()
            .map(|p| {
                let aos: Vec<[Vec<f64>; 4]> = p
                    .aos
                    .iter()
                    .map(|a| std::array::from_fn(|d| coefficients(&a[d])))
                    .collect();
                let x = descriptors(&aos, &base, order);
                PointPath {
                    aos,
                    descriptors: x,
                    weight: coefficients(&p.weight),
                }
            })
            .collect();
        Ok(Some(Self {
            functional,
            points: paths,
            order,
            active,
            dimension: density[0].re.rows,
            levels: Vec::with_capacity(2),
            first_rule,
        }))
    }
    pub(in crate::response::nuclear::spinor) fn supports(&self, rule: usize) -> bool {
        rule == self.first_rule || rule == 2 * self.first_rule
    }
    pub(in crate::response::nuclear::spinor) fn apply(
        &mut self,
        tangent: &[Cm],
        rule: usize,
    ) -> Result<Vec<Cm>> {
        if !self.supports(rule) {
            return Err(DftError::InvalidInput(
                "mixed cache is restricted to the two universal refinement rules".into(),
            ));
        }
        if tangent.len() <= self.order {
            return Err(DftError::InvalidInput(
                "mixed spinor cache needs the complete tangent path".into(),
            ));
        }
        let width = self.order + 1;
        let index = match self.levels.iter().position(|(r, _)| *r == rule) {
            Some(i) => i,
            None => {
                let active = self.active;
                let order = self.order;
                let functional = self.functional;
                let level = self
                    .points
                    .par_iter()
                    .map(|p| {
                        let mut values = vec![0.0; active * active * width];
                        let mut unit = std::array::from_fn(|_| vec![0.0; width]);
                        for column in 0..active {
                            unit[column][0] = 1.0;
                            let xc = spinor_semilocal_point_directional_taylor(
                                functional,
                                &p.descriptors,
                                &unit,
                                rule,
                            )?;
                            unit[column][0] = 0.0;
                            for row in 0..active {
                                for k in 0..=order {
                                    values[(row * active + column) * width + k] =
                                        xc.directional.gradient[row][k];
                                }
                            }
                        }
                        Ok(values)
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.levels.push((rule, level));
                self.levels.len() - 1
            }
        };
        let direction = tangent[..=self.order]
            .iter()
            .map(|p| {
                pauli_components(&BlochMatrix {
                    re: p.re.clone(),
                    im: p.im.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let zero = || vec![Cm::zeros(self.dimension, self.dimension); width];
        let active = self.active;
        let order = self.order;
        self.points
            .par_iter()
            .zip(&self.levels[index].1)
            .try_fold(zero, |mut out, (p, kernel)| {
                let y = descriptors(&p.aos, &direction, order);
                if y.iter().flatten().any(|v| !v.is_finite()) {
                    return Err(DftError::InvalidInput(
                        "non-finite spinor descriptor tangent".into(),
                    ));
                }
                let mut gradient: [Vec<f64>; 32] = std::array::from_fn(|_| vec![0.0; width]);
                for row in 0..active {
                    for column in 0..active {
                        for k in 0..=order {
                            for m in 0..=k {
                                gradient[row][k] +=
                                    kernel[(row * active + column) * width + m] * y[column][k - m];
                            }
                        }
                    }
                    gradient[row] = product(&p.weight, &gradient[row], order);
                }
                let fields = potential_matrices(&p.aos, &gradient, order)?;
                for k in 0..=order {
                    super::super::add(&mut out[k], &fields[k]);
                }
                Ok(out)
            })
            .try_reduce(zero, |mut a, b| {
                for k in 0..=order {
                    super::super::add(&mut a[k], &b[k]);
                }
                Ok(a)
            })
    }
}
