// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving real Gaussian quadrature coupled to every Pauli density block.
use super::{cm, Cm};
use crate::periodic::spinor_scf::{from_pauli, pauli_components};
use crate::periodic::spinor_xc::{
    lda_endpoint_taylor, spinor_semilocal_energy_taylor, spinor_semilocal_point,
    spinor_semilocal_point_kernel, spinor_semilocal_point_taylor, SpinorXcPoint,
    SpinorXcPointTaylor,
};
use crate::periodic::BlochMatrix;
use crate::response::nuclear::geometry::GridPointTaylor;
use crate::response::nuclear::{product, product_into};
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};
use rayon::prelude::*;

mod mixed;
pub(super) use mixed::{evaluate_linear, KernelCache};

struct PointCache {
    x: [f64; 32],
    energy: f64,
    gradient: [f64; 32],
    kernel: Vec<f64>,
    active: usize,
}
impl PointCache {
    fn taylor(&self, x: &[Vec<f64>; 32], order: usize, rule: usize) -> SpinorXcPointTaylor {
        // All paths share this exact base density and geometry. Differences
        // of floating-point contraction order are checked, never clipped.
        assert!((0..32).all(|a| (x[a][0] - self.x[a]).abs() <= 1e-10 * (1.0 + self.x[a].abs())));
        let mut out = SpinorXcPointTaylor {
            energy: vec![0.0; order + 1],
            gradient: std::array::from_fn(|_| vec![0.0; order + 1]),
            angular_order: rule,
        };
        out.energy[0] = self.energy;
        for a in 0..32 {
            out.gradient[a][0] = self.gradient[a];
            for k in 1..=order {
                out.energy[k] += self.gradient[a] * x[a][k];
            }
        }
        if !self.kernel.is_empty() && order > 0 {
            for a in 0..self.active {
                for b in 0..self.active {
                    let h = self.kernel[a * self.active + b];
                    out.gradient[a][1] += h * x[b][1];
                    if order >= 2 {
                        out.energy[2] += 0.5 * x[a][1] * h * x[b][1];
                    }
                }
            }
        }
        out
    }
}

/// Descriptor-space storage is independent of the fourth power of AO count.
/// Matrix-free application remains B^T H B on the actual grid.
pub(super) struct GridCache {
    points: Vec<PointCache>,
    pub(super) angular_history: Vec<[f64; 3]>,
    pub(super) rule: usize,
    pub(super) previous: Option<Box<GridCache>>,
    kernel: bool,
}
impl GridCache {
    pub(super) fn new(
        points: &[GridPointTaylor],
        functional: &ResolvedFunctional,
        reference: &Cm,
        initial_rule: usize,
        tolerance: f64,
        kernel: bool,
    ) -> Result<Self> {
        let components = pauli_components(&BlochMatrix {
            re: reference.re.clone(),
            im: reference.im.clone(),
        })?;
        let active = match functional.family {
            Family::Lda => 4,
            Family::Gga => 16,
            Family::MetaGga => 32,
        };
        let x: Vec<_> = points
            .iter()
            .map(|p| descriptors(&p.aos, std::slice::from_ref(&components), 0).map(|v| v[0]))
            .collect();
        let build = |rule| -> Result<Vec<PointCache>> {
            x.par_iter()
                .map(|x| {
                    let columns = if kernel { active } else { 1 };
                    let mut cache = PointCache {
                        x: *x,
                        energy: 0.0,
                        gradient: [0.0; 32],
                        kernel: if kernel {
                            vec![0.0; active * active]
                        } else {
                            Vec::new()
                        },
                        active,
                    };
                    if x[0] < x[1].hypot(x[2]).hypot(x[3]) {
                        return Err(DftError::InvalidInput(
                            "cached spinor density must be positive semidefinite".into(),
                        ));
                    }
                    if x[0] < crate::xc::RHO_CUTOFF {
                        return Ok(cache);
                    }
                    if functional.family == Family::Lda {
                        let mut complete = true;
                        for column in 0..columns {
                            let path = std::array::from_fn(|a| {
                                if kernel {
                                    vec![x[a], if a == column { 1.0 } else { 0.0 }]
                                } else {
                                    vec![x[a]]
                                }
                            });
                            let Some(value) = lda_endpoint_taylor(functional, &path, rule)? else {
                                complete = false;
                                break;
                            };
                            if column == 0 {
                                cache.energy = value.energy[0];
                                cache.gradient = std::array::from_fn(|a| value.gradient[a][0]);
                            }
                            if kernel {
                                for row in 0..active {
                                    cache.kernel[row * active + column] = value.gradient[row][1];
                                }
                            }
                        }
                        if complete {
                            return Ok(cache);
                        }
                    }
                    if !kernel {
                        let value = spinor_semilocal_point(
                            functional,
                            &SpinorXcPoint::from_descriptors(*x),
                            rule,
                        )?;
                        cache.energy = value.energy;
                        cache.gradient = value.gradient;
                        return Ok(cache);
                    }
                    if let Some(value) = spinor_semilocal_point_kernel(
                        functional,
                        &SpinorXcPoint::from_descriptors(*x),
                        rule,
                    )? {
                        cache.energy = value.energy;
                        cache.gradient = value.gradient;
                        cache.kernel = value.kernel;
                        return Ok(cache);
                    }
                    for column in 0..columns {
                        let path = std::array::from_fn(|a| {
                            if kernel {
                                vec![x[a], if a == column { 1.0 } else { 0.0 }]
                            } else {
                                vec![x[a]]
                            }
                        });
                        let value = spinor_semilocal_point_taylor(functional, &path, rule)?;
                        if column == 0 {
                            cache.energy = value.energy[0];
                            cache.gradient = std::array::from_fn(|a| value.gradient[a][0]);
                        }
                        if kernel {
                            for row in 0..active {
                                cache.kernel[row * active + column] = value.gradient[row][1];
                            }
                        }
                    }
                    Ok(cache)
                })
                .collect()
        };
        let mut rule = initial_rule.min(32);
        let mut previous = build(rule)?;
        let mut history = Vec::new();
        loop {
            rule *= 2;
            let current = build(rule)?;
            let mut energy = 0.0;
            let mut operator = 0.0;
            for ((a, b), point) in current.iter().zip(&previous).zip(points) {
                let phi = point
                    .aos
                    .iter()
                    .map(|p| p[0][0].abs())
                    .fold(0.0_f64, f64::max);
                let grad: [f64; 3] = std::array::from_fn(|d| {
                    point
                        .aos
                        .iter()
                        .map(|p| p[d + 1][0].abs())
                        .fold(0.0_f64, f64::max)
                });
                // Conservative maximum descriptor Jacobians for independent
                // Hermitian AO directions and spin/Pauli reconstruction.
                let bound: [f64; 32] = std::array::from_fn(|i| match i {
                    0..=3 => 4.0 * phi * phi,
                    4..=15 => 8.0 * phi * grad[(i - 4) % 3],
                    16..=19 => 2.0 * grad.iter().map(|g| g * g).sum::<f64>(),
                    _ => 4.0 * phi * grad[(i - 20) % 3],
                });
                let w = point.weight[0].abs();
                energy += w * (a.energy - b.energy).abs();
                let first: f64 = (0..active)
                    .map(|i| (a.gradient[i] - b.gradient[i]).abs() * bound[i])
                    .sum();
                let second: f64 = if kernel {
                    (0..active)
                        .map(|i| {
                            (0..active)
                                .map(|j| {
                                    (a.kernel[i * active + j] - b.kernel[i * active + j]).abs()
                                        * bound[i]
                                        * bound[j]
                                })
                                .sum::<f64>()
                        })
                        .sum()
                } else {
                    0.0
                };
                operator += w * (first + second);
            }
            history.push([rule as f64, energy, operator]);
            // These absolute descriptor bounds can remain conservative even
            // when the integrated operator has converged. Keep the final
            // pair of rules in that case: every actual energy/potential path
            // is independently compared before this cache is accepted.
            if (energy <= tolerance && operator <= tolerance)
                || rule == crate::periodic::spinor_xc::MAX_SPIN_ANGULAR_ORDER
            {
                return Ok(Self {
                    points: current,
                    angular_history: history,
                    rule,
                    kernel,
                    previous: Some(Box::new(Self {
                        points: previous,
                        angular_history: Vec::new(),
                        rule: rule / 2,
                        kernel,
                        previous: None,
                    })),
                });
            }
            previous = current;
        }
    }
}

fn descriptors(aos: &[[Vec<f64>; 4]], pauli: &[[BlochMatrix; 4]], order: usize) -> [Vec<f64>; 32] {
    let n = aos.len();
    let mut x: [Vec<f64>; 32] = std::array::from_fn(|_| vec![0.0; order + 1]);
    // Each AO pair overwrites the same four convolution buffers.
    let [mut pair, mut left, mut right, mut kinetic] =
        std::array::from_fn(|_| vec![0.0; order + 1]);
    for i in 0..n {
        for j in 0..n {
            product_into(&aos[i][0], &aos[j][0], &mut pair);
            for c in 0..4 {
                for k in 0..=order {
                    for m in 0..=k {
                        x[c][k] += pauli[m][c].re[(i, j)] * pair[k - m];
                    }
                }
            }
            for d in 0..3 {
                product_into(&aos[i][d + 1], &aos[j][0], &mut left);
                product_into(&aos[i][0], &aos[j][d + 1], &mut right);
                product_into(&aos[i][d + 1], &aos[j][d + 1], &mut kinetic);
                for c in 0..4 {
                    for k in 0..=order {
                        for m in 0..=k {
                            let (re, im) = (pauli[m][c].re[(i, j)], pauli[m][c].im[(i, j)]);
                            x[4 + 3 * c + d][k] += re * (left[k - m] + right[k - m]);
                            x[16 + c][k] += 0.5 * re * kinetic[k - m];
                            x[20 + 3 * c + d][k] += im * left[k - m];
                        }
                    }
                }
            }
        }
    }
    x
}

pub(super) fn evaluate(
    points: &[GridPointTaylor],
    functional: &ResolvedFunctional,
    density: &[Cm],
    order: usize,
    angular_order: usize,
    moving: bool,
    potentials: bool,
    cache: Option<&GridCache>,
) -> Result<(Vec<f64>, Vec<Cm>)> {
    let n = density[0].re.rows / 2;
    let pauli = density[..=order]
        .iter()
        .map(|p| {
            pauli_components(&BlochMatrix {
                re: p.re.clone(),
                im: p.im.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let zero = || {
        (
            vec![0.0; order + 1],
            vec![Cm::zeros(2 * n, 2 * n); order + 1],
        )
    };
    points
        .par_iter()
        .enumerate()
        .try_fold(zero, |(mut energy, mut fock), (index, point)| {
            let coefficients = |v: &[f64]| {
                (0..=order)
                    .map(|k| {
                        if k == 0 || moving {
                            v.get(k).copied().unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let weight = coefficients(&point.weight);
            let aos: Vec<[Vec<f64>; 4]> = point
                .aos
                .iter()
                .map(|a| std::array::from_fn(|d| coefficients(&a[d])))
                .collect();
            let x = descriptors(&aos, &pauli, order);
            let cached = cache
                .filter(|c| {
                    if potentials {
                        order == 0 || order == 1 && c.kernel
                    } else {
                        order <= 1 || order == 2 && c.kernel
                    }
                })
                .map(|c| c.points[index].taylor(&x, order, angular_order));
            if !potentials {
                let e = product(
                    &weight,
                    &match &cached {
                        Some(c) => c.energy.clone(),
                        None => match lda_endpoint_taylor(functional, &x, angular_order)? {
                            Some(exact) => exact.energy,
                            None => spinor_semilocal_energy_taylor(functional, &x, angular_order)?,
                        },
                    },
                    order,
                );
                for k in 0..=order {
                    energy[k] += e[k];
                }
                return Ok((energy, fock));
            }
            let xc = match cached {
                Some(c) => c,
                None => match lda_endpoint_taylor(functional, &x, angular_order)? {
                    Some(exact) => exact,
                    None => spinor_semilocal_point_taylor(functional, &x, angular_order)?,
                },
            };
            let e = product(&weight, &xc.energy, order);
            for k in 0..=order {
                energy[k] += e[k];
            }
            let potentials: [Vec<f64>; 32] =
                std::array::from_fn(|i| product(&weight, &xc.gradient[i], order));
            let fields = potential_matrices(&aos, &potentials, order)?;
            for k in 0..=order {
                super::add(&mut fock[k], &fields[k]);
            }
            Ok((energy, fock))
        })
        .try_reduce(zero, |(mut ea, mut va), (eb, vb)| {
            for k in 0..=order {
                ea[k] += eb[k];
                super::add(&mut va[k], &vb[k]);
            }
            Ok((ea, va))
        })
}

fn potential_matrices(
    aos: &[[Vec<f64>; 4]],
    potentials: &[Vec<f64>; 32],
    order: usize,
) -> Result<Vec<Cm>> {
    let n = aos.len();
    let mut v: Vec<[BlochMatrix; 4]> = (0..=order)
        .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(n, n)))
        .collect();
    let [mut pair, mut left, mut right, mut kinetic] =
        std::array::from_fn(|_| vec![0.0; order + 1]);
    for i in 0..n {
        for j in 0..n {
            product_into(&aos[i][0], &aos[j][0], &mut pair);
            for c in 0..4 {
                for k in 0..=order {
                    for m in 0..=k {
                        v[k][c].re[(i, j)] += 2.0 * potentials[c][m] * pair[k - m];
                    }
                }
            }
            for d in 0..3 {
                product_into(&aos[i][d + 1], &aos[j][0], &mut left);
                product_into(&aos[i][0], &aos[j][d + 1], &mut right);
                product_into(&aos[i][d + 1], &aos[j][d + 1], &mut kinetic);
                for c in 0..4 {
                    for k in 0..=order {
                        for m in 0..=k {
                            v[k][c].re[(i, j)] +=
                                2.0 * potentials[4 + 3 * c + d][m] * (left[k - m] + right[k - m])
                                    + potentials[16 + c][m] * kinetic[k - m];
                            v[k][c].im[(i, j)] +=
                                potentials[20 + 3 * c + d][m] * (left[k - m] - right[k - m]);
                        }
                    }
                }
            }
        }
    }
    v.iter().map(|v| from_pauli(v).map(|v| cm(&v))).collect()
}
