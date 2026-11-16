// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Covariant exact-exchange action between independent (k+q,k) AO spaces.
//! The same distance-selected quartets and Born-von Karman pair weights as
//! reference SCF are used. A finite-q transition density is not Hermitian.

use crate::basis::MolBasis;
use crate::integrals::{
    eri::{eri_quartet, Quartet as EriQuartet},
    CoulombKernel,
};
use crate::math::Vec3;
use crate::periodic::exchange::{born_von_karman_periods, minimum_image_weight};
use crate::periodic::exchange_quartets::ExchangeCells;
use crate::periodic::{BlochMatrix, Cell};
use crate::response::complex::Complex;
use crate::{DftError, Result};
use rayon::prelude::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

fn phase(angle: f64) -> Complex {
    Complex::new(angle.cos(), angle.sin())
}

const DEFAULT_INTEGRAL_CACHE_BYTES: usize = 32 * 1024 * 1024;

struct IntegralCache {
    targets: Vec<Vec<EriQuartet>>,
    storage_bound: usize,
}

/// Matrix-free exchange-density derivative without its functional coefficient.
/// Output K_q(k) is contracted with -cx or -clr by the common density solver.
pub struct FiniteQExchange {
    basis: MolBasis,
    cells: ExchangeCells,
    points: Vec<Vec3>,
    transfer: Vec3,
    kernel: CoulombKernel,
    /// [image][bra shell][ket shell], the same atom-pair weights as SCF.
    weights: Vec<Vec<f64>>,
    /// This operator owns immutable copies of its basis, translations, mesh,
    /// transfer and kernel. The cache cannot outlive or cross that identity.
    integral_cache: OnceLock<IntegralCache>,
    cache_budget: usize,
    _cache_allocation: crate::resources::MemoryReservation,
}

impl FiniteQExchange {
    pub fn new(
        cell: &Cell,
        points: &[Vec3],
        transfer: Vec3,
        cutoff: f64,
        kernel: CoulombKernel,
    ) -> Result<Self> {
        Self::with_cache_budget(
            cell,
            points,
            transfer,
            cutoff,
            kernel,
            DEFAULT_INTEGRAL_CACHE_BYTES,
        )
    }

    /// As `new`, with a maximum cache storage bound in bytes. Zero disables
    /// reuse. Uncached quartets are evaluated normally, never screened away.
    /// The bound includes conservative vector-capacity and metadata costs.
    pub fn with_cache_budget(
        cell: &Cell,
        points: &[Vec3],
        transfer: Vec3,
        cutoff: f64,
        kernel: CoulombKernel,
        cache_budget: usize,
    ) -> Result<Self> {
        if let CoulombKernel::Erf { omega } | CoulombKernel::Erfc { omega } = kernel {
            if !omega.is_finite() || omega < 0.0 {
                return Err(DftError::InvalidInput(
                    "finite-q exchange range must be finite and nonnegative".into(),
                ));
            }
        }
        if points.is_empty()
            || points
                .iter()
                .chain(std::iter::once(&transfer))
                .any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite())
            || !cutoff.is_finite()
            || cutoff <= 0.0
            || cell.nao() == 0
        {
            return Err(DftError::InvalidInput("finite-q exchange requires a finite nonempty k mesh, transfer, AO basis and positive cutoff".into()));
        }
        let cells = ExchangeCells::for_lattice(&cell.basis, &cell.lattice, cutoff);
        let periods = born_von_karman_periods(&cell.lattice, points);
        let nsh = cell.basis.shells.len();
        let weights = cells
            .near
            .iter()
            .map(|&image| {
                let mut weights = vec![1.0; nsh * nsh];
                if let Some(periods) = periods {
                    for (a, sa) in cell.basis.shells.iter().enumerate() {
                        for (b, sb) in cell.basis.shells.iter().enumerate() {
                            let displacement = cell.molecule.atoms[sb.atom].position + image
                                - cell.molecule.atoms[sa.atom].position;
                            weights[a * nsh + b] =
                                minimum_image_weight(&cell.lattice, periods, displacement);
                        }
                    }
                }
                weights
            })
            .collect();
        // Share one budget across full/range kernels and their spin actions.
        // An optional cache cannot consume the workspace needed to apply it.
        let cache_budget = crate::resources::remaining_bytes()
            .map_or(cache_budget, |remaining| cache_budget.min(remaining / 8));
        let (cache_budget, cache_allocation) = match crate::resources::reserve_cache(
            cache_budget,
            "finite-q exchange integral cache",
        )? {
            Some(lease) => (cache_budget, lease),
            None => (0, Default::default()),
        };
        Ok(Self {
            basis: cell.basis.clone(),
            cells,
            points: points.to_vec(),
            transfer,
            kernel,
            weights,
            integral_cache: OnceLock::new(),
            cache_budget,
            _cache_allocation: cache_allocation,
        })
    }

    /// Conservative retained-storage bound, zero before the first action.
    pub fn cached_integral_storage_bound(&self) -> usize {
        self.integral_cache
            .get()
            .map_or(0, |cache| cache.storage_bound)
    }

    fn prepare_integral_cache(&self, shell_of: &[usize]) -> IntegralCache {
        let empty = || IntegralCache {
            targets: Vec::new(),
            storage_bound: 0,
        };
        let outer = self
            .cells
            .near
            .len()
            .saturating_mul(4 * std::mem::size_of::<Vec<EriQuartet>>());
        if self.cache_budget <= outer {
            return empty();
        }
        let remaining = AtomicUsize::new(self.cache_budget - outer);
        let nsh = self.basis.shells.len();
        let targets = (0..self.cells.near.len())
            .into_par_iter()
            .map(|target| {
                let mut cached = Vec::new();
                let mut full = false;
                self.cells.visit(
                    &self.basis,
                    target,
                    |a, b| self.weights[target][a * nsh + b] != 0.0,
                    |_| true,
                    |quartet| {
                        if full {
                            return;
                        }
                        let [_, ol, os, _] = quartet.offsets;
                        if self.weights[quartet.source][shell_of[ol] * nsh + shell_of[os]] == 0.0 {
                            return;
                        }
                        let [a, b, c, d] = quartet.shells;
                        let elements = quartet.shells.iter().try_fold(1usize, |size, shell| {
                            size.checked_mul(crate::basis::n_sph(shell.l))
                        });
                        let Some(bytes) = elements
                            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
                            .and_then(|n| n.checked_add(4 * std::mem::size_of::<EriQuartet>()))
                        else {
                            full = true;
                            return;
                        };
                        if remaining
                            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |available| {
                                available.checked_sub(bytes)
                            })
                            .is_err()
                        {
                            full = true;
                            return;
                        }
                        cached.push(eri_quartet(a, b, c, d, self.kernel));
                    },
                );
                cached
            })
            .collect();
        IntegralCache {
            targets,
            storage_bound: self.cache_budget - remaining.into_inner(),
        }
    }

    pub fn apply(&self, density: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        let n = self.basis.nao_sph;
        if density.len() != self.points.len()
            || density.iter().any(|d| {
                d.rows() != n
                    || d.cols() != n
                    || d.im.rows != n
                    || d.im.cols != n
                    || d.re
                        .as_slice()
                        .iter()
                        .chain(d.im.as_slice())
                        .any(|v| !v.is_finite())
            })
        {
            return Err(DftError::InvalidInput("finite-q exchange needs one finite full complex AO transition density per reference k point".into()));
        }
        let nsh = self.basis.shells.len();
        let images = self
            .cells
            .near
            .iter()
            .map(|&translation| {
                let mut out = BlochMatrix::zeros(n, n);
                for (&k, d) in self.points.iter().zip(density) {
                    let factor = phase(-k.dot(translation)) / self.points.len() as f64;
                    for i in 0..n {
                        for j in 0..n {
                            let value = Complex::new(d.re[(i, j)], d.im[(i, j)]) * factor;
                            out.re[(i, j)] += value.re;
                            out.im[(i, j)] += value.im;
                        }
                    }
                }
                out
            })
            .collect::<Vec<_>>();
        let shell_of =
            self.basis
                .shells
                .iter()
                .enumerate()
                .fold(vec![0; n], |mut owner, (shell, s)| {
                    for value in &mut owner[s.sph_off..s.sph_off + crate::basis::n_sph(s.l)] {
                        *value = shell;
                    }
                    owner
                });
        let cache = self
            .integral_cache
            .get_or_init(|| self.prepare_integral_cache(&shell_of));
        let blocks = (0..self.cells.near.len())
            .into_par_iter()
            .map(|target| {
                let mut out = BlochMatrix::zeros(n, n);
                let mut ordinal = 0;
                self.cells.visit(
                    &self.basis,
                    target,
                    |a, b| self.weights[target][a * nsh + b] != 0.0,
                    |_| true,
                    |quartet| {
                        let [mu, lambda, sigma, nu] = quartet.shells;
                        let [om, ol, os, on] = quartet.offsets;
                        let outer = self.weights[target][shell_of[om] * nsh + shell_of[on]];
                        let inner = self.weights[quartet.source][shell_of[ol] * nsh + shell_of[os]];
                        if inner == 0.0 {
                            return;
                        }
                        // P_q(lambda C,sigma D)=exp(i q.C) sum_k exp(-i k.(D-C)) P_q(k).
                        let factor =
                            phase(self.transfer.dot(quartet.lambda_translation)) * (outer * inner);
                        let evaluated;
                        let integral = if let Some(saved) =
                            cache.targets.get(target).and_then(|v| v.get(ordinal))
                        {
                            saved
                        } else {
                            evaluated = eri_quartet(mu, lambda, sigma, nu, self.kernel);
                            &evaluated
                        };
                        ordinal += 1;
                        let source = &images[quartet.source];
                        for i in 0..integral.na {
                            for j in 0..integral.nd {
                                let mut value = Complex::ZERO;
                                for a in 0..integral.nb {
                                    for b in 0..integral.nc {
                                        value += Complex::new(
                                            source.re[(ol + a, os + b)],
                                            source.im[(ol + a, os + b)],
                                        ) * integral.get(i, a, b, j);
                                    }
                                }
                                value = value * factor;
                                out.re[(om + i, on + j)] += value.re;
                                out.im[(om + i, on + j)] += value.im;
                            }
                        }
                    },
                );
                out
            })
            .collect::<Vec<_>>();
        let result: Vec<BlochMatrix> = self
            .points
            .iter()
            .map(|&k| {
                let mut out = BlochMatrix::zeros(n, n);
                for (&translation, block) in self.cells.near.iter().zip(&blocks) {
                    let factor = phase(k.dot(translation));
                    for i in 0..n {
                        for j in 0..n {
                            let value = Complex::new(block.re[(i, j)], block.im[(i, j)]) * factor;
                            out.re[(i, j)] += value.re;
                            out.im[(i, j)] += value.im;
                        }
                    }
                }
                out
            })
            .collect();
        if result.iter().any(|b| {
            b.re.as_slice()
                .iter()
                .chain(b.im.as_slice())
                .any(|v| !v.is_finite())
        }) {
            return Err(DftError::InvalidInput(
                "finite-q exchange produced a nonfinite transition potential".into(),
            ));
        }
        Ok(result)
    }
}
