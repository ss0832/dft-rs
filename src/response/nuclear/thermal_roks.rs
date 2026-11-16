// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Taylor continuation of the common-orbital Mermin stationarity equations.

use super::geometry::GeometryTaylor;
use super::{add, density_path, matrix_congruence, matrix_product, metric_transport};
use crate::basis::MolBasis;
use crate::dual::Scalar;
use crate::grid::MolecularGrid;
use crate::linalg::{AccurateSum, Matrix};
use crate::response::complex::Complex;
use crate::response::krylov::{self, ResponseSolverConfig};
use crate::response::series::{evaluate_series_many_result, SeriesScalar};
use crate::response::ResponseSolver;
use crate::scf::{roks::thermal::Jacobian, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub(crate) struct ThermalRoksPath {
    pub energy: Vec<f64>,
    pub residuals: Vec<f64>,
    pub stationarity: Vec<f64>,
    pub number_defects: Vec<f64>,
    pub chemical: [Vec<f64>; 2],
}

pub(crate) struct ThermalRoksNuclearContext<'a> {
    molecule: &'a Molecule,
    basis: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    reference: &'a ScfResult,
    options: &'a ScfOptions,
    kernel: ResponseSolver<'a>,
    nonlocal_grid: Option<MolecularGrid>,
    pairs: Vec<(usize, usize)>,
    fock: [Matrix; 2],
}

impl<'a> ThermalRoksNuclearContext<'a> {
    pub(crate) fn new(
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        reference: &'a ScfResult,
        options: &'a ScfOptions,
    ) -> Result<Self> {
        if !reference.converged
            || reference.electronic_temperature <= 0.0
            || reference.mo_coeff[0].max_abs_diff(&reference.mo_coeff[1]) != 0.0
        {
            return Err(DftError::InvalidInput(
                "thermal common-orbital derivatives require a converged shared spatial frame"
                    .into(),
            ));
        }
        if options.density_fit.is_some()
            || options.pcm.is_some()
            || options.skala.is_some()
            || options.extra_hcore.is_some()
            || !matches!(options.kernel, crate::integrals::CoulombKernel::Coulomb)
        {
            return Err(DftError::Unsupported {
                feature:
                    "thermal common-orbital geometry Taylor response with this additional provider"
                        .into(),
                missing:
                    "the corresponding analytic moving-geometry terms through the requested order"
                        .into(),
            });
        }
        let c = &reference.mo_coeff[0];
        let pairs = (0..c.cols)
            .flat_map(|p| (p + 1..c.cols).map(move |q| (p, q)))
            .filter(|&(p, q)| reference.occupations.iter().any(|f| f[p] != f[q]))
            .collect();
        let fock = reference
            .fock
            .each_ref()
            .map(|f| c.transpose_matmul(&f.matmul(c)));
        let kernel =
            ResponseSolver::kernel_only(molecule, basis, functional, grid, reference, options)?;
        let nonlocal_grid = options
            .vv10
            .map(|_| options.nonlocal_grid(molecule))
            .transpose()?;
        Ok(Self {
            molecule,
            basis,
            functional,
            grid,
            reference,
            options,
            kernel,
            nonlocal_grid,
            pairs,
            fock,
        })
    }

    pub(crate) fn energy_taylor(
        &self,
        direction: &[[f64; 3]],
        order: usize,
        config: &ResponseSolverConfig,
    ) -> Result<ThermalRoksPath> {
        let r = self.reference;
        let c = &r.mo_coeff[0];
        let n = c.cols;
        let kt = r.electronic_temperature;
        let mut geometry = GeometryTaylor::new(
            self.molecule,
            self.basis,
            self.grid,
            self.functional,
            None,
            direction,
            self.options
                .external_field
                .map_or([0.0; 3], |f| f.to_array()),
            [0.0; 3],
            order,
            self.options.incore_mb,
        )?;
        if let Some((parameters, grid)) = self.options.vv10.zip(self.nonlocal_grid.as_ref()) {
            geometry.attach_molecular_nonlocal(
                self.molecule,
                self.basis,
                grid,
                parameters,
                1.0,
                direction,
                order,
                self.options.incore_mb,
            )?;
        }
        let frame = metric_transport(c, &geometry.overlap, order);
        let problem = Jacobian {
            c,
            occupations: &r.occupations,
            fock: &self.fock,
            pairs: &self.pairs,
            kernel: &self.kernel,
            kt,
        };
        let mut rotation = vec![Matrix::zeros(n, n); order + 1];
        let mut levels: [Vec<Vec<f64>>; 2] = std::array::from_fn(|s| {
            let mut values = vec![vec![0.0; order + 1]; n];
            for (v, &e) in values.iter_mut().zip(&r.mo_energy[s]) {
                v[0] = e;
            }
            values
        });
        let counts = [r.n_alpha, r.n_beta];
        let evaluate = |rotation: &[Matrix], levels: &[Vec<Vec<f64>>; 2], width| -> Result<_> {
            let common = matrix_product(&frame, &exponential(rotation, width), width);
            let mut populations = [Vec::new(), Vec::new()];
            let mut chemical = [Vec::new(), Vec::new()];
            for s in 0..2 {
                (populations[s], chemical[s]) = fermi(
                    &levels[s],
                    &r.occupations[s],
                    r.chemical_potentials[s],
                    counts[s],
                    kt,
                    width,
                )?;
            }
            let density = std::array::from_fn(|s| {
                let diag = (0..=width)
                    .map(|k| {
                        let mut d = Matrix::zeros(n, n);
                        for i in 0..n {
                            d[(i, i)] = populations[s][i][k];
                        }
                        d
                    })
                    .collect::<Vec<_>>();
                density_path(&common, &diag, width)
            });
            let values = geometry.evaluate(self.functional, &density, width)?;
            let fock = values
                .fock
                .each_ref()
                .map(|f| matrix_congruence(&common, f, width));
            Ok((values, populations, chemical, fock))
        };
        let residual = |populations: &[Vec<Vec<f64>>; 2], fock: &[Vec<Matrix>; 2], width| {
            self.pairs
                .iter()
                .map(|&(p, q)| {
                    (0..2)
                        .flat_map(|s| (0..=width).map(move |k| (s, k)))
                        .map(|(s, k)| {
                            (populations[s][p][k] - populations[s][q][k])
                                * fock[s][width - k][(p, q)]
                        })
                        .sum::<f64>()
                })
                .chain((0..2).flat_map(|s| (0..n).map(move |i| -fock[s][width][(i, i)])))
                .collect::<Vec<_>>()
        };
        let mut residuals = Vec::with_capacity(order);
        for width in 1..=order {
            let (_, p, _, f) = evaluate(&rotation, &levels, width)?;
            let rhs = residual(&p, &f, width)
                .into_iter()
                .map(|v| Complex::new(-v, 0.0))
                .collect::<Vec<_>>();
            let solution = krylov::solve(&problem, &rhs, None, config)?;
            for (&(p, q), v) in self.pairs.iter().zip(&solution.amplitudes) {
                rotation[width][(q, p)] = v.re;
                rotation[width][(p, q)] = -v.re;
            }
            for s in 0..2 {
                for i in 0..n {
                    levels[s][i][width] = solution.amplitudes[self.pairs.len() + s * n + i].re;
                }
            }
            residuals.push(solution.relative_residual);
        }
        let (values, p, chemical, fock) = evaluate(&rotation, &levels, order)?;
        let mut stationarity = Vec::with_capacity(order + 1);
        let mut number_defects = Vec::with_capacity(2 * order);
        for width in 0..=order {
            let mut check = residual(&p, &fock, width);
            for s in 0..2 {
                for i in 0..n {
                    check[self.pairs.len() + s * n + i] += levels[s][i][width];
                }
                let target = if width == 0 { counts[s] as f64 } else { 0.0 };
                number_defects.push((p[s].iter().map(|f| f[width]).sum::<f64>() - target).abs());
            }
            // Equal occupations remove a rotation at the reference. A path
            // lifting that degeneracy must still satisfy its full residual.
            let omitted = (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .filter(|&(i, j)| r.occupations.iter().all(|f| f[i] == f[j]));
            for (i, j) in omitted {
                let mut g = AccurateSum::default();
                for s in 0..2 {
                    for k in 0..=width {
                        g.add_product(p[s][i][k] - p[s][j][k], fock[s][width - k][(i, j)]);
                    }
                }
                check.push(g.value());
            }
            stationarity.push(check.into_iter().map(f64::abs).fold(0.0, f64::max));
        }
        let worst = stationarity
            .iter()
            .chain(&number_defects)
            .copied()
            .fold(0.0, f64::max);
        if worst > config.acceptance() {
            return Err(DftError::ResponseNotConverged {
                iterations: order,
                residual: worst,
            });
        }
        let mut energy = values.total_energy();
        energy[0] -= kt * r.electronic_entropy;
        for k in 1..=order {
            let mut entropy = AccurateSum::default();
            for s in 0..2 {
                for i in 0..n {
                    for j in 0..k {
                        // Integrate d(-T S) = -(epsilon - mu) df.
                        entropy.add_product(
                            -((k - j) as f64) / k as f64 * (levels[s][i][j] - chemical[s][j]),
                            p[s][i][k - j],
                        );
                    }
                }
            }
            energy[k] += entropy.value();
        }
        Ok(ThermalRoksPath {
            energy,
            residuals,
            stationarity,
            number_defects,
            chemical,
        })
    }
}

fn exponential(rotation: &[Matrix], order: usize) -> Vec<Matrix> {
    let n = rotation[0].rows;
    let mut power = vec![Matrix::zeros(n, n); order + 1];
    power[0] = Matrix::identity(n);
    let mut result = power.clone();
    let mut coefficient = 1.0;
    for k in 1..=order {
        power = matrix_product(&power, rotation, order);
        coefficient /= k as f64;
        for n in k..=order {
            add(&mut result[n], &power[n], coefficient);
        }
    }
    result
}

fn fermi(
    levels: &[Vec<f64>],
    initial: &[f64],
    mu: Option<f64>,
    count: usize,
    kt: f64,
    order: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut chemical = vec![0.0; order + 1];
    let mut occupations = vec![vec![0.0; order + 1]; levels.len()];
    for (f, &initial) in occupations.iter_mut().zip(initial) {
        f[0] = initial;
    }
    if count == 0 || count == levels.len() || order == 0 {
        chemical[0] = mu.unwrap_or(0.0);
        return Ok((occupations, chemical));
    }
    chemical[0] = mu.ok_or_else(|| {
        DftError::InvalidInput(
            "thermal common-orbital response requires both active Fermi levels".into(),
        )
    })?;
    let weight: Vec<_> = initial.iter().map(|f| f * (1.0 - f) / kt).collect();
    let sum: f64 = weight.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        return Err(DftError::Unsupported {
            feature: "common-orbital thermal response with unresolved Fermi weights".into(),
            missing: "a resolvable derivative of the spin electron count".into(),
        });
    }
    for k in 1..=order {
        let shifted = levels
            .iter()
            .map(|e| (0..=k).map(|j| (e[j] - chemical[j]) / kt).collect())
            .collect::<Vec<Vec<f64>>>();
        let bare = evaluate_series_many_result(&shifted, |x| {
            let one = SeriesScalar::from_f64(1.0);
            Ok(x.iter()
                .map(|&z| {
                    if z.value() > 0.0 {
                        let exp = (-z).exp();
                        exp / (one + exp)
                    } else {
                        (one + z.exp()).recip()
                    }
                })
                .collect())
        })?;
        chemical[k] = -bare.iter().map(|f| f[k]).sum::<f64>() / sum;
        for (i, f) in occupations.iter_mut().enumerate() {
            f[k] = bare[i][k] + weight[i] * chemical[k];
        }
    }
    Ok((occupations, chemical))
}
