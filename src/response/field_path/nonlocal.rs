// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The molecular VV10 two-endpoint body with the shared complex AO pullback.
//! Its dedicated quadrature and vacuum convention are identical to SCF.
use super::*;
use crate::nlc::Vv10Params;
use crate::response::nonlocal::vv10_density_taylor;

#[cfg(test)]
mod tests;

pub(super) struct NonlocalFieldProvider {
    grid: MolecularGrid,
    points: Vec<Point>,
    parameters: Vv10Params,
}
impl NonlocalFieldProvider {
    pub(super) fn new(
        molecule: &Molecule,
        basis: &MolBasis,
        parameters: Vv10Params,
        options: &ScfOptions,
        path: &FieldPath,
    ) -> Result<Self> {
        let grid = options.nonlocal_grid(molecule)?;
        Ok(Self::with_grid(basis, grid, parameters, path))
    }
    pub(super) fn with_grid(
        basis: &MolBasis,
        grid: MolecularGrid,
        parameters: Vv10Params,
        path: &FieldPath,
    ) -> Self {
        let points = make_points(basis, &grid, path);
        Self {
            grid,
            points,
            parameters,
        }
    }
    pub(super) fn evaluate(
        &self,
        p: &[Vec<Cm>; 2],
        order: usize,
        matrices: bool,
    ) -> Result<(Vec<f64>, Option<[Cm; 2]>)> {
        let series: Vec<_> = self
            .points
            .par_iter()
            .map(|p| point_series(p, order + 1))
            .collect();
        self.evaluate_series(&series, p, order, matrices)
    }
    fn evaluate_series(
        &self,
        series: &[PointSeries],
        p: &[Vec<Cm>; 2],
        order: usize,
        matrices: bool,
    ) -> Result<(Vec<f64>, Option<[Cm; 2]>)> {
        if p.iter().any(|s| s.len() <= order) || series.len() != self.grid.len() {
            return Err(DftError::InvalidInput(
                "nonlocal field density and quadrature coefficient dimensions disagree".into(),
            ));
        }
        let fields: Vec<_> = series
            .par_iter()
            .map(|point| density_fields(point, p, order, true, false, false))
            .collect::<Result<_>>()?;
        let rho: Vec<_> = fields
            .iter()
            .map(|v| {
                (0..=order)
                    .map(|n| v.variables[0][n] + v.variables[1][n])
                    .collect()
            })
            .collect();
        let gradient: Vec<[Vec<f64>; 3]> = fields
            .iter()
            .map(|v| {
                std::array::from_fn(|a| {
                    (0..=order)
                        .map(|n| v.gradient[0][a][n] + v.gradient[1][a][n])
                        .collect()
                })
            })
            .collect();
        let values = vv10_density_taylor(&self.grid, &rho, &gradient, self.parameters)?;
        if !matrices {
            return Ok((values.energy, None));
        }
        let nao = p[0][0].re.rows;
        let zero = || std::array::from_fn(|_| Cm::zeros(nao, nao));
        let fock = series
            .par_iter()
            .enumerate()
            .fold(zero, |mut total, (i, point)| {
                let g: [Vec<f64>; 3] = std::array::from_fn(|a| {
                    (0..=order)
                        .map(|n| {
                            2.0 * (0..=n)
                                .map(|k| values.vsigma[i][k] * gradient[i][a][n - k])
                                .sum::<f64>()
                        })
                        .collect()
                });
                let zeros = vec![0.0; order + 1];
                let potentials = PointPotentials {
                    rho: [&values.vrho[i], &values.vrho[i]],
                    gradient: std::array::from_fn(|_| std::array::from_fn(|a| g[a].as_slice())),
                    tau: [&zeros, &zeros],
                };
                let mut weight = vec![0.0; order + 1];
                weight[0] = self.grid.weights[i];
                let out = point_pullback(
                    point,
                    &weight,
                    &potentials,
                    &fields[i].kinetic,
                    order,
                    true,
                    false,
                );
                for s in 0..2 {
                    for (a, b) in total[s].re.as_mut_slice().iter_mut().zip(&out[s].0) {
                        *a += b;
                    }
                    for (a, b) in total[s].im.as_mut_slice().iter_mut().zip(&out[s].1) {
                        *a += b;
                    }
                }
                total
            })
            .reduce(zero, |mut a, b| {
                for s in 0..2 {
                    cadd(&mut a[s], &b[s], 1.0);
                }
                a
            });
        Ok((values.energy, Some(fock)))
    }
}

pub(crate) struct NonlocalFiniteField {
    provider: NonlocalFieldProvider,
    series: Vec<PointSeries>,
}
impl NonlocalFiniteField {
    pub(crate) fn new(
        molecule: &Molecule,
        basis: &MolBasis,
        parameters: Vv10Params,
        options: &ScfOptions,
        field: [f64; 3],
    ) -> Result<Self> {
        let provider = NonlocalFieldProvider::new(
            molecule,
            basis,
            parameters,
            options,
            &FieldPath {
                electric: [0.0; 3],
                magnetic: field,
                origin: [0.0; 3],
                london: true,
                order: 0,
            },
        )?;
        let series = finite_point_series(&provider.points);
        Ok(Self { provider, series })
    }
    pub(crate) fn evaluate(&self, density: &[Cm; 2]) -> Result<(f64, [Cm; 2])> {
        let p = std::array::from_fn(|s| vec![density[s].clone()]);
        let (energy, fock) = self.provider.evaluate_series(&self.series, &p, 0, true)?;
        Ok((
            energy[0],
            fock.expect("the nonlocal finite-field potential was requested"),
        ))
    }
}
