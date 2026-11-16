// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Global learned energy and its complete complex field-density pullback.
//! Network derivatives include every grid endpoint. The same AO/current
//! chain rule as the local providers is used by SCF and PT2 field paths.
use super::*;
use crate::dual::Taylor;
use crate::skala::{Number, SkalaInput, SkalaModel};
use std::sync::Arc;

#[cfg(test)]
mod tests;

trait CoefficientNumber: Number {
    fn from_series(values: &[f64]) -> Self;
    fn coefficients(&self) -> &[f64];
}
impl CoefficientNumber for f64 {
    fn from_series(values: &[f64]) -> Self {
        assert_eq!(values.len(), 1);
        values[0]
    }
    fn coefficients(&self) -> &[f64] {
        std::slice::from_ref(self)
    }
}
impl<const W: usize> CoefficientNumber for Taylor<W> {
    fn from_series(values: &[f64]) -> Self {
        Taylor::from_coefficients(values)
    }
    fn coefficients(&self) -> &[f64] {
        &self.c
    }
}

pub(super) struct LearnedFieldProvider {
    model: Arc<SkalaModel>,
    atom_of: Vec<usize>,
    points: Vec<[f64; 3]>,
    centers: Vec<[f64; 3]>,
    weights: Vec<f64>,
    raw_weights: Vec<f64>,
}
impl LearnedFieldProvider {
    pub(super) fn new(model: &Arc<SkalaModel>, molecule: &Molecule, grid: &MolecularGrid) -> Self {
        Self {
            model: model.clone(),
            atom_of: grid.atom_of.clone(),
            points: grid.points.iter().map(|p| p.to_array()).collect(),
            centers: molecule
                .atoms
                .iter()
                .map(|a| a.position.to_array())
                .collect(),
            weights: grid.weights.clone(),
            raw_weights: grid.raw_weights.clone(),
        }
    }
    pub(super) fn evaluate(
        &self,
        series: &[PointSeries],
        p: &[Vec<Cm>; 2],
        order: usize,
        matrices: bool,
        kinetic: MagneticKineticDensity,
    ) -> Result<(Vec<f64>, Option<[Cm; 2]>)> {
        if series.len() != self.points.len() || p.iter().any(|v| v.len() <= order) {
            return Err(DftError::InvalidInput(
                "global field grid and density series disagree".into(),
            ));
        }
        match order {
            0 => self.evaluate_number::<f64>(series, p, order, matrices, kinetic),
            1..=3 => self.evaluate_number::<Taylor<4>>(series, p, order, matrices, kinetic),
            4..=7 => self.evaluate_number::<Taylor<8>>(series, p, order, matrices, kinetic),
            _ => Err(DftError::Unsupported {
                feature: "learned complex field series above order seven".into(),
                missing: "a larger owned scalar width for the global network graph".into(),
            }),
        }
    }
    fn evaluate_number<S: CoefficientNumber>(
        &self,
        series: &[PointSeries],
        p: &[Vec<Cm>; 2],
        order: usize,
        matrices: bool,
        kinetic: MagneticKineticDensity,
    ) -> Result<(Vec<f64>, Option<[Cm; 2]>)> {
        let corrected = kinetic == MagneticKineticDensity::CurrentInvariant;
        let fields: Vec<_> = series
            .par_iter()
            .map(|s| density_fields(s, p, order, true, true, corrected))
            .collect::<Result<_>>()?;
        let rho: [Vec<S>; 2] = std::array::from_fn(|s| {
            fields
                .iter()
                .map(|f| S::from_series(&f.variables[s]))
                .collect()
        });
        let gradient: [Vec<[S; 3]>; 2] = std::array::from_fn(|s| {
            fields
                .iter()
                .map(|f| std::array::from_fn(|a| S::from_series(&f.gradient[s][a])))
                .collect()
        });
        let tau: [Vec<S>; 2] = std::array::from_fn(|s| {
            fields
                .iter()
                .map(|f| S::from_series(&f.variables[5 + s]))
                .collect()
        });
        let weights: Vec<S> = self.weights.iter().map(|&w| S::from_f64(w)).collect();
        let out = self.model.evaluate(
            &SkalaInput {
                atom_of: &self.atom_of,
                points: &self.points,
                centers: &self.centers,
                weights: &weights,
                atomic_weights: &self.raw_weights,
                rho: [&rho[0], &rho[1]],
                gradient: [&gradient[0], &gradient[1]],
                tau: [&tau[0], &tau[1]],
            },
            matrices,
        )?;
        let energy = out.energy.coefficients()[..=order].to_vec();
        if !matrices {
            return Ok((energy, None));
        }
        let nao = series.first().map_or(0, |s| s.w.len());
        let zero = || std::array::from_fn(|_| Cm::zeros(nao, nao));
        let mut unit_weight = vec![0.0; order + 1];
        unit_weight[0] = 1.0;
        let fock = series
            .par_iter()
            .zip(&fields)
            .enumerate()
            .fold(zero, |mut total, (i, (point, fields))| {
                // Network partials already include the quadrature weights and
                // all nonlocal neighbors, so the AO pullback uses unit weight.
                let potentials = PointPotentials {
                    rho: std::array::from_fn(|s| &out.d_rho[s][i].coefficients()[..=order]),
                    gradient: std::array::from_fn(|s| {
                        std::array::from_fn(|a| &out.d_gradient[s][i][a].coefficients()[..=order])
                    }),
                    tau: std::array::from_fn(|s| &out.d_tau[s][i].coefficients()[..=order]),
                };
                let point = point_pullback(
                    point,
                    &unit_weight,
                    &potentials,
                    &fields.kinetic,
                    order,
                    true,
                    true,
                );
                for s in 0..2 {
                    for (a, b) in total[s].re.as_mut_slice().iter_mut().zip(&point[s].0) {
                        *a += b;
                    }
                    for (a, b) in total[s].im.as_mut_slice().iter_mut().zip(&point[s].1) {
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
        Ok((energy, Some(fock)))
    }
}

/// Cached fixed-field basis/descriptor geometry. The finite SCF and the
/// Taylor response share one global functional and one exact AO pullback.
pub(crate) struct LearnedFiniteField {
    provider: LearnedFieldProvider,
    series: Vec<PointSeries>,
    kinetic: MagneticKineticDensity,
}
impl LearnedFiniteField {
    pub(crate) fn new(
        model: &Arc<SkalaModel>,
        molecule: &Molecule,
        basis: &MolBasis,
        grid: &MolecularGrid,
        field: [f64; 3],
        kinetic: MagneticKineticDensity,
    ) -> Self {
        let path = FieldPath {
            electric: [0.0; 3],
            magnetic: field,
            origin: [0.0; 3],
            london: true,
            order: 0,
        };
        let points = make_points(basis, grid, &path);
        let series = finite_point_series(&points);
        Self {
            provider: LearnedFieldProvider::new(model, molecule, grid),
            series,
            kinetic,
        }
    }
    pub(crate) fn evaluate(&self, density: &[Cm; 2]) -> Result<(f64, [Cm; 2])> {
        let p = std::array::from_fn(|s| vec![density[s].clone()]);
        let (energy, fock) = self
            .provider
            .evaluate(&self.series, &p, 0, true, self.kinetic)?;
        Ok((energy[0], fock.expect("finite-field Fock was requested")))
    }
}
