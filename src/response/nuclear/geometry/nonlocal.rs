// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The same two-endpoint functional supplies the energy, potential
//! and induced potential. Nested dual numbers differentiate the potential
//! with respect to a density change without a displaced calculation.

use super::{GridPointTaylor, Matrix};
use crate::dual::{DualN, Scalar, Taylor};
use crate::math::Vec3;
use crate::periodic::nonlocal::{vv10_fields, PeriodicVv10Options};
use crate::periodic::PeriodicGeometry;
use crate::{DftError, Result};
use rayon::prelude::*;

#[cfg(test)]
mod convergence {
    include!("../../../../tests/support/convergence.rs");
}

#[derive(Clone)]
enum NonlocalDomain {
    Periodic {
        geometry: Box<PeriodicGeometry>,
        lattice: Box<[[Vec<f64>; 3]; 3]>,
        options: PeriodicVv10Options,
    },
    Molecular {
        parameters: crate::nlc::Vv10Params,
    },
}

#[derive(Clone)]
pub(in crate::response::nuclear) struct NonlocalTaylor {
    domain: NonlocalDomain,
    reference_points: Vec<Vec3>,
    positions: Vec<[Vec<f64>; 3]>,
    molecular_grid: Option<Vec<GridPointTaylor>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::{features::PeriodicBoundary, Lattice};

    #[test]
    fn two_endpoint_potential_and_moving_kernel_are_variational_with_four_step_fits() {
        let rows = [[7.1, 0.0, 0.0], [0.7, 7.7, 0.0], [-0.2, 0.3, 8.3]];
        let geometry = PeriodicGeometry::new(
            Lattice::new(
                Vec3::new(7.1, 0.0, 0.0),
                Vec3::new(0.7, 7.7, 0.0),
                Vec3::new(-0.2, 0.3, 8.3),
            ),
            PeriodicBoundary::new([true; 3]),
        )
        .unwrap();
        let points = vec![
            Vec3::new(0.2, 0.3, 0.4),
            Vec3::new(1.1, -0.4, 0.8),
            Vec3::new(-0.8, 0.6, 0.7),
        ];
        let term = NonlocalTaylor::periodic(
            geometry,
            points.clone(),
            points
                .iter()
                .enumerate()
                .map(|(i, p)| std::array::from_fn(|a| vec![p.get(a), 0.02 * (i + a + 1) as f64]))
                .collect(),
            std::array::from_fn(|a| {
                std::array::from_fn(|b| vec![rows[a][b], 0.03 * (a + b + 1) as f64])
            }),
            PeriodicVv10Options {
                cutoff: 13.7,
                ..Default::default()
            },
        );
        let grid: Vec<_> = (0..3)
            .map(|i| GridPointTaylor {
                weight: vec![0.7 + 0.1 * i as f64, 0.02],
                aos: (0..2)
                    .map(|mu| {
                        std::array::from_fn(|a| {
                            vec![0.2 + 0.05 * (i + mu + a) as f64, 0.01 * (mu + a + 1) as f64]
                        })
                    })
                    .collect(),
            })
            .collect();
        let matrix = |values: [f64; 4]| Matrix::from_row_major(2, 2, values.to_vec());
        let p = [
            vec![
                matrix([0.8, 0.1, 0.1, 0.6]),
                matrix([0.02, 0.01, 0.01, -0.01]),
            ],
            vec![
                matrix([0.6, -0.05, -0.05, 0.9]),
                matrix([-0.01, 0.02, 0.02, 0.03]),
            ],
        ];
        let q = [
            vec![
                matrix([0.3, -0.1, -0.1, 0.2]),
                matrix([0.02, 0.01, 0.01, 0.03]),
            ],
            vec![
                matrix([-0.1, 0.15, 0.15, 0.25]),
                matrix([0.03, -0.01, -0.01, 0.02]),
            ],
        ];
        let (_, potential) = term.evaluate(&grid, &p, None, 1).unwrap();
        let (_, kernel) = term.evaluate(&grid, &p, Some(&q), 1).unwrap();
        let steps = [0.08, 0.04, 0.02, 0.01];
        for observable in 0..3 {
            let exact = if observable == 0 {
                (0..2)
                    .map(|s| potential[s][0].frobenius_dot(&q[s][0]))
                    .sum::<f64>()
            } else {
                kernel[0][observable - 1][(0, 1)]
            };
            let mut estimates = Vec::new();
            for h in steps {
                let mut values = Vec::new();
                for sign in [-1.0, 1.0] {
                    let mut density = p.clone();
                    for s in 0..2 {
                        for n in 0..2 {
                            for (v, d) in density[s][n]
                                .as_mut_slice()
                                .iter_mut()
                                .zip(q[s][n].as_slice())
                            {
                                *v += sign * h * d;
                            }
                        }
                    }
                    let (e, v) = term.evaluate(&grid, &density, None, 1).unwrap();
                    values.push(if observable == 0 {
                        e[0]
                    } else {
                        v[0][observable - 1][(0, 1)]
                    });
                }
                estimates.push((values[1] - values[0]) / (2.0 * h));
            }
            let errors: Vec<_> = estimates.iter().map(|v| (v - exact).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs();
            println!("periodic nonlocal variational observable={observable} value={exact:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
            assert!(rich < 1e-10);
        }
    }
}

impl NonlocalTaylor {
    pub fn approximate_bytes(&self) -> usize {
        let mut values = self.positions.iter().flatten().map(Vec::len).sum::<usize>();
        if let NonlocalDomain::Periodic { lattice, .. } = &self.domain {
            values += lattice.iter().flatten().map(Vec::len).sum::<usize>();
        }
        if let Some(grid) = &self.molecular_grid {
            values += grid
                .iter()
                .map(|p| p.weight.len() + p.aos.iter().flatten().map(Vec::len).sum::<usize>())
                .sum::<usize>();
        }
        8 * values
    }
    pub fn periodic(
        geometry: PeriodicGeometry,
        reference_points: Vec<Vec3>,
        positions: Vec<[Vec<f64>; 3]>,
        lattice: [[Vec<f64>; 3]; 3],
        options: PeriodicVv10Options,
    ) -> Self {
        Self {
            domain: NonlocalDomain::Periodic {
                geometry: Box::new(geometry),
                lattice: Box::new(lattice),
                options,
            },
            reference_points,
            positions,
            molecular_grid: None,
        }
    }

    pub fn molecular(
        mol: &crate::system::Molecule,
        basis: &crate::basis::MolBasis,
        grid: &crate::grid::MolecularGrid,
        parameters: crate::nlc::Vv10Params,
        direction: &[[f64; 3]],
        order: usize,
    ) -> Result<Self> {
        if direction.len() != mol.len() || direction.iter().flatten().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "molecular nonlocal paths need one finite direction per atom".into(),
            ));
        }
        let input = super::inputs(mol, direction, order);
        let points = super::moving_grid_taylor(mol, basis, grid, &input, order)?;
        let positions = grid
            .points
            .iter()
            .zip(&grid.atom_of)
            .map(|(point, &owner)| {
                std::array::from_fn(|axis| {
                    let mut series = vec![0.0; order + 1];
                    series[0] = point.get(axis);
                    if order > 0 {
                        series[1] = direction[owner][axis];
                    }
                    series
                })
            })
            .collect();
        Ok(Self {
            domain: NonlocalDomain::Molecular { parameters },
            reference_points: grid.points.clone(),
            positions,
            molecular_grid: Some(points),
        })
    }

    pub fn extend_fixed(&mut self, width: usize) {
        for axis in self.positions.iter_mut().flat_map(|p| p.iter_mut()) {
            axis.resize(width, 0.0);
        }
        if let NonlocalDomain::Periodic { lattice, .. } = &mut self.domain {
            for axis in lattice.iter_mut().flat_map(|p| p.iter_mut()) {
                axis.resize(width, 0.0);
            }
        }
        if let Some(grid) = &mut self.molecular_grid {
            for point in grid {
                point.weight.resize(width, 0.0);
                for ao in point.aos.iter_mut().flatten() {
                    ao.resize(width, 0.0);
                }
            }
        }
    }

    pub(in crate::response::nuclear) fn evaluate(
        &self,
        grid: &[GridPointTaylor],
        p: &[Vec<Matrix>; 2],
        q: Option<&[Vec<Matrix>; 2]>,
        order: usize,
    ) -> Result<(Vec<f64>, [Vec<Matrix>; 2])> {
        let grid = self.molecular_grid.as_deref().unwrap_or(grid);
        match order {
            0..=3 => self.evaluate_width::<4>(grid, p, q, order),
            4..=7 => self.evaluate_width::<8>(grid, p, q, order),
            _ => Err(DftError::Unsupported {
                feature: "nonlocal path response above order seven".into(),
                missing: "a wider scalar Taylor representation".into(),
            }),
        }
    }

    fn evaluate_width<const W: usize>(
        &self,
        grid: &[GridPointTaylor],
        p: &[Vec<Matrix>; 2],
        q: Option<&[Vec<Matrix>; 2]>,
        order: usize,
    ) -> Result<(Vec<f64>, [Vec<Matrix>; 2])> {
        let t = |v: &[f64]| Taylor::<W>::from_coefficients(&v[..=order]);
        let constant = |v: &[f64]| DualN::<Taylor<W>, 1>::constant(t(v));
        // The scalar density sees the symmetric part of a general AO
        // density change. Exchange retains its independent index order.
        let symmetric = q.map(|q| {
            std::array::from_fn(|s| {
                q[s][..=order]
                    .iter()
                    .map(|m| {
                        let mut out = m.clone();
                        for a in 0..m.rows {
                            for b in 0..m.cols {
                                out[(a, b)] = 0.5 * (m[(a, b)] + m[(b, a)]);
                            }
                        }
                        out
                    })
                    .collect::<Vec<_>>()
            })
        });
        let fields: Vec<_> = grid
            .par_iter()
            .map(|point| {
                let (base, grad) = point.fields(p, order, true, false);
                let change = symmetric
                    .as_ref()
                    .map(|q| point.fields(q, order, true, false));
                let total =
                    |a: &[Vec<f64>; 7]| (0..=order).map(|n| a[0][n] + a[1][n]).collect::<Vec<_>>();
                let rho = DualN {
                    re: t(&total(&base)),
                    im: [change
                        .as_ref()
                        .map(|c| t(&total(&c.0)))
                        .unwrap_or_else(|| Taylor::from_f64(0.0))],
                };
                let gradient: [_; 3] = std::array::from_fn(|a| {
                    let sum = |g: &[[Vec<f64>; 3]; 2]| {
                        (0..=order)
                            .map(|n| g[0][a][n] + g[1][a][n])
                            .collect::<Vec<_>>()
                    };
                    DualN {
                        re: t(&sum(&grad)),
                        im: [change
                            .as_ref()
                            .map(|c| t(&sum(&c.1)))
                            .unwrap_or_else(|| Taylor::from_f64(0.0))],
                    }
                });
                let sigma = gradient
                    .iter()
                    .fold(DualN::from_f64(0.0), |a, g| a + *g * *g);
                (rho, gradient, sigma)
            })
            .collect();
        let positions: Vec<_> = self
            .positions
            .iter()
            .map(|r| std::array::from_fn(|a| constant(&r[a])))
            .collect();
        let weights: Vec<_> = grid.iter().map(|g| constant(&g.weight)).collect();
        let rho: Vec<_> = fields.iter().map(|v| v.0).collect();
        let sigma: Vec<_> = fields.iter().map(|v| v.2).collect();
        let (energy, d_rho, d_sigma) = match &self.domain {
            NonlocalDomain::Periodic {
                geometry,
                lattice,
                options,
            } => {
                let result = vv10_fields(
                    geometry,
                    &self.reference_points,
                    &positions,
                    std::array::from_fn(|a| std::array::from_fn(|b| constant(&lattice[a][b]))),
                    &weights,
                    &rho,
                    &sigma,
                    *options,
                )?;
                (result.energy, result.d_rho, result.d_sigma)
            }
            NonlocalDomain::Molecular { parameters } => {
                let result = crate::nlc::vv10_moving_density_fields(
                    &positions,
                    &weights,
                    &rho,
                    &sigma,
                    *parameters,
                )?;
                (
                    result.energy,
                    result
                        .d_rho
                        .into_iter()
                        .zip(&weights)
                        .map(|(v, w)| v * *w)
                        .collect(),
                    result
                        .d_sigma
                        .into_iter()
                        .zip(&weights)
                        .map(|(v, w)| v * *w)
                        .collect(),
                )
            }
        };
        let nao = p[0][0].rows;
        let mut potential = vec![Matrix::zeros(nao, nao); order + 1];
        // The endpoint partial derivatives include the target quadrature
        // weight. Multiplying it again would break variational consistency.
        for (i, point) in grid.iter().enumerate() {
            for mu in 0..nao {
                for nu in 0..nao {
                    let a = constant(&point.aos[mu][0]);
                    let b = constant(&point.aos[nu][0]);
                    let mut v = d_rho[i] * a * b;
                    for axis in 0..3 {
                        let gradient_pair = constant(&point.aos[mu][axis + 1]) * b
                            + a * constant(&point.aos[nu][axis + 1]);
                        v = v + DualN::from_f64(2.0)
                            * d_sigma[i]
                            * fields[i].1[axis]
                            * gradient_pair;
                    }
                    let coefficients = if q.is_some() { v.im[0] } else { v.re };
                    for n in 0..=order {
                        potential[n][(mu, nu)] += coefficients.c[n];
                    }
                }
            }
        }
        Ok((
            energy.re.c[..=order].to_vec(),
            [potential.clone(), potential],
        ))
    }
}
