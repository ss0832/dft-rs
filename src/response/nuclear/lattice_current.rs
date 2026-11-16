// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Time-odd meta-GGA kernel for real Gamma-point lattice references.

use super::{Reference, ResolvedFunctional};
use crate::linalg::Matrix;
use crate::properties::giao_xc::MetaCurrentKernel;
use crate::response::nuclear::geometry::GridPointTaylor;
use crate::Result;
use rayon::prelude::*;

const BATCH: usize = 256;

pub(super) struct LatticeCurrentKernel<'a> {
    grid: &'a [GridPointTaylor],
    coefficients: [Vec<f64>; 2],
}

impl<'a> LatticeCurrentKernel<'a> {
    pub(super) fn new(functional: &ResolvedFunctional, reference: &'a Reference) -> Result<Self> {
        let grid = reference.geometry.grid_points();
        let ground = std::array::from_fn(|spin| vec![reference.density[spin].clone()]);
        let weights = grid
            .par_iter()
            .map(|point| point.meta_current_weight(functional, &ground))
            .collect::<Result<Vec<_>>>()?;
        let (alpha, beta) = weights.into_iter().map(|[a, b]| (a, b)).unzip();
        Ok(Self {
            grid,
            coefficients: [alpha, beta],
        })
    }

    pub(super) fn add_response(
        &self,
        output: &mut [Matrix; 2],
        density: &[Matrix; 2],
        spins: usize,
    ) {
        let nao = density[0].rows;
        let transposed = density.each_ref().map(Matrix::transpose);
        // Reuse the zero-order values of the stationary reference. The
        // four AO matrices occupy one block and are shared by both spins.
        for (block, points) in self.grid.chunks(BATCH).enumerate() {
            let mut batch: [Matrix; 4] = std::array::from_fn(|_| Matrix::zeros(points.len(), nao));
            for (g, point) in points.iter().enumerate() {
                for (mu, ao) in point.aos.iter().enumerate() {
                    for (matrix, series) in batch.iter_mut().zip(ao) {
                        matrix[(g, mu)] = series[0];
                    }
                }
            }
            let [phi, dx, dy, dz] = batch;
            let gradients = [dx, dy, dz];
            let start = block * BATCH;
            for s in 0..spins {
                MetaCurrentKernel::accumulate_response(
                    &mut output[s],
                    &phi,
                    &gradients,
                    &transposed[s],
                    &self.coefficients[s][start..start + points.len()],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::components::component_fit_of_order;
    use crate::acceptance::stencils::Stencil;
    use crate::basis::{g94, MolBasis};
    use crate::grid::eval::eval_aos;
    use crate::math::Vec3;
    use crate::periodic::spinor_scf::pauli_components;
    use crate::periodic::spinor_xc::spinor_semilocal_grid_response;
    use crate::periodic::BlochMatrix;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    #[test]
    fn tpss_time_odd_kernel_matches_independent_spinor_grid_slopes() {
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.11, -0.13, 0.17),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.54, 0.17, -0.12),
            },
        ]);
        let basis = g94::parse("current-kernel-check", "H 0\nS 1 1.0\n0.7 1.0\n****\n").unwrap();
        let basis = MolBasis::build(&molecule, &basis).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("tpss".into())).unwrap();
        let points = [
            Vec3::new(0.33, -0.28, 0.42),
            Vec3::new(0.68, 0.51, -0.29),
            Vec3::new(1.13, -0.41, 0.51),
            Vec3::new(1.42, 0.65, 0.13),
        ];
        let weights = [0.51, 0.23, 0.38, 0.44];
        let aos = eval_aos(&basis, &points, true);
        let derivatives = aos.grad.unwrap();
        let grid: Vec<GridPointTaylor> = weights
            .iter()
            .enumerate()
            .map(|(g, &weight)| GridPointTaylor {
                weight: vec![weight],
                aos: (0..2)
                    .map(|mu| {
                        [
                            vec![aos.phi[(g, mu)]],
                            vec![derivatives[0][(g, mu)]],
                            vec![derivatives[1][(g, mu)]],
                            vec![derivatives[2][(g, mu)]],
                        ]
                    })
                    .collect(),
            })
            .collect();
        let ground = [
            vec![Matrix::from_row_major(2, 2, vec![0.57, 0.11, 0.11, 0.42])],
            vec![Matrix::from_row_major(2, 2, vec![0.43, -0.07, -0.07, 0.33])],
        ];
        let coefficients: Vec<_> = grid
            .iter()
            .map(|point| point.meta_current_weight(&functional, &ground).unwrap())
            .collect();
        let kernel = LatticeCurrentKernel {
            grid: &grid,
            coefficients: std::array::from_fn(|s| coefficients.iter().map(|p| p[s]).collect()),
        };
        let phi = BlochMatrix {
            re: aos.phi,
            im: Matrix::zeros(points.len(), 2),
        };
        let momentum = derivatives.map(|mut im| {
            for value in im.as_mut_slice() {
                *value = -*value;
            }
            BlochMatrix {
                re: Matrix::zeros(points.len(), 2),
                im,
            }
        });
        let mut density = BlochMatrix::zeros(4, 4);
        for (spin, block) in ground.iter().enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    density.re[(2 * spin + i, 2 * spin + j)] = block[0][(i, j)];
                }
            }
        }
        let steps = [0.08, 0.04, 0.02, 0.01];
        // Each real antisymmetric 2 x 2 spin block has one independent
        // element. Exercise both diagonal and cross-spin kernel blocks.
        for source in 0..2 {
            let mut direction = [Matrix::zeros(2, 2), Matrix::zeros(2, 2)];
            direction[source][(0, 1)] = 1.0;
            direction[source][(1, 0)] = -1.0;
            let mut analytic = [Matrix::zeros(2, 2), Matrix::zeros(2, 2)];
            kernel.add_response(&mut analytic, &direction, 2);
            let evaluate = |h| {
                let mut displaced = density.clone();
                displaced.im[(2 * source, 2 * source + 1)] = h;
                displaced.im[(2 * source + 1, 2 * source)] = -h;
                let pauli = pauli_components(&displaced).unwrap();
                spinor_semilocal_grid_response(
                    &functional,
                    &weights,
                    &[&phi],
                    &[&momentum],
                    &[pauli],
                    12,
                )
                .unwrap()
                .potential
                .remove(0)
            };
            let estimates = steps.map(|h| {
                let plus = evaluate(h);
                let minus = evaluate(-h);
                let other = 1 - source;
                assert!(plus.im[(2 * other, 2 * other + 1)].abs() < 1e-14);
                assert!(minus.im[(2 * other, 2 * other + 1)].abs() < 1e-14);
                assert_eq!(analytic[other][(0, 1)], 0.0);
                let entry = (2 * source, 2 * source + 1);
                Stencil::Second.first(plus.im[entry], minus.im[entry], 0.0, 0.0, h)
            });
            let target = analytic[source][(0, 1)];
            let errors = estimates.map(|v| (v - target).abs());
            let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
            println!(
                "TPSS spin {source} current kernel {target:.12e}, Richardson {richardson:.3e}"
            );
            assert!(richardson < 1e-7);
            assert!(component_fit_of_order(
                &format!("TPSS spin {source} current kernel"),
                &steps,
                &errors,
                Stencil::Second.order()
            ));
        }
    }
}
