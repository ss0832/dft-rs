// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::super::point_lattice::fixed_point_cell_response;
use crate::acceptance::{components, stencils};
use crate::math::Vec3;
use crate::periodic::electrostatics::DimensionalCoulomb;
use crate::periodic::ewald::EwaldParams;
use crate::periodic::features::PeriodicBoundary;
use crate::periodic::gaussian_coulomb::lattice_green::fixed_displacement_cell_response;
use crate::periodic::{Lattice, PeriodicGeometry};
use stencils::Stencil;

const ORDERS: [[usize; 3]; 10] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [2, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 2, 0],
    [0, 1, 1],
    [0, 0, 2],
];

#[test]
fn tilted_dimensional_green_cell_partials_match_independent_deformed_kernels() {
    let lattice = [
        Vec3::new(5.61, 0.76, -0.42),
        Vec3::new(0.91, 5.43, 0.67),
        Vec3::new(-0.82, 0.48, 6.16),
    ];
    let r = Vec3::new(0.78, -0.46, 0.37);
    let width = 0.71;
    for (name, axes) in [
        ("wire", [false, true, false]),
        ("slab", [true, false, true]),
    ] {
        let kernel = |component: usize, h: f64| {
            let (a, b) = (component / 3, component % 3);
            let [x, y, z] = lattice.map(|v| {
                let mut r = v.to_array();
                r[a] += h * r[b];
                Vec3::new(r[0], r[1], r[2])
            });
            let geometry =
                PeriodicGeometry::new(Lattice::new(x, y, z), PeriodicBoundary::new(axes)).unwrap();
            DimensionalCoulomb::new(
                geometry,
                EwaldParams {
                    alpha: 0.7,
                    rcut: 11.0,
                    gcut: 10.0,
                },
                96,
            )
            .unwrap()
        };
        let base = kernel(0, 0.0);
        let gaussian =
            fixed_displacement_cell_response(&base, &base.reciprocal, r, width, 2).unwrap();
        let point = fixed_point_cell_response(&base, &base.reciprocal, r, false).unwrap();
        let exact: Vec<_> = (0..9)
            .map(|c| {
                let mut row = Vec::with_capacity(ORDERS.len() + 1);
                row.push(point[c].iter().sum::<f64>());
                row.extend(
                    ORDERS
                        .iter()
                        .map(|&m| gaussian.components(m).unwrap()[c].iter().sum::<f64>()),
                );
                row
            })
            .collect();
        let labels: Vec<_> = (0..9)
            .flat_map(|c| {
                std::iter::once(format!("{name} point affine[{c}]")).chain(
                    ORDERS
                        .into_iter()
                        .map(move |m| format!("{name} Gaussian {m:?} affine[{c}]")),
                )
            })
            .collect();
        let intervals = [
            ([0.02, 0.015, 0.01, 0.0075], Stencil::Second),
            ([0.01, 0.0075, 0.005, 0.00375], Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], Stencil::Fourth),
        ];
        let sample = |c, h| {
            let displaced = kernel(c, h);
            let mut result = vec![displaced.pair(r, 0.0, false).unwrap().0];
            let g = displaced.gaussian_derivatives(r, width, 2).unwrap();
            result.extend(
                ORDERS
                    .iter()
                    .map(|&m| g.components(m).unwrap().iter().sum::<f64>()),
            );
            result
        };
        let mut cache = std::collections::BTreeMap::new();
        let mut at = |c, h: f64| {
            cache
                .entry((c, h.to_bits()))
                .or_insert_with(|| sample(c, h))
                .clone()
        };
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &labels,
            &intervals,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; labels.len()];
                for (c, analytic) in exact.iter().enumerate() {
                    for (i, &h) in steps.iter().enumerate() {
                        let plus = at(c, h);
                        let minus = at(c, -h);
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (at(c, 2.0 * h), at(c, -2.0 * h))
                        } else {
                            (vec![0.0; analytic.len()], vec![0.0; analytic.len()])
                        };
                        for (k, &value) in analytic.iter().enumerate() {
                            let derivative =
                                stencil.first(plus[k], minus[k], plus2[k], minus2[k], h);
                            errors[c * analytic.len() + k][i] = (derivative - value).abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures(name, &failures);
    }
}
