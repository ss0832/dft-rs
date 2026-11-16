// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All-coordinate acceptance for cached molecular partition derivatives.
use super::*;
use crate::acceptance::{components, stencils};
use crate::grid::MolecularGrid;
use crate::system::Molecule;

fn fixture() -> Molecule {
    Molecule::new(
        [
            (6, [0.13, -0.21, 0.19]),
            (1, [1.57, 0.16, -0.08]),
            (7, [-0.19, 1.93, 0.37]),
            (9, [0.42, 0.29, 2.28]),
        ]
        .into_iter()
        .map(|(z, r)| crate::system::Atom {
            z,
            position: Vec3::new(r[0], r[1], r[2]),
        })
        .collect(),
    )
}

fn value_and_gradient(centers: &[Vec3], aij: &[f64], owner: usize, offset: Vec3) -> Vec<f64> {
    let (value, gradient) =
        becke_weight_gradient(centers, aij, centers.len(), owner, centers[owner] + offset);
    let sum = gradient.iter().copied().fold(Vec3::zero(), |a, b| a + b);
    let mut result = vec![value];
    for (atom, g) in gradient.iter().enumerate() {
        for k in 0..3 {
            result.push(g.get(k) - if atom == owner { sum.get(k) } else { 0.0 });
        }
    }
    result
}

#[test]
fn cached_weight_all_cartesian_components() {
    let molecule = fixture();
    let centers = molecule
        .atoms
        .iter()
        .map(|a| a.position)
        .collect::<Vec<_>>();
    let aij = MolecularGrid::becke_aij(&molecule);
    let n = 3 * centers.len();
    let offset = Vec3::new(0.53, -0.43, 0.61);
    let mut failures = Vec::new();
    for owner in 0..centers.len() {
        let (first, second) = MolecularWeights {
            centers: &centers,
            aij: &aij,
        }
        .derivatives(centers[owner] + offset, owner);
        let names = (0..n)
            .flat_map(|j| {
                (0..=n).map(move |i| {
                    format!("cached weight owner={owner} derivative={i} coordinate={j}")
                })
            })
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                (
                    [0.005, 0.0025, 0.00125, 0.000625],
                    stencils::Stencil::Second,
                ),
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; n * (n + 1)];
                for j in 0..n {
                    for (s, &h) in steps.iter().enumerate() {
                        let displaced = |shift| {
                            let mut cs = centers.clone();
                            let r = &mut cs[j / 3];
                            match j % 3 {
                                0 => r.x += shift,
                                1 => r.y += shift,
                                _ => r.z += shift,
                            };
                            value_and_gradient(&cs, &aij, owner, offset)
                        };
                        let plus = displaced(h);
                        let minus = displaced(-h);
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (displaced(2.0 * h), displaced(-2.0 * h))
                        } else {
                            (vec![0.0; n + 1], vec![0.0; n + 1])
                        };
                        for i in 0..=n {
                            let analytic = if i == 0 {
                                first[j]
                            } else {
                                second[(i - 1) * n + j]
                            };
                            let finite = stencil.first(plus[i], minus[i], plus2[i], minus2[i], h);
                            errors[j * (n + 1) + i][s] = (finite - analytic).abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("cached Becke Cartesian gradient and Hessian", &failures);
}

#[test]
fn cached_weights_match_dense_duals_and_zero_factor_limits() {
    let molecule = fixture();
    let centers = molecule
        .atoms
        .iter()
        .map(|a| a.position)
        .collect::<Vec<_>>();
    let aij = MolecularGrid::becke_aij(&molecule);
    let n = 3 * centers.len();
    for offset in [
        Vec3::new(0.53, -0.43, 0.61),
        Vec3::new(0.0001, -0.0002, 0.0003),
        Vec3::new(51.0, -42.0, 39.0),
    ] {
        for owner in 0..centers.len() {
            let point = centers[owner] + offset;
            let (_, second) = MolecularWeights {
                centers: &centers,
                aij: &aij,
            }
            .derivatives(point, owner);
            for a in 0..centers.len() {
                for b in 0..centers.len() {
                    let dense = becke_weight_second(&centers, &aij, owner, a, b, point);
                    for (x, row) in dense.iter().enumerate() {
                        for (y, expected) in row.iter().enumerate() {
                            let actual = second[(3 * a + x) * n + 3 * b + y];
                            assert!(
                                actual.is_finite() && (actual - expected).abs() < 5e-11,
                                "owner={owner} a={a} b={b} x={x} y={y}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
    let single = [Vec3::zero()];
    let (first, second) = MolecularWeights {
        centers: &single,
        aij: &[0.0],
    }
    .derivatives(Vec3::new(0.5, 0.2, -0.3), 0);
    assert!(first.iter().chain(&second).all(|&v| v == 0.0));
}
