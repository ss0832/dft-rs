// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent scalar-operator fits before electronic relaxation.

use super::*;
use crate::acceptance::{components, stencils};
use crate::periodic::path_series::LatticePath;
use crate::system::{Atom, Molecule};
use std::collections::BTreeMap;
use stencils::Stencil;

fn displaced_ecp_cell() -> impl Fn(&LatticePath, f64) -> Cell {
    let set = crate::basis::g94::parse(
        "scalar geometry fixture",
        include_str!("../../../tests/data/periodic_scalar_ecp.g94"),
    )
    .unwrap();
    let positions = [
        Vec3::new(0.13, -0.21, 0.07),
        Vec3::new(2.0, 0.45, -0.31),
        Vec3::new(-0.46, 1.29, 1.62),
    ];
    let vectors = [
        Vec3::new(4.8, 0.37, 0.18),
        Vec3::new(0.24, 5.18, 0.35),
        Vec3::new(-0.14, -0.22, 5.39),
    ];
    let cutoff = 4.17;
    move |path: &LatticePath, t: f64| {
        let deform = |v: Vec3| {
            let change: [f64; 3] = std::array::from_fn(|a| {
                path.deformation
                    .iter()
                    .enumerate()
                    .map(|(n, f)| {
                        t.powi(n as i32 + 1) * (0..3).map(|b| f[a][b] * v.get(b)).sum::<f64>()
                    })
                    .sum()
            });
            v + Vec3::new(change[0], change[1], change[2])
        };
        let molecule = Molecule::new(
            [11, 11, 1]
                .into_iter()
                .enumerate()
                .map(|(i, z)| {
                    let mut position = positions[i];
                    for (n, u) in path.displacement.iter().enumerate() {
                        position += Vec3::new(u[i][0], u[i][1], u[i][2]) * t.powi(n as i32 + 1);
                    }
                    Atom {
                        z,
                        position: deform(position),
                    }
                })
                .collect(),
        );
        let rows = vectors.map(deform);
        Cell::new(
            molecule,
            Lattice::new(rows[0], rows[1], rows[2]),
            &set,
            cutoff,
        )
        .unwrap()
    }
}

#[test]
fn translated_scalar_ecp_second_fits_complete_nuclear_and_affine_space() {
    let build = displaced_ecp_cell();
    let cell = build(&LatticePath::default(), 0.0);
    let direct = |path: &LatticePath, t| {
        let displaced = build(path, t);
        super::super::bloch(displaced.ecp_image_blocks(), Vec3::zero()).0
    };
    let zero = direct(&LatticePath::default(), 0.0);
    fit_contracted_nuclear_hessian(&cell, &build);
    // Q = I + v 1^T is invertible by the determinant lemma. The 18
    // q_i and 153 q_i+q_j recover all 171 symmetric quadratic components.
    let v: [f64; 18] = std::array::from_fn(|i| (i as f64 - 7.0) * 0.013);
    assert!((1.0 + v.iter().sum::<f64>()).abs() > 0.5);
    let q: [[f64; 18]; 18] =
        std::array::from_fn(|i| std::array::from_fn(|j| v[j] + f64::from(i == j)));
    let directions: Vec<[f64; 18]> = (0..18)
        .flat_map(|a| {
            (0..=a)
                .map(move |b| std::array::from_fn(|i| q[a][i] + if a == b { 0.0 } else { q[b][i] }))
        })
        .collect();
    assert_eq!(directions.len(), 171);
    let paths: Vec<_> = directions
        .iter()
        .map(|d| {
            let u: Vec<_> = d[..9].chunks_exact(3).map(|x| [x[0], x[1], x[2]]).collect();
            let f: [[f64; 3]; 3] =
                std::array::from_fn(|a| std::array::from_fn(|b| d[9 + 3 * a + b]));
            // Curvature also exercises u2, F2 and the F1 u1 chain-rule term.
            LatticePath {
                displacement: vec![u.clone(), u.iter().map(|x| x.map(|y| -0.07 * y)).collect()],
                deformation: vec![f, f.map(|x| x.map(|y| 0.11 * y))],
                ..LatticePath::default()
            }
        })
        .collect();
    let analytic: Vec<_> = paths
        .iter()
        .map(|path| gamma_ecp_second(&cell, path).unwrap())
        .collect();
    for matrix in &analytic {
        assert!(matrix.max_abs_diff(&matrix.transpose()) < 2e-12);
    }
    let entries: Vec<_> = (0..paths.len())
        .flat_map(|d| (0..3).flat_map(move |a| (0..=a).map(move |b| (d, a, b))))
        .collect();
    let labels: Vec<_> = entries
        .iter()
        .map(|(d, a, b)| format!("Gamma ECP second direction {d} AO [{a},{b}]"))
        .collect();
    let mut samples = BTreeMap::new();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.01, 0.007, 0.005, 0.0035], Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], Stencil::Fourth),
        ],
        |steps, stencil| {
            let estimates: Vec<_> = paths
                .iter()
                .enumerate()
                .map(|(d, path)| {
                    steps.map(|h| {
                        let mut secant = |t: f64| {
                            let plus = samples
                                .entry((d, t.to_bits()))
                                .or_insert_with(|| direct(path, t))
                                .clone();
                            let minus = samples
                                .entry((d, (-t).to_bits()))
                                .or_insert_with(|| direct(path, -t));
                            std::array::from_fn::<_, 9, _>(|i| {
                                (plus.as_slice()[i] - 2.0 * zero.as_slice()[i]
                                    + minus.as_slice()[i])
                                    / t
                            })
                        };
                        let first = secant(h);
                        let second = if stencil == Stencil::Fourth {
                            secant(2.0 * h)
                        } else {
                            [0.0; 9]
                        };
                        // Differentiating the antisymmetric secant yields the
                        // second derivative; divide by 2! for the Taylor value.
                        std::array::from_fn::<_, 9, _>(|i| {
                            0.5 * stencil.first(first[i], -first[i], second[i], -second[i], h)
                        })
                    })
                })
                .collect();
            entries
                .iter()
                .map(|&(d, a, b)| estimates[d].map(|x| (x[3 * a + b] - analytic[d][(a, b)]).abs()))
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("complete scalar ECP second operator", &failures);
    let translation = LatticePath {
        displacement: vec![vec![[0.2, -0.1, 0.3]; 3], vec![[-0.07, 0.09, 0.04]; 3]],
        ..LatticePath::default()
    };
    assert!(gamma_ecp_second(&cell, &translation)
        .unwrap()
        .as_slice()
        .iter()
        .all(|&x| x == 0.0));
}

#[test]
fn translated_scalar_ecp_cubic_and_quartic_fit_complete_operator() {
    use crate::acceptance::symmetric_cartesian_directions::directions;
    let build = displaced_ecp_cell();
    let cell = build(&LatticePath::default(), 0.0);
    let operator = |path: &LatticePath, h: f64| {
        let displaced = build(path, h);
        super::super::bloch(displaced.ecp_image_blocks(), Vec3::zero()).0
    };
    let centre = operator(&LatticePath::default(), 0.0);
    let nao = cell.nao();
    let mut failures = Vec::new();
    for order in [3, 4] {
        // Fix the first atom's displacement. Uniform translation is a
        // separate exact zero mode for the operator at every order.
        let probes = directions(15, order);
        for (p, d) in probes.into_iter().enumerate() {
            let mut u = vec![[0.0; 3]];
            u.extend(d[..6].chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
            let f: [[f64; 3]; 3] =
                std::array::from_fn(|a| std::array::from_fn(|b| d[6 + 3 * a + b]));
            let path = LatticePath {
                displacement: vec![u],
                deformation: vec![f],
                ..LatticePath::default()
            };
            let coefficients = gamma_ecp_path(&cell, &path, order).unwrap();
            let first = gamma_ecp_derivative(
                &cell,
                path.displacement.first().map(Vec::as_slice),
                Some(&f),
            )
            .unwrap();
            let second = gamma_ecp_second(&cell, &path).unwrap();
            assert!(first.max_abs_diff(&coefficients[1]) < 2e-12);
            assert!(second.max_abs_diff(&coefficients[2]) < 2e-12);
            let exact = &coefficients[order];
            assert!(exact.max_abs_diff(&exact.transpose()) < 2e-12);
            let entries: Vec<_> = (0..nao)
                .flat_map(|i| (0..=i).map(move |j| (i, j)))
                .collect();
            let names: Vec<_> = entries
                .iter()
                .map(|(i, j)| format!("Gamma scalar ECP order {order} direction {p} AO [{i},{j}]"))
                .collect();
            let mut samples = BTreeMap::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.024, 0.018, 0.012, 0.006], Stencil::Second),
                    ([0.012, 0.009, 0.006, 0.003], Stencil::Second),
                    ([0.024, 0.018, 0.012, 0.006], Stencil::Fourth),
                ],
                |steps, stencil| {
                    let estimates = steps.map(|h| {
                        let mut at = |t: f64| {
                            samples
                                .entry(t.to_bits())
                                .or_insert_with(|| operator(&path, t))
                                .clone()
                        };
                        let mut coefficient = |t: f64| {
                            let (p, m, p2, m2) = (at(t), at(-t), at(2.0 * t), at(-2.0 * t));
                            entries
                                .iter()
                                .map(|&(i, j)| {
                                    if order == 3 {
                                        (p2[(i, j)] - m2[(i, j)] - 2.0 * (p[(i, j)] - m[(i, j)]))
                                            / (12.0 * t.powi(3))
                                    } else {
                                        (p2[(i, j)] + m2[(i, j)] - 4.0 * (p[(i, j)] + m[(i, j)])
                                            + 6.0 * centre[(i, j)])
                                            / (24.0 * t.powi(4))
                                    }
                                })
                                .collect::<Vec<_>>()
                        };
                        let near = coefficient(h);
                        if stencil == Stencil::Fourth {
                            near.into_iter()
                                .zip(coefficient(2.0 * h))
                                .map(|(a, b)| (4.0 * a - b) / 3.0)
                                .collect()
                        } else {
                            near
                        }
                    });
                    entries
                        .iter()
                        .enumerate()
                        .map(|(k, &(i, j))| {
                            std::array::from_fn(|s| (estimates[s][k] - exact[(i, j)]).abs())
                        })
                        .collect()
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("complete scalar ECP cubic and quartic operators", &failures);
    let translation = LatticePath {
        displacement: vec![vec![[0.2, -0.1, 0.3]; 3]; 4],
        ..LatticePath::default()
    };
    assert!(gamma_ecp_path(&cell, &translation, 4).unwrap()[1..]
        .iter()
        .flat_map(Matrix::as_slice)
        .all(|&x| x == 0.0));
}

fn finite_k_density() -> [BlochMatrix; 3] {
    let re = Matrix::from_row_major(
        3,
        3,
        vec![0.67, 0.16, -0.09, 0.16, 0.54, 0.12, -0.09, 0.12, 0.28],
    );
    let im = Matrix::from_row_major(
        3,
        3,
        vec![0.0, 0.047, -0.06, -0.047, 0.0, 0.082, 0.06, -0.082, 0.0],
    );
    let mut conjugate = im.clone();
    for value in conjugate.as_mut_slice() {
        *value = -*value;
    }
    [
        BlochMatrix::from_pair((re.clone(), Matrix::zeros(3, 3))),
        BlochMatrix::from_pair((re.clone(), im)),
        BlochMatrix::from_pair((re, conjugate)),
    ]
}

fn finite_k_points(cell: &Cell) -> [Vec3; 3] {
    let reciprocal = cell.lattice().reciprocal();
    let k = reciprocal[0] * 0.17 - reciprocal[1] * 0.11 + reciprocal[2] * 0.08;
    [Vec3::zero(), k, k * -1.0]
}

fn finite_k_weights(
    blocks: &[ImageBlock],
    points: &[Vec3; 3],
    density: &[BlochMatrix; 3],
) -> Vec<ImageBlock> {
    blocks
        .iter()
        .map(|image| {
            let mut matrix = Matrix::zeros(3, 3);
            for (k, p) in points.iter().zip(density) {
                let (sin, cos) = k.dot(image.translation).sin_cos();
                for i in 0..3 {
                    for j in 0..3 {
                        matrix[(i, j)] += (cos * p.re[(i, j)] + sin * p.im[(i, j)]) / 3.0;
                    }
                }
            }
            ImageBlock {
                translation: image.translation,
                matrix,
            }
        })
        .collect()
}

/// Finite-k complex densities exercise the real image weights used by CPKS.
/// Differentiate all nine independently computed gradient components in
/// every nuclear coordinate, retaining all 81 ordered Hessian entries.
fn fit_contracted_nuclear_hessian(cell: &Cell, build: &impl Fn(&LatticePath, f64) -> Cell) {
    let density = finite_k_density();
    let k = cell.lattice().reciprocal()[0] * 0.17;
    let points = [Vec3::zero(), k, k * -1.0];
    let weights = finite_k_weights(cell.ecp_image_blocks(), &points, &density);
    let hessian = contract_ecp_hessian(cell, &weights).unwrap();
    assert!(hessian.max_abs_diff(&hessian.transpose()) < 1e-12);
    for i in 0..9 {
        for axis in 0..3 {
            assert!(
                (0..3)
                    .map(|a| hessian[(i, 3 * a + axis)])
                    .sum::<f64>()
                    .abs()
                    < 1e-12
            );
        }
    }
    let labels: Vec<_> = (0..9)
        .flat_map(|a| (0..9).map(move |b| format!("finite-k scalar ECP skeleton [{a},{b}]")))
        .collect();
    let mut samples = BTreeMap::new();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.01, 0.007, 0.005, 0.0035], Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 81];
            for b in 0..9 {
                let mut direction = vec![[0.0; 3]; 3];
                direction[b / 3][b % 3] = 1.0;
                let path = LatticePath::displacement(&direction);
                let mut gradient = |t: f64| {
                    samples
                        .entry((b, t.to_bits()))
                        .or_insert_with(|| {
                            let displaced = build(&path, t);
                            contract_ecp_geometry(
                                displaced.basis(),
                                EcpBoundary::Bulk(displaced.lattice()),
                                displaced.overlap_cutoff(),
                                3,
                                displaced.ecp_image_blocks(),
                                &points,
                                &density,
                            )
                            .unwrap()
                            .gradient
                            .into_iter()
                            .flat_map(|v| v.to_array())
                            .collect::<Vec<_>>()
                        })
                        .clone()
                };
                for (s, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (gradient(h), gradient(-h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (gradient(2.0 * h), gradient(-2.0 * h))
                    } else {
                        (vec![0.0; 9], vec![0.0; 9])
                    };
                    for a in 0..9 {
                        errors[9 * a + b][s] = (stencil
                            .first(plus[a], minus[a], plus2[a], minus2[a], h)
                            - hessian[(a, b)])
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("finite-k ECP fixed-density Hessian", &failures);
}

#[test]
fn translated_scalar_ecp_elastic_operator_fits_every_component() {
    use crate::periodic::strain_fock::strain_bloch;

    let build = displaced_ecp_cell();
    let cell = build(&LatticePath::default(), 0.0);
    let density = finite_k_density();
    let points = finite_k_points(&cell);
    let weights = finite_k_weights(cell.ecp_image_blocks(), &points, &density);
    let geometry = |cell: &Cell, points: &[Vec3; 3]| {
        contract_ecp_geometry(
            cell.basis(),
            EcpBoundary::Bulk(cell.lattice()),
            cell.overlap_cutoff(),
            3,
            cell.ecp_image_blocks(),
            points,
            &density,
        )
        .unwrap()
    };
    let second = ecp_strain_second(&cell, &weights).unwrap();
    let first = strain_bloch(&ecp_strain_blocks(&cell).unwrap(), points[1], 3);
    let unstrained = geometry(&cell, &points).strain;
    let mut labels = Vec::new();
    let mut exact = Vec::new();
    for column in 0..18 {
        for row in 0..9 {
            let (name, value) = if column < 9 {
                let c = column / 3;
                let d = column % 3;
                // Stress evaluated at the strained cell differentiates
                // (I + delta F)(I + F), so the first virial enters here.
                let chart = if row % 3 == c {
                    unstrained[row / 3][d]
                } else {
                    0.0
                };
                ("strain", second.strain[9 * row + column] + chart)
            } else {
                ("internal strain", second.mixed[9 * row + column - 9])
            };
            labels.push(format!("finite-k scalar ECP {name} [{row},{column}]"));
            exact.push(value);
        }
    }
    for row in 0..9 {
        for axis in 0..3 {
            let uniform = (0..3)
                .map(|atom| second.mixed[9 * row + 3 * atom + axis])
                .sum::<f64>();
            assert!(uniform.abs() < 1e-12, "{row} {axis}: {uniform}");
        }
    }
    let entries: Vec<_> = (0..3)
        .flat_map(|i| {
            (0..=i).flat_map(move |j| (0..1 + usize::from(i != j)).map(move |part| (i, j, part)))
        })
        .collect();
    for (column, matrix) in first.iter().enumerate() {
        for &(i, j, part) in &entries {
            labels.push(format!(
                "finite-k scalar ECP strain Fock [{column},{i},{j},{part}]"
            ));
            exact.push([&matrix.re, &matrix.im][part][(i, j)]);
        }
    }
    assert_eq!(exact.len(), 243);
    let mut cache = BTreeMap::new();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.01, 0.007, 0.005, 0.0035], Stencil::Second),
            ([0.005, 0.0025, 0.00125, 0.000625], Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], Stencil::Fourth),
            ([0.006, 0.004, 0.003, 0.002], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; exact.len()];
            for column in 0..18 {
                let path = if column < 9 {
                    let mut f = [[0.0; 3]; 3];
                    f[column / 3][column % 3] = 1.0;
                    LatticePath {
                        deformation: vec![f],
                        ..LatticePath::default()
                    }
                } else {
                    let mut u = [[0.0; 3]; 3];
                    u[column / 3 - 3][column % 3] = 1.0;
                    LatticePath::displacement(&u)
                };
                let mut at = |h: f64| {
                    *cache.entry((column, h.to_bits())).or_insert_with(|| {
                        let displaced = build(&path, h);
                        let k = finite_k_points(&displaced);
                        let strain = geometry(&displaced, &k).strain;
                        let (re, im) = super::super::bloch(displaced.ecp_image_blocks(), k[1]);
                        let mut values = [0.0; 18];
                        for (value, first) in values.iter_mut().zip(strain.into_iter().flatten()) {
                            *value = first;
                        }
                        for (value, &(i, j, part)) in values[9..].iter_mut().zip(&entries) {
                            *value = [&re, &im][part][(i, j)];
                        }
                        values
                    })
                };
                for (step, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (at(h), at(-h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        ([0.0; 18], [0.0; 18])
                    };
                    for row in 0..if column < 9 { 18 } else { 9 } {
                        let index = if row < 9 {
                            9 * column + row
                        } else {
                            162 + 9 * column + row - 9
                        };
                        let estimate =
                            stencil.first(plus[row], minus[row], plus2[row], minus2[row], h);
                        errors[index][step] = (estimate - exact[index]).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("complete finite-k ECP elastic operator", &failures);
}
