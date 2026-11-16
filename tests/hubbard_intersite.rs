// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent projected-density and variational tests for DFT+U+V.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::linalg::Matrix;
use dft_rs::periodic::BlochMatrix;
use dft_rs::scf::hubbard::{HubbardModel, Interaction, IntersiteInteraction, Site};

const V: f64 = 0.27;
const S: [[f64; 3]; 3] = [[1.21, 0.0, 0.12], [0.0, 0.81, -0.07], [0.12, -0.07, 1.44]];

fn model() -> HubbardModel {
    let overlap = Matrix::from_row_major(3, 3, S.into_iter().flatten().collect());
    HubbardModel::new(
        &overlap,
        [vec![0, 1], vec![2]]
            .into_iter()
            .map(|orbitals| Site {
                orbitals,
                interaction: Interaction::Dudarev { u_eff: 0.0 },
            })
            .collect(),
        1e-10,
    )
    .unwrap()
    .with_intersite(vec![IntersiteInteraction {
        sites: [1, 0],
        v: V,
    }])
    .unwrap()
}

fn bare_projected(block: &Matrix, left: usize, right: usize) -> f64 {
    (0..3)
        .flat_map(|i| (0..3).map(move |j| S[i][left] * block[(i, j)] * S[j][right]))
        .sum::<f64>()
        / (S[left][left] * S[right][right]).sqrt()
}

#[test]
fn all_collinear_hubbard_uv_field_and_kernel_components_have_independent_fits() {
    let model = model();
    let density: [Matrix; 2] = std::array::from_fn(|spin| {
        let d = 0.4 - 0.07 * spin as f64;
        Matrix::from_row_major(
            3,
            3,
            vec![d, 0.04, 0.18, 0.04, d + 0.06, -0.13, 0.18, -0.13, d + 0.02],
        )
    });
    let reference = model.evaluate(&density).unwrap();
    let bonds: f64 = density
        .iter()
        .flat_map(|p| (0..2).map(|i| bare_projected(p, i, 2).powi(2)))
        .sum();
    let left = density
        .iter()
        .flat_map(|p| (0..2).map(|i| bare_projected(p, i, i)))
        .sum::<f64>();
    let right = density.iter().map(|p| bare_projected(p, 2, 2)).sum::<f64>();
    let pair = &reference.pairs[0];
    assert_eq!(pair.pair.sites, [0, 1]);
    assert!((pair.double_counting - V * left * right).abs() < 5e-16);
    assert!((pair.interaction - V * (left * right - bonds)).abs() < 5e-16);
    assert!((reference.energy + V * bonds).abs() < 5e-16);
    println!(
        "DFT+U+V intersite interaction={:.16e} double_counting={:.16e} correction={:.16e}",
        pair.interaction, pair.double_counting, reference.energy
    );
    let entries: Vec<_> = (0..2)
        .flat_map(|s| (0..3).flat_map(move |i| (0..=i).map(move |j| (s, i, j))))
        .collect();
    let curve = &density;
    let stages = [
        ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
        ([0.16, 0.08, 0.04, 0.02], stencils::Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    for &(s, i, j) in &entries {
        let mut direction: [Matrix; 2] = std::array::from_fn(|_| Matrix::zeros(3, 3));
        direction[s][(i, j)] = 1.0;
        direction[s][(j, i)] = 1.0;
        let kernel = model.response(&direction).unwrap();
        let e_derivative = reference.potential[s].frobenius_dot(&direction[s]);
        let mut names = vec![format!("E density source {s}/{i}/{j}")];
        names.extend(
            entries
                .iter()
                .map(|(s2, i2, j2)| format!("F density source {s}/{i}/{j} output {s2}/{i2}/{j2}")),
        );
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let points = steps.map(|h| {
                    // A known nonzero higher tangent makes this polynomial
                    // functional's centered-stencil truncation measurable.
                    let sample = |t: f64| {
                        let changed = std::array::from_fn(|spin| {
                            let mut p = density[spin].clone();
                            for (k, value) in p.as_mut_slice().iter_mut().enumerate() {
                                *value += t * direction[spin].as_slice()[k]
                                    + (t.powi(3) + t.powi(5)) * curve[spin].as_slice()[k];
                            }
                            p
                        });
                        model.evaluate(&changed).unwrap()
                    };
                    let (plus, minus) = (sample(h), sample(-h));
                    let (plus2, minus2) = (sample(2.0 * h), sample(-2.0 * h));
                    let error =
                        |a, b, c, d, exact: f64| (stencil.first(a, b, c, d, h) - exact).abs();
                    let mut out = vec![error(
                        plus.energy,
                        minus.energy,
                        plus2.energy,
                        minus2.energy,
                        e_derivative,
                    )];
                    out.extend(entries.iter().map(|&(s, i, j)| {
                        error(
                            plus.potential[s][(i, j)],
                            minus.potential[s][(i, j)],
                            plus2.potential[s][(i, j)],
                            minus2.potential[s][(i, j)],
                            kernel[s][(i, j)],
                        )
                    }));
                    out
                });
                (0..names.len())
                    .map(|i| std::array::from_fn(|h| points[h][i]))
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures(
        "full collinear intersite Hubbard energy/Fock kernel",
        &failures,
    );
}

#[test]
fn complex_spinor_hubbard_uv_matches_closed_coherence_and_all_kernel_entries() {
    let model = model();
    let mut density = BlochMatrix::zeros(6, 6);
    for i in 0..6 {
        density.re[(i, i)] = 0.23 + 0.03 * i as f64;
        for j in 0..i {
            let re = 0.014 * (1 + i + j) as f64;
            let im = 0.007 * (i - j) as f64;
            density.re[(i, j)] = re;
            density.re[(j, i)] = re;
            density.im[(i, j)] = im;
            density.im[(j, i)] = -im;
        }
    }
    let expected_coherence = |p: &BlochMatrix, s: usize, t: usize, i| {
        [&p.re, &p.im].map(|source| {
            let block = Matrix::from_row_major(
                3,
                3,
                (0..3)
                    .flat_map(|a| (0..3).map(move |b| source[(3 * s + a, 3 * t + b)]))
                    .collect(),
            );
            bare_projected(&block, i, 2)
        })
    };
    let bonds = (0..2)
        .flat_map(|s| (0..2).flat_map(move |t| (0..2).map(move |i| (s, t, i))))
        .map(|(s, t, i)| {
            expected_coherence(&density, s, t, i)
                .iter()
                .map(|v| v.powi(2))
                .sum::<f64>()
        })
        .sum::<f64>();
    let initial = model.evaluate_spinor(&density).unwrap();
    assert!((initial.energy + V * bonds).abs() < 1e-15);
    let entries: Vec<_> = (0..6)
        .flat_map(|i| {
            (0..=i)
                .map(move |j| (i, j, false))
                .chain((0..i).map(move |j| (i, j, true)))
        })
        .collect();
    let stages = [
        ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
        ([0.16, 0.08, 0.04, 0.02], stencils::Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    let value = |matrix: &BlochMatrix, (i, j, imaginary)| {
        if imaginary {
            matrix.im[(i, j)]
        } else {
            matrix.re[(i, j)]
        }
    };
    for (source, &(i, j, imaginary)) in entries.iter().enumerate() {
        let mut direction = BlochMatrix::zeros(6, 6);
        if imaginary {
            direction.im[(i, j)] = 1.0;
            direction.im[(j, i)] = -1.0;
        } else {
            direction.re[(i, j)] = 1.0;
            direction.re[(j, i)] = 1.0;
        }
        let kernel = model.response_spinor(&direction).unwrap();
        let e_derivative = initial.potential.trace_product(&direction);
        let mut names = vec![format!("spinor E source {source}")];
        names.extend(
            (0..entries.len()).map(|target| format!("spinor F source {source} output {target}")),
        );
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let points = steps.map(|h| {
                    let sample = |t: f64| {
                        let mut shifted = density.clone();
                        shifted.add_scaled(&direction, t);
                        shifted.add_scaled(&density, t.powi(3) + t.powi(5));
                        let out = model.evaluate_spinor(&shifted).unwrap();
                        let mut values = vec![out.energy];
                        values.extend(entries.iter().map(|&entry| value(&out.potential, entry)));
                        values
                    };
                    let (plus, minus) = (sample(h), sample(-h));
                    let (plus2, minus2) = (sample(2.0 * h), sample(-2.0 * h));
                    let expected = std::iter::once(e_derivative)
                        .chain(entries.iter().map(|&entry| value(&kernel, entry)));
                    expected
                        .enumerate()
                        .map(|(k, e)| {
                            (stencil.first(plus[k], minus[k], plus2[k], minus2[k], h) - e).abs()
                        })
                        .collect::<Vec<_>>()
                });
                (0..names.len())
                    .map(|i| std::array::from_fn(|h| points[h][i]))
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures(
        "full spinor intersite Hubbard energy/Fock kernel",
        &failures,
    );
}
