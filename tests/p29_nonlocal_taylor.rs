// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::grid::MolecularGrid;
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::nlc::{vv10_fxc_contraction, vv10_potential, Vv10Params};
use dft_rs::response::nonlocal::vv10_taylor;
use dft_rs::system::{Atom, Molecule};

#[test]
fn nonlinear_nonlocal_energy_and_potential_match_independent_derivatives() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid {
        points: vec![
            Vec3::new(0.2, 0.0, -0.4),
            Vec3::new(-0.3, 0.1, 0.5),
            Vec3::new(0.0, 0.0, 1.1),
        ],
        weights: vec![0.3, 0.4, 0.2],
        raw_weights: vec![0.3, 0.4, 0.2],
        atom_of: vec![0, 1, 1],
    };
    let p = Matrix::from_row_major(2, 2, vec![0.8, 0.1, 0.1, 0.6]);
    let d = Matrix::from_row_major(2, 2, vec![0.07, -0.02, -0.02, -0.03]);
    let params = Vv10Params::ORIGINAL;
    let mut path = vec![p.clone(), d.clone()];
    path.resize(5, Matrix::zeros(2, 2));
    let result = vv10_taylor(&mb, &grid, &path, params).unwrap();
    let (energy, fock) = vv10_potential(&mb, &grid, &p, params).unwrap();
    assert!((energy - result.energy_coefficients[0]).abs() < 1e-14);
    assert!(fock.max_abs_diff(&result.fock_coefficients[0]) < 1e-14);
    assert!((fock.frobenius_dot(&d) - result.energy_coefficients[1]).abs() < 1e-14);
    let kernel = vv10_fxc_contraction(&mb, &grid, &p, &d, params).unwrap();
    assert!(kernel.max_abs_diff(&result.fock_coefficients[1]) < 1e-14);
    let at = |t: f64| {
        let mut pt = p.clone();
        for (a, b) in pt.as_mut_slice().iter_mut().zip(d.as_slice()) {
            *a += t * b;
        }
        vv10_potential(&mb, &grid, &pt, params).unwrap()
    };
    let second = |h: f64| {
        let (_, plus) = at(h);
        let (_, minus) = at(-h);
        Matrix::from_row_major(
            2,
            2,
            plus.as_slice()
                .iter()
                .zip(minus.as_slice())
                .zip(fock.as_slice())
                .map(|((&a, &b), &c)| (a + b - 2.0 * c) / (2.0 * h * h))
                .collect(),
        )
    };
    let h = 0.002;
    let a = second(h);
    let b = second(2.0 * h);
    let extrapolated = Matrix::from_row_major(
        2,
        2,
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(&a, &b)| (4.0 * a - b) / 3.0)
            .collect(),
    );
    assert!(extrapolated.max_abs_diff(&result.fock_coefficients[2]) < 1e-11);
    for n in 1..5 {
        assert!(
            (result.energy_coefficients[n]
                - result.fock_coefficients[n - 1].frobenius_dot(&d) / n as f64)
                .abs()
                < 1e-14
        );
    }
}

#[test]
fn molecular_vv10_fock_kernel_fits_all_density_hessian_components() {
    let numbers = [7, 1, 1, 1];
    let positions = [
        [0.13, -0.27, 0.19],
        [1.78, 0.34, -0.08],
        [-0.49, 1.65, 0.41],
        [0.31, -0.78, 1.91],
    ];
    c1_geometry::assert_c1(&numbers, &positions);
    let molecule = Molecule::new(
        numbers
            .into_iter()
            .zip(positions)
            .map(|(z, [x, y, z_coord])| Atom {
                z,
                position: Vec3::new(x, y, z_coord),
            })
            .collect(),
    );
    let mut basis = BasisSet::new("compact ammonia VV10 kernel");
    for (z, exponent) in [(1, 0.43), (7, 0.57)] {
        basis.insert(ElementBasis {
            z,
            shells: vec![Shell {
                l: 0,
                exps: vec![exponent],
                coefs: vec![1.0],
            }],
        });
    }
    let basis = MolBasis::build(&molecule, &basis).unwrap();
    let nao = basis.nao_sph;
    let grid = MolecularGrid::build(
        &molecule,
        &dft_rs::grid::GridOptions {
            level: 0,
            weight_cutoff: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let source_pairs: Vec<_> = (0..nao)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .collect();
    let n = source_pairs.len();
    // Rotate the complete symmetric AO coordinate basis so that every fit
    // probes a mixed variation, including the off-diagonal density entries.
    let modes: Vec<_> = (0..n)
        .map(|k| {
            let mut matrix = Matrix::zeros(nao, nao);
            let scale = if k == 0 { 1.0 } else { 2.0_f64.sqrt() } / (n as f64).sqrt();
            for (index, &(i, j)) in source_pairs.iter().enumerate() {
                let angle = std::f64::consts::PI * (index as f64 + 0.5) * k as f64 / n as f64;
                let value = scale * angle.cos();
                if i == j {
                    matrix[(i, i)] = value;
                } else {
                    matrix[(i, j)] = value * std::f64::consts::FRAC_1_SQRT_2;
                    matrix[(j, i)] = matrix[(i, j)];
                }
            }
            matrix
        })
        .collect();
    for (i, p) in modes.iter().enumerate() {
        for (j, q) in modes.iter().enumerate() {
            assert!((p.frobenius_dot(q) - f64::from(i == j)).abs() < 3e-15);
        }
    }
    let mut density = Matrix::identity(nao);
    for (k, mode) in modes.iter().enumerate() {
        let scale = 0.045 * (0.51 * (k + 1) as f64).cos();
        for (p, d) in density.as_mut_slice().iter_mut().zip(mode.as_slice()) {
            *p += scale * d;
        }
    }
    let slopes: Vec<_> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    for (name, params) in [
        ("omegaB97X-V", Vv10Params::WB97XV),
        ("VV10", Vv10Params::ORIGINAL),
    ] {
        let label = format!("C1 ammonia {name} VV10");
        let kernel: Vec<_> = modes
            .iter()
            .map(|mode| vv10_fxc_contraction(&basis, &grid, &density, mode, params).unwrap())
            .collect();
        let mut reciprocity = 0.0_f64;
        for &(i, j) in &slopes {
            reciprocity = reciprocity.max(
                (modes[j].frobenius_dot(&kernel[i]) - modes[i].frobenius_dot(&kernel[j])).abs(),
            );
        }
        let names: Vec<_> = slopes
            .iter()
            .map(|(i, j)| format!("{label} [{i},{j}]"))
            .collect();
        let potential = |mode: &Matrix, step: f64| {
            let mut trial = density.clone();
            for (p, d) in trial.as_mut_slice().iter_mut().zip(mode.as_slice()) {
                *p += step * d;
            }
            vv10_potential(&basis, &grid, &trial, params).unwrap().1
        };
        let mut richardson = f64::INFINITY;
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.09, 0.06, 0.04, 0.027], stencils::Stencil::Second),
                ([0.036, 0.026, 0.018, 0.012], stencils::Stencil::Second),
                ([0.10, 0.08, 0.06, 0.045], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                // Share each displaced Fock across all of its probe modes.
                let signed: Vec<_> = modes
                    .iter()
                    .map(|mode| {
                        steps.map(|h| {
                            let twice = (stencil == stencils::Stencil::Fourth)
                                .then(|| (potential(mode, 2.0 * h), potential(mode, -2.0 * h)));
                            (potential(mode, h), potential(mode, -h), twice)
                        })
                    })
                    .collect();
                let mut max_error = 0.0_f64;
                let errors = slopes
                    .iter()
                    .map(|&(i, j)| {
                        let exact = modes[j].frobenius_dot(&kernel[i]);
                        let probe = |matrix: &Matrix| modes[j].frobenius_dot(matrix);
                        let estimates = std::array::from_fn::<_, 4, _>(|k| {
                            let (plus, minus, twice) = &signed[i][k];
                            let (plus2, minus2) = twice
                                .as_ref()
                                .map_or((0.0, 0.0), |(p, m)| (probe(p), probe(m)));
                            stencil.first(probe(plus), probe(minus), plus2, minus2, steps[k])
                        });
                        let ratio = (steps[2] / steps[3]).powf(stencil.order());
                        let extrapolated = (ratio * estimates[3] - estimates[2]) / (ratio - 1.0);
                        max_error = max_error.max((extrapolated - exact).abs());
                        estimates.map(|value| (value - exact).abs())
                    })
                    .collect();
                println!("{label} {stencil:?} {steps:?} maximum Richardson error {max_error:.9e}");
                richardson = richardson.min(max_error);
                errors
            },
            &mut failures,
        );
        components::assert_no_failures(&label, &failures);
        println!(
            "{label} independent entries={} reciprocity={reciprocity:.9e}",
            slopes.len()
        );
        assert!(reciprocity < 1e-13 && richardson < 2e-8);
    }
}
