// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full scalar actions share a raw density, not independently converged Focks.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::overlap_matrix;
use dft_rs::linalg::{Matrix, Orthogonalizer};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction_reference::{
    ActiveSpaceScf, FrozenDensityEvaluator, MappedFrozenAction, WeightedFrozenAction,
};
use dft_rs::pseudo_sub::reduction_support::OrthogonalReducedChart;
use dft_rs::scf::{run_scf, EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

fn molecule() -> Molecule {
    Molecule::new(
        [
            [-1.31, 0.14, 0.03],
            [-0.22, 1.18, -0.12],
            [0.83, -0.27, 0.34],
            [1.56, 0.85, -0.43],
        ]
        .into_iter()
        .map(|r| Atom {
            z: 1,
            position: Vec3::new(r[0], r[1], r[2]),
        })
        .collect(),
    )
}

#[test]
fn mapped_scalar_fock_and_relaxed_weight_gradients_all_components() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(384))
        .unwrap()
        .run(|| {
            let mol = molecule();
            let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
            let grid = MolecularGrid::build(
                &mol,
                &GridOptions {
                    level: 1,
                    weight_cutoff: 0.,
                    ..Default::default()
                },
            )
            .unwrap();
            let xc = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
            let options = ScfOptions {
                guess: Guess::Core,
                max_iter: 300,
                e_tol: 1e-14,
                p_tol: 1e-13,
                conv_tol_grad: 1e-12,
                eri_mode: EriMode::InCore { limit_mb: 64 },
                ..Default::default()
            };
            let full = run_scf(&mol, &basis, &options, &xc, Some(&grid)).unwrap();
            let s = overlap_matrix(&basis);
            let mut raw = Matrix::zeros(3, 3);
            let mut cross = Matrix::zeros(4, 3);
            for i in 0..4 {
                for j in 0..3 {
                    cross[(i, j)] = s[(i, j)];
                    if i < 3 {
                        raw[(i, j)] = s[(i, j)];
                    }
                }
            }
            let frozen: Vec<_> = [0.0_f64, 0.23, -0.19]
                .into_iter()
                .map(|angle| {
                    let mut b = Matrix::zeros(4, 1);
                    for i in 0..4 {
                        b[(i, 0)] = angle.cos() * full.mo_coeff[0][(i, 0)]
                            + angle.sin() * full.mo_coeff[0][(i, 1)];
                    }
                    b
                })
                .collect();
            let charts: Vec<_> = frozen
                .iter()
                .map(|b| OrthogonalReducedChart::new(&s, &cross, &raw, b, 1e-10).unwrap())
                .collect();
            let evaluators: Vec<_> = frozen
                .iter()
                .map(|b| {
                    let mut p = b.matmul_transpose_seq(b);
                    for v in p.as_mut_slice() {
                        *v *= 2.;
                    }
                    FrozenDensityEvaluator::new(&mol, &basis, &grid, &xc, &options, &p).unwrap()
                })
                .collect();
            let terms: Vec<_> = evaluators
                .iter()
                .zip(&charts)
                .map(|(e, c)| MappedFrozenAction::new(e, c, 0.).unwrap())
                .collect();
            let weights = [0.46, 0.23, 0.31];
            let action = WeightedFrozenAction::new(&terms, &weights).unwrap();
            let mut p = Matrix::zeros(3, 3);
            for i in 0..3 {
                p[(i, i)] = 0.35 + 0.12 * i as f64;
            }
            p[(0, 1)] = 0.037;
            p[(1, 0)] = 0.037;
            p[(0, 2)] = -0.021;
            p[(2, 0)] = -0.021;
            p[(1, 2)] = 0.013;
            p[(2, 1)] = 0.013;
            let evaluated = action.evaluate(&p).unwrap();
            let scalar_only = action.evaluate_scalar(&p).unwrap();
            assert_eq!(scalar_only.energy, evaluated.evaluation.energy);
            assert_eq!(
                scalar_only.fock.as_slice(),
                evaluated.evaluation.fock.as_slice()
            );
            let pairs: Vec<_> = (0..3).flat_map(|i| (i..3).map(move |j| (i, j))).collect();
            let names: Vec<_> = pairs
                .iter()
                .map(|(i, j)| format!("mapped scalar dE/dP[{i},{j}]"))
                .collect();
            let mut failures = Vec::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    pairs
                        .iter()
                        .map(|&(i, j)| {
                            std::array::from_fn(|k| {
                                let h = steps[k];
                                let energy = |d: f64| {
                                    let mut trial = p.clone();
                                    trial[(i, j)] += d;
                                    if i != j {
                                        trial[(j, i)] += d;
                                    }
                                    action.evaluate(&trial).unwrap().evaluation.energy
                                };
                                let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                                    (energy(2. * h), energy(-2. * h))
                                } else {
                                    (0., 0.)
                                };
                                (stencil.first(energy(h), energy(-h), plus2, minus2, h)
                                    - evaluated.evaluation.fock[(i, j)]
                                        * if i == j { 1. } else { 2. })
                                .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            components::assert_no_failures("every mapped scalar Fock entry", &failures);

            let x = Orthogonalizer::canonical(&raw, 1e-10).unwrap().x;
            let no_frozen = Matrix::zeros(3, 0);
            let h = Matrix::identity(3);
            let problem = ActiveSpaceScf {
                overlap: &raw,
                complement: &x,
                frozen_orbitals: &no_frozen,
                active_electrons: 2,
                options: &options,
                initial_hcore: &h,
            };
            let solve = |q: [f64; 2]| {
                let w = [1. - q[0] - q[1], q[0], q[1]];
                let model = WeightedFrozenAction::new(&terms, &w).unwrap();
                let solution = problem
                    .solve(None, |p| Ok(model.evaluate(p)?.evaluation))
                    .unwrap();
                let value = model.evaluate(&solution.active_density).unwrap();
                println!(
                    "scalar weight q={q:?} E={:.16e} residual={:.12e} trace={:.12e}",
                    value.evaluation.energy,
                    solution.projected_orbital_residual,
                    solution.particle_number_error
                );
                (value.evaluation.energy, value.independent_weight_gradient())
            };
            let q = [0.23, 0.31];
            let (_, g) = solve(q);
            let names: Vec<_> = (0..2)
                .map(|i| format!("relaxed scalar weight[{i}]"))
                .collect();
            let mut failures = Vec::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    (0..2)
                        .map(|i| {
                            std::array::from_fn(|k| {
                                let h = steps[k];
                                let mut plus = q;
                                let mut minus = q;
                                plus[i] += h;
                                minus[i] -= h;
                                let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                                    let mut a = q;
                                    let mut b = q;
                                    a[i] += 2. * h;
                                    b[i] -= 2. * h;
                                    (solve(a).0, solve(b).0)
                                } else {
                                    (0., 0.)
                                };
                                (stencil.first(solve(plus).0, solve(minus).0, plus2, minus2, h)
                                    - g[i])
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            components::assert_no_failures("every relaxed scalar weight derivative", &failures);
            let vertex = WeightedFrozenAction::new(&terms, &[0., 1., 0.])
                .unwrap()
                .evaluate(&p)
                .unwrap();
            let vertex_scalar = WeightedFrozenAction::new(&terms, &[0., 1., 0.])
                .unwrap()
                .evaluate_scalar(&p)
                .unwrap();
            assert_eq!(vertex_scalar.energy, vertex.evaluation.energy);
            assert_eq!(
                vertex_scalar.fock.as_slice(),
                vertex.evaluation.fock.as_slice()
            );
            assert_eq!(
                vertex.evaluation.energy,
                terms[1].evaluate(&p).unwrap().energy
            );
            assert!(WeightedFrozenAction::new(&terms, &[0.4, 0.3, 0.4]).is_err());
            assert!(WeightedFrozenAction::new(&terms, &[-0.1, 0.5, 0.6]).is_err());
            let shifted = MappedFrozenAction::new(&evaluators[0], &charts[0], 7.3)
                .unwrap()
                .evaluate(&p)
                .unwrap();
            let base = terms[0].evaluate(&p).unwrap();
            assert_eq!(shifted.fock.as_slice(), base.fock.as_slice());
            assert_eq!(shifted.nuclear_energy, base.nuclear_energy);
            assert_eq!(shifted.electronic_energy, base.electronic_energy);
            assert_eq!(shifted.reference_energy, 7.3);
            assert!((shifted.energy - base.energy + 7.3).abs() < 1e-14);
            Ok(())
        })
        .unwrap();
}
