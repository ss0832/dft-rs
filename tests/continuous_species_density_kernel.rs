// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Every independent AO density-Hessian component against displaced Focks.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::overlap_matrix;
use dft_rs::linalg::{Matrix, Orthogonalizer};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction_reference::FrozenDensityEvaluator;
use dft_rs::response::density_kernel::SemilocalDensityKernel;
use dft_rs::scf::{EriMode, Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

#[test]
fn cached_density_kernel_all_800_ordered_density_hessian_components() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(384))
        .unwrap()
        .run(|| {
            let molecule = Molecule::new(
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
            .with_charge(1)
            .with_multiplicity(2);
            let basis = MolBasis::build(&molecule, &data::load("sto-3g")?)?;
            let grid = MolecularGrid::build(
                &molecule,
                &GridOptions {
                    level: 1,
                    weight_cutoff: 0.,
                    ..Default::default()
                },
            )?;
            let options = ScfOptions {
                reference: Reference::Uhf,
                eri_mode: EriMode::InCore { limit_mb: 64 },
                ..Default::default()
            };
            let x = Orthogonalizer::canonical(&overlap_matrix(&basis), options.orth_threshold)?.x;
            let inverse = x.matmul_transpose_seq(&x);
            let density: [Matrix; 2] = std::array::from_fn(|s| {
                let mut p = inverse.clone();
                for v in p.as_mut_slice() {
                    *v *= [0.36, 0.21][s];
                }
                p
            });
            let zero = Matrix::zeros(4, 4);
            let coordinates = (0..2)
                .flat_map(|s| (0..4).flat_map(move |i| (i..4).map(move |j| (s, i, j))))
                .collect::<Vec<_>>();
            for name in ["svwn5", "pbe"] {
                let functional = resolve(&FunctionalSpec::Combined(name.into()))?;
                let evaluator = FrozenDensityEvaluator::new_collinear(
                    &molecule,
                    &basis,
                    &grid,
                    &functional,
                    &options,
                    [&zero, &zero],
                )?;
                let kernel = SemilocalDensityKernel::new(
                    &molecule,
                    &basis,
                    &grid,
                    &functional,
                    [&density[0], &density[1]],
                    &options,
                )?;
                let analytic = coordinates
                    .iter()
                    .map(|&(spin, i, j)| {
                        let mut change = [zero.clone(), zero.clone()];
                        change[spin][(i, j)] = 1.;
                        change[spin][(j, i)] = 1.;
                        kernel.apply(&change)
                    })
                    .collect::<dft_rs::Result<Vec<_>>>()?;
                let indices = (0..20)
                    .flat_map(|input| (0..20).map(move |output| (input, output)))
                    .collect::<Vec<_>>();
                let names = indices
                    .iter()
                    .map(|(input, output)| {
                        format!("cached density Hessian {name} {output},{input}")
                    })
                    .collect::<Vec<_>>();
                let mut saved = std::collections::HashMap::new();
                let mut failures = Vec::new();
                stencils::fit_components_staged(
                    &names,
                    &[
                        ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                        ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                        ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
                        ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
                    ],
                    |steps, stencil| {
                        let mut finite = Vec::new();
                        for (input, &(spin, i, j)) in coordinates.iter().enumerate() {
                            let mut values = Vec::new();
                            for &h in steps {
                                let shifts = if stencil == stencils::Stencil::Fourth {
                                    vec![h, -h, 2. * h, -2. * h]
                                } else {
                                    vec![h, -h]
                                };
                                let focks = shifts
                                    .iter()
                                    .map(|step| {
                                        saved
                                            .entry((input, step.to_bits()))
                                            .or_insert_with(|| {
                                                let mut p = density.clone();
                                                p[spin][(i, j)] += step;
                                                if i != j {
                                                    p[spin][(j, i)] += step;
                                                }
                                                evaluator
                                                    .evaluate_collinear_with_reference(
                                                        [&p[0], &p[1]],
                                                        0.,
                                                    )
                                                    .unwrap()
                                                    .fock
                                            })
                                            .clone()
                                    })
                                    .collect::<Vec<_>>();
                                values.push(
                                    coordinates
                                        .iter()
                                        .map(|&(s, k, l)| {
                                            stencil.first(
                                                focks[0][s][(k, l)],
                                                focks[1][s][(k, l)],
                                                if focks.len() == 4 {
                                                    focks[2][s][(k, l)]
                                                } else {
                                                    0.
                                                },
                                                if focks.len() == 4 {
                                                    focks[3][s][(k, l)]
                                                } else {
                                                    0.
                                                },
                                                h,
                                            )
                                        })
                                        .collect::<Vec<_>>(),
                                );
                            }
                            finite.push(values);
                        }
                        indices
                            .iter()
                            .map(|&(input, output)| {
                                let (s, i, j) = coordinates[output];
                                // An off-diagonal symmetric density coordinate contributes
                                // two equal Fock elements to the scalar energy gradient.
                                let factor = if i == j { 1. } else { 2. };
                                std::array::from_fn(|k| {
                                    factor
                                        * (finite[input][k][output] - analytic[input][s][(i, j)])
                                            .abs()
                                })
                            })
                            .collect()
                    },
                    &mut failures,
                );
                components::assert_no_failures(
                    "all ordered spin-density Hessian entries",
                    &failures,
                );
            }
            Ok(())
        })
        .unwrap();
}
