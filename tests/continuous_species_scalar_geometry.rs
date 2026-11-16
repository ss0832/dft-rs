// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Every moving-coordinate component of the affine frozen scalar action.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::scalar_geometry::AffineFrozenGeometry;
use dft_rs::scf::{run_scf, EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

#[test]
fn affine_frozen_geometry_all_nine_components_and_density_gradient_parity() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(384))
        .unwrap()
        .run(|| {
            let full = Molecule::new(
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
            );
            let raw = Molecule::new(full.atoms[..3].to_vec()).with_charge(1);
            let set = data::load("sto-3g")?;
            let fb = MolBasis::build(&full, &set)?;
            let rb = MolBasis::build(&raw, &set)?;
            let xc = resolve(&FunctionalSpec::Combined("pbe".into()))?;
            let grid_options = GridOptions {
                level: 1,
                weight_cutoff: 0.,
                ..Default::default()
            };
            let grid = MolecularGrid::build(&full, &grid_options)?;
            let options = ScfOptions {
                guess: Guess::Core,
                max_iter: 300,
                e_tol: 1e-14,
                p_tol: 1e-13,
                conv_tol_grad: 1e-12,
                eri_mode: EriMode::InCore { limit_mb: 64 },
                ..Default::default()
            };
            let reference = run_scf(&full, &fb, &options, &xc, Some(&grid))?;
            // Exact delegation parity with the legacy explicit-weight assembly. This
            // is independent of the new chart response and preserves its old behavior.
            let zero = Matrix::zeros(4, 4);
            let old = dft_rs::gradient::assemble_gradient_weighted(
                &full,
                &fb,
                &xc,
                Some(&grid),
                &reference,
                &options,
                &zero,
            )?;
            let new = dft_rs::gradient::assemble_gradient_density(
                &full,
                &fb,
                &xc,
                Some(&grid),
                [&reference.density[0], &reference.density[1]],
                &options,
                &zero,
            )?;
            assert_eq!(old, new);
            let mut frozen = Matrix::zeros(4, 1);
            for i in 0..4 {
                frozen[(i, 0)] = 0.23_f64.cos() * reference.mo_coeff[0][(i, 0)]
                    + 0.23_f64.sin() * reference.mo_coeff[0][(i, 1)];
            }
            let model = AffineFrozenGeometry {
                full_reference: &full,
                full_basis: &fb,
                raw_reference: &raw,
                raw_basis: &rb,
                frozen_reference: &frozen,
                full_atom_owners: &[0, 1, 2, 2],
                functional: &xc,
                options: &options,
                grid_options: &grid_options,
                active_electrons: 2,
                reference_energy: reference.energy,
            };
            // A generic displacement makes the frozen neutral coefficient span
            // nonstationary in its new metric; it exercises the nonzero B' term.
            let positions = raw
                .atoms
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    a.position
                        + Vec3::new(
                            0.013 * (i + 1) as f64,
                            -0.007 * (i + 2) as f64,
                            0.011 * (i * i + 1) as f64,
                        )
                })
                .collect::<Vec<_>>();
            let evaluated = model.evaluate(&positions, None)?;
            let analytic = evaluated
                .gradient
                .iter()
                .flat_map(|v| v.to_array())
                .collect::<Vec<_>>();
            println!(
                "AFFINE_BASE E={:.16e} orbital={:.12e} chart={:.12e} frozen={:.12e}",
                evaluated.energy,
                evaluated.active_scf.projected_orbital_residual,
                evaluated.chart_metric_error,
                evaluated.frozen_overlap
            );
            let names = (0..9)
                .map(|i| format!("affine frozen Cartesian gradient {i}"))
                .collect::<Vec<_>>();
            let mut failures = Vec::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    (0..9)
                        .map(|i| {
                            std::array::from_fn(|k| {
                                let h = steps[k];
                                let energy = |d: f64| {
                                    let mut trial = positions.clone();
                                    match i % 3 {
                                        0 => trial[i / 3].x += d,
                                        1 => trial[i / 3].y += d,
                                        _ => trial[i / 3].z += d,
                                    };
                                    model.evaluate(&trial, None).unwrap().energy
                                };
                                let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                                    (energy(2. * h), energy(-2. * h))
                                } else {
                                    (0., 0.)
                                };
                                (stencil.first(energy(h), energy(-h), p2, m2, h) - analytic[i])
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            components::assert_no_failures(
                "all affine frozen nuclear gradient components",
                &failures,
            );
            let sum = evaluated.gradient.iter().fold(Vec3::zero(), |a, b| a + *b);
            assert!(sum.norm() < 1e-9, "translation sum {sum:?}");
            Ok(())
        })
        .unwrap();
}
