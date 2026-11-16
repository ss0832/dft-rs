// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Independent nuclear energy differences for both frozen spin charts.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::scalar_geometry::CollinearAffineFrozenGeometry;
use dft_rs::scf::{run_scf, EriMode, Guess, Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

#[test]
fn collinear_affine_gradient_all_nine_components_and_zero_frozen_limit() {
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
            )
            .with_charge(1)
            .with_multiplicity(2);
            let raw = Molecule::new(full.atoms[..3].to_vec()).with_charge(1);
            let set = data::load("sto-3g")?;
            let fb = MolBasis::build(&full, &set)?;
            let rb = MolBasis::build(&raw, &set)?;
            let functional = resolve(&FunctionalSpec::Combined("pbe".into()))?;
            let grid_options = GridOptions {
                level: 1,
                weight_cutoff: 0.,
                ..Default::default()
            };
            let grid = MolecularGrid::build(&full, &grid_options)?;
            let options = ScfOptions {
                reference: Reference::Uhf,
                guess: Guess::Core,
                max_iter: 300,
                e_tol: 1e-14,
                p_tol: 1e-13,
                conv_tol_grad: 1e-12,
                eri_mode: EriMode::InCore { limit_mb: 64 },
                ..Default::default()
            };
            let reference = run_scf(&full, &fb, &options, &functional, Some(&grid))?;
            let empty = Matrix::zeros(4, 0);
            let mut frozen = Matrix::zeros(4, 1);
            for i in 0..4 {
                frozen[(i, 0)] = 0.17_f64.cos() * reference.mo_coeff[0][(i, 0)]
                    + 0.17_f64.sin() * reference.mo_coeff[0][(i, 1)];
            }
            let model = CollinearAffineFrozenGeometry {
                full_reference: &full,
                full_basis: &fb,
                raw_reference: &raw,
                raw_basis: &rb,
                frozen_reference: [&frozen, &empty],
                full_atom_owners: &[0, 1, 2, 2],
                functional: &functional,
                options: &options,
                grid_options: &grid_options,
                active_electrons: [1, 1],
                reference_energy: reference.energy,
            };
            let coordinates = raw
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
            let base = model.evaluate(&coordinates, None)?;
            let analytic = base
                .gradient
                .iter()
                .flat_map(|v| v.to_array())
                .collect::<Vec<_>>();
            println!(
                "COLLINEAR_AFFINE E={:.16e} orbital={:.12e} chart={:?} frozen={:?} counts={:?}",
                base.energy,
                base.active_scf.projected_orbital_residual,
                base.chart_metric_error,
                base.frozen_overlap,
                base.active_scf.evaluation.active_electrons
            );
            let names = (0..9)
                .map(|i| format!("collinear affine Cartesian gradient {i}"))
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
                                let energy = |step: f64| {
                                    let mut r = coordinates.clone();
                                    match i % 3 {
                                        0 => r[i / 3].x += step,
                                        1 => r[i / 3].y += step,
                                        _ => r[i / 3].z += step,
                                    }
                                    model.evaluate(&r, None).unwrap().energy
                                };
                                let h = steps[k];
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
                "all collinear affine nuclear gradient components",
                &failures,
            );
            assert!(
                base.gradient
                    .iter()
                    .fold(Vec3::zero(), |sum, g| sum + *g)
                    .norm()
                    < 1e-9
            );
            let unfrozen = CollinearAffineFrozenGeometry {
                full_reference: &full,
                full_basis: &fb,
                raw_reference: &full,
                raw_basis: &fb,
                frozen_reference: [&empty, &empty],
                full_atom_owners: &[0, 1, 2, 3],
                functional: &functional,
                options: &options,
                grid_options: &grid_options,
                active_electrons: [2, 1],
                reference_energy: reference.energy,
            };
            let point = full.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
            let limit =
                unfrozen.evaluate(&point, Some([&reference.density[0], &reference.density[1]]))?;
            let ordinary = dft_rs::gradient::assemble_gradient(
                &full,
                &fb,
                &functional,
                Some(&grid),
                &reference,
                &options,
            )?;
            assert!(limit.energy.abs() < 1e-10);
            for (computed, expected) in limit.gradient.iter().zip(&ordinary) {
                assert!((*computed - *expected).norm() < 1e-9);
            }
            println!(
                "COLLINEAR_AFFINE_ZERO_FROZEN energy={:.12e} orbital={:.12e}",
                limit.energy, limit.active_scf.projected_orbital_residual
            );
            Ok(())
        })
        .unwrap();
}
