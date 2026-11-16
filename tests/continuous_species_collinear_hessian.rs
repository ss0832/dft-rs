// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Unequal active spin populations: every gradient and ordered Hessian component.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::scalar_geometry::CollinearAffineFrozenGeometry;
use dft_rs::response::krylov::ResponseSolverConfig;
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
fn collinear_affine_unequal_spins_all_nine_and_81_components() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(384))
        .unwrap()
        .run(|| {
            let full = Molecule::new(
                [
                    [-1.31, 0.14, 0.03],
                    [-0.22, 1.18, -0.12],
                    [0.83, -0.27, 0.34],
                    [1.56, 0.85, -0.43],
                    [2.07,-1.13,0.91],
                ]
                .into_iter()
                .map(|r| Atom {
                    z: 1,
                    position: Vec3::new(r[0], r[1], r[2]),
                })
                .collect(),
            )
            .with_multiplicity(2);
            let raw = Molecule::new(full.atoms[..3].to_vec()).with_multiplicity(2);
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
            let mut frozen_beta=Matrix::zeros(5,1);
            let mut frozen = Matrix::zeros(5, 1);
            for i in 0..5 {
                frozen[(i, 0)] = 0.17_f64.cos() * reference.mo_coeff[0][(i, 0)]
                    + 0.17_f64.sin() * reference.mo_coeff[0][(i, 1)];
            }
            for i in 0..5 {frozen_beta[(i,0)]=(-0.13_f64).cos()*reference.mo_coeff[1][(i,0)]+(-0.13_f64).sin()*reference.mo_coeff[1][(i,1)];}
            let model = CollinearAffineFrozenGeometry {
                full_reference: &full,
                full_basis: &fb,
                raw_reference: &raw,
                raw_basis: &rb,
                frozen_reference: [&frozen, &frozen_beta],
                full_atom_owners: &[0, 1, 2, 2, 1],
                functional: &functional,
                options: &options,
                grid_options: &grid_options,
                active_electrons: [2, 1],
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
                .map(|i| format!("collinear affine unequal Cartesian gradient {i}"))
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
            let config=ResponseSolverConfig{tolerance:1e-12,..Default::default()};
            let calculated=model.hessian(&coordinates,&config)?;
            assert_eq!(calculated.directional_results.len(),45);
            assert!(calculated.directional_results[9..].iter().all(|p|p.response_iterations==0),
                "pair curvatures must reuse first-order Cartesian responses and check their original equations");
            for(i,path)in calculated.directional_results.iter().enumerate(){
                println!("COLLINEAR_HESSIAN_PATH {i} first={:.16e} second={:.16e} original_residual={:.12e} relative={:.12e} gradient_defect={:.12e}",
                    path.first_derivative,path.second_derivative,path.orbital_response_residual,path.relative_response_residual,path.gradient_consistency_error);
            }
            let pairs=(0..9).flat_map(|i|(0..9).map(move |j|(i,j))).collect::<Vec<_>>();
            let names=pairs.iter().map(|(i,j)|format!("collinear affine Hessian {i},{j}")).collect::<Vec<_>>();
            let mut failures=Vec::new();
            stencils::fit_components_staged(&names,&[
                ([0.02,0.01,0.005,0.0025],stencils::Stencil::Second),
                ([0.01,0.005,0.0025,0.00125],stencils::Stencil::Second),
                ([0.02,0.01,0.005,0.0025],stencils::Stencil::Fourth)],|steps,stencil|{
                let mut errors=vec![[0.;4];81];
                for column in 0..9 {for k in 0..4 {
                    let h=steps[k];
                    let gradient=|d:f64|{
                        let mut trial=coordinates.clone();match column%3{
                            0=>trial[column/3].x+=d,1=>trial[column/3].y+=d,_=>trial[column/3].z+=d}
                        model.evaluate(&trial,None).unwrap().gradient.iter().flat_map(|g|g.to_array()).collect::<Vec<_>>()
                    };
                    let plus=gradient(h);let minus=gradient(-h);
                    let(plus2,minus2)=if stencil==stencils::Stencil::Fourth{(gradient(2.*h),gradient(-2.*h))}else{(vec![0.;9],vec![0.;9])};
                    for(index,(row,col))in pairs.iter().enumerate(){if *col==column{
                        errors[index][k]=(stencil.first(plus[*row],minus[*row],plus2[*row],minus2[*row],h)-calculated.raw_hessian[(*row,*col)]).abs();
                    }}
                }} errors
            },&mut failures);
            components::assert_no_failures("all ordered collinear affine Hessian elements",&failures);
            assert!(calculated.raw_asymmetry<1e-9);
            let mut translation_error=0_f64;
            for row in 0..9 {for axis in 0..3 {translation_error=translation_error.max((0..3).map(|a|calculated.hessian[(row,3*a+axis)]).sum::<f64>().abs());}}
            assert!(translation_error<1e-8);
            println!("COLLINEAR_HESSIAN raw_asymmetry={:.12e} translation_error={translation_error:.12e}",calculated.raw_asymmetry);
            Ok(())
        }).unwrap();
}
