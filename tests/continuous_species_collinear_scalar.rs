// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Every independent spin-density derivative and the unfrozen UKS limit.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::{Matrix, Orthogonalizer};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction_reference::{CollinearActiveSpaceScf, FrozenDensityEvaluator};
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
fn collinear_frozen_scalar_all_twenty_density_components_and_unfrozen_limit() {
    dft_rs::resources::ResourceLimits::new(Some(1),Some(384)).unwrap().run(|| {
        let full=Molecule::new([[-1.31,0.14,0.03],[-0.22,1.18,-0.12],
            [0.83,-0.27,0.34],[1.56,0.85,-0.43]].into_iter().map(|r|Atom{
                z:1,position:Vec3::new(r[0],r[1],r[2])}).collect()).with_charge(1).with_multiplicity(2);
        let basis=MolBasis::build(&full,&data::load("sto-3g")?)?;
        let functional=resolve(&FunctionalSpec::Combined("pbe".into()))?;
        let grid=MolecularGrid::build(&full,&GridOptions{level:1,weight_cutoff:0.,..Default::default()})?;
        let options=ScfOptions{reference:Reference::Uhf,guess:Guess::Core,max_iter:300,
            e_tol:1e-14,p_tol:1e-13,conv_tol_grad:1e-12,eri_mode:EriMode::InCore{limit_mb:64},..Default::default()};
        let reference=run_scf(&full,&basis,&options,&functional,Some(&grid))?;
        let zero=Matrix::zeros(4,4);
        let empty=Matrix::zeros(4,0);
        let unfrozen=FrozenDensityEvaluator::new_collinear(&full,&basis,&grid,&functional,&options,[&zero,&zero])?;
        let direct=unfrozen.evaluate_collinear_with_reference([&reference.density[0],&reference.density[1]],0.)?;
        assert!((direct.energy-reference.energy).abs()<1e-11);
        for spin in 0..2 {assert!(direct.fock[spin].max_abs_diff(&reference.fock[spin])<1e-10);}
        let x=Orthogonalizer::canonical(unfrozen.overlap(),options.orth_threshold)?.x;
        let state=CollinearActiveSpaceScf{overlap:unfrozen.overlap(),complements:[&x,&x],
            frozen_orbitals:[&empty,&empty],initial_hcore:[unfrozen.h_core(),unfrozen.h_core()],
            active_electrons:[2,1],options:&options}.solve(Some([&reference.density[0],&reference.density[1]]),
                |p|unfrozen.evaluate_collinear_with_reference(p,reference.energy))?;
        assert!(state.evaluation.energy.abs()<1e-10);
        assert!(state.projected_orbital_residual<options.conv_tol_grad);
        for spin in 0..2 {assert!(state.active_density[spin].max_abs_diff(&reference.density[spin])<1e-8);}
        println!("COLLINEAR_UNFROZEN energy_difference={:.16e} orbital_residual={:.12e} particle={:?} idempotency={:?}",
            state.evaluation.energy,state.projected_orbital_residual,state.particle_number_error,state.active_idempotency_error);

        // The frozen alpha electron is fixed. Positive definite trial spin
        // densities exercise every symmetric AO derivative off shell, where
        // stationarity cannot hide a missing spin or double-counting factor.
        let mut frozen=Matrix::zeros(4,4);
        for i in 0..4 {for j in 0..4 {frozen[(i,j)]=reference.mo_coeff[0][(i,0)]*reference.mo_coeff[0][(j,0)];}}
        let evaluator=FrozenDensityEvaluator::new_collinear(&full,&basis,&grid,&functional,&options,[&frozen,&zero])?;
        let metric_inverse=x.matmul_transpose_seq(&x);
        let density:[Matrix;2]=std::array::from_fn(|s| {
            let mut p=metric_inverse.clone();for value in p.as_mut_slice(){*value*=[0.36,0.21][s];}p
        });
        let base=evaluator.evaluate_collinear_with_reference([&density[0],&density[1]],reference.energy)?;
        let indices=(0..2).flat_map(|s|(0..4).flat_map(move |i|(i..4).map(move |j|(s,i,j)))).collect::<Vec<_>>();
        let names=indices.iter().map(|(s,i,j)|format!("collinear frozen Fock spin{s} {i},{j}")).collect::<Vec<_>>();
        let mut failures=Vec::new();
        stencils::fit_components_staged(&names,&[
            ([0.04,0.02,0.01,0.005],stencils::Stencil::Second),
            ([0.02,0.01,0.005,0.0025],stencils::Stencil::Second),
            ([0.04,0.02,0.01,0.005],stencils::Stencil::Fourth)],|steps,stencil| {
                indices.iter().map(|&(spin,i,j)| std::array::from_fn(|k| {
                    let energy=|step:f64| {
                        let mut trial=density.clone();trial[spin][(i,j)]+=step;
                        if i!=j {trial[spin][(j,i)]+=step;}
                        evaluator.evaluate_collinear_with_reference([&trial[0],&trial[1]],reference.energy).unwrap().energy
                    };
                    let h=steps[k];let (plus2,minus2)=if stencil==stencils::Stencil::Fourth {(energy(2.*h),energy(-2.*h))}else{(0.,0.)};
                    (stencil.first(energy(h),energy(-h),plus2,minus2,h)-base.fock[spin][(i,j)]*if i==j {1.}else{2.}).abs()
                })).collect()
            },&mut failures);
        components::assert_no_failures("every independent collinear scalar density derivative",&failures);
        // No spin-summed call can silently erase the unrestricted sector.
        assert!(evaluator.evaluate(&density[0]).is_err());
        Ok(())
    }).unwrap();
}
