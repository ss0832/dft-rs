// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-full-AO diagnostics separate frozen response from compression errors.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction_reference::{
    extract_methyl_frozen_space, ExtractionOptions, FrozenDensityEvaluator, FrozenReferenceModel,
    MethylBoundary,
};
use dft_rs::scf::{run_scf, EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

fn ethane() -> Molecule {
    let points = [
        [-1.46, -0.14, 0.23],
        [-2.13, 1.69, 0.68],
        [-2.28, -1.57, 1.31],
        [-2.19, -0.43, -1.65],
        [1.35, 0.17, -0.12],
        [2.05, 2.08, 0.16],
        [2.10, -0.92, 1.45],
        [2.01, -0.84, -1.67],
    ];
    Molecule::new(
        points
            .into_iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i == 0 || i == 4 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

#[test]
fn fixed_frozen_field_dipole_all_components_and_exact_limits() {
    dft_rs::resources::ResourceLimits::new(Some(1),Some(384)).unwrap().run(|| {
        let molecule = ethane();
        let basis = MolBasis::build(&molecule,&data::load("sto-3g").unwrap()).unwrap();
        let grid = MolecularGrid::build(&molecule,&GridOptions {
            level:1,weight_cutoff:0.0,..Default::default()
        }).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
        let options = ScfOptions { guess:Guess::Core,max_iter:300,e_tol:1e-13,p_tol:1e-12,
            conv_tol_grad:1e-11,eri_mode:EriMode::InCore{limit_mb:64},..Default::default() };
        let full = run_scf(&molecule,&basis,&options,&functional,Some(&grid)).unwrap();
        let selection = extract_methyl_frozen_space(&molecule,&basis,&full,MethylBoundary {
            carbon:4,hydrogens:[5,6,7],port_atom:0,
        },&ExtractionOptions::default()).unwrap();
        let model = FrozenReferenceModel::new(&molecule,&basis,&grid,&functional,&options,&selection).unwrap();
        let zero = model.solve(None).unwrap();
        let origin = Vec3::new(0.11,-0.07,0.13);
        let zero_field = model.solve_in_electric_field(Vec3::zero(),origin,None).unwrap();
        assert_eq!(zero.evaluation.energy.to_bits(),zero_field.evaluation.energy.to_bits());
        assert_eq!(zero.active_density.as_slice(),zero_field.active_density.as_slice());
        assert!((zero.evaluation.energy-full.energy).abs()<1e-10);
        let field = Vec3::new(0.003,-0.002,0.001);
        let state = model.solve_in_electric_field(field,origin,None).unwrap();
        let dipole = model.evaluator().electric_dipole(&state.active_density,origin).unwrap();
        println!("frozen field E={:.16e} dipole_e_bohr={:?} residual={:.12e} trace_error={:.12e} exclusion={:.12e}",
            state.evaluation.energy,dipole.to_array(),state.projected_orbital_residual,
            state.particle_number_error,state.active_frozen_overlap);
        let names:Vec<_>=(0..3).map(|axis|format!("frozen electric dipole[{axis}]")).collect();
        let mut failures=Vec::new();
        let mut smallest = [f64::INFINITY;3];
        stencils::fit_components_staged(&names,&[
            ([0.02,0.01,0.005,0.0025],stencils::Stencil::Second),
            ([0.005,0.0025,0.00125,0.000625],stencils::Stencil::Second),
            ([0.02,0.01,0.005,0.0025],stencils::Stencil::Fourth),
        ],|steps,stencil| {
            (0..3).map(|axis| std::array::from_fn(|k| {
                let h=steps[k];
                let energy=|step:f64| {
                    let mut shifted=field;
                    match axis {0=>shifted.x+=step,1=>shifted.y+=step,_=>shifted.z+=step}
                    let trial=model.solve_in_electric_field(shifted,origin,None).unwrap();
                    println!("frozen field oracle axis={axis} step={step:.12e} E={:.16e} residual={:.12e}",trial.evaluation.energy,trial.projected_orbital_residual);
                    trial.evaluation.energy
                };
                let plus=energy(h);let minus=energy(-h);
                let (plus2,minus2)=if stencil==stencils::Stencil::Fourth {(energy(2.0*h),energy(-2.0*h))}else{(0.0,0.0)};
                let error=(stencil.first(plus,minus,plus2,minus2,h)+dipole.get(axis)).abs();
                smallest[axis]=smallest[axis].min(error);
                error
            })).collect()
        },&mut failures);
        components::assert_no_failures("fixed frozen field all dipole components",&failures);
        // The fitted coarse interval resolves the asymptotic order. Retain its
        // errors and also check the absolute target at a finer, explicit step.
        for (axis, error) in smallest.iter_mut().enumerate() {
            let h=0.000625;
            let energy=|sign:f64| {
                let mut shifted=field;
                match axis {0=>shifted.x+=sign*h,1=>shifted.y+=sign*h,_=>shifted.z+=sign*h}
                model.solve_in_electric_field(shifted,origin,None).unwrap().evaluation.energy
            };
            *error=((energy(1.0)-energy(-1.0))/(2.0*h)+dipole.get(axis)).abs();
            println!("frozen dipole absolute check axis={axis} h={h:.12e} error={error:.12e}");
        }
        assert!(smallest.iter().all(|e|*e<1e-6),"dipole absolute errors {smallest:?}");
        let zero_frozen=FrozenDensityEvaluator::new(&molecule,&basis,&grid,&functional,&options,&Matrix::zeros(basis.nao_sph,basis.nao_sph)).unwrap();
        let mut field_options=options.clone();field_options.external_field=Some(field);
        let field_full=run_scf(&molecule,&basis,&field_options,&functional,Some(&grid)).unwrap();
        println!("matched fixed-AO field comparison full_E={:.16e} frozen_E={:.16e} full_delta_E={:.16e} frozen_delta_E={:.16e} full_dipole_e_bohr={:?} frozen_dipole_e_bohr={:?}",
            field_full.energy,state.evaluation.energy,field_full.energy-full.energy,
            state.evaluation.energy-zero.evaluation.energy,
            dft_rs::properties::dipole_moment(&molecule,&basis,&field_full).to_array(),dipole.to_array());
        let zero_frozen_value=zero_frozen.evaluate_in_electric_field(&field_full.p_total(),field,origin).unwrap();
        let energy_error=(zero_frozen_value.energy-field_full.energy).abs();
        let fock_error=zero_frozen_value.fock.max_abs_diff(&field_full.fock[0]);
        // Neutral total energy and dipole are origin invariant. The AO field
        // operator at a displaced origin differs by -(E dot origin) S.
        let expected_fock=zero_frozen.evaluate_in_electric_field(&field_full.p_total(),field,Vec3::zero()).unwrap();
        assert!(energy_error<1e-10);
        assert!(expected_fock.fock.max_abs_diff(&field_full.fock[0])<1e-10);
        assert!((zero_frozen.electric_dipole(&field_full.p_total(),origin).unwrap()
            - dft_rs::properties::dipole_moment(&molecule,&basis,&field_full)).norm()<1e-10);
        println!("zero-frozen field limit E_error={energy_error:.12e} displaced_origin_fock_shift={fock_error:.12e}");
        assert!(model.solve_in_electric_field(Vec3::new(f64::NAN,0.0,0.0),origin,None).is_err());
        Ok(())
    }).unwrap();
}
