// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Predefined construction ablation and untouched scaffold holdouts.
//! Chemical gates are measured trial outcomes, never assumptions of this harness.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/reduced_fragment.rs"]
mod fixture;
use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::continuous_species::reduced::{
    evaluate_application, evaluate_candidate, ReducedEvaluation,
};
use dft_rs::gradient::assemble_gradient;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{generate_methyl, ProjectionRule, ReducedFragment};
use dft_rs::pseudo_sub::reduction_application::bind_methyl;
use dft_rs::pseudo_sub::reduction_basis::OneCenterBasisTemplate;
use dft_rs::pseudo_sub::reduction_geometry::RigidExpansion;
use dft_rs::response::diagnostics::reference_diagnostics;
use dft_rs::scf::{run_scf, EriMode, Guess, ScfOptions};
use dft_rs::xc::{resolve, FunctionalSpec};
use dft_rs::{DftError, Result};
use std::sync::OnceLock;
use std::time::Instant;

fn origin() -> Vec3 {
    Vec3::new(0.11, 0.07, -0.09)
}
fn options() -> ScfOptions {
    ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..Default::default()
    }
}
struct Full {
    energy: f64,
    forces: Vec<Vec3>,
    dipole: Vec3,
    ne: usize,
    ao: usize,
    grid: usize,
    residual: f64,
}
fn full(map: &RigidExpansion, y: &[Vec3]) -> Result<Full> {
    let mol = map.expand(y)?;
    let basis = MolBasis::build(&mol, &data::load("sto-3g")?)?;
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.,
            ..Default::default()
        },
    )?;
    let functional = resolve(&FunctionalSpec::Combined("pbe".into()))?;
    let opts = options();
    let scf = run_scf(&mol, &basis, &opts, &functional, Some(&grid))?;
    let diagnostic = reference_diagnostics(&scf)?;
    if !scf.converged || diagnostic.orbital_residual > opts.conv_tol_grad {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: diagnostic.orbital_residual,
        });
    }
    let grad = assemble_gradient(&mol, &basis, &functional, Some(&grid), &scf, &opts)?;
    let forces: Vec<_> = grad.iter().map(|&g| g * (-1.)).collect();
    Ok(Full {
        energy: scf.energy,
        forces: map.pullback_forces(y, &forces)?,
        dipole: dft_rs::properties::dipole_moment(&mol, &basis, &scf),
        ne: mol.n_electrons()?,
        ao: basis.nao_sph,
        grid: grid.len(),
        residual: diagnostic.orbital_residual,
    })
}
#[derive(Debug)]
struct Metrics {
    mae: f64,
    p95: f64,
    max: f64,
    force_rmse: f64,
    force_max: f64,
    dipole_rmse: f64,
    dipole_max: f64,
}
fn report(
    label: &str,
    refs: &[std::result::Result<Full, String>],
    values: &[std::result::Result<ReducedEvaluation, String>],
) -> Option<Metrics> {
    assert_eq!(refs.len(), 9);
    assert_eq!(values.len(), 9);
    let mut failed = false;
    for i in 0..9 {
        if let Err(e) = &refs[i] {
            println!("{label} full-failure point={i} error={e}");
            failed = true;
        }
        if let Err(e) = &values[i] {
            println!("{label} reduced-failure point={i} error={e}");
            failed = true;
        }
    }
    if failed {
        println!("{label} status=numerical-failure metrics=unavailable no-imputation");
        return None;
    }
    let ef = refs[4].as_ref().unwrap().energy;
    let er = values[4].as_ref().unwrap().energy;
    let mut errors = Vec::new();
    let (mut f2, mut fmax, mut d2, mut dmax) = (0., 0_f64, 0., 0_f64);
    let mut nf = 0;
    for i in 0..9 {
        let a = refs[i].as_ref().unwrap();
        let b = values[i].as_ref().unwrap();
        let de = (b.energy - er) - (a.energy - ef);
        errors.push(de.abs());
        assert_eq!(a.forces.len(), b.forces.len());
        let mut pointmax = 0_f64;
        for (f, g) in a.forces.iter().zip(&b.forces) {
            for k in 0..3 {
                let e = g.get(k) - f.get(k);
                f2 += e * e;
                fmax = fmax.max(e.abs());
                pointmax = pointmax.max(e.abs());
                nf += 1;
            }
        }
        let dip = (b.model_dipole - a.dipole).norm();
        d2 += dip * dip;
        dmax = dmax.max(dip);
        println!("{label} point={i} offset={} Efull={:.16e} Ereduced={:.16e} relativeError={de:.16e} forceComponentMax={pointmax:.16e} dipoleVectorError={dip:.16e} fullResidual={:.9e} reducedResidual={:.9e}",fixture::OFFSETS[i],a.energy,b.energy,a.residual,b.scf_orbital_residual);
    }
    errors.sort_by(f64::total_cmp);
    let m = Metrics {
        mae: errors.iter().sum::<f64>() / 9.,
        p95: errors[(0.95_f64 * 9.).ceil() as usize - 1],
        max: errors[8],
        force_rmse: (f2 / nf as f64).sqrt(),
        force_max: fmax,
        dipole_rmse: (d2 / 9.).sqrt(),
        dipole_max: dmax,
    };
    println!("{label} metrics={m:?} units=Eh,Eh/bohr,e*bohr p95=nearest-rank energyMAE_kcal={:.12e} energyP95_kcal={:.12e} gates_MAE_1kcal={} gates_P95_2kcal={} gates_forceRMSE_1e-3={} status={} G=0 frozen_internal_response=absent",m.mae*627.5094740631,m.p95*627.5094740631,m.mae*627.5094740631<=1.,m.p95*627.5094740631<=2.,m.force_rmse<=1e-3,if m.mae*627.5094740631<=1.&&m.p95*627.5094740631<=2.&&m.force_rmse<=1e-3{"trial-gates-pass-not-general-validation"}else{"chemical-gates-fail"});
    // Explicit reads keep all separately reported quantities in the contract.
    assert!([m.max, m.force_max, m.dipole_rmse, m.dipole_max]
        .iter()
        .all(|v| v.is_finite()));
    Some(m)
}
struct Trial {
    model: ReducedFragment,
    generation_seconds: f64,
    metrics: Metrics,
}
struct Construction {
    trials: Vec<Trial>,
    selected: usize,
}
fn construction() -> &'static Construction {
    static RUN: OnceLock<Construction> = OnceLock::new();
    RUN.get_or_init(|| {
        let (source,spec)=fixture::ethane();let map=RigidExpansion::methyl(&source,spec.clone()).unwrap();
        let coordinates=fixture::scan(&map,spec.port.host_atom);
        // One ordinary full reference per geometry, shared by all four trials.
        let refs:Vec<_>=coordinates.iter().map(|y|full(&map,y).map_err(|e|e.to_string())).collect();
        let mut trials=Vec::new();
        for basis in [OneCenterBasisTemplate::Hydrogen1s,OneCenterBasisTemplate::CarbonValenceSp] {
            for rule in [ProjectionRule::FullAoMetric,ProjectionRule::PhysicalSupportPauliNormalized] {
                let label=format!("construction basis={} rule={}",basis.id(),rule.id());
                let mut opts=fixture::construction_options();opts.basis_template=basis;opts.projection_rule=rule;
                let start=Instant::now();let generated=generate_methyl(&source,spec.clone(),opts);let elapsed=start.elapsed().as_secs_f64();
                match generated {
                    Err(e)=>println!("{label} generation-failure={e} seconds={elapsed:.9e}"),
                    Ok(model)=> {
                        let frozen=model.to_json().unwrap();
                        assert_eq!(model.local_amplitudes.len()+model.nonlocal_coefficients.len(),39);
                        assert_eq!(model.gaussian_charges.len()+model.gaussian_body_dipoles.iter().map(|v|v.len()).sum::<usize>(),12);
                        let values:Vec<_>=coordinates.iter().map(|y|evaluate_candidate(&model,y,Vec3::zero(),origin()).map_err(|e|e.to_string())).collect();
                        assert_eq!(model.to_json().unwrap(),frozen);
                        if let Some(metrics)=report(&label,&refs,&values) {trials.push(Trial{model,generation_seconds:elapsed,metrics});}
                    }
                }
            }
        }
        assert!(!trials.is_empty(),"all four predefined construction candidates failed numerically; no selectable model");
        let selected=(0..trials.len()).min_by(|&a,&b|trials[a].metrics.mae.total_cmp(&trials[b].metrics.mae)).unwrap();
        let winner=&trials[selected];
        println!("selected construction-only basis={} rule={} mae={:.16e} generationSeconds={:.12e} diagnostics={:?} holdout_not_consulted=true",winner.model.options.basis_template.id(),winner.model.options.projection_rule.id(),winner.metrics.mae,winner.generation_seconds,winner.model.diagnostics);
        if let Ok(path)=std::env::var("DFT_RS_FRAGMENT_ABLATION_MODEL") {std::fs::write(path,winner.model.to_json().unwrap()).unwrap();}
        Construction{trials,selected}
    })
}
#[test]
fn predefined_four_candidate_construction_then_untouched_scaffold_holdouts() {
    let experiment = construction();
    let model = &experiment.trials[experiment.selected].model;
    let original = model.to_json().unwrap();
    for (name, (source, spec)) in [
        ("propane", fixture::propane()),
        ("isobutane", fixture::isobutane()),
    ] {
        let application = bind_methyl(model, &source, spec).unwrap();
        let y = fixture::scan(
            application.geometry_map(),
            application.application_spec().port.host_atom,
        );
        let refs: Vec<_> = y
            .iter()
            .map(|q| full(application.geometry_map(), q).map_err(|e| e.to_string()))
            .collect();
        let values: Vec<_> = y
            .iter()
            .map(|q| {
                evaluate_application(&application, q, Vec3::zero(), origin())
                    .map_err(|e| e.to_string())
            })
            .collect();
        report(&format!("heldout-{name} unchanged-model"), &refs, &values)
            .expect("heldout transfer must finish numerically; no missing-point acceptance");
        assert_eq!(model.to_json().unwrap(), original);
        assert_eq!(application.original_model_json(), original);
    }
}
fn selected_full_ao_reference_control(model: &ReducedFragment) {
    use dft_rs::pseudo_sub::reduction_reference::{
        extract_methyl_frozen_space, ExtractionOptions, FrozenReferenceModel, MethylBoundary,
    };
    let molecule = &model.source;
    let basis = MolBasis::build(molecule, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        molecule,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let opts = options();
    let full = run_scf(molecule, &basis, &opts, &functional, Some(&grid)).unwrap();
    let diagnostic = reference_diagnostics(&full).unwrap();
    assert!(full.converged && diagnostic.orbital_residual <= opts.conv_tol_grad);
    let s = &model.spec;
    let h: Vec<_> = s
        .explicit_atoms
        .iter()
        .copied()
        .filter(|&i| i != s.port.fragment_atom)
        .collect();
    let selection = extract_methyl_frozen_space(
        molecule,
        &basis,
        &full,
        MethylBoundary {
            carbon: s.port.fragment_atom,
            hydrogens: [h[0], h[1], h[2]],
            port_atom: s.port.host_atom,
        },
        &ExtractionOptions::default(),
    )
    .unwrap();
    let frozen =
        FrozenReferenceModel::new(molecule, &basis, &grid, &functional, &opts, &selection).unwrap();
    let initial = frozen
        .evaluator()
        .evaluate(&selection.active_reference_density)
        .unwrap();
    let solved = frozen.solve(None).unwrap();
    println!("selected fullAO reference_control extraction={:?} Efull={:.16e} Efrozen_at_full_density={:.16e} Efrozen_solved={:.16e} frozen_minus_full={:.16e} fockMaxDefect={:.16e} frozenResidual={:.16e} particleError={:.16e} activeIdempotencyError={:.16e} activeFrozenOverlap={:.16e} numerical_reference_control=true chemical_compression_acceptance=false",selection.diagnostics,full.energy,initial.energy,solved.evaluation.energy,solved.evaluation.energy-full.energy,initial.fock.max_abs_diff(&full.fock[0]),solved.projected_orbital_residual,solved.particle_number_error,solved.active_idempotency_error,solved.active_frozen_overlap);
    assert!((initial.energy - full.energy).abs() < 1e-10);
    assert!(initial.fock.max_abs_diff(&full.fock[0]) < 1e-9);
    assert!((solved.evaluation.energy - full.energy).abs() < 1e-10);
    assert!(solved.projected_orbital_residual < opts.conv_tol_grad);
}

#[test]
fn selected_candidate_full_ao_control_and_every_force_dipole_alpha_component() {
    use dft_rs::driver::continuous_species::reduced::evaluate_candidate_with_response;
    use dft_rs::response::ResponseSolverConfig;
    use std::collections::BTreeMap;
    let run = construction();
    let model = &run.trials[run.selected].model;
    selected_full_ao_reference_control(model);
    let saved = model.to_json().unwrap();
    let y = model.geometry_map.reference_coordinates();
    assert_eq!(y.len(), 5);
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let baseline = evaluate_candidate_with_response(
        model,
        &y,
        field,
        origin(),
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    let alpha = baseline
        .active_model_polarizability
        .expect("active-model analytic static alpha must be present");
    println!("selected derivative reference active-only-alpha={alpha:?} CPKSresiduals={:?} frozen_internal_response=absent",baseline.response_residuals);
    let intervals = [[0.02, 0.01, 0.005, 0.0025], [0.04, 0.02, 0.01, 0.005]];
    let names: Vec<_> = (0..27)
        .map(|i| {
            if i < 15 {
                format!("selected gradient[{},{}]", i / 3, i % 3)
            } else if i < 18 {
                format!("selected dipole[{}]", i - 15)
            } else {
                format!(
                    "selected active-only alpha[{},{}]",
                    (i - 18) / 3,
                    (i - 18) % 3
                )
            }
        })
        .collect();
    let mut cache = BTreeMap::<(usize, u64), ReducedEvaluation>::new();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &intervals,
        |steps| {
            let mut errors = vec![[0.; 4]; 27];
            for component in 0..27 {
                let (channel, analytic) = if component < 15 {
                    (
                        component,
                        baseline.gradient[component / 3].get(component % 3),
                    )
                } else if component < 18 {
                    (
                        15 + component - 15,
                        baseline.model_dipole.get(component - 15),
                    )
                } else {
                    (
                        15 + (component - 18) % 3,
                        alpha[(component - 18) / 3][(component - 18) % 3],
                    )
                };
                let mut estimates = [0.; 4];
                for (k, &h) in steps.iter().enumerate() {
                    let mut value = |amount: f64| {
                        let v=cache.entry((channel,amount.to_bits())).or_insert_with(|| {
                        let mut coords=y.clone();let mut f=field;
                        if channel<15 {let axis=channel%3;let mut p=coords[channel/3].to_array();p[axis]+=amount;coords[channel/3]=Vec3::new(p[0],p[1],p[2]);}
                        else {let mut p=f.to_array();p[channel-15]+=amount;f=Vec3::new(p[0],p[1],p[2]);}
                        evaluate_candidate(model,&coords,f,origin()).unwrap_or_else(|e|panic!("selected derivative channel={channel} amount={amount:.16e}: {e}"))
                    });
                        if component < 18 {
                            v.energy
                        } else {
                            v.model_dipole.get((component - 18) / 3)
                        }
                    };
                    estimates[k] = (value(h) - value(-h)) / (2. * h)
                        * if (15..18).contains(&component) {
                            -1.
                        } else {
                            1.
                        };
                    errors[component][k] = (estimates[k] - analytic).abs();
                }
                let extrapolated = (4. * estimates[3] - estimates[2]) / 3.;
                println!("{} interval={steps:?} analytic={analytic:.16e} finestAbsoluteError={:.16e} RichardsonError={:.16e} no_floor=true",names[component],errors[component][3],(extrapolated-analytic).abs());
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "selected candidate all15 force/3 dipole/9 active alpha components",
        &failures,
    );
    assert_eq!(saved, model.to_json().unwrap());
}
#[test]
fn selected_candidate_three_interleaved_energy_force_timings() {
    let run = construction();
    let trial = &run.trials[run.selected];
    let model = &trial.model;
    let y = model.geometry_map.reference_coordinates();
    let mut tf = Vec::new();
    let mut tr = Vec::new();
    for i in 0..3 {
        let mut pair = |reduced: bool| {
            let start = Instant::now();
            if reduced {
                let v = evaluate_candidate(model, &y, Vec3::zero(), origin()).unwrap();
                tr.push(start.elapsed().as_secs_f64());
                println!(
                    "timing run={i} reduced Ne={} AO={} grid={} occupied={} responseVariables={}",
                    v.active_electrons,
                    v.ao_count,
                    v.grid_points,
                    v.occupied_spatial_orbitals,
                    v.response_variables
                );
            } else {
                let v = full(&model.geometry_map, &y).unwrap();
                tf.push(start.elapsed().as_secs_f64());
                println!(
                    "timing run={i} full Ne={} AO={} grid={} occupied={} responseVariables={}",
                    v.ne,
                    v.ao,
                    v.grid,
                    v.ne / 2,
                    (v.ne / 2) * (v.ao - v.ne / 2)
                );
            }
        };
        pair(i % 2 == 1);
        pair(i % 2 == 0);
    }
    tf.sort_by(f64::total_cmp);
    tr.sort_by(f64::total_cmp);
    println!("energy+forces seconds full_min_median_max={tf:?} reduced_min_median_max={tr:?} repetitions=3 generationSeconds={:.12e} model_generation_excluded=true response_not_timed=true",trial.generation_seconds);
    if tf[1] > tr[1] {
        println!(
            "median amortization_calls={:.12e}",
            trial.generation_seconds / (tf[1] - tr[1])
        );
    } else {
        println!("amortization=unavailable reduced_not_faster=true");
    }
}
