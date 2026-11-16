// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every Cartesian gradient and full affine-strain component of a real ROKS SCF.
#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/stencils.rs"]
pub mod stencils;

use dft_rs::driver::periodic::{prepare_periodic, PeriodicMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::periodic::roks::GammaRoksProblem;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use stencils::Stencil;

fn system() -> System {
    System {
        numbers: vec![1; 3],
        positions: vec![[0.13, -0.21, 0.07], [1.91, 0.20, -0.19], [0.4, 1.7, 0.33]],
        cell: Some([[8.5, 0.0, 0.0], [0.31, 9.3, 0.0], [0.27, -0.19, 10.1]]),
        pbc: [true; 3],
    }
}

fn method(xc: &str) -> PeriodicMethod {
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", xc)
            .with("basis", "sto-3g")
            .with("dispersion", "none")
            .with("multiplicity", 2)
            .with("overlap_cutoff", 6.0)
            .with("grid_level", 1)
            .with("grid_shell", 1)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("conv_tol_dp", 1e-11)
            .with("conv_tol_grad", 1e-10)
            .with("max_iter", 300)
            .with("guess", "core")
            .with("damping", 0.0)
            .with("no_adiis", true),
    )
    .unwrap();
    PeriodicMethod::from_controls(&controls).unwrap()
}

fn displaced(kind: usize, index: usize, amount: f64) -> System {
    let mut next = system();
    if kind == 0 {
        next.positions[index / 3][index % 3] += amount;
    } else {
        let a = index / 3;
        let b = index % 3;
        let original = next.clone();
        for (position, old) in next.positions.iter_mut().zip(&original.positions) {
            position[a] += amount * old[b];
        }
        for (row, old) in next
            .cell
            .as_mut()
            .unwrap()
            .iter_mut()
            .zip(original.cell.unwrap())
        {
            row[a] += amount * old[b];
        }
    }
    next
}

#[derive(Clone, Copy)]
struct Sample {
    energy: f64,
    orbital: f64,
    density: f64,
    energy_change: f64,
    metric: f64,
    iterations: usize,
}

fn sample(xc: &str, kind: usize, index: usize, amount: f64) -> Sample {
    let checkpoint = std::env::var_os("DFT_RS_ROKS_VALIDATION_DIR").map(|directory| {
        let path = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&path).unwrap();
        path.join(format!(
            "{xc}-{kind}-{index}-{:016x}.json",
            amount.to_bits()
        ))
    });
    let identity = std::env::var("DFT_RS_ROKS_VALIDATION_ID").unwrap_or_default();
    if let Some(path) = checkpoint.as_ref().filter(|p| p.exists()) {
        assert!(
            !identity.is_empty(),
            "Reused scientific checkpoints require an exact source/library identity"
        );
        let record = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(
            record.get("identity").and_then(Value::as_str),
            Some(identity.as_str())
        );
        let number = |key| record.get(key).and_then(Value::as_f64).unwrap();
        let restored = Sample {
            energy: number("energy"),
            orbital: number("orbital"),
            density: number("density"),
            energy_change: number("energy_change"),
            metric: number("metric"),
            iterations: number("iterations") as usize,
        };
        assert!(
            restored.energy.is_finite()
                && restored.orbital < 1e-10
                && restored.density < 1e-11
                && restored.energy_change < 1e-13
                && restored.metric < 1e-12
        );
        println!("ROKS_FIRST_REUSED {xc} kind={kind} component={index} displacement={amount}");
        return restored;
    }
    let mut setup = prepare_periodic(&displaced(kind, index, amount), &method(xc)).unwrap();
    if xc == "pbe" {
        setup.options.xc_ao_cutoff = Some(16.0);
    }
    let problem = GammaRoksProblem::new(&setup.cell, &setup.functional, &setup.options).unwrap();
    let result = match problem.run() {
        Ok(result) => result,
        Err(error) => {
            if let Some(path) = &checkpoint {
                let record = Value::object()
                    .with("identity", identity.as_str())
                    .with("xc", xc)
                    .with("kind", kind)
                    .with("component", index)
                    .with("displacement", amount)
                    .with("error", error.to_string());
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                std::fs::write(
                    path.with_extension(format!("failure-{stamp}.json")),
                    record.to_json(),
                )
                .unwrap();
            }
            panic!("ROKS sample {xc} kind={kind} component={index} displacement={amount} failed: {error}");
        }
    };
    assert!(result.state.energy.is_finite() && result.state.orbital_residual < 1e-10);
    assert!(result.density_change < 1e-11 && result.energy_change < 1e-13);
    assert!(result.state.metric_defect < 1e-12);
    let sample = Sample {
        energy: result.state.energy,
        orbital: result.state.orbital_residual,
        density: result.density_change,
        energy_change: result.energy_change,
        metric: result.state.metric_defect,
        iterations: result.iterations,
    };
    if let Some(path) = checkpoint {
        assert!(!identity.is_empty());
        let record = Value::object()
            .with("identity", identity.as_str())
            .with("xc", xc)
            .with("kind", kind)
            .with("component", index)
            .with("displacement", amount)
            .with("energy", sample.energy)
            .with("orbital", sample.orbital)
            .with("density", sample.density)
            .with("energy_change", sample.energy_change)
            .with("metric", sample.metric)
            .with("iterations", sample.iterations);
        let temporary = path.with_extension("tmp");
        std::fs::write(&temporary, record.to_json()).unwrap();
        std::fs::rename(temporary, path).unwrap();
    }
    sample
}

#[test]
fn every_roks_cartesian_and_affine_strain_component_has_independent_four_step_fits() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        for xc in ["hf","pbe"] {
            let mut setup=prepare_periodic(&system(),&method(xc)).unwrap();
            if xc == "pbe" { setup.options.xc_ao_cutoff=Some(16.0); }
            let problem=GammaRoksProblem::new(&setup.cell,&setup.functional,&setup.options).unwrap();
            let result=problem.run().unwrap();
            let gradient=problem.nuclear_gradient(&result).unwrap();
            let stress=problem.stress(&result).unwrap();
            assert_eq!(gradient.len(),3);
            if xc == "pbe" {
                let mut range_options=setup.options.clone();range_options.xc_ao_cutoff=Some(20.0);
                let range_problem=GammaRoksProblem::new(&setup.cell,&setup.functional,&range_options).unwrap();
                let range_reference=range_problem.run().unwrap();
                let range_gradient=range_problem.nuclear_gradient(&range_reference).unwrap();
                let range_stress=range_problem.stress(&range_reference).unwrap();
                let de=(range_reference.state.energy-result.state.energy).abs();
                let dg=gradient.iter().zip(&range_gradient).map(|(a,b)|(*a-*b).norm()).fold(0.0_f64,f64::max);
                let ds=stress.strain_derivative.iter().flatten().zip(range_stress.strain_derivative.iter().flatten()).map(|(a,b)|(a-b).abs()).fold(0.0_f64,f64::max);
                println!("ROKS_AO_RANGE_CONVERGENCE from=16 to=20 energy={de} gradient={dg} strain={ds}");
                assert!(de<1e-13 && dg<1e-10 && ds<1e-10, "AO range must converge before derivative fits");
                assert!(range_reference.density_change<1e-11 && range_reference.energy_change<1e-13);
            }
            assert_eq!(gradient.len(),3);
            let mut analytic:Vec<f64>=gradient.iter().flat_map(|g|g.to_array()).collect();
            analytic.extend(stress.strain_derivative.into_iter().flatten());
            assert_eq!(analytic.len(),18);
            assert!(analytic.iter().all(|x|x.is_finite() && x.abs()>1e-8));
            let sum=gradient.iter().fold(dft_rs::math::Vec3::zero(),|a,b|a+*b);
            assert!(sum.norm()<1e-6);
            println!("ROKS_FIRST_REFERENCE {xc} energy={} residual={} translation={} rotation={}",result.state.energy,result.state.orbital_residual,sum.norm(),stress.rotation_residual);
            for a in 0..3 {for b in 0..3 {
                let terms:f64=stress.terms.iter().map(|(_,t)|t[a][b]).sum();
                assert!((terms-stress.strain_derivative[a][b]).abs()<1e-12);
                let expected=0.5*(stress.strain_derivative[a][b]+stress.strain_derivative[b][a])/stress.volume;
                assert!((stress.stress[a][b]-expected).abs()<1e-14);
            }}
            let names:Vec<String>=(0..18).map(|i|if i<9 {
                format!("ROKS {xc} gradient[{},{}]",i/3,i%3)
            } else {format!("ROKS {xc} strain[{},{}]",(i-9)/3,(i-9)%3)}).collect();
            let stages=[([0.02,0.01,0.005,0.0025],Stencil::Second),
                        ([0.01,0.005,0.0025,0.00125],Stencil::Second),
                        ([0.08,0.04,0.02,0.01],Stencil::Second),
                        ([0.04,0.03,0.02,0.01],Stencil::Fourth)];
            let mut cache:BTreeMap<(usize,usize,u64),Sample>=BTreeMap::new();
            let mut failures=Vec::new();
            stencils::fit_components_staged(&names,&stages,|steps,stencil| {
                let mut needed=BTreeSet::new();
                for kind in 0..2 {for index in 0..9 {for &h in steps {
                    for scale in [-1.0,1.0] {
                        needed.insert((kind,index,(h*scale).to_bits()));
                        if stencil==Stencil::Fourth {needed.insert((kind,index,(2.0*h*scale).to_bits()));}
                    }
                }}}
                let jobs:Vec<_>=needed.into_iter().filter(|key|!cache.contains_key(key)).collect();
                let completed:Vec<_>=jobs.par_iter().map(|&(kind,index,bits)| {
                    let value=sample(xc,kind,index,f64::from_bits(bits));
                    ((kind,index,bits),value)
                }).collect();
                for (key,value) in completed {
                    println!("ROKS_FIRST_SAMPLE {xc} kind={} component={} displacement={} energy={} orbital={} density={} energy_change={} metric={} iterations={}",key.0,key.1,f64::from_bits(key.2),value.energy,value.orbital,value.density,value.energy_change,value.metric,value.iterations);
                    cache.insert(key,value);
                }
                (0..18).map(|component| {
                    let kind=usize::from(component>=9);let index=component%9;
                    std::array::from_fn(|i| {
                        let h=steps[i];
                        let ep=cache[&(kind,index,h.to_bits())].energy;
                        let em=cache[&(kind,index,(-h).to_bits())].energy;
                        let (ep2,em2)=if stencil==Stencil::Fourth {
                            (cache[&(kind,index,(2.0*h).to_bits())].energy,cache[&(kind,index,(-2.0*h).to_bits())].energy)
                        } else {(ep,em)};
                        let estimate=stencil.first(ep,em,ep2,em2,h);
                        assert!(estimate.is_finite());
                        println!("ROKS_FIRST_SIGNED {xc} component={component} h={h} stencil={stencil:?} analytic={} reference={} error={}",analytic[component],estimate,(estimate-analytic[component]).abs());
                        (estimate-analytic[component]).abs()
                    })
                }).collect()
            },&mut failures);
            components::assert_no_failures(&format!("all ROKS {xc} first derivatives"),&failures);
        }
    });
}
