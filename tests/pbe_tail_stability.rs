// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Low-density PBE correlation energy, potential and kernel stability.
use dft_rs::xc::{resolve, FunctionalSpec, GridVars, ResolvedFunctional, XcOut, XcTaylorInput};
#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/stencils.rs"]
pub mod stencils;

struct Case {
    point: [f64; 7],
    energy: f64,
    gradient: [f64; 5],
    hessian: [[f64; 5]; 5],
}
struct Evaluation {
    energy: f64,
    gradient: [f64; 7],
    hessian: [[f64; 7]; 7],
}

fn energy(functional: &ResolvedFunctional, point: [f64; 7]) -> f64 {
    let rho = [[point[0]], [point[1]]];
    let sigma = [[point[2]], [point[3]], [point[4]]];
    let vars = GridVars {
        rho: [&rho[0], &rho[1]],
        sigma: Some([&sigma[0], &sigma[1], &sigma[2]]),
        tau: None,
    };
    let mut out = XcOut::default();
    functional.eval(&vars, 0, &mut out).unwrap();
    out.e[0]
}
fn evaluate(functional: &ResolvedFunctional, point: [f64; 7], runtime: bool) -> Evaluation {
    if runtime {
        let input = XcTaylorInput {
            variables: point.map(|x| vec![x]),
            parameters: None,
        };
        let value = functional.semilocal_taylor_potentials(&input).unwrap();
        let columns: [_; 7] = std::array::from_fn(|column| {
            let direction = std::array::from_fn(|i| vec![f64::from(i == column)]);
            functional
                .semilocal_taylor_directional_potentials(&input, &direction)
                .unwrap()
        });
        Evaluation {
            energy: value.energy[0],
            gradient: std::array::from_fn(|i| value.potentials[i][0]),
            hessian: std::array::from_fn(|i| {
                std::array::from_fn(|j| columns[j].directional.potentials[i][0])
            }),
        }
    } else {
        let rho = [[point[0]], [point[1]]];
        let sigma = [[point[2]], [point[3]], [point[4]]];
        let vars = GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: Some([&sigma[0], &sigma[1], &sigma[2]]),
            tau: None,
        };
        let mut value = XcOut::default();
        functional.eval(&vars, 2, &mut value).unwrap();
        let (gradient, hessian) = value.potential_and_kernel(0, functional.family).unwrap();
        Evaluation {
            energy: value.e[0],
            gradient,
            hessian,
        }
    }
}
fn relative(actual: f64, expected: f64) {
    assert!(actual.is_finite() && expected != 0.0);
    assert!(
        ((actual - expected) / expected).abs() < 2e-9,
        "{actual:.17e} versus {expected:.17e}"
    );
}

#[test]
fn pbe_tail_all_active_dense_and_runtime_derivatives() {
    dft_rs::resources::ResourceLimits::new(Some(1),Some(512)).unwrap().install(||{
        let functional=resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
        // Independent 80-digit scalar and second-order calculus at two
        // physical grid descriptors. Both spin densities are below the
        // per-channel cutoff, while their sum is above the total cutoff.
        let cases=[Case { point:[4.9562019750695374e-14,5.732908979697562e-14,8.170710334311418e-26,9.396871571416597e-26,1.0807076366137188e-25,0.0,0.0], energy:-4.099519953301759e-34, gradient:[-2.5851358793642213e-20,-2.522220090414749e-20,2.1706930844090238e-9,4.3413861688180476e-9,2.1706930844090238e-9], hessian:[[-1.3383236028616348e-6,-1.3917437774188453e-6,136882.77230768066,273765.5446153613,136882.77230768066],[-1.3917437774188453e-6,-1.2834930489611362e-6,133551.38547875866,267102.7709575173,133551.38547875866],[136882.77230768066,133551.38547875866,-1.7240708133056596e16,-3.448141626611319e16,-1.7240708133056596e16],[273765.5446153613,267102.7709575173,-3.448141626611319e16,-6.896283253222638e16,-3.448141626611319e16],[136882.77230768066,133551.38547875866,-1.7240708133056596e16,-3.448141626611319e16,-1.7240708133056596e16]]},Case { point:[5.5061236925026886e-14,6.348938145624551e-14,9.969971974278719e-26,1.1496147593150455e-25,1.3255947278915504e-25,0.0,0.0], energy:-5.452525713919299e-34, gradient:[-3.099139768190451e-20,-3.025301543502952e-20,2.3594705022650224e-9,4.718941004530045e-9,2.3594705022650224e-9], hessian:[[-1.4462424021818797e-6,-1.5046694776050919e-6,134109.0208826859,268218.0417653718,134109.0208826859],[-1.5046694776050919e-6,-1.3882252773730003e-6,130913.82067719041,261827.64135438082,130913.82067719041],[134109.0208826859,130913.82067719041,-1.5315198890801106e16,-3.0630397781602212e16,-1.5315198890801106e16],[268218.0417653718,261827.64135438082,-3.0630397781602212e16,-6.1260795563204424e16,-3.0630397781602212e16],[134109.0208826859,130913.82067719041,-1.5315198890801106e16,-3.0630397781602212e16,-1.5315198890801106e16]]}];
        for (index,case) in cases.iter().enumerate() {
            assert!(case.point[0]<1e-13 && case.point[1]<1e-13 && case.point[0]+case.point[1]>1e-13);
            for runtime in [false,true] {
                let analytic=evaluate(&functional,case.point,runtime);
                relative(analytic.energy,case.energy);
                for i in 0..5 {relative(analytic.gradient[i],case.gradient[i]);
                    for j in 0..5 {relative(analytic.hessian[i][j],case.hessian[i][j]);}}
                for i in 5..7 {assert_eq!(analytic.gradient[i],0.);
                    for j in 0..7 {assert_eq!(analytic.hessian[i][j],0.);assert_eq!(analytic.hessian[j][i],0.);}}
                let labels:Vec<_>=(0..30).map(|component|format!("PBE tail {index} runtime={runtime} component={component}")).collect();
                let mut failures=Vec::new();
                stencils::fit_components_staged(&labels,&[([0.04,0.02,0.01,0.005],stencils::Stencil::Second)],|steps,stencil| {
                    (0..30).map(|component|std::array::from_fn(|k|{
                        let h=steps[k];let mut plus=case.point;let mut minus=case.point;
                        let (target,reference)=if component<5 {
                            plus[component]*=1.+h;minus[component]*=1.-h;
                            (analytic.gradient[component]*case.point[component],
                             stencil.first(energy(&functional,plus),energy(&functional,minus),0.,0.,h))
                        } else {
                            let i=(component-5)/5;let j=(component-5)%5;
                            let reference=if i==j {
                                plus[i]*=1.+h;minus[i]*=1.-h;
                                (energy(&functional,plus)+energy(&functional,minus)-2.*energy(&functional,case.point))/(h*h)
                            } else {
                                [(1.,1.),(1.,-1.),(-1.,1.),(-1.,-1.)].iter().map(|&(si,sj)|{
                                    let mut point=case.point;point[i]*=1.+si*h;point[j]*=1.+sj*h;
                                    si*sj*energy(&functional,point)
                                }).sum::<f64>()/(4.*h*h)
                            };
                            (analytic.hessian[i][j]*case.point[i]*case.point[j],reference)
                        };
                        let error=(target-reference).abs();
                        println!("PBE_TAIL_SIGNED case={index} runtime={runtime} component={component} h={h:.17e} analytic={target:.17e} reference={reference:.17e} error={error:.17e}");error
                    })).collect()
                },&mut failures);
                components::assert_no_failures("all PBE tail active gradient and ordered Hessian entries",&failures);
            }
        }
    }).unwrap();
}
