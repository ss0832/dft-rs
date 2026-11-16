// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared full-component nuclear acceptance without symmetry reconstruction.
use super::{c1_geometry, components, quartic_oracle, stencils, symmetric_directions};
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::response::nuclear::nuclear_taylor;
use dft_rs::response::ResponseSolverConfig;
use std::collections::BTreeMap;

fn gradient(system: &System, controls: &Value) -> (f64, Vec<f64>) {
    let report = run("gradient", system, &Controls::from_value(controls).unwrap()).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    (
        report.get("energy").and_then(Value::as_f64).unwrap(),
        report.get("gradient").unwrap().numeric_array().unwrap().1,
    )
}

pub struct BasisContract<'a> {
    pub scalar_ecp: bool,
    pub highest_l: u8,
    pub ao_count: usize,
    pub required_l: &'a [u8],
}

pub fn fit_case(
    system: System,
    controls: Value,
    basis: BasisContract<'_>,
    label: String,
    order: usize,
) {
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let dimension = 3 * system.numbers.len();
    assert_eq!(dimension, 12);
    let method = MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    assert_eq!(setup.basis.max_l, basis.highest_l);
    assert_eq!(setup.basis.nao_sph, basis.ao_count);
    assert_eq!(!setup.basis.ecps.is_empty(), basis.scalar_ecp);
    for &l in basis.required_l {
        assert!(setup.basis.shells.iter().any(|s| s.l == l));
    }
    let mut failures = Vec::new();
    if order == 2 {
        let (_, reference_gradient) = gradient(&system, &controls);
        let hessian = run(
            "hessian",
            &system,
            &Controls::from_value(&controls).unwrap(),
        )
        .unwrap()
        .get("hessian")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
        assert_eq!(hessian.len(), dimension * dimension);
        for coordinate in 0..dimension {
            let names = std::iter::once(format!("{label} gradient {coordinate}"))
                .chain((0..dimension).map(|i| format!("{label} Hessian ({i},{coordinate})")))
                .collect::<Vec<_>>();
            let mut cache = BTreeMap::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    (
                        [0.005, 0.0025, 0.00125, 0.000625],
                        stencils::Stencil::Second,
                    ),
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    let mut at = |t: f64| {
                        cache
                            .entry(t.to_bits())
                            .or_insert_with(|| {
                                let mut displaced = system.clone();
                                displaced.positions[coordinate / 3][coordinate % 3] += t;
                                gradient(&displaced, &controls)
                            })
                            .clone()
                    };
                    let mut errors = vec![[0.0; 4]; dimension + 1];
                    for (step, &h) in steps.iter().enumerate() {
                        let (p, m) = (at(h), at(-h));
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            ((0.0, vec![0.0; dimension]), (0.0, vec![0.0; dimension]))
                        };
                        errors[0][step] = (stencil.first(p.0, m.0, p2.0, m2.0, h)
                            - reference_gradient[coordinate])
                            .abs();
                        for i in 0..dimension {
                            errors[i + 1][step] = (stencil
                                .first(p.1[i], m.1[i], p2.1[i], m2.1[i], h)
                                - hessian[i * dimension + coordinate])
                                .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
        println!(
            "{label} required gradient/Hessian entries={}",
            dimension + dimension * dimension
        );
    } else {
        let reference = run_molecular_reference(&setup).unwrap();
        assert!(reference.scf.converged);
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let directions = symmetric_directions::directions(dimension, order);
        assert_eq!(directions.len(), if order == 3 { 364 } else { 1365 });
        for (index, direction) in directions.iter().enumerate() {
            let pattern = direction
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect::<Vec<_>>();
            let report = nuclear_taylor(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                &setup.options,
                &pattern,
                order,
                &config,
            )
            .unwrap();
            assert!(
                report.converged,
                "{label} direction={index} order={order} stationarity={:?} projector={:?} metric={:?} particle_number={:?} linear={:?} reference_energy={:.12e} reference_fock={:.12e}",
                report.value.stationarity_defects,
                report.value.projector_defects,
                report.value.metric_defects,
                report.value.particle_number_coefficients,
                report.residuals,
                report.value.reference_energy_error,
                report.value.reference_fock_error,
            );
            let exact =
                report.value.energy_coefficients[order] * (1..=order).product::<usize>() as f64;
            let mut cache = BTreeMap::new();
            let mut at = |t: f64| {
                *cache.entry(t.to_bits()).or_insert_with(|| {
                    let mut displaced = system.clone();
                    for (i, value) in displaced.positions.iter_mut().flatten().enumerate() {
                        *value += t * direction[i];
                    }
                    gradient(&displaced, &controls)
                        .1
                        .iter()
                        .zip(direction)
                        .map(|(g, d)| g * d)
                        .sum::<f64>()
                })
            };
            let mut finite = |h: f64| {
                if order == 3 {
                    (at(h) - 2.0 * at(0.0) + at(-h)) / h.powi(2)
                } else {
                    (at(2.0 * h) - 2.0 * at(h) + 2.0 * at(-h) - at(-2.0 * h)) / (2.0 * h.powi(3))
                }
            };
            let names = [format!("{label} independent direction {index}")];
            let mut direction_failures = Vec::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                    ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                    ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
                    ([0.16, 0.12, 0.08, 0.06], stencils::Stencil::Second),
                    ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
                    // Retain every failed small-step interval. Larger fourth-order
                    // intervals separate a small quadratic coefficient from the
                    // amplified roundoff of third differences of gradients.
                    ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
                    ([0.12, 0.09, 0.06, 0.045], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    vec![std::array::from_fn(|step| {
                        let h = steps[step];
                        let near = finite(h);
                        let value = if stencil == stencils::Stencil::Fourth {
                            (4.0 * near - finite(2.0 * h)) / 3.0
                        } else {
                            near
                        };
                        (value - exact).abs()
                    })]
                },
                &mut direction_failures,
            );
            if !direction_failures.is_empty() && order == 4 {
                println!("{label} direction {index}: all failed gradient intervals retained; checking an independently displaced second response");
                direction_failures.clear();
                quartic_oracle::fit_from_second(
                    &names[0],
                    exact,
                    |t| {
                        let mut displaced = system.clone();
                        for (value, d) in displaced.positions.iter_mut().flatten().zip(direction) {
                            *value += t * d;
                        }
                        let local =
                            prepare_molecular(&displaced.numbers, &displaced.positions, &method)
                                .unwrap();
                        let reference = run_molecular_reference(&local).unwrap();
                        let value = nuclear_taylor(
                            &local.molecule,
                            &local.basis,
                            &local.functional,
                            local.grid.as_ref(),
                            &reference.scf,
                            &local.options,
                            &pattern,
                            2,
                            &config,
                        )
                        .unwrap();
                        assert!(value.converged);
                        2.0 * value.value.energy_coefficients[2]
                    },
                    &mut direction_failures,
                );
            }
            failures.extend(direction_failures);
            println!(
                "{label} progress={}/{} raw_response={:?} reference_energy_error={:.12e}",
                index + 1,
                directions.len(),
                report.residuals,
                report.value.reference_energy_error
            );
        }
    }
    components::assert_no_failures(&label, &failures);
}
