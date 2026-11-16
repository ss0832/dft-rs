// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete C1 scalar-ECP nuclear derivatives on constrained and complex references.
//! The independent oracle is a freshly converged displaced analytic gradient.
//! Every gradient/Hessian entry and every complete rank-three/rank-four
//! unisolvent direction is fitted; a partial run does not certify a tensor.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_reference.rs"]
mod nuclear_reference;
#[path = "support/quartic_oracle.rs"]
mod quartic_oracle;
#[path = "support/spdf_fixture.rs"]
mod spdf_fixture;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_cartesian_directions.rs"]
mod symmetric_directions;

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::periodic::spinor_scf::{run_spinor_scf, SpinorScfConfig};
use dft_rs::response::nuclear::{open_shell_nuclear_energy_taylor, SpinorNuclearContext};
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::response::ResponseSolverConfig;
use std::collections::BTreeMap;

fn fixture(reference: &str, xc: &str, spdf: bool) -> (System, Value) {
    let (system, controls) = spdf_fixture::fixture(true, true, xc);
    let basis = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(if spdf {
        "tests/data/c1_spdf_model.g94"
    } else {
        "tests/data/c1_compact_valence.g94"
    });
    let mut controls = controls
        .with("basis", basis.to_str().unwrap())
        .with("ecp", "lanl2dz")
        .with("reference", reference)
        .with("response_tolerance", 1e-12);
    if reference == "gks" {
        let operator = spinor_operator::source(if spdf { 20 } else { 5 });
        controls = controls
            .with("gks_operator_real", &operator.re)
            .with("gks_operator_imaginary", &operator.im);
    }
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    (system, controls)
}

fn hessian(reference: &str, xc: &str) {
    hessian_with_basis(reference, xc, false);
}
fn hessian_with_basis(reference: &str, xc: &str, spdf: bool) {
    let (system, controls) = fixture(reference, xc, spdf);
    nuclear_reference::fit(
        &format!("C1 ECP {reference} {xc} nuclear"),
        &system,
        &Controls::from_value(&controls).unwrap(),
        &[
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
        ],
        &[""],
        |feature, system, controls| run(feature, system, controls).unwrap(),
    );
}

fn higher(reference: &str, order: usize) {
    higher_with_basis(reference, order, false);
}
fn higher_with_basis(reference: &str, order: usize, spdf: bool) {
    let model = if spdf { "C1 s/p/d/f ECP" } else { "C1 ECP" };
    let (system, controls) = fixture(reference, "hf", spdf);
    let method = MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    assert_eq!(setup.basis.nao_sph, if spdf { 20 } else { 5 });
    assert_eq!(setup.basis.ecps.len(), 1);
    assert_eq!(setup.basis.n_core_electrons, 10);
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let common = if reference == "roks" {
        Some(
            dft_rs::scf::roks::run_roks_with_orbitals(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                setup.grid.as_ref(),
            )
            .unwrap(),
        )
    } else {
        None
    };
    let seed = run_molecular_reference(&setup).unwrap();
    let operator = spinor_operator::source(setup.basis.nao_sph);
    let backend = MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &seed.scf,
        &setup.options,
        operator.clone(),
        [0.0, 0.0, 1.0],
        MolecularSpinorFunctionalModel::SpinCovariant,
        1e-12,
    )
    .unwrap();
    let spinor = if common.is_none() {
        let result = run_spinor_scf(
            &backend,
            &SpinorScfConfig {
                energy_tolerance: 1e-14,
                density_tolerance: 1e-14,
                orbital_tolerance: 1e-14,
                max_iter: 500,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged);
        assert!(result.density[0]
            .im
            .as_slice()
            .iter()
            .any(|v| v.abs() > 1e-5));
        Some(result)
    } else {
        None
    };
    let context = spinor.as_ref().map(|state| {
        SpinorNuclearContext::new(
            &setup.molecule,
            &setup.basis,
            setup.grid.as_ref(),
            &setup.functional,
            state,
            &setup.options,
            &operator,
            order,
            &config,
        )
        .unwrap()
    });
    let controls = Controls::from_value(&controls).unwrap();
    let initial = run("gradient", &system, &controls).unwrap();
    let initial_gradient = initial.get("gradient").unwrap().numeric_array().unwrap().1;
    let directions = symmetric_directions::directions(12, order);
    assert_eq!(directions.len(), if order == 3 { 364 } else { 1365 });
    let mut failures = Vec::new();
    for (index, direction) in directions.iter().enumerate() {
        let pattern = direction
            .chunks_exact(3)
            .map(|v| [v[0], v[1], v[2]])
            .collect::<Vec<_>>();
        let (energy, residuals) = if let Some((state, orbitals)) = &common {
            let value = open_shell_nuclear_energy_taylor(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                state,
                orbitals,
                &setup.options,
                &pattern,
                order,
                &config,
            )
            .unwrap();
            assert!(value.converged);
            (value.energy[order], value.residuals)
        } else {
            let value = context
                .as_ref()
                .unwrap()
                .energy_taylor(&pattern, order)
                .unwrap();
            (value.energy[order], value.residuals)
        };
        let exact = energy * (1..=order).product::<usize>() as f64;
        let mut cache = BTreeMap::new();
        cache.insert(
            0.0_f64.to_bits(),
            initial_gradient
                .iter()
                .zip(direction)
                .map(|(g, d)| g * d)
                .sum::<f64>(),
        );
        let mut at = |t: f64| {
            *cache.entry(t.to_bits()).or_insert_with(|| {
                let mut displaced = system.clone();
                for (value, d) in displaced.positions.iter_mut().flatten().zip(direction) {
                    *value += t * d;
                }
                run("gradient", &displaced, &controls)
                    .unwrap()
                    .get("gradient")
                    .unwrap()
                    .numeric_array()
                    .unwrap()
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
        let names = [format!(
            "{model} {reference} order={order} independent direction {index}"
        )];
        let mut direction_failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
                ([0.16, 0.12, 0.08, 0.06], stencils::Stencil::Second),
                ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
                ([0.12, 0.09, 0.06, 0.045], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                vec![std::array::from_fn(|k| {
                    let h = steps[k];
                    let near = finite(h);
                    let measured = if stencil == stencils::Stencil::Fourth {
                        (4.0 * near - finite(2.0 * h)) / 3.0
                    } else {
                        near
                    };
                    (measured - exact).abs()
                })]
            },
            &mut direction_failures,
        );
        if !direction_failures.is_empty() && order == 4 {
            println!("{model} {reference} direction {index}: failed gradient intervals retained; independent Hessian oracle");
            direction_failures.clear();
            quartic_oracle::fit_from_second(
                &names[0],
                exact,
                |t| {
                    let mut displaced = system.clone();
                    for (value, d) in displaced.positions.iter_mut().flatten().zip(direction) {
                        *value += t * d;
                    }
                    let report = run("hessian", &displaced, &controls).unwrap();
                    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
                    let tensor = report.get("hessian").unwrap().numeric_array().unwrap().1;
                    (0..12)
                        .flat_map(|i| (0..12).map(move |j| (i, j)))
                        .map(|(i, j)| direction[i] * tensor[12 * i + j] * direction[j])
                        .sum::<f64>()
                },
                &mut direction_failures,
            );
        }
        failures.extend(direction_failures);
        println!(
            "{model} {reference} order={order} progress={}/{} raw_response={residuals:?}",
            index + 1,
            directions.len()
        );
    }
    components::assert_no_failures(&format!("{model} {reference} rank {order}"), &failures);
}

#[test]
fn rohf_hessian() {
    hessian("roks", "hf");
}
#[test]
fn ghf_hessian() {
    hessian("gks", "hf");
}
#[test]
fn roks_gga_hessian() {
    hessian("roks", "pbe");
}
#[test]
fn gks_gga_hessian() {
    hessian("gks", "pbe");
}
#[test]
fn rohf_cubic() {
    higher("roks", 3);
}
#[test]
fn ghf_cubic() {
    higher("gks", 3);
}
#[test]
fn rohf_quartic() {
    higher("roks", 4);
}
#[test]
fn ghf_quartic() {
    higher("gks", 4);
}

#[test]
fn rohf_spdf_hessian() {
    hessian_with_basis("roks", "hf", true);
}
#[test]
fn ghf_spdf_hessian() {
    hessian_with_basis("gks", "hf", true);
}
#[test]
fn rohf_spdf_cubic() {
    higher_with_basis("roks", 3, true);
}
#[test]
fn ghf_spdf_cubic() {
    higher_with_basis("gks", 3, true);
}
#[test]
fn rohf_spdf_quartic() {
    higher_with_basis("roks", 4, true);
}
#[test]
fn ghf_spdf_quartic() {
    higher_with_basis("gks", 4, true);
}
