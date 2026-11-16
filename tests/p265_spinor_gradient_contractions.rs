// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every complex-spinor gradient component against fresh SCF energies.
//! The complete four-step fits and reference diagnostics validate the shared
//! analytic integral contractions independently of the nuclear Taylor path.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/correlated_fixture.rs"]
mod correlated_fixture;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use std::collections::BTreeMap;

fn fixture() -> (System, Value) {
    let (system, mut options) = correlated_fixture::fixture("hf", false, "none");
    if let Value::Object(entries) = &mut options {
        entries.retain(|(k, _)| {
            !matches!(
                k.as_str(),
                "pt2_same_spin" | "pt2_opposite_spin" | "double_hybrid_target_xc"
            )
        });
    }
    options.set("reference", "gks");
    options.set("charge", 1);
    options.set("multiplicity", 2usize);
    options.set("spin_axis", [0.3, 0.4, 0.5]);
    let source = spinor_operator::source(4);
    options.set("gks_operator_real", &source.re);
    options.set("gks_operator_imaginary", &source.im);
    (system, options)
}

#[test]
fn every_gradient_component_matches_four_step_fresh_energy_fits() {
    let (system, options) = fixture();
    let report = run(
        "gradient",
        &system,
        &Controls::from_value(&options).unwrap(),
    )
    .unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    assert!(report
        .get("nuclear_model")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("Pauli"));
    let gradient = report.get("gradient").unwrap().numeric_array().unwrap().1;
    assert_eq!(gradient.len(), 12);
    // The existing moving-integral Taylor provider is an additional
    // independent contraction, before the displaced-energy error fits.
    let controls = Controls::from_value(&options).unwrap();
    let method = dft_rs::driver::method::MolecularMethod::from_controls(&controls).unwrap();
    let setup =
        dft_rs::driver::method::prepare_molecular(&system.numbers, &system.positions, &method)
            .unwrap();
    let seed = dft_rs::driver::method::run_molecular_reference(&setup).unwrap();
    let operator = spinor_operator::source(4);
    let backend = dft_rs::response::spinor::MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &seed.scf,
        &setup.options,
        operator.clone(),
        [0.3, 0.4, 0.5],
        dft_rs::response::spinor::MolecularSpinorFunctionalModel::SpinCovariant,
        1e-10,
    )
    .unwrap();
    let reference = dft_rs::periodic::spinor_scf::run_spinor_scf(
        &backend,
        &dft_rs::periodic::spinor_scf::SpinorScfConfig {
            max_iter: setup.options.max_iter,
            energy_tolerance: setup.options.e_tol,
            orbital_tolerance: setup.options.conv_tol_grad,
            density_tolerance: setup.options.p_tol,
            overlap_cutoff: setup.options.orth_threshold,
            diis_size: setup.options.diis_size,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(reference.converged);
    let config = dft_rs::response::ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let context = dft_rs::response::nuclear::SpinorNuclearContext::new(
        &setup.molecule,
        &setup.basis,
        setup.grid.as_ref(),
        &setup.functional,
        &reference,
        &setup.options,
        &operator,
        1,
        &config,
    )
    .unwrap();
    for coordinate in 0..12 {
        let mut direction = vec![[0.0; 3]; 4];
        direction[coordinate / 3][coordinate % 3] = 1.0;
        let taylor = context.energy_taylor(&direction, 1).unwrap();
        let error = (taylor.energy[1] - gradient[coordinate]).abs();
        println!("complex gradient coordinate={coordinate} independent Taylor defect={error:.12e}");
        assert!(error < 1e-10);
    }
    let mut energy_options = options.clone();
    if let Value::Object(entries) = &mut energy_options {
        entries.retain(|(key, _)| key != "response_tolerance");
    }
    let mut cache = BTreeMap::new();
    let mut energy = |coordinate: usize, displacement: f64| {
        *cache
            .entry((coordinate, displacement.to_bits()))
            .or_insert_with(|| {
                let mut shifted = system.clone();
                shifted.positions[coordinate / 3][coordinate % 3] += displacement;
                let value = run(
                    "energy",
                    &shifted,
                    &Controls::from_value(&energy_options).unwrap(),
                )
                .unwrap();
                assert_eq!(value.get("converged").and_then(Value::as_bool), Some(true));
                value.get("energy").and_then(Value::as_f64).unwrap()
            })
    };
    let names: Vec<_> = (0..12)
        .map(|i| format!("complex Pauli analytic gradient[{i}]"))
        .collect();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.01, 0.008, 0.006, 0.005], stencils::Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            (0..12)
                .map(|coordinate| {
                    std::array::from_fn(|s| {
                        let h = steps[s];
                        let (plus, minus) = (energy(coordinate, h), energy(coordinate, -h));
                        let (far_plus, far_minus) = if stencil == stencils::Stencil::Fourth {
                            (energy(coordinate, 2.0 * h), energy(coordinate, -2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(plus, minus, far_plus, far_minus, h) - gradient[coordinate])
                            .abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("all complex Pauli gradient components", &failures);
    let field = [0.003, -0.002, 0.004];
    for axis in 0..3 {
        let sum: f64 = (0..4).map(|a| gradient[3 * a + axis]).sum();
        println!(
            "charged complex gradient translation axis={axis} defect={:.12e}",
            (sum + field[axis]).abs()
        );
        assert!((sum + field[axis]).abs() < 1e-9);
    }
}
