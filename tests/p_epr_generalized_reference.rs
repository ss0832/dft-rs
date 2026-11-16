// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Actual GHF/GKS SCF states with explicit one-electron SOC and a declared
//! single-Kramers-pair EPR model. Not a spin-purified multiplet benchmark.

pub use dft_rs::properties::nmr;
pub use dft_rs::{
    basis, constants, dfpt, grid, integrals, linalg, logger, math, periodic, properties, resources,
    response, scf, system, xc, DftError, Result,
};
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "../src/properties/epr.rs"]
pub mod epr;

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::math::Vec3;
use dft_rs::periodic::spinor_scf::{run_spinor_scf, SpinorScfConfig};
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};

fn method(xc: &str) -> Value {
    Value::object()
        .with("xc", xc)
        .with("basis", "sto-3g")
        .with("multiplicity", 2usize)
        .with("unrestricted", true)
        .with("guess", "core")
        .with("grid_level", 2usize)
        .with("grid_weight_cutoff", 0.0)
        .with("dispersion", "none")
        .with("conv_tol", 1e-12)
        .with("conv_tol_grad", 1e-10)
        .with("conv_tol_dp", 1e-11)
        .with("max_iter", 500usize)
}

fn evaluate(system: &System, xc: &str, axis: [f64; 3]) -> (epr::spinor::ProjectedSpinorEpr, f64) {
    let parameters = method(xc);
    let method =
        MolecularMethod::from_controls(&Controls::from_value(&parameters).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    assert!(seed.scf.converged);
    let charges = system.numbers.iter().map(|&z| z as f64).collect::<Vec<_>>();
    let operator =
        epr::spinor::one_electron_soc_operator(&setup.basis, &setup.molecule, &charges).unwrap();
    assert!(operator.hermiticity_error() < 1e-12);
    assert!(epr::doublet::time_reversal_defect(&operator, 1).unwrap() < 1e-12);
    let backend = MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &seed.scf,
        &setup.options,
        operator.clone(),
        axis,
        MolecularSpinorFunctionalModel::SpinCovariant,
        1e-10,
    )
    .unwrap();
    let config = SpinorScfConfig {
        max_iter: 500,
        energy_tolerance: 1e-12,
        density_tolerance: 1e-11,
        orbital_tolerance: 1e-10,
        overlap_cutoff: setup.options.orth_threshold,
        ..Default::default()
    };
    let reference = run_spinor_scf(&backend, &config).unwrap();
    assert!(reference.converged, "{xc} generalized SCF did not converge");
    assert!(
        reference.fields.operator_converged,
        "{xc} generalized SCF operator did not converge: {:?}",
        reference.fields.xc_diagnostics
    );
    assert!(reference.orbital_residual < 1e-10);
    let result = epr::spinor::projected_spinor_epr(
        &setup.molecule,
        &setup.basis,
        &seed.scf.overlap,
        &reference,
        &operator,
        Vec3::new(
            system.positions[0][0],
            system.positions[0][1],
            system.positions[0][2],
        ),
        &charges,
        1e-8,
    )
    .unwrap();
    assert!(result.g_tensor.iter().flatten().all(|v| v.is_finite()));
    assert!(result
        .total_per_g_mhz
        .iter()
        .flatten()
        .flatten()
        .all(|v| v.is_finite()));
    assert!(result.spin_squared >= 0.75 - 1e-9);
    println!("{xc} generalized SOC SCF E={} residual={} S={:?} S2={} g={:?} nucleus0 A={:?}MHz/nuclear_g",
        reference.fields.energy,reference.orbital_residual,result.spin_expectation,result.spin_squared,result.g_tensor,result.total_per_g_mhz[0]);
    (result, reference.fields.energy)
}

#[test]
fn actual_ghf_gks_atomic_g_and_a_match_collinear_invariants() {
    resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let system = System::molecule(vec![1], vec![[0.13, -0.17, 0.23]]);
            for xc in ["hf", "pbe"] {
                let (result, _) = evaluate(&system, xc, [0.36, -0.48, 0.8]);
                assert!((result.spin_squared - 0.75).abs() < 1e-10);
                let mut options = method(xc);
                options.set("gauge", "common-origin");
                options.set("gauge_origin", system.positions[0]);
                let ordinary = dft_rs::driver::run(
                    "g-tensor",
                    &system,
                    &Controls::from_value(&options).unwrap(),
                )
                .unwrap();
                let g = ordinary.get("g_tensor").unwrap().numeric_array().unwrap().1;
                let hyperfine = dft_rs::driver::run(
                    "hyperfine",
                    &system,
                    &Controls::from_value(&method(xc)).unwrap(),
                )
                .unwrap();
                let contact = hyperfine
                    .get("isotropic_coupling_per_g_mhz")
                    .unwrap()
                    .as_array()
                    .unwrap()[0]
                    .as_f64()
                    .unwrap();
                for a in 0..3 {
                    for b in 0..3 {
                        let expected = (0..3).map(|c| g[3 * a + c] * g[3 * b + c]).sum::<f64>();
                        assert!(
                            (result.g_gram[a][b] - expected).abs() < 1e-9,
                            "{xc} atomic g Gram({a},{b})"
                        );
                        let actual = (0..3)
                            .map(|c| {
                                result.total_per_g_mhz[0][a][c] * result.total_per_g_mhz[0][b][c]
                            })
                            .sum::<f64>();
                        assert!(
                            (actual - if a == b { contact * contact } else { 0.0 }).abs() < 1e-6,
                            "{xc} atomic A Gram({a},{b})"
                        );
                    }
                }
            }
        })
        .unwrap();
}

#[test]
fn actual_c1_methyl_ghf_and_gks_keep_spin_orbit_and_complete_pair_tensors() {
    resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let system = System::molecule(
                vec![6, 1, 1, 1],
                vec![
                    [0.17, -0.11, 0.23],
                    [1.77, 0.18, -0.53],
                    [-0.31, 1.74, -0.45],
                    [-1.2, -1.31, -0.61],
                ],
            );
            c1_geometry::assert_c1(&system.numbers, &system.positions);
            for xc in ["hf", "pbe"] {
                let (result, _) = evaluate(&system, xc, [0., 0., 1.]);
                assert!(result.occupied_metric_defect < 1e-8);
                assert!(result.pair_overlap < 1e-8);
                assert!(result.hamiltonian_time_reversal_defect < 1e-12);
                for atom in 0..4 {
                    for a in 0..3 {
                        for b in 0..3 {
                            let sum = result.contact_per_g_mhz[atom][a][b]
                                + result.dipolar_per_g_mhz[atom][a][b]
                                + result.orbital_per_g_mhz[atom][a][b];
                            assert!((sum - result.total_per_g_mhz[atom][a][b]).abs() < 1e-8);
                        }
                    }
                }
            }
        })
        .unwrap();
}
