// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Distinct moving-grid descriptor paths on a bounded C1 molecular fixture.
//! The deliberately small fixed grid tests derivatives, not grid accuracy.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::grid::MolecularGrid;
use dft_rs::periodic::spinor_scf::{run_spinor_scf, SpinorScfConfig};
use dft_rs::response::nuclear::{spinor_excitation_path, spinor_excitations};
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::response::spinor_tda::SpinorTdaConfig;
use dft_rs::response::ResponseSolverConfig;

fn molecule() -> System {
    System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    )
}

fn preflight(name: &str) {
    let system = molecule();
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", name)
            .with("basis", "sto-3g")
            .with("guess", "core")
            .with("conv_tol", 1e-13)
            .with("conv_tol_dp", 1e-13)
            .with("conv_tol_grad", 1e-12)
            .with("max_iter", 400usize)
            .with("charge", 2i32)
            .with("grid_weight_cutoff", 0.0),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let mut setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    setup.grid_options.radial_override = Some(8);
    setup.grid_options.angular_override = Some(5);
    setup.grid_options.prune = false;
    setup.grid = Some(MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap());
    let seed = run_molecular_reference(&setup).unwrap();
    println!(
        "{name} stage=collinear_seed_complete energy={:.12e}",
        seed.scf.energy
    );
    let mut operator = spinor_operator::source(4);
    for v in operator
        .re
        .as_mut_slice()
        .iter_mut()
        .chain(operator.im.as_mut_slice())
    {
        *v *= 0.01;
    }
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
        1e-11,
    )
    .unwrap();
    let reference = run_spinor_scf(
        &backend,
        &SpinorScfConfig {
            energy_tolerance: 1e-13,
            density_tolerance: 1e-13,
            orbital_tolerance: 1e-12,
            max_iter: 400,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(reference.converged);
    println!(
        "{name} stage=spinor_reference_complete iterations={} residual={:.12e}",
        reference.iterations, reference.orbital_residual
    );
    let tda = SpinorTdaConfig {
        states: 12,
        memory_mb: 128,
        tolerance: 1e-9,
        minimum_root_gap: 1e-6,
    };
    let config = ResponseSolverConfig {
        tolerance: 1e-10,
        ..Default::default()
    };
    let spectrum = spinor_excitations(
        &setup.molecule,
        &setup.basis,
        setup.grid.as_ref(),
        &setup.functional,
        &reference,
        &setup.options,
        &operator,
        &tda,
        &config,
    )
    .unwrap();
    println!(
        "{name} stage=static_spectrum_complete first_pole={:.12e}",
        spectrum.states[0].energy
    );
    let direction: Vec<[f64; 3]> = (0..4)
        .map(|i| std::array::from_fn(|d| 0.13 * ((i * 3 + d + 1) as f64).sin()))
        .collect();
    let path = spinor_excitation_path(
        &setup.molecule,
        &setup.basis,
        setup.grid.as_ref(),
        &setup.functional,
        &reference,
        &setup.options,
        &operator,
        &direction,
        0,
        2,
        &tda,
        &config,
    )
    .unwrap();
    assert!((path.excitation[0] - spectrum.states[0].energy).abs() < 1e-9);
    assert!(path.excitation.iter().all(|v| v.is_finite()));
    assert!(path.residual <= config.acceptance());
    println!("{name} preflight poles={} gap={:.12e} ground={:?} pole={:?} relative_residual={:.12e} stationarity={:.12e} frame={:.12e}",
        spectrum.states.len(),spectrum.states[0].nearest_gap,path.ground,path.excitation,
        path.residual,path.stationarity_defect,path.frame_defect);
}

#[test]
fn lda_full_spinor_moving_kernel_preflight() {
    preflight("svwn5");
}
#[test]
fn gga_full_spinor_moving_kernel_preflight() {
    preflight("pbe");
}
#[test]
fn meta_gga_full_spinor_moving_kernel_preflight() {
    preflight("tpss");
}
