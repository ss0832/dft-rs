// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent finite-temperature Mermin response of a distorted molecule.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_cartesian_directions.rs"]
mod symmetric_cartesian_directions;

use dft_rs::driver::method::{prepare_molecular, MolecularMethod, MolecularSetup};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::math::Vec3;
use dft_rs::response::nuclear::{molecular_taylor, MolecularDirection};
use dft_rs::response::ResponseSolverConfig;

const ATOMS: [u8; 4] = [7, 1, 1, 1];
const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.21, 0.09],
    [1.61, 0.77, -0.29],
    [-1.34, 0.61, 0.33],
    [0.31, -1.74, 0.58],
];

fn controls() -> Value {
    Value::object()
        .with("xc", "pbe")
        .with("reference", "rks")
        .with("basis", "sto-3g")
        .with("dispersion", "none")
        .with("electronic_temperature", 100000.0)
        .with("grid_level", 1_usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-12)
        .with("conv_tol_dp", 1e-12)
        .with("max_iter", 500_usize)
        .with("cpu_threads", 1_usize)
}

fn setup() -> (MolecularMethod, MolecularSetup) {
    c1_geometry::assert_c1(&ATOMS, &POSITIONS);
    let method =
        MolecularMethod::from_controls(&Controls::from_value(&controls()).unwrap()).unwrap();
    let setup = prepare_molecular(&ATOMS, &POSITIONS, &method).unwrap();
    (method, setup)
}

fn reference(setup: &MolecularSetup, field: [f64; 3]) -> dft_rs::scf::ScfResult {
    let mut options = setup.options.clone();
    options.external_field = Some(Vec3::new(field[0], field[1], field[2]));
    let result = dft_rs::scf::run_scf(
        &setup.molecule,
        &setup.basis,
        &options,
        &setup.functional,
        setup.grid.as_ref(),
    )
    .unwrap();
    assert!(
        result.converged,
        "unconverged displaced SCF at field {field:?}"
    );
    result
}

fn contraction(tensor: &[f64], direction: &[f64]) -> f64 {
    tensor
        .iter()
        .enumerate()
        .map(|(mut flat, &value)| {
            let mut term = value;
            let mut width = tensor.len();
            while width > 1 {
                term *= direction[flat % direction.len()];
                flat /= direction.len();
                width /= direction.len();
            }
            term
        })
        .sum()
}

#[test]
fn mermin_beta_and_gamma_have_all_independent_electric_fits() {
    let (_, setup) = setup();
    let system = System::molecule(ATOMS.to_vec(), POSITIONS.to_vec());
    let controls = Controls::from_value(&controls()).unwrap();
    let centered = reference(&setup, [0.0; 3]);
    let initial = dft_rs::properties::dipole_moment(&setup.molecule, &setup.basis, &centered);
    let stages = [
        ([0.016, 0.012, 0.008, 0.006], stencils::Stencil::Second),
        ([0.008, 0.006, 0.004, 0.003], stencils::Stencil::Second),
        ([0.016, 0.012, 0.008, 0.006], stencils::Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    for (id, rank) in [("beta", 3), ("gamma", 4)] {
        let report = run(id, &system, &controls).unwrap();
        assert_eq!(
            report.get("converged").and_then(Value::as_bool),
            Some(true),
            "{}",
            report.to_json()
        );
        assert_eq!(
            report.get("projector_constraint").and_then(Value::as_bool),
            Some(false)
        );
        let (_, tensor) = report.get("tensor").unwrap().numeric_array().unwrap();
        assert_eq!(tensor.len(), 3_usize.pow(rank as u32));
        let directions = symmetric_cartesian_directions::directions(3, rank);
        println!("{id} E={:.15e} tensor={tensor:?}", centered.energy);
        let names: Vec<_> = (0..directions.len())
            .map(|i| format!("{id} direction {i}"))
            .collect();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                directions
                    .iter()
                    .map(|direction| {
                        let expected = contraction(&tensor, direction);
                        let projection = |moment: Vec3| {
                            moment.x * direction[0]
                                + moment.y * direction[1]
                                + moment.z * direction[2]
                        };
                        let dipole = |step| {
                            let field = std::array::from_fn(|axis| step * direction[axis]);
                            let state = reference(&setup, field);
                            projection(dft_rs::properties::dipole_moment(
                                &setup.molecule,
                                &setup.basis,
                                &state,
                            ))
                        };
                        let derivative = |h: f64| {
                            let plus = dipole(h);
                            let minus = dipole(-h);
                            if rank == 3 {
                                ((plus - projection(initial)) + (minus - projection(initial)))
                                    / h.powi(2)
                            } else {
                                let inner =
                                    stencils::Stencil::Second.first(plus, minus, 0.0, 0.0, h);
                                let outer = stencils::Stencil::Second.first(
                                    dipole(2.0 * h),
                                    dipole(-2.0 * h),
                                    0.0,
                                    0.0,
                                    2.0 * h,
                                );
                                2.0 * (outer - inner) / h.powi(2)
                            }
                        };
                        steps.map(|h| {
                            let value = match stencil {
                                stencils::Stencil::Second => derivative(h),
                                stencils::Stencil::Fourth => {
                                    (4.0 * derivative(h) - derivative(2.0 * h)) / 3.0
                                }
                            };
                            (value - expected).abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures("Mermin beta and gamma", &failures);
}

#[test]
fn mermin_fourth_electric_energy_matches_relaxed_dipole_and_entropy_ledger() {
    let (_, setup) = setup();
    let centered = reference(&setup, [0.0; 3]);
    let path = MolecularDirection {
        nuclear: vec![[0.0; 3]; ATOMS.len()],
        electric: [0.47, 0.31, -0.71],
    };
    let config = ResponseSolverConfig::default();
    let report = molecular_taylor(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &centered,
        &setup.options,
        &path,
        4,
        &config,
    )
    .unwrap();
    assert!(report.converged, "{report:?}");
    let values = report.value;
    let thermal = values.thermal_population.unwrap();
    assert_eq!(thermal.density_defects.len(), 5);
    assert!(thermal
        .density_defects
        .iter()
        .all(|r| *r <= config.acceptance()));
    assert!(values.energy_components[0].electronic_entropy_energy < 0.0);
    assert!((values.energy_coefficients[0] - centered.energy).abs() < 1e-10);
    for n in 1..=4 {
        let electronic_dipole = values.dipole_coefficients[n - 1]
            .iter()
            .zip(path.electric)
            .map(|(moment, field)| moment * field)
            .sum::<f64>();
        assert!((n as f64 * values.energy_coefficients[n] + electronic_dipole).abs() < 1e-9);
    }
    println!(
        "Mermin E={:?} entropy={:?} mu={:?} Fermi defects={:?} solve residuals={:?}",
        values.energy_coefficients,
        values
            .energy_components
            .iter()
            .map(|e| e.electronic_entropy_energy)
            .collect::<Vec<_>>(),
        thermal.chemical_potentials,
        thermal.density_defects,
        report.residuals
    );
}
