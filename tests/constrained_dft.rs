// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::overlap_matrix;
use dft_rs::linalg::Matrix;
use dft_rs::scf::run_scf;

const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.78, 0.34, -0.08],
    [-0.49, 1.65, 0.41],
    [0.31, -0.78, 1.91],
];
const NUMBERS: [u8; 4] = [7, 1, 1, 1];

fn value(report: &Value, key: &str) -> f64 {
    report.get(key).unwrap().as_f64().unwrap()
}

fn vector(report: &Value, key: &str) -> Vec<f64> {
    report
        .get(key)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

#[test]
fn restricted_charge_constraints_fit_every_multiplier_component() {
    fit_restricted_charge_constraints("wb97x-d");
}

#[test]
fn restricted_vv10_charge_constraints_fit_every_multiplier_component() {
    fit_restricted_charge_constraints("wb97x-v");
}

fn fit_restricted_charge_constraints(functional: &str) {
    let setup = prepare_molecular(
        &NUMBERS,
        &POSITIONS,
        &MolecularMethod {
            xc: Some(functional.into()),
            grid_level: 0,
            vv10_grid_level: 0,
            conv_tol: 1e-12,
            conv_tol_grad: Some(1e-11),
            conv_tol_dp: 1e-11,
            grid_weight_cutoff: 0.0,
            ..MolecularMethod::default()
        },
    )
    .unwrap();
    let overlap = overlap_matrix(&setup.basis);
    let owner = setup.basis.sph_ao_atoms();
    let operators: [Matrix; 2] = std::array::from_fn(|a| {
        let atom = a + 1;
        let mut result = overlap.clone();
        for i in 0..result.rows {
            for j in 0..result.cols {
                result[(i, j)] *= if owner[i] == atom && owner[j] == atom {
                    1.0
                } else if owner[i] == atom || owner[j] == atom {
                    0.5
                } else {
                    0.0
                };
            }
        }
        result
    });
    // This oracle drives ordinary spin-independent SCF with a linear AO
    // potential. It does not invoke the constrained implementation.
    let biased = |lambda: [f64; 2]| {
        let mut options = setup.options.clone();
        let mut potential = Matrix::zeros(overlap.rows, overlap.cols);
        for a in 0..2 {
            for (value, &q) in potential
                .as_mut_slice()
                .iter_mut()
                .zip(operators[a].as_slice())
            {
                *value += lambda[a] * q;
            }
        }
        options.extra_hcore = Some(potential);
        let scf = run_scf(
            &setup.molecule,
            &setup.basis,
            &options,
            &setup.functional,
            setup.grid.as_ref(),
        )
        .unwrap();
        let populations: [f64; 2] =
            std::array::from_fn(|a| operators[a].frobenius_dot(&scf.p_total()));
        (scf.energy, populations)
    };
    let lambda = [0.071, -0.043];
    let (biased_energy, populations) = biased(lambda);
    let evaluate = |targets: &[f64; 2]| {
        let constraints: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(i, &target)| {
                Value::object()
                    .with("atoms", vec![i + 1])
                    .with("target", target)
            })
            .collect();
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", functional)
                .with("grid_level", 0usize)
                .with("vv10_grid_level", 0usize)
                .with("reference", "rks")
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol", 1e-12)
                .with("conv_tol_grad", 1e-11)
                .with("conv_tol_dp", 1e-11)
                .with("constraint_tolerance", 1e-10)
                .with("density_constraints", constraints),
        )
        .unwrap();
        run(
            "constrained-dft",
            &System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec()),
            &controls,
        )
        .unwrap()
    };
    let result = evaluate(&populations);
    let dispersion = dft_rs::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )
    .unwrap();
    let expected_energy = biased_energy + dispersion
        - lambda
            .iter()
            .zip(populations)
            .map(|(a, b)| a * b)
            .sum::<f64>();
    assert!((value(&result, "energy") - expected_energy).abs() < 3e-10);
    let production_multipliers = vector(&result, "constraint_multipliers");
    for (actual, expected) in production_multipliers.iter().zip(lambda) {
        assert!((actual - expected).abs() < 5e-8);
    }
    let susceptibility = result
        .get("constraint_susceptibility")
        .unwrap()
        .as_array()
        .unwrap();
    let actual: [[f64; 2]; 2] = std::array::from_fn(|i| {
        std::array::from_fn(|j| susceptibility[i].as_array().unwrap()[j].as_f64().unwrap())
    });
    println!("cDFT {functional} raw lambda={lambda:?} Q={populations:?} J={actual:?} constraint={:?} response={:?} orbital={:.12e}",
        vector(&result, "constraint_residuals"), vector(&result, "constraint_response_residuals"),
        value(&result, "scf_orbital_residual"));
    let names = [
        "dQ0/dlambda0",
        "dQ0/dlambda1",
        "dQ1/dlambda0",
        "dQ1/dlambda1",
        "dE/dtarget0",
        "dE/dtarget1",
    ]
    .map(|component| format!("cDFT {functional} {component}"));
    let mut failed = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.012, 0.008, 0.005, 0.003], stencils::Stencil::Second),
            ([0.005, 0.0035, 0.0024, 0.0016], stencils::Stencil::Second),
            ([0.020, 0.015, 0.011, 0.008], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (k, &h) in steps.iter().enumerate() {
                for j in 0..2 {
                    let charge = |displacement| {
                        let mut changed = lambda;
                        changed[j] += displacement;
                        biased(changed).1
                    };
                    let qp = charge(h);
                    let qm = charge(-h);
                    let (qp2, qm2) = if stencil == stencils::Stencil::Fourth {
                        (charge(2.0 * h), charge(-2.0 * h))
                    } else {
                        ([0.0; 2], [0.0; 2])
                    };
                    for i in 0..2 {
                        errors[2 * i + j][k] =
                            (stencil.first(qp[i], qm[i], qp2[i], qm2[i], h) - actual[i][j]).abs();
                    }
                    let energy = |displacement| {
                        let mut changed = populations;
                        changed[j] += displacement;
                        value(&evaluate(&changed), "energy")
                    };
                    let (ep2, em2) = if stencil == stencils::Stencil::Fourth {
                        (energy(2.0 * h), energy(-2.0 * h))
                    } else {
                        (0.0, 0.0)
                    };
                    errors[4 + j][k] = (stencil.first(energy(h), energy(-h), ep2, em2, h)
                        + production_multipliers[j])
                        .abs();
                }
            }
            errors
        },
        &mut failed,
    );
    components::assert_no_failures("constrained range-separated DFT charge", &failed);
}

#[test]
fn carbon_monoxide_charge_constraint_range_separated_providers() {
    // Rounded gas-phase bond length of 1.128 Angstrom, without optimization.
    let system = System::molecule(vec![6, 8], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 2.132]]);
    for functional in ["wb97x-d", "wb97x-v", "wb97m-v"] {
        for basis in ["sto-3g", "6-31g"] {
            let evaluate = |target| {
                let constraints = vec![Value::object()
                    .with("atoms", vec![0usize])
                    .with("target", target)];
                let controls = Controls::from_value(
                    &Value::object()
                        .with("xc", functional)
                        .with("basis", basis)
                        .with("grid_level", 0usize)
                        .with("vv10_grid_level", 0usize)
                        .with("grid_weight_cutoff", 0.0)
                        .with("conv_tol", 1e-12)
                        .with("conv_tol_dp", 1e-11)
                        .with("conv_tol_grad", 1e-11)
                        .with("density_constraints", constraints),
                )
                .unwrap();
                run("constrained-dft", &system, &controls).unwrap()
            };
            let lower = evaluate(5.85);
            let upper = evaluate(6.05);
            let l = vector(&lower, "constraint_multipliers")[0];
            let u = vector(&upper, "constraint_multipliers")[0];
            assert!(
                u < l,
                "{functional} {basis} increasing population raised the multiplier"
            );
            for (target, result) in [(5.85, &lower), (6.05, &upper)] {
                let achieved = vector(result, "constraint_expectations")[0];
                let orbital = value(result, "scf_orbital_residual");
                let dispersion = value(result, "dispersion_energy");
                let nonlocal = value(result, "nonlocal_correlation_energy");
                assert!((achieved - target).abs() < 1e-9);
                if functional == "wb97x-d" {
                    assert!(dispersion < 0.0);
                } else {
                    assert_ne!(nonlocal, 0.0);
                }
                println!("CO {functional} {basis} target={target} Q={achieved:.12e} lambda={:.12e} E={:.12e} E_disp={dispersion:.12e} E_nlc={nonlocal:.12e} orbital={orbital:.12e} raw constraint={:?} response={:?}",
                    vector(result,"constraint_multipliers")[0],value(result,"energy"),
                    vector(result,"constraint_residuals"),vector(result,"constraint_response_residuals"));
            }
        }
    }
}

fn fragment_water_controls(xc: &str, gradient: bool) -> Value {
    Value::object()
        .with("xc", xc)
        .with("reference", "rks")
        .with("grid_level", 0usize)
        .with("vv10_grid_level", 0usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("constraint_tolerance", 1e-11)
        .with("gradient", gradient)
        .with(
            "density_constraints",
            vec![Value::object()
                .with("atoms", [0usize, 1])
                .with("electrons", 8.96)],
        )
}

fn fragment_water() -> System {
    System::molecule(
        vec![8, 1, 1],
        vec![
            [0.12, -0.17, 0.08],
            [1.68, 0.61, -0.29],
            [-0.66, 1.37, 0.67],
        ],
    )
}

#[test]
fn moving_fragment_charge_gradient_fits_all_cartesian_components() {
    fit_moving_fragment_gradient("wb97x-d");
}

#[test]
fn moving_vv10_fragment_charge_gradient_fits_all_cartesian_components() {
    fit_moving_fragment_gradient("wb97x-v");
}

fn fit_moving_fragment_gradient(xc: &str) {
    let system = fragment_water();
    let evaluate = |system: &System, controls: &Value| {
        run(
            "constrained-dft",
            system,
            &Controls::from_value(controls).unwrap(),
        )
        .unwrap()
    };
    let report = evaluate(&system, &fragment_water_controls(xc, true));
    let (shape, gradient) = report.get("gradient").unwrap().numeric_array().unwrap();
    assert_eq!(shape, [3, 3]);
    assert!(value(&report, "translation_residual") < 1e-8);
    println!(
        "water {xc} E={:.15e} gradient={gradient:?} lambda={:?} residuals={:?}",
        value(&report, "energy"),
        vector(&report, "constraint_multipliers"),
        vector(&report, "constraint_residuals"),
    );
    let oracle_controls = fragment_water_controls(xc, false).with(
        "initial_constraint_multipliers",
        vector(&report, "constraint_multipliers"),
    );
    let energy = |component: usize, delta: f64| {
        let mut displaced = system.clone();
        displaced.positions[component / 3][component % 3] += delta;
        let report = evaluate(&displaced, &oracle_controls);
        assert!((vector(&report, "constraint_expectations")[0] - 8.96).abs() < 1e-11);
        value(&report, "energy")
    };
    let names: Vec<_> = (0..9)
        .map(|i| format!("water {xc} fragment gradient {i}"))
        .collect();
    let mut failed = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.016, 0.011, 0.007, 0.004], stencils::Stencil::Second),
            ([0.008, 0.005, 0.003, 0.002], stencils::Stencil::Second),
            ([0.024, 0.017, 0.012, 0.008], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            gradient
                .iter()
                .enumerate()
                .map(|(component, expected)| {
                    steps.map(|h| {
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (energy(component, 2.0 * h), energy(component, -2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(
                            energy(component, h),
                            energy(component, -h),
                            plus2,
                            minus2,
                            h,
                        ) - expected)
                            .abs()
                    })
                })
                .collect()
        },
        &mut failed,
    );
    components::assert_no_failures("constrained water moving fragment", &failed);
}

#[test]
fn optimize_water_with_fragment_electron_count() {
    let system = fragment_water();
    let controls = fragment_water_controls("wb97x-d", true);
    let initial = run(
        "constrained-dft",
        &system,
        &Controls::from_value(&controls).unwrap(),
    )
    .unwrap();
    let optimized = run(
        "constrained-dft",
        &system,
        &Controls::from_value(
            &controls
                .with("optimize_geometry", true)
                .with("max_steps", 60usize),
        )
        .unwrap(),
    )
    .unwrap();
    println!("fragment constrained water E_initial={:.15e} E_final={:.15e} gradient={:.15e} population={:.15e} positions={:?}",
        value(&initial, "energy"), value(&optimized, "energy"), value(&optimized, "grad_max"),
        vector(&optimized, "constraint_expectations")[0], optimized.get("positions"));
    assert_eq!(
        optimized.get("geometry_converged").and_then(Value::as_bool),
        Some(true)
    );
    assert!(value(&optimized, "energy") < value(&initial, "energy"));
    assert!(value(&optimized, "grad_max") < 3e-4);
    assert!((vector(&optimized, "constraint_expectations")[0] - 8.96).abs() < 1e-11);
}

#[test]
fn ammonia_accepts_independent_multi_atom_electron_constraints() {
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let settings = Value::object()
        .with("xc", "wb97x-d")
        .with("reference", "rks")
        .with("grid_level", 0usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_grad", 1e-11)
        .with("conv_tol_dp", 1e-11)
        .with(
            "density_constraints",
            vec![
                Value::object()
                    .with("atoms", [0usize, 1])
                    .with("electrons", 8.05),
                Value::object()
                    .with("atoms", [1usize, 2])
                    .with("electrons", 1.90),
            ],
        );
    let report = run(
        "constrained-dft",
        &system,
        &Controls::from_value(&settings).unwrap(),
    )
    .unwrap();
    let populations = vector(&report, "constraint_expectations");
    assert!((populations[0] - 8.05).abs() < 1e-9);
    assert!((populations[1] - 1.90).abs() < 1e-9);
    println!(
        "ammonia fragment populations={populations:?} E={:.15e} multipliers={:?}",
        value(&report, "energy"),
        vector(&report, "constraint_multipliers")
    );
}

#[test]
fn static_fragment_electric_response_fits_all_tensor_entries() {
    let system = fragment_water();
    let fit = |xc: &str| {
        let input = fragment_water_controls(xc, false);
        let evaluate = |input: &Value| {
            run(
                "constrained-dft",
                &system,
                &Controls::from_value(input).unwrap(),
            )
            .unwrap()
        };
        let report = evaluate(&input.clone().with("constraint_electric_response", true));
        let (_, alpha) = report
            .get("polarizability")
            .unwrap()
            .numeric_array()
            .unwrap();
        assert!(value(&report, "electric_constraint_defect") < 1e-9);
        let setup = prepare_molecular(
            &system.numbers,
            &system.positions,
            &MolecularMethod::default(),
        )
        .unwrap();
        let dipole = dft_rs::integrals::dipole_matrices(&setup.basis, dft_rs::math::Vec3::zero());
        let input = input.with(
            "initial_constraint_multipliers",
            vector(&report, "constraint_multipliers"),
        );
        let moment = |axis: usize, field: f64| {
            let mut vector_field = [0.0; 3];
            vector_field[axis] = field;
            let report = evaluate(&input.clone().with("field", vector_field));
            std::array::from_fn::<_, 3, _>(|i| {
                ["density_alpha", "density_beta"]
                    .iter()
                    .map(|key| {
                        let (_, density) = report.get(key).unwrap().numeric_array().unwrap();
                        -density
                            .iter()
                            .zip(dipole[i].as_slice())
                            .map(|(p, r)| p * r)
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
        };
        let names: Vec<_> = (0..9)
            .map(|i| format!("water {xc} fixed-population alpha {i}"))
            .collect();
        println!(
            "water {xc} alpha={alpha:?} electric constraint defect={:.15e} response={:?}",
            value(&report, "electric_constraint_defect"),
            report.get("electric_response_residuals")
        );
        let mut failed = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.008, 0.005, 0.003, 0.002], stencils::Stencil::Second),
                ([0.005, 0.003, 0.002, 0.001], stencils::Stencil::Second),
                ([0.012, 0.009, 0.006, 0.004], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 9];
                for (k, &h) in steps.iter().enumerate() {
                    for j in 0..3 {
                        let (p, m) = (moment(j, h), moment(j, -h));
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (moment(j, 2.0 * h), moment(j, -2.0 * h))
                        } else {
                            ([0.0; 3], [0.0; 3])
                        };
                        for i in 0..3 {
                            errors[3 * i + j][k] = (stencil.first(p[i], m[i], p2[i], m2[i], h)
                                - alpha[3 * i + j])
                                .abs();
                        }
                    }
                }
                errors
            },
            &mut failed,
        );
        components::assert_no_failures("static fixed-fragment electric response", &failed);
    };
    for xc in ["wb97x-d", "wb97x-v"] {
        fit(xc);
    }
}

#[test]
fn spin_paired_gks_fragment_response_matches_the_collinear_limit() {
    let system = fragment_water();
    for xc in ["wb97x-d", "wb97x-v"] {
        let input = fragment_water_controls(xc, true).with("constraint_electric_response", true);
        let eval = |settings: &Value| {
            run(
                "constrained-dft",
                &system,
                &Controls::from_value(settings).unwrap(),
            )
            .unwrap()
        };
        let rks = eval(&input);
        let gks = eval(&input.with("reference", "gks"));
        assert_eq!(gks.get("reference_type").unwrap().as_str(), Some("gks"));
        assert!(value(&gks, "electric_constraint_defect") < 1e-9);
        assert!((value(&gks, "energy") - value(&rks, "energy")).abs() < 1e-8);
        for key in ["gradient", "polarizability"] {
            let (_, actual) = gks.get(key).unwrap().numeric_array().unwrap();
            let (_, expected) = rks.get(key).unwrap().numeric_array().unwrap();
            let defect = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            println!("constrained water {xc} GKS {key}={actual:?} RKS defect={defect:.15e}");
            assert!(defect < 2e-7, "{key} defect {defect}");
        }
    }
}
