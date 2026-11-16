// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static electric and mass response of molecular positron references.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::dipole_matrices;
use dft_rs::linalg::{solve_linear_matrix, Matrix};
use dft_rs::math::Vec3;
use dft_rs::multicomponent::correlation::{CorrelationModel, PositronRelaxation};
use dft_rs::multicomponent::response::CoupledResponseSolver;
use dft_rs::multicomponent::{
    gaussian_particle_basis, solve, InitialState, MulticomponentOptions, QuantumParticle,
};
use dft_rs::resources::ResourceLimits;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::DftError;
use stencils::{fit_components_staged, Stencil};

const NUMBERS: [u8; 4] = [7, 1, 1, 1];
const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.21, 0.09],
    [1.61, 0.77, -0.29],
    [-1.34, 0.61, 0.33],
    [0.31, -1.74, 0.58],
];
const POSITRON_CENTER: [f64; 3] = [0.45, -0.17, 1.8];

fn controls(correlation: &str) -> Value {
    Value::object()
        .with("xc", "m06")
        .with("reference", "rks")
        .with("basis", "sto-3g")
        .with("grid_level", 1_usize)
        .with("grid_weight_cutoff", 0.0)
        .with("dispersion", "none")
        .with("cpu_threads", 1_usize)
        .with("conv_tol", 1e-13)
        .with("conv_tol_grad", 1e-11)
        .with("conv_tol_dp", 1e-11)
        .with("species_tolerance", 1e-11)
        .with("species_max_iterations", 500_usize)
        .with(
            "particles",
            vec![Value::object()
                .with("kind", "positron")
                .with("position", POSITRON_CENTER)
                .with("exponents", [0.12, 0.38])
                .with("lmax", 1_usize)
                .with("correlation", correlation)],
        )
}

fn calculated(system: &System, options: &Value) -> Value {
    let report = run(
        "multicomponent",
        system,
        &Controls::from_value(options).unwrap(),
    )
    .unwrap();
    assert_eq!(
        report.get("converged").and_then(Value::as_bool),
        Some(true),
        "{}",
        report.to_json()
    );
    report
}

fn array(report: &Value, name: &str) -> Vec<f64> {
    report.get(name).unwrap().numeric_array().unwrap().1
}

#[test]
fn m06_positron_dynamic_dipole_recovers_the_static_limit_and_signed_frequency_symmetry() {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let endpoint = 0.036;
    let report = calculated(
        &system,
        &controls("drummond-2011")
            .with("species_polarizability", true)
            .with("quantum_position_constraint", true)
            .with("frequencies", [0.0, endpoint, -endpoint]),
    );
    let alpha = array(report.get("polarizability").unwrap(), "tensor");
    let dynamic = report.get("dynamic_polarizability").unwrap();
    let real = array(dynamic, "tensor_real");
    let imaginary = array(dynamic, "tensor_imaginary");
    let reference_residual = dynamic.get("reference_residual").unwrap().as_f64().unwrap();
    let stationary_residual = array(dynamic, "response_residuals");
    let position_residual = array(dynamic, "position_response_residuals");
    assert!(reference_residual < 1e-10);
    assert!(stationary_residual
        .iter()
        .chain(&position_residual)
        .all(|&x| x < 1e-10));
    assert_eq!(real.len(), 27);
    assert_eq!(imaginary.len(), 27);
    for a in 0..3 {
        for b in 0..3 {
            let i = a * 3 + b;
            assert!((real[i] - alpha[i]).abs() < 5e-9);
            assert!((real[9 + i] - real[18 + i]).abs() < 5e-9);
            assert!((real[9 + i] - real[9 + b * 3 + a]).abs() < 5e-9);
            assert!(imaginary[i].abs() < 5e-12);
            assert!((imaginary[9 + i] + imaginary[18 + i]).abs() < 5e-12);
        }
    }
    assert!((real[9] - alpha[0]).abs() > 1e-5);
    println!(
        "C1 NH3 plus constrained positron: omega={endpoint} Eh, alpha={:?} + i {:?}, reference={reference_residual:.6e}, response={stationary_residual:?}, position={position_residual:?}",
        &real[9..18], &imaginary[9..18],
    );
}

#[test]
fn m06_positron_static_polarizability_fits_all_nine_components() {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let setup = prepare_molecular(
        &NUMBERS,
        &POSITIONS,
        &MolecularMethod {
            xc: Some("m06".into()),
            dispersion: "none".into(),
            grid_level: 1,
            grid_weight_cutoff: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let electronic_dipoles = dipole_matrices(&setup.basis, Vec3::zero());
    let nuclear_dipole: [f64; 3] = std::array::from_fn(|axis| {
        NUMBERS
            .iter()
            .zip(POSITIONS)
            .map(|(&z, r)| f64::from(z) * r[axis])
            .sum()
    });
    let dipole = |report: &Value| -> [f64; 3] {
        // Reconstruct the finite-field dipole from the solved electronic AO
        // density and the positron expectation position in the laboratory frame.
        let density = report
            .get("electronic_density")
            .unwrap()
            .as_array()
            .unwrap();
        let quantum = report.get("quantum").unwrap().as_array().unwrap();
        let positron = array(&quantum[0], "position");
        std::array::from_fn(|axis| {
            let electronic = density
                .iter()
                .map(|spin| {
                    let p = spin.numeric_array().unwrap().1;
                    p.iter()
                        .zip(electronic_dipoles[axis].as_slice())
                        .map(|(p, r)| p * r)
                        .sum::<f64>()
                })
                .sum::<f64>();
            nuclear_dipole[axis] - electronic + positron[axis]
        })
    };
    let stages = [
        ([0.004, 0.003, 0.002, 0.001], Stencil::Second),
        ([0.002, 0.0015, 0.001, 0.0005], Stencil::Second),
        ([0.008, 0.006, 0.004, 0.002], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    for correlation in ["none", "drummond-2011"] {
        let options = controls(correlation);
        let reference = calculated(
            &system,
            &options.clone().with("species_polarizability", true),
        );
        let alpha = reference.get("polarizability").unwrap();
        let tensor = array(alpha, "tensor");
        assert_eq!(tensor.len(), 9);
        assert!(array(alpha, "response_residuals").iter().all(|x| *x < 1e-8));
        assert!(alpha.get("raw_asymmetry").unwrap().as_f64().unwrap() < 1e-8);
        let field_free_dipole = dipole(&reference);
        println!(
            "M06 NH3+positron correlation={correlation} E={} alpha={tensor:?} dipole={field_free_dipole:?} SCF_orbital={} SCF_density={} raw_asymmetry={} response_residuals={:?} positron_orbital={} positron_density={}",
            reference.get("energy").unwrap().as_f64().unwrap(),
            reference.get("electronic_orbital_residual").unwrap().as_f64().unwrap(),
            reference.get("electronic_density_residual").unwrap().as_f64().unwrap(),
            alpha.get("raw_asymmetry").unwrap().as_f64().unwrap(),
            array(alpha, "response_residuals"),
            reference.get("quantum").unwrap().as_array().unwrap()[0].get("orbital_residual").unwrap().as_f64().unwrap(),
            reference.get("quantum").unwrap().as_array().unwrap()[0].get("density_residual").unwrap().as_f64().unwrap(),
        );
        let mut labels = Vec::new();
        for b in 0..3 {
            labels.push(format!("M06/{correlation} field {b} energy derivative"));
            labels.extend((0..3).map(|a| format!("M06/{correlation} alpha[{a},{b}]")));
        }
        fit_components_staged(
            &labels,
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; labels.len()];
                for b in 0..3 {
                    let displaced = |h| {
                        let mut f = [0.0; 3];
                        f[b] = h;
                        let report = calculated(&system, &options.clone().with("field", f));
                        (
                            report.get("energy").unwrap().as_f64().unwrap(),
                            dipole(&report),
                        )
                    };
                    for (k, &h) in steps.iter().enumerate() {
                        let ((ep, dp), (em, dm)) = (displaced(h), displaced(-h));
                        let ((ep2, dp2), (em2, dm2)) = if stencil == Stencil::Fourth {
                            (displaced(2.0 * h), displaced(-2.0 * h))
                        } else {
                            ((0.0, [0.0; 3]), (0.0, [0.0; 3]))
                        };
                        errors[4 * b][k] =
                            (stencil.first(ep, em, ep2, em2, h) + field_free_dipole[b]).abs();
                        for a in 0..3 {
                            errors[4 * b + a + 1][k] = (stencil
                                .first(dp[a], dm[a], dp2[a], dm2[a], h)
                                - tensor[3 * a + b])
                                .abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("molecular positron static electric response", &failures);
}

#[test]
fn unsupported_positron_response_requests_have_named_errors() {
    let system = System::molecule(vec![1, 1], vec![[0.0; 3], [0.0, 0.0, 1.4]]);
    let options = controls("drummond-2011");
    let error = |request: Value, system: &System| {
        run(
            "multicomponent",
            system,
            &Controls::from_value(&request).unwrap(),
        )
        .unwrap_err()
    };
    for order in [1_usize, 2] {
        let result = error(
            options.clone().with("species_derivative_order", order),
            &system,
        );
        assert!(matches!(
            result,
            DftError::Unsupported { feature, missing }
                if feature.contains("added quantum particle") && missing.contains("Gaussian centers")
        ));
    }
    let dynamic = error(
        options
            .clone()
            .with("species_polarizability", true)
            .with("frequency", 0.02),
        &system,
    );
    assert!(dynamic.to_string().contains("frequency"), "{dynamic}");
    let quadrupole = error(
        options
            .clone()
            .with("species_quadrupole_polarizability", true),
        &system,
    );
    assert!(matches!(
        quadrupole,
        DftError::Unsupported { missing, .. } if missing.contains("field-gradient")
    ));
    let mut periodic = system;
    periodic.cell = Some([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]);
    periodic.pbc = [true; 3];
    let result = error(options, &periodic);
    assert!(result.to_string().contains("periodic"), "{result}");
}

fn constrained(options: Value) -> Value {
    options
        .with("quantum_position_constraint", true)
        .with("quantum_position_targets", vec![POSITRON_CENTER])
}

fn energy(report: &Value) -> f64 {
    report.get("energy").unwrap().as_f64().unwrap()
}

fn field_energy(system: &System, options: &Value, field: [f64; 3]) -> f64 {
    energy(&calculated(system, &options.clone().with("field", field)))
}

fn negative_mixed_field_derivative(energy: impl Fn(bool, bool) -> f64, step: f64) -> f64 {
    let slope = |positive_b| {
        Stencil::Second.first(
            energy(true, positive_b),
            energy(false, positive_b),
            0.0,
            0.0,
            step,
        )
    };
    -Stencil::Second.first(slope(true), slope(false), 0.0, 0.0, step)
}

// Eight independently reconverged fields determine all three distinct mixed
// Hessian entries. For each pair, averaging over the third field sign
// cancels its odd terms. Two shared central stencils then isolate the mixed
// coefficient. The third axis changes only the O(h^2) truncation term.
fn mixed_field_energy(system: &System, options: &Value, step: f64) -> [f64; 3] {
    let vertices: [f64; 8] = std::array::from_fn(|i| {
        let field = std::array::from_fn(|axis| if i & (1 << axis) == 0 { -step } else { step });
        field_energy(system, options, field)
    });
    [(0, 1), (0, 2), (1, 2)].map(|(a, b)| {
        let other = 3 - a - b;
        let symmetrized = |positive_a: bool, positive_b: bool| {
            let index = (usize::from(positive_a) << a) | (usize::from(positive_b) << b);
            0.5 * (vertices[index] + vertices[index | (1 << other)])
        };
        negative_mixed_field_derivative(symmetrized, step)
    })
}

fn pure_mixed_field_energy(system: &System, options: &Value, a: usize, b: usize, step: f64) -> f64 {
    negative_mixed_field_derivative(
        |positive_a, positive_b| {
            // Holding the other field at zero removes its contribution to
            // the leading truncation error of the sign-cube estimate.
            let mut field = [0.0; 3];
            field[a] = if positive_a { step } else { -step };
            field[b] = if positive_b { step } else { -step };
            field_energy(system, options, field)
        },
        step,
    )
}

#[test]
fn m06_positron_mixed_field_energy_fits_off_diagonal_alpha() {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let cube_steps = [0.004, 0.003, 0.002, 0.001];
    let pure_plane_steps = [0.0015, 0.001, 0.00075, 0.0005];
    let mut failures = Vec::new();
    for correlation in ["none", "drummond-2011"] {
        let options = controls(correlation);
        let reference = calculated(
            &system,
            &options.clone().with("species_polarizability", true),
        );
        let response = reference.get("polarizability").unwrap();
        let tensor = array(response, "tensor");
        let reciprocity = response.get("raw_asymmetry").unwrap().as_f64().unwrap();
        assert!(reciprocity < 1e-8);
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let expected = pairs.map(|(a, b)| 0.5 * (tensor[3 * a + b] + tensor[3 * b + a]));
        let labels = pairs.map(|(a, b)| format!("M06/{correlation} mixed energy alpha[{a},{b}]"));
        println!(
            "M06/{correlation} mixed energy alpha={expected:?} raw_asymmetry={reciprocity:.12e}"
        );
        let mut unresolved = Vec::new();
        fit_components_staged(
            &labels,
            &[(cube_steps, Stencil::Second)],
            |steps, _| {
                let samples = steps.map(|h| mixed_field_energy(&system, &options, h));
                (0..3)
                    .map(|i| samples.map(|values| (values[i] - expected[i]).abs()))
                    .collect()
            },
            &mut unresolved,
        );
        for (i, label) in labels
            .iter()
            .enumerate()
            .filter(|(_, label)| unresolved.contains(label))
        {
            let (a, b) = pairs[i];
            let signed_errors = pure_plane_steps
                .map(|h| pure_mixed_field_energy(&system, &options, a, b, h) - expected[i]);
            let fine_label = format!("{label} (fine pure-plane interval)");
            println!(
                "{fine_label}: analytic={:.16e} steps={pure_plane_steps:?} signed_errors={signed_errors:?}",
                expected[i]
            );
            if !components::component_fit_of_order(
                &fine_label,
                &pure_plane_steps,
                &signed_errors.map(f64::abs),
                Stencil::Second.order(),
            ) {
                failures.push(fine_label);
            }
        }
    }
    components::assert_no_failures("positron mixed electric energy response", &failures);
}

#[test]
fn expectation_constrained_positron_alpha_fits_the_complete_field_energy_hessian() {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let options = constrained(controls("drummond-2011"));
    let reference = calculated(
        &system,
        &options.clone().with("species_polarizability", true),
    );
    let e0 = energy(&reference);
    let response = reference.get("polarizability").unwrap();
    let tensor = array(response, "tensor");
    let residuals = array(response, "response_residuals");
    let position_residuals = array(response, "position_response_residuals");
    let positions = reference.get("position_constraints").unwrap();
    let constraint_residual = positions.get("residual").unwrap().as_f64().unwrap();
    assert!(residuals.iter().all(|r| *r < 1e-8));
    assert!(position_residuals.iter().all(|r| r.abs() < 1e-8));
    assert!(constraint_residual < 1e-8);
    assert!(response.get("raw_asymmetry").unwrap().as_f64().unwrap() < 1e-8);
    println!(
        "M06 constrained NH3+positron E={e0:.16e} alpha={tensor:?} residuals={residuals:?} position_response_residuals={position_residuals:?} position_residual={constraint_residual:.12e}"
    );
    let labels: Vec<_> = (0..3)
        .flat_map(|a| (0..3).map(move |b| format!("fixed expectation alpha[{a},{b}]")))
        .collect();
    let stages = [
        ([0.008, 0.006, 0.004, 0.002], Stencil::Second),
        ([0.004, 0.003, 0.002, 0.001], Stencil::Second),
    ];
    let mut failures = Vec::new();
    fit_components_staged(
        &labels,
        &stages,
        |steps, _| {
            let samples = steps.map(|h| {
                let mixed = mixed_field_energy(&system, &options, h);
                let mut alpha = [0.0; 9];
                for (i, (a, b)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                    alpha[3 * a + b] = mixed[i];
                    alpha[3 * b + a] = mixed[i];
                }
                for a in 0..3 {
                    let mut field = [0.0; 3];
                    field[a] = h;
                    let plus = field_energy(&system, &options, field);
                    field[a] = -h;
                    let minus = field_energy(&system, &options, field);
                    alpha[3 * a + a] = -((plus - e0) + (minus - e0)) / h.powi(2);
                }
                alpha
            });
            (0..9)
                .map(|i| samples.map(|values| (values[i] - tensor[i]).abs()))
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("position-constrained positron electric response", &failures);
}

#[test]
fn m06_positron_mass_gradient_and_hessian_fit_independent_mass_perturbations() {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let mut failures = Vec::new();
    let stages = [
        ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
        ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
    ];
    for fixed in [false, true] {
        let mut options = controls("drummond-2011");
        if fixed {
            options = constrained(options);
        }
        let particle = options.get("particles").unwrap().as_array().unwrap()[0].clone();
        let reference = calculated(
            &system,
            &options.clone().with("quantum_mass_response", true),
        );
        let mass = reference.get("mass_response").unwrap();
        let grad = array(mass, "gradient");
        let hess = array(mass, "hessian");
        let residuals = array(mass, "response_residuals");
        let position_residuals = array(mass, "position_response_residuals");
        assert_eq!(grad.len(), 1);
        assert_eq!(hess.len(), 1);
        assert!(residuals.iter().all(|v| v.abs() < 1e-8));
        assert!(position_residuals.iter().all(|v| v.abs() < 1e-8));
        println!(
            "M06 NH3+positron constraint={fixed} mass_gradient={grad:?} mass_hessian={hess:?} response_residuals={residuals:?} position_response_residuals={position_residuals:?}"
        );
        let sample = |m: f64| {
            let report = calculated(
                &system,
                &options
                    .clone()
                    .with("particles", vec![particle.clone().with("mass", m)]),
            );
            let quantum = &report.get("quantum").unwrap().as_array().unwrap()[0];
            let kinetic = quantum.get("kinetic_energy").unwrap().as_f64().unwrap();
            // The mass only scales the kinetic operator; the derivative of
            // the variational minimum at the displaced mass is -T/m.
            (energy(&report), -kinetic / m)
        };
        fit_components_staged(
            &[
                format!("positron constraint={fixed} mass gradient"),
                format!("positron constraint={fixed} mass Hessian"),
            ],
            &stages,
            |steps, stencil| {
                let values = steps.map(|h| {
                    let (ep, gp) = sample(1.0 + h);
                    let (em, gm) = sample(1.0 - h);
                    [
                        (stencil.first(ep, em, 0.0, 0.0, h) - grad[0]).abs(),
                        (stencil.first(gp, gm, 0.0, 0.0, h) - hess[0]).abs(),
                    ]
                });
                vec![values.map(|e| e[0]), values.map(|e| e[1])]
            },
            &mut failures,
        );
    }
    components::assert_no_failures("positron quantum mass response", &failures);
}

#[test]
fn h2_minimal_positron_stability_reports_the_only_available_root() {
    let numbers = [1, 1];
    let positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]];
    let center = [0.24, -0.31, 0.18];
    let exponents = [0.4];
    let tol = 1e-11;
    let controls = controls("none")
        .with(
            "particles",
            vec![Value::object()
                .with("kind", "positron")
                .with("position", center)
                .with("exponents", exponents)
                .with("lmax", 0_usize)],
        )
        .with("species_stability", "report")
        .with("species_stability_roots", 3_usize);
    let system = System::molecule(numbers.to_vec(), positions.to_vec());
    let report = calculated(&system, &controls);
    let stability = report.get("species_stability").unwrap();
    let eigenvalues = array(stability, "eigenvalues");
    let residuals = array(stability, "residuals");
    assert_eq!(stability.get("dimension").unwrap().as_f64(), Some(1.0));
    assert_eq!(eigenvalues.len(), 1);
    assert_eq!(residuals.len(), 1);
    assert!(residuals[0] < 1e-6);

    let method = MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
    let setup = prepare_molecular(&numbers, &positions, &method).unwrap();
    let positron = QuantumParticle {
        label: "positron".into(),
        charge: 1.0,
        mass: 1.0,
        basis: gaussian_particle_basis(Vec3::new(center[0], center[1], center[2]), &exponents, 0)
            .unwrap(),
        replaced_atom: None,
        correlation_channel: None,
    };
    let particles = [positron];
    let correlation = CorrelationModel::none();
    let reference = solve(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
        &particles,
        &correlation,
        &MulticomponentOptions {
            tolerance: tol,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(reference.converged);
    let solver = CoupledResponseSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
        &particles,
        &correlation,
        &reference,
    )
    .unwrap();
    assert_eq!(solver.dimension(), 1);
    // The only dense matrix entry is the eigenvalue, independent of the
    // iterative diagonalization and its trial-subspace size.
    let exact = solver.apply_real(&[1.0]).unwrap()[0];
    let defect = (eigenvalues[0] - exact).abs();
    println!(
        "M06 H2+positron stability eigenvalue={:.16e} dense={exact:.16e} defect={defect:.12e} residual={:.12e}",
        eigenvalues[0], residuals[0]
    );
    assert!(defect < tol);
}

#[test]
fn m06_positron_dense_orbital_solve_agrees_for_all_polarizability_entries() {
    check_m06_positron_dense_orbital_solve(false);
}

#[test]
fn m06_positron_tight_scf_and_response_agree_with_the_dense_orbital_solve() {
    check_m06_positron_dense_orbital_solve(true);
}

fn check_m06_positron_dense_orbital_solve(tight_reference: bool) {
    let tight_tolerance = 1e-14;
    let species_tolerance = if tight_reference {
        tight_tolerance
    } else {
        1e-11
    };
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    ResourceLimits::new(Some(1), None)
        .unwrap()
        .install(|| {
            for model in ["none", "drummond-2011"] {
                let mut request = controls(model);
                if tight_reference {
                    for name in ["conv_tol", "conv_tol_grad", "conv_tol_dp"] {
                        request.set(name, tight_tolerance);
                    }
                }
                let setup = prepare_molecular(
                    &NUMBERS,
                    &POSITIONS,
                    &MolecularMethod::from_controls(
                        &Controls::from_value(&request).unwrap(),
                    ).unwrap(),
                ).unwrap();
                let [x, y, z] = POSITRON_CENTER;
                let particles = [QuantumParticle {
                    label: "positron".into(),
                    charge: 1.0,
                    mass: 1.0,
                    basis: gaussian_particle_basis(Vec3::new(x, y, z), &[0.12, 0.38], 1).unwrap(),
                    replaced_atom: None,
                    correlation_channel: (model != "none").then_some(0),
                }];
                let correlation = if model == "none" {
                    CorrelationModel::none()
                } else {
                    CorrelationModel::single(Box::new(PositronRelaxation::drummond_2011()))
                };
                let reference = solve(
                    &setup.molecule,
                    &setup.basis,
                    &setup.options,
                    &setup.functional,
                    setup.grid.as_ref(),
                    &particles,
                    &correlation,
                    &MulticomponentOptions { tolerance: species_tolerance, ..Default::default() },
                ).unwrap();
                assert!(reference.converged);
                let solver = CoupledResponseSolver::new(
                    &setup.molecule,
                    &setup.basis,
                    &setup.options,
                    &setup.functional,
                    setup.grid.as_ref(),
                    &particles,
                    &correlation,
                    &reference,
                ).unwrap();
                let config = ResponseSolverConfig::default();
                let iterative = solver.polarizability(&config).unwrap();
                assert!(iterative.responses.iter().all(|r| r.converged));
                let e_moment = dipole_matrices(&setup.basis, Vec3::zero());
                let q_moment = dipole_matrices(&particles[0].basis, Vec3::zero());
                let project = |trial: &InitialState, j: usize| {
                    let electron = trial.electronic_density.iter()
                        .map(|d| d.frobenius_dot(&e_moment[j]))
                        .sum::<f64>();
                    let c = Matrix::from_row_major(trial.quantum[0].len(), 1, trial.quantum[0].clone());
                    let positron = c.frobenius_dot(&q_moment[j].matmul_seq(&c));
                    (electron, positron)
                };
                let n = solver.dimension();
                let mut hessian = Matrix::zeros(n, n);
                let mut dipole_jacobian = Matrix::zeros(n, 3);
                for j in 0..n {
                    let mut unit = vec![0.0; n];
                    unit[j] = 1.0;
                    for (i, v) in solver.apply_real(&unit).unwrap().into_iter().enumerate() {
                        hessian[(i, j)] = v;
                    }
                    let positive = solver.rotated_state(&unit, 1.0).unwrap();
                    let negative = solver.rotated_state(&unit, -1.0).unwrap();
                    for axis in 0..3 {
                        let (ep, qp) = project(&positive, axis);
                        let (em, qm) = project(&negative, axis);
                        // A one-slot rotation has (c_i ± s c_a)/sqrt(1+s²).
                        // The antisymmetric density difference is exactly
                        // 2/(1+s²) times its tangent in the unit amplitude.
                        // Here s²=1/2 electronically and 1 for the positron.
                        dipole_jacobian[(j, axis)] = (qp - qm) - 0.75 * (ep - em);
                    }
                }
                let x = solve_linear_matrix(&hessian, &dipole_jacobian).unwrap();
                let residual = hessian.matmul_seq(&x).max_abs_diff(&dipole_jacobian);
                let dense = dipole_jacobian.transpose_matmul_seq(&x);
                let symmetry = hessian.max_abs_diff(&hessian.transpose());
                assert!(residual < 1e-12 && symmetry < 1e-10);
                // Keep the stationary density and the operator fixed so this
                // measures linear-solve accuracy separately from SCF error.
                let tightened = solver.polarizability(&ResponseSolverConfig {
                    tolerance: tight_tolerance,
                    ..config.clone()
                }).unwrap();
                let cases = [
                    ("default", &iterative, config.tolerance, 1e-8),
                    ("tight", &tightened, tight_tolerance, 2e-12),
                ];
                for (setting, answer, tolerance, tensor_gate) in cases {
                    let errors: [[f64; 3]; 3] = std::array::from_fn(|a| {
                        std::array::from_fn(|b| 0.5 * dense[(a, b)] - answer.tensor[a][b])
                    });
                    let max_error = errors.iter().flatten().map(|v| v.abs()).fold(0.0_f64, f64::max);
                    let sources: Vec<_> = answer.responses.iter().map(|r| r.relative_residual).collect();
                    let scf: Vec<_> = answer.responses.iter().map(|r| r.reference_residual).collect();
                    let full_acceptance = answer.responses.iter().all(|r| r.converged);
                    println!(
                        "M06/{model} {setting} dense dimension={n} E={:.16e} residual={residual:.12e} raw_hessian_asymmetry={symmetry:.12e} signed_alpha_error={errors:?} max_abs_error={max_error:.12e} krylov_residuals={sources:?} reference_residuals={scf:?} full_acceptance={full_acceptance}",
                        reference.energy
                    );
                    assert!(sources.iter().all(|r| *r <= tolerance));
                    assert_eq!(full_acceptance, scf.iter().all(|r| *r <= tolerance));
                    if tight_reference {
                        assert!(full_acceptance, "tight reference rejected for {model}/{setting}");
                    }
                    assert!(max_error < tensor_gate);
                }
            }
        })
        .unwrap();
}
