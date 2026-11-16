// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete harmonic-mode and reference-route contracts, without symmetry fills.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::constants::{atomic_mass_au, AMU_TO_AU_MASS, HARTREE_TO_CM1};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::properties::vibrational::vibrational_analysis_with_projection;
use dft_rs::system::{Atom, Molecule};

fn c1() -> Molecule {
    Molecule::new(
        [
            (1, [0.13, -0.21, 0.19]),
            (6, [1.57, 0.16, -0.08]),
            (8, [-0.19, 1.93, 0.37]),
            (9, [0.42, 0.29, 2.28]),
        ]
        .into_iter()
        .map(|(z, r)| Atom {
            z,
            position: Vec3::new(r[0], r[1], r[2]),
        })
        .collect(),
    )
}

#[test]
fn complete_cartesian_modes_and_ir_match_a_closed_dense_spectrum() {
    let molecule = c1();
    let n = 3 * molecule.len();
    let mut u = (0..n).map(|i| 0.17 + 0.037 * i as f64).collect::<Vec<_>>();
    let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    for value in &mut u {
        *value /= norm;
    }
    let q = Matrix::from_row_major(
        n,
        n,
        (0..n)
            .flat_map(|i| {
                let u = &u;
                (0..n).map(move |j| {
                    if i == j {
                        1.0 - 2.0 * u[i] * u[j]
                    } else {
                        -2.0 * u[i] * u[j]
                    }
                })
            })
            .collect(),
    );
    let mass = (0..n)
        .map(|i| atomic_mass_au(molecule.atoms[i / 3].z))
        .collect::<Vec<_>>();
    let omega = (0..n)
        .map(|i| 0.015 + 0.0027 * i as f64)
        .collect::<Vec<_>>();
    let mut hessian = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            hessian[(i, j)] = (mass[i] * mass[j]).sqrt()
                * (0..n)
                    .map(|k| q[(i, k)] * omega[k].powi(2) * q[(j, k)])
                    .sum::<f64>();
        }
    }
    assert!(hessian.as_slice().iter().all(|&v| v != 0.0));
    let apt = Matrix::from_row_major(
        3,
        n,
        (0..3)
            .flat_map(|a| (0..n).map(move |i| (0.13 + 0.017 * i as f64 + 0.11 * a as f64).sin()))
            .collect(),
    );
    let result =
        vibrational_analysis_with_projection(&molecule, &hessian, Some(&apt), false).unwrap();
    assert_eq!(result.n_projected, 0);
    assert_eq!(result.frequencies_cm.len(), n);
    for k in 0..n {
        assert!((result.frequencies_cm[k] - omega[k] * HARTREE_TO_CM1).abs() < 1e-8);
        let strength = 974.8801
            * (0..3)
                .map(|a| {
                    (0..n)
                        .map(|i| apt[(a, i)] * q[(i, k)] / (mass[i] / AMU_TO_AU_MASS).sqrt())
                        .sum::<f64>()
                        .powi(2)
                })
                .sum::<f64>();
        assert!((result.ir_intensities[k] - strength).abs() < 1e-8);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (result.modes[(i, k)] * result.modes[(j, k)] - q[(i, k)] * q[(j, k)]).abs()
                        < 1e-12
                );
            }
        }
    }
}

#[test]
fn c1_spring_hessian_has_all_component_fits_and_the_independent_bond_spectrum() {
    let molecule = c1();
    let n = 3 * molecule.len();
    let mut bonds = Vec::new();
    let mut hessian = Matrix::zeros(n, n);
    for a in 0..molecule.len() {
        for b in 0..a {
            let r = molecule.atoms[a].position - molecule.atoms[b].position;
            let length = r.norm();
            let direction = r / length;
            let stiffness = 0.13 + 0.047 * bonds.len() as f64;
            bonds.push((a, b, length, stiffness, direction));
            for i in 0..3 {
                for j in 0..3 {
                    let value = stiffness * direction.to_array()[i] * direction.to_array()[j];
                    hessian[(3 * a + i, 3 * a + j)] += value;
                    hessian[(3 * b + i, 3 * b + j)] += value;
                    hessian[(3 * a + i, 3 * b + j)] -= value;
                    hessian[(3 * b + i, 3 * a + j)] -= value;
                }
            }
        }
    }
    let gradient = |geometry: &Molecule| {
        let mut out = vec![0.0; n];
        for &(a, b, length, stiffness, _) in &bonds {
            let r = geometry.atoms[a].position - geometry.atoms[b].position;
            let distance = r.norm();
            let force = r * (stiffness * (distance - length) / distance);
            for (i, v) in force.to_array().into_iter().enumerate() {
                out[3 * a + i] += v;
                out[3 * b + i] -= v;
            }
        }
        out
    };
    assert!(hessian.as_slice().iter().all(|&value| value != 0.0));
    let names = (0..n)
        .flat_map(|i| (0..n).map(move |j| format!("C1 harmonic spring Hessian({i},{j})")))
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; n * n];
            for (step, &h) in steps.iter().enumerate() {
                for column in 0..n {
                    let displaced = |offset: f64| {
                        let mut moved = molecule.clone();
                        let mut r = moved.atoms[column / 3].position.to_array();
                        r[column % 3] += offset;
                        moved.atoms[column / 3].position = Vec3::new(r[0], r[1], r[2]);
                        gradient(&moved)
                    };
                    let plus = displaced(h);
                    let minus = displaced(-h);
                    let plus2 = displaced(2.0 * h);
                    let minus2 = displaced(-2.0 * h);
                    for row in 0..n {
                        errors[row * n + column][step] =
                            (stencil.first(plus[row], minus[row], plus2[row], minus2[row], h)
                                - hessian[(row, column)])
                                .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("complete C1 spring Hessian", &failures);
    let result = vibrational_analysis_with_projection(&molecule, &hessian, None, true).unwrap();
    assert_eq!(result.n_projected, 6);
    assert_eq!(result.frequencies_cm.len(), bonds.len());
    let mut gram = Matrix::zeros(bonds.len(), bonds.len());
    for (p, &(a, b, _, kp, ep)) in bonds.iter().enumerate() {
        for (q, &(c, d, _, kq, eq)) in bonds.iter().enumerate() {
            for atom in 0..molecule.len() {
                let left = if atom == a {
                    1.0
                } else if atom == b {
                    -1.0
                } else {
                    0.0
                };
                let right = if atom == c {
                    1.0
                } else if atom == d {
                    -1.0
                } else {
                    0.0
                };
                gram[(p, q)] += (kp * kq).sqrt() * left * right * ep.dot(eq)
                    / atomic_mass_au(molecule.atoms[atom].z);
            }
        }
    }
    let (mut eigenvalues, _) = symmetric_eigen(&gram).unwrap();
    eigenvalues.sort_by(f64::total_cmp);
    for (actual, expected) in result.frequencies_cm.iter().zip(eigenvalues) {
        assert!((actual - expected.sqrt() * HARTREE_TO_CM1).abs() < 1e-8);
    }
}

#[test]
fn constrained_atomic_gibbs_uses_the_shared_energy_and_declared_degeneracy() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    let system = System::molecule(vec![1], vec![[0.13, -0.21, 0.19]]);
    for reference in ["rohf", "ghf"] {
        let value = Value::object()
            .with("reference", reference)
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("multiplicity", 2)
            .with("guess", "core")
            .with("temperature", 317.0)
            .with("electronic_degeneracy", 2)
            .with("response_tolerance", 1e-12);
        let report = run(
            "gibbs-free-energy",
            &system,
            &Controls::from_value(&value).unwrap(),
        )
        .unwrap();
        assert_eq!(report.get("n_projected").unwrap().as_i64(), Some(3));
        assert_eq!(
            report
                .get("frequencies_cm1")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert_eq!(report.get("converged").unwrap().as_bool(), Some(true));
        let one = run(
            "gibbs-free-energy",
            &system,
            &Controls::from_value(&value.with("electronic_degeneracy", 1)).unwrap(),
        )
        .unwrap();
        let delta = report.get("gibbs_free_energy").unwrap().as_f64().unwrap()
            - one.get("gibbs_free_energy").unwrap().as_f64().unwrap();
        let expected =
            -dft_rs::properties::thermochemistry::BOLTZMANN_HARTREE_K * 317.0 * 2.0_f64.ln();
        assert!((delta - expected).abs() < 1e-13);
        println!(
            "{reference} Gibbs electronic-degeneracy identity defect={:.12e}",
            (delta - expected).abs()
        );
    }
}

#[test]
fn c1_constrained_frequencies_reuse_the_same_full_analytic_hessian() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    let molecule = c1();
    let system = System::molecule(
        vec![1; 4],
        molecule
            .atoms
            .iter()
            .map(|atom| atom.position.to_array())
            .collect(),
    );
    for reference in ["rohf", "ghf"] {
        let mut options = Value::object()
            .with("reference", reference)
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("charge", 1)
            .with("multiplicity", 2)
            .with("guess", "core")
            .with("conv_tol", 1e-14)
            .with("conv_tol_grad", 1e-12)
            .with("conv_tol_dp", 1e-13)
            .with("response_tolerance", 1e-12)
            .with("max_iter", 600);
        if reference == "ghf" {
            let operator = spinor_operator::source(4);
            options = options
                .with("gks_operator_real", operator.re)
                .with("gks_operator_imaginary", operator.im);
        }
        let baseline = run("hessian", &system, &Controls::from_value(&options).unwrap()).unwrap();
        let report = run(
            "frequencies",
            &system,
            &Controls::from_value(&options.with("vibrational_projection", "eckart")).unwrap(),
        )
        .unwrap();
        assert_eq!(report.get("converged").unwrap().as_bool(), Some(true));
        let (_, hessian) = baseline.get("hessian").unwrap().numeric_array().unwrap();
        let geometry = Molecule::new(
            system
                .numbers
                .iter()
                .zip(&system.positions)
                .map(|(&z, r)| Atom {
                    z,
                    position: Vec3::new(r[0], r[1], r[2]),
                })
                .collect(),
        );
        let expected = vibrational_analysis_with_projection(
            &geometry,
            &Matrix::from_row_major(12, 12, hessian),
            None,
            true,
        )
        .unwrap();
        let (_, actual) = report
            .get("frequencies_cm1")
            .unwrap()
            .numeric_array()
            .unwrap();
        assert_eq!(actual.len(), 6);
        for (a, b) in actual.iter().zip(&expected.frequencies_cm) {
            assert!((a - b).abs() < 1e-8);
        }
        assert_eq!(
            report
                .get("normal_modes_cartesian")
                .unwrap()
                .numeric_array()
                .unwrap()
                .0,
            vec![6, 4, 3]
        );
        if reference == "ghf" {
            assert_eq!(report.get("ir_intensities_km_mol"), Some(&Value::Null));
        }
        println!("{reference} C1 all{} harmonic frequencies agree with the independent shared-Hessian contraction",actual.len());
    }
}

#[test]
fn double_hybrid_frequencies_include_the_full_correlation_hessian() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    let system = System::molecule(vec![1; 2], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "b2plyp")
            .with("reference", "rks")
            .with("basis", "sto-3g")
            .with("grid_level", 0)
            .with("grid_weight_cutoff", 0.0)
            .with("guess", "core")
            .with("conv_tol", 1e-13)
            .with("response_tolerance", 1e-12)
            .with("temperature", 0.0)
            .with("dispersion", "none"),
    )
    .unwrap();
    let report = run("frequencies", &system, &controls).unwrap();
    assert_eq!(
        report.get("converged").unwrap().as_bool(),
        Some(true),
        "{}",
        report.to_json_pretty()
    );
    let equations = report
        .get("coefficient_equations")
        .unwrap()
        .as_array()
        .unwrap();
    let residuals = report
        .get("response_residuals")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    assert_eq!(equations.len(), residuals.len());
    for (equation, &linear) in equations.iter().zip(&residuals) {
        assert!(linear.is_finite() && linear <= 1e-12);
        let inner = equation
            .get("inner_linear_relative_residual")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!(inner.is_finite() && inner <= 1e-12);
        assert_eq!(
            equation.get("linear_relative_residual").unwrap().as_f64(),
            Some(linear)
        );
        let source = equation.get("source_norm").unwrap().as_f64().unwrap();
        let norm = equation
            .get("reconstructed_norm")
            .unwrap()
            .as_f64()
            .unwrap();
        let ratio = equation
            .get("reconstructed_relative_residual")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!(source.is_finite() && source >= 0.0 && norm.is_finite() && norm <= 1e-12);
        assert_eq!(ratio, if source == 0.0 { norm } else { norm / source });
    }
    let (_, total) = report.get("hessian").unwrap().numeric_array().unwrap();
    let (_, parent) = report
        .get("parent_hessian")
        .unwrap()
        .numeric_array()
        .unwrap();
    let (_, pt2) = report.get("pt2_hessian").unwrap().numeric_array().unwrap();
    let (_, dispersion) = report
        .get("dispersion_hessian")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert!(pt2.iter().any(|value| value.abs() > 1e-10));
    let assembled = parent
        .iter()
        .zip(&pt2)
        .zip(&dispersion)
        .map(|((a, b), c)| a + b + c)
        .collect::<Vec<_>>();
    for (a, b) in total.iter().zip(&assembled) {
        assert!((a - b).abs() < 1e-12);
    }
    let molecule = Molecule::new(
        system
            .positions
            .iter()
            .map(|r| Atom {
                z: 1,
                position: Vec3::new(r[0], r[1], r[2]),
            })
            .collect(),
    );
    let expected = vibrational_analysis_with_projection(
        &molecule,
        &Matrix::from_row_major(6, 6, assembled),
        None,
        true,
    )
    .unwrap();
    let (_, actual) = report
        .get("frequencies_cm1")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert_eq!(actual.len(), 1);
    assert!((actual[0] - expected.frequencies_cm[0]).abs() < 1e-8);
    println!(
        "B2PLYP full PT2 harmonic frequency={:.12e} cm^-1",
        actual[0]
    );
}
