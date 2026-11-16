// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common-orbital flag TDA against an independently assembled determinant
//! Hamiltonian, with explicit S^2 checks of the scalar-excitation space.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::linalg::Matrix;
use dft_rs::response::nuclear::open_shell_tda_states;

fn operate(mut state: usize, operations: &[(usize, bool)]) -> Option<(usize, f64)> {
    let mut sign = 1.0;
    for &(orbital, creation) in operations {
        let bit = 1usize << orbital;
        if (state & bit != 0) == creation {
            return None;
        }
        if (state & (bit - 1)).count_ones() % 2 != 0 {
            sign = -sign;
        }
        state ^= bit;
    }
    Some((state, sign))
}
fn spin_step(vector: &[f64], n: usize, raising: bool) -> Vec<f64> {
    let mut out = vec![0.0; vector.len()];
    for (state, &weight) in vector.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        for orbital in 0..n {
            let (p, q) = if raising {
                (orbital, n + orbital)
            } else {
                (n + orbital, orbital)
            };
            if let Some((target, sign)) = operate(state, &[(q, false), (p, true)]) {
                out[target] += sign * weight;
            }
        }
    }
    out
}
fn verify_spin(vector: &[f64], n: usize, spin: f64) {
    let plus_minus = spin_step(&spin_step(vector, n, false), n, true);
    let minus_plus = spin_step(&spin_step(vector, n, true), n, false);
    for (state, &weight) in vector.iter().enumerate() {
        let up = (state & ((1usize << n) - 1)).count_ones() as f64;
        let down = (state >> n).count_ones() as f64;
        let sz = 0.5 * (up - down);
        let image = sz * sz * weight + 0.5 * (plus_minus[state] + minus_plus[state]);
        assert!((image - spin * (spin + 1.0) * weight).abs() < 1e-12);
    }
}

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

fn check_system(system: &System, charge: i32, multiplicity: usize) -> f64 {
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let values = Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("reference", if multiplicity == 1 { "rks" } else { "roks" })
        .with("charge", charge)
        .with("multiplicity", multiplicity)
        .with("guess", "core")
        .with("max_iter", 500usize)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-14)
        .with("conv_tol_dp", 1e-14)
        .with("grid_weight_cutoff", 0.0);
    let method = MolecularMethod::from_controls(&Controls::from_value(&values).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let (reference, orbitals) = if multiplicity == 1 {
        let reference = run_molecular_reference(&setup).unwrap().scf;
        let orbitals = reference.mo_coeff[0].clone();
        (reference, orbitals)
    } else {
        dft_rs::scf::roks::run_roks_with_orbitals(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )
        .unwrap()
    };
    assert!(reference.converged);
    let nc = reference.n_beta;
    let no = reference.n_alpha - nc;
    let n = orbitals.cols;
    assert_eq!(n, 4);
    let report = open_shell_tda_states(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference,
        &orbitals,
        &setup.options,
        32,
        64,
        &Default::default(),
    )
    .unwrap();
    let bits = 1usize << (2 * n);
    let determinant =
        ((1usize << reference.n_alpha) - 1) | (((1usize << reference.n_beta) - 1) << n);
    let mut reference_vector = vec![0.0; bits];
    reference_vector[determinant] = 1.0;
    let spin = 0.5 * (multiplicity - 1) as f64;
    verify_spin(&reference_vector, n, spin);
    let mut pairs = Vec::new();
    for i in 0..nc {
        for a in nc..nc + no {
            pairs.push((i, a));
        }
    }
    for i in 0..nc {
        for a in nc + no..n {
            pairs.push((i, a));
        }
    }
    for i in nc..nc + no {
        for a in nc + no..n {
            pairs.push((i, a));
        }
    }
    assert_eq!(pairs.len(), report.states.len());
    let basis = pairs
        .iter()
        .enumerate()
        .map(|(k, &(i, a))| {
            let mut vector = vec![0.0; bits];
            for spin in 0..2 {
                if let Some((target, sign)) =
                    operate(determinant, &[(spin * n + i, false), (spin * n + a, true)])
                {
                    vector[target] += sign;
                }
            }
            let norm = vector.iter().map(|x| x * x).sum::<f64>();
            assert!((norm - report.berry_weights[k]).abs() < 1e-14);
            for v in &mut vector {
                *v /= norm.sqrt();
            }
            verify_spin(&vector, n, spin);
            vector
        })
        .collect::<Vec<_>>();
    let h = orbitals
        .transpose()
        .matmul_seq(&reference.h_core)
        .matmul_seq(&orbitals);
    let eri =
        dft_rs::integrals::eri_incore(&setup.basis, dft_rs::integrals::CoulombKernel::Coulomb, 128)
            .unwrap();
    let mut g = vec![0.0; n.pow(4)];
    let ix = |i, j, k, l| ((i * n + j) * n + k) * n + l;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    for mu in 0..n {
                        for nu in 0..n {
                            for la in 0..n {
                                for si in 0..n {
                                    g[ix(i, j, k, l)] += orbitals[(mu, i)]
                                        * orbitals[(nu, j)]
                                        * orbitals[(la, k)]
                                        * orbitals[(si, l)]
                                        * eri.get(mu, nu, la, si);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let action = |vector: &[f64]| {
        let mut out = vec![0.0; bits];
        for (state, &weight) in vector.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            for spin in 0..2 {
                for p in 0..n {
                    for q in 0..n {
                        if let Some((target, sign)) =
                            operate(state, &[(spin * n + q, false), (spin * n + p, true)])
                        {
                            out[target] += weight * sign * h[(p, q)];
                        }
                    }
                }
            }
            for s in 0..2 {
                for t in 0..2 {
                    for p in 0..n {
                        for q in 0..n {
                            for r in 0..n {
                                for u in 0..n {
                                    if let Some((target, sign)) = operate(
                                        state,
                                        &[
                                            (s * n + q, false),
                                            (t * n + u, false),
                                            (t * n + r, true),
                                            (s * n + p, true),
                                        ],
                                    ) {
                                        out[target] += 0.5 * weight * sign * g[ix(p, q, r, u)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    };
    let ground = action(&reference_vector)[determinant];
    assert!((ground - reference.e_elec).abs() < 1e-10);
    let dimension = basis.len();
    let mut exact = Matrix::zeros(dimension, dimension);
    for (column, vector) in basis.iter().enumerate() {
        let image = action(vector);
        for row in 0..dimension {
            exact[(row, column)] = basis[row]
                .iter()
                .zip(&image)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                - if row == column { ground } else { 0.0 };
        }
    }
    let mut reconstructed = Matrix::zeros(dimension, dimension);
    for state in &report.states {
        for i in 0..dimension {
            for j in 0..dimension {
                reconstructed[(i, j)] += state.energy
                    * state.rotations[i]
                    * state.rotations[j]
                    * (report.berry_weights[i] * report.berry_weights[j]).sqrt();
            }
        }
    }
    let error = reconstructed.max_abs_diff(&exact);
    println!("multiplicity={multiplicity} flags={dimension} full_CIS_matrix_error={error:.12e} stationarity={:.12e} metric={:.12e}",report.reference_stationarity,report.metric_defect);
    assert!(error < 1e-10);
    assert!(report
        .states
        .iter()
        .all(|s| s.residual < 1e-10 && s.nearest_gap.is_finite()));
    let poles = dft_rs::linalg::symmetric_eigen(&exact).unwrap().0;
    reference.energy + poles[0]
}
fn check(charge: i32, multiplicity: usize) {
    let _ = check_system(&molecule(), charge, multiplicity);
}

#[test]
fn closed_shell_flag_tda_has_the_complete_singlet_hamiltonian_limit() {
    check(2, 1);
}
#[test]
fn doublet_flag_tda_has_the_complete_spin_pure_hamiltonian_limit() {
    check(1, 2);
}
#[test]
fn quartet_flag_tda_has_the_complete_maximal_spin_hamiltonian_limit() {
    check(1, 4);
}

fn derivative_controls(multiplicity: usize) -> Controls {
    Controls::from_value(
        &Value::object()
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("reference", "roks")
            .with("charge", 1i32)
            .with("multiplicity", multiplicity)
            .with("guess", "core")
            .with("max_iter", 500usize)
            .with("conv_tol", 1e-14)
            .with("conv_tol_grad", 1e-14)
            .with("conv_tol_dp", 1e-14)
            .with("grid_weight_cutoff", 0.0)
            .with("td_spin", "reference-spin")
            .with("td_tolerance", 1e-10)
            .with("response_tolerance", 1e-11),
    )
    .unwrap()
}
fn scalar(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}
fn numbers(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn check_nuclear_derivatives(multiplicity: usize) {
    let system = molecule();
    let controls = derivative_controls(multiplicity);
    let value = dft_rs::driver::run("td-hessian", &system, &controls).unwrap();
    assert_eq!(
        value.get("provenance").and_then(Value::as_str),
        Some("analytic")
    );
    let gradient = numbers(&value, "gradient");
    let hessian = numbers(&value, "hessian");
    assert_eq!(gradient.len(), 12);
    assert_eq!(hessian.len(), 144);
    let exact = check_system(&system, 1, multiplicity);
    assert!((exact - scalar(&value, "excited_state_energy")).abs() < 1e-10);
    let stages = [
        ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
        ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
        ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
        ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
    ];
    let mut failures = Vec::new();
    for column in 0..12 {
        let mut names = vec![format!("multiplicity{multiplicity} flag gradient {column}")];
        names.extend(
            (0..12).map(|row| format!("multiplicity{multiplicity} flag Hessian {row},{column}")),
        );
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 13];
                for (k, &h) in steps.iter().enumerate() {
                    let at = |displacement: f64| {
                        let mut point = system.clone();
                        point.positions[column / 3][column % 3] += displacement;
                        let report =
                            dft_rs::driver::run("td-state-gradient", &point, &controls).unwrap();
                        let energy = check_system(&point, 1, multiplicity);
                        assert!((energy - scalar(&report, "excited_state_energy")).abs() < 1e-10);
                        (energy, numbers(&report, "gradient"))
                    };
                    let (ep, gp) = at(h);
                    let (em, gm) = at(-h);
                    let (ep2, gp2, em2, gm2) = if stencil == stencils::Stencil::Fourth {
                        let (ep, gp) = at(2.0 * h);
                        let (em, gm) = at(-2.0 * h);
                        (ep, gp, em, gm)
                    } else {
                        (0.0, vec![0.0; 12], 0.0, vec![0.0; 12])
                    };
                    errors[0][k] = (stencil.first(ep, em, ep2, em2, h) - gradient[column]).abs();
                    for row in 0..12 {
                        errors[row + 1][k] = (stencil
                            .first(gp[row], gm[row], gp2[row], gm2[row], h)
                            - hessian[12 * row + column])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("common-orbital excited nuclear derivatives", &failures);
}

#[test]
fn doublet_flag_tda_nuclear_gradient_and_hessian_fit_every_entry() {
    check_nuclear_derivatives(2);
}
#[test]
fn quartet_flag_tda_nuclear_gradient_and_hessian_fit_every_entry() {
    check_nuclear_derivatives(4);
}

#[test]
fn unconnected_correlated_reference_models_cannot_return_parent_poles() {
    let system = molecule();
    for reference in ["roks", "gks"] {
        for id in ["excitations", "td-state-gradient", "td-hessian"] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("reference", reference)
                    .with("xc", "b2plyp")
                    .with("basis", "sto-3g")
                    .with("charge", 1i32)
                    .with("multiplicity", 2usize),
            )
            .unwrap();
            match dft_rs::driver::run(id,&system,&controls) {
                Err(dft_rs::DftError::Unsupported{feature,missing}) => {
                    assert!(feature.contains("correlated"),"{feature}");
                    assert!(missing.contains("propagator"),"{missing}");
                }
                result => panic!("{id}/{reference} must refuse the unconnected correlated pole model: {result:?}"),
            }
        }
    }
}
