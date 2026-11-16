// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All pole energies and all nine transition-vector components against
//! independent dense generalized eigenproblems at four signed interaction
//! strengths. Kernel contractions have their separate determinant oracle.
use super::*;
use crate::acceptance::components;
use crate::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use crate::driver::{Controls, Value};
use crate::integrals::{eri_incore, CoulombKernel};
use crate::properties::pt2::{from_ao, OrbitalSpace};

#[test]
fn common_transition_phase_is_observable_invariant_and_preserves_parent_continuation() {
    let make = |sign: f64| TdaState {
        energy: 0.7,
        x: std::array::from_fn(|_| Matrix::from_row_major(2, 1, vec![0.3 * sign, -0.4 * sign])),
        spin: TdaSpin::SpinResolved,
        tamm_dancoff: true,
        transition_dipole: [-0.1 * sign, 0.7 * sign, -0.2 * sign],
        transition_momentum: [0.4 * sign, -0.3 * sign, 0.2 * sign],
        transition_angular_momentum: [-0.3 * sign, 0.2 * sign, 0.6 * sign],
        oscillator_strength: 0.252,
        rotatory_strength: -0.025,
        rotatory_strength_velocity: 0.03,
    };
    let mut a = make(1.0);
    let mut b = make(-1.0);
    a.canonicalize_phase(None);
    b.canonicalize_phase(None);
    assert_eq!(state_components(&a), state_components(&b));
    for (a, b) in a.x.iter().zip(&b.x) {
        assert_eq!(a.as_slice(), b.as_slice());
    }
    assert_eq!(a.oscillator_strength, 0.252);
    assert_eq!(a.rotatory_strength, -0.025);
    assert_eq!(a.rotatory_strength_velocity, 0.03);
    // A strong coefficient correction may reverse the largest corrected
    // moment. Its phase remains anchored to the parent continuation.
    let mut continued = make(1.0);
    continued.transition_dipole[1] = -3.0;
    continued.canonicalize_phase(Some([[0.1, 0.7, -0.2], [0.0; 3], [0.0; 3]]));
    assert_eq!(continued.transition_dipole[1], -3.0);
    let mut dark = make(-1.0);
    dark.transition_dipole = [0.0; 3];
    dark.canonicalize_phase(None);
    assert!(dark.transition_momentum[0] > 0.0);
}

#[test]
fn complete_degenerate_block_metric_transport_has_independent_four_step_fits() {
    let metric = Matrix::from_row_major(
        3,
        3,
        vec![0.7, 0.13, -0.09, 0.13, 0.8, 0.17, -0.09, 0.17, 0.6],
    );
    let vectors: Vec<Vec<f64>> = (0..3)
        .map(|q| (0..3).map(|p| if p == q { 1.0 } else { 0.0 }).collect())
        .collect();
    let production: Vec<_> = (0..3)
        .flat_map(|q| block_metric_transport(&vectors, &metric, q))
        .collect();
    let exact = |t: f64| {
        let mut s = Matrix::identity(3);
        for i in 0..3 {
            for j in 0..3 {
                s[(i, j)] += t * t * metric[(i, j)];
            }
        }
        let (e, v) = symmetric_eigen(&s).unwrap();
        let mut x = Matrix::zeros(3, 3);
        for k in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    x[(i, j)] += v[(i, k)] * v[(j, k)] / e[k].sqrt();
                }
            }
        }
        (0..3)
            .flat_map(|q| (0..3).map(move |p| (p, q)))
            .map(|(p, q)| x[(p, q)])
            .collect::<Vec<_>>()
    };
    let names: Vec<_> = (0..9)
        .map(|k| format!("degenerate transition metric ({},{})", k % 3, k / 3))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.16, 0.12, 0.08, 0.06]],
        |steps| {
            let mut errors = vec![[0.0; 4]; 9];
            for (k, &h) in steps.iter().enumerate() {
                let plus = exact(h);
                let minus = exact(-h);
                for i in 0..9 {
                    let reference = if i / 3 == i % 3 { 1.0 } else { 0.0 };
                    errors[i][k] = ((plus[i] + minus[i] - 2.0 * reference) / (h * h)
                        - 2.0 * production[i])
                        .abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("complete degenerate metric transport", &failures);
    for p in 0..3 {
        for q in 0..3 {
            assert!((production[3 * q + p] + production[3 * p + q] + metric[(p, q)]).abs() < 1e-15);
        }
    }
}

fn state_components(state: &TdaState) -> Vec<f64> {
    std::iter::once(state.energy)
        .chain(state.transition_dipole)
        .chain(state.transition_angular_momentum)
        .chain(state.transition_momentum)
        .collect()
}
fn generalized(a: &Matrix, s: &Matrix) -> (Vec<f64>, Matrix) {
    let (values, vectors) = symmetric_eigen(s).unwrap();
    assert!(values.iter().all(|v| *v > 0.0));
    let mut x = vectors;
    for k in 0..values.len() {
        for i in 0..values.len() {
            x[(i, k)] /= values[k].sqrt();
        }
    }
    let h = x.transpose_matmul(&a.matmul(&x));
    let (e, v) = symmetric_eigen(&h).unwrap();
    (e, x.matmul(&v))
}

#[test]
fn every_correlated_tda_pole_and_transition_component_has_a_four_step_dense_fit() {
    for xc in ["hf", "pbe0"] {
        for u in [false, true] {
            let mut positions = vec![
                [0.13, -0.21, 0.19],
                [1.57, 0.16, -0.08],
                [-0.19, 1.93, 0.37],
                [0.42, 0.29, 2.28],
            ];
            if u {
                positions.push([1.41, 1.83, 1.59]);
            }
            let numbers = vec![1; positions.len()];
            let input = Value::object()
                .with("xc", xc)
                .with("basis", "sto-3g")
                .with("reference", if u { "uks" } else { "rks" })
                .with("multiplicity", if u { 2usize } else { 1usize })
                .with("guess", "core")
                .with("grid_level", 0usize)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol", 1e-13)
                .with("conv_tol_dp", 1e-13)
                .with("conv_tol_grad", 1e-13);
            let method =
                MolecularMethod::from_controls(&Controls::from_value(&input).unwrap()).unwrap();
            let setup = prepare_molecular(&numbers, &positions, &method).unwrap();
            let reference = run_molecular_reference(&setup).unwrap();
            let scf = &reference.scf;
            let spin = if u {
                TdaSpin::SpinResolved
            } else {
                TdaSpin::Singlet
            };
            let parent = TdaSolver::new(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                spin,
            )
            .unwrap();
            let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 128).unwrap();
            let n = eri.nao;
            let orbitals = std::array::from_fn(|s| OrbitalSpace {
                energies: scf.mo_energy[s].clone(),
                coefficients: scf.mo_coeff[s].as_slice().to_vec(),
                occupied: if s == 0 { scf.n_alpha } else { scf.n_beta },
            });
            let problem = from_ao(n, eri.as_slice(), &orbitals).unwrap();
            let coefficients = Pt2Coefficients {
                same_spin: 0.31,
                opposite_spin: 0.47,
            };
            let solution = problem.solve(coefficients).unwrap();
            let correlated = correlated_tda_states(
                &parent,
                &setup.basis,
                scf,
                &eri,
                &problem,
                &solution,
                coefficients,
                2,
                128,
            )
            .unwrap();
            let model = Pt2Transition::from_reference(
                scf,
                &eri,
                &problem,
                &solution.amplitudes,
                coefficients,
                128,
            )
            .unwrap();
            let dim = parent.dim();
            let parents = parent.solve_dense(dim).unwrap();
            let vectors: Vec<_> = parents.iter().map(flatten).collect();
            let occupied = problem.occupied;
            let ns = [problem.energies[0].len(), problem.energies[1].len()];
            let mut pairs = Vec::new();
            for (c, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                let v = [ns[0] - occupied[0], ns[1] - occupied[1]];
                for i in 0..occupied[s] {
                    for a in occupied[s]..ns[s] {
                        for j in 0..occupied[t] {
                            for b in occupied[t]..ns[t] {
                                if s == t && (i >= j || a >= b) {
                                    continue;
                                }
                                let index = ((i * v[s] + a - occupied[s]) * occupied[t] + j) * v[t]
                                    + b
                                    - occupied[t];
                                let gap = problem.energies[s][a] - problem.energies[s][i]
                                    + problem.energies[t][b]
                                    - problem.energies[t][j];
                                let w = if s == t {
                                    coefficients.same_spin
                                } else {
                                    coefficients.opposite_spin
                                };
                                pairs.push((c, index, gap, w.sqrt()));
                            }
                        }
                    }
                }
            }
            let full = dim + pairs.len();
            let mut a0 = Matrix::zeros(full, full);
            let mut a1 = a0.clone();
            let mut a2 = a0.clone();
            let mut s2 = a0.clone();
            for column in 0..dim {
                let mut unit = vec![0.0; dim];
                unit[column] = 1.0;
                let a = parent.a_vec(&unit).unwrap();
                let x = expand(&unit, scf, spin);
                let (second, _) = model.kernels(&x);
                let second = project(&second, spin);
                let metric = project(&model.metric(&x), spin);
                let sd = model.coupling_transpose(&x);
                for row in 0..dim {
                    a0[(row, column)] = a[row];
                    a2[(row, column)] = second[row];
                    s2[(row, column)] = metric[row];
                }
                for (d, &(c, k, _, w)) in pairs.iter().enumerate() {
                    a1[(dim + d, column)] = w * sd[c][k];
                    a1[(column, dim + d)] = w * sd[c][k];
                }
            }
            for (d, &(_, _, gap, _)) in pairs.iter().enumerate() {
                a0[(dim + d, dim + d)] = gap;
            }
            let mut properties = Vec::new();
            for ao in dipole_matrices(&setup.basis, Vec3::zero())
                .into_iter()
                .chain(angular_momentum_operator(&setup.basis, Vec3::zero()))
                .chain(gradient_operator(&setup.basis))
            {
                let op = std::array::from_fn(|s| {
                    scf.mo_coeff[s].transpose_matmul(&ao.matmul(&scf.mo_coeff[s]))
                });
                let (p2, pd) = model.property(&op);
                let p0 = std::array::from_fn(|s| {
                    let mut p = Matrix::zeros(ns[s] - occupied[s], occupied[s]);
                    for a in occupied[s]..ns[s] {
                        for i in 0..occupied[s] {
                            p[(a - occupied[s], i)] = op[s][(a, i)];
                        }
                    }
                    p
                });
                let mut zero = vec![0.0; full];
                zero[..dim].copy_from_slice(&project(&p0, spin));
                let mut first = vec![0.0; full];
                for (d, &(c, k, _, w)) in pairs.iter().enumerate() {
                    first[dim + d] = w * pd[c][k];
                }
                let mut second = vec![0.0; full];
                second[..dim].copy_from_slice(&project(&p2, spin));
                properties.push((zero, first, second));
            }
            let exact = |lambda: f64| -> Vec<f64> {
                let mut a = a0.clone();
                let mut s = Matrix::identity(full);
                for i in 0..full {
                    for j in 0..full {
                        a[(i, j)] += lambda * a1[(i, j)] + lambda * lambda * a2[(i, j)];
                        s[(i, j)] += lambda * lambda * s2[(i, j)];
                    }
                }
                let (energies, states) = generalized(&a, &s);
                let mut result = Vec::new();
                for parent in vectors.iter().take(2) {
                    let overlaps: Vec<f64> = (0..full)
                        .map(|r| (0..dim).map(|i| parent[i] * states[(i, r)]).sum())
                        .collect();
                    let root = (0..full)
                        .max_by(|&a, &b| overlaps[a].abs().total_cmp(&overlaps[b].abs()))
                        .unwrap();
                    assert!(
                        overlaps[root].abs() > 0.95,
                        "root tracking failed: {xc} U={u} lambda={lambda}"
                    );
                    let sign = overlaps[root].signum();
                    result.push(energies[root]);
                    for (p0, p1, p2) in &properties {
                        result.push(
                            (0..full)
                                .map(|i| {
                                    (p0[i] + lambda * p1[i] + lambda * lambda * p2[i])
                                        * states[(i, root)]
                                        * sign
                                })
                                .sum(),
                        );
                    }
                }
                result
            };
            let base = exact(0.0);
            let production: Vec<_> = correlated
                .states
                .iter()
                .take(2)
                .flat_map(state_components)
                .collect();
            let analytic: Vec<_> = production
                .iter()
                .zip(&base)
                .map(|(p, b)| 2.0 * (p - b))
                .collect();
            let names: Vec<_> = (0..analytic.len())
                .map(|k| {
                    format!(
                        "{xc} U={u} correlated TDA root {} component {}",
                        k / 10,
                        k % 10
                    )
                })
                .collect();
            let mut failures = Vec::new();
            components::fit_components(
                &names,
                &[
                    [0.2, 0.15, 0.1, 0.075],
                    [0.05, 0.0375, 0.025, 0.01875],
                    [0.4, 0.3, 0.2, 0.15],
                ],
                |steps| {
                    let mut errors = vec![[0.0; 4]; analytic.len()];
                    for (k, &h) in steps.iter().enumerate() {
                        let plus = exact(h);
                        let minus = exact(-h);
                        for i in 0..analytic.len() {
                            errors[i][k] = ((plus[i] + minus[i] - 2.0 * base[i]) / (h * h)
                                - analytic[i])
                                .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
            components::assert_no_failures(
                "all correlated TDA poles and transition vectors",
                &failures,
            );
        }
    }
}
