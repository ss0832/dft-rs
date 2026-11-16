// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete independent commutator oracle and interaction-strength fits of
//! every nonstructural singles A/B/metric, singles-doubles coupling, and
//! all electric/time-odd property-gradient components.
use super::*;
use crate::acceptance::components;
use crate::driver::Value;

fn values(v: &Value) -> Vec<f64> {
    v.numeric_array().unwrap().1
}
fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn integer(v: &Value) -> usize {
    v.as_f64().unwrap() as usize
}
fn flatten(x: &[Matrix; 2]) -> Vec<f64> {
    x.iter()
        .flat_map(|m| m.as_slice().iter().copied())
        .collect()
}

fn fixture() -> (Value, Pt2Problem<f64>, [Vec<f64>; 3]) {
    let data =
        Value::parse_json(include_str!("../../tests/oracle_data/pt2_propagator.json")).unwrap();
    let n = integer(data.get("orbitals").unwrap());
    let o = integer(data.get("occupied").unwrap());
    let g = values(data.get("integrals").unwrap());
    let e = array(data.get("energies").unwrap());
    let mut problem = Pt2Problem {
        energies: [values(&e[0]), values(&e[1])],
        occupied: [o; 2],
        ovov: std::array::from_fn(|_| Vec::new()),
    };
    for block in &mut problem.ovov {
        for i in 0..o {
            for a in o..n {
                for j in 0..o {
                    for b in o..n {
                        block.push(g[((i * n + a) * n + j) * n + b]);
                    }
                }
            }
        }
    }
    (data, problem, [g.clone(), g.clone(), g])
}
fn unique_pairs(model: &Pt2Transition, blocks: &[Vec<f64>; 3]) -> Vec<f64> {
    let n = model.dimensions();
    let o = model.occupied;
    let mut out = Vec::new();
    for (c, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        for i in 0..o[s] {
            for a in o[s]..n[s] {
                for j in 0..o[t] {
                    for b in o[t]..n[t] {
                        if s == t && (i >= j || a >= b) {
                            continue;
                        }
                        let v = [n[0] - o[0], n[1] - o[1]];
                        out.push(blocks[c][((i * v[s] + a - o[s]) * o[t] + j) * v[t] + b - o[t]]);
                    }
                }
            }
        }
    }
    out
}

#[test]
fn all_transition_blocks_and_property_gradients_have_independent_four_step_fits() {
    let (data, problem, g) = fixture();
    let coefficients = Pt2Coefficients {
        same_spin: 1.0,
        opposite_spin: 1.0,
    };
    let solution = problem.solve(coefficients).unwrap();
    let model = Pt2Transition::from_mo(&problem, g, solution.amplitudes, coefficients).unwrap();
    let n = model.dimensions();
    let o = model.occupied;
    let dim = (n[0] - o[0]) * o[0] + (n[1] - o[1]) * o[1];
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut s = Vec::new();
    let mut c = Vec::new();
    for column in 0..dim {
        let mut x = std::array::from_fn(|s| Matrix::zeros(n[s] - o[s], o[s]));
        let split = (n[0] - o[0]) * o[0];
        if column < split {
            x[0].as_mut_slice()[column] = 1.0;
        } else {
            x[1].as_mut_slice()[column - split] = 1.0;
        }
        let (ac, bc) = model.kernels(&x);
        a.push(flatten(&ac));
        b.push(flatten(&bc));
        s.push(flatten(&model.metric(&x)));
        c.push(unique_pairs(&model, &model.coupling_transpose(&x)));
    }
    let mut p = Vec::new();
    let mut d = Vec::new();
    for operator in array(data.get("operators").unwrap()) {
        let ops = array(operator);
        let ops = [
            Matrix::from_row_major(n[0], n[0], values(&ops[0])),
            Matrix::from_row_major(n[1], n[1], values(&ops[1])),
        ];
        let (pc, dc) = model.property(&ops);
        p.push(flatten(&pc));
        d.push(unique_pairs(&model, &dc));
    }
    let requests = array(data.get("requests").unwrap());
    let mut actual = Vec::new();
    let mut names = Vec::new();
    for r in requests {
        let row = integer(r.get("row").unwrap());
        let col = integer(r.get("column").unwrap());
        let kind = r.get("kind").unwrap().as_str().unwrap();
        let value = match kind {
            "A" => a[col][row],
            "B" => b[col][row],
            "S" => s[col][row],
            "C" => c[row][col],
            "P" => p[col][row],
            "D" => d[col][row],
            _ => panic!("invalid oracle request"),
        };
        let derivative = value
            * if integer(r.get("order").unwrap()) == 2 {
                2.0
            } else {
                1.0
            };
        let expected = r.get("target").unwrap().as_f64().unwrap();
        assert!(
            (derivative - expected).abs() < 2e-14,
            "{kind}({row},{col}) production={derivative:e} oracle={expected:e}"
        );
        actual.push(derivative);
        names.push(format!("PT2 propagator {kind} ({row},{col})"));
    }
    let intervals: Vec<[f64; 4]> = array(data.get("intervals").unwrap())
        .iter()
        .map(|v| values(v).try_into().unwrap())
        .collect();
    let estimates = array(data.get("estimates").unwrap());
    let mut stage = 0;
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &intervals,
        |_| {
            let finite = array(&estimates[stage]);
            stage += 1;
            let finite: Vec<_> = finite.iter().map(values).collect();
            (0..requests.len())
                .map(|i| std::array::from_fn(|k| (finite[k][i] - actual[i]).abs()))
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("complete PT2 transition propagator", &failures);
}

#[test]
fn transition_couplings_are_adjoint_and_signed_coefficients_are_linear() {
    let (_, problem, g) = fixture();
    let all = Pt2Coefficients {
        same_spin: 0.31,
        opposite_spin: -0.47,
    };
    let t = problem.solve(all).unwrap().amplitudes;
    let model = Pt2Transition::from_mo(&problem, g.clone(), t.clone(), all).unwrap();
    let n = model.dimensions();
    let o = model.occupied;
    let x = std::array::from_fn(|s| {
        let mut x = Matrix::zeros(n[s] - o[s], o[s]);
        for (k, v) in x.as_mut_slice().iter_mut().enumerate() {
            *v = ((k + 3 * s + 1) as f64).sin();
        }
        x
    });
    let coupled = model.coupling(&t);
    let forward = flatten(&x)
        .iter()
        .zip(flatten(&coupled))
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let reverse = model.pair_dot(&model.coupling_transpose(&x), &t);
    assert!(
        (forward - reverse).abs() < 2e-14,
        "SD adjoint defect {}",
        (forward - reverse).abs()
    );
    let first = Pt2Transition::from_mo(
        &problem,
        g.clone(),
        t.clone(),
        Pt2Coefficients {
            same_spin: all.same_spin,
            opposite_spin: 0.0,
        },
    )
    .unwrap();
    let second = Pt2Transition::from_mo(
        &problem,
        g,
        t,
        Pt2Coefficients {
            same_spin: 0.0,
            opposite_spin: all.opposite_spin,
        },
    )
    .unwrap();
    let (a, b) = model.kernels(&x);
    let (aa, ab) = first.kernels(&x);
    let (ba, bb) = second.kernels(&x);
    for (sum, (a, b)) in flatten(&a).into_iter().chain(flatten(&b)).zip(
        flatten(&aa)
            .into_iter()
            .chain(flatten(&ab))
            .zip(flatten(&ba).into_iter().chain(flatten(&bb))),
    ) {
        assert!((sum - a - b).abs() < 2e-14);
    }
}

#[test]
fn one_electron_limit_has_no_pauli_forbidden_double_poles() {
    let (_, original, g) = fixture();
    let n = original.energies[0].len();
    let occupied = [1, 0];
    let ovov = std::array::from_fn(|c| {
        let (s, t) = [(0, 0), (0, 1), (1, 1)][c];
        let mut values = Vec::new();
        for i in 0..occupied[s] {
            for a in occupied[s]..n {
                for j in 0..occupied[t] {
                    for b in occupied[t]..n {
                        values.push(g[c][((i * n + a) * n + j) * n + b]);
                    }
                }
            }
        }
        values
    });
    let problem = Pt2Problem {
        energies: original.energies,
        occupied,
        ovov,
    };
    let coefficients = Pt2Coefficients {
        same_spin: 1.0,
        opposite_spin: 1.0,
    };
    let solution = problem.solve(coefficients).unwrap();
    assert!(solution
        .amplitudes
        .iter()
        .flatten()
        .all(|v| v.abs() < 1e-15));
    let model = Pt2Transition::from_mo(&problem, g, solution.amplitudes, coefficients).unwrap();
    let x = [
        Matrix::from_row_major(n - 1, 1, vec![0.21, -0.13, 0.32]),
        Matrix::zeros(n, 0),
    ];
    // A formal repeated-hole denominator cannot be a physical double
    // excitation of a one-electron state.
    let frequency = problem.energies[0][1] + problem.energies[0][2] - 2.0 * problem.energies[0][0];
    let (doubles, norm, minimum) = model.doubles(frequency, &x).unwrap();
    assert!(doubles.iter().flatten().all(|v| *v == 0.0));
    assert_eq!(norm, 0.0);
    assert!(minimum.is_infinite());
    let (a, b) = model.kernels(&x);
    assert!(flatten(&a)
        .iter()
        .chain(flatten(&b).iter())
        .all(|v| v.abs() < 2e-14));
}
