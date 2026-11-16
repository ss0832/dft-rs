// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::driver::Value;
use dft_rs::linalg::Matrix;
use dft_rs::properties::pt2::Pt2Problem;
use dft_rs::xc::double_hybrid::Pt2Coefficients;
fn numbers(v: &Value) -> Vec<f64> {
    v.numeric_array().unwrap().1
}
#[test]
fn physical_zero_order_matches_all_independent_a_metric_coupling_targets() {
    let data = Value::parse_json(include_str!("oracle_data/pt2_propagator.json")).unwrap();
    let n = data.get("orbitals").unwrap().as_f64().unwrap() as usize;
    let o = data.get("occupied").unwrap().as_f64().unwrap() as usize;
    let g = numbers(data.get("integrals").unwrap());
    let ee = data.get("energies").unwrap().as_array().unwrap();
    let energies = [numbers(&ee[0]), numbers(&ee[1])];
    let mut problem = Pt2Problem {
        energies: energies.clone(),
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
    let coefficients = Pt2Coefficients {
        same_spin: 1.0,
        opposite_spin: 1.0,
    };
    let solution = problem.solve(coefficients).unwrap();
    let fock = std::array::from_fn(|s| {
        let mut v = vec![Matrix::zeros(n, n); 3];
        for i in 0..n {
            v[0][(i, i)] = energies[s][i];
        }
        v
    });
    let integrals = vec![
        [g.clone(), g.clone(), g],
        [
            vec![0.0; n.pow(4)],
            vec![0.0; n.pow(4)],
            vec![0.0; n.pow(4)],
        ],
        [
            vec![0.0; n.pow(4)],
            vec![0.0; n.pow(4)],
            vec![0.0; n.pow(4)],
        ],
    ];
    let zero = std::array::from_fn(|c| vec![0.0; solution.amplitudes[c].len()]);
    let amplitudes = vec![solution.amplitudes, zero.clone(), zero];
    let out = dft_rs::properties::collinear_transition_path(
        [o; 2],
        &fock,
        &integrals,
        &amplitudes,
        coefficients,
        128,
        1e-11,
    )
    .unwrap();
    assert!(out.coupling_duality_defect < 1e-13);
    let mut count = 0;
    let mut maximum = 0.0_f64;
    for request in data.get("requests").unwrap().as_array().unwrap() {
        let kind = request.get("kind").unwrap().as_str().unwrap();
        let row = request.get("row").unwrap().as_f64().unwrap() as usize;
        let col = request.get("column").unwrap().as_f64().unwrap() as usize;
        let value = match kind {
            "A" => out.singles[0].re[(row, col)],
            "S" => out.metric[0].re[(row, col)],
            "C" => out.coupling[0].re[(row, col)],
            _ => continue,
        };
        let factor = if request.get("order").unwrap().as_f64().unwrap() == 2.0 {
            2.0
        } else {
            1.0
        };
        let target = request.get("target").unwrap().as_f64().unwrap();
        let error = (factor * value - target).abs();
        maximum = maximum.max(error);
        assert!(
            error < 2e-14,
            "{kind} ({row},{col}) actual={} target={target}",
            factor * value
        );
        count += 1;
    }
    assert!(count > 100);
    assert!(
        out.doubles[0]
            .re
            .max_abs_diff(&out.doubles[0].re.transpose())
            < 1e-13
    );
    println!("independent targets={count} maximum_defect={maximum:e}");
}
