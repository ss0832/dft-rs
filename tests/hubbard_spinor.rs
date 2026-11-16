// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

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
use dft_rs::periodic::bloch::BlochMatrix;
use dft_rs::response::complex::Complex;
use dft_rs::scf::hubbard::{HubbardModel, Interaction, Site};
use stencils::Stencil;

const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.78, 0.34, -0.08],
    [-0.49, 1.65, 0.41],
    [0.31, -0.78, 1.91],
];
const U_EFF: f64 = 0.143;
const U: f64 = 0.2625;
const J: f64 = 0.052;
const PAIRS: [[usize; 2]; 2] = [[0, 1], [1, 2]];
const COULOMB: [[f64; 3]; 3] = [
    [0.34, 0.044, 0.21],
    [0.044, 0.052, 0.037],
    [0.21, 0.037, 0.29],
];

fn integral(a: usize, b: usize, c: usize, d: usize) -> f64 {
    COULOMB[PAIRS[a][b]][PAIRS[c][d]]
}

fn interactions() -> [(&'static str, Interaction); 2] {
    [
        ("Dudarev", Interaction::Dudarev { u_eff: U_EFF }),
        (
            "Liechtenstein",
            Interaction::Liechtenstein {
                u: U,
                j: J,
                integrals: (0..16)
                    .map(|index| integral(index / 8, index / 4 % 2, index / 2 % 2, index % 2))
                    .collect(),
            },
        ),
    ]
}

fn entry(p: &BlochMatrix, a: usize, b: usize) -> Complex {
    Complex::new(p.re[(a, b)], p.im[(a, b)])
}

fn set(p: &mut BlochMatrix, a: usize, b: usize, value: Complex) {
    p.re[(a, b)] = value.re;
    p.im[(a, b)] = value.im;
}

fn basis_directions(width: usize) -> Vec<(String, BlochMatrix, usize, usize, bool)> {
    let mut output = Vec::new();
    for i in 0..width {
        for j in 0..=i {
            let mut real = BlochMatrix::zeros(width, width);
            real.re[(i, j)] = 1.0;
            real.re[(j, i)] = 1.0;
            output.push((format!("R[{i},{j}]"), real, i, j, false));
            if i != j {
                let mut imaginary = BlochMatrix::zeros(width, width);
                imaginary.im[(i, j)] = 1.0;
                imaginary.im[(j, i)] = -1.0;
                output.push((format!("I[{i},{j}]"), imaginary, i, j, true));
            }
        }
    }
    output
}

fn small_projector(s: &Matrix, a: usize, b: usize) -> Vec<[f64; 2]> {
    // Closed two-by-two square root independent of the production eigensolver.
    let x = s[(a, a)];
    let y = s[(a, b)];
    let z = s[(b, b)];
    let determinant = (x * z - y * y).sqrt();
    let scale = (determinant * (x + z + 2.0 * determinant).sqrt()).recip();
    (0..s.rows)
        .map(|i| {
            let u = s[(i, a)];
            let v = s[(i, b)];
            [
                (u * (z + determinant) - v * y) * scale,
                (v * (x + determinant) - u * y) * scale,
            ]
        })
        .collect()
}

fn orbital_projection(p: &BlochMatrix, w: &[[f64; 2]]) -> [[Complex; 4]; 4] {
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let mut result = Complex::new(0.0, 0.0);
            for (i, left) in w.iter().enumerate() {
                for (j, right) in w.iter().enumerate() {
                    result += entry(p, a / 2 * w.len() + i, b / 2 * w.len() + j)
                        * (left[a % 2] * right[b % 2]);
                }
            }
            result
        })
    })
}

fn oracle(n: &[[Complex; 4]; 4], dudarev: bool) -> [f64; 2] {
    let total: f64 = (0..4).map(|a| n[a][a].re).sum();
    if dudarev {
        let norm: f64 = n.iter().flatten().map(|x| (x.conj() * *x).re).sum();
        return [0.5 * U_EFF * (total - norm), 0.0];
    }
    // Independent antisymmetrized spin-orbital two-particle energy.
    let mut interaction = 0.0;
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let direct = if a / 2 == c / 2 && b / 2 == d / 2 {
                        integral(a % 2, c % 2, b % 2, d % 2)
                    } else {
                        0.0
                    };
                    let exchange = if a / 2 == d / 2 && b / 2 == c / 2 {
                        integral(a % 2, d % 2, b % 2, c % 2)
                    } else {
                        0.0
                    };
                    interaction += 0.5 * (direct - exchange) * (n[c][a] * n[d][b]).re;
                }
            }
        }
    }
    let nx = n[0][2] + n[1][3];
    let nz = 0.5 * (n[0][0].re + n[1][1].re - n[2][2].re - n[3][3].re);
    let magnetization_squared = 4.0 * (nz * nz + (nx.conj() * nx).re);
    let dc = 0.5 * U * total * (total - 1.0)
        - 0.25 * J * (total * (total - 2.0) + magnetization_squared);
    [interaction, dc]
}

fn shifted(n: &[[Complex; 4]; 4], d: &[[Complex; 4]; 4], scale: f64) -> [[Complex; 4]; 4] {
    std::array::from_fn(|a| std::array::from_fn(|b| n[a][b] + d[a][b] * scale))
}

// Uses a complex SU(2) rotation with nonzero x, y and z generator components.
fn rotate(p: &BlochMatrix) -> BlochMatrix {
    let w = 0.63_f64;
    let z = 0.21_f64;
    let y = -0.37_f64;
    let x = (1.0 - w * w - z * z - y * y).sqrt();
    let u = [
        [Complex::new(w, z), Complex::new(y, x)],
        [Complex::new(-y, x), Complex::new(w, -z)],
    ];
    let n = p.rows() / 2;
    let mut result = BlochMatrix::zeros(2 * n, 2 * n);
    for a in 0..2 * n {
        for b in 0..2 * n {
            let mut value = Complex::new(0.0, 0.0);
            for s in 0..2 {
                for t in 0..2 {
                    value +=
                        u[a / n][s] * entry(p, s * n + a % n, t * n + b % n) * u[b / n][t].conj();
                }
            }
            set(&mut result, a, b, value);
        }
    }
    result
}

#[test]
fn all_spinor_density_components_with_independent_energy_and_fll() {
    let setup = prepare_molecular(
        &[1; 4],
        &POSITIONS,
        &MolecularMethod {
            grid_weight_cutoff: 0.0,
            ..MolecularMethod::default()
        },
    )
    .unwrap();
    let overlap = overlap_matrix(&setup.basis);
    let directions = basis_directions(2 * overlap.rows);
    let mut central = BlochMatrix::zeros(2 * overlap.rows, 2 * overlap.rows);
    for (index, (_, direction, a, b, _)) in directions.iter().enumerate() {
        let value = 0.0021 * (index + 1) as f64 + if a == b { 0.31 } else { 0.0 };
        central.add_scaled(direction, value);
    }
    let projectors = small_projector(&overlap, 0, 2);
    let central_occ = orbital_projection(&central, &projectors);
    let directions_occ: Vec<_> = directions
        .iter()
        .map(|(_, d, ..)| orbital_projection(d, &projectors))
        .collect();
    for (label, interaction) in interactions() {
        let dudarev = label == "Dudarev";
        let energy = |occupation: &[[Complex; 4]; 4]| {
            let [interaction, dc] = oracle(occupation, dudarev);
            interaction - dc
        };
        let fock = |occupation: &[[Complex; 4]; 4], direction: &[[Complex; 4]; 4]| {
            0.5 * (energy(&shifted(occupation, direction, 1.0))
                - energy(&shifted(occupation, direction, -1.0)))
        };
        let model = HubbardModel::new(
            &overlap,
            vec![Site {
                orbitals: vec![0, 2],
                interaction,
            }],
            1e-9,
        )
        .unwrap();
        let fields = model.evaluate_spinor(&central).unwrap();
        let rotated = model.evaluate_spinor(&rotate(&central)).unwrap();
        assert!(rotated.potential.rms_diff(&rotate(&fields.potential)) < 2e-15);
        assert!((rotated.energy - fields.energy).abs() < 2e-15);
        let [interaction, dc] = oracle(&central_occ, dudarev);
        assert!((interaction - fields.sites[0].interaction).abs() < 2e-15);
        assert!((dc - fields.sites[0].double_counting).abs() < 2e-15);
        let mut failed = Vec::new();
        for ((name, direction, ..), projection) in directions.iter().zip(&directions_occ) {
            let first = fields.potential.trace_product(direction);
            assert!((first - fock(&central_occ, projection)).abs() < 2e-15);
            let response = model.response_spinor(direction).unwrap();
            // Add a nonlinear chart term to measure truncation independently
            // even for exactly quadratic energies and zero kernel blocks.
            let probe = |t: f64| {
                let perturbed = shifted(&central_occ, projection, t);
                shifted(&perturbed, &central_occ, t.powi(3) + 0.2 * t.powi(5))
            };
            let mut labels = vec![format!("{label} spinor Fock {name}")];
            labels.extend(
                directions
                    .iter()
                    .map(|(output, ..)| format!("{label} spinor kernel {output}/{name}")),
            );
            let mut targets = vec![first];
            for ((_, output, ..), output_projection) in directions.iter().zip(&directions_occ) {
                let derivative = response.trace_product(output);
                let oracle_response = fock(projection, output_projection)
                    - fock(&[[Complex::new(0.0, 0.0); 4]; 4], output_projection);
                assert!((derivative - oracle_response).abs() < 2e-15);
                targets.push(derivative);
            }
            stencils::fit_components_staged(
                &labels,
                &[
                    ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], Stencil::Fourth),
                ],
                |steps, stencil| {
                    let differentiate = |target: f64, f: &dyn Fn(&[[Complex; 4]; 4]) -> f64| {
                        steps.map(|h| {
                            (stencil.first(
                                f(&probe(h)),
                                f(&probe(-h)),
                                f(&probe(2.0 * h)),
                                f(&probe(-2.0 * h)),
                                h,
                            ) - target)
                                .abs()
                        })
                    };
                    let mut errors = vec![differentiate(targets[0], &energy)];
                    errors.extend(
                        directions_occ
                            .iter()
                            .zip(&targets[1..])
                            .map(|(out, &target)| differentiate(target, &|p| fock(p, out))),
                    );
                    errors
                },
                &mut failed,
            );
        }
        components::assert_no_failures(label, &failed);

        let n = overlap.rows;
        let collinear = std::array::from_fn(|s| {
            Matrix::from_row_major(
                n,
                n,
                (0..n * n)
                    .map(|k| central.re[(s * n + k / n, s * n + k % n)])
                    .collect(),
            )
        });
        let mut diagonal = BlochMatrix::zeros(2 * n, 2 * n);
        for s in 0..2 {
            for a in 0..n {
                for b in 0..n {
                    diagonal.re[(s * n + a, s * n + b)] = collinear[s][(a, b)];
                }
            }
        }
        let spinor = model.evaluate_spinor(&diagonal).unwrap();
        let scalar = model.evaluate(&collinear).unwrap();
        assert!((spinor.energy - scalar.energy).abs() < 2e-15);
        for s in 0..2 {
            for a in 0..n {
                for b in 0..n {
                    assert!(
                        (spinor.potential.re[(s * n + a, s * n + b)] - scalar.potential[s][(a, b)])
                            .abs()
                            < 2e-15
                    );
                    assert_eq!(spinor.potential.re[(s * n + a, (1 - s) * n + b)], 0.0);
                }
            }
        }
    }
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn spinor_hubbard_stationarity_on_ammonia() {
    let system = System {
        numbers: vec![7, 1, 1, 1],
        positions: POSITIONS.to_vec(),
        cell: None,
        pbc: [false; 3],
    };
    for (xc, charge, multiplicity) in [("wb97x-d", 0, 1), ("wb97x-v", 1, 2)] {
        let base = Value::object()
            .with("xc", xc)
            .with("charge", charge)
            .with("multiplicity", multiplicity)
            .with("reference", "gks")
            .with("basis", "sto-3g")
            .with("grid_level", 0)
            .with("vv10_grid_level", 0)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol_grad", 1e-9);
        let parent = run("energy", &system, &Controls::from_value(&base).unwrap()).unwrap();
        for strength in [0.0, 1.0] {
            let sites = vec![
                Value::object()
                    .with("orbitals", vec![5_usize])
                    .with("u_eff", strength * U_EFF),
                Value::object()
                    .with("orbitals", vec![6_usize, 7])
                    .with("model", "liechtenstein")
                    .with("u", strength * U)
                    .with("j", strength * J)
                    .with(
                        "integrals",
                        (0..16)
                            .map(|i| strength * integral(i / 8, i / 4 % 2, i / 2 % 2, i % 2))
                            .collect::<Vec<_>>(),
                    ),
            ];
            let controls =
                Controls::from_value(&base.clone().with("hubbard_sites", sites)).unwrap();
            let report = run("dft-plus-u", &system, &controls).unwrap();
            let energy = number(&report, "energy");
            let correction = number(&report, "hubbard_energy");
            let residual = number(&report, "scf_orbital_residual");
            assert!(residual < 1e-9);
            if strength == 0.0 {
                assert_eq!(correction, 0.0);
                assert!((energy - number(&parent, "energy")).abs() < 1e-11);
            } else {
                assert!(correction.abs() > 1e-5);
            }
            let sum: f64 = report
                .get("hubbard_sites")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|site| number(site, "interaction") - number(site, "double_counting"))
                .sum();
            assert!((sum - correction).abs() < 1e-13);
            println!("Spinor Hubbard {xc}, charge {charge}, strength {strength}: E={energy:.16e}, Hubbard={correction:.16e}, residual={residual:.16e}");
        }
    }
}
