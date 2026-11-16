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
const PAIRS: [usize; 4] = [0, 1, 1, 2];
const COULOMB: [[f64; 3]; 3] = [
    [0.34, 0.044, 0.21],
    [0.044, 0.052, 0.037],
    [0.21, 0.037, 0.29],
];

fn integrals() -> Vec<f64> {
    PAIRS
        .iter()
        .flat_map(|&a| PAIRS.iter().map(move |&b| COULOMB[a][b]))
        .collect()
}

fn projectors(overlap: &Matrix, a: usize, b: usize) -> Vec<[f64; 2]> {
    let x = overlap[(a, a)];
    let y = overlap[(a, b)];
    let z = overlap[(b, b)];
    // Closed-form inverse square root of the positive two-dimensional metric.
    let det = (x * z - y * y).sqrt();
    let scale = 1.0 / (det * (x + z + 2.0 * det).sqrt());
    (0..overlap.rows)
        .map(|i| {
            let u = overlap[(i, a)];
            let v = overlap[(i, b)];
            [
                (u * (z + det) - v * y) * scale,
                (v * (x + det) - u * y) * scale,
            ]
        })
        .collect()
}

fn occupations(p: &Matrix, w: &[[f64; 2]]) -> [f64; 3] {
    let mut n = [0.0; 3];
    for (i, a) in w.iter().enumerate() {
        for (j, b) in w.iter().enumerate() {
            n[0] += a[0] * p[(i, j)] * b[0];
            n[1] += a[0] * p[(i, j)] * b[1];
            n[2] += a[1] * p[(i, j)] * b[1];
        }
    }
    n
}

fn liechtenstein(n: &[[f64; 3]; 2]) -> (f64, f64, [[f64; 3]; 2]) {
    let [a, c, e] = COULOMB[0];
    let [_, f, d] = COULOMB[1];
    let b = COULOMB[2][2];
    let [x, y, z] = std::array::from_fn(|i| n[0][i] + n[1][i]);
    let hartree =
        a * x * x + b * z * z + 2.0 * e * x * z + 4.0 * (c * x * y + d * z * y + f * y * y);
    let exchange = n.each_ref().map(|&[x, y, z]| {
        a * x * x
            + b * z * z
            + 2.0 * f * x * z
            + 4.0 * (c * x * y + d * z * y)
            + 2.0 * (e + f) * y * y
    });
    let ns = n.each_ref().map(|&[x, _, z]| x + z);
    let nt = x + z;
    let dc = 0.5 * (U * nt * (nt - 1.0) - J * ns.iter().map(|n| n * (n - 1.0)).sum::<f64>());
    let fock = std::array::from_fn(|s| {
        let [xs, ys, zs] = n[s];
        let shift = U * (nt - 0.5) - J * (ns[s] - 0.5);
        [
            a * (x - xs) + e * z - f * zs + 2.0 * c * (y - ys) - shift,
            c * (x - xs) + d * (z - zs) + 2.0 * f * y - (e + f) * ys,
            b * (z - zs) + e * x - f * xs + 2.0 * d * (y - ys) - shift,
        ]
    });
    (0.5 * (hartree - exchange[0] - exchange[1]), dc, fock)
}

fn dudarev(n: &[[f64; 3]; 2]) -> (f64, f64, [[f64; 3]; 2]) {
    let energy = n
        .iter()
        .map(|&[x, y, z]| x * (1.0 - x) + z * (1.0 - z) - 2.0 * y * y)
        .sum::<f64>();
    (
        0.5 * U_EFF * energy,
        0.0,
        n.each_ref()
            .map(|&[x, y, z]| [U_EFF * (0.5 - x), -U_EFF * y, U_EFF * (0.5 - z)]),
    )
}

#[test]
fn density_derivatives_match_scalar_polynomials_on_every_component() {
    // Distinct noncoplanar H coordinates give a C1 overlap problem. The
    // density probes test the full fixed-geometry AO correction.
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
    let size = overlap.rows;
    let w = projectors(&overlap, 0, 2);
    let mut central = std::array::from_fn(|_| Matrix::zeros(size, size));
    let coordinates: Vec<_> = (0..2)
        .flat_map(|s| (0..size).flat_map(move |a| (0..=a).map(move |b| (s, a, b))))
        .collect();
    for (index, &(s, a, b)) in coordinates.iter().enumerate() {
        let value = 0.016 * (index + 1) as f64 + if a == b { 0.125 } else { 0.0 };
        central[s][(a, b)] = value;
        central[s][(b, a)] = value;
    }
    for (label, interaction) in [
        ("Dudarev", Interaction::Dudarev { u_eff: U_EFF }),
        (
            "Liechtenstein",
            Interaction::Liechtenstein {
                u: U,
                j: J,
                integrals: integrals(),
            },
        ),
    ] {
        let model = HubbardModel::new(
            &overlap,
            vec![Site {
                orbitals: vec![0, 2],
                interaction,
            }],
            1e-9,
        )
        .unwrap();
        let oracle = |density: &[Matrix; 2]| {
            let n = density.each_ref().map(|p| occupations(p, &w));
            if label == "Dudarev" {
                dudarev(&n)
            } else {
                liechtenstein(&n)
            }
        };
        let ao_oracle = |p: &[Matrix; 2], (s, a, b): (usize, usize, usize)| {
            let [x, y, z] = oracle(p).2[s];
            w[a][0] * w[b][0] * x
                + (w[a][0] * w[b][1] + w[a][1] * w[b][0]) * y
                + w[a][1] * w[b][1] * z
        };
        let fields = model.evaluate(&central).unwrap();
        let (interaction, double_counting, _) = oracle(&central);
        assert!((fields.sites[0].interaction - interaction).abs() < 2e-15);
        assert!((fields.sites[0].double_counting - double_counting).abs() < 2e-15);
        assert!((fields.energy - (interaction - double_counting)).abs() < 2e-15);
        let mut failures = Vec::new();
        for &(s, a, b) in &coordinates {
            let mut variation = std::array::from_fn(|_| Matrix::zeros(size, size));
            variation[s][(a, b)] = 1.0;
            variation[s][(b, a)] = 1.0;
            let target = fields.potential[s][(a, b)];
            let multiplicity = if a == b { 1.0 } else { 2.0 };
            assert!((target - ao_oracle(&central, (s, a, b))).abs() < 2e-15);
            let induced = model.response(&variation).unwrap();
            let mut names = vec![format!("{label} Fock ({s},{a},{b})")];
            names.extend(
                coordinates
                    .iter()
                    .map(|(t, i, j)| format!("{label} kernel ({t},{i},{j}) / ({s},{a},{b})")),
            );
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], Stencil::Fourth),
                ],
                |steps, stencil| {
                    let probe = |t: f64| {
                        // The nonlinear chart has unit tangent. A direct linear
                        // chart would have zero truncation for a quadratic energy.
                        // A generic cubic term also resolves the known
                        // zero mixed-spin response blocks of Dudarev.
                        let mut p = central.each_ref().map(|matrix| {
                            let mut scaled = matrix.clone();
                            for entry in scaled.as_mut_slice() {
                                *entry *= 1.0 + t.powi(3) + 0.2 * t.powi(5);
                            }
                            scaled
                        });
                        p[s][(a, b)] += t;
                        p[s][(b, a)] = p[s][(a, b)];
                        p
                    };
                    let plus = steps.map(probe);
                    let minus = steps.map(|h| probe(-h));
                    let plus2 = steps.map(|h| probe(2.0 * h));
                    let minus2 = steps.map(|h| probe(-2.0 * h));
                    let error = |exact: f64, f: &dyn Fn(&[Matrix; 2]) -> f64| {
                        std::array::from_fn(|k| {
                            (stencil.first(
                                f(&plus[k]),
                                f(&minus[k]),
                                f(&plus2[k]),
                                f(&minus2[k]),
                                steps[k],
                            ) - exact)
                                .abs()
                        })
                    };
                    let mut errors = vec![error(multiplicity * target, &|p| {
                        let (ee, dc, _) = oracle(p);
                        ee - dc
                    })];
                    errors.extend(
                        coordinates.iter().map(|&(t, i, j)| {
                            error(induced[t][(i, j)], &|p| ao_oracle(p, (t, i, j)))
                        }),
                    );
                    errors
                },
                &mut failures,
            );
        }
        components::assert_no_failures(label, &failures);
    }
}

fn number(value: &Value, key: &str) -> f64 {
    value.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn isotropic_liechtenstein_reduces_to_dudarev() {
    let metric = Matrix::identity(2);
    let u_eff = 0.184;
    let mut integrals = vec![0.0; 16];
    for a in 0..2 {
        for b in 0..2 {
            integrals[3 * (4 * a + b)] = u_eff;
        }
    }
    let build = |interaction| {
        HubbardModel::new(
            &metric,
            vec![Site {
                orbitals: vec![0, 1],
                interaction,
            }],
            1e-9,
        )
        .unwrap()
    };
    let dudarev = build(Interaction::Dudarev { u_eff });
    let liechtenstein = build(Interaction::Liechtenstein {
        u: u_eff,
        j: 0.0,
        integrals,
    });
    for density in [
        [
            Matrix::from_row_major(2, 2, vec![0.32, 0.11, 0.11, 0.69]),
            Matrix::from_row_major(2, 2, vec![0.73, 0.05, 0.05, 0.84]),
        ],
        [
            Matrix::from_row_major(2, 2, vec![0.36, 0.48, 0.48, 0.64]),
            Matrix::zeros(2, 2),
        ],
    ] {
        let isotropic = dudarev.evaluate(&density).unwrap();
        let tensor = liechtenstein.evaluate(&density).unwrap();
        assert!((isotropic.energy - tensor.energy).abs() < 2e-16);
        for (a, b) in isotropic.potential.iter().zip(&tensor.potential) {
            assert!(a
                .as_slice()
                .iter()
                .zip(b.as_slice())
                .all(|(a, b)| (a - b).abs() < 2e-16));
        }
        if density[1].as_slice().iter().all(|&value| value == 0.0) {
            assert!(isotropic.energy.abs() < 2e-16);
        }
    }
}

#[test]
fn distorted_ammonia_reconverges_both_hubbard_models_on_required_parents() {
    let system = System::molecule(vec![7, 1, 1, 1], POSITIONS.to_vec());
    let dudarev_site = Value::object()
        .with("orbitals", [5usize, 6])
        .with("u_eff", U_EFF);
    let liechtenstein_site = Value::object()
        .with("model", "liechtenstein")
        .with("orbitals", [5usize, 6])
        .with("u", U)
        .with("j", J)
        .with("integrals", integrals());
    let zero = Value::object()
        .with("orbitals", [5usize, 6])
        .with("u_eff", 0.0);
    for (xc, reference, charge, multiplicity) in [
        ("wb97x-d", "rks", 0, 1usize),
        ("wb97x-v", "uks", 1, 2usize),
        ("wb97x-d", "roks", 1, 2usize),
        ("wb97x-v", "roks", 1, 2usize),
    ] {
        let settings = Value::object()
            .with("xc", xc)
            .with("reference", reference)
            .with("basis", "sto-3g")
            .with("charge", charge)
            .with("multiplicity", multiplicity)
            .with("grid_level", 0usize)
            .with("vv10_grid_level", 0usize)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-11)
            .with("conv_tol_dp", 1e-9)
            .with("conv_tol_grad", 1e-8);
        let parent = run("energy", &system, &Controls::from_value(&settings).unwrap()).unwrap();
        for (model, site) in [
            ("zero", &zero),
            ("dudarev", &dudarev_site),
            ("liechtenstein", &liechtenstein_site),
        ] {
            let input = settings.clone().with("hubbard_sites", vec![site.clone()]);
            let result = run(
                "dft-plus-u",
                &system,
                &Controls::from_value(&input).unwrap(),
            )
            .unwrap();
            let energy = number(&result, "energy");
            let correction = number(&result, "hubbard_energy");
            let residual = number(&result, "scf_orbital_residual");
            let entries = result.get("hubbard_sites").unwrap().as_array().unwrap();
            let item = &entries[0];
            let interaction = number(item, "interaction");
            let dc = number(item, "double_counting");
            println!("{xc} {reference} {model}: E={energy:.14e} E_U={correction:.14e} interaction={interaction:.14e} DC={dc:.14e} residual={residual:.9e}");
            assert!(residual < 1e-8);
            assert_eq!(
                result.get("reference_type").unwrap().as_str(),
                Some(reference)
            );
            assert!((correction - (interaction - dc)).abs() < 2e-15);
            if model == "zero" {
                assert_eq!(correction, 0.0);
                assert!((energy - number(&parent, "energy")).abs() < 2e-10);
            } else {
                let read_matrix = |key| {
                    let rows = item.get(key).unwrap().as_array().unwrap();
                    let at = |i: usize, j: usize| rows[i].as_array().unwrap()[j].as_f64().unwrap();
                    [at(0, 0), at(0, 1), at(1, 1)]
                };
                let n = [
                    read_matrix("occupations_alpha"),
                    read_matrix("occupations_beta"),
                ];
                let expected = if model == "dudarev" {
                    dudarev(&n)
                } else {
                    liechtenstein(&n)
                };
                assert!((expected.0 - interaction).abs() < 2e-15);
                assert!((expected.1 - dc).abs() < 2e-15);
            }
        }
    }
}
