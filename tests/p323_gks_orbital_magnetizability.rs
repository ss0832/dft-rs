// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Compare the common-origin spinor response with the curvature of fully
//! reconverged energies in independent static magnetic field components.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::MolBasis;
use dft_rs::driver::{method::build_basis, run, Controls, System, Value};
use dft_rs::integrals::one_electron::moment_matrices;
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::periodic::{spinor_scf::spin_diagonal, BlochMatrix};
use dft_rs::properties::nmr::angular_momentum_operator;
use dft_rs::system::{Atom, Molecule};
use std::collections::BTreeMap;
use stencils::Stencil;

const NUMBERS: [u8; 4] = [7, 1, 1, 1];
const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.78, 0.34, -0.08],
    [-0.49, 1.65, 0.41],
    [0.31, -0.78, 1.91],
];
const ORIGIN: [f64; 3] = [0.08, 0.01, 0.23];
const COMPONENTS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

#[derive(Clone, Copy)]
enum Perturbation {
    Full,
    Diamagnetic,
    Paramagnetic,
}

fn options() -> Value {
    Value::object()
        .with("xc", "hsesol")
        .with("reference", "gks")
        .with("basis", "sto-3g")
        .with("charge", 1)
        .with("multiplicity", 2)
        .with("grid_level", 0)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-12)
        .with("max_iter", 300)
        .with("cpu_threads", 1)
}

fn vector([x, y, z]: [f64; 3]) -> Vec3 {
    Vec3::new(x, y, z)
}

fn field_energy(
    system: &System,
    options: &Value,
    angular: &[Matrix; 3],
    moments: &[((i32, i32, i32), Matrix)],
    field: [f64; 3],
    part: Perturbation,
) -> f64 {
    // Insert a finite magnetic Hamiltonian before the SCF optimization.
    // Its linear term is B.L/2 and its quadratic term is |B x r|²/8.
    let [bx, by, bz] = field;
    let coefficients = [
        ((2, 0, 0), (by * by + bz * bz) / 8.0),
        ((0, 2, 0), (bx * bx + bz * bz) / 8.0),
        ((0, 0, 2), (bx * bx + by * by) / 8.0),
        ((1, 1, 0), -bx * by / 4.0),
        ((1, 0, 1), -bx * bz / 4.0),
        ((0, 1, 1), -by * bz / 4.0),
    ];
    let n = angular[0].rows;
    let mut re = Matrix::zeros(n, n);
    let mut im = Matrix::zeros(n, n);
    if !matches!(part, Perturbation::Paramagnetic) {
        for (power, coefficient) in coefficients {
            let matrix = &moments.iter().find(|(p, _)| *p == power).unwrap().1;
            for (value, integral) in re.as_mut_slice().iter_mut().zip(matrix.as_slice()) {
                *value += coefficient * integral;
            }
        }
    }
    if !matches!(part, Perturbation::Diamagnetic) {
        for (component, matrix) in field.into_iter().zip(angular) {
            for (value, integral) in im.as_mut_slice().iter_mut().zip(matrix.as_slice()) {
                *value -= 0.5 * component * integral;
            }
        }
    }
    let source = spin_diagonal(&BlochMatrix { re, im });
    let options = options
        .clone()
        .with("gks_operator_real", source.re)
        .with("gks_operator_imaginary", source.im);
    let report = run("energy", system, &Controls::from_value(&options).unwrap())
        .unwrap_or_else(|error| panic!("field {field:?}: {error}"));
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    report.get("energy").and_then(Value::as_f64).unwrap()
}

fn first(stencil: Stencil, h: f64, mut sample: impl FnMut(f64) -> f64) -> f64 {
    let plus = sample(h);
    let minus = sample(-h);
    let (plus2, minus2) = if stencil == Stencil::Fourth {
        (sample(2.0 * h), sample(-2.0 * h))
    } else {
        (0.0, 0.0)
    };
    stencil.first(plus, minus, plus2, minus2, h)
}

#[test]
fn all_six_static_spinor_orbital_components_agree_with_energy_curvatures() {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let options = options();
    let analytic_options = options
        .clone()
        .with("gauge", "common-origin")
        .with("gauge_origin", ORIGIN);
    let report = run(
        "magnetizability",
        &system,
        &Controls::from_value(&analytic_options).unwrap(),
    )
    .unwrap();
    println!("HSEsol GKS orbital magnetizability {}", report.to_json());
    let values = |key| report.get(key).unwrap().numeric_array().unwrap().1;
    let analytic = values("magnetizability");
    let dia = values("diamagnetic");
    let para = values("paramagnetic");
    for ((total, d), p) in analytic.iter().zip(&dia).zip(&para) {
        assert!((total - d - p).abs() < 2e-14);
    }
    assert!(report.get("raw_asymmetry").and_then(Value::as_f64).unwrap() < 1e-9);
    // Translating the molecule and origin together leaves the finite-basis
    // orbital observable invariant, even though the origin choice matters.
    let shift = [0.53, -0.41, 0.72];
    let moved = System::molecule(
        NUMBERS.to_vec(),
        POSITIONS
            .map(|r| std::array::from_fn(|a| r[a] + shift[a]))
            .to_vec(),
    );
    let shifted_origin: [f64; 3] = std::array::from_fn(|a| ORIGIN[a] + shift[a]);
    let moved_controls = analytic_options.with("gauge_origin", shifted_origin);
    let translated = run(
        "magnetizability",
        &moved,
        &Controls::from_value(&moved_controls).unwrap(),
    )
    .unwrap();
    for key in ["magnetizability", "diamagnetic", "paramagnetic"] {
        let value = translated.get(key).unwrap().numeric_array().unwrap().1;
        let defect = value
            .into_iter()
            .zip(values(key))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("Rigid shift of common-origin {key}: {defect:e}");
        assert!(defect < 2e-9, "translated {key} defect {defect:e}");
    }
    let atoms = NUMBERS
        .into_iter()
        .zip(POSITIONS)
        .map(|(z, position)| Atom {
            z,
            position: vector(position),
        })
        .collect();
    let basis =
        MolBasis::build(&Molecule::new(atoms), &build_basis("sto-3g", None).unwrap()).unwrap();
    let angular = angular_momentum_operator(&basis, vector(ORIGIN));
    let moments = moment_matrices(&basis, vector(ORIGIN), 2);
    let stages = [
        ([0.04, 0.028, 0.018, 0.012], Stencil::Second),
        ([0.02, 0.014, 0.009, 0.006], Stencil::Second),
        ([0.008, 0.006, 0.004, 0.003], Stencil::Second),
        ([0.025, 0.02, 0.015, 0.01], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    // Isolate the fixed-density and relaxed terms by including only the
    // quadratic or only the linear Hamiltonian term in the independent SCF.
    for (label, part, target) in [
        ("total", Perturbation::Full, analytic),
        ("diamagnetic", Perturbation::Diamagnetic, dia),
        ("paramagnetic", Perturbation::Paramagnetic, para),
    ] {
        let mut samples = BTreeMap::new();
        for (a, b) in COMPONENTS {
            let name = format!("GKS HSEsol {label} xi[{a},{b}]");
            stencils::fit_components_staged(
                &[name],
                &stages,
                |steps, stencil| {
                    vec![steps.map(|h| {
                        let numerical = -first(stencil, h, |x| {
                            first(stencil, h, |y| {
                                let mut field = [0.0; 3];
                                field[a] += x;
                                field[b] += y;
                                let key = field.map(f64::to_bits);
                                *samples.entry(key).or_insert_with(|| {
                                    field_energy(&system, &options, &angular, &moments, field, part)
                                })
                            })
                        });
                        (target[3 * a + b] - numerical).abs()
                    })]
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("static orbital GKS magnetizability", &failures);
}
