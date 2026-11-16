// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Weighted molecular XC: single-term SCF identity, Cartesian nuclear
//! derivatives and atomic polar tensors versus independently relaxed states.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_reference.rs"]
mod nuclear_reference;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::value::ToValue;
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::xc::{
    resolve, Family, FunctionalSpec, GridVars, ResolvedFunctional, WeightedFunctionalTerm, XcOut,
};
use stencils::{fit_components_staged, Stencil};

const POSITIONS: [[f64; 3]; 4] = [
    [0.12, -0.18, 0.24],
    [1.61, 0.12, -0.25],
    [-0.34, 2.82, 0.53],
    [0.83, 3.36, 1.14],
];

const CONSTRAINED_POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.21, 0.19],
    [1.57, 0.16, -0.08],
    [-0.19, 1.93, 0.37],
    [0.42, 0.29, 2.28],
];

fn controls(selection: &str, reference: &str) -> Controls {
    let mut controls = Controls::from_value(
        &Value::parse_json(&format!(
            r#"{{{selection},"reference":"{reference}","grid_level":1,"grid_weight_cutoff":0,
             "conv_tol":1e-13,"conv_tol_dp":1e-11,"conv_tol_grad":1e-11}}"#
        ))
        .unwrap(),
    )
    .unwrap();
    if reference != "rks" {
        controls.insert("charge", Value::Int(1)).unwrap();
        controls.insert("multiplicity", Value::Int(2)).unwrap();
    }
    if matches!(reference, "roks" | "gks") {
        controls.insert("max_iter", Value::Int(300)).unwrap();
        controls.insert("guess", Value::Str("core".into())).unwrap();
    }
    if reference == "gks" {
        let source = spinor_operator::source(4);
        controls
            .insert("gks_operator_real", source.re.to_value())
            .unwrap();
        controls
            .insert("gks_operator_imaginary", source.im.to_value())
            .unwrap();
    }
    controls
}

fn report(feature: &str, positions: [[f64; 3]; 4], controls: &Controls) -> Value {
    report_at(
        feature,
        &System::molecule(vec![1; 4], positions.to_vec()),
        controls,
    )
}

fn report_at(feature: &str, system: &System, controls: &Controls) -> Value {
    let result = run(feature, system, controls).unwrap();
    assert_eq!(result.get("converged").and_then(Value::as_bool), Some(true));
    if controls.contains("xc_terms") {
        let residual = result
            .get("scf_orbital_residual")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!(residual.is_finite() && residual < 1e-10, "{residual:.12e}");
    }
    result
}

fn energy(positions: [[f64; 3]; 4], controls: &Controls) -> f64 {
    report("energy", positions, controls)
        .get("energy")
        .unwrap()
        .as_f64()
        .unwrap()
}

#[test]
fn one_weighted_term_reproduces_the_named_reference() {
    for (reference, positions) in [
        ("rks", POSITIONS),
        ("uks", POSITIONS),
        ("roks", CONSTRAINED_POSITIONS),
        ("gks", CONSTRAINED_POSITIONS),
    ] {
        let named = controls(r#""xc":"pbe0""#, reference);
        let mixed = controls(r#""xc_terms":[{"xc":"pbe0","weight":1}]"#, reference);
        assert!((energy(positions, &named) - energy(positions, &mixed)).abs() < 1e-12);
    }
}

#[test]
fn weighted_point_energy_matches_separate_term_evaluations() {
    let terms = [("tpss", 0.4), ("pbe0", 0.75), ("svwn5", -0.2)].map(|(name, weight)| {
        WeightedFunctionalTerm {
            weight,
            functional: FunctionalSpec::Combined(name.into()),
        }
    });
    let vars = GridVars {
        rho: [&[0.31], &[0.22]],
        sigma: Some([&[0.041], &[0.017], &[0.029]]),
        tau: Some([&[0.14], &[0.11]]),
    };
    let mixed = ResolvedFunctional::weighted(&terms).unwrap();
    assert_eq!(mixed.family, Family::MetaGga);
    assert!((mixed.hybrid.cx - 0.1875).abs() < 1e-15);
    let mut result = XcOut::default();
    mixed.eval(&vars, 0, &mut result).unwrap();
    let expected = terms
        .iter()
        .map(|term| {
            let mut component = XcOut::default();
            resolve(&term.functional)
                .unwrap()
                .eval(&vars, 0, &mut component)
                .unwrap();
            println!(
                "{:?}: energy density {:.12e}",
                term.functional, component.e[0]
            );
            term.weight * component.e[0]
        })
        .sum::<f64>();
    assert!((result.e[0] - expected).abs() < 1e-14);
}

#[test]
fn weighted_local_and_exchange_gradient_fits_every_coordinate() {
    let recipe = r#""xc_terms":[{"x":"slater","c":"vwn5","weight":0.2},
                   {"xc":"pbe0","weight":0.65},{"xc":"hf","weight":0.15}]"#;
    let intervals = [
        ([0.032, 0.024, 0.016, 0.012], Stencil::Second),
        ([0.016, 0.012, 0.008, 0.006], Stencil::Second),
        ([0.064, 0.048, 0.032, 0.024], Stencil::Second),
        ([0.032, 0.024, 0.016, 0.012], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    for reference in ["rks", "uks", "roks", "gks"] {
        let positions = if matches!(reference, "roks" | "gks") {
            CONSTRAINED_POSITIONS
        } else {
            POSITIONS
        };
        let method = controls(recipe, reference);
        let base = report("gradient", positions, &method);
        let gradient: Vec<_> = base
            .get("gradient")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_f64().unwrap())
            })
            .collect();
        assert_eq!(gradient.len(), 12);
        let translation = (0..3)
            .map(|axis| gradient.iter().skip(axis).step_by(3).sum::<f64>().powi(2))
            .sum::<f64>()
            .sqrt();
        println!("{reference}: translation residual {translation:.12e} hartree/bohr");
        let orbital = base.get("scf_orbital_residual").unwrap().as_f64().unwrap();
        println!("{reference}: orbital residual {orbital:.12e} hartree");
        assert!(translation < 1e-9);
        let names = (0..gradient.len())
            .map(|index| format!("{reference} atom {} axis {}", index / 3, index % 3))
            .collect::<Vec<_>>();
        fit_components_staged(
            &names,
            &intervals,
            |steps, stencil| {
                gradient
                    .iter()
                    .enumerate()
                    .map(|(index, &analytic)| {
                        steps.map(|h| {
                            let displaced = |shift| {
                                let mut positions = positions;
                                positions[index / 3][index % 3] += shift;
                                energy(positions, &method)
                            };
                            let plus = displaced(h);
                            let minus = displaced(-h);
                            let estimate = match stencil {
                                Stencil::Second => stencil.first(plus, minus, 0.0, 0.0, h),
                                Stencil::Fourth => stencil.first(
                                    plus,
                                    minus,
                                    displaced(2.0 * h),
                                    displaced(-2.0 * h),
                                    h,
                                ),
                            };
                            (estimate - analytic).abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures("weighted XC analytic gradients", &failures);
}

const HESSIAN_STAGES: [([f64; 4], Stencil); 5] = [
    ([0.02, 0.015, 0.01, 0.0075], Stencil::Second),
    ([0.004, 0.003, 0.002, 0.0015], Stencil::Second),
    ([0.08, 0.06, 0.04, 0.03], Stencil::Second),
    ([0.02, 0.015, 0.01, 0.0075], Stencil::Fourth),
    ([0.04, 0.03, 0.02, 0.015], Stencil::Fourth),
];

const FIELD_INTERVALS: [[f64; 4]; 3] = [
    [0.008, 0.006, 0.004, 0.003],
    [0.004, 0.003, 0.002, 0.0015],
    [0.016, 0.012, 0.008, 0.006],
];

fn fit_weighted_hessian(reference: &str) {
    let positions = if reference == "rks" {
        POSITIONS
    } else {
        CONSTRAINED_POSITIONS
    };
    let system = System::molecule(vec![1; 4], positions.to_vec());
    let controls = controls(
        r#""xc_terms":[{"xc":"hf","weight":0.65},{"xc":"svwn5","weight":0.35}]"#,
        reference,
    );
    let label = format!("weighted {reference}");
    let report = nuclear_reference::fit(
        &label,
        &system,
        &controls,
        &HESSIAN_STAGES,
        &[""],
        report_at,
    );
    let hessian = report.get("hessian").unwrap().numeric_array().unwrap().1;
    let dim = 3 * positions.len();
    let mut acoustic_defect = 0.0_f64;
    for row in hessian.chunks_exact(dim) {
        for axis in 0..3 {
            let sum = row.iter().skip(axis).step_by(3).sum::<f64>();
            acoustic_defect = acoustic_defect.max(sum.abs());
        }
    }
    let orbital = report
        .get("scf_orbital_residual")
        .unwrap()
        .as_f64()
        .unwrap();
    println!("{label} Hessian acoustic defect={acoustic_defect:.12e} orbital={orbital:.12e}");
    assert!(acoustic_defect < 1e-8);
    if reference != "gks" {
        fit_atomic_polar_tensor(&label, &system, &controls, &report, dim);
    }
    if reference == "roks" {
        fit_open_shell_polarizability(&label, &system, &controls, &report);
    }
}

fn field_report(feature: &str, system: &System, controls: &Controls, field: [f64; 3]) -> Value {
    let mut request = controls.to_value();
    request.set("field", field);
    report_at(feature, system, &Controls::from_value(&request).unwrap())
}

fn fit_atomic_polar_tensor(
    label: &str,
    system: &System,
    controls: &Controls,
    hessian: &Value,
    dim: usize,
) {
    let (shape, analytic) = hessian
        .get("dipole_derivatives")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert_eq!(shape, [3, dim]);
    let charge = controls.opt_i64("charge").unwrap().unwrap_or(0) as f64;
    let mut translation_defect = 0.0_f64;
    for (axis, row) in analytic.chunks_exact(dim).enumerate() {
        for coordinate in 0..3 {
            let expected = if axis == coordinate { charge } else { 0.0 };
            let sum = row.iter().skip(coordinate).step_by(3).sum::<f64>();
            translation_defect = translation_defect.max((sum - expected).abs());
        }
    }
    println!("{label} atomic polar tensor translation defect={translation_defect:.12e}");
    assert!(translation_defect < 1e-8);
    let names: Vec<_> = (0..3)
        .flat_map(|axis| {
            (0..dim).map(move |coordinate| format!("{label} d dipole[{axis}]/d R[{coordinate}]"))
        })
        .collect();
    let mut failures = Vec::new();
    fit_components_staged(
        &names,
        &[
            (FIELD_INTERVALS[0], Stencil::Second),
            (FIELD_INTERVALS[1], Stencil::Second),
            ([0.002, 0.0015, 0.001, 0.00075], Stencil::Second),
            (FIELD_INTERVALS[2], Stencil::Second),
            (FIELD_INTERVALS[0], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; analytic.len()];
            for axis in 0..3 {
                let gradient = |strength| {
                    let mut field = [0.0; 3];
                    field[axis] = strength;
                    field_report("gradient", system, controls, field)
                        .get("gradient")
                        .unwrap()
                        .numeric_array()
                        .unwrap()
                        .1
                };
                for (i, &step) in steps.iter().enumerate() {
                    let (plus, minus) = (gradient(step), gradient(-step));
                    let (plus2, minus2) = match stencil {
                        Stencil::Second => (vec![0.0; dim], vec![0.0; dim]),
                        Stencil::Fourth => (gradient(2.0 * step), gradient(-2.0 * step)),
                    };
                    for coordinate in 0..dim {
                        let index = axis * dim + coordinate;
                        // The field coupling is -mu.E, so the mixed energy
                        // derivative of a nuclear gradient is -d(mu)/dR.
                        let measured = -stencil.first(
                            plus[coordinate],
                            minus[coordinate],
                            plus2[coordinate],
                            minus2[coordinate],
                            step,
                        );
                        errors[index][i] = (measured - analytic[index]).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(label, &failures);
}

fn fit_open_shell_polarizability(
    label: &str,
    system: &System,
    controls: &Controls,
    hessian: &Value,
) {
    let (shape, analytic) = hessian
        .get("polarizability")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert_eq!(shape, [3, 3]);
    let energy = |field| {
        field_report("energy", system, controls, field)
            .get("energy")
            .unwrap()
            .as_f64()
            .unwrap()
    };
    let center = energy([0.0; 3]);
    let axes = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let names: Vec<_> = axes
        .iter()
        .map(|(a, b)| format!("{label} alpha[{a},{b}]"))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &FIELD_INTERVALS,
        |steps| {
            axes.iter()
                .map(|&(a, b)| {
                    assert!((analytic[3 * a + b] - analytic[3 * b + a]).abs() < 1e-8);
                    steps.map(|h| {
                        let measured = if a == b {
                            let mut field = [0.0; 3];
                            field[a] = h;
                            let plus = energy(field);
                            field[a] = -h;
                            (2.0 * center - plus - energy(field)) / (h * h)
                        } else {
                            let mut mixed = 0.0;
                            for sa in [-1.0, 1.0] {
                                for sb in [-1.0, 1.0] {
                                    let mut field = [0.0; 3];
                                    field[a] = sa * h;
                                    field[b] = sb * h;
                                    mixed -= sa * sb * energy(field);
                                }
                            }
                            mixed / (4.0 * h * h)
                        };
                        (measured - analytic[3 * a + b]).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures(label, &failures);
}

#[test]
fn weighted_rks_hessian_and_atomic_polar_tensor_fit_every_component() {
    fit_weighted_hessian("rks");
}

#[test]
fn weighted_uks_hessian_and_atomic_polar_tensor_fit_every_component() {
    fit_weighted_hessian("uks");
}

#[test]
fn weighted_roks_hessian_and_atomic_polar_tensor_fit_every_component() {
    fit_weighted_hessian("roks");
}

#[test]
fn weighted_gks_hessian_fits_every_component() {
    fit_weighted_hessian("gks");
}
