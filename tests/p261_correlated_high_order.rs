// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete biased static beta/gamma and complete contracted cubic/quartic
//! matrices. Every SS/OS and target channel is fitted independently. Direct
//! displaced analytic gradients or the older real-orbital alpha provider are
//! the oracles; no output component is copied by symmetry in these checks.
//! Initial direct third-energy differences lost the error order to SCF
//! cancellation on seven entries; those failed intervals are retained.
//! Bounded acceptance, 2026-10-26: 1,750 complete static beta/gamma
//! component/channel fits across HF/SVWN5/PBE/TPSS/PBE0 parents and RKS/UKS;
//! HF uses a distinct PBE target. The contracted nuclear matrices have
//! 2,304 RKS and 3,600 UKS fits, including two resolved UKS entries. Original
//! failures, resolved step intervals and raw residuals are in docs/response.md.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_directions.rs"]
mod symmetric_directions;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::london::FieldPath;
use dft_rs::response::correlated_field::correlated_field_taylor;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::{double_hybrid::Pt2Coefficients, resolve, FunctionalSpec};
#[path = "support/correlated_fixture.rs"]
mod correlated_fixture;
use correlated_fixture::fixture;
use std::collections::BTreeMap;

const MODELS: [&str; 5] = ["hf", "svwn5", "pbe", "tpss", "pbe0"];
const PREFIXES: [&str; 8] = [
    "",
    "parent_",
    "target_",
    "pt2_",
    "pt2_same_spin_alpha_",
    "pt2_opposite_spin_",
    "pt2_same_spin_beta_",
    "dispersion_",
];
fn selected(model: &str, u: bool) -> bool {
    std::env::var("DFT_RS_TEST_MODEL")
        .ok()
        .is_none_or(|s| s == model)
        && std::env::var("DFT_RS_TEST_UNRESTRICTED")
            .ok()
            .is_none_or(|s| (s == "1") == u)
}
fn calculate(id: &str, system: &System, options: &Value) -> Value {
    let mut options = options.clone();
    if id == "energy" {
        if let Value::Object(entries) = &mut options {
            entries.retain(|(key, _)| key != "response_tolerance");
        }
    }
    let report = run(id, system, &Controls::from_value(&options).unwrap()).unwrap();
    let diagnostics: Vec<_> = [
        "response_residuals",
        "amplitude_response_residuals",
        "amplitude_relative_residuals",
        "projector_defects",
        "nuclear_stationarity_defects",
        "metric_defects",
        "response_number_defects",
        "orbital_frame_defects",
        "reference_defects",
    ]
    .into_iter()
    .filter_map(|key| {
        report.get(key).map(|value| {
            (
                key,
                value
                    .numeric_array()
                    .unwrap()
                    .1
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0, f64::max),
            )
        })
    })
    .collect();
    assert_eq!(
        report.get("converged").and_then(Value::as_bool),
        Some(true),
        "{id}: {diagnostics:?}"
    );
    report
}
fn scalar(report: &Value, key: &str) -> f64 {
    report.get(key).unwrap().as_f64().unwrap()
}
fn array(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}
fn contraction(tensor: &[f64], direction: [f64; 3]) -> f64 {
    tensor
        .iter()
        .enumerate()
        .map(|(mut flat, &value)| {
            let mut factor = 1.0;
            let mut size = tensor.len();
            while size > 1 {
                factor *= direction[flat % 3];
                flat /= 3;
                size /= 3;
            }
            value * factor
        })
        .sum()
}

#[test]
fn every_biased_dipole_channel_and_constant_energy_matches_independent_finite_scf() {
    for model in MODELS {
        for u in [false, true] {
            if !selected(model, u) {
                continue;
            }
            let (system, options) = fixture(model, u, "none");
            let method =
                MolecularMethod::from_controls(&Controls::from_value(&options).unwrap()).unwrap();
            let mut setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
            if setup.grid.is_none() {
                setup.grid = Some(
                    dft_rs::grid::MolecularGrid::build(&setup.molecule, &setup.grid_options)
                        .unwrap(),
                );
            }
            let reference = run_molecular_reference(&setup).unwrap();
            assert!(reference.scf.converged);
            let target =
                (model == "hf").then(|| resolve(&FunctionalSpec::Combined("pbe".into())).unwrap());
            let center = calculate("energy", &system, &options);
            let channels = |value: &Value| {
                [
                    scalar(value, "energy"),
                    scalar(value, "orbital_reference_energy"),
                    scalar(value, "reference_energy"),
                    scalar(value, "pt2_correlation"),
                    scalar(value, "pt2_same_spin_alpha"),
                    scalar(value, "pt2_opposite_spin"),
                    scalar(value, "pt2_same_spin_beta"),
                ]
            };
            let config = ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            };
            let mut failures = Vec::new();
            for a in 0..3 {
                let mut direction = [0.0; 3];
                direction[a] = 1.0;
                let value = correlated_field_taylor(
                    &setup.molecule,
                    &setup.basis,
                    &setup.functional,
                    target.as_ref(),
                    setup.grid.as_ref(),
                    &reference.scf,
                    &setup.options,
                    Pt2Coefficients {
                        same_spin: 0.31,
                        opposite_spin: 0.47,
                    },
                    &FieldPath {
                        electric: direction,
                        magnetic: [0.0; 3],
                        origin: [0.0; 3],
                        london: false,
                        order: 1,
                    },
                    None,
                    2048,
                    &config,
                )
                .unwrap();
                let paths = [
                    &value.energy,
                    &value.parent.parent.energy,
                    &value.parent.target_energy,
                    &value.correlation,
                    &value.same_spin_alpha,
                    &value.opposite_spin,
                    &value.same_spin_beta,
                ];
                for (p, e) in paths.iter().zip(channels(&center)) {
                    assert!((p[0] - e).abs() < 1e-10);
                }
                let names: Vec<_> = PREFIXES[..7]
                    .iter()
                    .map(|p| format!("{model} U={u} biased electric energy derivative {p}axis={a}"))
                    .collect();
                components::fit_components(
                    &names,
                    &[
                        [0.02, 0.015, 0.01, 0.0075],
                        [0.005, 0.00375, 0.0025, 0.001875],
                        [0.04, 0.03, 0.02, 0.015],
                    ],
                    |steps| {
                        let mut errors = vec![[0.0; 4]; 7];
                        for (k, &h) in steps.iter().enumerate() {
                            let mut values = Vec::new();
                            for sign in [-1.0, 1.0] {
                                let mut controls = options.clone();
                                let mut field = [0.003, -0.002, 0.004];
                                field[a] += sign * h;
                                controls.set("reference_field", field);
                                values.push(channels(&calculate("energy", &system, &controls)));
                            }
                            for c in 0..7 {
                                errors[c][k] =
                                    ((values[1][c] - values[0][c]) / (2.0 * h) - paths[c][1]).abs();
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
            }
            components::assert_no_failures("every biased dipole energy channel", &failures);
        }
    }
}

#[test]
fn all_static_beta_gamma_components_have_independent_lower_response_fits() {
    for model in MODELS {
        for u in [false, true] {
            if !selected(model, u) {
                continue;
            }
            let (system, options) = fixture(model, u, "none");
            for rank in [3usize, 4] {
                let id = if rank == 3 { "beta" } else { "gamma" };
                println!("begin {model} U={u} complete {id}");
                let report = calculate(id, &system, &options);
                for key in [
                    "amplitude_response_residuals",
                    "amplitude_relative_residuals",
                ] {
                    println!(
                        "{model} U={u} rank={rank} {key} maximum {:.12e}",
                        array(&report, key)
                            .iter()
                            .map(|v| v.abs())
                            .fold(0.0, f64::max)
                    );
                }
                assert_eq!(
                    report.get("shape").unwrap().numeric_array().unwrap().1,
                    vec![3.0; rank]
                );
                assert_eq!(report.get("energy_permutation_defect"), Some(&Value::Null));
                assert_eq!(
                    report
                        .get("energy_permutation_comparisons")
                        .and_then(Value::as_i64),
                    Some(0)
                );
                let tensors: Vec<_> = PREFIXES[..7]
                    .iter()
                    .map(|p| array(&report, &format!("{p}tensor")))
                    .collect();
                for key in ["parent_nonlocal_tensor", "target_nonlocal_tensor"] {
                    assert!(array(&report, key).iter().all(|&v| v == 0.0));
                }
                let mut failures = Vec::new();
                for (index, direction) in symmetric_directions::directions(rank)
                    .into_iter()
                    .enumerate()
                {
                    let exact: Vec<_> = tensors.iter().map(|t| contraction(t, direction)).collect();
                    let mut cache = BTreeMap::<u64, [f64; 7]>::new();
                    let mut alpha = |t: f64| {
                        if let Some(v) = cache.get(&t.to_bits()) {
                            return *v;
                        }
                        let mut controls = options.clone();
                        controls.set(
                            "reference_field",
                            [
                                0.003 + direction[0] * t,
                                -0.002 + direction[1] * t,
                                0.004 + direction[2] * t,
                            ],
                        );
                        let report = calculate("polarizability", &system, &controls);
                        let parent =
                            contraction(&array(&report, "parent_polarizability"), direction);
                        let values = [
                            contraction(&array(&report, "polarizability"), direction),
                            parent,
                            parent
                                + contraction(
                                    &array(&report, "target_polarizability_correction"),
                                    direction,
                                ),
                            contraction(
                                &array(&report, "pt2_polarizability_correction"),
                                direction,
                            ),
                            contraction(
                                &array(&report, "pt2_same_spin_alpha_polarizability"),
                                direction,
                            ),
                            contraction(
                                &array(&report, "pt2_opposite_spin_polarizability"),
                                direction,
                            ),
                            contraction(
                                &array(&report, "pt2_same_spin_beta_polarizability"),
                                direction,
                            ),
                        ];
                        cache.insert(t.to_bits(), values);
                        values
                    };
                    let names: Vec<_> = PREFIXES[..7]
                        .iter()
                        .map(|p| format!("{model} U={u} rank={rank} generic={index} {p}electric"))
                        .collect();
                    stencils::fit_components_staged(
                        &names,
                        &[
                            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                            (
                                [0.005, 0.00375, 0.0025, 0.001875],
                                stencils::Stencil::Second,
                            ),
                            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
                            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
                            (
                                [0.005, 0.00375, 0.0025, 0.001875],
                                stencils::Stencil::Fourth,
                            ),
                            ([0.00125, 0.001, 0.00075, 0.0005], stencils::Stencil::Second),
                        ],
                        |steps, stencil| {
                            let center = alpha(0.0);
                            let mut errors = vec![[0.0; 4]; 7];
                            for (k, &h) in steps.iter().enumerate() {
                                let plus = alpha(h);
                                let minus = alpha(-h);
                                let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                                    (alpha(2.0 * h), alpha(-2.0 * h))
                                } else {
                                    ([0.0; 7], [0.0; 7])
                                };
                                for c in 0..7 {
                                    let value = if rank == 3 {
                                        stencil.first(plus[c], minus[c], plus2[c], minus2[c], h)
                                    } else if stencil == stencils::Stencil::Second {
                                        (plus[c] - 2.0 * center[c] + minus[c]) / (h * h)
                                    } else {
                                        (-plus2[c] + 16.0 * plus[c] - 30.0 * center[c]
                                            + 16.0 * minus[c]
                                            - minus2[c])
                                            / (12.0 * h * h)
                                    };
                                    errors[c][k] = (value - exact[c]).abs();
                                }
                            }
                            errors
                        },
                        &mut failures,
                    );
                }
                components::assert_no_failures(
                    "complete correlated static electric tensor",
                    &failures,
                );
            }
        }
    }
}

#[test]
fn every_cubic_quartic_contracted_matrix_entry_has_independent_gradient_fits() {
    for model in MODELS {
        for u in [false, true] {
            if !selected(model, u) {
                continue;
            }
            let (system, options) = fixture(model, u, "d3bj");
            let dim = 3 * system.numbers.len();
            let probe: Vec<_> = (0..dim)
                .map(|a| 0.17 + 0.09 * (a as f64 * 1.7).sin())
                .collect();
            let second: Vec<_> = (0..dim)
                .map(|a| -0.13 + 0.08 * (a as f64 * 0.9).cos())
                .collect();
            let mut cache = BTreeMap::<Vec<u64>, [Vec<f64>; 8]>::new();
            let rank_filter = std::env::var("DFT_RS_TEST_NUCLEAR_RANK").ok();
            let component_filter = std::env::var("DFT_RS_TEST_NUCLEAR_COMPONENTS").ok();
            if rank_filter.is_some() || component_filter.is_some() {
                println!(
                    "diagnostic subset: rank={rank_filter:?}, components={component_filter:?}"
                );
            }
            for rank in [3usize, 4] {
                if rank_filter
                    .as_ref()
                    .is_some_and(|value| value != &rank.to_string())
                {
                    continue;
                }
                let id = if rank == 3 {
                    "cubic-fc-along"
                } else {
                    "quartic-fc-along"
                };
                let mut controls = options.clone();
                controls.set("probe", probe.clone());
                if rank == 4 {
                    controls.set("second_probe", second.clone());
                }
                println!("begin {model} U={u} complete {id} matrix");
                let report = calculate(id, &system, &controls);
                let exact: Vec<_> = PREFIXES
                    .iter()
                    .map(|p| array(&report, &format!("{p}contracted_force_constants")))
                    .collect();
                assert!(exact.iter().all(|v| v.len() == dim * dim));
                for (channel, tensor) in exact.iter().enumerate() {
                    let mut defect = 0.0_f64;
                    for a in 0..3 {
                        for j in 0..dim {
                            let row = (0..system.numbers.len())
                                .map(|atom| tensor[(3 * atom + a) * dim + j])
                                .sum::<f64>();
                            let column = (0..system.numbers.len())
                                .map(|atom| tensor[j * dim + 3 * atom + a])
                                .sum::<f64>();
                            defect = defect.max(row.abs()).max(column.abs());
                        }
                    }
                    let scale = tensor.iter().map(|v| v.abs()).fold(1.0, f64::max);
                    println!("{model} U={u} nuclear rank={rank} channel={channel} translation defect {defect:.12e}");
                    assert!(defect < 1e-8 * scale);
                }
                let mut failures = Vec::new();
                for i in 0..dim {
                    for j in 0..dim {
                        if component_filter.as_ref().is_some_and(|values| {
                            !values.split(',').any(|value| value == format!("{i}:{j}"))
                        }) {
                            continue;
                        }
                        let mut directions = vec![probe.clone()];
                        if rank == 4 {
                            directions.push(second.clone());
                        }
                        let mut d = vec![0.0; dim];
                        d[j] = 1.0;
                        directions.push(d);
                        let mut finite = |h: f64| {
                            let mut sum = [0.0; 8];
                            for signs in 0..1usize << directions.len() {
                                let mut displaced = system.clone();
                                let mut sign = 1.0;
                                for (k, d) in directions.iter().enumerate() {
                                    let s = if signs & (1 << k) == 0 { -1.0 } else { 1.0 };
                                    sign *= s;
                                    for a in 0..dim {
                                        displaced.positions[a / 3][a % 3] += s * h * d[a];
                                    }
                                }
                                let key: Vec<_> = displaced
                                    .positions
                                    .iter()
                                    .flatten()
                                    .map(|v| v.to_bits())
                                    .collect();
                                let value = cache.entry(key).or_insert_with(|| {
                                    let gradient = calculate("gradient", &displaced, &options);
                                    std::array::from_fn(|c| {
                                        array(&gradient, &format!("{}gradient", PREFIXES[c]))
                                    })
                                });
                                for c in 0..8 {
                                    sum[c] += sign * value[c][i];
                                }
                            }
                            sum.map(|v| v / (2.0 * h).powi((rank - 1) as i32))
                        };
                        let names: Vec<_> = PREFIXES
                            .iter()
                            .map(|p| format!("{model} U={u} nuclear rank={rank} {p}({i},{j})"))
                            .collect();
                        stencils::fit_components_staged(
                            &names,
                            &[
                                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                                ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                                ([0.16, 0.12, 0.08, 0.06], stencils::Stencil::Second),
                                ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
                                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
                                (
                                    [0.005, 0.00375, 0.0025, 0.001875],
                                    stencils::Stencil::Second,
                                ),
                                ([0.04, 0.035, 0.03, 0.025], stencils::Stencil::Fourth),
                                ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
                                ([0.01, 0.008, 0.006, 0.005], stencils::Stencil::Second),
                                ([0.025, 0.0225, 0.02, 0.0175], stencils::Stencil::Fourth),
                            ],
                            |steps, stencil| {
                                let mut errors = vec![[0.0; 4]; 8];
                                for (k, &h) in steps.iter().enumerate() {
                                    let near = finite(h);
                                    let far = if stencil == stencils::Stencil::Fourth {
                                        finite(2.0 * h)
                                    } else {
                                        [0.0; 8]
                                    };
                                    for c in 0..8 {
                                        let v = if stencil == stencils::Stencil::Fourth {
                                            (4.0 * near[c] - far[c]) / 3.0
                                        } else {
                                            near[c]
                                        };
                                        errors[c][k] = (v - exact[c][i * dim + j]).abs();
                                    }
                                }
                                errors
                            },
                            &mut failures,
                        );
                    }
                }
                components::assert_no_failures(
                    "complete correlated contracted nuclear matrices",
                    &failures,
                );
            }
        }
    }
}
