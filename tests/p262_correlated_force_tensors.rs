// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete cubic/quartic Cartesian and three-pattern correlated tensors.
//! Every independent component of every parent/target/SS/OS/dispersion channel
//! is constrained by a complete unisolvent set of generic directions. The
//! oracle differentiates independently displaced SCF first derivatives only.
//! Pattern acceptance never substitutes for complete Cartesian acceptance.
//! Bounded HF-parent/PBE-target acceptance, 2026-10-26: 44,152 Cartesian
//! and pattern component/channel fits across RKS/UKS. The initial complete
//! quartic runs left three fits each; independently recomputed full tensors
//! and the retained resolved directions complete the evidence. There are
//! 44,055 second-order fits (p=1.800342..2.199896, minimum R2=0.99900149)
//! and 97 fourth-order-stencil fits (p=3.806484..4.133323). Every attempted
//! interval is retained; raw resolved errors and all intervals are in
//! docs/response.md. Other functional/reference combinations are separate.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/correlated_fixture.rs"]
mod correlated_fixture;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_cartesian_directions.rs"]
mod symmetric_directions;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::response::nuclear::{correlated_molecular_taylor, MolecularDirection};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::{double_hybrid::Pt2Coefficients, resolve, FunctionalSpec};
use std::collections::BTreeMap;

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
fn patterns(atoms: usize) -> Vec<Vec<[f64; 3]>> {
    (0..3)
        .map(|p| {
            (0..atoms)
                .map(|atom| {
                    std::array::from_fn(|axis| {
                        let a = (3 * atom + axis + 1) as f64;
                        0.51 * (0.7 * a * (p + 1) as f64).sin()
                            + 0.23 * (0.31 * (a + 1.0) * (p + 2) as f64).cos()
                    })
                })
                .collect()
        })
        .collect()
}
fn contract(tensor: &[f64], weights: &[f64], order: usize) -> f64 {
    tensor
        .iter()
        .enumerate()
        .map(|(mut index, &v)| {
            let mut product = v;
            for _ in 0..order {
                product *= weights[index % weights.len()];
                index /= weights.len();
            }
            product
        })
        .sum()
}
fn gradient_oracle(system: &System, options: &Value, d: &[f64], t: f64) -> [f64; 8] {
    let positions: Vec<_> = system
        .positions
        .iter()
        .enumerate()
        .map(|(a, r)| std::array::from_fn(|i| r[i] + t * d[3 * a + i]))
        .collect();
    let controls = Controls::from_value(options).unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let mut setup = prepare_molecular(&system.numbers, &positions, &method).unwrap();
    if setup.grid.is_none() {
        setup.grid =
            Some(dft_rs::grid::MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap());
    }
    let reference = run_molecular_reference(&setup).unwrap();
    assert!(reference.scf.converged);
    let target = options
        .get("double_hybrid_target_xc")
        .and_then(Value::as_str)
        .map(|name| resolve(&FunctionalSpec::Combined(name.into())).unwrap());
    let nuclear: Vec<_> = d.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
    let response = correlated_molecular_taylor(
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
        &MolecularDirection::nuclear(nuclear),
        1,
        &ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        },
    )
    .unwrap();
    let gradient = dft_rs::dispersion::dispersion_gradient(
        &setup.molecule,
        &setup.dispersion,
        &method.dispersion_functional_name(),
    )
    .unwrap();
    let dispersion: f64 = gradient
        .iter()
        .enumerate()
        .map(|(a, v)| (0..3).map(|i| v.to_array()[i] * d[3 * a + i]).sum::<f64>())
        .sum();
    [
        response.energy[1] + dispersion,
        response.parent[1],
        response.target[1],
        response.correlation[1],
        response.same_spin_alpha[1],
        response.opposite_spin[1],
        response.same_spin_beta[1],
        dispersion,
    ]
}

#[test]
fn complete_cartesian_and_pattern_tensors_have_all_independent_component_fits() {
    let model = std::env::var("DFT_RS_TEST_MODEL").unwrap_or_else(|_| "hf".into());
    let rank_filter = std::env::var("DFT_RS_TEST_TENSOR_RANK").ok();
    let direction_filter = std::env::var("DFT_RS_TEST_TENSOR_DIRECTIONS").ok();
    if rank_filter.is_some() || direction_filter.is_some() {
        println!("diagnostic subset: rank={rank_filter:?}, directions={direction_filter:?}; default acceptance covers every independent component");
    }
    for unrestricted in [false, true] {
        if std::env::var("DFT_RS_TEST_UNRESTRICTED")
            .ok()
            .is_some_and(|s| (s == "1") != unrestricted)
        {
            continue;
        }
        let (system, options) = correlated_fixture::fixture(&model, unrestricted, "d3bj");
        let atoms = system.numbers.len();
        for space in ["patterns", "cartesian"] {
            if std::env::var("DFT_RS_TEST_TENSOR_SPACE")
                .ok()
                .is_some_and(|s| s != space)
            {
                continue;
            }
            for order in [3usize, 4] {
                if rank_filter
                    .as_ref()
                    .is_some_and(|s| s != &order.to_string())
                {
                    continue;
                }
                let mut controls = options.clone();
                controls.set("response_order", order);
                if space == "patterns" {
                    controls.set("patterns", patterns(atoms));
                }
                println!("begin complete {model} U={unrestricted} {space} rank={order}");
                let report = run(
                    "force-constants",
                    &system,
                    &Controls::from_value(&controls).unwrap(),
                )
                .unwrap();
                if let Ok(path) = std::env::var("DFT_RS_TEST_REPORT_DIR") {
                    let path = std::path::PathBuf::from(path);
                    std::fs::create_dir_all(&path).unwrap();
                    std::fs::write(
                        path.join(format!(
                            "complete-{space}-{}-{order}.json",
                            if unrestricted { "uks" } else { "rks" }
                        )),
                        report.to_json(),
                    )
                    .unwrap();
                }
                assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
                assert_eq!(report.get("permutation_defect"), Some(&Value::Null));
                let (_, raw) = report.get("patterns").unwrap().numeric_array().unwrap();
                let basis: Vec<Vec<f64>> =
                    raw.chunks_exact(3 * atoms).map(|p| p.to_vec()).collect();
                if space == "cartesian" {
                    assert_eq!(basis.len(), 3 * atoms);
                    for (i, row) in basis.iter().enumerate() {
                        for (j, &value) in row.iter().enumerate() {
                            assert_eq!(value, if i == j { 1.0 } else { 0.0 });
                        }
                    }
                } else {
                    let expected: Vec<Vec<f64>> = patterns(atoms)
                        .iter()
                        .map(|p| p.iter().flatten().copied().collect())
                        .collect();
                    assert_eq!(basis, expected);
                }
                let size = basis.len().pow(order as u32);
                let tensors: Vec<_> = PREFIXES
                    .iter()
                    .map(|p| {
                        let values = report
                            .get(&format!("{p}tensor"))
                            .unwrap()
                            .numeric_array()
                            .unwrap()
                            .1;
                        assert_eq!(values.len(), size);
                        values
                    })
                    .collect();
                let mut failures = Vec::new();
                for (index, weights) in symmetric_directions::directions(basis.len(), order)
                    .iter()
                    .enumerate()
                {
                    if direction_filter
                        .as_ref()
                        .is_some_and(|s| !s.split(',').any(|v| v == index.to_string()))
                    {
                        continue;
                    }
                    let direction: Vec<f64> = (0..3 * atoms)
                        .map(|a| weights.iter().zip(&basis).map(|(w, p)| w * p[a]).sum())
                        .collect();
                    let exact: Vec<_> = tensors
                        .iter()
                        .map(|t| contract(t, weights, order))
                        .collect();
                    let mut cache = BTreeMap::<u64, [f64; 8]>::new();
                    let mut gradient = |t: f64| {
                        *cache
                            .entry(t.to_bits())
                            .or_insert_with(|| gradient_oracle(&system, &options, &direction, t))
                    };
                    let mut finite = |h: f64| {
                        let plus = gradient(h);
                        let minus = gradient(-h);
                        if order == 3 {
                            let center = gradient(0.0);
                            std::array::from_fn::<_, 8, _>(|c| {
                                (plus[c] - 2.0 * center[c] + minus[c]) / h.powi(2)
                            })
                        } else {
                            let plus2 = gradient(2.0 * h);
                            let minus2 = gradient(-2.0 * h);
                            std::array::from_fn(|c| {
                                stencils::Stencil::Second.first(
                                    plus2[c] - 2.0 * plus[c],
                                    minus2[c] - 2.0 * minus[c],
                                    0.0,
                                    0.0,
                                    h,
                                ) / h.powi(2)
                            })
                        }
                    };
                    let names: Vec<_> = PREFIXES.iter().map(|p| format!("{model} U={unrestricted} {space} rank={order} generic={index} {p}nuclear")).collect();
                    stencils::fit_components_staged(
                        &names,
                        &[
                            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                            ([0.16, 0.12, 0.08, 0.06], stencils::Stencil::Second),
                            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
                            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
                            ([0.01, 0.008, 0.006, 0.005], stencils::Stencil::Second),
                            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
                            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
                            ([0.12, 0.10, 0.08, 0.06], stencils::Stencil::Fourth),
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
                                    let value = if stencil == stencils::Stencil::Fourth {
                                        (4.0 * near[c] - far[c]) / 3.0
                                    } else {
                                        near[c]
                                    };
                                    errors[c][k] = (value - exact[c]).abs();
                                }
                            }
                            errors
                        },
                        &mut failures,
                    );
                }
                components::assert_no_failures("complete correlated nuclear tensor", &failures);
            }
        }
    }
}

#[test]
fn full_tensor_selection_refuses_incompatible_requests_before_calculation() {
    let (system, base) = correlated_fixture::fixture("hf", false, "none");
    for change in [
        Value::object().with("response_order", 2usize),
        Value::object().with("coordinates", Vec::<usize>::new()),
        Value::object().with("coordinates", vec![0usize, 0]),
        Value::object().with("coordinates", vec![100usize]),
        Value::object().with("patterns", Vec::<f64>::new()),
        Value::object()
            .with("patterns", patterns(system.numbers.len()))
            .with("coordinates", vec![0usize]),
    ] {
        let mut controls = base.clone();
        if let Value::Object(entries) = change {
            for (key, value) in entries {
                controls.set(&key, value);
            }
        }
        assert!(run(
            "force-constants",
            &system,
            &Controls::from_value(&controls).unwrap()
        )
        .is_err());
    }
}

#[test]
fn complete_cubic_blocks_and_patterns_match_all_cartesian_channels_in_caller_order() {
    for unrestricted in [false, true] {
        let (system, mut controls) = correlated_fixture::fixture("hf", unrestricted, "d3bj");
        controls.set("response_order", 3usize);
        let full = run(
            "force-constants",
            &system,
            &Controls::from_value(&controls).unwrap(),
        )
        .unwrap();
        assert_eq!(full.get("converged").and_then(Value::as_bool), Some(true));
        if let Ok(path) = std::env::var("DFT_RS_TEST_REPORT_DIR") {
            let path = std::path::PathBuf::from(path);
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(
                path.join(format!(
                    "complete-cubic-{}.json",
                    if unrestricted { "uks" } else { "rks" }
                )),
                full.to_json(),
            )
            .unwrap();
        }
        for key in [
            "translation_residual",
            "response_residuals",
            "amplitude_response_residuals",
            "amplitude_relative_residuals",
            "metric_defects",
        ] {
            let values = full.get(key).unwrap().numeric_array().unwrap().1;
            println!(
                "complete cubic U={unrestricted} {key} maximum {:.12e}",
                values.iter().map(|v| v.abs()).fold(0.0, f64::max)
            );
        }
        let coordinates = [8usize, 0, 4];
        let mut block_controls = controls.clone();
        block_controls.set("coordinates", coordinates);
        let block = run(
            "force-constants",
            &system,
            &Controls::from_value(&block_controls).unwrap(),
        )
        .unwrap();
        let supplied = patterns(system.numbers.len());
        let mut pattern_controls = controls.clone();
        pattern_controls.set("patterns", supplied.clone());
        let projected = run(
            "force-constants",
            &system,
            &Controls::from_value(&pattern_controls).unwrap(),
        )
        .unwrap();
        let dimension = 3 * system.numbers.len();
        for prefix in PREFIXES {
            let key = format!("{prefix}tensor");
            let complete = full.get(&key).unwrap().numeric_array().unwrap().1;
            let block = block.get(&key).unwrap().numeric_array().unwrap().1;
            let projected = projected.get(&key).unwrap().numeric_array().unwrap().1;
            for flat in 0..27 {
                let (a, b, c) = (flat / 9, (flat / 3) % 3, flat % 3);
                let direct = complete
                    [(coordinates[a] * dimension + coordinates[b]) * dimension + coordinates[c]];
                assert!((block[flat] - direct).abs() < 1e-9);
                let mut contraction = 0.0;
                for i in 0..dimension {
                    for j in 0..dimension {
                        for k in 0..dimension {
                            contraction += complete[(i * dimension + j) * dimension + k]
                                * supplied[a][i / 3][i % 3]
                                * supplied[b][j / 3][j % 3]
                                * supplied[c][k / 3][k % 3];
                        }
                    }
                }
                assert!((projected[flat] - contraction).abs() < 1e-9);
            }
        }
    }
}
