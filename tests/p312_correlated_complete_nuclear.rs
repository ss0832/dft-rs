// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete C1 restricted cubic/quartic acceptance with responding PT2.
//! Contract the actual common-driver Cartesian tensors along a complete
//! unisolvent set, then fit against separately displaced SCF directional
//! Hessians. All seven total/parent/target/SS/OS channels are independent
//! acceptance entries. Default coverage is 48,412 fits across four parents.
//! Numerical acceptance is pending until every direction and raw residual
//! passes; diagnostic filters never certify the complete tensor.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/correlated_fixture.rs"]
mod correlated_fixture;
#[path = "support/symmetric_cartesian_directions.rs"]
mod symmetric_directions;

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::response::nuclear::{
    correlated_molecular_taylor, MolecularCorrelatedTaylor, MolecularDirection,
};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::double_hybrid::Pt2Coefficients;
use std::collections::BTreeMap;

const MODELS: [&str; 4] = ["hf", "pbe", "tpss", "pbe0"];
const PREFIXES: [&str; 7] = [
    "",
    "parent_",
    "target_",
    "pt2_",
    "pt2_same_spin_alpha_",
    "pt2_opposite_spin_",
    "pt2_same_spin_beta_",
];
const TOLERANCE: f64 = 1e-12;
const SECOND_ORDER_INTERVALS: [[f64; 4]; 5] = [
    [0.08, 0.06, 0.04, 0.03],
    [0.02, 0.015, 0.01, 0.0075],
    [0.16, 0.12, 0.08, 0.06],
    [0.04, 0.03, 0.02, 0.015],
    [0.01, 0.008, 0.006, 0.005],
];

fn fixture(model: &str) -> (System, Value) {
    let (system, mut options) = correlated_fixture::fixture(model, false, "none");
    assert_c1_fixture(&system);
    if let Value::Object(entries) = &mut options {
        entries.retain(|(key, _)| {
            !matches!(key.as_str(), "reference_field" | "double_hybrid_target_xc")
        });
    }
    options.set("double_hybrid_model", "custom");
    options.set("pt2_same_spin", 0.37);
    options.set("pt2_opposite_spin", 0.61);
    options.set("conv_tol", 1e-13);
    options.set("conv_tol_grad", 1e-12);
    options.set("conv_tol_dp", 1e-12);
    options.set("max_iter", 300usize);
    (system, options)
}

fn maximum(values: &[f64]) -> f64 {
    assert!(
        !values.is_empty(),
        "Required response diagnostics are absent"
    );
    assert!(values.iter().all(|v| v.is_finite()));
    values.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

fn assert_c1_fixture(system: &System) {
    assert_eq!(system.numbers, vec![1; 4]);
    assert_eq!(system.positions.len(), 4);
    let origin = system.positions[0];
    let edges: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| system.positions[i + 1][j] - origin[j]));
    let [a, b, c] = edges;
    let volume = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    assert!(volume.is_finite() && volume != 0.0);
    let mut distances = Vec::new();
    for (i, left) in system.positions.iter().enumerate() {
        for right in system.positions.iter().skip(i + 1) {
            let squared: f64 = left.iter().zip(right).map(|(x, y)| (x - y).powi(2)).sum();
            assert!(squared.is_finite() && squared > 0.0);
            distances.push(squared);
        }
    }
    distances.sort_by(f64::total_cmp);
    assert_eq!(distances.len(), 6);
    assert!(distances.windows(2).all(|pair| pair[0] < pair[1]));
    println!("fixture C1 signed_six_volume={volume:.16e} squared_pair_distances={distances:?}");
}

fn check_tensor_permutations(tensor: &[f64], dimension: usize, rank: usize, label: &str) {
    assert_eq!(tensor.len(), dimension.pow(rank as u32));
    let mut largest = 0.0_f64;
    for (index, &value) in tensor.iter().enumerate() {
        assert!(value.is_finite());
        let mut remainder = index;
        let mut indices = vec![0; rank];
        for coordinate in &mut indices {
            *coordinate = remainder % dimension;
            remainder /= dimension;
        }
        indices.sort_unstable();
        let canonical = indices.iter().fold(0usize, |flat, &i| flat * dimension + i);
        largest = largest.max((value - tensor[canonical]).abs());
    }
    assert!(
        largest <= TOLERANCE,
        "{label} permutation defect={largest:.16e}"
    );
    println!("tensor-permutation {label} maximum={largest:.16e}");
}

fn check_path(value: &MolecularCorrelatedTaylor, label: &str) {
    for (name, defect) in [
        ("response", maximum(&value.response_residuals)),
        ("amplitude", maximum(&value.amplitude_residuals)),
        (
            "amplitude_relative",
            maximum(&value.amplitude_relative_residuals),
        ),
        ("projector", maximum(&value.projector_defects)),
        ("stationarity", maximum(&value.stationarity_defects)),
        ("metric", maximum(&value.metric_defects)),
        (
            "number",
            maximum(
                &value
                    .number_defects
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
            ),
        ),
        ("frame", value.orbital_frame_defect),
        ("reference_energy", value.reference_energy_defect),
        ("reference_fock", value.reference_fock_defect),
    ] {
        assert!(
            defect.is_finite() && defect <= TOLERANCE,
            "{label} {name}={defect:.16e}"
        );
        println!("sample {label} {name}={defect:.16e}");
    }
    let equations = value.coefficient_equations.as_ref().unwrap();
    assert_eq!(equations.len(), 2);
    for (order, equation) in equations.iter().enumerate() {
        assert!(equation.source_norm.is_finite());
        assert!(equation.reconstructed_norm.is_finite());
        assert!(equation.reconstructed_relative_residual.is_finite());
        for residual in [
            equation.linear_relative_residual,
            equation.inner_linear_relative_residual,
        ] {
            assert!(
                residual.is_finite() && residual <= TOLERANCE,
                "{label} order={} linear residual={residual:.16e}",
                order + 1
            );
        }
        println!("sample {label} equation_order={} source={:.16e} rebuilt={:.16e} ratio={:.16e} returned={:.16e} inner={:.16e}", order + 1,
            equation.source_norm, equation.reconstructed_norm,
            equation.reconstructed_relative_residual, equation.linear_relative_residual,
            equation.inner_linear_relative_residual);
    }
}

fn hessian_oracle(
    system: &System,
    options: &Value,
    direction: &[f64],
    offset: f64,
    label: &str,
) -> [f64; 7] {
    let positions: Vec<_> = system
        .positions
        .iter()
        .enumerate()
        .map(|(a, r)| std::array::from_fn(|axis| r[axis] + offset * direction[3 * a + axis]))
        .collect();
    let method = MolecularMethod::from_controls(&Controls::from_value(options).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &positions, &method).unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    assert!(
        reference.scf.converged,
        "{label} offset={offset:.16e} SCF did not converge"
    );
    let path = correlated_molecular_taylor(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        None,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        Pt2Coefficients {
            same_spin: 0.37,
            opposite_spin: 0.61,
        },
        &MolecularDirection::nuclear(
            direction
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
        ),
        2,
        &ResponseSolverConfig {
            tolerance: TOLERANCE,
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("{label} offset={offset:.16e}: {error}"));
    check_path(&path, &format!("{label} offset={offset:.16e}"));
    let values = [
        path.energy[2],
        path.parent[2],
        path.target[2],
        path.correlation[2],
        path.same_spin_alpha[2],
        path.opposite_spin[2],
        path.same_spin_beta[2],
    ]
    .map(|coefficient| 2.0 * coefficient);
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} offset={offset:.16e} contains a nonfinite Hessian channel"
    );
    println!("hessian-sample {label} offset={offset:.16e} values={values:?}");
    values
}

fn contract(tensor: &[f64], direction: &[f64], rank: usize) -> f64 {
    assert_eq!(tensor.len(), direction.len().pow(rank as u32));
    tensor
        .iter()
        .enumerate()
        .map(|(mut index, &value)| {
            let mut weight = 1.0;
            for _ in 0..rank {
                weight *= direction[index % direction.len()];
                index /= direction.len();
            }
            value * weight
        })
        .sum()
}

#[test]
fn every_restricted_correlated_cubic_and_quartic_component_has_an_independent_fit() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    let model_filter = std::env::var("DFT_RS_TEST_MODEL").ok();
    let rank_filter = std::env::var("DFT_RS_TEST_TENSOR_RANK").ok();
    let direction_filter = std::env::var("DFT_RS_TEST_TENSOR_DIRECTIONS").ok();
    assert!(model_filter
        .as_ref()
        .is_none_or(|s| MODELS.contains(&s.as_str())));
    assert!(rank_filter
        .as_ref()
        .is_none_or(|s| matches!(s.as_str(), "3" | "4")));
    let selected_directions = direction_filter.map(|s| {
        let mut ids: Vec<_> = s.split(',').map(|i| i.parse::<usize>().unwrap()).collect();
        ids.sort_unstable();
        assert!(!ids.is_empty() && ids.windows(2).all(|w| w[0] != w[1]));
        ids
    });
    if model_filter.is_some() || rank_filter.is_some() || selected_directions.is_some() {
        println!("diagnostic selection model={model_filter:?} rank={rank_filter:?} directions={selected_directions:?}; complete acceptance requires the union of all four parents, both ranks and every direction");
    }
    for model in MODELS {
        if model_filter.as_ref().is_some_and(|s| s != model) {
            continue;
        }
        let (system, options) = fixture(model);
        for rank in [3usize, 4] {
            if rank_filter.as_ref().is_some_and(|s| s != &rank.to_string()) {
                continue;
            }
            let directions = symmetric_directions::directions(12, rank);
            assert_eq!(directions.len(), if rank == 3 { 364 } else { 1365 });
            assert!(selected_directions
                .as_ref()
                .is_none_or(|ids| ids.iter().all(|&i| i < directions.len())));
            let mut controls = options.clone();
            controls.set("response_order", rank);
            println!(
                "begin complete-correlated {model} rank={rank} independent={} channels=7",
                directions.len()
            );
            let report = run(
                "force-constants",
                &system,
                &Controls::from_value(&controls).unwrap(),
            )
            .unwrap();
            assert_eq!(
                report.get("converged").and_then(Value::as_bool),
                Some(true),
                "{}",
                report.to_json_pretty()
            );
            assert_eq!(
                report.get("independent_components").unwrap().as_i64(),
                Some(directions.len() as i64)
            );
            for name in [
                "response_residuals",
                "amplitude_response_residuals",
                "amplitude_relative_residuals",
                "projector_defects",
                "nuclear_stationarity_defects",
                "metric_defects",
                "response_number_defects",
                "orbital_frame_defects",
                "reference_defects",
            ] {
                let residual = maximum(&report.get(name).unwrap().numeric_array().unwrap().1);
                println!("complete-correlated {model} rank={rank} {name}={residual:.16e}");
                assert!(
                    residual <= TOLERANCE,
                    "{model} rank={rank} {name}={residual:.16e}"
                );
            }
            if let Ok(directory) = std::env::var("DFT_RS_TEST_REPORT_DIR") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(
                    directory.join(format!("complete-correlated-{model}-{rank}.json")),
                    report.to_json(),
                )
                .unwrap();
            }
            let tensors: Vec<_> = PREFIXES
                .iter()
                .map(|prefix| {
                    report
                        .get(&format!("{prefix}tensor"))
                        .unwrap()
                        .numeric_array()
                        .unwrap()
                        .1
                })
                .collect();
            for (prefix, tensor) in PREFIXES.iter().zip(&tensors) {
                check_tensor_permutations(
                    tensor,
                    12,
                    rank,
                    &format!("{model} rank={rank} {prefix}nuclear"),
                );
            }
            let mut failures = Vec::new();
            for (index, direction) in directions.iter().enumerate() {
                if selected_directions
                    .as_ref()
                    .is_some_and(|ids| !ids.contains(&index))
                {
                    continue;
                }
                let label = format!("{model} complete-correlated rank={rank} generic={index}");
                let exact: Vec<_> = tensors
                    .iter()
                    .map(|tensor| contract(tensor, direction, rank))
                    .collect();
                assert!(exact.iter().all(|value| value.is_finite()));
                println!("analytic {label} direction={direction:?} values={exact:?}");
                let mut cache = BTreeMap::<u64, [f64; 7]>::new();
                let mut sample = |offset: f64| {
                    *cache.entry(offset.to_bits()).or_insert_with(|| {
                        hessian_oracle(&system, &options, direction, offset, &label)
                    })
                };
                let mut finite = |h: f64| {
                    let plus = sample(h);
                    let minus = sample(-h);
                    let center = if rank == 4 { sample(0.0) } else { [0.0; 7] };
                    std::array::from_fn::<_, 7, _>(|channel| {
                        if rank == 3 {
                            (plus[channel] - minus[channel]) / (2.0 * h)
                        } else {
                            (plus[channel] - 2.0 * center[channel] + minus[channel]) / h.powi(2)
                        }
                    })
                };
                let names: Vec<_> = PREFIXES
                    .iter()
                    .map(|prefix| format!("{label} {prefix}nuclear"))
                    .collect();
                components::fit_components(
                    &names,
                    &SECOND_ORDER_INTERVALS,
                    |steps| {
                        let mut errors = vec![[0.0; 4]; 7];
                        for (k, &h) in steps.iter().enumerate() {
                            let measured = finite(h);
                            for channel in 0..7 {
                                errors[channel][k] = (measured[channel] - exact[channel]).abs();
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
            }
            components::assert_no_failures(
                &format!("{model} complete correlated rank={rank}"),
                &failures,
            );
            println!("end complete-correlated {model} rank={rank} all selected fits passed");
        }
    }
}
