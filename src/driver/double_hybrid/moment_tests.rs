// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::{calculate_prepared, prepare_with_core};
use crate::acceptance::{components, stencils};
use crate::driver::{run, Controls, System, Value};
use crate::linalg::Matrix;
use stencils::Stencil;

pub(in crate::driver) const ORIGIN: [f64; 3] = [0.23, -0.50, 0.42];
pub(in crate::driver) const EXPONENT: f64 = 0.8;
pub(in crate::driver) const COMPONENTS: [(usize, usize); 6] =
    [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).unwrap().as_f64().unwrap()
}

fn tensor(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

pub(in crate::driver) fn system(unrestricted: bool) -> System {
    let positions = [
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
        [-1.25, 0.39, 1.03],
        [1.36, 2.45, 1.38],
    ];
    let atoms = if unrestricted { 5 } else { 6 };
    System::molecule(vec![1; atoms], positions[..atoms].to_vec())
}

/// Normalized equal-exponent Gaussian products independently supply
/// the applied monomial potential. Its ionic term is supplied separately.
pub(in crate::driver) fn perturbation(system: &System, a: usize, b: usize) -> (Matrix, f64) {
    let n = system.positions.len();
    let mut operator = Matrix::zeros(n, n);
    for (i, ri) in system.positions.iter().enumerate() {
        for (j, rj) in system.positions.iter().enumerate() {
            let center = |axis| 0.5 * (ri[axis] + rj[axis]) - ORIGIN[axis];
            let distance2: f64 = (0..3).map(|axis| (ri[axis] - rj[axis]).powi(2)).sum();
            let variance = if a == b { 0.25 / EXPONENT } else { 0.0 };
            operator[(i, j)] =
                (-0.5 * EXPONENT * distance2).exp() * (center(a) * center(b) + variance);
        }
    }
    let ionic = system
        .positions
        .iter()
        .map(|r| (r[a] - ORIGIN[a]) * (r[b] - ORIGIN[b]))
        .sum();
    (operator, ionic)
}

pub(in crate::driver) fn correlated_controls(
    model: &str,
    unrestricted: bool,
    basis: &str,
) -> Value {
    let mut controls = Value::object()
        .with("basis", basis)
        .with("guess", "core")
        .with("grid_level", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("field", [0.0012, -0.0017, 0.0008])
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-11)
        .with("max_iter", 300)
        .with("multiplicity", if unrestricted { 2 } else { 1 })
        .with("unrestricted", unrestricted);
    if model == "custom" {
        controls.set("xc", "hf");
        controls.set("double_hybrid_target_xc", "svwn5");
        controls.set("pt2_same_spin", 0.28);
        controls.set("pt2_opposite_spin", 0.49);
    } else {
        controls.set("xc", model);
    }
    controls
}

fn fit_quadrupole(model: &str, unrestricted: bool, basis: &str) {
    let system = system(unrestricted);
    let controls = correlated_controls(model, unrestricted, basis);
    let moment_controls = controls.clone().with("origin", ORIGIN).with("max_rank", 2);
    let report = run(
        "molecular-multipole",
        &system,
        &Controls::from_value(&moment_controls).unwrap(),
    )
    .unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let ranks = report.get("ranks").unwrap().as_array().unwrap();
    assert_eq!(ranks.len(), 3);
    let rank = &ranks[2];
    let ss = number(&report, "pt2_same_spin_coefficient");
    let os = number(&report, "pt2_opposite_spin_coefficient");
    let mut channels = vec![
        ("cartesian", tensor(rank, "cartesian")),
        ("parent_cartesian", tensor(rank, "parent_cartesian")),
        (
            "pt2_cartesian_correction",
            tensor(rank, "pt2_cartesian_correction"),
        ),
        ("pt2_same_spin_alpha", tensor(rank, "pt2_same_spin_alpha")),
        ("pt2_same_spin_beta", tensor(rank, "pt2_same_spin_beta")),
        ("pt2_opposite_spin", tensor(rank, "pt2_opposite_spin")),
    ];
    let target = tensor(rank, "target_cartesian_correction");
    for i in 0..6 {
        assert!(
            (channels[2].1[i] - ss * (channels[3].1[i] + channels[4].1[i]) - os * channels[5].1[i])
                .abs()
                < 1e-12
        );
        assert!((channels[0].1[i] - channels[1].1[i] - channels[2].1[i] - target[i]).abs() < 1e-12);
    }
    if model == "custom" {
        channels.push(("target_cartesian_correction", target));
    } else {
        assert_eq!(target, vec![0.0; 6]);
    }
    let reference = if unrestricted { "uks" } else { "rks" };
    println!(
        "{model} {reference} quadrupole {} orbital={:.12e}",
        rank.to_json(),
        number(&report, "reference_orbital_residual")
    );
    let names = COMPONENTS
        .iter()
        .flat_map(|&(a, b)| {
            channels
                .iter()
                .map(move |(key, _)| format!("{model} {reference} {key}[{a},{b}]"))
        })
        .collect::<Vec<_>>();
    let stages = [
        ([0.004, 0.003, 0.002, 0.0015], Stencil::Second),
        ([0.002, 0.0015, 0.001, 0.00075], Stencil::Second),
        ([0.008, 0.006, 0.004, 0.003], Stencil::Second),
        ([0.004, 0.003, 0.002, 0.0015], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    let mut cache = std::collections::BTreeMap::new();
    stencils::fit_components_staged(
        &names,
        &stages,
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (index, &(a, b)) in COMPONENTS.iter().enumerate() {
                let (operator, ionic) = perturbation(&system, a, b);
                let mut sample = |step: f64| {
                    *cache.entry((index, step.to_bits())).or_insert_with(|| {
                        let request = Controls::from_value(&controls).unwrap();
                        let state = prepare_with_core(&system, &request, "energy", |_| {
                            let mut core = operator.clone();
                            for value in core.as_mut_slice() {
                                *value *= step;
                            }
                            Some(core)
                        })
                        .unwrap();
                        let sample = calculate_prepared(state, "energy").unwrap();
                        assert_eq!(sample.get("converged").and_then(Value::as_bool), Some(true));
                        let parent = number(&sample, "orbital_reference_energy");
                        [
                            number(&sample, "energy") - step * ionic,
                            parent - step * ionic,
                            number(&sample, "pt2_correlation"),
                            number(&sample, "pt2_same_spin_alpha"),
                            number(&sample, "pt2_same_spin_beta"),
                            number(&sample, "pt2_opposite_spin"),
                            number(&sample, "reference_energy") - parent,
                        ]
                    })
                };
                for (step_index, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (sample(h), sample(-h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (sample(2.0 * h), sample(-2.0 * h))
                    } else {
                        ([0.0; 7], [0.0; 7])
                    };
                    for (channel, (_, analytic)) in channels.iter().enumerate() {
                        let measured = -stencil.first(
                            plus[channel],
                            minus[channel],
                            plus2[channel],
                            minus2[channel],
                            h,
                        );
                        errors[index * channels.len() + channel][step_index] =
                            (measured - analytic[index]).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(model, &failures);
}

#[test]
fn correlated_quadrupole_components_and_spin_channels_pass_independent_fits() {
    let basis =
        std::env::temp_dir().join(format!("dft-rs-pt2-quadrupole-{}.g94", std::process::id()));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    for model in ["b2plyp", "custom"] {
        for unrestricted in [false, true] {
            fit_quadrupole(model, unrestricted, basis.to_str().unwrap());
        }
    }
    std::fs::remove_file(basis).unwrap();
}
