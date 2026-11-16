// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All nine constrained common-orbital susceptibilities: independent static
//! energy differences, zero-frequency matching and real/imaginary frequency
//! limits with four-step fits. The C1 geometry has no vanishing components.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::{run, Controls, System, Value};

fn system() -> System {
    System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.20, -1.31, -0.61],
        ],
    )
}
fn controls(xc: &str) -> Value {
    Value::object()
        .with("xc", xc)
        .with("basis", "sto-3g")
        .with("reference", "roks")
        .with("charge", 1i32)
        .with("multiplicity", 2usize)
        .with("grid_level", 0usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_grad", 1e-12)
        .with("conv_tol_dp", 1e-12)
        .with("response_tolerance", 1e-12)
        .with("max_iter", 300usize)
        .with("guess", "core")
}
fn evaluate(id: &str, system: &System, values: &Value) -> Value {
    run(id, system, &Controls::from_value(values).unwrap()).unwrap()
}
fn tensor(value: &Value, key: &str) -> Vec<f64> {
    value.get(key).unwrap().numeric_array().unwrap().1
}

fn validate(label: &str, system: &System, xc: &str, model: Option<&std::path::Path>) {
    let mut base = controls(xc);
    if let Some(path) = model {
        base.set("skala_model", path.to_str().unwrap());
    }
    let static_report = evaluate("polarizability", system, &base);
    let expected = tensor(&static_report, "polarizability");
    let paired = evaluate("dynamic-polarizability", system, &base);
    let actual = tensor(&paired, "alpha_real");
    let defect = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "{label} {xc} constrained static match {defect:.12e}, residuals {:?}",
        tensor(&paired, "residuals")
    );
    assert!(defect < 1e-9);
    let mut failures = Vec::new();
    for imaginary in [false, true] {
        let names = (0..9)
            .map(|i| {
                format!(
                    "{label} {xc} RO alpha({},{}) imaginary={imaginary}",
                    i / 3,
                    i % 3
                )
            })
            .collect::<Vec<_>>();
        components::fit_components(
            &names,
            &[
                [0.04, 0.03, 0.02, 0.015],
                [0.004, 0.003, 0.002, 0.0015],
                [0.001, 0.00075, 0.0005, 0.000375],
            ],
            |steps| {
                let mut errors = vec![[0.0; 4]; 9];
                for (k, h) in steps.iter().enumerate() {
                    let mut v = base.clone();
                    v.set("frequency", *h);
                    v.set("imaginary", imaginary);
                    let report = evaluate("dynamic-polarizability", system, &v);
                    let response = tensor(&report, "alpha_real");
                    for i in 0..9 {
                        errors[i][k] = (response[i] - actual[i]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    let intervals = [
        [0.016, 0.008, 0.004, 0.002],
        [0.008, 0.004, 0.002, 0.001],
        [0.032, 0.016, 0.008, 0.004],
        [0.008, 0.006, 0.004, 0.003],
        [0.0008, 0.0006, 0.0004, 0.0003],
        [0.0002, 0.00015, 0.0001, 0.000075],
    ];
    let energy = |a: usize, x: f64, b: usize, y: f64| {
        let mut v = base.clone();
        let mut f = [0.0; 3];
        f[a] += x;
        f[b] += y;
        v.set("field", f);
        evaluate("energy", system, &v)
            .get("energy")
            .unwrap()
            .as_f64()
            .unwrap()
    };
    let centre = energy(0, 0.0, 0, 0.0);
    for a in 0..3 {
        for b in a..3 {
            let mut names = vec![format!("{label} {xc} RO alpha({a},{b}) field-energy")];
            if a != b {
                names.push(format!("{label} {xc} RO alpha({b},{a}) field-energy"));
            }
            components::fit_components(
                &names,
                &intervals,
                |steps| {
                    let mut errors = vec![[0.0; 4]; names.len()];
                    for (k, &h) in steps.iter().enumerate() {
                        let fd = if a == b {
                            -(energy(a, h, b, 0.0) + energy(a, -h, b, 0.0) - 2.0 * centre) / (h * h)
                        } else {
                            -(energy(a, h, b, h) - energy(a, h, b, -h) - energy(a, -h, b, h)
                                + energy(a, -h, b, -h))
                                / (4.0 * h * h)
                        };
                        errors[0][k] = (fd - actual[3 * a + b]).abs();
                        if a != b {
                            errors[1][k] = (fd - actual[3 * b + a]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    // A small h^2 coefficient can cancel its h^4 correction above the SCF
    // cancellation range. Keep every failed interval and resolve such
    // components with an independent fourth-order energy stencil.
    for a in 0..3 {
        for b in a..3 {
            let names = if a == b {
                vec![format!("{label} {xc} RO alpha({a},{b}) field-energy")]
            } else {
                vec![
                    format!("{label} {xc} RO alpha({a},{b}) field-energy"),
                    format!("{label} {xc} RO alpha({b},{a}) field-energy"),
                ]
            };
            if !names.iter().any(|name| failures.contains(name)) {
                continue;
            }
            let steps = [0.016, 0.012, 0.008, 0.006];
            let mut errors = vec![[0.0; 4]; names.len()];
            for (k, &h) in steps.iter().enumerate() {
                let fd = if a == b {
                    (energy(a, 2.0 * h, b, 0.0) - 16.0 * energy(a, h, b, 0.0) + 30.0 * centre
                        - 16.0 * energy(a, -h, b, 0.0)
                        + energy(a, -2.0 * h, b, 0.0))
                        / (12.0 * h * h)
                } else {
                    let mut second = 0.0;
                    for (x, wx) in [
                        (1.0, 2.0 / 3.0),
                        (-1.0, -2.0 / 3.0),
                        (2.0, -1.0 / 12.0),
                        (-2.0, 1.0 / 12.0),
                    ] {
                        for (y, wy) in [
                            (1.0, 2.0 / 3.0),
                            (-1.0, -2.0 / 3.0),
                            (2.0, -1.0 / 12.0),
                            (-2.0, 1.0 / 12.0),
                        ] {
                            second -= wx * wy * energy(a, x * h, b, y * h) / (h * h);
                        }
                    }
                    second
                };
                errors[0][k] = (fd - actual[3 * a + b]).abs();
                if a != b {
                    errors[1][k] = (fd - actual[3 * b + a]).abs();
                }
            }
            for (i, name) in names.iter().enumerate() {
                if failures.contains(name)
                    && components::component_fit_of_order(
                        &format!("{name} fourth-order stencil"),
                        &steps,
                        &errors[i],
                        4.0,
                    )
                {
                    failures.retain(|failed| failed != name);
                }
            }
        }
    }
    components::assert_no_failures(&format!("{xc} constrained susceptibility"), &failures);
    let mut v = base.clone();
    v.set("frequency", 0.11);
    v.set("frequency_damping", 0.02);
    let positive = evaluate("dynamic-polarizability", system, &v);
    v.set("frequency", -0.11);
    let negative = evaluate("dynamic-polarizability", system, &v);
    assert!(tensor(&positive, "alpha_real")
        .iter()
        .zip(tensor(&negative, "alpha_real"))
        .all(|(a, b)| (a - b).abs() < 1e-9));
    assert!(tensor(&positive, "alpha_imag")
        .iter()
        .zip(tensor(&negative, "alpha_imag"))
        .all(|(a, b)| (a + b).abs() < 1e-9));
}

#[test]
fn common_orbital_all_components_static_and_causal_frequency_limits() {
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0"] {
        if std::env::var("DFT_RS_TEST_MODEL")
            .ok()
            .is_some_and(|v| v != xc)
        {
            continue;
        }
        validate("ammonia cation", &system(), xc, None);
    }
}
#[test]
fn soft_hydrogen_cation_retains_the_failed_intervals_and_resolves_them() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
    );
    validate("soft hydrogen cation", &system, "hf", None);
}

#[test]
fn one_electron_common_orbital_dynamics_equals_unrestricted_dynamics() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
    );
    for xc in ["hf", "pbe0"] {
        for (frequency, damping, imaginary) in
            [(0.0, 0.0, false), (0.07, 0.01, false), (0.13, 0.0, true)]
        {
            let mut v = controls(xc);
            v.set("charge", 3i32);
            v.set("frequency", frequency);
            v.set("frequency_damping", damping);
            v.set("imaginary", imaginary);
            let constrained = evaluate("dynamic-polarizability", &system, &v);
            v.set("reference", "uks");
            let unrestricted = evaluate("dynamic-polarizability", &system, &v);
            for key in ["alpha_real", "alpha_imag"] {
                let defect = tensor(&constrained, key)
                    .iter()
                    .zip(tensor(&unrestricted, key))
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                println!(
                    "{xc} one-electron {key} at {frequency} imaginary={imaginary}: {defect:.12e}"
                );
                assert!(defect < 1e-9);
            }
        }
    }
}

#[test]
fn learned_common_orbital_reference_uses_the_generic_learned_hessian() {
    use dft_rs::skala::{random_tensors, write_tensors, SkalaSizes};
    let path = std::env::temp_dir().join(format!(
        "dft_rs_common_orbital_{}.model",
        std::process::id()
    ));
    write_tensors(
        &path,
        &random_tensors(
            7,
            SkalaSizes {
                width: 6,
                channels: 2,
                shells: 1,
                correlation: 2,
                layers: 1,
                post: 2,
            },
        ),
    )
    .unwrap();
    validate("learned ammonia cation", &system(), "skala", Some(&path));
    std::fs::remove_file(path).unwrap();
}
