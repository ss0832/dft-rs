// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! DSD-PBEB95-D3(BJ): exact recipe, local derivatives, independent spin-PT2
//! and dispersion terms, and every C1 restricted/unrestricted force entry.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::xc::double_hybrid::{DoubleHybridFunctional, DOUBLE_HYBRID_NAMES};
use dft_rs::xc::{Family, GridVars, ResolvedFunctional, XcOut};
fn recipe() -> DoubleHybridFunctional {
    DoubleHybridFunctional::named("dsd-pbeb95").unwrap()
}
fn rows(v: &Value) -> &[Value] {
    let Value::Array(v) = v else {
        panic!("array expected")
    };
    v
}
fn number(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}
fn array(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}
fn oracle() -> Value {
    Value::parse_json(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/oracle_data/dsd_pbeb95.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
fn evaluate(f: &ResolvedFunctional, point: &[f64], order: usize) -> XcOut {
    let p: Vec<_> = point.iter().map(|v| [*v]).collect();
    let vars = GridVars {
        rho: [&p[0], &p[1]],
        sigma: Some([&p[2], &p[3], &p[4]]),
        tau: Some([&p[5], &p[6]]),
    };
    let mut out = XcOut::default();
    f.eval(&vars, order, &mut out).unwrap();
    out
}
fn system(entry: &Value) -> System {
    let numbers = entry.get("numbers").unwrap().numeric_array().unwrap().1;
    let Value::Array(rows) = entry.get("positions").unwrap() else {
        panic!("positions")
    };
    System::molecule(
        numbers.iter().map(|z| *z as u8).collect(),
        rows.iter()
            .map(|r| {
                let r = r.numeric_array().unwrap().1;
                [r[0], r[1], r[2]]
            })
            .collect(),
    )
}

#[test]
fn named_double_hybrids_match_independent_term_oracles() {
    let oracle = oracle();
    let molecules = oracle.get("molecules").unwrap();
    let Value::Array(cases) = oracle.get("cases").unwrap() else {
        panic!("cases")
    };
    assert!(!cases.is_empty());
    let mut worst = 0.0_f64;
    for case in cases {
        let model = case.get("model").and_then(Value::as_str).unwrap();
        let name = case.get("molecule").and_then(Value::as_str).unwrap();
        assert!(case.get("converged").and_then(Value::as_bool).unwrap());
        let spin = number(case, "spin");
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", model)
                .with("basis", "sto-3g")
                .with("grid_level", 9usize)
                .with("conv_tol", 1e-12)
                .with("conv_tol_grad", 1e-9)
                .with("charge", number(case, "charge") as i64)
                .with("multiplicity", spin as i64 + 1)
                .with("unrestricted", spin > 0.0),
        )
        .unwrap();
        let report = run("energy", &system(molecules.get(name).unwrap()), &controls).unwrap();
        let dispersion = number(case, "dispersion");
        let Value::Array(parts) = report.get("semilocal_components").unwrap() else {
            panic!("semilocal components")
        };
        let Value::Array(expected_parts) = case.get("parts").unwrap() else {
            panic!("oracle parts")
        };
        assert_eq!(parts.len(), expected_parts.len(), "{model} parts");
        let mut checks = vec![
            ("parent", number(&report, "orbital_reference_energy")),
            ("target", number(&report, "reference_energy") - dispersion),
            ("energy", number(&report, "energy")),
            ("one", number(&report, "one_electron_energy")),
            ("hartree", number(&report, "hartree_energy")),
            ("exchange", number(&report, "exact_exchange_energy")),
            (
                "same_spin",
                number(&report, "pt2_same_spin_alpha") + number(&report, "pt2_same_spin_beta"),
            ),
            ("opposite_spin", number(&report, "pt2_opposite_spin")),
            ("dispersion", number(&report, "model_dispersion_energy")),
        ];
        let part_values: Vec<f64> = parts.iter().map(|p| number(p, "energy")).collect();
        let tol = 5e-7;
        for (key, actual) in checks.drain(..) {
            let expected = number(case, key);
            let defect = (actual - expected).abs();
            worst = worst.max(defect);
            println!("{model} {name} {key} computed={actual:0.12e} oracle={expected:0.12e} defect={defect:0.3e}");
            assert!(defect < tol, "{model} {name} {key}: {defect:e}");
        }
        for (k, (actual, expected)) in part_values.iter().zip(expected_parts).enumerate() {
            let expected = expected.as_f64().unwrap();
            let defect = (actual - expected).abs();
            worst = worst.max(defect);
            println!("{model} {name} part{k} computed={actual:0.12e} oracle={expected:0.12e} defect={defect:0.3e}");
            assert!(defect < tol, "{model} {name} part {k}: {defect:e}");
        }
        for key in [
            "pt2_amplitude_residual",
            "semilocal_component_sum_residual",
            "reference_energy_sum_residual",
        ] {
            let raw = number(&report, key);
            println!("{model} {name} {key}={raw:0.3e}");
            assert!(raw < 1e-9, "{model} {name} {key}: {raw:e}");
        }
    }
    println!("named double hybrids: largest term defect {worst:0.3e}");
}

fn symmetric_third(output: &XcOut, i: usize, j: usize, k: usize) -> f64 {
    assert!(i <= j && j <= k && k < 7);
    let pair = |a, b, n| (0..a).map(|p| n - p).sum::<usize>() + b - a;
    let triple = |a, b, c, n| {
        let mut index = 0;
        for p in 0..n {
            for q in p..n {
                for r in q..n {
                    if (p, q, r) == (a, b, c) {
                        return index;
                    }
                    index += 1;
                }
            }
        }
        panic!("invalid symmetric triple")
    };
    let indices = [i, j, k];
    let rho = indices.iter().filter(|&&v| v < 2).count();
    let sigma = indices.iter().filter(|&&v| (2..5).contains(&v)).count();
    match (rho, sigma) {
        (3, 0) => output.v3rho3[i + j + k][0],
        (2, 1) => output.v3rho2sigma[3 * (i + j) + k - 2][0],
        (1, 2) => output.v3rhosigma2[6 * i + pair(j - 2, k - 2, 3)][0],
        (0, 3) => output.v3sigma3[triple(i - 2, j - 2, k - 2, 3)][0],
        (2, 0) => output.v3rho2tau[2 * (i + j) + k - 5][0],
        (1, 1) => output.v3rhosigmatau[6 * i + 2 * (j - 2) + k - 5][0],
        (1, 0) => output.v3rhotau2[3 * i + j + k - 10][0],
        (0, 2) => output.v3sigma2tau[2 * pair(i - 2, j - 2, 3) + k - 5][0],
        (0, 1) => output.v3sigmatau2[3 * (i - 2) + j + k - 10][0],
        (0, 0) => output.v3tau3[i + j + k - 15][0],
        _ => panic!("invalid derivative block"),
    }
}

#[test]
fn published_recipe_and_local_component_oracles_match() {
    let r = recipe();
    assert_eq!(r.parent.family, Family::MetaGga);
    assert_eq!(r.parent.hybrid.cx, 0.66);
    assert_eq!(r.parent.parts.len(), 2);
    assert_eq!(r.parent.parts[0].0, 0.34);
    assert_eq!(r.parent.parts[1].0, 0.55);
    assert_eq!(r.parent.parts[1].1.params(), &[0.038, 0.0031]);
    assert_eq!(r.correlation.same_spin, 0.09);
    assert_eq!(r.correlation.opposite_spin, 0.46);
    assert!(
        (r.parent.parts[1].0 + r.correlation.same_spin + r.correlation.opposite_spin - 1.10).abs()
            < 1e-15
    );
    let d = r.dispersion.unwrap();
    assert_eq!([d.s6, d.s8, d.a1, d.a2], [0.61, 0., 0., 6.2]);
    assert!(r.target.is_none() && r.target_vv10.is_none() && r.reference_vv10.is_none());
    assert_eq!(
        DOUBLE_HYBRID_NAMES
            .iter()
            .filter(|&&n| n == "dsd-pbeb95")
            .count(),
        1
    );
    for name in ["DSD-PBEB95", "dsd-pbeb95-d3bj", "dsd-pbeb95-2013"] {
        let alias = DoubleHybridFunctional::named(name).unwrap();
        assert_eq!(alias.correlation.opposite_spin, 0.46);
        assert_eq!(alias.parent.hybrid.cx, 0.66);
    }
    assert!(
        dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("dsd-pbeb95".into())).is_err()
    );
    let data = oracle();
    for (index, point) in rows(data.get("points").unwrap()).iter().enumerate() {
        let p = point.numeric_array().unwrap().1;
        for (label, f) in [
            ("parent", recipe().parent),
            (
                "exchange",
                ResolvedFunctional::custom(
                    "PBE",
                    vec![(1., r.parent.parts[0].1.clone())],
                    dft_rs::xc::HybridParams::PURE,
                    Family::Gga,
                ),
            ),
            (
                "correlation",
                ResolvedFunctional::custom(
                    "B95",
                    vec![(1., r.parent.parts[1].1.clone())],
                    dft_rs::xc::HybridParams::PURE,
                    Family::MetaGga,
                ),
            ),
        ] {
            let output = evaluate(&f, &p, if p[1] > 0. { 2 } else { 1 });
            let expected = &rows(data.get("components").unwrap().get(label).unwrap())[index];
            let e = (output.e[0] - number(expected, "energy")).abs();
            assert!(e < 1e-11, "{label} point{index} energy {e:e}");
            let v = [
                output.vrho[0][0],
                output.vrho[1][0],
                output.vsigma[0][0],
                output.vsigma[1][0],
                output.vsigma[2][0],
                if f.family == Family::MetaGga {
                    output.vtau[0][0]
                } else {
                    0.
                },
                if f.family == Family::MetaGga {
                    output.vtau[1][0]
                } else {
                    0.
                },
            ];
            for (i, expected) in array(expected, "potential").into_iter().enumerate() {
                if p[1] == 0. && [1, 3, 4, 6].contains(&i) {
                    continue;
                }
                assert!(
                    (v[i] - expected).abs() < 1e-10 * (1. + expected.abs()),
                    "{label} point{index} potential{i}"
                );
            }
            if let Some(k) = expected.get("kernel") {
                let (_, kernel) = output.potential_and_kernel(0, f.family).unwrap();
                for (i, row) in rows(k).iter().enumerate() {
                    for (j, value) in row.numeric_array().unwrap().1.iter().enumerate() {
                        assert!(
                            (kernel[i][j] - value).abs() < 1e-9 * (1. + value.abs()),
                            "{label} point{index} kernel{i},{j}: {} vs{value}",
                            kernel[i][j]
                        );
                    }
                }
            }
        }
    }
}

const ACTIVE: [usize; 6] = [0, 1, 2, 4, 5, 6];
fn active_partial(indices: &[usize]) -> bool {
    if indices.contains(&3) {
        return false;
    }
    let tau: Vec<_> = indices.iter().copied().filter(|&i| i >= 5).collect();
    if tau.is_empty() {
        return true;
    }
    if tau.len() > 1 {
        return false;
    }
    indices.iter().all(|&i| {
        i == tau[0]
            || if tau[0] == 5 {
                [0, 2].contains(&i)
            } else {
                [1, 4].contains(&i)
            }
    })
}

#[test]
fn every_active_parent_first_second_and_third_partial_has_four_step_fits() {
    let f = recipe().parent;
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let u = [0.23, -0.17, 0.11, 0.13, -0.19, 0.07];
    let output = evaluate(&f, &point, 3);
    let (potential, kernel) = output.potential_and_kernel(0, Family::MetaGga).unwrap();
    assert_eq!(potential[3], 0.);
    let mut targets = Vec::new();
    let mut labels = Vec::new();
    let mut zeros = 0;
    // Each target is a derivative of energy, potential, or kernel.
    for &i in &ACTIVE {
        targets.push((0, 0, 0, i, potential[i]));
        labels.push(format!("DSD-PBEB95 parent first{i}"));
    }
    for i in 0..7 {
        for j in i..7 {
            if active_partial(&[i, j]) {
                targets.push((1, i, 0, j, kernel[i][j]));
                labels.push(format!("DSD-PBEB95 parent second{i},{j}"));
            } else {
                assert_eq!(kernel[i][j], 0.);
                zeros += 1;
            }
            for k in j..7 {
                let actual = symmetric_third(&output, i, j, k);
                if active_partial(&[i, j, k]) {
                    targets.push((2, i, j, k, actual));
                    labels.push(format!("DSD-PBEB95 parent third{i},{j},{k}"));
                } else {
                    assert_eq!(actual, 0.);
                    zeros += 1;
                }
            }
        }
    }
    assert_eq!(targets.len(), 46);
    assert_eq!(zeros, 72);
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
            ([0.004, 0.002, 0.001, 0.0005], stencils::Stencil::Second),
            ([0.006, 0.003, 0.0015, 0.00075], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.; 4]; targets.len()];
            for (step, &h) in steps.iter().enumerate() {
                let mut directions = vec![[0.; 6]; targets.len()];
                for axis in 0..6 {
                    let at = |scale: f64| {
                        let mut p = point;
                        for j in 0..6 {
                            p[ACTIVE[j]] += scale * (u[j] + if axis == j { 1. } else { 0. });
                        }
                        let out = evaluate(&f, &p, 2);
                        let (v, k) = out.potential_and_kernel(0, Family::MetaGga).unwrap();
                        targets
                            .iter()
                            .map(|&(order, i, j, _, _)| match order {
                                0 => out.e[0],
                                1 => v[i],
                                _ => k[i][j],
                            })
                            .collect::<Vec<_>>()
                    };
                    let plus = at(h);
                    let minus = at(-h);
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (at(2. * h), at(-2. * h))
                    } else {
                        (vec![0.; targets.len()], vec![0.; targets.len()])
                    };
                    for target in 0..targets.len() {
                        directions[target][axis] = stencil.first(
                            plus[target],
                            minus[target],
                            plus2[target],
                            minus2[target],
                            h,
                        );
                    }
                }
                for (target, &(_, _, _, axis, actual)) in targets.iter().enumerate() {
                    let shared = (0..6).map(|j| u[j] * directions[target][j]).sum::<f64>()
                        / (1. + u.iter().sum::<f64>());
                    let axis = ACTIVE.iter().position(|&j| j == axis).unwrap();
                    errors[target][step] = (directions[target][axis] - shared - actual).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("DSD-PBEB95 complete local parent partials", &failures);
}

#[test]
fn spin_exchange_vacuum_and_zero_pt2_limits_hold() {
    let r = recipe();
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let permutation = [1, 0, 4, 3, 2, 6, 5];
    let swapped: Vec<_> = permutation.iter().map(|&i| point[i]).collect();
    let a = evaluate(&r.parent, &point, 2);
    let b = evaluate(&r.parent, &swapped, 2);
    assert!((a.e[0] - b.e[0]).abs() < 1e-14);
    let (va, ka) = a.potential_and_kernel(0, Family::MetaGga).unwrap();
    let (vb, kb) = b.potential_and_kernel(0, Family::MetaGga).unwrap();
    for i in 0..7 {
        assert!((va[i] - vb[permutation[i]]).abs() < 1e-13);
        for j in 0..7 {
            assert!((ka[i][j] - kb[permutation[i]][permutation[j]]).abs() < 1e-12);
        }
    }
    assert_eq!(evaluate(&r.parent, &[0.; 7], 2).e[0], 0.);
    assert!((r.parent.hybrid.cx + r.parent.parts[0].0 - 1.).abs() < 1e-15);
    let system = System::molecule(vec![1, 1], vec![[0., 0., -0.7], [0., 0., 0.7]]);
    let report = run(
        "energy",
        &system,
        &Controls::from_value(
            &Value::object()
                .with("xc", "dsd-pbeb95")
                .with("basis", "sto-3g")
                .with("pt2_same_spin", 0.)
                .with("pt2_opposite_spin", 0.)
                .with("dispersion", "none")
                .with("conv_tol", 1e-13),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(number(&report, "pt2_correlation"), 0.);
    assert!((number(&report, "energy") - number(&report, "reference_energy")).abs() < 1e-12);
}

#[test]
fn every_c1_restricted_and_unrestricted_gradient_has_four_step_fits() {
    let original = vec![
        [0.17, -0.11, 0.23],
        [1.43, 0.81, -0.62],
        [-0.63, 1.59, -0.43],
        [-1.12, -1.36, -0.71],
    ];
    for (charge, multiplicity) in [(0i64, 1i64), (1, 2)] {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "dsd-pbeb95")
                .with("basis", "sto-3g")
                .with("grid_level", 5usize)
                .with("grid_weight_cutoff", 0.)
                .with("charge", charge)
                .with("multiplicity", multiplicity)
                .with("unrestricted", multiplicity > 1)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-10)
                .with("conv_tol_dp", 1e-12)
                .with("max_iter", 500usize),
        )
        .unwrap();
        let system = |positions| System::molecule(vec![7, 1, 1, 1], positions);
        let report = run("gradient", &system(original.clone()), &controls).unwrap();
        let analytic = report.get("gradient").unwrap().numeric_array().unwrap().1;
        assert_eq!(analytic.len(), 12);
        let labels: Vec<_> = (0..12)
            .map(|i| format!("DSD-PBEB95 multiplicity{multiplicity} C1 gradient{i}"))
            .collect();
        let mut failures = Vec::new();
        components::fit_components(
            &labels,
            &[
                [0.04, 0.02, 0.01, 0.005],
                [0.02, 0.01, 0.005, 0.0025],
                [0.08, 0.04, 0.02, 0.01],
            ],
            |steps| {
                let mut errors = vec![[0.; 4]; 12];
                for (coordinate, &expected) in analytic.iter().enumerate() {
                    for (step, &h) in steps.iter().enumerate() {
                        let at = |sign: f64| {
                            let mut p = original.clone();
                            p[coordinate / 3][coordinate % 3] += sign * h;
                            number(&run("energy", &system(p), &controls).unwrap(), "energy")
                        };
                        errors[coordinate][step] = ((at(1.) - at(-1.)) / (2. * h) - expected).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures(
            "DSD-PBEB95 full restricted/unrestricted C1 forces",
            &failures,
        );
    }
}
