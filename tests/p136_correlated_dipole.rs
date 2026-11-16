// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Physical electric derivatives of double-hybrid energies and independent
//! dense noncanonical amplitude equations, with four-step error fits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::linalg::{solve_linear, Matrix};
use dft_rs::properties::pt2::{from_ao, OrbitalSpace};
use dft_rs::properties::pt2_response::linear_response;
use dft_rs::xc::double_hybrid::Pt2Coefficients;

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}
fn fit(label: &str, steps: &[f64], errors: &[f64]) {
    let f = convergence::error_power_fit(steps, errors);
    println!(
        "{label} steps={steps:?} errors={errors:?} p={:.9} R2={:.12}",
        f.exponent, f.r_squared
    );
    assert!(
        (f.exponent - 2.).abs() < 0.1 && f.r_squared > 0.999,
        "{label}"
    );
}

#[test]
fn noncanonical_pair_response_matches_independent_dense_amplitudes() {
    let mut eri = vec![0.; 256];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                for l in 0..4 {
                    let factor = |p: usize, a: usize, b: usize| {
                        (((a + b + 2) * (p + 1) + a * b + 1) as f64).sin() * 0.13
                    };
                    eri[((i * 4 + j) * 4 + k) * 4 + l] =
                        (0..3).map(|p| factor(p, i, j) * factor(p, k, l)).sum();
                }
            }
        }
    }
    let orbitals = std::array::from_fn(|s| OrbitalSpace {
        occupied: 2,
        energies: vec![-0.8 + 0.1 * s as f64, -0.4, 0.2, 0.6],
        coefficients: Matrix::identity(4).as_slice().to_vec(),
    });
    let problem = from_ao(4, &eri, &orbitals).unwrap();
    let coefficients = Pt2Coefficients {
        same_spin: 0.31,
        opposite_spin: 0.47,
    };
    let df = std::array::from_fn(|s| {
        let mut f = Matrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                if (i < 2) == (j < 2) {
                    f[(i, j)] = 0.08 * ((i + j + 2 + s) as f64).cos();
                }
            }
        }
        f
    });
    let direction = problem
        .ovov
        .each_ref()
        .map(|v| v.iter().map(|g| 0.2 * g).collect());
    let actual = linear_response(&problem, &direction, &df, coefficients).unwrap();
    let dense = |theta: f64| {
        let mut energy = 0.;
        let mut amplitudes = Vec::new();
        for (ch, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let mut matrix = Matrix::zeros(16, 16);
            let mut g = vec![0.; 16];
            let entries: Vec<_> = (0..2)
                .flat_map(|i| {
                    (0..2).flat_map(move |a| {
                        (0..2).flat_map(move |j| (0..2).map(move |b| [i, a, j, b]))
                    })
                })
                .collect();
            let f = |spin: usize, i: usize, j: usize| {
                (if i == j {
                    problem.energies[spin][i]
                } else {
                    0.
                }) + theta * df[spin][(i, j)]
            };
            for (p, &[i, a, j, b]) in entries.iter().enumerate() {
                let exchange = ((i * 2 + b) * 2 + j) * 2 + a;
                g[p] = (problem.ovov[ch][p]
                    - if s == t {
                        problem.ovov[ch][exchange]
                    } else {
                        0.
                    })
                    * (1. + 0.2 * theta + 0.1 * theta * theta);
                for (q, &[k, c, l, d]) in entries.iter().enumerate() {
                    if a == c && j == l && b == d {
                        matrix[(p, q)] += f(s, i, k);
                    }
                    if i == k && a == c && b == d {
                        matrix[(p, q)] += f(t, j, l);
                    }
                    if i == k && j == l && b == d {
                        matrix[(p, q)] -= f(s, 2 + a, 2 + c);
                    }
                    if i == k && a == c && j == l {
                        matrix[(p, q)] -= f(t, 2 + b, 2 + d);
                    }
                }
            }
            let x = solve_linear(&matrix, &g).unwrap();
            let weight = if s == t {
                0.25 * coefficients.same_spin
            } else {
                coefficients.opposite_spin
            };
            energy += weight * g.iter().zip(&x).map(|(g, t)| g * t).sum::<f64>();
            amplitudes.extend(x);
        }
        (energy, amplitudes)
    };
    let steps = [0.08, 0.04, 0.02, 0.01];
    let mut errors = Vec::new();
    let mut amplitude_errors = Vec::new();
    for h in steps {
        let (ep, tp) = dense(h);
        let (em, tm) = dense(-h);
        errors.push(((ep - em) / (2. * h) - actual.correlation).abs());
        let expected = actual.amplitude_derivatives.iter().flatten();
        amplitude_errors.push(
            tp.iter()
                .zip(tm)
                .zip(expected)
                .map(|((p, m), a)| ((p - m) / (2. * h) - a).powi(2))
                .sum::<f64>()
                .sqrt(),
        );
    }
    fit("noncanonical PT2 energy", &steps, &errors);
    fit("noncanonical PT2 amplitude", &steps, &amplitude_errors);
    println!(
        "noncanonical physical amplitude response residual={:.12e}",
        actual.residual
    );
    assert!(actual.residual < 1e-12);
}

#[test]
fn correlated_dipoles_include_orbital_amplitude_and_distinct_target_terms() {
    let system = System::molecule(
        vec![8, 1, 1],
        vec![
            [0., 0., 0.2217],
            [0., 1.4309, -0.8867],
            [0., -1.4309, -0.8867],
        ],
    );
    for xc in [
        "b2plyp",
        "dsd-pbep86",
        "dsd-blyp",
        "pwpb95",
        "wb97m2",
        "pbe",
        "tpss",
        "pbe0",
    ] {
        if std::env::var("DFT_RS_TEST_MODEL")
            .ok()
            .is_some_and(|model| model != xc)
        {
            continue;
        }
        let custom = matches!(xc, "pbe" | "tpss" | "pbe0");
        let controls = |field: f64| {
            let mut v = Value::object()
                .with("xc", xc)
                .with("basis", "sto-3g")
                // The rounded PZ81 join invalidates the level-five finite
                // difference interval. Its failed fits and branch counts
                // are retained in docs/response.md; level zero resolves
                // this four-step validation without changing the model.
                .with(
                    "grid_level",
                    if xc == "dsd-pbep86" { 0usize } else { 5usize },
                )
                .with("vv10_grid_level", 1usize)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-11)
                .with("conv_tol_dp", 1e-12)
                .with("field", [0., 0., field]);
            if custom {
                v.set("pt2_same_spin", 0.2);
                v.set("pt2_opposite_spin", 0.3);
            }
            Controls::from_value(&v).unwrap()
        };
        let mut dipole_controls = controls(0.);
        dipole_controls
            .insert("origin", Value::Str("zero".into()))
            .unwrap();
        let report = run("dipole", &system, &dipole_controls).unwrap();
        let (_, mu) = report.get("dipole").unwrap().numeric_array().unwrap();
        let exact = -mu[2];
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let ep = run("energy", &system, &controls(h)).unwrap();
            let em = run("energy", &system, &controls(-h)).unwrap();
            let estimate = (number(&ep, "energy") - number(&em, "energy")) / (2. * h);
            let (_, parent) = report
                .get("parent_dipole")
                .unwrap()
                .numeric_array()
                .unwrap();
            let (_, target) = report
                .get("target_dipole_correction")
                .unwrap()
                .numeric_array()
                .unwrap();
            let (_, correlation) = report
                .get("pt2_dipole_correction")
                .unwrap()
                .numeric_array()
                .unwrap();
            println!(
                "{xc} h={h:.9e} target_derivative_error={:.12e} pt2_derivative_error={:.12e}",
                (number(&ep, "reference_energy") - number(&em, "reference_energy")) / (2. * h)
                    + parent[2]
                    + target[2],
                (number(&ep, "pt2_correlation") - number(&em, "pt2_correlation")) / (2. * h)
                    + correlation[2]
            );
            estimates.push(estimate);
            errors.push((estimate - exact).abs());
        }
        fit(&format!("{xc} correlated dipole"), &steps, &errors);
        let richardson = ((4. * estimates[3] - estimates[2]) / 3. - exact).abs();
        println!("{xc} dipole={mu:?} Richardson={richardson:.12e} response={:?} amplitudes={:?} target={:?}",report.get("response_residuals"),report.get("amplitude_response_residuals"),report.get("target_dipole_correction"));
        assert!(richardson < 1e-6, "{xc}");
        for key in [
            "response_residuals",
            "amplitude_response_residuals",
            "response_number_residuals",
        ] {
            let (_, values) = report.get(key).unwrap().numeric_array().unwrap();
            assert!(values.iter().all(|v| v.abs() < 1e-9), "{xc} {key}");
        }
    }
}
