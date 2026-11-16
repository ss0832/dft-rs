// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::electric_tests::fixture;
use super::*;
use crate::acceptance::{components, stencils::Stencil};

fn numbers(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn scalar(report: &Value, key: &str) -> f64 {
    report.get(key).unwrap().as_f64().unwrap()
}

fn quadrupole(report: &Value) -> &Value {
    report.get("electric_quadrupole").unwrap()
}

fn moments(report: &Value) -> [f64; 8] {
    let dipole = numbers(report, "dipole");
    let moment = numbers(quadrupole(report), "moment");
    std::array::from_fn(|a| {
        if a < 3 {
            dipole[a]
        } else {
            moment[a - 3] / 3.0
        }
    })
}

fn susceptibility(report: &Value, part: &str) -> [[f64; 8]; 8] {
    let q = quadrupole(report);
    let alpha = numbers(
        report.get("polarizability").unwrap(),
        &format!("tensor_{part}"),
    );
    let dipole_gradient = numbers(q, &format!("dipole_from_field_gradient_{part}"));
    let gradient_dipole = numbers(q, &format!("field_conjugate_moment_from_field_{part}"));
    let field_gradient = numbers(q, &format!("polarizability_{part}"));
    std::array::from_fn(|a| {
        std::array::from_fn(|b| match (a < 3, b < 3) {
            (true, true) => alpha[3 * a + b],
            (true, false) => dipole_gradient[5 * a + b - 3],
            (false, true) => gradient_dipole[3 * (a - 3) + b],
            (false, false) => field_gradient[5 * (a - 3) + b - 3] / 3.0,
        })
    })
}

#[test]
fn public_joint_dirac_quadrupole_fits_all_components() {
    let (system, path, options) = fixture("coulomb", "quadrupole");
    let origin = [0.14, -0.18, 0.23];
    let bias = [0.013, -0.009, 0.007, 0.004, -0.006, 0.003, -0.005, 0.002];
    let solve = |input: Value| {
        let report = crate::driver::run(
            "multicomponent",
            &system,
            &Controls::from_value(&input).unwrap(),
        )
        .unwrap_or_else(|error| panic!("joint quadrupole response {input:?}: {error}"));
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        let problems = crate::driver::units::complete(&mut report.clone());
        assert!(problems.is_empty(), "{problems:?}");
        report
    };
    let at_field = |input: Value, field: [f64; 8], origin: [f64; 3]| {
        input
            .with("field", &field[..3])
            .with("species_field_gradient", &field[3..])
            .with("origin", origin)
    };
    let with_response = options
        .clone()
        .with("species_polarizability", true)
        .with("species_quadrupole_polarizability", true);
    let base = at_field(with_response.clone(), bias, origin);
    let analytic = solve(base.clone());
    let target_moments = moments(&analytic);
    let target_tensor = susceptibility(&analytic, "real");
    let coupling = scalar(analytic.get("energy_components").unwrap(), "external_field");
    assert!(
        (coupling
            + bias
                .iter()
                .zip(target_moments)
                .map(|(f, m)| f * m)
                .sum::<f64>())
        .abs()
            < 2e-13
    );
    assert!(scalar(quadrupole(&analytic), "cross_raw_asymmetry") < 2e-9);
    println!(
        "Joint quadrupole static energy={:.17e} {}",
        scalar(&analytic, "energy"),
        quadrupole(&analytic).to_json()
    );

    let labels: Vec<_> = (0..8)
        .map(|a| format!("joint quadrupole moment[{a}] from energy"))
        .chain((0..8).flat_map(|a| (0..8).map(move |b| format!("joint quadrupole chi[{a},{b}]"))))
        .collect();
    // Form four full-rank pairs of mixed fields. Inverting each pair
    // recovers the ordinary partials without single-axis SCF sampling.
    let partner = [3, 4, 5, 0, 1, 2, 7, 6];
    let mix: [f64; 8] = [0.37, 0.37, 0.37, -0.37, -0.37, -0.37, 0.37, -0.37];
    let undo_mix = |values: [f64; 8], b: usize| {
        (values[b] - mix[b] * values[partner[b]]) / (1.0 + mix[b].powi(2))
    };
    let mut failed = Vec::new();
    components::fit_components(
        &labels,
        &[
            [0.014, 0.010, 0.007, 0.004],
            [0.009, 0.0065, 0.0045, 0.003],
            [0.024, 0.018, 0.013, 0.009],
        ],
        |steps| {
            let mut errors = vec![[0.0; 4]; labels.len()];
            let sample = |b: usize, h: f64| {
                let mut field = bias;
                field[b] += h;
                field[partner[b]] += mix[b] * h;
                solve(at_field(options.clone(), field, origin))
            };
            for (step, &h) in steps.iter().enumerate() {
                let mut directional_moments = [0.0; 8];
                let mut directional_response = [[0.0; 8]; 8];
                for b in 0..8 {
                    let plus = sample(b, h);
                    let minus = sample(b, -h);
                    let first = |p, m| Stencil::Second.first(p, m, 0.0, 0.0, h);
                    directional_moments[b] =
                        -first(scalar(&plus, "energy"), scalar(&minus, "energy"));
                    let (mp, mm) = (moments(&plus), moments(&minus));
                    for a in 0..8 {
                        directional_response[a][b] = first(mp[a], mm[a]);
                    }
                }
                for b in 0..8 {
                    errors[b][step] = (undo_mix(directional_moments, b) - target_moments[b]).abs();
                    for a in 0..8 {
                        errors[8 + 8 * a + b][step] =
                            (undo_mix(directional_response[a], b) - target_tensor[a][b]).abs();
                    }
                }
            }
            errors
        },
        &mut failed,
    );
    components::assert_no_failures("joint electric quadrupole and mixed response", &failed);

    let at_frequency = |omega, damping, imaginary| {
        solve(
            base.clone()
                .with("frequency", omega)
                .with("frequency_damping", damping)
                .with("imaginary_frequency", imaginary),
        )
    };
    let plus = at_frequency(0.08, 0.008, false);
    let minus = at_frequency(-0.08, 0.008, false);
    let near_zero = at_frequency(1e-5, 0.0, true);
    let plus_real = susceptibility(&plus, "real");
    let plus_imag = susceptibility(&plus, "imag");
    let minus_real = susceptibility(&minus, "real");
    let minus_imag = susceptibility(&minus, "imag");
    let limit_real = susceptibility(&near_zero, "real");
    let limit_imag = susceptibility(&near_zero, "imag");
    let defect = |a: [[f64; 8]; 8], b: [[f64; 8]; 8], sign: f64| {
        a.into_iter()
            .flatten()
            .zip(b.into_iter().flatten())
            .map(|(a, b)| (a - sign * b).abs())
            .fold(0.0_f64, f64::max)
    };
    let conjugation = defect(plus_real, minus_real, 1.0).max(defect(plus_imag, minus_imag, -1.0));
    let static_limit =
        defect(limit_real, target_tensor, 1.0).max(defect(limit_imag, [[0.0; 8]; 8], 1.0));
    assert!(conjugation < 1e-8);
    assert!(static_limit < 1e-7);
    assert!((0..8).all(|a| plus_imag[a][a] > 0.0));

    let delta = [0.19, -0.23, 0.31];
    let modes = quadrupole_modes();
    let t_delta: [[f64; 3]; 5] =
        modes.map(|mode| mode.map(|row| row.iter().zip(delta).map(|(t, d)| t * d).sum::<f64>()));
    let g_delta: [f64; 3] =
        std::array::from_fn(|a| t_delta.iter().zip(&bias[3..]).map(|(v, &b)| b * v[a]).sum());
    let mut shifted_bias = bias;
    for (amplitude, change) in shifted_bias.iter_mut().zip(g_delta) {
        *amplitude += change;
    }
    let shifted_origin = std::array::from_fn(|a| origin[a] + delta[a]);
    let shifted = solve(at_field(with_response, shifted_bias, shifted_origin));
    let shifted_moments = moments(&shifted);
    let charge = 3.0;
    let expected_energy_change = charge
        * (0..3)
            .map(|a| delta[a] * (bias[a] + 0.5 * g_delta[a]))
            .sum::<f64>();
    assert!(
        (scalar(&shifted, "energy") - scalar(&analytic, "energy") - expected_energy_change).abs()
            < 2e-11
    );
    for a in 0..3 {
        assert!((shifted_moments[a] - target_moments[a] + charge * delta[a]).abs() < 2e-10);
    }
    for a in 0..5 {
        let change = (0..3)
            .map(|b| t_delta[a][b] * (0.5 * charge * delta[b] - target_moments[b]))
            .sum::<f64>();
        assert!((shifted_moments[a + 3] - target_moments[a + 3] - change).abs() < 2e-10);
    }
    // Translating the origin mixes each gradient moment with the three
    // dipole moments. Both response indices transform with this same map.
    let mut translate = std::array::from_fn::<_, 8, _>(|a| {
        std::array::from_fn::<_, 8, _>(|b| f64::from(u8::from(a == b)))
    });
    for (row, change) in translate[3..].iter_mut().zip(t_delta) {
        for (value, term) in row.iter_mut().zip(change) {
            *value = -term;
        }
    }
    let translated = std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            (0..8)
                .flat_map(|i| {
                    (0..8).map(move |j| translate[a][i] * target_tensor[i][j] * translate[b][j])
                })
                .sum::<f64>()
        })
    });
    let origin_defect = defect(susceptibility(&shifted, "real"), translated, 1.0);
    assert!(origin_defect < 2e-9);
    std::fs::remove_file(path).unwrap();
    println!(
        "Joint quadrupole energy={:.17e} moments={target_moments:?} alpha00={:.17e} C00={:.17e} C00_retarded={:.17e}+{:.17e}i conjugation={conjugation:.17e} static_limit={static_limit:.17e} origin_defect={origin_defect:.17e}",
        scalar(&analytic, "energy"), target_tensor[0][0], 3.0 * target_tensor[3][3],
        3.0 * plus_real[3][3], 3.0 * plus_imag[3][3],
    );
    println!("Joint quadrupole retarded {}", quadrupole(&plus).to_json());
}
