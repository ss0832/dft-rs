// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electric moments of independently relaxed collinear and spinor densities across XC classes.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use stencils::Stencil;

const EXPONENT: f64 = 0.8;
const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.21, 0.19],
    [1.57, 0.16, -0.08],
    [-0.19, 1.93, 0.37],
    [0.42, 0.29, 2.28],
];
const ORIGIN: [f64; 3] = [0.23, -0.5, 0.42];
const SHIFT: [f64; 3] = [0.32, 0.19, -0.11];
const FIELD: [f64; 3] = [0.002, -0.004, 0.003];

fn get(feature: &str, system: &System, options: &Value) -> Value {
    let report = run(feature, system, &Controls::from_value(options).unwrap())
        .unwrap_or_else(|err| panic!("{feature} {options:?}: {err}"));
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    if let Some(residual) = report
        .get("scf_orbital_residual")
        .or_else(|| report.get("orbital_residual"))
    {
        let residual = residual.as_f64().unwrap();
        assert!(residual < 1e-10, "{feature}: {residual:.12e}");
    }
    report
}

fn options(reference: &str, basis: &str, model: &str) -> Value {
    let mut value = Value::object()
        .with("basis", basis)
        .with("reference", reference)
        .with("grid_level", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("guess", "core")
        .with("conv_tol", 1e-13)
        .with("conv_tol_grad", 1e-11)
        .with("conv_tol_dp", 1e-11)
        .with("max_iter", 300)
        .with("field", FIELD);
    if model == "weighted" {
        value.set(
            "xc_terms",
            vec![
                Value::object().with("xc", "hf").with("weight", 0.65),
                Value::object()
                    .with("x", "slater")
                    .with("c", "vwn5")
                    .with("weight", 0.35),
            ],
        );
    } else {
        value.set("xc", model);
    }
    if reference != "rks" {
        value.set("charge", 1);
        value.set("multiplicity", 2);
    }
    if reference == "gks" {
        let source = spinor_operator::source(POSITIONS.len());
        value.set("gks_operator_real", &source.re);
        value.set("gks_operator_imaginary", &source.im);
    }
    value
}

fn values(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn spatial_density(report: &Value, reference: &str) -> [[f64; 4]; 4] {
    let channels = report.get("channels").unwrap().as_array().unwrap();
    if reference == "gks" {
        let spin = values(report, "spin_expectation");
        assert!(spin[0].abs() > 1e-5 && spin[1].abs() > 1e-5);
        let matrix = values(&channels[0], "density_real");
        assert_eq!(matrix.len(), 64);
        std::array::from_fn(|i| {
            std::array::from_fn(|j| matrix[8 * i + j] + matrix[8 * (i + 4) + j + 4])
        })
    } else {
        let alpha = values(&channels[0], "density");
        let beta = if reference == "rks" {
            assert_eq!(channels.len(), 1);
            alpha.clone()
        } else {
            assert_eq!(channels.len(), 2);
            values(&channels[1], "density")
        };
        assert_eq!(alpha.len(), 16);
        assert_eq!(beta.len(), 16);
        std::array::from_fn(|i| std::array::from_fn(|j| alpha[4 * i + j] + beta[4 * i + j]))
    }
}

// The product of the two normalized equal-exponent s Gaussians has
// overlap exp(-a |Ri-Rj|^2/2), midpoint mean and variance 1/(4a).
// This closed form is independent of the moment integral implementation.
fn gaussian_moment(density: &[[f64; 4]; 4], origin: [f64; 3], powers: [i32; 3]) -> f64 {
    let nuclear: f64 = POSITIONS
        .iter()
        .map(|r| {
            (0..3)
                .map(|a| (r[a] - origin[a]).powi(powers[a]))
                .product::<f64>()
        })
        .sum();
    let mut electronic = 0.0;
    for (i, ri) in POSITIONS.iter().enumerate() {
        for (j, rj) in POSITIONS.iter().enumerate() {
            let distance2 = (0..3).map(|a| (ri[a] - rj[a]).powi(2)).sum::<f64>();
            let average = (0..3)
                .map(|a| {
                    let mean = 0.5 * (ri[a] + rj[a]) - origin[a];
                    match powers[a] {
                        0 => 1.0,
                        1 => mean,
                        2 => mean * mean + 0.25 / EXPONENT,
                        _ => unreachable!(),
                    }
                })
                .product::<f64>();
            electronic += density[i][j] * (-0.5 * EXPONENT * distance2).exp() * average;
        }
    }
    nuclear - electronic
}

fn component_axes(powers: &[f64]) -> (usize, usize) {
    let mut axes = powers
        .iter()
        .enumerate()
        .flat_map(|(axis, &power)| (0..power as usize).map(move |_| axis));
    (axes.next().unwrap(), axes.next().unwrap())
}

fn raw_tensor(rank: &Value) -> [[f64; 3]; 3] {
    let powers = values(rank, "monomials");
    let mut tensor = [[0.0; 3]; 3];
    for (power, value) in powers.chunks_exact(3).zip(values(rank, "cartesian")) {
        let (a, b) = component_axes(power);
        tensor[a][b] = value;
        tensor[b][a] = value;
    }
    tensor
}

fn fit_dipole(label: &str, system: &System, options: &Value, dipole: &[f64]) {
    let names = (0..3)
        .map(|axis| format!("{label} dipole[{axis}]"))
        .collect::<Vec<_>>();
    let stages = [
        ([0.008, 0.006, 0.004, 0.003], Stencil::Second),
        ([0.004, 0.003, 0.002, 0.0015], Stencil::Second),
        ([0.016, 0.012, 0.008, 0.006], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &stages,
        |steps, stencil| {
            (0..3)
                .map(|axis| {
                    let sample = |offset: f64| {
                        let mut field = FIELD;
                        field[axis] += offset;
                        let moved = options.clone().with("field", field);
                        get("energy", system, &moved)
                            .get("energy")
                            .unwrap()
                            .as_f64()
                            .unwrap()
                    };
                    steps.map(|h| {
                        let (plus, minus) = (sample(h), sample(-h));
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (sample(2.0 * h), sample(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        // The static field couples as -F.mu. Its energy derivative
                        // measures the zero-origin dipole of the relaxed state.
                        (stencil.first(plus, minus, plus2, minus2, h) + dipole[axis]).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures(label, &failures);
}

fn check_moments(model: &str) {
    let path =
        std::env::temp_dir().join(format!("dft-rs-{model}-moments-{}.g94", std::process::id()));
    std::fs::write(&path, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    let system = System::molecule(vec![1; 4], POSITIONS.to_vec());
    for reference in ["rks", "uks", "roks", "gks"] {
        let options = options(reference, path.to_str().unwrap(), model);
        let orbitals = get("orbitals", &system, &options.clone().with("density", true));
        let density = spatial_density(&orbitals, reference);
        let charge = if reference == "rks" { 0.0 } else { 1.0 };
        let dipole = get("dipole", &system, &options.clone().with("origin", ORIGIN));
        let dipole_values = values(&dipole, "dipole");
        let origin2: [f64; 3] = std::array::from_fn(|a| ORIGIN[a] + SHIFT[a]);
        let mut quadrupoles = Vec::new();
        let mut gaussian_defect = 0.0_f64;
        let mut convention_defect = 0.0_f64;
        for origin in [ORIGIN, origin2] {
            let report = get(
                "molecular-multipole",
                &system,
                &options.clone().with("max_rank", 2).with("origin", origin),
            );
            assert_eq!(
                report.get("reference_type").and_then(Value::as_str),
                Some(reference)
            );
            let ranks = report.get("ranks").unwrap().as_array().unwrap();
            assert_eq!(ranks.len(), 3);
            for rank in ranks {
                let cart = values(rank, "cartesian");
                let powers = values(rank, "monomials");
                for (p, moment) in powers.chunks_exact(3).zip(cart) {
                    let expected =
                        gaussian_moment(&density, origin, [p[0] as i32, p[1] as i32, p[2] as i32]);
                    gaussian_defect = gaussian_defect.max((moment - expected).abs());
                    assert!(
                        (moment - expected).abs() < 5e-11,
                        "{reference} {p:?}: {moment} != {expected}"
                    );
                }
            }
            assert!((values(&ranks[0], "cartesian")[0] - charge).abs() < 1e-11);
            let raw = raw_tensor(&ranks[2]);
            let trace = (0..3).map(|i| raw[i][i]).sum::<f64>();
            for (p, moment) in values(&ranks[2], "monomials")
                .chunks_exact(3)
                .zip(values(&ranks[2], "traceless"))
            {
                let (a, b) = component_axes(p);
                let expected = 1.5 * (raw[a][b] - if a == b { trace / 3.0 } else { 0.0 });
                convention_defect = convention_defect.max((moment - expected).abs());
                assert!((moment - expected).abs() < 5e-11);
            }
            let root3 = 3.0_f64.sqrt();
            let spherical = [
                raw[2][2] - 0.5 * (raw[0][0] + raw[1][1]),
                root3 * raw[0][2],
                root3 * raw[1][2],
                0.5 * root3 * (raw[0][0] - raw[1][1]),
                root3 * raw[0][1],
            ];
            for (actual, expected) in values(&ranks[2], "spherical").iter().zip(spherical) {
                convention_defect = convention_defect.max((actual - expected).abs());
                assert!((actual - expected).abs() < 5e-11);
            }
            quadrupoles.push(raw);
        }
        let mut origin_defect = 0.0_f64;
        for a in 0..3 {
            let mut powers = [0; 3];
            powers[a] = 1;
            let expected = gaussian_moment(&density, ORIGIN, powers);
            gaussian_defect = gaussian_defect.max((dipole_values[a] - expected).abs());
            assert!((dipole_values[a] - expected).abs() < 5e-11);
            for b in 0..3 {
                let shifted = quadrupoles[0][a][b]
                    - SHIFT[a] * dipole_values[b]
                    - SHIFT[b] * dipole_values[a]
                    + charge * SHIFT[a] * SHIFT[b];
                origin_defect = origin_defect.max((quadrupoles[1][a][b] - shifted).abs());
                assert!((quadrupoles[1][a][b] - shifted).abs() < 5e-11);
            }
        }
        let zero_origin: Vec<_> = dipole_values
            .iter()
            .zip(ORIGIN)
            .map(|(v, r)| v + charge * r)
            .collect();
        println!(
            "{model} {reference} moments orbital={:.12e} gaussian={gaussian_defect:.12e} convention={convention_defect:.12e} origin={origin_defect:.12e} dipole={dipole_values:?}",
            dipole
                .get("scf_orbital_residual")
                .unwrap()
                .as_f64()
                .unwrap()
        );
        fit_dipole(
            &format!("{model} {reference}"),
            &system,
            &options,
            &zero_origin,
        );
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn weighted_electric_moments_use_the_relaxed_density_in_every_reference() {
    check_moments("weighted");
}

#[test]
fn hf_electric_moments_use_the_relaxed_density_in_every_reference() {
    check_moments("hf");
}

#[test]
fn lda_electric_moments_use_the_relaxed_density_in_every_reference() {
    check_moments("svwn5");
}

#[test]
fn gga_electric_moments_use_the_relaxed_density_in_every_reference() {
    check_moments("pbe");
}

#[test]
fn meta_gga_electric_moments_use_the_relaxed_density_in_every_reference() {
    check_moments("tpss");
}

#[test]
fn global_hybrid_electric_moments_use_the_relaxed_density_in_every_reference() {
    check_moments("pbe0");
}

#[test]
fn missing_correlated_moment_densities_are_named_errors() {
    let system = System::molecule(vec![1; 4], POSITIONS.to_vec());
    for reference in ["roks", "gks"] {
        let request = options(reference, "sto-3g", "b2plyp");
        let error = run(
            "molecular-multipole",
            &system,
            &Controls::from_value(&request).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(error, dft_rs::DftError::Unsupported { .. }));
        assert!(error.to_string().contains("relaxed PT2 density"));
        let error = run("dipole", &system, &Controls::from_value(&request).unwrap()).unwrap_err();
        assert!(matches!(error, dft_rs::DftError::Unsupported { .. }));
        assert!(error.to_string().contains(reference));
    }
}
