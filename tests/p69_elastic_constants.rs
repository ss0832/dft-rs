// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic elastic constants on the common driver: independent four-step
//! fits of central differences of the analytic SCF strain derivative
//! (validation only) under homogeneous strain and under Gamma-point atomic
//! displacements, Richardson checks, the relaxed-ion assembly identities and
//! the CLI route. All 81 strain-Hessian and 54 internal-strain entries are
//! fitted independently for each parent model by default. Retained failed
//! intervals resolve on staged refinements without changing the original
//! p/R2/Richardson gates. Diagnostic environment filters are explicitly partial.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use std::collections::BTreeMap;
use std::process::Command;
use stencils::Stencil;

/// The hydrogen chain of the stress tests: two atoms per cell along x in a
/// skewed cell, so that every strain component carries a response.
const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.5, 0.1, 0.05]];

/// The cell strained by `eps` after displacing `atom` by `u` along `axis`.
fn system(eps: [[f64; 3]; 3], atom: usize, axis: usize, u: f64) -> System {
    let map = |v: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|a| v[a] + (0..3).map(|b| eps[a][b] * v[b]).sum::<f64>())
    };
    let mut positions = POSITIONS;
    positions[atom][axis] += u;
    System {
        numbers: vec![1, 1],
        positions: positions.iter().map(|p| map(*p)).collect(),
        cell: Some(CELL.map(map)),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(18.0)),
        ("grid_level".to_string(), Value::Int(4)),
        ("grid_shell".to_string(), Value::Int(4)),
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        ("conv_tol".to_string(), Value::Float(1e-13)),
        ("conv_tol_dp".to_string(), Value::Float(1e-12)),
        ("max_iter".to_string(), Value::Int(500)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn elastic_and_stress_controls(shared: &[(&str, Value)]) -> (Controls, Controls) {
    let mut response = shared.to_vec();
    response.push(("response_tolerance", Value::Float(1e-11)));
    // The finite-difference oracle differentiates only the strain derivative.
    let mut stress = shared.to_vec();
    stress.push(("forces", Value::Bool(false)));
    (controls(&response), controls(&stress))
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn tensor(v: &Value, key: &str) -> [[f64; 3]; 3] {
    let data = flat(v, key);
    std::array::from_fn(|a| std::array::from_fn(|b| data[3 * a + b]))
}

/// Explicit diagnostic filters never establish complete-tensor acceptance.
fn selected_column(kind: &str, column: usize) -> bool {
    let key = format!("DFT_RS_TEST_ELASTIC_{kind}_COLUMNS");
    match std::env::var(key) {
        Ok(value) => value.split(',').filter(|v| !v.is_empty()).any(|v| {
            v.parse::<usize>()
                .expect("elastic diagnostic columns must be integer indices")
                == column
        }),
        Err(_) => true,
    }
}

fn selected_model(name: &str) -> bool {
    match std::env::var("DFT_RS_TEST_ELASTIC_FUNCTIONAL") {
        Ok(value) => {
            println!("Diagnostic elastic subset requested: {value}; full tensor acceptance remains separate");
            value == name
        }
        Err(_) => true,
    }
}

/// One displaced pair supplies all nine rows. The shared component fit is
/// followed by the original stricter p/R2/Richardson gates, per entry.
fn fit_nine(
    label: &str,
    exact: [f64; 9],
    stages: &[([f64; 4], Stencil)],
    mut evaluate: impl FnMut(f64) -> [[f64; 3]; 3],
    failures: &mut Vec<String>,
) {
    let names: Vec<_> = (0..9)
        .map(|i| format!("{label} row [{},{}]", i / 3, i % 3))
        .collect();
    let mut accepted = [false; 9];
    let mut cache = BTreeMap::<u64, [[f64; 3]; 3]>::new();
    for (stage, (steps, stencil)) in stages.iter().enumerate() {
        let mut at = |t: f64| *cache.entry(t.to_bits()).or_insert_with(|| evaluate(t));
        let estimates: [[f64; 9]; 4] = std::array::from_fn(|k| {
            let h = steps[k];
            let (p, m) = (at(h), at(-h));
            let (p2, m2) = if *stencil == Stencil::Fourth {
                (at(2.0 * h), at(-2.0 * h))
            } else {
                ([[0.0; 3]; 3], [[0.0; 3]; 3])
            };
            std::array::from_fn(|i| {
                stencil.first(
                    p[i / 3][i % 3],
                    m[i / 3][i % 3],
                    p2[i / 3][i % 3],
                    m2[i / 3][i % 3],
                    h,
                )
            })
        });
        let errors: Vec<[f64; 4]> = (0..9)
            .map(|i| std::array::from_fn(|k| (estimates[k][i] - exact[i]).abs()))
            .collect();
        let labels: Vec<_> = names
            .iter()
            .map(|n| format!("{n} strict stage {}", stage + 1))
            .collect();
        let mut common_failures = Vec::new();
        stencils::fit_components_staged(
            &labels,
            &[(*steps, *stencil)],
            |_, _| errors.clone(),
            &mut common_failures,
        );
        for i in 0..9 {
            if accepted[i] || errors[i].iter().any(|e| !e.is_finite() || *e == 0.0) {
                continue;
            }
            let fit = convergence::error_power_fit(steps, &errors[i]);
            // Eliminate the leading error using the actual step ratio.
            // The fourth-order fallback also supports non-dyadic intervals.
            let scale = (steps[2] / steps[3]).powf(stencil.order());
            let coarse_error = estimates[2][i] - exact[i];
            let fine_error = estimates[3][i] - exact[i];
            let richardson = ((scale * fine_error - coarse_error) / (scale - 1.0)).abs();
            let pass = !common_failures.contains(&labels[i])
                && (fit.exponent - stencil.order()).abs() < 0.1
                && fit.r_squared > 0.9999
                && richardson < 2e-5;
            println!("{} strict gates steps={steps:?} p={:.12} R2={:.12} C={:.12e} Richardson={richardson:.12e}{}",
                labels[i],fit.exponent,fit.r_squared,fit.prefactor,if pass { "" } else { " FAILED" });
            accepted[i] = pass;
        }
        if accepted.iter().all(|v| *v) {
            return;
        }
    }
    failures.extend(names.into_iter().enumerate().filter_map(|(i, n)| {
        if accepted[i] {
            None
        } else {
            Some(n)
        }
    }));
}

/// Complete 9x9 composed-strain and 9x(3N) internal-strain tensors.
fn fit_all_elastic_entries(
    label: &str,
    report: &Value,
    strain_steps: [f64; 4],
    displacement_steps: [f64; 4],
    mut first: impl FnMut([[f64; 3]; 3], usize, usize, f64) -> [[f64; 3]; 3],
) {
    let w = flat(report, "strain_hessian");
    let lambda = flat(report, "internal_strain");
    let s0 = tensor(report, "strain_derivative");
    let zero = [[0.0; 3]; 3];
    let mut failures = Vec::new();
    let strain_stages = [
        (strain_steps, Stencil::Second),
        ([0.01, 0.005, 0.0025, 0.00125], Stencil::Second),
        ([0.005, 0.0025, 0.00125, 0.000625], Stencil::Second),
        ([0.0025, 0.00125, 0.000625, 0.0003125], Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
        ([0.02, 0.01, 0.005, 0.0025], Stencil::Fourth),
    ];
    for column in 0..9 {
        if !selected_column("STRAIN", column) {
            continue;
        }
        let (c, d) = (column / 3, column % 3);
        // A fresh strain is composed with the prestrained cell, giving
        // W_ab,cd + delta_bc E_,ad rather than W alone.
        let exact = std::array::from_fn(|i| {
            w[i * 9 + column] + if i % 3 == c { s0[i / 3][d] } else { 0.0 }
        });
        fit_nine(
            &format!("{label} strain [{c},{d}]"),
            exact,
            &strain_stages,
            |h| {
                let mut eps = zero;
                eps[c][d] = h;
                first(eps, 0, 0, 0.0)
            },
            &mut failures,
        );
    }
    let displacement_stages = [
        (displacement_steps, Stencil::Second),
        ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
        ([0.005, 0.0025, 0.00125, 0.000625], Stencil::Second),
        ([0.16, 0.08, 0.04, 0.02], Stencil::Second),
        ([0.04, 0.02, 0.01, 0.005], Stencil::Fourth),
    ];
    for column in 0..6 {
        if !selected_column("DISPLACEMENT", column) {
            continue;
        }
        let exact = std::array::from_fn(|i| lambda[i * 6 + column]);
        fit_nine(
            &format!("{label} internal column {column}"),
            exact,
            &displacement_stages,
            |h| first(zero, column / 3, column % 3, h),
            &mut failures,
        );
    }
    components::assert_no_failures("complete elastic/internal-strain entries", &failures);
}

#[test]
fn elastic_constants_pass_four_step_fits_of_the_analytic_stress() {
    if !selected_model("svwn5") {
        return;
    }
    let zero = [[0.0; 3]; 3];
    let report = run("elastic", &system(zero, 0, 0, 0.0), &controls(&[])).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let lambda = flat(&report, "internal_strain");
    let relaxation = flat(&report, "internal_relaxation");
    let phi = flat(&report, "force_constants");
    for key in [
        "raw_asymmetry",
        "internal_strain_asymmetry",
        "rotation_residual",
        "force_constant_asymmetry",
        "acoustic_residual",
        "smallest_optical_eigenvalue",
    ] {
        println!("{key} = {:e}", number(&report, key));
    }
    println!("clamped-ion (GPa) {:?}", flat(&report, "clamped_ion_gpa"));
    println!("relaxed-ion (GPa) {:?}", flat(&report, "relaxed_ion_gpa"));
    println!("strain solves {:?}", report.get("strain_solves"));
    let first = |eps: [[f64; 3]; 3], atom: usize, axis: usize, u: f64| -> [[f64; 3]; 3] {
        tensor(
            &run("stress", &system(eps, atom, axis, u), &controls(&[])).unwrap(),
            "strain_derivative",
        )
    };
    fit_all_elastic_entries(
        "svwn5",
        &report,
        [0.04, 0.02, 0.01, 0.005],
        [0.08, 0.04, 0.02, 0.01],
        first,
    );
    // The ionic relaxation solves Phi du = -Lambda^T outside the uniform
    // translations: the residual has equal components on both atoms.
    for x in 0..9 {
        let residual: Vec<f64> = (0..6)
            .map(|u| {
                (0..6)
                    .map(|v| phi[6 * u + v] * relaxation[9 * v + x])
                    .sum::<f64>()
                    + lambda[6 * x + u]
            })
            .collect();
        for axis in 0..3 {
            let mean = 0.5 * (residual[axis] + residual[3 + axis]);
            assert!(
                (residual[axis] - mean).abs() < 1e-8,
                "relaxation residual {residual:?}"
            );
        }
    }
    assert!(number(&report, "raw_asymmetry") < 1e-6);
    assert!(number(&report, "internal_strain_asymmetry") < 1e-6);
}

/// Elastic constants with exact exchange: the strain Hessian and the
/// internal-strain coupling of Hartree-Fock and a global hybrid against
/// central differences of the analytic strain derivative. The exchange
/// cutoff is large enough for the truncation asymmetry of the image sum to
/// stay below the finite-difference errors.
#[test]
fn hybrid_elastic_constants_pass_four_step_fits_of_the_analytic_stress() {
    let zero = [[0.0; 3]; 3];
    for xc in ["hf", "pbe0", "wb97x-d"] {
        if !selected_model(xc) {
            continue;
        }
        // A generic rotation of the polar bond makes the displaced
        // molecular-crystal geometry asymmetric in the triclinic cell.
        let fixture = |eps: [[f64; 3]; 3], atom: usize, axis: usize, u: f64| {
            if xc != "wb97x-d" {
                return system(eps, atom, axis, u);
            }
            let cell = [[11.0, 0.0, 0.0], [0.31, 12.0, 0.0], [0.22, 0.45, 12.7]];
            let positions = [[0.19, -0.24, 0.37], [1.64, 0.79, -0.63]];
            let deform = |p: [f64; 3]| {
                std::array::from_fn(|a| p[a] + (0..3).map(|b| eps[a][b] * p[b]).sum::<f64>())
            };
            let mut positions = positions;
            positions[atom][axis] += u;
            System {
                numbers: vec![1, 9],
                positions: positions.map(deform).to_vec(),
                cell: Some(cell.map(deform)),
                pbc: [true; 3],
            }
        };
        let settings = || {
            controls(&[
                ("xc", Value::Str(xc.into())),
                (
                    "k_mesh",
                    Value::Array(vec![
                        Value::Int(if xc == "wb97x-d" { 1 } else { 2 }),
                        Value::Int(1),
                        Value::Int(1),
                    ]),
                ),
                (
                    "overlap_cutoff",
                    Value::Float(if xc == "wb97x-d" { 7.0 } else { 14.0 }),
                ),
                (
                    "exchange_cutoff",
                    Value::Float(if xc == "wb97x-d" { 7.0 } else { 12.0 }),
                ),
                (
                    "xc_ao_cutoff",
                    Value::Float(if xc == "wb97x-d" { 12.0 } else { 14.0 }),
                ),
                ("disp_cutoff", Value::Float(14.4)),
                ("grid_level", Value::Int(2)),
                ("grid_shell", Value::Int(1)),
            ])
        };
        let report = run("elastic", &fixture(zero, 0, 0, 0.0), &settings()).unwrap();
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        if xc == "wb97x-d" {
            let same_cell = run("stress", &fixture(zero, 0, 0, 0.0), &settings()).unwrap();
            let dispersion = number(&report, "dispersion_energy");
            assert!(dispersion < 0.0);
            assert!((dispersion - number(&same_cell, "dispersion_energy")).abs() < 1e-13);
            assert!((number(&report, "energy") - number(&same_cell, "energy")).abs() < 1e-11);
            println!(
                "omegaB97X-D SOEC E={:.17e} dispersion={dispersion:.17e} strain solves={:?}",
                number(&report, "energy"),
                report.get("strain_solves")
            );
        }
        println!(
            "{xc}: raw asymmetry {:e}, internal-strain asymmetry {:e}, rotation residual {:e}, force-constant asymmetry {:e}, clamped-ion (GPa) {:?}",
            number(&report, "raw_asymmetry"),
            number(&report, "internal_strain_asymmetry"),
            number(&report, "rotation_residual"),
            number(&report, "force_constant_asymmetry"),
            flat(&report, "clamped_ion_gpa")
        );
        let first = |eps: [[f64; 3]; 3], atom: usize, axis: usize, u: f64| -> [[f64; 3]; 3] {
            tensor(
                &run("stress", &fixture(eps, atom, axis, u), &settings()).unwrap(),
                "strain_derivative",
            )
        };
        fit_all_elastic_entries(
            xc,
            &report,
            [0.01, 0.005, 0.0025, 0.00125],
            [0.04, 0.02, 0.01, 0.005],
            first,
        );
        assert!(number(&report, "raw_asymmetry") < 1e-6);
        assert!(number(&report, "internal_strain_asymmetry") < 1e-6);
    }
}

#[test]
fn charged_soec_fits_all_ordered_strain_entries() {
    // A closed-shell lithium ion in a triclinic cell has no optical
    // internal coordinates. All nine affine directions remain independent.
    let lattice = [[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]];
    let position = [0.17, 0.41, -0.19];
    let geometry = |column: usize, h: f64| {
        let deform = |mut point: [f64; 3]| {
            point[column / 3] += h * point[column % 3];
            point
        };
        System {
            numbers: vec![3],
            positions: vec![deform(position)],
            cell: Some(lattice.map(deform)),
            pbc: [true; 3],
        }
    };
    let shared = [
        ("reference", Value::Str("rks".into())),
        ("charge", Value::Int(1)),
        ("k_mesh", Value::Array(vec![Value::Int(1); 3])),
        // The previous 12-Bohr cutoff introduced the +/- (1,0,-1) images
        // under compression within the stencil. For every affine component
        // and |h| <= 0.08, the last retained and first omitted images are
        // bounded by 13.628846 and 14.018473 Bohr, respectively.
        ("overlap_cutoff", Value::Float(13.8)),
        // At 16 Bohr, shell/point pairs cross the XC AO cutoff while their
        // tails still contribute to the stress. The 24-to-28 Bohr comparison
        // bounds this truncation on the retained xx diagnostic interval.
        ("xc_ao_cutoff", Value::Float(24.0)),
        ("grid_level", Value::Int(1)),
        ("grid_shell", Value::Int(1)),
        ("conv_tol_grad", Value::Float(1e-11)),
    ];
    let (response, settings) = elastic_and_stress_controls(&shared);
    let report = run("elastic", &geometry(0, 0.0), &response).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let hessian = flat(&report, "strain_hessian");
    let first = tensor(&report, "strain_derivative");
    assert_eq!(hessian.len(), 81);
    assert!(number(&report, "raw_asymmetry") < 1e-6);
    let clamped = flat(&report, "clamped_ion_gpa");
    let relaxed = flat(&report, "relaxed_ion_gpa");
    assert_eq!(clamped, relaxed);
    let orbital_residual = number(&report, "orbital_residual");
    assert!(orbital_residual < 1e-11);
    let strain_solves = report.get("strain_solves").unwrap().as_array().unwrap();
    assert_eq!(strain_solves.len(), 9);
    assert!(strain_solves
        .iter()
        .all(|solve| number(solve, "relative_residual") < 1e-11));
    println!("charged Li SOEC E={:.17e} orbital residual={orbital_residual:.12e} strain Hessian={hessian:?} C(GPa)={clamped:?} strain solves={strain_solves:?}",
        number(&report, "energy"));
    let mut failures = Vec::new();
    for column in 0..9 {
        if !selected_column("STRAIN", column) {
            continue;
        }
        let (c, d) = (column / 3, column % 3);
        let exact = std::array::from_fn(|i| {
            hessian[9 * i + column] + if i % 3 == c { first[i / 3][d] } else { 0.0 }
        });
        fit_nine(
            &format!("charged Li SOEC [{c},{d}]"),
            exact,
            &[
                ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
                ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
                ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
            ],
            |h| {
                let report = run("stress", &geometry(column, h), &settings).unwrap();
                assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
                let residual = number(&report, "orbital_residual");
                assert!(residual < 1e-11);
                println!(
                    "charged Li strain [{c},{d}] h={h:+.10e} orbital residual={residual:.12e}"
                );
                tensor(&report, "strain_derivative")
            },
            &mut failures,
        );
    }
    components::assert_no_failures("charged Li complete ordered strain Hessian", &failures);
}

#[test]
fn scalar_ecp_nah_soec_fits_all_strain_and_internal_entries() {
    // An oriented NaH trial bond in a triclinic molecular-crystal model.
    // The trial distance probes the stretched side of the NaH bond.
    let lattice = [[16.0, 0.9, 0.5], [0.6, 17.1, 0.8], [0.4, -0.7, 17.8]];
    let positions = [[0.13, -0.21, 0.07], [2.885, -1.879, 2.282]];
    let geometry = |eps: [[f64; 3]; 3], atom: usize, axis: usize, u: f64| {
        let deform = |point: [f64; 3]| {
            std::array::from_fn(|i| point[i] + (0..3).map(|j| eps[i][j] * point[j]).sum::<f64>())
        };
        let mut shifted = positions;
        shifted[atom][axis] += u;
        System {
            numbers: vec![11, 1],
            positions: shifted.map(deform).to_vec(),
            cell: Some(lattice.map(deform)),
            pbc: [true; 3],
        }
    };
    let shared = [
        ("reference", Value::Str("rks".into())),
        ("xc", Value::Str("pbe".into())),
        ("basis", Value::Str("lanl2dz".into())),
        ("k_mesh", Value::Array(vec![Value::Int(1); 3])),
        ("overlap_cutoff", Value::Float(18.0)),
        ("xc_ao_cutoff", Value::Float(20.0)),
        ("grid_level", Value::Int(2)),
        ("grid_shell", Value::Int(2)),
        ("conv_tol_grad", Value::Float(1e-11)),
    ];
    let (response, settings) = elastic_and_stress_controls(&shared);
    let report = run("elastic", &geometry([[0.0; 3]; 3], 0, 0, 0.0), &response).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    for key in [
        "raw_asymmetry",
        "internal_strain_asymmetry",
        "acoustic_residual",
    ] {
        let residual = number(&report, key);
        println!("NaH scalar-ECP SOEC {key}={residual:.17e}");
        assert!(residual < 1e-6, "{key}: {residual:e}");
    }
    assert!(number(&report, "smallest_optical_eigenvalue") > 0.0);
    println!(
        "NaH scalar-ECP SOEC E={:.17e} C_clamped(GPa)={:?} C_relaxed(GPa)={:?} strain solves={:?} displacement solves={:?}",
        number(&report, "energy"),
        flat(&report, "clamped_ion_gpa"),
        flat(&report, "relaxed_ion_gpa"),
        report.get("strain_solves"),
        report.get("displacement_solves"),
    );
    fit_all_elastic_entries(
        "NaH scalar-ECP PBE SOEC",
        &report,
        [0.04, 0.02, 0.01, 0.005],
        [0.08, 0.04, 0.02, 0.01],
        |eps, atom, axis, u| {
            tensor(
                &run("stress", &geometry(eps, atom, axis, u), &settings).unwrap(),
                "strain_derivative",
            )
        },
    );
}

#[test]
fn elastic_cli_route_matches_the_rust_driver() {
    let coarse = controls(&[
        ("overlap_cutoff", Value::Float(12.0)),
        ("grid_level", Value::Int(2)),
        ("grid_shell", Value::Int(1)),
        (
            "k_mesh",
            Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
        ),
    ]);
    let report = run("elastic", &system([[0.0; 3]; 3], 0, 0, 0.0), &coarse).unwrap();
    let path = std::env::temp_dir().join(format!("dft_elastic_{}.xyz", std::process::id()));
    let a = dft_rs::constants::BOHR_TO_ANGSTROM;
    let mut xyz = String::from("2\nchain\n");
    for p in POSITIONS {
        xyz.push_str(&format!(
            "H {:.16} {:.16} {:.16}\n",
            p[0] * a,
            p[1] * a,
            p[2] * a
        ));
    }
    std::fs::write(&path, xyz).unwrap();
    let rows: Vec<String> = CELL
        .iter()
        .map(|r| format!("{:.16},{:.16},{:.16}", r[0] * a, r[1] * a, r[2] * a))
        .collect();
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .args([
            "elastic",
            path.to_str().unwrap(),
            "--cell",
            &rows.join(";"),
            "--xc",
            "svwn5",
            "--basis",
            "sto-3g",
            "--k-mesh",
            "2,1,1",
            "--overlap-cutoff",
            "12",
            "--grid-level",
            "2",
            "--grid-shell",
            "1",
            "--grid-weight-cutoff",
            "0",
            "--conv-tol",
            "1e-13",
            "--conv-tol-dp",
            "1e-12",
            "--max-iter",
            "500",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json = Value::parse_json(&String::from_utf8_lossy(&output.stdout)).unwrap();
    for key in ["strain_hessian", "internal_strain", "relaxed_ion_gpa"] {
        let (_, cli) = json.get(key).unwrap().numeric_array().unwrap();
        let (_, rust) = report.get(key).unwrap().numeric_array().unwrap();
        for (x, y) in cli.iter().zip(&rust) {
            assert!((x - y).abs() < 1e-8 * (1.0 + y.abs()), "{key}: {x} vs {y}");
        }
    }
}
