// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent relaxed energy differences through the public periodic driver.
#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/stencils.rs"]
pub mod stencils;

use dft_rs::driver::{periodic::lattice_of, run, Controls, System, Value};
use dft_rs::math::Vec3;
use rayon::prelude::*;
use std::collections::BTreeMap;
use stencils::Stencil;

const BASIS: &str = "H 0\nS 1 1.0\n0.8 1.0\n****\nHe 0\nS 1 1.0\n1.1 1.0\n****\nLi 0\nS 1 1.0\n4.0 1.0\nS 1 1.0\n0.65 1.0\n****\n";
const GRADIENT_COMPONENTS: usize = 6;
const COMPONENTS: usize = GRADIENT_COMPONENTS + 9;
const GEOMETRY_FIT_STAGES: [([f64; 4], Stencil); 4] = [
    ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
    ([0.01, 0.005, 0.0025, 0.00125], Stencil::Second),
    ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
    ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
];

fn system(reference: &str, component: usize, step: f64) -> System {
    dimensional_system(reference, component, step, [true; 3])
}

fn dimensional_system(reference: &str, component: usize, step: f64, pbc: [bool; 3]) -> System {
    let mut result = System {
        numbers: vec![1, if reference == "rks" { 3 } else { 2 }],
        positions: vec![[0.13, -0.21, 0.07], [2.0, 0.45, -0.31]],
        cell: Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
        pbc,
    };
    if !pbc.into_iter().all(|active| active) {
        for (vector, active) in result.cell.as_mut().unwrap().iter_mut().zip(pbc) {
            if active {
                *vector = vector.map(|x| 0.55 * x);
            }
        }
    }
    displace(result, component, step)
}

fn displace(mut result: System, component: usize, step: f64) -> System {
    let gradient_components = 3 * result.positions.len();
    if component < gradient_components {
        result.positions[component / 3][component % 3] += step;
    } else if component < gradient_components + 9 {
        let strain = component - gradient_components;
        let (a, b) = (strain / 3, strain % 3);
        for vector in result
            .positions
            .iter_mut()
            .chain(result.cell.iter_mut().flatten())
        {
            vector[a] += step * vector[b];
        }
    }
    result
}

fn controls(reference: &str, basis: &str) -> Value {
    let mut options = Value::object()
        .with("basis", basis)
        .with("multiplicity", if reference == "rks" { 1 } else { 2 })
        .with("overlap_cutoff", 3.5)
        .with("exchange_cutoff", 3.5)
        .with("xc_ao_cutoff", 12.0)
        .with("grid_level", 2)
        .with("grid_weight_cutoff", 0.0)
        .with("guess", "core")
        .with("max_iter", 300)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-10)
        .with("no_adiis", true);
    options.set("reference", reference);
    match reference {
        "rks" | "uks" => options.set("grid_shell", 1),
        "roks" => {
            options.set("grid_shell", 1);
            options.set("damping", 0.0);
        }
        "gks" => {
            options.set("grid_level", 4);
            options.set("grid_cutoff", 6.0);
            options.set("ewald_alpha", 0.5);
            options.set("ewald_real_cutoff", 12.0);
            options.set("ewald_reciprocal_cutoff", 7.0);
            options.set("electrostatic_order", 16);
            options.set("damping", 0.2);
        }
        _ => unreachable!(),
    }
    options
}

fn terms() -> Vec<Value> {
    vec![
        Value::object().with("xc", "hf").with("weight", 0.65),
        Value::object().with("xc", "svwn5").with("weight", 0.35),
    ]
}

fn report(feature: &str, reference: &str, component: usize, step: f64, options: &Value) -> Value {
    dimensional_report(feature, reference, component, step, options, [true; 3])
}

fn dimensional_report(
    feature: &str,
    reference: &str,
    component: usize,
    step: f64,
    options: &Value,
    pbc: [bool; 3],
) -> Value {
    checked_report(
        feature,
        reference,
        component,
        step,
        options,
        &dimensional_system(reference, component, step, pbc),
    )
}

fn checked_report(
    feature: &str,
    reference: &str,
    component: usize,
    step: f64,
    options: &Value,
    system: &System,
) -> Value {
    let result =
        run(feature, system, &Controls::from_value(options).unwrap()).unwrap_or_else(|error| {
            panic!("{reference} {feature}, component {component}, h={step}: {error}")
        });
    assert_eq!(result.get("converged").and_then(Value::as_bool), Some(true));
    match reference {
        "rks" | "uks" => assert_eq!(
            result.get("unrestricted").and_then(Value::as_bool),
            Some(reference == "uks")
        ),
        "roks" | "gks" => assert_eq!(
            result.get("reference_type").and_then(Value::as_str),
            Some(reference)
        ),
        _ => unreachable!(),
    }
    let residual = number(&result, "orbital_residual");
    println!("PERIODIC_WEIGHTED_SCF {reference} {feature} component={component} h={step:.17e} energy={:.17e} orbital={residual:.17e}", number(&result, "energy"));
    assert!(residual.is_finite() && residual < 1e-10);
    result
}

fn number(report: &Value, key: &str) -> f64 {
    report
        .get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing {key}"))
}

fn tensor(report: &Value, key: &str) -> Vec<f64> {
    report
        .get(key)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
        })
        .collect()
}

fn with_basis<T>(name: &str, check: impl FnOnce(&str) -> T) -> T {
    with_basis_text(name, BASIS, check)
}

fn with_basis_text<T>(name: &str, text: &str, check: impl FnOnce(&str) -> T) -> T {
    let path =
        std::env::temp_dir().join(format!("dft-rs-periodic-{name}-{}.g94", std::process::id()));
    std::fs::write(&path, text).unwrap();
    let value = check(path.to_str().unwrap());
    std::fs::remove_file(path).unwrap();
    value
}

#[test]
fn periodic_weighted_controls_reject_ambiguous_and_unconnected_models() {
    use dft_rs::DftError;

    let system = system("rks", COMPONENTS, 0.0);
    let weighted = Value::object().with("xc_terms", terms());
    for (feature, options, invalid_input) in [
        ("stress", weighted.clone().with("xc", "pbe"), true),
        ("stress", weighted.clone().with("dispersion", "d3bj"), false),
        ("stress", weighted.clone().with("pt2_same_spin", 0.2), false),
        ("toec", weighted, false),
    ] {
        let error = run(feature, &system, &Controls::from_value(&options).unwrap()).unwrap_err();
        if invalid_input {
            assert!(matches!(error, DftError::InvalidInput(_)), "{error}");
        } else {
            assert!(matches!(error, DftError::Unsupported { .. }), "{error}");
        }
    }
}

#[test]
fn weighted_periodic_stress_uses_the_existing_reference_routes() {
    with_basis("weighted-routes", |basis| {
        for reference in ["rks", "uks", "roks", "gks"] {
            let options = controls(reference, basis);
            let named = options.clone().with("xc", "hf");
            let weighted = options.with(
                "xc_terms",
                vec![Value::object().with("xc", "hf").with("weight", 1.0)],
            );
            let named = report("stress", reference, COMPONENTS, 0.0, &named);
            let weighted = report("stress", reference, COMPONENTS, 0.0, &weighted);
            assert!((number(&named, "energy") - number(&weighted, "energy")).abs() < 1e-12);
            for (key, len) in [
                ("gradient", GRADIENT_COMPONENTS),
                ("strain_derivative", 9),
                ("stress", 9),
            ] {
                let a = tensor(&named, key);
                let b = tensor(&weighted, key);
                assert_eq!(a.len(), len);
                assert_eq!(b.len(), len);
                for (a, b) in a.iter().zip(b) {
                    assert!((a - b).abs() < 1e-11, "{reference} {key}: {a} != {b}");
                }
            }
        }
    });
}

#[test]
fn weighted_periodic_cartesian_and_affine_components_pass_independent_fits() {
    check_fits("weighted-collinear-fits", &["rks", "uks", "roks"]);
}

#[test]
fn weighted_periodic_generalized_cartesian_and_affine_components_pass_independent_fits() {
    check_fits("weighted-generalized-fits", &["gks"]);
}

fn check_fits(name: &str, references: &[&str]) {
    with_basis(name, |basis| {
        for &reference in references {
            let options = controls(reference, basis).with("xc_terms", terms());
            let analytic = report("stress", reference, COMPONENTS, 0.0, &options);
            let gradient = tensor(&analytic, "gradient");
            let strain = tensor(&analytic, "strain_derivative");
            assert_eq!(gradient.len(), GRADIENT_COMPONENTS);
            assert_eq!(strain.len(), 9);
            let analytic: Vec<_> = gradient.into_iter().chain(strain).collect();
            assert!(analytic.iter().all(|x| x.is_finite() && x.abs() > 1e-8));
            let names: Vec<_> = (0..COMPONENTS)
                .map(|index| {
                    let (label, index) = if index < GRADIENT_COMPONENTS {
                        ("gradient", index)
                    } else {
                        ("strain", index - GRADIENT_COMPONENTS)
                    };
                    format!(
                        "weighted periodic {reference} {label}[{},{}]",
                        index / 3,
                        index % 3
                    )
                })
                .collect();
            fit_energies(reference, &names, &analytic, |component, h| {
                number(
                    &report("periodic-energy", reference, component, h, &options),
                    "energy",
                )
            });
        }
    });
}

fn fit_energies(
    label: &str,
    names: &[String],
    analytic: &[f64],
    sample: impl Fn(usize, f64) -> f64 + Sync,
) {
    fit_energy_stages(label, names, analytic, &GEOMETRY_FIT_STAGES, sample);
}

fn fit_energy_stages(
    label: &str,
    names: &[String],
    analytic: &[f64],
    stages: &[([f64; 4], Stencil)],
    sample: impl Fn(usize, f64) -> f64 + Sync,
) {
    let mut samples = BTreeMap::new();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        names,
        stages,
        |steps, stencil| {
            let multipliers: &[f64] = match stencil {
                Stencil::Second => &[-1.0, 1.0],
                Stencil::Fourth => &[-2.0, -1.0, 1.0, 2.0],
            };
            let mut jobs: Vec<_> = (0..analytic.len())
                .flat_map(|component| {
                    steps.iter().flat_map(move |&h| {
                        multipliers
                            .iter()
                            .map(move |scale| (component, (scale * h).to_bits()))
                    })
                })
                .filter(|key| !samples.contains_key(key))
                .collect();
            jobs.sort_unstable();
            jobs.dedup();
            let values: Vec<_> = jobs
                .into_par_iter()
                .map(|(component, bits)| {
                    let energy = sample(component, f64::from_bits(bits));
                    ((component, bits), energy)
                })
                .collect();
            samples.extend(values);
            (0..analytic.len())
                .map(|component| {
                    std::array::from_fn(|i| {
                        let h = steps[i];
                        let energy = |step: f64| samples[&(component, step.to_bits())];
                        let (plus2, minus2) = match stencil {
                            Stencil::Second => (0.0, 0.0),
                            Stencil::Fourth => (energy(2.0 * h), energy(-2.0 * h)),
                        };
                        (stencil.first(energy(h), energy(-h), plus2, minus2, h)
                            - analytic[component])
                            .abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures(label, &failures);
}

#[test]
fn lower_dimensional_spinor_affine_stress_matches_relaxed_finite_k_energies() {
    with_basis("dimensional-spinor-strain", |basis| {
        for (label, pbc, unit) in [
            ("wire", [true, false, false], "hartree/bohr"),
            ("slab", [true, false, true], "hartree/bohr^2"),
        ] {
            let options = controls("gks", basis)
                .with("xc", "hf")
                .with("k_mesh", [2, 1, 1])
                .with("overlap_cutoff", 3.8)
                .with("exchange_cutoff", 3.8)
                .with("electrostatic_order", 48);
            let report = dimensional_report("stress", "gks", COMPONENTS, 0.0, &options, pbc);
            let derivative = tensor(&report, "strain_derivative");
            let stress = tensor(&report, "stress");
            let measure = number(&report, "stress_periodic_measure");
            assert_eq!(
                report
                    .get("units")
                    .and_then(|v| v.get("stress"))
                    .and_then(Value::as_str),
                Some(unit)
            );
            assert!(report.get("stress_gpa").is_none());
            assert!(derivative
                .iter()
                .zip(stress)
                .all(|(d, s)| (d / measure - s).abs() < 1e-14));
            let base = dimensional_system("gks", COMPONENTS, 0.0, pbc);
            let mut padding = base.clone();
            for (vector, active) in padding.cell.as_mut().unwrap().iter_mut().zip(pbc) {
                if !active {
                    *vector = vector.map(|x| 1.9 * x);
                }
            }
            let padded = run("stress", &padding, &Controls::from_value(&options).unwrap()).unwrap();
            assert!((number(&padded, "energy") - number(&report, "energy")).abs() < 1e-12);
            assert!(tensor(&padded, "stress")
                .iter()
                .zip(tensor(&report, "stress"))
                .all(|(a, b)| (a - b).abs() < 1e-12));
            println!("Dimensional spinor stress {label} {}", report.to_json());
            let labels: Vec<_> = (0..9)
                .map(|c| format!("{label} GKS affine[{},{}]", c / 3, c % 3))
                .collect();
            fit_energies(label, &labels, &derivative, |c, h| {
                number(
                    &dimensional_report(
                        "periodic-energy",
                        "gks",
                        GRADIENT_COMPONENTS + c,
                        h,
                        &options,
                        pbc,
                    ),
                    "energy",
                )
            });
        }
    });
}

fn check_dimensional_geometry(
    reference: &str,
    xc: &str,
    pbc: [bool; 3],
    unit: &str,
    k_mesh: [usize; 3],
) {
    with_basis(
        &format!("dimensional-{reference}-{xc}-{pbc:?}-{k_mesh:?}"),
        |basis| {
            let (base, options) = dimensional_fixture(reference, xc, pbc, k_mesh, basis);
            fit_geometry(reference, xc, unit, &base, options);
        },
    );
}

fn dimensional_fixture(
    reference: &str,
    xc: &str,
    pbc: [bool; 3],
    k_mesh: [usize; 3],
    basis: &str,
) -> (System, Value) {
    let base_controls = if pbc == [true; 3] {
        controls(reference, basis)
    } else {
        controls("gks", basis).with("electrostatic_order", 48)
    };
    let mut options = base_controls
        .with("reference", reference)
        .with("multiplicity", if reference == "rks" { 1 } else { 2 })
        .with("xc", xc)
        .with("k_mesh", k_mesh)
        .with("overlap_cutoff", 3.8)
        .with("exchange_cutoff", 3.8)
        .with("forces", true);
    if dft_rs::nlc::Vv10Params::for_functional(xc).is_some() {
        options.set("vv10_cutoff", 6.4);
        options.set("grid_level", 0);
        // This compact grid checks derivatives of the discrete energy.
        // Its integrated density is checked separately from stationarity.
        if pbc != [true; 3] {
            options.set("grid_tolerance", 0.1);
        }
    }
    // A tilted wire with only two atoms always has a mirror plane.
    let mut base = dimensional_system(reference, COMPONENTS, 0.0, pbc);
    base.numbers.push(2);
    base.positions.push([-0.46, 1.29, 1.62]);
    if reference == "roks" {
        base.numbers = vec![1, 3, 1];
        options.set("damping", 0.0);
    }
    if pbc == [true; 3] {
        for vector in base.cell.as_mut().unwrap() {
            *vector = vector.map(|x| x * 0.55);
        }
    }
    (base, options)
}

fn fit_geometry(reference: &str, xc: &str, unit: &str, base: &System, options: Value) {
    fit_geometry_with_stages(reference, xc, unit, base, options, None);
}

fn fit_geometry_with_stages(
    reference: &str,
    xc: &str,
    unit: &str,
    base: &System,
    options: Value,
    stages: Option<&[([f64; 4], Stencil)]>,
) {
    let pbc = base.pbc;
    let gradient_components = 3 * base.positions.len();
    let analytic = checked_report(
        "stress",
        reference,
        gradient_components + 9,
        0.0,
        &options,
        base,
    );
    if dft_rs::nlc::Vv10Params::for_functional(xc).is_some() {
        let contribution = if reference == "roks" {
            number(&analytic, "nonlocal_correlation_energy")
        } else {
            let energy = analytic.get("energy_components").unwrap();
            let Value::Object(terms) = energy else {
                panic!("expected a periodic energy term object");
            };
            let total: f64 = terms.iter().map(|(_, term)| term.as_f64().unwrap()).sum();
            assert!((total - number(&analytic, "energy")).abs() < 1e-11);
            number(energy, "nonlocal_correlation")
        };
        assert!(contribution.abs() > 1e-8);
        if let Some(diagnostics) = analytic.get("stress_diagnostics") {
            assert!(number(diagnostics, "xc_reference_energy_defect") < 1e-10);
        }
        if reference == "gks" {
            assert_eq!(
                number(&analytic, "nonlocal_correlation_energy"),
                contribution
            );
            for key in ["strain_components", "gradient_components"] {
                let term = tensor(analytic.get(key).unwrap(), "nonlocal_correlation");
                assert!(term.iter().any(|v| v.abs() > 1e-8));
            }
        }
        println!("{reference} {xc} {pbc:?} VV10={contribution:.16e}");
    }
    if reference == "roks" {
        if analytic.get("density_conjugation_defect").is_some() {
            assert!(number(&analytic, "density_conjugation_defect") < 1e-12);
            assert!(number(&analytic, "fock_conjugation_defect") < 1e-10);
        }
        assert!(number(&analytic, "closed_subspace_defect") < 1e-12);
        assert!(number(&analytic, "metric_defect") < 1e-12);
        let valence = if base.numbers.contains(&11) { 3.0 } else { 5.0 };
        assert_eq!(number(&analytic, "n_alpha"), 0.5 * (valence + 1.0));
        assert_eq!(number(&analytic, "n_beta"), 0.5 * (valence - 1.0));
    }
    if base.numbers.contains(&11) {
        let terms = if pbc == [true; 3] {
            "strain_derivative_terms"
        } else {
            "strain_components"
        };
        let name = if pbc == [true; 3] { "ECP" } else { "ecp" };
        let ecp_strain = tensor(analytic.get(terms).unwrap(), name);
        assert!(ecp_strain.iter().any(|v| v.abs() > 1e-6));
        if pbc != [true; 3] {
            let ecp_gradient = tensor(analytic.get("gradient_components").unwrap(), "ecp");
            assert!(ecp_gradient.iter().any(|v| v.abs() > 1e-6));
            for axis in 0..3 {
                assert!(ecp_gradient.iter().skip(axis).step_by(3).sum::<f64>().abs() < 1e-12);
            }
        }
    }
    let strain = tensor(&analytic, "strain_derivative");
    let stress = tensor(&analytic, "stress");
    let dimensional = analytic.get("stress_periodic_measure").is_some();
    let measure = number(
        &analytic,
        if dimensional {
            "stress_periodic_measure"
        } else {
            "volume"
        },
    );
    if options.get("field_scalar").is_some() {
        assert!(
            number(
                analytic.get("stress_diagnostics").unwrap(),
                "external_field_energy_defect"
            ) < 1e-12
        );
        assert!(
            number(
                analytic.get("gradient_diagnostics").unwrap(),
                "external_field_energy_defect"
            ) < 1e-12
        );
        for key in ["strain_components", "gradient_components"] {
            assert!(tensor(analytic.get(key).unwrap(), "external_field")
                .iter()
                .any(|v| v.abs() > 1e-5));
        }
        assert_eq!(
            analytic
                .get("stress_field_convention")
                .and_then(Value::as_str),
            Some("fixed reduced wave indices and fixed Cartesian Fourier amplitudes")
        );
    }
    assert_eq!(
        analytic
            .get("units")
            .and_then(|v| v.get("stress"))
            .and_then(Value::as_str)
            .and_then(|value| value.split_whitespace().next()),
        Some(unit)
    );
    assert_eq!(analytic.get("stress_gpa").is_some(), pbc == [true; 3]);
    for (component, s) in stress.into_iter().enumerate() {
        let a = component / 3;
        let b = component % 3;
        let raw = if dimensional {
            strain[component]
        } else {
            0.5 * (strain[component] + strain[3 * b + a])
        };
        assert!((raw / measure - s).abs() < 1e-14);
    }
    let mut derivatives: Vec<_> = tensor(&analytic, "forces").iter().map(|f| -f).collect();
    derivatives.extend(strain);
    let label = format!("{reference}/{xc} {pbc:?}");
    let names = (0..gradient_components)
        .map(|i| format!("{label} gradient[{},{}]", i / 3, i % 3))
        .chain((0..9).map(|i| format!("{label} affine[{},{}]", i / 3, i % 3)))
        .collect::<Vec<_>>();
    println!("Dimensional geometry {label} {}", analytic.to_json());
    let energy_options = options.with("forces", false);
    let sample = |i, h| {
        number(
            &checked_report(
                "periodic-energy",
                reference,
                i,
                h,
                &energy_options,
                &displace(base.clone(), i, h),
            ),
            "energy",
        )
    };
    if let Some(stages) = stages {
        fit_energy_stages(&label, &names, &derivatives, stages, sample);
    } else {
        fit_energies(&label, &names, &derivatives, sample);
    }
}

#[test]
fn dimensional_collinear_rks_wire_geometry() {
    check_dimensional_geometry(
        "rks",
        "pbe0",
        [true, false, false],
        "hartree/bohr",
        [2, 1, 1],
    );
}

#[test]
fn dimensional_collinear_rks_slab_geometry() {
    check_dimensional_geometry(
        "rks",
        "hf",
        [true, false, true],
        "hartree/bohr^2",
        [2, 1, 1],
    );
}

#[test]
fn dimensional_collinear_uks_wire_geometry() {
    check_dimensional_geometry("uks", "hf", [true, false, false], "hartree/bohr", [2, 1, 1]);
}

#[test]
fn dimensional_collinear_uks_slab_geometry() {
    check_dimensional_geometry(
        "uks",
        "tpss",
        [true, false, true],
        "hartree/bohr^2",
        [2, 1, 1],
    );
}

#[test]
#[ignore = "Long finite-k VV10 wire scan over all force and strain components"]
fn dimensional_vv10_rks_wire_geometry() {
    check_dimensional_geometry(
        "rks",
        "wb97x-v",
        [true, false, false],
        "hartree/bohr",
        [3, 1, 1],
    );
}

#[test]
#[ignore = "Long finite-k VV10 slab scan over all force and strain components"]
fn dimensional_vv10_uks_slab_geometry() {
    check_dimensional_geometry(
        "uks",
        "wb97m-v",
        [true, false, true],
        "hartree/bohr^2",
        [3, 1, 1],
    );
}

#[test]
#[ignore = "Long finite-k VV10 scan over all force and strain components"]
fn periodic_vv10_roks_wire_geometry() {
    check_dimensional_geometry(
        "roks",
        "wb97x-v",
        [true, false, false],
        "hartree/bohr",
        [3, 1, 1],
    );
}

#[test]
#[ignore = "Long Gamma VV10 scan over all force and strain components"]
fn periodic_vv10_roks_slab_geometry() {
    check_dimensional_geometry(
        "roks",
        "wb97m-v",
        [true, false, true],
        "hartree/bohr^2",
        [1, 1, 1],
    );
}

#[test]
#[ignore = "Long Gamma VV10 scan over all force and strain components"]
fn periodic_vv10_roks_bulk_geometry() {
    check_dimensional_geometry("roks", "wb97x-v", [true; 3], "hartree/bohr^3", [1; 3]);
}

fn nearest_atom_images(system: &System) -> Vec<[i64; 3]> {
    let lattice = lattice_of(system).unwrap();
    let positions: Vec<_> = system
        .positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let reference_lattice = &lattice;
    positions
        .iter()
        .flat_map(|&a| {
            positions.iter().map(move |&b| {
                let separation = b - a;
                reference_lattice
                    .to_fractional(reference_lattice.minimum_image(separation) - separation)
                    .map(|index| index.round() as i64)
            })
        })
        .collect()
}

#[test]
#[ignore = "Long Gamma VV10 scan with audited fixed exchange images"]
fn periodic_vv10_roks_bulk_local_chart_geometry() {
    with_basis("bulk-vv10-roks-local-chart", |basis| {
        let (base, options) = dimensional_fixture("roks", "wb97x-v", [true; 3], [1; 3], basis);
        let reference = nearest_atom_images(&base);
        // Beyond h = -0.00394308 for affine[z,x], the Li--H pair changes
        // its minimum image. Keep every independent channel in the same
        // exchange chart and validate the identical unperturbed reference.
        let steps = [0.0035, 0.0025, 0.00175, 0.00125];
        for component in 0..(3 * base.positions.len() + 9) {
            for h in steps.into_iter().flat_map(|h| [h, -h]) {
                let displaced = displace(base.clone(), component, h);
                assert_eq!(
                    nearest_atom_images(&displaced),
                    reference,
                    "exchange images changed at component={component} h={h}"
                );
            }
        }
        let old_crossing = displace(base.clone(), 3, 0.01);
        assert_ne!(nearest_atom_images(&old_crossing), reference);
        fit_geometry_with_stages(
            "roks",
            "wb97x-v",
            "hartree/bohr^3",
            &base,
            options,
            Some(&[(steps, Stencil::Second)]),
        );
    });
}

#[test]
#[ignore = "Independent Gamma SCF at four steps in every force and strain component"]
fn periodic_chg_roks_gamma_force_and_stress_fits() {
    check_roks_chg_geometry([1; 3]);
}

#[test]
#[ignore = "Independent finite-k SCF at four steps in every force and strain component"]
fn periodic_chg_roks_mesh_force_and_stress_fits() {
    check_roks_chg_geometry([2, 1, 1]);
}

fn check_roks_chg_geometry(mesh: [usize; 3]) {
    with_basis(&format!("bulk-chg-roks-{mesh:?}"), |basis| {
        let (mut base, options) = dimensional_fixture("roks", "wb97x-d", [true; 3], mesh, basis);
        base.cell = Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]);
        let options = options.with("grid_level", 0).with("disp_cutoff", 5.4);
        let centre = run("stress", &base, &Controls::from_value(&options).unwrap()).unwrap();
        let correction = number(&centre, "dispersion_energy");
        assert!(correction < 0.0);
        assert!(
            (number(&centre, "energy")
                - number(&centre, "electronic_energy")
                - number(&centre, "nuclear_energy")
                - correction)
                .abs()
                < 1e-12
        );
        let settings = dft_rs::periodic::dispersion::PeriodicDispersion::from_model(
            &dft_rs::dispersion::Dispersion::Chg,
            "wb97x-d",
            40.0,
            5.4,
        )
        .unwrap()
        .unwrap();
        let positions: Vec<_> = base
            .positions
            .iter()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();
        let direct = dft_rs::periodic::dispersion::periodic_d3(
            &base.numbers,
            &positions,
            &lattice_of(&base).unwrap(),
            &settings,
        )
        .unwrap();
        assert!((direct.energy - correction).abs() < 1e-14);
        let terms = centre.get("strain_derivative_terms").unwrap();
        let reported = tensor(terms, "dispersion");
        for (actual, expected) in reported
            .into_iter()
            .zip(direct.strain_derivative.into_iter().flatten())
        {
            assert!((actual - expected).abs() < 1e-14);
        }
        fit_geometry("roks", "wb97x-d", "hartree/bohr^3", &base, options);
    });
}

#[test]
#[ignore = "Independent wire and slab SCF for every force and strain component"]
fn lower_dimensional_chg_roks_force_and_stress_fits() {
    with_basis("dimensional-chg-roks", |basis| {
        // For slab zx the coarse central errors contain sizable fourth-order
        // terms. Keep the original stages and check the resolved local window.
        let slab_stages: Vec<_> = GEOMETRY_FIT_STAGES
            .into_iter()
            .chain([([0.0035, 0.0025, 0.00175, 0.00125], Stencil::Second)])
            .collect();
        for (pbc, unit) in [
            ([true, false, false], "hartree/bohr"),
            ([true, false, true], "hartree/bohr^2"),
        ] {
            let (base, options) = dimensional_fixture("roks", "wb97x-d", pbc, [2, 1, 1], basis);
            let options = options
                .with("grid_level", 0)
                .with("grid_tolerance", 0.1)
                .with("disp_cutoff", 5.4);
            fit_geometry_with_stages(
                "roks",
                "wb97x-d",
                unit,
                &base,
                options,
                pbc[2].then_some(slab_stages.as_slice()),
            );
        }
    });
}

#[test]
#[ignore = "Independent reconverged spinor SCFs for all VV10 force and strain components"]
fn periodic_gks_vv10_fits_all_force_and_strain_components() {
    with_basis("periodic-gks-vv10", |basis| {
        for (pbc, unit, xc) in [
            ([true, false, false], "hartree/bohr", "wb97x-v"),
            ([true, false, true], "hartree/bohr^2", "wb97m-v"),
            ([true, true, true], "hartree/bohr^3", "wb97x-v"),
        ] {
            let (base, controls) = dimensional_fixture("gks", xc, pbc, [2, 1, 1], basis);
            fit_geometry("gks", xc, unit, &base, controls.with("grid_tolerance", 0.1));
        }
    });
}

#[test]
fn dimensional_dispersion_fits_against_an_independent_vacuum_embedding() {
    use dft_rs::dispersion::Dispersion;
    use dft_rs::driver::periodic::dimensional_geometry_of;
    use dft_rs::periodic::dispersion::{dimensional_dispersion, periodic_d3, PeriodicDispersion};
    use dft_rs::periodic::Lattice;

    let as_vec = |p: [f64; 3]| Vec3::new(p[0], p[1], p[2]);
    for pbc in [[true, false, false], [true, false, true]] {
        let mut base = dimensional_system("roks", COMPONENTS, 0.0, pbc);
        base.numbers = vec![6, 7, 8];
        base.positions.push([-0.46, 1.29, 1.62]);
        let geometry = dimensional_geometry_of(&base).unwrap();
        for kind in [Dispersion::D2, Dispersion::D3BjAbc, Dispersion::Chg] {
            let mut model = PeriodicDispersion::from_model(&kind, "pbe", 6.4, 6.4)
                .unwrap()
                .unwrap();
            if let Some(term) = &mut model.three_body {
                term.cutoff = 6.4;
            }
            let positions: Vec<_> = base.positions.iter().copied().map(as_vec).collect();
            let result =
                dimensional_dispersion(&base.numbers, &positions, &geometry, &model).unwrap();
            assert!(result.energy < 0.0);
            let derivatives: Vec<_> = result
                .forces
                .iter()
                .flat_map(|force| (-*force).to_array())
                .chain(result.strain_derivative.into_iter().flatten())
                .collect();
            let label = format!("{kind:?} geometry {pbc:?}");
            let labels: Vec<_> = (0..derivatives.len())
                .map(|i| format!("{label} component {i}"))
                .collect();
            fit_energy_stages(
                &label,
                &labels,
                &derivatives,
                &[([0.008, 0.006, 0.004, 0.002], Stencil::Second)],
                |i, h| {
                    let sample = displace(base.clone(), i, h);
                    let positions: Vec<_> = sample.positions.iter().copied().map(as_vec).collect();
                    let [a, _, c] = sample.cell.unwrap().map(as_vec);
                    let normal = if pbc[2] {
                        a.cross(c)
                    } else {
                        a.cross(Vec3::new(0.0, 0.0, 1.0))
                    };
                    let b = normal * (40.0 / normal.norm());
                    let c = if pbc[2] {
                        c
                    } else {
                        let cd = a.cross(b);
                        cd * (40.0 / cd.norm())
                    };
                    periodic_d3(&sample.numbers, &positions, &Lattice::new(a, b, c), &model)
                        .unwrap()
                        .energy
                },
            );
        }
    }
}

fn mixed_difference(
    positions: &mut [[f64; 3]],
    coordinates: &[usize],
    h: f64,
    stencil: Stencil,
    oracle: &impl Fn(&[[f64; 3]]) -> f64,
) -> f64 {
    let Some((&coordinate, rest)) = coordinates.split_first() else {
        return oracle(positions);
    };
    let (atom, axis) = (coordinate / 3, coordinate % 3);
    let origin = positions[atom][axis];
    let mut evaluate = |offset| {
        positions[atom][axis] = origin + offset * h;
        mixed_difference(positions, rest, h, stencil, oracle)
    };
    let (plus, minus) = (evaluate(1.0), evaluate(-1.0));
    let (plus2, minus2) = if stencil == Stencil::Fourth {
        (evaluate(2.0), evaluate(-2.0))
    } else {
        (0.0, 0.0)
    };
    positions[atom][axis] = origin;
    stencil.first(plus, minus, plus2, minus2, h)
}

fn d1_pair_oracle(
    atoms: [dft_rs::dispersion::d1::AtomParameters; 2],
    displacement: [f64; 3],
    scale: f64,
    gradient_axis: Option<usize>,
) -> f64 {
    let [a, b] = atoms;
    let radius = a.radius + b.radius;
    let distance = displacement.iter().map(|x| x * x).sum::<f64>().sqrt();
    let damping = 1.0 / (1.0 + (23.0 * (1.0 - distance / radius)).exp());
    let c6 = 2.0 * a.c6 * b.c6 / (a.c6 + b.c6);
    let energy = -scale * c6 * damping / distance.powi(6);
    gradient_axis.map_or(energy, |axis| {
        energy * (23.0 * (1.0 - damping) / radius - 6.0 / distance) * displacement[axis] / distance
    })
}

#[test]
fn molecular_d1_nuclear_derivatives_through_order_four() {
    use dft_rs::dispersion::{
        d1, dispersion_energy, dispersion_gradient, dispersion_hessian, Dispersion,
    };
    use dft_rs::response::series::evaluate_series_many_result;
    use dft_rs::system::{Atom, Molecule};

    // Noncoplanar distorted ammonia. Keep the same free-atom inputs on all H sites.
    let positions = [
        [0.13, -0.18, 0.24],
        [1.74, 0.53, -0.48],
        [-1.19, 1.36, -0.16],
        [-0.60, -1.68, -0.79],
    ];
    let molecule = Molecule::new(
        [7, 1, 1, 1]
            .into_iter()
            .zip(positions)
            .map(|(z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    );
    let mut atoms = [d1::AtomParameters {
        c6: 6.5,
        radius: 2.4,
    }; 4];
    atoms[0] = d1::AtomParameters {
        c6: 23.0,
        radius: 3.0,
    };
    let scale = 0.81;
    let model = Dispersion::D1 {
        atoms: atoms.to_vec(),
        s6: scale,
    };
    let oracle = |coords: &[[f64; 3]], component: Option<usize>| {
        let mut total = 0.0;
        for i in 0..coords.len() {
            for j in 0..i {
                let owners =
                    component.map_or(1.0, |k| f64::from(k / 3 == i) - f64::from(k / 3 == j));
                let displacement = std::array::from_fn(|a| coords[i][a] - coords[j][a]);
                total += owners
                    * d1_pair_oracle(
                        [atoms[i], atoms[j]],
                        displacement,
                        scale,
                        component.map(|k| k % 3),
                    );
            }
        }
        total
    };
    let energy = dispersion_energy(&molecule, &model, "pbe").unwrap();
    assert!((energy - oracle(&positions, None)).abs() < 2e-14);
    let gradient = dispersion_gradient(&molecule, &model, "pbe").unwrap();
    let hessian = dispersion_hessian(&molecule, &model, "pbe").unwrap();
    let dimension = 3 * positions.len();
    let mut failures = Vec::new();
    let stages = [
        ([0.018, 0.012, 0.008, 0.005], Stencil::Second),
        ([0.008, 0.006, 0.004, 0.003], Stencil::Second),
        ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
        ([0.075, 0.05, 0.035, 0.025], Stencil::Fourth),
    ];
    for order in 1..=4_usize {
        let mut comps = Vec::new();
        let mut zeros = 0;
        for flat in 0..dimension.pow(order as u32) {
            let mut index = flat;
            let mut axes = vec![0; order];
            for coordinate in axes.iter_mut().rev() {
                *coordinate = index % dimension;
                index /= dimension;
            }
            if !axes.windows(2).all(|p| p[0] <= p[1]) {
                continue;
            }
            let analytic = match axes.as_slice() {
                [k] => gradient[*k / 3].to_array()[*k % 3],
                [k, m] => hessian[(*k, *m)],
                _ => (1_usize..(1 << order))
                    .map(|mask| {
                        // Polarize the factorial-normalized coefficient over every
                        // subset of these nuclear displacement directions.
                        let mut inputs = vec![vec![0.0; order + 1]; dimension];
                        for (input, value) in inputs.iter_mut().zip(positions.iter().flatten()) {
                            input[0] = *value;
                        }
                        for (slot, &coordinate) in axes.iter().enumerate() {
                            if mask & (1 << slot) != 0 {
                                inputs[coordinate][1] += 1.0;
                            }
                        }
                        let coeff = evaluate_series_many_result(&inputs, |variables| {
                            let coords: Vec<_> = variables
                                .chunks_exact(3)
                                .map(|c| [c[0], c[1], c[2]])
                                .collect();
                            Ok(vec![d1::energy_geometry(&coords, &atoms, scale)?])
                        })
                        .unwrap()[0][order];
                        if (order - mask.count_ones() as usize).is_multiple_of(2) {
                            coeff
                        } else {
                            -coeff
                        }
                    })
                    .sum(),
            };
            let owners = axes.iter().fold(0_u8, |mask, &k| mask | (1 << (k / 3)));
            if owners.count_ones() > 2 {
                assert!(
                    analytic.abs() < 2e-14,
                    "structural zero {axes:?}: {analytic}"
                );
                zeros += 1;
            } else {
                comps.push((axes, analytic));
            }
        }
        println!("Molecular D1 energy={energy:.17e} order {order}: {} fitted and {zeros} structural zeros", comps.len());
        let names: Vec<_> = comps
            .iter()
            .map(|(axes, _)| format!("molecular D1 {axes:?}"))
            .collect();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                comps
                    .iter()
                    .map(|(axes, value)| {
                        let (coordinate, derivatives) = if order == 1 {
                            (None, axes.as_slice())
                        } else {
                            (axes.last().copied(), &axes[..order - 1])
                        };
                        steps.map(|h| {
                            let mut displaced = positions;
                            (mixed_difference(
                                &mut displaced,
                                derivatives,
                                h,
                                stencil,
                                &|r| oracle(r, coordinate),
                            ) - value)
                                .abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures("molecular D1 nuclear derivatives", &failures);
}

#[test]
fn dimensional_d1_nuclear_derivatives_through_order_four() {
    use dft_rs::dispersion::d1::AtomParameters;
    use dft_rs::driver::periodic::dimensional_geometry_of;
    use dft_rs::periodic::dispersion::{
        dimensional_dispersion_force_constant_tensor, PeriodicDispersion,
    };

    let numbers = [6, 7, 8];
    let atoms = [
        AtomParameters {
            c6: 19.0,
            radius: 2.8,
        },
        AtomParameters {
            c6: 23.0,
            radius: 2.95,
        },
        AtomParameters {
            c6: 17.0,
            radius: 3.1,
        },
    ];
    let (scale, cutoff) = (0.81, 6.4);
    let settings = PeriodicDispersion::d1(&numbers, &atoms, scale, cutoff).unwrap();
    let stages = [
        ([0.018, 0.012, 0.008, 0.005], Stencil::Second),
        ([0.008, 0.006, 0.004, 0.003], Stencil::Second),
        ([0.028, 0.021, 0.014, 0.009], Stencil::Fourth),
        ([0.016, 0.012, 0.009, 0.006], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    for pbc in [[true, false, false], [true, false, true], [true; 3]] {
        let mut base = dimensional_system("roks", COMPONENTS, 0.0, pbc);
        base.numbers = numbers.to_vec();
        base.positions.push([-0.46, 1.29, 1.62]);
        if pbc == [true; 3] {
            for row in base.cell.as_mut().unwrap() {
                *row = row.map(|x| x * 0.55);
            }
        }
        let geometry = dimensional_geometry_of(&base).unwrap();
        let positions: Vec<_> = base
            .positions
            .iter()
            .map(|&[x, y, z]| Vec3::new(x, y, z))
            .collect();
        let lattice = base.cell.unwrap();
        let mut pairs = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for a in -3_i32..=3 {
                    for b in -3_i32..=3 {
                        for c in -3_i32..=3 {
                            let image = [a, b, c];
                            if (i == j && image == [0; 3])
                                || image.iter().zip(pbc).any(|(&m, active)| m != 0 && !active)
                            {
                                continue;
                            }
                            let t: [f64; 3] = std::array::from_fn(|axis| {
                                (0..3).map(|k| f64::from(image[k]) * lattice[k][axis]).sum()
                            });
                            let distance2: f64 = (0..3)
                                .map(|axis| {
                                    (base.positions[j][axis] - base.positions[i][axis] + t[axis])
                                        .powi(2)
                                })
                                .sum();
                            if distance2 < cutoff * cutoff {
                                assert!(image.into_iter().all(|m| m.abs() < 3));
                                pairs.push((i, j, t));
                            }
                        }
                    }
                }
            }
        }
        // Explicit image sum and independent closed-form Cartesian gradient.
        let oracle = |coords: &[[f64; 3]], component: Option<usize>| {
            pairs
                .iter()
                .map(|&(i, j, t)| {
                    let d: [f64; 3] = std::array::from_fn(|a| coords[j][a] - coords[i][a] + t[a]);
                    let sign =
                        component.map_or(1.0, |k| f64::from(k / 3 == j) - f64::from(k / 3 == i));
                    0.5 * sign
                        * d1_pair_oracle([atoms[i], atoms[j]], d, scale, component.map(|k| k % 3))
                })
                .sum::<f64>()
        };
        for order in 1..=4 {
            let result = dimensional_dispersion_force_constant_tensor(
                &numbers, &positions, &geometry, &settings, order,
            )
            .unwrap();
            assert!((result.energy - oracle(&base.positions, None)).abs() < 2e-13);
            assert!(result.permutation_residual < 2e-11);
            assert!(result.acoustic_residual < 2e-11);
            println!(
                "D1 {pbc:?} order {order}: energy={:.17e} permutation={:.3e} acoustic={:.3e}",
                result.energy, result.permutation_residual, result.acoustic_residual
            );
            let mut entries = Vec::new();
            let mut zeros = 0;
            for (flat, &analytic) in result.tensor.iter().enumerate() {
                let mut remainder = flat;
                let mut indices = vec![0; order];
                for index in indices.iter_mut().rev() {
                    *index = remainder % result.dimension;
                    remainder /= result.dimension;
                }
                if !indices.windows(2).all(|pair| pair[0] <= pair[1]) {
                    continue;
                }
                // No pair term depends on three distinct nuclear centres.
                let owners = indices.iter().fold(0_u8, |mask, &k| mask | (1 << (k / 3)));
                if owners.count_ones() > 2 {
                    assert_eq!(analytic, 0.0);
                    zeros += 1;
                } else {
                    entries.push((indices, analytic));
                }
            }
            println!(
                "D1 {pbc:?} order {order}: {} fitted and {zeros} structural zeros",
                entries.len()
            );
            let names: Vec<_> = entries
                .iter()
                .map(|(indices, _)| format!("D1 {pbc:?} {indices:?}"))
                .collect();
            stencils::fit_components_staged(
                &names,
                &stages,
                |steps, stencil| {
                    entries
                        .iter()
                        .map(|(indices, analytic)| {
                            let (component, coordinates) = if order == 1 {
                                (None, indices.as_slice())
                            } else {
                                (indices.last().copied(), &indices[..order - 1])
                            };
                            steps.map(|h| {
                                let mut coords = base.positions.clone();
                                (mixed_difference(&mut coords, coordinates, h, stencil, &|x| {
                                    oracle(x, component)
                                }) - analytic)
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("periodic D1 nuclear derivatives", &failures);
}

#[test]
fn dimensional_roks_wire_geometry() {
    check_dimensional_geometry(
        "roks",
        "hf",
        [true, false, false],
        "hartree/bohr",
        [1, 1, 1],
    );
}

#[test]
fn dimensional_roks_slab_geometry() {
    check_dimensional_geometry(
        "roks",
        "pbe",
        [true, false, true],
        "hartree/bohr^2",
        [1, 1, 1],
    );
}

#[test]
fn mesh_roks_wire_geometry() {
    check_dimensional_geometry(
        "roks",
        "hf",
        [true, false, false],
        "hartree/bohr",
        [3, 1, 1],
    );
}

#[test]
fn mesh_roks_slab_geometry() {
    check_dimensional_geometry(
        "roks",
        "pbe",
        [true, false, true],
        "hartree/bohr^2",
        [3, 1, 1],
    );
}

#[test]
fn mesh_roks_bulk_geometry() {
    check_dimensional_geometry("roks", "hf", [true; 3], "hartree/bohr^3", [3, 1, 1]);
}

fn fourier_field_geometry(reference: &str, pbc: [bool; 3], unit: &str, mesh: [usize; 3]) {
    with_basis(&format!("fourier-field-{reference}-{pbc:?}"), |basis| {
        let mut base = dimensional_system(reference, COMPONENTS, 0.0, pbc);
        base.numbers = if reference == "rks" {
            vec![1, 3, 2]
        } else {
            vec![1, 2, 2]
        };
        base.positions.push([-0.46, 1.29, 1.62]);
        let indices =
            std::array::from_fn::<_, 3, _>(|axis| if pbc[axis] { [1, -1, 1][axis] } else { 0 });
        let options = controls("gks", basis)
            .with("reference", reference)
            .with("multiplicity", if reference == "rks" { 1 } else { 2 })
            .with("xc", "hf")
            .with("k_mesh", mesh)
            .with("electrostatic_order", 48)
            .with("overlap_cutoff", 3.8)
            .with("exchange_cutoff", 3.8)
            .with("forces", true)
            .with("field_wave_indices", indices)
            .with("field_scalar", [0.065, -0.047])
            .with("field_vector_real", [0.023, -0.039, 0.028])
            .with("field_vector_imag", [-0.017, 0.012, 0.031])
            .with("field_amplitude", 0.74);
        fit_geometry(
            reference,
            "hf-fourier-electric-orbital",
            unit,
            &base,
            options,
        );
    });
}

#[test]
fn fourier_field_uks_wire_geometry() {
    fourier_field_geometry("uks", [true, false, false], "hartree/bohr", [3, 1, 1]);
}

#[test]
fn fourier_field_rks_slab_geometry() {
    fourier_field_geometry("rks", [true, false, true], "hartree/bohr^2", [1; 3]);
}

#[test]
fn fourier_field_rks_bulk_geometry() {
    fourier_field_geometry("rks", [true; 3], "hartree/bohr^3", [1; 3]);
}

fn check_scalar_ecp_geometry(reference: &str, pbc: [bool; 3], unit: &str) {
    let basis_text = format!(
        "{BASIS}Na 0\nS 1 1.0\n0.7 1.0\n****\nNa 0\nNa-ECP 1 10\np potential\n1\n2 0.65 0.75\ns-p potential\n1\n2 0.91 -0.40\n"
    );
    with_basis_text(&format!("ecp-{reference}-{pbc:?}"), &basis_text, |basis| {
        let bulk = pbc == [true; 3];
        let mut options = if bulk {
            controls(reference, basis)
        } else {
            controls("gks", basis).with("electrostatic_order", 48)
        }
        .with("reference", reference)
        .with("multiplicity", if reference == "rks" { 1 } else { 2 })
        .with("xc", "hf")
        .with(
            "k_mesh",
            if reference == "rks" || reference == "uks" {
                [3, 1, 1]
            } else {
                [1; 3]
            },
        )
        .with("overlap_cutoff", 3.8)
        .with("exchange_cutoff", 3.8)
        .with("forces", true);
        if reference == "roks" {
            options.set("damping", 0.0);
        }
        let mut base = dimensional_system(reference, COMPONENTS, 0.0, pbc);
        base.numbers = vec![1, 11, if reference == "rks" { 2 } else { 1 }];
        base.positions.push([-0.46, 1.29, 1.62]);
        if bulk {
            for vector in base.cell.as_mut().unwrap() {
                *vector = vector.map(|x| x * 0.55);
            }
        }
        fit_geometry(reference, "hf-scalar-ecp", unit, &base, options);
    });
}

#[test]
fn scalar_ecp_rks_bulk_geometry() {
    check_scalar_ecp_geometry("rks", [true; 3], "hartree/bohr^3");
}

#[test]
fn scalar_ecp_uks_wire_geometry() {
    check_scalar_ecp_geometry("uks", [true, false, false], "hartree/bohr");
}

#[test]
fn scalar_ecp_roks_slab_geometry() {
    check_scalar_ecp_geometry("roks", [true, false, true], "hartree/bohr^2");
}

#[test]
fn scalar_ecp_gks_wire_geometry() {
    check_scalar_ecp_geometry("gks", [true, false, false], "hartree/bohr");
}

fn dos_fixture(reference: &str, pbc: [bool; 3], open: bool) -> Value {
    with_basis(&format!("dos-{reference}-{pbc:?}-{open}"), |basis| {
        let dimensional = reference == "gks" || pbc != [true; 3];
        let mut settings = if dimensional {
            controls("gks", basis).with("electrostatic_order", 48)
        } else {
            controls(reference, basis)
        }
        .with("reference", reference)
        .with("multiplicity", if open { 2 } else { 1 })
        .with("xc", "hf")
        .with("k_mesh", [3, 1, 1])
        .with("sigma", 0.06)
        .with("points", 1201)
        .with("energy_window", [-8.0, 8.0])
        .with("projected", true);
        if reference == "roks" {
            settings.set("damping", 0.0);
        }
        let fixture = System {
            numbers: vec![1, if open { 2 } else { 1 }],
            positions: vec![[0.13, -0.21, 0.07], [1.50, 0.27, -0.31]],
            cell: Some([[6.20, 0.46, 0.27], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
            pbc,
        };
        run("dos", &fixture, &Controls::from_value(&settings).unwrap()).unwrap()
    })
}

fn dos_curve_and_sum_rule(report: &Value, reference: &str) -> Vec<f64> {
    assert_eq!(
        report.get("reference_type").and_then(Value::as_str),
        Some(reference)
    );
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    assert_eq!(number(report, "sampling_k_points"), 3.0);
    let energies = report.get("energies").unwrap().as_array().unwrap();
    let channels = report.get("channels").unwrap().as_array().unwrap();
    assert_eq!(
        channels.len(),
        if matches!(reference, "uks" | "roks") {
            2
        } else {
            1
        }
    );
    let mut total_curve = vec![0.0; energies.len()];
    let mut count = 0.0;
    for channel in channels {
        let expected = if channels.len() == 1 { 4.0 } else { 2.0 };
        assert!((number(channel, "state_count") - expected).abs() < 1e-12);
        assert!((number(channel, "window_state_count") - expected).abs() < 1e-12);
        assert!(number(channel, "integration_residual").abs() < 1e-10);
        assert!(number(channel, "projection_sum_residual").abs() < 1e-10);
        let last = channel
            .get("integrated")
            .unwrap()
            .as_array()
            .unwrap()
            .last()
            .unwrap();
        count += last.as_f64().unwrap();
        for (total, value) in total_curve
            .iter_mut()
            .zip(channel.get("dos").unwrap().as_array().unwrap())
        {
            *total += value.as_f64().unwrap();
        }
    }
    assert!((count - 4.0).abs() < 1e-10);
    println!(
        "Electronic DOS {reference} energy={:.17e} count={count:.17e} peak={:.17e}",
        number(report, "energy"),
        total_curve.iter().copied().fold(0.0, f64::max)
    );
    total_curve
}

#[test]
fn periodic_dos_closed_shell_reference_limit_and_state_count() {
    let pbc = [true, false, false];
    let rks = dos_fixture("rks", pbc, false);
    let expected = dos_curve_and_sum_rule(&rks, "rks");
    for reference in ["uks", "roks", "gks"] {
        let report = dos_fixture(reference, pbc, false);
        let curve = dos_curve_and_sum_rule(&report, reference);
        assert!((number(&rks, "energy") - number(&report, "energy")).abs() < 1e-9);
        assert!(expected
            .iter()
            .zip(curve)
            .all(|(a, b)| (a - b).abs() < 1e-7));
    }
}

#[test]
fn periodic_dos_open_shell_slab_and_bulk_state_counts() {
    dos_curve_and_sum_rule(&dos_fixture("roks", [true, false, true], true), "roks");
    dos_curve_and_sum_rule(&dos_fixture("uks", [true; 3], true), "uks");
}
