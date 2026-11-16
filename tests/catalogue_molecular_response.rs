// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Catalogue-driven all-component tests of two common molecular routes.
//! Every canonical named provider is exercised with its physical terms.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::xc::catalogue::{catalogue, FunctionalClass, FunctionalInfo};
use std::cell::Cell;
use stencils::Stencil;

const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.78, 0.34, -0.08],
    [-0.49, 1.65, 0.41],
    [0.31, -0.78, 1.91],
];
const NUMBERS: [u8; 4] = [7, 1, 1, 1];

fn request(functional: &FunctionalInfo, reference: &str) -> Value {
    let mut options = Value::object()
        .with("reference", reference)
        .with("basis", "sto-3g")
        .with("grid_level", 0)
        .with("vv10_grid_level", 0)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-12)
        .with("max_iter", 300);
    if reference != "rks" {
        options.set("charge", 1);
        options.set("multiplicity", 2);
    }
    match functional.class {
        FunctionalClass::Exchange => {
            options.set("x", functional.name);
            options.set("c", "none");
        }
        FunctionalClass::Correlation => {
            options.set("x", "slater");
            options.set("c", functional.name);
        }
        _ => options.set("xc", functional.name),
    }
    options
}

fn calculate(feature: &str, system: &System, options: &Value) -> Result<Value, String> {
    let result = run(
        feature,
        system,
        &Controls::from_value(options).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if result.get("converged").and_then(Value::as_bool) != Some(true) {
        return Err(format!("{feature} did not converge"));
    }
    Ok(result)
}

fn entries(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn orbital_residual(report: &Value) -> Result<f64, String> {
    [
        "scf_orbital_residual",
        "orbital_residual",
        "reference_orbital_residual",
        "stationarity_defect",
    ]
    .into_iter()
    .find_map(|key| report.get(key).and_then(Value::as_f64))
    .filter(|value| value.is_finite() && *value >= 0.0)
    .ok_or_else(|| "missing finite orbital residual in Hartree".into())
}

fn fit(
    label: &str,
    target: &[f64],
    width: usize,
    samples: impl Fn(usize, f64) -> Result<Vec<f64>, String>,
    stages: &[([f64; 4], Stencil)],
    failures: &mut Vec<String>,
) {
    assert_eq!(target.len() % width, 0);
    let sources = target.len() / width;
    let names: Vec<_> = (0..sources)
        .flat_map(|j| (0..width).map(move |i| format!("{label} [{i},{j}]")))
        .collect();
    let mut cache = std::collections::BTreeMap::new();
    let mut sample_errors = Vec::new();
    stencils::fit_components_staged_with_accuracy(
        &names,
        stages,
        |steps, stencil| {
            let mut estimates = vec![[f64::NAN; 4]; target.len()];
            for j in 0..sources {
                for (k, &h) in steps.iter().enumerate() {
                    let offsets = match stencil {
                        Stencil::Second => vec![h, -h],
                        Stencil::Fourth => vec![h, -h, 2.0 * h, -2.0 * h],
                    };
                    let mut values = Vec::with_capacity(offsets.len());
                    for offset in offsets {
                        let sample = cache.entry((j, offset.to_bits())).or_insert_with(|| {
                            let result = samples(j, offset);
                            if let Err(error) = &result {
                                let detail = format!("{label} source={j} h={offset}: {error}");
                                println!("{detail}");
                                sample_errors.push(detail);
                            }
                            result
                        });
                        if let Ok(sample) = sample {
                            values.push(sample.clone());
                        }
                    }
                    if values.len() != if stencil == Stencil::Second { 2 } else { 4 } {
                        continue;
                    }
                    for i in 0..width {
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (values[2][i], values[3][i])
                        } else {
                            (0.0, 0.0)
                        };
                        estimates[j * width + i][k] =
                            stencil.first(values[0][i], values[1][i], plus2, minus2, h);
                    }
                }
            }
            let ratio = (steps[2] / steps[3]).powf(stencil.order());
            estimates
                .into_iter()
                .enumerate()
                .map(|(entry, values)| {
                    let exact = target[(entry % width) * sources + entry / width];
                    let error = ((ratio * values[3] - values[2]) / (ratio - 1.0) - exact).abs();
                    // Use the same absolute-plus-relative gate as the
                    // nuclear Taylor tests.
                    stencils::FitErrors {
                        errors: values.map(|value| (value - exact).abs()),
                        richardson: Some((error, 2e-7 + 1e-8 * exact.abs())),
                    }
                })
                .collect()
        },
        failures,
    );
    if !sample_errors.is_empty() {
        println!("{label} displaced failures retained: {sample_errors:?}");
    }
}

#[test]
fn an_interval_must_pass_both_convergence_and_accuracy() {
    let name = ["one component".to_string()];
    let coarse = [0.04, 0.03, 0.02, 0.01];
    let fine = [0.008, 0.006, 0.004, 0.002];
    let stages = [(coarse, Stencil::Second), (fine, Stencil::Second)];
    let mut failures = Vec::new();
    stencils::fit_components_staged(&name, &stages, |h, _| vec![h.map(|v| v * v)], &mut failures);
    assert!(failures.is_empty());
    stencils::fit_components_staged_with_accuracy(
        &name,
        &stages,
        |h, _| {
            vec![if *h == coarse {
                stencils::FitErrors {
                    errors: h.map(|v| v * v),
                    richardson: Some((4e-7, 2e-7)),
                }
            } else {
                stencils::FitErrors {
                    errors: h.map(|v| v * v * v * v),
                    richardson: Some((1e-8, 2e-7)),
                }
            }]
        },
        &mut failures,
    );
    assert_eq!(failures, name);
}

fn check_catalogue(reference: &str, electric: bool) {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let mut failures = Vec::new();
    let mut count = 0;
    let providers = catalogue().unwrap();
    let selected = std::env::var("DFT_RS_TEST_FUNCTIONAL").ok();
    // Record the mandatory DFT witnesses early, then exercise the whole
    // catalogue without treating other entries as optional.
    let mut ordered: Vec<_> = providers
        .iter()
        .filter(|f| selected.as_deref().is_none_or(|name| f.name == name))
        .collect();
    assert!(
        !ordered.is_empty(),
        "Unknown functional selection {selected:?}"
    );
    let expected = ordered.len();
    ordered.sort_by_key(|f| match f.name {
        "wb97x-d" => 0,
        "wb97x-v" => 1,
        name if name.starts_with("m06") || name.starts_with("revm06") => 2,
        "pbe" | "tpss" => 3,
        _ => 4,
    });
    for functional in ordered {
        // The ROKS P86 field oracle works inside one local-correlation
        // branch. Tighten the stationary solves to resolve its small
        // off-diagonal finite-field signal without changing the fit gates.
        let precise_field = electric && reference == "roks" && functional.name == "p86";
        let mut options = request(functional, reference);
        if precise_field {
            options.set("conv_tol", 1e-14);
            options.set("conv_tol_dp", 1e-13);
            options.set("conv_tol_grad", 1e-13);
            options.set("max_iter", 500);
        }
        let mut central = options.clone();
        if precise_field {
            central.set("response_tol", 1e-12);
        }
        let (feature, field) = if electric {
            ("polarizability", "polarizability")
        } else {
            ("gradient", "gradient")
        };
        let label = format!(
            "{reference} {} {} {feature}",
            functional.class.as_str(),
            functional.name
        );
        println!("Starting {label}");
        let report = match calculate(feature, &system, &central) {
            Ok(report) => report,
            Err(error) => {
                failures.push(format!("{label}: {error}"));
                continue;
            }
        };
        let tensor = entries(&report, field);
        assert_eq!(tensor.len(), if electric { 9 } else { 12 });
        let orbital = match orbital_residual(&report) {
            Ok(value) => value,
            Err(error) => {
                failures.push(format!("{label}: {error}"));
                continue;
            }
        };
        println!(
            "{label}: values={tensor:?} raw_energy={:?} orbital_residual={orbital:.9e} response={:?}",
            report.get("energy"),
            report
                .get("response_absolute_residuals")
                .or_else(|| report.get("response_residuals"))
                .or_else(|| report.get("response_residual"))
                .or_else(|| report.get("residuals"))
        );
        let stages = if electric {
            &[
                ([0.016, 0.011, 0.007, 0.004], Stencil::Second),
                ([0.008, 0.005, 0.003, 0.002], Stencil::Second),
                ([0.022, 0.016, 0.011, 0.008], Stencil::Fourth),
                // A small-field window resolves higher-order dipole terms
                // that still exceed the Richardson gate on coarser fields.
                ([0.004, 0.0025, 0.0015, 0.001], Stencil::Second),
                // This interval resolves the local derivative below the
                // observed correlation-branch crossings in the P86 test.
                ([0.0005, 0.0003, 0.0002, 0.00015], Stencil::Second),
            ][..]
        } else {
            &[
                ([0.02, 0.014, 0.009, 0.006], Stencil::Second),
                ([0.01, 0.007, 0.004, 0.0025], Stencil::Second),
                ([0.03, 0.021, 0.014, 0.009], Stencil::Fourth),
                // Steep functional interpolations can still dominate the
                // earlier intervals with higher terms. Require the same
                // rate and accuracy gates in the small-step regime.
                ([0.003, 0.002, 0.0014, 0.001], Stencil::Second),
                // Moving quadrature points can cross a piecewise local
                // fit even on the preceding interval. Keep those results
                // and check the derivative in a smaller neighbourhood.
                ([0.001, 0.00075, 0.00055, 0.0004], Stencil::Second),
                ([0.0006, 0.00043, 0.00031, 0.00022], Stencil::Second),
                ([0.00032, 0.00027, 0.00022, 0.00017], Stencil::Second),
            ][..]
        };
        let max_displaced_residual = Cell::new(None::<f64>);
        fit(
            &label,
            &tensor,
            if electric { 3 } else { 1 },
            |component, h| {
                let (report, quantity) = if electric {
                    let mut field = [0.0; 3];
                    field[component] = h;
                    (
                        calculate("dipole", &system, &options.clone().with("field", field))?,
                        "dipole",
                    )
                } else {
                    let mut displaced = system.clone();
                    displaced.positions[component / 3][component % 3] += h;
                    (calculate("energy", &displaced, &options)?, "energy")
                };
                let residual = orbital_residual(&report)?;
                max_displaced_residual.set(Some(
                    max_displaced_residual
                        .get()
                        .map_or(residual, |previous| previous.max(residual)),
                ));
                Ok(entries(&report, quantity))
            },
            stages,
            &mut failures,
        );
        println!(
            "{label}: max displaced orbital residual (Hartree)={:?}",
            max_displaced_residual.get()
        );
        count += 1;
    }
    println!("Catalogue {reference} electric={electric} evaluated {count}/{expected} providers");
    components::assert_no_failures("named molecular catalogue", &failures);
    assert_eq!(count, expected);
}

#[test]
fn distinct_target_spinor_double_hybrid_gradient_matches_all_energy_slopes() {
    let settings = named_spinor_request("pbe0")
        .with("double_hybrid_model", "custom")
        .with("double_hybrid_target_xc", "pbe")
        .with("pt2_same_spin", 0.27)
        .with("pt2_opposite_spin", 0.27)
        .with("dispersion", "d3bj");
    check_spinor_double_hybrid_gradient(&settings, 0.27, None);
}

#[test]
#[ignore = "All-component acceptance with independently reconverged VV10 geometries"]
fn nonlocal_target_spinor_double_hybrid_gradient_matches_all_energy_slopes() {
    check_spinor_double_hybrid_gradient(
        &named_spinor_request("wb97m(2)"),
        0.34096,
        Some([0.65904, 10.0, 0.01]),
    );
}

fn named_spinor_request(name: &str) -> Value {
    let providers = catalogue().unwrap();
    request(providers.iter().find(|f| f.name == name).unwrap(), "gks")
}

fn check_spinor_double_hybrid_gradient(settings: &Value, weight: f64, nonlocal: Option<[f64; 3]>) {
    c1_geometry::assert_c1(&NUMBERS, &POSITIONS);
    let system = System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec());
    let mut channels = vec![
        "gradient",
        "parent_gradient",
        "generalized_pt2_gradient",
        "pt2_gradient",
        "target_gradient",
    ];
    if nonlocal.is_none() {
        channels.push("dispersion_gradient");
    }
    let number = |report: &Value, key| report.get(key).and_then(Value::as_f64).unwrap();
    let energy_channels = |report: &Value| {
        let terms = report.get("energy_components").unwrap();
        let pair = number(report, "generalized_pt2_correlation");
        let mut values = vec![
            number(report, "energy"),
            entries(terms, "spinor").iter().sum(),
            pair,
            weight * pair,
            number(terms, "target_correction"),
        ];
        if nonlocal.is_none() {
            values.push(number(terms, "dispersion") + number(terms, "model_dispersion"));
        }
        values
    };
    let gradient = calculate("gradient", &system, settings).unwrap();
    let energy = calculate("energy", &system, settings).unwrap();
    let reference = energy_channels(&energy);
    assert!((number(&gradient, "energy") - reference[0]).abs() < 1e-10);
    assert!(orbital_residual(&gradient).unwrap() < 1e-12);
    let exact: Vec<_> = channels.iter().map(|key| entries(&gradient, key)).collect();
    if let Some(parameters) = nonlocal {
        assert_eq!(entries(&gradient, "dispersion_gradient"), vec![0.0; 12]);
        for (key, expected) in ["target_vv10_scale", "target_vv10_b", "target_vv10_c"]
            .into_iter()
            .zip(parameters)
        {
            assert_eq!(number(&energy, key), expected);
            assert_eq!(number(&gradient, key), expected);
        }
    }
    let residuals = entries(&gradient, "response_residuals");
    assert_eq!(residuals.len(), 12);
    println!("NH3+ generalized double-hybrid channels={channels:?} energies={reference:?} gradients={exact:?} residuals={residuals:?}");
    // The fitter reads a row-major matrix with energy channels as rows.
    let targets: Vec<f64> = exact.iter().flatten().copied().collect();
    let mut failures = Vec::new();
    fit(
        "generalized double-hybrid nuclear gradient",
        &targets,
        channels.len(),
        |component, h| {
            let mut displaced = system.clone();
            displaced.positions[component / 3][component % 3] += h;
            let energy = calculate("energy", &displaced, settings)?;
            let residual = orbital_residual(&energy)?;
            if residual >= 1e-12 {
                return Err(format!(
                    "displaced orbital residual {residual:.12e} Hartree"
                ));
            }
            Ok(energy_channels(&energy))
        },
        &[
            ([0.020, 0.014, 0.009, 0.006], Stencil::Second),
            ([0.010, 0.007, 0.004, 0.0025], Stencil::Second),
        ],
        &mut failures,
    );
    components::assert_no_failures("generalized double-hybrid nuclear gradient", &failures);
}

#[test]
#[ignore = "Full catalogue acceptance sweep with reconverged displaced geometries"]
fn named_catalogue_rks_gradients() {
    check_catalogue("rks", false);
}

#[test]
#[ignore = "Full catalogue acceptance sweep with reconverged finite-field dipoles"]
fn named_catalogue_rks_polarizabilities() {
    check_catalogue("rks", true);
}

macro_rules! open_shell_catalogue {
    ($reference:literal, $gradient:ident, $polarizability:ident) => {
        #[test]
        #[ignore = "Full catalogue acceptance sweep with reconverged displaced geometries"]
        fn $gradient() {
            check_catalogue($reference, false);
        }

        #[test]
        #[ignore = "Full catalogue acceptance sweep with reconverged finite-field dipoles"]
        fn $polarizability() {
            check_catalogue($reference, true);
        }
    };
}

open_shell_catalogue!(
    "uks",
    named_catalogue_uks_gradients,
    named_catalogue_uks_polarizabilities
);
open_shell_catalogue!(
    "roks",
    named_catalogue_roks_gradients,
    named_catalogue_roks_polarizabilities
);
open_shell_catalogue!(
    "gks",
    named_catalogue_gks_gradients,
    named_catalogue_gks_polarizabilities
);
