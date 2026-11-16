// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Functional extensibility: five functionals added through the generic
//! drivers (PBEsol, RPBE, PW91, SCAN, LRC-wPBEh), with PBE and PBE0 as
//! controls, reach every response property through the same code.
//!
//! Oracles: frozen libxc energy densities and first derivatives of every
//! new component at nine points (`tools/oracle/gen_functional_points.py`;
//! the SCF energies are gated by `p3_dft_oracle` against
//! `tools/oracle/gen_dft.py`); four-step fits of every component of the
//! analytic gradient (9), Hessian (81), static polarizability (9), first
//! hyperpolarizability (27), periodic strain derivative (9) and the coupled
//! quantum-proton gradient (9) and Hessian (81) against central differences
//! of energies, gradients and field dipoles (a component whose fit fails on
//! the first step interval is recomputed on a finer and then a coarser one,
//! and every fit is printed);
//! the Taylor series of the semilocal kernels at order 4 against the dense
//! order-3 derivatives for every combined registry entry; the EPC17-2
//! interspecies functional against its closed form.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/molecular_fits.rs"]
mod molecular_fits;
#[path = "support/stencils.rs"]
mod stencils;

use components::{assert_no_failures, fit_components};

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::xc::taylor::XcTaylorInput;
use dft_rs::xc::{resolve, FunctionalSpec, GridVars, XcOut, COMBINED_NAMES};

const NEW: [&str; 5] = ["pbesol", "rpbe", "pw91", "scan", "lrc-wpbeh"];
const CONTROLS: [&str; 2] = ["pbe", "pbe0"];
/// One entry per component class added by the functional catalogue that the
/// five functionals above do not cover: a B95 meta-GGA hybrid and a B97
/// power-series hybrid. The regularized meta-GGA rSCAN joins its polynomial
/// switch with only three continuous derivatives, so its field and nuclear
/// derivatives do not follow a power law beyond that order; its response is
/// validated in `p280_rscan_validation`.
const CLASSES: [&str; 2] = ["pw6b95", "b97-1"];

fn molecular_set() -> impl Iterator<Item = &'static str> {
    NEW.iter().chain(&CONTROLS).chain(&CLASSES).copied()
}

const WATER: [(u8, [f64; 3]); 3] = [
    (8, [0.0, 0.0, 0.2217]),
    (1, [0.0, 1.4309, -0.8867]),
    (1, [0.13, -1.4309, -0.8867]),
];

fn system(atoms: &[(u8, [f64; 3])]) -> System {
    System::molecule(
        atoms.iter().map(|(z, _)| *z).collect(),
        atoms.iter().map(|(_, r)| *r).collect(),
    )
}

fn controls(xc: &str, pairs: &[(&str, Value)]) -> Controls {
    let mut all = vec![
        ("xc".to_string(), Value::Str(xc.into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("grid_level".to_string(), Value::Int(7)),
        // H[3][1] (d2E/dx(H1)dy(O)) has a truncation coefficient near 1e-6,
        // five orders below the other elements, so its central differences
        // reach the noise of the displaced gradients near 1e-13 at
        // h = 0.0025: PBEsol errors 8.90e-10, 2.34e-10, 7.79e-11, 1.42e-11
        // (p = 1.949, R2 = 0.9925) and no interval resolved it (finer: noise
        // growing as 1/h; coarser: h^4 terms). Tightening the SCF to
        // conv_tol = 1e-13 and conv_tol_grad = 1e-11 left it unchanged
        // (errors 8.90e-10, 2.27e-10, 5.76e-11, 1.79e-11; PW91 likewise), so
        // the noise is not the SCF: it is the partition-weight cutoff, which
        // drops and admits points as the atoms move. The fits therefore keep
        // every quadrature point (grid_weight_cutoff = 0), which makes the
        // energy a smooth function of the nuclear positions.
        ("conv_tol".to_string(), Value::Float(1e-13)),
        ("conv_tol_grad".to_string(), Value::Float(1e-11)),
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        ("dispersion".to_string(), Value::Str("none".into())),
    ];
    for (k, v) in pairs {
        all.retain(|(key, _)| key != k);
        all.push((k.to_string(), v.clone()));
    }
    Controls::from_value(&Value::Object(all)).unwrap()
}

fn floats(v: &Value, key: &str) -> Vec<f64> {
    v.get(key)
        .unwrap_or_else(|| panic!("report has no `{key}`"))
        .numeric_array()
        .unwrap()
        .1
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("report has no number `{key}`"))
}

fn vec3(v: [f64; 3]) -> Value {
    Value::Array(v.iter().map(|x| Value::Float(*x)).collect())
}

const AXES: [&str; 3] = ["x", "y", "z"];

/// Every new component against the frozen libxc energy density and first
/// derivatives, point by point.
#[test]
fn new_components_match_libxc_points() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/functional_points.json"),
    )
    .unwrap();
    let oracle = Value::parse_json(&text).unwrap();
    let points: Vec<Vec<f64>> = match oracle.get("points").unwrap() {
        Value::Array(rows) => rows
            .iter()
            .map(|row| row.numeric_array().unwrap().1)
            .collect(),
        _ => panic!("points"),
    };
    let mut failures = Vec::new();
    let Value::Array(components) = oracle.get("components").unwrap() else {
        panic!("components")
    };
    for component in components {
        let table = component.get("table").and_then(Value::as_str).unwrap();
        let name = component.get("name").and_then(Value::as_str).unwrap();
        // A correlation component is the difference between Slater
        // exchange with and without it (the split registry has no
        // correlation-only entry).
        let (spec, minus) = match table {
            "x" => (
                FunctionalSpec::Split {
                    x: name.into(),
                    c: "none".into(),
                },
                None,
            ),
            "c" => (
                FunctionalSpec::Split {
                    x: "slater".into(),
                    c: name.into(),
                },
                Some(FunctionalSpec::Split {
                    x: "slater".into(),
                    c: "none".into(),
                }),
            ),
            _ => (FunctionalSpec::Combined(name.into()), None),
        };
        let functional = resolve(&spec).unwrap();
        let minus = minus.map(|spec| resolve(&spec).unwrap());
        let Value::Array(values) = component.get("values").unwrap() else {
            panic!("values")
        };
        let mut worst = [0.0_f64; 4];
        let mut by_point = Vec::new();
        for (index, (p, expected)) in points.iter().zip(values).enumerate() {
            let before = worst;
            worst = [0.0; 4];
            let (ra, rb) = ([p[0]], [p[1]]);
            let (saa, sab, sbb) = ([p[2]], [p[3]], [p[4]]);
            let (ta, tb) = ([p[5]], [p[6]]);
            let vars = GridVars {
                rho: [&ra, &rb],
                sigma: Some([&saa, &sab, &sbb]),
                tau: Some([&ta, &tb]),
            };
            let mut out = XcOut::default();
            functional.eval(&vars, 1, &mut out).unwrap();
            if let Some(slater) = &minus {
                let mut other = XcOut::default();
                slater.eval(&vars, 1, &mut other).unwrap();
                out.e[0] -= other.e[0];
                for s in 0..2 {
                    out.vrho[s][0] -= other.vrho[s][0];
                }
            }
            let relative = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
            let e = number(expected, "e");
            worst[0] = worst[0].max(relative(out.e[0], e));
            let vrho = floats(expected, "vrho");
            let vsigma = floats(expected, "vsigma");
            let vtau = floats(expected, "vtau");
            for s in 0..2 {
                // A spin channel without density has no defined potential.
                if p[s] > 0.0 {
                    worst[1] = worst[1].max(relative(out.vrho[s][0], vrho[s]));
                }
            }
            for k in 0..3 {
                if !out.vsigma[k].is_empty()
                    && ((k == 0 && p[0] > 0.0)
                        || (k == 2 && p[1] > 0.0)
                        || (k == 1 && p[0] * p[1] > 0.0))
                {
                    worst[2] = worst[2].max(relative(out.vsigma[k][0], vsigma[k]));
                }
            }
            if !out.vtau[0].is_empty() {
                for s in 0..2 {
                    if p[s] > 0.0 {
                        worst[3] = worst[3].max(relative(out.vtau[s][0], vtau[s]));
                    }
                }
            }
            let here = worst.iter().fold(0.0_f64, |m, w| m.max(*w));
            if here >= 1e-8 {
                by_point.push(format!("point {index}: {here:.3e}"));
            }
            for (w, b) in worst.iter_mut().zip(before) {
                *w = w.max(b);
            }
        }
        println!(
            "{table} {name}: largest relative deviation e {:.3e}, vrho {:.3e}, vsigma {:.3e}, vtau {:.3e}",
            worst[0], worst[1], worst[2], worst[3]
        );
        if worst.iter().any(|w| *w >= 1e-8) {
            failures.push(format!(
                "{table} {name}: {worst:?} ({})",
                by_point.join(", ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "pointwise failures:
{}",
        failures.join(
            "
"
        )
    );
}

/// The order-4 Taylor coefficients of every combined registry entry along a
/// density path agree with the dense order-3 derivatives at third order
/// (a functional that bypassed the generic drivers would differ or fail).
#[test]
fn every_registry_entry_has_consistent_taylor_and_dense_derivatives() {
    let mut checked = 0;
    for name in COMBINED_NAMES {
        let functional = resolve(&FunctionalSpec::Combined((*name).into())).unwrap();
        if functional.parts.is_empty() {
            continue;
        }
        let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
        let direction = [0.013, -0.007, 0.004, 0.002, -0.003, 0.011, 0.005];
        let at = |t: f64| -> f64 {
            let v: Vec<f64> = point
                .iter()
                .zip(direction)
                .map(|(p, d)| p + t * d)
                .collect();
            let (ra, rb) = ([v[0]], [v[1]]);
            let (saa, sab, sbb) = ([v[2]], [v[3]], [v[4]]);
            let (ta, tb) = ([v[5]], [v[6]]);
            let vars = GridVars {
                rho: [&ra, &rb],
                sigma: Some([&saa, &sab, &sbb]),
                tau: Some([&ta, &tb]),
            };
            let mut out = XcOut::default();
            functional.eval(&vars, 0, &mut out).unwrap();
            out.e[0]
        };
        let input = XcTaylorInput::linear(point, direction, 4).unwrap();
        let series = functional.semilocal_taylor_energy(&input).unwrap();
        // Independent check of the third-order coefficient: central
        // differences of the energy along the direction at four steps; a
        // coarser interval when the first one ends in round-off, and the
        // fourth-order stencil when the h^2 and h^4 terms cancel inside both
        // (revM11: errors 1.75e-12, 2.40e-11, 7.52e-12, 1.96e-12, 8.45e-13
        // from 0.8 to 0.05).
        let fit_on = |steps: [f64; 4], fourth: bool| {
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| {
                    let third = if fourth {
                        (-at(3.0 * h) + 8.0 * at(2.0 * h) - 13.0 * at(h) + 13.0 * at(-h)
                            - 8.0 * at(-2.0 * h)
                            + at(-3.0 * h))
                            / (8.0 * h * h * h)
                    } else {
                        (at(2.0 * h) - 2.0 * at(h) + 2.0 * at(-h) - at(-2.0 * h))
                            / (2.0 * h * h * h)
                    };
                    (third / 6.0 - series[3]).abs()
                })
                .collect();
            let order = if fourth { 4.0 } else { 2.0 };
            let f = convergence::error_power_fit(&steps, &errors);
            let pass = (f.exponent - order).abs() < 0.3 && f.r_squared > 0.99;
            println!(
                "{name}: Taylor coefficients {series:?}; third order against {} differences: steps {steps:?} errors {errors:?} p={:.4} (expected {order}) R2={:.6}{}",
                if fourth { "fourth-order" } else { "central" },
                f.exponent,
                f.r_squared,
                if pass { "" } else { " FAILED" }
            );
            pass
        };
        assert!(
            fit_on([0.4, 0.2, 0.1, 0.05], false)
                || fit_on([0.8, 0.4, 0.2, 0.1], false)
                || fit_on([0.4, 0.2, 0.1, 0.05], true)
                // revM11's finer fourth-order interval reaches cancellation:
                // errors [1.90e-11, 1.19e-12, 4.10e-14, 6.41e-13]. Retain
                // it above; the resolved coarser interval has p=3.9890 and
                // R2=0.999997, with the original acceptance bounds unchanged.
                || fit_on([1.6, 0.8, 0.4, 0.2], true),
            "{name}"
        );
        checked += 1;
    }
    assert!(checked > 40);
}

/// Every component of the analytic gradient and Hessian of every
/// functional (`support/molecular_fits.rs`).
#[test]
fn gradient_and_hessian_pass_four_step_fits_in_every_component() {
    let mut failures = Vec::new();
    for xc in molecular_set() {
        molecular_fits::gradient_and_hessian(xc, &|extra| controls(xc, extra), &mut failures);
    }
    assert_no_failures("gradient and Hessian", &failures);
}

/// Every component of the static polarizability and of the first
/// hyperpolarizability of every functional (`support/molecular_fits.rs`).
#[test]
fn polarizability_and_hyperpolarizability_pass_four_step_fits_in_every_component() {
    let mut failures = Vec::new();
    for xc in molecular_set() {
        molecular_fits::polarizability_and_hyperpolarizability(
            xc,
            &|extra| controls(xc, extra),
            &mut failures,
        );
    }
    assert_no_failures("polarizability and hyperpolarizability", &failures);
}

/// Excitation energies (TDA and RPA; RPA lies below TDA for the lowest
/// singlet) and London NMR shielding, which must not depend on the gauge
/// origin, for every functional.
#[test]
fn excitations_and_origin_independent_shielding_for_every_functional() {
    for xc in molecular_set() {
        let tda = run(
            "excitations",
            &system(&WATER),
            &controls(xc, &[("nstates", Value::Int(3))]),
        )
        .unwrap();
        let rpa = run(
            "excitations",
            &system(&WATER),
            &controls(
                xc,
                &[
                    ("nstates", Value::Int(3)),
                    ("td_method", Value::Str("rpa".into())),
                ],
            ),
        )
        .unwrap();
        let energies = |report: &Value| -> Vec<f64> {
            let Some(Value::Array(states)) = report.get("states") else {
                panic!("excitations report has no states")
            };
            states.iter().map(|s| number(s, "energy")).collect()
        };
        let (t, r) = (energies(&tda), energies(&rpa));
        // London (GIAO) shielding does not depend on the gauge origin.
        let shieldings = |origin: [f64; 3]| {
            floats(
                &run(
                    "nmr-shielding",
                    &system(&WATER),
                    &controls(xc, &[("gauge_origin", vec3(origin))]),
                )
                .unwrap(),
                "isotropic_shieldings",
            )
        };
        let (s0, s1) = (shieldings([0.0; 3]), shieldings([3.0, -2.0, 1.0]));
        let origin_change = s0
            .iter()
            .zip(&s1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("{xc}: isotropic shieldings {s0:?}, origin change {origin_change:.3e}");
        println!("{xc}: TDA {t:?}, RPA {r:?}");
        assert!(t[0] > 0.0 && r[0] > 0.0 && r[0] <= t[0] + 1e-10);
        assert!(origin_change < 1e-6 * s0.iter().fold(1.0_f64, |m, v| m.max(v.abs())));
    }
}

/// EPC17-2 changes only the interspecies parameter c, which lowers the
/// magnitude of the local correlation energy density at every density.
#[test]
fn epc17_2_is_the_closed_form_with_its_parameters() {
    use dft_rs::multicomponent::correlation::{interspecies_correlation, Epc17};
    let one = interspecies_correlation("epc17").unwrap();
    let two = interspecies_correlation("epc17-2").unwrap();
    for (e, p) in [(0.3, 0.02), (1.2, 0.4), (0.01, 3.0)] {
        let x: f64 = e * p;
        let closed = |c: f64| -x / (2.35 - 2.4 * x.sqrt() + c * x);
        let a = one.evaluate(e, p).unwrap();
        let b = two.evaluate(e, p).unwrap();
        assert!((a.energy - closed(3.2)).abs() < 1e-14);
        assert!((b.energy - closed(6.6)).abs() < 1e-14);
        assert!(b.energy.abs() < a.energy.abs());
    }
    let _ = Epc17::epc17_2();
    assert!(interspecies_correlation("epc18").is_err());
}

/// Every component of the periodic strain derivative dE/d(epsilon_ab) of a
/// two-atom hydrogen chain against central differences of the energy of
/// the cell deformed by `v_a -> v_a + h v_b` (atoms and lattice vectors), for
/// the new semilocal functionals.
#[test]
fn periodic_strain_derivative_passes_four_step_fits_in_every_component() {
    const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
    const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.5, 0.1, 0.05]];
    const INTERVALS: [[f64; 4]; 3] = [
        [0.04, 0.02, 0.01, 0.005],
        [0.02, 0.01, 0.005, 0.0025],
        [0.08, 0.04, 0.02, 0.01],
    ];
    let deformed = |a: usize, b: usize, h: f64| -> System {
        let map = |v: [f64; 3]| -> [f64; 3] {
            let mut w = v;
            w[a] += h * v[b];
            w
        };
        System {
            numbers: vec![1, 1],
            positions: POSITIONS.iter().map(|p| map(*p)).collect(),
            cell: Some(CELL.map(map)),
            pbc: [true; 3],
        }
    };
    let periodic = |xc: &str| {
        Controls::from_value(&Value::Object(vec![
            ("xc".into(), Value::Str(xc.into())),
            ("basis".into(), Value::Str("sto-3g".into())),
            (
                "k_mesh".into(),
                Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
            ),
            ("overlap_cutoff".into(), Value::Float(18.0)),
            ("grid_level".into(), Value::Int(4)),
            ("grid_shell".into(), Value::Int(4)),
            ("conv_tol".into(), Value::Float(1e-12)),
            ("conv_tol_dp".into(), Value::Float(1e-10)),
            ("max_iter".into(), Value::Int(200)),
        ]))
        .unwrap()
    };
    let mut failures = Vec::new();
    for xc in ["pbesol", "rpbe", "pw91", "pbe"] {
        let report = run("stress", &deformed(0, 0, 0.0), &periodic(xc)).unwrap();
        let derivative = floats(&report, "strain_derivative");
        assert_eq!(derivative.len(), 9);
        for a in 0..3 {
            for b in 0..3 {
                let energy = |h: f64| {
                    number(
                        &run("periodic-energy", &deformed(a, b, h), &periodic(xc)).unwrap(),
                        "energy",
                    )
                };
                fit_components(
                    &[format!("{xc} periodic dE/d eps_{}{}", AXES[a], AXES[b])],
                    &INTERVALS,
                    |steps| {
                        let mut errors = [0.0; 4];
                        for (s, &h) in steps.iter().enumerate() {
                            errors[s] = ((energy(h) - energy(-h)) / (2.0 * h)
                                - derivative[3 * a + b])
                                .abs();
                        }
                        vec![errors]
                    },
                    &mut failures,
                );
            }
        }
    }
    assert_no_failures("periodic strain derivative", &failures);
}

/// Simultaneous electronic/quantum-proton SCF with EPC17-2 for new
/// functionals: every component of the coupled Cartesian gradient against
/// central differences of the energy, and every component of the coupled
/// Hessian against central differences of the gradient (atom 1 carries the
/// quantum proton; its coordinates are the centre of the proton basis).
#[test]
fn quantum_nuclear_gradient_and_hessian_pass_four_step_fits_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.02, 0.01, 0.005, 0.0025],
        [0.01, 0.005, 0.0025, 0.00125],
        [0.08, 0.04, 0.02, 0.01],
    ];
    let mut failures = Vec::new();
    for xc in ["pbesol", "scan"] {
        let species = |atoms: &[(u8, [f64; 3])], order: usize| {
            run(
                "multicomponent",
                &system(atoms),
                &controls(
                    xc,
                    &[
                        ("quantum_atoms", Value::Array(vec![Value::Int(1)])),
                        ("interspecies_correlation", Value::Str("epc17-2".into())),
                        ("species_tolerance", Value::Float(1e-11)),
                        ("species_derivative_order", Value::Int(order as i64)),
                    ],
                ),
            )
            .unwrap()
        };
        let reference = species(&WATER, 2);
        let derivatives = reference.get("geometry_derivatives").unwrap();
        let gradient = floats(derivatives, "gradient");
        let hessian = floats(derivatives, "hessian");
        assert_eq!((gradient.len(), hessian.len()), (9, 81));
        println!(
            "{xc} with a quantum proton and EPC17-2: energy {:.10}, interspecies correlation {:.6e}",
            number(&reference, "energy"),
            number(reference.get("energy_components").unwrap(), "interspecies_correlation")
        );
        for coord in 0..9 {
            let (atom, axis) = (coord / 3, coord % 3);
            let displaced = |h: f64| {
                let mut atoms = WATER;
                atoms[atom].1[axis] += h;
                let report = species(&atoms, 1);
                (
                    number(&report, "energy"),
                    floats(report.get("geometry_derivatives").unwrap(), "gradient"),
                )
            };
            let mut names = vec![format!("{xc} quantum-proton dE/d{}{atom}", AXES[axis])];
            names.extend((0..9).map(|r| format!("{xc} quantum-proton H[{r}][{coord}]")));
            fit_components(
                &names,
                &INTERVALS,
                |steps| {
                    let mut errors = vec![[0.0; 4]; 10];
                    for (s, &h) in steps.iter().enumerate() {
                        let (ep, gp) = displaced(h);
                        let (em, gm) = displaced(-h);
                        errors[0][s] = ((ep - em) / (2.0 * h) - gradient[coord]).abs();
                        for r in 0..9 {
                            errors[1 + r][s] =
                                ((gp[r] - gm[r]) / (2.0 * h) - hessian[9 * r + coord]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    assert_no_failures("quantum-proton gradient and Hessian", &failures);
}
