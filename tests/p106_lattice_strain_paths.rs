// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Strain paths of a periodic cell on the common driver, every component
//! against independent finite differences (validation only):
//!
//! - third- and fourth-order elastic constants at clamped ions through
//!   their contractions along 56 and 126 linearly independent generic Voigt
//!   directions `u` (as many as the symmetric tensors have independent
//!   components), against third and fourth central differences of the
//!   self-consistent energy of the cells deformed by
//!   `F(t) = (1 + 2 t eta(u))^(1/2)`;
//! - the same constants at relaxed ions along the analytic relaxed paths:
//!   `t u'(u)` for the third order and `t u'(u) + t^2 u''(u, u) / 2` for the
//!   fourth. The energy is stationary in the internal coordinates, so a path
//!   error of order t^2 (t^3) changes the energy only at order t^4 (t^6)
//!   and leaves the third (fourth) coefficient unchanged; the analytic
//!   relaxation itself is validated by the change of the forces along the
//!   path, `F(t) - F(0)`, which must vanish as t^2 (t^3);
//! - every mode Grueneisen parameter of the optical modes for all six
//!   strains against the analytic frequencies of strained, first-order
//!   relaxed cells;
//! - all 216 cubic force constants against the analytic force constants of
//!   displaced cells.
//!
//! A component that fails on the first interval is recomputed on a finer
//! and then a coarser one. Independent energy evaluations run on worker
//! threads.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/quartic_oracle.rs"]
mod quartic_oracle;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_cartesian_directions.rs"]
mod symmetric_cartesian_directions;

use components::{assert_no_failures, component_fit_of_order, fit_components};
use dft_rs::driver::{run, Controls, System, Value};

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]];
const VOIGT: [&str; 6] = ["xx", "yy", "zz", "yz", "xz", "xy"];
const AXES: [&str; 3] = ["x", "y", "z"];

/// Symmetric strain tensor of the Voigt components; shear components are
/// twice the tensor components (engineering convention of the reports).
fn tensor(voigt: &[f64; 6]) -> [[f64; 3]; 3] {
    [
        [voigt[0], 0.5 * voigt[5], 0.5 * voigt[4]],
        [0.5 * voigt[5], voigt[1], 0.5 * voigt[3]],
        [0.5 * voigt[4], 0.5 * voigt[3], voigt[2]],
    ]
}

fn product(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// `F = (1 + 2 eta)^(1/2)` by its binomial series, whose Lagrangian strain
/// is exactly `eta` (the strains here keep `|2 eta| < 0.5`).
fn deformation(voigt: &[f64; 6]) -> [[f64; 3]; 3] {
    let two_eta = tensor(voigt).map(|row| row.map(|v| 2.0 * v));
    let mut f = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut power = f;
    let mut coefficient = 1.0;
    for n in 1..80 {
        coefficient *= (0.5 - (n as f64 - 1.0)) / n as f64;
        power = product(&power, &two_eta);
        for i in 0..3 {
            for j in 0..3 {
                f[i][j] += coefficient * power[i][j];
            }
        }
    }
    f
}

/// The cell deformed by the Lagrangian strain `voigt` at clamped fractional
/// coordinates, with the atoms displaced by `shift` (six values) before the
/// deformation: `r' = F (r + shift)`.
fn strained(voigt: &[f64; 6], shift: &[f64]) -> System {
    deform_system(
        &System {
            numbers: vec![1, 1],
            positions: POSITIONS.to_vec(),
            cell: Some(CELL),
            pbc: [true; 3],
        },
        voigt,
        shift,
    )
}

fn deform_system(system: &System, voigt: &[f64; 6], shift: &[f64]) -> System {
    let f = deformation(voigt);
    let apply =
        |v: [f64; 3]| -> [f64; 3] { std::array::from_fn(|i| (0..3).map(|k| f[i][k] * v[k]).sum()) };
    let positions = system
        .positions
        .iter()
        .enumerate()
        .map(|(atom, p)| apply(std::array::from_fn(|axis| p[axis] + shift[3 * atom + axis])))
        .collect();
    System {
        numbers: system.numbers.clone(),
        positions,
        cell: system.cell.map(|cell| cell.map(apply)),
        pbc: system.pbc,
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions and of the strain (see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(20.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(2)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(200)),
        // SAD starts the strained-cell SCFs far closer to the fixed point
        // than the core guess, cutting their iterations; the fixed point,
        // the thresholds and the fits below are unchanged.
        ("guess".to_string(), Value::Str("sad".into())),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn values(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

/// `count` independent evaluations on a few worker threads, in order.
fn parallel<T: Send, F: Fn(usize) -> T + Sync>(count: usize, f: F) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<T>>> =
        (0..count).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= count {
                    break;
                }
                *slots[i].lock().unwrap() = Some(f(i));
            });
        }
    });
    slots
        .into_iter()
        .map(|s| s.into_inner().unwrap().unwrap())
        .collect()
}

/// Energy and forces (six values) of a strained, displaced cell.
fn state(voigt: &[f64; 6], shift: &[f64]) -> (f64, Vec<f64>) {
    let report = run(
        "periodic-energy",
        &strained(voigt, shift),
        &controls(&[("forces", Value::Bool(true))]),
    )
    .unwrap();
    (number(&report, "energy"), values(&report, "forces"))
}

/// Complete full-rank set of six-dimensional strain directions.
fn directions(order: usize) -> Vec<[f64; 6]> {
    symmetric_cartesian_directions::directions(6, order)
        .into_iter()
        .map(|v| v.try_into().unwrap())
        .collect()
}

/// `sum C[i1..ik] u_i1 .. u_ik` of a symmetric Voigt tensor, last index
/// fastest.
fn contract(tensor: &[f64], u: &[f64; 6]) -> f64 {
    let order = (tensor.len() as f64).log(6.0).round() as u32;
    let mut total = 0.0;
    for (flat, value) in tensor.iter().enumerate() {
        let mut weight = 1.0;
        let mut rest = flat;
        for _ in 0..order {
            weight *= u[rest % 6];
            rest /= 6;
        }
        total += value * weight;
    }
    total
}

/// Shift of the six coordinates along the relaxed path at `t`:
/// `t u' u + t^2 u''(u, u) / 2` (`second` empty for the first-order path).
fn path_shift(relaxation: &[f64], second: &[f64], u: &[f64; 6], t: f64) -> Vec<f64> {
    let coordinates = relaxation.len() / 6;
    (0..coordinates)
        .map(|k| {
            let first: f64 = (0..6).map(|i| relaxation[6 * k + i] * u[i]).sum();
            let curvature: f64 = if second.is_empty() {
                0.0
            } else {
                (0..6)
                    .flat_map(|i| (0..6).map(move |j| (i, j)))
                    .map(|(i, j)| second[(6 * i + j) * coordinates + k] * u[i] * u[j])
                    .sum()
            };
            t * first + 0.5 * t * t * curvature
        })
        .collect()
}

/// Directional fits of a symmetric elastic tensor of order 2, 3 or 4 along
/// `dirs`, with the atoms on `shift(u, t)`. With `force_order`, the norm of
/// the change of the forces along the path is also fitted against that
/// order.
#[allow(clippy::too_many_arguments)]
fn elastic_fits(
    label: &str,
    order: usize,
    analytic: &[f64],
    volume: f64,
    dirs: &[[f64; 6]],
    intervals: &[[f64; 4]],
    state: &(impl Fn(&[f64; 6], &[f64]) -> (f64, Vec<f64>) + Sync),
    shift: &(dyn Fn(&[f64; 6], f64) -> Vec<f64> + Sync),
    force_order: Option<f64>,
    failures: &mut Vec<String>,
) {
    // The reference cell need not be at internal equilibrium: the residual
    // is the change of the forces along the path, F(t) - F(0), whose linear
    // (and, on the second-order path, quadratic) term vanishes by the
    // definition of the analytic relaxation.
    let (centre, centre_forces) = state(&[0.0; 6], &[0.0; 6]);
    for (n, u) in dirs.iter().enumerate() {
        let expected = volume * contract(analytic, u);
        // Residual force norms at +h of the first interval, taken from the
        // same self-consistent states as the energies.
        let mut residual: Option<([f64; 4], [f64; 4])> = None;
        fit_components(
            &[format!("{label} along direction {n}")],
            intervals,
            |steps| {
                // E at +-h and +-2h for every step.
                let points: Vec<f64> = steps
                    .iter()
                    .flat_map(|&h| [h, -h, 2.0 * h, -2.0 * h])
                    .collect();
                let states = parallel(points.len(), |i| {
                    let t = points[i];
                    state(&u.map(|x| t * x), &shift(u, t))
                });
                if residual.is_none() {
                    residual = Some((
                        *steps,
                        std::array::from_fn(|s| {
                            states[4 * s]
                                .1
                                .iter()
                                .zip(&centre_forces)
                                .map(|(f, f0)| (f - f0) * (f - f0))
                                .sum::<f64>()
                                .sqrt()
                        }),
                    ));
                }
                let errors: [f64; 4] = std::array::from_fn(|s| {
                    let h = steps[s];
                    let (p, m, p2, m2) = (
                        states[4 * s].0,
                        states[4 * s + 1].0,
                        states[4 * s + 2].0,
                        states[4 * s + 3].0,
                    );
                    let fd = match order {
                        2 => (p + m - 2.0 * centre) / h.powi(2),
                        3 => (p2 - 2.0 * p + 2.0 * m - m2) / (2.0 * h.powi(3)),
                        4 => (p2 - 4.0 * p + 6.0 * centre - 4.0 * m + m2) / h.powi(4),
                        _ => unreachable!("elastic energy stencil order"),
                    };
                    (fd - expected).abs()
                });
                vec![errors]
            },
            failures,
        );
        if let (Some(force), Some((steps, norms))) = (force_order, residual) {
            if !component_fit_of_order(
                &format!("{label} residual force along direction {n}"),
                &steps,
                &norms,
                force,
            ) {
                failures.push(format!("{label} residual force along direction {n}"));
            }
        }
    }
}

#[test]
fn third_order_elastic_constants_pass_directional_fits_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.04, 0.03, 0.02, 0.015],
        [0.02, 0.015, 0.01, 0.0075],
        [0.08, 0.06, 0.04, 0.03],
    ];
    let report = run("toec", &strained(&[0.0; 6], &[0.0; 6]), &controls(&[])).unwrap();
    let volume = number(&report, "volume");
    let clamped = values(&report, "high_order_clamped");
    let relaxed = values(&report, "high_order_relaxed");
    let relaxation = values(&report, "internal_relaxation");
    println!("periodic TOEC energy={:.17e} volume={volume:.17e} clamped={clamped:?} relaxed={relaxed:?} response_residuals={:?}",
        number(&report, "energy"), values(&report, "response_residuals"));
    assert_eq!(
        (clamped.len(), relaxed.len(), relaxation.len()),
        (216, 216, 36)
    );
    let dirs = directions(3);
    let mut failures = Vec::new();
    elastic_fits(
        "clamped C_IJK",
        3,
        &clamped,
        volume,
        &dirs,
        &INTERVALS,
        &state,
        &|_, _| vec![0.0; 6],
        None,
        &mut failures,
    );
    elastic_fits(
        "relaxed C_IJK",
        3,
        &relaxed,
        volume,
        &dirs,
        &INTERVALS,
        &state,
        &|u, t| path_shift(&relaxation, &[], u, t),
        Some(2.0),
        &mut failures,
    );
    assert_no_failures("third-order elastic constants", &failures);
}

#[test]
fn fourth_order_elastic_constants_pass_directional_fits_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.06, 0.045, 0.035, 0.025],
        [0.035, 0.025, 0.0175, 0.0125],
        [0.08, 0.06, 0.045, 0.035],
    ];
    let report = run("foec", &strained(&[0.0; 6], &[0.0; 6]), &controls(&[])).unwrap();
    let volume = number(&report, "volume");
    let clamped = values(&report, "high_order_clamped");
    let relaxed = values(&report, "high_order_relaxed");
    let relaxation = values(&report, "internal_relaxation");
    let second = values(&report, "second_internal_relaxation");
    println!("periodic FOEC energy={:.17e} volume={volume:.17e} clamped={clamped:?} relaxed={relaxed:?} response_residuals={:?}",
        number(&report, "energy"), values(&report, "response_residuals"));
    assert_eq!(
        (clamped.len(), relaxed.len(), relaxation.len(), second.len()),
        (1296, 1296, 36, 216)
    );
    let dirs = directions(4);
    let mut failures = Vec::new();
    elastic_fits(
        "clamped C_IJKL",
        4,
        &clamped,
        volume,
        &dirs,
        &INTERVALS,
        &state,
        &|_, _| vec![0.0; 6],
        None,
        &mut failures,
    );
    elastic_fits(
        "relaxed C_IJKL",
        4,
        &relaxed,
        volume,
        &dirs,
        &INTERVALS,
        &state,
        &|u, t| path_shift(&relaxation, &second, u, t),
        Some(3.0),
        &mut failures,
    );
    assert_no_failures("fourth-order elastic constants", &failures);
}

#[test]
fn dispersion_corrected_elastic_constants_fit_every_independent_direction() {
    // HeH+ removes the inversion of the identical-atom fixture. The cutoff
    // falls between the two closest pair distances and the first self image.
    let system = |strain: &[f64; 6], shift: &[f64]| {
        let mut cell = strained(strain, shift);
        cell.numbers = vec![2, 1];
        cell
    };
    let settings = |extra: &[(&str, Value)]| {
        let mut pairs = vec![
            ("xc", Value::Str("pbe".into())),
            ("charge", Value::Int(1)),
            ("k_mesh", Value::Array(vec![Value::Int(1); 3])),
            ("dispersion", Value::Str("d3bj".into())),
            ("disp_cutoff", Value::Float(2.85)),
            ("cn_cutoff", Value::Float(2.85)),
        ];
        pairs.extend_from_slice(extra);
        controls(&pairs)
    };
    let force_settings = settings(&[("forces", Value::Bool(true))]);
    let displaced = |strain: &[f64; 6], shift: &[f64]| {
        let settings = settings(&[("forces", Value::Bool(true))]);
        let report = run("periodic-energy", &system(strain, shift), &settings).unwrap();
        (number(&report, "energy"), values(&report, "forces"))
    };
    let center = system(&[0.0; 6], &[0.0; 6]);
    let energy = run("periodic-energy", &center, &force_settings).unwrap();
    let mut failures = Vec::new();
    for (order, id) in [(3, "toec"), (4, "foec")] {
        let report = run(id, &center, &settings(&[])).unwrap();
        let dispersion = number(&report, "dispersion_energy_per_primitive_cell");
        let volume = number(&report, "volume");
        let clamped = values(&report, "high_order_clamped");
        let relaxed = values(&report, "high_order_relaxed");
        let relaxation = values(&report, "internal_relaxation");
        let second = if order == 4 {
            values(&report, "second_internal_relaxation")
        } else {
            Vec::new()
        };
        let residuals = values(&report, "response_residuals");
        println!("D3(BJ) {id}: energy={:.17e} dispersion={dispersion:.17e} volume={volume:.17e} clamped_xx={:.17e} relaxed_xx={:.17e} orbital_residuals={residuals:?}", number(&report, "energy"), clamped[0], relaxed[0]);
        assert!(dispersion < 0.0);
        assert!((dispersion - number(&energy, "dispersion_energy")).abs() < 1e-13);
        assert!((number(&report, "energy") - number(&energy, "energy")).abs() < 1e-10);
        let dirs = directions(order);
        let intervals = [
            [0.06, 0.045, 0.035, 0.025],
            [0.035, 0.025, 0.0175, 0.0125],
            [0.08, 0.06, 0.045, 0.035],
        ];
        elastic_fits(
            &format!("D3(BJ) clamped {id}"),
            order,
            &clamped,
            volume,
            &dirs,
            &intervals,
            &displaced,
            &|_, _| vec![0.0; 6],
            None,
            &mut failures,
        );
        elastic_fits(
            &format!("D3(BJ) relaxed {id}"),
            order,
            &relaxed,
            volume,
            &dirs,
            &intervals,
            &displaced,
            &|u, t| path_shift(&relaxation, &second, u, t),
            Some((order - 1) as f64),
            &mut failures,
        );
    }
    assert_no_failures("D3(BJ) higher elastic constants", &failures);
}

#[test]
fn biased_elastic_constants_fit_every_independent_direction() {
    let reference = System {
        numbers: vec![3, 1],
        positions: vec![[0.17, -0.25, 0.08], [3.30, 0.44, -0.57]],
        cell: Some([[9.6, 0.7, 0.3], [0.4, 10.4, 0.6], [0.2, -0.5, 11.8]]),
        pbc: [true; 3],
    };
    let field = [0.001, -0.0015, 0.002];
    let mut settings = vec![
        ("xc", Value::Str("wb97x-d".into())),
        (
            "field",
            Value::Array(field.into_iter().map(Value::Float).collect()),
        ),
        ("k_mesh", Value::Array(vec![Value::Int(1); 3])),
        ("overlap_cutoff", Value::Float(4.2)),
        ("exchange_cutoff", Value::Float(4.2)),
        ("disp_cutoff", Value::Float(4.2)),
        ("grid_level", Value::Int(0)),
        ("grid_shell", Value::Int(1)),
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-11)),
        ("conv_tol_grad", Value::Float(1e-11)),
    ];
    let elastic_controls = controls(&settings);
    settings.push(("forces", Value::Bool(true)));
    let state = |strain: &[f64; 6], shift: &[f64]| {
        let sample = run(
            "periodic-efield",
            &deform_system(&reference, strain, shift),
            &controls(&settings),
        )
        .unwrap();
        // The oracle returns physical forces. F^T converts them to the
        // material-coordinate forces held fixed on the relaxation path.
        let f = deformation(strain);
        let material_forces = values(&sample, "forces")
            .chunks_exact(3)
            .flat_map(|force| {
                std::array::from_fn::<_, 3, _>(|i| (0..3).map(|j| f[j][i] * force[j]).sum::<f64>())
            })
            .collect();
        (number(&sample, "field_energy"), material_forces)
    };
    let (energy, _) = state(&[0.0; 6], &[0.0; 6]);
    let toec = run("toec", &reference, &elastic_controls).unwrap();
    let foec = run("foec", &reference, &elastic_controls).unwrap();
    let intervals = [
        [0.06, 0.045, 0.035, 0.025],
        [0.035, 0.025, 0.0175, 0.0125],
        [0.08, 0.06, 0.045, 0.035],
    ];
    let mut failures = Vec::new();
    for (order, report) in [(2, &toec), (3, &toec), (4, &foec)] {
        let prefix = if order == 2 {
            "elastic_constants"
        } else {
            "high_order"
        };
        let clamped = values(report, &format!("{prefix}_clamped"));
        let relaxed = values(report, &format!("{prefix}_relaxed"));
        let volume = number(report, "volume");
        let relaxation = values(report, "internal_relaxation");
        let second = if order == 4 {
            values(report, "second_internal_relaxation")
        } else {
            Vec::new()
        };
        assert_eq!(values(report, "reference_field"), field);
        assert!((number(report, "field_energy") - energy).abs() < 1e-12);
        println!("Fixed-field order {order}: enthalpy={energy:.17e} volume={volume:.17e} clamped_xx={:.17e} relaxed_xx={:.17e} series_residual={:.17e} response_residuals={:?}", clamped[0], relaxed[0], number(report, "series_reference_residual"), values(report, "response_residuals"));
        let directions = directions(order);
        elastic_fits(
            &format!("fixed-field clamped order {order}"),
            order,
            &clamped,
            volume,
            &directions,
            &intervals,
            &state,
            &|_, _| vec![0.0; 6],
            None,
            &mut failures,
        );
        elastic_fits(
            &format!("fixed-field relaxed order {order}"),
            order,
            &relaxed,
            volume,
            &directions,
            &intervals,
            &state,
            &|u, t| path_shift(&relaxation, &second, u, t),
            Some(if order == 4 { 3.0 } else { 2.0 }),
            &mut failures,
        );
    }
    assert_no_failures("fixed-field elastic constants", &failures);
}

#[test]
fn mode_gruneisen_parameters_pass_four_step_fits_in_every_component() {
    check_mode_gruneisen(&strained, &[]);
}

#[test]
fn dispersion_corrected_mode_gruneisen_parameters_fit_all_strains() {
    let reference = System {
        numbers: vec![1, 1, 2],
        positions: vec![[0.11, -0.17, 0.23], [1.24, 0.19, 0.57], [-0.37, 1.59, 0.42]],
        cell: Some([[7.3, 0.21, 0.17], [0.35, 7.9, 0.41], [0.13, 0.29, 8.6]]),
        pbc: [true; 3],
    };
    let system = |strain: &[f64; 6], shift: &[f64]| deform_system(&reference, strain, shift);
    check_mode_gruneisen(
        &system,
        &[
            ("xc", Value::Str("pbe".into())),
            ("dispersion", Value::Str("d2".into())),
            ("disp_cutoff", Value::Float(3.5)),
            ("overlap_cutoff", Value::Float(3.0)),
            ("k_mesh", Value::Array(vec![Value::Int(1); 3])),
        ],
    );
}

fn check_mode_gruneisen(
    system: &(impl Fn(&[f64; 6], &[f64]) -> System + Sync),
    settings: &[(&str, Value)],
) {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.02, 0.014, 0.01, 0.007],
        [0.01, 0.007, 0.005, 0.0035],
        [0.04, 0.028, 0.02, 0.014],
    ];
    let reference = system(&[0.0; 6], &[0.0; 9]);
    let mut with_shear = settings.to_vec();
    with_shear.push(("gruneisen_shear", Value::Bool(true)));
    let report = run("mode-gruneisen", &reference, &controls(&with_shear)).unwrap();
    let strain = values(&report, "mode_gruneisen_strain");
    let frequencies = values(&report, "frequencies_cm1");
    assert_eq!(strain.len(), 6 * frequencies.len());
    // The signed frequencies use |omega| for unstable optical branches.
    let Some(Value::Array(acoustic)) = report.get("acoustic_modes") else {
        panic!("acoustic mode flags are missing");
    };
    let optical: Vec<usize> = (0..frequencies.len())
        .filter(|&m| matches!(acoustic[m], Value::Bool(false)))
        .collect();
    assert_eq!(
        optical.len() + 3,
        frequencies.len(),
        "frequencies {frequencies:?}"
    );
    assert!(optical.iter().all(|&m| frequencies[m].abs() > 1e-8));
    if settings.is_empty() {
        assert_eq!(strain.len(), 36);
        assert_eq!(optical.len(), 3);
        assert!(optical.iter().all(|&m| frequencies[m] > 50.0));
    }
    println!("strain mode Gruneisen: frequencies={frequencies:?} gamma={strain:?} response_residuals={:?}", values(&report, "response_residuals"));
    let relaxation = if settings.is_empty() {
        values(
            &run("toec", &reference, &controls(settings)).unwrap(),
            "internal_relaxation",
        )
    } else {
        // Obtain the PBE-D2 relaxation from the independent conventional
        // elastic solver in its nine affine strain coordinates. The first
        // Lagrangian and affine strain derivatives agree at the reference;
        // each engineering shear is the average of two ordered affine seeds.
        let affine = values(
            &run("elastic", &reference, &controls(settings)).unwrap(),
            "internal_relaxation",
        );
        const VOIGT_AXES: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
        assert_eq!(affine.len(), 9 * frequencies.len());
        affine
            .chunks_exact(9)
            .flat_map(|row| VOIGT_AXES.map(|(a, b)| 0.5 * (row[3 * a + b] + row[3 * b + a])))
            .collect()
    };
    let mut failures = Vec::new();
    for i in 0..6 {
        let mut unit = [0.0; 6];
        unit[i] = 1.0;
        let names: Vec<String> = optical
            .iter()
            .map(|m| format!("mode Grueneisen gamma[{m}][{}]", VOIGT[i]))
            .collect();
        fit_components(
            &names,
            &INTERVALS,
            |steps| {
                let points: Vec<f64> = steps.iter().flat_map(|&h| [h, -h]).collect();
                let spectra = parallel(points.len(), |k| {
                    let t = points[k];
                    let r = run(
                        "phonons",
                        &system(
                            &unit.map(|x| t * x),
                            &path_shift(&relaxation, &[], &unit, t),
                        ),
                        &controls(settings),
                    )
                    .unwrap();
                    values(&r, "frequencies_cm1")
                });
                optical
                    .iter()
                    .map(|&m| {
                        std::array::from_fn(|s| {
                            let h = steps[s];
                            let fd = -(spectra[2 * s][m].abs().ln()
                                - spectra[2 * s + 1][m].abs().ln())
                                / (2.0 * h);
                            (fd - strain[6 * m + i]).abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    assert_no_failures("mode Grueneisen parameters", &failures);
}

#[test]
fn complete_cubic_force_constants_pass_four_step_fits_in_every_component() {
    const INTERVALS: [[f64; 4]; 3] = [
        [0.08, 0.04, 0.02, 0.01],
        [0.04, 0.02, 0.01, 0.005],
        [0.16, 0.08, 0.04, 0.02],
    ];
    let report = run(
        "periodic-cubic-fc",
        &strained(&[0.0; 6], &[0.0; 6]),
        &controls(&[]),
    )
    .unwrap();
    let tensor = values(&report, "force_constant_tensor");
    assert_eq!(tensor.len(), 216);
    // The tensor is symmetric in all three indices.
    let mut asymmetry = 0.0_f64;
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                asymmetry =
                    asymmetry.max((tensor[36 * i + 6 * j + k] - tensor[36 * k + 6 * j + i]).abs());
            }
        }
    }
    println!("complete cubic force constants: asymmetry {asymmetry:e}");
    assert!(asymmetry < 1e-12);
    let mut failures = Vec::new();
    for coordinate in 0..6 {
        let names: Vec<String> = (0..36)
            .map(|jk| {
                format!(
                    "cubic force constant d Phi[{}][{}] / d{}{}",
                    jk / 6,
                    jk % 6,
                    AXES[coordinate % 3],
                    coordinate / 3
                )
            })
            .collect();
        fit_components(
            &names,
            &INTERVALS,
            |steps| {
                let points: Vec<f64> = steps.iter().flat_map(|&h| [h, -h]).collect();
                let hessians = parallel(points.len(), |k| {
                    let mut shift = [0.0; 6];
                    shift[coordinate] = points[k];
                    values(
                        &run("phonons", &strained(&[0.0; 6], &shift), &controls(&[])).unwrap(),
                        "force_constants",
                    )
                });
                (0..36)
                    .map(|jk| {
                        std::array::from_fn(|s| {
                            let h = steps[s];
                            let fd = (hessians[2 * s][jk] - hessians[2 * s + 1][jk]) / (2.0 * h);
                            (fd - tensor[36 * coordinate + jk]).abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    assert_no_failures("complete cubic force constants", &failures);
}

fn charged_force_constant_fixture(extra: &[(&str, Value)]) -> (System, Controls) {
    // The heterogeneous cell removes the inversion of the neutral H2
    // fixture. Uniform charge compensation is held at the fixed lattice.
    let mut reference = strained(&[0.0; 6], &[0.0; 6]);
    reference.numbers[0] = 2;
    let mut settings = vec![
        ("charge", Value::Int(1)),
        ("k_mesh", Value::Array(vec![Value::Int(1); 3])),
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-11)),
        ("conv_tol_grad", Value::Float(1e-11)),
    ];
    settings.extend_from_slice(extra);
    (reference, controls(&settings))
}

fn displaced_hessian(reference: &System, controls: &Controls, direction: &[f64; 6], h: f64) -> f64 {
    let mut geometry = reference.clone();
    for (a, xyz) in geometry.positions.iter_mut().enumerate() {
        for j in 0..3 {
            xyz[j] += h * direction[3 * a + j];
        }
    }
    let hessian = run("phonons", &geometry, controls).unwrap();
    contract(&values(&hessian, "raw_force_constants"), direction)
}

#[test]
fn charged_cubic_force_constants_fit_all_independent_directions() {
    let (reference, controls) = charged_force_constant_fixture(&[]);
    cubic_force_constant_fits("charged periodic FC3", &reference, &controls);
}

fn cubic_force_constant_fits(label: &str, reference: &System, controls: &Controls) {
    use stencils::Stencil;
    let report = run("periodic-cubic-fc", reference, controls).unwrap();
    let tensor = values(&report, "force_constant_tensor");
    assert_eq!(tensor.len(), 6_usize.pow(3));
    let acoustic = (0..3)
        .flat_map(|a| (0..36).map(move |b| (a, b)))
        .map(|(a, b)| (tensor[36 * a + b] + tensor[36 * (a + 3) + b]).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "{label} energy={:.17e} acoustic={acoustic:.17e} residuals={:?} tensor={tensor:?}",
        number(&report, "energy"),
        values(&report, "response_residuals")
    );
    assert!(acoustic < 1e-8);
    let mut failures = Vec::new();
    for (i, direction) in directions(3).into_iter().enumerate() {
        let exact = contract(&tensor, &direction);
        let mut samples = std::collections::BTreeMap::new();
        stencils::fit_components_staged(
            &[format!("{label} direction {i}")],
            &[
                ([0.04, 0.03, 0.02, 0.01], Stencil::Second),
                ([0.02, 0.015, 0.01, 0.005], Stencil::Second),
                ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut at = |h: f64| {
                    *samples
                        .entry(h.to_bits())
                        .or_insert_with(|| displaced_hessian(reference, controls, &direction, h))
                };
                vec![steps.map(|h| {
                    let (plus, minus) = (at(h), at(-h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        (0.0, 0.0)
                    };
                    (stencil.first(plus, minus, plus2, minus2, h) - exact).abs()
                })]
            },
            &mut failures,
        );
    }
    assert_no_failures(label, &failures);
}

#[test]
fn charged_quartic_force_constants_fit_all_independent_directions() {
    let (reference, controls) = charged_force_constant_fixture(&[]);
    quartic_force_constant_fits("charged periodic FC4", &reference, &controls);
}

fn quartic_force_constant_fits(label: &str, reference: &System, controls: &Controls) {
    let report = run("periodic-quartic-fc", reference, controls).unwrap();
    let tensor = values(&report, "force_constant_tensor");
    assert_eq!(tensor.len(), 6_usize.pow(4));
    let acoustic = (0..3)
        .flat_map(|a| (0..216).map(move |b| (a, b)))
        .map(|(a, b)| (tensor[216 * a + b] + tensor[216 * (a + 3) + b]).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "{label} energy={:.17e} acoustic={acoustic:.17e} residuals={:?} tensor={tensor:?}",
        number(&report, "energy"),
        values(&report, "response_residuals")
    );
    assert!(acoustic < 1e-8);
    let mut failures = Vec::new();
    for (i, direction) in directions(4).into_iter().enumerate() {
        quartic_oracle::fit_from_second(
            &format!("{label} direction {i}"),
            contract(&tensor, &direction),
            |h| displaced_hessian(reference, controls, &direction, h),
            &mut failures,
        );
    }
    assert_no_failures(label, &failures);
}

#[test]
fn implicit_chg_cubic_force_constants_fit_all_independent_directions() {
    let (reference, controls) = charged_force_constant_fixture(&[
        ("xc", Value::Str("wb97x-d".into())),
        ("disp_cutoff", Value::Float(2.85)),
    ]);
    cubic_force_constant_fits("omegaB97X-D periodic FC3", &reference, &controls);
}

#[test]
fn implicit_chg_quartic_force_constants_fit_all_independent_directions() {
    let (reference, controls) = charged_force_constant_fixture(&[
        ("xc", Value::Str("wb97x-d".into())),
        ("disp_cutoff", Value::Float(2.85)),
    ]);
    quartic_force_constant_fits("omegaB97X-D periodic FC4", &reference, &controls);
}
