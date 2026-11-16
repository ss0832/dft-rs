// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic Hessian of the restricted open-shell energy on the common
//! driver: a column of the Hessian against central differences of the
//! analytic restricted open-shell gradient by a four-step fit (finite
//! differences are validation only), the gradient of the path response
//! against the analytic gradient, and the unrestricted Hessian of a
//! one-electron molecule, where the two references coincide. Measured
//! (docs/response.md, "Restricted open-shell and spin-flip surfaces"): the
//! one-electron limit passes with maximum difference 1.027e-15
//! Hartree/Bohr^2; both OH tests fail at the restricted open-shell SCF
//! after 150 iterations, so their four-step fit is not measured yet.
//! A common-orbital Newton fallback was subsequently added and converges
//! the OH reference. The next run failed the column fit with errors
//! [6.473e-4, 1.644e-4, 6.362e-5, 3.945e-5] over [0.04, 0.02, 0.01, 0.005]
//! (p = 1.3): with only conv_tol set, the orbital residual tolerance was the
//! default 1e-6, and the displaced analytic gradients carried that noise.
//! The controls now set conv_tol_grad = 1e-10; the steps and the fit bounds
//! are unchanged. The rerun with it failed the same way (errors
//! [6.496e-4, 1.607e-4, 6.362e-5, 3.945e-5], p = 1.35, R^2 = 0.954): the
//! second differences of the OH energies scatter at 3e-8 Hartree because
//! the 2Pi ground state has a degenerate pi pair and every displaced SCF
//! settles on a hole orientation of its own, which the angular grid tells
//! apart. The tests now use the water cation (2B1, not degenerate).
//! The water-cation rerun passes all three tests: column p=1.999885801,
//! R^2=0.999999999256, response residual 9.993455098941e-11;
//! path/analytic-gradient defect 7.455e-12 Hartree/Bohr. The OH failures
//! above remain degenerate-state regressions, not successful acceptance.
//! The fits now cover every component: each of the nine gradient
//! components against central differences of the energy and all 81
//! Hessian elements against central differences of the analytic gradient
//! (a component that fails on the first interval is recomputed on a finer
//! and then a coarser one).

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use components::{assert_no_failures, fit_components};
use stencils::{fit_components_staged, Stencil};

use dft_rs::driver::{run, Controls, System, Value};

/// Water cation (Bohr), distorted away from C2v, with Cartesian coordinate
/// `coord` (3 atom + axis) moved by `shift`: an open shell whose singly
/// occupied orbital is not degenerate.
fn water_cation(coord: usize, shift: f64) -> System {
    let mut positions = vec![
        [0.0, 0.0, 0.23],
        [0.0, 1.4309, -0.8867],
        [0.12, -1.4809, -0.9067],
    ];
    positions[coord / 3][coord % 3] += shift;
    System {
        numbers: vec![8, 1, 1],
        positions,
        cell: None,
        pbc: [false; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("pbe".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        // Every quadrature point is kept, so the energy is a smooth function
        // of the nuclear positions (the partition-weight cutoff otherwise
        // adds noise near 1e-13 to displaced gradients; see p225).
        ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
        ("charge".to_string(), Value::Int(1)),
        ("multiplicity".to_string(), Value::Int(2)),
        // With conv_tol_grad = 1e-10 the displaced gradients carried noise
        // near 2e-12: H[1][3] errors 1.39e-8, 1.97e-9, 3.83e-10, 1.14e-10,
        // 3.67e-10 from h = 0.04 to 0.0025 (the last rising), and no
        // interval resolved it. The orbitals are now converged to 1e-12.
        ("conv_tol".to_string(), Value::Float(1e-13)),
        ("conv_tol_grad".to_string(), Value::Float(1e-12)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn gradient_and_hessian_pass_four_step_fits_in_every_component() {
    const INTERVALS: [[f64; 4]; 4] = [
        [0.04, 0.02, 0.01, 0.005],
        [0.02, 0.01, 0.005, 0.0025],
        [0.01, 0.005, 0.0025, 0.00125],
        [0.08, 0.04, 0.02, 0.01],
    ];
    // The coarse differences of H[1][3] and H[0][4] mix h^2 and h^4 terms
    // (error ratios 8.8, 7.0, 5.1 over 0.08 ... 0.01), so a component that
    // fails every second-order interval is fitted with the fourth-order
    // stencil against order 4.
    const LAST: [([f64; 4], Stencil); 1] = [([0.08, 0.04, 0.02, 0.01], Stencil::Fourth)];
    const AXES: [&str; 3] = ["x", "y", "z"];
    let report = run("roks-hessian", &water_cation(0, 0.0), &controls(&[])).unwrap();
    for key in [
        "response_residual",
        "flag_defect",
        "stationarity_defect",
        "reference_stationarity",
        "reference_energy_error",
    ] {
        let value = report.get(key).and_then(Value::as_f64).unwrap();
        println!("restricted open-shell Hessian diagnostic {key}={value:.12e}");
    }
    let hessian = values(&report, "hessian");
    let reference = run("roks", &water_cation(0, 0.0), &controls(&[])).unwrap();
    let gradient = values(&reference, "gradient");
    assert_eq!((gradient.len(), hessian.len()), (9, 81));
    println!("restricted open-shell gradient {gradient:?}");
    println!("restricted open-shell Hessian {hessian:?}");
    let state = |coord: usize, shift: f64| {
        let report = run("roks", &water_cation(coord, shift), &controls(&[])).unwrap();
        (
            report.get("energy").and_then(Value::as_f64).unwrap(),
            values(&report, "gradient"),
        )
    };
    let mut failures = Vec::new();
    for coord in 0..9 {
        let mut names = vec![format!("gradient dE/d{}{}", AXES[coord % 3], coord / 3)];
        names.extend((0..9).map(|r| format!("Hessian H[{r}][{coord}]")));
        let errors_at = |steps: &[f64; 4], stencil: Stencil| -> Vec<[f64; 4]> {
            let mut errors = vec![[0.0; 4]; 10];
            for (s, &h) in steps.iter().enumerate() {
                let ((ep, gp), (em, gm)) = (state(coord, h), state(coord, -h));
                let ((ep2, gp2), (em2, gm2)) = if stencil == Stencil::Fourth {
                    (state(coord, 2.0 * h), state(coord, -2.0 * h))
                } else {
                    ((0.0, vec![0.0; 9]), (0.0, vec![0.0; 9]))
                };
                errors[0][s] = (stencil.first(ep, em, ep2, em2, h) - gradient[coord]).abs();
                for r in 0..9 {
                    errors[1 + r][s] = (stencil.first(gp[r], gm[r], gp2[r], gm2[r], h)
                        - hessian[9 * r + coord])
                        .abs();
                }
            }
            errors
        };
        let mut open = Vec::new();
        fit_components(
            &names,
            &INTERVALS,
            |steps| errors_at(steps, Stencil::Second),
            &mut open,
        );
        if !open.is_empty() {
            let index: Vec<usize> = open
                .iter()
                .map(|name| names.iter().position(|n| n == name).unwrap())
                .collect();
            fit_components_staged(
                &open,
                &LAST,
                |steps, stencil| {
                    let all = errors_at(steps, stencil);
                    index.iter().map(|&k| all[k]).collect()
                },
                &mut failures,
            );
        }
    }
    assert_no_failures("restricted open-shell gradient and Hessian", &failures);
}

#[test]
fn path_gradient_is_the_analytic_gradient_and_translations_cost_nothing() {
    let report = run("roks-hessian", &water_cation(0, 0.0), &controls(&[])).unwrap();
    let analytic = values(
        &run("roks", &water_cation(0, 0.0), &controls(&[])).unwrap(),
        "gradient",
    );
    let path = values(&report, "gradient");
    let difference = path
        .iter()
        .zip(&analytic)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let translation = report
        .get("translation_residual")
        .and_then(Value::as_f64)
        .unwrap();
    println!(
        "path against analytic gradient {difference:.3e}, translation residual {translation:.3e}"
    );
    assert!(difference < 1e-6);
    assert!(translation < 1e-5);
}

#[test]
fn one_electron_molecule_has_the_unrestricted_hessian() {
    // The hydrogen molecular ion: one electron, so the restricted open-shell
    // and the unrestricted reference are the same state.
    let system = System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        cell: None,
        pbc: [false; 3],
    };
    let charge = ("charge", Value::Int(1));
    let open = run(
        "roks-hessian",
        &system,
        &controls(std::slice::from_ref(&charge)),
    )
    .unwrap();
    let unrestricted = run(
        "hessian",
        &system,
        &controls(&[charge, ("unrestricted", Value::Bool(true))]),
    )
    .unwrap();
    let difference = values(&open, "hessian")
        .iter()
        .zip(&values(&unrestricted, "hessian"))
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    println!("restricted open-shell against unrestricted Hessian {difference:.3e}");
    assert!(difference < 1e-6);
}
