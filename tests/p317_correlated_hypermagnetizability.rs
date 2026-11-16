// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All independent static E/E/B/B components against relaxed SCF/PT2 energies.
//! The finite differences below are acceptance oracles, never production values.
#[path = "support/complex_pt2.rs"]
mod complex_pt2;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/finite_correlated_field.rs"]
mod finite_correlated_field;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::properties::correlated_magnetic::correlated_hypermagnetizability_for_model;
use dft_rs::properties::magnetic::Gauge;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::double_hybrid::Pt2Coefficients;
use dft_rs::Vec3;
use finite_correlated_field::RelaxedFieldOracle;
use stencils::Stencil;

const CHANNELS: [usize; 5] = [0, 1, 4, 5, 6];
const CHANNEL_NAMES: [&str; 5] = ["total", "parent", "SS-alpha", "OS", "SS-beta"];
const FIELD_SCALES: [f64; 6] = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0];

#[test]
fn restricted_common_origin_complete_mixed_tensor() {
    fit_complete_tensor(false, false);
}

#[test]
fn unrestricted_common_origin_complete_mixed_tensor() {
    fit_complete_tensor(true, false);
}

#[test]
fn restricted_london_complete_mixed_tensor() {
    fit_complete_tensor(false, true);
}

#[test]
fn unrestricted_london_complete_mixed_tensor() {
    fit_complete_tensor(true, true);
}

fn fit_complete_tensor(unrestricted: bool, london: bool) {
    let origin = if london {
        [0.0; 3]
    } else {
        [0.17, -0.29, 0.31]
    };
    let mut positions = vec![
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
    ];
    if unrestricted {
        positions.push([1.41, 1.83, 1.59]);
    }
    // Use the same centred C1 geometry as the finite London SCF engine.
    for axis in 0..3 {
        let centre = positions.iter().map(|p| p[axis]).sum::<f64>() / positions.len() as f64;
        for p in &mut positions {
            p[axis] -= centre;
        }
    }
    let setup = prepare_molecular(
        &vec![1; positions.len()],
        &positions,
        &MolecularMethod {
            xc: Some("hf".into()),
            basis: "sto-3g".into(),
            unrestricted,
            multiplicity: if unrestricted { 2 } else { 1 },
            grid_weight_cutoff: 0.0,
            dispersion: "none".into(),
            max_iter: 300,
            conv_tol: 1e-14,
            conv_tol_dp: 1e-13,
            conv_tol_grad: Some(1e-13),
            guess: Some("core".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    assert!(reference.scf.converged);
    let coefficients = Pt2Coefficients {
        same_spin: 0.31,
        opposite_spin: 0.47,
    };
    let value = correlated_hypermagnetizability_for_model(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        None,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        coefficients,
        if london {
            Gauge::Giao
        } else {
            Gauge::CommonOrigin
        },
        Vec3::new(origin[0], origin[1], origin[2]),
        128,
        &ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        },
    )
    .unwrap();
    let label = format!("HF PT2 U={unrestricted} London={london}");
    println!(
        "{label} energy={:.16e} alpha={:?} xi={:?} eta={:?}",
        reference.scf.energy, value.polarizability[0], value.magnetic.total, value.tensor[0]
    );
    println!("{label} field_scales={FIELD_SCALES:?} paths={} roundoff={:?} field={:?} pair={:?} frame={:?}",
        value.directional_evaluations, value.accumulation_roundoff,
        value.magnetic.response_residuals, value.magnetic.amplitude_residuals,
        value.magnetic.orbital_frame_defects);
    assert!(value.magnetic.response_residuals.iter().all(|v| *v < 1e-10));
    assert!(value
        .magnetic
        .amplitude_residuals
        .iter()
        .all(|v| *v < 1e-10));
    assert!(value
        .magnetic
        .orbital_frame_defects
        .iter()
        .all(|v| *v < 1e-10));
    let mut oracle = RelaxedFieldOracle::new(&setup, &reference.scf, coefficients, london, origin);
    let magnetic = [
        value.magnetic.total,
        value.magnetic.parent,
        value.magnetic.same_spin_alpha,
        value.magnetic.opposite_spin,
        value.magnetic.same_spin_beta,
    ];
    let mut failures = Vec::new();
    for a in 0..3 {
        for b in a..3 {
            // Every independent alpha/xi entry is also fitted: Cotton-Mouton
            // orientational terms depend on these complete correlated tensors.
            fit_entry(
                &format!("{label} alpha({a},{b})"),
                &[a, b],
                std::array::from_fn(|i| -value.polarizability[CHANNELS[i]][a][b]),
                &mut oracle,
                &mut failures,
            );
            fit_entry(
                &format!("{label} xi({a},{b})"),
                &[3 + a, 3 + b],
                std::array::from_fn(|i| -magnetic[i][a][b]),
                &mut oracle,
                &mut failures,
            );
            for c in 0..3 {
                for d in c..3 {
                    let exact = std::array::from_fn(|i| -value.tensor[CHANNELS[i]][a][b][c][d]);
                    fit_entry(
                        &format!("{label} eta({a},{b},{c},{d})"),
                        &[a, b, 3 + c, 3 + d],
                        exact,
                        &mut oracle,
                        &mut failures,
                    );
                    // The full 81-entry representation must preserve both
                    // static pair permutations and the SS/OS composition.
                    for k in 0..9 {
                        assert_eq!(value.tensor[k][a][b][c][d], value.tensor[k][b][a][d][c]);
                    }
                    let composed = exact[1]
                        + coefficients.same_spin * (exact[2] + exact[4])
                        + coefficients.opposite_spin * exact[3];
                    assert!((exact[0] - composed).abs() < 1e-8);
                    assert_eq!(value.tensor[7][a][b][c][d], 0.0);
                    assert_eq!(value.tensor[8][a][b][c][d], 0.0);
                }
            }
        }
    }
    components::assert_no_failures(&label, &failures);
}

fn fit_entry(
    label: &str,
    axes: &[usize],
    exact: [f64; 5],
    oracle: &mut RelaxedFieldOracle<'_>,
    failures: &mut Vec<String>,
) {
    println!("{label} analytic_energy_derivatives={exact:?}");
    let names = CHANNEL_NAMES.map(|c| format!("{label} {c}"));
    stencils::fit_components_staged(
        &names,
        &[
            ([0.08, 0.06, 0.04, 0.03], Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], Stencil::Second),
            ([0.16, 0.12, 0.08, 0.06], Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], Stencil::Second),
            ([0.005, 0.00375, 0.0025, 0.001875], Stencil::Second),
            ([0.00125, 0.0009375, 0.000625, 0.00046875], Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 5];
            for (k, &h) in steps.iter().enumerate() {
                let finite = differentiate(oracle, axes, [0.0; 6], h, stencil);
                for i in 0..5 {
                    errors[i][k] = (finite[i] - exact[i]).abs();
                }
            }
            errors
        },
        failures,
    );
}

/// Tensor product of independent first-difference operators. Repeated axes
/// sum their displacements, so diagonal and mixed fourth derivatives have the
/// same independently derived stencil. No Taylor coefficients enter this path.
fn differentiate(
    oracle: &mut RelaxedFieldOracle<'_>,
    axes: &[usize],
    field: [f64; 6],
    h: f64,
    stencil: Stencil,
) -> [f64; 5] {
    let Some((&axis, rest)) = axes.split_first() else {
        return oracle.at(field);
    };
    let step = h * FIELD_SCALES[axis];
    let mut at = |offset: f64| {
        let mut shifted = field;
        shifted[axis] += offset * step;
        differentiate(oracle, rest, shifted, h, stencil)
    };
    let plus = at(1.0);
    let minus = at(-1.0);
    let (plus2, minus2) = if stencil == Stencil::Fourth {
        (at(2.0), at(-2.0))
    } else {
        ([0.0; 5], [0.0; 5])
    };
    std::array::from_fn(|i| stencil.first(plus[i], minus[i], plus2[i], minus2[i], step))
}
