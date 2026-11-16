// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Finite electric bias: analytic nuclear coefficients against independently
//! displaced stationary enthalpies and forces. Differences are validation only.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::response::{
    nuclear::{lattice_finite_field, lattice_path_energy_at_field},
    ResponseSolverConfig,
};
use dft_rs::xc::ResolvedFunctional;
use dft_rs::{
    basis::data,
    math::Vec3,
    periodic::{Cell, Lattice},
    system::{Atom, Molecule},
};

const FIELD: [f64; 3] = [0.001, -0.0015, 0.002];

#[test]
fn implicit_chg_biased_phonons_fit_all_cartesian_hessian_entries() {
    use stencils::Stencil;

    // The oblique field and tilted LiH in a triclinic cell leave no mirror
    // axis or inversion. The two routes independently form first and second
    // derivatives of the dispersion term and the stationary enthalpy.
    let cell = System {
        numbers: vec![3, 1],
        positions: vec![[0.17, -0.25, 0.08], [2.89, 0.44, -0.57]],
        cell: Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
        pbc: [true; 3],
    };
    let controls = |forces| {
        let mut values = Value::object()
            .with("xc", "wb97x-d")
            .with("basis", "sto-3g")
            .with("k_mesh", [1_usize; 3])
            .with("field", FIELD)
            .with("disp_cutoff", 12.3)
            .with("exchange_cutoff", 5.6)
            .with("overlap_cutoff", 6.0)
            .with("grid_shell", 1)
            .with("grid_level", 0)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("conv_tol_dp", 1e-11)
            .with("conv_tol_grad", 1e-11);
        if forces {
            values.set("forces", true);
        }
        Controls::from_value(&values).unwrap()
    };
    let numeric = |report: &Value, key: &str| report.get(key).unwrap().as_f64().unwrap();
    let array = |report: &Value, key: &str| report.get(key).unwrap().numeric_array().unwrap().1;
    let phonons = run("phonons", &cell, &controls(false)).unwrap();
    let field_controls = controls(true);
    let reference = run("periodic-efield", &cell, &field_controls).unwrap();
    let hessian = array(&phonons, "force_constants");
    let force = array(&reference, "forces");
    let dispersion = numeric(&phonons, "dispersion_energy_per_primitive_cell");
    assert!(dispersion < 0.0);
    assert!(
        (dispersion - numeric(&reference, "dispersion_energy_per_primitive_cell")).abs() < 1e-13
    );
    assert!(
        (numeric(&phonons, "field_energy") - numeric(&reference, "field_energy")).abs() < 1e-12
    );
    assert!(numeric(&phonons, "acoustic_residual") < 1e-8);
    assert_eq!(hessian.len(), 36);
    println!("Biased omegaB97X-D LiH enthalpy={:.17e} dispersion={dispersion:.17e} reference_residual={:.17e} response={:?} hessian={hessian:?}", numeric(&phonons, "field_energy"), numeric(&phonons, "series_reference_residual"), array(&phonons, "response_residuals"));

    let stages = [
        ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
        ([0.02, 0.015, 0.01, 0.005], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    for column in 0..6 {
        let mut samples = std::collections::BTreeMap::new();
        let mut point = |h: f64| {
            *samples.entry(h.to_bits()).or_insert_with(|| {
                let mut geometry = cell.clone();
                geometry.positions[column / 3][column % 3] += h;
                let report = run("periodic-efield", &geometry, &field_controls).unwrap();
                let energy = numeric(&report, "field_energy");
                println!("Biased LiH displaced column={column} h={h:.17e} E={energy:.17e} residual={:.17e}", numeric(&report, "series_reference_residual"));
                (energy, <[f64; 6]>::try_from(array(&report, "forces")).unwrap())
            })
        };
        components::fit_components(
            &[format!("Biased LiH CHG force [{column}]")],
            &[stages[0].0, [0.08, 0.06, 0.04, 0.02]],
            |steps| {
                vec![steps.map(|h| {
                    let difference = -Stencil::Second.first(point(h).0, point(-h).0, 0.0, 0.0, h);
                    (difference - force[column]).abs()
                })]
            },
            &mut failures,
        );
        let names: Vec<_> = (0..6)
            .map(|row| format!("Biased LiH CHG Hessian [{row},{column}]"))
            .collect();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 6];
                for (i, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (point(h).1, point(-h).1);
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (point(2.0 * h).1, point(-2.0 * h).1)
                    } else {
                        ([0.0; 6], [0.0; 6])
                    };
                    for row in 0..6 {
                        let difference =
                            -stencil.first(plus[row], minus[row], plus2[row], minus2[row], h);
                        errors[row][i] = (difference - hessian[6 * row + column]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("implicit CHG finite-field Gamma Hessian", &failures);
}

fn reference(
    positions: &[[f64; 3]],
    numbers: &[u8],
) -> (
    Cell,
    dft_rs::periodic::scf::PeriodicScfResult,
    PeriodicScfOptions,
) {
    let molecule = Molecule::new(
        positions
            .iter()
            .zip(numbers)
            .map(|(p, z)| Atom {
                z: *z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    );
    let cell = Cell::new(
        molecule,
        Lattice::cubic(10.0),
        &data::load("sto-3g").unwrap(),
        5.0,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        max_iter: 300,
        ..Default::default()
    };
    let state = run_periodic_scf(&cell, &ResolvedFunctional::hartree_fock(), &options).unwrap();
    assert!(state.converged);
    (cell, state, options)
}

#[test]
fn finite_bias_nuclear_orders_one_through_four_have_independent_force_fits() {
    let positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]];
    let direction = [[-0.2, 0.1, -0.3], [0.2, -0.1, 0.3]];
    let functional = ResolvedFunctional::hartree_fock();
    let (cell, state, options) = reference(&positions, &[1, 1]);
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let series = lattice_path_energy_at_field(
        &cell,
        &functional,
        &state,
        &options,
        1,
        &direction,
        2,
        FIELD,
        &config,
    )
    .unwrap();
    assert!(series.converged && series.residuals.iter().all(|r| *r <= 1e-12));
    let point = |t: f64| {
        let moved: Vec<_> = positions
            .iter()
            .zip(&direction)
            .map(|(p, d)| std::array::from_fn(|a| p[a] + t * d[a]))
            .collect();
        let (cell, state, options) = reference(&moved, &[1, 1]);
        let report =
            lattice_finite_field(&cell, &functional, &state, &options, 1, FIELD, true).unwrap();
        println!(
            "finite-bias displaced reference t={t:.16e} residual={:.16e} iterations={}",
            report.residual, report.iterations
        );
        let gradient = -report
            .forces
            .unwrap()
            .iter()
            .zip(&direction)
            .map(|(f, d)| (0..3).map(|a| f[a] * d[a]).sum::<f64>())
            .sum::<f64>();
        (report.energy, gradient)
    };
    let mut cache = std::collections::BTreeMap::new();
    let mut cached_point = |t: f64| {
        *cache.entry(t.to_bits()).or_insert_with(|| {
            let value = point(t);
            println!(
                "finite-bias displaced point t={t:.16e} energy={:.16e} gradient={:.16e}",
                value.0, value.1
            );
            value
        })
    };
    let zero = cached_point(0.0);
    // Retain the failed coarse Hessian interval as evidence. Its small h^2
    // coefficient is contaminated by h^4 terms; resolve a finer interval
    // without changing the production derivative or acceptance thresholds.
    let steps = [0.12, 0.09, 0.06, 0.03, 0.015, 0.0075, 0.00375];
    let mut estimates: Vec<[f64; 4]> = Vec::new();
    for h in steps {
        let (plus, minus, twice_plus, twice_minus) = (
            cached_point(h),
            cached_point(-h),
            cached_point(2.0 * h),
            cached_point(-2.0 * h),
        );
        estimates.push([
            (plus.0 - minus.0) / (2.0 * h),
            (plus.1 - minus.1) / (2.0 * h),
            (plus.1 - 2.0 * zero.1 + minus.1) / (h * h),
            (twice_plus.1 - 2.0 * plus.1 + 2.0 * minus.1 - twice_minus.1) / (2.0 * h * h * h),
        ]);
    }
    let mut checks = Vec::new();
    for order in 1..=4 {
        let factorial = (1..=order).product::<usize>() as f64;
        let target = factorial * series.energy[order];
        let errors: Vec<_> = estimates
            .iter()
            .map(|v| (v[order - 1] - target).abs())
            .collect();
        println!("finite-bias nuclear order={order} target={target:.16e} all_steps={steps:?} all_errors={errors:?}");
        let coarse = convergence::error_power_fit(&steps[..4], &errors[..4]);
        println!("finite-bias nuclear order={order} interval=original_coarse steps={:?} errors={:?} p={:.9} R2={:.12}", &steps[..4], &errors[..4], coarse.exponent, coarse.r_squared);
        if errors[3..].iter().all(|e| e.is_finite() && *e > 0.0) {
            let finer = convergence::error_power_fit(&steps[3..], &errors[3..]);
            println!("finite-bias nuclear order={order} interval=finer_diagnostic steps={:?} errors={:?} p={:.9} R2={:.12}", &steps[3..], &errors[3..], finer.exponent, finer.r_squared);
        } else {
            println!("finite-bias nuclear order={order} interval=finer_diagnostic invalid_nonpositive_or_nonfinite_error steps={:?} errors={:?}; no error floor applied", &steps[3..], &errors[3..]);
        }
        let (start, last, previous) = if order == 2 { (3, 6, 5) } else { (0, 3, 2) };
        let fit = convergence::error_power_fit(&steps[start..start + 4], &errors[start..start + 4]);
        let rich = ((4.0 * estimates[last][order - 1] - estimates[previous][order - 1]) / 3.0
            - target)
            .abs();
        println!("finite-bias nuclear order={order} interval=acceptance target={target:.16e} steps={:?} errors={:?} p={:.9} R2={:.12} Richardson={rich:.12e} residuals={:?}",&steps[start..start+4],&errors[start..start+4],fit.exponent,fit.r_squared,series.residuals);
        checks.push((order, fit, rich));
    }
    // Print every measured order before asserting, so a failed fit cannot
    // hide the remaining expensive independently displaced results.
    for (order, fit, rich) in checks {
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 2e-6,
            "finite-bias nuclear order={order}"
        );
    }
}
