// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The stationary Dirac energy is reconverged independently at each field.
//! Every independent Cartesian entry is fitted without subtracting a floor.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use std::cell::RefCell;
use std::collections::BTreeMap;
use stencils::Stencil;

const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.77, 0.36, -0.08],
    [2.44, 2.35, 1.20],
    [3.97, 2.88, 1.67],
];
const BIAS: [f64; 3] = [0.009, -0.007, 0.012];
const SITE: usize = 0;
const RADIO: [f64; 4] = [0.08, 0.10, 0.115, 0.094];

fn scalar(value: &Value, key: &str) -> f64 {
    value.get(key).and_then(Value::as_f64).unwrap()
}

fn values(value: &Value, key: &str) -> Vec<f64> {
    value.get(key).unwrap().numeric_array().unwrap().1
}

fn verify_report(report: &Value, property: &str, absolute: &str, relaxed: &str) {
    assert_eq!(
        report.get("provenance").and_then(Value::as_str),
        Some("analytic")
    );
    assert!(scalar(report, "response_residual") < 1e-10);
    assert!(scalar(report, "magnetic_response_reciprocity_defect") < 1e-10);
    assert!(scalar(report, "magnetic_response_sector_metric_defect") < 1e-10);
    assert!(scalar(report, "magnetic_response_sector_operator_residual") < 1e-8);
    assert!(scalar(report, "magnetic_response_reference_energy_defect") < 1e-10);
    assert!(scalar(report, "magnetic_response_reference_fock_defect") < 1e-10);
    let tensor = values(report, property);
    let explicit = values(report, absolute);
    let relaxation = values(report, relaxed);
    assert_eq!(tensor.len(), 9);
    assert!(tensor
        .iter()
        .zip(explicit)
        .zip(relaxation)
        .all(|((value, clamped), relaxed)| (value - clamped - relaxed).abs() < 2e-10));
    println!("{property} {tensor:?}; CPKS {:.6e}, electronic-sector metric {:.6e}, projector {:.6e} Hartree, stationary energy {:.6e} Hartree, Fock {:.6e} Hartree, reciprocity {:.6e}",
        scalar(report,"response_residual"),
        scalar(report,"magnetic_response_sector_metric_defect"),
        scalar(report,"magnetic_response_sector_operator_residual"),
        scalar(report,"magnetic_response_reference_energy_defect"),
        scalar(report,"magnetic_response_reference_fock_defect"),
        scalar(report,"magnetic_response_reciprocity_defect"));
}

#[test]
fn c1_dirac_uniform_and_mixed_magnetic_stationary_energy() {
    stationary_derivatives("coulomb");
}

#[test]
fn c1_dirac_gaunt_uniform_and_mixed_magnetic_energy() {
    stationary_derivatives("gaunt");
}

#[test]
fn c1_dirac_breit_uniform_and_mixed_magnetic_energy() {
    stationary_derivatives("breit");
}

fn stationary_derivatives(interaction: &str) {
    c1_geometry::assert_c1(&[1; 4], &POSITIONS);
    let molecule = System::molecule(vec![1; 4], POSITIONS.to_vec());
    // Analytic and reconverged energies use the same finite Gaussian basis,
    // nominal speed of light and phase expansion. One s primitive per H
    // keeps the independent finite-field SCFs inexpensive.
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-dirac-magnetic-{interaction}-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, "H 0\nS 1 1.0\n0.59 1.0\n****\n").unwrap();
    let settings = Value::object()
        .with("xc", "hf")
        .with("dirac_interaction", interaction)
        .with("basis", basis.to_str().unwrap())
        .with("dispersion", "none")
        .with("speed_of_light", 6.8)
        .with("nuclear_rms_radii", [0.09, 0.083, 0.075, 0.088])
        .with("dirac_integrals", "london-rmb")
        .with("magnetic_phase_order", 2_usize)
        .with("cpu_threads", 1_usize)
        .with("grid_level", 0_usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 2e-14)
        .with("conv_tol_grad", 5e-12)
        .with("conv_tol_dp", 5e-12);
    let run_it = |feature, settings| {
        run(
            feature,
            &molecule,
            &Controls::from_value(&settings).unwrap(),
        )
        .unwrap()
    };
    let settings = settings.with("magnetic_field", BIAS);
    let derivatives = settings
        .clone()
        .with("relativistic_method", "dirac")
        .with("response_tolerance", 1e-11);
    let magnet = run_it("magnetizability", derivatives.clone());
    let shielding = run_it(
        "nmr-shielding",
        derivatives
            .with("nuclei", [SITE])
            .with("nuclear_magnetic_rms_radii", RADIO),
    );
    verify_report(
        &magnet,
        "magnetizability",
        "magnetizability_explicit_sector",
        "magnetizability_orbital_relaxation",
    );
    verify_report(
        &shielding,
        "shielding_tensors",
        "shielding_explicit_sector_tensors",
        "shielding_orbital_relaxation_tensors",
    );
    let moments = values(&magnet, "magnetic_dipole");
    println!("C1 4c magnetic moment {moments:?} at {BIAS:?}");
    let xi = values(&magnet, "magnetizability");
    let sigma = values(&shielding, "shielding_tensors");
    let cache = RefCell::new(BTreeMap::new());
    let energy = |b: [f64; 3], mu: [f64; 3]| {
        *cache
            .borrow_mut()
            .entry((b.map(f64::to_bits), mu.map(f64::to_bits)))
            .or_insert_with(|| {
                let field = std::array::from_fn::<_, 3, _>(|a| BIAS[a] + b[a]);
                let mut moments = [[0.0; 3]; 4];
                moments[SITE] = mu;
                let value = run_it(
                    "dirac-dks",
                    settings
                        .clone()
                        .with("magnetic_field", field)
                        .with("nuclear_magnetic_rms_radii", RADIO)
                        .with("nuclear_magnetic_moments", moments),
                );
                assert_eq!(value.get("converged").and_then(Value::as_bool), Some(true));
                scalar(&value, "energy")
            })
    };
    let mut directions = Vec::new();
    for (i, &value) in moments.iter().enumerate() {
        directions.push((format!("moment {i}"), (0, i, 0), value));
    }
    for i in 0..3 {
        for j in 0..=i {
            directions.push((format!("magnetizability {i} {j}"), (1, i, j), xi[3 * i + j]));
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            directions.push((
                format!("shielding {i} {j}"),
                (2, i, j),
                sigma[3 * i + j] * 1e-6,
            ));
        }
    }
    let central = |(kind, i, j): (usize, usize, usize), h: f64| {
        let e = |left: f64, right: f64| {
            let mut b = [0.0; 3];
            let mut m = [0.0; 3];
            if kind == 2 {
                m[i] = left;
            } else {
                b[i] = left;
            }
            b[j] += right;
            energy(b, m)
        };
        if kind == 0 {
            -Stencil::Second.first(e(h, 0.0), e(-h, 0.0), 0.0, 0.0, h)
        } else if kind == 1 && i == j {
            -(e(h, 0.0) + e(-h, 0.0) - 2.0 * e(0.0, 0.0)) / h.powi(2)
        } else {
            let derivative = |r| Stencil::Second.first(e(h, r), e(-h, r), 0.0, 0.0, h);
            let value = Stencil::Second.first(derivative(h), derivative(-h), 0.0, 0.0, h);
            if kind == 1 {
                -value
            } else {
                value
            }
        }
    };
    let estimate = |direction, h, stencil| {
        let center = central(direction, h);
        if stencil == Stencil::Fourth {
            (4.0 * center - central(direction, 2.0 * h)) / 3.0
        } else {
            center
        }
    };
    let mut failures = Vec::new();
    let names = directions.iter().map(|d| d.0.clone()).collect::<Vec<_>>();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.044, 0.032, 0.023, 0.016], Stencil::Second),
            ([0.032, 0.023, 0.016, 0.011], Stencil::Second),
            ([0.038, 0.028, 0.020, 0.014], Stencil::Fourth),
        ],
        |steps, stencil| {
            directions
                .iter()
                .map(|(_, d, v)| steps.map(|h| (estimate(*d, h, stencil) - v).abs()))
                .collect()
        },
        &mut failures,
    );
    if !failures.is_empty() {
        // With a small h^2 coefficient, an h^6 term can still bias the
        // first fourth-order interval. Keep those failed fits in the log
        // and isolate the h^4 term on a finer interval for the open slots.
        let unresolved = std::mem::take(&mut failures);
        let remaining = directions
            .iter()
            .filter(|(name, _, _)| unresolved.contains(name))
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &unresolved,
            &[([0.027, 0.021, 0.016, 0.0125], Stencil::Fourth)],
            |steps, stencil| {
                remaining
                    .iter()
                    .map(|(_, d, value)| steps.map(|h| (estimate(*d, h, stencil) - value).abs()))
                    .collect()
            },
            &mut failures,
        );
    }
    let absolute_h = 0.004;
    let absolute = directions
        .iter()
        .map(|(_, d, v)| (estimate(*d, absolute_h, Stencil::Fourth) - v).abs())
        .collect::<Vec<_>>();
    println!(
        "C1 4c fourth-order absolute energy derivative defects at h={absolute_h}: {absolute:?}"
    );
    std::fs::remove_file(basis).unwrap();
    components::assert_no_failures(
        "4c magnetic moments, magnetizability and NMR shielding",
        &failures,
    );
    assert!(absolute.into_iter().all(|error| error < 2e-9));
}
