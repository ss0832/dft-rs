// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete Cartesian PT2 magnetic energy Hessians against independently
//! converged finite-field determinants and direct complex AO-to-MO sums.
//! Separate SS-alpha/OS/SS-beta channels are fitted without symmetry copying.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::integrals::london::FieldPath;
use dft_rs::response::correlated_field::correlated_field_taylor;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::double_hybrid::Pt2Coefficients;

#[path = "support/complex_pt2.rs"]
mod complex_pt2;
#[path = "support/finite_correlated_field.rs"]
mod finite_correlated_field;
use finite_correlated_field::RelaxedFieldOracle;

#[test]
fn every_magnetic_hessian_component_includes_complex_pt2_amplitude_and_orbital_response() {
    fit_magnetic_hessians(true);
}

#[test]
fn every_common_origin_magnetic_hessian_component_has_independent_energy_fits() {
    fit_magnetic_hessians(false);
}

fn fit_magnetic_hessians(london: bool) {
    let origin = if london {
        [0.0; 3]
    } else {
        [0.17, -0.29, 0.31]
    };

    let models = ["hf", "svwn5", "pbe", "tpss", "pbe0"];
    if let Ok(model) = std::env::var("DFT_RS_TEST_MODEL") {
        assert!(
            models.contains(&model.as_str()),
            "unknown acceptance model: {model}"
        );
    }
    for xc in models {
        for unrestricted in [false, true] {
            if std::env::var("DFT_RS_TEST_MODEL")
                .ok()
                .is_some_and(|v| v != xc)
            {
                continue;
            }
            let mut positions = vec![
                [0.13, -0.21, 0.19],
                [1.57, 0.16, -0.08],
                [-0.19, 1.93, 0.37],
                [0.42, 0.29, 2.28],
            ];
            if unrestricted {
                positions.push([1.41, 1.83, 1.59]);
            }
            // The finite-field engine centres its basis. Starting from
            // precisely that same geometry makes the direct integral oracle
            // independent of any untested AO phase-transport convention.
            for axis in 0..3 {
                let centre =
                    positions.iter().map(|p| p[axis]).sum::<f64>() / positions.len() as f64;
                for p in &mut positions {
                    p[axis] -= centre;
                }
            }
            let method = MolecularMethod {
                xc: Some(xc.into()),
                basis: "sto-3g".into(),
                unrestricted,
                multiplicity: if unrestricted { 2 } else { 1 },
                grid_level: 0,
                grid_weight_cutoff: 0.0,
                dispersion: "none".into(),
                max_iter: 300,
                conv_tol: 1e-14,
                conv_tol_dp: 1e-13,
                conv_tol_grad: Some(1e-13),
                guess: Some("core".into()),
                ..Default::default()
            };
            let setup = prepare_molecular(&vec![1; positions.len()], &positions, &method).unwrap();
            let reference = run_molecular_reference(&setup).unwrap();
            assert!(reference.scf.converged);
            let gaps: [f64; 2] = std::array::from_fn(|s| {
                let occupied = if s == 0 {
                    reference.scf.n_alpha
                } else {
                    reference.scf.n_beta
                };
                reference.scf.mo_energy[s][occupied] - reference.scf.mo_energy[s][occupied - 1]
            });
            println!("{xc} London={london} U={unrestricted} parent occupied-virtual gaps {gaps:?} Hartree");
            let coefficients = Pt2Coefficients {
                same_spin: 0.31,
                opposite_spin: 0.47,
            };
            let config = ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            };
            let series = |direction: [f64; 3]| {
                let result = correlated_field_taylor(
                    &setup.molecule,
                    &setup.basis,
                    &setup.functional,
                    None,
                    setup.grid.as_ref(),
                    &reference.scf,
                    &setup.options,
                    coefficients,
                    &FieldPath {
                        electric: [0.0; 3],
                        magnetic: direction,
                        origin,
                        london,
                        order: 2,
                    },
                    None,
                    2048,
                    &config,
                )
                .unwrap();
                println!("{xc} London={london} U={unrestricted} B={direction:?} raw amplitude={:?} field={:?} frame={:.12e} imaginary={:.12e}",
                    result.amplitude_residuals, result.parent.parent.residuals,
                    result.parent.orbital_frame_defect, result.imaginary_energy_defect);
                assert!(result.amplitude_residuals.iter().all(|r| *r < 1e-10));
                assert!(result.parent.orbital_frame_defect < 1e-10);
                [
                    result.energy[2],
                    result.parent.parent.energy[2],
                    result.same_spin_alpha[2],
                    result.opposite_spin[2],
                    result.same_spin_beta[2],
                ]
            };
            let mut oracle =
                RelaxedFieldOracle::new(&setup, &reference.scf, coefficients, london, origin);
            let mut finite =
                |field: [f64; 3]| oracle.at([0.0, 0.0, 0.0, field[0], field[1], field[2]]);
            let tensor =
                dft_rs::properties::correlated_magnetic::correlated_magnetizability_for_model(
                    &setup.molecule,
                    &setup.basis,
                    &setup.functional,
                    None,
                    setup.grid.as_ref(),
                    &reference.scf,
                    &setup.options,
                    coefficients,
                    if london {
                        dft_rs::properties::magnetic::Gauge::Giao
                    } else {
                        dft_rs::properties::magnetic::Gauge::CommonOrigin
                    },
                    dft_rs::Vec3::new(origin[0], origin[1], origin[2]),
                    128,
                    &config,
                )
                .unwrap();
            let response_tensors = [
                tensor.total,
                tensor.parent,
                tensor.same_spin_alpha,
                tensor.opposite_spin,
                tensor.same_spin_beta,
            ];
            let zero = finite([0.0; 3]);
            let diagonal: Vec<_> = (0..3)
                .map(|axis| {
                    let mut d = [0.0; 3];
                    d[axis] = 1.0;
                    series(d)
                })
                .collect();
            let mut failures = Vec::new();
            for a in 0..3 {
                for b in a..3 {
                    let exact = if a == b {
                        diagonal[a].map(|v| 2.0 * v)
                    } else {
                        let mut d = [0.0; 3];
                        d[a] = 1.0;
                        d[b] = 1.0;
                        let sum = series(d);
                        std::array::from_fn(|c| sum[c] - diagonal[a][c] - diagonal[b][c])
                    };
                    // Check the common tensor assembler against the original
                    // path contraction before fitting it to independent SCF energies.
                    let assembled: [f64; 5] = std::array::from_fn(|c| -response_tensors[c][a][b]);
                    for c in 0..5 {
                        assert!((assembled[c] - exact[c]).abs() < 1e-9);
                        assert!(
                            (response_tensors[c][a][b] - response_tensors[c][b][a]).abs() < 1e-14
                        );
                    }
                    let exact = assembled;
                    let names: Vec<_> = ["total", "parent", "SS-alpha", "OS", "SS-beta"]
                        .iter()
                        .flat_map(|channel| {
                            let mut v = vec![format!(
                                "{xc} London={london} U={unrestricted} {channel} B({a},{b})"
                            )];
                            if a != b {
                                v.push(format!(
                                    "{xc} London={london} U={unrestricted} {channel} B({b},{a})"
                                ));
                            }
                            v
                        })
                        .collect();
                    components::fit_components(
                        &names,
                        &[
                            [0.08, 0.06, 0.04, 0.03],
                            [0.02, 0.015, 0.01, 0.0075],
                            [0.16, 0.12, 0.08, 0.06],
                            [0.005, 0.00375, 0.0025, 0.001875],
                            [0.00125, 0.0009375, 0.000625, 0.00046875],
                        ],
                        |steps| {
                            let mut errors = vec![[0.0; 4]; names.len()];
                            for (k, &h) in steps.iter().enumerate() {
                                let mut at = |x: f64, y: f64| {
                                    let mut d = [0.0; 3];
                                    d[a] += x;
                                    d[b] += y;
                                    finite(d)
                                };
                                let (plus, minus, pm, mp) = if a == b {
                                    (at(h, 0.0), at(-h, 0.0), zero, zero)
                                } else {
                                    (at(h, h), at(-h, -h), at(h, -h), at(-h, h))
                                };
                                for c in 0..5 {
                                    let fd = if a == b {
                                        (plus[c] + minus[c] - 2.0 * zero[c]) / (h * h)
                                    } else {
                                        (plus[c] + minus[c] - pm[c] - mp[c]) / (4.0 * h * h)
                                    };
                                    let stride = if a == b { 1 } else { 2 };
                                    errors[stride * c][k] = (fd - exact[c]).abs();
                                    if a != b {
                                        errors[stride * c + 1][k] = (fd - exact[c]).abs();
                                    }
                                }
                            }
                            errors
                        },
                        &mut failures,
                    );
                }
            }
            components::assert_no_failures("complex PT2 magnetic Hessian", &failures);
        }
    }
}
