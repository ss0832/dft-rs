// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every independent molecular magnetic energy-Hessian component and all
//! target/nonlocal/SS/OS channels against independent finite London SCF,
//! direct occupied-orbital nonlocal energy and semicanonical PT2 sums.
#[path = "support/complex_pt2.rs"]
mod complex_pt2;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nonlocal_complex_oracle.rs"]
mod nonlocal_complex_oracle;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::integrals::london::{repulsion_series, ComplexMatrix, FieldPath};
use dft_rs::nlc::Vv10Params;
use dft_rs::response::complex::Complex;
use dft_rs::response::correlated_field::correlated_field_taylor_for_model;
use dft_rs::response::field_path::FieldEnergyModel;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::magnetic::{magnetic_field_scf_with_core, MagneticFieldOptions};
use dft_rs::xc::double_hybrid::Pt2Coefficients;
use dft_rs::Vec3;

#[test]
fn every_nonlocal_target_and_parent_magnetic_hessian_channel_has_independent_fits() {
    for xc in ["hf", "pbe0"] {
        for unrestricted in [false, true] {
            for parent_nonlocal in [false, true] {
                if std::env::var("DFT_RS_TEST_MODEL")
                    .ok()
                    .is_some_and(|s| s != xc)
                {
                    continue;
                }
                if std::env::var("DFT_RS_TEST_PARENT_NLC")
                    .ok()
                    .is_some_and(|s| (s == "1") != parent_nonlocal)
                {
                    continue;
                }
                let mut positions = vec![
                    [0.13, -0.21, 0.19],
                    [1.5, -0.04, 0.06],
                    [0.44, 3.02, 0.66],
                    [1.94, 3.24, 1.15],
                ];
                if unrestricted {
                    positions.push([1.41, 1.83, -1.59]);
                }
                for a in 0..3 {
                    let center =
                        positions.iter().map(|r| r[a]).sum::<f64>() / positions.len() as f64;
                    for r in &mut positions {
                        r[a] -= center;
                    }
                }
                let method = MolecularMethod {
                    xc: Some(xc.into()),
                    basis: "sto-3g".into(),
                    unrestricted,
                    multiplicity: if unrestricted { 2 } else { 1 },
                    conv_tol: 1e-14,
                    conv_tol_dp: 1e-14,
                    conv_tol_grad: Some(1e-14),
                    max_iter: 400,
                    guess: Some("core".into()),
                    dispersion: if parent_nonlocal { "vv10" } else { "none" }.into(),
                    grid_level: 0,
                    vv10_grid_level: 0,
                    grid_weight_cutoff: 0.0,
                    ..Default::default()
                };
                let setup =
                    prepare_molecular(&vec![1; positions.len()], &positions, &method).unwrap();
                let reference = run_molecular_reference(&setup).unwrap();
                assert!(reference.scf.converged);
                let config = ResponseSolverConfig {
                    tolerance: 1e-12,
                    ..Default::default()
                };
                let coefficients = Pt2Coefficients {
                    same_spin: 0.31,
                    opposite_spin: 0.47,
                };
                let target_coefficient = 0.37;
                let target_parameters = Vv10Params::WB97XV;
                let target = FieldEnergyModel::new(&setup.functional)
                    .with_vv10(target_coefficient, target_parameters)
                    .unwrap();
                let nlgrid = setup.options.nonlocal_grid(&setup.molecule).unwrap();
                let response = |field| {
                    let value = correlated_field_taylor_for_model(
                        &setup.molecule,
                        &setup.basis,
                        &setup.functional,
                        Some(target.clone()),
                        setup.grid.as_ref(),
                        &reference.scf,
                        &setup.options,
                        coefficients,
                        &FieldPath {
                            electric: [0.0; 3],
                            magnetic: field,
                            origin: [0.0; 3],
                            london: true,
                            order: 2,
                        },
                        None,
                        2048,
                        &config,
                    )
                    .unwrap();
                    let raw = value
                        .parent
                        .parent
                        .residuals
                        .iter()
                        .chain(&value.amplitude_residuals)
                        .copied()
                        .fold(0.0_f64, f64::max);
                    println!("{xc} U={unrestricted} parentNLC={parent_nonlocal} field={field:?} raw residual {raw:.12e}");
                    assert!(raw <= config.acceptance());
                    [
                        value.energy[2],
                        value.parent.parent.energy[2],
                        value.parent.target_energy[2],
                        value.correlation[2],
                        value.same_spin_alpha[2],
                        value.opposite_spin[2],
                        value.same_spin_beta[2],
                        value.parent.target_nonlocal_energy[2],
                        value.parent.parent_nonlocal_energy[2],
                    ]
                };
                let mut diagonal = Vec::new();
                for a in 0..3 {
                    let mut d = [0.0; 3];
                    d[a] = 1.0;
                    diagonal.push(response(d));
                }
                let n = setup.basis.nao_sph;
                let finite_cache = std::cell::RefCell::new(std::collections::BTreeMap::<
                    [u64; 3],
                    [f64; 9],
                >::new());
                let finite = |field: [f64; 3]| {
                    let key = field.map(f64::to_bits);
                    if let Some(e) = finite_cache.borrow().get(&key) {
                        return *e;
                    }
                    let state = magnetic_field_scf_with_core(
                        &setup.molecule,
                        &setup.basis,
                        &setup.functional,
                        setup.grid.as_ref(),
                        &reference.scf,
                        &setup.options,
                        &MagneticFieldOptions {
                            field,
                            spin_zeeman: false,
                            series_order: Some(16),
                            series_tolerance: 1e-15,
                        },
                        &[ComplexMatrix::zeros(n, n), ComplexMatrix::zeros(n, n)],
                    )
                    .unwrap();
                    let shift = Vec3::new(state.shift[0], state.shift[1], state.shift[2]);
                    let mut basis = setup.basis.clone();
                    for shell in &mut basis.shells {
                        shell.center += shift;
                    }
                    let mut grid = nlgrid.clone();
                    for r in &mut grid.points {
                        *r += shift;
                    }
                    let target_nlc = nonlocal_complex_oracle::energy(
                        &basis,
                        &grid,
                        field,
                        &state.coefficients,
                        [state.n_alpha, state.n_beta],
                        target_parameters,
                    );
                    if parent_nonlocal {
                        let parent_nlc = nonlocal_complex_oracle::energy(
                            &basis,
                            &grid,
                            field,
                            &state.coefficients,
                            [state.n_alpha, state.n_beta],
                            Vv10Params::ORIGINAL,
                        );
                        assert!((parent_nlc - state.e_nlc).abs() < 1e-12);
                    }
                    let target_energy =
                        state.energy - state.e_nlc + target_coefficient * target_nlc;
                    let raw = repulsion_series(
                        &basis,
                        &FieldPath {
                            electric: [0.0; 3],
                            magnetic: field,
                            origin: [0.0; 3],
                            london: true,
                            order: 16,
                        },
                        None,
                    )
                    .unwrap();
                    let mut g = vec![Complex::ZERO; n.pow(4)];
                    for (re, im) in raw {
                        for (k, v) in g.iter_mut().enumerate() {
                            *v += Complex::new(re[k], im[k]);
                        }
                    }
                    let e = complex_pt2::finite_channels(
                        n,
                        &g,
                        &state.coefficients,
                        &state.orbital_energies,
                        [state.n_alpha, state.n_beta],
                    );
                    let correlation =
                        coefficients.same_spin * (e[0] + e[2]) + coefficients.opposite_spin * e[1];
                    let values = [
                        target_energy + correlation,
                        state.energy,
                        target_energy,
                        correlation,
                        e[0],
                        e[1],
                        e[2],
                        target_coefficient * target_nlc,
                        state.e_nlc,
                    ];
                    finite_cache.borrow_mut().insert(key, values);
                    values
                };
                let zero = finite([0.0; 3]);
                let mut failures = Vec::new();
                for a in 0..3 {
                    for b in a..3 {
                        let exact = if a == b {
                            diagonal[a].map(|v| 2.0 * v)
                        } else {
                            let mut direction = [0.0; 3];
                            direction[a] = 1.0;
                            direction[b] = 1.0;
                            let both = response(direction);
                            std::array::from_fn(|c| both[c] - diagonal[a][c] - diagonal[b][c])
                        };
                        let count = if parent_nonlocal { 9 } else { 8 };
                        let names:Vec<_>=["total","parent","target","correlation","SS-alpha","OS","SS-beta","target-NLC","parent-NLC"].iter().take(count).map(|channel|
                format!("{xc} U={unrestricted} parentNLC={parent_nonlocal} nonlocal {channel} B({a},{b})")).collect();
                        stencils::fit_components_staged(
                            &names,
                            &[
                                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                                ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                                ([0.16, 0.12, 0.08, 0.06], stencils::Stencil::Second),
                                (
                                    [0.005, 0.00375, 0.0025, 0.001875],
                                    stencils::Stencil::Second,
                                ),
                                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
                            ],
                            |steps, stencil| {
                                let mut errors = vec![[0.0; 4]; count];
                                for (k, &h) in steps.iter().enumerate() {
                                    let at = |x: f64, y: f64| {
                                        let mut d = [0.0; 3];
                                        d[a] += x;
                                        d[b] += y;
                                        finite(d)
                                    };
                                    let values = if a == b {
                                        let plus = at(h, 0.0);
                                        let minus = at(-h, 0.0);
                                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                                            (at(2.0 * h, 0.0), at(-2.0 * h, 0.0))
                                        } else {
                                            ([0.0; 9], [0.0; 9])
                                        };
                                        std::array::from_fn::<_, 9, _>(|c| {
                                            if stencil == stencils::Stencil::Second {
                                                (plus[c] + minus[c] - 2.0 * zero[c]) / (h * h)
                                            } else {
                                                (-p2[c] + 16.0 * plus[c] - 30.0 * zero[c]
                                                    + 16.0 * minus[c]
                                                    - m2[c])
                                                    / (12.0 * h * h)
                                            }
                                        })
                                    } else {
                                        let along = |x| {
                                            let p = at(x, h);
                                            let m = at(x, -h);
                                            let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                                                (at(x, 2.0 * h), at(x, -2.0 * h))
                                            } else {
                                                ([0.0; 9], [0.0; 9])
                                            };
                                            std::array::from_fn::<_, 9, _>(|c| {
                                                stencil.first(p[c], m[c], p2[c], m2[c], h)
                                            })
                                        };
                                        let p = along(h);
                                        let m = along(-h);
                                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                                            (along(2.0 * h), along(-2.0 * h))
                                        } else {
                                            ([0.0; 9], [0.0; 9])
                                        };
                                        std::array::from_fn(|c| {
                                            stencil.first(p[c], m[c], p2[c], m2[c], h)
                                        })
                                    };
                                    for c in 0..count {
                                        errors[c][k] = (values[c] - exact[c]).abs();
                                    }
                                }
                                errors
                            },
                            &mut failures,
                        );
                    }
                }
                components::assert_no_failures(
                    "full nonlocal composite magnetic Hessian",
                    &failures,
                );
            }
        }
    }
}
